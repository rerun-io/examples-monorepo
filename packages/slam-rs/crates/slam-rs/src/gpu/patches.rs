//! The GPU [`SourcePatches`]: one device buffer per patch set.

use cubecl::prelude::*;
use nalgebra::Vector2;

use super::kernels::{self, PatchShape, PositionBases};
use super::pyramid::GpuPyramid;
use super::{GpuError, guarded};
use crate::frontend::patterns::Pattern;
use crate::frontend::tracker::{
    PointsSoA, SourcePatches, TrackerError, check_patch_inputs, checked_patch_shape,
};
use crate::pyramid::Pyramid;

/// Layout of the one `store` buffer a patch set owns.
///
/// The three sections `kernels` documents, in one allocation so the per-patch
/// kernels need one binding for all of it: `data`, then `H^-1 J^T`, then the
/// per-level validity flag. Every section has the patch index fast-varying.
///
/// Public because a downloaded `store` is a flat `Vec<f32>` and something has
/// to say where a coefficient sits in it: the tolerance tests read it through
/// [`GpuPatches::layout`] rather than through one accessor per section on the
/// shipped type.
#[derive(Debug, Clone, Copy)]
pub struct StoreLayout {
    /// Patch slots, the stride between two taps of one row.
    pub capacity: usize,
    /// Pattern taps.
    pub taps: usize,
    /// Pyramid levels.
    pub num_levels: usize,
}

impl StoreLayout {
    /// Elements in the `data` section, which is also the `H^-1 J^T` offset.
    fn data_len(&self) -> usize {
        self.num_levels * self.taps * self.capacity
    }

    /// Elements in the whole buffer.
    ///
    /// `elements` rather than `len`: a layout is never empty, so the `len`/
    /// `is_empty` pair a public `len` implies would be a method that always
    /// answers `false`.
    pub fn elements(&self) -> usize {
        // data + three Jacobian rows + one flag per (level, patch)
        4 * self.data_len() + self.num_levels * self.capacity
    }

    /// Index of `data[level][tap][patch]`.
    pub fn data(&self, level: usize, tap: usize, patch: usize) -> usize {
        (level * self.taps + tap) * self.capacity + patch
    }

    /// Index of `h_inv_jt[level][row][tap][patch]`.
    pub fn jacobian(&self, level: usize, row: usize, tap: usize, patch: usize) -> usize {
        self.data_len() + ((level * 3 + row) * self.taps + tap) * self.capacity + patch
    }

    /// Index of `valid[level][patch]`.
    pub fn valid(&self, level: usize, patch: usize) -> usize {
        4 * self.data_len() + level * self.capacity + patch
    }
}

/// One camera's source patches for every pyramid level, resident on the device.
///
/// The positions are the one thing kept on the host as well: the
/// [`SourcePatches`] trait hands them back one at a time and they arrive from
/// the host to begin with, so mirroring them costs two `f32` per patch and
/// saves a download per frame.
pub struct GpuPatches<P: Pattern, R: Runtime> {
    client: ComputeClient<R>,
    layout: StoreLayout,
    len: usize,
    /// `data`, `H^-1 J^T` and the validity flags, in that order.
    store: cubecl::server::Handle,
    /// `[x; capacity]`, `[y; capacity]`, `[selected; capacity]`, then the
    /// backward pass's guess offsets `[off_x; capacity]`, `[off_y; capacity]`.
    ///
    /// One buffer, because the patch-build kernel reads the positions and the
    /// selection flag from the same binding, and the tracker's own kernels read
    /// the source position and the offset from it too.
    positions: cubecl::server::Handle,
    /// The host mirror of the positions, for [`SourcePatches::position`].
    host: PointsSoA,
    /// The staging buffer the positions are uploaded from.
    staging: Vec<f32>,
    pattern: std::marker::PhantomData<P>,
}

/// Runs inside the positions buffer, each `len` long.
///
/// Strided by the live patch count, not by the capacity: the transform buffers
/// in `super::track` already are, and at the shipped `max_keypoints = 3000` a
/// capacity stride meant every upload zeroed and shipped 5 x 3000 floats — 60 kB,
/// copied twice on the host inside `create_from_slice` — to carry about two
/// hundred patches, five to seven times a frameset. The cost is ~0.05 ms, which
/// is under this host's noise; what the change buys is that it stops scaling
/// with a config knob and starts scaling with the live patch count, which is
/// what matters on a machine with a tenth of the bandwidth.
pub(super) const POSITION_RUNS: usize = 5;
/// Run index of the selection flag.
pub(super) const SELECTED_RUN: usize = 2;
/// Run index of the backward guess offset's `x`.
pub(super) const OFFSET_RUN: usize = 3;

impl<P: Pattern, R: Runtime> GpuPatches<P, R> {
    /// Storage for `capacity` patches over `num_levels` pyramid levels.
    ///
    /// # Errors
    ///
    /// Whatever the frontend's `checked_patch_shape` refuses, which is what
    /// [`crate::frontend::tracker::PatchSoA::new`] refuses: a GPU tracker turns
    /// down exactly what the CPU one turns down rather than asking the device
    /// for a buffer no allocation could hold. Only the last product differs —
    /// this lane folds `4 * taps + flags` into one buffer. And
    /// [`super::GpuError::DeviceLost`], because the two allocations are device
    /// operations: CubeCL panics rather than returning when the worker
    /// submission fails, so the body runs inside the module's guard (D32).
    pub fn new(
        client: ComputeClient<R>,
        capacity: usize,
        num_levels: usize,
    ) -> Result<Self, TrackerError> {
        guarded(
            GpuError::DeviceLost {
                what: "patch allocation",
            },
            || {
                let (flags, taps): (usize, usize) =
                    checked_patch_shape(capacity, num_levels, P::SIZE)?;
                let elements: usize = taps
                    .checked_mul(4)
                    .and_then(|body| body.checked_add(flags))
                    .ok_or(TrackerError::BufferShapeOverflow {
                        capacity,
                        num_levels,
                        taps: P::SIZE,
                    })?;

                let layout: StoreLayout = StoreLayout {
                    capacity,
                    taps: P::SIZE,
                    num_levels,
                };
                let mut host: PointsSoA = PointsSoA::with_capacity(capacity);
                host.resize(capacity);
                Ok(Self {
                    layout,
                    len: 0,
                    store: client.empty(elements * size_of::<f32>()),
                    positions: client.empty(POSITION_RUNS * capacity * size_of::<f32>()),
                    host,
                    staging: vec![0.0; POSITION_RUNS * capacity],
                    pattern: std::marker::PhantomData,
                    client,
                })
            },
        )
    }

    /// Patches this set can hold.
    pub fn capacity(&self) -> usize {
        self.layout.capacity
    }

    /// Pyramid levels each patch is built at.
    pub fn num_levels(&self) -> usize {
        self.layout.num_levels
    }

    /// The `store` buffer and its element count.
    pub(super) fn store(&self) -> (&cubecl::server::Handle, usize) {
        (&self.store, self.layout.elements())
    }

    /// The positions buffer and its element count.
    pub(super) fn position_buffer(&self) -> (&cubecl::server::Handle, usize) {
        (&self.positions, POSITION_RUNS * self.len)
    }

    /// Where the kernels find the source positions and the selection flag.
    pub(super) fn bases(&self) -> PositionBases {
        PositionBases {
            x: 0,
            y: self.len,
            selected: SELECTED_RUN * self.len,
        }
    }

    /// Where the kernels find the backward pass's guess offsets.
    ///
    /// Beside [`GpuPatches::bases`] so the positions buffer's run layout stays
    /// one module's knowledge: the tracker asks for the offsets rather than
    /// rebuilding them out of [`OFFSET_RUN`] and the stride.
    pub(super) fn offset_bases(&self) -> PositionBases {
        PositionBases {
            x: OFFSET_RUN * self.len,
            y: (OFFSET_RUN + 1) * self.len,
            selected: SELECTED_RUN * self.len,
        }
    }

    /// The shape the launchers need for a call over `self.len` patches.
    pub(super) fn shape(&self) -> PatchShape {
        PatchShape {
            capacity: self.layout.capacity,
            taps: self.layout.taps,
            num_levels: self.layout.num_levels,
            count: self.len,
        }
    }

    /// Upload the source positions and the selection flags.
    fn upload_positions(&mut self, positions: &PointsSoA, selected: Option<&[bool]>) {
        let count: usize = self.len;
        let staging: &mut [f32] = &mut self.staging[..POSITION_RUNS * count];
        staging.fill(0.0);
        staging[..count].copy_from_slice(&positions.xs()[..count]);
        staging[count..2 * count].copy_from_slice(&positions.ys()[..count]);
        for index in 0..count {
            let on: bool = selected.is_none_or(|flags| flags[index]);
            staging[SELECTED_RUN * count + index] = f32::from(u8::from(on));
        }
        self.upload_staging();
    }

    /// Upload only the backward pass's guess offsets, with an all-selected mask.
    ///
    /// The offset is `source position - guess` per patch, which the backward
    /// pass adds back to the forward result
    /// (`frame_to_frame_optical_flow.h:357`). It is computed on the host because
    /// both terms are already there, and it rides in this buffer because the
    /// build needs the buffer anyway. The position runs stay zero on purpose: a
    /// backward build reads its positions out of the *forward result*, on the
    /// device, so nothing would look at them.
    pub(super) fn upload_offsets(&mut self, offset_x: &[f32], offset_y: &[f32]) {
        let count: usize = self.len;
        let staging: &mut [f32] = &mut self.staging[..POSITION_RUNS * count];
        staging.fill(0.0);
        for index in 0..count {
            staging[SELECTED_RUN * count + index] = 1.0;
        }
        staging[OFFSET_RUN * count..(OFFSET_RUN + 1) * count].copy_from_slice(&offset_x[..count]);
        staging[(OFFSET_RUN + 1) * count..POSITION_RUNS * count]
            .copy_from_slice(&offset_y[..count]);
        self.upload_staging();
    }

    /// Replace the device positions buffer with this call's runs of the staging
    /// buffer, which is as long as the capacity but only filled to `len`.
    fn upload_staging(&mut self) {
        self.positions = super::seam::UPLOAD.measure(|| {
            self.client
                .create_from_slice(f32::as_bytes(&self.staging[..POSITION_RUNS * self.len]))
        });
    }

    /// Sample every filled patch at every level of `pyramid`, without waiting.
    pub(super) fn launch_build(&self, pyramid: &GpuPyramid<R>, bases: PositionBases) {
        self.launch_build_from(pyramid, self.position_buffer(), bases);
    }

    /// The same build, reading the positions out of a buffer the caller names.
    ///
    /// This is what keeps the backward pass on the device: its patches sit at
    /// the *forward* translations, which are already in the forward result
    /// buffer, so `positions` points there and nothing round-trips through the
    /// host (`frame_to_frame_optical_flow.h:355`).
    pub(super) fn launch_build_from(
        &self,
        pyramid: &GpuPyramid<R>,
        positions: (&cubecl::server::Handle, usize),
        bases: PositionBases,
    ) {
        if self.len == 0 {
            return;
        }
        kernels::launch_patch_build::<R>(
            &self.client,
            pyramid.buffers(),
            pyramid.meta(),
            positions,
            self.store(),
            self.shape(),
            bases,
        );
    }

    /// Record how many patches the next launch covers, and check the shapes.
    ///
    /// # Errors
    ///
    /// [`TrackerError`] when the positions do not fit, the selection mask is
    /// shorter than the positions, or the pyramid is too shallow — the same
    /// three the CPU patch set refuses.
    pub(super) fn accept(
        &mut self,
        count: usize,
        selected: Option<&[bool]>,
        pyramid_levels: usize,
    ) -> Result<(), TrackerError> {
        check_patch_inputs(
            count,
            self.layout.capacity,
            selected,
            pyramid_levels,
            self.layout.num_levels,
        )?;
        self.len = count;
        Ok(())
    }

    /// Download the whole `store` buffer, for the tolerance tests.
    ///
    /// # Errors
    ///
    /// [`TrackerError::LengthMismatch`] when the device returns the wrong
    /// number of bytes, which is what an incomplete CubeCL runtime does instead
    /// of failing, [`super::GpuError::DeviceReadFailed`] when the read itself
    /// fails, and [`super::GpuError::DeviceLost`] when it panics instead of
    /// failing — this download is not on the per-frame path, so it carries its
    /// own guard rather than sitting inside a stage's (decision D32).
    pub fn read_store(&self) -> Result<Vec<f32>, TrackerError> {
        guarded(
            GpuError::DeviceLost {
                what: "the patch store read",
            },
            || {
                let bytes = self
                    .client
                    .read_one(self.store.clone())
                    .map_err(|error| super::read_failed("the patch store", &error))?;
                let expected: usize = self.layout.elements() * size_of::<f32>();
                if bytes.len() != expected {
                    return Err(TrackerError::LengthMismatch {
                        first_name: "store bytes expected",
                        first: expected,
                        second_name: "returned",
                        second: bytes.len(),
                    });
                }
                Ok(f32::from_bytes(&bytes).to_vec())
            },
        )
    }

    /// Where each coefficient of a downloaded [`GpuPatches::read_store`] sits.
    pub fn layout(&self) -> StoreLayout {
        self.layout
    }
}

impl<P: Pattern, R: Runtime> SourcePatches for GpuPatches<P, R> {
    type Pyramid = GpuPyramid<R>;

    /// Build every patch at every level, one launch, no wait.
    fn build(
        &mut self,
        pyramid: &GpuPyramid<R>,
        positions: &PointsSoA,
        selected: Option<&[bool]>,
    ) -> Result<(), TrackerError> {
        self.prepare(pyramid, positions, selected)?;
        guarded(
            GpuError::DeviceLost {
                what: "patch build",
            },
            || {
                self.launch_build(pyramid, self.bases());
                Ok(())
            },
        )
    }

    fn prepare(
        &mut self,
        pyramid: &GpuPyramid<R>,
        positions: &PointsSoA,
        selected: Option<&[bool]>,
    ) -> Result<(), TrackerError> {
        guarded(
            GpuError::DeviceLost {
                what: "patch build",
            },
            || {
                self.accept(positions.len(), selected, pyramid.num_levels())?;
                for index in 0..positions.len() {
                    self.host.set(index, positions.get(index));
                }
                // `accept` has set `len`, which is what the runs are now strided by.
                self.upload_positions(positions, selected);
                Ok(())
            },
        )
    }

    fn len(&self) -> usize {
        self.len
    }

    fn position(&self, patch: usize) -> Vector2<f32> {
        self.host.get(patch)
    }
}

impl<P: Pattern, R: Runtime> std::fmt::Debug for GpuPatches<P, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPatches")
            .field("layout", &self.layout)
            .field("len", &self.len)
            .finish()
    }
}
