//! The GPU [`PatchTracker`]: both passes and the recovered-distance test on the
//! device, one wait per call.

use cubecl::prelude::*;

use super::kernels::{self, PositionBases};
use super::patches::GpuPatches;
use super::pyramid::GpuPyramid;
use super::{GpuError, guarded};
use crate::frontend::patterns::Pattern;
use crate::frontend::tracker::{
    FlowResult, FlowTransforms, PatchTracker, SourcePatches, TrackerError, check_track_inputs,
};
use crate::pyramid::Pyramid;

/// Runs of `count` in a transform buffer: the six warp coefficients and the
/// validity flag.
const TRANSFORM_RUNS: usize = 7;

/// The CPU tracker's counterpart, with the two passes and the check between them
/// on the device.
///
/// One [`PatchTracker::track`] call is five launches and one wait: the forward
/// sweep, the backward pass's inputs, the backward patch build, the backward
/// sweep, and the recovered-distance test that combines them. Nothing crosses
/// back to the host in between — the backward pass reads the forward result
/// where it lies — so the only download is the packed result, seven `f32` per
/// keypoint.
///
/// The buffers a call needs are allocated once at construction and reused; the
/// one exception is the forward pass's inputs, which arrive from the host and so
/// must be a fresh `create_from_slice` every call.
pub struct GpuPatchTracker<P: Pattern, R: Runtime> {
    client: ComputeClient<R>,
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    /// Backward source patches, built from `next` at the forward result.
    backward_patches: GpuPatches<P, R>,
    /// The backward pass's inputs, then its results in place.
    backward: cubecl::server::Handle,
    /// The combined result the host downloads.
    result: cubecl::server::Handle,
    /// The staging buffer the transform inputs are uploaded from.
    staging: Vec<f32>,
    /// `source position - guess` per patch, the offset the backward guess adds.
    offset_x: Vec<f32>,
    /// The `y` half of the same offset.
    offset_y: Vec<f32>,
}

impl<P: Pattern, R: Runtime> GpuPatchTracker<P, R> {
    /// A tracker sized for `capacity` keypoints over `num_levels` levels.
    ///
    /// # Errors
    ///
    /// As [`GpuPatches::new`]: every buffer product is checked against the
    /// `usize` range and the two ceilings before anything is allocated, and the
    /// three allocations run inside the module's guard, so a device that dies
    /// under them is [`super::GpuError::DeviceLost`] and not an unwind through
    /// the caller (decision D32).
    pub fn new(
        client: ComputeClient<R>,
        capacity: usize,
        num_levels: usize,
        max_iterations: usize,
        max_recovered_dist2: f32,
    ) -> Result<Self, TrackerError> {
        guarded(
            GpuError::DeviceLost {
                what: "tracker allocation",
            },
            || {
                let backward_patches: GpuPatches<P, R> =
                    GpuPatches::new(client.clone(), capacity, num_levels)?;
                let transform_bytes: usize = TRANSFORM_RUNS * capacity * size_of::<f32>();
                Ok(Self {
                    capacity,
                    num_levels,
                    max_iterations,
                    max_recovered_dist2,
                    backward_patches,
                    backward: client.empty(transform_bytes),
                    result: client.empty(transform_bytes),
                    staging: vec![0.0; TRANSFORM_RUNS * capacity],
                    offset_x: vec![0.0; capacity],
                    offset_y: vec![0.0; capacity],
                    client,
                })
            },
        )
    }
}

impl<P: Pattern, R: Runtime> PatchTracker for GpuPatchTracker<P, R> {
    type Pattern = P;
    type Pyramid = GpuPyramid<R>;
    type Patches = GpuPatches<P, R>;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn make_patches(&self) -> Result<GpuPatches<P, R>, TrackerError> {
        guarded(
            GpuError::DeviceLost {
                what: "patch allocation",
            },
            || GpuPatches::new(self.client.clone(), self.capacity, self.num_levels),
        )
    }

    /// `trackPoints` (`frame_to_frame_optical_flow.h:294-375`) on the device.
    fn track(
        &mut self,
        prev: &GpuPyramid<R>,
        next: &GpuPyramid<R>,
        patches: &GpuPatches<P, R>,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        // The one call per frameset that waits on the device, and the one the
        // frontend makes with the GIL released: a lost device panics inside
        // CubeCL's own client, and the guard is what turns that into this
        // method's error rather than a `PanicException` in Python (decision D32).
        guarded(GpuError::DeviceLost { what: "tracker" }, || {
            let count: usize = transforms_in.len();
            check_track_inputs(
                count,
                patches.len(),
                patches.num_levels(),
                prev.num_levels(),
                next.num_levels(),
                self.capacity,
                self.num_levels,
            )?;

            out.reset(count);
            if count == 0 {
                out.finish(0);
                return Ok(());
            }

            // ── the forward pass's inputs: the source warps and the guesses.
            for index in 0..count {
                let coefficients: [f32; 6] = transforms_in.coefficients(index);
                for (run, value) in coefficients.into_iter().enumerate() {
                    self.staging[run * count + index] = value;
                }
            }
            self.staging[6 * count..TRANSFORM_RUNS * count].fill(0.0);
            // A local, not a field beside `backward` and `result`. Those two are
            // `client.empty` once at construction and reused; this one is a
            // host-to-device write, and `create_from_slice` is the only one CubeCL
            // 0.10 has, so a fresh buffer is allocated on every call whatever holds
            // it — and as a field it also meant one allocation at construction that
            // no path ever read (the first call replaced it, and a `count == 0` call
            // returns before touching it). Measured: this form costs about 0.05 ms
            // of the lane's 5.9 and an `Option` field about 0.14, both inside this
            // host's drift and above its 0.02 ms pair-to-pair floor.
            let forward: cubecl::server::Handle = self
                .client
                .create_from_slice(f32::as_bytes(&self.staging[..TRANSFORM_RUNS * count]));

            // `off = source position - guess` (`:339`), which the backward guess
            // adds back (`:357`). Both terms are on the host already, so the offset
            // rides along in the backward patch set's positions buffer instead of
            // costing a kernel.
            let guess_x: &[f32] = transforms_in.translations_x();
            let guess_y: &[f32] = transforms_in.translations_y();
            for index in 0..count {
                let source = patches.position(index);
                self.offset_x[index] = source.x - guess_x[index];
                self.offset_y[index] = source.y - guess_y[index];
            }

            // `shape.count` is `patches.len()`, which the guard above proved equal
            // to `count`.
            let shape = patches.shape();
            let forward_view = (&forward, TRANSFORM_RUNS * count);

            // ── forward: `trackPoint(pyr_1, pyr_2, transform_1, transform_2)` (`:349`).
            kernels::launch_klt::<R>(
                &self.client,
                next.buffers(),
                next.meta(),
                patches.store(),
                forward_view,
                shape,
                self.max_iterations,
                true,
            );

            // ── the backward source patches, from `next` at the forward result.
            self.backward_patches
                .accept(count, None, next.num_levels())?;
            self.backward_patches
                .upload_offsets(&self.offset_x[..count], &self.offset_y[..count]);
            kernels::launch_prepare_backward::<R>(
                &self.client,
                forward_view,
                self.backward_patches.position_buffer(),
                (&self.backward, TRANSFORM_RUNS * count),
                count,
                self.backward_patches.offset_bases(),
            );
            // The backward patches sit at the forward translations, which live in
            // runs 4 and 5 of the forward buffer, with its validity flag as the
            // selection mask — `patches.build(next, forward, Some(forward_valid))`
            // in the CPU tracker, without the round trip.
            self.backward_patches.launch_build_from(
                next,
                forward_view,
                PositionBases {
                    x: 4 * count,
                    y: 5 * count,
                    selected: 6 * count,
                },
            );

            // ── backward: `trackPoint(pyr_2, pyr_1, transform_2, recovered)` (`:359`).
            let backward_view = (&self.backward, TRANSFORM_RUNS * count);
            kernels::launch_klt::<R>(
                &self.client,
                prev.buffers(),
                prev.meta(),
                self.backward_patches.store(),
                backward_view,
                shape,
                self.max_iterations,
                false,
            );

            // ── `dist2 = (t1 - t1_recovered).squaredNorm() < max` (`:362`).
            kernels::launch_finish::<R>(
                &self.client,
                forward_view,
                backward_view,
                patches.position_buffer(),
                (&self.result, TRANSFORM_RUNS * count),
                count,
                patches.bases(),
                self.max_recovered_dist2,
            );

            // ── the one wait of the call.
            let bytes = self
                .client
                .read_one(self.result.clone())
                .map_err(|error| super::read_failed("the tracker result", &error))?;
            // `<`, where every sibling download checks `!=`: `self.result` is
            // allocated at `capacity` and read whole, while `expected` is sized by
            // `count`, so a full frame returns more bytes than this call reads
            // (28,672 against 700 on the tracker's own tolerance test). What an
            // incomplete runtime does — return fewer — is what this refuses.
            let expected: usize = TRANSFORM_RUNS * count * size_of::<f32>();
            if bytes.len() < expected {
                return Err(TrackerError::LengthMismatch {
                    first_name: "result bytes expected",
                    first: expected,
                    second_name: "returned",
                    second: bytes.len(),
                });
            }
            let values: &[f32] = f32::from_bytes(&bytes);
            let (valid, transforms) = out.parts_mut();
            let [m00, m01, m10, m11, tx, ty] = transforms.coefficients_mut();
            for index in 0..count {
                m00[index] = values[index];
                m01[index] = values[count + index];
                m10[index] = values[2 * count + index];
                m11[index] = values[3 * count + index];
                tx[index] = values[4 * count + index];
                ty[index] = values[5 * count + index];
                valid[index] = values[6 * count + index] != 0.0;
            }
            out.finish(count);
            Ok(())
        })
    }
}

impl<P: Pattern, R: Runtime> std::fmt::Debug for GpuPatchTracker<P, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPatchTracker")
            .field("capacity", &self.capacity)
            .field("num_levels", &self.num_levels)
            .field("max_iterations", &self.max_iterations)
            .field("max_recovered_dist2", &self.max_recovered_dist2)
            .finish()
    }
}
