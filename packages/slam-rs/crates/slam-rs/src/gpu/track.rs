//! The GPU [`PatchTracker`]: both passes and the recovered-distance test on the
//! device, one wait per call.

mod batch;

use cubecl::prelude::*;

use super::kernels::klt_fused::{
    CachedU32Upload, FUSED_RUNS, FusedParams, decode_point, encode_point, launch_fused,
};
use super::patches::GpuPatchSources;
use super::pyramid::GpuPyramid;
use super::{GpuError, guarded};
use crate::frontend::tracker::{
    FlowTransforms, PatchTracker, SourcePatches, TrackInput, TrackerError, check_track_inputs,
};
use crate::pyramid::Pyramid;
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;

pub(super) struct FusedLaunch {
    pub(super) buffers: [(cubecl::server::Handle, usize); 4],
    pub(super) meta: (cubecl::server::Handle, usize),
    pub(super) io: cubecl::server::Handle,
    pub(super) count: usize,
    pub(super) cameras: usize,
    pub(super) params: FusedParams,
}
impl FusedLaunch {
    fn new(
        buffers: [super::kernels::Buffer<'_>; 4],
        meta: &CachedU32Upload,
        io: &cubecl::server::Handle,
        count: usize,
        cameras: usize,
        params: FusedParams,
    ) -> Self {
        Self {
            buffers: buffers.map(|(handle, len)| (handle.clone(), len)),
            meta: (meta.handle.clone(), meta.len()),
            io: io.clone(),
            count,
            cameras,
            params,
        }
    }

    pub(super) fn run<R: Runtime>(self, client: &ComputeClient<R>) {
        launch_fused(
            client,
            self.buffers.each_ref().map(|(handle, len)| (handle, *len)),
            (&self.meta.0, self.meta.1),
            &self.io,
            self.count,
            self.cameras,
            self.params,
        );
    }
}

pub(super) struct PackedFused {
    pub(super) io: cubecl::server::Handle,
    meta: CachedU32Upload,
    pub(super) count: usize,
}

/// Fused forward/backward KLT on every supported GPU. Templates stay in
/// registers; only source positions and point results cross the host boundary.
/// A batch uses shared pyramid arenas when available, or per-camera bindings.
pub struct GpuPatchTracker<P: Pattern, R: Runtime> {
    launches: super::submission::LaunchList,
    pub(super) client: ComputeClient<R>,
    pub(super) capacity: usize,
    pub(super) num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    exit_step_px: Option<f32>,
    /// One combined result per lane; a batch writes `results[lane]` and
    /// [`PatchTracker::collect`] downloads every one of them at once.
    results: Vec<cubecl::server::Handle>,
    /// The keypoint counts of the passes submitted since the last collect, in
    /// submission order, which is also their lane order.
    pub(super) pending: Vec<usize>,
    pub(super) batch: crate::frontend::tracker::TrackBatch,
    /// The staging buffer the transform inputs are uploaded from.
    staging: Vec<f32>,
    pub(super) packed_fused: PackedFused,
    /// Source/target geometry, cached and uploaded only when a lane changes shape.
    fused_meta: Vec<CachedU32Upload>,
    geometry: Vec<u32>,
    pattern: std::marker::PhantomData<P>,
}

impl<P: Pattern, R: Runtime> GpuPatchTracker<P, R> {
    pub(super) fn fused_params(&self) -> FusedParams {
        FusedParams {
            levels: self.num_levels,
            taps: P::OFFSETS.len(),
            iterations: self.max_iterations,
            max_dist2: self.max_recovered_dist2,
            exit_step_px: self.exit_step_px,
        }
    }

    /// Allocate point buffers and verify the device's subgroup operations.
    pub fn new(
        client: ComputeClient<R>,
        capacity: usize,
        num_levels: usize,
        max_iterations: usize,
        max_recovered_dist2: f32,
        lanes: usize,
        launches: super::submission::LaunchList,
    ) -> Result<Self, TrackerError> {
        guarded(
            GpuError::DeviceLost {
                what: "tracker allocation",
            },
            || {
                kornia_staging_imgproc::optical_flow::patch_se2::validate_pattern::<P>()?;
                crate::frontend::tracker::checked_patch_shape(capacity, num_levels, P::SIZE)?;
                super::runtime::probe_subgroups(&client)?;

                let lanes = lanes.max(1);
                let transform_bytes: usize = FUSED_RUNS * capacity * size_of::<f32>();
                // Allocated here and never again: the pool has to be flat from
                // the first frameset, which a lane allocated on first use would
                // break (`the_whole_gpu_path_holds_the_pool_flat`).
                let results: Vec<cubecl::server::Handle> =
                    (0..lanes).map(|_| client.empty(transform_bytes)).collect();
                let meta_len = super::kernels::klt_fused::meta_len(num_levels, 1, P::OFFSETS.len());
                let fused_meta = (0..lanes)
                    .map(|_| CachedU32Upload::new(&client, meta_len))
                    .collect();
                let packed_fused = PackedFused {
                    io: client.empty(transform_bytes * lanes),
                    meta: CachedU32Upload::new(
                        &client,
                        super::kernels::klt_fused::meta_len(num_levels, lanes, P::OFFSETS.len()),
                    ),
                    count: 0,
                };
                Ok(Self {
                    launches,
                    capacity,
                    num_levels,
                    max_iterations,
                    max_recovered_dist2,
                    exit_step_px: None,
                    results,
                    pending: Vec::with_capacity(lanes),
                    batch: crate::frontend::tracker::TrackBatch::default(),
                    staging: vec![0.0; FUSED_RUNS * capacity],
                    packed_fused,
                    fused_meta,
                    geometry: Vec::with_capacity(meta_len),
                    client,
                    pattern: std::marker::PhantomData,
                })
            },
        )
    }
}

impl<P: Pattern, R: Runtime> PatchTracker for GpuPatchTracker<P, R> {
    fn set_klt_exit_step_px(&mut self, threshold: Option<f32>) {
        self.exit_step_px = threshold;
    }

    fn batch(&self) -> &crate::frontend::tracker::TrackBatch {
        &self.batch
    }
    fn batch_mut(&mut self) -> &mut crate::frontend::tracker::TrackBatch {
        &mut self.batch
    }

    type Pattern = P;
    type Pyramid = GpuPyramid<R>;
    type Patches = GpuPatchSources<P, R>;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn make_patches(&self) -> Result<GpuPatchSources<P, R>, TrackerError> {
        GpuPatchSources::new(self.capacity, self.num_levels)
    }

    /// `trackPoints` on the device,
    /// launched into the next free lane and left there.
    fn submit(
        &mut self,
        prev: &GpuPyramid<R>,
        next: &GpuPyramid<R>,
        patches: &GpuPatchSources<P, R>,
        transforms_in: &FlowTransforms,
    ) -> Result<usize, TrackerError> {
        // Launches only, but they still reach the device, and a lost one panics
        // inside CubeCL's own client: the guard is what turns that into this
        // method's error rather than a `PanicException` through the released GIL
        // (decision D32).
        guarded(GpuError::DeviceLost { what: "tracker" }, || {
            // The general path can sample separate allocations. Execute queued
            // shared-pyramid work before launching any of its readers.
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
            let lane: usize = self.pending.len();
            if lane >= self.results.len() {
                return Err(TrackerError::TooManyPasses {
                    submitted: lane + 1,
                    lanes: self.results.len(),
                });
            }
            self.batch.slot_mut(lane, self.capacity);
            self.pending.push(count);
            if count == 0 {
                return Ok(lane);
            }

            self.staging.clear();
            self.geometry.clear();
            prev.append_geometry(&mut self.geometry, None);
            next.append_geometry(&mut self.geometry, None);
            self.geometry
                .extend(P::OFFSETS.iter().flat_map(|tap| tap.map(f32::to_bits)));
            self.fused_meta[lane].update(&self.client, &self.geometry);
            stage_points(&mut self.staging, transforms_in, 0, |index| {
                (patches.selected(index), patches.position(index))
            });
            self.client.write(
                &self.results[lane],
                cubecl::bytes::Bytes::from_elems(self.staging.clone()),
            );
            let a = prev.buffers();
            let b = next.buffers();
            self.launches
                .defer(super::submission::Launch::Klt(FusedLaunch::new(
                    [a[0], a[1], b[0], b[1]],
                    &self.fused_meta[lane],
                    &self.results[lane],
                    count,
                    1,
                    self.fused_params(),
                )));
            Ok(lane)
        })
    }

    fn submit_batch(
        &mut self,
        prev: &[GpuPyramid<R>],
        next: &[GpuPyramid<R>],
        inputs: &mut [TrackInput],
        patches: &mut GpuPatchSources<P, R>,
        temporal: bool,
    ) -> Result<(), TrackerError> {
        if self.submit_packed(prev, next, inputs)? {
            return Ok(());
        }
        crate::frontend::tracker::submit_each(self, prev, next, inputs, patches, temporal)
    }

    /// Every submitted pass's result in one download, decoded in order.
    fn collect(&mut self) -> Result<(), TrackerError> {
        // The one call per frameset that waits on the device, and the one the
        // frontend makes with the GIL released: a lost device panics inside
        // CubeCL's own client, and the guard is what turns that into this
        // method's error rather than a `PanicException` in Python (decision D32).
        let outcome: Result<(), TrackerError> =
            guarded(GpuError::DeviceLost { what: "tracker" }, || {
                let reads = self.read_handles();
                let bytes = if reads.is_empty() {
                    Vec::new()
                } else {
                    super::submission::read_blocking(
                        &self.client,
                        &self.launches,
                        reads,
                        "the tracker result",
                    )?
                };
                self.decode_results(&bytes)
            });
        self.reset_batch();
        outcome
    }

    fn discard(&mut self) {
        self.reset_batch();
    }
}

impl<P: Pattern, R: Runtime> GpuPatchTracker<P, R> {
    fn reset_batch(&mut self) {
        self.launches.clear();
        self.pending.clear();
        self.packed_fused.count = 0;
    }

    /// The batch's one wait: every submitted lane's packed result downloaded
    /// together, then decoded into the batch slots in submission order.
    pub(super) fn read_handles(&mut self) -> Vec<cubecl::server::Handle> {
        self.launches.flush(&self.client);

        for (lane, count) in self.pending.iter().enumerate() {
            self.batch.slots[lane].reset(*count);
        }
        // A pass that offered nothing launched nothing, so it has no buffer to
        // read; the download is over the lanes that do.
        let packed_count = self.packed_fused.count;
        if packed_count != 0 {
            let packed = &self.packed_fused;
            let expected = (FUSED_RUNS * packed_count * size_of::<f32>()) as u64;
            vec![
                packed
                    .io
                    .clone()
                    .offset_end(packed.io.size_in_used() - expected),
            ]
        } else {
            self.pending
                .iter()
                .enumerate()
                .filter(|(_, count)| **count != 0)
                .map(|(lane, count)| {
                    let expected = (FUSED_RUNS * count * size_of::<f32>()) as u64;
                    let handle = &self.results[lane];
                    handle.clone().offset_end(handle.size_in_used() - expected)
                })
                .collect()
        }
    }

    pub(super) fn decode_results(
        &mut self,
        bytes: &[cubecl::bytes::Bytes],
    ) -> Result<(), TrackerError> {
        let packed_count = self.packed_fused.count;
        let mut read: usize = 0;
        let mut packed_offset = 0;
        for (lane, count) in self.pending.iter().enumerate() {
            let count: usize = *count;
            if count == 0 {
                self.batch.slots[lane].finish(0);
                continue;
            }
            let expected: usize = FUSED_RUNS * count * size_of::<f32>();
            // One buffer per non-empty lane, in the order they were asked for;
            // anything else is the runtime breaking its own contract.
            let Some(buffer) = bytes.get(if packed_count == 0 { read } else { 0 }) else {
                return Err(super::GpuError::DeviceReadFailed {
                    what: "a tracker lane's result",
                }
                .into());
            };
            read += 1;
            // The kernel packs by count; the unused capacity tail stays on the
            // device. Keep the short-read refusal before decoding any values.
            if buffer.len() < expected + packed_offset {
                return Err(TrackerError::LengthMismatch {
                    first_name: "result bytes expected",
                    first: expected,
                    second_name: "returned",
                    second: buffer.len(),
                });
            }
            let values: &[f32] = f32::from_bytes(&buffer[packed_offset..packed_offset + expected]);
            if packed_count != 0 {
                packed_offset += expected;
            }
            for (index, point) in values.as_chunks::<{ FUSED_RUNS }>().0.iter().enumerate() {
                decode_point(point, &mut self.batch.slots[lane], index);
            }
            self.batch.slots[lane].finish(count);
        }
        Ok(())
    }
}

impl<P: Pattern, R: Runtime> std::fmt::Debug for GpuPatchTracker<P, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPatchTracker")
            .field("capacity", &self.capacity)
            .field("num_levels", &self.num_levels)
            .field("max_iterations", &self.max_iterations)
            .field("max_recovered_dist2", &self.max_recovered_dist2)
            .field("lanes", &self.results.len())
            .finish()
    }
}

/// Append one camera's records without changing its selection flags or column order.
fn stage_points(
    staging: &mut Vec<f32>,
    guesses: &FlowTransforms,
    camera: usize,
    source: impl Fn(usize) -> (f32, nalgebra::Vector2<f32>),
) {
    for index in 0..guesses.len() {
        let (selected, position) = source(index);
        staging.extend(encode_point(
            guesses.coefficients(index),
            selected,
            position,
            camera,
        ));
    }
}
