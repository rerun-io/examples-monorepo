//! The GPU [`PatchTracker`]: both passes and the recovered-distance test on the
//! device, one wait per call.

use crate::frontend::flow::FrontendError;
mod batch;

use cubecl::prelude::*;

use super::patches::GpuPatchSources;
use super::pyramid::GpuPyramid;
use kornia_staging_gpu::optical_flow::{FUSED_RUNS, FusedKltPlan, TrackPoints};
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;
use kornia_staging_imgproc::optical_flow::patch_tracker::{FlowTransforms, TrackerError};
use kornia_staging_slam::tracking::optical_flow::{PatchTracker, TrackPhase};

pub(super) struct PackedFused<P: Pattern, R: Runtime> {
    pub(super) plan: FusedKltPlan<P, R>,
    // Fixed device binding; only host readbacks trim to the current point count.
    pub(super) io: cubecl::server::Handle,
    pub(super) count: usize,
}

/// Fused forward/backward KLT on every supported GPU. Templates stay in
/// registers; only source positions and point results cross the host boundary.
/// A batch uses shared pyramid arenas when available, or per-camera bindings.
pub struct GpuPatchTracker<P: Pattern, R: Runtime> {
    launches: super::submission::LaunchList,
    client: ComputeClient<R>,
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    exit_step_px: Option<f32>,
    /// One combined result per lane; a batch writes `results[lane]` and
    /// [`PatchTracker::collect`] downloads every one of them at once.
    results: Vec<FusedKltPlan<P, R>>,
    /// The keypoint counts of the passes submitted since the last collect, in
    /// submission order, which is also their lane order.
    pending: Vec<usize>,
    batch: kornia_staging_slam::tracking::optical_flow::TrackBatch,
    packed_fused: PackedFused<P, R>,
    pattern: std::marker::PhantomData<P>,
}

impl<P: Pattern, R: Runtime> GpuPatchTracker<P, R> {
    /// Allocate point buffers and verify the device's subgroup operations.
    pub fn new(
        client: ComputeClient<R>,
        capacity: usize,
        num_levels: usize,
        max_iterations: usize,
        max_recovered_dist2: f32,
        lanes: usize,
        launches: super::submission::LaunchList,
    ) -> Result<Self, FrontendError> {
        let lanes = lanes.max(1);
        let make_plan = |cameras| {
            FusedKltPlan::new_batched(
                client.clone(),
                capacity,
                num_levels,
                max_iterations,
                max_recovered_dist2,
                cameras,
            )
        };
        let results = (0..lanes)
            .map(|_| make_plan(1))
            .collect::<Result<Vec<_>, _>>()?;
        let plan = make_plan(lanes)?;
        let packed_fused = PackedFused {
            io: plan.result_handle(capacity * lanes)?,
            plan,
            count: 0,
        };
        Ok(Self {
            launches,
            client,
            capacity,
            num_levels,
            max_iterations,
            max_recovered_dist2,
            exit_step_px: None,
            results,
            pending: Vec::with_capacity(lanes),
            batch: Default::default(),
            packed_fused,
            pattern: std::marker::PhantomData,
        })
    }
}

impl<P: Pattern, R: Runtime> PatchTracker for GpuPatchTracker<P, R> {
    fn set_klt_exit_step_px(&mut self, threshold: Option<f32>) -> Result<(), FrontendError> {
        for plan in &mut self.results {
            plan.set_exit_step_px(threshold)?;
        }
        self.packed_fused.plan.set_exit_step_px(threshold)?;
        self.exit_step_px = threshold;
        Ok(())
    }

    fn batch(&self) -> &kornia_staging_slam::tracking::optical_flow::TrackBatch {
        &self.batch
    }
    fn batch_mut(&mut self) -> &mut kornia_staging_slam::tracking::optical_flow::TrackBatch {
        &mut self.batch
    }

    type Error = FrontendError;
    type Pattern = P;
    type Pyramid = GpuPyramid<R>;
    type Patches = GpuPatchSources<P, R>;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn make_patches(&self) -> Result<GpuPatchSources<P, R>, FrontendError> {
        GpuPatchSources::new(self.capacity, self.num_levels)
    }

    fn submit_batch(
        &mut self,
        prev: &[GpuPyramid<R>],
        next: &[GpuPyramid<R>],
        phase: TrackPhase<'_>,
        patches: &mut GpuPatchSources<P, R>,
        slots: &mut [usize],
    ) -> Result<(), FrontendError> {
        use crate::pyramid::Pyramid;
        if !self.pending.is_empty() { return Err(TrackerError::PendingBatch.into()); }
        let inputs = phase.validate(
            self.capacity,
            self.num_levels,
            slots.len(),
            matches!(phase, TrackPhase::Matching { .. }).then_some(patches.num_levels()),
            |i| prev.get(i).map(Pyramid::num_levels),
            |i| next.get(i).map(Pyramid::num_levels),
        )?;
        if self.pending.len() + inputs.len() > self.results.len() {
            return Err(TrackerError::TooManyPasses {
                submitted: self.pending.len() + inputs.len(),
                lanes: self.results.len(),
            }
            .into());
        }
        if self.submit_packed(prev, next, inputs, slots)? {
            return Ok(());
        }
        kornia_staging_slam::tracking::optical_flow::submit_each(inputs, slots, |input| {
            use kornia_staging_slam::tracking::optical_flow::SourcePatches;
            if inputs.is_temporal() || input.destination == 1 {
                patches.build(&prev[input.source], input.positions, None)?;
            }
            self.submit_pass(
                &prev[input.source],
                &next[input.destination],
                patches,
                input.guesses,
            )
        })
    }

    /// Every submitted pass's result in one download, decoded in order.
    fn collect(&mut self) -> Result<(), FrontendError> {
        self.collect_with(Vec::new(), || Ok(())).map(|_| ())
    }

    fn discard(&mut self) {
        self.reset_batch();
    }
}

impl<P: Pattern, R: Runtime> GpuPatchTracker<P, R> {
    /// Collect tracker results and any frame-owned handles in one readback.
    pub(super) fn collect_with(
        &mut self,
        extra: Vec<cubecl::server::Handle>,
        after_copy: impl FnOnce() -> Result<(), kornia_staging_gpu::runtime::GpuError>,
    ) -> Result<Vec<cubecl::bytes::Bytes>, FrontendError> {
        let outcome = (|| {
            let mut reads = self.read_handles()?;
            let lanes = reads.len();
            reads.extend(extra);
            let mut bytes = if reads.is_empty() { Vec::new() } else {
                super::submission::read_with_lookahead(&self.client, &self.launches, reads, "the tracker result", after_copy)?
            };
            let extra = bytes.split_off(lanes);
            self.decode_results(&bytes)?;
            Ok(extra)
        })();
        self.reset_batch();
        outcome
    }

    pub(super) fn plan_like(&self, capacity: usize, lanes: usize) -> Result<FusedKltPlan<P, R>, FrontendError> {
        let mut plan = FusedKltPlan::new_batched(self.client.clone(), capacity, self.num_levels, self.max_iterations, self.max_recovered_dist2, lanes)?;
        plan.set_exit_step_px(self.exit_step_px)?;
        Ok(plan)
    }

    pub(super) fn packed_result(&self) -> (&cubecl::server::Handle, usize, usize) {
        (&self.packed_fused.io, self.packed_fused.count, self.pending.first().copied().unwrap_or(0))
    }

    pub(super) fn exit_step_px(&self) -> Option<f32> { self.exit_step_px }

    pub(super) fn publish_lane(&mut self, lane: usize, count: usize, bytes: &[u8]) -> Result<(), FrontendError> {
        self.batch.slot_mut(lane, self.capacity);
        let result = self.batch.result_mut(lane);
        result.reset(count);
        FusedKltPlan::<P, R>::decode(bytes, count, result)?;
        Ok(())
    }

    /// `trackPoints` on the device,
    /// launched into the next free lane and left there.
    fn submit_pass(
        &mut self,
        prev: &GpuPyramid<R>,
        next: &GpuPyramid<R>,
        patches: &GpuPatchSources<P, R>,
        transforms_in: &FlowTransforms,
    ) -> Result<usize, FrontendError> {
        let count = transforms_in.len();
        let lane = self.pending.len();
        if lane >= self.results.len() {
            return Err(TrackerError::TooManyPasses {
                submitted: lane + 1,
                lanes: self.results.len(),
            }
            .into());
        }
        self.batch.slot_mut(lane, self.capacity);
        let launch = self.results[lane].prepare(
            &[(prev, next)],
            None,
            &[TrackPoints {
                positions: &patches.positions,
                guesses: transforms_in,
                selected: Some(&patches.selected),
            }],
        )?;
        if count != 0 {
            self.launches.defer(super::submission::Launch::Klt(launch));
        }
        self.pending.push(count);
        Ok(lane)
    }

    fn reset_batch(&mut self) {
        self.launches.clear();
        self.pending.clear();
        self.packed_fused.count = 0;
    }

    /// The batch's one wait: every submitted lane's packed result downloaded
    /// together, then decoded into the batch slots in submission order.
    fn read_handles(&mut self) -> Result<Vec<cubecl::server::Handle>, FrontendError> {
        self.launches.flush(&self.client)?;

        for (lane, count) in self.pending.iter().enumerate() {
            self.batch.result_mut(lane).reset(*count);
        }
        // A pass that offered nothing launched nothing, so it has no buffer to
        // read; the download is over the lanes that do.
        if self.packed_fused.count != 0 {
            Ok(vec![
                self.packed_fused
                    .plan
                    .result_handle(self.packed_fused.count)?,
            ])
        } else {
            self.pending
                .iter()
                .enumerate()
                .filter(|(_, count)| **count != 0)
                .map(|(lane, count)| self.results[lane].result_handle(*count).map_err(Into::into))
                .collect()
        }
    }

    fn decode_results(
        &mut self,
        bytes: &[cubecl::bytes::Bytes],
    ) -> Result<(), FrontendError> {
        let packed_count = self.packed_fused.count;
        let mut read: usize = 0;
        let mut packed_offset = 0;
        for (lane, count) in self.pending.iter().enumerate() {
            let count: usize = *count;
            if count == 0 {
                self.batch.result_mut(lane).clear();
                continue;
            }
            let expected: usize = FUSED_RUNS * count * size_of::<f32>();
            // One buffer per non-empty lane, in the order they were asked for;
            // anything else is the runtime breaking its own contract.
            let Some(buffer) = bytes.get(if packed_count == 0 { read } else { 0 }) else {
                return Err(kornia_staging_gpu::runtime::GpuError::DeviceReadFailed {
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
                }
                .into());
            }
            FusedKltPlan::<P, R>::decode(
                &buffer[packed_offset..packed_offset + expected],
                count,
                self.batch.result_mut(lane),
            )?;
            if packed_count != 0 {
                packed_offset += expected;
            }
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

impl From<kornia_staging_gpu::optical_flow::TrackingError> for FrontendError {
    fn from(error: kornia_staging_gpu::optical_flow::TrackingError) -> Self {
        match error {
            kornia_staging_gpu::optical_flow::TrackingError::Input(error) => error.into(),
            kornia_staging_gpu::optical_flow::TrackingError::Gpu(error) => error.into(),
        }
    }
}
