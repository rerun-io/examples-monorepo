//! Stateful template caches and camera-order submission.
use super::*;
use crate::parallel::try_for_each_mut;
use kornia_staging_imgproc::optical_flow::patch_tracker::limits::check_patch_inputs;
use kornia_staging_imgproc::optical_flow::patch_tracker::TrackingSteps;
use kornia_staging_imgproc::pyramid::{BuildGeneration, PyramidPlanU16};
use rayon::ThreadPool;
use std::sync::Arc;
/// The CPU tracker: `trackPoints` with the same arithmetic and a fixed thread budget.
///
/// Template reuse follows [`PatchTracker::submit_batch`]'s caller rules: carry
/// committed pyramids forward and discard failed or abandoned batches. Cache
/// hits require the id, position bits, and source pyramid build generation.
#[derive(Debug)]
pub struct CpuPatchTracker<P: Pattern> {
    capacity: usize,
    num_levels: usize,
    steps: TrackingSteps,
    pool: Option<Arc<ThreadPool>>,
    /// Forward result per input, before the backward check.
    forward: FlowTransforms,
    /// Whether the forward track succeeded, per input.
    forward_valid: Vec<bool>,
    batch: TrackBatch,
    /// Temporal source templates and the last backward templates per camera.
    cameras: Vec<CpuCameraBuffers<P>>,
    temporal: Vec<CpuBackwardCache<P>>,
    /// Stereo backward templates for newly detected ids in the same frame.
    /// Those ids are disjoint from the temporal cache's surviving ids.
    stereo: Vec<CpuBackwardCache<P>>,
    offsets: Vec<usize>,
    batch_result: FlowResult,
}

#[derive(Debug)]
struct CpuBackwardCache<P: Pattern> {
    patches: PatchSoA<P>,
    generation: BuildGeneration,
    ids: Vec<u64>,
    positions: PointsSoA,
    valid: Vec<bool>,
}

impl<P: Pattern> CpuBackwardCache<P> {
    fn build_backward(
        &mut self,
        pyramid: &PyramidPlanU16,
        input: TrackLane<'_>,
        forward: &FlowTransforms,
        forward_valid: &[bool],
        start: usize,
    ) -> Result<(), TrackerError> {
        let count = input.guesses.len();
        self.positions.resize(count);
        for index in 0..count {
            self.positions
                .set(index, forward.translation(start + index));
        }
        self.patches
            .build(pyramid, &self.positions, Some(forward_valid))?;
        self.ids.clear();
        self.ids.extend_from_slice(input.ids);
        self.generation = pyramid.generation();
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum TemplateSource {
    Built,
    Temporal(usize),
    Stereo(usize),
}

#[derive(Debug)]
struct CpuCameraBuffers<P: Pattern> {
    source: PatchSoA<P>,
    build: Vec<bool>,
    sources: Vec<TemplateSource>,
}

impl<P: Pattern> CpuBackwardCache<P> {
    fn new(patches: PatchSoA<P>) -> Self {
        Self {
            patches,
            generation: BuildGeneration::NONE,
            ids: Vec::new(),
            positions: PointsSoA::default(),
            valid: Vec::new(),
        }
    }
}

impl<P: Pattern> CpuPatchTracker<P> {
    /// A tracker sized for `capacity` keypoints over `num_levels` pyramid levels.
    ///
    /// # Arguments
    /// * `capacity` - Maximum points in a camera pass, at most [`MAX_CAPACITY`].
    /// * `num_levels` - Positive level count, at most [`MAX_LEVELS`].
    /// * `max_iterations` - Positive per-level iteration budget.
    /// * `max_recovered_dist2` - Finite nonnegative squared backward-error limit.
    /// * `pool` - Caller-owned worker policy, shared with patch builds.
    ///
    /// # Errors
    ///
    /// Invalid iteration or recovery settings, or as [`PatchSoA::new`]: the capacity is checked against [`MAX_CAPACITY`] and
    /// every buffer product against the `usize` range before anything is
    /// allocated.
    pub fn new(
        capacity: usize,
        num_levels: usize,
        max_iterations: usize,
        max_recovered_dist2: f32,
        pool: Option<Arc<ThreadPool>>,
    ) -> Result<Self, TrackerError> {
        kornia_staging_imgproc::optical_flow::patch_tracker::limits::checked_patch_shape(
            capacity,
            num_levels,
            P::SIZE,
        )?;
        let steps = TrackingSteps::new(num_levels, max_iterations, max_recovered_dist2)?;
        // First, so the smaller buffers below cannot be sized from a capacity the
        // patch storage would have refused.
        let mut forward: FlowTransforms = FlowTransforms::with_capacity(capacity);
        forward.resize(capacity);
        Ok(Self {
            capacity,
            num_levels,
            steps,
            pool,
            forward,
            forward_valid: vec![false; capacity],
            batch: TrackBatch::default(),
            cameras: Vec::new(),
            temporal: Vec::new(),
            stereo: Vec::new(),
            offsets: Vec::new(),
            batch_result: FlowResult::default(),
        })
    }

    /// Workers the tracking passes run on.
    pub fn threads(&self) -> usize {
        self.pool
            .as_ref()
            .map_or(1, |pool| pool.current_num_threads())
    }
}

impl<P: Pattern> PatchTracker for CpuPatchTracker<P> {
    fn set_klt_exit_step_px(&mut self, threshold: Option<f32>) -> Result<(), TrackerError> {
        self.steps.set_exit_step(threshold)
    }

    fn batch(&self) -> &TrackBatch {
        &self.batch
    }
    fn batch_mut(&mut self) -> &mut TrackBatch {
        &mut self.batch
    }

    type Pattern = P;
    type Error = TrackerError;
    type Pyramid = PyramidPlanU16;
    type Patches = PatchSoA<P>;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn make_patches(&self) -> Result<PatchSoA<P>, TrackerError> {
        Ok(PatchSoA::new(self.capacity, self.num_levels)?.with_pool(self.pool.clone()))
    }

    fn discard(&mut self) {
        self.batch.submitted = 0;
        // A refused frame may have replaced some backward stores. Rebuild from
        // the committed pyramids on retry instead of reusing uncommitted data.
        for camera in self.temporal.iter_mut().chain(&mut self.stereo) {
            camera.ids.clear();
        }
    }

    fn submit_batch(
        &mut self,
        prev: &[PyramidPlanU16],
        next: &[PyramidPlanU16],
        phase: TrackPhase<'_>,
        patches: &mut PatchSoA<P>,
        slots: &mut [usize],
    ) -> Result<(), TrackerError> {
        if self.batch.submitted != 0 {
            return Err(TrackerError::PendingBatch);
        }
        let inputs = phase.validate(
            self.capacity,
            self.num_levels,
            slots.len(),
            matches!(phase, TrackPhase::Matching { .. }).then_some(patches.num_levels()),
            |i| prev.get(i).map(|p| p.levels().len()),
            |i| next.get(i).map(|p| p.levels().len()),
        )?;
        let temporal = inputs.is_temporal();
        if inputs.is_empty() {
            return Ok(());
        }
        if !temporal {
            let first = inputs.lane(0);
            check_patch_inputs(
                first.positions.len(),
                patches.capacity(),
                None,
                prev[first.source].levels().len(),
                patches.num_levels(),
            )?;
        }
        while self.cameras.len() < next.len() {
            self.cameras.push(CpuCameraBuffers {
                source: self.make_patches()?,
                build: Vec::new(),
                sources: Vec::new(),
            });
            self.temporal
                .push(CpuBackwardCache::new(self.make_patches()?));
            self.stereo
                .push(CpuBackwardCache::new(self.make_patches()?));
        }
        self.offsets.clear();
        self.offsets.push(0);
        let mut total = 0;
        for input in inputs.iter() {
            total += input.guesses.len();
            self.offsets.push(total);
        }
        self.forward.resize(total);
        self.forward_valid.resize(total, false);
        self.batch_result.reset(total);

        self.prepare_sources(prev, inputs, patches, temporal)?;

        let steps = self.steps;
        let offsets = &self.offsets;
        let cameras = &self.cameras;
        let source = |camera: usize, index: usize| {
            if temporal {
                match cameras[camera].sources[index] {
                    TemplateSource::Built => (&cameras[camera].source, index),
                    TemplateSource::Temporal(column) => (&self.temporal[camera].patches, column),
                    TemplateSource::Stereo(column) => (&self.stereo[camera].patches, column),
                }
            } else {
                (&*patches, index)
            }
        };
        // One point range across all cameras. Empty lanes have equal offsets;
        // partition_point skips them without scheduling empty work.
        let first_target = steps.borrow_target_unchecked(&next[inputs.lane(0).destination]);
        self.forward.fill_with(
            self.pool.as_deref(),
            total,
            &mut self.forward_valid[..total],
            || (0, first_target),
            |(target_camera, target), slot| {
                let camera = offsets.partition_point(|&start| start <= slot) - 1;
                let index = slot - offsets[camera];
                let input = inputs.lane(camera);
                if *target_camera != camera {
                    *target = steps.borrow_target_unchecked(&next[input.destination]);
                    *target_camera = camera;
                }
                let (patches, column) = source(camera, index);
                steps.forward_slot_unchecked(patches, column, target, input.guesses.get(index))
            },
        );

        if temporal {
            // Forward reads are complete; consume the one-frame stereo cache.
            for cache in &mut self.stereo {
                cache.ids.clear();
            }
        }

        // Build all backward stores before the backward KLT region starts.
        let forward = &self.forward;
        let forward_valid = &self.forward_valid;
        let caches = if temporal {
            &mut self.temporal[..inputs.len()]
        } else {
            &mut self.stereo[1..1 + inputs.len()]
        };
        try_for_each_mut(self.pool.as_deref(), caches, |camera, cache| {
            let input = inputs.lane(camera);
            let start = offsets[camera];
            cache.build_backward(
                &next[input.destination],
                input,
                forward,
                &forward_valid[start..start + input.guesses.len()],
                start,
            )
        })?;

        let first_target = steps.borrow_target_unchecked(&prev[inputs.lane(0).source]);
        self.batch_result.fill_with(
            self.pool.as_deref(),
            || (0, first_target),
            |(target_camera, target), slot| {
                let camera = offsets.partition_point(|&start| start <= slot) - 1;
                let index = slot - offsets[camera];
                let input = inputs.lane(camera);
                if *target_camera != camera {
                    *target = steps.borrow_target_unchecked(&prev[input.source]);
                    *target_camera = camera;
                }
                let backward = if temporal {
                    &self.temporal[camera].patches
                } else {
                    &self.stereo[input.destination].patches
                };
                steps.backward_slot_unchecked(
                    backward,
                    index,
                    target,
                    input.positions.get(index),
                    input.guesses.translation(index),
                    (forward.get(slot), forward_valid[slot]),
                )
            },
        );
        // Publish in camera order, independent of which worker finished first.
        for (camera, input) in inputs.iter().enumerate() {
            let (pass, result) = self.batch.submit_slot(self.capacity);
            slots[camera] = pass;
            result.reset(input.guesses.len());
            let cache = if temporal {
                &mut self.temporal[camera]
            } else {
                &mut self.stereo[input.destination]
            };
            cache.valid.resize(input.guesses.len(), false);
            for index in 0..input.guesses.len() {
                let slot = offsets[camera] + index;
                let valid = self.batch_result.is_valid(slot);
                cache.valid[index] = valid;
                result.set_track(index, valid, &forward.get(slot));
            }
            result.finish();
        }
        if !temporal {
            // Camera zero's detections become next frame's temporal sources.
            let input = inputs.lane(0);
            let cache = &mut self.stereo[input.source];
            std::mem::swap(&mut cache.patches, patches);
            cache.generation = prev[input.source].generation();
            cache.ids.clear();
            cache.ids.extend_from_slice(input.ids);
            cache.positions.clone_from(input.positions);
            cache.valid.clear();
            cache.valid.resize(input.positions.len(), true);
        }
        Ok(())
    }
}

impl<P: Pattern> CpuPatchTracker<P> {
    fn prepare_sources(
        &mut self,
        prev: &[PyramidPlanU16],
        inputs: ValidatedPhase<'_>,
        patches: &mut PatchSoA<P>,
        temporal: bool,
    ) -> Result<(), TrackerError> {
        // Forward sources: retain surviving columns in their cache stores, and
        // sample detections with no matching template. The cache key is id,
        // position bits, and the committed source pyramid's build generation.
        if temporal {
            let prepare = |camera: usize, buffers: &mut CpuCameraBuffers<P>| {
                let input = inputs.lane(camera);
                let temporal = &self.temporal[input.destination];
                let stereo = &self.stereo[input.destination];
                buffers.build.clear();
                buffers.sources.clear();
                for (slot, id) in input.ids.iter().enumerate() {
                    let position = input.positions.get(slot);
                    let cached = |cache: &CpuBackwardCache<P>| {
                        if cache.generation != prev[input.source].generation() {
                            return None;
                        }
                        cache.ids.binary_search(id).ok().filter(|&column| {
                            let old = cache.patches.position(column);
                            cache.valid[column]
                                && old[0].to_bits() == position[0].to_bits()
                                && old[1].to_bits() == position[1].to_bits()
                        })
                    };
                    if let Some(column) = cached(temporal) {
                        buffers.sources.push(TemplateSource::Temporal(column));
                        buffers.build.push(false);
                    } else if let Some(column) = cached(stereo) {
                        buffers.sources.push(TemplateSource::Stereo(column));
                        buffers.build.push(false);
                    } else {
                        buffers.sources.push(TemplateSource::Built);
                        buffers.build.push(true);
                    }
                }
                buffers
                    .source
                    .build(&prev[input.source], input.positions, Some(&buffers.build))
            };
            try_for_each_mut(
                self.pool.as_deref(),
                &mut self.cameras[..inputs.len()],
                prepare,
            )?;
        } else {
            patches.build(&prev[inputs.lane(0).source], inputs.lane(0).positions, None)?;
        }

        Ok(())
    }
}
