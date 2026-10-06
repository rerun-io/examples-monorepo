//! CPU KLT tracking and source-template caches.

use super::*;
use crate::frontend::parallel::WorkPool;
use crate::frontend::patch::{patch_increment_rows, patch_residual_taps};
use crate::frontend::patterns::MAX_PATTERN_SIZE;
use crate::frontend::se2::se2_exp;
use crate::pyramid::PyramidU16;
use kornia_image::Image;
use nalgebra::{Matrix2, Vector3};

/// The CPU tracker: `trackPoints` with the same arithmetic and a fixed thread budget.
///
/// Template reuse follows [`PatchTracker::submit_batch`]'s caller rules: carry
/// committed pyramids forward and discard failed or abandoned batches. Cache
/// hits require the id, position bits, and source pyramid build generation.
#[derive(Debug)]
pub struct CpuPatchTracker<P: Pattern> {
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    exit_step_px: Option<f32>,
    pool: WorkPool,
    /// Backward source patches, built from `next` at the forward result.
    backward: PatchSoA<P>,
    /// Forward result per input, before the backward check.
    forward: FlowTransforms,
    /// Whether the forward track succeeded, per input.
    forward_valid: Vec<bool>,
    /// The positions the backward patches are built at.
    backward_positions: PointsSoA,
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
    generation: crate::pyramid::PyramidGeneration,
    ids: Vec<crate::types::KeypointId>,
    positions: PointsSoA,
    valid: Vec<bool>,
}

impl<P: Pattern> CpuBackwardCache<P> {
    fn build_backward(
        &mut self,
        pyramid: &PyramidU16,
        input: &TrackInput,
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
        self.ids.clone_from(&input.ids);
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

impl<P: Pattern> CpuPatchTracker<P> {
    /// A tracker sized for `capacity` keypoints over `num_levels` pyramid levels.
    ///
    /// `max_iterations` is `optical_flow_max_iterations`,
    /// `max_recovered_dist2` is `optical_flow_max_recovered_dist2`, and `pool`
    /// carries the explicit thread budget (decision D31).
    ///
    /// # Errors
    ///
    /// As [`PatchSoA::new`]: the capacity is checked against [`MAX_CAPACITY`] and
    /// every buffer product against the `usize` range before anything is
    /// allocated.
    pub fn new(
        capacity: usize,
        num_levels: usize,
        max_iterations: usize,
        max_recovered_dist2: f32,
        pool: WorkPool,
    ) -> Result<Self, TrackerError> {
        // First, so the smaller buffers below cannot be sized from a capacity the
        // patch storage would have refused.
        let backward: PatchSoA<P> = PatchSoA::new(capacity, num_levels)?.with_pool(pool.clone());
        let mut forward: FlowTransforms = FlowTransforms::with_capacity(capacity);
        forward.resize(capacity);
        let mut backward_positions: PointsSoA = PointsSoA::with_capacity(capacity);
        backward_positions.resize(capacity);
        Ok(Self {
            capacity,
            num_levels,
            max_iterations,
            max_recovered_dist2,
            exit_step_px: None,
            pool,
            backward,
            forward,
            forward_valid: vec![false; capacity],
            backward_positions,
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
        self.pool.threads()
    }

    fn steps(&self) -> TrackingSteps {
        TrackingSteps {
            num_levels: self.num_levels,
            max_iterations: self.max_iterations,
            max_recovered_dist2: self.max_recovered_dist2,
            exit_step_px: self.exit_step_px,
        }
    }
}

impl<P: Pattern> PatchTracker for CpuPatchTracker<P> {
    fn set_klt_exit_step_px(&mut self, threshold: Option<f32>) {
        self.exit_step_px = threshold;
    }

    fn batch(&self) -> &TrackBatch {
        &self.batch
    }
    fn batch_mut(&mut self) -> &mut TrackBatch {
        &mut self.batch
    }

    type Pattern = P;
    type Pyramid = PyramidU16;
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

    fn submit(
        &mut self,
        prev: &Self::Pyramid,
        next: &Self::Pyramid,
        patches: &Self::Patches,
        transforms_in: &FlowTransforms,
    ) -> Result<usize, TrackerError> {
        let capacity = self.capacity();
        let batch = self.batch_mut();
        let pass = batch.submitted;
        let mut result = std::mem::take(batch.slot_mut(pass, capacity));
        let outcome = self.track_into(prev, next, patches, transforms_in, &mut result);
        self.batch_mut().slots[pass] = result;
        outcome?;
        self.batch_mut().submitted += 1;
        Ok(pass)
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
        prev: &[PyramidU16],
        next: &[PyramidU16],
        inputs: &mut [TrackInput],
        patches: &mut PatchSoA<P>,
        temporal: bool,
    ) -> Result<(), TrackerError> {
        if inputs.is_empty() {
            return Ok(());
        }
        self.validate_batch(prev, next, inputs, temporal)?;
        while self.cameras.len() < next.len() {
            self.cameras.push(CpuCameraBuffers {
                source: self.make_patches()?,
                build: Vec::new(),
                sources: Vec::new(),
            });
            self.temporal.push(CpuBackwardCache {
                patches: self.make_patches()?,
                generation: crate::pyramid::PyramidGeneration::default(),
                ids: Vec::new(),
                positions: PointsSoA::default(),
                valid: Vec::new(),
            });
            self.stereo.push(CpuBackwardCache {
                patches: self.make_patches()?,
                generation: crate::pyramid::PyramidGeneration::default(),
                ids: Vec::new(),
                positions: PointsSoA::default(),
                valid: Vec::new(),
            });
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

        let steps = self.steps();
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
        self.pool.for_each_warp(
            self.forward.coefficients_prefix_mut(total),
            &mut self.forward_valid[..total],
            |slot| {
                let camera = offsets.partition_point(|&start| start <= slot) - 1;
                let index = slot - offsets[camera];
                let input = &inputs[camera];
                let (patches, column) = source(camera, index);
                steps.forward_slot(
                    patches,
                    column,
                    &next[input.destination],
                    input.guesses.get(index),
                )
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
        self.pool.try_for_each_mut(caches, |camera, cache| {
            let input = &inputs[camera];
            let start = offsets[camera];
            cache.build_backward(
                &next[input.destination],
                input,
                forward,
                &forward_valid[start..start + input.guesses.len()],
                start,
            )
        })?;

        let (valid, transforms) = self.batch_result.parts_mut();
        self.pool.for_each_warp(
            transforms.coefficients_prefix_mut(total),
            &mut valid[..total],
            |slot| {
                let camera = offsets.partition_point(|&start| start <= slot) - 1;
                let index = slot - offsets[camera];
                let input = &inputs[camera];
                let backward = if temporal {
                    &self.temporal[camera].patches
                } else {
                    &self.stereo[input.destination].patches
                };
                steps.backward_slot(
                    backward,
                    index,
                    &prev[input.source],
                    input.positions.get(index),
                    input.guesses.translation(index),
                    (forward.get(slot), forward_valid[slot]),
                )
            },
        );
        // Publish in camera order, independent of which worker finished first.
        for (camera, input) in inputs.iter_mut().enumerate() {
            let (pass, result) = self.batch.submit_slot(self.capacity);
            input.result = pass;
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
            result.finish(input.guesses.len());
        }
        if !temporal {
            // Camera zero's detections become next frame's temporal sources.
            let input = &inputs[0];
            let cache = &mut self.stereo[input.source];
            std::mem::swap(&mut cache.patches, patches);
            cache.generation = prev[input.source].generation();
            cache.ids.clone_from(&input.ids);
            cache.positions.clone_from(&input.positions);
            cache.valid.clear();
            cache.valid.resize(input.positions.len(), true);
        }
        Ok(())
    }
}

impl<P: Pattern> CpuPatchTracker<P> {
    fn validate_batch(
        &self,
        prev: &[PyramidU16],
        next: &[PyramidU16],
        inputs: &[TrackInput],
        temporal: bool,
    ) -> Result<(), TrackerError> {
        // Validate the whole batch before modifying either cache. Source and
        // destination indices come from the driver, but are checked at this
        // public boundary just like the ordinary single-pass inputs.
        for (camera, input) in inputs.iter().enumerate() {
            let destination = camera + usize::from(!temporal);
            let source = if temporal { camera } else { 0 };
            for (actual, expected, name) in [
                (input.source, source, "source camera"),
                (input.destination, destination, "destination camera"),
            ] {
                if actual != expected {
                    return Err(TrackerError::LengthMismatch {
                        first_name: name,
                        first: actual,
                        second_name: "camera in batch order",
                        second: expected,
                    });
                }
            }
            for (index, len, name) in [
                (input.source, prev.len(), "source camera"),
                (input.destination, next.len(), "destination camera"),
            ] {
                if index >= len {
                    return Err(TrackerError::LengthMismatch {
                        first_name: name,
                        first: index,
                        second_name: "cameras",
                        second: len,
                    });
                }
            }
            check_track_inputs(
                input.guesses.len(),
                input.positions.len(),
                self.num_levels,
                prev[input.source].num_levels(),
                next[input.destination].num_levels(),
                self.capacity,
                self.num_levels,
            )?;
            if input.ids.len() != input.positions.len() {
                return Err(TrackerError::LengthMismatch {
                    first_name: "keypoint ids",
                    first: input.ids.len(),
                    second_name: "positions",
                    second: input.positions.len(),
                });
            }
            if !temporal {
                if input.positions.len() != inputs[0].positions.len() {
                    return Err(TrackerError::LengthMismatch {
                        first_name: "patches",
                        first: inputs[0].positions.len(),
                        second_name: "transforms",
                        second: input.guesses.len(),
                    });
                }
                debug_assert_eq!(input.ids, inputs[0].ids);
                debug_assert_eq!(input.positions, inputs[0].positions);
            }
        }
        Ok(())
    }

    fn prepare_sources(
        &mut self,
        prev: &[PyramidU16],
        inputs: &[TrackInput],
        patches: &mut PatchSoA<P>,
        temporal: bool,
    ) -> Result<(), TrackerError> {
        // Forward sources: retain surviving columns in their cache stores, and
        // sample detections with no matching template. The cache key is id,
        // position bits, and the committed source pyramid's build generation.
        if temporal {
            let prepare = |camera: usize, buffers: &mut CpuCameraBuffers<P>| {
                let input = &inputs[camera];
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
                                && old.x.to_bits() == position.x.to_bits()
                                && old.y.to_bits() == position.y.to_bits()
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
                    .build(&prev[input.source], &input.positions, Some(&buffers.build))
            };
            self.pool
                .try_for_each_mut(&mut self.cameras[..inputs.len()], prepare)?;
        } else {
            patches.build(&prev[inputs[0].source], &inputs[0].positions, None)?;
        }

        Ok(())
    }

    fn track_into(
        &mut self,
        prev: &PyramidU16,
        next: &PyramidU16,
        patches: &PatchSoA<P>,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
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
        // A previous batch may have left these at a different live length.
        self.forward.resize(count);
        self.forward_valid.resize(count, false);

        // ── forward: `trackPoint(pyr_1, pyr_2, transform_1, transform_2)`
        let steps = self.steps();
        {
            self.pool.for_each_warp(
                self.forward.coefficients_prefix_mut(count),
                &mut self.forward_valid[..count],
                |index| steps.forward_slot(patches, index, next, transforms_in.get(index)),
            );
        }

        // ── the backward source patches, from `next` at the forward result
        {
            // `build` wants `&mut self.backward` and `&self.backward_positions`
            // at once, which is what the destructure is for; `resize(count)` is
            // what `PatchSoA::build` reads as the patch count, and it comes
            // before the writes so the next call's may be longer.
            let Self {
                backward,
                backward_positions,
                forward,
                forward_valid,
                ..
            } = self;
            backward_positions.resize(count);
            for index in 0..count {
                backward_positions.set(index, forward.translation(index));
            }
            backward.build(next, backward_positions, Some(&forward_valid[..count]))?;
        }

        // ── backward: `trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered)`
        let backward: &PatchSoA<P> = &self.backward;
        let forward: &FlowTransforms = &self.forward;
        let forward_valid: &[bool] = &self.forward_valid[..count];
        {
            let (valid, transforms) = out.parts_mut();
            self.pool.for_each_warp(
                transforms.coefficients_prefix_mut(count),
                &mut valid[..count],
                |index| {
                    steps.backward_slot(
                        backward,
                        index,
                        prev,
                        patches.position(index),
                        transforms_in.translation(index),
                        (forward.get(index), forward_valid[index]),
                    )
                },
            );
        }

        out.finish(count);
        Ok(())
    }
}

/// Per-point KLT arithmetic shared by single-pass and cached batch tracking.
struct TrackingSteps {
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    exit_step_px: Option<f32>,
}

impl TrackingSteps {
    fn forward_slot<P: Pattern>(
        &self,
        patches: &PatchSoA<P>,
        index: usize,
        target: &PyramidU16,
        guess: AffineCompact2f,
    ) -> ([f32; 6], bool) {
        let (width, height) = level0_size(target);
        let position = guess.translation;
        if position.x < 0.0 || position.y < 0.0 || position.x >= width || position.y >= height {
            return (AffineCompact2f::identity().coefficients(), false);
        }
        let (tracked, ok) = track_point::<P>(
            patches,
            index,
            target,
            guess.linear,
            position,
            self.num_levels,
            self.max_iterations,
            self.exit_step_px,
        );
        (tracked.coefficients(), ok)
    }

    fn backward_slot<P: Pattern>(
        &self,
        patches: &PatchSoA<P>,
        index: usize,
        target: &PyramidU16,
        source: Vector2<f32>,
        guess: Vector2<f32>,
        (forward, valid): (AffineCompact2f, bool),
    ) -> ([f32; 6], bool) {
        let kept = forward.coefficients();
        if !valid {
            return (kept, false);
        }
        // `off = source position - guess`; `t1_recovered += off`.
        let offset = source - guess;
        let recovered_guess = forward.translation + offset;
        let (recovered, ok) = track_point::<P>(
            patches,
            index,
            target,
            forward.linear,
            recovered_guess,
            self.num_levels,
            self.max_iterations,
            self.exit_step_px,
        );
        if !ok {
            return (kept, false);
        }
        let dist2 = (source - recovered.translation).norm_squared();
        (kept, dist2 < self.max_recovered_dist2)
    }
}

/// Target camera's level-zero dimensions as floats, supporting mixed-resolution rigs.
fn level0_size(pyramid: &PyramidU16) -> (f32, f32) {
    // `check_track_inputs` refuses a pyramid with fewer levels than the patch
    // set, and the patch set always has at least one, so level 0 is there. The
    // `(0.0, 0.0)` would fail every keypoint's bounds test silently, so the
    // debug build says so instead (decision D32).
    debug_assert!(pyramid.level_size(0).is_some(), "no level 0 to size");
    match pyramid.level_size(0) {
        Some((width, height, _)) => (width as f32, height as f32),
        None => (0.0, 0.0),
    }
}

/// `trackPoint`.
///
/// Coarse to fine, with the translation divided by `1 << level` on the way in and
/// multiplied back on the way out — both exact in `f32`, since
/// the scale is a power of two. The linear part starts at the identity
/// and is composed with the source's at the end, so the SE(2) rotation
/// is re-estimated from scratch at every frame pair and never warm-started
/// (`papers-part2.md` §13 deviation D7).
#[allow(clippy::too_many_arguments)]
fn track_point<P: Pattern>(
    patches: &PatchSoA<P>,
    index: usize,
    target: &PyramidU16,
    old_linear: Matrix2<f32>,
    guess: Vector2<f32>,
    num_levels: usize,
    max_iterations: usize,
    exit_step_px: Option<f32>,
) -> (AffineCompact2f, bool) {
    let mut transform: AffineCompact2f = AffineCompact2f {
        linear: Matrix2::identity(),
        translation: guess,
    };
    let mut patch_valid: bool = true;

    for level in (0..num_levels).rev() {
        if !patch_valid {
            // Idle through remaining levels after failure to keep loop bounds fixed.
            continue;
        }
        let scale: f32 = (1u32 << level) as f32;
        transform.translation /= scale;

        patch_valid &= patches.valid(level, index);
        if patch_valid && let Some(image) = target.level(level) {
            patch_valid &= track_point_at_level::<P>(
                image,
                patches,
                level,
                index,
                &mut transform,
                max_iterations,
                exit_step_px,
            );
        }

        transform.translation *= scale;
    }

    transform.linear = old_linear * transform.linear;
    (transform, patch_valid)
}

/// `trackPointAtLevel`.
///
/// One Gauss-Newton step is: warp the pattern, take the mean-normalised residual,
/// `inc = -H_se2^-1 J_se2^T r`, reject a non-finite or huge increment
/// (because `SE2::exp` crashes on NaN), apply it on the right
/// (`transform *= SE2::exp(inc)`) and require the new centre to stay two
/// pixels inside the image.
fn track_point_at_level<P: Pattern>(
    image: &Image<u16, 1>,
    patches: &PatchSoA<P>,
    level: usize,
    index: usize,
    transform: &mut AffineCompact2f,
    max_iterations: usize,
    exit_step_px: Option<f32>,
) -> bool {
    let mut residual: [f32; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
    let data_offset: usize = patches.data_offset(level, index);
    let jacobian_offset: usize = patches.jacobian_offset(level, index);
    let element_stride: usize = 4;
    let row_stride: usize = P::SIZE * element_stride;
    let mut patch_valid: bool = true;

    for _ in 0..max_iterations {
        if !patch_valid {
            // `for (iteration = 0; patch_valid &&...)`, as a no-op.
            continue;
        }

        patch_valid &= patch_residual_taps::<P>(
            &patches.data[data_offset..],
            element_stride,
            image,
            transform,
            &mut residual,
        );

        if patch_valid {
            let increment: Vector3<f32> = -patch_increment_rows::<P>(
                &patches.h_inv_jt[jacobian_offset..],
                element_stride,
                row_stride,
                &residual,
            );

            patch_valid &= increment.iter().all(|value| value.is_finite());
            // Fold absolute coefficients from element zero, retaining a left-hand NaN.
            let mut infinity_norm: f32 = increment[0].abs();
            for row in 1..3 {
                let candidate: f32 = increment[row].abs();
                if infinity_norm < candidate {
                    infinity_norm = candidate;
                }
            }
            patch_valid &= infinity_norm < MAX_INCREMENT_INFINITY_NORM;

            if patch_valid {
                *transform = transform.compose(&se2_exp(&increment));
                patch_valid &= kornia_staging_imgproc::interpolation::in_bounds_u16(
                    image,
                    transform.translation.x,
                    transform.translation.y,
                    FILTER_MARGIN,
                );
                // Keep the accepted update and all existing validity checks.
                // SIMD samples taps of this point, so no other point waits for it.
                if let Some(threshold) = exit_step_px
                    && increment[0] * increment[0] + increment[1] * increment[1]
                        < threshold * threshold
                    && increment[2].abs() * 4.0 < threshold
                {
                    break;
                }
            }
        }
    }

    patch_valid
}
