//! Reusable memory for one forward/backward pass.
use super::*;
use crate::interpolation::U16View;
use crate::optical_flow::patch_se2::{
    patch_increment_rows, patch_residual_taps, se2_exp, Pattern, MAX_PATTERN_SIZE,
};
use crate::pyramid::PyramidPlanU16;
use nalgebra::{Matrix2, Vector3};
use rayon::ThreadPool;
use std::sync::Arc;
/// Reusable storage for a stateless forward/backward tracking pass.
/// No identities or frame history are retained between calls.
#[derive(Debug)]
pub struct PatchTrackerPlan<P: Pattern> {
    capacity: usize,
    steps: TrackingSteps,
    pool: Option<Arc<ThreadPool>>,
    /// Backward source patches, built from `next` at the forward result.
    backward: PatchSoA<P>,
    /// Forward result per input, before the backward check.
    forward: FlowTransforms,
    /// Whether the forward track succeeded, per input.
    forward_valid: Vec<bool>,
    /// The positions the backward patches are built at.
    backward_positions: PointsSoA,
}
impl<P: Pattern> PatchTrackerPlan<P> {
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
        validate_tracking_parameters(max_iterations, max_recovered_dist2)?;
        // First, so the smaller buffers below cannot be sized from a capacity the
        // patch storage would have refused.
        let backward: PatchSoA<P> = PatchSoA::new(capacity, num_levels)?.with_pool(pool.clone());
        let mut forward: FlowTransforms = FlowTransforms::with_capacity(capacity);
        forward.resize(capacity);
        let mut backward_positions: PointsSoA = PointsSoA::with_capacity(capacity);
        backward_positions.resize(capacity);
        Ok(Self {
            capacity,
            steps: TrackingSteps::new(num_levels, max_iterations, max_recovered_dist2)?,
            pool,
            backward,
            forward,
            forward_valid: vec![false; capacity],
            backward_positions,
        })
    }

    /// Allocate a stateless convenience pass with already validated settings.
    /// # Errors
    /// Rejects an invalid patch capacity.
    pub fn with_steps(
        capacity: usize,
        steps: TrackingSteps,
        pool: Option<Arc<ThreadPool>>,
    ) -> Result<Self, TrackerError> {
        let mut plan = Self::new(
            capacity,
            steps.num_levels,
            steps.max_iterations,
            steps.max_recovered_dist2,
            pool,
        )?;
        plan.steps.exit_step_px = steps.exit_step_px;
        Ok(plan)
    }

    /// Workers the tracking passes run on.
    pub fn threads(&self) -> usize {
        self.pool
            .as_ref()
            .map_or(1, |pool| pool.current_num_threads())
    }

    /// Validated per-point iteration settings, shared by independent passes.
    pub fn steps(&self) -> TrackingSteps {
        self.steps
    }
}

impl<P: Pattern> PatchTrackerPlan<P> {
    /// Set the optional positive finite convergence threshold.
    /// # Errors
    /// Rejects a nonfinite or nonpositive threshold.
    pub fn set_klt_exit_step_px(&mut self, threshold: Option<f32>) -> Result<(), TrackerError> {
        validate_exit_step(threshold)?;
        self.steps.exit_step_px = threshold;
        Ok(())
    }
    /// Track templates forward and backward, publishing surviving input indices.
    /// # Arguments
    /// * `prev`, `next` - Source and target image pyramids.
    /// * `patches` - Templates sampled in source-point order.
    /// * `transforms_in` - Initial destination warps.
    /// * `out` - Reusable output, overwritten for this pass.
    /// # Errors
    /// Rejects mismatched point counts, capacity, or pyramid depth before writes.
    pub fn track(
        &mut self,
        prev: &PyramidPlanU16,
        next: &PyramidPlanU16,
        patches: &PatchSoA<P>,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        let count: usize = transforms_in.len();
        check_track_inputs(
            count,
            patches.len(),
            patches.num_levels(),
            prev.levels().len(),
            next.levels().len(),
            self.capacity,
            self.steps.num_levels,
        )?;

        out.reset(count);
        // A previous batch may have left these at a different live length.
        self.forward.resize(count);
        self.forward_valid.resize(count, false);

        // ── forward: `trackPoint(pyr_1, pyr_2, transform_1, transform_2)`
        let steps = self.steps();
        {
            let target = steps.borrow_target_unchecked(next);
            self.forward.fill_with(
                self.pool.as_deref(),
                count,
                &mut self.forward_valid[..count],
                || (),
                |_, index| {
                    steps.forward_slot_unchecked(patches, index, &target, transforms_in.get(index))
                },
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
            let target = steps.borrow_target_unchecked(prev);
            out.fill_with(
                self.pool.as_deref(),
                || (),
                |_, index| {
                    steps.backward_slot_unchecked(
                        backward,
                        index,
                        &target,
                        patches.position(index),
                        transforms_in.translation(index),
                        (forward.get(index), forward_valid[index]),
                    )
                },
            );
        }

        out.finish();
        Ok(())
    }
}

/// Borrowed target levels for one tracking pass or worker chunk.
/// Pixel storage is resolved once when this value is built. Its lifetime keeps
/// the source pyramid immutable until all point operations finish; it allocates
/// no storage and holds no frame identity or cache policy.
#[derive(Clone, Copy)]
pub struct TrackingTarget<'a> {
    levels: [U16View<'a>; MAX_LEVELS],
}

/// Per-point KLT arithmetic shared by single-pass and cached batch tracking.
#[derive(Clone, Copy, Debug)]
pub struct TrackingSteps {
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    exit_step_px: Option<f32>,
}

impl TrackingSteps {
    /// Construct checked iteration settings without allocating patch storage.
    /// # Errors
    /// Rejects invalid depth, iteration budget or recovery distance.
    pub fn new(
        num_levels: usize,
        max_iterations: usize,
        max_recovered_dist2: f32,
    ) -> Result<Self, TrackerError> {
        limits::checked_patch_shape(0, num_levels, 0)?;
        validate_tracking_parameters(max_iterations, max_recovered_dist2)?;
        Ok(Self {
            num_levels,
            max_iterations,
            max_recovered_dist2,
            exit_step_px: None,
        })
    }
    /// Set the optional finite, positive convergence threshold.
    /// # Errors
    /// Rejects invalid thresholds without changing the configuration.
    pub fn set_exit_step(&mut self, threshold: Option<f32>) -> Result<(), TrackerError> {
        validate_exit_step(threshold)?;
        self.exit_step_px = threshold;
        Ok(())
    }

    /// Borrow configured target levels before processing points.
    ///
    /// # Arguments
    /// * `target` - Pyramid whose depth was checked at the pass boundary.
    /// # Preconditions
    /// The target must carry this configuration's level count. Use the returned
    /// value only with the configuration that created it.
    /// # Panics
    /// If the target has fewer than the configured levels.
    pub fn borrow_target_unchecked<'a>(&self, target: &'a PyramidPlanU16) -> TrackingTarget<'a> {
        let mut levels = [U16View {
            pixels: &[],
            width: 0,
            height: 0,
        }; MAX_LEVELS];
        for (view, image) in levels.iter_mut().zip(&target.levels()[..self.num_levels]) {
            *view = U16View::new(image);
        }
        TrackingTarget { levels }
    }

    /// Track one template toward its initial destination guess.
    /// # Arguments
    /// * `patches`, `index` - Source template store and point index.
    /// * `target` - Target pyramid with the configured levels.
    /// * `guess` - Initial warp in full-resolution pixels.
    /// # Preconditions
    /// The caller must validate the template index and both level counts once before the pass.
    /// # Panics
    /// If a referenced index or level is absent.
    pub fn forward_slot_unchecked<P: Pattern>(
        &self,
        patches: &PatchSoA<P>,
        index: usize,
        target: &TrackingTarget<'_>,
        guess: AffineCompact2f,
    ) -> ([f32; 6], bool) {
        let (width, height) = (
            target.levels[0].width as f32,
            target.levels[0].height as f32,
        );
        let position = guess.translation;
        if !guess.coefficients().iter().all(|v| v.is_finite())
            || position[0] < 0.0
            || position[1] < 0.0
            || position[0] >= width
            || position[1] >= height
        {
            return (AffineCompact2f::identity().coefficients(), false);
        }
        let (tracked, ok) = track_point::<P>(
            patches,
            index,
            target,
            guess,
            self.num_levels,
            self.max_iterations,
            self.exit_step_px,
        );
        (tracked.coefficients(), ok)
    }

    /// Track backward and reject a point whose recovery error exceeds the limit.
    /// # Arguments
    /// * `patches`, `index` - Backward template store and point index.
    /// * `target` - Source pyramid with the configured levels.
    /// * `source`, `guess` - Original source centre and initial destination centre.
    /// * `forward` - Forward warp and validity flag.
    /// # Preconditions
    /// The caller must validate the template index and both level counts once before the pass.
    /// # Panics
    /// If a referenced index or level is absent.
    pub fn backward_slot_unchecked<P: Pattern>(
        &self,
        patches: &PatchSoA<P>,
        index: usize,
        target: &TrackingTarget<'_>,
        source: [f32; 2],
        guess: [f32; 2],
        (forward, valid): (AffineCompact2f, bool),
    ) -> ([f32; 6], bool) {
        let kept = forward.coefficients();
        if !valid {
            return (kept, false);
        }
        // `off = source position - guess`; `t1_recovered += off`.
        let recovered_guess = [
            forward.translation[0] + (source[0] - guess[0]),
            forward.translation[1] + (source[1] - guess[1]),
        ];
        let (recovered, ok) = track_point::<P>(
            patches,
            index,
            target,
            AffineCompact2f {
                linear: forward.linear,
                translation: recovered_guess,
            },
            self.num_levels,
            self.max_iterations,
            self.exit_step_px,
        );
        if !ok {
            return (kept, false);
        }
        let dx = source[0] - recovered.translation[0];
        let dy = source[1] - recovered.translation[1];
        let dist2 = dx * dx + dy * dy;
        (kept, dist2 < self.max_recovered_dist2)
    }
}

/// `trackPoint`.
///
/// Coarse to fine, with the translation divided by `1 << level` on the way in and
/// multiplied back on the way out — both exact in `f32`, since
/// the scale is a power of two. The linear part starts at the identity
/// and is composed with the source's at the end, so the SE(2) rotation
/// is re-estimated from scratch rather than warm-started.
#[allow(clippy::too_many_arguments)]
fn track_point<P: Pattern>(
    patches: &PatchSoA<P>,
    index: usize,
    target: &TrackingTarget<'_>,
    guess: AffineCompact2f,
    num_levels: usize,
    max_iterations: usize,
    exit_step_px: Option<f32>,
) -> (AffineCompact2f, bool) {
    let mut transform: AffineCompact2f = AffineCompact2f {
        linear: Matrix2::identity().into(),
        translation: guess.translation,
    };
    let mut patch_valid: bool = true;

    for level in (0..num_levels).rev() {
        if !patch_valid {
            // Idle through remaining levels after failure to keep loop bounds fixed.
            continue;
        }
        let scale: f32 = (1u32 << level) as f32;
        transform.translation[0] /= scale;
        transform.translation[1] /= scale;

        patch_valid &= patches.valid(level, index);
        if patch_valid {
            let image = target.levels[level];
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

        transform.translation[0] *= scale;
        transform.translation[1] *= scale;
    }

    transform.linear = (Matrix2::from(guess.linear) * Matrix2::from(transform.linear)).into();
    if !transform.coefficients().iter().all(|v| v.is_finite()) {
        return (AffineCompact2f::identity(), false);
    }
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
    image: U16View<'_>,
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
            let increment: Vector3<f32> = -Vector3::from(patch_increment_rows::<P>(
                &patches.h_inv_jt[jacobian_offset..],
                element_stride,
                row_stride,
                &residual,
            ));

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
                *transform = transform.compose(&se2_exp(increment.as_ref()));
                patch_valid &= image.in_bounds(
                    transform.translation[0],
                    transform.translation[1],
                    FILTER_MARGIN,
                );
                // Keep the accepted update and all existing validity checks.
                // SIMD samples taps of this point, so no other point waits for it.
                if exit_step_px.is_some_and(|threshold| {
                    increment[0] * increment[0] + increment[1] * increment[1]
                        < threshold * threshold
                        && increment[2].abs() * 4.0 < threshold
                }) {
                    break;
                }
            }
        }
    }

    patch_valid
}
