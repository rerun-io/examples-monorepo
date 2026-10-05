//! CPU KLT tracking and source-template caches.

use super::*;
use crate::frontend::parallel::WorkPool;
use crate::frontend::patch::{patch_increment_rows, patch_residual_taps};
use crate::frontend::patterns::MAX_PATTERN_SIZE;
use crate::frontend::se2::se2_exp;
use crate::image::ImageU16;
use crate::pyramid::PyramidU16;
use nalgebra::{Matrix2, Vector3};

/// The CPU tracker: `trackPoints` with the same arithmetic and a fixed thread budget.
///
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
        })
    }

    /// Workers the tracking passes run on.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Stop KLT after applying an update below this threshold; None keeps fixed iterations.
    pub fn set_klt_exit_step_px(&mut self, threshold: Option<f32>) {
        self.exit_step_px = threshold;
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
    fn configure_klt_exit(&mut self, threshold: Option<f32>) -> Result<(), TrackerError> {
        self.set_klt_exit_step_px(threshold);
        Ok(())
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

    fn submit_prepared(
        &mut self,
        prev: &Self::Pyramid,
        next: &Self::Pyramid,
        patches: &Self::Patches,
        transforms_in: &FlowTransforms,
    ) -> Result<usize, TrackerError> {
        let capacity = self.capacity();
        let batch = self.batch_mut();
        let pass = batch.submitted;
        if pass == batch.slots.len() {
            batch.slots.push(FlowResult::with_capacity(capacity));
        }
        let mut result = std::mem::take(&mut batch.slots[pass]);
        let outcome = self.track_into(prev, next, patches, transforms_in, &mut result);
        self.batch_mut().slots[pass] = result;
        outcome?;
        self.batch_mut().submitted += 1;
        Ok(pass)
    }
}

impl<P: Pattern> CpuPatchTracker<P> {
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
    image: &ImageU16,
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
                patch_valid &= image.in_bounds(
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
