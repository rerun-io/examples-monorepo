//! The SE(2) inverse-compositional KLT tracker, ported from
//! `frame_to_frame_optical_flow.h:294-438`.
//!
//! Three C++ functions come across:
//!
//! * `trackPointAtLevel` (`:404-438`) — up to `optical_flow_max_iterations`
//!   Gauss-Newton steps on one pyramid level;
//! * `trackPoint` (`:377-402`) — the coarse-to-fine sweep, with the source patch
//!   rebuilt from the previous frame's pyramid at every level;
//! * `trackPoints` (`:294-375`) — the whole keypoint set forward, then backward,
//!   keeping only the tracks that come back to where they started.
//!
//! ## What the port moves, and why nothing moves numerically
//!
//! **The source patches are hoisted into a [`PatchSoA`].** `trackPoint` builds
//! `PatchT p(old_pyr.lvl(level), old_transform.translation() / scale)` inside its
//! level loop (`:388`). That patch depends only on the previous pyramid, the
//! source position and the level — never on anything the tracker computes — so
//! building all of them up front is the same arithmetic in a different order, and
//! it is the split a GPU wants: one kernel over (patch, level), then one over
//! patches. The forward patches come in as an argument; the backward ones are
//! built inside [`CpuPatchTracker`], because their positions are the forward
//! result.
//!
//! **The three passes are separated.** basalt does forward track, mask test and
//! backward track inside one per-point body. Here the forward pass runs over all
//! points, then the backward patches are built, then the backward pass runs.
//! Points are independent, so per-point values are identical; the mask test that
//! sat between the two passes moves to the caller, which drops the same points
//! one step later (see [`super::flow`]).
//!
//! **Loops have fixed bounds.** The C++ conditions its `for` on `patch_valid`
//! (`:383`, `:408`). Breaking out early and running to the end with a no-op give
//! the same numbers here, and only the second form ports to a GPU where the whole
//! warp runs the maximum count anyway (`cubecl-portability.md` §12.2). Invalid
//! points therefore idle instead of exiting.
//!
//! **The masks and the depth guess stay outside.** `trackPoints` reads
//! `masks1`/`masks2` and calls `calib.projectBetweenCams` (`:329`, `:335-342`),
//! which would drag the calibration into the tracker. The driver applies both and
//! hands over the already-offset guesses; the offset itself is recovered here as
//! `source position - guess`, exactly the `off` the C++ adds back before the
//! backward track (`:357`).

use nalgebra::{Matrix2, Vector2, Vector3};

use crate::frontend::parallel::WorkPool;
use crate::frontend::patch::{OpticalFlowPatch, patch_increment, patch_residual};
use crate::frontend::patterns::{MAX_PATTERN_SIZE, Pattern};
use crate::frontend::se2::{AffineCompact2f, se2_exp};
use crate::image::ImageU16;
use crate::pyramid::{Pyramid, PyramidU16};

/// The increment guard at `frame_to_frame_optical_flow.h:425`.
const MAX_INCREMENT_INFINITY_NORM: f32 = 1e6;

/// `const int filter_margin = 2` (`frame_to_frame_optical_flow.h:430`).
const FILTER_MARGIN: f32 = 2.0;

/// What the tracker can refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TrackerError {
    /// More keypoints were offered than the preallocated buffers hold.
    #[error("{offered} keypoints do not fit the tracker's capacity of {capacity}")]
    CapacityExceeded {
        /// Keypoints offered.
        offered: usize,
        /// Slots the tracker was built for.
        capacity: usize,
    },
    /// The patch set and the transform list disagree on how many keypoints there are.
    #[error("{patches} patches against {transforms} transforms")]
    LengthMismatch {
        /// Patches offered.
        patches: usize,
        /// Transforms offered.
        transforms: usize,
    },
    /// A pyramid does not carry the levels the tracker was built for.
    #[error("the pyramid holds {actual} levels, the tracker needs {expected}")]
    LevelMismatch {
        /// Levels the tracker runs over.
        expected: usize,
        /// Levels the pyramid holds.
        actual: usize,
    },
}

/// One camera's source patches for every pyramid level, in structure-of-arrays form.
///
/// The layout puts the **patch index fast-varying** in every array, so a GPU
/// thread per patch reads consecutive addresses (`cubecl-portability.md` §12.2).
/// Capacity is fixed at construction and the per-frame path never allocates.
#[derive(Debug, Clone)]
pub struct PatchSoA<P: Pattern> {
    capacity: usize,
    num_levels: usize,
    len: usize,
    /// Source position at level 0, one per patch.
    pos_x: Vec<f32>,
    /// Source position at level 0, one per patch.
    pos_y: Vec<f32>,
    /// `data[(level * P::SIZE + tap) * capacity + patch]`.
    data: Vec<f32>,
    /// `h_inv_jt[((level * 3 + row) * P::SIZE + tap) * capacity + patch]`.
    h_inv_jt: Vec<f32>,
    /// `valid[level * capacity + patch]`.
    valid: Vec<bool>,
    /// `mean[level * capacity + patch]`.
    mean: Vec<f32>,
    pattern: std::marker::PhantomData<P>,
}

impl<P: Pattern> PatchSoA<P> {
    /// Storage for `capacity` patches over `num_levels` pyramid levels.
    ///
    /// `num_levels` is `optical_flow_levels + 1`, matching
    /// [`crate::pyramid::Pyramid::num_levels`].
    pub fn new(capacity: usize, num_levels: usize) -> Self {
        Self {
            capacity,
            num_levels,
            len: 0,
            pos_x: vec![0.0; capacity],
            pos_y: vec![0.0; capacity],
            data: vec![0.0; num_levels * P::SIZE * capacity],
            h_inv_jt: vec![0.0; num_levels * 3 * P::SIZE * capacity],
            valid: vec![false; num_levels * capacity],
            mean: vec![0.0; num_levels * capacity],
            pattern: std::marker::PhantomData,
        }
    }

    /// Patches this set can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Pyramid levels each patch is built at.
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Patches currently filled.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no patch is filled.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The level-0 source position of one patch.
    pub fn position(&self, patch: usize) -> Vector2<f32> {
        Vector2::new(self.pos_x[patch], self.pos_y[patch])
    }

    /// Whether one patch at one level may be tracked (`patch.h:164-165`).
    pub fn valid(&self, level: usize, patch: usize) -> bool {
        self.valid[level * self.capacity + patch]
    }

    /// The mean of one patch at one level (`patch.h:129`).
    pub fn mean(&self, level: usize, patch: usize) -> f32 {
        self.mean[level * self.capacity + patch]
    }

    /// Build every patch at every level from `pyramid`.
    ///
    /// One patch per entry of `positions`, at `position / (1 << level)` — the
    /// `old_transform.translation() / scale` of `frame_to_frame_optical_flow.h:388`.
    /// `selected` may switch patches off; a patch that is switched off is marked
    /// invalid at every level and never sampled.
    ///
    /// This runs on the calling thread. It is a pure per-patch map, so moving it
    /// onto [`WorkPool`] later cannot change a value; it is left sequential in V0
    /// because the SoA's patch-fast-varying layout has no chunk-contiguous
    /// mutable split, and the tracking passes are the ones the thread budget is
    /// spent on.
    ///
    /// # Errors
    ///
    /// [`TrackerError::CapacityExceeded`] when there are more positions than
    /// slots, [`TrackerError::LevelMismatch`] when the pyramid is the wrong depth.
    pub fn build(
        &mut self,
        pyramid: &PyramidU16,
        positions: &[Vector2<f32>],
        selected: Option<&[bool]>,
    ) -> Result<(), TrackerError> {
        if positions.len() > self.capacity {
            return Err(TrackerError::CapacityExceeded {
                offered: positions.len(),
                capacity: self.capacity,
            });
        }
        if pyramid.num_levels() < self.num_levels {
            return Err(TrackerError::LevelMismatch {
                expected: self.num_levels,
                actual: pyramid.num_levels(),
            });
        }
        self.len = positions.len();

        let mut patch: OpticalFlowPatch<P> = OpticalFlowPatch::default();
        for level in 0..self.num_levels {
            let Some(image) = pyramid.level(level) else {
                return Err(TrackerError::LevelMismatch {
                    expected: self.num_levels,
                    actual: pyramid.num_levels(),
                });
            };
            // `const Scalar scale = 1 << level` (`frame_to_frame_optical_flow.h:384`).
            let scale: f32 = (1u32 << level) as f32;
            for (index, position) in positions.iter().enumerate() {
                let wanted: bool = selected.is_none_or(|flags| flags[index]);
                if !wanted {
                    self.valid[level * self.capacity + index] = false;
                    continue;
                }
                patch.set_from_image(image, position / scale);
                self.scatter(level, index, &patch);
            }
        }

        for (index, position) in positions.iter().enumerate() {
            self.pos_x[index] = position.x;
            self.pos_y[index] = position.y;
        }
        Ok(())
    }

    /// Write one built patch into the structure-of-arrays.
    fn scatter(&mut self, level: usize, patch_index: usize, patch: &OpticalFlowPatch<P>) {
        let capacity: usize = self.capacity;
        self.valid[level * capacity + patch_index] = patch.valid;
        self.mean[level * capacity + patch_index] = patch.mean;
        for tap in 0..P::SIZE {
            self.data[(level * P::SIZE + tap) * capacity + patch_index] = patch.data[tap];
            for row in 0..3 {
                self.h_inv_jt[((level * 3 + row) * P::SIZE + tap) * capacity + patch_index] =
                    patch.h_se2_inv_j_se2_t[row][tap];
            }
        }
    }

    /// Offset of tap 0 of one patch's `data` at one level; taps are `capacity` apart.
    #[inline]
    fn data_offset(&self, level: usize, patch: usize) -> usize {
        level * P::SIZE * self.capacity + patch
    }

    /// Offset of row 0, tap 0 of one patch's `H^-1 J^T`; rows are
    /// `P::SIZE * capacity` apart and taps `capacity` apart.
    #[inline]
    fn jacobian_offset(&self, level: usize, patch: usize) -> usize {
        level * 3 * P::SIZE * self.capacity + patch
    }
}

/// What one call to [`PatchTracker::track`] produced.
///
/// Dense per-input arrays plus a compacted list of the inputs that survived, all
/// preallocated: there is no map keyed by keypoint id anywhere on this path, and
/// no `push` inside the tracking loop (`cubecl-portability.md` §12.2, §12.3).
#[derive(Debug, Clone, Default)]
pub struct FlowResult {
    valid: Vec<bool>,
    transforms: Vec<AffineCompact2f>,
    tracked: Vec<u32>,
}

impl FlowResult {
    /// Room for `capacity` inputs.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            valid: vec![false; capacity],
            transforms: vec![AffineCompact2f::identity(); capacity],
            tracked: Vec::with_capacity(capacity),
        }
    }

    /// Whether input `index` was tracked.
    pub fn is_valid(&self, index: usize) -> bool {
        self.valid[index]
    }

    /// The tracked warp of input `index`; meaningless unless [`FlowResult::is_valid`].
    pub fn transform(&self, index: usize) -> AffineCompact2f {
        self.transforms[index]
    }

    /// The input indices that survived, ascending.
    pub fn tracked(&self) -> &[u32] {
        &self.tracked
    }

    /// How many inputs survived.
    pub fn len(&self) -> usize {
        self.tracked.len()
    }

    /// Whether nothing survived.
    pub fn is_empty(&self) -> bool {
        self.tracked.is_empty()
    }
}

/// The frontend's tracking stage: one call moves a whole camera's patches.
///
/// The trait takes the entire keypoint set, never one point, so the CPU
/// implementation here and a later CubeCL one can be swapped without touching the
/// driver (`cubecl-portability.md` §12.1).
pub trait PatchTracker {
    /// The sampling pattern this tracker was built for.
    type Pattern: Pattern;

    /// Track every patch of `patches` from `prev` into `next`.
    ///
    /// `transforms_in[i]` is basalt's `transform_2` before tracking: the linear
    /// part of the source keypoint and the translation the caller guesses. The
    /// source position itself is `patches.position(i)`.
    ///
    /// # Errors
    ///
    /// [`TrackerError`] when the inputs do not match the shape the tracker was
    /// built for.
    fn track(
        &mut self,
        prev: &PyramidU16,
        next: &PyramidU16,
        patches: &PatchSoA<Self::Pattern>,
        transforms_in: &[AffineCompact2f],
        out: &mut FlowResult,
    ) -> Result<(), TrackerError>;
}

/// The CPU tracker: `trackPoints` with the same arithmetic and a fixed thread budget.
#[derive(Debug)]
pub struct CpuPatchTracker<P: Pattern> {
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    pool: WorkPool,
    /// Backward source patches, built from `next` at the forward result.
    backward: PatchSoA<P>,
    /// Forward result per input, before the backward check.
    forward: Vec<AffineCompact2f>,
    /// Whether the forward track succeeded, per input.
    forward_valid: Vec<bool>,
    /// The positions the backward patches are built at.
    backward_positions: Vec<Vector2<f32>>,
}

impl<P: Pattern> CpuPatchTracker<P> {
    /// A tracker sized for `capacity` keypoints over `num_levels` pyramid levels.
    ///
    /// `max_iterations` is `optical_flow_max_iterations`,
    /// `max_recovered_dist2` is `optical_flow_max_recovered_dist2`, and `pool`
    /// carries the explicit thread budget (decision D31).
    pub fn new(
        capacity: usize,
        num_levels: usize,
        max_iterations: usize,
        max_recovered_dist2: f32,
        pool: WorkPool,
    ) -> Self {
        Self {
            capacity,
            num_levels,
            max_iterations,
            max_recovered_dist2,
            pool,
            backward: PatchSoA::new(capacity, num_levels),
            forward: vec![AffineCompact2f::identity(); capacity],
            forward_valid: vec![false; capacity],
            backward_positions: vec![Vector2::zeros(); capacity],
        }
    }

    /// Keypoints this tracker can carry in one call.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Workers the tracking passes run on.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }
}

impl<P: Pattern> PatchTracker for CpuPatchTracker<P> {
    type Pattern = P;

    fn track(
        &mut self,
        prev: &PyramidU16,
        next: &PyramidU16,
        patches: &PatchSoA<P>,
        transforms_in: &[AffineCompact2f],
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        let count: usize = transforms_in.len();
        if count != patches.len() {
            return Err(TrackerError::LengthMismatch {
                patches: patches.len(),
                transforms: count,
            });
        }
        if count > self.capacity {
            return Err(TrackerError::CapacityExceeded {
                offered: count,
                capacity: self.capacity,
            });
        }
        for pyramid in [prev, next] {
            if pyramid.num_levels() < self.num_levels {
                return Err(TrackerError::LevelMismatch {
                    expected: self.num_levels,
                    actual: pyramid.num_levels(),
                });
            }
        }

        if out.valid.len() < count {
            out.valid.resize(count, false);
        }
        if out.transforms.len() < count {
            out.transforms.resize(count, AffineCompact2f::identity());
        }
        out.valid[..count].fill(false);
        out.tracked.clear();

        // ── forward: `trackPoint(pyr_1, pyr_2, transform_1, transform_2)` (`:349`)
        let max_iterations: usize = self.max_iterations;
        let num_levels: usize = self.num_levels;
        let (target_width, target_height): (f32, f32) = level0_size(next);
        self.pool.for_each_indexed(
            &mut self.forward[..count],
            &mut self.forward_valid[..count],
            |index, transform, valid| {
                let guess: Vector2<f32> = transforms_in[index].translation;
                // `valid = t2(0) >= 0 && t2(1) >= 0 && t2(0) < w && t2(1) < h` (`:346`).
                if guess.x < 0.0
                    || guess.y < 0.0
                    || guess.x >= target_width
                    || guess.y >= target_height
                {
                    *valid = false;
                    return;
                }
                let (tracked, ok) = track_point::<P>(
                    patches,
                    index,
                    next,
                    transforms_in[index].linear,
                    guess,
                    num_levels,
                    max_iterations,
                );
                *transform = tracked;
                *valid = ok;
            },
        );

        // ── the backward source patches, from `next` at the forward result
        for index in 0..count {
            self.backward_positions[index] = self.forward[index].translation;
        }
        self.backward.build(
            next,
            &self.backward_positions[..count],
            Some(&self.forward_valid[..count]),
        )?;

        // ── backward: `trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered)` (`:359`)
        let backward: &PatchSoA<P> = &self.backward;
        let forward: &[AffineCompact2f] = &self.forward[..count];
        let forward_valid: &[bool] = &self.forward_valid[..count];
        let max_recovered_dist2: f32 = self.max_recovered_dist2;
        self.pool.for_each_indexed(
            &mut out.valid[..count],
            &mut out.transforms[..count],
            |index, valid, transform| {
                if !forward_valid[index] {
                    *valid = false;
                    return;
                }
                // `off = t2 - t2_guess` with `t2 == t1` at that point (`:339`),
                // so `off == source position - guess`; `t1_recovered += off` (`:357`).
                let source: Vector2<f32> = patches.position(index);
                let offset: Vector2<f32> = source - transforms_in[index].translation;
                let recovered_guess: Vector2<f32> = forward[index].translation + offset;
                let (recovered, ok) = track_point::<P>(
                    backward,
                    index,
                    prev,
                    forward[index].linear,
                    recovered_guess,
                    num_levels,
                    max_iterations,
                );
                if !ok {
                    *valid = false;
                    return;
                }
                // `dist2 = (t1 - t1_recovered).squaredNorm()` (`:362`).
                let dist2: f32 = (source - recovered.translation).norm_squared();
                *valid = dist2 < max_recovered_dist2;
                *transform = forward[index];
            },
        );

        for index in 0..count {
            if out.valid[index] {
                out.tracked.push(index as u32);
            }
        }
        Ok(())
    }
}

/// Level 0's `(width, height)` as floats, standing in for basalt's `w`, `h`.
///
/// basalt reads `calib.resolution.at(0)` for every camera
/// (`frame_to_frame_optical_flow.h:108-109`, trap 16); the port reads the target
/// camera's own level-0 size, which is the same number for a rig whose cameras
/// share a resolution and the right one for msd-g2, whose cameras do not
/// (decision D30).
fn level0_size(pyramid: &PyramidU16) -> (f32, f32) {
    match pyramid.level_size(0) {
        Some((width, height, _)) => (width as f32, height as f32),
        None => (0.0, 0.0),
    }
}

/// `trackPoint` (`frame_to_frame_optical_flow.h:377-402`).
///
/// Coarse to fine, with the translation divided by `1 << level` on the way in and
/// multiplied back on the way out (`:386`, `:396`) — both exact in `f32`, since
/// the scale is a power of two. The linear part starts at the identity (`:381`)
/// and is composed with the source's at the end (`:399`), so the SE(2) rotation
/// is re-estimated from scratch at every frame pair and never warm-started
/// (`papers-part2.md` §13 deviation D7).
fn track_point<P: Pattern>(
    patches: &PatchSoA<P>,
    index: usize,
    target: &PyramidU16,
    old_linear: Matrix2<f32>,
    guess: Vector2<f32>,
    num_levels: usize,
    max_iterations: usize,
) -> (AffineCompact2f, bool) {
    let mut transform: AffineCompact2f = AffineCompact2f {
        linear: Matrix2::identity(),
        translation: guess,
    };
    let mut patch_valid: bool = true;

    for level in (0..num_levels).rev() {
        if !patch_valid {
            // The C++ `for` exits here (`:383`); running the remaining levels as
            // a no-op keeps the bound fixed and the numbers identical.
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
            );
        }

        transform.translation *= scale;
    }

    transform.linear = old_linear * transform.linear;
    (transform, patch_valid)
}

/// `trackPointAtLevel` (`frame_to_frame_optical_flow.h:404-438`).
///
/// One Gauss-Newton step is: warp the pattern, take the mean-normalised residual,
/// `inc = -H_se2^-1 J_se2^T r`, reject a non-finite or huge increment
/// (`:422-425`, because `SE2::exp` crashes on NaN), apply it on the right
/// (`transform *= SE2::exp(inc)`, `:428`) and require the new centre to stay two
/// pixels inside the image (`:430-432`).
fn track_point_at_level<P: Pattern>(
    image: &ImageU16,
    patches: &PatchSoA<P>,
    level: usize,
    index: usize,
    transform: &mut AffineCompact2f,
    max_iterations: usize,
) -> bool {
    let mut residual: [f32; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
    let data_offset: usize = patches.data_offset(level, index);
    let jacobian_offset: usize = patches.jacobian_offset(level, index);
    let capacity: usize = patches.capacity;
    let row_stride: usize = P::SIZE * capacity;
    let mut patch_valid: bool = true;

    for _ in 0..max_iterations {
        if !patch_valid {
            // `for (iteration = 0; patch_valid && ...)` (`:408`), as a no-op.
            continue;
        }

        patch_valid &= patch_residual::<P, ImageU16>(
            &patches.data[data_offset..],
            capacity,
            image,
            transform,
            &mut residual,
        );

        if patch_valid {
            let increment: Vector3<f32> = -patch_increment::<P>(
                &patches.h_inv_jt[jacobian_offset..],
                capacity,
                row_stride,
                &residual,
            );

            patch_valid &= increment.iter().all(|value| value.is_finite());
            // `inc.lpNorm<Eigen::Infinity>()` is `cwiseAbs().maxCoeff()`, whose
            // reduction is `(a < b) ? b : a` from element 0 — spelled out so a
            // NaN takes the same branch it takes in C++.
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
            }
        }
    }

    patch_valid
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::frontend::patterns::Pattern51;
    use crate::pyramid::{CpuPyramidBuilder, PyramidBuilder};
    use proptest::prelude::*;

    /// A band-limited texture: twelve plane waves with wavelengths between 16
    /// and 56 pixels, in fixed pseudo-random directions and phases.
    ///
    /// Band-limited matters twice over. Below the Nyquist of the finest pyramid
    /// level the `[1,4,6,4,1]` subsample does not alias, so the coarse levels
    /// really do carry the same shift; and away from the sampling limit bilinear
    /// interpolation reconstructs the field closely, so the residual's fixed
    /// point sits near the true shift rather than a fraction of a pixel off it.
    /// Twelve components in different directions also keep every patch's `H_se2`
    /// well conditioned: a single wave, or a field that is locally almost affine,
    /// is the aperture problem and no tracker recovers a shift from it.
    fn texture(x: f64, y: f64) -> f64 {
        // (wavelength, direction in turns, phase in turns)
        const WAVES: [(f64, f64, f64); 16] = [
            (22.0000, 0.000000, 0.000000),
            (23.5218, 0.381966, 0.618034),
            (25.1489, 0.763932, 0.236068),
            (26.8886, 0.145898, 0.854102),
            (28.7486, 0.527864, 0.472136),
            (30.7373, 0.909830, 0.090170),
            (32.8635, 0.291796, 0.708204),
            (35.1368, 0.673762, 0.326238),
            (37.5674, 0.055728, 0.944272),
            (40.1661, 0.437694, 0.562306),
            (42.9446, 0.819660, 0.180340),
            (45.9153, 0.201626, 0.798374),
            (49.0914, 0.583592, 0.416408),
            (52.4873, 0.965558, 0.034442),
            (56.1181, 0.347524, 0.652476),
            (60.0000, 0.729490, 0.270510),
        ];
        let tau: f64 = std::f64::consts::TAU;
        let mut sum: f64 = 0.0;
        for (wavelength, direction, phase) in WAVES {
            let angle: f64 = tau * direction;
            let projection: f64 = x * angle.cos() + y * angle.sin();
            sum += (tau * (projection / wavelength + phase)).sin();
        }
        sum / WAVES.len() as f64
    }

    /// A textured frame, shifted by `(dx, dy)`: the same continuous field
    /// resampled at `(x - dx, y - dy)`, so the shift is exact by construction.
    fn shifted_image(width: usize, height: usize, dx: f32, dy: f32) -> ImageU16 {
        let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
        for y in 0..height {
            for x in 0..width {
                let fx: f64 = f64::from(x as f32 - dx);
                let fy: f64 = f64::from(y as f32 - dy);
                let value: f64 = 32_000.0 + 28_000.0 * texture(fx, fy);
                image.set(x, y, value as u16);
            }
        }
        image
    }

    fn pyramid_of(image: &ImageU16, levels: usize) -> PyramidU16 {
        let mut pyramid: PyramidU16 =
            PyramidU16::with_capacity(image.width(), image.height(), levels).unwrap();
        CpuPyramidBuilder::new().build(image, &mut pyramid).unwrap();
        pyramid
    }

    struct Fixture {
        prev: PyramidU16,
        next: PyramidU16,
        patches: PatchSoA<Pattern51>,
        transforms: Vec<AffineCompact2f>,
        positions: Vec<Vector2<f32>>,
    }

    fn fixture(dx: f32, dy: f32, levels: usize) -> Fixture {
        let base: ImageU16 = shifted_image(160, 160, 0.0, 0.0);
        let moved: ImageU16 = shifted_image(160, 160, dx, dy);
        let prev: PyramidU16 = pyramid_of(&base, levels);
        let next: PyramidU16 = pyramid_of(&moved, levels);

        let mut positions: Vec<Vector2<f32>> = Vec::new();
        for y in (40..120).step_by(16) {
            for x in (40..120).step_by(16) {
                positions.push(Vector2::new(x as f32, y as f32));
            }
        }
        let transforms: Vec<AffineCompact2f> = positions
            .iter()
            .map(|position| AffineCompact2f::at(*position))
            .collect();

        let mut patches: PatchSoA<Pattern51> = PatchSoA::new(positions.len(), levels + 1);
        patches.build(&prev, &positions, None).unwrap();

        Fixture {
            prev,
            next,
            patches,
            transforms,
            positions,
        }
    }

    fn tracker(capacity: usize, levels: usize, threads: usize) -> CpuPatchTracker<Pattern51> {
        CpuPatchTracker::new(
            capacity,
            levels + 1,
            5,
            0.04,
            WorkPool::new(threads).unwrap(),
        )
    }

    #[test]
    fn an_integer_shift_is_recovered() {
        let levels: usize = 3;
        let scene: Fixture = fixture(2.0, -1.0, levels);
        let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut out,
            )
            .unwrap();

        assert!(
            out.len() >= scene.positions.len() / 2,
            "tracked {}",
            out.len()
        );
        for index in out.tracked() {
            let index: usize = *index as usize;
            let moved: Vector2<f32> = out.transform(index).translation - scene.positions[index];
            // An integer shift moves the samples themselves, so the two
            // bilinear reconstructions are exact translates of each other and
            // the residual's fixed point is the true shift.
            assert!(
                (moved.x - 2.0).abs() < 0.01 && (moved.y + 1.0).abs() < 0.01,
                "patch {index} moved by {moved:?}, expected (2, -1)"
            );
        }
    }

    /// A sub-pixel shift, up to the pattern's own radius.
    ///
    /// The tolerance is not the tracker's convergence — it converges to five
    /// decimal places in three iterations — but the **bias of the fixed point
    /// itself**. `interp` reconstructs the image bilinearly and `interpGrad`
    /// differentiates that reconstruction by central differences
    /// (`image.h:396-469`), so for a shift that is not a whole number of samples
    /// the residual vanishes not at the true shift but a little beside it.
    ///
    /// The size of that displacement depends only on the **fractional** part of
    /// the shift, not on its magnitude: on this texture an exactly integer shift
    /// is recovered to `0.0000` px, a shift of 0.02 px to 0.0004, and a shift of
    /// half a pixel to 0.035 on the median patch and 0.13 on the worst — the same
    /// numbers whether the shift is 0.5 or 3.5 pixels. Shortening the texture's
    /// wavelengths raises the floor and lengthening them makes the patches
    /// ill-conditioned instead; basalt's C++ has the same property, because this
    /// is its arithmetic. The gate is therefore the median, with a cap on the tail.
    fn sub_pixel_shift_error(dx: f32, dy: f32) -> (f32, f32, usize, usize) {
        let levels: usize = 3;
        let scene: Fixture = fixture(dx, dy, levels);
        let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut out,
            )
            .unwrap();

        let mut errors: Vec<f32> = Vec::new();
        for index in out.tracked() {
            let index: usize = *index as usize;
            let moved: Vector2<f32> = out.transform(index).translation - scene.positions[index];
            errors.push((moved.x - dx).abs().max((moved.y - dy).abs()));
        }
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median: f32 = errors
            .get(errors.len() / 2)
            .copied()
            .unwrap_or(f32::INFINITY);
        let worst: f32 = errors.last().copied().unwrap_or(f32::INFINITY);
        (median, worst, out.len(), scene.positions.len())
    }

    #[test]
    fn a_sub_pixel_shift_is_recovered() {
        let (median, worst, tracked, total) = sub_pixel_shift_error(0.6, 1.4);
        assert_eq!(tracked, total);
        assert!(median < 0.05, "median error {median}");
        assert!(worst < 0.2, "worst error {worst}");
    }

    /// The forward-backward gate (`frame_to_frame_optical_flow.h:362-364`) is
    /// what rejects a track onto an unrelated image.
    #[test]
    fn a_mismatched_pair_is_rejected() {
        let levels: usize = 3;
        let scene: Fixture = fixture(0.0, 0.0, levels);
        // A different texture entirely, not a shift of the first.
        let mut other: ImageU16 = ImageU16::zeros(160, 160).unwrap();
        for y in 0..160 {
            for x in 0..160 {
                let value: f64 = 25_000.0
                    + 9_000.0 * ((x as f64) * 0.61).cos()
                    + 6_000.0 * ((y as f64) * 0.47).sin();
                other.set(x, y, value as u16);
            }
        }
        let unrelated: PyramidU16 = pyramid_of(&other, levels);

        let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker
            .track(
                &scene.prev,
                &unrelated,
                &scene.patches,
                &scene.transforms,
                &mut out,
            )
            .unwrap();
        assert!(
            out.len() * 4 < scene.positions.len(),
            "{} of {} tracks survived an unrelated image",
            out.len(),
            scene.positions.len()
        );
    }

    #[test]
    fn one_thread_and_four_threads_agree_exactly() {
        let levels: usize = 3;
        let scene: Fixture = fixture(1.3, -0.7, levels);

        let mut single: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker(scene.positions.len(), levels, 1)
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut single,
            )
            .unwrap();

        let mut wide: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker(scene.positions.len(), levels, 4)
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut wide,
            )
            .unwrap();

        assert_eq!(single.tracked(), wide.tracked());
        assert!(!single.is_empty());
        for index in single.tracked() {
            let index: usize = *index as usize;
            assert_eq!(single.transform(index), wide.transform(index));
        }
    }

    #[test]
    fn two_runs_of_the_same_tracker_agree_exactly() {
        let levels: usize = 3;
        let scene: Fixture = fixture(0.9, 0.4, levels);
        let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 4);

        let mut first: FlowResult = FlowResult::with_capacity(scene.positions.len());
        let mut second: FlowResult = FlowResult::with_capacity(scene.positions.len());
        for out in [&mut first, &mut second] {
            tracker
                .track(
                    &scene.prev,
                    &scene.next,
                    &scene.patches,
                    &scene.transforms,
                    out,
                )
                .unwrap();
        }
        assert_eq!(first.tracked(), second.tracked());
        for index in first.tracked() {
            let index: usize = *index as usize;
            assert_eq!(first.transform(index), second.transform(index));
        }
    }

    #[test]
    fn more_keypoints_than_capacity_is_refused() {
        let levels: usize = 1;
        let scene: Fixture = fixture(0.0, 0.0, levels);
        let mut tracker: CpuPatchTracker<Pattern51> = tracker(2, levels, 1);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        let error = tracker
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut out,
            )
            .unwrap_err();
        assert_eq!(
            error,
            TrackerError::CapacityExceeded {
                offered: scene.positions.len(),
                capacity: 2
            }
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(12))]

        /// Any shift up to the pattern's radius (3.5 px for `Pattern51`) is
        /// recovered, with the interpolation bias documented on
        /// [`sub_pixel_shift_error`] as the tolerance.
        #[test]
        fn any_shift_within_the_pattern_radius_is_recovered(
            dx in -3.5f32..3.5,
            dy in -3.5f32..3.5,
        ) {
            let (median, worst, tracked, total) = sub_pixel_shift_error(dx, dy);
            prop_assert!(tracked * 4 >= total * 3, "tracked {tracked} of {total}");
            prop_assert!(median < 0.05, "median error {median} for ({dx}, {dy})");
            prop_assert!(worst < 0.2, "worst error {worst} for ({dx}, {dy})");
        }
    }
}
