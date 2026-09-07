//! `basalt::OpticalFlowPatch`, ported from `optical_flow/patch.h`.
//!
//! A patch is a pattern of taps sampled around a keypoint, mean-normalised, plus
//! the cached inverse-compositional factor `H_se2^-1 J_se2^T` that turns a
//! residual into an SE(2) increment. The three pieces are:
//!
//! * [`set_data`] — `patch.h:73-99`, sample and mean-normalise, optionally
//!   through an SE(2) warp. Only the ported `test_patch.cpp` uses it.
//! * [`set_data_jac_se2`] — `patch.h:101-142`, the same sampling with the
//!   gradient, the SE(2) warp Jacobian, and the product-rule term that comes
//!   from differentiating the `1/mean` factor.
//! * [`OpticalFlowPatch::set_from_image`] — `patch.h:144-166`, which forms
//!   `H = J^T J`, inverts it with Eigen's pivoted LDLT and caches `H^-1 J^T`.
//! * [`OpticalFlowPatch::residual`] — `patch.h:168-202`, the mean-normalised SSD
//!   residual of a warped pattern against the stored data.
//!
//! ## Two deliberate departures from the C++ shape
//!
//! **The 52x3 Jacobian is never materialised.** `setDataJacSe2` builds
//! `MatrixP3 J_se2` and `setFromImage` then forms `J^T J` and `H^-1 J^T` from it
//! (`patch.h:147-156`). Here the *transpose* `J^T` (3xP) is written straight into
//! the buffer that will hold `H^-1 J^T`, the 3x3 `H` is accumulated one rank-1
//! outer product per tap while the rows are finalised, and the final product is
//! applied one column at a time in place. The equivalence is exact as algebra —
//! `J^T J = sum_i row_i^T row_i` and `(H^-1 J^T)[:, i] = H^-1 (J^T[:, i])` — and
//! it keeps the largest transient per patch at a 3x3 matrix instead of the 624
//! bytes a 52x3 `f32` Jacobian needs, which is over the ~512-byte per-thread
//! private budget that goes racy on Vulkan/SPIR-V
//! (`cubecl-portability.md` §12.2, CubeCL issue #1336). The summation order of
//! `J^T J` is tap index ascending; Eigen's blocked product may add in another
//! order, so the 3x3 can differ in the last `f32` bits.
//!
//! **`residual` takes the warp, not a pre-multiplied pattern.** The C++ builds
//! `transformed_pat = transform.linear() * pattern2` and adds the translation
//! columnwise (`frame_to_frame_optical_flow.h:411-412`) before calling
//! `residual(img, transformed_pattern, res)`. [`AffineCompact2::warp_tap`]
//! performs the same two multiplies and one add per tap in the same order, so
//! fusing them removes a 2xP transient without changing a bit.

use std::marker::PhantomData;

use nalgebra::{Matrix3, Vector2, Vector3};

use crate::frontend::ldlt::ldlt_inverse3;
use crate::frontend::patterns::{MAX_PATTERN_SIZE, Pattern};
use crate::frontend::se2::AffineCompact2;
use crate::image::ImageU16;
use crate::lie::LieScalar;

/// The border every patch tap is sampled with, `img.InBounds(p, 2)`
/// (`patch.h:87`, `:117`, `:173`).
///
/// `interpGrad` alone needs 1 (`image.h:424`); basalt asks for 2 everywhere on
/// the patch path so that a tap and its gradient stencil are both inside.
pub const PATCH_BORDER: f32 = 2.0;

/// What a patch can be sampled from.
///
/// basalt templates `setData`/`setDataJacSe2` on the image type
/// (`patch.h:73`, `:101`) so its own tests can substitute an analytic function
/// for a real image (`test/src/test_patch.cpp:11-30`), and the port does the
/// same. Monomorphised, so a tap costs no more than a direct call; the GPU seam
/// rule against per-pixel virtual dispatch (`cubecl-portability.md` §12.3 item 5)
/// is about the *stage* traits, and this is not one.
pub trait PatchSource<S: LieScalar> {
    /// `img.InBounds(p, border)` (`image.h:694-705`).
    fn in_bounds(&self, x: S, y: S, border: S) -> bool;

    /// `img.interp<Scalar>(p)` (`image.h:396-415`).
    fn interp(&self, x: S, y: S) -> S;

    /// `img.interpGrad<Scalar>(p)` (`image.h:418-469`): `[value, d/dx, d/dy]`.
    fn interp_grad(&self, x: S, y: S) -> (S, [S; 2]);
}

impl PatchSource<f32> for ImageU16 {
    #[inline]
    fn in_bounds(&self, x: f32, y: f32, border: f32) -> bool {
        ImageU16::in_bounds(self, x, y, border)
    }

    #[inline]
    fn interp(&self, x: f32, y: f32) -> f32 {
        ImageU16::interp(self, x, y)
    }

    #[inline]
    fn interp_grad(&self, x: f32, y: f32) -> (f32, [f32; 2]) {
        ImageU16::interp_grad(self, x, y)
    }
}

/// `OpticalFlowPatch::setData` (`patch.h:73-99`).
///
/// Samples the pattern around `pos`, optionally through the SE(2) warp `se2`
/// (`p = pos + (*se2) * pattern2.col(i)`, `patch.h:82`), marks out-of-bounds taps
/// with `-1` (`:93`) and divides every tap by the mean of the valid ones (`:98`).
/// Note that `data /= mean` scales the `-1` markers too, exactly as the C++ does;
/// the sign is what later tests look at, and it survives a positive mean.
///
/// Returns the number of taps that were in bounds. `data` must be at least
/// `P::SIZE` long; only that prefix is written.
///
/// # Panics
///
/// If `data` is shorter than `P::SIZE`.
// The loop variable is the pattern tap index, shared by `data`, `P::OFFSETS`
// and the C++ `for (int i = 0; i < PATTERN_SIZE; i++)` this mirrors; an
// `enumerate` over one of them would name the wrong thing.
#[allow(clippy::needless_range_loop)]
pub fn set_data<P: Pattern, S: LieScalar, Src: PatchSource<S>>(
    source: &Src,
    pos: &Vector2<S>,
    se2: Option<&AffineCompact2<S>>,
    data: &mut [S],
) -> (S, usize) {
    let border: S = S::from_literal(f64::from(PATCH_BORDER));
    let mut num_valid_points: usize = 0;
    let mut sum: S = S::zero();

    for i in 0..P::SIZE {
        let tap: [S; 2] = [
            S::from_literal(f64::from(P::OFFSETS[i][0])),
            S::from_literal(f64::from(P::OFFSETS[i][1])),
        ];
        let warped: [S; 2] = match se2 {
            Some(warp) => warp.warp_tap(tap),
            None => tap,
        };
        let px: S = pos.x + warped[0];
        let py: S = pos.y + warped[1];

        if source.in_bounds(px, py, border) {
            let value: S = source.interp(px, py);
            data[i] = value;
            sum += value;
            num_valid_points += 1;
        } else {
            data[i] = -S::one();
        }
    }

    // `mean = sum / num_valid_points; data /= mean;` (`patch.h:97-98`). The int
    // is converted to the scalar before the division, as C++ does.
    let mean: S = sum / S::from_literal(num_valid_points as f64);
    for value in data.iter_mut().take(P::SIZE) {
        *value /= mean;
    }
    (mean, num_valid_points)
}

/// `OpticalFlowPatch::setDataJacSe2` (`patch.h:101-142`), writing `J^T` not `J`.
///
/// `jacobian_transpose[r][i]` is `J_se2(i, r)`: three rows of `P::SIZE` taps,
/// tap index fast-varying, which is both the transpose the caller needs next and
/// the SoA major axis the GPU seam wants (`cubecl-portability.md` §12.2).
///
/// The four steps, in the C++'s order:
///
/// 1. per tap, `Jw_se2` is `[[1, 0, -p_y], [0, 1, p_x]]` for the *pattern* offset
///    `p` (`patch.h:107-115`), and `J.row(i) = grad^T Jw_se2` (`:121`);
///    `grad_sum_se2` accumulates the raw rows (`:122`);
/// 2. `mean = sum / n` and `mean_inv = n / sum` (`:129`, `:131`) — two separate
///    divisions, not reciprocals of each other;
/// 3. the product-rule correction `J.row(i) -= grad_sum^T * data[i] / sum`
///    applied with the **raw** `data[i]`, then `data[i] *= mean_inv` (`:135-136`);
///    rows of invalid taps are zeroed (`:138`);
/// 4. `J_se2 *= mean_inv` (`:141`), folded into step 3's loop here — every
///    element is multiplied once either way.
///
/// Step 3 is the term the papers never write out; dropping it gives a Jacobian
/// that looks right and converges to the wrong warp (`papers-part2.md` §12.1).
///
/// # Panics
///
/// If `data` is shorter than `P::SIZE`.
pub fn set_data_jac_se2<P: Pattern, S: LieScalar, Src: PatchSource<S>>(
    source: &Src,
    pos: &Vector2<S>,
    data: &mut [S],
    jacobian_transpose: &mut [[S; MAX_PATTERN_SIZE]; 3],
) -> (S, usize) {
    let border: S = S::from_literal(f64::from(PATCH_BORDER));
    let mut num_valid_points: usize = 0;
    let mut sum: S = S::zero();
    let mut grad_sum_se2: Vector3<S> = Vector3::zeros();

    for i in 0..P::SIZE {
        let tap_x: S = S::from_literal(f64::from(P::OFFSETS[i][0]));
        let tap_y: S = S::from_literal(f64::from(P::OFFSETS[i][1]));
        let px: S = pos.x + tap_x;
        let py: S = pos.y + tap_y;

        if source.in_bounds(px, py, border) {
            let (value, grad): (S, [S; 2]) = source.interp_grad(px, py);
            data[i] = value;
            sum += value;
            // `valGrad.tail<2>().transpose() * Jw_se2` with
            // `Jw_se2 = [[1, 0, -tap_y], [1 -> 0, 1, tap_x]]` (`patch.h:107-115`).
            let row: [S; 3] = [grad[0], grad[1], grad[0] * -tap_y + grad[1] * tap_x];
            jacobian_transpose[0][i] = row[0];
            jacobian_transpose[1][i] = row[1];
            jacobian_transpose[2][i] = row[2];
            grad_sum_se2.x += row[0];
            grad_sum_se2.y += row[1];
            grad_sum_se2.z += row[2];
            num_valid_points += 1;
        } else {
            data[i] = -S::one();
        }
    }

    let mean: S = sum / S::from_literal(num_valid_points as f64);
    let mean_inv: S = S::from_literal(num_valid_points as f64) / sum;

    for i in 0..P::SIZE {
        if data[i] >= S::zero() {
            let raw: S = data[i];
            for r in 0..3 {
                jacobian_transpose[r][i] =
                    (jacobian_transpose[r][i] - grad_sum_se2[r] * raw / sum) * mean_inv;
            }
            data[i] = raw * mean_inv;
        } else {
            for row in jacobian_transpose.iter_mut() {
                row[i] = S::zero();
            }
        }
    }

    (mean, num_valid_points)
}

/// `basalt::OpticalFlowPatch<Scalar, Pattern>` (`patch.h:45-214`), `Scalar = f32`.
///
/// The stored state is `patch.h:204-213`: the source position, the
/// mean-normalised taps with `-1` marking the ones that fell outside, the cached
/// `H_se2^-1 J_se2^T`, the mean, and the validity flag. The buffers are sized for
/// the largest pattern and only the first `P::SIZE` entries are used, so one
/// layout serves every pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalFlowPatch<P: Pattern> {
    /// `pos`, the centre this patch was sampled at, in the level's pixel frame.
    pub pos: Vector2<f32>,
    /// `data`: mean-normalised taps; negative marks a tap that was out of bounds.
    pub data: [f32; MAX_PATTERN_SIZE],
    /// `H_se2_inv_J_se2_T` (3xP), row `r`, tap `i`.
    pub h_se2_inv_j_se2_t: [[f32; MAX_PATTERN_SIZE]; 3],
    /// `mean`, the average of the taps that were in bounds.
    pub mean: f32,
    /// `valid`: whether this patch may be tracked (`patch.h:164-165`).
    pub valid: bool,
    /// The pattern this patch was sampled with.
    pattern: PhantomData<P>,
}

impl<P: Pattern> Default for OpticalFlowPatch<P> {
    fn default() -> Self {
        Self {
            pos: Vector2::zeros(),
            data: [0.0; MAX_PATTERN_SIZE],
            h_se2_inv_j_se2_t: [[0.0; MAX_PATTERN_SIZE]; 3],
            mean: 0.0,
            valid: false,
            pattern: PhantomData,
        }
    }
}

impl<P: Pattern> OpticalFlowPatch<P> {
    /// `OpticalFlowPatch(img, pos)` (`patch.h:71`): build a patch in one step.
    pub fn new<Src: PatchSource<f32>>(source: &Src, pos: Vector2<f32>) -> Self {
        let mut patch: Self = Self::default();
        patch.set_from_image(source, pos);
        patch
    }

    /// `setFromImage` (`patch.h:144-166`).
    ///
    /// Samples the pattern with [`set_data_jac_se2`], forms `H_se2 = J^T J`,
    /// inverts it with Eigen's pivoted LDLT ([`ldlt_inverse3`], `patch.h:151-154`)
    /// and caches `H_se2^-1 J_se2^T` (`:156`). A patch is `valid` only when the
    /// mean is above the scalar epsilon and both the cached factor and the data
    /// are finite (`:164-165`) — an all-black patch cannot be normalised and
    /// would otherwise carry `inf` into the tracker.
    pub fn set_from_image<Src: PatchSource<f32>>(&mut self, source: &Src, pos: Vector2<f32>) {
        self.pos = pos;

        // `J_se2^T` goes straight into the buffer that ends up holding
        // `H^-1 J^T`; see the module note on why `J` is never materialised.
        let (mean, _) = set_data_jac_se2::<P, f32, Src>(
            source,
            &pos,
            &mut self.data,
            &mut self.h_se2_inv_j_se2_t,
        );
        self.mean = mean;

        // `H_se2 = J_se2.transpose() * J_se2` (`patch.h:151`), accumulated one
        // rank-1 outer product per tap, in tap order.
        let mut h_se2: Matrix3<f32> = Matrix3::zeros();
        for i in 0..P::SIZE {
            let row: [f32; 3] = [
                self.h_se2_inv_j_se2_t[0][i],
                self.h_se2_inv_j_se2_t[1][i],
                self.h_se2_inv_j_se2_t[2][i],
            ];
            for r in 0..3 {
                for c in 0..3 {
                    h_se2[(r, c)] += row[r] * row[c];
                }
            }
        }

        let h_se2_inv: Matrix3<f32> = ldlt_inverse3(&h_se2);

        // `H_se2_inv_J_se2_T = H_se2_inv * J_se2.transpose()` (`patch.h:156`),
        // one column at a time so the 3xP buffer is updated in place.
        let mut finite: bool = true;
        for i in 0..P::SIZE {
            let column: Vector3<f32> = Vector3::new(
                self.h_se2_inv_j_se2_t[0][i],
                self.h_se2_inv_j_se2_t[1][i],
                self.h_se2_inv_j_se2_t[2][i],
            );
            let product: Vector3<f32> = h_se2_inv * column;
            for r in 0..3 {
                self.h_se2_inv_j_se2_t[r][i] = product[r];
                finite &= product[r].is_finite();
            }
            finite &= self.data[i].is_finite();
        }

        self.valid = mean > f32::EPSILON && finite;
    }

    /// `residual` (`patch.h:168-202`), with the warp applied per tap.
    ///
    /// Samples `img` at `transform * pattern2.col(i)`; a tap outside the image is
    /// marked `-1` and contributes nothing (`:173-179`). An all-black target
    /// (`sum < epsilon`) zeroes the residual and fails (`:183-186`). Every tap
    /// valid in **both** the target and the stored source becomes
    /// `num_valid_points * val / sum - data[i]` (`:193`) — note that the target's
    /// count and sum come from the taps in bounds in the *target*, while
    /// `data[i]` was normalised by the source's own count, so the two
    /// normalisations differ whenever the patch straddles a border
    /// (`papers-part2.md` §13 deviation D5). Everything else is zeroed (`:197`).
    ///
    /// Returns `true` only when more than half the pattern survived (`:201`).
    ///
    /// # Panics
    ///
    /// If `residual` is shorter than `P::SIZE`.
    #[inline]
    pub fn residual<Src: PatchSource<f32>>(
        &self,
        source: &Src,
        transform: &AffineCompact2<f32>,
        residual: &mut [f32],
    ) -> bool {
        patch_residual::<P, Src>(&self.data, 1, source, transform, residual)
    }

    /// `inc = -H_se2_inv_J_se2_T * res` (`frame_to_frame_optical_flow.h:419`).
    #[inline]
    pub fn increment(&self, residual: &[f32]) -> Vector3<f32> {
        patch_increment::<P>(
            self.h_se2_inv_j_se2_t.as_flattened(),
            1,
            MAX_PATTERN_SIZE,
            residual,
        )
    }
}

/// [`OpticalFlowPatch::residual`] over a strided `data` array.
///
/// `stride` is the distance between two taps of the same patch: `1` for the
/// per-patch struct, and the patch capacity for the SoA the tracker reads
/// (patch index fast-varying, `cubecl-portability.md` §12.2). One body serves
/// both so the arithmetic cannot drift apart.
///
/// # Panics
///
/// If `data` or `residual` is too short for `P::SIZE` taps at `stride`.
// As `set_data`: `i` is the tap index of `residual`, `data` and `P::OFFSETS`.
#[allow(clippy::needless_range_loop)]
#[inline]
pub fn patch_residual<P: Pattern, Src: PatchSource<f32>>(
    data: &[f32],
    stride: usize,
    source: &Src,
    transform: &AffineCompact2<f32>,
    residual: &mut [f32],
) -> bool {
    let mut sum: f32 = 0.0;
    let mut num_valid_points: i32 = 0;

    for i in 0..P::SIZE {
        let warped: [f32; 2] = transform.warp_tap(P::OFFSETS[i]);
        if source.in_bounds(warped[0], warped[1], PATCH_BORDER) {
            let value: f32 = source.interp(warped[0], warped[1]);
            residual[i] = value;
            sum += value;
            num_valid_points += 1;
        } else {
            residual[i] = -1.0;
        }
    }

    if sum < f32::EPSILON {
        residual[..P::SIZE].fill(0.0);
        return false;
    }

    let mut num_residuals: usize = 0;
    let count: f32 = num_valid_points as f32;
    for i in 0..P::SIZE {
        let stored: f32 = data[i * stride];
        if residual[i] >= 0.0 && stored >= 0.0 {
            let value: f32 = residual[i];
            residual[i] = count * value / sum - stored;
            num_residuals += 1;
        } else {
            residual[i] = 0.0;
        }
    }

    num_residuals > P::SIZE / 2
}

/// `H_se2_inv_J_se2_T * res` over a strided 3xP array
/// (`frame_to_frame_optical_flow.h:419`, without the leading minus).
///
/// `element_stride` separates two taps of one row and `row_stride` two rows, so
/// the same body reads the packed per-patch buffer and the tracker's SoA. The
/// sum runs over taps in ascending index order, which fixes the reduction order
/// the way decision D31 asks; Eigen's vectorised `3xP * P` may add in another.
///
/// # Panics
///
/// If `h_inv_jt` or `residual` is too short for three rows of `P::SIZE` taps.
#[inline]
pub fn patch_increment<P: Pattern>(
    h_inv_jt: &[f32],
    element_stride: usize,
    row_stride: usize,
    residual: &[f32],
) -> Vector3<f32> {
    let mut increment: Vector3<f32> = Vector3::zeros();
    for r in 0..3 {
        let base: usize = r * row_stride;
        let mut sum: f32 = 0.0;
        for i in 0..P::SIZE {
            sum += h_inv_jt[base + i * element_stride] * residual[i];
        }
        increment[r] = sum;
    }
    increment
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::frontend::patterns::{Pattern51, Pattern52};
    use crate::frontend::se2::se2_exp;
    use approx::assert_abs_diff_eq;

    /// `test/src/test_patch.cpp:11-30`: `sin(x/100 + y/20)`, always in bounds,
    /// with the analytic value and gradient.
    struct SmoothFunction;

    impl PatchSource<f64> for SmoothFunction {
        fn in_bounds(&self, _x: f64, _y: f64, _border: f64) -> bool {
            true
        }

        fn interp(&self, x: f64, y: f64) -> f64 {
            (x / 100.0 + y / 20.0).sin()
        }

        fn interp_grad(&self, x: f64, y: f64) -> (f64, [f64; 2]) {
            let angle: f64 = x / 100.0 + y / 20.0;
            (angle.sin(), [angle.cos() / 100.0, angle.cos() / 20.0])
        }
    }

    /// The probe point of both C++ tests: `(231, 123) + (0.4, 0.34345)`
    /// (`test_patch.cpp:33-37`, `:49-53`).
    fn probe() -> Vector2<f64> {
        Vector2::new(231.0 + 0.4, 123.0 + 0.34345)
    }

    /// Port of `TEST(Patch, ImageInterpolateGrad)` (`test_patch.cpp:32-46`):
    /// the gradient `interpGrad` returns is the derivative of `interp`.
    #[test]
    fn image_interpolate_grad() {
        let image: SmoothFunction = SmoothFunction;
        let point: Vector2<f64> = probe();
        let (_, grad): (f64, [f64; 2]) = image.interp_grad(point.x, point.y);

        // `test_jacobian`'s central difference, `test/include/test_utils.h`.
        let step: f64 = 1e-6;
        for axis in 0..2 {
            let mut plus: Vector2<f64> = point;
            let mut minus: Vector2<f64> = point;
            plus[axis] += step;
            minus[axis] -= step;
            let numeric: f64 =
                (image.interp(plus.x, plus.y) - image.interp(minus.x, minus.y)) / (2.0 * step);
            assert_abs_diff_eq!(grad[axis], numeric, epsilon = 1e-9);
        }
    }

    /// Port of `TEST(Patch, PatchSe2Jac)` (`test_patch.cpp:48-83`), first half:
    /// `setDataJacSe2` and `setData` agree on the mean and the normalised data.
    #[test]
    fn patch_se2_jac_data_matches_set_data() {
        let image: SmoothFunction = SmoothFunction;
        let point: Vector2<f64> = probe();

        let mut data_jac: [f64; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
        let mut jacobian: [[f64; MAX_PATTERN_SIZE]; 3] = [[0.0; MAX_PATTERN_SIZE]; 3];
        let (mean_jac, valid_jac) = set_data_jac_se2::<Pattern52, f64, SmoothFunction>(
            &image,
            &point,
            &mut data_jac,
            &mut jacobian,
        );

        let mut data: [f64; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
        let (mean, valid) =
            set_data::<Pattern52, f64, SmoothFunction>(&image, &point, None, &mut data);

        assert_eq!(valid_jac, Pattern52::SIZE);
        assert_eq!(valid, Pattern52::SIZE);
        assert_abs_diff_eq!(mean_jac, mean, epsilon = 1e-8);
        for i in 0..Pattern52::SIZE {
            assert_abs_diff_eq!(data_jac[i], data[i], epsilon = 1e-12);
        }
    }

    /// Port of `TEST(Patch, PatchSe2Jac)` (`test_patch.cpp:70-82`), second half:
    /// `J_se2` is the derivative of the normalised data with respect to the SE(2)
    /// warp, at the identity.
    #[test]
    fn patch_se2_jac_is_the_derivative_of_the_warped_data() {
        let image: SmoothFunction = SmoothFunction;
        let point: Vector2<f64> = probe();

        let mut data: [f64; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
        let mut jacobian: [[f64; MAX_PATTERN_SIZE]; 3] = [[0.0; MAX_PATTERN_SIZE]; 3];
        set_data_jac_se2::<Pattern52, f64, SmoothFunction>(
            &image,
            &point,
            &mut data,
            &mut jacobian,
        );

        let step: f64 = 1e-6;
        for column in 0..3 {
            let mut plus_tangent: Vector3<f64> = Vector3::zeros();
            let mut minus_tangent: Vector3<f64> = Vector3::zeros();
            plus_tangent[column] = step;
            minus_tangent[column] = -step;

            let mut plus: [f64; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
            let mut minus: [f64; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
            set_data::<Pattern52, f64, SmoothFunction>(
                &image,
                &point,
                Some(&se2_exp(&plus_tangent)),
                &mut plus,
            );
            set_data::<Pattern52, f64, SmoothFunction>(
                &image,
                &point,
                Some(&se2_exp(&minus_tangent)),
                &mut minus,
            );

            for i in 0..Pattern52::SIZE {
                let numeric: f64 = (plus[i] - minus[i]) / (2.0 * step);
                assert_abs_diff_eq!(jacobian[column][i], numeric, epsilon = 1e-6);
            }
        }
    }

    /// A synthetic textured image: a smooth ramp plus a sinusoid, so every patch
    /// has gradient in both directions and `H_se2` is well conditioned.
    fn textured_image(width: usize, height: usize) -> ImageU16 {
        let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
        for y in 0..height {
            for x in 0..width {
                let fx: f64 = x as f64;
                let fy: f64 = y as f64;
                let value: f64 = 20_000.0
                    + 8_000.0 * (fx * 0.31).sin() * (fy * 0.23).cos()
                    + 3_000.0 * ((fx + fy) * 0.11).sin();
                image.set(x, y, value as u16);
            }
        }
        image
    }

    #[test]
    fn a_textured_patch_is_valid_and_its_residual_against_itself_is_zero() {
        let image: ImageU16 = textured_image(64, 64);
        let patch: OpticalFlowPatch<Pattern51> =
            OpticalFlowPatch::new(&image, Vector2::new(32.0, 32.0));
        assert!(patch.valid);
        assert!(patch.mean > 0.0);

        let mut residual: [f32; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
        let survived: bool = patch.residual(
            &image,
            &AffineCompact2::at(Vector2::new(32.0, 32.0)),
            &mut residual,
        );
        assert!(survived);
        for value in residual.iter().take(Pattern51::SIZE) {
            assert_abs_diff_eq!(*value, 0.0, epsilon = 1e-5);
        }
    }

    /// `patch.h:164-165`: an all-black patch has mean zero, so it is not valid.
    #[test]
    fn an_all_black_patch_is_not_valid() {
        let image: ImageU16 = ImageU16::zeros(64, 64).unwrap();
        let patch: OpticalFlowPatch<Pattern51> =
            OpticalFlowPatch::new(&image, Vector2::new(32.0, 32.0));
        assert!(!patch.valid);
    }

    /// `patch.h:183-186`: a residual against an all-black target zeroes and fails.
    #[test]
    fn a_residual_against_black_fails_and_zeroes() {
        let image: ImageU16 = textured_image(64, 64);
        let black: ImageU16 = ImageU16::zeros(64, 64).unwrap();
        let patch: OpticalFlowPatch<Pattern51> =
            OpticalFlowPatch::new(&image, Vector2::new(32.0, 32.0));

        let mut residual: [f32; MAX_PATTERN_SIZE] = [1.0; MAX_PATTERN_SIZE];
        let survived: bool = patch.residual(
            &black,
            &AffineCompact2::at(Vector2::new(32.0, 32.0)),
            &mut residual,
        );
        assert!(!survived);
        assert_eq!(&residual[..Pattern51::SIZE], &[0.0; Pattern51::SIZE][..]);
    }

    /// `patch.h:201`: with most of the pattern off the edge, the residual is
    /// rejected however good the surviving taps are.
    #[test]
    fn a_residual_with_half_the_pattern_outside_is_rejected() {
        let image: ImageU16 = textured_image(64, 64);
        let patch: OpticalFlowPatch<Pattern51> =
            OpticalFlowPatch::new(&image, Vector2::new(32.0, 32.0));

        let mut residual: [f32; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
        let survived: bool = patch.residual(
            &image,
            &AffineCompact2::at(Vector2::new(1.0, 32.0)),
            &mut residual,
        );
        assert!(!survived);
    }

    /// `patch.h:93`, `:125`: a tap outside the image is marked negative, and the
    /// mean-normalisation of `setData` keeps the sign.
    #[test]
    fn out_of_bounds_taps_stay_negative() {
        let image: ImageU16 = textured_image(64, 64);
        let mut data: [f32; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
        let (mean, valid) =
            set_data::<Pattern51, f32, ImageU16>(&image, &Vector2::new(4.0, 32.0), None, &mut data);
        assert!(mean > 0.0);
        assert!(valid < Pattern51::SIZE);
        assert!(data.iter().take(Pattern51::SIZE).any(|value| *value < 0.0));
    }
}
