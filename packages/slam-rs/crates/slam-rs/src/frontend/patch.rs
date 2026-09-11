//! Mean-normalized patches and cached inverse-compositional SE(2) factors.
//!
//! Sampling computes raw taps and gradients, then applies the derivative of the
//! normalization factor. Build `H = JᵀJ` in tap order using rank-one updates,
//! invert the 3x3 factor with guarded pivoted LDLT, then apply it to each stored
//! Jacobian column. The product-rule correction is required for the right warp.
//!
//! [`build_patch`] writes into caller storage with explicit strides, serving
//! both packed test records and the tracker's structure-of-arrays layout.
//! The largest temporary is a 3x3 matrix plus a 3-vector, avoiding a full patch
//! Jacobian in per-thread GPU storage. Residual sampling warps one tap at a time,
//! avoiding a transient transformed-pattern matrix.

use std::marker::PhantomData;

use nalgebra::{Matrix3, Vector2, Vector3};

use crate::frontend::ldlt::ldlt_inverse3;
use crate::frontend::patterns::{MAX_PATTERN_SIZE, Pattern};
use crate::frontend::se2::AffineCompact2;
use crate::image::ImageU16;
use crate::lie::LieScalar;

/// Two-pixel patch border, keeping taps and their gradient stencils inside the image.
pub const PATCH_BORDER: f32 = 2.0;

/// A patch sampling source. Tests can supply analytic functions in place of images.
/// Static dispatch avoids per-tap virtual calls.
pub trait PatchSource<S: LieScalar> {
    /// `img.InBounds(p, border)`.
    fn in_bounds(&self, x: S, y: S, border: S) -> bool;

    /// `img.interp<Scalar>(p)`.
    fn interp(&self, x: S, y: S) -> S;

    /// `img.interpGrad<Scalar>(p)` : `[value, d/dx, d/dy]`.
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

/// Write a strided transposed SE(2) sampling Jacobian.
/// `data[i * data_stride]` is tap i; row/element strides address the three Jacobian rows.
///
/// First form `gradᵀ [[1, 0, -p_y], [0, 1, p_x]]` and accumulate raw row sums.
/// Compute `mean = sum/n` and `mean_inv = n/sum` separately. Subtract the
/// product-rule term `grad_sumᵀ * raw_data[i] / sum`, zero invalid rows, then
/// scale valid data and Jacobians by `mean_inv`. Omitting the product-rule term
/// converges to the wrong warp (`papers-part2.md` §12.1).
/// Returns the mean and valid tap count.
///
/// # Panics
/// If either array is too short for the pattern and strides.
#[allow(clippy::too_many_arguments)]
pub fn set_data_jac_se2<P: Pattern, S: LieScalar, Src: PatchSource<S>>(
    source: &Src,
    pos: &Vector2<S>,
    data: &mut [S],
    data_stride: usize,
    jacobian_transpose: &mut [S],
    jt_element_stride: usize,
    jt_row_stride: usize,
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
            data[i * data_stride] = value;
            sum += value;
            // `valGrad.tail<2>().transpose() * Jw_se2` with
            // `Jw_se2 = [[1, 0, -tap_y], [0, 1, tap_x]]`.
            let row: [S; 3] = [grad[0], grad[1], grad[0] * -tap_y + grad[1] * tap_x];
            for r in 0..3 {
                jacobian_transpose[r * jt_row_stride + i * jt_element_stride] = row[r];
                grad_sum_se2[r] += row[r];
            }
            num_valid_points += 1;
        } else {
            data[i * data_stride] = -S::one();
        }
    }

    let mean: S = sum / S::from_literal(num_valid_points as f64);
    let mean_inv: S = S::from_literal(num_valid_points as f64) / sum;

    for i in 0..P::SIZE {
        let raw: S = data[i * data_stride];
        if raw >= S::zero() {
            for r in 0..3 {
                let slot: usize = r * jt_row_stride + i * jt_element_stride;
                jacobian_transpose[slot] =
                    (jacobian_transpose[slot] - grad_sum_se2[r] * raw / sum) * mean_inv;
            }
            data[i * data_stride] = raw * mean_inv;
        } else {
            for r in 0..3 {
                jacobian_transpose[r * jt_row_stride + i * jt_element_stride] = S::zero();
            }
        }
    }

    (mean, num_valid_points)
}

/// Build a patch factor in caller storage at the supplied strides.
/// On return the Jacobian buffer contains `H^-1 Jᵀ`, formed by tap-ordered
/// rank-one updates, guarded 3x3 LDLT and column-wise multiplication.
/// Return `(mean, valid)`; validity requires positive mean above epsilon and
/// finite values, so a black patch cannot introduce infinities into tracking.
///
/// # Panics
/// If either array is too short for the pattern and strides.
pub fn build_patch<P: Pattern, Src: PatchSource<f32>>(
    source: &Src,
    pos: &Vector2<f32>,
    data: &mut [f32],
    data_stride: usize,
    jacobian_transpose: &mut [f32],
    jt_element_stride: usize,
    jt_row_stride: usize,
) -> (f32, bool) {
    let (mean, _) = set_data_jac_se2::<P, f32, Src>(
        source,
        pos,
        data,
        data_stride,
        jacobian_transpose,
        jt_element_stride,
        jt_row_stride,
    );

    let mut h_se2: Matrix3<f32> = Matrix3::zeros();
    for i in 0..P::SIZE {
        let row: Vector3<f32> =
            Vector3::from_fn(|r, _| jacobian_transpose[r * jt_row_stride + i * jt_element_stride]);
        // `H += row row^T`. nalgebra walks the columns and does
        // `h[(r, c)] += (1 · row[c]) · row[r]` where the nested loop below it
        // did `row[r] · row[c]`, so the accumulation is bit-identical, not
        // merely equivalent (S33 audit B, row 7).
        h_se2.ger(1.0, &row, &row, 1.0);
    }

    let h_se2_inv: Matrix3<f32> = ldlt_inverse3(&h_se2);

    let mut finite: bool = true;
    for i in 0..P::SIZE {
        let column: Vector3<f32> =
            Vector3::from_fn(|r, _| jacobian_transpose[r * jt_row_stride + i * jt_element_stride]);
        let product: Vector3<f32> = h_se2_inv * column;
        for r in 0..3 {
            jacobian_transpose[r * jt_row_stride + i * jt_element_stride] = product[r];
            finite &= product[r].is_finite();
        }
        finite &= data[i * data_stride].is_finite();
    }

    (mean, mean > f32::EPSILON && finite)
}

/// Packed f32 patch record used by tests.
/// Tracking builds directly into strided arrays through the same [`build_patch`].
/// The record stores source position, normalized taps, cached `H^-1 Jᵀ`, mean
/// and validity. Negative taps mark invalid samples. Only `P::SIZE` entries of
/// the maximum-sized buffers are used.
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
    /// `valid`: whether this patch may be tracked.
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
    /// `OpticalFlowPatch(img, pos)` : build a patch in one step.
    pub fn new<Src: PatchSource<f32>>(source: &Src, pos: Vector2<f32>) -> Self {
        let mut patch: Self = Self::default();
        patch.set_from_image(source, pos);
        patch
    }

    /// `setFromImage`, into this record's packed storage.
    ///
    /// A thin call into [`build_patch`] at stride 1. The tracking path does not
    /// come through here: it points [`build_patch`] straight at its
    /// structure-of-arrays, so no packed record is ever built per patch.
    pub fn set_from_image<Src: PatchSource<f32>>(&mut self, source: &Src, pos: Vector2<f32>) {
        self.pos = pos;
        let Self {
            data,
            h_se2_inv_j_se2_t,
            ..
        } = self;
        let (mean, valid) = build_patch::<P, Src>(
            source,
            &pos,
            data,
            1,
            h_se2_inv_j_se2_t.as_flattened_mut(),
            1,
            MAX_PATTERN_SIZE,
        );
        self.mean = mean;
        self.valid = valid;
    }

    /// `residual`, with the warp applied per tap.
    ///
    /// Samples `img` at `transform * pattern2.col(i)`; a tap outside the image is
    /// marked `-1` and contributes nothing. An all-black target
    /// (`sum < epsilon`) zeroes the residual and fails. Every tap
    /// valid in **both** the target and the stored source becomes
    /// `num_valid_points * val / sum - data[i]` — note that the target's
    /// count and sum come from the taps in bounds in the *target*, while
    /// `data[i]` was normalised by the source's own count, so the two
    /// normalisations differ whenever the patch straddles a border
    /// (`papers-part2.md` §13 deviation D5). Everything else is zeroed.
    ///
    /// Returns `true` only when more than half the pattern survived.
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

/// Multiply a strided 3xP factor by the residual, without a leading minus.
/// The same routine reads packed and structure-of-arrays storage. Sum taps in
/// ascending order for determinism (D31).
///
/// # Panics
/// If factor or residual storage is too short.
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

    /// Test sampling without Jacobians, optionally through an SE(2) warp.
    /// Mark out-of-bounds taps negative, then normalize by the valid-tap mean.
    /// Scaling the markers preserves their sign for a positive mean.
    /// Return the valid count and write only the pattern-sized prefix.
    ///
    /// # Panics
    /// If `data` is shorter than `P::SIZE`.
    #[allow(clippy::needless_range_loop)]
    fn set_data<P: Pattern, S: LieScalar, Src: PatchSource<S>>(
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

        // Convert the valid count to the scalar before mean division.
        let mean: S = sum / S::from_literal(num_valid_points as f64);
        for value in data.iter_mut().take(P::SIZE) {
            *value /= mean;
        }
        (mean, num_valid_points)
    }

    use super::*;
    use crate::frontend::patterns::{Pattern51, Pattern52};
    use crate::frontend::se2::se2_exp;
    use approx::assert_abs_diff_eq;

    /// `sin(x/100 + y/20)`, always in bounds,
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

    /// Subpixel probe at `(231, 123) + (0.4, 0.34345)`.
    fn probe() -> Vector2<f64> {
        Vector2::new(231.0 + 0.4, 123.0 + 0.34345)
    }

    /// Port of `TEST(Patch, ImageInterpolateGrad)` :
    /// the gradient `interpGrad` returns is the derivative of `interp`.
    #[test]
    fn image_interpolate_grad() {
        let image: SmoothFunction = SmoothFunction;
        let point: Vector2<f64> = probe();
        let (_, grad): (f64, [f64; 2]) = image.interp_grad(point.x, point.y);

        // `test_jacobian`'s central difference.
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

    /// Port of `TEST(Patch, PatchSe2Jac)`, first half:
    /// `setDataJacSe2` and `setData` agree on the mean and the normalised data.
    #[test]
    fn patch_se2_jac_data_matches_set_data() {
        let image: SmoothFunction = SmoothFunction;
        let point: Vector2<f64> = probe();

        let mut data_jac: [f64; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
        let mut jacobian: [f64; 3 * MAX_PATTERN_SIZE] = [0.0; 3 * MAX_PATTERN_SIZE];
        let (mean_jac, valid_jac) = set_data_jac_se2::<Pattern52, f64, SmoothFunction>(
            &image,
            &point,
            &mut data_jac,
            1,
            &mut jacobian,
            1,
            MAX_PATTERN_SIZE,
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

    /// Port of `TEST(Patch, PatchSe2Jac)`, second half:
    /// `J_se2` is the derivative of the normalised data with respect to the SE(2)
    /// warp, at the identity.
    #[test]
    fn patch_se2_jac_is_the_derivative_of_the_warped_data() {
        let image: SmoothFunction = SmoothFunction;
        let point: Vector2<f64> = probe();

        let mut data: [f64; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
        let mut jacobian: [f64; 3 * MAX_PATTERN_SIZE] = [0.0; 3 * MAX_PATTERN_SIZE];
        set_data_jac_se2::<Pattern52, f64, SmoothFunction>(
            &image,
            &point,
            &mut data,
            1,
            &mut jacobian,
            1,
            MAX_PATTERN_SIZE,
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
                assert_abs_diff_eq!(
                    jacobian[column * MAX_PATTERN_SIZE + i],
                    numeric,
                    epsilon = 1e-6
                );
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

    /// an all-black patch has mean zero, so it is not valid.
    #[test]
    fn an_all_black_patch_is_not_valid() {
        let image: ImageU16 = ImageU16::zeros(64, 64).unwrap();
        let patch: OpticalFlowPatch<Pattern51> =
            OpticalFlowPatch::new(&image, Vector2::new(32.0, 32.0));
        assert!(!patch.valid);
    }

    /// a residual against an all-black target zeroes and fails.
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

    /// with most of the pattern off the edge, the residual is
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

    /// : a tap outside the image is marked negative, and the
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
    /// The **accounting** behind the module's private-state claim, not the
    /// claim itself.
    ///
    /// The GPU seam budgets about 512 bytes of thread-private state before
    /// Vulkan/SPIR-V goes racy (`cubecl-portability.md` §12.2, CubeCL #1336).
    /// What a compiler actually puts on the stack is a property of the compiled
    /// code — measured at 36 bytes for the largest local and 344 for the frame,
    /// quoted in the module docs — and no `size_of` can stand in for that. What
    /// this pins is the *shape*: the pieces the build and the tracking loop hold
    /// are all small and none is sized by the pattern, and the packed record
    /// that is sized by the pattern is over the budget, so an edit that put one
    /// back on the hot path would have to change this test to pass.
    #[test]
    fn the_documented_private_state_accounting_holds() {
        // `build_patch`: the Hessian, its inverse, one column and one row.
        const BUILD: usize =
            2 * size_of::<Matrix3<f32>>() + size_of::<Vector3<f32>>() + size_of::<[f32; 3]>();
        // `track_point_at_level`: one residual vector, one warp, one increment.
        const TRACK: usize = size_of::<[f32; MAX_PATTERN_SIZE]>()
            + size_of::<AffineCompact2<f32>>()
            + size_of::<Vector3<f32>>();
        const {
            assert!(BUILD <= 512, "the patch build holds too much private state");
        };
        const {
            assert!(
                TRACK <= 512,
                "one tracking iteration holds too much private state"
            );
        };

        // The packed record is one patch's *storage*, the same bytes the
        // structure-of-arrays holds in its columns — not a transient. It is over
        // the budget, which is exactly why the tracking path never builds one:
        // `PatchSoA::build` points `build_patch` at its own arrays instead.
        const {
            assert!(size_of::<OpticalFlowPatch<Pattern51>>() > 512);
        };
    }
}
