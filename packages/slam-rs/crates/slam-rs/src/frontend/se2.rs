//! The 2-D warp the KLT tracker optimises: `Eigen::AffineCompact2f` and `Sophus::SE2::exp`.
//!
//! basalt stores a keypoint as an `Eigen::AffineCompact2f` — a 2x3 matrix whose
//! left 2x2 block is the linear part and whose last column is the translation
//! (`optical_flow.h:66`). The tracker's update is
//! `transform *= SE2::exp(inc).matrix()` (`frame_to_frame_optical_flow.h:428`),
//! where Eigen composes a compact 2x3 with a homogeneous 3x3 as the plain matrix
//! product `2x3 * 3x3`, which is the same thing as composing the two 3x3
//! transforms and dropping the last row.
//!
//! Only `exp` is needed: nothing on the tracking path takes a `log`, an inverse
//! or an adjoint of an SE(2) element.
//!
//! The type is generic in the scalar for the same reason basalt's is
//! (`OpticalFlowTyped<Scalar, Pattern>`, `optical_flow.h:184`): the shipped
//! frontend runs `f32` ([`AffineCompact2f`], decision D05), and the ported
//! `test_patch.cpp` Jacobian checks run `f64`, where a central difference is
//! sharp enough to be a real test.

use nalgebra::{Matrix2, Vector2, Vector3};

use crate::lie::LieScalar;

/// `Eigen::AffineCompact2<S>`: a 2x2 linear part and a translation.
///
/// The C++ type is a 2x3 matrix; keeping the two blocks apart costs nothing and
/// names them the way `frame_to_frame_optical_flow.h` does
/// (`transform.linear()`, `transform.translation()`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineCompact2<S: LieScalar> {
    /// `transform.linear()`, the 2x2 block.
    pub linear: Matrix2<S>,
    /// `transform.translation()`, the last column.
    pub translation: Vector2<S>,
}

/// The frontend's warp type, `Eigen::AffineCompact2f` (`optical_flow.h:66`).
pub type AffineCompact2f = AffineCompact2<f32>;

impl<S: LieScalar> Default for AffineCompact2<S> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<S: LieScalar> AffineCompact2<S> {
    /// `Eigen::AffineCompact2f::Identity()`.
    pub fn identity() -> Self {
        Self {
            linear: Matrix2::identity(),
            translation: Vector2::zeros(),
        }
    }

    /// An identity rotation at `position`, as `addPointsForCamera` builds one
    /// (`frame_to_frame_optical_flow.h:600-601`).
    pub fn at(position: Vector2<S>) -> Self {
        Self {
            linear: Matrix2::identity(),
            translation: position,
        }
    }

    /// `*this * other`, the 2x3-by-3x3 product Eigen performs for `operator*=`.
    ///
    /// `linear = self.linear * other.linear` and
    /// `translation = self.linear * other.translation + self.translation`.
    #[inline]
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            linear: self.linear * other.linear,
            translation: self.linear * other.translation + self.translation,
        }
    }

    /// The six coefficients, `[m00, m01, m10, m11, tx, ty]`.
    ///
    /// The order the structure-of-arrays layout stores them in
    /// ([`crate::frontend::tracker::FlowTransforms`]): row-major linear part,
    /// then the translation.
    pub fn coefficients(&self) -> [S; 6] {
        [
            self.linear[(0, 0)],
            self.linear[(0, 1)],
            self.linear[(1, 0)],
            self.linear[(1, 1)],
            self.translation.x,
            self.translation.y,
        ]
    }

    /// The affine image of a pattern tap: `linear * tap + translation`.
    ///
    /// This is one column of the `transformed_pat` the tracker builds at
    /// `frame_to_frame_optical_flow.h:411-412`
    /// (`transform.linear().matrix() * pattern2`, then `colwise() += translation`),
    /// in the same multiply-then-add order, so fusing the two statements into
    /// this call is bit-identical to materialising the 2xP matrix first.
    #[inline]
    pub fn warp_tap(&self, tap: [S; 2]) -> [S; 2] {
        [
            self.linear[(0, 0)] * tap[0] + self.linear[(0, 1)] * tap[1] + self.translation.x,
            self.linear[(1, 0)] * tap[0] + self.linear[(1, 1)] * tap[1] + self.translation.y,
        ]
    }
}

/// `Sophus::SE2<S>::exp` (`Sophus/sophus/se2.hpp:609-630`), as an affine warp.
///
/// The tangent is `(t_x, t_y, theta)`. Three details are load-bearing and are
/// reproduced literally:
///
/// * `SO2::exp(theta)` builds `SO2(cos theta, sin theta)` (`so2.hpp:453-457`),
///   and that two-argument constructor **normalises** (`so2.hpp:415-418`,
///   `:173-180`, `length = hypot(re, im)`). In `f32` the normalisation is not a
///   no-op, and the normalised components are the ones the `V` factor below
///   divides by.
/// * The small-angle branch triggers at `Sophus::Constants<S>::epsilon()`
///   (`common.hpp:166`, `:182-186`): `1e-10` in `f64` and `1e-5` in `f32`, both
///   far larger than the machine epsilon `nalgebra` would pick, which is why
///   [`LieScalar::SOPHUS_EPSILON`] exists.
/// * The translation is `V(theta) * upsilon` written out as two scalar
///   expressions (`se2.hpp:626-628`), not as a matrix product.
#[inline]
pub fn se2_exp<S: LieScalar>(tangent: &Vector3<S>) -> AffineCompact2<S> {
    let one: S = S::one();
    let theta: S = tangent[2];
    // `SO2<Scalar>::exp(theta)` — cos/sin, then normalise (`so2.hpp:415-418`).
    let cos_theta: S = theta.cos();
    let sin_theta: S = theta.sin();
    let length: S = cos_theta.hypot(sin_theta);
    let real: S = cos_theta / length;
    let imaginary: S = sin_theta / length;

    let (sin_theta_by_theta, one_minus_cos_theta_by_theta): (S, S) =
        if theta.abs() < S::SOPHUS_EPSILON {
            let theta_sq: S = theta * theta;
            (
                one - S::from_literal(1.0 / 6.0) * theta_sq,
                S::from_literal(0.5) * theta - S::from_literal(1.0 / 24.0) * theta * theta_sq,
            )
        } else {
            (imaginary / theta, (one - real) / theta)
        };

    AffineCompact2 {
        // `SO2::matrix()` is `[[re, -im], [im, re]]`.
        linear: Matrix2::new(real, -imaginary, imaginary, real),
        translation: Vector2::new(
            sin_theta_by_theta * tangent[0] - one_minus_cos_theta_by_theta * tangent[1],
            one_minus_cos_theta_by_theta * tangent[0] + sin_theta_by_theta * tangent[1],
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn exp_of_zero_is_the_identity() {
        let warp: AffineCompact2f = se2_exp(&Vector3::zeros());
        assert_eq!(warp.linear, Matrix2::identity());
        assert_eq!(warp.translation, Vector2::zeros());
    }

    #[test]
    fn pure_translation_passes_straight_through() {
        let warp: AffineCompact2f = se2_exp(&Vector3::new(0.25, -0.5, 0.0));
        assert_eq!(warp.linear, Matrix2::identity());
        assert_eq!(warp.translation, Vector2::new(0.25, -0.5));
    }

    /// A pure rotation of `theta` rotates by `theta` and leaves the origin fixed.
    #[test]
    fn pure_rotation_rotates_and_does_not_translate() {
        let theta: f32 = 0.3;
        let warp: AffineCompact2f = se2_exp(&Vector3::new(0.0, 0.0, theta));
        assert_abs_diff_eq!(warp.translation.x, 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(warp.translation.y, 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(warp.linear[(0, 0)], theta.cos(), epsilon = 1e-6);
        assert_abs_diff_eq!(warp.linear[(1, 0)], theta.sin(), epsilon = 1e-6);
        assert_abs_diff_eq!(warp.linear[(0, 1)], -theta.sin(), epsilon = 1e-6);
    }

    /// The small-angle branch is not a cosmetic optimisation: it is the only
    /// branch that survives.
    ///
    /// `(1 - cos theta) / theta` cancels catastrophically as `theta` shrinks. At
    /// Sophus's own threshold — `1e-10` in `f64`, `1e-5` in `f32` — `cos theta`
    /// has already rounded to exactly `1`, so the closed form returns `0` where
    /// the true value is `theta / 2`. The series returns `theta / 2`. Picking the
    /// branch at any *smaller* angle would hand the warp a translation factor of
    /// zero, which is why the threshold is where it is and why the port takes it
    /// from `Sophus::Constants` rather than from the machine epsilon.
    #[test]
    fn the_small_angle_branch_is_the_one_that_survives_the_cancellation() {
        let boundary: f64 = f64::SOPHUS_EPSILON;
        let series: AffineCompact2<f64> = se2_exp(&Vector3::new(1.0, 2.0, boundary * 0.999));
        let closed: AffineCompact2<f64> = se2_exp(&Vector3::new(1.0, 2.0, boundary * 1.001));
        // `x = 1 * sin(t)/t - 2 * (1 - cos t)/t`, so the series carries `-2 * t/2`.
        assert_abs_diff_eq!(
            series.translation.x,
            1.0 - boundary * 0.999,
            epsilon = 1e-20
        );
        assert_eq!(closed.translation.x, 1.0);

        let boundary: f32 = f32::SOPHUS_EPSILON;
        let series: AffineCompact2f = se2_exp(&Vector3::new(1.0, 2.0, boundary * 0.999));
        let closed: AffineCompact2f = se2_exp(&Vector3::new(1.0, 2.0, boundary * 1.001));
        assert_abs_diff_eq!(series.translation.x, 1.0 - boundary * 0.999, epsilon = 1e-9);
        assert_eq!(closed.translation.x, 1.0);
    }

    /// `warp_tap` is the fused form of `linear * pattern2` then `+= translation`.
    #[test]
    fn warp_tap_matches_the_matrix_form() {
        let warp: AffineCompact2f = se2_exp(&Vector3::new(3.0, -1.0, 0.2));
        let tap: [f32; 2] = [-3.5, 2.5];
        let warped: [f32; 2] = warp.warp_tap(tap);
        let expected: Vector2<f32> = warp.linear * Vector2::new(tap[0], tap[1]) + warp.translation;
        assert_abs_diff_eq!(warped[0], expected.x, epsilon = 1e-6);
        assert_abs_diff_eq!(warped[1], expected.y, epsilon = 1e-6);
    }
}
