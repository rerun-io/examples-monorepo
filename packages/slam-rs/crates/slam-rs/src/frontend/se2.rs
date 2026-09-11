//! The KLT tracker's 2-D affine warp and SE(2) exponential.
//! A 2x2 linear part and translation compose as the first two rows of a
//! homogeneous matrix product. Tracking needs only the exponential.
//! The frontend uses f32; f64 supports sharp finite-difference Jacobian tests.

use nalgebra::{Matrix2, Vector2, Vector3};

use crate::lie::LieScalar;

/// A 2x2 linear part and a translation, stored as separate blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineCompact2<S: LieScalar> {
    /// `transform.linear()`, the 2x2 block.
    pub linear: Matrix2<S>,
    /// `transform.translation()`, the last column.
    pub translation: Vector2<S>,
}

/// The frontend's f32 affine warp.
pub type AffineCompact2f = AffineCompact2<f32>;

impl<S: LieScalar> Default for AffineCompact2<S> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<S: LieScalar> AffineCompact2<S> {
    /// Identity warp.
    pub fn identity() -> Self {
        Self {
            linear: Matrix2::identity(),
            translation: Vector2::zeros(),
        }
    }

    /// An identity rotation at `position`, as `addPointsForCamera` builds one
    pub fn at(position: Vector2<S>) -> Self {
        Self {
            linear: Matrix2::identity(),
            translation: position,
        }
    }

    /// Compose warps: `linear = self.linear * other.linear` and
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

/// SE(2) exponential for tangent `(t_x, t_y, theta)`.
/// Normalize the sine/cosine pair before using it in the translation factor.
/// The small-angle thresholds are `1e-10` in f64 and `1e-5` in f32.
/// Translation is `V(theta) * upsilon`, evaluated as two scalar expressions.
#[inline]
pub fn se2_exp<S: LieScalar>(tangent: &Vector3<S>) -> AffineCompact2<S> {
    let one: S = S::one();
    let theta: S = tangent[2];
    // `SO2<Scalar>::exp(theta)` — cos/sin, then normalise.
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
