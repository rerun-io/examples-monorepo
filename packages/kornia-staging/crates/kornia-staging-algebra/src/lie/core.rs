//! Scalar-preserving arithmetic shared by the concrete Kornia APIs.
use super::Rotation3;
use crate::Scalar;
use nalgebra::{Matrix3, Vector3};

pub(super) fn hat<S: Scalar>(v: &Vector3<S>) -> Matrix3<S> {
    let zero = S::zero();
    Matrix3::new(zero, -v.z, v.y, v.z, zero, -v.x, -v.y, v.x, zero)
}

/// Coupled SE(3) exponential with translation first.
#[inline]
pub(super) fn exp<S: Scalar>(
    translation: &Vector3<S>,
    omega: &Vector3<S>,
) -> (Rotation3<S>, Vector3<S>) {
    let theta_sq = omega.norm_squared();
    let epsilon = S::SOPHUS_EPSILON;
    let theta = if theta_sq < epsilon * epsilon {
        S::zero()
    } else {
        theta_sq.sqrt()
    };
    let rotation = Rotation3::exp(omega);
    let v = sophus_left_jacobian_so3(omega, theta);
    (rotation, v * translation)
}

/// Coupled SE(3) logarithm for a rotation and translation.
#[inline]
pub(super) fn log<S: Scalar>(
    rotation: &Rotation3<S>,
    translation: &Vector3<S>,
) -> (Vector3<S>, Vector3<S>) {
    let (omega, theta) = log_and_theta(rotation);
    let v_inv = sophus_left_jacobian_inv_so3(&omega, theta);
    (v_inv * translation, omega)
}

fn log_and_theta<S: Scalar>(rotation: &Rotation3<S>) -> (Vector3<S>, S) {
    let q = rotation.quaternion();
    let squared_n = q.vector().norm_squared();
    let w = q.w;
    let epsilon = S::SOPHUS_EPSILON;
    let omega = rotation.log();
    let theta = if squared_n < epsilon * epsilon {
        S::from_literal(2.0) * squared_n / w
    } else if w < S::zero() {
        -omega.norm()
    } else {
        omega.norm()
    };
    (omega, theta)
}

/// Right SO(3) Jacobian: `exp(phi + eps) ~ exp(phi) exp(J eps)`.
pub fn right_jacobian_so3<S: Scalar>(phi: &Vector3<S>) -> Matrix3<S> {
    let phi_norm2: S = phi.norm_squared();
    let phi_hat: Matrix3<S> = hat(phi);
    let phi_hat2: Matrix3<S> = phi_hat * phi_hat;

    let mut j: Matrix3<S> = Matrix3::identity();
    if phi_norm2 > S::SOPHUS_EPSILON {
        let phi_norm: S = phi_norm2.sqrt();
        let phi_norm3: S = phi_norm2 * phi_norm;
        j -= phi_hat * ((S::from_literal(1.0) - phi_norm.cos()) / phi_norm2);
        j += phi_hat2 * ((phi_norm - phi_norm.sin()) / phi_norm3);
    } else {
        // Taylor expansion around 0.
        j -= phi_hat / S::from_literal(2.0);
        j += phi_hat2 / S::from_literal(6.0);
    }
    j
}

/// Inverse right SO(3) Jacobian: `log(exp(phi) exp(eps)) ~ phi + J eps`.
/// Over-rotated inputs use the same zeroth-order branch as pi, without panic.
pub fn right_jacobian_inv_so3<S: Scalar>(phi: &Vector3<S>) -> Matrix3<S> {
    let phi_norm2: S = phi.norm_squared();
    let phi_hat: Matrix3<S> = hat(phi);
    let phi_hat2: Matrix3<S> = phi_hat * phi_hat;

    let mut j: Matrix3<S> = Matrix3::identity() + phi_hat / S::from_literal(2.0);
    j += inverse_jacobian_second_order_term(&phi_hat2, phi_norm2);
    j
}

/// Inverse left SO(3) Jacobian: `log(exp(eps) exp(phi)) ~ phi + J eps`.
pub fn left_jacobian_inv_so3<S: Scalar>(phi: &Vector3<S>) -> Matrix3<S> {
    let phi_norm2: S = phi.norm_squared();
    let phi_hat: Matrix3<S> = hat(phi);
    let phi_hat2: Matrix3<S> = phi_hat * phi_hat;

    let mut j: Matrix3<S> = Matrix3::identity() - phi_hat / S::from_literal(2.0);
    j += inverse_jacobian_second_order_term(&phi_hat2, phi_norm2);
    j
}

/// The `hat(phi)^2` term in both inverse SO(3) Jacobians.
/// Use the closed form on `(0, pi)`, a zeroth-order expansion at pi where sine
/// vanishes, and `1/12` at zero. Thresholds and denominators use the input scalar.
fn inverse_jacobian_second_order_term<S: Scalar>(
    phi_hat2: &Matrix3<S>,
    phi_norm2: S,
) -> Matrix3<S> {
    if phi_norm2 <= S::SOPHUS_EPSILON {
        // Taylor expansion around 0.
        return phi_hat2 / S::from_literal(12.0);
    }
    let phi_norm: S = phi_norm2.sqrt();
    let pi: S = S::from_literal(std::f64::consts::PI);
    let threshold: S = pi - S::sophus_epsilon_sqrt();
    if phi_norm < threshold {
        // Regular case on (0, pi).
        phi_hat2
            * (S::from_literal(1.0) / phi_norm2
                - (S::from_literal(1.0) + phi_norm.cos())
                    / (S::from_literal(2.0) * phi_norm * phi_norm.sin()))
    } else {
        // 0th-order Taylor expansion around pi.
        phi_hat2 / (pi * pi)
    }
}

/// Left Jacobian for coupled SE(3) exp. Its Taylor branch stops at
/// `I + Omega/2`, unlike the standalone Jacobian's additional `Omega^2/6` term.
fn sophus_left_jacobian_so3<S: Scalar>(omega: &Vector3<S>, theta: S) -> Matrix3<S> {
    let theta_sq: S = theta * theta;
    let big_omega: Matrix3<S> = hat(omega);
    let epsilon: S = S::SOPHUS_EPSILON;

    if theta_sq < epsilon * epsilon {
        Matrix3::identity() + big_omega * S::from_literal(0.5)
    } else {
        Matrix3::identity()
            + big_omega * ((S::from_literal(1.0) - theta.cos()) / theta_sq)
            + big_omega * big_omega * ((theta - theta.sin()) / (theta_sq * theta))
    }
}

/// Sophus's own inverse left Jacobian, `SO3::leftJacobianInverse`
/// Used only by SE(3) log.
fn sophus_left_jacobian_inv_so3<S: Scalar>(omega: &Vector3<S>, theta: S) -> Matrix3<S> {
    let theta_sq: S = theta * theta;
    let big_omega: Matrix3<S> = hat(omega);
    let epsilon: S = S::SOPHUS_EPSILON;

    let identity: Matrix3<S> = Matrix3::identity();
    if theta_sq < epsilon * epsilon {
        identity - big_omega * S::from_literal(0.5)
            + big_omega * big_omega * S::from_literal(1.0 / 12.0)
    } else {
        let half_theta: S = S::from_literal(0.5) * theta;
        let factor: S = (S::from_literal(1.0)
            - S::from_literal(0.5) * theta * half_theta.cos() / half_theta.sin())
            / (theta * theta);
        identity - big_omega * S::from_literal(0.5) + big_omega * big_omega * factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn logarithm_keeps_signed_and_small_angle_theta() {
        for angle in [std::f64::consts::PI - 1e-6, std::f64::consts::PI] {
            let (omega, theta) = log_and_theta(&Rotation3::exp(&Vector3::new(0.0, 0.0, angle)));
            assert_abs_diff_eq!(theta, angle, epsilon = 1e-12);
            assert_abs_diff_eq!(theta, omega.norm(), epsilon = 1e-15);
        }
        let (omega, theta) = log_and_theta(&Rotation3::exp(&Vector3::new(
            0.0,
            0.0,
            std::f64::consts::PI + 0.5,
        )));
        assert_abs_diff_eq!(omega.norm(), std::f64::consts::PI - 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(theta, -omega.norm(), epsilon = 1e-15);
        let (omega, theta) = log_and_theta(&Rotation3::exp(&Vector3::new(0.0, 0.0, 1e-12)));
        assert_abs_diff_eq!(omega.norm(), 1e-12, epsilon = 1e-24);
        assert_abs_diff_eq!(theta, 5e-25, epsilon = 1e-27);
    }
}
