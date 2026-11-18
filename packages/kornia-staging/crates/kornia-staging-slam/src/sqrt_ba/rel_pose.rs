//! Relative-pose (pose-graph) rows in rootba's own residual form.
//!
//! Keyframes hold `T_i_w` (world to body) and move by the decoupled left
//! increment `T_i_w <- expd(xi) T_i_w`, `xi = [rho; phi]` (translation first).
//! With `T_a_b_est = T_a_w T_b_w^-1` the residual is
//! `logd(T_a_b_meas T_a_b_est^-1)` (rootba `rel_pose_constraint.hpp`).
//! Its translation row carries the rotation error times the edge length, so it
//! differs from RootSLAM's PGO term (Ceres `PoseGraph3dErrorTerm`).
use kornia_staging_algebra::{
    lie::{right_jacobian_inv_so3, RigidTransform, Rotation3},
    Scalar,
};
use nalgebra::{Matrix6, Vector6};

/// A relative pose measurement between two keyframes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelPoseConstraint<S: Scalar> {
    /// Index of keyframe `a`.
    pub frame_a: usize,
    /// Index of keyframe `b`.
    pub frame_b: usize,
    /// Measured pose of `b`'s body frame in `a`'s body frame.
    pub a_from_b: RigidTransform<S>,
    /// Square-root information; the whitened residual is `sqrt_info * res`.
    pub sqrt_info: Matrix6<S>,
}

impl<S: Scalar> RelPoseConstraint<S> {
    /// A constraint with independent translation (map units) and rotation (rad) sigmas.
    pub fn with_sigmas(
        frame_a: usize,
        frame_b: usize,
        a_from_b: RigidTransform<S>,
        sigma_translation: S,
        sigma_rotation: S,
    ) -> Self {
        let mut sqrt_info = Matrix6::zeros();
        for i in 0..3 {
            sqrt_info[(i, i)] = S::one() / sigma_translation;
            sqrt_info[(i + 3, i + 3)] = S::one() / sigma_rotation;
        }
        Self {
            frame_a,
            frame_b,
            a_from_b,
            sqrt_info,
        }
    }

    /// The constraint in another scalar.
    pub fn cast<T: Scalar>(&self) -> RelPoseConstraint<T> {
        RelPoseConstraint {
            frame_a: self.frame_a,
            frame_b: self.frame_b,
            a_from_b: self.a_from_b.cast(),
            sqrt_info: self.sqrt_info.map(|v| T::from_literal(v.to_f64())),
        }
    }
}

/// `logd(T_a_b_meas (T_a_w T_b_w^-1)^-1)` and its Jacobians against the two
/// decoupled left increments.
///
/// ```
/// use kornia_staging_algebra::lie::RigidTransform;
/// use kornia_staging_slam::sqrt_ba::relative_pose_error;
/// let pose = RigidTransform::<f64>::identity();
/// let res = relative_pose_error(&pose, &pose, &pose, None, None);
/// assert_eq!(res, nalgebra::Vector6::zeros());
/// ```
pub fn relative_pose_error<S: Scalar>(
    a_from_b_meas: &RigidTransform<S>,
    a_from_w: &RigidTransform<S>,
    b_from_w: &RigidTransform<S>,
    d_res_d_xi_a: Option<&mut Matrix6<S>>,
    d_res_d_xi_b: Option<&mut Matrix6<S>>,
) -> Vector6<S> {
    let a_from_b_est: RigidTransform<S> = *a_from_w * b_from_w.inverse();
    let err: RigidTransform<S> = *a_from_b_meas * a_from_b_est.inverse();
    let res: Vector6<S> = err.log_decoupled();
    if d_res_d_xi_a.is_none() && d_res_d_xi_b.is_none() {
        return res;
    }
    // basalt `rightJacobianInvSE3Decoupled`: blkdiag(exp(omega), Jr^-1(omega)).
    let omega = res.fixed_rows::<3>(3).into_owned();
    let mut j: Matrix6<S> = Matrix6::zeros();
    j.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&Rotation3::exp(&omega).matrix());
    j.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&right_jacobian_inv_so3(&omega));
    if let Some(out) = d_res_d_xi_a {
        *out = -j;
    }
    if let Some(out) = d_res_d_xi_b {
        // Decoupled adjoint of T_a_b_est: T_b_a expd(xi) T_a_b = expd(adj xi).
        *out = j * a_from_b_est.adjoint();
    }
    res
}

/// Apply the decoupled left increment `T <- expd(xi) T` used by every GBA pose.
pub fn apply_left_increment<S: Scalar>(
    pose: &RigidTransform<S>,
    xi: &Vector6<S>,
) -> RigidTransform<S> {
    RigidTransform::exp_decoupled(xi) * *pose
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(seed: f64) -> RigidTransform<f64> {
        let xi = Vector6::from_fn(|i, _| ((i as f64 + 1.0) * 1.7 + seed * 3.1).sin() * 0.8);
        let mut p = RigidTransform::exp_decoupled(&xi);
        p.translation *= 4.0;
        p
    }

    #[test]
    fn jacobians_match_central_differences() {
        for seed in 0..16 {
            let a = pose(seed as f64);
            let b = pose(seed as f64 + 0.37);
            // A measurement near the estimate, as on a real edge.
            let noise = Vector6::from_fn(|i, _| ((i + seed) as f64).cos() * 0.05);
            let meas = apply_left_increment(&(a * b.inverse()), &noise);
            let (mut ja, mut jb) = (Matrix6::zeros(), Matrix6::zeros());
            relative_pose_error(&meas, &a, &b, Some(&mut ja), Some(&mut jb));
            let h = 1e-6;
            for k in 0..6 {
                let mut d = Vector6::zeros();
                d[k] = h;
                let na =
                    (relative_pose_error(&meas, &apply_left_increment(&a, &d), &b, None, None)
                        - relative_pose_error(
                            &meas,
                            &apply_left_increment(&a, &(-d)),
                            &b,
                            None,
                            None,
                        ))
                        / (2.0 * h);
                let nb =
                    (relative_pose_error(&meas, &a, &apply_left_increment(&b, &d), None, None)
                        - relative_pose_error(
                            &meas,
                            &a,
                            &apply_left_increment(&b, &(-d)),
                            None,
                            None,
                        ))
                        / (2.0 * h);
                assert!(
                    (na - ja.column(k)).amax() < 1e-7,
                    "a col {k}: {na} vs {}",
                    ja.column(k)
                );
                assert!(
                    (nb - jb.column(k)).amax() < 1e-7,
                    "b col {k}: {nb} vs {}",
                    jb.column(k)
                );
            }
        }
    }

    #[test]
    fn translation_row_carries_rotation_error_times_edge_length() {
        // rootba's form: a 5 m edge with a 0.01 rad yaw error moves the
        // translation row by ~5 cm (the PGO Ceres term would not).
        let a = RigidTransform::<f64>::identity();
        let b_in_a =
            RigidTransform::new(Rotation3::identity(), nalgebra::Vector3::new(5.0, 0.0, 0.0));
        let b = b_in_a.inverse();
        let yaw = RigidTransform::new(
            Rotation3::exp(&nalgebra::Vector3::new(0.0, 0.0, 0.01)),
            nalgebra::Vector3::zeros(),
        );
        let meas = b_in_a * yaw;
        let res = relative_pose_error(&meas, &a, &b, None, None);
        assert!((res[1].abs() - 0.05).abs() < 1e-3, "{res}");
        assert!((res[5].abs() - 0.01).abs() < 1e-9);
        let c = RelPoseConstraint::with_sigmas(0, 1, meas, 0.05, 0.01);
        let whitened = (c.sqrt_info * res).norm_squared();
        assert!((whitened - 2.0).abs() < 0.05, "{whitened}");
    }
}
