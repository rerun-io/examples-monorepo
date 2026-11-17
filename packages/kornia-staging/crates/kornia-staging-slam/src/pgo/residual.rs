//! The relative-pose residual and its analytic Jacobians.

use kornia_staging_algebra::lie::{right_jacobian_inv_so3, RigidTransform, Rotation3};
use kornia_staging_algebra::Scalar;
use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};

/// A linearized relative-pose residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelPoseLinearization<S: Scalar> {
    /// `logd(from_t_to_measured^-1 * world_from_a^-1 * world_from_b)`, translation first.
    pub residual: Vector6<S>,
    /// Derivative with respect to the increment of `world_from_a`.
    pub d_from: Matrix6<S>,
    /// Derivative with respect to the increment of `world_from_b`.
    pub d_to: Matrix6<S>,
}

/// Relative-pose residual between two `world_from_body` poses, with analytic Jacobians.
///
/// The residual is the decoupled SE(3) log of the error in the measurement frame,
/// `r = logd(T_ab_meas^-1 * T_wa^-1 * T_wb)` (translation first), as rootba's `relative_pose_error`.
/// Both Jacobians are with respect to [`RigidTransform::apply_inc`] (`t += dt`, `R = exp(dw) R`),
/// which is RootSLAM's Ceres parameterization of its PGO (Euclidean position, left-multiplied
/// quaternion manifold). Ceres' `PoseGraph3dErrorTerm` is the same residual rotated by `R_ab_meas`
/// with `2 vec(q)` for the rotation, so with an isotropic information per 3-block both costs agree to
/// second order in the rotation error.
///
/// # Arguments
/// * `from_t_to` - Measured `a_from_b`.
/// * `world_from_a` - Pose of the edge's first keyframe.
/// * `world_from_b` - Pose of the edge's second keyframe.
pub fn relative_pose_residual<S: Scalar>(
    from_t_to: &RigidTransform<S>,
    world_from_a: &RigidTransform<S>,
    world_from_b: &RigidTransform<S>,
) -> RelPoseLinearization<S> {
    let r_meas_t: Matrix3<S> = from_t_to.rotation.inverse().matrix();
    let r_a_t: Matrix3<S> = world_from_a.rotation.inverse().matrix();
    let r_b_t: Matrix3<S> = world_from_b.rotation.inverse().matrix();
    let delta: Vector3<S> = world_from_b.translation - world_from_a.translation;

    // E = T_meas^-1 T_a^-1 T_b = (R_m^T R_a^T R_b, R_m^T (R_a^T (p_b - p_a) - t_m)).
    let error_rotation =
        from_t_to.rotation.inverse() * world_from_a.rotation.inverse() * world_from_b.rotation;
    let r_t: Vector3<S> = r_meas_t * (r_a_t * delta - from_t_to.translation);
    let r_w: Vector3<S> = error_rotation.log();

    let mr: Matrix3<S> = r_meas_t * r_a_t;
    let jr_inv_rb: Matrix3<S> = right_jacobian_inv_so3(&r_w) * r_b_t;

    let mut d_from = Matrix6::zeros();
    d_from.fixed_view_mut::<3, 3>(0, 0).copy_from(&-mr);
    d_from
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(mr * Rotation3::hat(&delta)));
    d_from.fixed_view_mut::<3, 3>(3, 3).copy_from(&-jr_inv_rb);

    let mut d_to = Matrix6::zeros();
    d_to.fixed_view_mut::<3, 3>(0, 0).copy_from(&mr);
    d_to.fixed_view_mut::<3, 3>(3, 3).copy_from(&jr_inv_rb);

    RelPoseLinearization {
        residual: Vector6::new(r_t.x, r_t.y, r_t.z, r_w.x, r_w.y, r_w.z),
        d_from,
        d_to,
    }
}
