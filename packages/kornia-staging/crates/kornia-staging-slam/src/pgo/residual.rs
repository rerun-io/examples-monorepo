//! The relative-pose residual and its analytic Jacobians.

use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use kornia_staging_algebra::Scalar;
use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};

/// A linearized relative-pose residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelPoseLinearization<S: Scalar> {
    /// `[R_a^T (p_b - p_a) - t_ab; 2 vec(q_ab q_est^-1)]`, translation first (see [`relative_pose_residual`]).
    pub residual: Vector6<S>,
    /// Derivative with respect to the increment of `world_from_a`.
    pub d_from: Matrix6<S>,
    /// Derivative with respect to the increment of `world_from_b`.
    pub d_to: Matrix6<S>,
}

/// Relative-pose residual between two `world_from_body` poses, with analytic Jacobians.
///
/// This is RootSLAM's PGO term, the Ceres `PoseGraph3dErrorTerm` that `loop_closing.cpp` (`buildCeresProblem`)
/// adds for every pose-graph edge, value for value:
///
/// `r = [R_a^T (p_b - p_a) - t_ab; 2 vec(q_ab * q_est^-1)]` with `q_est = q_a^-1 q_b`.
///
/// The translation row is the position error in frame `a`; the rotation row is twice the vector part of the
/// rotation error quaternion (no sign canonicalization, as in Ceres; the cost does not depend on the sign).
/// Both Jacobians are with respect to [`RigidTransform::apply_inc`] (`t += dt`, `R = exp(dw) R`), which is
/// the parameterization of RootSLAM's Ceres PGO (Euclidean position, left-multiplied quaternion manifold).
///
/// It is *not* rootba's GBA pose-graph row `logd(T_ab_meas * T_ab_est^-1)` (`rel_pose_constraint.hpp`):
/// there the translation row is `t_ab - R_ab R_est^T t_est`, which carries the rotation error times the
/// edge length (5 cm on a 5 m edge with 0.01 rad of yaw error), where this row carries none. The GBA rows
/// that must match rootba's `bal` need their own residual.
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
    let r_a_t: Matrix3<S> = world_from_a.rotation.inverse().matrix();
    let delta: Vector3<S> = world_from_b.translation - world_from_a.translation;
    let r_t: Vector3<S> = r_a_t * delta - from_t_to.translation;

    // q_err = q_ab q_est^-1 = q_ab q_b^-1 q_a. A left increment exp(w) of q_a (or of q_b, with -w) turns it into
    // q_err * [1, R_a^T w / 2], whose vector part moves by (q_err.w I + hat(q_err.vec)) R_a^T w / 2.
    let q_err = *from_t_to.rotation.quaternion()
        * world_from_b.rotation.quaternion().inverse()
        * *world_from_a.rotation.quaternion();
    let two = S::one() + S::one();
    let vec: Vector3<S> = q_err.imag();
    let r_w: Vector3<S> = vec * two;
    let d_rot: Matrix3<S> = (Matrix3::identity() * q_err.w + Rotation3::hat(&vec)) * r_a_t;

    let mut d_from = Matrix6::zeros();
    d_from.fixed_view_mut::<3, 3>(0, 0).copy_from(&-r_a_t);
    d_from
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(r_a_t * Rotation3::hat(&delta)));
    d_from.fixed_view_mut::<3, 3>(3, 3).copy_from(&d_rot);

    let mut d_to = Matrix6::zeros();
    d_to.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_a_t);
    d_to.fixed_view_mut::<3, 3>(3, 3).copy_from(&-d_rot);

    RelPoseLinearization {
        residual: Vector6::new(r_t.x, r_t.y, r_t.z, r_w.x, r_w.y, r_w.z),
        d_from,
        d_to,
    }
}
