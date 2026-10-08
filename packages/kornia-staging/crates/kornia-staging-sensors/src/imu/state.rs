//! Numerical pose and velocity values, without estimator lifecycle.
use kornia_staging_algebra::{lie::RigidTransform, Scalar};
use nalgebra::{SVector, Vector3};
type Vector9<S> = SVector<S, 9>;
/// Pose and world-frame velocity at a timestamp.
/// Preintegrated delta states use elapsed nanoseconds instead of absolute time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavState<S: Scalar> {
    /// Timestamp of the state, in nanoseconds.
    pub timestamp_ns: i64,
    /// Pose of the IMU (rig) frame in the world frame.
    pub t_w_i: RigidTransform<S>,
    /// Linear velocity in the world frame, m/s.
    pub vel_w_i: Vector3<S>,
}

impl<S: Scalar> Default for NavState<S> {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            t_w_i: RigidTransform::identity(),
            vel_w_i: Vector3::zeros(),
        }
    }
}

impl<S: Scalar> NavState<S> {
    /// A pose-velocity state from its parts.
    pub fn new(timestamp_ns: i64, t_w_i: RigidTransform<S>, vel_w_i: Vector3<S>) -> Self {
        Self {
            timestamp_ns,
            t_w_i,
            vel_w_i,
        }
    }

    /// Apply a 9-vector increment, `NavState::applyInc`.
    ///
    /// The layout is `[trans(3), rot(3), vel(3)]`; the pose goes through
    /// [`RigidTransform::apply_inc`] and the velocity is added.
    pub fn apply_inc(&mut self, inc: &Vector9<S>) {
        self.t_w_i.apply_inc(&inc.fixed_rows::<6>(0).into_owned());
        self.vel_w_i += inc.fixed_rows::<3>(6);
    }

    /// The increment that takes `self` to `other`, `NavState::diff`
    /// the inverse of [`NavState::apply_inc`].
    ///
    /// No production caller: the estimator's states are 15-dof and use
    /// the full-state difference. This is the 9-dof one, and it is what the
    /// preintegration's finite-difference tests measure their Jacobians with —
    /// the residual they check is `delta_state.diff(propagated)`.
    pub fn diff(&self, other: &Self) -> Vector9<S> {
        let mut res: Vector9<S> = Vector9::zeros();
        res.fixed_rows_mut::<3>(0)
            .copy_from(&(other.t_w_i.translation - self.t_w_i.translation));
        res.fixed_rows_mut::<3>(3)
            .copy_from(&(other.t_w_i.rotation * self.t_w_i.rotation.inverse()).log());
        res.fixed_rows_mut::<3>(6)
            .copy_from(&(other.vel_w_i - self.vel_w_i));
        res
    }
}
