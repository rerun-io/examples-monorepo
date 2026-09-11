//! Construction of the public window snapshot.

use super::SqrtKeypointVio;
use crate::landmark::StereographicParam;
use crate::lie::{LieScalar, Se3};
use crate::types::{FrameId, LandmarkId, PoseVelBiasState, TimeCamId};
use nalgebra::{Vector3, Vector4};

/// The nine degrees of freedom a 15-dof state carries beyond its pose.
///
/// A pose block (`frame_poses`) has none of them, a state block
/// (`frame_states`) has all three, so they travel as one value rather than as
/// three `Option`s that are always all-`Some` or all-`None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelBias<S: LieScalar> {
    /// World-frame velocity.
    pub vel_w_i: Vector3<S>,
    /// Gyroscope bias.
    pub bias_gyro: Vector3<S>,
    /// Accelerometer bias.
    pub bias_accel: Vector3<S>,
}

/// One window state, as the S9 Rerun rung needs it.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowState<S: LieScalar> {
    /// State timestamp.
    pub t_ns: i64,
    /// `T_w_i`, the rig pose in the world frame.
    pub t_w_i: Se3<S>,
    /// The nine dof beyond the pose, `None` for a pose-only block.
    pub vel_bias: Option<VelBias<S>>,
    /// Whether the linearization point is frozen.
    pub linearized: bool,
    /// Whether this frame is a keyframe.
    pub keyframe: bool,
    /// Whether this frame is a long-term keyframe.
    pub long_term_keyframe: bool,
    /// Monotonic frame index for the UI.
    pub frame_index: Option<usize>,
}

/// One landmark as the V2 rung draws it.
///
/// The host is carried because the rung colours the point cloud by the keyframe
/// that hosts it: the position alone cannot say which frame's bearing it is a
/// distance along, and `lmdb` keys on the host rather than storing it per point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotLandmark<S: LieScalar> {
    /// The landmark's id, which is the id of the keypoint that spawned it.
    pub id: LandmarkId,
    /// Host keyframe and camera: the image the inverse distance is measured from.
    pub host: TimeCamId,
    /// Position in the world frame, metres.
    pub position_w: Vector3<S>,
}

/// The window and its landmarks, for the V2 visual-validation rung (D51).
///
/// `getAllPosesMap`, `get_current_points` and the `VioVisualizationData` fields
///  collapsed into one value the caller reads once per frame. The
/// core logs nothing itself (D03).
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSnapshot<S: LieScalar> {
    /// Frameset timestamp of the newest state.
    pub t_ns: i64,
    /// The 15-dof states, oldest first.
    pub states: Vec<WindowState<S>>,
    /// The pose-only blocks, oldest first.
    pub poses: Vec<WindowState<S>>,
    /// Landmarks the window currently holds, in `lmdb` order.
    pub landmarks: Vec<SnapshotLandmark<S>>,
    /// Frames the last marginalization removed from the window.
    pub marginalized: Vec<FrameId>,
}

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// The window, its landmarks in the world frame and the frames the last
    /// marginalization removed, for the V2 Rerun rung (D51).
    ///
    /// The landmark position is `host_pose * T_i_c * unproject(direction) /
    /// inv_dist`, exactly as the landmark bundle is built, and a
    /// landmark whose host has left the window is skipped as it is there.
    pub fn snapshot(&self) -> WindowSnapshot<S> {
        let states: Vec<WindowState<S>> = self
            .ba
            .frame_states
            .iter()
            .map(|(t_ns, state)| {
                let inner: &PoseVelBiasState<S> = state.state();
                WindowState {
                    t_ns: *t_ns,
                    t_w_i: inner.t_w_i,
                    vel_bias: Some(VelBias {
                        vel_w_i: inner.vel_w_i,
                        bias_gyro: inner.bias_gyro,
                        bias_accel: inner.bias_accel,
                    }),
                    linearized: state.is_linearized(),
                    keyframe: self.kf_ids.contains(t_ns),
                    long_term_keyframe: self.ltkfs.contains(t_ns),
                    frame_index: self.frame_idx.get(t_ns).copied(),
                }
            })
            .collect();
        let poses: Vec<WindowState<S>> = self
            .ba
            .frame_poses
            .iter()
            .map(|(t_ns, pose)| WindowState {
                t_ns: *t_ns,
                t_w_i: *pose.pose(),
                vel_bias: None,
                linearized: pose.is_linearized(),
                keyframe: self.kf_ids.contains(t_ns),
                long_term_keyframe: self.ltkfs.contains(t_ns),
                frame_index: self.frame_idx.get(t_ns).copied(),
            })
            .collect();

        let mut landmarks: Vec<SnapshotLandmark<S>> =
            Vec::with_capacity(self.ba.lmdb.num_landmarks());
        for lm in self.ba.lmdb.landmarks() {
            let Ok(host) = self.ba.get_pose_state_with_lin(lm.host_kf_id.frame_id) else {
                continue;
            };
            let Some(t_i_c) = self.ba.calib.t_i_c.get(lm.host_kf_id.cam_id) else {
                continue;
            };
            let t_w_c: Se3<S> = *host.pose() * *t_i_c;
            let bearing: Vector4<S> = StereographicParam::unproject(&lm.direction);
            let scale: S = S::one() / lm.inv_dist;
            let point_c: Vector3<S> =
                Vector3::new(bearing[0] * scale, bearing[1] * scale, bearing[2] * scale);
            landmarks.push(SnapshotLandmark {
                id: lm.id,
                host: lm.host_kf_id,
                position_w: t_w_c * point_c,
            });
        }

        WindowSnapshot {
            t_ns: self.last_state_t_ns,
            states,
            poses,
            landmarks,
            marginalized: self.last_marginalized.clone(),
        }
    }
}
