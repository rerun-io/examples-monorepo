//! Per-frame poses re-anchored on the corrected keyframes (RootSLAM §IV;
//! DESIGN §2, §3 row 15, §4 B13, §8, PR f1).
//!
//! The window never reads the map (DESIGN §11 Q3), so the raw VIO pose of a
//! frame keeps the drift it had when it was tracked. [`reanchor`] moves the
//! frames onto the keyframes the back end corrected, two-sided and non-causal:
//! each frame between keyframes `i` and `i+1` takes both keyframes'
//! corrections and blends the two results, `T_left = T_i_opt · T_i_raw⁻¹ ·
//! T_f_raw`, `T_right = T_i+1_opt · T_i+1_raw⁻¹ · T_f_raw`, translation lerped
//! and rotation slerped at `α = (t_f − t_i) / (t_i+1 − t_i)`. This is
//! `merge_frames_keyframes.py:63-150`. A keyframe's raw pose is the frame's own
//! raw pose, so a keyframe lands on its optimized pose exactly.
//!
//! There is no online corrected pose (DESIGN §1, §8, f1 review): the pose-graph
//! gauge fixes the window and the newest map keyframe, so a live frame's
//! correction is the identity and the pose `track` returns is already the
//! corrected pose of the current frame. The corrected history so far is
//! [`super::MapSnapshot::reanchor`] on the frames tracked up to the newest map
//! keyframe (all of them after `Vio::flush`).
//!
//! B13: RootSLAM's online path (`vio.cpp:642-680`) interpolates the two
//! keyframe poses and drops the frame's own raw motion; [`reanchor`] moves the
//! frame by the relative correction instead.

use crate::lie::{Se3, So3};

/// Why [`reanchor`] refused its inputs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnchorError {
    /// Frame or keyframe timestamps must increase strictly.
    #[error("{what} timestamps must increase strictly: {t_ns} ns follows {previous_t_ns} ns")]
    NotIncreasing {
        /// `"frame"` or `"keyframe"`.
        what: &'static str,
        /// The earlier timestamp.
        previous_t_ns: i64,
        /// The one that does not follow it.
        t_ns: i64,
    },
    /// A keyframe is not one of the frames, so it has no raw pose.
    #[error("keyframe at {t_ns} ns is not one of the frames")]
    KeyframeNotAFrame {
        /// The keyframe.
        t_ns: i64,
    },
    /// A frame's quaternion is zero or not finite, so it holds no rotation.
    #[error("no unit quaternion at {t_ns} ns")]
    NoRotation {
        /// The frame.
        t_ns: i64,
    },
}

fn strictly_increasing(what: &'static str, poses: &[(i64, Se3<f64>)]) -> Result<(), AnchorError> {
    match poses.windows(2).find(|pair| pair[1].0 <= pair[0].0) {
        Some(pair) => Err(AnchorError::NotIncreasing {
            what,
            previous_t_ns: pair[0].0,
            t_ns: pair[1].0,
        }),
        None => Ok(()),
    }
}

/// The final, non-causal trajectory: every frame re-anchored on the optimized
/// keyframes on both sides (module docs). Frames before the first keyframe or
/// after the last take that keyframe's correction alone; with no keyframe the
/// frames come back unchanged.
///
/// `frames` are the raw per-frame poses and `keyframes` the optimized keyframe
/// poses, each `(t_ns, T_w_i)` in strictly increasing time; every keyframe must
/// be one of the frames.
///
/// # Errors
///
/// [`AnchorError`] when the timestamps do not increase or a keyframe is not a
/// frame.
pub fn reanchor(
    frames: &[(i64, Se3<f64>)],
    keyframes: &[(i64, Se3<f64>)],
) -> Result<Vec<Se3<f64>>, AnchorError> {
    strictly_increasing("frame", frames)?;
    strictly_increasing("keyframe", keyframes)?;
    // `T_opt · T_raw⁻¹` per keyframe, the left factor of both candidates.
    let corrections: Vec<(i64, Se3<f64>)> = keyframes
        .iter()
        .map(|&(t_ns, optimized)| {
            let row: usize = frames
                .binary_search_by_key(&t_ns, |&(frame_t_ns, _)| frame_t_ns)
                .map_err(|_| AnchorError::KeyframeNotAFrame { t_ns })?;
            Ok((t_ns, optimized * frames[row].1.inverse()))
        })
        .collect::<Result<_, AnchorError>>()?;
    if corrections.is_empty() {
        return Ok(frames.iter().map(|&(_, pose)| pose).collect());
    }
    Ok(frames
        .iter()
        .map(|&(t_ns, raw)| {
            let next: usize = corrections.partition_point(|&(kf_t_ns, _)| kf_t_ns < t_ns);
            if let Some(&(kf_t_ns, _)) = corrections.get(next)
                && kf_t_ns == t_ns
            {
                return keyframes[next].1;
            }
            if next == 0 {
                return corrections[0].1 * raw;
            }
            let Some(&(t1, right)) = corrections.get(next) else {
                return corrections[next - 1].1 * raw;
            };
            let (t0, left) = corrections[next - 1];
            let alpha: f64 = (t_ns - t0) as f64 / (t1 - t0) as f64;
            interpolate(&(left * raw), &(right * raw), alpha)
        })
        .collect())
}

impl super::MapSnapshot {
    /// [`reanchor`] on this map's keyframes, for frames as
    /// [`crate::VioPose::world_from_rig`] holds them: `(t_ns, [tx, ty, tz, qx,
    /// qy, qz, qw])`.
    ///
    /// Call it after `Vio::flush`, when every map keyframe is one of the
    /// frames. Mid-run, take the snapshot first and pass the frames up to the
    /// newest frame `track` has returned: a keyframe the frames do not reach
    /// fails the whole call.
    ///
    /// # Errors
    ///
    /// As [`reanchor`], and [`AnchorError::NoRotation`] for a zero or
    /// non-finite quaternion.
    pub fn reanchor(&self, frames: &[(i64, [f64; 7])]) -> Result<Vec<[f64; 7]>, AnchorError> {
        let frames: Vec<(i64, Se3<f64>)> = frames
            .iter()
            .map(|&(t_ns, pose)| {
                let rotation = So3::from_quaternion_xyzw(pose[3], pose[4], pose[5], pose[6])
                    .ok_or(AnchorError::NoRotation { t_ns })?;
                Ok((
                    t_ns,
                    Se3::new(rotation, nalgebra::Vector3::new(pose[0], pose[1], pose[2])),
                ))
            })
            .collect::<Result<_, AnchorError>>()?;
        let keyframes: Vec<(i64, Se3<f64>)> = self
            .keyframe_t_ns
            .iter()
            .copied()
            .zip(self.keyframe_poses.iter().copied())
            .collect();
        Ok(reanchor(&frames, &keyframes)?
            .iter()
            .map(pose_to_array)
            .collect())
    }
}

fn pose_to_array(pose: &Se3<f64>) -> [f64; 7] {
    let [x, y, z, w] = pose.rotation.quaternion_xyzw();
    [
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        x,
        y,
        z,
        w,
    ]
}

/// Translation lerped, rotation slerped on the shortest arc: `scipy`'s
/// `Slerp([0, 1], [left, right])(alpha)`.
fn interpolate(left: &Se3<f64>, right: &Se3<f64>, alpha: f64) -> Se3<f64> {
    let relative: So3<f64> = left.rotation.inverse() * right.rotation;
    Se3::new(
        left.rotation * So3::exp(&(relative.log() * alpha)),
        left.translation + (right.translation - left.translation) * alpha,
    )
}

#[cfg(test)]
mod tests;
