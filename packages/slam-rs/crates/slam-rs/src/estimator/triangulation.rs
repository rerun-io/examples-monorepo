//! Initialize landmarks from unconnected observations in the window.

use super::{EstimatorError, FlowObservations, SqrtKeypointVio, cast_pixel};
use crate::ba_base::triangulate;
use crate::camera::CameraEnum;
use crate::landmark::{Landmark, StereographicParam};
use crate::lie::{LieScalar, Se3};
use crate::types::{KeypointId, LandmarkId, TimeCamId};
use nalgebra::{Vector2, Vector3, Vector4};
use std::collections::{BTreeMap, BTreeSet};

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// Triangulate unconnected observations into landmarks hosted by this frameset.
    ///
    /// Gather all observations of each id in the live window and try second views
    /// in `TimeCamId` order. Unproject both pixels, form
    /// `T_0_1 = T_i_c[host]⁻¹ · T_i0_i1 · T_i_c[other]`, reject short baselines,
    /// and accept finite DLT results only when `0 < inv_dist < 3`.
    /// On acceptance, file all gathered observations, not just the successful pair.
    /// Each camera may host landmarks.
    ///
    /// All camera ids have been validated against the rig. Host pose and calibration
    /// stay constant throughout the loop, so their inverses are computed once.
    pub(super) fn triangulate_unconnected(
        &mut self,
        frame: &FlowObservations,
        unconnected_obs: &[BTreeSet<KeypointId>],
    ) -> Result<usize, EstimatorError> {
        debug_assert_eq!(unconnected_obs.len(), self.ba.calib.t_i_c.len());
        // the squared threshold is formed in `double` and cast, so the
        // `f32` instantiation compares against `(float)(0.05 * 0.05)`.
        let min_triang_distance2: S = S::from_literal(
            self.config.vio_min_triangulation_dist * self.config.vio_min_triangulation_dist,
        );
        let mut num_points_added: usize = 0;
        // 's `T_i0_inv`: the host is this frameset, for every landmark
        // and every pair.
        let t_i0_inv: Se3<S> = self
            .ba
            .get_pose_state_with_lin(frame.t_ns)?
            .pose()
            .inverse();

        for (cam_id, ids) in unconnected_obs.iter().enumerate() {
            let tcidl: TimeCamId = TimeCamId::new(frame.t_ns, cam_id);
            let host_keypoints: &BTreeMap<KeypointId, Vector2<f32>> = &frame.cameras[cam_id];
            let cam0: CameraEnum<S> = self.ba.cameras()[cam_id];
            let t_i_c0_inv: Se3<S> = self.ba.calib.t_i_c[cam_id].inverse();
            for kpt_id in ids {
                let lm_id: LandmarkId = LandmarkId::from(*kpt_id);
                // another camera of this frameset may have hosted it
                // already.
                if self.ba.lmdb.landmark_exists(lm_id) {
                    continue;
                }
                // 's `.at(lm_id)`: `measure` took this id out of
                // `host_keypoints` itself, so a miss is an invariant break, not
                // a landmark to skip (D32).
                let Some(p0_pixel) = host_keypoints.get(kpt_id) else {
                    return Err(EstimatorError::UnconnectedKeypointMissing {
                        cam_id,
                        kpt_id: *kpt_id,
                    });
                };
                let p0: Vector2<S> = cast_pixel::<S>(p0_pixel);

                // Visit every image of this id in the live window in `TimeCamId` order.
                let mut kp_obs: BTreeMap<TimeCamId, Vector2<S>> = BTreeMap::new();
                for (other_t_ns, other) in &self.prev_opt_flow_res {
                    for (other_cam, keypoints) in other.cameras.iter().enumerate() {
                        if let Some(pixel) = keypoints.get(kpt_id) {
                            kp_obs.insert(
                                TimeCamId::new(*other_t_ns, other_cam),
                                cast_pixel::<S>(pixel),
                            );
                        }
                    }
                }

                let mut accepted: Option<Landmark<S>> = None;
                for (tcido, p1) in &kp_obs {
                    // an unprojection the camera rejects skips this
                    // pair, not the landmark.
                    let mut p0_3d: Vector4<S> = Vector4::zeros();
                    let mut p1_3d: Vector4<S> = Vector4::zeros();
                    let cam1: CameraEnum<S> = self.ba.cameras()[tcido.cam_id];
                    let valid0: bool = cam0.unproject(&p0, &mut p0_3d);
                    let valid1: bool = cam1.unproject(p1, &mut p1_3d);
                    if !valid0 || !valid1 {
                        continue;
                    }

                    let other_pose: Se3<S> =
                        *self.ba.get_pose_state_with_lin(tcido.frame_id)?.pose();
                    let t_i0_i1: Se3<S> = t_i0_inv * other_pose;
                    let t_0_1: Se3<S> = t_i_c0_inv * t_i0_i1 * self.ba.calib.t_i_c[tcido.cam_id];

                    // Require enough squared translation baseline to triangulate.
                    let t: Vector3<S> = t_0_1.translation;
                    let baseline2: S = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
                    if baseline2 < min_triang_distance2 {
                        continue;
                    }

                    // A refused DLT skips this observation pair.
                    let Some(triangulated): Option<Vector4<S>> = triangulate(
                        &Vector3::new(p0_3d[0], p0_3d[1], p0_3d[2]),
                        &Vector3::new(p1_3d[0], p1_3d[1], p1_3d[2]),
                        &t_0_1,
                    ) else {
                        continue;
                    };
                    let finite: bool = triangulated.iter().all(|v| v.is_finite());
                    // What decides this gate is the triangulated value's own
                    // reduction order (D47), not the comparison.
                    let inv_dist: S = triangulated[3];
                    // `3.0` is a `double` literal, so the `f32` instantiation
                    // promotes and compares in `double`.
                    if finite && inv_dist > S::zero() && inv_dist.to_f64() < 3.0 {
                        accepted = Some(Landmark::new(
                            lm_id,
                            tcidl,
                            StereographicParam::project(&triangulated),
                            inv_dist,
                        ));
                        break;
                    }
                }

                if let Some(landmark) = accepted {
                    self.ba.lmdb.add_landmark(lm_id, &landmark);
                    num_points_added += 1;
                    for (tcido, pixel) in &kp_obs {
                        self.ba.lmdb.add_observation(*tcido, lm_id, *pixel)?;
                    }
                }
            }
        }
        Ok(num_points_added)
    }
}
