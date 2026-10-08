//! Tracking pass submission, collection, and camera geometry.

use super::{FrameToFrameOpticalFlow, NO_RESPONSE, PosePrediction};
use crate::camera::RigCamera;
use crate::config::MatchingGuessType;
use crate::frontend::stages::FrameStages;
use crate::lie::{Se3, So3};
use kornia_staging_imgproc::optical_flow::patch_se2::AffineCompact2f;
use kornia_staging_slam::tracking::optical_flow::{PatchTracker, TrackInput};
use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

impl<
    P: crate::frontend::patterns::ConfiguredPattern,
    F: FrameStages<Tracker: PatchTracker<Pattern = P>>,
> FrameToFrameOpticalFlow<P, F>
{
    /// Mask and predict every camera's inputs independently, then submit them
    /// together. `None` selects camera-zero stereo matches of new detections.
    pub(super) fn prepare_tracks(&mut self, prediction: Option<&PosePrediction>) {
        let Self {
            passes,
            matching_guesses,
            config,
            cameras,
            calib,
            frame,
            new_cam0,
            masks,
            depth_guess,
            host_pool,
            ..
        } = self;
        let predict = |camera: usize,
                       source_camera: usize,
                       position: Vector2<f32>,
                       transform: AffineCompact2f,
                       pose: Se3<f32>,
                       use_depth: bool| {
            let translation = if use_depth {
                let (valid, pixel) = project_between_cams(
                    cameras,
                    &position,
                    *depth_guess,
                    &pose,
                    source_camera,
                    camera,
                );
                if valid {
                    pixel
                } else {
                    Vector2::repeat(-1.0e6)
                }
            } else {
                position
            };
            AffineCompact2f {
                linear: transform.linear,
                translation: translation.into(),
            }
        };
        if let Some(prediction) = prediction {
            let prepare = |(camera, pass): (usize, &mut TrackInput)| {
                let previous = prediction.t_w_i_previous * calib.t_i_c[camera];
                let current = prediction.t_w_i_current * calib.t_i_c[camera];
                let pose = previous.inverse() * current;
                pass.ids.clear();
                pass.positions.clear();
                pass.guesses.clear();
                for (index, id) in frame.cameras[camera].ids.iter().enumerate() {
                    let transform = frame.cameras[camera].transforms.get(index);
                    let position = Vector2::from(transform.translation);
                    if masks[camera].in_bounds(position.x, position.y) {
                        continue;
                    }
                    pass.ids.push(id.0);
                    pass.positions.push(position);
                    pass.guesses
                        .push(&predict(camera, camera, position, transform, pose, true));
                }
            };
            if host_pool
                .install(|| {
                    use rayon::prelude::*;
                    passes.par_iter_mut().enumerate().for_each(prepare);
                })
                .is_none()
            {
                passes.iter_mut().enumerate().for_each(prepare);
            }
        } else {
            // Mask camera zero once. Every destination borrows these same slots.
            let source = &mut passes[0];
            source.ids.clear();
            source.positions.clear();
            source.guesses.clear();
            for (index, id) in new_cam0.ids.iter().enumerate() {
                let transform = new_cam0.transforms.get(index);
                let position = Vector2::from(transform.translation);
                if masks[0].in_bounds(position.x, position.y) {
                    continue;
                }
                source.ids.push(id.0);
                source.positions.push(position);
                source.guesses.push(&transform);
            }
            let use_depth = config.optical_flow_matching_guess_type != MatchingGuessType::SamePixel;
            let prepare = |(lane, guesses): (
                usize,
                &mut kornia_staging_imgproc::optical_flow::patch_tracker::FlowTransforms,
            )| {
                let camera = lane + 1;
                let pose = calib.t_i_c[0].inverse() * calib.t_i_c[camera];
                guesses.clear();
                for index in 0..source.ids.len() {
                    guesses.push(&predict(
                        camera,
                        0,
                        Vector2::from(source.positions.get(index)),
                        source.guesses.get(index),
                        pose,
                        use_depth,
                    ));
                }
            };
            if host_pool
                .install(|| {
                    use rayon::prelude::*;
                    matching_guesses
                        .par_iter_mut()
                        .enumerate()
                        .for_each(prepare);
                })
                .is_none()
            {
                matching_guesses.iter_mut().enumerate().for_each(prepare);
            }
        }
    }

    /// The tail of `trackPoints` for one camera, once its lane has arrived.
    pub(super) fn finish_camera(&mut self, camera: usize) {
        self.finish_track_points(camera, camera);
        self.frame.cameras[camera].clear();
        for (slot, id) in self.tracked_ids.iter().enumerate() {
            // `keypoint_map_2.insert(result.begin(), result.end())`; the
            // cell counts are rebuilt afterwards by `updateCellCounts`.
            self.frame.cameras[camera].set(*id, &self.tracked.get(slot), NO_RESPONSE);
        }
    }

    /// The reading half of `trackPoints`: `masks2` over one collected lane,
    /// leaving the survivors in `tracked_ids` and `tracked`.
    pub(super) fn finish_track_points(&mut self, source: usize, lane: usize) {
        self.tracked_ids.clear();
        self.tracked.clear();
        let pass = &self.passes[source];
        let cam2 = lane;
        let result = self.stages.tracker().result(self.result_slots[lane]);
        for slot in result.tracked() {
            let slot: usize = *slot as usize;
            let transform: AffineCompact2f = result.transform(slot);
            // `if (masks2.inBounds(t2.x(), t2.y())) continue;`.
            if self.masks[cam2].in_bounds(transform.translation[0], transform.translation[1]) {
                continue;
            }
            self.tracked_ids
                .push(crate::types::KeypointId(pass.ids[slot]));
            self.tracked.push(&transform);
        }
    }
}

/// `computeEssential`.
///
/// `E.topLeftCorner<3,3>() = hat(t.normalized()) * R`, computed in `double`
/// because that is what casts to before calling it, and
/// cast back to the estimator's scalar afterwards.
pub(super) fn compute_essential(t_0_1: &Se3<f64>) -> Matrix4<f64> {
    let translation: Vector3<f64> = t_0_1.translation;
    let rotation = t_0_1.rotation.matrix();
    let mut essential: Matrix4<f64> = Matrix4::zeros();
    // Normalizing a zero baseline yields NaNs: coincident cameras have no epipolar geometry.
    let normalized: Vector3<f64> = translation / translation.norm();
    essential
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(So3::hat(&normalized) * rotation));
    essential
}

/// `Ed.cast<Scalar>()`.
pub(super) fn cast_matrix4(matrix: &Matrix4<f64>) -> Matrix4<f32> {
    Matrix4::from_iterator(matrix.iter().map(|value| *value as f32))
}

/// Project between cameras, returning both validity and the written pixel.
/// Both tracking and overlap masking must check validity before using the pixel.
///
/// # Panics
/// If either camera index is outside the rig.
pub fn project_between_cams(
    cameras: &[RigCamera<f32>],
    ci_uv: &Vector2<f32>,
    ci_depth: f32,
    t_ci_cj: &Se3<f32>,
    i: usize,
    j: usize,
) -> (bool, Vector2<f32>) {
    let mut ci_xyzw: Vector4<f32> = Vector4::zeros();
    let mut valid: bool = cameras[i].model.unproject(ci_uv, &mut ci_xyzw);
    ci_xyzw *= ci_depth;
    ci_xyzw.w = 1.0;

    // `T_ci_cj.inverse() * ci_xyzw`: Sophus rotates and translates the first
    // three components and carries the fourth through unchanged.
    let inverse: Se3<f32> = t_ci_cj.inverse();
    let point: Vector3<f32> = inverse * Vector3::new(ci_xyzw.x, ci_xyzw.y, ci_xyzw.z);
    let cj_xyzw: Vector4<f32> = Vector4::new(point.x, point.y, point.z, ci_xyzw.w);

    let mut cj_uv: Vector2<f32> = Vector2::zeros();
    valid &= cameras[j].model.project_point(&cj_xyzw, &mut cj_uv, None);
    (valid, cj_uv)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::calib::Calibration;
    use crate::camera::SlamCamera;
    use crate::config::VioConfig;
    use crate::frontend::flow::FrontendOptions;
    use crate::types::KeypointId;
    use kornia_staging_3d::camera::{CameraModelKind, KannalaBrandt4};
    use kornia_staging_imgproc::optical_flow::patch_se2::Pattern51;
    use nalgebra::Matrix2;

    #[test]
    fn tracking_rejects_kb4_guesses_without_changing_source_slots() {
        let calibration =
            Calibration::from_json_str(include_str!("../../../tests/fixtures/msdmg_calib.json"))
                .unwrap();
        let mut flow = FrameToFrameOpticalFlow::<Pattern51>::new(
            VioConfig::default(),
            &calibration,
            FrontendOptions::default(),
        )
        .unwrap();
        flow.cameras[0].model = SlamCamera {
            inner: CameraModelKind::Kb4(
                KannalaBrandt4::new([100.0, 100.0, 480.0, 480.0, -1.0, 0.0, 0.0, 0.0]).unwrap(),
            ),
        };
        for (id, x) in [(1, 480.0), (2, 800.0)] {
            flow.frame.cameras[0].set(
                KeypointId(id),
                &AffineCompact2f {
                    linear: Matrix2::identity().into(),
                    translation: Vector2::new(x, 480.0).into(),
                },
                NO_RESPONSE,
            );
        }
        flow.prepare_tracks(Some(&PosePrediction::default()));
        assert_eq!(flow.passes[0].ids, vec![1, 2]);
        assert_eq!(flow.passes[0].guesses.get(1).translation, [-1.0e6; 2]);
        assert_eq!(flow.passes[0].guesses.get(0).translation, [480.0, 480.0]);
    }
}
