//! Tracking pass submission, collection, and camera geometry.

use super::{FrameToFrameOpticalFlow, NO_RESPONSE, PosePrediction};
use crate::camera::RigCamera;
use crate::config::MatchingGuessType;
use crate::frontend::patterns::Pattern;
use crate::frontend::se2::AffineCompact2f;
use crate::frontend::stages::FrameStages;
use crate::frontend::tracker::{PatchTracker, TrackInput};
use crate::lie::{Se3, So3};
use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

impl<P: Pattern, F: FrameStages<Tracker: PatchTracker<Pattern = P>>> FrameToFrameOpticalFlow<P, F> {
    /// Mask and predict every camera's inputs independently, then submit them
    /// together. `None` selects camera-zero stereo matches of new detections.
    pub(super) fn prepare_tracks(&mut self, prediction: Option<&PosePrediction>) {
        let Self {
            passes,
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
        let first = usize::from(prediction.is_none());
        let use_depth = prediction.is_some()
            || config.optical_flow_matching_guess_type != MatchingGuessType::SamePixel;
        let prepare = |(index, pass): (usize, &mut TrackInput)| {
            let camera = first + index;
            let (source, cam1, t_c1_c2) = match prediction {
                Some(prediction) => {
                    let t_c1 = prediction.t_w_i_previous * calib.t_i_c[camera];
                    let t_c2 = prediction.t_w_i_current * calib.t_i_c[camera];
                    (&frame.cameras[camera], camera, t_c1.inverse() * t_c2)
                }
                None => (
                    &*new_cam0,
                    0,
                    calib.t_i_c[0].inverse() * calib.t_i_c[camera],
                ),
            };
            pass.source = cam1;
            pass.destination = camera;
            pass.ids.clear();
            pass.positions.clear();
            pass.guesses.clear();
            for (index, id) in source.ids.iter().enumerate() {
                let transform = source.transforms.get(index);
                let position = transform.translation;
                if masks[cam1].in_bounds(position.x, position.y) {
                    continue;
                }
                let translation = if use_depth {
                    project_between_cams(cameras, &position, *depth_guess, &t_c1_c2, cam1, camera).1
                } else {
                    position
                };
                pass.ids.push(*id);
                pass.positions.push(position);
                pass.guesses.push(&AffineCompact2f {
                    linear: transform.linear,
                    translation,
                });
            }
        };
        if host_pool
            .install(|| {
                use rayon::prelude::*;
                passes[first..].par_iter_mut().enumerate().for_each(prepare);
            })
            .is_none()
        {
            passes[first..].iter_mut().enumerate().for_each(prepare);
        }
    }

    /// The tail of `trackPoints` for one camera, once its lane has arrived.
    pub(super) fn finish_camera(&mut self, camera: usize) {
        self.finish_track_points(camera);
        self.frame.cameras[camera].clear();
        for (slot, id) in self.tracked_ids.iter().enumerate() {
            // `keypoint_map_2.insert(result.begin(), result.end())`; the
            // cell counts are rebuilt afterwards by `updateCellCounts`.
            self.frame.cameras[camera].set(*id, &self.tracked.get(slot), NO_RESPONSE);
        }
    }

    /// The reading half of `trackPoints`: `masks2` over one collected lane,
    /// leaving the survivors in `tracked_ids` and `tracked`.
    pub(super) fn finish_track_points(&mut self, lane: usize) {
        self.tracked_ids.clear();
        self.tracked.clear();
        let pass = &self.passes[lane];
        let cam2 = pass.destination;
        let result = self.stages.tracker().result(pass.result);
        for slot in result.tracked() {
            let slot: usize = *slot as usize;
            let transform: AffineCompact2f = result.transform(slot);
            // `if (masks2.inBounds(t2.x(), t2.y())) continue;`.
            if self.masks[cam2].in_bounds(transform.translation.x, transform.translation.y) {
                continue;
            }
            self.tracked_ids.push(pass.ids[slot]);
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
/// Tracking uses the pixel; overlap masking also checks validity.
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
    valid &= cameras[j].model.project(&cj_xyzw, &mut cj_uv);
    (valid, cj_uv)
}
