//! Tracking pass submission, collection, and camera geometry.

use super::{FrameToFrameOpticalFlow, FrontendError, Keypoints, NO_RESPONSE};
use crate::camera::RigCamera;
use crate::config::MatchingGuessType;
use crate::frontend::patterns::Pattern;
use crate::frontend::se2::AffineCompact2f;
use crate::frontend::tracker::{PatchTracker, SourcePatches};
use crate::lie::{Se3, So3};
use crate::pyramid::PyramidBuilder;
use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

impl<P: Pattern, B: PyramidBuilder, T: PatchTracker<Pattern = P, Pyramid = B::Pyramid>>
    FrameToFrameOpticalFlow<P, B, T>
{
    /// One camera's frame-to-frame track launched into its own lane:
    /// `trackPoints(..., cam, cam)` up to the download.
    pub(super) fn submit_camera(
        &mut self,
        camera: usize,
        t_c1_c2: &Se3<f32>,
    ) -> Result<(), FrontendError> {
        // Source and destination are the same slot, so the
        // ids and warps are copied out first — and the slot is only cleared once
        // the track has succeeded, so a refused frame does not lose the camera's
        // keypoints.
        self.passes[camera].ids.clear();
        let source: &Keypoints = &self.frame.cameras[camera];
        self.passes[camera].ids.extend_from_slice(&source.ids);
        self.source.clone_from(&source.transforms);

        self.submit_track_points(camera, camera, camera, t_c1_c2, true)
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

    /// The launching half of `trackPoints` : the mask test, the
    /// guesses, the patch build and the tracker's own kernels, into `lane`.
    ///
    /// Reads the pass source IDs and shared source warps, then records its
    /// offered-index map and the result slot returned by the tracker. Split from [`FrameToFrameOpticalFlow::finish_track_points`]
    /// because `trackPoints` serves two purposes — carrying a camera's own
    /// keypoints forward in time, and matching camera 0's new keypoints into
    /// camera *i* — and both run every camera of the frameset before reading any
    /// of them.
    pub(super) fn submit_track_points(
        &mut self,
        lane: usize,
        cam1: usize,
        cam2: usize,
        t_c1_c2: &Se3<f32>,
        tracking: bool,
    ) -> Result<(), FrontendError> {
        // `use_depth = tracking || (matching && guess_type != SAME_PIXEL)`.
        let use_depth: bool = tracking
            || self.config.optical_flow_matching_guess_type != MatchingGuessType::SamePixel;
        let depth: f32 = self.depth_guess;

        self.passes[lane].offered.clear();
        self.positions.clear();
        self.guesses.clear();

        for index in 0..self.source.len() {
            let transform_1: AffineCompact2f = self.source.get(index);
            let t1: Vector2<f32> = transform_1.translation;
            // `if (masks1.inBounds(t1.x(), t1.y())) continue;`.
            if self.masks[cam1].in_bounds(t1.x, t1.y) {
                continue;
            }
            // `off = t2 - t2_guess` with `t2 == t1`, then `t2 -= off`
            // so the guess is simply `t2_guess`.
            let translation: Vector2<f32> = if use_depth {
                project_between_cams(&self.cameras, &t1, depth, t_c1_c2, cam1, cam2).1
            } else {
                t1
            };
            self.passes[lane].offered.push(index);
            self.positions.push(t1);
            self.guesses.push(&AffineCompact2f {
                linear: transform_1.linear,
                translation,
            });
        }

        // The forward source patches come from the previous frame when tracking
        // and from this frame's camera 0 when matching. This
        // frame is `staging` until the call commits.
        let source_pyramid: &B::Pyramid = if tracking {
            &self.pyramid[cam1]
        } else {
            &self.staging[cam1]
        };
        self.patches
            .prepare(source_pyramid, &self.positions, None)?;
        self.passes[lane].destination = cam2;
        self.passes[lane].result = self.tracker.submit_prepared(
            source_pyramid,
            &self.staging[cam2],
            &self.patches,
            &self.guesses,
        )?;
        Ok(())
    }

    /// The reading half of `trackPoints`: `masks2` over one collected lane,
    /// leaving the survivors in `tracked_ids` and `tracked`.
    pub(super) fn finish_track_points(&mut self, lane: usize) {
        self.tracked_ids.clear();
        self.tracked.clear();
        let pass = &self.passes[lane];
        let cam2 = pass.destination;
        let result = self.tracker.result(pass.result);
        for slot in result.tracked() {
            let slot: usize = *slot as usize;
            let transform: AffineCompact2f = result.transform(slot);
            // `if (masks2.inBounds(t2.x(), t2.y())) continue;`.
            if self.masks[cam2].in_bounds(transform.translation.x, transform.translation.y) {
                continue;
            }
            self.tracked_ids
                .push(self.passes[lane].ids[self.passes[lane].offered[slot]]);
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
