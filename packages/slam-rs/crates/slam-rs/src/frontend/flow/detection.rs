//! Detection, occupancy bookkeeping, cross-camera matching, and filtering.

use super::{FrameToFrameOpticalFlow, FrontendError, project_between_cams};
use crate::duration_ns;
use crate::frontend::detect::{
    CellGrid, DetectError, DetectorConfig, DetectorScratch, KeypointsData, Masks, Occupancy, Rect,
    detect_keypoints_with_cells,
};
use crate::frontend::parallel::WorkPool;
use crate::frontend::patterns::Pattern;
use crate::frontend::se2::AffineCompact2f;
use crate::frontend::stages::FrameStages;
use crate::frontend::tracker::PatchTracker;
use crate::lie::Se3;
use crate::types::KeypointId;
use kornia_image::Image;
use nalgebra::{Matrix4, Vector2, Vector4};

impl<P: Pattern, F: FrameStages<Tracker: PatchTracker<Pattern = P>>> FrameToFrameOpticalFlow<P, F> {
    /// `updateCellCounts` : rebuild one camera's occupancy from scratch.
    pub(super) fn update_cell_counts(&mut self, camera: usize) {
        self.cells[camera].fill(0);
        for index in 0..self.frame.cameras[camera].len() {
            let position: Vector2<f32> = self.frame.cameras[camera].transforms.translation(index);
            // `if (p[0] < x_start ||... || p[1] >= y_stop + c) continue;`.
            if !self.occupancy_grid.contains(position.x, position.y) {
                continue;
            }
            let (row, column) = self.occupancy_grid.cell_of(position.x, position.y);
            self.cells[camera][row * self.occupancy_grid.columns + column] += 1;
        }
    }

    /// The `cells(y, x)++` half of `addKeypoint`/`addKeypoints`.
    fn bump_cell(&mut self, camera: usize, transform: &AffineCompact2f) {
        let (row, column) = self
            .occupancy_grid
            .cell_of(transform.translation.x, transform.translation.y);
        self.cells[camera][row * self.occupancy_grid.columns + column] += 1;
    }

    /// `removeKeypoint` : drop a keypoint and decrement its cell.
    fn remove_keypoint(&mut self, camera: usize, id: KeypointId) {
        let Some(transform) = self.frame.cameras[camera].remove(id) else {
            return;
        };
        let (row, column) = self
            .occupancy_grid
            .cell_of(transform.translation.x, transform.translation.y);
        self.cells[camera][row * self.occupancy_grid.columns + column] -= 1;
    }

    /// `detectKeypointsWithCells`' own configuration, from this frontend's.
    pub(super) fn detector_config(&self) -> DetectorConfig {
        DetectorConfig {
            num_points_cell: self.config.optical_flow_detection_num_points_cell as usize,
            min_threshold: self.config.optical_flow_detection_min_threshold,
            max_threshold: self.config.optical_flow_detection_max_threshold,
            safe_radius: self.config.optical_flow_image_safe_radius,
        }
    }

    /// Detect with the selected scanner, then assign IDs in camera order.
    /// With `side_pool` (the side-camera pass, `cameras` = `1..`), independent
    /// side scanners run on its workers; otherwise, and for scanners with
    /// shared preparation, every camera runs on the caller.
    fn add_points_for_cameras(
        &mut self,
        cameras: std::ops::Range<usize>,
        side_pool: Option<WorkPool>,
        images: &[Image<u16, 1>],
    ) -> Result<(), FrontendError> {
        let config = self.detector_config();
        let mark = std::time::Instant::now();
        let Self {
            stages,
            side_detectors,
            detected,
            frame,
            options,
            detection_grids,
            cells,
            occupancy_grid,
            masks,
            ..
        } = self;
        let detector = stages.detector();
        let occupancy_grid: &CellGrid = occupancy_grid;
        let detect =
            |camera: usize, scratch: &mut DetectorScratch<F::Scanner>, out: &mut KeypointsData| {
                out.corners.clear();
                out.responses.clear();
                let budget = options
                    .max_keypoints
                    .saturating_sub(frame.cameras[camera].len());
                if budget == 0 {
                    return Ok(());
                }
                // Level 0 is the unchanged input image, including on the device lane.
                detect_keypoints_with_cells(
                    &images[camera],
                    camera,
                    &detection_grids[camera],
                    &Occupancy {
                        counts: &cells[camera],
                        rows: occupancy_grid.rows,
                        columns: occupancy_grid.columns,
                    },
                    &config,
                    &masks[camera],
                    budget,
                    scratch,
                    out,
                )
            };
        let parallel =
            side_pool
                .as_ref()
                .zip(side_detectors.as_mut())
                .and_then(|(pool, scanners)| {
                    pool.install(|| {
                        use rayon::prelude::*;
                        scanners
                            .par_iter_mut()
                            .zip(detected[cameras.clone()].par_iter_mut())
                            .enumerate()
                            .map(|(slot, (scratch, out))| detect(slot + 1, scratch, out))
                            .collect::<Vec<Result<(), DetectError>>>()
                    })
                });
        match parallel {
            Some(results) => {
                for result in results {
                    result?;
                }
            }
            None => {
                for camera in cameras.clone() {
                    detect(camera, detector, &mut detected[camera])?;
                }
            }
        }
        self.timings.detect_ns += duration_ns(mark);
        self.new_cam0.clear();
        let detected = std::mem::take(&mut self.detected);
        for camera in cameras {
            self.file_detected(camera, &detected[camera]);
        }
        self.detected = detected;
        Ok(())
    }

    /// The second half of `addPointsForCamera`: register `detected` on camera
    /// `camera` under fresh ids, bumping cells.
    fn file_detected(&mut self, camera: usize, detected: &KeypointsData) {
        for index in 0..detected.corners.len() {
            let corner: [f32; 2] = detected.corners[index];
            let response: f32 = detected.responses[index];
            let transform: AffineCompact2f =
                AffineCompact2f::at(Vector2::new(corner[0], corner[1]));
            let id: KeypointId = KeypointId(self.last_keypoint_id);
            // `addKeypoint` : bump the cell, then register.
            self.bump_cell(camera, &transform);
            self.frame.cameras[camera].set(id, &transform, response);
            if camera == 0 {
                self.new_cam0.set(id, &transform, response);
            }
            // `last_keypoint_id++`, the global landmark id space.
            self.last_keypoint_id += 1;
        }
    }

    /// Bump occupancy for each offered keypoint, then insert ids that are new.
    /// A point both tracked and matched increments twice while retaining its existing
    /// entry. Once capacity is reached, further matches neither insert nor bump cells.
    fn add_keypoints(&mut self, camera: usize) {
        let count: usize = self.tracked_ids.len();
        for slot in 0..count {
            if self.frame.cameras[camera].len() >= self.options.max_keypoints {
                break;
            }
            let transform: AffineCompact2f = self.tracked.get(slot);
            self.bump_cell(camera, &transform);
            self.frame.cameras[camera].insert_if_absent(self.tracked_ids[slot], &transform);
        }
    }

    /// Mask camera cells that project into camera 0 at `depth_guess`.
    /// Use the frontend grid bounds and append to reusable per-camera mask lists.
    /// Reusing those lists avoids allocating hundreds of rectangles each frameset.
    fn append_cam0_overlap_masks(&mut self, camera: usize) {
        let Self {
            masks,
            cameras,
            calib,
            occupancy_grid,
            depth_guess,
            ..
        } = self;
        let grid: CellGrid = *occupancy_grid;
        let cell: usize = grid.cell;
        let half: usize = cell / 2;
        let x_first: usize = grid.x_start + half;
        let y_first: usize = grid.y_start + half;
        let x_last: usize = grid.x_stop + half;
        let y_last: usize = grid.y_stop + half;

        let width: f32 = cameras[0].resolution[0] as f32;
        let height: f32 = cameras[0].resolution[1] as f32;
        let t_ci_c0: Se3<f32> = calib.t_i_c[camera].inverse() * calib.t_i_c[0];

        let out: &mut Masks = &mut masks[camera];
        let mut y: usize = y_first;
        while y <= y_last {
            let mut x: usize = x_first;
            while x <= x_last {
                let ci_uv: Vector2<f32> = Vector2::new(x as f32, y as f32);
                let (projected, c0_uv) =
                    project_between_cams(cameras, &ci_uv, *depth_guess, &t_ci_c0, camera, 0);
                let in_bounds: bool =
                    c0_uv.x >= 0.0 && c0_uv.x < width && c0_uv.y >= 0.0 && c0_uv.y < height;
                if projected && in_bounds {
                    out.masks.push(Rect {
                        x: (x - half) as f32,
                        y: (y - half) as f32,
                        w: cell as f32,
                        h: cell as f32,
                    });
                }
                x += cell;
            }
            y += cell;
        }
    }

    /// Decide whether the whole frameset detects again (D75).
    /// A zero survivor-ratio setting detects every frame. Above zero, camera 0 must
    /// fall below the configured fraction of its count after the last detection.
    /// This relative budget needs no per-rig target count.
    /// Gate the whole frameset so new camera-0 ids can be matched into every other
    /// camera. Only camera 0 votes on keyframes, so its count is the relevant input.
    /// No timing or arrival-order state participates; replay remains deterministic.
    pub(super) fn should_detect(&self) -> bool {
        let ratio: f32 = self.config.port_redetect_survivor_ratio;
        // A ratio that is not a positive number is the knob switched off, NaN
        // included: `x < NaN` is false, which would stop detection for good.
        if !ratio.is_finite() || ratio <= 0.0 {
            return true;
        }
        // Nothing has been detected yet, so there is no survivor fraction to
        // take: the first frameset of a run, and the first after one that was
        // refused before it detected.
        if self.last_detect_count == 0 {
            return true;
        }
        let survivors: f32 = self.frame.cameras[0].len() as f32;
        survivors < ratio * self.last_detect_count as f32
    }

    /// `addPoints` : detect on camera 0, match onward, then detect
    /// again on the cameras that do not overlap camera 0.
    pub(super) fn add_points(&mut self, images: &[Image<u16, 1>]) -> Result<(), FrontendError> {
        // Camera 0's cell winners are already on the host: `run_passes`
        // launched them before the temporal tracks and the tracks' own download
        // brought them back (D78). A backend without a device path prepared
        // nothing and every `detect_keypoints_with_cells` answers for itself.
        self.add_points_for_cameras(0..1, None, images)?;

        // `for (i = 1; i < getNumCams(); i++) trackPoints(pyr0, pyri, kpts0, ...)`
        // With one camera there is nothing to match into (trap 17).
        // One batch again: camera *i*'s match reads camera 0's new keypoints and
        // writes camera *i* alone, so the whole rig is launched before any of it
        // is downloaded.
        if self.cameras.len() > 1 {
            self.prepare_tracks(None);
        }
        self.stages.stereo(
            &mut self.passes[1..],
            images,
            &self.cell_selects,
            self.config.optical_flow_detection_nonoverlap,
            &mut self.timings,
        )?;
        let mark = std::time::Instant::now();
        for camera in 1..self.cameras.len() {
            self.finish_track_points(camera);
            self.add_keypoints(camera);
        }
        self.timings.stereo_ns += duration_ns(mark);

        // `if (!config.optical_flow_detection_nonoverlap) continue;`.
        if self.config.optical_flow_detection_nonoverlap {
            for camera in 1..self.cameras.len() {
                self.append_cam0_overlap_masks(camera);
            }
            self.add_points_for_cameras(
                1..self.cameras.len(),
                Some(self.host_pool.clone()),
                images,
            )?;
        }
        Ok(())
    }

    /// `filterPointsForCam` : drop the keypoints camera `camera`
    /// shares with camera 0 whose epipolar error is too large.
    fn filter_points_for_cam(&mut self, camera: usize) {
        self.to_remove.clear();
        let essential: Matrix4<f32> = self.essential[camera];
        let threshold: f32 = self.config.optical_flow_epipolar_error;

        for index in 0..self.frame.cameras[camera].ids.len() {
            let id: KeypointId = self.frame.cameras[camera].ids[index];
            let Some(in_cam0) = self.frame.cameras[0].get(id) else {
                continue;
            };
            let proj1: Vector2<f32> = self.frame.cameras[camera].transforms.translation(index);

            let mut p3d0: Vector4<f32> = Vector4::zeros();
            let mut p3d1: Vector4<f32> = Vector4::zeros();
            let ok0: bool = self.cameras[0]
                .model
                .unproject(&in_cam0.translation, &mut p3d0);
            let ok1: bool = self.cameras[camera].model.unproject(&proj1, &mut p3d1);

            if ok0 && ok1 {
                // `std::abs(p3d0.transpose() * E[cam] * p3d1)`, in the
                // estimator's scalar and only then widened for the comparison.
                let error: f32 = (p3d0.transpose() * essential * p3d1)[(0, 0)].abs();
                if error > threshold {
                    self.to_remove.push(id);
                }
            } else {
                self.to_remove.push(id);
            }
        }

        // `for (int id : kp_to_remove) removeKeypoint(cam_id, id);`, a
        // `std::set`, so ascending; `self.to_remove` is built in id order
        // already. Indexed rather than drained because `remove_keypoint` takes
        // `&mut self`, and the list does not change while it runs.
        for slot in 0..self.to_remove.len() {
            let id: KeypointId = self.to_remove[slot];
            self.remove_keypoint(camera, id);
        }
        self.to_remove.clear();
    }

    /// `filterPoints` : every camera but camera 0.
    pub(super) fn filter_points(&mut self) {
        for camera in 1..self.cameras.len() {
            self.filter_points_for_cam(camera);
        }
    }
}
