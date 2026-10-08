//! KeyNet-F on perspective crops as the tracker's keypoint stage (handtrack `keynet_perspective.py`,
//! `PerspectiveKeyNetEstimator`).
//!
//! A tracked view's crop camera fits the planning pose's landmarks in that camera (no jitter); an acquisition's is aimed through
//! its DetNet circle's centre. The crop samples the native frame through the lens; the keypoint input is the pose's landmarks in
//! that crop (zeros on acquisition); decoded keypoints go back through the crop camera and the lens to the net frame.

use std::time::Instant;

use kornia_image::Image;
use nalgebra::{Isometry3, Vector2, Vector3};

use super::HandsError;
use super::camera::{CameraError, RigCameraModel, in_front, rig_models};
use super::detect::sigmoid;
use super::heatmaps::{
    decode_distance, decode_heatmaps, keypoint_input, nan_to_num, relative_distances,
};
use super::letterbox::BarLetterbox;
use super::perspective::{
    CROP_IMAGE_SIZE, CROP_MARGIN, CropCamera, CropMaps, crop_camera_from_circle,
    crop_camera_from_points, placeholder_crop, sample_crop,
};
use super::{CropSource, RIGHT};
use crate::frame::{NUM_CAMERAS, Rig};
use crate::nets::{HandNets, NUM_LANDMARKS};
use kornia_staging_sensors::CameraFrame;

/// One KeyNet view to estimate: tracker.py's `CropRequest` row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewRequest {
    /// Camera index 0..6.
    pub camera: usize,
    /// Hand slot: `hands::LEFT` (0) or `hands::RIGHT` (1, mirrored crop).
    pub side: usize,
    /// The crop-planning pose's 21 landmarks in the world frame (metres), for a tracked hand; `None` on acquisition.
    pub planning_pose_landmarks_world: Option<[[f64; 3]; NUM_LANDMARKS]>,
    /// The view's hand circle (cx, cy, r) in the net frame: the DetNet circle on acquisition, the projected pose's enclosing
    /// circle for a tracked view (the fallback when the pose gives no usable crop camera). Pass it whenever the tracker has one.
    pub circle_net: Option<[f32; 3]>,
    /// What the crop is aimed at (the tracker sets it where it makes the request; the estimator does not read it).
    pub source: CropSource,
}

/// KeyNet's answer for one view (tracker.py's `KeypointEstimate` row).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeypointEstimate {
    /// Keypoints in the 640x480 net frame (crop and mirror undone); 0 where not finite.
    pub points_net: [[f32; 2]; NUM_LANDMARKS],
    /// The same keypoints in the camera's full-resolution pixels (`letterbox.from_net(points_net)`, the fit's input).
    pub points_px: [[f32; 2]; NUM_LANDMARKS],
    /// Relative distances, millimetres of the generic hand.
    pub d_rel_mm: [f32; NUM_LANDMARKS],
    /// Crop presence probability; 0 for a view without a usable crop camera.
    pub presence: f32,
    /// Per keypoint: the heatmap's peak value; 0 for an unusable view.
    pub confidence: [f32; NUM_LANDMARKS],
    /// Thumb-index contact probability, with the pinch head.
    pub pinch: Option<f32>,
    /// Whether the view had a usable crop camera and finite keypoints.
    pub usable: bool,
    /// The crop camera the view sampled (`None` from estimators without one).
    pub crop: Option<CropCamera>,
}

/// One planned view: its crop camera and keypoint input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedView {
    /// The crop camera (NaN focal when unusable).
    pub crop: CropCamera,
    /// The 63-value keypoint input (zeros on acquisition).
    pub keypoint_input: [f32; 3 * NUM_LANDMARKS],
}

/// `PerspectiveKeyNetEstimator` for the RoboCap rig.
pub struct PerspectiveKeyNet {
    models: Vec<RigCameraModel>,
    letterbox: BarLetterbox,
    phi: f64,
    maps: CropMaps,
    /// Crop buffers, kept at the largest batch seen; the first `active` are the last `estimate` call's.
    crops: Vec<Image<f32, 1>>,
    active: usize,
}

impl PerspectiveKeyNet {
    /// The estimator for `rig` at hand scale `phi` (KeyNet's d_rel unit), roll 0 for every camera (RoboCap).
    ///
    /// # Errors
    ///
    /// [`HandsError::Invalid`] when the rig's cameras are not supported (see [`CameraError`]).
    pub fn new(rig: &Rig, phi: f64) -> Result<Self, HandsError> {
        let models =
            rig_models(rig).map_err(|error: CameraError| HandsError::Invalid(error.to_string()))?;
        let maps = CropMaps::new().map_err(|error| HandsError::Invalid(error.to_string()))?;
        Ok(Self {
            models,
            letterbox: BarLetterbox::robocap(),
            phi,
            maps,
            crops: Vec::new(),
            active: 0,
        })
    }

    /// Change the hand scale (after the live calibration).
    pub fn set_phi(&mut self, phi: f64) {
        self.phi = phi;
    }

    /// The hand scale in use.
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// The camera models, in index order.
    pub fn models(&self) -> &[RigCameraModel] {
        &self.models
    }

    /// The letterbox between full-resolution pixels and the net frame.
    pub fn letterbox(&self) -> &BarLetterbox {
        &self.letterbox
    }

    /// The crops of the last `estimate` call, in request order (96 x 96, [0, 1], left-hand orientation).
    pub fn crops(&self) -> &[Image<f32, 1>] {
        &self.crops[..self.active]
    }

    /// `_cameras`: one view's crop camera and keypoint input.
    ///
    /// # Errors
    ///
    /// [`HandsError::Invalid`] for a bad camera index, or an acquisition (or unusable tracked view) without a circle.
    pub fn plan(
        &self,
        world_from_rig: &Isometry3<f64>,
        request: &ViewRequest,
    ) -> Result<PlannedView, HandsError> {
        let model = self.models.get(request.camera).ok_or_else(|| {
            HandsError::Invalid(format!("camera {} out of range", request.camera))
        })?;
        let mirror = request.side == RIGHT;
        // RoboCap's cameras need no crop roll (handtrack's per-camera angle is for UmeTrack's rig).
        let roll = 0.0;
        if let Some(world) = &request.planning_pose_landmarks_world {
            let points_cam: [Vector3<f64>; NUM_LANDMARKS] = std::array::from_fn(|i| {
                model.cam_from_world_point(
                    world_from_rig,
                    &Vector3::new(world[i][0], world[i][1], world[i][2]),
                )
            });
            let valid: [bool; NUM_LANDMARKS] = std::array::from_fn(|i| in_front(&points_cam[i]));
            let crop = crop_camera_from_points(&points_cam, &valid, roll, mirror, CROP_MARGIN);
            if crop.focal.is_finite() {
                let uv: [Vector2<f64>; NUM_LANDMARKS] =
                    std::array::from_fn(|i| crop.to_crop(&points_cam[i]).0);
                let input = keypoint_input(&uv, &relative_distances(&points_cam, self.phi));
                return Ok(PlannedView {
                    crop,
                    keypoint_input: input,
                });
            }
        }
        let circle = request.circle_net.ok_or_else(|| {
            HandsError::Invalid("an untracked view needs its DetNet circle".into())
        })?;
        Ok(PlannedView {
            crop: crop_camera_from_circle(model, &self.letterbox, circle, roll, mirror),
            keypoint_input: [0.0; 3 * NUM_LANDMARKS],
        })
    }

    /// Estimate keypoints for `requests` (at most a few views; one batched KeyNet call).
    ///
    /// `full` holds each camera's 1920x1080 frame (`HandInputs::full`); a frame turned 180 degrees is sampled upright. A view whose
    /// camera frame is missing, or whose crop camera is unusable, samples a placeholder crop and reports presence 0 (as handtrack
    /// zeroes such views after the net).
    ///
    /// # Returns
    ///
    /// One estimate per request, in order, and the wall-clock milliseconds spent planning and sampling the crops.
    ///
    /// # Errors
    ///
    /// [`HandsError::Invalid`] for bad requests or images, [`HandsError::Nets`] when the backend fails.
    pub fn estimate(
        &mut self,
        nets: &mut dyn HandNets,
        full: &[Option<&CameraFrame>; NUM_CAMERAS],
        turned_180: &[bool; NUM_CAMERAS],
        world_from_rig: &Isometry3<f64>,
        requests: &[ViewRequest],
    ) -> Result<(Vec<KeypointEstimate>, f64), HandsError> {
        let invalid =
            |error: kornia_image::ImageError| HandsError::Invalid(format!("KeyNet crop: {error}"));
        let start = Instant::now();
        let planned = requests
            .iter()
            .map(|request| self.plan(world_from_rig, request))
            .collect::<Result<Vec<_>, _>>()?;
        while self.crops.len() < requests.len() {
            self.crops
                .push(Image::from_size_val(CROP_IMAGE_SIZE, 0.0).map_err(invalid)?);
        }
        self.active = requests.len();
        let mut usable: Vec<bool> = Vec::with_capacity(requests.len());
        for ((request, plan), crop) in requests
            .iter()
            .zip(&planned)
            .zip(self.crops[..self.active].iter_mut())
        {
            let model = &self.models[request.camera];
            let frame = full[request.camera];
            usable.push(plan.crop.usable() && frame.is_some());
            match frame {
                Some(frame) => {
                    let camera = if plan.crop.usable() {
                        plan.crop
                    } else {
                        placeholder_crop(plan.crop.mirror)
                    };
                    sample_crop(
                        &frame.full,
                        model,
                        &camera,
                        &mut self.maps,
                        crop,
                        turned_180[request.camera],
                    )
                    .map_err(invalid)?;
                }
                None => crop.as_slice_mut().fill(0.0),
            }
        }
        let crops_ms = start.elapsed().as_secs_f64() * 1e3;

        let inputs: Vec<&[f32]> = self.crops[..self.active]
            .iter()
            .map(|crop| crop.as_slice())
            .collect();
        let features: Vec<[f32; 3 * NUM_LANDMARKS]> =
            planned.iter().map(|plan| plan.keypoint_input).collect();
        let raw = if requests.is_empty() {
            Vec::new()
        } else {
            nets.keynet(&inputs, &features)?
        };
        if raw.len() != requests.len() {
            return Err(HandsError::Invalid(format!(
                "KeyNet returned {} outputs for {} crops",
                raw.len(),
                requests.len()
            )));
        }

        let mut out = Vec::with_capacity(requests.len());
        for (((request, plan), raw), usable) in requests.iter().zip(&planned).zip(&raw).zip(usable)
        {
            let model = &self.models[request.camera];
            let (points_crop, confidence) = decode_heatmaps(&raw.heatmaps).ok_or_else(|| {
                HandsError::Invalid(format!("KeyNet heatmaps: {} values", raw.heatmaps.len()))
            })?;
            let d_rel = decode_distance(&raw.distance).ok_or_else(|| {
                HandsError::Invalid(format!("KeyNet distance: {} values", raw.distance.len()))
            })?;
            let mut points_net = [[f32::NAN; 2]; NUM_LANDMARKS];
            let mut points_px = [[f32::NAN; 2]; NUM_LANDMARKS];
            for landmark in 0..NUM_LANDMARKS {
                let [u, v] = points_crop[landmark];
                let ray = plan
                    .crop
                    .from_crop(&Vector2::new(f64::from(u), f64::from(v)));
                if let Some(native) = model.project(&ray) {
                    let net = self.letterbox.to_net(&native);
                    points_net[landmark] = [net.x as f32, net.y as f32];
                    points_px[landmark] = [native.x as f32, native.y as f32];
                }
            }
            let usable = usable && points_net.iter().flatten().all(|v| v.is_finite());
            out.push(KeypointEstimate {
                points_net: points_net.map(|p| p.map(nan_to_num)),
                points_px: points_px.map(|p| p.map(nan_to_num)),
                d_rel_mm: d_rel.map(nan_to_num),
                presence: if usable {
                    sigmoid(raw.presence_logit)
                } else {
                    0.0
                },
                confidence: if usable {
                    confidence
                } else {
                    [0.0; NUM_LANDMARKS]
                },
                pinch: raw.pinch_logit.map(sigmoid),
                usable,
                crop: Some(plan.crop),
            });
        }
        Ok((out, crops_ms))
    }
}
