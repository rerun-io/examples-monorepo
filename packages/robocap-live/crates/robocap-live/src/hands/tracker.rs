//! The hand tracker: the Rust port of handtrack's `Tracker.step` (`packages/handtrack/handtrack/tracker.py`, commit 54eaf309)
//! with `ROBUST_TRACKER_CONFIG` and `fit_backend = "native"`, generalised from four to the six RoboCap cameras.
//!
//! Per hand the state is θ(t−1) and θ(t−2) (none while untracked). On each frameset:
//!
//! - **Tracked hand:** plan from θ̂ (θ(t−1) while the track is young, else the damped extrapolation
//!   θ(t−1) + 0.5·(θ(t−1) − θ(t−2)) with the wrist step clamped to 0.15 m), project its 21 keypoints into every configured
//!   camera, and ask KeyNet for the (at most two) cameras with the most keypoints inside the image (ties: lower camera index).
//!   A hand that no camera sees is dropped without KeyNet.
//! - **Untracked hands:** while at least one hand is untracked, DetNet runs on one of `detnet_groups` interleaved groups of the DetNet
//!   cameras per frameset (one camera per frameset with one group per camera: ROBUST_TRACKER_CONFIG's round robin; every camera
//!   with one group: handtrack's `detnet_all_cameras`). A hand above 0.8 gets an acquisition view in each camera that reports it
//!   (best first, at most `max_views`); one re-crop pass re-cuts it around KeyNet's own keypoints.
//! - **Presence rules:** KeyNet presence below 0.5 keeps a view out of the fit; below it in every view ends the track; a tracked
//!   hand that loses one of two views ends too (`end_on_view_rejection`, patience 1).
//! - **Fit:** handfit's LM from θ(t−1) with the temporal prior (tracked), or handfit's cold `initial_pose` (acquisitions); keypoints
//!   whose heatmap peak is under 0.05, or that the planning pose puts outside the image, weigh 0.
//! - **Gates:** an acquisition must converge and leave an RMS residual under 0.08 of its keypoints' spread; any fit must be
//!   finite with the wrist within 1 m of the headset. A new track is reported from its third frame (`confirm_frames` 2).
//!
//! The DetNet and KeyNet stages sit behind [`Perception`] (handtrack's `Detector` / `KeypointEstimator` protocols), so the
//! tests drive the tracker with fakes and the runtime with the perspective-crop estimator on the NPU.
//!
//! Numerics follow the Python reference where it is cheap: the fit sees float32 inputs and returns float32 poses, as
//! handfit's Python binding does, so a run driven by the same estimator outputs reproduces the Python poses.

use std::time::Instant;

use handfit::cold::initial_pose_parallel;
use handfit::nalgebra::{Matrix3, Matrix4, Vector2, Vector3};
use handfit::{Config, JacobianMode, Model, Pose, View, fit};
use nalgebra::Isometry3;

use super::camera::{RigCameraModel, f32_round, fit_view, world_matrix};
use super::detect::Detections;
use super::estimator::{KeypointEstimate, ViewRequest};
use super::model::{GenericHandModel, landmarks_world, mirror, pose_f32, pose_finite};
use super::scale::{CalibrationBlock, LiveScale, ScaleOutcome};
use super::{
    CropSource, DetNetHit, HandFrameResult, HandInputs, HandOutput, HandTimings, HandTracking,
    HandsConfig, HandsError, KeyNetView, LEFT, RIGHT, ScaleMode, ViewOutcome,
};
use crate::frame::{CameraFrame, Luma, NUM_CAMERAS, Rig};
use crate::nets::{HandNets, NUM_LANDMARKS};
use crate::hands::circles::min_enclosing_circle;

/// The tracker's thresholds: the fields of handtrack's `TrackerConfig` that `ROBUST_TRACKER_CONFIG` uses or sets.
/// [`TrackerConfig::robust`] (the [`Default`]) is `ROBUST_TRACKER_CONFIG` exactly; [`TrackerConfig::handtrack_default`] is
/// handtrack's `TrackerConfig()` (for the ported tests). The views, DetNet's cameras and the cold fit's threads are [`HandsConfig`]'s.
#[derive(Clone, Debug)]
pub struct TrackerConfig {
    /// DetNet reports a hand when its presence exceeds this.
    pub detnet_threshold: f64,
    /// KeyNet presence below this keeps a view out of the fit; below it in every view ends the track.
    pub presence_threshold: f64,
    /// Plan from the extrapolated θ̂ (false: θ(t−1)).
    pub extrapolate: bool,
    /// θ̂ = θ(t−1) + gain·(θ(t−1) − θ(t−2)).
    pub extrapolation_gain: f64,
    /// Plan from θ(t−1) until the track has been fitted more than this many frames.
    pub extrapolate_min_age: u32,
    /// Clamp θ̂'s wrist step to this length (metres per frame).
    pub extrapolation_max_step_m: Option<f64>,
    /// A fitted wrist farther than this from the headset ends the track.
    pub max_reach_m: f64,
    /// A keypoint whose heatmap peak is below this weighs 0 in the fit.
    pub min_keypoint_confidence: f64,
    /// A keypoint that the planning pose projects outside the image (or within `image_margin_px` of its edge) weighs 0.
    pub mask_out_of_image: bool,
    /// See `mask_out_of_image`.
    pub image_margin_px: f64,
    /// With `end_on_view_rejection`: end the track after this many consecutive frames with a rejected view.
    pub rejection_patience: u32,
    /// Re-crop passes of an acquisition around KeyNet's own keypoints before its fit.
    pub acquire_recrop: u32,
    /// Reject an acquisition whose fit leaves an RMS 2D residual above this (pixels).
    pub acquire_max_rms_px: f64,
    /// Reject an acquisition whose RMS residual exceeds this share of its keypoints' RMS spread.
    pub acquire_max_relative_rms: f64,
    /// A new track is reported from its (confirm_frames+1)-th frame.
    pub confirm_frames: u32,
    /// End a tracked hand when KeyNet rejects one of its two views.
    pub end_on_view_rejection: bool,
    /// handfit's solver settings (handtrack `FitConfig()`); `phi` is overwritten with the tracker's scale.
    pub fit: Config,
}

impl TrackerConfig {
    /// handtrack's `TrackerConfig()` defaults (`DEFAULT_TRACKER_CONFIG`), restricted to the ported fields.
    pub fn handtrack_default() -> Self {
        Self {
            detnet_threshold: 0.5,
            presence_threshold: 0.5,
            extrapolate: true,
            extrapolation_gain: 1.0,
            extrapolate_min_age: 0,
            extrapolation_max_step_m: None,
            max_reach_m: 1.0,
            min_keypoint_confidence: 0.05,
            mask_out_of_image: false,
            image_margin_px: 0.0,
            rejection_patience: 1,
            acquire_recrop: 0,
            acquire_max_rms_px: 1e9,
            acquire_max_relative_rms: 1e9,
            confirm_frames: 0,
            end_on_view_rejection: false,
            fit: Config::default(),
        }
    }

    /// `ROBUST_TRACKER_CONFIG` (commit 54eaf309).
    pub fn robust() -> Self {
        Self {
            end_on_view_rejection: true,
            acquire_recrop: 1,
            mask_out_of_image: true,
            confirm_frames: 2,
            detnet_threshold: 0.8,
            extrapolation_gain: 0.5,
            extrapolate_min_age: 2,
            extrapolation_max_step_m: Some(0.15),
            acquire_max_relative_rms: 0.08,
            ..Self::handtrack_default()
        }
    }
}

impl Default for TrackerConfig {
    /// `ROBUST_TRACKER_CONFIG` (the views, DetNet's cameras and the cold fit's threads are `HandsConfig`'s).
    fn default() -> Self {
        Self::robust()
    }
}

/// DetNet and KeyNet as the tracker sees them (handtrack's `Detector` and `KeypointEstimator` protocols), in the estimator's
/// types: [`Detections`] per camera, [`ViewRequest`] in and [`KeypointEstimate`] out per view.
pub trait Perception: Send {
    /// DetNet on several cameras' 640x360 small images (one batch), one answer per camera in order.
    ///
    /// # Errors
    ///
    /// Any backend or input error.
    fn detect(
        &mut self,
        nets: &mut dyn HandNets,
        cameras: &[usize],
        small: &[&Luma],
    ) -> Result<Vec<Detections>, HandsError>;

    /// KeyNet on `views` (one batched call), in order, and the milliseconds of the call spent cutting crops (0 when not measured).
    ///
    /// # Errors
    ///
    /// Any backend or input error.
    fn estimate(
        &mut self,
        nets: &mut dyn HandNets,
        full: &[Option<&CameraFrame>; NUM_CAMERAS],
        world_from_rig: &Isometry3<f64>,
        views: &[ViewRequest],
    ) -> Result<(Vec<KeypointEstimate>, f64), HandsError>;

    /// The hand scale changed (the keypoint input's relative distances are divided by it).
    fn set_phi(&mut self, phi: f64);
}

/// A finite circle as a view's `circle_net` (`None` when any value is not finite).
fn circle_net(circle: [f64; 3]) -> Option<[f32; 3]> {
    circle
        .iter()
        .all(|x| x.is_finite())
        .then(|| circle.map(|x| x as f32))
}

/// θ̂ = θ(t−1) + gain·(θ(t−1) − θ(t−2)) (handtrack `hand.pose.extrapolate`): the rotation step D = R(t−1)·R(t−2)ᵀ scaled towards the
/// identity and projected back onto SO(3), the wrist step clamped to `max_step_m`, the joint angles with the same gain.
pub fn extrapolate(previous: &Pose, before: &Pose, gain: f64, max_step_m: Option<f64>) -> Pose {
    let mut delta: Matrix3<f64> = previous.rotation * before.rotation.transpose();
    if gain != 1.0 {
        let eye = Matrix3::identity();
        let svd = (eye + gain * (delta - eye)).svd(true, true);
        if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
            let mut flip = Matrix3::identity();
            flip[(2, 2)] = (u * v_t).determinant();
            delta = u * flip * v_t;
        }
    }
    let mut step: Vector3<f64> = gain * (previous.translation - before.translation);
    if let Some(limit) = max_step_m {
        step *= (limit / step.norm().max(1e-9)).min(1.0);
    }
    Pose {
        rotation: delta * previous.rotation,
        translation: previous.translation + step,
        angles: previous.angles + gain * (previous.angles - before.angles),
    }
}

/// RMS distance of a hand's observed keypoints from their weighted centre in each view (handtrack `_keypoint_spread_px`).
fn keypoint_spread_px(views: &[ViewObservation]) -> f64 {
    let mut total = 0.0;
    let mut weighted = 0.0;
    for view in views {
        let sum: f64 = view.weights.iter().sum();
        if sum <= 0.0 {
            continue;
        }
        let mut centre = [0.0, 0.0];
        for i in 0..NUM_LANDMARKS {
            centre[0] += view.weights[i] * view.keypoints_px[i][0];
            centre[1] += view.weights[i] * view.keypoints_px[i][1];
        }
        centre = [centre[0] / sum, centre[1] / sum];
        for i in 0..NUM_LANDMARKS {
            total += view.weights[i]
                * ((view.keypoints_px[i][0] - centre[0]).powi(2)
                    + (view.keypoints_px[i][1] - centre[1]).powi(2));
        }
        weighted += sum;
    }
    (total / weighted.max(1.0)).sqrt()
}

#[derive(Clone, Debug, Default)]
struct History {
    /// θ(t−1); None while untracked.
    previous: Option<Pose>,
    /// θ(t−2); None after the first tracked frame.
    before: Option<Pose>,
    /// Consecutive frames with a rejected view.
    rejections: u32,
    /// Frames this track has been fitted.
    age: u32,
    /// Frames this track stays tentative.
    confirm: u32,
}

/// One camera's projection of a hand pose.
#[derive(Clone, Copy, Debug)]
struct CameraProjection {
    /// Keypoints in front and inside the image.
    inside: usize,
    /// Smallest enclosing circle of the in-front keypoints in the net frame.
    circle: Option<[f64; 3]>,
}

/// One camera's evidence for one hand (handtrack `ViewObservation`).
#[derive(Clone, Debug)]
struct ViewObservation {
    camera: usize,
    keypoints_px: [[f64; 2]; NUM_LANDMARKS],
    weights: [f64; NUM_LANDMARKS],
    d_rel_mm: [f64; NUM_LANDMARKS],
}

#[derive(Clone, Debug)]
struct HandObservation {
    side: usize,
    views: Vec<ViewObservation>,
}

/// The fit's answer for one hand.
#[derive(Clone, Debug)]
struct FitAnswer {
    pose: Pose,
    e_2d: f64,
    converged: bool,
}

/// One frameset while the tracker works on it: the networks, the images, the headset pose (as given and as handtrack's float32
/// 4x4), and what the step fills in.
struct Step<'a> {
    nets: &'a mut dyn HandNets,
    inputs: &'a HandInputs<'a>,
    world_from_rig: &'a Isometry3<f64>,
    world: Matrix4<f64>,
    out: FrameOutput,
    timings: HandTimings,
}

/// What a frameset fills in while it runs (handtrack `_FrameOutput`, the parts the runtime reports).
#[derive(Clone, Default)]
struct FrameOutput {
    detnet_camera: Option<usize>,
    reported: [bool; 2],
    calibration: Vec<CalibrationBlock>,
    detnet_hits: [Vec<DetNetHit>; 2],
    keynet_views: [Vec<KeyNetView>; 2],
    predicted: [Option<[[f64; 3]; NUM_LANDMARKS]>; 2],
}

/// The hand tracker (see the module docs).
pub struct Tracker {
    cameras: Vec<RigCameraModel>,
    /// `hands.cameras`, sorted, without repeats.
    active: Vec<usize>,
    /// `hands.detnet_cameras` (else `active`), sorted, without repeats.
    detnet_cameras: Vec<usize>,
    /// The views per hand, DetNet's groups and the cold fit's threads.
    hands: HandsConfig,
    config: TrackerConfig,
    generic: GenericHandModel,
    model: Model,
    phi: f64,
    history: [History; 2],
    next_detnet: usize,
    perception: Box<dyn Perception>,
    scale: LiveScale,
}

impl Tracker {
    /// A tracker for `rig`, using `hands.cameras` (views and DetNet's camera policy from `hands` too), at the scale `hands.scale`
    /// asks for.
    ///
    /// # Errors
    ///
    /// [`HandsError::Invalid`] for an empty or out-of-range camera list, an unsupported camera, `max_views` outside 1..=2 or a
    /// non-positive fixed scale; [`HandsError::Model`] when the hand model asset is broken.
    pub fn new(
        rig: &Rig,
        hands: &HandsConfig,
        config: TrackerConfig,
        mut perception: Box<dyn Perception>,
    ) -> Result<Self, HandsError> {
        if hands.detnet_groups == 0 {
            return Err(HandsError::Invalid(
                "detnet_groups must be at least 1".into(),
            ));
        }
        if !(1..=2).contains(&hands.max_views) {
            return Err(HandsError::Invalid(format!(
                "max_views must be between 1 and 2, got {}",
                hands.max_views
            )));
        }
        if hands.cameras.is_empty()
            || hands
                .cameras
                .iter()
                .any(|&c| c >= rig.cameras.len() || c >= NUM_CAMERAS)
        {
            return Err(HandsError::Invalid(format!(
                "hand cameras {:?} do not fit a rig of {} cameras",
                hands.cameras,
                rig.cameras.len()
            )));
        }
        let mut active = hands.cameras.clone();
        active.sort_unstable();
        active.dedup();
        let mut detnet_cameras = hands
            .detnet_cameras
            .clone()
            .unwrap_or_else(|| active.clone());
        detnet_cameras.sort_unstable();
        detnet_cameras.dedup();
        if detnet_cameras.is_empty()
            || detnet_cameras
                .iter()
                .any(|&c| c >= rig.cameras.len() || c >= NUM_CAMERAS)
        {
            return Err(HandsError::Invalid(format!(
                "DetNet cameras {detnet_cameras:?} do not fit a rig of {} cameras",
                rig.cameras.len()
            )));
        }
        let cameras: Vec<RigCameraModel> = rig
            .cameras
            .iter()
            .map(RigCameraModel::from_rig_camera)
            .collect::<Result<_, _>>()
            .map_err(|error| HandsError::Invalid(error.to_string()))?;
        if active
            .iter()
            .map(|&c| cameras[c].fit.is_fisheye())
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1
        {
            return Err(HandsError::Invalid(
                "the hand cameras mix pinhole and Fisheye62 lenses".into(),
            ));
        }
        let (phi, scale) = match hands.scale {
            ScaleMode::Fixed(phi) if phi.is_finite() && phi > 0.0 => (phi, LiveScale::fixed()),
            ScaleMode::Fixed(phi) => {
                return Err(HandsError::Invalid(format!(
                    "hand scale {phi} must be positive"
                )));
            }
            ScaleMode::Auto { seconds } => (1.0, LiveScale::auto(seconds, hands.scale_wait)),
        };
        let generic = GenericHandModel::load()?;
        perception.set_phi(phi);
        Ok(Self {
            model: generic.scaled(phi),
            generic,
            cameras,
            active,
            detnet_cameras,
            hands: hands.clone(),
            config,
            phi,
            history: [History::default(), History::default()],
            next_detnet: 0,
            perception,
            scale,
        })
    }

    /// The scale in use.
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// The live calibration's outcome, once it finished.
    pub fn scale_outcome(&self) -> Option<&ScaleOutcome> {
        self.scale.outcome()
    }

    /// Whether the hand is tracked internally (reported or tentative).
    pub fn is_tracked(&self, side: usize) -> bool {
        self.history.get(side).is_some_and(|h| h.previous.is_some())
    }

    /// θ(t) of a tracked hand.
    pub fn pose(&self, side: usize) -> Option<&Pose> {
        self.history.get(side).and_then(|h| h.previous.as_ref())
    }

    /// Switch to the live calibration's scale and say so once: phi, how it was chosen (calibrated, clamped, fallback), the solve.
    fn set_scale(&mut self, outcome: &ScaleOutcome) {
        eprintln!(
            "robocap-live: hands: scale phi {:.4}, {} ({} stereo observations, solve {:.3} s)",
            outcome.phi, outcome.note, outcome.blocks, outcome.solve_s
        );
        self.phi = outcome.phi;
        self.model = self.generic.scaled(outcome.phi);
        self.perception.set_phi(outcome.phi);
    }

    fn drop_track(&mut self, side: usize) {
        let history = &mut self.history[side];
        history.previous = None;
        history.before = None;
        history.rejections = 0;
        history.age = 0;
    }

    /// The pose's keypoints in every active camera with an image (handtrack `_project`).
    fn project(
        &self,
        landmarks: &[[f64; 3]; NUM_LANDMARKS],
        world: &Matrix4<f64>,
        available: &[bool; NUM_CAMERAS],
    ) -> [Option<CameraProjection>; NUM_CAMERAS] {
        let mut out = [None; NUM_CAMERAS];
        for &c in &self.active {
            if !available[c] {
                continue;
            }
            let camera = &self.cameras[c];
            let mut inside = 0;
            let mut front: Vec<[f64; 2]> = Vec::with_capacity(NUM_LANDMARKS);
            for point in landmarks {
                let p = camera.fit.cam_from_world_point(world, point);
                if !(p.iter().all(|x| x.is_finite()) && p[2] > 0.0) {
                    continue;
                }
                let uv = camera.fit.project(&p);
                if camera.inside_image(&uv) {
                    inside += 1;
                }
                let net = camera.net.to_net(&Vector2::new(uv[0], uv[1]));
                if net.iter().all(|x| x.is_finite()) {
                    front.push([net.x, net.y]);
                }
            }
            out[c] = Some(CameraProjection {
                inside,
                circle: min_enclosing_circle(&front).map(|circle| circle.to_array()),
            });
        }
        out
    }

    /// Boxes from θ̂ in every camera that sees it and the KeyNet views; drops a hand that no camera sees (handtrack `_plan_tracked`).
    fn plan_tracked(
        &mut self,
        side: usize,
        world: &Matrix4<f64>,
        available: &[bool; NUM_CAMERAS],
    ) -> Vec<ViewRequest> {
        let history = &self.history[side];
        let Some(previous) = history.previous.as_ref() else {
            return Vec::new();
        };
        let guess: Pose = match history.before.as_ref() {
            Some(before)
                if self.config.extrapolate && history.age > self.config.extrapolate_min_age =>
            {
                extrapolate(
                    previous,
                    before,
                    self.config.extrapolation_gain,
                    self.config.extrapolation_max_step_m,
                )
            }
            _ => previous.clone(),
        };
        let landmarks = landmarks_world(&self.model, &guess, side);
        let projection = self.project(&landmarks, world, available);
        let mut order: Vec<(usize, usize, [f64; 3])> = self
            .active
            .iter()
            .filter_map(|&c| {
                projection[c].and_then(|p| {
                    if p.inside > 0 {
                        p.circle.map(|circle| (c, p.inside, circle))
                    } else {
                        None
                    }
                })
            })
            .collect();
        order.sort_by_key(|&(c, inside, _)| (std::cmp::Reverse(inside), c));
        if order.is_empty() {
            self.drop_track(side);
            return Vec::new();
        }
        order
            .iter()
            .take(self.hands.max_views)
            .map(|&(camera, _, circle)| ViewRequest {
                camera,
                side,
                planning_pose_landmarks_world: Some(landmarks),
                circle_net: circle_net(circle),
                source: CropSource::Pose,
            })
            .collect()
    }

    /// DetNet on this frameset's group of DetNet cameras (those with an image); each untracked hand gets a view in each camera that
    /// reports it, the most probable first, at most `max_views` (handtrack `_detect` / `_detect_all`), and its `detnet_hits`. The
    /// result's `detnet_camera` is the camera of the strongest detection (else the first camera DetNet ran on).
    fn detect(
        &mut self,
        step: &mut Step<'_>,
        untracked: &[usize],
    ) -> Result<Vec<ViewRequest>, HandsError> {
        let groups = self.hands.detnet_groups.clamp(1, self.detnet_cameras.len());
        let group = self.next_detnet % groups;
        self.next_detnet = (self.next_detnet + 1) % groups;
        let mut cameras = Vec::new();
        let mut images = Vec::new();
        for (i, &camera) in self.detnet_cameras.iter().enumerate() {
            if i % groups != group {
                continue;
            }
            if let Some(small) = step.inputs.small[camera] {
                cameras.push(camera);
                images.push(small);
            }
        }
        if cameras.is_empty() {
            return Ok(Vec::new());
        }
        let detections = self.perception.detect(step.nets, &cameras, &images)?;
        if detections.len() != cameras.len() {
            return Err(HandsError::Invalid(format!(
                "DetNet answered {} of {} cameras",
                detections.len(),
                cameras.len()
            )));
        }
        let probability = |k: usize, side: usize| f64::from(detections[k].probability[side]);
        let mut views = Vec::new();
        let mut strongest: Option<(f64, usize)> = None;
        for &side in untracked {
            let hits: Vec<DetNetHit> = (0..cameras.len())
                .map(|k| DetNetHit {
                    camera: cameras[k],
                    circle: detections[k].circle_net[side],
                    probability: detections[k].probability[side],
                    accepted: probability(k, side) > self.config.detnet_threshold,
                })
                .collect();
            let mut found: Vec<usize> = (0..cameras.len()).filter(|&k| hits[k].accepted).collect();
            found.sort_by(|&a, &b| probability(b, side).total_cmp(&probability(a, side)));
            step.out.detnet_hits[side] = hits;
            for &k in found.iter().take(self.hands.max_views) {
                let circle = detections[k].circle_net[side].map(f64::from);
                views.push(ViewRequest {
                    camera: cameras[k],
                    side,
                    planning_pose_landmarks_world: None,
                    circle_net: circle_net(circle),
                    source: CropSource::DetNet,
                });
                if strongest.is_none_or(|(p, _)| probability(k, side) > p) {
                    strongest = Some((probability(k, side), k));
                }
            }
        }
        let shown = strongest.map_or(0, |(_, k)| k);
        step.out.detnet_camera = Some(cameras[shown]);
        Ok(views)
    }

    fn estimate(
        &mut self,
        step: &mut Step<'_>,
        plans: &[ViewRequest],
    ) -> Result<Vec<KeypointEstimate>, HandsError> {
        let begin = Instant::now();
        let (estimates, crops) =
            self.perception
                .estimate(step.nets, &step.inputs.full, step.world_from_rig, plans)?;
        step.timings.crops_ms += crops;
        step.timings.keynet_ms += begin.elapsed().as_secs_f64() * 1e3 - crops;
        if estimates.len() != plans.len() {
            return Err(HandsError::Invalid(format!(
                "the estimator answered {} of {} views",
                estimates.len(),
                plans.len()
            )));
        }
        Ok(estimates)
    }

    /// 1 per keypoint, or 0 where the planning pose projects it outside this camera's image (handtrack `_clear`, `mask_out_of_image`).
    fn clear(&self, view: &ViewRequest, world: &Matrix4<f64>) -> [f64; NUM_LANDMARKS] {
        let mut clear = [1.0; NUM_LANDMARKS];
        let Some(landmarks) = view.planning_pose_landmarks_world.as_ref() else {
            return clear;
        };
        if !self.config.mask_out_of_image {
            return clear;
        }
        let camera = &self.cameras[view.camera];
        let (width, height) = camera.size();
        let margin = self.config.image_margin_px;
        for (i, point) in landmarks.iter().enumerate() {
            let p = camera.fit.cam_from_world_point(world, point);
            let uv = camera.fit.project(&p);
            let inside = uv[0] >= margin - 0.5
                && uv[1] >= margin - 0.5
                && uv[0] < f64::from(width) - 0.5 - margin
                && uv[1] < f64::from(height) - 0.5 - margin
                && p[2] > 0.0;
            if !inside {
                clear[i] = 0.0;
            }
        }
        clear
    }

    /// KeyNet's answer as the fit's observation of `view` (handtrack `_seen` and `_weights`).
    fn seen(
        &self,
        view: &ViewRequest,
        estimate: &KeypointEstimate,
        world: &Matrix4<f64>,
    ) -> ViewObservation {
        let clear = self.clear(view, world);
        let weights: [f64; NUM_LANDMARKS] = std::array::from_fn(|i| {
            let usable = f64::from(estimate.confidence[i]) >= self.config.min_keypoint_confidence
                && estimate.points_px[i].iter().all(|x| x.is_finite())
                && estimate.d_rel_mm[i].is_finite();
            if usable { clear[i] } else { 0.0 }
        });
        ViewObservation {
            camera: view.camera,
            keypoints_px: estimate.points_px.map(|p| p.map(f64::from)),
            weights,
            d_rel_mm: estimate.d_rel_mm.map(f64::from),
        }
    }

    /// KeyNet on `views`; the hands with at least one view above the presence threshold, and the hands whose every view fell below
    /// it (handtrack `_observe`, without the UmeTrack estimator's DetNet confirmation).
    fn observe(
        &mut self,
        step: &mut Step<'_>,
        views: &[ViewRequest],
    ) -> Result<(Vec<HandObservation>, Vec<usize>), HandsError> {
        let estimates = self.estimate(step, views)?;
        let world = &step.world;
        let mut hands = Vec::new();
        let mut rejected = Vec::new();
        for side in [LEFT, RIGHT] {
            let mine: Vec<usize> = (0..views.len())
                .filter(|&i| views[i].side == side)
                .collect();
            let mut good: Vec<usize> = mine
                .iter()
                .copied()
                .filter(|&i| f64::from(estimates[i].presence) >= self.config.presence_threshold)
                .collect();
            let history = &mut self.history[side];
            if self.config.end_on_view_rejection && history.previous.is_some() && mine.len() >= 2 {
                if good.len() == 1 {
                    history.rejections += 1;
                    if history.rejections >= self.config.rejection_patience {
                        good.clear();
                    }
                } else {
                    history.rejections = 0;
                }
            }
            step.out.keynet_views[side] = mine
                .iter()
                .map(|&i| KeyNetView {
                    camera: views[i].camera,
                    keypoints_px: estimates[i].points_px,
                    confidence: estimates[i].confidence,
                    d_rel_mm: estimates[i].d_rel_mm,
                    presence: estimates[i].presence,
                    pinch: estimates[i].pinch,
                    outcome: if f64::from(estimates[i].presence) < self.config.presence_threshold
                        || estimates[i].presence.is_nan()
                    {
                        ViewOutcome::LowPresence
                    } else if good.contains(&i) {
                        ViewOutcome::Fitted
                    } else {
                        ViewOutcome::LostPair
                    },
                    crop: estimates[i].crop,
                    crop_source: views[i].source,
                })
                .collect();
            if !mine.is_empty() && good.is_empty() {
                rejected.push(side);
            }
            if !good.is_empty() {
                hands.push(HandObservation {
                    side,
                    views: good
                        .iter()
                        .map(|&i| self.seen(&views[i], &estimates[i], world))
                        .collect(),
                });
            }
        }
        Ok((hands, rejected))
    }

    /// One `acquire_recrop` pass: acquisition views whose KeyNet presence passes get the enclosing circle of KeyNet's keypoints
    /// (source [`CropSource::Recrop`]).
    /// Only KeyNet runs here: the presence rules of [`Self::observe`] change state only for tracked hands, and these are not.
    fn recrop_acquisitions(
        &mut self,
        step: &mut Step<'_>,
        mut views: Vec<ViewRequest>,
    ) -> Result<Vec<ViewRequest>, HandsError> {
        let fresh: Vec<usize> = (0..views.len())
            .filter(|&i| views[i].planning_pose_landmarks_world.is_none())
            .collect();
        if fresh.is_empty() {
            return Ok(views);
        }
        let subset: Vec<ViewRequest> = fresh.iter().map(|&i| views[i]).collect();
        let estimates = self.estimate(step, &subset)?;
        for (estimate, &index) in estimates.iter().zip(&fresh) {
            let presence = f64::from(estimate.presence);
            if !presence.is_finite()
                || presence < self.config.presence_threshold
                || !estimate.points_net.iter().flatten().all(|x| x.is_finite())
            {
                continue;
            }
            let points_net: [[f64; 2]; NUM_LANDMARKS] =
                estimate.points_net.map(|p| p.map(f64::from));
            if let Some(circle) = min_enclosing_circle(&points_net).map(|circle| circle.to_array())
                && circle.iter().all(|x| x.is_finite())
            {
                views[index].circle_net = circle_net(circle);
                views[index].source = CropSource::Recrop;
            }
        }
        Ok(views)
    }

    /// handfit Views of a hand, the inputs rounded to float32 as handfit's Python binding receives them.
    fn fit_views(&self, hand: &HandObservation, world: &Matrix4<f64>) -> Vec<View> {
        hand.views
            .iter()
            .map(|view| {
                fit_view(
                    &self.cameras[view.camera],
                    world,
                    &view.keypoints_px,
                    &view.weights,
                    &view.d_rel_mm,
                )
            })
            .collect()
    }

    /// The native fit of each hand: warm from θ(t−1) for a tracked hand, handfit's cold start for an acquisition (handtrack `_fit`).
    /// [`HandsError::Invalid`] when handfit refuses a hand's views (more than two; `max_views` keeps them to two).
    fn fit_hands(
        &self,
        views: &[Vec<View>],
        hands: &[HandObservation],
    ) -> Result<Vec<FitAnswer>, HandsError> {
        let config = Config {
            phi: self.phi,
            ..self.config.fit.clone()
        };
        hands
            .iter()
            .zip(views)
            .map(|(hand, views)| {
                let result = match self.history[hand.side].previous.as_ref() {
                    Some(prior) => fit(
                        &self.model,
                        &config,
                        prior,
                        mirror(hand.side),
                        views,
                        JacobianMode::Analytic,
                    ),
                    None => initial_pose_parallel(
                        &self.model,
                        &config,
                        mirror(hand.side),
                        views,
                        JacobianMode::Analytic,
                        self.hands.acquire_threads,
                    ),
                };
                let result = result.map_err(|e| HandsError::Invalid(format!("hand fit: {e}")))?;
                Ok(FitAnswer {
                    pose: pose_f32(&result.pose),
                    e_2d: f32_round(result.energies[0]),
                    converged: result.converged,
                })
            })
            .collect()
    }

    /// KeyNet on every view, the presence rules, the fit of the hands that remain and the acceptance gates (handtrack
    /// `_keypoints_and_fit`).
    fn keypoints_and_fit(
        &mut self,
        step: &mut Step<'_>,
        mut views: Vec<ViewRequest>,
    ) -> Result<(), HandsError> {
        for _ in 0..self.config.acquire_recrop {
            views = self.recrop_acquisitions(step, views)?;
        }
        let (hands, rejected) = self.observe(step, &views)?;
        let world = &step.world;
        for side in rejected {
            self.drop_track(side);
        }
        if hands.is_empty() {
            return Ok(());
        }
        let begin = Instant::now();
        let fit_views: Vec<Vec<View>> = hands
            .iter()
            .map(|hand| self.fit_views(hand, world))
            .collect();
        let answers = self.fit_hands(&fit_views, &hands)?;
        step.timings.fit_ms += begin.elapsed().as_secs_f64() * 1e3;
        let headset = world.fixed_view::<3, 1>(0, 3).into_owned();
        for ((hand, answer), views) in hands.iter().zip(answers).zip(fit_views) {
            let side = hand.side;
            let reach = (answer.pose.translation - headset).norm();
            let acquiring = self.history[side].previous.is_none();
            let weighted: f64 = hand
                .views
                .iter()
                .map(|v| v.weights.iter().sum::<f64>())
                .sum();
            let rms = if answer.e_2d.is_finite() {
                (answer.e_2d / weighted.max(1.0)).sqrt()
            } else {
                f64::INFINITY
            };
            let relative_limit =
                self.config.acquire_max_relative_rms * keypoint_spread_px(&hand.views);
            let rejection = if acquiring && rms > self.config.acquire_max_rms_px {
                Some(ViewOutcome::FitResidual {
                    rms_px: rms as f32,
                    limit_px: self.config.acquire_max_rms_px as f32,
                })
            } else if acquiring && rms > relative_limit {
                Some(ViewOutcome::FitResidual {
                    rms_px: rms as f32,
                    limit_px: relative_limit as f32,
                })
            } else if (acquiring && !answer.converged)
                || !pose_finite(&answer.pose)
                || reach > self.config.max_reach_m
            {
                Some(ViewOutcome::FitFailed)
            } else {
                None
            };
            if let Some(outcome) = rejection {
                for view in step.out.keynet_views[side]
                    .iter_mut()
                    .filter(|v| v.outcome == ViewOutcome::Fitted)
                {
                    view.outcome = outcome;
                }
                self.drop_track(side);
                continue;
            }
            let history = &mut self.history[side];
            history.before = history.previous.take();
            history.previous = Some(answer.pose.clone());
            history.age += 1;
            if acquiring {
                history.confirm = self.config.confirm_frames;
            }
            if history.age <= history.confirm {
                continue; // tentative: tracked internally, not reported yet
            }
            step.out.reported[side] = true;
            step.out.calibration.push(CalibrationBlock {
                mirror: mirror(side),
                views,
                initial: answer.pose,
            });
        }
        Ok(())
    }

    /// The live calibration's bookkeeping after a frameset; switch to its phi when it finished.
    fn advance_scale(&mut self, t_ns: i64, blocks: Vec<CalibrationBlock>) {
        if let Some(outcome) = self
            .scale
            .advance(t_ns, blocks, self.generic.model(), self.phi)
        {
            self.set_scale(&outcome);
        }
    }

    /// Wait for a running calibration to finish and switch to its scale (tests and replays that want a deterministic switch).
    pub fn finish_scale(&mut self) {
        if let Some(outcome) = self.scale.finish(self.phi) {
            self.set_scale(&outcome);
        }
    }

    /// Track one frameset (handtrack `Tracker.step`).
    ///
    /// # Errors
    ///
    /// Errors of the perception backends.
    pub fn track(
        &mut self,
        inputs: &HandInputs<'_>,
        world_from_rig: &Isometry3<f64>,
        nets: &mut dyn HandNets,
    ) -> Result<HandFrameResult, HandsError> {
        let start = Instant::now();
        let world = world_matrix(world_from_rig);
        let mut step = Step {
            nets,
            inputs,
            world_from_rig,
            world,
            out: FrameOutput::default(),
            timings: HandTimings::default(),
        };
        if world.iter().all(|x| x.is_finite()) {
            let available: [bool; NUM_CAMERAS] = std::array::from_fn(|c| inputs.full[c].is_some());
            let untracked: Vec<usize> = [LEFT, RIGHT]
                .into_iter()
                .filter(|&side| self.history[side].previous.is_none())
                .collect();
            let mut views = Vec::new();
            for side in [LEFT, RIGHT] {
                if self.history[side].previous.is_some() {
                    let planned = self.plan_tracked(side, &world, &available);
                    step.out.predicted[side] = planned
                        .first()
                        .and_then(|view| view.planning_pose_landmarks_world);
                    views.extend(planned);
                }
            }
            if !untracked.is_empty() {
                let begin = Instant::now();
                views.extend(self.detect(&mut step, &untracked)?);
                step.timings.detnet_ms += begin.elapsed().as_secs_f64() * 1e3;
            }
            if !views.is_empty() {
                self.keypoints_and_fit(&mut step, views)?;
            }
        } else {
            self.drop_track(LEFT);
            self.drop_track(RIGHT);
        }
        let Step {
            mut out,
            mut timings,
            ..
        } = step;
        let blocks = std::mem::take(&mut out.calibration);
        self.advance_scale(inputs.t_ns, blocks);
        let hands: [HandOutput; 2] = std::array::from_fn(|side| {
            let pose = self.history[side].previous.as_ref();
            let hits = std::mem::take(&mut out.detnet_hits[side]);
            // The hand's strongest accepted detection, ties to the lower camera (for display).
            let strongest = hits.iter().filter(|hit| hit.accepted).reduce(|best, hit| {
                if hit.probability > best.probability {
                    hit
                } else {
                    best
                }
            });
            HandOutput {
                tracked: pose.is_some(),
                reported: out.reported[side],
                landmarks_world: pose.map(|pose| landmarks_world(&self.model, pose, side)),
                pose: pose.cloned(),
                detnet_circle: strongest.map(|hit| hit.circle),
                detnet_camera: strongest.map(|hit| hit.camera),
                detnet_hits: hits,
                keynet_views: std::mem::take(&mut out.keynet_views[side]),
                predicted_landmarks_world: out.predicted[side],
            }
        });
        let spent = timings.detnet_ms + timings.crops_ms + timings.keynet_ms + timings.fit_ms;
        timings.tracker_ms = (start.elapsed().as_secs_f64() * 1e3 - spent).max(0.0);
        Ok(HandFrameResult {
            hands,
            detnet_camera: out.detnet_camera,
            scale: self.phi,
            scale_final: self.scale.is_final(),
            timings,
            world_from_rig: Some(*world_from_rig),
        })
    }
}

impl HandTracking for Tracker {
    fn step(
        &mut self,
        inputs: &HandInputs<'_>,
        world_from_rig: &Isometry3<f64>,
        nets: &mut dyn HandNets,
    ) -> Result<HandFrameResult, HandsError> {
        self.track(inputs, world_from_rig, nets)
    }
}

#[cfg(test)]
#[path = "tracker_tests.rs"]
mod tests;
