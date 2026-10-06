//! Hand tracking: the Rust port of handtrack's `Tracker` with `ROBUST_TRACKER_CONFIG` (commit 54eaf309, including
//! `acquire_max_relative_rms = 0.08`) and the native (handfit) fit, generalised to all six RoboCap cameras.
//!
//! This file is the interface the runtime builds against; the perception stages and the tracker live in the submodules.

use nalgebra::Isometry3;

use crate::frame::Luma;
use crate::frame::{NUM_CAMERAS, Rig};
use crate::nets::{HandNets, NUM_LANDMARKS, NetsError};
use kornia_staging_sensors::CameraFrame;

// Perception: cameras, the DetNet letterbox and decode, perspective KeyNet crops and decode.
pub mod camera;
pub mod detect;
pub mod estimator;
pub mod heatmaps;
pub mod letterbox;
pub mod mesh;
pub mod perspective;
// Tracking: the tracker state machine, the hand model, enclosing circles, the scale calibration.
pub mod model;
pub mod perception;
pub mod scale;
pub mod tracker;

/// Hand slot of the left hand (handtrack `Side.LEFT`).
pub const LEFT: usize = 0;
/// Hand slot of the right hand.
pub const RIGHT: usize = 1;

/// Errors of the hand tracker.
#[derive(Debug, thiserror::Error)]
pub enum HandsError {
    /// A network backend failed.
    #[error(transparent)]
    Nets(#[from] NetsError),
    /// The hand model could not be built or loaded.
    #[error("hand model: {0}")]
    Model(String),
    /// Inputs or configuration were invalid.
    #[error("{0}")]
    Invalid(String),
}

/// How the hand model's scale phi is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaleMode {
    /// Calibrate live over the first `seconds` of tracking (handtrack's section 3.6 calibration), then refit with it.
    Auto {
        /// Seconds of tracking to calibrate over.
        seconds: f64,
    },
    /// A known scale (parity runs; fallback 0.971 = Pablo's hand on s7).
    Fixed(f64),
}

/// Tracker settings; the defaults are ROBUST_TRACKER_CONFIG plus the gate.
#[derive(Clone, Debug)]
pub struct HandsConfig {
    /// How phi is chosen.
    pub scale: ScaleMode,
    /// Cameras the hands may use (subset of 0..6), e.g. `[0, 1, 2, 3]` for parity with the 4-camera Python run.
    pub cameras: Vec<usize>,
    /// At most this many views per hand (handtrack `max_views`).
    pub max_views: usize,
    /// The cameras DetNet looks at while a hand is untracked; `None` = `cameras`.
    pub detnet_cameras: Option<Vec<usize>>,
    /// Split the DetNet cameras into this many interleaved groups (camera i of the sorted list in group i % groups) and run one
    /// group per frameset.
    /// - 1 (the default) = every camera every frameset, handtrack's `detnet_all_cameras` (on full s66 with six cameras it reports
    ///   left 0.708 / right 0.659 / both 0.620 against round robin's 0.580 / 0.606 / 0.438);
    /// - one group per camera (or more) = one camera per frameset, ROBUST_TRACKER_CONFIG's round robin (parity runs against
    ///   s66-cams4/cams6.jsonl);
    /// - 2 on six cameras = {0, 2, 4} (the left cameras) and {1, 3, 5} (the right) on alternate framesets, half the NPU work
    ///   (Cap B, A55 cores: a 6-camera DetNet pass costs ~24 ms of a 33 ms frame).
    pub detnet_groups: usize,
    /// With `ScaleMode::Auto`: wait for the scale solve in the step that starts it, so a replay switches scale on the same
    /// frameset every run (the step then takes the solve's time: ~0.2-1.4 s on an A76). False (live): solve in the background.
    pub scale_wait: bool,
    /// Threads for an acquisition's cold fit (handfit's parallel cold start gives the same result for any count): the hands
    /// stage's cores, so a cold fit does not spill onto the downsample's.
    pub acquire_threads: usize,
}

impl Default for HandsConfig {
    fn default() -> Self {
        Self {
            scale: ScaleMode::Auto { seconds: 10.0 },
            cameras: (0..NUM_CAMERAS).collect(),
            max_views: 2,
            detnet_cameras: None,
            detnet_groups: 1,
            scale_wait: false,
            acquire_threads: 4,
        }
    }
}

/// One frameset's images for the tracker.
pub struct HandInputs<'a> {
    /// Cameras whose full-resolution pixels need a half-turn to match calibration.
    pub turned_180: [bool; NUM_CAMERAS],
    /// Frameset index.
    pub index: u64,
    /// Frameset time, nanoseconds.
    pub t_ns: i64,
    /// The 1920x1080 frame per camera (crop source); KeyNet's crops of a frame turned 180 degrees are sampled upright. DetNet
    /// needs nothing: it sees the small images, which the downsample turns upright.
    pub full: [Option<&'a CameraFrame>; NUM_CAMERAS],
    /// 640x360 luma per camera (DetNet letterbox content).
    pub small: [Option<&'a Luma>; NUM_CAMERAS],
}

/// What DetNet said about one untracked hand in one camera this frameset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetNetHit {
    /// Camera index.
    pub camera: usize,
    /// Circle (cx, cy, r) in the camera's DetNet net frame.
    pub circle: [f32; 3],
    /// Probability that the hand is in the image.
    pub probability: f32,
    /// Above the DetNet threshold (a candidate view; at most `max_views` of them are tried).
    pub accepted: bool,
}

/// What the tracker did with one KeyNet view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViewOutcome {
    /// In the fit, and the fit was kept.
    Fitted,
    /// KeyNet presence below the threshold.
    LowPresence,
    /// Presence passed, but the track's other view failed it and `end_on_view_rejection` ended the track.
    LostPair,
    /// In an acquisition fit whose residual was too large.
    FitResidual {
        /// The fit's RMS residual, pixels.
        rms_px: f32,
        /// The limit it broke, pixels (absolute, or relative to the keypoints' spread).
        limit_px: f32,
    },
    /// In a fit that did not converge, came out non-finite or out of reach.
    FitFailed,
}

/// What a KeyNet crop was aimed at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CropSource {
    /// An acquisition: DetNet's circle.
    DetNet,
    /// An acquisition after `acquire_recrop`: the circle around KeyNet's first answer.
    Recrop,
    /// A tracked hand: its predicted (extrapolated) pose.
    Pose,
}

/// One KeyNet answer for one hand in one camera this frameset, used or not.
#[derive(Clone, Debug)]
pub struct KeyNetView {
    /// Camera index.
    pub camera: usize,
    /// KeyNet's 21 keypoints in the camera's full-resolution pixels.
    pub keypoints_px: [[f32; 2]; NUM_LANDMARKS],
    /// Per keypoint: the heatmap's peak value.
    pub confidence: [f32; NUM_LANDMARKS],
    /// Per keypoint: KeyNet's distance relative to the hand's mean, millimetres of the generic hand.
    pub d_rel_mm: [f32; NUM_LANDMARKS],
    /// KeyNet presence probability.
    pub presence: f32,
    /// KeyNet pinch probability, with the pinch head.
    pub pinch: Option<f32>,
    /// What the tracker did with the view.
    pub outcome: ViewOutcome,
    /// The perspective crop KeyNet saw (`None` from perception without one, e.g. test fakes).
    pub crop: Option<perspective::CropCamera>,
    /// What the crop was aimed at.
    pub crop_source: CropSource,
}

/// One hand's state after a frameset.
#[derive(Clone, Debug, Default)]
pub struct HandOutput {
    /// Tracked internally this frame.
    pub tracked: bool,
    /// Reported (tracked and past `confirm_frames`); only reported hands are drawn.
    pub reported: bool,
    /// 21 landmarks in the world frame (metres), when tracked.
    pub landmarks_world: Option<[[f64; 3]; NUM_LANDMARKS]>,
    /// The fitted pose (wrist in the world, metres; joint angles), when tracked: what the viewer skins the hand mesh on.
    pub pose: Option<handfit::Pose>,
    /// DetNet circle (cx, cy, r) in the net frame of this hand's `detnet_camera`, when this hand was being acquired: its strongest
    /// accepted entry of `detnet_hits` (ties: the lower camera).
    pub detnet_circle: Option<[f32; 3]>,
    /// The camera of `detnet_circle`.
    pub detnet_camera: Option<usize>,
    /// Every DetNet answer for this hand while it was untracked, in camera order.
    pub detnet_hits: Vec<DetNetHit>,
    /// Every KeyNet view of this hand that reached the presence rules, with what the tracker did with it.
    pub keynet_views: Vec<KeyNetView>,
    /// A tracked hand: the landmarks (world, metres) of the predicted pose its KeyNet crops were planned from.
    pub predicted_landmarks_world: Option<[[f64; 3]; NUM_LANDMARKS]>,
}

impl HandOutput {
    /// The views the kept fit used: `keynet_views` with [`ViewOutcome::Fitted`], in request order.
    pub fn fitted_views(&self) -> impl Iterator<Item = &KeyNetView> {
        self.keynet_views
            .iter()
            .filter(|view| view.outcome == ViewOutcome::Fitted)
    }
}

/// Wall-clock milliseconds per stage, this frameset.
#[derive(Clone, Copy, Debug, Default)]
pub struct HandTimings {
    /// Letterbox + DetNet + decode.
    pub detnet_ms: f64,
    /// Crop planning + sampling.
    pub crops_ms: f64,
    /// KeyNet + decode.
    pub keynet_ms: f64,
    /// handfit fits.
    pub fit_ms: f64,
    /// Everything else in the step.
    pub tracker_ms: f64,
}

/// The tracker's output for one frameset.
#[derive(Clone, Debug, Default)]
pub struct HandFrameResult {
    /// Left, right.
    pub hands: [HandOutput; 2],
    /// The camera DetNet ran on this frame, if any.
    pub detnet_camera: Option<usize>,
    /// The scale in use.
    pub scale: f64,
    /// Whether live calibration has finished.
    pub scale_final: bool,
    /// Stage timings.
    pub timings: HandTimings,
    /// The headset pose the step used (the world frame of `landmarks_world`).
    pub world_from_rig: Option<Isometry3<f64>>,
}

/// The tracker (implementation in `hands/tracker.rs`).
pub trait HandTracking: Send {
    /// Track one frameset.
    ///
    /// # Errors
    ///
    /// [`HandsError::Nets`] when a network fails (the caller may rebuild the networks), [`HandsError::Invalid`] for bad inputs.
    fn step(
        &mut self,
        inputs: &HandInputs<'_>,
        world_from_rig: &Isometry3<f64>,
        nets: &mut dyn HandNets,
    ) -> Result<HandFrameResult, HandsError>;
}

/// Build the tracker for a rig: handtrack's `Tracker` with `ROBUST_TRACKER_CONFIG` and the native fit
/// ([`tracker::Tracker`]), on DetNet + the perspective-crop KeyNet ([`perception::NetsPerception`]). DetNet's camera policy
/// comes from `config.detnet_cameras` / `detnet_groups`.
///
/// # Errors
///
/// [`HandsError::Invalid`] for a bad camera list, scale or rig; [`HandsError::Model`] when the hand model asset is broken.
pub fn new_tracker(rig: &Rig, config: HandsConfig) -> Result<Box<dyn HandTracking>, HandsError> {
    let perception = Box::new(perception::NetsPerception::new(rig)?);
    Ok(Box::new(tracker::Tracker::new(
        rig,
        &config,
        tracker::TrackerConfig::default(),
        perception,
    )?))
}
