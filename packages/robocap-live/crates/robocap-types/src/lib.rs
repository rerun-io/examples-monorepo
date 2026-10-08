//! RoboCap camera constants and SLAM result types. Sensor payloads live in kornia-staging-sensors.

use kornia_image::ImageSize;
use nalgebra::Isometry3;
use serde::Serialize;

/// Number of cameras on the RoboCap.
pub const NUM_CAMERAS: usize = 6;
/// Camera names in index order (= catalog `/world/rig_00/cam_00..cam_05`; V4L2 mainpaths 75, 111, 84, 66, 102, 93).
pub const CAMERA_NAMES: [&str; NUM_CAMERAS] = [
    "left_front",
    "right_front",
    "left_eye",
    "right_eye",
    "left",
    "right",
];
/// The cameras slam-rs uses, in its input order (left_front, right_front, left, right), with the front stereo pair first.
pub const SLAM_CAMERAS: [usize; 4] = [0, 1, 4, 5];
/// Native capture size.
pub const FULL_SIZE: ImageSize = ImageSize {
    width: 1920,
    height: 1080,
};
/// The "small" image: SLAM input, DetNet letterbox content and viewer video.
pub const SMALL_SIZE: ImageSize = ImageSize {
    width: 640,
    height: 360,
};

/// What one SLAM step produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlamStatus {
    /// The world has not started, or the pose lacks visual support (PR #270's rule: tracking, >= 10 landmarks, >= 10 tracked
    /// observations, optimisation started, finite).
    NoVisualFeatures,
    /// A visually supported pose.
    Tracking,
    /// slam-rs failed on this frameset; the estimator was reset.
    Failed,
    /// `--slam off`.
    Off,
    /// `--slam reference`.
    Reference,
}

impl SlamStatus {
    /// A short name for logs and the record.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoVisualFeatures => "no_visual_features",
            Self::Tracking => "tracking",
            Self::Failed => "failed",
            Self::Off => "off",
            Self::Reference => "reference",
        }
    }
}

/// One SLAM pose.
#[derive(Clone, Copy, Debug)]
pub struct SlamPose {
    /// The frameset's index.
    pub index: u64,
    /// The frameset's time.
    pub t_ns: i64,
    /// Rig (= IMU0) pose in the SLAM world.
    pub world_from_rig: Isometry3<f64>,
    /// Whether the pose is usable (visually supported tracking, or a reference pose).
    pub ok: bool,
    /// What the estimator decided.
    pub status: SlamStatus,
    /// Wall time of the call that returned this pose, ms (0 without SLAM); see [`SlamStages`].
    pub compute_ms: f64,
    /// Landmarks in the window.
    pub landmarks: usize,
    /// Observations of window landmarks in this frameset.
    pub tracked: usize,
    /// Whether the window optimisation has started (5 states).
    pub optimised: bool,
    /// Timings of the call that returned this pose; see [`SlamStages`].
    pub stages: SlamStages,
    /// Estimator restarts so far.
    pub resets: u64,
}

impl SlamPose {
    /// A pose no `Vio::track` produced (SLAM off, a reference pose, a failed step): identity, not ok, no landmarks or timings.
    pub fn untracked(index: u64, t_ns: i64, status: SlamStatus) -> Self {
        Self {
            index,
            t_ns,
            world_from_rig: Isometry3::identity(),
            ok: false,
            status,
            compute_ms: 0.0,
            landmarks: 0,
            tracked: 0,
            optimised: false,
            stages: SlamStages::default(),
            resets: 0,
        }
    }
}

/// Work performed by one track/flush call, ms. With lag, frontend work belongs to the next submitted frameset,
/// while estimator work and keyframe flags belong to the returned pose. These stages overlap and must not be summed.
/// Flush does only estimator work, with zero frontend time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SlamStages {
    /// The frontend: pyramids + FAST + KLT + stereo + the IMU prediction.
    pub frontend_ms: f64,
    /// The estimator's LM loop.
    pub optimize_ms: f64,
    /// Marginalisation.
    pub marginalize_ms: f64,
    /// Keyframe decision + landmark initialisation.
    pub keyframe_ms: f64,
    /// Whether the frameset took a keyframe.
    pub keyframe: bool,
    /// Of `frontend_ms`: pyramids, FAST detection, temporal KLT, stereo matching.
    pub pyramid_ms: f64,
    /// FAST detection.
    pub detect_ms: f64,
    /// Temporal KLT.
    pub track_ms: f64,
    /// Cross-camera matching + epipolar filter.
    pub stereo_ms: f64,
    /// The previous keyframe's deferred joint solve (triangulation, LM, marginalisation), which ran on a second thread beside
    /// this frameset's frontend; 0 when none was pending. Not part of `compute_ms` except for `deferred_wait_ms`.
    pub deferred_ms: f64,
    /// Of `compute_ms`: how long this frameset waited for that solve after its own frontend was done.
    pub deferred_wait_ms: f64,
    /// Whether this keyframe frameset left its joint solve to the next frameset (its pose is the frame update's).
    pub keyframe_deferred: bool,
}
