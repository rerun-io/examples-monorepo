//! robocap-types: the RoboCap's data types, with no runtime, no device and no estimator. A frameset of six cameras with its
//! frames' capture metadata, an IMU sample, what a source yields, and what one SLAM step produced. Drivers produce them, the
//! pipeline passes them on, and sinks read them; nothing here starts a thread or touches a device.
//!
//! The split follows kornia: `cu-stereo-payloads` (cu-kornia-vio) and sensor-rt's types keep the wire types apart from the
//! estimator and from any runtime, so a driver can name what it produces without depending on the pipeline. These are the
//! types `UPSTREAM.md` proposes for kornia-sensors.

use std::sync::Arc;

use kornia_image::{Image, ImageSize};
use nalgebra::Isometry3;
use serde::Serialize;

/// Number of cameras on the RoboCap.
pub const NUM_CAMERAS: usize = 6;
/// Camera names in index order (= catalog `/world/rig_00/cam_00..cam_05`; V4L2 mainpaths 75, 111, 84, 66, 102, 93).
pub const CAMERA_NAMES: [&str; NUM_CAMERAS] = ["left_front", "right_front", "left_eye", "right_eye", "left", "right"];
/// The cameras slam-rs uses, in its input order (left, left_front, right_front, right), as in PR #270's live adapter.
pub const SLAM_CAMERAS: [usize; 4] = [4, 0, 1, 5];
/// Native capture size.
pub const FULL_SIZE: ImageSize = ImageSize { width: 1920, height: 1080 };
/// The "small" image: SLAM input, DetNet letterbox content and viewer video.
pub const SMALL_SIZE: ImageSize = ImageSize { width: 640, height: 360 };

/// An 8-bit luma image (kornia-rs), shared.
pub type Luma = Arc<Image<u8, 1>>;

/// Where and when a frame was captured (the shape of sensor-rt's `FrameMeta`, with slam-rs's `i64` nanoseconds). The default is
/// camera 0's upright frame 0 at time 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameMeta {
    /// The driver's frame sequence number (replay: the frameset index).
    pub seq: u64,
    /// Capture time, nanoseconds: CLOCK_MONOTONIC on the cap; the catalog's video time in replay.
    pub pts_ns: i64,
    /// Camera index 0..6 (see [`CAMERA_NAMES`]).
    pub source_id: u32,
    /// The image is the camera's view turned 180 degrees (live: a camera the vendor turns, robocap-live's `capture::vendor_turned_180`);
    /// its consumers read it upright. Replay frames are upright.
    pub turned_180: bool,
}

/// One camera's frame of a frameset.
#[derive(Clone)]
pub struct CameraFrame {
    /// Sequence number, capture time and camera.
    pub meta: FrameMeta,
    /// Full-resolution luma, 1920x1080.
    pub full: Luma,
}

/// The six cameras' frames of one trigger instant (cameras whose frame is missing are `None`).
#[derive(Clone)]
pub struct Frameset {
    /// Running frameset number from the source's start.
    pub index: u64,
    /// The frameset's time: the earliest present camera's `t_ns`.
    pub t_ns: i64,
    /// The frames by camera index; `None` for a camera whose frame is missing.
    pub cameras: [Option<CameraFrame>; NUM_CAMERAS],
}

/// One IMU0 measurement in the IMU frame, SI units: gyro and accel together, accel interpolated onto the gyro timestamp, as PR #270's
/// live adapter feeds slam-rs (`Vio::push_imu`) and as kornia-sensors' `ImuMeasurement` holds them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuSample {
    /// Time of the gyro sample, nanoseconds (the frames' clock).
    pub t_ns: i64,
    /// rad/s
    pub gyro: [f64; 3],
    /// m/s^2
    pub accel: [f64; 3],
}

/// What a source yields, in time order (IMU samples up to a frameset's time come before it).
#[derive(Clone)]
pub enum SourceEvent {
    /// One trigger instant's frames.
    Frameset(Frameset),
    /// One IMU0 sample.
    Imu(ImuSample),
}


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
