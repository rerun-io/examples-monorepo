//! Inputs, results, and errors at the pipeline boundary.

#[cfg(doc)]
use crate::Vio;
use crate::{estimator, frontend, image, imu};
use serde::{Deserialize, Serialize};

/// Offline status: a measured frame has a state; an uncovered frame needs more IMU.
/// Initialization happens in the same call that first measures, so no separate
/// initializing status is exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VioStatus {
    /// The frame arrived before the IMU samples that cover it. Nothing moved —
    /// not the frontend, neither IMU buffer, not the estimator — and the pose on
    /// the result is the last one, not this frame's: push the missing samples
    /// and call `track` again with the same frameset (D17 — no arrival order may
    /// reach the trajectory).
    NeedMoreImu,
    /// The returned pose is an estimate of this frameset's rig pose.
    Tracking,
}

/// Wall time the frontend lane spent on the last tracked frameset, nanoseconds.
///
/// The three phases the frontend measures itself
/// ([`frontend::flow::FlowTimings`]) and the preintegration [`Vio::track`] runs
/// to seed the KLT with a pose prediction (D24) — the frontend's own inertial
/// work, and no part of the estimator's stages
/// ([`estimator::StageTimings`]). Reported, never compared, like those.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrontendTimings {
    /// Building this frameset's pyramids, every camera.
    pub pyramid_ns: u64,
    /// FAST detection with cells, every camera.
    pub detect_ns: u64,
    /// Temporal KLT calls only.
    pub track_ns: u64,
    /// Cross-camera matching and epipolar filtering.
    pub stereo_ns: u64,
    /// Preintegrating the samples since the previous frameset into the KLT's prediction.
    pub imu_ns: u64,
}

/// A borrowed grayscale image: `height` rows of `width` bytes, `stride` bytes apart.
#[derive(Debug, Clone, Copy)]
pub struct ImageView<'a> {
    /// Row length in pixels.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
    /// Distance between the starts of two rows, in bytes.
    pub stride: usize,
    /// Pixel bytes, at least `stride * height` long.
    pub data: &'a [u8],
}

/// What one `track` call produced.
///
/// The pose is `world_from_rig` as `[tx, ty, tz, qx, qy, qz, qw]` (translation
/// in metres, quaternion xyzw, as the Python boundary expects); the rig frame
/// is the IMU frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VioResult {
    /// Estimator state for this frame.
    pub status: VioStatus,
    /// Timestamp of the frameset, in nanoseconds.
    pub t_ns: i64,
    /// Rig pose in the world frame, `[tx, ty, tz, qx, qy, qz, qw]`.
    pub world_from_rig: [f64; 7],
    /// Rig velocity in the world frame, m/s.
    pub velocity: [f64; 3],
    /// Gyroscope bias estimate, rad/s.
    pub gyro_bias: [f64; 3],
    /// Accelerometer bias estimate, m/s².
    pub accel_bias: [f64; 3],
}

/// Everything that can go wrong at the API boundary.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum VioError {
    /// IMU samples must arrive strictly ordered; duplicates are rejected too.
    #[error("imu sample at {t_ns} ns does not follow the previous sample at {previous_t_ns} ns")]
    NonMonotonicImu {
        /// Timestamp of the last accepted sample.
        previous_t_ns: i64,
        /// Timestamp of the rejected sample.
        t_ns: i64,
    },
    /// An IMU sample has a non-finite component. Refuse it at the boundary before
    /// it contaminates both preintegrators and all following states (D32).
    #[error("imu sample at {t_ns} ns has a non-finite {field}")]
    NonFiniteImu {
        /// Timestamp of the rejected sample.
        t_ns: i64,
        /// Which of `gyro` and `accel` carries it.
        field: &'static str,
    },
    /// A GPU backend was asked for in a build without the `gpu-wgpu` feature.
    ///
    /// The variant exists in every build so the Python surface and its stub
    /// carry the same signature whether or not the feature is on: asking for a
    /// backend that is not compiled in is a refusal, not a missing argument.
    #[error("this build has no GPU backend; rebuild with the `gpu-wgpu` cargo feature")]
    GpuUnavailable,
    /// The frameset does not hold one image per configured camera.
    #[error("expected {expected} images, got {actual}")]
    CameraCountMismatch {
        /// Cameras in the configured rig.
        expected: usize,
        /// Images in the frameset.
        actual: usize,
    },
    /// A row stride cannot be shorter than the row.
    #[error("camera {index}: stride {stride} is shorter than width {width}")]
    StrideTooSmall {
        /// Index of the offending camera.
        index: usize,
        /// Declared row length in pixels.
        width: usize,
        /// Declared row pitch in bytes.
        stride: usize,
    },
    /// `stride * height` does not fit in a `usize`, so no buffer can satisfy it.
    #[error("camera {index}: {height} rows of stride {stride} overflow the address space")]
    ImageSizeOverflow {
        /// Index of the offending camera.
        index: usize,
        /// Declared number of rows.
        height: usize,
        /// Declared row pitch in bytes.
        stride: usize,
    },
    /// The buffer does not hold `stride * height` bytes.
    #[error("camera {index}: {height} rows of stride {stride} do not fit in {len} bytes")]
    ShortImage {
        /// Index of the offending camera.
        index: usize,
        /// Declared number of rows.
        height: usize,
        /// Declared row pitch in bytes.
        stride: usize,
        /// Bytes actually supplied.
        len: usize,
    },
    /// The frontend refused the configuration, the rig or a frame.
    #[error("frontend: {0}")]
    Frontend(#[from] frontend::flow::FrontendError),
    /// The estimator refused the configuration or a frame.
    #[error("estimator: {0}")]
    Estimator(#[from] estimator::EstimatorError),
    /// An image could not be widened into the frontend's `u16` buffer.
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    /// The frontend's own preintegration (D24) refused a sample.
    #[error("imu: {0}")]
    Imu(#[from] imu::ImuError),
    /// The operating system refused the thread for a deferred keyframe's joint
    /// solve (D84).
    #[error("the deferred keyframe solve thread did not start: {0}")]
    DeferredSolveThread(String),
    /// The thread running a deferred keyframe's joint solve (D84) panicked.
    #[error("the deferred keyframe solve panicked")]
    DeferredSolvePanicked,
}

/// Which frontend backend a [`Vio`] runs.
///
/// The stage traits make the choice a construction-time one (decision D21): the
/// CPU implementations stay in the crate permanently and a GPU build only adds a
/// second pair. `Cpu` is the default everywhere — the fleet's installs, the
/// gates and the accuracy references all run it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// The ported CPU frontend.
    #[default]
    Cpu,
    /// The CubeCL frontend on this host's GPU.
    ///
    /// Available only in a build with the `gpu-wgpu` feature; [`Vio::with_backend`]
    /// returns [`VioError::GpuUnavailable`] otherwise, so the Python surface
    /// carries the same signature either way.
    Gpu,
}
