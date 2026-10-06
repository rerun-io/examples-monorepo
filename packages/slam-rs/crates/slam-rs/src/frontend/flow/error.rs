//! Failures reported by the frontend driver and its backends.

use crate::camera::CameraError;
#[cfg(doc)]
use crate::frontend::parallel::MAX_THREADS;
use crate::pyramid::PyramidError;
use kornia_staging_imgproc::features::CenteredCellError;
#[cfg(doc)]
use kornia_staging_imgproc::features::LOWEST_THRESHOLD_RUNG;
use kornia_staging_imgproc::optical_flow::patch_tracker::TrackerError;

/// What the frontend can refuse.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FrontendError {
    /// Host image allocation failed while preparing a CPU fallback.
    #[error(transparent)]
    Image(#[from] crate::image::IngestError),
    /// Device backend failure, retained at the application boundary.
    #[cfg(feature = "gpu-core")]
    #[error(transparent)]
    Gpu(#[from] kornia_staging_gpu::runtime::GpuError),

    /// The calibration carries no cameras.
    #[error("the calibration carries no cameras")]
    NoCameras,
    /// The calibration has fewer extrinsics than cameras.
    #[error("the calibration has {intrinsics} cameras but {extrinsics} extrinsics")]
    RaggedExtrinsics {
        /// Camera models the calibration carries.
        intrinsics: usize,
        /// `T_i_c` entries it carries.
        extrinsics: usize,
    },
    /// A frameset arrived at or before the last accepted one.
    #[error("frameset timestamps must increase: got {t_ns} after {previous_t_ns}")]
    NonMonotonicFrameset {
        /// Timestamp of the last accepted frameset.
        previous_t_ns: i64,
        /// Timestamp of the frameset handed in.
        t_ns: i64,
    },
    /// The frameset does not hold one image per camera.
    #[error("expected {expected} images, got {actual}")]
    CameraCountMismatch {
        /// Cameras in the rig.
        expected: usize,
        /// Images in the frameset.
        actual: usize,
    },
    /// `optical_flow_pattern` does not name the pattern this instance runs.
    #[error("config asks for pattern {config}, this frontend runs pattern {built}")]
    PatternMismatch {
        /// `optical_flow_pattern` from the config file.
        config: i32,
        /// `Pattern::CODE` of the type parameter.
        built: i32,
    },
    /// `optical_flow_type` names an implementation that is not ported.
    #[error("optical flow type {0:?} is not ported; only frame_to_frame is")]
    UnsupportedFlowType(String),
    /// The staged detector grid rejected the image geometry.
    #[error(transparent)]
    Grid(#[from] kornia_staging_imgproc::features::CellGridError),
    /// A config field that indexes or counts is negative.
    #[error("{field} must not be negative, got {value}")]
    NegativeConfig {
        /// The config key.
        field: &'static str,
        /// What it holds.
        value: i32,
    },
    /// The keypoint budget is larger than the tracker can carry.
    #[error("max_keypoints is {max_keypoints}, the tracker's capacity is {capacity}")]
    BudgetExceedsCapacity {
        /// What the options ask for.
        max_keypoints: usize,
        /// What the tracker was built for.
        capacity: usize,
    },
    /// The tracker was built for a different pyramid depth than the config asks.
    #[error("config asks for {config} pyramid levels, the tracker runs {tracker}")]
    LevelMismatch {
        /// `optical_flow_levels + 1`.
        config: usize,
        /// What the tracker was built for.
        tracker: usize,
    },
    /// `optical_flow_detection_min_threshold` cannot stop the halving ladder.
    #[error(
        "optical_flow_detection_min_threshold is {min_threshold}, which must be at least {rung}: \
         the detector halves the FAST threshold until it drops below it, and integer division \
         never gets a threshold of zero past zero"
    )]
    ThresholdLadderNeverEnds {
        /// `optical_flow_detection_min_threshold` from the config file.
        min_threshold: i32,
        /// The lowest rung the ladder can stop at ([`LOWEST_THRESHOLD_RUNG`]).
        rung: i32,
    },
    /// The threshold ladder starts below where it stops, so it never runs.
    #[error(
        "optical_flow_detection_max_threshold is {max_threshold} and \
         optical_flow_detection_min_threshold is {min_threshold}: the ladder starts below where it \
         stops, so the detector can never add a keypoint"
    )]
    EmptyThresholdLadder {
        /// `optical_flow_detection_min_threshold` from the config file.
        min_threshold: i32,
        /// `optical_flow_detection_max_threshold` from the config file.
        max_threshold: i32,
    },
    /// A frameset image is not the size the calibration gives that camera.
    #[error(
        "camera {camera}: the calibration is for {expected_width}x{expected_height} frames, \
         got {actual_width}x{actual_height}"
    )]
    FrameSizeMismatch {
        /// Which camera.
        camera: usize,
        /// Width the calibration gives the camera.
        expected_width: usize,
        /// Height the calibration gives the camera.
        expected_height: usize,
        /// Width of the image handed in.
        actual_width: usize,
        /// Height of the image handed in.
        actual_height: usize,
    },
    /// A camera model the projection layer does not implement.
    #[error("camera: {0}")]
    Camera(#[from] CameraError),
    /// The pyramid refused the geometry.
    #[error("pyramid: {0}")]
    Pyramid(#[from] PyramidError),
    /// The tracker refused the inputs.
    #[error("tracker: {0}")]
    Tracker(#[from] TrackerError),
    /// The detector refused the inputs.
    #[error("detector: {0}")]
    Detect(#[from] CenteredCellError),
    /// The thread pool could not be built.
    #[error("could not build a pool of {threads} threads")]
    ThreadPool {
        /// Threads asked for.
        threads: usize,
    },
    /// No workers were asked for, which is not a pool anything can run on.
    #[error("threads must be at least 1")]
    NoThreads,
    /// More workers were asked for than [`MAX_THREADS`].
    #[error("threads is {threads}, the ceiling is {ceiling}")]
    TooManyThreads {
        /// Workers asked for.
        threads: usize,
        /// [`MAX_THREADS`].
        ceiling: usize,
    },
}
