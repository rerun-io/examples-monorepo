//! Visual-inertial odometry core.
//!
//! The crate is deliberately free of Python, Rerun and GPU code: it consumes
//! grayscale images and IMU samples and returns plain values. Python plumbing
//! (catalog feed, evaluation, logging) lives in the `slam_rs` package and the
//! bindings in `slam-rs-py`; `slam-rs-cli` is a placeholder binary whose only
//! working subcommand is `version`.
//!
//! [`Vio`] is the Offline driver (D17): [`Vio::push_imu`] buffers samples and
//! [`Vio::track`] runs the frontend and then the estimator to completion in the
//! calling thread, so every result is final and a repeat run over the same input
//! is bit-identical. Realtime mode — basalt's two threads joined by bounded
//! queues — is stage S10's.

pub mod ba_base;
pub mod calib;
pub mod camera;
pub mod config;
pub mod eigen;
pub mod estimator;
pub mod frontend;
#[cfg(feature = "gpu-core")]
pub mod gpu;

// `gpu-core` is the kernels and the seam; a runtime comes from `gpu` (CUDA) or
// `gpu-wgpu`. Enabled on its own there would be no client to build one on, and
// the failure would be a wall of missing items rather than a sentence.
#[cfg(all(feature = "gpu-core", not(any(feature = "gpu", feature = "gpu-wgpu"))))]
compile_error!(
    "feature `gpu-core` carries the CubeCL kernels but no runtime: enable \
     `gpu` for the NVIDIA lane or `gpu-wgpu` for the portable one"
);
pub mod image;
pub mod imu;
pub mod landmark;
pub mod lie;
pub mod linearize;
pub mod marg;
pub mod pyramid;
pub mod types;

use nalgebra::{Isometry3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// Version of the core, as declared in `crates/slam-rs/Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Which GPU runtime this build's frontend carries, or `None` for the CPU-only
/// default.
///
/// The `gpu` and `gpu-wgpu` features are two builds of one source behind one
/// `gpu: bool`, so nothing a caller can pass says which of them it is running.
/// This is that fact, and it is read on the Python side (`_core.gpu_backend`) to
/// name the lane a fleet row was measured on — which the two lanes need, because
/// they do not agree on every clip.
///
/// [`gpu::BACKEND_NAME`] is the same name; this wrapper is what a build without
/// the feature can still answer.
#[cfg(feature = "gpu-core")]
pub const GPU_BACKEND: Option<&str> = Some(gpu::BACKEND_NAME);

/// Which GPU runtime this build's frontend carries: none, this being the
/// off-by-default CPU port the fleet installs.
#[cfg(not(feature = "gpu-core"))]
pub const GPU_BACKEND: Option<&str> = None;

/// Elapsed nanoseconds, saturating rather than panicking on an absurd clock.
///
/// The one place a stage mark is taken: the estimator's six
/// ([`estimator::StageTimings`]) and the frontend's three
/// ([`frontend::flow::FlowTimings`]) are the same measurement of different work.
pub(crate) fn duration_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// How far the estimator has got.
///
/// Offline mode has exactly these two states: basalt's estimator initialises
/// inside the same `process_frame` that measures
/// (`sqrt_keypoint_vio.cpp:263-296`), so a measured frameset always has a state
/// and an uncovered one never does. There is no third, "initialising" status a
/// caller could branch on.
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
/// work, and no part of the estimator's six stages
/// ([`estimator::StageTimings`]). Reported, never compared, like those.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrontendTimings {
    /// Building this frameset's pyramids, every camera.
    pub pyramid_ns: u64,
    /// FAST detection with cells, every camera.
    pub detect_ns: u64,
    /// Every KLT call: frame to frame, then camera 0 into the others.
    pub track_ns: u64,
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
    /// A sample carries a value that is not a number.
    ///
    /// basalt does not check: its samples come from a device driver. Here they
    /// come from a caller, and one non-finite component reaches both
    /// preintegrators and poisons every state after it with nothing to undo it —
    /// so it is bad input, refused at the boundary (D32), not a NaN the
    /// estimator is asked to survive.
    #[error("imu sample at {t_ns} ns has a non-finite {field}")]
    NonFiniteImu {
        /// Timestamp of the rejected sample.
        t_ns: i64,
        /// Which of `gyro` and `accel` carries it.
        field: &'static str,
    },
    /// A GPU backend was asked for in a build without the `gpu` feature.
    ///
    /// The variant exists in every build so the Python surface and its stub
    /// carry the same signature whether or not the feature is on: asking for a
    /// backend that is not compiled in is a refusal, not a missing argument.
    #[error("this build has no GPU backend; rebuild with the `gpu` cargo feature")]
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
    /// Available only in a build with the `gpu` feature; [`Vio::with_backend`]
    /// returns [`VioError::GpuUnavailable`] otherwise, so the Python surface
    /// carries the same signature either way.
    Gpu,
}

/// The frontend of a [`Vio`], on whichever backend it was built for.
///
/// A two-arm enum rather than a type parameter on [`Vio`]: the estimator is
/// already generic over its scalar, the dispatch happens once per frameset, and
/// every method below returns a type neither backend owns — so the whole cost of
/// the choice is this forwarding.
// 2.8 kB against 3.2 kB, and a pipeline holds exactly one, so boxing a variant
// would buy an indirection on the per-frame path and nothing else.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum FrontendLane {
    /// The CPU pyramid builder and patch tracker.
    Cpu(frontend::flow::FrameToFrameOpticalFlow<frontend::patterns::Pattern51>),
    /// The CubeCL pyramid builder and patch tracker.
    #[cfg(feature = "gpu-core")]
    Gpu(
        frontend::flow::FrameToFrameOpticalFlow<
            frontend::patterns::Pattern51,
            gpu::LanePyramidBuilder,
            gpu::LanePatchTracker<frontend::patterns::Pattern51>,
        >,
    ),
}

/// Run the same expression against whichever backend the lane holds.
macro_rules! on_lane {
    ($lane:expr, |$flow:ident| $body:expr) => {
        match $lane {
            FrontendLane::Cpu($flow) => $body,
            #[cfg(feature = "gpu-core")]
            FrontendLane::Gpu($flow) => $body,
        }
    };
}

impl FrontendLane {
    // ── the seven [`Vio`] drives ──────────────────────────────────────────

    /// Which backend this lane runs.
    pub fn backend(&self) -> Backend {
        match self {
            Self::Cpu(_) => Backend::Cpu,
            #[cfg(feature = "gpu-core")]
            Self::Gpu(_) => Backend::Gpu,
        }
    }

    /// `FrameToFrameOpticalFlow::check_frameset`.
    ///
    /// # Errors
    ///
    /// What the frontend refuses: a frameset of the wrong shape or a timestamp
    /// that does not follow the last one.
    pub fn check_frameset(
        &self,
        t_ns: i64,
        sizes: impl ExactSizeIterator<Item = (usize, usize)>,
    ) -> Result<(), frontend::flow::FrontendError> {
        on_lane!(self, |flow| flow.check_frameset(t_ns, sizes))
    }

    /// `FrameToFrameOpticalFlow::process_frame`.
    ///
    /// # Errors
    ///
    /// What the frontend refuses, plus a device failure on the GPU lane.
    pub fn process_frame(
        &mut self,
        t_ns: i64,
        images: &[image::ImageU16],
        prediction: &frontend::flow::PosePrediction,
        masks: &[frontend::detect::Masks],
    ) -> Result<&frontend::flow::FlowFrame, frontend::flow::FrontendError> {
        on_lane!(self, |flow| flow
            .process_frame(t_ns, images, prediction, masks))
    }

    /// What the last frame's phases cost.
    pub fn timings(&self) -> frontend::flow::FlowTimings {
        on_lane!(self, |flow| flow.timings())
    }

    /// The last committed frame's tracked keypoints.
    pub fn frame(&self) -> &frontend::flow::FlowFrame {
        on_lane!(self, |flow| flow.frame())
    }

    /// The config the frontend was built from.
    pub fn config(&self) -> &config::VioConfig {
        on_lane!(self, |flow| flow.config())
    }

    /// Publish a new average scene depth.
    pub fn set_depth_guess(&mut self, depth: f32) {
        on_lane!(self, |flow| flow.set_depth_guess(depth));
    }

    // ── the rest, which only `slam-rs-py`'s standalone `OpticalFlow` reads ─

    /// Cameras in the rig.
    pub fn camera_count(&self) -> usize {
        on_lane!(self, |flow| flow.camera_count())
    }

    /// Framesets committed so far.
    pub fn frame_counter(&self) -> u64 {
        on_lane!(self, |flow| flow.frame_counter())
    }

    /// The high-water mark of the keypoint id space.
    pub fn last_keypoint_id(&self) -> u64 {
        on_lane!(self, |flow| flow.last_keypoint_id())
    }

    /// The same mark as it stood before the last committed frameset.
    pub fn last_keypoint_id_before_frame(&self) -> u64 {
        on_lane!(self, |flow| flow.last_keypoint_id_before_frame())
    }

    /// One camera's occupancy counts.
    pub fn cell_counts(&self, camera: usize) -> &[i32] {
        on_lane!(self, |flow| flow.cell_counts(camera))
    }

    /// The occupancy grid's geometry.
    pub fn occupancy_grid(&self) -> frontend::detect::CellGrid {
        on_lane!(self, |flow| flow.occupancy_grid())
    }

    /// The last committed frameset's timestamp, `None` before the first.
    pub fn t_ns(&self) -> Option<i64> {
        on_lane!(self, |flow| flow.t_ns())
    }
}

/// Build the frontend lane a [`Backend`] names.
///
/// The CPU arm is `FrameToFrameOpticalFlow::new`. The GPU arm makes the two
/// CubeCL stage backends on one shared client and hands them to
/// `with_backends`, which is the whole of what selecting a backend costs.
fn build_frontend(
    config: &config::VioConfig,
    calibration: &calib::Calibration<f64>,
    options: frontend::flow::FrontendOptions,
    backend: Backend,
) -> Result<FrontendLane, VioError> {
    // The two counts a backend-specific arm casts to `usize` to size its
    // buffers, checked here rather than after the cast: the GPU arm's
    // `optical_flow_levels as usize + 1` panics on `-1` in a debug build and
    // wraps to zero in a release one, either way before
    // `FrameToFrameOpticalFlow::with_backends` can run the frontend's own
    // refusal. Both arms return that refusal now, on the same field and value,
    // and no device is constructed for a config no backend can run.
    for (field, value) in [
        ("optical_flow_levels", config.optical_flow_levels),
        (
            "optical_flow_max_iterations",
            config.optical_flow_max_iterations,
        ),
    ] {
        if value < 0 {
            return Err(frontend::flow::FrontendError::NegativeConfig { field, value }.into());
        }
    }

    match backend {
        Backend::Cpu => Ok(FrontendLane::Cpu(
            frontend::flow::FrameToFrameOpticalFlow::new(config.clone(), calibration, options)?,
        )),
        #[cfg(feature = "gpu-core")]
        Backend::Gpu => {
            let num_levels: usize = config.optical_flow_levels as usize + 1;
            let (pyramid, tracker, scanner) = gpu::gpu_backends::<frontend::patterns::Pattern51>(
                options.max_keypoints,
                num_levels,
                config.optical_flow_max_iterations as usize,
                config.optical_flow_max_recovered_dist2,
            )
            .map_err(frontend::flow::FrontendError::from)?;
            Ok(FrontendLane::Gpu(
                frontend::flow::FrameToFrameOpticalFlow::with_backends(
                    config.clone(),
                    calibration,
                    options,
                    pyramid,
                    tracker,
                    scanner,
                )?,
            ))
        }
        #[cfg(not(feature = "gpu-core"))]
        Backend::Gpu => Err(VioError::GpuUnavailable),
    }
}

/// The estimator, driven one frameset at a time (D17, D24).
///
/// One `track` call is basalt's whole pipeline for one frameset, in the calling
/// thread and in basalt's order:
///
/// 1. the frontend's own preintegration over `(t_prev, t_now]` and the pose
///    prediction it feeds the KLT (`frame_to_frame_optical_flow.h:138-152`) —
///    the estimator runs a **second, independent** preintegrator (D24) and the
///    two are deliberately not shared;
/// 2. `processFrame`, which produces the tracked keypoints;
/// 3. the estimator's own IMU consumption and `measure`, which optimises and
///    marginalizes;
/// 4. the two feedback values basalt pushes back to the frontend: the newest
///    state, and — because every shipped config sets
///    `optical_flow_matching_guess_type = REPROJ_AVG_DEPTH` — the average scene
///    depth from `computeProjections` (`sqrt_keypoint_vio.cpp:583-604`).
///
/// Nothing about arrival order can reach a decision: there are no queues, no
/// drops (`vio_enforce_realtime` is refused) and no threads, which is what makes
/// a repeat run bit-identical.
///
/// The frontend is `f32` throughout, as `FrameToFrameOpticalFlow<float,
/// Pattern51>` is; the estimator's scalar is the type parameter, and `f32` is
/// the shipped precision the reference lane runs (Q07).
#[derive(Debug)]
pub struct Vio<S: lie::LieScalar = f32> {
    frontend: FrontendLane,
    estimator: estimator::SqrtKeypointVio<S>,
    /// The frontend's own IMU buffer (D24). The same samples reach the
    /// estimator through its own queue.
    frontend_imu: std::collections::VecDeque<imu::ImuSample>,
    /// The frontend's already-popped sample, `processImu`'s `data` (`:169`).
    frontend_pending: Option<imu::Popped<f64>>,
    /// `latest_state` (`frame_to_frame_optical_flow.h:141-146`), which doubles
    /// as basalt's `first_state_arrived` (`:143`): `None` until the estimator
    /// has published one. `predicted_state` (`:150`) is a member there and a
    /// local here — nothing outside the prediction that produces it reads it.
    latest_state: Option<types::PoseVelBiasState<f64>>,
    /// The frontend's own preintegration noise, `accel_cov`/`gyro_cov` at
    /// `frame_to_frame_optical_flow.h:105-106`.
    frontend_noise: imu::ImuNoise<f64>,
    /// The static bias calibration, applied to the frontend's samples in `f32`
    /// and cast back to `f64` (`:171-178`).
    calib_f32: calib::Calibration<f32>,
    /// Widened frames, reused so a steady-state `track` does not allocate.
    frames: Vec<image::ImageU16>,
    /// `img->masks`, always empty here: masks come from Monado.
    masks: Vec<frontend::detect::Masks>,
    /// Cameras in the rig; every frameset must carry exactly this many.
    camera_count: usize,
    /// The last frameset's timestamp, `t_ns` in the frontend (`:172`).
    last_frame_t_ns: Option<i64>,
    /// What the last `track` decided; the S9 Rerun rung reads this.
    last_stats: Option<Box<estimator::FrameStats<S>>>,
    /// What the last tracked frameset's frontend lane cost; see [`FrontendTimings`].
    frontend_timings: FrontendTimings,
}

impl<S: lie::LieScalar> Vio<S> {
    /// Build the pipeline from basalt's own config and calibration (D18).
    ///
    /// # Errors
    ///
    /// [`VioError::Frontend`] when the config names another flow type or
    /// pattern or the rig is unusable, and [`VioError::Estimator`] when the
    /// config asks for a path this port does not have — `vio_linearization_type`
    /// other than `ABS_QR`, `vio_sqrt_marg` false, or `vio_enforce_realtime`.
    pub fn new(
        config: config::VioConfig,
        calibration: calib::Calibration<f64>,
        options: frontend::flow::FrontendOptions,
    ) -> Result<Self, VioError> {
        Self::with_backend(config, calibration, options, Backend::Cpu)
    }

    /// The same pipeline on a named frontend backend (decision D21).
    ///
    /// # Errors
    ///
    /// As [`Vio::new`], plus [`VioError::GpuUnavailable`] when `Gpu` is asked
    /// for and this build has no `gpu` feature, and [`VioError::Frontend`] when
    /// the device cannot size the frontend's buffers.
    pub fn with_backend(
        config: config::VioConfig,
        calibration: calib::Calibration<f64>,
        options: frontend::flow::FrontendOptions,
        backend: Backend,
    ) -> Result<Self, VioError> {
        let camera_count: usize = calibration.t_i_c.len();
        let frontend: FrontendLane = build_frontend(&config, &calibration, options, backend)?;
        let calib_f32: calib::Calibration<f32> = calibration.cast();
        let frontend_noise: imu::ImuNoise<f64> = imu::ImuNoise::from_calibration(&calibration);
        let estimator: estimator::SqrtKeypointVio<S> =
            estimator::SqrtKeypointVio::with_default_gravity(calibration.cast(), config)?;
        Ok(Self {
            frontend,
            estimator,
            frontend_imu: std::collections::VecDeque::new(),
            frontend_pending: None,
            latest_state: None,
            frontend_noise,
            calib_f32,
            frames: Vec::new(),
            masks: vec![frontend::detect::Masks::default(); camera_count],
            camera_count,
            last_frame_t_ns: None,
            last_stats: None,
            frontend_timings: FrontendTimings::default(),
        })
    }

    /// The estimator, for callers that want the window or the snapshot.
    pub fn estimator(&self) -> &estimator::SqrtKeypointVio<S> {
        &self.estimator
    }

    /// The frontend, for callers that want the tracked keypoints.
    pub fn frontend(&self) -> &FrontendLane {
        &self.frontend
    }

    /// Which frontend backend this pipeline runs.
    pub fn backend(&self) -> Backend {
        self.frontend.backend()
    }

    /// What the frontend lane cost on the last frameset that ran it.
    ///
    /// All zero before the first one. A frameset the IMU does not cover is
    /// refused before any of this work happens, so what stands then is the
    /// previous frameset's, exactly as the keypoints and the snapshot are.
    pub fn frontend_timings(&self) -> FrontendTimings {
        self.frontend_timings
    }

    /// What the last `track` decided, or `None` before the first one.
    pub fn last_stats(&self) -> Option<&estimator::FrameStats<S>> {
        self.last_stats.as_deref()
    }

    /// The last inertial timestamp accepted, or `None` before the first sample.
    ///
    /// The frontier [`check_imu_sample`] measures against; a caller pushing a
    /// batch reads it to check the whole batch before pushing any of it. It is
    /// the estimator's own frontier, not a second copy: `push_imu` hands every
    /// accepted sample to both preintegrators, so the two could only differ by
    /// a bug.
    pub fn last_imu_t_ns(&self) -> Option<i64> {
        self.estimator.newest_imu_t_ns()
    }

    /// Add one IMU sample: `gyro` in rad/s, `accel` in m/s², both in the rig
    /// frame, uncalibrated (the static bias calibration is applied inside).
    ///
    /// The sample reaches both preintegrators (D24).
    ///
    /// # Errors
    ///
    /// Whatever [`check_imu_sample`] refuses: a duplicate or out-of-order
    /// timestamp, never a silent reorder, and a non-finite component.
    pub fn push_imu(&mut self, t_ns: i64, gyro: [f64; 3], accel: [f64; 3]) -> Result<(), VioError> {
        check_imu_sample(t_ns, &gyro, &accel, self.last_imu_t_ns())?;
        let sample: imu::ImuSample = imu::ImuSample {
            t_ns,
            gyro: Vector3::new(gyro[0], gyro[1], gyro[2]),
            accel: Vector3::new(accel[0], accel[1], accel[2]),
        };
        self.frontend_imu.push_back(sample);
        self.estimator.push_imu(sample);
        Ok(())
    }

    /// Process one frameset: one image per camera, oldest to newest in time.
    ///
    /// Returns [`VioStatus::NeedMoreImu`] without touching anything — the
    /// frontend, both IMU buffers, the estimator — when the buffered IMU does
    /// not yet reach past `t_ns`; the same frameset may then be tracked again
    /// once the samples arrive, and the result is the one a run that had them
    /// all along would have produced (D17).
    ///
    /// A refused frameset leaves the pipeline exactly as the last accepted one
    /// did, whether it is refused for want of samples or because it is not a
    /// frameset this rig can take: every rule is checked before anything moves,
    /// so a caller may correct it and call again.
    ///
    /// # Errors
    ///
    /// [`VioError`] when the frameset is the wrong width or geometry, is not the
    /// size the calibration gives its cameras, does not follow the last accepted
    /// frameset, or when the estimator refuses it.
    pub fn track(&mut self, t_ns: i64, images: &[ImageView<'_>]) -> Result<VioResult, VioError> {
        // Every refusal is decided here, before **any** mutation, and the
        // coverage test is one of them. Everything below moves the pipeline
        // forward irreversibly — the frontend's own preintegrator eats its
        // buffer to seed the KLT, then the frontend swaps its pyramids and
        // advances its clock and counter, then the estimator eats its own queue
        // — so a rule checked halfway down spends what a retry needs: it would
        // make `NeedMoreImu` a status the caller cannot act on, and a corrected
        // frame a different trajectory from the one it belongs to. The buffer
        // geometry is this level's (the widening reads it); the clock and the
        // calibrated frame sizes are the frontend's own precondition, which
        // `process_frame` still re-runs for callers that hold it directly; the
        // coverage predicate is the estimator's, and `process_frame` still runs
        // it as its second line.
        check_frameset(images, self.camera_count)?;
        self.frontend
            .check_frameset(t_ns, images.iter().map(|image| (image.width, image.height)))?;
        if !self.estimator.imu_covers_frame(t_ns) {
            return Ok(self.result(VioStatus::NeedMoreImu, t_ns));
        }

        // `frame_to_frame_optical_flow.h:138-152`: the prediction the KLT is
        // seeded with. Until the estimator has produced a state both poses are
        // the identity, which is basalt's `first_state_arrived == false` path —
        // and here that flag is `latest_state` being `None`.
        let mark: std::time::Instant = std::time::Instant::now();
        let prediction: frontend::flow::PosePrediction = match self.latest_state {
            Some(latest) => {
                let pim: imu::IntegratedImuMeasurement<f64> =
                    self.frontend_preintegrate(t_ns, &latest)?;
                let predicted: types::PoseVelState<f64> =
                    pim.predict_state(&latest.pose_vel_state(), &imu::gravity::<f64>());
                frontend::flow::PosePrediction {
                    t_w_i_previous: latest.t_w_i.cast(),
                    t_w_i_current: predicted.t_w_i.cast(),
                }
            }
            None => frontend::flow::PosePrediction::default(),
        };
        self.frontend_timings.imu_ns = duration_ns(mark);

        // `vit_tracker.cpp:534`: the `u8 << 8` widening the whole frontend
        // assumes, into buffers that are reused frame to frame.
        self.frames
            .resize_with(images.len(), image::ImageU16::default);
        for (frame, view) in self.frames.iter_mut().zip(images.iter()) {
            frame.fill_from_u8_strided(view.data, view.width, view.height, view.stride)?;
        }
        self.frontend
            .process_frame(t_ns, &self.frames, &prediction, &self.masks)?;
        let flow: frontend::flow::FlowTimings = self.frontend.timings();
        self.frontend_timings.pyramid_ns = flow.pyramid_ns;
        self.frontend_timings.detect_ns = flow.detect_ns;
        self.frontend_timings.track_ns = flow.track_ns;
        self.last_frame_t_ns = Some(t_ns);

        // The estimator reads only the ids and the observed pixels
        // (`optical_flow.h:186-215`).
        let mut observations: estimator::FlowObservations =
            estimator::FlowObservations::new(t_ns, self.camera_count);
        debug_assert_eq!(
            observations.cameras.len(),
            self.frontend.frame().cameras.len()
        );
        for (slot, keypoints) in observations
            .cameras
            .iter_mut()
            .zip(self.frontend.frame().cameras.iter())
        {
            for (index, id) in keypoints.ids.iter().enumerate() {
                let warp: frontend::se2::AffineCompact2f = keypoints.transform(index);
                slot.insert(*id, warp.translation);
            }
        }
        let outcome: estimator::FrameOutcome<S> = self
            .estimator
            .process_frame(std::sync::Arc::new(observations))?;

        // The estimator initialises inside the same `process_frame` that
        // measures (`:263-296`), so a `Measured` outcome always has a state and
        // the outcome alone decides the status.
        let status: VioStatus = match outcome {
            estimator::FrameOutcome::NeedMoreImu => VioStatus::NeedMoreImu,
            estimator::FrameOutcome::Measured(stats) => {
                self.last_stats = Some(stats);
                // `:592-620`: the two feedback values, in basalt's order.
                self.publish_state();
                self.publish_depth_guess()?;
                VioStatus::Tracking
            }
        };

        Ok(self.result(status, t_ns))
    }

    /// The estimator's newest state as one [`VioResult`], or the identity pose
    /// before the window has one.
    fn result(&self, status: VioStatus, t_ns: i64) -> VioResult {
        let (world_from_rig, velocity, gyro_bias, accel_bias) = match self.estimator.state() {
            Some(state) => (
                pose_to_array(&Isometry3::from_parts(
                    state.t_w_i.translation.map(lie::LieScalar::to_f64).into(),
                    *state.t_w_i.rotation.cast::<f64>().quaternion(),
                )),
                state.vel_w_i.map(lie::LieScalar::to_f64).into(),
                state.bias_gyro.map(lie::LieScalar::to_f64).into(),
                state.bias_accel.map(lie::LieScalar::to_f64).into(),
            ),
            None => (
                pose_to_array(&Isometry3::identity()),
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
            ),
        };
        VioResult {
            status,
            t_ns,
            world_from_rig,
            velocity,
            gyro_bias,
            accel_bias,
        }
    }

    /// `processImu(curr_t_ns)` (`frame_to_frame_optical_flow.h:157-201`).
    ///
    /// The same three-part loop the estimator runs, over the frontend's own
    /// buffer and at `f64`: skip up to the previous frame, integrate up to this
    /// one, then close the interval by retiming the next sample. The bias
    /// calibration happens in `f32` and is cast back (`:171-178`), which is what
    /// `Calibration<Scalar>` with `Scalar = float` means here.
    /// # Errors
    ///
    /// [`VioError::Imu`] when a sample does not follow the interval: the KLT is
    /// seeded from this prediction, so a truncated preintegration would move
    /// the whole trajectory silently (D32).
    fn frontend_preintegrate(
        &mut self,
        curr_t_ns: i64,
        latest: &types::PoseVelBiasState<f64>,
    ) -> Result<imu::IntegratedImuMeasurement<f64>, VioError> {
        let prev_t_ns: i64 = self.last_frame_t_ns.unwrap_or(-1);
        let mut pim: imu::IntegratedImuMeasurement<f64> =
            imu::IntegratedImuMeasurement::new(prev_t_ns, &latest.bias_gyro, &latest.bias_accel);
        // `:190-198`, the same three-part loop the estimator's own
        // preintegration runs, through `IntegratedImuMeasurement::accumulate_to`.
        let noise: imu::ImuNoise<f64> = self.frontend_noise;
        let pending: Option<imu::Popped<f64>> = self.frontend_pending.take();
        self.frontend_pending = pim.accumulate_to(
            pending,
            || self.frontend_pop(),
            prev_t_ns,
            curr_t_ns,
            &noise,
        )?;
        Ok(pim)
    }

    /// One sample off the frontend's buffer, calibrated in `f32` and cast back
    /// to `f64` (`frame_to_frame_optical_flow.h:171-178`).
    fn frontend_pop(&mut self) -> Option<imu::Popped<f64>> {
        let sample: imu::ImuSample = self.frontend_imu.pop_front()?;
        let accel: Vector3<f32> = self
            .calib_f32
            .calib_accel_bias
            .calibrated(&sample.accel.cast());
        let gyro: Vector3<f32> = self
            .calib_f32
            .calib_gyro_bias
            .calibrated(&sample.gyro.cast());
        Some((sample.t_ns, gyro.cast(), accel.cast()))
    }

    /// `opt_flow_state_queue->push(data)` (`sqrt_keypoint_vio.cpp:620`).
    fn publish_state(&mut self) {
        if let Some(state) = self.estimator.state() {
            self.latest_state = Some(types::PoseVelBiasState {
                t_ns: state.t_ns,
                t_w_i: state.t_w_i.cast(),
                vel_w_i: state.vel_w_i.map(lie::LieScalar::to_f64),
                bias_gyro: state.bias_gyro.map(lie::LieScalar::to_f64),
                bias_accel: state.bias_accel.map(lie::LieScalar::to_f64),
            });
        }
    }

    /// `opt_flow_depth_guess_queue->push(avg_depth)` (`:583-604`).
    ///
    /// `num_features / Σ inverse-depth`, or `optical_flow_matching_default_depth`
    /// when the sum is not positive. Only computed when the config asks for
    /// `REPROJ_AVG_DEPTH`, which every shipped config does.
    ///
    /// **The reduction is `f64` on both instantiations because the C++ one is**
    /// (D42): `computeProjections` is called with `Scalar2 = double`
    /// (`ba_base.cpp:540,558`), so every `proj` is widened to `Vector4d` on the
    /// way into the vector, and `:592-593` declares `avg_invdepth` and
    /// `num_features` as `double` whatever `Scalar` is. `to_f64()` here is that
    /// `cast<double>`, and the division follows it.
    ///
    /// One deviation, and it is on the way **out**: C++ carries the guess as a
    /// `double` from the queue (`vio_estimator.h:108`) into
    /// `OpticalFlowBase::depth_guess` (`optical_flow.h:164`), while the port's
    /// frontend holds it as `f32` (`FrameToFrameOpticalFlow::depth_guess`), so
    /// the quotient is rounded once here. The guess only seeds the KLT's
    /// matching window, and the f64 backend lane reproduces the C++ trajectory
    /// to 2.5e-13 m over 4,095 framesets with the same rounding in place, so it
    /// reaches no decision on the shipped path; widening the frontend's field is
    /// the fix if one ever does.
    fn publish_depth_guess(&mut self) -> Result<(), VioError> {
        // No `t_i_c.is_empty()` clause: `SqrtKeypointVio::new` refuses a rig of
        // fewer than two cameras and is the only way to build the estimator a
        // `Vio` holds, so the rig cannot be empty here.
        if self.frontend.config().optical_flow_matching_guess_type
            != config::MatchingGuessType::ReprojAvgDepth
        {
            return Ok(());
        }
        let projections: Vec<Vec<nalgebra::Vector4<S>>> = self
            .estimator
            .ba
            .compute_projections(self.estimator.last_state_t_ns())
            .map_err(estimator::EstimatorError::from)?;
        let mut avg_invdepth: f64 = 0.0;
        let mut num_features: f64 = 0.0;
        for cam in &projections {
            for entry in cam {
                avg_invdepth += entry[2].to_f64();
            }
            num_features += cam.len() as f64;
        }
        let valid: bool = avg_invdepth > 0.0 && num_features > 0.0;
        let default_depth: f32 = self.frontend.config().optical_flow_matching_default_depth;
        let avg_depth: f64 = if valid {
            num_features / avg_invdepth
        } else {
            f64::from(default_depth)
        };
        self.frontend.set_depth_guess(avg_depth as f32);
        Ok(())
    }
}

/// The rule every inertial sample meets, wherever it enters.
///
/// `previous_t_ns` is the frontier it must follow: [`Vio::last_imu_t_ns`] for
/// the first sample of a batch, and the sample before it for the rest. It is a
/// free function so a caller holding a whole batch can decide the batch before
/// pushing any of it — a batch that pushed as it went would leave the samples
/// before the bad one behind and move the frontier past them, and the caller
/// could then neither retry the batch nor correct it.
///
/// # Errors
///
/// [`VioError::NonMonotonicImu`] on a timestamp that does not strictly follow
/// the frontier, and [`VioError::NonFiniteImu`] on a component that is not a
/// number.
pub fn check_imu_sample(
    t_ns: i64,
    gyro: &[f64; 3],
    accel: &[f64; 3],
    previous_t_ns: Option<i64>,
) -> Result<(), VioError> {
    if let Some(previous_t_ns) = previous_t_ns
        && t_ns <= previous_t_ns
    {
        return Err(VioError::NonMonotonicImu {
            previous_t_ns,
            t_ns,
        });
    }
    for (field, values) in [("gyro", gyro), ("accel", accel)] {
        if !values.iter().all(|value| value.is_finite()) {
            return Err(VioError::NonFiniteImu { t_ns, field });
        }
    }
    Ok(())
}

/// The frameset geometry checks [`Vio::track`] runs before it widens anything.
///
/// Caller-controlled `width`, `height` and `stride`: an unchecked
/// `stride * height` panics in debug and wraps to an accepted zero in release,
/// so the product is checked (D32).
///
/// # Errors
///
/// [`VioError::CameraCountMismatch`], [`VioError::StrideTooSmall`],
/// [`VioError::ImageSizeOverflow`] or [`VioError::ShortImage`].
fn check_frameset(images: &[ImageView<'_>], camera_count: usize) -> Result<(), VioError> {
    if images.len() != camera_count {
        return Err(VioError::CameraCountMismatch {
            expected: camera_count,
            actual: images.len(),
        });
    }
    for (index, image) in images.iter().enumerate() {
        if image.stride < image.width {
            return Err(VioError::StrideTooSmall {
                index,
                width: image.width,
                stride: image.stride,
            });
        }
        let needed: usize =
            image
                .stride
                .checked_mul(image.height)
                .ok_or(VioError::ImageSizeOverflow {
                    index,
                    height: image.height,
                    stride: image.stride,
                })?;
        if image.data.len() < needed {
            return Err(VioError::ShortImage {
                index,
                height: image.height,
                stride: image.stride,
                len: image.data.len(),
            });
        }
    }
    Ok(())
}

/// Flatten an isometry into `[tx, ty, tz, qx, qy, qz, qw]`.
fn pose_to_array(pose: &Isometry3<f64>) -> [f64; 7] {
    let rotation: UnitQuaternion<f64> = pose.rotation;
    [
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        rotation.i,
        rotation.j,
        rotation.k,
        rotation.w,
    ]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    fn image(bytes: &[u8], width: usize, height: usize) -> ImageView<'_> {
        ImageView {
            width,
            height,
            stride: width,
            data: bytes,
        }
    }

    /// The package's own MSDMI config, the file `reference_segments.toml` names
    /// and the C++ reference runs loaded.
    const MSDMI_CONFIG: &str = include_str!("../../../configs/msdmi_config.json");

    fn pipeline() -> Vio<f32> {
        let directory: std::path::PathBuf =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let config: config::VioConfig = config::VioConfig::from_json_str(MSDMI_CONFIG).unwrap();
        let calibration: calib::Calibration<f64> = calib::Calibration::from_json_str(
            &std::fs::read_to_string(directory.join("msdmi_calib.json")).unwrap(),
        )
        .unwrap();
        Vio::new(
            config,
            calibration,
            frontend::flow::FrontendOptions {
                threads: 1,
                ..frontend::flow::FrontendOptions::default()
            },
        )
        .unwrap()
    }

    /// A frameset that arrives before the IMU covering it reports
    /// [`VioStatus::NeedMoreImu`] and the frontend does **not** run: it would
    /// swap its pyramids and advance its clock and counter, and the retry the
    /// status invites would then track the frameset against itself (D17).
    /// `vio_parity.rs` proves the retry itself; this proves nothing moved.
    #[test]
    fn a_frameset_ahead_of_the_imu_needs_more_imu() {
        let mut vio: Vio<f32> = pipeline();
        let blank: Vec<u8> = vec![0; 960 * 960];
        let views: [ImageView<'_>; 2] = [image(&blank, 960, 960), image(&blank, 960, 960)];
        let before: String = format!("{vio:?}");
        let result: VioResult = vio.track(1_000, &views).unwrap();
        assert_eq!(result.status, VioStatus::NeedMoreImu);
        assert!(!vio.estimator().is_initialized());
        assert_abs_diff_eq!(result.world_from_rig[6], 1.0, epsilon = 1e-12);
        assert_eq!(vio.frontend().frame_counter(), 0);
        // `None` rather than a sentinel: the frontend has seen no frameset.
        assert_eq!(vio.frontend().t_ns(), None);
        assert_eq!(
            format!("{vio:?}"),
            before,
            "the refused frameset moved a field"
        );
    }

    /// `vio_enforce_realtime` drops framesets, which Offline mode cannot do
    /// without letting arrival order reach a decision, so it is refused at
    /// construction rather than silently ignored.
    #[test]
    fn realtime_frame_dropping_is_refused() {
        let directory: std::path::PathBuf =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut config: config::VioConfig = config::VioConfig::from_json_str(MSDMI_CONFIG).unwrap();
        config.vio_enforce_realtime = true;
        let calibration: calib::Calibration<f64> = calib::Calibration::from_json_str(
            &std::fs::read_to_string(directory.join("msdmi_calib.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            Vio::<f32>::new(
                config,
                calibration,
                frontend::flow::FrontendOptions::default()
            )
            .err(),
            Some(VioError::Estimator(
                estimator::EstimatorError::EnforceRealtime
            ))
        );
    }

    #[test]
    fn version_is_the_crate_version() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn repeated_imu_timestamps_are_rejected() {
        let mut vio: Vio<f32> = pipeline();
        vio.push_imu(5, [0.0; 3], [0.0; 3]).unwrap();
        assert_eq!(
            vio.push_imu(5, [0.0; 3], [0.0; 3]),
            Err(VioError::NonMonotonicImu {
                previous_t_ns: 5,
                t_ns: 5
            })
        );
    }

    /// A component that is not a number is bad input, not a value to integrate:
    /// it reaches both preintegrators and there is nothing that undoes it (D32).
    #[test]
    fn a_non_finite_imu_sample_is_rejected() {
        let mut vio: Vio<f32> = pipeline();
        assert_eq!(
            vio.push_imu(5, [0.0, f64::NAN, 0.0], [0.0; 3]),
            Err(VioError::NonFiniteImu {
                t_ns: 5,
                field: "gyro"
            })
        );
        assert_eq!(
            vio.push_imu(5, [0.0; 3], [f64::NEG_INFINITY, 0.0, 0.0]),
            Err(VioError::NonFiniteImu {
                t_ns: 5,
                field: "accel"
            })
        );
        // The guard is the only thing that moved, so the same timestamp is
        // still the one the next sample has to take.
        assert!(vio.push_imu(5, [0.0; 3], [0.0; 3]).is_ok());
    }

    #[test]
    fn a_frameset_of_the_wrong_width_is_rejected() {
        let mut vio: Vio<f32> = pipeline();
        let pixels: Vec<u8> = vec![0; 16];
        assert_eq!(
            vio.track(0, &[image(&pixels, 4, 4)]),
            Err(VioError::CameraCountMismatch {
                expected: 2,
                actual: 1
            })
        );
    }

    /// The geometry of the buffer is checked before a pixel is read, so an
    /// under-long or a narrow-stride frameset is a typed refusal rather than the
    /// out-of-range read the widening would otherwise make (D32).
    #[test]
    fn a_short_buffer_is_rejected() {
        let mut vio: Vio<f32> = pipeline();
        let pixels: Vec<u8> = vec![0; 8];
        let short: [ImageView<'_>; 2] = [image(&pixels, 4, 4), image(&pixels, 4, 4)];
        assert_eq!(
            vio.track(0, &short),
            Err(VioError::ShortImage {
                index: 0,
                height: 4,
                stride: 4,
                len: 8
            })
        );
        let narrow: [ImageView<'_>; 2] = [
            ImageView {
                width: 4,
                height: 2,
                stride: 2,
                data: &pixels,
            },
            image(&pixels, 4, 2),
        ];
        assert_eq!(
            vio.track(0, &narrow),
            Err(VioError::StrideTooSmall {
                index: 0,
                width: 4,
                stride: 2,
            })
        );
    }

    #[test]
    fn an_image_whose_size_overflows_is_rejected() {
        let mut vio: Vio<f32> = pipeline();
        let huge: [ImageView<'_>; 2] = [
            ImageView {
                width: 1,
                height: 2,
                stride: 1 << 63,
                data: &[],
            },
            image(&[], 0, 0),
        ];
        assert_eq!(
            vio.track(0, &huge),
            Err(VioError::ImageSizeOverflow {
                index: 0,
                height: 2,
                stride: 1 << 63,
            })
        );
    }

    proptest! {
        // The pipeline reads two JSON fixtures and builds a frontend per case,
        // so the case count is cut to what the ordering guard needs.
        #![proptest_config(ProptestConfig::with_cases(16))]

        /// Whatever the first timestamp is, a second one that does not strictly
        /// follow it is rejected, and the accepted state does not move.
        #[test]
        fn non_monotonic_imu_is_always_rejected(first in -1_000_000i64..1_000_000, back in 0i64..1_000_000) {
            let mut vio: Vio<f32> = pipeline();
            vio.push_imu(first, [0.0; 3], [0.0; 3]).unwrap();
            prop_assert_eq!(
                vio.push_imu(first - back, [0.0; 3], [0.0; 3]),
                Err(VioError::NonMonotonicImu { previous_t_ns: first, t_ns: first - back })
            );
            // The guard is the only thing that moved, so the next in-order
            // sample is still accepted.
            prop_assert!(vio.push_imu(first + 1, [0.0; 3], [0.0; 3]).is_ok());
        }
    }
}
