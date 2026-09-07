//! Visual-inertial odometry core.
//!
//! The crate is deliberately free of Python, Rerun and GPU code: it consumes
//! grayscale images and IMU samples and returns plain values. Python plumbing
//! (catalog feed, evaluation, logging) lives in the `slam_rs` package and the
//! bindings in `slam-rs-py`; a native runner lives in `slam-rs-cli`.
//!
//! [`Vio`] is the Offline driver (D17): [`Vio::push_imu`] buffers samples and
//! [`Vio::track`] runs the frontend and then the estimator to completion in the
//! calling thread, so every result is final and a repeat run over the same input
//! is bit-identical. Realtime mode — basalt's two threads joined by bounded
//! queues — is stage S9's.

pub mod ba_base;
pub mod calib;
pub mod camera;
pub mod config;
pub(crate) mod eigen_blas;
pub mod estimator;
pub mod frontend;
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

/// How far the estimator has got.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VioStatus {
    /// No pose yet: the estimator has not initialised.
    NotInitialised,
    /// The frame arrived before the IMU samples that cover it. Nothing moved:
    /// push the missing samples and call `track` again with the same frameset
    /// (D17 — no arrival order may reach the trajectory).
    NeedMoreImu,
    /// The returned pose is an estimate.
    Tracking,
}

/// The pre-estimator stub's configuration.
///
/// Not basalt's config — that is [`config::VioConfig`], which is what the real
/// [`Vio`] takes. This is the two numbers [`StubVio`] needs, and it goes when
/// the PyO3 class stops wrapping the stub (stage S9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Number of cameras in the rig; every frameset carries exactly this many images.
    pub camera_count: usize,
    /// IMU samples needed before a frame can be processed.
    pub min_imu_samples: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_count: 2,
            min_imu_samples: 1,
        }
    }
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

/// The frameset validator the PyO3 class still wraps.
///
/// Everything about it is real except the estimator: it enforces the IMU
/// ordering and the frameset geometry, and reports
/// [`VioStatus::NeedMoreImu`]/[`VioStatus::NotInitialised`] accordingly, but it
/// never tracks. The bindings and the Python replay call this while stage S9
/// moves them onto [`Vio`]; the boundary tests that pin the error taxonomy are
/// written against it.
#[derive(Debug, Clone)]
pub struct StubVio {
    config: Config,
    last_imu_t_ns: Option<i64>,
    imu_count: usize,
}

impl StubVio {
    /// Build a validator that has seen nothing yet.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            last_imu_t_ns: None,
            imu_count: 0,
        }
    }

    /// The configuration this estimator runs with.
    pub fn config(&self) -> Config {
        self.config
    }

    /// Timestamp of the last accepted IMU sample, if any.
    pub fn last_imu_t_ns(&self) -> Option<i64> {
        self.last_imu_t_ns
    }

    /// Add one IMU sample: `gyro` in rad/s, `accel` in m/s², both in the rig frame.
    ///
    /// Samples must be strictly increasing in time; a duplicate or out-of-order
    /// timestamp is a [`VioError::NonMonotonicImu`], never a silent reorder.
    pub fn push_imu(&mut self, t_ns: i64, gyro: [f64; 3], accel: [f64; 3]) -> Result<(), VioError> {
        if let Some(previous_t_ns) = self.last_imu_t_ns
            && t_ns <= previous_t_ns
        {
            return Err(VioError::NonMonotonicImu {
                previous_t_ns,
                t_ns,
            });
        }
        let _ = (gyro, accel); // the stub counts samples; it never integrates
        self.last_imu_t_ns = Some(t_ns);
        self.imu_count += 1;
        Ok(())
    }

    /// Process one frameset: one image per camera, oldest to newest in time.
    pub fn track(&mut self, t_ns: i64, images: &[ImageView<'_>]) -> Result<VioResult, VioError> {
        check_frameset(images, self.config.camera_count)?;

        let status: VioStatus = if self.imu_count < self.config.min_imu_samples {
            VioStatus::NeedMoreImu
        } else {
            // The estimator arrives with the frontend and backend PRs; until then
            // the state stays at the identity and no frame ever tracks.
            VioStatus::NotInitialised
        };
        log::debug!(
            "frame {t_ns} ns: {} images, status {status:?}",
            images.len()
        );

        Ok(VioResult {
            status,
            t_ns,
            // The stub never tracks, so the state is the identity every frame.
            world_from_rig: pose_to_array(&Isometry3::identity()),
            velocity: [0.0; 3],
            gyro_bias: [0.0; 3],
            accel_bias: [0.0; 3],
        })
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
    frontend: frontend::flow::FrameToFrameOpticalFlow<frontend::patterns::Pattern51>,
    estimator: estimator::SqrtKeypointVio<S>,
    /// The frontend's own IMU buffer (D24). The same samples reach the
    /// estimator through its own queue.
    frontend_imu: std::collections::VecDeque<imu::ImuSample>,
    /// The frontend's already-popped sample, `processImu`'s `data` (`:169`).
    frontend_pending: Option<imu::ImuSample>,
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
    /// The last IMU timestamp accepted, for the ordering check.
    last_imu_t_ns: Option<i64>,
    /// Cameras in the rig; every frameset must carry exactly this many.
    camera_count: usize,
    /// The last frameset's timestamp, `t_ns` in the frontend (`:172`).
    last_frame_t_ns: Option<i64>,
    /// What the last `track` decided; the S9 Rerun rung reads this.
    last_stats: Option<Box<estimator::FrameStats<S>>>,
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
        let camera_count: usize = calibration.t_i_c.len();
        let frontend: frontend::flow::FrameToFrameOpticalFlow<frontend::patterns::Pattern51> =
            frontend::flow::FrameToFrameOpticalFlow::new(config.clone(), &calibration, options)?;
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
            last_imu_t_ns: None,
            camera_count,
            last_frame_t_ns: None,
            last_stats: None,
        })
    }

    /// The estimator, for callers that want the window or the snapshot.
    pub fn estimator(&self) -> &estimator::SqrtKeypointVio<S> {
        &self.estimator
    }

    /// The frontend, for callers that want the tracked keypoints.
    pub fn frontend(
        &self,
    ) -> &frontend::flow::FrameToFrameOpticalFlow<frontend::patterns::Pattern51> {
        &self.frontend
    }

    /// What the last `track` decided, or `None` before the first one.
    pub fn last_stats(&self) -> Option<&estimator::FrameStats<S>> {
        self.last_stats.as_deref()
    }

    /// Add one IMU sample: `gyro` in rad/s, `accel` in m/s², both in the rig
    /// frame, uncalibrated (the static bias calibration is applied inside).
    ///
    /// The sample reaches both preintegrators (D24).
    ///
    /// # Errors
    ///
    /// [`VioError::NonMonotonicImu`] on a duplicate or out-of-order timestamp,
    /// never a silent reorder.
    pub fn push_imu(&mut self, t_ns: i64, gyro: [f64; 3], accel: [f64; 3]) -> Result<(), VioError> {
        if let Some(previous_t_ns) = self.last_imu_t_ns
            && t_ns <= previous_t_ns
        {
            return Err(VioError::NonMonotonicImu {
                previous_t_ns,
                t_ns,
            });
        }
        let sample: imu::ImuSample = imu::ImuSample {
            t_ns,
            gyro: Vector3::new(gyro[0], gyro[1], gyro[2]),
            accel: Vector3::new(accel[0], accel[1], accel[2]),
        };
        self.frontend_imu.push_back(sample);
        self.estimator.push_imu(sample);
        self.last_imu_t_ns = Some(t_ns);
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
    /// # Errors
    ///
    /// [`VioError`] when the frameset is the wrong width or geometry, or when
    /// the frontend or the estimator refuses it.
    pub fn track(&mut self, t_ns: i64, images: &[ImageView<'_>]) -> Result<VioResult, VioError> {
        check_frameset(images, self.camera_count)?;

        // D17: the coverage test comes before **any** mutation. Everything
        // below moves the pipeline forward irreversibly — the frontend swaps
        // its pyramids, advances `t_ns` and the frame counter, and both
        // preintegrators eat their buffers — so asking the estimator after all
        // that would make `NeedMoreImu` a status the caller cannot act on: the
        // retry would track the frameset against itself. The predicate is the
        // estimator's own, and `process_frame` still runs it as its second
        // line.
        if !self.estimator.imu_covers_frame(t_ns) {
            return Ok(self.result(VioStatus::NeedMoreImu, t_ns));
        }

        // `frame_to_frame_optical_flow.h:138-152`: the prediction the KLT is
        // seeded with. Until the estimator has produced a state both poses are
        // the identity, which is basalt's `first_state_arrived == false` path —
        // and here that flag is `latest_state` being `None`.
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

        // `vit_tracker.cpp:534`: the `u8 << 8` widening the whole frontend
        // assumes, into buffers that are reused frame to frame.
        self.frames
            .resize_with(images.len(), image::ImageU16::default);
        for (frame, view) in self.frames.iter_mut().zip(images.iter()) {
            frame.fill_from_u8_strided(view.data, view.width, view.height, view.stride)?;
        }
        self.frontend
            .process_frame(t_ns, &self.frames, &prediction, &self.masks)?;
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
        // the outcome alone decides the status: `VioStatus::NotInitialised` is
        // reachable from [`StubVio`] only.
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
        if self.frontend_pending.is_none() {
            self.frontend_pending = self.frontend_pop();
        }
        while let Some(sample) = self.frontend_pending {
            if sample.t_ns > prev_t_ns {
                break;
            }
            self.frontend_pending = self.frontend_pop();
        }
        while let Some(sample) = self.frontend_pending {
            if sample.t_ns > curr_t_ns {
                break;
            }
            pim.integrate(
                &sample,
                &self.frontend_noise.accel_cov,
                &self.frontend_noise.gyro_cov,
            )?;
            self.frontend_pending = self.frontend_pop();
        }
        // `:195-198`: "Pretend last IMU sample before now happened now".
        if pim.get_start_t_ns() + pim.get_dt_ns() < curr_t_ns
            && let Some(sample) = self.frontend_pending
        {
            let retimed: imu::ImuSample = imu::ImuSample {
                t_ns: curr_t_ns,
                ..sample
            };
            pim.integrate(
                &retimed,
                &self.frontend_noise.accel_cov,
                &self.frontend_noise.gyro_cov,
            )?;
        }
        Ok(pim)
    }

    /// One sample off the frontend's buffer, calibrated in `f32` and cast back
    /// to `f64` (`frame_to_frame_optical_flow.h:171-178`).
    fn frontend_pop(&mut self) -> Option<imu::ImuSample> {
        let sample: imu::ImuSample = self.frontend_imu.pop_front()?;
        let accel: Vector3<f32> = self
            .calib_f32
            .calib_accel_bias
            .calibrated(&sample.accel.cast());
        let gyro: Vector3<f32> = self
            .calib_f32
            .calib_gyro_bias
            .calibrated(&sample.gyro.cast());
        Some(imu::ImuSample {
            t_ns: sample.t_ns,
            gyro: gyro.cast(),
            accel: accel.cast(),
        })
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
    fn publish_depth_guess(&mut self) -> Result<(), VioError> {
        if self.estimator.ba.calib.t_i_c.is_empty()
            || self.frontend.config().optical_flow_matching_guess_type
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

/// The frameset geometry checks both [`Vio::track`] and [`StubVio::track`] run.
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

    fn pipeline() -> Vio<f32> {
        let directory: std::path::PathBuf =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let config: config::VioConfig = config::VioConfig::from_json_str(
            &std::fs::read_to_string(directory.join("msdmi_config.json")).unwrap(),
        )
        .unwrap();
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
        assert_eq!(vio.frontend().t_ns(), -1);
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
        let mut config: config::VioConfig = config::VioConfig::from_json_str(
            &std::fs::read_to_string(directory.join("msdmi_config.json")).unwrap(),
        )
        .unwrap();
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
    fn config_round_trips_through_json() {
        let config: Config = Config {
            camera_count: 4,
            min_imu_samples: 8,
        };
        let text: String = serde_json::to_string(&config).unwrap();
        assert_eq!(serde_json::from_str::<Config>(&text).unwrap(), config);
    }

    #[test]
    fn a_frame_without_imu_needs_more_imu() {
        let mut vio: StubVio = StubVio::new(Config::default());
        let pixels: Vec<u8> = vec![0; 16];
        let images: [ImageView<'_>; 2] = [image(&pixels, 4, 4), image(&pixels, 4, 4)];
        let result: VioResult = vio.track(1_000, &images).unwrap();
        assert_eq!(result.status, VioStatus::NeedMoreImu);
        assert_eq!(result.t_ns, 1_000);
        assert_abs_diff_eq!(result.world_from_rig[6], 1.0, epsilon = 1e-12);
        assert_eq!(result.velocity, [0.0; 3]);
    }

    #[test]
    fn imu_lifts_the_frame_out_of_need_more_imu() {
        let mut vio: StubVio = StubVio::new(Config {
            camera_count: 1,
            min_imu_samples: 2,
        });
        let pixels: Vec<u8> = vec![0; 16];
        vio.push_imu(0, [0.0; 3], [0.0, 0.0, 9.81]).unwrap();
        assert_eq!(
            vio.track(10, &[image(&pixels, 4, 4)]).unwrap().status,
            VioStatus::NeedMoreImu
        );
        vio.push_imu(1_000, [0.0; 3], [0.0, 0.0, 9.81]).unwrap();
        assert_eq!(
            vio.track(2_000, &[image(&pixels, 4, 4)]).unwrap().status,
            VioStatus::NotInitialised
        );
    }

    #[test]
    fn repeated_imu_timestamps_are_rejected() {
        let mut vio: StubVio = StubVio::new(Config::default());
        vio.push_imu(5, [0.0; 3], [0.0; 3]).unwrap();
        assert_eq!(
            vio.push_imu(5, [0.0; 3], [0.0; 3]),
            Err(VioError::NonMonotonicImu {
                previous_t_ns: 5,
                t_ns: 5
            })
        );
        assert_eq!(vio.last_imu_t_ns(), Some(5));
    }

    #[test]
    fn a_frameset_of_the_wrong_width_is_rejected() {
        let mut vio: StubVio = StubVio::new(Config::default());
        let pixels: Vec<u8> = vec![0; 16];
        assert_eq!(
            vio.track(0, &[image(&pixels, 4, 4)]),
            Err(VioError::CameraCountMismatch {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn a_short_buffer_is_rejected() {
        let mut vio: StubVio = StubVio::new(Config {
            camera_count: 1,
            min_imu_samples: 1,
        });
        let pixels: Vec<u8> = vec![0; 8];
        let short: [ImageView<'_>; 1] = [image(&pixels, 4, 4)];
        assert_eq!(
            vio.track(0, &short),
            Err(VioError::ShortImage {
                index: 0,
                height: 4,
                stride: 4,
                len: 8
            })
        );
        let narrow: [ImageView<'_>; 1] = [ImageView {
            width: 4,
            height: 2,
            stride: 2,
            data: &pixels,
        }];
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
        let mut vio: StubVio = StubVio::new(Config {
            camera_count: 1,
            min_imu_samples: 1,
        });
        let huge: [ImageView<'_>; 1] = [ImageView {
            width: 1,
            height: 2,
            stride: 1 << 63,
            data: &[],
        }];
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
        /// Whatever the first timestamp is, a second one that does not strictly
        /// follow it is rejected, and the accepted state does not move.
        #[test]
        fn non_monotonic_imu_is_always_rejected(first in -1_000_000i64..1_000_000, back in 0i64..1_000_000) {
            let mut vio: StubVio = StubVio::new(Config::default());
            vio.push_imu(first, [0.0; 3], [0.0; 3]).unwrap();
            let result: Result<(), VioError> = vio.push_imu(first - back, [0.0; 3], [0.0; 3]);
            prop_assert_eq!(
                result,
                Err(VioError::NonMonotonicImu { previous_t_ns: first, t_ns: first - back })
            );
            prop_assert_eq!(vio.last_imu_t_ns(), Some(first));
        }
    }
}
