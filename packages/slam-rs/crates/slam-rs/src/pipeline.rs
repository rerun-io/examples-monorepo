//! Vio sequencing, optional frontend lag, and deferred-solver ownership.

use kornia_staging_algebra::Scalar;
mod boundary;
mod lag;
mod lane;
#[cfg(test)]
mod tests;

pub use boundary::{Backend, FrontendTimings, ImageView, VioError, VioPose, VioResult, VioStatus};
pub use lane::FrontendLane;
use lane::build_frontend;

use crate::{calib, config, duration_ns, estimator, frontend, image, imu, types};
use nalgebra::{Isometry3, UnitQuaternion, Vector3};

/// Frameset pipeline with optional one-frame estimator lag (D17, D24, M7).
/// First preintegrate for the frontend prediction, run optical flow, then run
/// the estimator's independent preintegrator, measurement, optimization and
/// marginalization. Feed back the newest state and average scene depth.
/// The two preintegrators remain independent. No timing or queue dropping enters
/// a decision. The frontend uses f32; the estimator scalar is selectable.
/// With `port.frontend_lag`, frontend(t) runs beside estimator(t-1); results
/// carry t-1's timestamp. Drain the final result with [`Self::flush`].
pub struct Vio<S: Scalar = f32> {
    frontend: FrontendLane,
    estimator: estimator::SqrtKeypointVio<S>,
    /// The frontend's own IMU buffer (D24). The same samples reach the
    /// estimator through its own queue.
    frontend_imu: std::collections::VecDeque<imu::ImuSample>,
    /// The frontend's already-popped sample, `processImu`'s `data`.
    frontend_pending: Option<imu::Popped<f64>>,
    /// Most recent estimated state; absent until the estimator publishes one.
    latest_state: Option<types::PoseVelBiasState<f64>>,
    /// The frontend's own accelerometer and gyroscope preintegration noise.
    frontend_noise: imu::ImuNoise<f64>,
    /// The static bias calibration, applied to the frontend's samples in `f32`
    /// and cast back to `f64`.
    calib_f32: calib::Calibration<f32>,
    /// Owned frames: reusable dense widened buffers.
    frames: Vec<kornia_image::Image<u16, 1>>,
    /// Reusable lookahead input, filled by the same lane as the current frame.
    next_frames: Vec<kornia_image::Image<u16, 1>>,
    /// `img->masks`, always empty here: masks come from Monado.
    masks: Vec<kornia_staging_imgproc::features::Masks>,
    /// Cameras in the rig; every frameset must carry exactly this many.
    camera_count: usize,
    /// The last frameset's timestamp, `t_ns` in the frontend.
    last_frame_t_ns: Option<i64>,
    /// What the last `track` decided; the S9 Rerun rung reads this.
    last_stats: Option<Box<estimator::FrameStats<S>>>,
    /// What the last tracked frameset's frontend lane cost; see [`FrontendTimings`].
    frontend_timings: FrontendTimings,
    /// The deferred keyframe (D84) the last `track` finished beside its
    /// frontend, if it finished one.
    last_deferred: Option<Box<estimator::DeferredKeyframeStats<S>>>,
    /// One frameset waiting for the estimator when frontend lag is enabled.
    pending_observations: Option<std::sync::Arc<estimator::FlowObservations>>,
    /// Estimator wall time and exposed join wait on the last overlapping call.
    overlap_timings: lag::OverlapTimings,
}

pub use lag::OverlapTimings;

/// A validated frameset whose pixels are owned by the pipeline.
///
/// The mutable borrow prevents another call from replacing those pixels before
/// computation finishes. No source image borrow is retained.
pub struct PreparedTrack<'a, S: Scalar> {
    vio: &'a mut Vio<S>,
    t_ns: i64,
    prediction: Option<frontend::flow::PosePrediction>,
}

impl<S: Scalar> PreparedTrack<'_, S> {
    /// Finish tracking, or return the unchanged IMU-coverage refusal.
    ///
    /// # Errors
    ///
    /// The frontend or estimator errors returned by [`Vio::track`].
    pub fn finish(self) -> Result<VioResult, VioError> {
        match self.prediction {
            Some(prediction) => self.vio.finish_track(self.t_ns, &prediction),
            None => Ok(self.vio.result(VioStatus::NeedMoreImu, self.t_ns)),
        }
    }
}

impl<S: Scalar> Vio<S> {
    /// Build the pipeline from configuration and calibration (D18).
    ///
    /// # Errors
    /// Reject unsupported flow types or patterns, unusable rigs, linearization other
    /// than `ABS_QR`, disabled square-root marginalization and realtime enforcement.
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
            next_frames: Vec::new(),
            masks: vec![kornia_staging_imgproc::features::Masks::default(); camera_count],
            camera_count,
            last_frame_t_ns: None,
            last_stats: None,
            frontend_timings: FrontendTimings::default(),
            last_deferred: None,
            pending_observations: None,
            overlap_timings: lag::OverlapTimings::default(),
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
    ///
    /// On a keyframe frameset whose solve was deferred (D84),
    /// [`estimator::FrameStats::keyframe_deferred`] is set and the joint solve
    /// is reported by the next frameset's [`Self::last_deferred_keyframe`].
    pub fn last_stats(&self) -> Option<&estimator::FrameStats<S>> {
        self.last_stats.as_deref()
    }

    /// The deferred keyframe solve (D84) the last `track` ran beside its
    /// frontend, or `None` when nothing was pending.
    pub fn last_deferred_keyframe(&self) -> Option<&estimator::DeferredKeyframeStats<S>> {
        self.last_deferred.as_deref()
    }

    /// Timestamp of the frameset still held by the lagged estimator.
    pub fn pending_t_ns(&self) -> Option<i64> {
        self.pending_observations.as_ref().map(|frame| frame.t_ns)
    }

    /// Time waiting for estimator work after the frontend, in nanoseconds.
    /// Includes both D84 and one-frame lag; equal to `overlap_timings().wait_ns`.
    pub fn deferred_wait_ns(&self) -> u64 {
        self.overlap_timings.wait_ns
    }

    /// Finish a pending deferred keyframe (D84) now, on this thread. The next
    /// `track` would do it anyway; call this before reading the window through
    /// [`Self::estimator`] or at the end of a stream.
    /// In lag mode use [`Self::flush`] to also estimate the queued frameset.
    ///
    /// # Errors
    ///
    /// What the joint solve or the marginalization refuse.
    pub fn finish_deferred_keyframe(&mut self) -> Result<(), VioError> {
        if let Some(stats) = self
            .estimator
            .finish_deferred_keyframe(self.frontend.cpu_pool().as_ref())?
        {
            self.last_deferred = Some(Box::new(stats));
        }
        Ok(())
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
    /// With `port.frontend_lag`, the first accepted frame returns
    /// [`VioStatus::Buffered`]. Later calls return the previous frame's pose and
    /// timestamp. Call [`Self::flush`] at stream end to receive the final pose.
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
        self.prepare_track(t_ns, images)?.finish()
    }

    /// Track this frame and prepare the next frame's image-only GPU work while
    /// waiting for this one's result. CPU tracking ignores the valid hint.
    ///
    /// The next call still supplies its images normally. A skipped timestamp or
    /// changed image discards cached work, as do plain `track` and `prepare_track`.
    /// No IMU coverage is required for
    /// the lookahead, and the returned result has the same timestamp and values
    /// as [`Self::track`]. Callers without future images keep using `track`.
    ///
    /// # Errors
    ///
    /// The errors from [`Self::track`], or invalid lookahead geometry/timestamp.
    /// Invalid hints are refused before this frame consumes any IMU.
    pub fn track_with_lookahead(
        &mut self,
        t_ns: i64,
        images: &[ImageView<'_>],
        lookahead: Option<(i64, &[ImageView<'_>])>,
    ) -> Result<VioResult, VioError> {
        if let Some((next_t_ns, next_images)) = lookahead {
            check_frameset(next_images, self.camera_count)?;
            self.frontend.check_frameset(
                next_t_ns,
                next_images.iter().map(|image| (image.width, image.height)),
            )?;
            if next_t_ns <= t_ns {
                return Err(frontend::flow::FrontendError::NonMonotonicFrameset {
                    previous_t_ns: t_ns,
                    t_ns: next_t_ns,
                }
                .into());
            }
        }
        let prepared = self.prepare_track_hinted(t_ns, images)?;
        if prepared.prediction.is_some()
            && prepared.vio.frontend.backend() == Backend::Gpu
            && let Some((next_t_ns, next_images)) = lookahead
        {
            let vio = &mut *prepared.vio;
            vio.next_frames.resize_with(next_images.len(), image::empty);
            for (frame, view) in vio.next_frames.iter_mut().zip(next_images) {
                vio.frontend.fill_frame(frame, view)?;
            }
            vio.frontend
                .queue_lookahead(next_t_ns, &mut vio.next_frames, next_images)?;
        }
        prepared.finish()
    }

    /// Validate and widen borrowed pixels before releasing their owner.
    ///
    /// Checks, IMU prediction and widening run in the same order as [`Self::track`].
    /// The returned value borrows only this pipeline, not `images`.
    ///
    /// # Errors
    ///
    /// The input and IMU prediction errors returned by [`Self::track`].
    pub fn prepare_track(
        &mut self,
        t_ns: i64,
        images: &[ImageView<'_>],
    ) -> Result<PreparedTrack<'_, S>, VioError> {
        let prepared = self.prepare_track_hinted(t_ns, images)?;
        if prepared.prediction.is_some() {
            prepared.vio.frontend.discard_lookahead();
        }
        Ok(prepared)
    }

    fn prepare_track_hinted(
        &mut self,
        t_ns: i64,
        images: &[ImageView<'_>],
    ) -> Result<PreparedTrack<'_, S>, VioError> {
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
            return Ok(PreparedTrack {
                vio: self,
                t_ns,
                prediction: None,
            });
        }

        // Before the first estimated state, both prediction poses are identity.
        let mark: std::time::Instant = std::time::Instant::now();
        let prediction: frontend::flow::PosePrediction = match self.latest_state {
            Some(latest) if self.frontend.config().port_frontend_lag => {
                self.lagged_prediction(t_ns, &latest)?
            }
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
            None => {
                // Waiting for stereo can take arbitrarily many frames. Keep
                // only the interval a lagged first state could still need.
                let keep_after = self.last_frame_t_ns.unwrap_or(t_ns);
                self.drop_frontend_imu_through(keep_after);
                frontend::flow::PosePrediction::default()
            }
        };
        self.frontend_timings.imu_ns = duration_ns(mark);

        // Widen and densify directly into reusable image buffers.
        self.frames.resize_with(images.len(), image::empty);
        for (frame, view) in self.frames.iter_mut().zip(images.iter()) {
            self.frontend.fill_frame(frame, view)?;
        }
        self.frontend.prepare_packed_inputs(images);
        Ok(PreparedTrack {
            vio: self,
            t_ns,
            prediction: Some(prediction),
        })
    }

    fn finish_track(
        &mut self,
        t_ns: i64,
        prediction: &frontend::flow::PosePrediction,
    ) -> Result<VioResult, VioError> {
        self.last_deferred = None;
        let lagged_outcome = self.process_frame_beside_estimator(t_ns, prediction)?;
        self.frontend_timings.flow = self.frontend.timings();
        self.last_frame_t_ns = Some(t_ns);

        // The estimator reads only the ids and the observed pixels
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
        let observations = std::sync::Arc::new(observations);
        let (result_t_ns, outcome) = if self.frontend.config().port_frontend_lag {
            self.pending_observations = Some(observations);
            let Some(completed) = lagged_outcome else {
                return Ok(self.result(VioStatus::Buffered, t_ns));
            };
            completed
        } else {
            (
                t_ns,
                self.estimator
                    .process_frame(observations, self.frontend.cpu_pool().as_ref())?,
            )
        };

        self.accept_outcome(result_t_ns, outcome)
    }

    fn accept_outcome(
        &mut self,
        t_ns: i64,
        outcome: estimator::FrameOutcome<S>,
    ) -> Result<VioResult, VioError> {
        // The estimator initialises inside the same `process_frame` that
        // measures, so a `Measured` outcome always has a state and
        // the outcome alone decides the status.
        let status: VioStatus = match outcome {
            estimator::FrameOutcome::NeedMoreImu => VioStatus::NeedMoreImu,
            estimator::FrameOutcome::NoVisualFeatures => {
                self.last_stats = None;
                VioStatus::NoVisualFeatures
            }
            estimator::FrameOutcome::Measured(stats) => {
                self.last_stats = Some(stats);
                // Return the newest state, then the depth feedback.
                self.publish_state();
                self.publish_depth_guess()?;
                VioStatus::Tracking
            }
        };

        Ok(self.result(status, t_ns))
    }

    /// Only a tracked frameset publishes the estimator state.
    fn result(&self, status: VioStatus, t_ns: i64) -> VioResult {
        let pose = self
            .estimator
            .state()
            .filter(|_| status == VioStatus::Tracking)
            .map(|state| VioPose {
                world_from_rig: pose_to_array(&Isometry3::from_parts(
                    state
                        .t_w_i
                        .translation
                        .map(kornia_staging_algebra::Scalar::to_f64)
                        .into(),
                    *state.t_w_i.rotation.cast::<f64>().quaternion(),
                )),
                velocity: state
                    .vel_w_i
                    .map(kornia_staging_algebra::Scalar::to_f64)
                    .into(),
                gyro_bias: state
                    .bias_gyro
                    .map(kornia_staging_algebra::Scalar::to_f64)
                    .into(),
                accel_bias: state
                    .bias_accel
                    .map(kornia_staging_algebra::Scalar::to_f64)
                    .into(),
            });
        VioResult { status, t_ns, pose }
    }

    /// `processImu(curr_t_ns)`.
    ///
    /// The same three-part loop the estimator runs, over the frontend's own
    /// buffer and at `f64`: skip up to the previous frame, integrate up to this
    /// one, then close the interval by retiming the next sample. The bias
    /// calibration happens in `f32` and is cast back, which is what
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
        // the same three-part loop the estimator's own
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
    /// to `f64`.
    fn drop_frontend_imu_through(&mut self, t_ns: i64) {
        while self
            .frontend_imu
            .front()
            .is_some_and(|sample| sample.t_ns <= t_ns)
        {
            self.frontend_imu.pop_front();
        }
    }

    fn frontend_pop(&mut self) -> Option<imu::Popped<f64>> {
        let sample: imu::ImuSample = self.frontend_imu.pop_front()?;
        Some(self.calibrated(&sample))
    }

    fn calibrated(&self, sample: &imu::ImuSample) -> imu::Popped<f64> {
        let accel: Vector3<f32> = self
            .calib_f32
            .calib_accel_bias
            .calibrated(&sample.accel.cast());
        let gyro: Vector3<f32> = self
            .calib_f32
            .calib_gyro_bias
            .calibrated(&sample.gyro.cast());
        (sample.t_ns, gyro.cast(), accel.cast())
    }

    /// `opt_flow_state_queue->push(data)`.
    fn publish_state(&mut self) {
        if let Some(state) = self.estimator.state() {
            self.latest_state = Some(types::PoseVelBiasState {
                t_ns: state.t_ns,
                t_w_i: state.t_w_i.cast(),
                vel_w_i: state.vel_w_i.map(kornia_staging_algebra::Scalar::to_f64),
                bias_gyro: state.bias_gyro.map(kornia_staging_algebra::Scalar::to_f64),
                bias_accel: state.bias_accel.map(kornia_staging_algebra::Scalar::to_f64),
            });
        }
    }

    /// Average scene depth is `num_features / Σ inverse-depth`, or the configured
    /// default when the sum is not positive. Compute only for `REPROJ_AVG_DEPTH`.
    /// Both estimator lanes reduce in f64, then round once into the frontend's f32
    /// depth guess, which seeds the KLT matching window.
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

impl<S: Scalar> std::fmt::Debug for Vio<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Exhaustive destructuring keeps retry fingerprints complete when fields are added.
        let Self {
            frontend,
            estimator,
            frontend_imu,
            frontend_pending,
            latest_state,
            frontend_noise,
            calib_f32,
            frames,
            next_frames,
            masks,
            camera_count,
            last_frame_t_ns,
            last_stats,
            frontend_timings,
            last_deferred,
            pending_observations,
            overlap_timings,
        } = self;
        f.debug_struct("Vio")
            .field("frontend", frontend)
            .field("estimator", estimator)
            .field("frontend_imu", frontend_imu)
            .field("frontend_pending", frontend_pending)
            .field("latest_state", latest_state)
            .field("frontend_noise", frontend_noise)
            .field("calib_f32", calib_f32)
            .field(
                "frames",
                &frames
                    .iter()
                    .map(|image| (image.size(), image.as_slice()))
                    .collect::<Vec<_>>(),
            )
            .field(
                "next_frames",
                &next_frames
                    .iter()
                    .map(|image| (image.size(), image.as_slice()))
                    .collect::<Vec<_>>(),
            )
            .field("masks", masks)
            .field("camera_count", camera_count)
            .field("last_frame_t_ns", last_frame_t_ns)
            .field("last_stats", last_stats)
            .field("frontend_timings", frontend_timings)
            .field("last_deferred", last_deferred)
            .field("pending_observations", pending_observations)
            .field("overlap_timings", overlap_timings)
            .finish()
    }
}
