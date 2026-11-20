//! Vio sequencing, optional frontend lag, and deferred-solver ownership.

use kornia_staging_algebra::Scalar;
pub mod bootstrap;
mod boundary;
mod lag;
mod lane;
mod rolling_shutter;
mod select;
#[cfg(test)]
mod tests;

pub use boundary::{
    Backend, FrontendTimings, ImageView, VioError, VioPose, VioResult, VioStatus, pose_from_xyzw,
    pose_to_xyzw,
};
pub use lane::FrontendLane;
use lane::build_frontend;
pub use rolling_shutter::RollingShutter;
pub use select::{
    BackendRequest, Backends, GbaRequest, device_probes_run, follow_frontend, gba_probes_run,
    gpu_probe, probe_back_end, select_backend,
};

use crate::{calib, config, duration_ns, estimator, frontend, imu, types};
use nalgebra::{Isometry3, UnitQuaternion, Vector3};

/// Widest camera image the frontend tracks at full resolution. A rig whose every camera is wider (the Insta360
/// X5 and the X4 Air's high-resolution mode, 3840 px lenses) is tracked on its images reduced by two, on the device
/// on the GPU lane: the calibration is halved once here ([`calib::Calibration::half_resolution`]), so the tracker,
/// the estimator and every pose see a 1920 px rig, and the tracker's cost stays that of the 1920 px mode.
/// Keypoints and `Vio::calibration` are in the reduced pixels.
pub const MAX_FULL_RESOLUTION_WIDTH: u32 = 2048;

/// Frameset pipeline with optional one-frame estimator lag (D17, D24, M7).
/// First preintegrate for the frontend prediction, run optical flow, then run
/// the estimator's independent preintegrator, measurement, optimization and
/// marginalization. Feed back the newest state and average scene depth.
/// The two preintegrators remain independent. No timing or queue dropping enters
/// a decision. The frontend uses f32; the estimator scalar is selectable.
/// With `port.frontend_lag`, frontend(t) runs beside estimator(t-1); results
/// carry t-1's timestamp. Drain the final result with [`Self::flush`].
#[derive(Debug)]
pub struct Vio<S: Scalar = f32> {
    frontend: FrontendLane,
    estimator: estimator::SqrtKeypointVio<S>,
    /// The frontend's own IMU buffer (D24). The same samples reach the
    /// estimator through its own queue.
    frontend_imu: std::collections::VecDeque<kornia_staging_sensors::imu::CombinedImuSample>,
    /// The frontend's already-popped sample, `processImu`'s `data`.
    frontend_pending: Option<imu::Popped<f64>>,
    /// Most recent estimated state; absent until the estimator publishes one.
    latest_state: Option<types::PoseVelBiasState<f64>>,
    /// The frontend's own accelerometer and gyroscope preintegration noise.
    frontend_noise: kornia_staging_sensors::imu::ImuNoise<f64>,
    /// The static bias calibration, applied to the frontend's samples in `f32`
    /// and cast back to `f64`.
    calib_f32: calib::Calibration<f32>,
    /// Owned dense CPU inputs or retained packed GPU inputs.
    frames: frontend::input::FrameStorage,
    /// Reusable lookahead input, filled by the same lane as the current frame.
    next_frames: frontend::input::PackedImages,
    /// Every camera is tracked on its images reduced by two; see [`MAX_FULL_RESOLUTION_WIDTH`].
    half: bool,
    /// `img->masks`: whole detection cells of each camera where the frontend neither detects nor keeps a track;
    /// empty unless [`Self::set_frame_masks`] filled them.
    masks: Vec<kornia_staging_imgproc::features::CellMasks>,
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
    /// The rolling-shutter correction of the estimator's keypoints, when a camera states a readout.
    rolling_shutter: Option<RollingShutter>,
    /// The inertial bootstrap, when `config.vio_bootstrap` selects it.
    bootstrap: bootstrap::BootstrapPhase,
    /// The back end over the persistent keyframe map, inline (Offline) or on
    /// its own thread (Realtime); `None` when `backend.enabled` is off.
    #[cfg(feature = "backend")]
    map_runner: Option<crate::backend::BackendRunner<crate::backend::MapBackend>>,
    /// Images of keyframes whose stamp has not gone to the back end yet, for
    /// loop closing (DESIGN §2.2: the stamp carries its own keyframe's images).
    #[cfg(feature = "backend")]
    keyframe_images: std::collections::BTreeMap<i64, crate::estimator::KeyframeImages>,
    /// With `port.frontend_lag` and loop closing on: the previous frameset's
    /// input and its timestamp, which the estimator finishes beside the
    /// frontend of the next. Swapped with `frames`, never copied.
    #[cfg(feature = "backend")]
    lagged_frames: Option<(i64, frontend::input::FrameStorage)>,
    /// The image slots of the keyframe stamps sent to the back end, bounded
    /// by the Realtime query cap.
    #[cfg(feature = "backend")]
    image_slots: crate::estimator::ImageSlots,
    /// Where the global BA's PCG runs.
    gba_lane: Backend,
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

    /// The same pipeline on a named frontend backend (decision D21). With
    /// `backend.enabled`, the global BA's PCG follows the frontend
    /// ([`follow_frontend`], DESIGN Q2/Q7).
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
        let gba: Backend = if config.backend_enabled {
            follow_frontend(backend)
        } else {
            Backend::Cpu
        };
        Self::with_backends(
            config,
            calibration,
            options,
            Backends {
                frontend: backend,
                gba,
            },
        )
    }

    /// [`Self::with_backend`] with the global BA's PCG lane named as well
    /// ([`select_backend`] resolves both).
    ///
    /// # Errors
    /// As [`Self::with_backend`].
    #[cfg_attr(not(feature = "backend"), allow(unused_variables))]
    pub fn with_backends(
        config: config::VioConfig,
        calibration: calib::Calibration<f64>,
        options: frontend::flow::FrontendOptions,
        backends: Backends,
    ) -> Result<Self, VioError> {
        let backend: Backend = backends.frontend;
        let camera_count: usize = calibration.t_i_c.len();
        let half: bool = !calibration.resolution.is_empty()
            && calibration
                .resolution
                .iter()
                .all(|[width, _]| *width > MAX_FULL_RESOLUTION_WIDTH);
        let calibration = if half {
            calibration.half_resolution()
        } else {
            calibration
        };
        let bootstrap = bootstrap::BootstrapPhase::for_config(&config, &calibration)?;
        let frontend: FrontendLane = build_frontend(&config, &calibration, options, backend)?;
        let rolling_shutter: Option<RollingShutter> = RollingShutter::for_calibration(&calibration)
            .map_err(frontend::flow::FrontendError::from)?;
        let calib_f32: calib::Calibration<f32> = calibration.cast();
        let frontend_noise: kornia_staging_sensors::imu::ImuNoise<f64> =
            crate::imu::noise_from_calibration(&calibration)?;
        #[cfg(feature = "backend")]
        let query_cap: usize = config.backend_realtime_max_queued_queries.get();
        #[cfg(feature = "backend")]
        let map_runner = config
            .backend_enabled
            .then(|| {
                let rig = crate::backend::BackendRig::new(&calibration)
                    .map_err(|error| VioError::BackendRig(error.to_string()))?;
                crate::backend::BackendRunner::new(
                    crate::backend::MapBackend::new(&config, rig, backends.gba),
                    config.backend_mode,
                    query_cap,
                )
                .map_err(|error| VioError::BackendThread(error.to_string()))
            })
            .transpose()?;
        #[cfg_attr(not(feature = "backend"), allow(unused_mut))]
        let mut estimator: estimator::SqrtKeypointVio<S> =
            estimator::SqrtKeypointVio::new(calibration.cast(), config)?;
        #[cfg(feature = "backend")]
        estimator.emit_map_stamps(map_runner.is_some());
        Ok(Self {
            frontend,
            estimator,
            frontend_imu: std::collections::VecDeque::new(),
            frontend_pending: None,
            latest_state: None,
            frontend_noise,
            calib_f32,
            frames: frontend::input::FrameStorage::new(backend == Backend::Gpu),
            next_frames: frontend::input::PackedImages::default(),
            half,
            masks: vec![kornia_staging_imgproc::features::CellMasks::default(); camera_count],
            camera_count,
            last_frame_t_ns: None,
            last_stats: None,
            frontend_timings: FrontendTimings::default(),
            last_deferred: None,
            pending_observations: None,
            overlap_timings: lag::OverlapTimings::default(),
            rolling_shutter,
            bootstrap,
            #[cfg(feature = "backend")]
            map_runner,
            #[cfg(feature = "backend")]
            keyframe_images: std::collections::BTreeMap::new(),
            #[cfg(feature = "backend")]
            lagged_frames: None,
            #[cfg(feature = "backend")]
            image_slots: crate::estimator::ImageSlots::new(query_cap),
            gba_lane: backends.gba,
        })
    }

    /// The rig's calibration, as the frontend and the estimator read it.
    pub fn calibration(&self) -> &calib::Calibration<f32> {
        &self.calib_f32
    }

    /// The rolling-shutter correction, when the rig has one: its frame counts say how many frames were corrected
    /// and how many stayed raw because the gyro did not cover their time.
    pub fn rolling_shutter(&self) -> Option<&RollingShutter> {
        self.rolling_shutter.as_ref()
    }

    /// Gravity in the world frame, `(0, 0, -calibration.gravity_ms2)`.
    fn gravity(&self) -> Vector3<f64> {
        crate::imu::gravity(self.calib_f32.gravity_ms2)
    }

    /// The map's keyframes at the end of a sequence, after a global BA on the
    /// map when `run_gba` (DESIGN §3.1, §8). Every stamp the window has handed
    /// over is taken first. The window is untouched, so the raw VIO poses stay
    /// what [`Self::track`] returned; the map keeps the GBA's poses. Call it
    /// after [`Self::flush`]: a later keyframe stamp writes the window's poses
    /// over the GBA's for the keyframes still in the window. `None` when
    /// `backend.enabled` is off. A GBA that fails costs only the GBA:
    /// [`crate::backend::MapResult::gba_failure`] says why, and the keyframes
    /// are the map's after loops and PGO.
    ///
    /// # Panics
    ///
    /// Re-raises a panic of the back-end thread.
    #[cfg(feature = "backend")]
    pub fn finish_map(&mut self, run_gba: bool) -> Option<crate::backend::MapResult> {
        self.step_map();
        let config: &config::VioConfig = self.frontend.config();
        let runner = self.map_runner.as_mut()?;
        Some(runner.update(|backend| backend.finish(run_gba, config)))
    }

    /// Read the persistent keyframe map itself; `None` when `backend.enabled`
    /// is off. In Realtime this waits for the step in progress: tests and
    /// diagnostics only. Tracking-side readers take [`Self::map_view`].
    #[cfg(feature = "backend")]
    pub fn read_map<R>(&self, read: impl FnOnce(&crate::backend::MapBackend) -> R) -> Option<R> {
        self.map_runner
            .as_ref()
            .map(|backend| backend.read(|map_backend, _| read(map_backend)))
    }

    /// A copy of the map and the back end's counters at the moment it was
    /// taken; `None` when `backend.enabled` is off. Never waits for the back
    /// end ([`crate::backend::BackendRunner::view`]).
    ///
    /// Current up to the last solved keyframe the back end has taken: a
    /// keyframe whose solve is still deferred (D84) joins once
    /// [`Self::finish_deferred_keyframe`], the next `track` or [`Self::flush`]
    /// solves it. In Realtime it is the newest copy the back-end thread
    /// posted, which can trail the stamps it still holds;
    /// [`Self::flush_backend`] makes it current.
    #[cfg(feature = "backend")]
    pub fn map_view(&self) -> Option<crate::backend::BackendView<crate::backend::MapSnapshot>> {
        self.map_runner.as_ref().map(|runner| runner.view())
    }

    /// The map of [`Self::map_view`].
    #[cfg(feature = "backend")]
    pub fn map_snapshot(&self) -> Option<std::sync::Arc<crate::backend::MapSnapshot>> {
        self.map_view().map(|view| view.view)
    }

    /// What the back end has done so far: stamps, queries run and dropped,
    /// timings. `None` when `backend.enabled` is off. Never waits.
    #[cfg(feature = "backend")]
    pub fn backend_counters(&self) -> Option<crate::backend::BackendCounters> {
        self.map_runner.as_ref().map(|runner| runner.counters())
    }

    /// Where the back end runs; `None` when `backend.enabled` is off.
    #[cfg(feature = "backend")]
    pub fn backend_mode(&self) -> Option<config::BackendMode> {
        self.map_runner.as_ref().map(|runner| runner.mode())
    }

    /// Wait until the back end has taken every stamp the window has handed it.
    /// Immediate in Offline or with the back end off. [`Self::flush`] calls it.
    ///
    /// # Panics
    ///
    /// Re-raises a panic of the back-end thread.
    pub fn flush_backend(&mut self) {
        self.step_map();
        #[cfg(feature = "backend")]
        if let Some(runner) = self.map_runner.as_mut() {
            runner.flush();
        }
    }

    /// Hand the window's new stamps to the back end, oldest first. The window
    /// never reads the map back, so this cannot move a VIO pose. Realtime only
    /// sends them: tracking never waits for the back end.
    fn step_map(&mut self) {
        #[cfg(feature = "backend")]
        if let Some(runner) = self.map_runner.as_mut() {
            for mut stamp in self.estimator.take_map_stamps() {
                if let crate::estimator::MapStamp::Keyframe(keyframe) = &mut stamp {
                    keyframe.images = self
                        .keyframe_images
                        .remove(&keyframe.t_ns)
                        .map(|images| self.image_slots.hold(images));
                    self.keyframe_images.retain(|&t_ns, _| t_ns > keyframe.t_ns);
                }
                runner.submit(stamp);
            }
        }
    }

    /// Whether keyframe images go to the back end: loop closing is on.
    #[cfg(feature = "backend")]
    fn loop_closing_on(&self) -> bool {
        self.map_runner.is_some() && self.frontend.config().backend_loop_closing.enabled
    }

    /// Keep a copy of the frameset's images when the estimator made it a
    /// keyframe and loop closing will describe it; its stamp (now, or after a
    /// deferred solve) takes them. A lagged pipeline finishes the previous
    /// frameset, whose input `lagged_frames` holds. A camera image that fails
    /// to fill holds none, so no image shifts to another camera.
    #[cfg(feature = "backend")]
    fn hold_keyframe_images(&mut self, t_ns: i64, outcome: &estimator::FrameOutcome<S>) {
        let keyframe =
            matches!(outcome, estimator::FrameOutcome::Measured(stats) if stats.took_keyframe);
        if !keyframe || !self.loop_closing_on() {
            return;
        }
        let frames = if self.last_frame_t_ns == Some(t_ns) {
            &self.frames
        } else {
            match &self.lagged_frames {
                Some((lagged_t_ns, frames)) if *lagged_t_ns == t_ns => frames,
                _ => return,
            }
        };
        let Ok(images) = frames
            .view()
            .iter()
            .map(|frame| frame.with_dense(Clone::clone))
            .collect::<Result<Vec<kornia_image::Image<u16, 1>>, _>>()
        else {
            // Counted later as the query's `without_images`.
            log::warn!("loop closing: keyframe {t_ns}'s images could not be copied");
            return;
        };
        self.keyframe_images.insert(
            t_ns,
            crate::estimator::KeyframeImages(std::sync::Arc::new(images)),
        );
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

    /// Where the global BA's PCG runs.
    pub fn gba_lane(&self) -> Backend {
        self.gba_lane
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
        self.step_map();
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
        self.bootstrap.newest_imu_t_ns(&self.estimator)
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
        let sample: kornia_staging_sensors::imu::CombinedImuSample =
            kornia_staging_sensors::imu::CombinedImuSample {
                timestamp_ns: t_ns,
                gyro: gyro.into(),
                accel: accel.into(),
            };
        self.frontend_imu.push_back(sample);
        self.bootstrap
            .push_imu(&mut self.estimator, sample, &self.calib_f32);
        if let Some(rolling_shutter) = &mut self.rolling_shutter {
            rolling_shutter.push_gyro(t_ns, calibrated(&self.calib_f32, &sample).1);
        }
        Ok(())
    }

    /// Masks for the framesets that follow, until the next call: one optional mask per camera, at any resolution
    /// (it is read through its scale to the camera's tracked image, see [`frontend::masks`]). `None` masks nothing.
    ///
    /// # Errors
    ///
    /// [`VioError::CameraCountMismatch`] when there is not one entry per camera, [`VioError::FrameMask`] when a mask
    /// is empty or its buffer is not its size. Nothing is changed then.
    pub fn set_frame_masks(
        &mut self,
        masks: &[Option<frontend::masks::PixelMask<'_>>],
    ) -> Result<(), VioError> {
        if masks.len() != self.camera_count {
            return Err(VioError::CameraCountMismatch {
                expected: self.camera_count,
                actual: masks.len(),
            });
        }
        for (camera, mask) in masks.iter().enumerate() {
            if let Some(mask) = mask
                && (mask.width == 0
                    || mask.height == 0
                    || mask.width.checked_mul(mask.height) != Some(mask.data.len()))
            {
                return Err(VioError::FrameMask {
                    camera,
                    width: mask.width,
                    height: mask.height,
                    len: mask.data.len(),
                });
            }
        }
        for (camera, mask) in masks.iter().enumerate() {
            self.masks[camera] = match mask {
                Some(mask) => {
                    let [width, height] = self.calib_f32.resolution[camera];
                    let image = kornia_image::ImageSize {
                        width: width as usize,
                        height: height as usize,
                    };
                    frontend::masks::masked_cells(
                        *mask,
                        image,
                        &self.frontend.detection_grid(camera),
                    )
                }
                None => kornia_staging_imgproc::features::CellMasks::default(),
            };
        }
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
            let shift = usize::from(self.half);
            self.frontend.check_frameset(
                next_t_ns,
                next_images.iter().map(|image| (image.width >> shift, image.height >> shift)),
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
            vio.next_frames.fill(next_images, vio.half);
            vio.frontend
                .queue_lookahead(next_t_ns, &mut vio.next_frames)?;
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
        // A half-resolution rig takes full-size frames and tracks them reduced by two.
        let shift = usize::from(self.half);
        self.frontend.check_frameset(
            t_ns,
            images.iter().map(|image| (image.width >> shift, image.height >> shift)),
        )?;
        if !self.bootstrap.imu_covers_frame(&self.estimator, t_ns) {
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
                let pim: kornia_staging_sensors::imu::IntegratedImuMeasurement<f64> =
                    self.frontend_preintegrate(t_ns, &latest)?;
                let predicted: kornia_staging_sensors::imu::NavState<f64> =
                    pim.predict_state(&latest.pose_vel_state(), &self.gravity());
                frontend::flow::PosePrediction {
                    t_w_i_previous: latest.t_w_i.cast(),
                    t_w_i_current: predicted.t_w_i.cast(),
                }
            }
            None => match self.last_frame_t_ns {
                Some(previous_t_ns) if self.bootstrap.predicts_rotation() => {
                    self.gyro_prediction(previous_t_ns, t_ns)?
                }
                _ => {
                    // Waiting for stereo can take arbitrarily many frames. Keep
                    // only the interval a lagged first state could still need.
                    let keep_after = self.last_frame_t_ns.unwrap_or(t_ns);
                    self.drop_frontend_imu_through(keep_after);
                    frontend::flow::PosePrediction::default()
                }
            },
        };
        self.frontend_timings.imu_ns = duration_ns(mark);

        // A lagged estimator finishes the frameset `frames` holds now: keep it.
        #[cfg(feature = "backend")]
        if self.frontend.config().port_frontend_lag
            && self.loop_closing_on()
            && let Some(previous_t_ns) = self.last_frame_t_ns
        {
            let mut spare = match self.lagged_frames.take() {
                Some((_, spare)) => spare,
                None => frontend::input::FrameStorage::new(matches!(
                    self.frames,
                    frontend::input::FrameStorage::Packed(_)
                )),
            };
            std::mem::swap(&mut self.frames, &mut spare);
            self.lagged_frames = Some((previous_t_ns, spare));
        }
        self.frames.fill(images, self.half)?;
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
                let warp: kornia_staging_imgproc::optical_flow::patch_se2::AffineCompact2f =
                    keypoints.transform(index);
                slot.insert(*id, warp.translation.into());
            }
        }
        if let Some(rolling_shutter) = &mut self.rolling_shutter {
            rolling_shutter.correct(
                &mut observations,
                self.latest_state.as_ref().map(|state| state.bias_gyro),
            );
        }
        let observations = std::sync::Arc::new(observations);
        let (result_t_ns, outcome) = if self.frontend.config().port_frontend_lag {
            self.pending_observations = Some(observations);
            let Some(completed) = lagged_outcome else {
                return Ok(self.result(VioStatus::Buffered, t_ns));
            };
            completed
        } else if self.bootstrap.is_running() {
            return self.bootstrap_track(observations);
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
        #[cfg(feature = "backend")]
        self.hold_keyframe_images(t_ns, &outcome);
        self.step_map();
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
    ) -> Result<kornia_staging_sensors::imu::IntegratedImuMeasurement<f64>, VioError> {
        let prev_t_ns: i64 = self.last_frame_t_ns.unwrap_or(-1);
        let mut pim: kornia_staging_sensors::imu::IntegratedImuMeasurement<f64> =
            kornia_staging_sensors::imu::IntegratedImuMeasurement::new(
                prev_t_ns,
                &latest.bias_gyro,
                &latest.bias_accel,
            );
        // the same three-part loop the estimator's own
        // preintegration runs, through `IntegratedImuMeasurement::accumulate_to`.
        let noise: kornia_staging_sensors::imu::ImuNoise<f64> = self.frontend_noise;
        let pending: Option<imu::Popped<f64>> = self.frontend_pending.take();
        self.frontend_pending = crate::imu::accumulate_to(
            &mut pim,
            pending,
            || self.frontend_pop(),
            prev_t_ns,
            curr_t_ns,
            &noise,
        )?;
        Ok(pim)
    }

    /// The gyro-only rotation from `previous_t_ns` to `t_ns`, before any
    /// state exists (the inertial bootstrap): the frontend preintegration from
    /// an identity state at rest with zero biases, of which only the rotation
    /// is kept.
    ///
    /// # Errors
    ///
    /// What [`Self::frontend_preintegrate`] refuses.
    fn gyro_prediction(
        &mut self,
        previous_t_ns: i64,
        t_ns: i64,
    ) -> Result<frontend::flow::PosePrediction, VioError> {
        let rest: types::PoseVelBiasState<f64> = types::PoseVelBiasState {
            t_ns: previous_t_ns,
            ..types::PoseVelBiasState::default()
        };
        let pim: kornia_staging_sensors::imu::IntegratedImuMeasurement<f64> =
            self.frontend_preintegrate(t_ns, &rest)?;
        let predicted: kornia_staging_sensors::imu::NavState<f64> =
            pim.predict_state(&rest.pose_vel_state(), &self.gravity());
        Ok(frontend::flow::PosePrediction {
            t_w_i_previous: crate::lie::Se3::identity(),
            t_w_i_current: crate::lie::Se3::new(predicted.t_w_i.rotation, Vector3::zeros()).cast(),
        })
    }

    fn drop_frontend_imu_through(&mut self, t_ns: i64) {
        while self
            .frontend_imu
            .front()
            .is_some_and(|sample| sample.timestamp_ns <= t_ns)
        {
            self.frontend_imu.pop_front();
        }
    }

    /// One sample off the frontend's buffer, calibrated.
    fn frontend_pop(&mut self) -> Option<imu::Popped<f64>> {
        let sample: kornia_staging_sensors::imu::CombinedImuSample =
            self.frontend_imu.pop_front()?;
        Some(calibrated(&self.calib_f32, &sample))
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

/// One raw sample calibrated in `f32` (what `Calibration<Scalar>` with
/// `Scalar = float` means) and cast back to `f64`: the frontend's
/// preintegration and the inertial bootstrap both read it.
fn calibrated(
    calibration: &calib::Calibration<f32>,
    sample: &kornia_staging_sensors::imu::CombinedImuSample,
) -> imu::Popped<f64> {
    let accel: Vector3<f32> = calibration
        .calib_accel_bias
        .calibrated(&Vector3::from(sample.accel.to_array()).cast());
    let gyro: Vector3<f32> = calibration
        .calib_gyro_bias
        .calibrated(&Vector3::from(sample.gyro.to_array()).cast());
    (sample.timestamp_ns, gyro.cast::<f64>(), accel.cast::<f64>())
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
