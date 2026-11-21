//! Python bindings for the slam-rs core, exposed as `slam_rs._core`.
//!
//! Every array crossing the boundary is validated here and copied into owned
//! Rust storage, so the GIL is released for the whole core call and a producer
//! thread can keep decoding. A bad dtype, rank, shape or memory layout raises
//! `ValueError`, never a panic.

mod bootstrap;
mod catalog;
mod gba;
mod map;

use kornia_image::Image;
use kornia_staging_imgproc::features::CellGrid;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2,
    PyUntypedArrayMethods, ToPyArray,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use slam_rs::calib::{
    Calibration as CoreCalibration, CameraParts as CoreCameraParts, ImuParts as CoreImuParts,
};
use slam_rs::config::VioConfig as CoreVioConfig;
use slam_rs::frontend::flow::{
    FlowFrame as CoreFlowFrame, FrameToFrameOpticalFlow, FrontendError, FrontendOptions,
    PosePrediction,
};
use slam_rs::{Backend, BackendRequest, Backends, FrontendLane, GbaRequest, ImageView, VioError};

/// Map any core error onto `ValueError`, which is what every refusal here is.
fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// How far the estimator has got.
#[pyclass(module = "slam_rs._core", eq, eq_int, skip_from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VioStatus {
    /// The frame arrived before the IMU samples that cover it.
    NeedMoreImu,
    /// The returned pose is an estimate.
    Tracking,
    /// The lagged frontend accepted a frame; no estimate is ready yet.
    Buffered,
    /// The first keyframe lacked stereo structure; no pose exists yet.
    NoVisualFeatures,
}

impl From<slam_rs::VioStatus> for VioStatus {
    fn from(status: slam_rs::VioStatus) -> Self {
        match status {
            slam_rs::VioStatus::NeedMoreImu => Self::NeedMoreImu,
            slam_rs::VioStatus::Tracking => Self::Tracking,
            slam_rs::VioStatus::Buffered => Self::Buffered,
            slam_rs::VioStatus::NoVisualFeatures => Self::NoVisualFeatures,
        }
    }
}

/// What one `track` call produced.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct VioResult {
    inner: slam_rs::VioResult,
}

#[pymethods]
impl VioResult {
    /// Estimator state for this frame.
    #[getter]
    fn status(&self) -> VioStatus {
        self.inner.status.into()
    }

    /// Timestamp of the frameset, in nanoseconds.
    #[getter]
    fn t_ns(&self) -> i64 {
        self.inner.t_ns
    }

    /// Rig pose in the world frame as `[tx, ty, tz, qx, qy, qz, qw]`: the raw,
    /// causal VIO, which is also this frame's corrected pose (the pose-graph
    /// gauge fixes the window). The corrected history is
    /// `slam_rs.tracking.reanchored(poses, vio.map_snapshot())`.
    #[getter]
    fn world_from_rig<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .pose
            .as_ref()
            .map(|pose| pose.world_from_rig.as_slice().to_pyarray(py))
    }

    /// Rig velocity in the world frame, m/s.
    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .pose
            .as_ref()
            .map(|pose| pose.velocity.as_slice().to_pyarray(py))
    }

    /// Gyroscope bias estimate, rad/s.
    #[getter]
    fn gyro_bias<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .pose
            .as_ref()
            .map(|pose| pose.gyro_bias.as_slice().to_pyarray(py))
    }

    /// Accelerometer bias estimate, m/s².
    #[getter]
    fn accel_bias<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .pose
            .as_ref()
            .map(|pose| pose.accel_bias.as_slice().to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "VioResult(status=VioStatus.{:?}, t_ns={})",
            self.inner.status, self.inner.t_ns
        )
    }
}

/// The estimator, driven one frameset at a time (D16, D17).
///
/// The default runs frontend and estimator in sequence. With `port.frontend_lag`,
/// each call tracks this frame beside the previous frame's estimate, returning
/// the previous timestamp. `flush` drains the final result. Both schedules are
/// deterministic. The GIL is released around frontend and estimator compute.
#[pyclass(module = "slam_rs._core")]
pub struct Vio {
    inner: slam_rs::Vio<f32>,
}

#[pymethods]
impl Vio {
    /// Build an f32 pipeline from calibration and configuration.
    /// Inputs can come from JSON or catalog dataclasses. `backend` is `"auto"`
    /// (the default), `"cpu"` or `"gpu"`, resolved by `slam_rs::select_backend`:
    /// auto runs the CubeCL frontend when this host can and otherwise the CPU,
    /// with one warning on stderr; `"gpu"` raises `ValueError` instead of falling
    /// back. Invalid input and runtime initialization panics also raise
    /// `ValueError` (D32). `threads` controls only the CPU pool and is accepted
    /// but inert on the GPU lane. `ba_backend` is where `finish_map`'s global BA
    /// runs its PCG: `"follow"` (the default, the frontend's lane), `"cpu"` or
    /// `"gpu"`, for benches and tests.
    #[new]
    #[pyo3(signature = (calibration, config, *, threads = 1, max_keypoints = None, backend = "auto", ba_backend = "follow"))]
    fn new(
        calibration: PyRef<'_, Calibration>,
        config: PyRef<'_, VioConfig>,
        threads: usize,
        max_keypoints: Option<usize>,
        backend: &str,
        ba_backend: &str,
    ) -> PyResult<Self> {
        let request: BackendRequest = backend.parse().map_err(PyValueError::new_err)?;
        let gba: GbaRequest = ba_backend.parse().map_err(PyValueError::new_err)?;
        let backends: Backends = slam_rs::select_backend(request, gba).map_err(value_error)?;
        Ok(Self {
            inner: slam_rs::Vio::with_backends(
                config.inner.clone(),
                calibration.inner.clone(),
                frontend_options(threads, max_keypoints),
                backends,
            )
            .map_err(value_error)?,
        })
    }

    /// `(corrected, no_coverage)`: frames whose keypoints the rolling-shutter
    /// correction moved, and frames left raw because the gyro did not cover their
    /// time; `None` when no camera states a readout.
    #[getter]
    fn rolling_shutter_frames(&self) -> Option<(usize, usize)> {
        self.inner.rolling_shutter().map(slam_rs::RollingShutter::counts)
    }

    /// `(width, height)` of the frames each camera is tracked on: the calibration's, or half of it when every
    /// camera is wider than 2048 px (keypoints are then in the reduced pixels).
    #[getter]
    fn tracked_resolution(&self) -> Vec<(u32, u32)> {
        self.inner
            .calibration()
            .resolution
            .iter()
            .map(|&[width, height]| (width, height))
            .collect()
    }

    /// Whether the frontend runs on the GPU.
    #[getter]
    fn gpu(&self) -> bool {
        self.inner.backend() == Backend::Gpu
    }

    /// The lane this pipeline runs: `"cpu"` or `"gpu"`.
    #[getter]
    fn backend(&self) -> &'static str {
        self.inner.backend().name()
    }

    /// Where the global BA's PCG runs: `"cpu"` or `"gpu"`.
    #[getter]
    fn gba_lane(&self) -> &'static str {
        self.inner.gba_lane().name()
    }

    /// Cameras this estimator expects in every frameset.
    ///
    /// Read off the frontend rather than mirrored, so it cannot drift from the
    /// calibration the rig was built with.
    #[getter]
    fn camera_count(&self) -> usize {
        self.inner.frontend().camera_count()
    }

    /// Unit bearings of camera `camera`'s pixels (`float32[n, 2]`, as
    /// `FlowFrame.positions` gives them) in camera 0's frame (rotation only):
    /// `float64[n, 3]`, NaN where the pixel does not unproject.
    fn bearings<'py>(
        &self,
        py: Python<'py>,
        camera: usize,
        pixels: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let array: &Bound<'_, PyArray2<f32>> = pixels
            .cast::<PyArray2<f32>>()
            .map_err(|_| PyValueError::new_err("pixels must be a 2-D float32 numpy array"))?;
        if array.shape()[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "pixels must have shape (n, 2), got {:?}",
                array.shape()
            )));
        }
        let readonly = array.readonly();
        let values: &[f32] = contiguous(&readonly, "pixels", "pixels")?;
        let calibration: CoreCalibration<f64> = self.inner.calibration().cast();
        if camera >= calibration.intrinsics.len() {
            return Err(PyIndexError::new_err(format!(
                "camera {camera} out of range for {} cameras",
                calibration.intrinsics.len()
            )));
        }
        let model = slam_rs::camera::SlamCamera::from_model(&calibration.intrinsics[camera])
            .map_err(value_error)?;
        let to_first = calibration.t_i_c[0].rotation.inverse() * calibration.t_i_c[camera].rotation;
        let bearings: Vec<f64> = values
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[x, y]| {
                let mut point = Default::default();
                model.unproject(&[f64::from(x), f64::from(y)].into(), &mut point);
                let bearing = to_first * point.xyz();
                [bearing.x, bearing.y, bearing.z]
            })
            .collect();
        bearings.into_pyarray(py).reshape((values.len() / 2, 3))
    }

    /// Add one IMU sample: `gyro` in rad/s, `accel` in m/s², both in the rig
    /// frame and uncalibrated — the static bias calibration is applied inside.
    fn push_imu(&mut self, t_ns: i64, gyro: [f64; 3], accel: [f64; 3]) -> PyResult<()> {
        self.inner.push_imu(t_ns, gyro, accel).map_err(value_error)
    }

    /// Add `n` IMU samples at once: `t_ns` is `int64[n]`, `gyro` and `accel` are `float64[n, 3]`.
    ///
    /// All or nothing: a batch the core would refuse anywhere is refused whole,
    /// so the estimator is left where it was and the batch can be corrected and
    /// pushed again.
    fn push_imu_batch(
        &mut self,
        py: Python<'_>,
        t_ns: &Bound<'_, PyAny>,
        gyro: &Bound<'_, PyAny>,
        accel: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let times: Vec<i64> = int64_column(t_ns, "t_ns")?;
        let gyro_rows: Vec<[f64; 3]> = float64_triples(gyro, "gyro")?;
        let accel_rows: Vec<[f64; 3]> = float64_triples(accel, "accel")?;
        if times.len() != gyro_rows.len() || times.len() != accel_rows.len() {
            return Err(PyValueError::new_err(format!(
                "t_ns, gyro and accel must have the same length, got {}, {} and {}",
                times.len(),
                gyro_rows.len(),
                accel_rows.len()
            )));
        }
        py.detach(|| {
            // All or nothing. Pushing as it goes leaves the samples before a bad
            // one in the estimator and the frontier past them, so the caller can
            // neither retry the batch nor correct it: `[10, 20, 20, 30]` raised
            // at the duplicate and then refused 10 and 20 as too old. The whole
            // batch is decided first, each sample against the one before it and
            // the frontier.
            let mut previous_t_ns: Option<i64> = self.inner.last_imu_t_ns();
            for ((&t, gyro_sample), accel_sample) in
                times.iter().zip(gyro_rows.iter()).zip(accel_rows.iter())
            {
                slam_rs::check_imu_sample(t, gyro_sample, accel_sample, previous_t_ns)?;
                previous_t_ns = Some(t);
            }
            for ((&t, &gyro_sample), &accel_sample) in
                times.iter().zip(gyro_rows.iter()).zip(accel_rows.iter())
            {
                self.inner.push_imu(t, gyro_sample, accel_sample)?;
            }
            Ok::<(), VioError>(())
        })
        .map_err(value_error)
    }

    /// Process one frameset: `images` holds one `uint8[h, w]` array per camera.
    ///
    /// On the GPU lane, pixels widen directly into reused core storage while
    /// the GIL is held. Computation then runs without a NumPy borrow. The CPU
    /// lane copies the bytes before detaching. A refused frameset leaves the
    /// estimator exactly as the last accepted one did.
    ///
    /// On a GPU lane a device that dies mid-run is a `ValueError` here as well:
    /// CubeCL panics on a lost device rather than returning an error, and every
    /// GPU stage runs inside a guard that turns that into the core's typed error
    /// before it can unwind through the released GIL (decision D32).
    ///
    /// `masks`, when given, holds one entry per camera: `None` or a `uint8[h, w]` segmentation mask at any
    /// resolution, nonzero = no features there; it is read through its scale to the camera's tracked image
    /// (`slam_rs::frontend::masks`). The frontend neither detects in nor keeps a track in a detection cell that
    /// holds a masked pixel. `masks=None` masks nothing.
    #[pyo3(signature = (t_ns, images, masks = None))]
    fn track(
        &mut self,
        py: Python<'_>,
        t_ns: i64,
        images: Vec<Bound<'_, PyAny>>,
        masks: Option<Vec<Option<Bound<'_, PyAny>>>>,
    ) -> PyResult<VioResult> {
        self.set_frame_masks(masks)?;
        if self.inner.backend() == slam_rs::Backend::Gpu {
            let arrays: Vec<PyReadonlyArray2<'_, u8>> = images
                .iter()
                .enumerate()
                .map(|(index, image)| {
                    let array = gray_array(image, index, "image")?;
                    gray_pixels(&array, index, "image")?;
                    Ok(array)
                })
                .collect::<PyResult<_>>()?;
            let views: Vec<ImageView<'_>> = arrays
                .iter()
                .enumerate()
                .map(|(index, array)| {
                    let (data, width, height) = gray_pixels(array, index, "image")?;
                    Ok(ImageView {
                        width,
                        height,
                        stride: width,
                        data,
                    })
                })
                .collect::<PyResult<_>>()?;
            let prepared = self
                .inner
                .prepare_track(t_ns, &views)
                .map_err(value_error)?;
            drop(views);
            drop(arrays);
            let inner = py.detach(|| prepared.finish()).map_err(value_error)?;
            return Ok(VioResult { inner });
        }
        let mut frames: Vec<GrayImage> = Vec::with_capacity(images.len());
        for (index, image) in images.iter().enumerate() {
            frames.push(gray_image(image, index)?);
        }
        let result: slam_rs::VioResult = py
            .detach(|| {
                let views: Vec<ImageView<'_>> = frames
                    .iter()
                    .map(|frame| ImageView {
                        width: frame.width,
                        height: frame.height,
                        stride: frame.width,
                        data: &frame.pixels,
                    })
                    .collect();
                self.inner.track(t_ns, &views)
            })
            .map_err(value_error)?;
        Ok(VioResult { inner: result })
    }

    /// The keypoints the frontend tracked on the last accepted frameset.
    ///
    /// `None` before the first one. The estimator drives its own frontend, so
    /// this is where a Rerun rung reads the keypoints it draws over the images;
    /// they are copied out on every call, as [`OpticalFlow::process`] copies them
    /// out of the same buffers.
    fn flow_frame(&self) -> PyResult<Option<FlowFrame>> {
        let Some(t_ns) = self.inner.frontend().t_ns() else {
            return Ok(None);
        };
        flow_frame(self.inner.frontend(), t_ns)
            .map(Some)
            .map_err(PyErr::from)
    }

    /// Drain the final lagged pose once, or return `None` if none is queued.
    fn flush(&mut self, py: Python<'_>) -> PyResult<Option<VioResult>> {
        py.detach(|| self.inner.flush())
            .map(|result| result.map(|inner| VioResult { inner }))
            .map_err(value_error)
    }

    /// What the frontend lane cost on the last frameset that ran it, in ms:
    /// `imu` (its motion preintegration), `pyramid`, `detect`, `track` and
    /// `stereo`. Unlike a snapshot's timings it exists before the estimator has
    /// a state, so a bootstrapping run can be timed too.
    #[getter]
    fn frontend_timings_ms(&self) -> std::collections::BTreeMap<&'static str, f64> {
        let timings = self.inner.frontend_timings();
        [
            ("imu", timings.imu_ns),
            ("pyramid", timings.flow.pyramid_ns),
            ("detect", timings.flow.detect_ns),
            ("track", timings.flow.track_ns),
            ("stereo", timings.flow.stereo_ns),
        ]
        .into_iter()
        .map(|(name, ns)| (name, ns as f64 / 1e6))
        .collect()
    }

    /// The inertial bootstrap's attempts so far (`config.vio_bootstrap`), or
    /// None on the stereo start.
    fn bootstrap_report(&self) -> Option<bootstrap::BootstrapReport> {
        self.inner.bootstrap_report().map(Into::into)
    }

    /// The window, its landmarks and the last measured frame's statistics.
    ///
    /// `None` until a frameset has been measured: before that the window is
    /// empty and there are no statistics to report. Everything is copied, so a
    /// snapshot stays valid across the next `track`.
    /// With frontend lag, frontend timers describe the newer input images;
    /// the window and estimator statistics describe the returned pose.
    fn snapshot(&self) -> PyResult<Option<VioSnapshot>> {
        let Some(stats) = self.inner.last_stats() else {
            return Ok(None);
        };
        VioSnapshot::build(
            &self.inner.estimator().snapshot(),
            stats,
            self.inner.frontend_timings(),
        )
        .map(Some)
    }

    /// What changed in the map since revision ``since`` (``0``: everything):
    /// every keyframe pose and the loops, and the points of only the keyframes
    /// a write touched since. Costs the changed points, not the whole map.
    /// ``None`` with ``backend.enabled`` off, or in Realtime while the back-end
    /// thread holds the map (it never waits; ask again later).
    fn map_delta(&self, py: Python<'_>, since: u64) -> Option<map::MapDelta> {
        py.detach(|| self.inner.map_delta(since)).map(map::MapDelta::new)
    }

    /// A copy of the persistent keyframe map and the back end's counters at the
    /// moment it was taken, or `None` when ``backend.enabled`` is off.
    ///
    /// Never waits for the back end: in Realtime it is the newest copy the
    /// back-end thread posted (``flush`` makes it current).
    fn map_snapshot(&self, py: Python<'_>) -> PyResult<Option<map::MapSnapshot>> {
        let (Some(view), Some(mode)) = (
            py.detach(|| self.inner.map_view()),
            self.inner.backend_mode(),
        ) else {
            return Ok(None);
        };
        Ok(Some(map::MapSnapshot::new(view.view, view.counters, mode)))
    }

    /// The map's keyframes at the end of a sequence, after a global BA on the
    /// map when ``run_gba``, or ``None`` when ``backend.enabled`` is off.
    ///
    /// Takes every stamp first. The window is untouched, so the poses ``track``
    /// returned stay the raw VIO; the map keeps the GBA's. Call it after
    /// ``flush``. A GBA that fails is reported in ``gba_failure``, and the
    /// keyframes are then the map's after loops and PGO.
    fn finish_map(&mut self, py: Python<'_>, run_gba: bool) -> Option<gba::MapResult> {
        py.detach(|| self.inner.finish_map(run_gba))
            .map(gba::MapResult::new)
    }

    fn __repr__(&self) -> String {
        format!("Vio(camera_count={})", self.camera_count())
    }
}

/// The estimator's window and last frame, copied out for the Rerun rung (D51).
///
/// Everything is a flat array or a scalar: the core logs nothing itself (D03),
/// so this is what the Python layer draws from. The window is the 15-dof states
/// followed by the pose-only blocks, each oldest first; `keyframe`/`long_term`
/// say what each frame is, and `kf_ids` is the keyframes as an id list, which is
/// what a count wants.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
#[derive(Debug)]
pub struct VioSnapshot {
    t_ns: i64,
    window_t_ns: Vec<i64>,
    /// `[tx, ty, tz, qx, qy, qz, qw]` per window frame.
    window_poses: Vec<f64>,
    window_keyframe: Vec<bool>,
    window_long_term: Vec<bool>,
    kf_ids: Vec<i64>,
    marginalized: Vec<i64>,
    landmark_ids: Vec<i64>,
    landmark_hosts: Vec<i64>,
    /// `[x, y, z]` per landmark, world frame.
    landmark_positions: Vec<f64>,
    lm_iterations: usize,
    lm_lambda: f64,
    lm_error_before: f64,
    lm_error_after: f64,
    num_observations: usize,
    frame_update: &'static str,
    timings: slam_rs::estimator::StageTimings,
    frontend_timings: slam_rs::FrontendTimings,
}

impl VioSnapshot {
    /// Flatten one window snapshot and the frame that produced it.
    ///
    /// # Errors
    ///
    /// `ValueError` when a landmark id has outgrown the `int64` the boundary
    /// hands to numpy, which is the same refusal `OpticalFlow` makes for the
    /// keypoint ids the landmarks inherit.
    fn build(
        window: &slam_rs::estimator::WindowSnapshot<f32>,
        stats: &slam_rs::estimator::FrameStats<f32>,
        frontend_timings: slam_rs::FrontendTimings,
    ) -> PyResult<Self> {
        let frames: usize = window.states.len() + window.poses.len();
        let mut window_t_ns: Vec<i64> = Vec::with_capacity(frames);
        let mut window_poses: Vec<f64> = Vec::with_capacity(7 * frames);
        let mut window_keyframe: Vec<bool> = Vec::with_capacity(frames);
        let mut window_long_term: Vec<bool> = Vec::with_capacity(frames);
        for state in window.states.iter().chain(window.poses.iter()) {
            let quaternion: [f32; 4] = state.t_w_i.rotation.quaternion_xyzw();
            window_t_ns.push(state.t_ns);
            window_poses.extend_from_slice(&[
                f64::from(state.t_w_i.translation.x),
                f64::from(state.t_w_i.translation.y),
                f64::from(state.t_w_i.translation.z),
                f64::from(quaternion[0]),
                f64::from(quaternion[1]),
                f64::from(quaternion[2]),
                f64::from(quaternion[3]),
            ]);
            window_keyframe.push(state.keyframe);
            window_long_term.push(state.long_term_keyframe);
        }
        let mut landmark_ids: Vec<i64> = Vec::with_capacity(window.landmarks.len());
        let mut landmark_hosts: Vec<i64> = Vec::with_capacity(window.landmarks.len());
        let mut landmark_positions: Vec<f64> = Vec::with_capacity(3 * window.landmarks.len());
        for landmark in &window.landmarks {
            landmark_ids.push(i64::try_from(landmark.id.0).map_err(|_| {
                PyValueError::new_err(format!(
                    "landmark id {} does not fit in an int64",
                    landmark.id.0
                ))
            })?);
            landmark_hosts.push(landmark.host.frame_id);
            landmark_positions.extend_from_slice(&[
                f64::from(landmark.position_w.x),
                f64::from(landmark.position_w.y),
                f64::from(landmark.position_w.z),
            ]);
        }
        Ok(Self {
            t_ns: window.t_ns,
            window_t_ns,
            window_poses,
            window_keyframe,
            window_long_term,
            kf_ids: stats.kf_ids.clone(),
            marginalized: window.marginalized.clone(),
            landmark_ids,
            landmark_hosts,
            landmark_positions,
            lm_iterations: stats.lm.len(),
            // The trail is empty for the first four framesets, where `opt_started`
            // is still false and no linearization ran at all.
            lm_lambda: stats.lm.last().map_or(0.0, |step| f64::from(step.lambda)),
            lm_error_before: stats
                .lm
                .first()
                .map_or(0.0, |step| f64::from(step.error_before)),
            lm_error_after: stats
                .lm
                .last()
                .map_or(0.0, |step| f64::from(step.error_after)),
            num_observations: stats.num_observations,
            frame_update: stats.frame_update.as_str(),
            timings: stats.timings,
            frontend_timings,
        })
    }
}

#[pymethods]
impl VioSnapshot {
    /// Frameset timestamp of the newest state in the window.
    #[getter]
    fn t_ns(&self) -> i64 {
        self.t_ns
    }

    /// Timestamps of the window's frames: `int64[n]`.
    #[getter]
    fn window_t_ns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.window_t_ns.to_pyarray(py)
    }

    /// Rig poses of the window's frames, `[tx, ty, tz, qx, qy, qz, qw]`: `float64[n, 7]`.
    #[getter]
    fn window_poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.window_poses
            .to_pyarray(py)
            .reshape((self.window_t_ns.len(), 7))
    }

    /// Whether each window frame is a keyframe: `bool[n]`.
    ///
    /// The estimator answers this per frame, so nothing downstream has to join
    /// [`VioSnapshot::kf_ids`] back onto [`VioSnapshot::window_t_ns`].
    #[getter]
    fn window_keyframe<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.window_keyframe.to_pyarray(py)
    }

    /// Whether each window frame is a long-term keyframe: `bool[n]`.
    #[getter]
    fn window_long_term<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.window_long_term.to_pyarray(py)
    }

    /// The keyframes' timestamps, oldest first: `int64[k]`.
    #[getter]
    fn kf_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.kf_ids.to_pyarray(py)
    }

    /// Frames the last marginalization removed from the window: `int64[m]`.
    #[getter]
    fn marginalized<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.marginalized.to_pyarray(py)
    }

    /// Landmark ids, which are the ids of the keypoints that spawned them: `int64[p]`.
    #[getter]
    fn landmark_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.landmark_ids.to_pyarray(py)
    }

    /// Timestamp of the keyframe hosting each landmark: `int64[p]`.
    #[getter]
    fn landmark_hosts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.landmark_hosts.to_pyarray(py)
    }

    /// Landmark positions in the world frame, metres: `float64[p, 3]`.
    #[getter]
    fn landmark_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.landmark_positions
            .to_pyarray(py)
            .reshape((self.landmark_ids.len(), 3))
    }

    /// Levenberg-Marquardt steps the last frame took, accepted and rejected alike.
    #[getter]
    fn lm_iterations(&self) -> usize {
        self.lm_iterations
    }

    /// Damping the last step solved with; zero when no step ran.
    #[getter]
    fn lm_lambda(&self) -> f64 {
        self.lm_lambda
    }

    /// Total cost before the first step; zero when no step ran.
    #[getter]
    fn lm_error_before(&self) -> f64 {
        self.lm_error_before
    }

    /// Total cost after the last step; zero when no step ran.
    #[getter]
    fn lm_error_after(&self) -> f64 {
        self.lm_error_after
    }

    /// Landmark observations the window holds after the last frame.
    #[getter]
    fn num_observations(&self) -> usize {
        self.num_observations
    }

    /// What D76's frame update did with this frameset: `not_attempted`,
    /// `taken`, or `declined_<precondition>`. A clip whose median frame is slow
    /// with the knob on says which precondition refused it here.
    #[getter]
    fn frame_update(&self) -> &'static str {
        self.frame_update
    }

    /// The GPU lane collected this frameset's temporal and stereo results in
    /// one device wait (one-wait); always false on the CPU lane.
    #[getter]
    fn gpu_one_wait(&self) -> bool {
        self.frontend_timings.flow.gpu_one_wait
    }

    /// The GPU lane reused this frameset's timestamp-bound lookahead image
    /// upload; always false on the CPU lane.
    #[getter]
    fn gpu_lookahead(&self) -> bool {
        self.frontend_timings.flow.gpu_lookahead
    }

    /// Wall time each stage took on the last frame, milliseconds.
    ///
    /// The estimator's nine — `predict`, `keyframe`, `optimize`, `linearize`,
    /// `solver`, `back_substitution`, `error`, `marginalize`, `measure` — and
    /// the frontend lane's five under a `frontend_` prefix: the pyramid build,
    /// the FAST detection, the temporal KLT calls, the cross-camera stereo
    /// match, and the preintegration that seeds the KLT. The five are the same
    /// kind of measurement as the nine and are read the same way, which is why
    /// they come back in one map. They do not add up to the frame: the timers
    /// nest — `optimize` covers `linearize`, `solver`, `back_substitution` and
    /// `error`, and `measure` covers `keyframe`, `optimize`, `marginalize` and
    /// state prediction — and the bookkeeping between the phases is nobody's
    /// stage. With frontend lag the frontend is one image ahead of the
    /// estimator, and their work overlaps. `slam_rs/_core.pyi` carries the
    /// same contract for Python readers.
    #[getter]
    fn timings_ms(&self) -> std::collections::BTreeMap<&'static str, f64> {
        [
            ("back_substitution", self.timings.back_substitution_ns),
            ("error", self.timings.error_ns),
            ("linearize", self.timings.linearize_ns),
            ("marginalize", self.timings.marginalize_ns),
            ("measure", self.timings.measure_ns),
            ("solver", self.timings.solver_ns),
            ("predict", self.timings.predict_ns),
            ("keyframe", self.timings.keyframe_ns),
            ("optimize", self.timings.optimize_ns),
            ("frontend_stereo", self.frontend_timings.flow.stereo_ns),
            ("frontend_pyramid", self.frontend_timings.flow.pyramid_ns),
            ("frontend_detect", self.frontend_timings.flow.detect_ns),
            ("frontend_track", self.frontend_timings.flow.track_ns),
            ("frontend_imu", self.frontend_timings.imu_ns),
        ]
        .into_iter()
        .map(|(name, ns)| (name, ns as f64 / 1e6))
        .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "VioSnapshot(t_ns={}, window={}, keyframes={}, landmarks={})",
            self.t_ns,
            self.window_t_ns.len(),
            self.kf_ids.len(),
            self.landmark_ids.len()
        )
    }
}

/// The name `backend.mode` gives a mode.
fn backend_mode_name(mode: slam_rs::config::BackendMode) -> &'static str {
    match mode {
        slam_rs::config::BackendMode::Offline => "offline",
        slam_rs::config::BackendMode::Realtime => "realtime",
    }
}

/// `[tx, ty, tz, qx, qy, qz, qw]` per pose, flat.
fn flat_poses(poses: &[slam_rs::lie::Se3<f64>]) -> Vec<f64> {
    poses.iter().flat_map(slam_rs::pose_to_xyzw).collect()
}

/// The frontend options both entry points build, from the two knobs they expose.
///
/// Everything else in `FrontendOptions` is a property of the port rather than of
/// a run, so `None` means the default rather than "unset". On the CPU lane,
/// `threads` sizes the workers shared by tracking, pyramids, detection and the
/// estimator. The estimator runs serially on the GPU lane and during a D84
/// deferred solve; `threads` has no effect on the GPU lane.
fn frontend_options(threads: usize, max_keypoints: Option<usize>) -> FrontendOptions {
    let defaults: FrontendOptions = FrontendOptions::default();
    FrontendOptions {
        threads,
        max_keypoints: max_keypoints.unwrap_or(defaults.max_keypoints),
    }
}

/// One grayscale frame copied out of Python, packed (stride == width).
struct GrayImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

/// The C-contiguous elements of an array borrowed out of Python.
///
/// Both halves of one check, so the refusal has one wording. `as_slice` alone
/// accepts Fortran order, whose bytes run down the columns and would transpose
/// every row-major read here, so the C flag is asked explicitly; `as_slice`'s
/// own refusal — which the flag has already ruled out — lands on the same
/// sentence rather than a second copy of it. `what` names the array in the
/// message and `hint` is what the caller should wrap.
fn contiguous<'a, T, D>(
    readonly: &'a numpy::PyReadonlyArray<'_, T, D>,
    what: &str,
    hint: &str,
) -> PyResult<&'a [T]>
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    let refused = || {
        PyValueError::new_err(format!(
            "{what} must be C-contiguous; pass numpy.ascontiguousarray({hint})"
        ))
    };
    if !readonly.is_c_contiguous() {
        return Err(refused());
    }
    readonly.as_slice().map_err(|_| refused())
}

impl Vio {
    /// The masks `track` was given, one optional `uint8[h, w]` per camera, or none at all.
    fn set_frame_masks(&mut self, masks: Option<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<()> {
        let Some(masks) = masks else {
            let none: Vec<Option<slam_rs::frontend::masks::PixelMask<'_>>> =
                vec![None; self.inner.frontend().camera_count()];
            return self.inner.set_frame_masks(&none).map_err(value_error);
        };
        let arrays: Vec<Option<PyReadonlyArray2<'_, u8>>> = masks
            .iter()
            .enumerate()
            .map(|(index, mask)| {
                mask.as_ref()
                    .map(|mask| gray_array(mask, index, "mask"))
                    .transpose()
            })
            .collect::<PyResult<_>>()?;
        let views: Vec<Option<slam_rs::frontend::masks::PixelMask<'_>>> = arrays
            .iter()
            .enumerate()
            .map(|(index, array)| {
                array
                    .as_ref()
                    .map(|array| {
                        gray_pixels(array, index, "mask").map(|(data, width, height)| {
                            slam_rs::frontend::masks::PixelMask {
                                width,
                                height,
                                data,
                            }
                        })
                    })
                    .transpose()
            })
            .collect::<PyResult<_>>()?;
        self.inner.set_frame_masks(&views).map_err(value_error)
    }
}

/// Borrow one `uint8[h, w]` array out of Python: rank and dtype checked.
///
/// Consumers either copy the bytes or widen them into reused [`Image<u16, 1>`]
/// storage. The layout is [`gray_pixels`]' half of the same pair.
fn gray_array<'py>(
    object: &Bound<'py, PyAny>,
    index: usize,
    what: &str,
) -> PyResult<PyReadonlyArray2<'py, u8>> {
    let array: &Bound<'py, PyArray2<u8>> = object.cast::<PyArray2<u8>>().map_err(|_| {
        PyValueError::new_err(format!("{what} {index} must be a 2-D uint8 numpy array"))
    })?;
    Ok(array.readonly())
}

/// The pixels, width and height of an array [`gray_array`] has accepted.
fn gray_pixels<'a>(
    readonly: &'a PyReadonlyArray2<'_, u8>,
    index: usize,
    what: &str,
) -> PyResult<(&'a [u8], usize, usize)> {
    let shape: &[usize] = readonly.shape();
    let pixels: &[u8] = contiguous(readonly, &format!("{what} {index}"), what)?;
    Ok((pixels, shape[1], shape[0]))
}

/// Copy one C-contiguous `uint8[h, w]` array out of Python.
fn gray_image(object: &Bound<'_, PyAny>, index: usize) -> PyResult<GrayImage> {
    let readonly: PyReadonlyArray2<'_, u8> = gray_array(object, index, "image")?;
    let (pixels, width, height) = gray_pixels(&readonly, index, "image")?;
    Ok(GrayImage {
        width,
        height,
        pixels: pixels.to_vec(),
    })
}

/// Copy a C-contiguous `int64[n]` array out of Python.
fn int64_column(object: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    let array: &Bound<'_, PyArray1<i64>> = object
        .cast::<PyArray1<i64>>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a 1-D int64 numpy array")))?;
    let readonly = array.readonly();
    Ok(contiguous(&readonly, name, name)?.to_vec())
}

/// Copy a C-contiguous `float64[n, 3]` array out of Python, row by row.
fn float64_triples(object: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<[f64; 3]>> {
    let array: &Bound<'_, PyArray2<f64>> = object
        .cast::<PyArray2<f64>>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a 2-D float64 numpy array")))?;
    let shape: &[usize] = array.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n, 3), got {shape:?}"
        )));
    }
    // Fortran order passes as_slice() but its bytes run down the columns, which
    // would turn the chunks below into transposed samples.
    let readonly = array.readonly();
    let values: &[f64] = contiguous(&readonly, name, name)?;
    Ok(values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

/// VIO configuration read from JSON. Omitted keys retain constructor defaults.
#[pyclass(module = "slam_rs._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct VioConfig {
    inner: CoreVioConfig,
}

#[pymethods]
impl VioConfig {
    /// Default VIO configuration.
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreVioConfig::default(),
        }
    }

    /// Read configuration JSON, retaining defaults for omitted keys.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreVioConfig::from_json_str(text).map_err(value_error)?,
        })
    }

    /// Write configuration JSON with the `value0` wrapper.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(value_error)
    }

    /// `config.optical_flow_image_safe_radius`: the circular mask that hides a
    /// fisheye's black corners, in pixels. Zero disables it.
    ///
    /// This is the one frontend field that differs per device rather than per
    /// dataset — 472 on Index, 340 on G2 — and the reference manifest carries it
    /// per segment, so it is settable here rather than only through a file.
    #[getter]
    fn optical_flow_image_safe_radius(&self) -> f32 {
        self.inner.optical_flow_image_safe_radius
    }

    #[setter]
    fn set_optical_flow_image_safe_radius(&mut self, radius: f32) {
        self.inner.optical_flow_image_safe_radius = radius;
    }

    /// `config.vio_bootstrap`: `"stereo"` (the default) or `"inertial"`.
    #[getter]
    fn vio_bootstrap(&self) -> &'static str {
        match self.inner.vio_bootstrap {
            slam_rs::config::VioBootstrap::Stereo => "stereo",
            slam_rs::config::VioBootstrap::Inertial => "inertial",
        }
    }

    #[setter]
    fn set_vio_bootstrap(&mut self, value: &str) -> PyResult<()> {
        self.inner.vio_bootstrap = match value {
            "stereo" => slam_rs::config::VioBootstrap::Stereo,
            "inertial" => slam_rs::config::VioBootstrap::Inertial,
            other => {
                return Err(PyValueError::new_err(format!(
                    "vio_bootstrap must be stereo or inertial, not {other:?}"
                )));
            }
        };
        Ok(())
    }

    /// ``backend.enabled``: whether `Vio` builds the persistent keyframe map.
    #[getter]
    fn backend_enabled(&self) -> bool {
        self.inner.backend_enabled
    }

    /// ``backend.mode``: ``"offline"`` (inline) or ``"realtime"`` (own thread).
    #[getter]
    fn backend_mode(&self) -> &'static str {
        backend_mode_name(self.inner.backend_mode)
    }

    /// The same configuration with the back end ``"off"`` (``backend.mode``
    /// kept) or on in ``"offline"`` or ``"realtime"``; every other key unchanged.
    fn with_backend(&self, mode: &str) -> PyResult<Self> {
        let mode: Option<slam_rs::config::BackendMode> = match mode {
            "off" => None,
            "offline" => Some(slam_rs::config::BackendMode::Offline),
            "realtime" => Some(slam_rs::config::BackendMode::Realtime),
            other => {
                return Err(PyValueError::new_err(format!(
                    "back-end mode {other:?} is not \"off\", \"offline\" or \"realtime\""
                )));
            }
        };
        Ok(Self {
            inner: self.inner.with_backend(mode),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "VioConfig(optical_flow_type={:?}, optical_flow_pattern={}, optical_flow_levels={}, optical_flow_image_safe_radius={})",
            self.inner.optical_flow_type,
            self.inner.optical_flow_pattern,
            self.inner.optical_flow_levels,
            self.inner.optical_flow_image_safe_radius
        )
    }
}

/// Camera-IMU calibration: extrinsics, intrinsics and noise model.
/// Read JSON or copy catalog dataclass fields into the same Rust implementation.
/// Rotation validation, model names and isotropic noise conversion stay in one place.
#[pyclass(module = "slam_rs._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct Calibration {
    inner: CoreCalibration<f64>,
}

#[pymethods]
impl Calibration {
    /// Read calibration JSON.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCalibration::<f64>::from_json_str(text).map_err(value_error)?,
        })
    }

    /// Build calibration from cameras in rig order and IMU fields.
    /// `imu_T_body` is ignored because the recording rig frame is the IMU frame.
    #[staticmethod]
    fn from_catalog(cameras: Vec<Bound<'_, PyAny>>, imu: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut parts: Vec<CoreCameraParts<f64>> = Vec::with_capacity(cameras.len());
        for (index, camera) in cameras.iter().enumerate() {
            parts.push(camera_parts(camera, index)?);
        }
        Ok(Self {
            inner: CoreCalibration::from_catalog_parts(&parts, &imu_parts(imu)?)
                .map_err(value_error)?,
        })
    }

    /// Write calibration JSON with the `value0` wrapper.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(value_error)
    }

    /// Cameras on the rig.
    #[getter]
    fn camera_count(&self) -> usize {
        self.inner.camera_count()
    }

    /// Each camera's `(width, height)` in pixels, in rig order.
    #[getter]
    fn resolution(&self) -> Vec<(u32, u32)> {
        self.inner
            .resolution
            .iter()
            .map(|size| (size[0], size[1]))
            .collect()
    }

    /// Camera model names in rig order.
    #[getter]
    fn camera_models(&self) -> Vec<&'static str> {
        self.inner
            .intrinsics
            .iter()
            .map(|lens| lens.name())
            .collect()
    }

    /// Per-camera parameters: fx, fy, cx, cy, then model-specific terms.
    #[getter]
    fn intrinsics(&self) -> Vec<Vec<f64>> {
        self.inner
            .intrinsics
            .iter()
            .map(|lens| lens.params())
            .collect()
    }

    /// Camera-to-IMU transforms, copied as row-major float64[4, 4] arrays.
    #[getter(imu_T_cam)]
    fn imu_t_cam<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        self.inner
            .t_i_c
            .iter()
            .map(|pose| {
                let matrix = pose.matrix();
                let values: Vec<f64> = (0..4)
                    .flat_map(|row| (0..4).map(move |col| matrix[(row, col)]))
                    .collect();
                values.to_pyarray(py).reshape((4, 4))
            })
            .collect()
    }

    /// Gravity magnitude, m/s²: the recording's, or 9.81 when it states none.
    #[getter]
    fn gravity_ms2(&self) -> f64 {
        self.inner.gravity_ms2
    }

    /// Rolling-shutter readout per camera, seconds, `None` where the camera states none.
    #[getter]
    fn readout_s(&self) -> Vec<Option<f64>> {
        self.inner.readout_s.clone()
    }

    /// IMU sample rate in Hz.
    #[getter]
    fn imu_update_rate(&self) -> f64 {
        self.inner.imu_update_rate
    }

    /// Gyroscope noise density per axis.
    #[getter]
    fn gyro_noise_std(&self) -> Vec<f64> {
        self.inner.gyro_noise_std.as_slice().to_vec()
    }

    /// Accelerometer noise density per axis.
    #[getter]
    fn accel_noise_std(&self) -> Vec<f64> {
        self.inner.accel_noise_std.as_slice().to_vec()
    }

    /// Gyroscope bias random walk per axis.
    #[getter]
    fn gyro_bias_std(&self) -> Vec<f64> {
        self.inner.gyro_bias_std.as_slice().to_vec()
    }

    /// Accelerometer bias random walk per axis.
    #[getter]
    fn accel_bias_std(&self) -> Vec<f64> {
        self.inner.accel_bias_std.as_slice().to_vec()
    }

    /// Offset added to camera timestamps to reach the IMU clock.
    #[getter]
    fn cam_time_offset_ns(&self) -> i64 {
        self.inner.cam_time_offset_ns
    }

    fn __repr__(&self) -> String {
        format!("Calibration(camera_count={})", self.inner.camera_count())
    }
}

/// One camera's keypoints for one frameset, copied out of the frontend.
///
/// Owned rather than borrowed: the frontend's buffers are reused frame to frame,
/// so a view would change under the caller on the next `process`.
#[derive(Debug, Default)]
struct CameraKeypoints {
    /// Keypoint ids, ascending.
    ids: Vec<i64>,
    /// `[m00, m01, tx, m10, m11, ty]` per keypoint, keypoint index slow-varying.
    ///
    /// The keypoint's pixel is `[tx, ty]`, so this is where a position comes
    /// from as well ([`FlowFrame::positions`]).
    transforms: Vec<f32>,
    /// Occupancy counts, row-major over the frame's grid.
    occupancy: Vec<i32>,
    /// Ids handed out on this frameset: detections plus stereo matches.
    num_new: usize,
}

/// What one `OpticalFlow::process` call produced.
#[pyclass(module = "slam_rs._core", frozen)]
#[derive(Debug)]
pub struct FlowFrame {
    t_ns: i64,
    grid: CellGrid,
    cameras: Vec<CameraKeypoints>,
}

impl FlowFrame {
    /// One camera's keypoints, or an `IndexError`.
    fn camera(&self, camera: usize) -> PyResult<&CameraKeypoints> {
        self.cameras.get(camera).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "camera {camera} is past the end of a {}-camera rig",
                self.cameras.len()
            ))
        })
    }
}

#[pymethods]
impl FlowFrame {
    /// Frameset timestamp, nanoseconds.
    #[getter]
    fn t_ns(&self) -> i64 {
        self.t_ns
    }

    /// Cameras on the rig.
    #[getter]
    fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// `optical_flow_detection_grid_size`: the occupancy cell's side in pixels.
    #[getter]
    fn cell_size(&self) -> usize {
        self.grid.cell
    }

    /// Top-left grid origin; split leftover pixels between the two image edges.
    #[getter]
    fn cell_origin(&self) -> (usize, usize) {
        (self.grid.x_start, self.grid.y_start)
    }

    /// One camera's keypoint ids, ascending: `int64[n]`.
    fn ids<'py>(&self, py: Python<'py>, camera: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(self.camera(camera)?.ids.to_pyarray(py))
    }

    /// One camera's keypoint positions in pixels: `float32[n, 2]`.
    ///
    /// Read out of the warps' translation column rather than stored beside them:
    /// they are the same two floats.
    fn positions<'py>(
        &self,
        py: Python<'py>,
        camera: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let keypoints: &CameraKeypoints = self.camera(camera)?;
        let positions: Vec<f32> = keypoints
            .transforms
            .as_chunks::<6>()
            .0
            .iter()
            .flat_map(|warp| [warp[2], warp[5]])
            .collect();
        positions.to_pyarray(py).reshape((keypoints.ids.len(), 2))
    }

    /// One camera's 2x3 warps, `[[m00, m01, tx], [m10, m11, ty]]`: `float32[n, 2, 3]`.
    fn transforms<'py>(
        &self,
        py: Python<'py>,
        camera: usize,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let keypoints: &CameraKeypoints = self.camera(camera)?;
        keypoints
            .transforms
            .to_pyarray(py)
            .reshape((keypoints.ids.len(), 2, 3))
    }

    /// Occupancy counts with camera 0's grid shape, regardless of this camera's resolution.
    fn occupancy<'py>(
        &self,
        py: Python<'py>,
        camera: usize,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let keypoints: &CameraKeypoints = self.camera(camera)?;
        keypoints
            .occupancy
            .to_pyarray(py)
            .reshape((self.grid.rows, self.grid.columns))
    }

    /// Ids one camera gained on this frameset: detections plus stereo matches.
    fn num_new(&self, camera: usize) -> PyResult<usize> {
        Ok(self.camera(camera)?.num_new)
    }

    /// Keypoints one camera carries.
    fn num_tracks(&self, camera: usize) -> PyResult<usize> {
        Ok(self.camera(camera)?.ids.len())
    }

    fn __repr__(&self) -> String {
        let counts: Vec<usize> = self.cameras.iter().map(|camera| camera.ids.len()).collect();
        format!("FlowFrame(t_ns={}, keypoints={counts:?})", self.t_ns)
    }
}

/// Frame-to-frame optical flow, driven one frameset at a time.
/// Only Pattern51 is supported; other configured patterns are refused.
#[pyclass(module = "slam_rs._core")]
pub struct OpticalFlow {
    inner: FrontendLane,
    /// The widened frameset, reused so a steady stream never reallocates.
    images: Vec<Image<u16, 1>>,
}

#[pymethods]
impl OpticalFlow {
    /// Build a frontend from calibration and configuration objects loaded through their JSON APIs.
    #[new]
    #[pyo3(signature = (calibration, config, *, threads = 1, max_keypoints = None))]
    fn new(
        calibration: PyRef<'_, Calibration>,
        config: PyRef<'_, VioConfig>,
        threads: usize,
        max_keypoints: Option<usize>,
    ) -> PyResult<Self> {
        // The standalone frontend is the CPU lane: the GPU choice belongs on
        // `Vio`, which is what a caller runs a whole pipeline through.
        let inner: FrontendLane = FrontendLane::Cpu(
            FrameToFrameOpticalFlow::new(
                config.inner.clone(),
                &calibration.inner,
                frontend_options(threads, max_keypoints),
            )
            .map_err(value_error)?,
        );
        let cameras: usize = inner.camera_count();
        Ok(Self {
            inner,
            images: vec![slam_rs::image::empty(); cameras],
        })
    }

    /// Cameras this frontend expects in every frameset.
    #[getter]
    fn camera_count(&self) -> usize {
        self.inner.camera_count()
    }

    /// The next keypoint id that will be handed out.
    #[getter]
    fn last_keypoint_id(&self) -> u64 {
        self.inner.last_keypoint_id()
    }

    /// Timestamp of the last accepted frameset, or None before the first.
    ///
    /// The core's own clock, not a copy of it: the two cannot drift apart when a
    /// frameset is refused.
    #[getter]
    fn t_ns(&self) -> Option<i64> {
        self.inner.t_ns()
    }

    /// Track and detect on one frameset of `camera_count` `uint8[h, w]` images.
    ///
    /// The images are widened into the frontend's own buffers while the GIL is
    /// held and the whole tracking pass then runs without it, so a decoder thread
    /// keeps running. A frameset the frontend refuses leaves it exactly as the
    /// last accepted one did.
    fn process(
        &mut self,
        py: Python<'_>,
        t_ns: i64,
        images: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<FlowFrame> {
        // Refuse the frameset's width before widening anything: the core would
        // refuse it too, but only after every image had been copied in. The
        // buffer is sized once, at construction, so nothing here resizes it.
        let cameras: usize = self.inner.camera_count();
        if images.len() != cameras {
            return Err(value_error(FrontendError::CameraCountMismatch {
                expected: cameras,
                actual: images.len(),
            }));
        }
        // Widen here, where the GIL is still held: the numpy borrow cannot
        // outlive it. `fill_from_u8_strided` reuses the buffer whenever the
        // geometry is unchanged, which is every frame after the first.
        for (index, image) in images.iter().enumerate() {
            let readonly: PyReadonlyArray2<'_, u8> = gray_array(image, index, "image")?;
            let (pixels, width, height) = gray_pixels(&readonly, index, "image")?;
            slam_rs::image::fill_from_u8_strided(
                &mut self.images[index],
                pixels,
                width,
                height,
                width,
            )
            .map_err(|error| PyValueError::new_err(format!("image {index}: {error}")))?;
        }

        let Self { inner, images } = self;
        py.detach(|| -> Result<FlowFrame, ProcessError> {
            inner
                .process_frame(t_ns, images, &PosePrediction::default(), &[])
                .map_err(ProcessError::Frontend)?;
            flow_frame(inner, t_ns)
        })
        .map_err(PyErr::from)
    }

    fn __repr__(&self) -> String {
        format!(
            "OpticalFlow(camera_count={}, frame_counter={})",
            self.inner.camera_count(),
            self.inner.frame_counter()
        )
    }
}

/// What a `process` call can refuse once the GIL is already released.
enum ProcessError {
    /// The frontend itself refused the frameset.
    Frontend(FrontendError),
    /// A keypoint id outgrew the `int64` the boundary hands to numpy.
    IdOverflow(u64),
}

impl From<ProcessError> for PyErr {
    fn from(error: ProcessError) -> Self {
        match error {
            ProcessError::Frontend(inner) => value_error(inner),
            ProcessError::IdOverflow(id) => {
                PyValueError::new_err(format!("keypoint id {id} does not fit in an int64"))
            }
        }
    }
}

/// Copy the frontend's committed frame into an owned [`FlowFrame`].
///
/// A pure read: `num_new` comes from the frontend's own watermark
/// ([`FrameToFrameOpticalFlow::last_keypoint_id_before_frame`]), so this answers
/// the same thing inside the `process` that produced the frame and afterwards
/// through [`Vio::flow_frame`].
///
/// `t_ns` is the timestamp `process_frame` has just accepted. The core carries
/// its own as an `Option` — `None` until the first frameset commits — and this
/// runs only after a commit, so taking the value the caller passed keeps the
/// Python-facing `FlowFrame.t_ns` a plain `int` with no impossible branch.
fn flow_frame(flow: &FrontendLane, t_ns: i64) -> Result<FlowFrame, ProcessError> {
    let previous_last_id: u64 = flow.last_keypoint_id_before_frame();
    let grid: CellGrid = flow.occupancy_grid();
    let frame: &CoreFlowFrame = flow.frame();
    let mut cameras: Vec<CameraKeypoints> = Vec::with_capacity(frame.cameras.len());
    for (index, keypoints) in frame.cameras.iter().enumerate() {
        let count: usize = keypoints.len();
        let mut ids: Vec<i64> = Vec::with_capacity(count);
        let mut transforms: Vec<f32> = Vec::with_capacity(6 * count);
        let mut num_new: usize = 0;
        for (slot, id) in keypoints.ids.iter().enumerate() {
            ids.push(i64::try_from(id.0).map_err(|_| ProcessError::IdOverflow(id.0))?);
            // Ids are handed out in one ascending run, so everything at or above
            // the counter's value before the call is new on this frameset.
            if id.0 >= previous_last_id {
                num_new += 1;
            }
            // `[m00, m01, m10, m11, tx, ty]`, into row-major 2x3.
            let warp: [f32; 6] = keypoints.transforms.coefficients(slot);
            transforms.extend_from_slice(&[warp[0], warp[1], warp[4], warp[2], warp[3], warp[5]]);
        }
        cameras.push(CameraKeypoints {
            ids,
            transforms,
            occupancy: flow.cell_counts(index).to_vec(),
            num_new,
        });
    }
    Ok(FlowFrame {
        t_ns,
        grid,
        cameras,
    })
}

/// One `slam_rs.catalog_feed.CameraCalib`, read attribute by attribute.
fn camera_parts(object: &Bound<'_, PyAny>, index: usize) -> PyResult<CoreCameraParts<f64>> {
    let what: String = format!("camera {index}");
    let rows: Vec<Vec<f64>> = extract_attribute(object, "imu_T_cam", &what, "a 4x4 float matrix")?;
    if rows.len() != 4 || rows.iter().any(|row| row.len() != 4) {
        return Err(wrong_type(&what, "imu_T_cam", "a 4x4 float matrix"));
    }
    let mut imu_t_cam_row_major: [f64; 16] = [0.0; 16];
    for (row, values) in rows.iter().enumerate() {
        imu_t_cam_row_major[4 * row..4 * row + 4].copy_from_slice(values);
    }
    Ok(CoreCameraParts {
        width: extract_attribute(object, "width", &what, "a non-negative integer")?,
        height: extract_attribute(object, "height", &what, "a non-negative integer")?,
        fx: extract_attribute(object, "fx", &what, "a float")?,
        fy: extract_attribute(object, "fy", &what, "a float")?,
        cx: extract_attribute(object, "cx", &what, "a float")?,
        cy: extract_attribute(object, "cy", &what, "a float")?,
        model: extract_attribute(object, "model", &what, "a string")?,
        distortion: extract_attribute(object, "distortion", &what, "a float sequence")?,
        distortion_valid_radius: extract_attribute(
            object,
            "distortion_valid_radius",
            &what,
            "a float or None",
        )?,
        readout_s: extract_attribute(object, "readout_s", &what, "a float or None")?,
        imu_t_cam_row_major,
    })
}

/// Read catalog IMU fields. `imu_T_body` is unused because the rig frame is the IMU frame.
fn imu_parts(object: &Bound<'_, PyAny>) -> PyResult<CoreImuParts<f64>> {
    let what: &str = "imu";
    Ok(CoreImuParts {
        frequency_hz: extract_attribute(object, "frequency_hz", what, "a float")?,
        gyro_noise_std: extract_attribute(object, "gyro_noise_std", what, "a float")?,
        accel_noise_std: extract_attribute(object, "accel_noise_std", what, "a float")?,
        gyro_bias_std: extract_attribute(object, "gyro_bias_std", what, "a float")?,
        accel_bias_std: extract_attribute(object, "accel_bias_std", what, "a float")?,
        cam_time_offset_ns: extract_attribute(object, "cam_time_offset_ns", what, "an integer")?,
        gravity_ms2: extract_attribute(object, "gravity_ms2", what, "a float or None")?,
    })
}

/// One attribute of a calibration dataclass, named in the error when it is absent.
fn attribute<'py>(
    object: &Bound<'py, PyAny>,
    name: &str,
    what: &str,
) -> PyResult<Bound<'py, PyAny>> {
    object
        .getattr(name)
        .map_err(|_| PyValueError::new_err(format!("{what}: no attribute {name:?}")))
}

/// One attribute of a calibration dataclass, as the type the field it feeds asks for.
///
/// `expected` is how that type reads in the refusal: the caller names it, because
/// `u32` is "a non-negative integer" to a caller who typed a pixel count and
/// `Option<f64>` is "a float or None".
fn extract_attribute<'py, T: FromPyObjectOwned<'py>>(
    object: &Bound<'py, PyAny>,
    name: &str,
    what: &str,
    expected: &str,
) -> PyResult<T> {
    attribute(object, name, what)?
        .extract()
        .map_err(|_| wrong_type(what, name, expected))
}

/// The one shape every calibration-attribute error takes.
fn wrong_type(what: &str, name: &str, expected: &str) -> PyErr {
    PyValueError::new_err(format!("{what}.{name} must be {expected}"))
}

/// Resolve a backend request (`"auto"`, `"cpu"` or `"gpu"`) to the lane `Vio`
/// would run, `"cpu"` or `"gpu"`, through `slam_rs::select_backend`.
///
/// Auto never raises: it returns `"cpu"` and warns once on stderr when the GPU
/// frontend cannot run here. `"gpu"` raises `ValueError` in that case.
#[pyfunction]
fn select_backend(request: &str) -> PyResult<&'static str> {
    let request: BackendRequest = request.parse().map_err(PyValueError::new_err)?;
    // The frontend's lane only: a CPU GBA request never runs the back-end probe.
    Ok(slam_rs::select_backend(request, GbaRequest::Cpu)
        .map_err(value_error)?
        .frontend
        .name())
}

/// How many device probes this process has run: capability validations plus
/// GBA round trips.
///
/// Each runs once per process, so this is 0 before the first `auto` or `gpu`
/// request and stays at its first value after it. The unit test tier asserts
/// it stays at 0.
#[pyfunction]
fn device_probes_run() -> usize {
    slam_rs::device_probes_run() + slam_rs::gba_probes_run()
}

/// `(t_ns, [tx, ty, tz, qx, qy, qz, qw])` rows from parallel timestamp and pose arrays.
fn stamped_rows(
    t_ns: &[i64],
    poses: &PyReadonlyArray2<'_, f64>,
    what: &str,
) -> PyResult<Vec<(i64, [f64; 7])>> {
    let poses = poses.as_array();
    if poses.ncols() != 7 || poses.nrows() != t_ns.len() {
        return Err(PyValueError::new_err(format!(
            "{what}: {} timestamps need {} x 7 poses, got {:?}",
            t_ns.len(),
            t_ns.len(),
            poses.shape()
        )));
    }
    Ok(t_ns
        .iter()
        .zip(poses.rows())
        .map(|(&t_ns, row)| (t_ns, std::array::from_fn(|i| row[i])))
        .collect())
}

/// The final trajectory: every raw frame pose re-anchored on the optimized
/// keyframes on both sides (`slam_rs::backend::reanchor_arrays`, PR f1).
#[pyfunction]
fn reanchor<'py>(
    py: Python<'py>,
    frame_t_ns: numpy::PyReadonlyArray1<'_, i64>,
    frame_poses: PyReadonlyArray2<'_, f64>,
    keyframe_t_ns: numpy::PyReadonlyArray1<'_, i64>,
    keyframe_poses: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let frames = stamped_rows(frame_t_ns.as_slice()?, &frame_poses, "frames")?;
    let keyframes = stamped_rows(keyframe_t_ns.as_slice()?, &keyframe_poses, "keyframes")?;
    let keyframe_poses: Vec<slam_rs::lie::Se3<f64>> = keyframes
        .iter()
        .map(|(t_ns, pose)| {
            slam_rs::pose_from_xyzw(pose).ok_or_else(|| {
                PyValueError::new_err(format!("keyframes: no unit quaternion at {t_ns} ns"))
            })
        })
        .collect::<PyResult<_>>()?;
    let keyframe_t_ns: Vec<i64> = keyframes.iter().map(|&(t_ns, _)| t_ns).collect();
    let anchored = slam_rs::backend::reanchor_arrays(&frames, &keyframe_t_ns, &keyframe_poses)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    numpy::ndarray::Array2::from_shape_vec((anchored.len(), 7), anchored.concat())
        .map(|array| array.to_pyarray(py))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// The compiled core of the `slam_rs` package.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    catalog::register(module)?;
    module.add("__version__", slam_rs::VERSION)?;
    // Report the wgpu runtime, or None for a CPU-only build.
    module.add("gpu_backend", slam_rs::GPU_BACKEND)?;
    module.add_function(wrap_pyfunction!(select_backend, module)?)?;
    module.add_function(wrap_pyfunction!(device_probes_run, module)?)?;
    module.add_function(wrap_pyfunction!(reanchor, module)?)?;
    module.add_class::<Vio>()?;
    module.add_class::<VioResult>()?;
    module.add_class::<bootstrap::BootstrapAttempt>()?;
    module.add_class::<bootstrap::BootstrapReport>()?;
    module.add_class::<VioStatus>()?;
    module.add_class::<Calibration>()?;
    module.add_class::<VioConfig>()?;
    module.add_class::<OpticalFlow>()?;
    module.add_class::<FlowFrame>()?;
    module.add_class::<VioSnapshot>()?;
    module.add_class::<map::MapSnapshot>()?;
    module.add_class::<map::MapDelta>()?;
    module.add_class::<map::LoopCounts>()?;
    module.add_class::<gba::MapResult>()?;
    module.add_class::<gba::GbaRun>()?;
    Ok(())
}
