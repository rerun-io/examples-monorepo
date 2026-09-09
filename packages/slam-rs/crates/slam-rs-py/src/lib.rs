//! Python bindings for the slam-rs core, exposed as `slam_rs._core`.
//!
//! Every array crossing the boundary is validated here and copied into owned
//! Rust storage, so the GIL is released for the whole core call and a producer
//! thread can keep decoding. A bad dtype, rank, shape or memory layout raises
//! `ValueError`, never a panic.

use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods,
    ToPyArray,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use slam_rs::calib::{
    Calibration as CoreCalibration, CameraParts as CoreCameraParts, ImuParts as CoreImuParts,
};
use slam_rs::config::VioConfig as CoreVioConfig;
use slam_rs::frontend::detect::CellGrid;
use slam_rs::frontend::flow::{
    FlowFrame as CoreFlowFrame, FrameToFrameOpticalFlow, FrontendError, FrontendOptions,
    PosePrediction,
};
use slam_rs::image::ImageU16;
use slam_rs::{Backend, FrontendLane, ImageView, VioError};

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
}

impl From<slam_rs::VioStatus> for VioStatus {
    fn from(status: slam_rs::VioStatus) -> Self {
        match status {
            slam_rs::VioStatus::NeedMoreImu => Self::NeedMoreImu,
            slam_rs::VioStatus::Tracking => Self::Tracking,
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

    /// Rig pose in the world frame as `[tx, ty, tz, qx, qy, qz, qw]`.
    #[getter]
    fn world_from_rig<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.world_from_rig.as_slice().to_pyarray(py)
    }

    /// Rig velocity in the world frame, m/s.
    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.velocity.as_slice().to_pyarray(py)
    }

    /// Gyroscope bias estimate, rad/s.
    #[getter]
    fn gyro_bias<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.gyro_bias.as_slice().to_pyarray(py)
    }

    /// Accelerometer bias estimate, m/s².
    #[getter]
    fn accel_bias<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.accel_bias.as_slice().to_pyarray(py)
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
/// Offline mode: the frontend and the backend run to completion in the calling
/// thread, so every `track` result is final and a repeat run over the same input
/// is bit-identical. The GIL is released around frontend and estimator compute,
/// so a decoder thread keeps running while the frameset is tracked.
#[pyclass(module = "slam_rs._core")]
pub struct Vio {
    inner: slam_rs::Vio<f32>,
}

#[pymethods]
impl Vio {
    /// Build the pipeline from basalt's own calibration and config.
    ///
    /// The two objects are the ones [`OpticalFlow`] takes: basalt's files arrive
    /// through [`Calibration::from_json`] and [`VioConfig::from_json`], and the
    /// catalog's own dataclasses through [`Calibration::from_catalog`]. The
    /// estimator runs in single precision, which is the precision every
    /// reference run was produced at (`use-double` false).
    /// `gpu` runs the frontend's pyramid, patch build and KLT tracker through
    /// CubeCL on this host's GPU instead of the CPU port (decision D21). The
    /// default is the CPU, which is what every accuracy reference was produced
    /// on; a build without the `gpu` cargo feature refuses `gpu=True` rather
    /// than ignoring it, and so does a build that has the feature on a host
    /// with no usable GPU — a missing driver library, a driver that will not
    /// initialise, no visible device, no adapter — each a `ValueError` naming
    /// what is absent rather than the `PanicException` CubeCL's own unwrapped
    /// bring-up produces (decision D32). A failure no probe anticipates is
    /// caught rather than raised, so it is a `ValueError` too — with the
    /// runtime's own panic message left on stderr, which is the only account of
    /// a case the probe did not know to ask about.
    ///
    /// `threads` is **inert on the GPU lane**: it reaches
    /// `FrontendOptions::threads`, which only `CpuPatchTracker::new` reads, and
    /// the GPU tracker holds no work pool. It is accepted rather than refused
    /// together with `gpu=True` so the same call site can select either lane.
    #[new]
    #[pyo3(signature = (calibration, config, *, threads = 1, max_keypoints = None, gpu = false))]
    fn new(
        calibration: PyRef<'_, Calibration>,
        config: PyRef<'_, VioConfig>,
        threads: usize,
        max_keypoints: Option<usize>,
        gpu: bool,
    ) -> PyResult<Self> {
        let backend: Backend = if gpu { Backend::Gpu } else { Backend::Cpu };
        Ok(Self {
            inner: slam_rs::Vio::with_backend(
                config.inner.clone(),
                calibration.inner.clone(),
                frontend_options(threads, max_keypoints),
                backend,
            )
            .map_err(value_error)?,
        })
    }

    /// Whether the frontend runs on the GPU.
    #[getter]
    fn gpu(&self) -> bool {
        self.inner.backend() == Backend::Gpu
    }

    /// Cameras this estimator expects in every frameset.
    ///
    /// Read off the frontend rather than mirrored, so it cannot drift from the
    /// calibration the rig was built with.
    #[getter]
    fn camera_count(&self) -> usize {
        self.inner.frontend().camera_count()
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
    fn track(
        &mut self,
        py: Python<'_>,
        t_ns: i64,
        images: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<VioResult> {
        if self.inner.backend() == slam_rs::Backend::Gpu {
            let arrays: Vec<PyReadonlyArray2<'_, u8>> = images
                .iter()
                .enumerate()
                .map(|(index, image)| {
                    let array = gray_array(image, index)?;
                    gray_pixels(&array, index)?;
                    Ok(array)
                })
                .collect::<PyResult<_>>()?;
            let views: Vec<ImageView<'_>> = arrays
                .iter()
                .enumerate()
                .map(|(index, array)| {
                    let (data, width, height) = gray_pixels(array, index)?;
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

    /// The window, its landmarks and the last measured frame's statistics.
    ///
    /// `None` until a frameset has been measured: before that the window is
    /// empty and there are no statistics to report. Everything is copied, so a
    /// snapshot stays valid across the next `track`.
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

    /// Wall time each stage took on the last frame, milliseconds.
    ///
    /// The estimator's six, and the frontend lane's four under a
    /// `frontend_` prefix: the pyramid build, the FAST detection, every KLT
    /// call, and the preintegration that seeds the KLT. The four are the same
    /// kind of measurement as the six and are read the same way, which is why
    /// they come back in one map; they do not add up to the frame, because the
    /// bookkeeping between the phases is nobody's stage.
    #[getter]
    fn timings_ms(&self) -> std::collections::BTreeMap<&'static str, f64> {
        [
            ("back_substitution", self.timings.back_substitution_ns),
            ("error", self.timings.error_ns),
            ("linearize", self.timings.linearize_ns),
            ("marginalize", self.timings.marginalize_ns),
            ("measure", self.timings.measure_ns),
            ("solver", self.timings.solver_ns),
            ("frontend_pyramid", self.frontend_timings.pyramid_ns),
            ("frontend_detect", self.frontend_timings.detect_ns),
            ("frontend_track", self.frontend_timings.track_ns),
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

/// The frontend options both entry points build, from the two knobs they expose.
///
/// Everything else in `FrontendOptions` is a property of the port rather than of
/// a run, so `None` means the default rather than "unset". `threads` is read by
/// `CpuPatchTracker::new` alone, so it does nothing on the GPU lane.
fn frontend_options(threads: usize, max_keypoints: Option<usize>) -> FrontendOptions {
    let defaults: FrontendOptions = FrontendOptions::default();
    FrontendOptions {
        threads,
        max_keypoints: max_keypoints.unwrap_or(defaults.max_keypoints),
        ..defaults
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

/// Borrow one `uint8[h, w]` array out of Python: rank and dtype checked.
///
/// Consumers either copy the bytes or widen them into reused [`ImageU16`]
/// storage. The layout is [`gray_pixels`]' half of the same pair.
fn gray_array<'py>(
    object: &Bound<'py, PyAny>,
    index: usize,
) -> PyResult<PyReadonlyArray2<'py, u8>> {
    let array: &Bound<'py, PyArray2<u8>> = object.cast::<PyArray2<u8>>().map_err(|_| {
        PyValueError::new_err(format!("image {index} must be a 2-D uint8 numpy array"))
    })?;
    Ok(array.readonly())
}

/// The pixels, width and height of an array [`gray_array`] has accepted.
fn gray_pixels<'a>(
    readonly: &'a PyReadonlyArray2<'_, u8>,
    index: usize,
) -> PyResult<(&'a [u8], usize, usize)> {
    let shape: &[usize] = readonly.shape();
    let pixels: &[u8] = contiguous(readonly, &format!("image {index}"), "image")?;
    Ok((pixels, shape[1], shape[0]))
}

/// Copy one C-contiguous `uint8[h, w]` array out of Python.
fn gray_image(object: &Bound<'_, PyAny>, index: usize) -> PyResult<GrayImage> {
    let readonly: PyReadonlyArray2<'_, u8> = gray_array(object, index)?;
    let (pixels, width, height) = gray_pixels(&readonly, index)?;
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
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

/// basalt's `VioConfig`, as `data/**/*_config.json` carries it.
///
/// A fresh instance is `VioConfig::VioConfig()`, the same defaults the C++
/// constructor sets; [`VioConfig::from_json`] then overwrites whatever keys a
/// file names, leaving the rest alone, as cereal does.
#[pyclass(module = "slam_rs._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct VioConfig {
    inner: CoreVioConfig,
}

#[pymethods]
impl VioConfig {
    /// basalt's own defaults (`src/utils/vio_config.cpp:47-128`).
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreVioConfig::default(),
        }
    }

    /// Read one of basalt's config files; keys it omits keep their default.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreVioConfig::from_json_str(text).map_err(value_error)?,
        })
    }

    /// Write the config back in basalt's shape, `value0` wrapper and all.
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

/// basalt's camera-IMU calibration: extrinsics, intrinsics and the noise model.
///
/// Either read from one of basalt's calibration files, or built from what the
/// catalog feed reports. The second path takes
/// `slam_rs.catalog_feed.CameraCalib` and `ImuCalib` field by field, so the
/// catalog-to-basalt rules — the rotation-matrix check, the model names, the
/// isotropic noise densities — stay in the Rust that is tested against basalt's
/// own JSON rather than being written a second time in Python.
#[pyclass(module = "slam_rs._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct Calibration {
    inner: CoreCalibration<f64>,
}

#[pymethods]
impl Calibration {
    /// Read one of basalt's calibration files.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCalibration::<f64>::from_json_str(text).map_err(value_error)?,
        })
    }

    /// Build the calibration from the feed's dataclasses.
    ///
    /// `cameras` are `CameraCalib` in rig order and `imu` is an `ImuCalib`; only
    /// the fields basalt models are read, so `ImuCalib.imu_T_body` is ignored —
    /// the rig reference frame *is* the IMU on every recording the feed reads.
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

    /// Write the calibration back in basalt's shape, `value0` wrapper and all.
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

/// What one [`OpticalFlow::process`] call produced.
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

    /// `(x_start, y_start)`: the top-left corner of cell `(0, 0)` in pixels.
    ///
    /// basalt centres the grid on the frame, so the leftover `width % cell` is
    /// split between the two edges (`keypoints.cpp:140-144`).
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
            .chunks_exact(6)
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

    /// One camera's occupancy counts: `int32[rows, columns]`.
    ///
    /// The grid is camera 0's, as basalt's is (`frame_to_frame_optical_flow.h:119`),
    /// whatever the camera's own resolution is.
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

/// basalt's `FrameToFrameOpticalFlow`, driven one frameset at a time.
///
/// Pattern 51 only: every shipped config sets `optical_flow_pattern = 51`, and a
/// config asking for another one is refused rather than silently tracked with
/// the wrong pattern.
#[pyclass(module = "slam_rs._core")]
pub struct OpticalFlow {
    inner: FrontendLane,
    /// The widened frameset, reused so a steady stream never reallocates.
    images: Vec<ImageU16>,
}

#[pymethods]
impl OpticalFlow {
    /// Build a frontend for one rig.
    ///
    /// One of basalt's own files arrives through [`Calibration::from_json`] and
    /// [`VioConfig::from_json`], so this takes the two classes only.
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
            images: vec![ImageU16::default(); cameras],
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
            let readonly: PyReadonlyArray2<'_, u8> = gray_array(image, index)?;
            let (pixels, width, height) = gray_pixels(&readonly, index)?;
            self.images[index]
                .fill_from_u8_strided(pixels, width, height, width)
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
        imu_t_cam_row_major,
    })
}

/// One `slam_rs.catalog_feed.ImuCalib`, read attribute by attribute.
///
/// `imu_T_body` is not read: basalt's calibration has no such field, because the
/// rig reference frame *is* the IMU on every recording the feed reads.
fn imu_parts(object: &Bound<'_, PyAny>) -> PyResult<CoreImuParts<f64>> {
    let what: &str = "imu";
    Ok(CoreImuParts {
        frequency_hz: extract_attribute(object, "frequency_hz", what, "a float")?,
        gyro_noise_std: extract_attribute(object, "gyro_noise_std", what, "a float")?,
        accel_noise_std: extract_attribute(object, "accel_noise_std", what, "a float")?,
        gyro_bias_std: extract_attribute(object, "gyro_bias_std", what, "a float")?,
        accel_bias_std: extract_attribute(object, "accel_bias_std", what, "a float")?,
        cam_time_offset_ns: extract_attribute(object, "cam_time_offset_ns", what, "an integer")?,
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

/// The compiled core of the `slam_rs` package.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", slam_rs::VERSION)?;
    // Report the wgpu runtime, or None for a CPU-only build.
    module.add("gpu_backend", slam_rs::GPU_BACKEND)?;
    module.add_class::<Vio>()?;
    module.add_class::<VioResult>()?;
    module.add_class::<VioStatus>()?;
    module.add_class::<Calibration>()?;
    module.add_class::<VioConfig>()?;
    module.add_class::<OpticalFlow>()?;
    module.add_class::<FlowFrame>()?;
    module.add_class::<VioSnapshot>()?;
    Ok(())
}
