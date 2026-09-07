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
    CalibError, Calibration as CoreCalibration, CameraParts as CoreCameraParts,
    ImuParts as CoreImuParts,
};
use slam_rs::config::VioConfig as CoreVioConfig;
use slam_rs::frontend::detect::CellGrid;
use slam_rs::frontend::flow::{
    FlowFrame as CoreFlowFrame, FrameToFrameOpticalFlow, FrontendError, FrontendOptions,
    PosePrediction,
};
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::image::ImageU16;
use slam_rs::{Config, ImageView, VioError};

/// Map a core error onto `ValueError`.
fn to_py_err(error: VioError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// How far the estimator has got.
#[pyclass(module = "slam_rs._core", eq, eq_int, skip_from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VioStatus {
    /// No pose yet: the estimator has not initialised.
    NotInitialised,
    /// The frame arrived before the IMU samples that cover it.
    NeedMoreImu,
    /// The returned pose is an estimate.
    Tracking,
}

impl From<slam_rs::VioStatus> for VioStatus {
    fn from(status: slam_rs::VioStatus) -> Self {
        match status {
            slam_rs::VioStatus::NotInitialised => Self::NotInitialised,
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

/// The estimator, driven one frameset at a time.
#[pyclass(module = "slam_rs._core")]
pub struct Vio {
    inner: slam_rs::Vio,
}

#[pymethods]
impl Vio {
    /// Build an estimator for a rig of `camera_count` cameras.
    #[new]
    #[pyo3(signature = (camera_count = 2, min_imu_samples = 1))]
    fn new(camera_count: usize, min_imu_samples: usize) -> PyResult<Self> {
        if camera_count == 0 {
            return Err(PyValueError::new_err("camera_count must be at least 1"));
        }
        Ok(Self {
            inner: slam_rs::Vio::new(Config {
                camera_count,
                min_imu_samples,
            }),
        })
    }

    /// Number of cameras this estimator expects in every frameset.
    #[getter]
    fn camera_count(&self) -> usize {
        self.inner.config().camera_count
    }

    /// Add one IMU sample: `gyro` in rad/s, `accel` in m/s², both in the rig frame.
    fn push_imu(&mut self, t_ns: i64, gyro: [f64; 3], accel: [f64; 3]) -> PyResult<()> {
        self.inner.push_imu(t_ns, gyro, accel).map_err(to_py_err)
    }

    /// Add `n` IMU samples at once: `t_ns` is `int64[n]`, `gyro` and `accel` are `float64[n, 3]`.
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
            for ((&t, &gyro_sample), &accel_sample) in
                times.iter().zip(gyro_rows.iter()).zip(accel_rows.iter())
            {
                self.inner.push_imu(t, gyro_sample, accel_sample)?;
            }
            Ok::<(), VioError>(())
        })
        .map_err(to_py_err)
    }

    /// Process one frameset: `images` holds one `uint8[h, w]` array per camera.
    fn track(
        &mut self,
        py: Python<'_>,
        t_ns: i64,
        images: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<VioResult> {
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
            .map_err(to_py_err)?;
        Ok(VioResult { inner: result })
    }
}

/// One grayscale frame copied out of Python, packed (stride == width).
struct GrayImage {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

/// Borrow one `uint8[h, w]` array out of Python: rank, dtype and layout checked.
///
/// The two consumers differ only in what they do with the bytes — [`Vio::track`]
/// copies them into a `Vec`, [`OpticalFlow::process`] widens them straight into a
/// reused [`ImageU16`] — so the checks live here and neither repeats them.
fn gray_array<'py>(
    object: &Bound<'py, PyAny>,
    index: usize,
) -> PyResult<PyReadonlyArray2<'py, u8>> {
    let array: &Bound<'py, PyArray2<u8>> = object.cast::<PyArray2<u8>>().map_err(|_| {
        PyValueError::new_err(format!("image {index} must be a 2-D uint8 numpy array"))
    })?;
    // as_slice() alone accepts Fortran order, whose bytes are transposed with
    // respect to the row-major copies below, so check the C flag explicitly.
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "image {index} must be C-contiguous; pass numpy.ascontiguousarray(image)"
        )));
    }
    Ok(array.readonly())
}

/// Copy one C-contiguous `uint8[h, w]` array out of Python.
fn gray_image(object: &Bound<'_, PyAny>, index: usize) -> PyResult<GrayImage> {
    let readonly: PyReadonlyArray2<'_, u8> = gray_array(object, index)?;
    let shape: Vec<usize> = readonly.shape().to_vec();
    let pixels: &[u8] = readonly.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "image {index} must be C-contiguous; pass numpy.ascontiguousarray(image)"
        ))
    })?;
    Ok(GrayImage {
        width: shape[1],
        height: shape[0],
        pixels: pixels.to_vec(),
    })
}

/// Copy a C-contiguous `int64[n]` array out of Python.
fn int64_column(object: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    let array: &Bound<'_, PyArray1<i64>> = object
        .cast::<PyArray1<i64>>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a 1-D int64 numpy array")))?;
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous; pass numpy.ascontiguousarray({name})"
        )));
    }
    let readonly = array.readonly();
    let values: &[i64] = readonly.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be C-contiguous; pass numpy.ascontiguousarray({name})"
        ))
    })?;
    Ok(values.to_vec())
}

/// Copy a C-contiguous `float64[n, 3]` array out of Python, row by row.
fn float64_triples(object: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<[f64; 3]>> {
    let array: &Bound<'_, PyArray2<f64>> = object
        .cast::<PyArray2<f64>>()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a 2-D float64 numpy array")))?;
    let shape: Vec<usize> = array.shape().to_vec();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n, 3), got {shape:?}"
        )));
    }
    // Fortran order passes as_slice() but its bytes run down the columns, which
    // would turn the chunks below into transposed samples.
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous; pass numpy.ascontiguousarray({name})"
        )));
    }
    let readonly = array.readonly();
    let values: &[f64] = readonly.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be C-contiguous; pass numpy.ascontiguousarray({name})"
        ))
    })?;
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
            inner: CoreVioConfig::from_json_str(text)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Write the config back in basalt's shape, `value0` wrapper and all.
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json_string()
            .map_err(|error| PyValueError::new_err(error.to_string()))
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
            inner: CoreCalibration::<f64>::from_json_str(text).map_err(calib_error)?,
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
                .map_err(calib_error)?,
        })
    }

    /// Write the calibration back in basalt's shape, `value0` wrapper and all.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(calib_error)
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
    /// `[x, y]` per keypoint, keypoint index slow-varying.
    positions: Vec<f32>,
    /// `[m00, m01, tx, m10, m11, ty]` per keypoint, keypoint index slow-varying.
    transforms: Vec<f32>,
    /// Detector response per keypoint, `-1` where basalt records none.
    responses: Vec<f32>,
    /// `OpticalFlowResult::pyramid_levels`, empty for this flow type.
    levels: Vec<u32>,
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
    fn positions<'py>(
        &self,
        py: Python<'py>,
        camera: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let keypoints: &CameraKeypoints = self.camera(camera)?;
        keypoints
            .positions
            .to_pyarray(py)
            .reshape((keypoints.ids.len(), 2))
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

    /// One camera's detector responses: `float32[n]`, `-1` where basalt records none.
    fn responses<'py>(
        &self,
        py: Python<'py>,
        camera: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(self.camera(camera)?.responses.to_pyarray(py))
    }

    /// `OpticalFlowResult::pyramid_levels`: `uint32[0]` for `frame_to_frame`.
    ///
    /// basalt only fills this in the multiscale variant, which is not ported, so
    /// the array is empty rather than one entry per keypoint. It is exposed
    /// because the core carries it, not because it holds anything today.
    fn levels<'py>(&self, py: Python<'py>, camera: usize) -> PyResult<Bound<'py, PyArray1<u32>>> {
        Ok(self.camera(camera)?.levels.to_pyarray(py))
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
    inner: FrameToFrameOpticalFlow<Pattern51>,
    /// The widened frameset, reused so a steady stream never reallocates.
    images: Vec<ImageU16>,
    /// Timestamp of the last accepted frameset; the clock must move forward.
    last_t_ns: Option<i64>,
}

#[pymethods]
impl OpticalFlow {
    /// Build a frontend for one rig.
    ///
    /// `calibration` is a [`Calibration`] or one of basalt's calibration files as
    /// text; `config` is a [`VioConfig`] or one of its config files as text.
    #[new]
    #[pyo3(signature = (calibration, config, *, threads = 1, epipolar_per_camera = true, max_keypoints = None))]
    fn new(
        calibration: &Bound<'_, PyAny>,
        config: &Bound<'_, PyAny>,
        threads: usize,
        epipolar_per_camera: bool,
        max_keypoints: Option<usize>,
    ) -> PyResult<Self> {
        if threads == 0 {
            return Err(PyValueError::new_err("threads must be at least 1"));
        }
        let defaults: FrontendOptions = FrontendOptions::default();
        let options: FrontendOptions = FrontendOptions {
            epipolar_per_camera,
            threads,
            max_keypoints: max_keypoints.unwrap_or(defaults.max_keypoints),
        };
        let inner: FrameToFrameOpticalFlow<Pattern51> = FrameToFrameOpticalFlow::new(
            config_argument(config)?,
            &calibration_argument(calibration)?,
            options,
        )
        .map_err(frontend_error)?;
        let cameras: usize = inner.camera_count();
        Ok(Self {
            inner,
            images: vec![ImageU16::default(); cameras],
            last_t_ns: None,
        })
    }

    /// Cameras this frontend expects in every frameset.
    #[getter]
    fn camera_count(&self) -> usize {
        self.inner.camera_count()
    }

    /// Framesets accepted so far.
    #[getter]
    fn frame_counter(&self) -> u64 {
        self.inner.frame_counter()
    }

    /// The next keypoint id that will be handed out.
    #[getter]
    fn last_keypoint_id(&self) -> u64 {
        self.inner.last_keypoint_id()
    }

    /// Timestamp of the last accepted frameset, or None before the first.
    #[getter]
    fn t_ns(&self) -> Option<i64> {
        self.last_t_ns
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
        if let Some(last) = self.last_t_ns
            && t_ns <= last
        {
            return Err(PyValueError::new_err(format!(
                "frameset timestamps must increase: got {t_ns} after {last}"
            )));
        }
        // Widen here, where the GIL is still held: the numpy borrow cannot
        // outlive it. `fill_from_u8_strided` reuses the buffer whenever the
        // geometry is unchanged, which is every frame after the first.
        self.images.resize_with(images.len(), ImageU16::default);
        for (index, image) in images.iter().enumerate() {
            let readonly: PyReadonlyArray2<'_, u8> = gray_array(image, index)?;
            let shape: Vec<usize> = readonly.shape().to_vec();
            let pixels: &[u8] = readonly.as_slice().map_err(|_| {
                PyValueError::new_err(format!(
                    "image {index} must be C-contiguous; pass numpy.ascontiguousarray(image)"
                ))
            })?;
            self.images[index]
                .fill_from_u8_strided(pixels, shape[1], shape[0], shape[1])
                .map_err(|error| PyValueError::new_err(format!("image {index}: {error}")))?;
        }

        let previous_last_id: u64 = self.inner.last_keypoint_id();
        let Self { inner, images, .. } = self;
        let frame: FlowFrame = py.detach(|| -> Result<FlowFrame, ProcessError> {
            inner
                .process_frame(t_ns, images, &PosePrediction::default(), &[])
                .map_err(ProcessError::Frontend)?;
            flow_frame(inner, previous_last_id)
        })?;
        self.last_t_ns = Some(t_ns);
        Ok(frame)
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
            ProcessError::Frontend(inner) => frontend_error(inner),
            ProcessError::IdOverflow(id) => {
                PyValueError::new_err(format!("keypoint id {id} does not fit in an int64"))
            }
        }
    }
}

/// Copy the frontend's committed frame into an owned [`FlowFrame`].
fn flow_frame(
    flow: &FrameToFrameOpticalFlow<Pattern51>,
    previous_last_id: u64,
) -> Result<FlowFrame, ProcessError> {
    let grid: CellGrid = flow.occupancy_grid();
    let frame: &CoreFlowFrame = flow.frame();
    let mut cameras: Vec<CameraKeypoints> = Vec::with_capacity(frame.cameras.len());
    for (index, keypoints) in frame.cameras.iter().enumerate() {
        let count: usize = keypoints.len();
        let mut ids: Vec<i64> = Vec::with_capacity(count);
        let mut positions: Vec<f32> = Vec::with_capacity(2 * count);
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
            positions.extend_from_slice(&[warp[4], warp[5]]);
            transforms.extend_from_slice(&[warp[0], warp[1], warp[4], warp[2], warp[3], warp[5]]);
        }
        cameras.push(CameraKeypoints {
            ids,
            positions,
            transforms,
            responses: keypoints.responses.clone(),
            levels: keypoints.pyramid_levels.clone(),
            occupancy: flow.cell_counts(index).to_vec(),
            num_new,
        });
    }
    Ok(FlowFrame {
        t_ns: frame.t_ns,
        grid,
        cameras,
    })
}

/// A [`Calibration`] or one of basalt's calibration files as text.
fn calibration_argument(object: &Bound<'_, PyAny>) -> PyResult<CoreCalibration<f64>> {
    if let Ok(calibration) = object.extract::<PyRef<'_, Calibration>>() {
        return Ok(calibration.inner.clone());
    }
    let text: String = object.extract().map_err(|_| {
        PyValueError::new_err(
            "calibration must be a Calibration or a basalt calibration JSON string",
        )
    })?;
    CoreCalibration::<f64>::from_json_str(&text).map_err(calib_error)
}

/// A [`VioConfig`] or one of basalt's config files as text.
fn config_argument(object: &Bound<'_, PyAny>) -> PyResult<CoreVioConfig> {
    if let Ok(config) = object.extract::<PyRef<'_, VioConfig>>() {
        return Ok(config.inner.clone());
    }
    let text: String = object.extract().map_err(|_| {
        PyValueError::new_err("config must be a VioConfig or a basalt config JSON string")
    })?;
    CoreVioConfig::from_json_str(&text).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// One `slam_rs.catalog_feed.CameraCalib`, read attribute by attribute.
fn camera_parts(object: &Bound<'_, PyAny>, index: usize) -> PyResult<CoreCameraParts<f64>> {
    let what: String = format!("camera {index}");
    let rows: Vec<Vec<f64>> = attribute(object, "imu_T_cam", &what)?
        .extract()
        .map_err(|_| wrong_type(&what, "imu_T_cam", "a 4x4 float matrix"))?;
    if rows.len() != 4 || rows.iter().any(|row| row.len() != 4) {
        return Err(wrong_type(&what, "imu_T_cam", "a 4x4 float matrix"));
    }
    let mut imu_t_cam_row_major: [f64; 16] = [0.0; 16];
    for (row, values) in rows.iter().enumerate() {
        imu_t_cam_row_major[4 * row..4 * row + 4].copy_from_slice(values);
    }
    Ok(CoreCameraParts {
        width: size_attribute(object, "width", &what)?,
        height: size_attribute(object, "height", &what)?,
        fx: float_attribute(object, "fx", &what)?,
        fy: float_attribute(object, "fy", &what)?,
        cx: float_attribute(object, "cx", &what)?,
        cy: float_attribute(object, "cy", &what)?,
        model: attribute(object, "model", &what)?
            .extract()
            .map_err(|_| wrong_type(&what, "model", "a string"))?,
        distortion: attribute(object, "distortion", &what)?
            .extract()
            .map_err(|_| wrong_type(&what, "distortion", "a float sequence"))?,
        distortion_valid_radius: attribute(object, "distortion_valid_radius", &what)?
            .extract()
            .map_err(|_| wrong_type(&what, "distortion_valid_radius", "a float or None"))?,
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
        frequency_hz: float_attribute(object, "frequency_hz", what)?,
        gyro_noise_std: float_attribute(object, "gyro_noise_std", what)?,
        accel_noise_std: float_attribute(object, "accel_noise_std", what)?,
        gyro_bias_std: float_attribute(object, "gyro_bias_std", what)?,
        accel_bias_std: float_attribute(object, "accel_bias_std", what)?,
        cam_time_offset_ns: attribute(object, "cam_time_offset_ns", what)?
            .extract()
            .map_err(|_| wrong_type(what, "cam_time_offset_ns", "an integer"))?,
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

/// One `float` attribute.
fn float_attribute(object: &Bound<'_, PyAny>, name: &str, what: &str) -> PyResult<f64> {
    attribute(object, name, what)?
        .extract()
        .map_err(|_| wrong_type(what, name, "a float"))
}

/// One pixel-count attribute, which must be a non-negative `int`.
fn size_attribute(object: &Bound<'_, PyAny>, name: &str, what: &str) -> PyResult<u32> {
    attribute(object, name, what)?
        .extract()
        .map_err(|_| wrong_type(what, name, "a non-negative integer"))
}

/// The one shape every calibration-attribute error takes.
fn wrong_type(what: &str, name: &str, expected: &str) -> PyErr {
    PyValueError::new_err(format!("{what}.{name} must be {expected}"))
}

/// Map a calibration error onto `ValueError`.
fn calib_error(error: CalibError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Map a frontend error onto `ValueError`.
fn frontend_error(error: FrontendError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// The compiled core of the `slam_rs` package.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", slam_rs::VERSION)?;
    module.add_class::<Vio>()?;
    module.add_class::<VioResult>()?;
    module.add_class::<VioStatus>()?;
    module.add_class::<Calibration>()?;
    module.add_class::<VioConfig>()?;
    module.add_class::<OpticalFlow>()?;
    module.add_class::<FlowFrame>()?;
    Ok(())
}
