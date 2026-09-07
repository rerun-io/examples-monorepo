//! Python bindings for the slam-rs core, exposed as `slam_rs._core`.
//!
//! Every array crossing the boundary is validated here and copied into owned
//! Rust storage, so the GIL is released for the whole core call and a producer
//! thread can keep decoding. A bad dtype, rank, shape or memory layout raises
//! `ValueError`, never a panic.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
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

/// Copy one C-contiguous `uint8[h, w]` array out of Python.
fn gray_image(object: &Bound<'_, PyAny>, index: usize) -> PyResult<GrayImage> {
    let array: &Bound<'_, PyArray2<u8>> = object.cast::<PyArray2<u8>>().map_err(|_| {
        PyValueError::new_err(format!("image {index} must be a 2-D uint8 numpy array"))
    })?;
    // as_slice() alone accepts Fortran order, whose bytes are transposed with
    // respect to the row-major copy below, so check the C flag explicitly.
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "image {index} must be C-contiguous; pass numpy.ascontiguousarray(image)"
        )));
    }
    let shape: Vec<usize> = array.shape().to_vec();
    let readonly = array.readonly();
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

/// The compiled core of the `slam_rs` package.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", slam_rs::VERSION)?;
    module.add_class::<Vio>()?;
    module.add_class::<VioResult>()?;
    module.add_class::<VioStatus>()?;
    Ok(())
}
