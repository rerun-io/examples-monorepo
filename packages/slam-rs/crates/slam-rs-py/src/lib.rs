//! Python bindings for the slam-rs core, exposed as `slam_rs._core`.
//!
//! Every array crossing the boundary is validated here and copied into owned
//! Rust storage, so the GIL is released for the whole core call and a producer
//! thread can keep decoding. A bad dtype, rank, shape or memory layout raises
//! `ValueError`, never a panic.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use slam_rs::calib::Calibration as CoreCalibration;
use slam_rs::config::VioConfig as CoreVioConfig;

/// Map any core error onto `ValueError`, which is what every refusal here is.
fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
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

/// The compiled core of the `slam_rs` package.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", slam_rs::VERSION)?;
    module.add_class::<Calibration>()?;
    module.add_class::<VioConfig>()?;
    Ok(())
}
