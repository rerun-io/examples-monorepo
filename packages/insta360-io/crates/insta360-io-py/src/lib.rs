//! The `insta360_io` Python module: an Insta360 `.insv` clip reader (`kornia_staging_io::insta360`) and the CPU
//! stitch of its two lens images into the hemisphere tiles and the viewing panorama
//! (`kornia_staging_imgproc::panorama`).
//!
//! `MeiCamera` is a lens camera as a calibration states it; `PanoramaStitch` takes two of them and the pose of camera 1
//! in camera 0. Every refusal raises `ValueError`, never a panic.

mod camera;
mod insv;
mod panorama;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Map any core error onto `ValueError`, which is what every refusal here is.
fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn insta360_io(module: &Bound<'_, PyModule>) -> PyResult<()> {
    camera::register(module)?;
    insv::register(module)?;
    panorama::register(module)
}
