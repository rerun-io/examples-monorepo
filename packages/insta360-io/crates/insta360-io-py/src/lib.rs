//! `insta360_io._insta360_io`, the compiled half of the `insta360_io` Python package: the Insta360 `.insv` clip
//! (`kornia_staging_io::insta360`), the calibrated rig the reader yields Rerun chunks from (`insta360_io_core::rig`),
//! and the CPU stitch of the two lens images into the hemisphere tiles and the viewing panorama
//! (`kornia_staging_imgproc::panorama`). The package's Python half (`python/insta360_io`) holds `InsvReader`.
//!
//! `MeiCamera` is a lens camera as a calibration states it; `PanoramaStitch` takes two of them and the pose of camera 1
//! in camera 0. Every refusal raises `ValueError`, never a panic.

mod camera;
mod insv;
mod panorama;
mod rig;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Map any core error onto `ValueError`, which is what every refusal here is.
fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
#[pyo3(name = "_insta360_io")]
fn insta360_io(module: &Bound<'_, PyModule>) -> PyResult<()> {
    camera::register(module)?;
    insv::register(module)?;
    rig::register(module)?;
    panorama::register(module)
}
