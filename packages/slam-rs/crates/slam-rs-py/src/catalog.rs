//! Python adapters for the catalog rules owned by slam-rs.
use crate::value_error;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};
use slam_rs::catalog_timing;

/// Return the feed's typed adapter for the core-owned rig facts.
#[pyfunction]
fn catalog_rig_profile<'py>(py: Python<'py>, dataset: &str) -> PyResult<Bound<'py, PyAny>> {
    let profile = match dataset {
        "robocap" => &catalog_timing::ROBOCAP,
        "msd-g2" => &catalog_timing::MSD_G2,
        _ => return Err(value_error(format!("unknown catalog rig {dataset}"))),
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("camera_names", PyTuple::new(py, profile.camera_names)?)?;
    kwargs.set_item("downscale", profile.downscale)?;
    kwargs.set_item("interpolate_accel_onto_gyro", profile.interpolate_accel)?;
    kwargs.set_item("frameset_tolerance_ns", profile.tolerance_ns)?;
    kwargs.set_item("video_time_is_absolute", profile.video_time_is_absolute)?;
    py.import("slam_rs.catalog_feed")?
        .getattr("RigProfile")?
        .call((), Some(&kwargs))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(catalog_rig_profile, module)?)?;
    Ok(())
}
