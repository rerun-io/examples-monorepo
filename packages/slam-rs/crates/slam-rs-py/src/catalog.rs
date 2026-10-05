//! Python adapters for the catalog rules owned by slam-rs.
use crate::value_error;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};
use slam_rs::calib::{
    CameraParts, ImuParts,
    catalog::{CameraStatics, ImuStatics},
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

#[pyfunction]
fn catalog_camera_calib<'py>(
    py: Python<'py>,
    index: usize,
    statics: &Bound<'py, PyAny>,
    downscale: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let raw = CameraStatics {
        distortion_model: statics.getattr("distortion_model")?.extract()?,
        distortion_coefficients: statics.getattr("distortion_coefficients")?.extract()?,
        image_from_camera: statics.getattr("image_from_camera")?.extract()?,
        resolution_wh: statics.getattr("resolution_wh")?.extract()?,
        transform_mat3x3: statics.getattr("transform_mat3x3")?.extract()?,
        transform_translation: statics.getattr("transform_translation")?.extract()?,
        transform_relation: statics.getattr("transform_relation")?.extract()?,
        distortion_valid_radius: statics.getattr("distortion_valid_radius")?.extract()?,
    };
    let camera = CameraParts::from_catalog_statics(index, &raw, downscale).map_err(value_error)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("index", index)?;
    kwargs.set_item("width", camera.width)?;
    kwargs.set_item("height", camera.height)?;
    kwargs.set_item("fx", camera.fx)?;
    kwargs.set_item("fy", camera.fy)?;
    kwargs.set_item("cx", camera.cx)?;
    kwargs.set_item("cy", camera.cy)?;
    kwargs.set_item("model", camera.model)?;
    kwargs.set_item("distortion", camera.distortion.to_pyarray(py))?;
    kwargs.set_item("distortion_valid_radius", camera.distortion_valid_radius)?;
    let pose: Vec<Vec<f64>> = camera
        .imu_t_cam_row_major
        .chunks_exact(4)
        .map(|r| r.to_vec())
        .collect();
    kwargs.set_item("imu_T_cam", PyArray2::from_vec2(py, &pose)?)?;
    py.import("slam_rs.rig")?
        .getattr("CameraCalib")?
        .call((), Some(&kwargs))
}

#[pyfunction]
fn catalog_imu_calib<'py>(
    py: Python<'py>,
    calibration: &Bound<'py, PyAny>,
    imu_t_body: &Bound<'py, PyAny>,
    applied_time_shift_ns: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let raw = ImuStatics {
        rate_hz: calibration.getattr("rate_hz")?.extract()?,
        gyro_noise_density: calibration.getattr("gyro_noise_density")?.extract()?,
        accel_noise_density: calibration.getattr("accel_noise_density")?.extract()?,
        gyro_bias_random_walk: calibration.getattr("gyro_bias_random_walk")?.extract()?,
        accel_bias_random_walk: calibration.getattr("accel_bias_random_walk")?.extract()?,
        applied_time_shift_ns,
    };
    let imu = ImuParts::from_catalog_statics(&raw).map_err(value_error)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("frequency_hz", imu.frequency_hz)?;
    kwargs.set_item("gyro_noise_std", imu.gyro_noise_std)?;
    kwargs.set_item("accel_noise_std", imu.accel_noise_std)?;
    kwargs.set_item("gyro_bias_std", imu.gyro_bias_std)?;
    kwargs.set_item("accel_bias_std", imu.accel_bias_std)?;
    kwargs.set_item("cam_time_offset_ns", imu.cam_time_offset_ns)?;
    kwargs.set_item("imu_T_body", imu_t_body)?;
    py.import("slam_rs.rig")?
        .getattr("ImuCalib")?
        .call((), Some(&kwargs))
}

#[pyfunction]
fn catalog_frame_nearest_anchor(
    times: Vec<i64>,
    cursor: usize,
    anchor_t_ns: i64,
    tolerance_ns: i64,
) -> (Option<usize>, usize) {
    catalog_timing::frame_nearest_anchor(&times, cursor, anchor_t_ns, tolerance_ns)
}

type FrameMatches<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<i64>>);
#[pyfunction]
fn catalog_match_framesets(
    py: Python<'_>,
    camera_t_ns: Vec<Vec<i64>>,
    tolerance_ns: i64,
) -> PyResult<FrameMatches<'_>> {
    let cameras: Vec<&[i64]> = camera_t_ns.iter().map(Vec::as_slice).collect();
    let sets = catalog_timing::match_framesets(&cameras, tolerance_ns).map_err(value_error)?;
    if sets.is_empty() {
        return Err(value_error(format!(
            "no frameset has all {} cameras within {tolerance_ns} ns of camera 0",
            cameras.len()
        )));
    }
    let times: Vec<i64> = sets.iter().map(|s| s.0).collect();
    let indices: Vec<Vec<i64>> = sets
        .iter()
        .map(|s| s.1.iter().map(|&i| i as i64).collect())
        .collect();
    Ok((times.to_pyarray(py), PyArray2::from_vec2(py, &indices)?))
}

#[pyfunction]
fn catalog_pair_imu<'py>(
    py: Python<'py>,
    gyro_t_ns: Vec<i64>,
    gyro_rad_s: Vec<[f64; 3]>,
    accel_t_ns: Vec<i64>,
    accel_m_s2: Vec<[f64; 3]>,
    interpolate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if gyro_t_ns.len() != gyro_rad_s.len() || accel_t_ns.len() != accel_m_s2.len() {
        return Err(value_error("IMU timestamp and measurement counts differ"));
    }
    let rows = catalog_timing::pair_imu(
        gyro_t_ns.into_iter().zip(gyro_rad_s).collect(),
        accel_t_ns.into_iter().zip(accel_m_s2).collect(),
        interpolate,
    )
    .map_err(value_error)?;
    let times: Vec<i64> = rows.iter().map(|r| r.t_ns).collect();
    let gyro: Vec<[f64; 3]> = rows.iter().map(|r| r.gyro).collect();
    let accel: Vec<[f64; 3]> = rows.iter().map(|r| r.accel).collect();
    // Empty synchronized streams still have the same (0, 3) shape.
    let gyro = numpy::ndarray::Array2::from_shape_vec(
        (rows.len(), 3),
        gyro.into_iter().flatten().collect(),
    )
    .map_err(value_error)?;
    let accel = numpy::ndarray::Array2::from_shape_vec(
        (rows.len(), 3),
        accel.into_iter().flatten().collect(),
    )
    .map_err(value_error)?;
    py.import("slam_rs.catalog_timing")?
        .getattr("ImuStream")?
        .call1((
            times.to_pyarray(py),
            gyro.to_pyarray(py),
            accel.to_pyarray(py),
        ))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(catalog_rig_profile, module)?)?;
    module.add_function(wrap_pyfunction!(catalog_camera_calib, module)?)?;
    module.add_function(wrap_pyfunction!(catalog_imu_calib, module)?)?;
    module.add_function(wrap_pyfunction!(catalog_frame_nearest_anchor, module)?)?;
    module.add_function(wrap_pyfunction!(catalog_match_framesets, module)?)?;
    module.add_function(wrap_pyfunction!(catalog_pair_imu, module)?)?;
    Ok(())
}
