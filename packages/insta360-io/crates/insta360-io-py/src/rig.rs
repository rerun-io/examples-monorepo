//! Python access to the calibrated rig of a clip (`insta360_io_core::rig`): what `InsvReader` turns into Rerun chunks.
use crate::insv::{Imu, InsvLens};
use crate::value_error;
use insta360_io_core::profile::ImuNoise as CoreImuNoise;
use insta360_io_core::rig::{InsvRig as CoreRig, Packed, RigError};
use kornia_staging_io::container::mp4::VideoCodec;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

/// The IMU noise model of a device profile.
#[pyclass(module = "insta360_io", frozen, skip_from_py_object, get_all)]
#[derive(Debug, Clone)]
pub struct ImuNoise {
    /// rad/s/sqrt(Hz).
    gyro_noise_density: f64,
    /// m/s^2/sqrt(Hz).
    accel_noise_density: f64,
    /// rad/s^2/sqrt(Hz).
    gyro_bias_random_walk: f64,
    /// m/s^3/sqrt(Hz).
    accel_bias_random_walk: f64,
    /// Sample rate the densities are stated for.
    rate_hz: f64,
    /// Where the numbers come from.
    source: String,
}

impl From<&CoreImuNoise> for ImuNoise {
    fn from(n: &CoreImuNoise) -> Self {
        Self {
            gyro_noise_density: n.gyro_noise_density,
            accel_noise_density: n.accel_noise_density,
            gyro_bias_random_walk: n.gyro_bias_random_walk,
            accel_bias_random_walk: n.accel_bias_random_walk,
            rate_hz: n.rate_hz,
            source: n.source.clone(),
        }
    }
}

/// A clip with its device profile applied: two lens cameras in the IMU frame, times on one clock, the GOP demux.
#[pyclass(module = "insta360_io", frozen, skip_from_py_object)]
pub struct InsvRig {
    rig: CoreRig,
}

type PackedArrays<'py> = (
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<bool>>,
);

impl InsvRig {
    fn lens(&self, lens: usize) -> PyResult<usize> {
        (lens < 2)
            .then_some(lens)
            .ok_or_else(|| PyIndexError::new_err(format!("lens {lens}; a clip has lenses 0 and 1")))
    }
}

/// `IndexError` for a lens or range outside the rig, `ValueError` for every other refusal.
fn rig_error(error: RigError) -> PyErr {
    match error {
        RigError::OutOfRange(why) => PyIndexError::new_err(why),
        other => value_error(other),
    }
}

#[pymethods]
impl InsvRig {
    /// Read the clip and select its profile; `frame_limit` keeps the first frames and the IMU up to 1 s past them.
    #[new]
    #[pyo3(signature = (path, frame_limit=None))]
    fn new(py: Python<'_>, path: std::path::PathBuf, frame_limit: Option<usize>) -> PyResult<Self> {
        let rig = py
            .detach(|| CoreRig::open(path, frame_limit))
            .map_err(value_error)?;
        Ok(Self { rig })
    }

    /// Camera model name, e.g. `Insta360 X4 Air`.
    #[getter]
    fn camera_type(&self) -> &str {
        &self.rig.clip.trailer.metadata.camera_type
    }

    /// Firmware version string.
    #[getter]
    fn firmware(&self) -> &str {
        &self.rig.clip.trailer.metadata.firmware
    }

    /// Name of the selected device profile; `None` when no measured profile fits (`profile_note` says why).
    #[getter]
    fn profile_name(&self) -> Option<&str> {
        self.rig.profile.as_ref().ok().map(|p| p.name.as_str())
    }

    /// Version of the selected device profile.
    #[getter]
    fn profile_version(&self) -> Option<u32> {
        self.rig.profile.as_ref().ok().map(|p| p.version)
    }

    /// Empty with a profile; without one, why there is none.
    #[getter]
    fn profile_note(&self) -> &str {
        self.rig.profile.as_ref().err().map_or("", String::as_str)
    }

    /// The selected profile's IMU noise model.
    #[getter]
    fn imu_noise(&self) -> Option<ImuNoise> {
        self.rig
            .profile
            .as_ref()
            .ok()
            .map(|p| ImuNoise::from(&p.imu_noise))
    }

    /// Rolling-shutter readout of each lens image, seconds (first row to last); `None` when neither the clip nor a
    /// profile states one.
    #[getter]
    fn readout_s(&self) -> Option<f64> {
        self.rig.readout_s
    }

    /// `R_P_imu`, row-major: the profile's rotation, orthonormalised; `None` without a profile.
    #[getter]
    fn pano_from_imu(&self) -> Option<[[f64; 3]; 3]> {
        self.rig.profile.as_ref().ok().map(|p| {
            let r = p.imu_from_pano().transpose();
            std::array::from_fn(|row| std::array::from_fn(|column| r[(row, column)]))
        })
    }

    /// Codec of both lens tracks, as FFmpeg names it: `h264` or `hevc`.
    #[getter]
    fn codec(&self) -> &'static str {
        match self.rig.clip.tracks[0].codec {
            VideoCodec::H264 => "h264",
            VideoCodec::H265 => "hevc",
        }
    }

    /// Lens 0 and lens 1 (V6 when the clip has a V6 string, else V3), at the track resolution.
    #[getter]
    fn lenses(&self) -> Vec<InsvLens> {
        self.rig.lenses.iter().map(InsvLens::from).collect()
    }

    /// `T_rig_lens` of one lens, a row-major 4x4 matrix: rig = IMU frame with a profile, vendor frame P without.
    fn rig_from_lens(&self, lens: usize) -> PyResult<[[f64; 4]; 4]> {
        let m = self.rig.rig_from_lens[self.lens(lens)?].to_homogeneous();
        Ok(std::array::from_fn(|row| {
            std::array::from_fn(|column| m[(row, column)])
        }))
    }

    /// Video track index of each lens.
    #[getter]
    fn lens_track(&self) -> (usize, usize) {
        (self.rig.clip.lens_track[0], self.rig.clip.lens_track[1])
    }

    /// Track width and height in pixels.
    #[getter]
    fn resolution(&self) -> (u16, u16) {
        (
            self.rig.clip.tracks[0].width,
            self.rig.clip.tracks[0].height,
        )
    }

    /// Frames of the rig.
    #[getter]
    fn num_frames(&self) -> usize {
        self.rig.frames()
    }

    /// The earliest frame or IMU time on the trailer clock, ns.
    #[getter]
    fn start_ns(&self) -> i64 {
        self.rig.start_ns()
    }

    /// Frame times on the trailer clock, ns.
    fn frame_ns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.rig.frame_ns.to_pyarray(py)
    }

    /// Exposure time per frame, `(2, n)` seconds: lens 0 (record 4) and lens 1 (record 12).
    fn exposure_s<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(PyArray2::from_vec2(py, &self.rig.exposure_s)?)
    }

    /// IMU samples: times (trailer clock, ns), accel `(n, 3)` m/s^2, gyro `(n, 3)` rad/s, IMU axes.
    fn imu<'py>(&self, py: Python<'py>) -> PyResult<Imu<'py>> {
        let rows = |v: &[[f64; 3]]| {
            PyArray1::from_iter(py, v.iter().flatten().copied()).reshape([v.len(), 3])
        };
        Ok((
            self.rig.imu_ns.to_pyarray(py),
            rows(&self.rig.accel)?,
            rows(&self.rig.gyro)?,
        ))
    }

    /// GOPs of one lens as `(n, 2)` `[start, stop)` frame ranges: a keyframe and the samples up to the next one.
    fn gops<'py>(&self, py: Python<'py>, lens: usize) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let gops = self.rig.gops(lens).map_err(rig_error)?;
        PyArray1::from_iter(py, gops.iter().flat_map(|g| [g.start as i64, g.end as i64]))
            .reshape([gops.len(), 2])
    }

    /// Samples `start..stop` of one lens as Annex B access units, packed: bytes back to back, `stop - start + 1`
    /// int32 offsets, IRAP flags. The read runs without the GIL; the arrays take the buffers without a copy.
    fn read_packed<'py>(
        &self,
        py: Python<'py>,
        lens: usize,
        start: usize,
        stop: usize,
    ) -> PyResult<PackedArrays<'py>> {
        let Packed {
            data,
            offsets,
            keyframes,
        } = py
            .detach(|| self.rig.read_packed(lens, start..stop))
            .map_err(rig_error)?;
        Ok((
            PyArray1::from_vec(py, data),
            PyArray1::from_vec(py, offsets),
            PyArray1::from_vec(py, keyframes),
        ))
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<InsvRig>()?;
    module.add_class::<ImuNoise>()?;
    Ok(())
}
