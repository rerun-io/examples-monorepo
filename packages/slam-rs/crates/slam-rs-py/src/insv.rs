//! Python access to an Insta360 `.insv` clip through `kornia_staging_io::insta360`.
use crate::value_error;
use kornia_staging_io::insta360::{Insta360Clip, LensCalibration};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// One lens of the factory calibration at the track resolution, window crop applied.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct InsvLens {
    /// Lens-string version, 3 or 6.
    #[pyo3(get)]
    version: u8,
    /// Mei sphere offset.
    #[pyo3(get)]
    xi: f64,
    /// Mei focal along x, track pixels.
    #[pyo3(get)]
    fx: f64,
    /// Mei focal along y, track pixels.
    #[pyo3(get)]
    fy: f64,
    /// Principal point x, track pixels.
    #[pyo3(get)]
    cx: f64,
    /// Principal point y, track pixels.
    #[pyo3(get)]
    cy: f64,
    /// Pixels per radian at the optical axis.
    #[pyo3(get)]
    px_per_rad: f64,
    /// Radial coefficients (V3 `k1 k2 k3`, V6 `d0..d4`).
    #[pyo3(get)]
    radial: Vec<f64>,
    /// Tangential coefficients (V3 `p1 p2`, V6 `d5..d8`).
    #[pyo3(get)]
    tangential: Vec<f64>,
    /// Thin-prism coefficients (V6 `d9..d12`).
    #[pyo3(get)]
    prism: Vec<f64>,
    /// Factory yaw, pitch, roll, degrees.
    #[pyo3(get)]
    ypr_deg: [f64; 3],
    /// Factory translation, metres.
    #[pyo3(get)]
    translation_m: [f64; 3],
    /// Rotation rig -> lens, row-major 3x3.
    #[pyo3(get)]
    lens_from_rig_rotation: [[f64; 3]; 3],
    /// Translation of rig -> lens, metres; the rig origin is the midpoint of the lens centres.
    #[pyo3(get)]
    lens_from_rig_translation_m: [f64; 3],
    core: LensCalibration,
}

#[pymethods]
impl InsvLens {
    /// Image radius in track pixels of rays `theta` radians off the axis, through the radial part of the model.
    fn radius_px<'py>(&self, py: Python<'py>, theta: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        theta
            .iter()
            .map(|&t| self.core.radius_px(t))
            .collect::<Vec<_>>()
            .to_pyarray(py)
    }
}

impl From<&LensCalibration> for InsvLens {
    fn from(l: &LensCalibration) -> Self {
        Self {
            version: l.version,
            xi: l.xi,
            fx: l.fx,
            fy: l.fy,
            cx: l.cx,
            cy: l.cy,
            px_per_rad: l.px_per_rad(),
            radial: l.radial.clone(),
            tangential: l.tangential.clone(),
            prism: l.prism.clone(),
            ypr_deg: l.ypr_deg,
            translation_m: l.translation_m,
            lens_from_rig_rotation: l.lens_from_rig_rotation,
            lens_from_rig_translation_m: l.lens_from_rig_translation_m,
            core: l.clone(),
        }
    }
}

/// An open `.insv` clip: metadata, factory lenses, IMU, frame times, and the two HEVC lens tracks.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
pub struct InsvClip {
    clip: Insta360Clip,
}

type Imu<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);
type Samples<'py> = (Vec<Bound<'py, PyBytes>>, Bound<'py, PyArray1<bool>>);

#[pymethods]
impl InsvClip {
    #[new]
    fn new(path: std::path::PathBuf) -> PyResult<Self> {
        Ok(Self {
            clip: Insta360Clip::open(path).map_err(value_error)?,
        })
    }

    /// Camera model name, e.g. `Insta360 X4 Air`.
    #[getter]
    fn camera_type(&self) -> &str {
        &self.clip.trailer.metadata.camera_type
    }

    /// Firmware version string.
    #[getter]
    fn firmware(&self) -> &str {
        &self.clip.trailer.metadata.firmware
    }

    /// The first frame's time on the trailer clock, microseconds.
    #[getter]
    fn first_frame_timestamp_us(&self) -> i64 {
        self.clip.trailer.metadata.first_frame_timestamp_us
    }

    /// Rolling-shutter readout time, milliseconds.
    #[getter]
    fn rolling_shutter_ms(&self) -> Option<f64> {
        self.clip.trailer.metadata.rolling_shutter_ms
    }

    /// The vendor's gyro timestamp shift, milliseconds.
    #[getter]
    fn gyro_timestamp_ms(&self) -> Option<f64> {
        self.clip.trailer.metadata.gyro_timestamp_ms
    }

    /// The raw factory lens strings by version name (`v1`, `v2`, `v3`, `v6`), those present.
    #[getter]
    fn factory_strings(&self) -> Vec<(String, String)> {
        let m = &self.clip.trailer.metadata;
        [
            ("v1", &m.offset_v1),
            ("v2", &m.offset_v2),
            ("v3", &m.offset_v3),
            ("v6", &m.offset_v6),
        ]
        .into_iter()
        .filter_map(|(name, text)| text.clone().map(|t| (name.to_string(), t)))
        .collect()
    }

    /// Video track index of lens 0 and lens 1.
    #[getter]
    fn lens_track(&self) -> (usize, usize) {
        (self.clip.lens_track[0], self.clip.lens_track[1])
    }

    /// V3 factory calibration per lens, in lens order.
    #[getter]
    fn lenses(&self) -> Vec<InsvLens> {
        self.clip.lenses.iter().map(InsvLens::from).collect()
    }

    /// Track width and height in pixels.
    #[getter]
    fn resolution(&self) -> (u16, u16) {
        (self.clip.tracks[0].width, self.clip.tracks[0].height)
    }

    /// Frames per track.
    #[getter]
    fn num_frames(&self) -> usize {
        self.clip
            .tracks
            .iter()
            .map(|t| t.samples.len())
            .min()
            .unwrap_or(0)
    }

    /// IMU samples: times (trailer clock, us), accel `(n, 3)` m/s^2, gyro `(n, 3)` rad/s.
    fn imu<'py>(&self, py: Python<'py>) -> PyResult<Imu<'py>> {
        let imu = &self.clip.trailer.imu;
        let t: Vec<i64> = imu.iter().map(|s| s.t_us).collect();
        let accel: Vec<Vec<f64>> = imu.iter().map(|s| s.accel_m_s2.to_vec()).collect();
        let gyro: Vec<Vec<f64>> = imu.iter().map(|s| s.gyro_rad_s.to_vec()).collect();
        Ok((
            t.to_pyarray(py),
            PyArray2::from_vec2(py, &accel)?,
            PyArray2::from_vec2(py, &gyro)?,
        ))
    }

    /// Per-frame exposure entries of one sensor record (`vin1=False`: record 4): times (us) and exposure (s).
    fn frame_exposures<'py>(
        &self,
        py: Python<'py>,
        vin1: bool,
    ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
        let e = self.clip.frame_exposures(vin1);
        (
            e.iter().map(|x| x.t_us).collect::<Vec<_>>().to_pyarray(py),
            e.iter()
                .map(|x| x.exposure_s)
                .collect::<Vec<_>>()
                .to_pyarray(py),
        )
    }

    /// Samples `start..stop` of one track as Annex B access units (parameter sets on IRAPs) and their IRAP flags.
    /// Read in batches: the samples stay on disk until asked for.
    fn read_samples<'py>(
        &self,
        py: Python<'py>,
        track: usize,
        start: usize,
        stop: usize,
    ) -> PyResult<Samples<'py>> {
        let t = self
            .clip
            .tracks
            .get(track)
            .ok_or_else(|| PyIndexError::new_err(format!("track {track}")))?;
        let stop = stop.min(t.samples.len());
        let mut reader = self.clip.sample_reader().map_err(value_error)?;
        let (mut samples, mut keyframes, mut out) = (Vec::new(), Vec::new(), Vec::new());
        for index in start..stop {
            out.clear();
            keyframes.push(
                reader
                    .read_annex_b(t, index, &mut out)
                    .map_err(value_error)?,
            );
            samples.push(PyBytes::new(py, &out));
        }
        Ok((samples, keyframes.to_pyarray(py)))
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<InsvClip>()?;
    module.add_class::<InsvLens>()?;
    Ok(())
}
