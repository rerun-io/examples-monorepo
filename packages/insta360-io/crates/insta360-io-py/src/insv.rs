//! Python access to an Insta360 `.insv` clip through `kornia_staging_io::insta360`.
use crate::camera::MeiCamera;
use crate::value_error;
use kornia_staging_3d::camera::formats::insta360::FactoryLens;
use kornia_staging_io::insta360::Insta360Clip;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;

/// One lens of the factory calibration at the track resolution, window crop applied.
#[pyclass(module = "insta360_io", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct InsvLens {
    core: FactoryLens,
}

#[pymethods]
impl InsvLens {
    /// The lens camera, for the video track's pixels.
    #[getter]
    fn camera(&self) -> MeiCamera {
        MeiCamera {
            camera: self.core.camera,
        }
    }

    /// `R_P_lens`, row-major: lens vectors into the vendor rig frame `P` (+Z lens-0 axis, +Y toward the handle).
    #[getter]
    fn pano_from_lens_rotation(&self) -> [[f64; 3]; 3] {
        let r = self.core.pano_from_lens.rotation;
        std::array::from_fn(|row| std::array::from_fn(|column| r[(row, column)]))
    }

    /// Lens centre in `P`, metres; the origin is the midpoint of the lens centres.
    #[getter]
    fn pano_centre_m(&self) -> [f64; 3] {
        self.core.pano_from_lens.translation.vector.into()
    }
}

impl From<&FactoryLens> for InsvLens {
    fn from(l: &FactoryLens) -> Self {
        Self { core: *l }
    }
}

/// An open `.insv` clip as the file states it (no device profile): metadata, factory lenses, IMU, frame times and the
/// two HEVC lens tracks' layout. `InsvRig` applies a profile and reads the samples.
#[pyclass(module = "insta360_io", frozen, skip_from_py_object)]
pub struct InsvClip {
    clip: Insta360Clip,
}

pub(crate) type Imu<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

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

    /// V6 factory calibration per lens, in lens order; empty when the clip has no V6 string.
    #[getter]
    fn lenses_v6(&self) -> Vec<InsvLens> {
        self.clip
            .lenses_v6
            .iter()
            .flatten()
            .map(InsvLens::from)
            .collect()
    }

    /// The window crop the lens strings are stated through: `(src_w, src_h, dst_w, dst_h, offset_x, offset_y)`.
    #[getter]
    fn window_crop(&self) -> Option<(u32, u32, u32, u32, i64, i64)> {
        self.clip.trailer.metadata.window_crop.map(|c| {
            (
                c.src[0],
                c.src[1],
                c.dst[0],
                c.dst[1],
                c.offset[0],
                c.offset[1],
            )
        })
    }

    /// Track width and height in pixels.
    #[getter]
    fn resolution(&self) -> (u16, u16) {
        (self.clip.tracks[0].width, self.clip.tracks[0].height)
    }

    /// Frames of the clip: those present in both lens tracks.
    #[getter]
    fn num_frames(&self) -> usize {
        self.clip.frames()
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
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<InsvClip>()?;
    module.add_class::<InsvLens>()?;
    Ok(())
}
