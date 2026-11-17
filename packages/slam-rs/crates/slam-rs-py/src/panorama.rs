//! The dual-fisheye lane's stitch without a pipeline, for the derived viewing panorama.
use crate::{gray_array, gray_pixels, value_error, window_crop};
use kornia_image::ImageSize;
use kornia_staging_imgproc::panorama::TileGeometry;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use slam_rs::ImageView;
use slam_rs::frontend::input::LensStitch;

/// Two lens images of an Insta360 frameset in, the 1920x960 `uint8` panorama out:
/// the tables and the host remap of the dual-fisheye lane (`LensStitch`), built from
/// the same factory inputs `Vio(dual_fisheye=...)` takes. Independent of `Vio`.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
pub struct PanoramaStitch {
    stitch: LensStitch,
}

#[pymethods]
impl PanoramaStitch {
    /// `lens_string` is the raw V3/V6 factory string, `window_crop` the
    /// `(src_w, src_h, dst_w, dst_h, offset_x, offset_y)` it is stated through or
    /// `None`, `lens_size` each lens image's `(width, height)`. The tiles take the
    /// default margin; the panorama's pixels do not depend on it.
    #[new]
    fn new(
        lens_string: &str,
        window_crop_px: Option<[i64; 6]>,
        lens_size: (u32, u32),
    ) -> PyResult<Self> {
        let (width, height) = lens_size;
        let lenses = slam_rs::DualFisheyeLenses::parse(
            lens_string,
            window_crop(window_crop_px)?,
            [width, height],
            0.0,
        )
        .map_err(value_error)?;
        let lens = ImageSize {
            width: width as usize,
            height: height as usize,
        };
        Ok(Self {
            stitch: lenses
                .stitch(lens, TileGeometry::default().margin)
                .map_err(value_error)?,
        })
    }

    /// The panorama of lens 0's and lens 1's C-contiguous `uint8[h, w]` images
    /// (`uint8[960, 1920]`, lens 0's hemisphere on the left half).
    fn panorama<'py>(
        &self,
        py: Python<'py>,
        lens0: &Bound<'py, PyAny>,
        lens1: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let arrays = [gray_array(lens0, 0)?, gray_array(lens1, 1)?];
        let mut views: Vec<ImageView<'_>> = Vec::with_capacity(2);
        for (index, array) in arrays.iter().enumerate() {
            let (data, width, height) = gray_pixels(array, index)?;
            views.push(ImageView {
                width,
                height,
                stride: width,
                data,
            });
        }
        let pano = py
            .detach(|| self.stitch.panorama(&views))
            .map_err(value_error)?;
        Ok(
            numpy::ndarray::Array2::from_shape_vec((pano.height(), pano.width()), pano.into_vec())
                .map_err(value_error)?
                .into_pyarray(py),
        )
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PanoramaStitch>()
}
