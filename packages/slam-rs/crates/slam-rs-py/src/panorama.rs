//! The dual-fisheye stitch without a pipeline, for the derived viewing panorama.
use crate::{gray_array, gray_pixels, value_error, window_crop};
use kornia_image::ImageSize;
use kornia_staging_imgproc::panorama::{LensStitch, TileGeometry};
use kornia_staging_3d::camera::CameraModel;
use kornia_staging_3d::camera::formats::insta360::{FactoryLens, parse_lens_string};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Two lens images of an Insta360 frameset in, the 1920x960 `uint8` panorama out,
/// through `kornia_staging_imgproc`'s stitch plan (`LensStitch`): the same tables
/// and host remap as the dual-fisheye lane of `Vio(dual_fisheye=...)`, built from
/// the same factory inputs. Independent of `Vio`.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
pub struct PanoramaStitch {
    stitch: LensStitch,
}

#[pymethods]
impl PanoramaStitch {
    /// `lens_string` is the raw V3/V6 factory string, `window_crop` the
    /// `(src_w, src_h, dst_w, dst_h, offset_x, offset_y)` it is stated through or
    /// `None`, `lens_size` each lens image's `(width, height)`. The tiles have no
    /// margin: the panorama is the two tile centres.
    #[new]
    fn new(
        lens_string: &str,
        window_crop_px: Option<[i64; 6]>,
        lens_size: (u32, u32),
    ) -> PyResult<Self> {
        let (width, height) = lens_size;
        let lenses: Vec<FactoryLens> =
            parse_lens_string(lens_string, window_crop(window_crop_px)?, [width, height])
                .map_err(value_error)?;
        let lenses: [FactoryLens; 2] = lenses.try_into().map_err(|all: Vec<_>| {
            PyValueError::new_err(format!(
                "a dual fisheye has two lenses; the string holds {}",
                all.len()
            ))
        })?;
        let geometry = TileGeometry {
            hemisphere_width: 960,
            height: 960,
            margin: 0,
        };
        let lens = ImageSize {
            width: width as usize,
            height: height as usize,
        };
        let stitch = LensStitch::new(geometry, lens, |k, ray| {
            let local = lenses[k]
                .pano_from_lens
                .rotation
                .inverse_transform_vector(&ray.into());
            lenses[k].camera.project(local.into()).ok()
        })
        .map_err(value_error)?;
        Ok(Self { stitch })
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
        let lens = self.stitch.lens_size();
        let mut data: [&[u8]; 2] = [&[], &[]];
        for (index, array) in arrays.iter().enumerate() {
            let (pixels, width, height) = gray_pixels(array, index)?;
            if (width, height) != (lens.width, lens.height) {
                return Err(PyValueError::new_err(format!(
                    "lens image {index} is {width}x{height}; the stitch reads {}x{}",
                    lens.width, lens.height
                )));
            }
            data[index] = pixels;
        }
        let pano = py
            .detach(|| self.stitch.panorama(data, lens.width))
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
