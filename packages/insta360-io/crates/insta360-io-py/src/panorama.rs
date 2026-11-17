//! The dual-fisheye stitch on the CPU: two lens images of an Insta360 frameset into the two hemisphere tiles, then
//! the 1920x960 viewing panorama.
//!
//! The same steps as slam-rs's `LensStitch` (the lane's host remap): tables from `build_tile_lut` through each lens's
//! `project_from_panorama`, then per frameset `lens_level1_into` + `remap_tile` per tile and `assemble_panorama`.
use crate::value_error;
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::panorama::{
    assemble_panorama, build_tile_lut, lens_level1_into, remap_tile, TileGeometry, TileLut,
};
use kornia_staging_io::insta360::{parse_lens_string, LensCalibration, WindowCrop};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Two lens images of an Insta360 frameset in, the 1920x960 `uint8` panorama out: the tables and the host remap of
/// slam-rs's dual-fisheye lane, built from the same factory inputs.
#[pyclass(module = "insta360_io", frozen, skip_from_py_object)]
pub struct PanoramaStitch {
    luts: [TileLut; 2],
    lens: ImageSize,
}

#[pymethods]
impl PanoramaStitch {
    /// `lens_string` is the raw V3/V6 factory string, `window_crop_px` the
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
        if width < 4 || height < 4 {
            return Err(PyValueError::new_err(format!(
                "a {width}x{height} lens image is below 4x4"
            )));
        }
        let lenses: Vec<LensCalibration> =
            parse_lens_string(lens_string, window_crop(window_crop_px)?, [width, height])
                .map_err(|error| value_error(format!("factory lens string: {error}")))?;
        let lenses: [LensCalibration; 2] = lenses.try_into().map_err(|all: Vec<_>| {
            value_error(format!(
                "a dual fisheye has two lenses; the string holds {}",
                all.len()
            ))
        })?;
        let lens = ImageSize {
            width: width as usize,
            height: height as usize,
        };
        let geometry = TileGeometry::default();
        let luts: [TileLut; 2] = std::array::from_fn(|k| {
            build_tile_lut(geometry, k, lens, |ray| {
                lenses[k].project_from_panorama(ray)
            })
        });
        Ok(Self { luts, lens })
    }

    /// The panorama of lens 0's and lens 1's C-contiguous `uint8[h, w]` images
    /// (`uint8[960, 1920]`, lens 0's hemisphere on the left half).
    fn panorama<'py>(
        &self,
        py: Python<'py>,
        lens0: &Bound<'py, PyAny>,
        lens1: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let arrays = [self.lens_array(lens0, 0)?, self.lens_array(lens1, 1)?];
        let pixels: [&[u8]; 2] = [
            arrays[0].as_slice().map_err(value_error)?,
            arrays[1].as_slice().map_err(value_error)?,
        ];
        let pano = py.detach(|| self.stitch(pixels)).map_err(value_error)?;
        Ok(
            numpy::ndarray::Array2::from_shape_vec((pano.height(), pano.width()), pano.into_vec())
                .map_err(value_error)?
                .into_pyarray(py),
        )
    }
}

impl PanoramaStitch {
    /// Lens `index` as a C-contiguous `uint8` array of the tables' lens size.
    fn lens_array<'py>(
        &self,
        object: &Bound<'py, PyAny>,
        index: usize,
    ) -> PyResult<PyReadonlyArray2<'py, u8>> {
        let array: &Bound<'py, PyArray2<u8>> = object.cast::<PyArray2<u8>>().map_err(|_| {
            PyValueError::new_err(format!("image {index} must be a 2-D uint8 numpy array"))
        })?;
        let readonly = array.readonly();
        let shape: &[usize] = readonly.shape();
        if shape != [self.lens.height, self.lens.width] {
            return Err(PyValueError::new_err(format!(
                "a {}x{} lens image; the stitch was built for {}x{}",
                shape[1], shape[0], self.lens.width, self.lens.height
            )));
        }
        if !readonly.is_c_contiguous() {
            return Err(PyValueError::new_err(format!(
                "image {index} must be C-contiguous; pass numpy.ascontiguousarray(image)"
            )));
        }
        Ok(readonly)
    }

    /// Tile `k` from lens `k` (`lens_level1_into` + `remap_tile`), then the two tiles' centre columns.
    fn stitch(&self, lenses: [&[u8]; 2]) -> Result<Image<u8, 1>, kornia_image::ImageError> {
        let geometry: TileGeometry = self.luts[0].geometry;
        let tile_size = ImageSize {
            width: geometry.width(),
            height: geometry.height,
        };
        let mut level0 = Image::from_size_val(self.lens, 0u16)?;
        let mut level1 = Image::from_size_val(self.luts[0].source_size, 0u16)?;
        let mut tiles = [
            Image::from_size_val(tile_size, 0u16)?,
            Image::from_size_val(tile_size, 0u16)?,
        ];
        for (k, tile) in tiles.iter_mut().enumerate() {
            lens_level1_into(lenses[k], self.lens.width, &mut level0, &mut level1)?;
            remap_tile(&level1, &self.luts[k], tile)?;
        }
        assemble_panorama([&tiles[0], &tiles[1]], geometry)
    }
}

/// A factory window crop `(src_w, src_h, dst_w, dst_h, offset_x, offset_y)` from Python.
fn window_crop(crop: Option<[i64; 6]>) -> PyResult<Option<WindowCrop>> {
    let size = |value: i64| {
        u32::try_from(value).map_err(|_| PyValueError::new_err(format!("{value} is not a size")))
    };
    crop.map(|[sw, sh, dw, dh, ox, oy]| {
        Ok(WindowCrop {
            src: [size(sw)?, size(sh)?],
            dst: [size(dw)?, size(dh)?],
            offset: [ox, oy],
        })
    })
    .transpose()
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PanoramaStitch>()
}
