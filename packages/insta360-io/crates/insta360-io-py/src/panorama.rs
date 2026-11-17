//! The dual-fisheye stitch on the CPU: two lens images into the two hemisphere tiles, then the 1920x960 viewing
//! panorama, through `kornia_staging_imgproc`'s stitch plan (`LensStitch`).
//!
//! The panorama frame `P` is stated by the two cameras alone (slam-rs's rule): axes = camera 0's, origin at the
//! midpoint of the two camera centres. Tile 0 looks along camera 0's `+Z`, tile 1 along `-Z`, each tile centred on its
//! own camera, so only the rotation of camera 1 in camera 0 enters the tables.
use crate::camera::MeiCamera;
use crate::value_error;
use kornia_image::ImageSize;
use kornia_staging_3d::camera::{CameraModel, Mei};
use kornia_staging_imgproc::panorama::{LensStitch, StitchError, TileGeometry};
use nalgebra::{Matrix3, Rotation3, Vector3};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Two 960x960 hemispheres side by side, no margin: the panorama is the two tile centres.
pub const GEOMETRY: TileGeometry = TileGeometry {
    hemisphere_width: 960,
    height: 960,
    margin: 0,
};

/// The stitch plan of `cameras` (camera 0, camera 1) with `cam0_from_cam1` the rotation of camera 1 in camera 0.
pub fn stitch_plan(
    cameras: [&Mei<f64>; 2],
    cam0_from_cam1: &Rotation3<f64>,
    lens: ImageSize,
) -> Result<LensStitch, StitchError> {
    LensStitch::new(GEOMETRY, lens, |k, ray| {
        let local = if k == 0 {
            ray
        } else {
            cam0_from_cam1
                .inverse_transform_vector(&Vector3::from(ray))
                .into()
        };
        cameras[k].project(local).ok()
    })
}

/// Two lens images of a dual-fisheye frameset in, the 1920x960 `uint8` panorama out.
#[pyclass(module = "insta360_io", frozen, skip_from_py_object)]
pub struct PanoramaStitch {
    stitch: LensStitch,
}

#[pymethods]
impl PanoramaStitch {
    /// `camera0`, `camera1`: the two lens cameras; `cam0_from_cam1`: the pose of camera 1 in camera 0 as a 4x4
    /// row-major matrix (its translation does not enter the tables); `lens_size`: each lens image's `(width, height)`.
    /// Raises `ValueError` on a rotation that is not one or a lens below 4x4.
    #[new]
    fn new(
        camera0: MeiCamera,
        camera1: MeiCamera,
        cam0_from_cam1: [[f64; 4]; 4],
        lens_size: (u32, u32),
    ) -> PyResult<Self> {
        let matrix = Matrix3::from_fn(|row, column| cam0_from_cam1[row][column]);
        let orthonormal = (matrix.transpose() * matrix - Matrix3::identity())
            .abs()
            .max()
            < 1e-5
            && (matrix.determinant() - 1.0).abs() < 1e-5;
        if !orthonormal || cam0_from_cam1[3] != [0.0, 0.0, 0.0, 1.0] {
            return Err(PyValueError::new_err(format!(
                "cam0_from_cam1 {cam0_from_cam1:?} is not a rigid transform"
            )));
        }
        let lens = ImageSize {
            width: lens_size.0 as usize,
            height: lens_size.1 as usize,
        };
        let rotation = Rotation3::from_matrix_unchecked(matrix);
        let stitch = stitch_plan([&camera0.camera, &camera1.camera], &rotation, lens)
            .map_err(value_error)?;
        Ok(Self { stitch })
    }

    /// The panorama of lens 0's and lens 1's C-contiguous `uint8[h, w]` images
    /// (`uint8[960, 1920]`, camera 0's hemisphere on the left half).
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
        let width = self.stitch.lens_size().width;
        let pano = py
            .detach(|| self.stitch.panorama(pixels, width))
            .map_err(value_error)?;
        Ok(
            numpy::ndarray::Array2::from_shape_vec((pano.height(), pano.width()), pano.into_vec())
                .map_err(value_error)?
                .into_pyarray(py),
        )
    }
}

impl PanoramaStitch {
    /// Lens `index` as a C-contiguous `uint8` array of the plan's lens size.
    fn lens_array<'py>(
        &self,
        object: &Bound<'py, PyAny>,
        index: usize,
    ) -> PyResult<PyReadonlyArray2<'py, u8>> {
        let array: &Bound<'py, PyArray2<u8>> = object.cast::<PyArray2<u8>>().map_err(|_| {
            PyValueError::new_err(format!("image {index} must be a 2-D uint8 numpy array"))
        })?;
        let readonly = array.readonly();
        let lens = self.stitch.lens_size();
        let shape: &[usize] = readonly.shape();
        if shape != [lens.height, lens.width] {
            return Err(PyValueError::new_err(format!(
                "a {}x{} lens image; the stitch was built for {}x{}",
                shape[1], shape[0], lens.width, lens.height
            )));
        }
        if !readonly.is_c_contiguous() {
            return Err(PyValueError::new_err(format!(
                "image {index} must be C-contiguous; pass numpy.ascontiguousarray(image)"
            )));
        }
        Ok(readonly)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PanoramaStitch>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_staging_3d::camera::formats::insta360::{
        parse_lens_string, FactoryLens, WindowCrop,
    };

    /// The X4 Air clip's V6 factory string (VID_20261116_125200_00_003).
    const V6: &str = "2_2.314940_7096.760_7095.390_3834.830_3864.120_0.119_0.407_90.629_0.000000_0.000000_0.000000_0.93861496_-1.68402135_2.19166017_3.85762072_0.00000000_-0.00495962_0.00156076_-0.00330722_0.00397486_0.00827898_-0.00125315_0.02416405_-0.00143547_15360_7680_131_2.314940_7149.220_7149.700_11543.300_3858.430_-0.212_0.339_88.279_0.000132_-0.000043_-0.033291_0.95675713_-1.80496192_2.88544273_1.76948690_0.00000000_0.00131924_0.00023973_0.00279902_-0.00570679_0.00069985_0.00064811_0.00799088_0.02677096_15360_7680_131_394240";

    #[test]
    fn the_plan_of_two_cameras_and_their_relative_pose_is_lens_stitch_of_the_same_cameras() {
        let crop = WindowCrop {
            src: [7680, 7680],
            dst: [7424, 7424],
            offset: [0, 0],
        };
        let lenses: Vec<FactoryLens> = parse_lens_string(V6, Some(crop), [96, 96]).unwrap();
        // Camera 1 in camera 0, from the factory poses in the vendor frame: R_c0_c1 = R_P_c0^T R_P_c1.
        let cam0_from_cam1 =
            lenses[0].pano_from_lens.rotation.inverse() * lenses[1].pano_from_lens.rotation;
        let lens = ImageSize {
            width: 96,
            height: 96,
        };
        let plan = stitch_plan(
            [&lenses[0].camera, &lenses[1].camera],
            &cam0_from_cam1,
            lens,
        )
        .unwrap();
        // P = camera 0's axes: lens 0 sees P directly, lens 1 through R_c1_c0.
        let direct = LensStitch::new(GEOMETRY, lens, |k, ray| {
            let rotation = [Rotation3::identity(), cam0_from_cam1][k];
            let local = rotation.inverse_transform_vector(&ray.into());
            lenses[k].camera.project(local.into()).ok()
        })
        .unwrap();
        assert_eq!(plan.tables(), direct.tables());
        // Both tiles see their lens: most of each table maps a pixel.
        for table in plan.tables() {
            let valid = table
                .samples()
                .iter()
                .filter(|s| **s != kornia_staging_imgproc::panorama::RemapTable::INVALID)
                .count();
            assert!(
                valid * 2 > table.samples().len(),
                "{valid} of {}",
                table.samples().len()
            );
        }
    }
}
