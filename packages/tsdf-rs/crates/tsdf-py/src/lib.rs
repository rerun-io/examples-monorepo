//! The `tsdf_rs` Python module: NumPy bindings for the `tsdf` core.
//!
//! Inputs are C-contiguous, row-major NumPy arrays (a Fortran-ordered or
//! strided array is refused, never read transposed). They are copied before
//! the GIL is released while the core integrates or extracts, so another
//! Python thread cannot write them mid-call.

use numpy::ndarray::Dimension;
use numpy::{
    Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tsdf::{Camera, DepthFrame, DepthRange, DirtyMeshes, Mesh, TsdfError, TsdfVolume};

/// `(vertices, triangles, colors | None, normals)`.
type PyMesh<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<u32>>,
    Option<Bound<'py, PyArray2<u8>>>,
    Bound<'py, PyArray2<f32>>,
);
type PyBlockMeshes<'py> = Vec<((i64, i64, i64), PyMesh<'py>)>;

/// Sparse colour TSDF volume (see `tsdf_rs.pyi`).
#[pyclass(name = "TsdfVolume", module = "tsdf_rs")]
struct PyTsdfVolume {
    inner: TsdfVolume,
}

#[pymethods]
impl PyTsdfVolume {
    #[new]
    #[pyo3(signature = (voxel_size, sdf_trunc, fuse_color))]
    fn new(voxel_size: f32, sdf_trunc: f32, fuse_color: bool) -> PyResult<Self> {
        let inner = TsdfVolume::new(voxel_size, sdf_trunc, fuse_color).map_err(value_error)?;
        Ok(Self { inner })
    }

    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[pyo3(signature = (
        depth_hw, K_33, cam_T_world_44, color_hw3=None, *, min_depth=0.0, max_depth=f32::INFINITY
    ))]
    fn integrate(
        &mut self,
        py: Python<'_>,
        depth_hw: PyReadonlyArray2<'_, f32>,
        K_33: PyReadonlyArray2<'_, f32>,
        cam_T_world_44: PyReadonlyArray2<'_, f32>,
        color_hw3: Option<PyReadonlyArray3<'_, u8>>,
        min_depth: f32,
        max_depth: f32,
    ) -> PyResult<()> {
        let [height, width] = [depth_hw.shape()[0], depth_hw.shape()[1]];
        let depth = contiguous(&depth_hw, "depth_hw")?.to_vec();
        let camera = Camera {
            intrinsics: matrix(&K_33, "K_33")?,
            camera_from_world: matrix(&cam_T_world_44, "cam_T_world_44")?,
        };
        let color = match &color_hw3 {
            Some(array) => {
                check_shape(array, "color_hw3", &[height, width, 3])?;
                Some(contiguous(array, "color_hw3")?.to_vec())
            }
            None => None,
        };
        let frame = DepthFrame {
            depth: &depth,
            image_size: [height, width],
            color: color.as_deref(),
            depth_range: DepthRange {
                min: min_depth,
                max: max_depth,
            },
        };
        let volume = &mut self.inner;
        py.detach(|| volume.integrate(&frame, &camera))
            .map_err(value_error)
    }

    fn extract_mesh<'py>(&self, py: Python<'py>) -> PyResult<PyMesh<'py>> {
        let volume = &self.inner;
        let mesh = py.detach(|| volume.extract_mesh()).map_err(value_error)?;
        mesh_to_numpy(py, mesh)
    }

    fn extract_dirty_block_meshes<'py>(&mut self, py: Python<'py>) -> PyResult<PyBlockMeshes<'py>> {
        let volume = &mut self.inner;
        let DirtyMeshes { meshes, delivery } = py.detach(|| volume.dirty_block_meshes());
        // Confirm only after conversion: a NumPy failure drops the delivery,
        // so the meshes stay pending and the call can be retried.
        let converted = meshes
            .into_iter()
            .map(|([x, y, z], mesh)| Ok(((x, y, z), mesh_to_numpy(py, mesh)?)))
            .collect::<PyResult<PyBlockMeshes<'py>>>()?;
        delivery.confirm();
        Ok(converted)
    }
}

/// The array's elements in C order; refuses Fortran-ordered and strided
/// arrays, which `as_slice` alone would accept or read transposed.
fn contiguous<'a, T: Element, D: Dimension>(
    array: &'a PyReadonlyArray<'_, T, D>,
    name: &str,
) -> PyResult<&'a [T]> {
    let refused = || {
        PyValueError::new_err(format!(
            "{name} must be C-contiguous; pass numpy.ascontiguousarray({name})"
        ))
    };
    if !array.is_c_contiguous() {
        return Err(refused());
    }
    array.as_slice().map_err(|_| refused())
}

fn check_shape<T: Element, D: Dimension>(
    array: &PyReadonlyArray<'_, T, D>,
    name: &str,
    expected: &[usize],
) -> PyResult<()> {
    if array.shape() == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must have shape {expected:?}; got {:?}",
            array.shape()
        )))
    }
}

/// A square row-major matrix of `N = n * n` values.
fn matrix<const N: usize>(array: &PyReadonlyArray2<'_, f32>, name: &str) -> PyResult<[f32; N]> {
    let side = N.isqrt();
    check_shape(array, name, &[side, side])?;
    let values = contiguous(array, name)?;
    // The shape check fixed the length.
    values
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must hold {N} values")))
}

fn rows<T: Element, const C: usize>(
    py: Python<'_>,
    rows: Vec<[T; C]>,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    let count = rows.len();
    PyArray1::from_vec(py, rows.into_flattened()).reshape([count, C])
}

fn mesh_to_numpy(py: Python<'_>, mesh: Mesh) -> PyResult<PyMesh<'_>> {
    Ok((
        rows(py, mesh.vertices)?,
        rows(py, mesh.triangles)?,
        mesh.colors.map(|colors| rows(py, colors)).transpose()?,
        rows(py, mesh.normals)?,
    ))
}

fn value_error(error: TsdfError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn tsdf_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTsdfVolume>()
}
