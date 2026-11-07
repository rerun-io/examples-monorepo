//! Deterministic sparse TSDF fusion and marching-cubes extraction.
//!
//! The Rust API contains the core algorithms. This crate also builds a PyO3
//! extension with the same volume and dense marching-cubes operations.

#![deny(missing_docs)]

mod error;
mod marching_cubes;
mod tables;
mod volume;

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub use error::TsdfError;
pub use marching_cubes::{marching_cubes, Mesh};
pub use volume::TsdfVolume;

type PyMesh<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u32>>);
type PyColoredMesh<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<u32>>,
    Option<Bound<'py, PyArray2<u8>>>,
);
type PyBlockMeshes<'py> = Vec<((i64, i64, i64), PyColoredMesh<'py>)>;

#[pyclass(name = "TsdfVolume")]
struct PyTsdfVolume {
    inner: TsdfVolume,
}

#[pymethods]
impl PyTsdfVolume {
    #[new]
    #[pyo3(signature = (voxel_size, sdf_trunc, fuse_color))]
    fn new(voxel_size: f32, sdf_trunc: f32, fuse_color: bool) -> PyResult<Self> {
        Ok(Self {
            inner: TsdfVolume::new(voxel_size, sdf_trunc, fuse_color).map_err(py_error)?,
        })
    }

    #[allow(non_snake_case)]
    #[pyo3(signature = (depth_hw, K_33, cam_T_world_44, color_hw3=None))]
    fn integrate(
        &mut self,
        depth_hw: PyReadonlyArray2<'_, f32>,
        K_33: PyReadonlyArray2<'_, f32>,
        cam_T_world_44: PyReadonlyArray2<'_, f32>,
        color_hw3: Option<PyReadonlyArray3<'_, u8>>,
    ) -> PyResult<()> {
        let depth_shape = depth_hw.shape();
        let image_size = [depth_shape[0], depth_shape[1]];
        let depth = depth_hw
            .as_slice()
            .map_err(|_| py_error(TsdfError::NonContiguous { name: "depth_hw" }))?;
        let intrinsics = read_matrix::<9>("K_33", &K_33, [3, 3])?;
        let camera_from_world = read_matrix::<16>("cam_T_world_44", &cam_T_world_44, [4, 4])?;
        let color = color_slice(color_hw3.as_ref(), image_size)?;
        self.inner
            .integrate(depth, image_size, intrinsics, camera_from_world, color)
            .map_err(py_error)
    }

    fn extract_mesh<'py>(&self, py: Python<'py>) -> PyResult<PyColoredMesh<'py>> {
        let mesh = self.inner.extract_mesh().map_err(py_error)?;
        mesh_to_python(py, mesh)
    }

    fn extract_dirty_block_meshes<'py>(&mut self, py: Python<'py>) -> PyResult<PyBlockMeshes<'py>> {
        // Convert before committing so a NumPy failure leaves the change
        // tracker intact and the call can be retried.
        let meshes = self
            .inner
            .dirty_block_meshes_uncommitted()
            .map_err(py_error)?;
        let outcomes: Vec<([i64; 3], bool)> = meshes
            .iter()
            .map(|(key, mesh)| (*key, !mesh.triangles.is_empty()))
            .collect();
        let converted: PyBlockMeshes<'py> = meshes
            .into_iter()
            .map(|(key, mesh)| Ok(((key[0], key[1], key[2]), mesh_to_python(py, mesh)?)))
            .collect::<PyResult<_>>()?;
        self.inner.commit_dirty_extraction(&outcomes);
        Ok(converted)
    }
}

#[pyfunction(name = "marching_cubes")]
fn marching_cubes_py<'py>(
    py: Python<'py>,
    tsdf_xyz: PyReadonlyArray3<'_, f32>,
    voxel_size: f32,
    origin_xyz: PyReadonlyArray1<'_, f32>,
) -> PyResult<PyMesh<'py>> {
    let values = tsdf_xyz
        .as_slice()
        .map_err(|_| py_error(TsdfError::NonContiguous { name: "tsdf_xyz" }))?;
    if origin_xyz.shape() != [3] {
        return Err(py_error(TsdfError::ShapeMismatch {
            name: "origin_xyz",
            expected: vec![3],
            got: origin_xyz.shape().to_vec(),
        }));
    }
    let origin = origin_xyz
        .as_slice()
        .map_err(|_| py_error(TsdfError::NonContiguous { name: "origin_xyz" }))?;
    let shape = tsdf_xyz.shape();
    let mesh = marching_cubes(
        values,
        [shape[0], shape[1], shape[2]],
        voxel_size,
        [origin[0], origin[1], origin[2]],
    )
    .map_err(py_error)?;
    let (vertices, triangles, _) = mesh_to_python(py, mesh)?;
    Ok((vertices, triangles))
}

fn read_matrix<const N: usize>(
    name: &'static str,
    array: &PyReadonlyArray2<'_, f32>,
    expected_shape: [usize; 2],
) -> PyResult<[f32; N]> {
    if array.shape() != expected_shape {
        return Err(py_error(TsdfError::ShapeMismatch {
            name,
            expected: expected_shape.to_vec(),
            got: array.shape().to_vec(),
        }));
    }
    let values = array
        .as_slice()
        .map_err(|_| py_error(TsdfError::NonContiguous { name }))?;
    values.try_into().map_err(|_| {
        py_error(TsdfError::ShapeMismatch {
            name,
            expected: expected_shape.to_vec(),
            got: vec![values.len()],
        })
    })
}

fn color_slice<'a>(
    color: Option<&'a PyReadonlyArray3<'_, u8>>,
    image_size: [usize; 2],
) -> PyResult<Option<&'a [u8]>> {
    let Some(array) = color else {
        return Ok(None);
    };
    let expected_shape = [image_size[0], image_size[1], 3];
    if array.shape() != expected_shape {
        return Err(py_error(TsdfError::ShapeMismatch {
            name: "color_hw3",
            expected: expected_shape.to_vec(),
            got: array.shape().to_vec(),
        }));
    }
    array
        .as_slice()
        .map(Some)
        .map_err(|_| py_error(TsdfError::NonContiguous { name: "color_hw3" }))
}

fn mesh_to_python<'py>(py: Python<'py>, mesh: Mesh) -> PyResult<PyColoredMesh<'py>> {
    let vertex_count = mesh.vertices.len();
    let triangle_count = mesh.triangles.len();
    let vertices =
        PyArray1::from_vec(py, mesh.vertices.into_flattened()).reshape([vertex_count, 3])?;
    let triangles =
        PyArray1::from_vec(py, mesh.triangles.into_flattened()).reshape([triangle_count, 3])?;
    let colors = mesh
        .colors
        .map(|rows| {
            let row_count = rows.len();
            PyArray1::from_vec(py, rows.into_flattened()).reshape([row_count, 3])
        })
        .transpose()?;
    Ok((vertices, triangles, colors))
}

fn py_error(error: TsdfError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn tsdf_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(marching_cubes_py, module)?)?;
    module.add_class::<PyTsdfVolume>()?;
    Ok(())
}
