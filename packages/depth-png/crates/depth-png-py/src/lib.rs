//! The `depth_png` Python module: NumPy bindings for the `depth-png` core.
//!
//! `encode` takes a C-contiguous `(h, w)` uint16 or uint8 array. It is copied before the GIL is
//! released (for uint16 that copy is the big-endian conversion PNG needs anyway), so another
//! Python thread cannot write it mid-call. `decode` reads the `bytes` in place, since they are
//! immutable, and hands the decoded buffer to NumPy without a copy.

use depth_png::{DepthImage, DepthPngError};
use numpy::{
    Element, PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Encode a `(h, w)` uint16 or uint8 depth image as a lossless greyscale PNG with the Up row filter.
#[pyfunction]
#[pyo3(signature = (depth, level = 4))]
fn encode<'py>(
    py: Python<'py>,
    depth: &Bound<'py, PyUntypedArray>,
    level: u8,
) -> PyResult<Bound<'py, PyBytes>> {
    let png_bytes = if let Ok(array) = depth.extract::<PyReadonlyArray2<'_, u16>>() {
        let (samples, size) = owned_samples(&array)?;
        py.detach(|| depth_png::encode_u16(&samples, size, level))
    } else if let Ok(array) = depth.extract::<PyReadonlyArray2<'_, u8>>() {
        let (samples, size) = owned_samples(&array)?;
        py.detach(|| depth_png::encode_u8(&samples, size, level))
    } else {
        return Err(PyTypeError::new_err(format!(
            "depth must be a 2D uint16 or uint8 array; got {}D {}",
            depth.ndim(),
            depth.dtype()
        )));
    }
    .map_err(value_error)?;
    Ok(PyBytes::new(py, &png_bytes))
}

/// Decode an 8- or 16-bit greyscale PNG into a `(h, w)` uint8 or uint16 array.
#[pyfunction]
fn decode<'py>(py: Python<'py>, png: &[u8]) -> PyResult<Bound<'py, PyUntypedArray>> {
    let image = py.detach(|| depth_png::decode(png)).map_err(value_error)?;
    Ok(match image {
        DepthImage::U8 {
            width,
            height,
            data,
        } => to_numpy(py, data, width, height)?,
        DepthImage::U16 {
            width,
            height,
            data,
        } => to_numpy(py, data, width, height)?,
    })
}

/// The array's samples in C order, copied, and its `[width, height]`; refuses Fortran-ordered and strided arrays.
fn owned_samples<T: Element + Copy>(
    array: &PyReadonlyArray2<'_, T>,
) -> PyResult<(Vec<T>, [u32; 2])> {
    let [height, width] = [array.shape()[0], array.shape()[1]];
    let size = [dimension(width, "width")?, dimension(height, "height")?];
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "depth must be C-contiguous; pass numpy.ascontiguousarray(depth)",
        ));
    }
    let samples = array.as_slice().map_err(|_| {
        PyValueError::new_err("depth must be C-contiguous; pass numpy.ascontiguousarray(depth)")
    })?;
    Ok((samples.to_vec(), size))
}

fn dimension(pixels: usize, name: &str) -> PyResult<u32> {
    u32::try_from(pixels)
        .ok()
        .filter(|&pixels| pixels > 0)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "depth {name} must be 1 to {} pixels; got {pixels}",
                u32::MAX
            ))
        })
}

fn to_numpy<T: Element>(
    py: Python<'_>,
    data: Vec<T>,
    width: u32,
    height: u32,
) -> PyResult<Bound<'_, PyUntypedArray>> {
    let image =
        PyArray1::from_vec(py, data).reshape::<[usize; 2]>([height as usize, width as usize])?;
    Ok(image.as_untyped().clone())
}

fn value_error(error: DepthPngError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

// Not named `depth_png`, which would shadow the core crate of that name.
#[pymodule(name = "depth_png")]
fn depth_png_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(encode, module)?)?;
    module.add_function(wrap_pyfunction!(decode, module)?)
}
