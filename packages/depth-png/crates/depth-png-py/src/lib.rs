//! The `depth_png` Python module: NumPy bindings for the `depth-png` core.
//!
//! Arrays are read and written in place with the GIL released, as NumPy's own functions do, so
//! another Python thread must not write them during a call. `decode` hands its buffer to NumPy
//! without a copy; `decode_into` writes into a caller's array, e.g. `pinned_tensor.numpy()`.

use depth_png::{DepthImage, DepthPngError, EncodeOptions};
use numpy::{
    Element, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2,
    PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Encode a `(h, w)` uint16 or uint8 depth image as a lossless greyscale PNG with the Up row filter.
#[pyfunction]
#[pyo3(signature = (depth, level = 4, threads = None))]
fn encode<'py>(
    py: Python<'py>,
    depth: &Bound<'py, PyUntypedArray>,
    level: u8,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyBytes>> {
    let options = EncodeOptions {
        level,
        threads: threads.unwrap_or(0),
        band_rows: None,
    };
    let png_bytes = if let Ok(array) = depth.extract::<PyReadonlyArray2<'_, u16>>() {
        let (samples, size) = samples_and_size(&array)?;
        py.detach(|| depth_png::encode_u16(samples, size, &options))
    } else if let Ok(array) = depth.extract::<PyReadonlyArray2<'_, u8>>() {
        let (samples, size) = samples_and_size(&array)?;
        py.detach(|| depth_png::encode_u8(samples, size, &options))
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

/// Decode an 8- or 16-bit greyscale PNG (bytes or a 1D uint8 array) into a `(h, w)` uint8 or uint16 array.
#[pyfunction]
#[pyo3(signature = (png, threads = None))]
fn decode<'py>(
    py: Python<'py>,
    png: &Bound<'py, PyAny>,
    threads: Option<usize>,
) -> PyResult<Bound<'py, PyUntypedArray>> {
    let threads = threads.unwrap_or(0);
    let image = with_png_bytes(png, |png_bytes| {
        py.detach(|| depth_png::decode(png_bytes, threads))
            .map_err(value_error)
    })?;
    match image {
        DepthImage::U8 {
            width,
            height,
            data,
        } => to_numpy(py, data, width, height),
        DepthImage::U16 {
            width,
            height,
            data,
        } => to_numpy(py, data, width, height),
    }
}

/// Decode an 8- or 16-bit greyscale PNG into `out`, a C-contiguous `(h, w)` uint8 or uint16 array of the image's size.
#[pyfunction]
#[pyo3(signature = (png, out, threads = None))]
fn decode_into(
    py: Python<'_>,
    png: &Bound<'_, PyAny>,
    out: &Bound<'_, PyUntypedArray>,
    threads: Option<usize>,
) -> PyResult<()> {
    let threads = threads.unwrap_or(0);
    with_png_bytes(png, |png_bytes| {
        if let Ok(mut array) = out.extract::<PyReadwriteArray2<'_, u16>>() {
            let shape = (array.shape()[0], array.shape()[1]);
            let samples = writable_samples(&mut array)?;
            let size = py
                .detach(|| depth_png::decode_into_u16(png_bytes, samples, threads))
                .map_err(value_error)?;
            check_shape(size, shape)
        } else if let Ok(mut array) = out.extract::<PyReadwriteArray2<'_, u8>>() {
            let shape = (array.shape()[0], array.shape()[1]);
            let samples = writable_samples(&mut array)?;
            let size = py
                .detach(|| depth_png::decode_into_u8(png_bytes, samples, threads))
                .map_err(value_error)?;
            check_shape(size, shape)
        } else {
            Err(PyTypeError::new_err(format!(
                "out must be a writable 2D uint16 or uint8 array; got {}D {}",
                out.ndim(),
                out.dtype()
            )))
        }
    })
}

/// `(height, width, bit_depth)` of an 8- or 16-bit greyscale PNG, read from its header alone.
#[pyfunction]
fn header(png: &Bound<'_, PyAny>) -> PyResult<(u32, u32, u8)> {
    with_png_bytes(png, |png_bytes| {
        let (width, height, sixteen_bit) = depth_png::header(png_bytes).map_err(value_error)?;
        Ok((height, width, if sixteen_bit { 16 } else { 8 }))
    })
}

/// Runs `body` on the PNG's bytes, read in place from `bytes` or a C-contiguous 1D uint8 array.
fn with_png_bytes<T>(
    png: &Bound<'_, PyAny>,
    body: impl FnOnce(&[u8]) -> PyResult<T>,
) -> PyResult<T> {
    if let Ok(bytes) = png.cast::<PyBytes>() {
        return body(bytes.as_bytes());
    }
    if let Ok(array) = png.extract::<PyReadonlyArray1<'_, u8>>() {
        return body(array.as_slice().map_err(|_| not_contiguous("png"))?);
    }
    Err(PyTypeError::new_err(
        "png must be bytes or a 1D uint8 array",
    ))
}

/// The array's samples in C order and its `[width, height]`; refuses Fortran-ordered and strided arrays.
fn samples_and_size<'a, T: Element>(
    array: &'a PyReadonlyArray2<'_, T>,
) -> PyResult<(&'a [T], [u32; 2])> {
    let [height, width] = [array.shape()[0], array.shape()[1]];
    let size = [dimension(width, "width")?, dimension(height, "height")?];
    if !array.is_c_contiguous() {
        return Err(not_contiguous("depth"));
    }
    let samples = array.as_slice().map_err(|_| not_contiguous("depth"))?;
    Ok((samples, size))
}

fn writable_samples<'a, T: Element>(
    array: &'a mut PyReadwriteArray2<'_, T>,
) -> PyResult<&'a mut [T]> {
    if !array.is_c_contiguous() {
        return Err(not_contiguous("out"));
    }
    array.as_slice_mut().map_err(|_| not_contiguous("out"))
}

/// `decode_into_*` refused an output of the wrong length; this refuses a transposed one.
fn check_shape([width, height]: [u32; 2], shape: (usize, usize)) -> PyResult<()> {
    if shape == (height as usize, width as usize) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "out has shape {shape:?}; the PNG is ({height}, {width})"
        )))
    }
}

fn not_contiguous(name: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{name} must be C-contiguous; pass numpy.ascontiguousarray({name})"
    ))
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
    module.add_function(wrap_pyfunction!(decode, module)?)?;
    module.add_function(wrap_pyfunction!(decode_into, module)?)?;
    module.add_function(wrap_pyfunction!(header, module)?)
}
