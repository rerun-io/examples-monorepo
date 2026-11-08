//! `asmk_rs`: the API of ASMK's Cython module `asmk.hamming`, on the `hamming` crate.
//!
//! Inputs must be C-contiguous arrays of the Cython signature's dtype (float32 rows, uint32 packed rows), as the
//! typed memoryviews require; anything else raises.

use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn contiguous<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    array: &'a numpy::PyReadonlyArray<'_, T, D>,
) -> PyResult<&'a [T]> {
    // `as_slice` also accepts Fortran order; the Cython memoryviews (`[:, ::1]`) take C order only.
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err("ndarray is not C-contiguous"));
    }
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err("ndarray is not C-contiguous"))
}

/// Binarize a 1-D float32 array by `arr > threshold` and pack it into a uint32 array (one word per started 32 bits).
#[pyfunction]
#[pyo3(signature = (arr, threshold = 0))]
fn binarize_and_pack<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f32>,
    threshold: i32,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let row = contiguous(&arr)?;
    let mut out = Array1::<u32>::zeros(hamming::packed_len(row.len()));
    hamming::binarize_and_pack(
        row,
        threshold,
        out.as_slice_mut().expect("fresh array is contiguous"),
    );
    Ok(out.into_pyarray(py))
}

/// Binarize every row of a 2-D float32 array by `arr > threshold` and pack it into uint32 words.
#[pyfunction]
#[pyo3(name = "binarize_and_pack_2D", signature = (arr, threshold = 0))]
fn binarize_and_pack_2d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
    threshold: i32,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let (rows, cols) = arr.as_array().dim();
    let data = contiguous(&arr)?;
    let words = hamming::packed_len(cols);
    let mut out = Array2::<u32>::zeros((rows, words));
    let out_data = out.as_slice_mut().expect("fresh array is contiguous");
    if cols > 0 {
        for (row, packed) in data.chunks(cols).zip(out_data.chunks_mut(words)) {
            hamming::binarize_and_pack(row, threshold, packed);
        }
    }
    Ok(out.into_pyarray(py))
}

/// Hamming distance of two packed uint32 arrays divided by `normalization` (0 = the number of bits).
#[pyfunction]
#[pyo3(signature = (n1, n2, normalization = 0.0))]
fn hamming_dist_packed(
    n1: PyReadonlyArray1<'_, u32>,
    n2: PyReadonlyArray1<'_, u32>,
    normalization: f32,
) -> PyResult<f32> {
    let (a, b) = (contiguous(&n1)?, contiguous(&n2)?);
    if a.len() != b.len() {
        return Err(PyValueError::new_err(
            "packed arrays must have the same length",
        ));
    }
    Ok(hamming::hamming_dist_packed(a, b, normalization))
}

/// Hamming distances between every row of arr1 and every row of arr2 (packed uint32), as a float32 (n1, n2) array.
#[pyfunction]
#[pyo3(signature = (arr1, arr2, normalization = 0.0))]
fn hamming_cdist_packed<'py>(
    py: Python<'py>,
    arr1: PyReadonlyArray2<'py, u32>,
    arr2: PyReadonlyArray2<'py, u32>,
    normalization: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let ((n1, words), (n2, words2)) = (arr1.as_array().dim(), arr2.as_array().dim());
    if words != words2 {
        return Err(PyValueError::new_err(
            "packed arrays must have the same number of words",
        ));
    }
    let (a, b) = (contiguous(&arr1)?, contiguous(&arr2)?);
    let mut out = Array2::<f32>::zeros((n1, n2));
    if words > 0 {
        for (row, dists) in a.chunks(words).zip(out.rows_mut()) {
            for (other, dist) in b.chunks(words).zip(dists) {
                *dist = hamming::hamming_dist_packed(row, other, normalization);
            }
        }
    } else {
        // Zero-width rows: 0 / 0 in float32, as the Cython module.
        out.fill(
            0.0 / if normalization == 0.0 {
                0.0
            } else {
                normalization
            },
        );
    }
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn asmk_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binarize_and_pack, m)?)?;
    m.add_function(wrap_pyfunction!(binarize_and_pack_2d, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_dist_packed, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_cdist_packed, m)?)?;
    Ok(())
}
