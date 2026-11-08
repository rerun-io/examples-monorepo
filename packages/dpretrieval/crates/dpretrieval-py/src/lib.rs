//! `dpretrieval._core`: the DBoW2 database and the cross-checked Hamming matcher behind `dpretrieval.DPRetrieval`.
//! Descriptors arrive as C-contiguous (n, 32) uint8 arrays (OpenCV ORB's output).

use std::path::PathBuf;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

fn descriptors(array: &PyReadonlyArray2<'_, u8>) -> PyResult<Vec<dbow::Descriptor>> {
    let (rows, cols) = array.as_array().dim();
    if rows == 0 {
        return Ok(Vec::new());
    }
    if cols != 32 || !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "descriptors must be a C-contiguous (n, 32) uint8 array, got shape ({rows}, {cols})"
        )));
    }
    let data = array.as_slice()?;
    Ok(data
        .chunks_exact(32)
        .map(|c| c.try_into().expect("32 bytes"))
        .collect())
}

/// A DBoW2 database over an ORB vocabulary; entries are the bags of words of the images added, in order.
#[pyclass(module = "dpretrieval._core")]
struct Database {
    vocabulary: dbow::Vocabulary,
    database: dbow::Database,
    bows: Vec<dbow::BowVector>,
}

#[pymethods]
impl Database {
    /// Load a DBoW2 text vocabulary (`ORBvoc.txt`).
    #[new]
    fn new(py: Python<'_>, vocab_path: PathBuf) -> PyResult<Self> {
        let vocabulary = py
            .detach(|| dbow::Vocabulary::load(&vocab_path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let database = dbow::Database::new(&vocabulary);
        Ok(Self {
            vocabulary,
            database,
            bows: Vec::new(),
        })
    }

    /// Add one image's descriptors; returns its entry id.
    fn add(&mut self, py: Python<'_>, descs: PyReadonlyArray2<'_, u8>) -> PyResult<u32> {
        let descs = descriptors(&descs)?;
        let bow = py.detach(|| self.vocabulary.transform(&descs));
        let entry = self.database.add(&bow);
        self.bows.push(bow);
        Ok(entry)
    }

    /// The `max_results` best entries for entry `i`'s bag of words, best first, as `(entry, score)`.
    fn query(&self, py: Python<'_>, i: usize, max_results: usize) -> PyResult<Vec<(u32, f64)>> {
        let bow = self
            .bows
            .get(i)
            .ok_or_else(|| PyIndexError::new_err("index invalid"))?;
        let results = py.detach(|| self.database.query(bow, max_results));
        Ok(results.into_iter().map(|r| (r.entry, r.score)).collect())
    }

    fn __len__(&self) -> usize {
        self.database.len()
    }
}

/// `cv::BFMatcher(NORM_HAMMING, crossCheck=true).knnMatch(query, train, 1)` as `(query idx, train idx, distance)`.
#[pyfunction]
fn cross_check_match(
    py: Python<'_>,
    query: PyReadonlyArray2<'_, u8>,
    train: PyReadonlyArray2<'_, u8>,
) -> PyResult<Vec<(usize, usize, u32)>> {
    let (query, train) = (descriptors(&query)?, descriptors(&train)?);
    Ok(py.detach(|| dbow::cross_check_match(&query, &train)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    m.add_function(wrap_pyfunction!(cross_check_match, m)?)?;
    Ok(())
}
