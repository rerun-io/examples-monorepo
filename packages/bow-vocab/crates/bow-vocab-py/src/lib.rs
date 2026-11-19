//! The `bow_vocab` Python module: kornia-staging-bow's seeded hierarchical k-means vocabulary for float descriptors
//! (`kornia_staging_bow::vocabulary`) and the dense DBoW2 L1 score matrix of a set of images
//! (`kornia_staging_bow::l1_similarity_matrix`).
//!
//! Descriptor arrays are borrowed, not copied: a C-contiguous `n x dim` float32 array of any width is read in place.
//! Rust reads the arrays with the GIL released, so the caller must not write to them (or to memory they alias)
//! during a call. A wrong width or a non-C-contiguous array raises `ValueError`; an array of another dtype (float64,
//! say) raises `TypeError` from PyO3's argument extraction.

use kornia_staging_bow::l1_similarity_matrix;
use kornia_staging_bow::vocabulary::{tf_idf, Vocabulary as Tree, VocabularyConfig};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Borrow a C-contiguous float32 array of `dim` columns as its flat row-major values; refuse any other width or
/// layout (a Fortran-ordered or strided array would otherwise be read with its rows mixed). `dim = None` accepts
/// any width of at least 1 and returns it.
fn rows<'a>(
    array: &'a PyReadonlyArray2<'_, f32>,
    dim: Option<usize>,
) -> PyResult<(&'a [f32], usize)> {
    let cols = array.shape()[1];
    match dim {
        Some(d) if d != cols => {
            return Err(PyValueError::new_err(format!(
                "expected {d} columns, got {cols}"
            )))
        }
        None if cols == 0 => {
            return Err(PyValueError::new_err(
                "descriptors need at least one column",
            ))
        }
        _ => {}
    }
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "descriptors must be a C-contiguous float32 array",
        ));
    }
    let values = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("descriptors must be a C-contiguous float32 array"))?;
    Ok((values, cols))
}

/// A seeded hierarchical k-means vocabulary over float descriptors, trained once per run.
#[pyclass(module = "bow_vocab", frozen)]
struct Vocabulary {
    tree: Tree,
}

#[pymethods]
impl Vocabulary {
    /// Train on `sample` (C-contiguous `n x dim` float32): k-means++ then up to `iterations` Lloyd iterations per
    /// node, `branching` children per node, `depth` levels (lowered when the sample cannot fill them). The same
    /// `seed` gives the same tree for any thread count.
    #[new]
    #[pyo3(signature = (sample, *, branching, depth, iterations, seed))]
    fn new(
        py: Python<'_>,
        sample: PyReadonlyArray2<'_, f32>,
        branching: usize,
        depth: usize,
        iterations: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let config = VocabularyConfig {
            branching,
            depth,
            iterations,
            seed,
        };
        let (data, dim) = rows(&sample, None)?;
        let tree = py
            .detach(|| Tree::train(data, dim, &config))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { tree })
    }

    /// Size of the word-id range, `branching ** depth` (depth as built).
    #[getter]
    fn n_words(&self) -> usize {
        self.tree.n_words()
    }

    /// The word of every row of `descriptors`, by greedy descent.
    fn words<'py>(
        &self,
        py: Python<'py>,
        descriptors: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let (data, _) = rows(&descriptors, Some(self.tree.dim()))?;
        let words = py.detach(|| self.tree.words(data));
        Ok(PyArray1::from_vec(py, words))
    }

    /// The symmetric `n x n` float32 DBoW2 L1 score matrix of `images` (one `n_i x dim` float32 array each): every
    /// descriptor's word, TF-IDF bags (`ln(n_images / df)`, L1-normalized), scores in `[0, 1]`, 1.0 on the diagonal,
    /// 0 for an image whose bag is empty.
    fn similarity<'py>(
        &self,
        py: Python<'py>,
        images: Vec<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = images.len();
        let data: Vec<&[f32]> = images
            .iter()
            .map(|image| rows(image, Some(self.tree.dim())).map(|(values, _)| values))
            .collect::<PyResult<_>>()?;
        let scores = py.detach(|| {
            let words: Vec<Vec<u32>> = data
                .par_iter()
                .map(|image| self.tree.words(image))
                .collect();
            l1_similarity_matrix(&tf_idf(&words))
        });
        PyArray1::from_vec(py, scores).reshape([n, n])
    }
}

#[pymodule]
fn bow_vocab(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Vocabulary>()
}
