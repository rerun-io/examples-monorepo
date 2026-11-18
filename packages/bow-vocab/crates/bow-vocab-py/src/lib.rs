//! The `bow_vocab` Python module: kornia-staging-bow's seeded hierarchical k-means vocabulary for float descriptors
//! (`kornia_staging_bow::vocabulary`) and the dense DBoW2 L1 score matrix of a set of images.
//!
//! Descriptor arrays are borrowed, not copied: a C-contiguous `n x D` float32 array is read as `n` rows of
//! `[f32; D]`. `D` is 64, 128 or 256. The GIL is released while Rust works. Every refusal raises `ValueError`.

use kornia_staging_bow::vocabulary::{similarity_matrix, Vocabulary as Tree, VocabularyConfig};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Borrow a C-contiguous `n x D` float32 array as `n` rows of `[f32; D]`.
fn rows<'a, const D: usize>(array: &'a PyReadonlyArray2<'_, f32>) -> PyResult<&'a [[f32; D]]> {
    let (_, cols) = array.as_array().dim();
    if cols != D {
        return Err(PyValueError::new_err(format!(
            "expected {D} columns, got {cols}"
        )));
    }
    let (chunks, rest) = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("descriptors must be a C-contiguous float32 array"))?
        .as_chunks::<D>();
    debug_assert!(rest.is_empty());
    Ok(chunks)
}

/// Train a `D`-dimensional tree on `sample` without the GIL.
fn train<const D: usize>(
    py: Python<'_>,
    sample: &PyReadonlyArray2<'_, f32>,
    config: &VocabularyConfig,
) -> PyResult<Tree<D>> {
    let data = rows::<D>(sample)?;
    py.detach(|| Tree::train(data, config))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The trained tree, for one of the supported descriptor sizes.
enum Inner {
    D64(Tree<64>),
    D128(Tree<128>),
    D256(Tree<256>),
}

/// Run `$body` with `$tree` bound to the tree of the matching size and `$d` to that size as a const.
macro_rules! with_tree {
    ($inner:expr, $tree:ident, $d:ident, $body:expr) => {
        match $inner {
            Inner::D64($tree) => {
                const $d: usize = 64;
                $body
            }
            Inner::D128($tree) => {
                const $d: usize = 128;
                $body
            }
            Inner::D256($tree) => {
                const $d: usize = 256;
                $body
            }
        }
    };
}

/// A seeded hierarchical k-means vocabulary over float descriptors, trained once per run.
#[pyclass(module = "bow_vocab", frozen)]
struct Vocabulary {
    inner: Inner,
}

#[pymethods]
impl Vocabulary {
    /// Train on `sample` (C-contiguous `n x D` float32, D in 64/128/256): k-means++ then up to `iterations` Lloyd
    /// iterations per node, `branching` children per node, `depth` levels (lowered when the sample cannot fill
    /// them). The same `seed` gives the same tree for any thread count.
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
        let inner = match sample.as_array().dim().1 {
            64 => Inner::D64(train::<64>(py, &sample, &config)?),
            128 => Inner::D128(train::<128>(py, &sample, &config)?),
            256 => Inner::D256(train::<256>(py, &sample, &config)?),
            d => {
                return Err(PyValueError::new_err(format!(
                    "descriptor size {d} is not supported (64, 128 or 256)"
                )))
            }
        };
        Ok(Self { inner })
    }

    /// Size of the word-id range, `branching ** depth` (depth as built).
    #[getter]
    fn n_words(&self) -> usize {
        with_tree!(&self.inner, tree, _D, tree.n_words())
    }

    /// The word of every row of `descriptors`, by greedy descent.
    fn words<'py>(
        &self,
        py: Python<'py>,
        descriptors: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let words = with_tree!(&self.inner, tree, D, {
            let data = rows::<D>(&descriptors)?;
            py.detach(|| tree.words(data))
        });
        Ok(PyArray1::from_vec(py, words))
    }

    /// The symmetric `n x n` float32 DBoW2 L1 score matrix of `images` (one `n_i x D` float32 array each): every
    /// descriptor's word, TF-IDF bags (`ln(n_images / df)`, L1-normalized), scores in `[0, 1]`, 1.0 on the diagonal,
    /// 0 for an image whose bag is empty.
    fn similarity<'py>(
        &self,
        py: Python<'py>,
        images: Vec<PyReadonlyArray2<'py, f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = images.len();
        let scores = with_tree!(&self.inner, tree, D, {
            let data: Vec<&[[f32; D]]> = images.iter().map(rows::<D>).collect::<PyResult<_>>()?;
            py.detach(|| {
                let words: Vec<Vec<u32>> = data.par_iter().map(|image| tree.words(image)).collect();
                similarity_matrix(&words)
            })
        });
        PyArray1::from_vec(py, scores).reshape([n, n])
    }
}

#[pymodule]
fn bow_vocab(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Vocabulary>()
}
