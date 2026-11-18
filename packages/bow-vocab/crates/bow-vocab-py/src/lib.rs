//! The `bow_vocab` Python module: kornia-staging-bow's seeded hierarchical k-means vocabulary for float descriptors
//! (`kornia_staging_bow::vocabulary`), TF-IDF bags of words and DBoW2 L1 scores through the inverted-file
//! `kornia_staging_bow::Database`.
//!
//! Descriptor arrays are borrowed, not copied: a C-contiguous `n x D` float32 array is read as `n` rows of
//! `[f32; D]`. `D` is 64, 128 or 256. The GIL is released while Rust works. Every refusal raises `ValueError`.

use kornia_bow::BoW;
use kornia_staging_bow::vocabulary::{tf_idf, Vocabulary as Tree, VocabularyConfig};
use kornia_staging_bow::Database;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

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
    #[pyo3(signature = (sample, branching = 10, depth = 5, iterations = 10, seed = 0))]
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
            64 => {
                let data = rows::<64>(&sample)?;
                Inner::D64(
                    py.detach(|| Tree::train(data, &config))
                        .map_err(value_error)?,
                )
            }
            128 => {
                let data = rows::<128>(&sample)?;
                Inner::D128(
                    py.detach(|| Tree::train(data, &config))
                        .map_err(value_error)?,
                )
            }
            256 => {
                let data = rows::<256>(&sample)?;
                Inner::D256(
                    py.detach(|| Tree::train(data, &config))
                        .map_err(value_error)?,
                )
            }
            d => {
                return Err(PyValueError::new_err(format!(
                    "descriptor size {d} is not supported (64, 128 or 256)"
                )))
            }
        };
        Ok(Self { inner })
    }

    /// Children per node.
    #[getter]
    fn branching(&self) -> usize {
        with_tree!(&self.inner, tree, _D, tree.branching())
    }

    /// Levels actually built.
    #[getter]
    fn depth(&self) -> usize {
        with_tree!(&self.inner, tree, _D, tree.depth())
    }

    /// Size of the word-id range, `branching ** depth`.
    #[getter]
    fn n_words(&self) -> usize {
        with_tree!(&self.inner, tree, _D, tree.n_words())
    }

    /// Columns the descriptors must have.
    #[getter]
    fn descriptor_dim(&self) -> usize {
        with_tree!(&self.inner, _tree, D, D)
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

    /// Index a set of images, one `n_i x D` float32 array each: their words, TF-IDF bags of words
    /// (`ln(n_images / df)`, L1-normalized) and an inverted-file database. Entry `i` is `images[i]`.
    fn index(&self, py: Python<'_>, images: Vec<PyReadonlyArray2<'_, f32>>) -> PyResult<BowIndex> {
        let words: Vec<Vec<u32>> = with_tree!(&self.inner, tree, D, {
            let data: Vec<&[[f32; D]]> = images.iter().map(rows::<D>).collect::<PyResult<_>>()?;
            py.detach(|| data.par_iter().map(|image| tree.words(image)).collect())
        });
        py.detach(|| {
            let mut offsets = Vec::with_capacity(words.len() + 1);
            offsets.push(0usize);
            for image in &words {
                offsets.push(offsets.last().copied().unwrap_or(0) + image.len());
            }
            let flat: Vec<u32> = words.concat();
            let bows = tf_idf(&flat, &offsets).map_err(value_error)?;
            let mut database = Database::new();
            for bow in &bows {
                database.add(bow);
            }
            Ok(BowIndex { bows, database })
        })
    }
}

/// TF-IDF bags of words of a fixed set of images in an inverted-file database.
#[pyclass(module = "bow_vocab", frozen)]
struct BowIndex {
    bows: Vec<BoW>,
    database: Database,
}

impl BowIndex {
    fn bow(&self, i: usize) -> PyResult<&BoW> {
        self.bows
            .get(i)
            .ok_or_else(|| PyIndexError::new_err(format!("image {i} is not indexed")))
    }
}

#[pymethods]
impl BowIndex {
    fn __len__(&self) -> usize {
        self.bows.len()
    }

    /// The symmetric `n x n` float32 matrix of DBoW2 L1 scores (`sum_w min(q_w, d_w)`), 1.0 on the diagonal, in
    /// `[0, 1]`. Each row is one inverted-file query; rows run in parallel.
    fn similarity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = self.bows.len();
        let scores = py.detach(|| {
            let mut scores = vec![0.0f32; n * n];
            if n > 0 {
                scores.par_chunks_mut(n).enumerate().for_each(|(q, row)| {
                    for hit in self.database.query(&self.bows[q], 0) {
                        row[hit.entry as usize] = hit.score.clamp(0.0, 1.0);
                    }
                    row[q] = 1.0;
                });
            }
            scores
        });
        PyArray1::from_vec(py, scores).reshape([n, n])
    }

    /// The DBoW2 L1 score of images `a` and `b`, as `similarity()[a, b]`: 1.0 when `a == b`, 0.0 when either bag
    /// is empty (kornia-bow's `l1_similarity` would read an empty bag as 0.5 or 1).
    fn score(&self, a: usize, b: usize) -> PyResult<f32> {
        let (first, second) = (self.bow(a)?, self.bow(b)?);
        if a == b {
            return Ok(1.0);
        }
        if first.0.is_empty() || second.0.is_empty() {
            return Ok(0.0);
        }
        Ok(first.l1_similarity(second).clamp(0.0, 1.0))
    }

    /// The `top_k` images most similar to image `i` (itself excluded), best first, as `(image, score)`.
    fn query(&self, py: Python<'_>, i: usize, top_k: usize) -> PyResult<Vec<(usize, f32)>> {
        let bow = self.bow(i)?;
        let hits = py.detach(|| self.database.query_where(bow, top_k, |e| e as usize != i));
        Ok(hits
            .into_iter()
            .map(|h| (h.entry as usize, h.score))
            .collect())
    }
}

#[pymodule]
fn bow_vocab(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Vocabulary>()?;
    module.add_class::<BowIndex>()
}
