//! `dpretrieval._core`: image retrieval for DPV-SLAM's classical loop closure, on kornia-rs.
//!
//! Frames arrive as their ORB features (OpenCV's, from `dpretrieval.DPRetrieval`). Each gets a kornia-bow bag of
//! words over ORB-SLAM3's `ORBvoc.txt` and an entry in kornia-staging-bow's inverted-file database; pairs of frames
//! are matched with kornia-imgproc's cross-checked brute-force Hamming matcher.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use kornia_bow::metric::Feature;
use kornia_bow::orb_slam3::{
    load_orb_slam3_vocabulary, pack_orb_descriptor, OrbVocabulary, ORB_WORDS,
};
use kornia_bow::BoW;
use kornia_imgproc::features::{hamming_distance, match_descriptors};
use kornia_staging_bow::Database as BowDatabase;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

/// Parsing `ORBvoc.txt` (1M words) takes about half a second; its kornia-bow binary form loads in 0.05 s. The
/// binary copy is written next to the text file on first load (`<vocabulary>.kbow`) and used while it is newer.
fn load_vocabulary(path: &Path) -> Result<OrbVocabulary, String> {
    let cache = PathBuf::from(format!("{}.kbow", path.display()));
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    let text_time =
        modified(path).ok_or_else(|| format!("cannot read the vocabulary {}", path.display()))?;
    if modified(&cache).is_some_and(|t: SystemTime| t >= text_time) {
        if let Ok(vocabulary) = OrbVocabulary::load(&cache.to_string_lossy()) {
            return Ok(vocabulary);
        }
    }
    let vocabulary = load_orb_slam3_vocabulary(path)
        .map_err(|e| format!("cannot load the ORB vocabulary {}: {e}", path.display()))?;
    // Write, then rename: a concurrent reader never sees half a file. A read-only directory only costs speed.
    let partial = format!("{}.{}.partial", cache.display(), std::process::id());
    if vocabulary.save(&partial).is_ok() {
        let _ = std::fs::rename(&partial, &cache);
    } else {
        let _ = std::fs::remove_file(&partial);
    }
    Ok(vocabulary)
}

/// `(train_x, train_y, query_x, query_y, hamming distance)` of one match.
type Match = (f32, f32, f32, f32, f32);

/// One frame's ORB features.
struct Frame {
    keypoints_xy: Vec<[f32; 2]>,
    descriptors: Vec<[u8; 32]>,
}

/// The frames added so far (entry id = insertion order): their features, bags of words and database.
#[pyclass(module = "dpretrieval._core")]
struct Database {
    vocabulary: OrbVocabulary,
    database: BowDatabase,
    bows: Vec<BoW>,
    frames: Vec<Frame>,
}

impl Database {
    fn frame(&self, i: usize) -> PyResult<&Frame> {
        self.frames
            .get(i)
            .ok_or_else(|| PyIndexError::new_err("index invalid"))
    }
}

#[pymethods]
impl Database {
    /// Load an ORB-SLAM3 / DBoW2 text vocabulary (`ORBvoc.txt`).
    #[new]
    fn new(py: Python<'_>, vocab_path: PathBuf) -> PyResult<Self> {
        let vocabulary = py
            .detach(|| load_vocabulary(&vocab_path))
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            vocabulary,
            database: BowDatabase::new(),
            bows: Vec::new(),
            frames: Vec::new(),
        })
    }

    /// Add a frame from its ORB features, (n, 2) float32 keypoints and C-contiguous (n, 32) uint8 descriptors;
    /// returns its entry id.
    fn add(
        &mut self,
        py: Python<'_>,
        keypoints_xy: PyReadonlyArray2<'_, f32>,
        descriptors: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<u32> {
        let (n, cols) = descriptors.as_array().dim();
        if (n > 0 && cols != 32)
            || !descriptors.is_c_contiguous()
            || keypoints_xy.as_array().dim() != (n, 2)
        {
            return Err(PyValueError::new_err(format!(
                "expected (n, 2) keypoints and C-contiguous (n, 32) descriptors, got {:?} and {:?}",
                keypoints_xy.as_array().dim(),
                (n, cols)
            )));
        }
        let keypoints_xy: Vec<[f32; 2]> = keypoints_xy
            .as_array()
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1]])
            .collect();
        let descriptors: Vec<[u8; 32]> = descriptors
            .as_slice()?
            .chunks_exact(32)
            .map(|c| c.try_into().expect("32 bytes"))
            .collect();
        let bow = py.detach(|| {
            let packed: Vec<Feature<u64, ORB_WORDS>> =
                descriptors.iter().map(pack_orb_descriptor).collect();
            // A frame without features gets an empty bag of words (kornia-bow reports it as an error).
            self.vocabulary
                .transform(&packed)
                .unwrap_or(BoW(Vec::new()))
        });
        let entry = self.database.add(&bow);
        self.bows.push(bow);
        self.frames.push(Frame {
            keypoints_xy,
            descriptors,
        });
        Ok(entry)
    }

    /// The `max_results` entries most similar to entry `i`, best first, as `(entry, L1 score)`.
    fn query(&self, py: Python<'_>, i: usize, max_results: usize) -> PyResult<Vec<(u32, f32)>> {
        let bow = self
            .bows
            .get(i)
            .ok_or_else(|| PyIndexError::new_err("index invalid"))?;
        let results = py.detach(|| self.database.query(bow, max_results));
        Ok(results.into_iter().map(|r| (r.entry, r.score)).collect())
    }

    /// Mutual-nearest Hamming matches of frame `qi`'s ORB features against frame `ti`'s, as
    /// `(train_x, train_y, query_x, query_y, distance)` in query order.
    fn match_pair(&self, py: Python<'_>, ti: usize, qi: usize) -> PyResult<Vec<Match>> {
        let (train, query) = (self.frame(ti)?, self.frame(qi)?);
        Ok(py.detach(|| {
            match_descriptors(&query.descriptors, &train.descriptors, None, true, None)
                .into_iter()
                .map(|(q, t)| {
                    let ([tx, ty], [qx, qy]) = (train.keypoints_xy[t], query.keypoints_xy[q]);
                    (
                        tx,
                        ty,
                        qx,
                        qy,
                        hamming_distance(&query.descriptors[q], &train.descriptors[t]) as f32,
                    )
                })
                .collect()
        }))
    }

    fn __len__(&self) -> usize {
        self.frames.len()
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    Ok(())
}
