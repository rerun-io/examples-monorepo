//! A seeded hierarchical k-means vocabulary for float descriptors, with TF-IDF bags of words.
//!
//! kornia-bow's [`kornia_bow::Vocabulary::train`] clusters each node with k-means++ seeding only (no Lloyd
//! iterations, so every centroid is a training descriptor) and an unseeded generator, and weights words by
//! `ln(training descriptors / training descriptors in the leaf)`. On learned float descriptors (ALIKED, 128-d) that
//! tree finds 9-14 % fewer true loop pairs than one refined with Lloyd iterations, and differs from run to run.
//! [`Vocabulary`] adds what is missing for a per-run retrieval vocabulary (cuSfM, arXiv 2510.15271, 3.2-3.3):
//!
//! * k-means++ seeding followed by Lloyd iterations at every node, under squared L2;
//! * one [`ChaCha8Rng`] per node, seeded from the run seed and the node's position in the tree, so the tree is
//!   the same for a given seed whatever the thread count or the order rayon runs the nodes in (every reduction
//!   sums fixed chunks in a fixed order);
//! * greedy descent to a leaf word, as DBoW2's `transform` and kornia-bow's `traverse` do;
//! * [`tf_idf`]: term counts weighted by `ln(n_images / document frequency)`, L1-normalized, ready for
//!   [`crate::Database`], [`crate::l1_similarity_matrix`] (every pair's score at once, the per-run retrieval index)
//!   and [`BoW::l1_similarity`].
//!
//! Descriptors are `[f32; D]` rows, the layout [`kornia_bow::metric::Feature<f32, D>`] dereferences to, so a
//! C-contiguous `n x D` float32 buffer is borrowed with `<[f32]>::as_chunks::<D>()` instead of copied.
//!
//! The tree keeps `branching^(level + 1)` centroid slots per level: the children of node `n` are slots
//! `n * branching .. (n + 1) * branching` of the next level, and a word id is a slot of the last level. A slot no
//! training descriptor needed repeats an earlier slot of its node, and ties go to the first slot, so it is never
//! chosen.

use std::collections::HashMap;
use std::fmt;

use kornia_bow::BoW;
use rand::rngs::ChaCha8Rng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Rows per parallel task: reductions add these chunks in index order, which keeps them deterministic.
const CHUNK: usize = 1024;

/// Settings of [`Vocabulary::train`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VocabularyConfig {
    /// Children per node, at least 2.
    pub branching: usize,
    /// Levels; the vocabulary holds up to `branching^depth` words. Lowered to the deepest tree the training set can
    /// fill (`branching^depth <= n_training`), at least 1.
    pub depth: usize,
    /// Lloyd iterations per node after k-means++ seeding; 0 keeps the seeds. A node stops early when no
    /// assignment changes.
    pub iterations: usize,
    /// Seed of every random draw.
    pub seed: u64,
}

impl Default for VocabularyConfig {
    /// 10 x 5 (100 000 words), 10 iterations, seed 0.
    fn default() -> Self {
        Self {
            branching: 10,
            depth: 5,
            iterations: 10,
            seed: 0,
        }
    }
}

/// Invalid vocabulary settings or inputs.
#[derive(Debug, PartialEq, Eq)]
pub enum VocabularyError {
    /// The branching factor must be at least 2.
    Branching(usize),
    /// The depth must be at least 1.
    Depth(usize),
    /// `branching^depth` must fit word ids (`u32`).
    TooManyWords,
    /// Training needs at least one descriptor.
    NoTrainingData,
}

impl fmt::Display for VocabularyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Branching(b) => write!(f, "branching must be at least 2, got {b}"),
            Self::Depth(d) => write!(f, "depth must be at least 1, got {d}"),
            Self::TooManyWords => write!(f, "branching^depth does not fit a u32 word id"),
            Self::NoTrainingData => write!(f, "no training descriptors"),
        }
    }
}

impl std::error::Error for VocabularyError {}

/// A trained hierarchical k-means vocabulary over `D`-dimensional float descriptors (squared L2).
///
/// ```
/// use kornia_staging_bow::vocabulary::{Vocabulary, VocabularyConfig};
///
/// let data: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]];
/// let config = VocabularyConfig { branching: 2, depth: 1, iterations: 5, seed: 7 };
/// let vocabulary = Vocabulary::train(&data, &config).unwrap();
/// assert_eq!(vocabulary.word(&[0.05, 0.0]), vocabulary.word(&[0.0, 0.0]));
/// assert_ne!(vocabulary.word(&[0.0, 0.0]), vocabulary.word(&[5.0, 5.0]));
/// ```
#[derive(Clone, Debug)]
pub struct Vocabulary<const D: usize> {
    branching: usize,
    centroids: Vec<Vec<[f32; D]>>,
}

/// Squared L2 distance.
#[inline]
fn distance<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Index of the nearest of `centroids`; the first wins ties, so a slot that repeats an earlier one is never chosen.
#[inline]
fn nearest<const D: usize>(point: &[f32; D], centroids: &[[f32; D]]) -> usize {
    let mut best = 0;
    let mut best_distance = f32::INFINITY;
    for (i, centroid) in centroids.iter().enumerate() {
        let d = distance(point, centroid);
        if d < best_distance {
            best_distance = d;
            best = i;
        }
    }
    best
}

/// SplitMix64: decorrelates the per-node seeds.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Nearest-centroid labels of `points`, in parallel chunks.
fn assign<const D: usize>(points: &[[f32; D]], centroids: &[[f32; D]], labels: &mut [u32]) {
    labels
        .par_chunks_mut(CHUNK)
        .zip(points.par_chunks(CHUNK))
        .for_each(|(l, p)| {
            for (label, point) in l.iter_mut().zip(p) {
                *label = nearest(point, centroids) as u32;
            }
        });
}

/// k-means++ seeding and up to `iterations` Lloyd iterations of `points` (at least `k` of them) into `k` clusters.
///
/// Returns the centroids and each point's label (nearest final centroid).
fn kmeans<const D: usize>(
    points: &[[f32; D]],
    k: usize,
    iterations: usize,
    rng: &mut ChaCha8Rng,
) -> (Vec<[f32; D]>, Vec<u32>) {
    let n = points.len();
    let mut centroids: Vec<[f32; D]> = Vec::with_capacity(k);
    centroids.push(points[rng.random_range(0..n)]);
    let mut nearest_distance: Vec<f32> = vec![f32::INFINITY; n];
    while centroids.len() < k {
        let last = *centroids.last().expect("one centroid");
        nearest_distance
            .par_chunks_mut(CHUNK)
            .zip(points.par_chunks(CHUNK))
            .for_each(|(d, p)| {
                for (d, p) in d.iter_mut().zip(p) {
                    *d = d.min(distance(p, &last));
                }
            });
        let total: f64 = nearest_distance.iter().map(|&d| f64::from(d)).sum();
        let chosen = if total > 0.0 {
            let mut target = rng.random::<f64>() * total;
            let mut chosen = n - 1;
            for (i, &d) in nearest_distance.iter().enumerate() {
                target -= f64::from(d);
                if target <= 0.0 && d > 0.0 {
                    chosen = i;
                    break;
                }
            }
            chosen
        } else {
            // Every point coincides with a centroid already: duplicates collapse, as in DBoW2.
            rng.random_range(0..n)
        };
        centroids.push(points[chosen]);
    }

    let mut labels = vec![0u32; n];
    assign(points, &centroids, &mut labels);
    for _ in 0..iterations {
        // Per-chunk sums, then added in chunk order: the same result for any thread count.
        let partial: Vec<(Vec<[f64; D]>, Vec<usize>)> = points
            .par_chunks(CHUNK)
            .zip(labels.par_chunks(CHUNK))
            .map(|(p, l)| {
                let mut sums = vec![[0.0f64; D]; k];
                let mut counts = vec![0usize; k];
                for (point, &label) in p.iter().zip(l) {
                    let sum = &mut sums[label as usize];
                    for (s, &x) in sum.iter_mut().zip(point) {
                        *s += f64::from(x);
                    }
                    counts[label as usize] += 1;
                }
                (sums, counts)
            })
            .collect();
        let mut sums = vec![[0.0f64; D]; k];
        let mut counts = vec![0usize; k];
        for (chunk_sums, chunk_counts) in partial {
            for ((sum, count), (chunk_sum, chunk_count)) in sums
                .iter_mut()
                .zip(counts.iter_mut())
                .zip(chunk_sums.iter().zip(&chunk_counts))
            {
                *count += chunk_count;
                for (total, x) in sum.iter_mut().zip(chunk_sum) {
                    *total += x;
                }
            }
        }
        for ((centroid, sum), &count) in centroids.iter_mut().zip(&sums).zip(&counts) {
            // An empty cluster keeps its centroid, as scipy's kmeans2 does.
            if count > 0 {
                let inverse = 1.0 / count as f64;
                for (c, s) in centroid.iter_mut().zip(sum) {
                    *c = (s * inverse) as f32;
                }
            }
        }
        let previous = labels.clone();
        assign(points, &centroids, &mut labels);
        if labels == previous {
            break;
        }
    }
    (centroids, labels)
}

impl<const D: usize> Vocabulary<D> {
    /// Train a vocabulary on `data`, level by level; the nodes of a level run in parallel.
    ///
    /// A node with fewer descriptors than `branching` keeps each of them as a child and fills the other slots with
    /// repeats of the last; a node no descriptor reached repeats its parent's centroid, so every descent ends at a
    /// leaf.
    ///
    /// # Errors
    ///
    /// [`VocabularyError`] when the settings are invalid or `data` is empty.
    pub fn train(data: &[[f32; D]], config: &VocabularyConfig) -> Result<Self, VocabularyError> {
        let branching = config.branching;
        if branching < 2 {
            return Err(VocabularyError::Branching(branching));
        }
        if config.depth < 1 {
            return Err(VocabularyError::Depth(config.depth));
        }
        if branching
            .checked_pow(config.depth as u32)
            .is_none_or(|w| w > u32::MAX as usize)
        {
            return Err(VocabularyError::TooManyWords);
        }
        if data.is_empty() {
            return Err(VocabularyError::NoTrainingData);
        }
        let mut depth = 1;
        while depth < config.depth && branching.pow(depth as u32 + 1) <= data.len() {
            depth += 1;
        }

        let mut centroids: Vec<Vec<[f32; D]>> = Vec::with_capacity(depth);
        let mut node_of: Vec<u32> = vec![0; data.len()];
        let mut n_nodes = 1usize;
        let mut first_node_id = 0u64; // nodes above this level, so (level, node) maps to one tree-wide id
        for _ in 0..depth {
            // Counting sort of the descriptors by node: each node's members are contiguous and in index order.
            let mut bounds = vec![0usize; n_nodes + 1];
            for &node in &node_of {
                bounds[node as usize + 1] += 1;
            }
            for i in 0..n_nodes {
                bounds[i + 1] += bounds[i];
            }
            let mut cursor = bounds.clone();
            let mut order = vec![0usize; data.len()];
            for (i, &node) in node_of.iter().enumerate() {
                order[cursor[node as usize]] = i;
                cursor[node as usize] += 1;
            }

            // Each node's child slots and its members' child labels. Unused slots repeat a used one, which `nearest`
            // never picks (ties go to the first slot): a node with `m < branching` members repeats slot `m - 1`, an
            // empty node (only below the root, which holds every descriptor) repeats its parent's centroid.
            let fitted: Vec<(Vec<[f32; D]>, Vec<u32>)> = (0..n_nodes)
                .into_par_iter()
                .map(|node| {
                    let members = &order[bounds[node]..bounds[node + 1]];
                    if members.is_empty() {
                        let parent = centroids.last().map_or([0.0f32; D], |level| level[node]);
                        return (vec![parent; branching], Vec::new());
                    }
                    let points: Vec<[f32; D]> = members.iter().map(|&i| data[i]).collect();
                    if points.len() < branching {
                        let mut slots = points.clone();
                        slots.resize(branching, points[points.len() - 1]);
                        return (slots, (0..points.len() as u32).collect());
                    }
                    let mut rng = ChaCha8Rng::seed_from_u64(mix(
                        config.seed ^ mix(first_node_id + node as u64)
                    ));
                    kmeans(&points, branching, config.iterations, &mut rng)
                })
                .collect();

            let mut level_centroids = Vec::with_capacity(n_nodes * branching);
            let mut child_of = vec![0u32; data.len()];
            for (node, (slots, labels)) in fitted.into_iter().enumerate() {
                for (&i, &label) in order[bounds[node]..bounds[node + 1]].iter().zip(&labels) {
                    child_of[i] = (node * branching) as u32 + label;
                }
                level_centroids.extend(slots);
            }
            centroids.push(level_centroids);
            node_of = child_of;
            first_node_id += n_nodes as u64;
            n_nodes *= branching;
        }
        Ok(Self {
            branching,
            centroids,
        })
    }

    /// Children per node.
    pub fn branching(&self) -> usize {
        self.branching
    }

    /// Levels actually built (the configured depth, lowered when the training set could not fill it).
    pub fn depth(&self) -> usize {
        self.centroids.len()
    }

    /// Size of the word-id range, `branching^depth`, including leaves no descriptor reached.
    pub fn n_words(&self) -> usize {
        self.branching.pow(self.depth() as u32)
    }

    /// The word of one descriptor: greedy descent, the nearest child at every level (the first of equal slots).
    pub fn word(&self, descriptor: &[f32; D]) -> u32 {
        let b = self.branching;
        let mut node = 0usize;
        for centroids in &self.centroids {
            let base = node * b;
            node = base + nearest(descriptor, &centroids[base..base + b]);
        }
        node as u32
    }

    /// [`Vocabulary::word`] of every descriptor, in parallel.
    pub fn words(&self, descriptors: &[[f32; D]]) -> Vec<u32> {
        let mut words = vec![0u32; descriptors.len()];
        words
            .par_chunks_mut(CHUNK)
            .zip(descriptors.par_chunks(CHUNK))
            .for_each(|(w, d)| {
                for (w, d) in w.iter_mut().zip(d) {
                    *w = self.word(d);
                }
            });
        words
    }
}

/// TF-IDF bags of words of a set of images, from the words of each image's descriptors.
///
/// Word `w` of image `i` weighs `count_i(w) * ln(n_images / df(w))`, where `df(w)` is the number of images
/// containing `w`; words of weight 0 (in every image) are dropped and each bag is L1-normalized. An image whose every
/// word is dropped gets an empty bag.
///
/// ```
/// use kornia_staging_bow::vocabulary::tf_idf;
///
/// // Word 7 is in both images (idf 0, dropped); words 1 and 2 each in one.
/// let bows = tf_idf(&[vec![1, 1, 7], vec![2, 7]]);
/// assert_eq!(bows[0].0, vec![(1, 1.0)]);
/// assert_eq!(bows[1].0, vec![(2, 1.0)]);
/// ```
pub fn tf_idf<W: AsRef<[u32]> + Sync>(images: &[W]) -> Vec<BoW> {
    let counts: Vec<Vec<(u32, u32)>> = images
        .par_iter()
        .map(|image| {
            let mut image = image.as_ref().to_vec();
            image.sort_unstable();
            let mut runs: Vec<(u32, u32)> = Vec::new();
            for word in image {
                match runs.last_mut() {
                    Some((last, count)) if *last == word => *count += 1,
                    _ => runs.push((word, 1)),
                }
            }
            runs
        })
        .collect();
    let mut document_frequency: HashMap<u32, u32> = HashMap::new();
    for image in &counts {
        for &(word, _) in image {
            *document_frequency.entry(word).or_insert(0) += 1;
        }
    }
    let n_images = counts.len() as f64;
    counts
        .into_par_iter()
        .map(|image| {
            let mut bow = BoW(image
                .into_iter()
                .map(|(word, count)| {
                    let idf = (n_images / f64::from(document_frequency[&word])).ln();
                    (word, (f64::from(count) * idf) as f32)
                })
                .filter(|&(_, weight)| weight > 0.0)
                .collect());
            bow.normalize_l1();
            bow
        })
        .collect()
}

#[cfg(test)]
mod tests;
