//! RootSLAM's HashBoW (Basalt `hash_bow.h`): a vocabulary-free bag of words for 256-bit binary descriptors.
//!
//! A descriptor's word is up to 32 of its bits, picked by a fixed permutation; an image's bag is the L1-normalized
//! histogram of its words (term frequency, no inverse document frequency). Entries are keyed by frame and camera,
//! queried with DBoW2's L1 score through [`Database`], and can be removed when a keyframe is culled.
//!
//! Differences from `hash_bow.h`, all deterministic where it is not: results are always sorted, best first, with
//! ties broken by key (it returns at most N results in hash-map order when there are no more than N), and
//! [`HashBow::query_frames`] ranks frames by their best camera before taking the top N (it takes the top N over
//! (frame, camera) entries, so a 4-camera rig gets about N/4 distinct frames). Weights are f32 ([`BoW`]), scores
//! are accumulated in f64; Basalt uses f64 throughout, so scores agree to about 1e-7.

use std::collections::{BTreeMap, HashMap};

use kornia_bow::BoW;

use crate::database::{Database, EntryId};

/// The largest word size: words are `u32`, as Basalt's `FeatureHash` is a 32-bit `std::bitset`.
pub const HASH_BOW_MAX_BITS: usize = 32;

/// Basalt's word bits for 256-bit descriptors: the first 32 entries below 256 of `hash_bow.h`'s
/// `random_bit_permutation`. Word bit `i` is descriptor bit `WORD_BITS[i]`.
const WORD_BITS: [u8; HASH_BOW_MAX_BITS] = [
    170, 215, 41, 38, 96, 172, 52, 1, 182, 89, 234, 98, 217, 73, 195, 113, 161, 247, 24, 75, 7,
    232, 49, 196, 144, 69, 3, 86, 94, 201, 107, 251,
];

/// A HashBoW entry: one camera of one frame. Ordered by frame, then camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameCamId {
    /// Frame id; RootSLAM uses the keyframe timestamp in nanoseconds.
    pub frame: i64,
    /// Camera index in the rig.
    pub cam: u32,
}

/// One [`HashBow::query`] result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EntryHit {
    /// The matching entry.
    pub id: FrameCamId,
    /// L1 similarity in `[0, 1]`.
    pub score: f32,
}

/// One [`HashBow::query_frames`] result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameHit {
    /// The matching frame.
    pub frame: i64,
    /// The best score over the frame's cameras.
    pub score: f32,
}

/// Invalid HashBoW settings.
#[derive(Debug, PartialEq, Eq)]
pub enum HashBowError {
    /// The word size must be in `1..=32`.
    WordBits(usize),
}

impl std::fmt::Display for HashBowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WordBits(bits) => write!(
                f,
                "HashBoW word size {bits} is outside 1..={HASH_BOW_MAX_BITS}"
            ),
        }
    }
}

impl std::error::Error for HashBowError {}

/// An inverted-file HashBoW database keyed by [`FrameCamId`].
///
/// ```
/// use kornia_staging_bow::hash_bow::{FrameCamId, HashBow};
///
/// let mut db = HashBow::new(16)?;
/// let bag = db.bag(&[[1, 2, 3, 4], [5, 6, 7, 8]]);
/// db.insert(FrameCamId { frame: 10, cam: 0 }, &bag);
/// db.insert(FrameCamId { frame: 90, cam: 0 }, &bag);
/// // Only frames older than 50 are candidates.
/// let hits = db.query(&bag, 10, Some(50));
/// assert_eq!(hits.len(), 1);
/// assert_eq!(hits[0].id.frame, 10);
/// # Ok::<(), kornia_staging_bow::hash_bow::HashBowError>(())
/// ```
#[derive(Clone, Debug)]
pub struct HashBow {
    num_bits: usize,
    database: Database,
    entries: HashMap<FrameCamId, EntryId>,
    keys: HashMap<EntryId, FrameCamId>,
}

impl HashBow {
    /// An empty database whose words take `num_bits` descriptor bits (RootSLAM `loop_closing_bow_num_bits`, 16).
    ///
    /// # Errors
    ///
    /// [`HashBowError::WordBits`] when `num_bits` is 0 or above [`HASH_BOW_MAX_BITS`] (Basalt clamps it to 32).
    pub fn new(num_bits: usize) -> Result<Self, HashBowError> {
        if !(1..=HASH_BOW_MAX_BITS).contains(&num_bits) {
            return Err(HashBowError::WordBits(num_bits));
        }
        Ok(Self {
            num_bits,
            database: Database::new(),
            entries: HashMap::new(),
            keys: HashMap::new(),
        })
    }

    /// The word of a descriptor: bit `i` is descriptor bit `WORD_BITS[i]`, for `i < num_bits`.
    ///
    /// Descriptor bit `b` is bit `b % 64` of `descriptor[b / 64]`.
    pub fn word(&self, descriptor: &[u64; 4]) -> u32 {
        WORD_BITS[..self.num_bits]
            .iter()
            .enumerate()
            .fold(0, |word, (i, &bit)| {
                let bit = usize::from(bit);
                word | (((descriptor[bit / 64] >> (bit % 64)) & 1) as u32) << i
            })
    }

    /// The bag of words of an image's descriptors: each word's count over the number of descriptors, sorted by
    /// word. Empty for no descriptors.
    pub fn bag(&self, descriptors: &[[u64; 4]]) -> BoW {
        let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
        for descriptor in descriptors {
            *counts.entry(self.word(descriptor)).or_default() += 1;
        }
        let total = descriptors.len() as f64;
        BoW(counts
            .into_iter()
            .map(|(word, count)| (word, (f64::from(count) / total) as f32))
            .collect())
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when there is no entry.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `true` when `id` is in the database.
    pub fn contains(&self, id: FrameCamId) -> bool {
        self.entries.contains_key(&id)
    }

    /// Add the bag of words of one camera of one frame.
    ///
    /// # Returns
    ///
    /// `false`, leaving the database unchanged, when `id` is already present (RootSLAM skips indexed keyframes).
    pub fn insert(&mut self, id: FrameCamId, bow: &BoW) -> bool {
        if self.entries.contains_key(&id) {
            return false;
        }
        let entry = self.database.add(bow);
        self.entries.insert(id, entry);
        self.keys.insert(entry, id);
        true
    }

    /// Remove one entry. Returns `false` when it is absent.
    pub fn remove(&mut self, id: FrameCamId) -> bool {
        let Some(entry) = self.entries.remove(&id) else {
            return false;
        };
        self.keys.remove(&entry);
        self.database.remove(entry)
    }

    /// Remove every camera of a culled frame. Returns how many entries were removed.
    pub fn remove_frame(&mut self, frame: i64) -> usize {
        let ids: Vec<FrameCamId> = self
            .entries
            .keys()
            .filter(|id| id.frame == frame)
            .copied()
            .collect();
        ids.into_iter().filter(|&id| self.remove(id)).count()
    }

    /// The entries most similar to `bow`, best first, ties by [`FrameCamId`].
    ///
    /// # Arguments
    ///
    /// * `bow` - The query bag of words, from [`HashBow::bag`].
    /// * `max_results` - The number of results to keep; `0` keeps all.
    /// * `before` - When set, only entries of frames strictly below it are scored: RootSLAM's minimum age,
    ///   `query frame - loop_closing_frame_time_margin_s`.
    pub fn query(&self, bow: &BoW, max_results: usize, before: Option<i64>) -> Vec<EntryHit> {
        let mut hits: Vec<EntryHit> = self
            .database
            .query_where(bow, 0, |entry| {
                before.is_none_or(|b| self.keys[&entry].frame < b)
            })
            .into_iter()
            .map(|r| EntryHit {
                id: self.keys[&r.entry],
                score: r.score,
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        if max_results > 0 {
            hits.truncate(max_results);
        }
        hits
    }

    /// The frames most similar to `bow`: each frame scores its best camera, then the best `max_results` frames
    /// are kept, best first, ties by frame. Arguments as [`HashBow::query`].
    pub fn query_frames(
        &self,
        bow: &BoW,
        max_results: usize,
        before: Option<i64>,
    ) -> Vec<FrameHit> {
        let mut best: BTreeMap<i64, f32> = BTreeMap::new();
        for hit in self.query(bow, 0, before) {
            let score = best.entry(hit.id.frame).or_insert(hit.score);
            *score = score.max(hit.score);
        }
        let mut frames: Vec<FrameHit> = best
            .into_iter()
            .map(|(frame, score)| FrameHit { frame, score })
            .collect();
        frames.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.frame.cmp(&b.frame)));
        if max_results > 0 {
            frames.truncate(max_results);
        }
        frames
    }
}

#[cfg(test)]
mod tests;
