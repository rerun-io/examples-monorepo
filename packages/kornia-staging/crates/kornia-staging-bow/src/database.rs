//! An inverted-file database of bags of words with DBoW2's L1 query (`TemplatedDatabase::queryL1`).

use std::collections::HashMap;

use kornia_bow::BoW;

/// Id of a database entry: the order in which it was added, from 0.
pub type EntryId = u32;

/// One query result: an entry and its L1 similarity, in `[0, 1]` (1 for identical bags of words).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryResult {
    /// The matching entry.
    pub entry: EntryId,
    /// `1 - |q - d|_1 / 2` for the L1-normalized bags of words `q` and `d`.
    pub score: f32,
}

/// Bags of words indexed by word: for each word, the entries that contain it and their weights.
///
/// A query visits only the entries that share at least one word with it, so its cost grows with the number of
/// matching entries rather than the database size. Words of weight 0 carry no information and are not indexed.
///
/// ```
/// use kornia_bow::BoW;
/// use kornia_staging_bow::Database;
///
/// let mut db = Database::new();
/// let kitchen = BoW(vec![(3, 0.5), (7, 0.5)]);
/// let garden = BoW(vec![(1, 1.0)]);
/// assert_eq!(db.add(&kitchen), 0);
/// assert_eq!(db.add(&garden), 1);
///
/// let results = db.query(&kitchen, 4);
/// assert_eq!(results.len(), 1); // the garden shares no word with the kitchen
/// assert_eq!(results[0].entry, 0);
/// assert!((results[0].score - 1.0).abs() < 1e-6);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Database {
    inverted: HashMap<u32, Vec<(EntryId, f32)>>,
    len: u32,
}

impl Database {
    /// An empty database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries added.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` when no entry has been added.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Add an L1-normalized bag of words (as [`kornia_bow::Vocabulary::transform`] returns it).
    ///
    /// # Returns
    ///
    /// The id of the new entry.
    pub fn add(&mut self, bow: &BoW) -> EntryId {
        let entry = self.len;
        self.len += 1;
        for &(word, weight) in bow.0.iter().filter(|&&(_, w)| w != 0.0) {
            self.inverted.entry(word).or_default().push((entry, weight));
        }
        entry
    }

    /// The entries most similar to `bow` by L1 similarity, best first.
    ///
    /// Only entries that share a word with `bow` are scored. For L1-normalized vectors
    /// `|q - d|_1 = 2 + sum over shared words of (|q_w - d_w| - |q_w| - |d_w|)`, so each entry accumulates that
    /// sum over the shared words alone (DBoW2's `queryL1`). Equal scores keep entry order.
    ///
    /// # Arguments
    ///
    /// * `bow` - The L1-normalized query bag of words.
    /// * `max_results` - The number of results to keep; `0` keeps every scored entry.
    pub fn query(&self, bow: &BoW, max_results: usize) -> Vec<QueryResult> {
        let mut sums = vec![0.0f32; self.len()];
        let mut scored = vec![false; self.len()];
        for &(word, q) in bow.0.iter().filter(|&&(_, w)| w != 0.0) {
            for &(entry, d) in self.inverted.get(&word).into_iter().flatten() {
                sums[entry as usize] += (q - d).abs() - q.abs() - d.abs();
                scored[entry as usize] = true;
            }
        }
        let mut results: Vec<QueryResult> = (0..self.len)
            .filter(|&entry| scored[entry as usize])
            .map(|entry| QueryResult {
                entry,
                score: -sums[entry as usize] / 2.0,
            })
            .collect();
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        if max_results > 0 {
            results.truncate(max_results);
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn bow(words: &[(u32, f32)]) -> BoW {
        let mut bow = BoW(words.to_vec());
        bow.normalize_l1();
        bow
    }

    #[test]
    fn ranks_by_shared_weight_and_skips_disjoint_entries() {
        let mut db = Database::new();
        db.add(&bow(&[(1, 1.0), (2, 1.0)]));
        db.add(&bow(&[(1, 1.0), (3, 1.0)]));
        db.add(&bow(&[(9, 1.0)]));
        let results = db.query(&bow(&[(1, 1.0), (2, 1.0)]), 0);
        assert_eq!(
            results.iter().map(|r| r.entry).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert!((results[1].score - 0.5).abs() < 1e-6);
        assert_eq!(db.query(&bow(&[(1, 1.0)]), 1).len(), 1);
    }

    #[test]
    fn zero_weight_words_do_not_match() {
        let mut db = Database::new();
        db.add(&BoW(vec![(4, 0.0), (5, 1.0)]));
        assert!(db.query(&BoW(vec![(4, 1.0)]), 0).is_empty());
        assert!(db.query(&BoW(vec![(6, 1.0)]), 0).is_empty());
    }

    #[test]
    fn ties_keep_entry_order() {
        let mut db = Database::new();
        for _ in 0..3 {
            db.add(&bow(&[(1, 1.0)]));
        }
        let entries: Vec<_> = db
            .query(&bow(&[(1, 1.0)]), 0)
            .iter()
            .map(|r| r.entry)
            .collect();
        assert_eq!(entries, vec![0, 1, 2]);
    }

    fn arbitrary_bow() -> impl Strategy<Value = BoW> {
        proptest::collection::btree_map(0u32..40, 0.01f32..1.0, 1..12).prop_map(|words| {
            let mut bow = BoW(words.into_iter().collect());
            bow.normalize_l1();
            bow
        })
    }

    proptest! {
        /// The inverted-file score equals kornia-bow's pairwise L1 similarity for every entry it returns.
        #[test]
        fn query_scores_match_pairwise_l1_similarity(entries in proptest::collection::vec(arbitrary_bow(), 1..8), q in arbitrary_bow()) {
            let mut db = Database::new();
            for e in &entries {
                db.add(e);
            }
            for r in db.query(&q, 0) {
                let expected = q.l1_similarity(&entries[r.entry as usize]);
                prop_assert!((r.score - expected).abs() < 1e-5, "{} vs {}", r.score, expected);
            }
        }
    }
}
