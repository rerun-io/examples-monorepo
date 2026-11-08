//! DBoW2 (lahavlipson/DBoW2@89f8b27) for ORB descriptors, the subset DPV-SLAM's classical loop closure uses,
//! plus OpenCV's cross-checked brute-force Hamming matcher.
//!
//! - [`Vocabulary`]: the k-ary tree of an `ORBvoc.txt` text vocabulary. A descriptor descends from the root to
//!   the child at the smallest Hamming distance (the first child on ties) until it reaches a leaf, the word.
//! - [`BowVector`]: the TF-IDF bag of words of one image, L1-normalized, as `TemplatedVocabulary::transform`
//!   with L1 scoring and TF-IDF weighting builds it.
//! - [`Database`]: the inverted file and `TemplatedDatabase::queryL1` (no direct index).
//! - [`cross_check_match`]: `cv::BFMatcher(NORM_HAMMING, crossCheck=true).knnMatch(k=1)`.
//!
//! Every sum runs in the order of the C++ code (words in ascending id, entries in ascending id), so the scores
//! are bit-identical. DBoW2's text loader also appends a phantom child to the root when the file ends with a
//! newline (its `while(!f.eof())` loop parses one empty line), with an uninitialized descriptor; it is left out.

use std::collections::BTreeMap;
use std::path::Path;

/// An ORB descriptor: 256 bits.
pub type Descriptor = [u8; 32];
/// Word id in the vocabulary.
pub type WordId = u32;
/// Database entry id (insertion order).
pub type EntryId = u32;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot read the vocabulary {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("vocabulary line {line}: {reason}")]
    Parse { line: usize, reason: String },
    #[error("vocabulary uses scoring {scoring} and weighting {weighting}; only L1 scoring (0) with TF-IDF weighting (0) is supported")]
    Unsupported { scoring: i32, weighting: i32 },
}

/// Hamming distance between two ORB descriptors (`FORB::distance`).
#[inline]
pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    let mut bits = 0;
    for i in 0..4 {
        let x = u64::from_ne_bytes(a[8 * i..8 * i + 8].try_into().expect("8 bytes"));
        let y = u64::from_ne_bytes(b[8 * i..8 * i + 8].try_into().expect("8 bytes"));
        bits += (x ^ y).count_ones();
    }
    bits
}

struct Node {
    descriptor: Descriptor,
    children: Vec<u32>,
    /// `Some((word id, idf weight))` for a leaf.
    word: Option<(WordId, f64)>,
}

/// An ORB vocabulary tree read from DBoW2's text format.
pub struct Vocabulary {
    nodes: Vec<Node>,
    num_words: usize,
}

impl Vocabulary {
    /// Read `ORBvoc.txt`: a `k L scoring weighting` header, then one node per line in id order (from 1; 0 is the
    /// root): `parent is_leaf d0 .. d31 weight`.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self, Error> {
        let mut lines = text.lines();
        let header: Vec<i32> = lines
            .next()
            .unwrap_or_default()
            .split_ascii_whitespace()
            .take(4)
            .map(|s| s.parse::<i32>())
            .collect::<Result<_, _>>()
            .map_err(|e| Error::Parse {
                line: 1,
                reason: e.to_string(),
            })?;
        let [k, levels, scoring, weighting] = header[..] else {
            return Err(Error::Parse {
                line: 1,
                reason: "expected `k L scoring weighting`".into(),
            });
        };
        if !(0..=20).contains(&k)
            || !(1..=10).contains(&levels)
            || !(0..=5).contains(&scoring)
            || !(0..=3).contains(&weighting)
        {
            return Err(Error::Parse {
                line: 1,
                reason: "this is not a DBoW2 text vocabulary".into(),
            });
        }
        if scoring != 0 || weighting != 0 {
            return Err(Error::Unsupported { scoring, weighting });
        }
        let mut nodes = vec![Node {
            descriptor: [0; 32],
            children: Vec::new(),
            word: None,
        }];
        let mut num_words = 0usize;
        for (i, line) in lines.enumerate() {
            let line_no = i + 2;
            let mut fields = line.split_ascii_whitespace();
            let bad = |reason: &str| Error::Parse {
                line: line_no,
                reason: reason.into(),
            };
            let Some(parent) = fields.next() else {
                continue;
            };
            let parent: usize = parent.parse().map_err(|_| bad("bad parent id"))?;
            let is_leaf: i32 = fields
                .next()
                .ok_or_else(|| bad("missing leaf flag"))?
                .parse()
                .map_err(|_| bad("bad leaf flag"))?;
            let mut descriptor = [0u8; 32];
            for byte in descriptor.iter_mut() {
                let v: i32 = fields
                    .next()
                    .ok_or_else(|| bad("short descriptor"))?
                    .parse()
                    .map_err(|_| bad("bad descriptor byte"))?;
                *byte = v as u8; // `(unsigned char)n`, as FORB::fromString
            }
            let weight: f64 = fields
                .next()
                .ok_or_else(|| bad("missing weight"))?
                .parse()
                .map_err(|_| bad("bad weight"))?;
            let id = nodes.len() as u32;
            if parent >= nodes.len() {
                return Err(bad("parent after child"));
            }
            nodes[parent].children.push(id);
            let word = if is_leaf > 0 {
                num_words += 1;
                Some(((num_words - 1) as WordId, weight))
            } else {
                None
            };
            nodes.push(Node {
                descriptor,
                children: Vec::new(),
                word,
            });
        }
        Ok(Self { nodes, num_words })
    }

    pub fn num_words(&self) -> usize {
        self.num_words
    }

    /// The word of one descriptor and its idf weight (`TemplatedVocabulary::transform(feature, id, weight)`).
    pub fn word(&self, descriptor: &Descriptor) -> (WordId, f64) {
        let mut id = 0usize;
        loop {
            let children = &self.nodes[id].children;
            let Some((&first, rest)) = children.split_first() else {
                break;
            };
            let mut best = first as usize;
            let mut best_d = hamming(descriptor, &self.nodes[best].descriptor);
            for &child in rest {
                let d = hamming(descriptor, &self.nodes[child as usize].descriptor);
                if d < best_d {
                    best_d = d;
                    best = child as usize;
                }
            }
            id = best;
        }
        // A childless inner node (not in a well-formed file) is a stopped word, as DBoW2's default Node.
        self.nodes[id].word.unwrap_or((0, 0.0))
    }

    /// The L1-normalized TF-IDF bag of words of one image's descriptors.
    pub fn transform(&self, descriptors: &[Descriptor]) -> BowVector {
        let mut words: BTreeMap<WordId, f64> = BTreeMap::new();
        for d in descriptors {
            let (id, weight) = self.word(d);
            if weight > 0.0 {
                *words.entry(id).or_insert(0.0) += weight;
            }
        }
        let norm: f64 = words.values().map(|v| v.abs()).sum();
        if norm > 0.0 {
            for v in words.values_mut() {
                *v /= norm;
            }
        }
        BowVector(words.into_iter().collect())
    }
}

/// A bag of words: (word id, weight) in ascending word id.
#[derive(Clone, Debug, PartialEq)]
pub struct BowVector(pub Vec<(WordId, f64)>);

/// One query result: an entry and its L1 similarity in [0 worst, 1 best].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryResult {
    pub entry: EntryId,
    pub score: f64,
}

/// DBoW2's database without the direct index: one inverted row per word.
pub struct Database {
    inverted: Vec<Vec<(EntryId, f64)>>,
    entries: u32,
}

impl Database {
    pub fn new(vocabulary: &Vocabulary) -> Self {
        Self {
            inverted: vec![Vec::new(); vocabulary.num_words()],
            entries: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries as usize
    }

    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }

    /// Add one image's bag of words; returns its entry id.
    pub fn add(&mut self, bow: &BowVector) -> EntryId {
        let entry = self.entries;
        self.entries += 1;
        for &(word, weight) in &bow.0 {
            self.inverted[word as usize].push((entry, weight));
        }
        entry
    }

    /// The `max_results` best entries for `bow` (all when 0), best first (`queryL1`).
    pub fn query(&self, bow: &BowVector, max_results: usize) -> Vec<QueryResult> {
        let mut pairs: BTreeMap<EntryId, f64> = BTreeMap::new();
        for &(word, q) in &bow.0 {
            for &(entry, d) in &self.inverted[word as usize] {
                let value = (q - d).abs() - q.abs() - d.abs();
                *pairs.entry(entry).or_insert(0.0) += value;
            }
        }
        // Ascending "score" (-2 best .. 0 worst), as std::sort over entries in id order; ties keep id order.
        let mut results: Vec<QueryResult> = pairs
            .into_iter()
            .map(|(entry, score)| QueryResult { entry, score })
            .collect();
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).expect("finite scores"));
        if max_results > 0 {
            results.truncate(max_results);
        }
        for r in &mut results {
            r.score = -r.score / 2.0;
        }
        results
    }
}

/// `cv::BFMatcher(NORM_HAMMING, crossCheck=true).knnMatch(query, train, 1)`: for every query descriptor that
/// gets a match, `(query index, train index, distance)`, in query order.
///
/// OpenCV's cross-check (`batchDistance`): a query and a train descriptor match when each is the other's nearest
/// neighbour, the first one on ties in both directions.
pub fn cross_check_match(query: &[Descriptor], train: &[Descriptor]) -> Vec<(usize, usize, u32)> {
    let nearest = |from: &Descriptor, to: &[Descriptor]| -> Option<(usize, u32)> {
        let mut best: Option<(usize, u32)> = None;
        for (j, d) in to.iter().enumerate() {
            let dist = hamming(from, d);
            if best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((j, dist));
            }
        }
        best
    };
    let train_to_query: Vec<Option<(usize, u32)>> =
        train.iter().map(|t| nearest(t, query)).collect();
    query
        .iter()
        .enumerate()
        .filter_map(|(q, query_d)| {
            let (t, d) = nearest(query_d, train)?;
            (train_to_query[t].map(|(back, _)| back) == Some(q)).then_some((q, t, d))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(fill: u8) -> Descriptor {
        [fill; 32]
    }

    /// Root with two children; the first has one leaf, the second two.
    fn vocab() -> Vocabulary {
        let node = |parent: u32, leaf: u32, byte: u8, weight: &str| {
            format!(
                "{parent} {leaf} {} {weight}\n",
                vec![byte.to_string(); 32].join(" ")
            )
        };
        let text = [
            "2 2 0 0\n".to_string(),
            node(0, 0, 0, "0"),
            node(0, 0, 255, "0"),
            node(1, 1, 0, "0.5"),
            node(2, 1, 255, "2.0"),
            node(2, 1, 15, "1.0"),
        ]
        .concat();
        Vocabulary::parse(&text).unwrap()
    }

    #[test]
    fn descends_to_the_nearest_leaf_and_ties_keep_the_first_child() {
        let voc = vocab();
        assert_eq!(voc.num_words(), 3);
        assert_eq!(voc.word(&desc(0)), (0, 0.5));
        assert_eq!(voc.word(&desc(255)), (1, 2.0));
        // 0x0F: 128 bits from both root children -> the first (node 1), then its only leaf.
        assert_eq!(voc.word(&desc(15)), (0, 0.5));
    }

    #[test]
    fn transform_is_l1_normalized_tf_idf() {
        let voc = vocab();
        let bow = voc.transform(&[desc(0), desc(0), desc(255)]);
        assert_eq!(bow.0, vec![(0, 1.0 / 3.0), (1, 2.0 / 3.0)]);
    }

    #[test]
    fn query_ranks_identical_images_first() {
        let voc = vocab();
        let mut db = Database::new(&voc);
        let a = voc.transform(&[desc(0)]);
        let b = voc.transform(&[desc(255)]);
        db.add(&a);
        db.add(&b);
        let results = db.query(&a, 4);
        assert_eq!(
            results,
            vec![QueryResult {
                entry: 0,
                score: 1.0
            }]
        );
    }

    #[test]
    fn cross_check_keeps_mutual_choices_only() {
        // Query 0 and train 1 are mutual nearest neighbours. Query 1's nearest train is train 0 (0x07: 5 bits per
        // byte away, against 6 for 0x03), but train 0's nearest query is query 0, so query 1 stays unmatched.
        let query = [desc(0b0000_0001), desc(0b1111_1111)];
        let train = [desc(0b0000_0111), desc(0b0000_0011)];
        assert_eq!(cross_check_match(&query, &train), vec![(0, 1, 32)]);
    }
}
