#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::*;

fn id(frame: i64, cam: u32) -> FrameCamId {
    FrameCamId { frame, cam }
}

/// A descriptor whose word under 16 bits is `word`.
fn with_word(word: u32) -> [u64; 4] {
    let mut descriptor = [0u64; 4];
    for (i, &bit) in WORD_BITS.iter().enumerate().take(16) {
        if word >> i & 1 == 1 {
            descriptor[usize::from(bit) / 64] |= 1 << (bit % 64);
        }
    }
    descriptor
}

/// A bag over the given words, one descriptor per listed word.
fn bag_of(db: &HashBow, words: &[u32]) -> BoW {
    db.bag(&words.iter().map(|&w| with_word(w)).collect::<Vec<_>>())
}

#[test]
fn word_size_is_checked() {
    assert_eq!(HashBow::new(0).unwrap_err(), HashBowError::WordBits(0));
    assert_eq!(HashBow::new(33).unwrap_err(), HashBowError::WordBits(33));
    assert!(HashBow::new(32).is_ok());
}

#[test]
fn words_pick_the_permuted_bits() {
    let mut descriptor = [0u64; 4];
    descriptor[170 / 64] |= 1 << (170 % 64);
    descriptor[251 / 64] |= 1 << (251 % 64);
    descriptor[0] |= 1; // bit 0 is not a word bit
    assert_eq!(HashBow::new(16).unwrap().word(&descriptor), 1);
    assert_eq!(HashBow::new(32).unwrap().word(&descriptor), 1 | 1 << 31);
    assert_eq!(HashBow::new(16).unwrap().word(&with_word(0xBEEF)), 0xBEEF);
}

#[test]
fn bags_are_term_frequencies_sorted_by_word() {
    let db = HashBow::new(16).unwrap();
    assert_eq!(bag_of(&db, &[9, 2, 9, 9]).0, vec![(2, 0.25), (9, 0.75)]);
    assert!(db.bag(&[]).0.is_empty());
}

#[test]
fn results_are_sorted_even_when_fewer_than_requested() {
    // B3: hash_bow.h sorts only when it has more hits than requested, and RootSLAM takes the first candidate
    // that validates, so a weaker one could win.
    let mut db = HashBow::new(16).unwrap();
    db.insert(id(1, 0), &bag_of(&db, &[1, 2, 3, 4]));
    db.insert(id(2, 0), &bag_of(&db, &[1, 5, 6, 7]));
    db.insert(id(3, 0), &bag_of(&db, &[1, 2, 3, 8]));
    let hits = db.query(&bag_of(&db, &[1, 2, 3, 4]), 10, None);
    assert_eq!(
        hits.iter().map(|h| h.id.frame).collect::<Vec<_>>(),
        [1, 3, 2]
    );
    assert!(
        (hits[0].score - 1.0).abs() < 1e-6
            && (hits[1].score - 0.75).abs() < 1e-6
            && (hits[2].score - 0.25).abs() < 1e-6
    );
}

#[test]
fn ties_are_broken_by_frame_then_camera_whatever_the_insertion_order() {
    let mut db = HashBow::new(16).unwrap();
    let bag = bag_of(&db, &[4, 5]);
    for key in [id(7, 1), id(3, 0), id(7, 0), id(5, 2)] {
        db.insert(key, &bag);
    }
    let order: Vec<FrameCamId> = db.query(&bag, 0, None).iter().map(|h| h.id).collect();
    assert_eq!(order, [id(3, 0), id(5, 2), id(7, 0), id(7, 1)]);
    assert_eq!(db.query(&bag, 2, None).len(), 2);
}

#[test]
fn top_n_counts_frames_not_cameras() {
    // B4: with N = 2, the entry ranking spends both places on frame 1's cameras.
    let mut db = HashBow::new(16).unwrap();
    let query = bag_of(&db, &[1, 2, 3, 4]);
    db.insert(id(1, 0), &bag_of(&db, &[1, 2, 3, 4]));
    db.insert(id(1, 1), &bag_of(&db, &[1, 2, 3, 9]));
    db.insert(id(2, 0), &bag_of(&db, &[1, 2, 8, 9]));
    db.insert(id(2, 1), &bag_of(&db, &[1, 7, 8, 9]));
    let entries: Vec<i64> = db
        .query(&query, 2, None)
        .iter()
        .map(|h| h.id.frame)
        .collect();
    assert_eq!(entries, [1, 1]);
    let frames = db.query_frames(&query, 2, None);
    assert_eq!(frames.iter().map(|h| h.frame).collect::<Vec<_>>(), [1, 2]);
    assert!((frames[1].score - 0.5).abs() < 1e-6);
}

#[test]
fn the_minimum_age_filter_is_strict() {
    let mut db = HashBow::new(16).unwrap();
    let bag = bag_of(&db, &[1]);
    for frame in [10, 20, 30] {
        db.insert(id(frame, 0), &bag);
    }
    let frames = |before| {
        db.query_frames(&bag, 0, before)
            .iter()
            .map(|h| h.frame)
            .collect::<Vec<_>>()
    };
    assert_eq!(frames(Some(20)), [10]);
    assert_eq!(frames(Some(21)), [10, 20]);
    assert_eq!(frames(None), [10, 20, 30]);
}

#[test]
fn culled_frames_leave_the_database() {
    let mut db = HashBow::new(16).unwrap();
    let bag = bag_of(&db, &[1, 2]);
    for key in [id(1, 0), id(1, 1), id(2, 0)] {
        assert!(db.insert(key, &bag));
    }
    assert!(!db.insert(id(1, 0), &bag_of(&db, &[3])));
    assert_eq!(db.remove_frame(1), 2);
    assert_eq!(db.remove_frame(1), 0);
    assert!(!db.remove(id(1, 0)));
    assert_eq!(db.len(), 1);
    assert_eq!(
        db.query(&bag, 0, None)
            .iter()
            .map(|h| h.id)
            .collect::<Vec<_>>(),
        [id(2, 0)]
    );
    // A culled id can be indexed again.
    assert!(db.insert(id(1, 0), &bag));
}

/// Parity with Basalt's `hash_bow.h` on the descriptors of `fixtures/basalt_brief/basalt_brief_dump.cpp`.
mod golden {
    use super::*;

    /// Basalt's results of one query: entry and score, in its order.
    type Hits = Vec<(FrameCamId, f64)>;
    /// An image's width, height and described keypoints (position, descriptor).
    type GoldenImage = (usize, usize, Vec<([f32; 2], [u64; 4])>);

    struct Golden {
        /// Per image: width, height and the described keypoints (position, descriptor).
        images: Vec<GoldenImage>,
        bags: BTreeMap<FrameCamId, Vec<(u32, f64)>>,
        /// `(query, all results, top-10 results older than query - 5)`, in Basalt's order.
        queries: Vec<(FrameCamId, Hits, Hits)>,
    }

    fn results(fields: &[&str]) -> (FrameCamId, Hits) {
        let key = id(fields[1].parse().unwrap(), fields[2].parse().unwrap());
        let hits: Hits = fields[4..]
            .iter()
            .map(|h| {
                let parts: Vec<&str> = h.split(',').collect();
                (
                    id(parts[0].parse().unwrap(), parts[1].parse().unwrap()),
                    parts[2].parse().unwrap(),
                )
            })
            .collect();
        assert_eq!(hits.len(), fields[3].parse::<usize>().unwrap());
        (key, hits)
    }

    fn load() -> Option<Golden> {
        let path = std::env::var_os("KORNIA_STAGING_BASALT_BRIEF")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    "/mnt/nas/ssd-8tb/scratch/slam-rs-rootslam/fixtures/b1/basalt_brief.txt",
                )
            });
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skipped: Basalt BRIEF golden {} is absent", path.display());
            return None;
        };
        let mut golden = Golden {
            images: Vec::new(),
            bags: BTreeMap::new(),
            queries: Vec::new(),
        };
        for line in text.lines() {
            let fields: Vec<&str> = line.split(' ').collect();
            match fields[0] {
                "image" => golden.images.push((
                    fields[2].parse().unwrap(),
                    fields[3].parse().unwrap(),
                    Vec::new(),
                )),
                "kp" if fields[3] != "out" => {
                    let words: Vec<u64> = fields[4..8]
                        .iter()
                        .map(|w| u64::from_str_radix(w, 16).unwrap())
                        .collect();
                    let position = [fields[1].parse().unwrap(), fields[2].parse().unwrap()];
                    golden
                        .images
                        .last_mut()
                        .unwrap()
                        .2
                        .push((position, words.try_into().unwrap()));
                }
                "bag" => {
                    let words = fields[4..]
                        .iter()
                        .map(|w| {
                            let (word, weight) = w.split_once(':').unwrap();
                            (word.parse().unwrap(), weight.parse().unwrap())
                        })
                        .collect();
                    golden.bags.insert(
                        id(fields[1].parse().unwrap(), fields[2].parse().unwrap()),
                        words,
                    );
                }
                "query_all" => {
                    let (key, all) = results(&fields);
                    golden.queries.push((key, all, Vec::new()));
                }
                "query_top" => {
                    let (key, top) = results(&fields);
                    let last = golden.queries.last_mut().unwrap();
                    assert_eq!(last.0, key);
                    last.2 = top;
                }
                _ => {}
            }
        }
        Some(golden)
    }

    /// The dump's entries: a 4x4 tiling of each image, frame `tile * 10 + image / 2`, camera `image % 2`.
    fn our_bags(db: &HashBow, golden: &Golden) -> BTreeMap<FrameCamId, BoW> {
        let mut bags = BTreeMap::new();
        for (k, (width, height, keypoints)) in golden.images.iter().enumerate() {
            for tile in 0..16usize {
                let descriptors: Vec<[u64; 4]> = keypoints
                    .iter()
                    .filter(|([x, y], _)| {
                        (*y as usize * 4 / height) * 4 + *x as usize * 4 / width == tile
                    })
                    .map(|&(_, d)| d)
                    .collect();
                bags.insert(
                    id((tile * 10 + k / 2) as i64, (k % 2) as u32),
                    db.bag(&descriptors),
                );
            }
        }
        bags
    }

    #[test]
    fn bags_match_basalt() {
        let Some(golden) = load() else { return };
        let db = HashBow::new(16).unwrap();
        let ours = our_bags(&db, &golden);
        assert_eq!(ours.len(), golden.bags.len());
        for (key, basalt) in &golden.bags {
            let words: Vec<(u32, f32)> = basalt.iter().map(|&(w, x)| (w, x as f32)).collect();
            assert_eq!(ours[key].0, words, "bag {key:?}");
        }
    }

    #[test]
    fn queries_match_basalt_scores_and_sorted_order() {
        let Some(golden) = load() else { return };
        let mut db = HashBow::new(16).unwrap();
        let bags = our_bags(&db, &golden);
        for (&key, bag) in &bags {
            db.insert(key, bag);
        }
        let (mut compared, mut reordered) = (0usize, 0usize);
        for (key, all, top) in &golden.queries {
            // Every score, over every entry that shares a word.
            let ours: BTreeMap<FrameCamId, f32> = db
                .query(&bags[key], 0, None)
                .iter()
                .map(|h| (h.id, h.score))
                .collect();
            assert_eq!(ours.len(), all.len(), "query {key:?}");
            for (id, score) in all {
                assert!(
                    (f64::from(ours[id]) - score).abs() < 1e-6,
                    "query {key:?} entry {id:?}: {} vs {score}",
                    ours[id]
                );
            }
            // The top 10 older than the query: the same entries, best first. Basalt sorts only when it had more
            // than 10 candidates (B3), so compare as a set and count how often its order was not by score.
            let before = key.frame - 5;
            let ours = db.query(&bags[key], 10, Some(before));
            assert_eq!(ours.len(), top.len(), "query {key:?}");
            assert!(ours.windows(2).all(|w| w[0].score >= w[1].score));
            let kth = ours.last().map_or(0.0, |h| f64::from(h.score));
            for (id, score) in top {
                // A Basalt entry missing from ours can only be a tie at the cut.
                if !ours.iter().any(|h| h.id == *id) {
                    assert!(
                        (score - kth).abs() < 1e-6,
                        "query {key:?}: {id:?} {score} dropped"
                    );
                }
            }
            reordered += usize::from(top.windows(2).any(|w| w[0].1 < w[1].1));
            compared += top.len();
        }
        eprintln!("{compared} top results compared; Basalt returned {reordered} unsorted lists");
        assert!(compared > 300);
    }
}
