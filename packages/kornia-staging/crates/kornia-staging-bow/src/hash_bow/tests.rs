#![allow(clippy::unwrap_used)]

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
