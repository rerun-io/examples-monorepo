//! HashBoW parity with Basalt's `hash_bow.h` on the descriptors of `fixtures/basalt_brief/basalt_brief_dump.cpp`,
//! stored as an `.rrd` on the NAS. Run with the `kornia-staging-golden` Pixi task (release, `--ignored`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use kornia_bow::BoW;
use kornia_staging_bow::hash_bow::{FrameCamId, HashBow};
use kornia_staging_golden::BasaltBrief;

fn id(frame: i64, cam: u32) -> FrameCamId {
    FrameCamId { frame, cam }
}

/// The dump's entries: a 4x4 tiling of each image's described keypoints, frame `tile * 10 + image / 2`, camera
/// `image % 2`.
fn our_bags(db: &HashBow, golden: &BasaltBrief) -> BTreeMap<FrameCamId, BoW> {
    let mut bags = BTreeMap::new();
    for (k, image) in golden.images.iter().enumerate() {
        for tile in 0..16usize {
            let descriptors: Vec<[u64; 4]> = image
                .keypoints
                .iter()
                .filter_map(|kp| {
                    let [x, y] = kp.position;
                    let (_, descriptor) = kp.described?;
                    ((y as usize * 4 / image.height) * 4 + x as usize * 4 / image.width == tile)
                        .then_some(descriptor)
                })
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
#[ignore = "needs the Basalt BRIEF golden .rrd on the NAS (kornia-staging-golden task)"]
fn bags_match_basalt() {
    let golden = BasaltBrief::load_default();
    let db = HashBow::new(16).unwrap();
    let ours = our_bags(&db, &golden);
    assert_eq!(ours.len(), golden.bags.len());
    for (&(frame, cam), basalt) in &golden.bags {
        let words: Vec<(u32, f32)> = basalt.iter().map(|&(w, x)| (w, x as f32)).collect();
        assert_eq!(ours[&id(frame, cam)].0, words, "bag {frame}/{cam}");
    }
    eprintln!("{} bags identical", golden.bags.len());
}

#[test]
#[ignore = "needs the Basalt BRIEF golden .rrd on the NAS (kornia-staging-golden task)"]
fn queries_match_basalt_scores_and_sorted_order() {
    let golden = BasaltBrief::load_default();
    let mut db = HashBow::new(16).unwrap();
    let bags = our_bags(&db, &golden);
    for (&key, bag) in &bags {
        db.insert(key, bag);
    }
    let (mut compared, mut reordered) = (0usize, 0usize);
    for query in &golden.queries {
        let key = id(query.frame, query.cam);
        // Every score, over every entry that shares a word.
        let ours: BTreeMap<FrameCamId, f32> = db
            .query(&bags[&key], 0, None)
            .iter()
            .map(|h| (h.id, h.score))
            .collect();
        assert_eq!(ours.len(), query.all.len(), "query {key:?}");
        for &(frame, cam, score) in &query.all {
            let ours = ours[&id(frame, cam)];
            assert!(
                (f64::from(ours) - score).abs() < 1e-6,
                "query {key:?} entry {frame}/{cam}: {ours} vs {score}"
            );
        }
        // The top 10 older than the query: the same entries, best first. Basalt sorts only when it had more
        // than 10 candidates (B3), so compare as a set and count how often its order was not by score.
        let ours = db.query(&bags[&key], 10, Some(key.frame - 5));
        assert_eq!(ours.len(), query.top.len(), "query {key:?}");
        assert!(ours.windows(2).all(|w| w[0].score >= w[1].score));
        let kth = ours.last().map_or(0.0, |h| f64::from(h.score));
        for &(frame, cam, score) in &query.top {
            // A Basalt entry missing from ours can only be a tie at the cut.
            if !ours.iter().any(|h| h.id == id(frame, cam)) {
                assert!(
                    (score - kth).abs() < 1e-6,
                    "query {key:?}: {frame}/{cam} {score} dropped"
                );
            }
        }
        reordered += usize::from(query.top.windows(2).any(|w| w[0].2 < w[1].2));
        compared += query.top.len();
    }
    eprintln!("{compared} top results compared; Basalt returned {reordered} unsorted lists");
    assert!(compared > 300);
}
