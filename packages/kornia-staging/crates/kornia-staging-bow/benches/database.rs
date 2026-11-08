//! Inverted-file query against 4,000 entries of 500 words each from a 1M-word vocabulary (DPV-SLAM's KITTI scale).

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_bow::BoW;
use kornia_staging_bow::Database;

/// A deterministic bag of `n` words; a small word range makes entries overlap, as frames of one trajectory do.
fn bag(seed: u64, n: usize) -> BoW {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut words: Vec<(u32, f32)> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (
                (state % 1_000_000) as u32,
                ((state >> 40) % 1000) as f32 + 1.0,
            )
        })
        .collect();
    words.sort_unstable_by_key(|w| w.0);
    words.dedup_by_key(|w| w.0);
    let mut bow = BoW(words);
    bow.normalize_l1();
    bow
}

fn query(c: &mut Criterion) {
    let mut db = Database::new();
    for i in 0..4000 {
        db.add(&bag(i, 500));
    }
    let q = bag(17, 500);
    c.bench_function("database_query_4000x500", |b| b.iter(|| db.query(&q, 4)));
}

criterion_group!(benches, query);
criterion_main!(benches);
