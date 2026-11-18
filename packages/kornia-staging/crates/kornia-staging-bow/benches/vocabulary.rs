//! Per-run vocabulary build on real learned descriptors: train (200 000-descriptor sample, 10 x 5, 10 Lloyd
//! iterations), assign every descriptor to its word, and score (TF-IDF, inverted-file database, one query per image
//! into a dense `n x n` matrix, as colsfm's retrieval index does).
//!
//! Reads 128-d float32 descriptors from a COLMAP database named by `BOW_BENCH_DB` (image names `<camera>/<t_ns>`,
//! e.g. a colsfm run's `database.db`), and skips when it is unset. Two sets:
//! * `2k`: the images with `120 s <= t <= 230 s` (clip B: 1 974 images, 3.7 M descriptors);
//! * `18k`: every image four times, each copy jittered by a seeded +-0.01 per coordinate (clip B: 18 168 images,
//!   34 M descriptors, about 18 GB), standing in for a long clip when no 18k-image database is at hand.
//!
//! Threads: rayon's default pool; set `RAYON_NUM_THREADS` to compare.

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_bow::vocabulary::{tf_idf, Vocabulary, VocabularyConfig};
use kornia_staging_bow::Database;
use rand::rngs::ChaCha8Rng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use rusqlite::{Connection, OpenFlags};

const DIM: usize = 128;
const SAMPLE: usize = 200_000;

/// `(t_ns, descriptors)` of every image in the database, in image-id order.
fn read_images(path: &str) -> Vec<(i64, Vec<[f32; DIM]>)> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("open the database");
    let mut statement = conn
        .prepare("SELECT i.name, d.data FROM images i JOIN descriptors d USING (image_id) ORDER BY i.image_id")
        .expect("query descriptors");
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .expect("read descriptors");
    rows.map(|row| {
        let (name, blob) = row.expect("a descriptor row");
        let t_ns: i64 = name
            .rsplit('/')
            .next()
            .and_then(|t| t.parse().ok())
            .unwrap_or(0);
        let (bytes, _) = blob.as_chunks::<4>();
        let values: Vec<f32> = bytes.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let (rows, _) = values.as_chunks::<DIM>();
        let descriptors = rows.to_vec();
        (t_ns, descriptors)
    })
    .collect()
}

/// An even-stride training sample of about `SAMPLE` descriptors, the same quota from every image.
fn sample(images: &[Vec<[f32; DIM]>]) -> Vec<[f32; DIM]> {
    let quota = SAMPLE.div_ceil(images.len()).max(1);
    images
        .iter()
        .flat_map(|image| {
            let step = (image.len() / quota).max(1);
            image.iter().step_by(step).take(quota).copied()
        })
        .collect()
}

fn assign(vocabulary: &Vocabulary<DIM>, images: &[Vec<[f32; DIM]>]) -> Vec<Vec<u32>> {
    images
        .par_iter()
        .map(|image| vocabulary.words(image))
        .collect()
}

/// TF-IDF, the database, and the dense score matrix (one query per image).
fn score(words: &[Vec<u32>]) -> Vec<f32> {
    let mut offsets = vec![0usize];
    for image in words {
        offsets.push(offsets[offsets.len() - 1] + image.len());
    }
    let bows = tf_idf(&words.concat(), &offsets).expect("valid offsets");
    let mut database = Database::new();
    for bow in &bows {
        database.add(bow);
    }
    let n = bows.len();
    let mut scores = vec![0.0f32; n * n];
    scores.par_chunks_mut(n).enumerate().for_each(|(q, row)| {
        for hit in database.query(&bows[q], 0) {
            row[hit.entry as usize] = hit.score;
        }
    });
    scores
}

fn bench_set(c: &mut Criterion, name: &str, images: &[Vec<[f32; DIM]>]) {
    let config = VocabularyConfig::default();
    let training = sample(images);
    let vocabulary = Vocabulary::train(&training, &config).expect("train");
    let words = assign(&vocabulary, images);
    let mut group = c.benchmark_group(format!("vocabulary_{name}_{}_images", images.len()));
    group.sample_size(10);
    group.bench_function("train", |b| {
        b.iter(|| Vocabulary::train(&training, &config).expect("train"))
    });
    group.bench_function("assign", |b| b.iter(|| assign(&vocabulary, images)));
    group.bench_function("score", |b| b.iter(|| score(&words)));
    group.finish();
}

fn vocabulary(c: &mut Criterion) {
    let Ok(path) = std::env::var("BOW_BENCH_DB") else {
        eprintln!("BOW_BENCH_DB is not set: skipping the vocabulary bench (point it at a colsfm database.db)");
        return;
    };
    let all = read_images(&path);
    let two_k: Vec<Vec<[f32; DIM]>> = all
        .iter()
        .filter(|(t, _)| (120_000_000_000..=230_000_000_000).contains(t))
        .map(|(_, d)| d.clone())
        .collect();
    bench_set(c, "2k", &two_k);
    drop(two_k);

    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut eighteen_k: Vec<Vec<[f32; DIM]>> = Vec::with_capacity(4 * all.len());
    for copy in 0..4 {
        for (_, image) in &all {
            eighteen_k.push(if copy == 0 {
                image.clone()
            } else {
                image
                    .iter()
                    .map(|d| d.map(|x| x + rng.random_range(-0.01f32..0.01)))
                    .collect()
            });
        }
    }
    drop(all);
    bench_set(c, "18k", &eighteen_k);
}

criterion_group!(benches, vocabulary);
criterion_main!(benches);
