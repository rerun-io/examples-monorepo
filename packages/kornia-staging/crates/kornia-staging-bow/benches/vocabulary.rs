//! Per-run vocabulary build, sized like a retrieval index over a video: train (a 200 000-descriptor sample, 10 x 5,
//! 10 Lloyd iterations), assign every descriptor to its word, and the dense score matrix ([`tf_idf`] then
//! [`l1_similarity_matrix`]).
//!
//! Synthetic clustered data: `n_images x 2 048` 128-d descriptors. Image `i` draws its descriptors around 2 048
//! centers picked from a window of 8 192 consecutive centers starting at `16 * i`, so neighbouring images share most of
//! their structure and distant ones little, as frames along a trajectory do. Two sizes: 2 000 images (4.1 M
//! descriptors, 2 GB) and 18 000 images (36.9 M descriptors, 19 GB; set `BOW_BENCH_SMALL_ONLY=1` to skip it).
//!
//! Threads: rayon's default pool; set `RAYON_NUM_THREADS` to compare.

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_bow::l1_similarity_matrix;
use kornia_staging_bow::vocabulary::{tf_idf, Vocabulary, VocabularyConfig};
use rand::rngs::ChaCha8Rng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

const DIM: usize = 128;
const PER_IMAGE: usize = 2048;
const WINDOW: usize = 8192;
const STRIDE: usize = 16;
const SAMPLE: usize = 200_000;

/// `n_images` images of clustered descriptors (module docs), deterministic.
fn images(n_images: usize) -> Vec<Vec<[f32; DIM]>> {
    let n_centers = STRIDE * n_images + WINDOW;
    let centers: Vec<[f32; DIM]> = (0..n_centers)
        .into_par_iter()
        .map(|c| {
            let mut rng = ChaCha8Rng::seed_from_u64(c as u64);
            std::array::from_fn(|_| rng.random_range(-1.0f32..1.0))
        })
        .collect();
    (0..n_images)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64((1 << 40) | i as u64);
            (0..PER_IMAGE)
                .map(|_| {
                    let center = &centers[STRIDE * i + rng.random_range(0..WINDOW)];
                    center.map(|x| x + rng.random_range(-0.1f32..0.1))
                })
                .collect()
        })
        .collect()
}

/// An even-stride training sample of about `SAMPLE` descriptors, the same quota from every image.
fn sample(images: &[Vec<[f32; DIM]>]) -> Vec<[f32; DIM]> {
    let quota = SAMPLE.div_ceil(images.len()).max(1);
    let step = (PER_IMAGE / quota).max(1);
    images
        .iter()
        .flat_map(|image| image.iter().step_by(step).take(quota).copied())
        .collect()
}

fn assign(vocabulary: &Vocabulary<DIM>, images: &[Vec<[f32; DIM]>]) -> Vec<Vec<u32>> {
    images
        .par_iter()
        .map(|image| vocabulary.words(image))
        .collect()
}

fn bench_size(c: &mut Criterion, n_images: usize) {
    let images = images(n_images);
    let config = VocabularyConfig::default();
    let training = sample(&images);
    let vocabulary = Vocabulary::train(&training, &config).expect("train");
    let words = assign(&vocabulary, &images);
    let mut group = c.benchmark_group(format!("vocabulary_{n_images}_images"));
    group.sample_size(10);
    group.bench_function("train", |b| {
        b.iter(|| Vocabulary::train(&training, &config).expect("train"))
    });
    group.bench_function("assign", |b| b.iter(|| assign(&vocabulary, &images)));
    group.bench_function("similarity_matrix", |b| {
        b.iter(|| l1_similarity_matrix(&tf_idf(&words)))
    });
    group.finish();
}

fn vocabulary(c: &mut Criterion) {
    bench_size(c, 2_000);
    if std::env::var_os("BOW_BENCH_SMALL_ONLY").is_none() {
        bench_size(c, 18_000);
    }
}

criterion_group!(benches, vocabulary);
criterion_main!(benches);
