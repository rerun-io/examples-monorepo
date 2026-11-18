#![allow(clippy::unwrap_used)]

use super::*;
use crate::Database;

/// `per_cluster` points around each of `centers`, jittered by a seeded generator within +-0.05.
fn clustered<const D: usize>(centers: &[[f32; D]], per_cluster: usize, seed: u64) -> Vec<[f32; D]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut points = Vec::new();
    for _ in 0..per_cluster {
        for center in centers {
            let mut p = *center;
            for x in p.iter_mut() {
                *x += rng.random_range(-0.05f32..0.05);
            }
            points.push(p);
        }
    }
    points
}

fn config(branching: usize, depth: usize, iterations: usize, seed: u64) -> VocabularyConfig {
    VocabularyConfig {
        branching,
        depth,
        iterations,
        seed,
    }
}

fn random_points<const D: usize>(n: usize, seed: u64) -> Vec<[f32; D]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| std::array::from_fn(|_| rng.random_range(-1.0f32..1.0)))
        .collect()
}

#[test]
fn rejects_invalid_settings() {
    let data = random_points::<4>(10, 0);
    assert_eq!(
        Vocabulary::train(&data, &config(1, 2, 1, 0)).unwrap_err(),
        VocabularyError::Branching(1)
    );
    assert_eq!(
        Vocabulary::train(&data, &config(2, 0, 1, 0)).unwrap_err(),
        VocabularyError::Depth(0)
    );
    assert_eq!(
        Vocabulary::train(&data, &config(10, 10, 1, 0)).unwrap_err(),
        VocabularyError::TooManyWords
    );
    assert_eq!(
        Vocabulary::<4>::train(&[], &config(2, 2, 1, 0)).unwrap_err(),
        VocabularyError::NoTrainingData
    );
}

#[test]
fn a_different_seed_gives_a_different_tree() {
    // Same seed across thread counts is the proptest `same_seed_same_vocabulary_across_thread_counts`.
    let data = random_points::<8>(5000, 1);
    let a = Vocabulary::train(&data, &config(4, 3, 5, 42)).unwrap();
    let b = Vocabulary::train(&data, &config(4, 3, 5, 43)).unwrap();
    assert_ne!(a.centroids, b.centroids);
}

#[test]
fn lloyd_iterations_put_centroids_on_cluster_means() {
    // Recovery of well-separated clusters is the proptest `well_separated_clusters_are_recovered`; this checks that
    // the centroids are the cluster means (k-means++ alone would leave them on training points).
    let centers = [[0.0f32, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]];
    let data = clustered(&centers, 50, 3);
    let vocabulary = Vocabulary::train(&data, &config(4, 1, 10, 0)).unwrap();
    for center in &centers {
        let closest = vocabulary.centroids[0]
            .iter()
            .map(|c| distance(c, center))
            .fold(f32::INFINITY, f32::min);
        assert!(closest < 1e-3, "{closest}");
    }
}

#[test]
fn word_is_the_greedy_descent() {
    let data = random_points::<6>(3000, 5);
    let vocabulary = Vocabulary::train(&data, &config(3, 3, 4, 9)).unwrap();
    assert_eq!(vocabulary.depth(), 3);
    assert_eq!(vocabulary.n_words(), 27);
    for point in random_points::<6>(200, 6) {
        let mut node = 0usize;
        for (centroids, live) in vocabulary.centroids.iter().zip(&vocabulary.live) {
            let children = node * 3..node * 3 + 3;
            node = children
                .filter(|&c| live[c])
                .min_by(|&a, &b| {
                    distance(&point, &centroids[a]).total_cmp(&distance(&point, &centroids[b]))
                })
                .unwrap();
        }
        assert_eq!(vocabulary.word(&point) as usize, node);
    }
    assert_eq!(
        vocabulary.words(&data),
        data.iter().map(|p| vocabulary.word(p)).collect::<Vec<_>>()
    );
}

#[test]
fn shallow_sample_lowers_the_depth_and_small_nodes_keep_their_points() {
    let data = random_points::<2>(30, 2);
    let vocabulary = Vocabulary::train(&data, &config(10, 5, 3, 0)).unwrap();
    assert_eq!(vocabulary.depth(), 1); // 10^2 > 30
    let five = random_points::<2>(5, 4);
    let vocabulary = Vocabulary::train(&five, &config(10, 1, 3, 0)).unwrap();
    for (i, point) in five.iter().enumerate() {
        assert_eq!(vocabulary.word(point) as usize, i);
    }
}

#[test]
fn duplicates_do_not_break_training() {
    let data = vec![[1.0f32, 2.0]; 100];
    let vocabulary = Vocabulary::train(&data, &config(4, 2, 5, 0)).unwrap();
    let word = vocabulary.word(&[1.0, 2.0]);
    assert!((word as usize) < vocabulary.n_words());
}

#[test]
fn tf_idf_matches_a_hand_computed_case() {
    // 3 images. Word 1 in images 0 and 1 (idf ln 1.5), word 2 only in image 0 (idf ln 3), word 3 everywhere (0).
    let bows = tf_idf(&[vec![1, 1, 2, 3], vec![1, 3], vec![3]]);
    let (a, b) = (2.0 * 1.5f64.ln(), 3.0f64.ln());
    let expected0 = [(1u32, a / (a + b)), (2, b / (a + b))];
    assert_eq!(bows[0].0.len(), 2);
    for (got, want) in bows[0].0.iter().zip(expected0) {
        assert_eq!(got.0, want.0);
        assert!((f64::from(got.1) - want.1).abs() < 1e-6);
    }
    assert_eq!(bows[1].0, vec![(1, 1.0)]);
    assert!(bows[2].0.is_empty());

    // DBoW2 L1 between images 0 and 1: sum over shared words of min(q, d) = a / (a + b).
    let score = a / (a + b);
    assert!((f64::from(bows[0].l1_similarity(&bows[1])) - score).abs() < 1e-6);
    let mut database = Database::new();
    for bow in &bows {
        database.add(bow);
    }
    let results = database.query(&bows[1], 0);
    assert_eq!(results[0].entry, 1);
    assert_eq!(results[1].entry, 0);
    assert!((f64::from(results[1].score) - score).abs() < 1e-6);
}

mod properties {
    use super::*;
    use proptest::prelude::*;

    fn train_in(threads: usize, data: &[[f32; 4]], cfg: &VocabularyConfig) -> Vocabulary<4> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| Vocabulary::train(data, cfg).unwrap())
    }

    /// Small random corpora: 4-d points on a coarse grid, so duplicates and ties occur.
    fn points(max: usize) -> impl Strategy<Value = Vec<[f32; 4]>> {
        proptest::collection::vec(
            proptest::array::uniform4((-8i8..8).prop_map(|v| f32::from(v) / 4.0)),
            1..max,
        )
    }

    /// Images as runs of words over a small vocabulary, so documents share words.
    fn images() -> impl Strategy<Value = Vec<Vec<u32>>> {
        proptest::collection::vec(proptest::collection::vec(0u32..12, 0..20), 1..8)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The same seed gives the same tree and the same words, with 1 thread and with 8.
        #[test]
        fn same_seed_same_vocabulary_across_thread_counts(
            data in points(600), branching in 2usize..5, depth in 1usize..4, iterations in 0usize..4, seed in any::<u64>()
        ) {
            let cfg = VocabularyConfig { branching, depth, iterations, seed };
            let one = train_in(1, &data, &cfg);
            let eight = train_in(8, &data, &cfg);
            prop_assert_eq!(&one.centroids, &eight.centroids);
            prop_assert_eq!(&one.live, &eight.live);
            prop_assert_eq!(one.words(&data), eight.words(&data));
        }

        /// Every descriptor, seen or not, lands on exactly one live leaf inside the word range.
        #[test]
        fn every_descriptor_maps_to_one_live_leaf(
            data in points(400), queries in points(50), branching in 2usize..5, depth in 1usize..4, seed in any::<u64>()
        ) {
            let vocabulary = Vocabulary::train(&data, &VocabularyConfig { branching, depth, iterations: 3, seed }).unwrap();
            let leaves = vocabulary.live.last().unwrap();
            for q in data.iter().chain(&queries) {
                let word = vocabulary.word(q) as usize;
                prop_assert!(word < vocabulary.n_words());
                prop_assert!(leaves[word], "word {} is a dead leaf", word);
            }
        }

        /// TF-IDF bags are non-negative, sorted by word, and sum to 1 unless every word was dropped.
        #[test]
        fn tf_idf_rows_are_l1_normalised_and_non_negative(images in images()) {
            for bow in tf_idf(&images) {
                prop_assert!(bow.0.iter().all(|&(_, w)| w > 0.0));
                prop_assert!(bow.0.windows(2).all(|p| p[0].0 < p[1].0));
                if !bow.0.is_empty() {
                    let total: f32 = bow.0.iter().map(|&(_, w)| w).sum();
                    prop_assert!((total - 1.0).abs() < 1e-5, "{}", total);
                }
            }
        }

        /// The L1 score is symmetric and in [0, 1], and a bag scores 1 against itself. An image whose every word was
        /// dropped has an empty bag: the Database never returns it (it has no word to share), while kornia-bow's
        /// `l1_similarity` reads it as `1 - |q - d|_1 / 2` (1 against another empty bag, 0.5 against any other);
        /// [`similarity_matrix`] scores it 0, as the Database does.
        #[test]
        fn score_is_symmetric_bounded_and_one_on_itself(images in images()) {
            let bows = tf_idf(&images);
            let mut database = Database::new();
            for bow in &bows {
                database.add(bow);
            }
            for (i, a) in bows.iter().enumerate() {
                prop_assert!((a.l1_similarity(a) - 1.0).abs() < 1e-5);
                let hits = database.query(a, 0);
                prop_assert_eq!(a.0.is_empty(), hits.is_empty());
                for hit in hits {
                    prop_assert!(!bows[hit.entry as usize].0.is_empty());
                    prop_assert!((-1e-6..=1.0 + 1e-6).contains(&hit.score), "{}", hit.score);
                    let back = database.query(&bows[hit.entry as usize], 0);
                    let reverse = back.iter().find(|h| h.entry as usize == i).unwrap();
                    prop_assert!((reverse.score - hit.score).abs() < 1e-6);
                    if hit.entry as usize == i {
                        prop_assert!((hit.score - 1.0).abs() < 1e-5);
                    }
                }
            }
        }

        /// The inverted-file top-k equals a brute-force ranking by pairwise L1 similarity (ties by entry order).
        #[test]
        fn database_top_k_equals_brute_force_ranking(images in images(), k in 1usize..6) {
            let bows = tf_idf(&images);
            let mut database = Database::new();
            for bow in &bows {
                database.add(bow);
            }
            for q in &bows {
                let mut brute: Vec<(usize, f32)> = bows.iter().enumerate()
                    .filter(|(_, d)| !d.0.is_empty() && !q.0.is_empty())
                    .map(|(e, d)| (e, q.l1_similarity(d)))
                    .filter(|&(_, s)| s > 1e-6)
                    .collect();
                brute.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                let hits = database.query(q, k);
                prop_assert!(hits.len() <= k);
                for (rank, hit) in hits.iter().enumerate() {
                    // Same score at every rank; the entry may differ only inside a tie.
                    prop_assert!((hit.score - brute[rank].1).abs() < 1e-5, "rank {}: {} vs {}", rank, hit.score, brute[rank].1);
                    prop_assert!((q.l1_similarity(&bows[hit.entry as usize]) - hit.score).abs() < 1e-5);
                }
                prop_assert_eq!(hits.len(), brute.len().min(k));
            }
        }

        /// The dense matrix equals the Database's scores on the same bags: symmetric, diagonal 1, in [0, 1], and an
        /// empty bag scores 0 against every other image.
        #[test]
        fn similarity_matrix_equals_database_scores(images in images()) {
            let scores = similarity_matrix(&images);
            let bows = tf_idf(&images);
            let n = bows.len();
            prop_assert_eq!(scores.len(), n * n);
            let mut database = Database::new();
            for bow in &bows {
                database.add(bow);
            }
            for q in 0..n {
                let mut expected = vec![0.0f32; n];
                for hit in database.query(&bows[q], 0) {
                    expected[hit.entry as usize] = hit.score.clamp(0.0, 1.0);
                }
                expected[q] = 1.0;
                prop_assert_eq!(&scores[q * n..(q + 1) * n], &expected[..]);
                for d in 0..n {
                    let s = scores[q * n + d];
                    prop_assert!((0.0..=1.0).contains(&s));
                    // f64 sums of `|q-d| - |q| - |d|` and `|d-q| - |d| - |q|` can differ in the last bit.
                    prop_assert!((s - scores[d * n + q]).abs() < 1e-6);
                    if q != d && (bows[q].0.is_empty() || bows[d].0.is_empty()) {
                        prop_assert_eq!(s, 0.0);
                    }
                }
            }
        }

        /// Well-separated clusters (spacing 10, jitter 0.05) are recovered: one word per cluster, every point on it.
        #[test]
        fn well_separated_clusters_are_recovered(
            n_clusters in 2usize..7, per_cluster in 5usize..30, seed in any::<u64>(), placement in any::<u64>()
        ) {
            let mut rng = ChaCha8Rng::seed_from_u64(placement);
            let mut centers: Vec<[f32; 3]> = Vec::new();
            while centers.len() < n_clusters {
                let c = [rng.random_range(0..6) as f32 * 10.0, rng.random_range(0..6) as f32 * 10.0, rng.random_range(0..6) as f32 * 10.0];
                if !centers.contains(&c) {
                    centers.push(c);
                }
            }
            let data = clustered(&centers, per_cluster, placement);
            let vocabulary = Vocabulary::train(&data, &VocabularyConfig { branching: n_clusters, depth: 1, iterations: 10, seed }).unwrap();
            let words: Vec<u32> = centers.iter().map(|c| vocabulary.word(c)).collect();
            let mut distinct = words.clone();
            distinct.sort_unstable();
            distinct.dedup();
            prop_assert_eq!(distinct.len(), n_clusters, "{:?}", words);
            for (point, center) in data.iter().zip(centers.iter().cycle()) {
                prop_assert_eq!(vocabulary.word(point), vocabulary.word(center));
            }
        }
    }
}
