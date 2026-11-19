//! 256-bit binary descriptors, their Hamming distance and Basalt's mutual brute-force matcher.

/// A 256-bit binary descriptor: test `i` is bit `i % 64` of word `i / 64` (the bit order of Basalt's
/// `std::bitset<256>`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Descriptor256(pub [u64; 4]);

impl Descriptor256 {
    /// The result of binary test `index` (tests only: the bit order is checked against `std::bitset`).
    #[cfg(test)]
    fn bit(&self, index: usize) -> bool {
        self.0[index / 64] >> (index % 64) & 1 == 1
    }

    /// Number of differing bits.
    ///
    /// ```
    /// use kornia_staging_imgproc::features::Descriptor256;
    /// let a = Descriptor256([0b1011, 0, 0, u64::MAX]);
    /// let b = Descriptor256([0b0001, 0, 0, 0]);
    /// assert_eq!(a.hamming(&b), 2 + 64);
    /// ```
    #[inline]
    pub fn hamming(&self, other: &Self) -> u32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

/// Acceptance rule of [`match_mutual`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchConfig {
    /// A match needs a best distance strictly below this (Basalt `loop_closing_max_hamming_distance`).
    pub max_distance: u32,
    /// A match needs `best * second_best_ratio <= second best` (Basalt `loop_closing_second_best_test_ratio`).
    pub second_best_ratio: f64,
}

impl Default for MatchConfig {
    /// RootSLAM's loop-closing values: distance below 70, second best at least 1.2 times the best.
    fn default() -> Self {
        Self {
            max_distance: 70,
            second_best_ratio: 1.2,
        }
    }
}

/// Largest possible 256-bit Hamming distance plus one.
const NO_DISTANCE: u32 = 257;

#[derive(Clone, Copy)]
struct Nearest {
    index: usize,
    best: u32,
    second: u32,
}

impl Nearest {
    const EMPTY: Self = Self {
        index: 0,
        best: NO_DISTANCE,
        second: NO_DISTANCE,
    };

    #[inline]
    fn insert(&mut self, index: usize, distance: u32) {
        if distance < self.best {
            self.second = self.best;
            self.best = distance;
            self.index = index;
        } else {
            self.second = self.second.min(distance);
        }
    }

    fn matching_index(self, count: usize, config: MatchConfig) -> Option<usize> {
        // A single candidate has no second best, so the ratio test does not apply.
        let separated = count == 1
            || (self.best < self.second
                && f64::from(self.best) * config.second_best_ratio <= f64::from(self.second));
        (self.best < NO_DISTANCE && self.best < config.max_distance && separated)
            .then_some(self.index)
    }
}

// Write contiguous distances so LLVM can vectorize popcount independently
// of the best/second-best updates.
#[inline(always)]
fn distance_row(query: &Descriptor256, candidates: &[Descriptor256], row: &mut [u32]) {
    for (out, candidate) in row.iter_mut().zip(candidates) {
        *out = query.hamming(candidate);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn fill_distances(query: &Descriptor256, candidates: &[Descriptor256], row: &mut [u32]) {
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    unsafe fn avx512(query: &Descriptor256, candidates: &[Descriptor256], row: &mut [u32]) {
        distance_row(query, candidates, row);
    }

    #[target_feature(enable = "popcnt")]
    unsafe fn popcnt(query: &Descriptor256, candidates: &[Descriptor256], row: &mut [u32]) {
        distance_row(query, candidates, row);
    }

    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vpopcntdq")
    {
        // SAFETY: Both CPU features and OS support were checked above.
        unsafe { avx512(query, candidates, row) }
    } else if std::is_x86_feature_detected!("popcnt") {
        // SAFETY: The CPU supports POPCNT.
        unsafe { popcnt(query, candidates, row) }
    } else {
        distance_row(query, candidates, row);
    }
}

#[cfg(not(target_arch = "x86_64"))]
use distance_row as fill_distances;

/// Mutual nearest-neighbour matches between two descriptor sets, as Basalt's `matchDescriptors`, ordered by `a`
/// index.
///
/// A pair `(i, j)` is kept when `a[i]` matches `b[j]` and `b[j]` matches `a[i]` under [`MatchConfig`]. Unlike
/// Basalt, a descriptor whose reverse search finds no match never passes the mutual check (Basalt reads the
/// missing reverse entry as index 0), and ties for the best distance never match (B17): Basalt's `matchFastHelper`
/// accepts them when the ratio is 1.0 or the distance is 0 and keeps the last tied index.
///
/// ```
/// use kornia_staging_imgproc::features::{match_mutual, Descriptor256, MatchConfig};
/// let a = [Descriptor256([0, 0, 0, 0]), Descriptor256([u64::MAX; 4])];
/// let b = [Descriptor256([u64::MAX, u64::MAX, u64::MAX, 1]), Descriptor256([1, 0, 0, 0])];
/// assert_eq!(match_mutual(&a, &b, MatchConfig::default()), vec![(0, 1), (1, 0)]);
/// ```
pub fn match_mutual(
    a: &[Descriptor256],
    b: &[Descriptor256],
    config: MatchConfig,
) -> Vec<(usize, usize)> {
    if config.max_distance == 0 || a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    // A distance above this bound cannot be an accepted best, tie the best,
    // or make an accepted best fail the ratio test. In that case a missing
    // second distance can stand in for any ignored distance.
    let best_limit = config.max_distance - 1;
    let ratio_limit = f64::from(best_limit) * config.second_best_ratio;
    let limit = best_limit.max(ratio_limit as u32).min(NO_DISTANCE - 1);
    let mut a_nearest = Vec::with_capacity(a.len());
    let mut b_nearest = vec![Nearest::EMPTY; b.len()];
    let mut row = vec![0; b.len()];
    for (i, query) in a.iter().enumerate() {
        fill_distances(query, b, &mut row);
        let mut nearest = Nearest::EMPTY;
        for (j, (&distance, reverse)) in row.iter().zip(&mut b_nearest).enumerate() {
            if distance <= limit {
                nearest.insert(j, distance);
                reverse.insert(i, distance);
            }
        }
        a_nearest.push(nearest);
    }

    a_nearest
        .into_iter()
        .enumerate()
        .filter_map(|(i, nearest)| {
            let j = nearest.matching_index(b.len(), config)?;
            (b_nearest[j].matching_index(a.len(), config) == Some(i)).then_some((i, j))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// A descriptor with the first `ones` bits set.
    fn ones(ones: u32) -> Descriptor256 {
        let mut words = [0u64; 4];
        for bit in 0..ones as usize {
            words[bit / 64] |= 1 << (bit % 64);
        }
        Descriptor256(words)
    }

    #[test]
    fn bits_follow_bitset_order() {
        let d = Descriptor256([1 << 3, 0, 1 << 63, 0]);
        assert!(d.bit(3) && d.bit(191));
        assert_eq!((0..256).filter(|&i| d.bit(i)).count(), 2);
    }

    #[test]
    fn ties_for_the_best_never_match() {
        let config = MatchConfig {
            max_distance: 70,
            second_best_ratio: 1.0,
        };
        // B17: Basalt with ratio 1.0 accepts the tie (best * 1.0 <= second) and keeps the last index.
        assert!(match_mutual(&[ones(0)], &[ones(5), ones(5)], config).is_empty());
        // Also at distance 0, where Basalt accepts the tie for any ratio.
        assert!(match_mutual(&[ones(0)], &[ones(0), ones(0)], MatchConfig::default()).is_empty());
        assert_eq!(
            match_mutual(&[ones(0)], &[ones(5), ones(6)], config),
            vec![(0, 0)]
        );
    }

    #[test]
    fn ratio_and_distance_gates() {
        let config = MatchConfig::default();
        assert_eq!(
            match_mutual(&[ones(0)], &[ones(10), ones(12)], config),
            vec![(0, 0)]
        );
        assert!(match_mutual(&[ones(0)], &[ones(11), ones(13)], config).is_empty());
        // The runner-up can fail the ratio test even if its own distance
        // would be too large to match.
        assert!(match_mutual(&[ones(0)], &[ones(69), ones(82)], config).is_empty());
        assert_eq!(
            match_mutual(&[ones(0)], &[ones(69), ones(83)], config),
            vec![(0, 0)]
        );
        assert!(match_mutual(&[ones(0)], &[ones(70)], config).is_empty());
        assert_eq!(match_mutual(&[ones(0)], &[ones(69)], config), vec![(0, 0)]);
        assert!(match_mutual(&[ones(0)], &[], config).is_empty());
    }

    #[test]
    fn a_missing_reverse_match_fails_the_mutual_check() {
        // B6: a[0] matches b[0], but b[0] ties between a[0] and a[1], so it has no match. Basalt's
        // `matches_2_1[kv.second] == kv.first` default-inserts 0 for the missing key and keeps (0, 0).
        let a = [ones(0), ones(0)];
        let b = [ones(3), ones(100)];
        assert!(match_mutual(&a, &b, MatchConfig::default()).is_empty());
    }

    fn descriptor() -> impl Strategy<Value = Descriptor256> {
        proptest::array::uniform4(any::<u64>()).prop_map(Descriptor256)
    }

    // The original two directional scans are the oracle for the optimized matcher.
    fn reference_best_match(
        query: &Descriptor256,
        candidates: &[Descriptor256],
        config: MatchConfig,
    ) -> Option<usize> {
        let mut best: Option<(usize, u32)> = None;
        let mut second: Option<u32> = None;
        for (index, candidate) in candidates.iter().enumerate() {
            let distance = query.hamming(candidate);
            match best {
                Some((_, best_distance)) if distance >= best_distance => {
                    if second.is_none_or(|s| distance < s) {
                        second = Some(distance);
                    }
                }
                _ => {
                    second = best.map(|(_, d)| d);
                    best = Some((index, distance));
                }
            }
        }
        let (index, distance) = best?;
        let separated = second.is_none_or(|s| {
            distance < s && f64::from(distance) * config.second_best_ratio <= f64::from(s)
        });
        (distance < config.max_distance && separated).then_some(index)
    }

    proptest! {
        #[test]
        fn mutual_matches_equal_the_original_two_pass_matcher(
            center in descriptor(),
            a_masks in proptest::collection::vec(0..=256u32, 0..32),
            b_masks in proptest::collection::vec(0..=256u32, 0..32),
            a_random in proptest::collection::vec(descriptor(), 0..8),
            b_random in proptest::collection::vec(descriptor(), 0..8),
            config in prop_oneof![
                Just(MatchConfig::default()),
                (0..=258u32, prop_oneof![
                    -2.0f64..=4.0,
                    Just(1.0),
                    Just(f64::NAN),
                    Just(f64::INFINITY),
                    Just(f64::NEG_INFINITY),
                ]).prop_map(|(max_distance, second_best_ratio)| MatchConfig {
                    max_distance,
                    second_best_ratio,
                }),
            ],
        ) {
            // Clustered descriptors produce close matches, ties and ratio-test failures.
            // Independent descriptors exercise the remaining word/bit patterns.
            let cluster = |masks: Vec<u32>, mut descriptors: Vec<Descriptor256>| {
                descriptors.extend(masks.into_iter().map(|n| {
                    let mask = ones(n).0;
                    Descriptor256(std::array::from_fn(|i| center.0[i] ^ mask[i]))
                }));
                descriptors
            };
            let a = cluster(a_masks, a_random);
            let b = cluster(b_masks, b_random);
            let expected: Vec<_> = a.iter().enumerate()
                .filter_map(|(i, d)| reference_best_match(d, &b, config).map(|j| (i, j)))
                .filter(|&(i, j)| reference_best_match(&b[j], &a, config) == Some(i))
                .collect();
            prop_assert_eq!(match_mutual(&a, &b, config), expected);
        }
    }

    proptest! {
        #[test]
        fn hamming_is_a_metric(a in descriptor(), b in descriptor(), c in descriptor()) {
            prop_assert_eq!(a.hamming(&a), 0);
            prop_assert_eq!(a.hamming(&b), b.hamming(&a));
            prop_assert!(a.hamming(&c) <= a.hamming(&b) + b.hamming(&c));
            prop_assert_eq!(a.hamming(&b) as usize, (0..256).filter(|&i| a.bit(i) != b.bit(i)).count());
        }

        /// Every mutual match is the unique nearest neighbour in both directions.
        #[test]
        fn mutual_matches_are_unique_nearest_neighbours(
            a in proptest::collection::vec(descriptor(), 0..12),
            b in proptest::collection::vec(descriptor(), 0..12),
        ) {
            let config = MatchConfig { max_distance: 257, second_best_ratio: 1.0 };
            for (i, j) in match_mutual(&a, &b, config) {
                let d = a[i].hamming(&b[j]);
                prop_assert!(b.iter().enumerate().all(|(k, x)| k == j || a[i].hamming(x) > d));
                prop_assert!(a.iter().enumerate().all(|(k, x)| k == i || x.hamming(&b[j]) > d));
            }
        }
    }
}
