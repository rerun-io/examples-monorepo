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

/// The index in `candidates` that `query` matches under `config`, if any.
///
/// The best distance must be below `config.max_distance`, strictly below the second best, and pass the ratio test.
/// Requiring a strict gap means a tie for the best candidate never matches, whatever the ratio; Basalt's
/// `matchFastHelper` accepts ties when the ratio is 1.0 or the distance is 0 and keeps the last tied index.
fn best_match(
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

/// Mutual nearest-neighbour matches between two descriptor sets, as Basalt's `matchDescriptors`, ordered by `a`
/// index.
///
/// A pair `(i, j)` is kept when `a[i]` matches `b[j]` and `b[j]` matches `a[i]` under [`MatchConfig`]. Unlike
/// Basalt, a descriptor whose reverse search finds no match never passes the mutual check (Basalt reads the
/// missing reverse entry as index 0), and ties for the best distance are rejected (see [`MatchConfig`]).
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
    let b_to_a: Vec<Option<usize>> = b.iter().map(|d| best_match(d, a, config)).collect();
    a.iter()
        .enumerate()
        .filter_map(|(i, d)| best_match(d, b, config).map(|j| (i, j)))
        .filter(|&(i, j)| b_to_a[j] == Some(i))
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
        // B17: Basalt with ratio 1.0 accepts the tie (best * 1.0 <= second) and keeps the last index.
        let config = MatchConfig {
            max_distance: 70,
            second_best_ratio: 1.0,
        };
        assert_eq!(best_match(&ones(0), &[ones(5), ones(5)], config), None);
        // Also at distance 0, where Basalt accepts the tie for any ratio.
        assert_eq!(
            best_match(&ones(0), &[ones(0), ones(0)], MatchConfig::default()),
            None
        );
        assert_eq!(best_match(&ones(0), &[ones(5), ones(6)], config), Some(0));
    }

    #[test]
    fn ratio_and_distance_gates() {
        let config = MatchConfig::default();
        assert_eq!(best_match(&ones(0), &[ones(10), ones(12)], config), Some(0)); // 12 <= 12
        assert_eq!(best_match(&ones(0), &[ones(11), ones(13)], config), None); // 13.2 > 13
        assert_eq!(best_match(&ones(0), &[ones(70)], config), None);
        assert_eq!(best_match(&ones(0), &[ones(69)], config), Some(0));
        assert_eq!(best_match(&ones(0), &[], config), None);
    }

    #[test]
    fn a_missing_reverse_match_fails_the_mutual_check() {
        // B6: a[0] matches b[0], but b[0] ties between a[0] and a[1], so it has no match. Basalt's
        // `matches_2_1[kv.second] == kv.first` default-inserts 0 for the missing key and keeps (0, 0).
        let a = [ones(0), ones(0)];
        let b = [ones(3), ones(100)];
        assert_eq!(best_match(&a[0], &b, MatchConfig::default()), Some(0));
        assert_eq!(best_match(&b[0], &a, MatchConfig::default()), None);
        assert!(match_mutual(&a, &b, MatchConfig::default()).is_empty());
    }

    fn descriptor() -> impl Strategy<Value = Descriptor256> {
        proptest::array::uniform4(any::<u64>()).prop_map(Descriptor256)
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
