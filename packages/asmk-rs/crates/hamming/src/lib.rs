//! ASMK's binary signatures (jenicek/asmk@2a96d9c `cython/hamming.pyx`).
//!
//! A float vector is binarized by `x > threshold` and packed MSB-first into `u32` words; the last word of a
//! row whose length is not a multiple of 32 is left-aligned (its low bits are zero). The Hamming distance of
//! two packed rows is the popcount of their XOR divided by a normalization, by default the number of packed
//! bits (32 per word), in `f32` as the Cython module computes it.

/// Number of `u32` words that hold `len` bits.
pub fn packed_len(len: usize) -> usize {
    len.div_ceil(32)
}

/// Binarize `row > threshold` and pack it into `out` (`packed_len(row.len())` words), MSB first.
pub fn binarize_and_pack(row: &[f32], threshold: i32, out: &mut [u32]) {
    assert_eq!(
        out.len(),
        packed_len(row.len()),
        "output must hold packed_len(row.len()) words"
    );
    let threshold = threshold as f32;
    for (word, chunk) in out.iter_mut().zip(row.chunks(32)) {
        let bits = chunk
            .iter()
            .fold(0u32, |acc, &x| (acc << 1) | u32::from(x > threshold));
        // The Cython module shifts the last, partial word left by the missing bit count.
        *word = bits << (32 - chunk.len());
    }
}

/// Hamming distance of two packed rows divided by `normalization` (0 = the number of bits, 32 per word).
pub fn hamming_dist_packed(a: &[u32], b: &[u32], normalization: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "packed rows must have the same length");
    let normalization = if normalization == 0.0 {
        (a.len() * 32) as f32
    } else {
        normalization
    };
    let bits: i32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x ^ y).count_ones() as i32)
        .sum();
    bits as f32 / normalization
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_msb_first_and_left_aligns_the_last_word() {
        let row = [1.0, -1.0, 1.0];
        let mut out = [0u32; 1];
        binarize_and_pack(&row, 0, &mut out);
        assert_eq!(out[0], 0b101 << 29);
    }

    #[test]
    fn full_words_are_not_shifted() {
        let row: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut out = [0u32; 2];
        binarize_and_pack(&row, 0, &mut out);
        assert_eq!(out, [0xAAAA_AAAA, 0xAAAA_AAAA]);
    }

    #[test]
    fn threshold_is_strict_and_nan_is_false() {
        let row = [0.0, 0.5, f32::NAN, 2.0];
        let mut out = [0u32; 1];
        binarize_and_pack(&row, 0, &mut out);
        assert_eq!(out[0], 0b0101 << 28);
    }

    #[test]
    fn distance_matches_the_docstring_examples() {
        assert_eq!(hamming_dist_packed(&[3], &[1], 2.0), 0.5);
        assert_eq!(hamming_dist_packed(&[1], &[2], 2.0), 1.0);
        assert_eq!(hamming_dist_packed(&[u32::MAX, 0], &[0, 0], 0.0), 0.5);
    }
}
