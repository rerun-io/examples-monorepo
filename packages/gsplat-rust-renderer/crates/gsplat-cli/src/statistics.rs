//! Wall-clock sample statistics.
/// Equal weight per repeat; the midpoint is averaged for an even count.
pub fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());
    let mut values = samples.to_vec();
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) * 0.5
}

#[cfg(test)]
mod tests {
    #[test]
    fn repeat_headline_weights_repeats_equally() {
        assert_eq!(super::median(&[15.95, 2.15, 3.10]), 3.10);
        assert_eq!(super::median(&[2.0, 4.0]), 3.0);
    }
}
