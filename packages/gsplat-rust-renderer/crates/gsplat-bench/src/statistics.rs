//! Wall-clock sample statistics (nearest-rank percentiles).
use serde::{Deserialize, Serialize};
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statistics {
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub fps: f64,
}
pub fn summarize(samples: &[f64]) -> crate::Result<Statistics> {
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(crate::Error::Invalid(
            "timings must be finite, positive, and nonempty".into(),
        ));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |q: f64| sorted[((q * sorted.len() as f64).ceil() as usize).saturating_sub(1)];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    Ok(Statistics {
        median_ms: percentile(0.5),
        p95_ms: percentile(0.95),
        p99_ms: percentile(0.99),
        mean_ms: mean,
        fps: 1000.0 / mean,
    })
}
/// Equal weight per repeat; the midpoint is averaged for an even count.
pub fn median(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty());
    let mut values = samples.to_vec();
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_percentiles_and_throughput() {
        let s = summarize(&(1..=100).map(f64::from).collect::<Vec<_>>()).unwrap();
        assert_eq!(s.median_ms, 50.0);
        assert_eq!(s.p95_ms, 95.0);
        assert_eq!(s.p99_ms, 99.0);
        assert_eq!(s.mean_ms, 50.5);
        assert_eq!(s.fps, 1000.0 / 50.5);
    }
    #[test]
    fn rejects_invalid_samples() {
        assert!(summarize(&[]).is_err());
        assert!(summarize(&[f64::NAN]).is_err());
    }
    #[test]
    fn repeat_headline_weights_repeats_equally() {
        assert_eq!(super::median(&[15.95, 2.15, 3.10]), 3.10);
        assert_eq!(super::median(&[2.0, 4.0]), 3.0);
    }
}
