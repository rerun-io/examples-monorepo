//! Shared input and timing conventions for native replay tools.

/// Nearest-rank percentile of sorted measurements; an empty sample is unscored.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Feed samples through the first one strictly after the frame, retaining the next cursor.
/// A failed push leaves that sample available for a retry.
pub fn feed_imu_through<T, E>(
    samples: &[T],
    cursor: &mut usize,
    t_ns: i64,
    timestamp: impl Fn(&T) -> i64,
    mut push: impl FnMut(&T) -> Result<(), E>,
) -> Result<(), E> {
    while let Some(sample) = samples.get(*cursor) {
        push(sample)?;
        *cursor += 1;
        if timestamp(sample) > t_ns {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_the_nearest_rank_and_keeps_empty_reports_unscored() {
        assert_eq!(percentile(&[2.0, 3.0, 9.0, 20.0], 0.5), 9.0);
        assert_eq!(percentile(&[2.0, 3.0, 9.0, 20.0], 0.95), 20.0);
        assert!(percentile(&[], 0.5).is_nan());
    }

    #[test]
    fn imu_feed_includes_one_sample_after_the_frame_without_repeating_it() -> Result<(), ()> {
        let samples = [10, 20, 30, 40];
        let mut cursor = 0;
        let mut fed = Vec::new();
        feed_imu_through(
            &samples,
            &mut cursor,
            20,
            |&t| t,
            |&t| {
                fed.push(t);
                Ok(())
            },
        )?;
        assert_eq!(fed, [10, 20, 30]);
        feed_imu_through(
            &samples,
            &mut cursor,
            40,
            |&t| t,
            |&t| {
                fed.push(t);
                Ok(())
            },
        )?;
        assert_eq!(fed, [10, 20, 30, 40]);
        Ok(())
    }
}
