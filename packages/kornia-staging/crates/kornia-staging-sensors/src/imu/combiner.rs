//! Acceleration interpolated onto gyro timestamps on one monotonic nanosecond clock.
use crate::SensorError;
use kornia_algebra::Vec3F64;
use std::collections::VecDeque;

/// Combined IMU reading, with acceleration interpolated at the gyro time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CombinedImuSample {
    /// Gyro timestamp on the frames' monotonic nanosecond clock.
    pub timestamp_ns: i64,
    /// Body angular velocity in rad/s.
    pub gyro: Vec3F64,
    /// Body acceleration in m/s².
    pub accel: Vec3F64,
}

/// Response to a sample that violates stream ordering or an interpolation gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapPolicy {
    /// Drop the affected sample and increment its counter.
    DropAndCount,
    /// Stop with a typed error.
    Error,
}
/// Limits and failure policy selected by the source application.
#[derive(Clone, Copy, Debug)]
pub struct ImuCombinerConfig {
    /// Maximum spacing of the two bracketing acceleration samples, positive nanoseconds.
    pub max_accel_gap_ns: i64,
    /// Maximum queued samples per channel, at least two for an acceleration bracket.
    pub max_queue_len: usize,
    /// Gap and non-increasing sample policy.
    pub gap_policy: GapPolicy,
}
/// Samples that were dropped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CombinerCounts {
    /// Non-increasing gyro or acceleration timestamps.
    pub unordered: u64,
    /// Gyro timestamps before the first acceleration timestamp.
    pub before_accel: u64,
    /// Gyro samples inside an acceleration gap wider than the configured bound.
    pub accel_gap: u64,
}
/// Stateful combiner; both input streams must use the same clock and SI units.
/// ```
/// use kornia_staging_sensors::imu::{GapPolicy, ImuCombiner, ImuCombinerConfig};
/// use kornia_algebra::Vec3F64;
/// let mut combiner = ImuCombiner::new(ImuCombinerConfig { max_accel_gap_ns: 50,
///     max_queue_len: 8, gap_policy: GapPolicy::Error })?;
/// combiner.push_accel(0, Vec3F64::ZERO)?;
/// combiner.push_accel(20, Vec3F64::new(2.0, 4.0, 6.0))?;
/// combiner.push_gyro(10, Vec3F64::ZERO)?;
/// assert_eq!(combiner.pop()?.unwrap().accel.to_array(), [1.0, 2.0, 3.0]);
/// # Ok::<(), kornia_staging_sensors::SensorError>(())
/// ```
#[derive(Debug)]
pub struct ImuCombiner {
    config: ImuCombinerConfig,
    gyro: VecDeque<(i64, Vec3F64)>,
    accel: VecDeque<(i64, Vec3F64)>,
    last_gyro_ns: Option<i64>,
    last_accel_ns: Option<i64>,
    /// Drop counts since construction.
    pub counts: CombinerCounts,
}
impl ImuCombiner {
    /// Validate configuration before buffering samples.
    /// # Arguments
    /// * `config` - source-owned limits and policies.
    /// # Errors
    /// Rejects nonpositive gaps or queue limits below two.
    pub fn new(config: ImuCombinerConfig) -> Result<Self, SensorError> {
        if config.max_accel_gap_ns <= 0 || config.max_queue_len < 2 {
            return Err(SensorError::InvalidConfig(
                "positive IMU gap and at least two queued samples required",
            ));
        }
        Ok(Self {
            config,
            gyro: VecDeque::new(),
            accel: VecDeque::new(),
            last_gyro_ns: None,
            last_accel_ns: None,
            counts: CombinerCounts::default(),
        })
    }
    /// Queue angular velocity in rad/s.
    /// # Arguments
    /// * `timestamp_ns` - strictly increasing gyro time, nanoseconds.
    /// * `gyro` - finite three-axis angular velocity.
    /// # Errors
    /// Nonfinite input, exhausted queue, or unordered input under error policy.
    #[inline]
    pub fn push_gyro(&mut self, timestamp_ns: i64, gyro: Vec3F64) -> Result<(), SensorError> {
        self.push(timestamp_ns, gyro, true)
    }
    /// Queue acceleration in m/s².
    /// # Arguments
    /// * `timestamp_ns` - strictly increasing acceleration time, nanoseconds.
    /// * `accel` - finite three-axis acceleration.
    /// # Errors
    /// Nonfinite input, exhausted queue, or unordered input under error policy.
    #[inline]
    pub fn push_accel(&mut self, timestamp_ns: i64, accel: Vec3F64) -> Result<(), SensorError> {
        self.push(timestamp_ns, accel, false)
    }
    fn push(&mut self, timestamp_ns: i64, value: Vec3F64, gyro: bool) -> Result<(), SensorError> {
        if value.to_array().iter().any(|v| !v.is_finite()) {
            return Err(SensorError::NonFiniteInput);
        }
        let (queue, last) = if gyro {
            (&mut self.gyro, &mut self.last_gyro_ns)
        } else {
            (&mut self.accel, &mut self.last_accel_ns)
        };
        if let Some(previous_timestamp_ns) = last.filter(|last| timestamp_ns <= *last) {
            if self.config.gap_policy == GapPolicy::Error {
                return Err(SensorError::NonMonotonicSample {
                    previous_timestamp_ns,
                    timestamp_ns,
                });
            }
            self.counts.unordered += 1;
            return Ok(());
        }
        if queue.len() == self.config.max_queue_len {
            return Err(SensorError::ImuQueueFull);
        }
        *last = Some(timestamp_ns);
        queue.push_back((timestamp_ns, value));
        Ok(())
    }
    /// Produce the next gyro sample once acceleration brackets it.
    /// Waits for an acceleration sample strictly after the gyro time. Exact matches
    /// use that acceleration even when the following gap exceeds the bound.
    /// # Errors
    /// A wide acceleration gap under error policy, or a nonfinite interpolation result.
    pub fn pop(&mut self) -> Result<Option<CombinedImuSample>, SensorError> {
        loop {
            let Some(&(gt, gyro)) = self.gyro.front() else {
                return Ok(None);
            };
            let Some(&(first, _)) = self.accel.front() else {
                return Ok(None);
            };
            if first > gt {
                self.gyro.pop_front();
                self.counts.before_accel += 1;
                continue;
            }
            while self.accel.get(1).is_some_and(|a| a.0 <= gt) {
                self.accel.pop_front();
            }
            let (Some(&(at, a)), Some(&(bt, b))) = (self.accel.front(), self.accel.get(1)) else {
                return Ok(None);
            };
            if at == gt {
                self.gyro.pop_front();
                return Ok(Some(CombinedImuSample {
                    timestamp_ns: gt,
                    gyro,
                    accel: a,
                }));
            }
            let gap = bt.abs_diff(at);
            if gap > self.config.max_accel_gap_ns as u64 {
                if self.config.gap_policy == GapPolicy::Error {
                    return Err(SensorError::AccelGap { gap_ns: gap });
                }
                self.gyro.pop_front();
                self.counts.accel_gap += 1;
                continue;
            }
            let alpha = (gt as i128 - at as i128) as f64 / (bt as i128 - at as i128) as f64;
            let (a, b) = (a.to_array(), b.to_array());
            let accel = Vec3F64::from_array(std::array::from_fn(|i| a[i] + alpha * (b[i] - a[i])));
            if accel.to_array().iter().any(|v| !v.is_finite()) {
                return Err(SensorError::NonFiniteInput);
            }
            self.gyro.pop_front();
            return Ok(Some(CombinedImuSample {
                timestamp_ns: gt,
                gyro,
                accel,
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MAX_ACCEL_GAP_NS: i64 = 50_000_000;
    fn combiner(gap_policy: GapPolicy) -> ImuCombiner {
        ImuCombiner::new(ImuCombinerConfig {
            max_accel_gap_ns: MAX_ACCEL_GAP_NS,
            max_queue_len: 512,
            gap_policy,
        })
        .unwrap()
    }
    #[test]
    fn accel_is_interpolated_onto_gyro_stamps_and_uncovered_gyro_is_dropped() {
        let mut combiner = combiner(GapPolicy::DropAndCount);
        combiner.push_gyro(5, [1.0; 3].into()).unwrap();
        combiner.push_accel(10, [0.0, 0.0, 10.0].into()).unwrap();
        assert_eq!(
            combiner.pop().unwrap(),
            None,
            "gyro 5 waits: is it before the first accel?"
        );
        combiner.push_gyro(12, [2.0; 3].into()).unwrap();
        combiner.push_gyro(20, [3.0; 3].into()).unwrap();
        combiner.push_accel(20, [10.0, 0.0, 20.0].into()).unwrap();
        let first = combiner.pop().unwrap();
        assert_eq!(combiner.counts.before_accel, 1);
        assert_eq!(
            first,
            Some(CombinedImuSample {
                timestamp_ns: 12,
                gyro: [2.0; 3].into(),
                accel: [2.0, 0.0, 12.0].into()
            })
        );
        assert_eq!(
            combiner.pop().unwrap(),
            None,
            "gyro 20 needs the accel bracket after it"
        );
        combiner.push_accel(30, [0.0; 3].into()).unwrap();
        assert_eq!(
            combiner.pop().unwrap(),
            Some(CombinedImuSample {
                timestamp_ns: 20,
                gyro: [3.0; 3].into(),
                accel: [10.0, 0.0, 20.0].into()
            })
        );
        combiner.push_gyro(20, [9.0; 3].into()).unwrap();
        assert_eq!(combiner.counts.unordered, 1);
        combiner
            .push_accel(30 + MAX_ACCEL_GAP_NS + 1, [0.0; 3].into())
            .unwrap();
        combiner.push_gyro(40, [4.0; 3].into()).unwrap();
        combiner
            .push_gyro(MAX_ACCEL_GAP_NS + 40, [5.0; 3].into())
            .unwrap();
        assert_eq!(
            combiner.pop().unwrap(),
            None,
            "gyro 40 is dropped in the gap; the next one waits for a bracket"
        );
        assert_eq!(combiner.counts.accel_gap, 1);
    }

    #[test]
    fn error_policy_checks_interpolation_gaps_and_order() -> Result<(), SensorError> {
        let mut stream = combiner(GapPolicy::Error);
        stream.push_accel(0, Vec3F64::ZERO)?;
        stream.push_accel(MAX_ACCEL_GAP_NS + 1, Vec3F64::ZERO)?;
        stream.push_gyro(1, Vec3F64::ZERO)?;
        assert!(matches!(stream.pop(), Err(SensorError::AccelGap { .. })));
        assert_eq!(
            stream.push_gyro(0, Vec3F64::ZERO),
            Err(SensorError::NonMonotonicSample {
                previous_timestamp_ns: 1,
                timestamp_ns: 0
            })
        );
        assert_eq!(
            stream.push_accel(1, Vec3F64::new(f64::NAN, 0.0, 0.0)),
            Err(SensorError::NonFiniteInput)
        );
        let mut live = combiner(GapPolicy::Error);
        live.push_accel(0, Vec3F64::ZERO)?;
        live.push_accel(MAX_ACCEL_GAP_NS + 1, Vec3F64::ZERO)?;
        live.push_gyro(0, Vec3F64::ZERO)?;
        assert!(
            live.pop()?.is_some(),
            "exact samples do not interpolate across the following gap"
        );
        Ok(())
    }
    #[test]
    fn extreme_timestamp_gap_and_queue_limits_do_not_overflow() -> Result<(), SensorError> {
        let mut stream = combiner(GapPolicy::Error);
        stream.push_accel(i64::MIN, Vec3F64::ZERO)?;
        stream.push_accel(i64::MAX, Vec3F64::ZERO)?;
        stream.push_gyro(0, Vec3F64::ZERO)?;
        assert_eq!(
            stream.pop(),
            Err(SensorError::AccelGap { gap_ns: u64::MAX })
        );
        let config = ImuCombinerConfig {
            max_accel_gap_ns: 10,
            max_queue_len: 2,
            gap_policy: GapPolicy::Error,
        };
        let mut stream = ImuCombiner::new(config)?;
        stream.push_gyro(0, Vec3F64::ZERO)?;
        stream.push_gyro(1, Vec3F64::ZERO)?;
        assert_eq!(
            stream.push_gyro(2, Vec3F64::ZERO),
            Err(SensorError::ImuQueueFull)
        );
        assert!(ImuCombiner::new(ImuCombinerConfig {
            max_queue_len: 1,
            ..config
        })
        .is_err());
        assert!(ImuCombiner::new(ImuCombinerConfig {
            max_accel_gap_ns: 0,
            ..config
        })
        .is_err());
        Ok(())
    }
}
