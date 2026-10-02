//! IMU0 sample assembly: accel linearly interpolated onto the gyro timestamps (PR #270's `LiveSlam::push` loop, the same rule
//! the replay dumps' `imu.bin` uses), and the IMU clock guard.

use std::collections::VecDeque;

use crate::frame::ImuSample;

/// Largest accel spacing to interpolate across (PR #270: 50 ms); a wider gap drops the gyro samples inside it.
pub const MAX_ACCEL_GAP_NS: i64 = 50_000_000;

/// Counters of samples the combiner could not use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CombinerCounts {
    /// Gyro or accel samples whose timestamp did not follow the previous one of their stream.
    pub unordered: u64,
    /// Gyro samples before the first accel sample.
    pub before_accel: u64,
    /// Gyro samples inside an accel gap wider than [`MAX_ACCEL_GAP_NS`].
    pub accel_gap: u64,
}

/// Combines separate gyro and accel streams into [`ImuSample`]s on the gyro timestamps.
#[derive(Debug, Default)]
pub struct ImuCombiner {
    gyro: VecDeque<(i64, [f64; 3])>,
    accel: VecDeque<(i64, [f64; 3])>,
    last_gyro_ns: Option<i64>,
    last_accel_ns: Option<i64>,
    /// What was dropped so far.
    pub counts: CombinerCounts,
}

impl ImuCombiner {
    /// Add a gyro sample (rad/s).
    pub fn push_gyro(&mut self, t_ns: i64, gyro: [f64; 3]) {
        if self.last_gyro_ns.is_some_and(|last| t_ns <= last) {
            self.counts.unordered += 1;
            return;
        }
        self.last_gyro_ns = Some(t_ns);
        self.gyro.push_back((t_ns, gyro));
    }

    /// Add an accel sample (m/s^2).
    pub fn push_accel(&mut self, t_ns: i64, accel: [f64; 3]) {
        if self.last_accel_ns.is_some_and(|last| t_ns <= last) {
            self.counts.unordered += 1;
            return;
        }
        self.last_accel_ns = Some(t_ns);
        self.accel.push_back((t_ns, accel));
    }

    /// The next combined sample, once an accel sample at or after its gyro time has arrived.
    pub fn pop(&mut self) -> Option<ImuSample> {
        loop {
            let &(gt, gyro) = self.gyro.front()?;
            if self.accel.front().is_none_or(|a| a.0 > gt) {
                if self.accel.is_empty() {
                    return None;
                }
                self.gyro.pop_front();
                self.counts.before_accel += 1;
                continue;
            }
            while self.accel.get(1).is_some_and(|a| a.0 <= gt) {
                self.accel.pop_front();
            }
            let (&(at, a), &(bt, b)) = (self.accel.front()?, self.accel.get(1)?);
            if at == gt {
                self.gyro.pop_front();
                return Some(ImuSample { t_ns: gt, gyro, accel: a });
            }
            if bt - at > MAX_ACCEL_GAP_NS {
                self.gyro.pop_front();
                self.counts.accel_gap += 1;
                continue;
            }
            let alpha = (gt - at) as f64 / (bt - at) as f64;
            self.gyro.pop_front();
            return Some(ImuSample { t_ns: gt, gyro, accel: std::array::from_fn(|i| a[i] + alpha * (b[i] - a[i])) });
        }
    }
}

/// What the clock guard decided for one sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClockCheck {
    /// The sample is in the past, less than a second old.
    Ok,
    /// The sample is up to the allowed skew in the future (counted, kept).
    FutureSkew,
    /// The sample is further in the future than the allowed skew, but by less than a wrong clock would be (counted, dropped).
    FutureDropped,
    /// The sample is at least the clock-error bound in the future: the clock is wrong, and the run stops.
    WrongClock,
    /// The sample is at least the age bound old (a clock mismatch or a backlog): the run stops.
    TooOld,
}

/// The IMU clock guard: a sample up to `max_future_ns` in the future of CLOCK_MONOTONIC is kept, one further ahead but by less
/// than `max_clock_error_ns` is dropped, and one beyond that, or `max_age_ns` old (a backlog), stops the run. PR #270 stopped
/// at any future sample and saw 0.72 ms on Cap A; Cap B's IMU FIFO timestamps run 2.2-20 ms ahead at times (2026-10-01). A wrong
/// clock (IIO's default `current_timestamp_clock` is realtime) would be off by years, not milliseconds.
#[derive(Clone, Copy, Debug)]
pub struct ClockGuard {
    /// Largest future skew kept.
    pub max_future_ns: i64,
    /// Future skew from which the clock itself is wrong (the run stops).
    pub max_clock_error_ns: i64,
    /// Largest accepted age.
    pub max_age_ns: i64,
}

impl Default for ClockGuard {
    fn default() -> Self {
        Self { max_future_ns: 2_000_000, max_clock_error_ns: 1_000_000_000, max_age_ns: 1_000_000_000 }
    }
}

impl ClockGuard {
    /// Check one sample's timestamp against `now_ns`.
    pub fn check(&self, sample_ns: i64, now_ns: i64) -> ClockCheck {
        let age = now_ns - sample_ns;
        if age <= -self.max_clock_error_ns {
            ClockCheck::WrongClock
        } else if age < -self.max_future_ns {
            ClockCheck::FutureDropped
        } else if age >= self.max_age_ns {
            ClockCheck::TooOld
        } else if age < 0 {
            ClockCheck::FutureSkew
        } else {
            ClockCheck::Ok
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accel_is_interpolated_onto_gyro_stamps_and_uncovered_gyro_is_dropped() {
        let mut combiner = ImuCombiner::default();
        combiner.push_gyro(5, [1.0; 3]);
        combiner.push_accel(10, [0.0, 0.0, 10.0]);
        assert_eq!(combiner.pop(), None, "gyro 5 waits: is it before the first accel?");
        combiner.push_gyro(12, [2.0; 3]);
        combiner.push_gyro(20, [3.0; 3]);
        combiner.push_accel(20, [10.0, 0.0, 20.0]);
        let first = combiner.pop();
        assert_eq!(combiner.counts.before_accel, 1);
        assert_eq!(first, Some(ImuSample { t_ns: 12, gyro: [2.0; 3], accel: [2.0, 0.0, 12.0] }));
        assert_eq!(combiner.pop(), None, "gyro 20 needs the accel bracket after it");
        combiner.push_accel(30, [0.0; 3]);
        assert_eq!(combiner.pop(), Some(ImuSample { t_ns: 20, gyro: [3.0; 3], accel: [10.0, 0.0, 20.0] }));
        combiner.push_gyro(20, [9.0; 3]);
        assert_eq!(combiner.counts.unordered, 1);
        combiner.push_accel(30 + MAX_ACCEL_GAP_NS + 1, [0.0; 3]);
        combiner.push_gyro(40, [4.0; 3]);
        combiner.push_gyro(MAX_ACCEL_GAP_NS + 40, [5.0; 3]);
        assert_eq!(combiner.pop(), None, "gyro 40 is dropped in the gap; the next one waits for a bracket");
        assert_eq!(combiner.counts.accel_gap, 1);
    }

    #[test]
    fn the_clock_guard_counts_small_future_skew_and_stops_beyond_it() {
        let guard = ClockGuard::default();
        assert_eq!(guard.check(1_000, 2_000), ClockCheck::Ok);
        assert_eq!(guard.check(2_000_000 + 1_500_000, 2_000_000), ClockCheck::FutureSkew);
        // Cap B, 2026-10-01: a gyro sample 2.199 ms and an accel sample 20.205 ms ahead of CLOCK_MONOTONIC (the driver's FIFO
        // timestamps run ahead) each ended a live run. They are dropped and counted; the run goes on.
        assert_eq!(guard.check(2_000_000 + 2_199_000, 2_000_000), ClockCheck::FutureDropped);
        assert_eq!(guard.check(2_000_000 + 20_205_000, 2_000_000), ClockCheck::FutureDropped);
        // A second or more ahead is a wrong clock (e.g. CLOCK_REALTIME), not jitter, and a second old is a backlog: the run stops.
        assert_eq!(guard.check(2_000_000 + 1_000_000_001, 2_000_000), ClockCheck::WrongClock);
        assert_eq!(guard.check(0, 1_000_000_000), ClockCheck::TooOld);
        assert_eq!(guard.check(1, 1_000_000_000), ClockCheck::Ok);
    }
}
