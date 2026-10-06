//! RoboCap IMU gap limit and clock guard.

/// Maximum permitted acceleration bracket, nanoseconds.
pub const MAX_ACCEL_GAP_NS: i64 = 50_000_000;

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
        Self {
            max_future_ns: 2_000_000,
            max_clock_error_ns: 1_000_000_000,
            max_age_ns: 1_000_000_000,
        }
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
    fn the_clock_guard_counts_small_future_skew_and_stops_beyond_it() {
        let guard = ClockGuard::default();
        assert_eq!(guard.check(1_000, 2_000), ClockCheck::Ok);
        assert_eq!(
            guard.check(2_000_000 + 1_500_000, 2_000_000),
            ClockCheck::FutureSkew
        );
        // Cap B, 2026-10-01: a gyro sample 2.199 ms and an accel sample 20.205 ms ahead of CLOCK_MONOTONIC (the driver's FIFO
        // timestamps run ahead) each ended a live run. They are dropped and counted; the run goes on.
        assert_eq!(
            guard.check(2_000_000 + 2_199_000, 2_000_000),
            ClockCheck::FutureDropped
        );
        assert_eq!(
            guard.check(2_000_000 + 20_205_000, 2_000_000),
            ClockCheck::FutureDropped
        );
        // A second or more ahead is a wrong clock (e.g. CLOCK_REALTIME), not jitter, and a second old is a backlog: the run stops.
        assert_eq!(
            guard.check(2_000_000 + 1_000_000_001, 2_000_000),
            ClockCheck::WrongClock
        );
        assert_eq!(guard.check(0, 1_000_000_000), ClockCheck::TooOld);
        assert_eq!(guard.check(1, 1_000_000_000), ClockCheck::Ok);
    }
}
