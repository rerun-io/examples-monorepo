//! Capture clock health and bounded trigger recovery (live sources only).

use crate::frame::NUM_CAMERAS;
use std::collections::VecDeque;
use std::time::Duration;

/// Health of the live camera clocks. Replay has no trigger recovery.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(into = "&'static str")]
pub enum CaptureSync {
    /// All six cameras are within the matcher tolerance.
    #[default]
    InSync,
    /// A sustained split or absent camera needs attention.
    OutOfSync,
    /// Trigger recovery is in progress; fresh good ticks have not yet confirmed it.
    Resyncing,
}

impl CaptureSync {
    /// Stable text used in diagnostics and the panel.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InSync => "in_sync",
            Self::OutOfSync => "out_of_sync",
            Self::Resyncing => "resyncing",
        }
    }
}

impl From<CaptureSync> for &'static str {
    fn from(state: CaptureSync) -> Self { state.as_str() }
}

const PERIOD_NS: i64 = 33_333_333;
const REQUIRED_TICKS: u32 = 15;

struct Tick {
    anchor: i64,
    stamps: [Option<i64>; NUM_CAMERAS],
}

/// Timestamp-only observer: it never holds camera buffers or changes matcher membership.
pub(crate) struct SyncMonitor {
    tolerance: i64,
    ticks: VecDeque<Tick>,
    bad: [u32; NUM_CAMERAS],
    good: u32,
    incompatible: u32,
    state: CaptureSync,
    newest: i64,
    delivery: Option<(u64, Duration)>,
}

impl SyncMonitor {
    pub(crate) fn new(tolerance: i64) -> Self {
        Self {
            tolerance,
            ticks: VecDeque::new(),
            bad: [0; NUM_CAMERAS],
            good: 0,
            incompatible: 0,
            state: CaptureSync::InSync,
            newest: i64::MIN,
            delivery: None,
        }
    }

    /// Delivery counters advance on camera threads even while the consumer is busy.
    pub(crate) fn delivery(&mut self, now: Duration, frames: u64) {
        match self.delivery {
            Some((previous, _)) if previous != frames => self.delivery = Some((frames, now)),
            Some((_, last)) if now.saturating_sub(last) >= Duration::from_secs(1) => self.missing(),
            None if frames > 0 => self.delivery = Some((frames, now)),
            _ => {}
        }
    }

    pub(crate) fn restart(&mut self) {
        *self = Self { state: CaptureSync::Resyncing, ..Self::new(self.tolerance) };
    }

    pub(crate) fn missing(&mut self) {
        self.state = CaptureSync::OutOfSync;
    }

    pub(crate) fn state(&self) -> CaptureSync {
        self.state
    }

    pub(crate) fn frame(&mut self, camera: usize, stamp: i64) {
        self.newest = self.newest.max(stamp);
        // Associate EOF stamps over half a trigger period for diagnosis only. The real matcher
        // still uses 3 ms. Sequence numbers cannot identify a shared tick after a driver drop.
        if let Some(tick) = self.ticks.iter_mut().find(|tick| {
            tick.stamps[camera].is_none() && stamp.abs_diff(tick.anchor) <= (PERIOD_NS / 2) as u64
        }) {
            tick.stamps[camera] = Some(stamp);
        } else {
            let mut stamps = [None; NUM_CAMERAS];
            stamps[camera] = Some(stamp);
            let index = self.ticks.partition_point(|tick| tick.anchor < stamp);
            self.ticks.insert(
                index,
                Tick {
                    anchor: stamp,
                    stamps,
                },
            );
        }
        while self.ticks.front().is_some_and(|tick| {
            tick.stamps.iter().all(Option::is_some)
                || self.newest.saturating_sub(tick.anchor) > 2 * PERIOD_NS
        }) {
            let tick = self.ticks.pop_front().expect("front exists");
            let mut stamps: Vec<i64> = tick.stamps.iter().flatten().copied().collect();
            stamps.sort_unstable();
            let median = stamps[stamps.len() / 2];
            // Use the middle pair for six samples, without adding two large absolute timestamps.
            let median = if stamps.len() == NUM_CAMERAS {
                stamps[NUM_CAMERAS / 2 - 1] + (median - stamps[NUM_CAMERAS / 2 - 1]) / 2
            } else {
                median
            };
            // A balanced 3+3 split can put every median offset below tolerance while
            // still being incompatible with the matcher's first-arrival anchor.
            let compatible = tick.stamps.iter().all(|stamp| {
                stamp.is_some_and(|stamp| stamp.abs_diff(tick.anchor) <= self.tolerance as u64)
            });
            self.incompatible = if compatible {
                0
            } else {
                self.incompatible.saturating_add(1)
            };
            let mut good = compatible;
            for (camera, stamp) in tick.stamps.iter().enumerate() {
                let outside =
                    stamp.is_none_or(|stamp| stamp.abs_diff(median) > self.tolerance as u64);
                self.bad[camera] = if outside {
                    self.bad[camera].saturating_add(1)
                } else {
                    0
                };
                good &= !outside;
            }
            self.good = if good { self.good.saturating_add(1) } else { 0 };
            if self.incompatible >= REQUIRED_TICKS || self.bad.iter().any(|&n| n >= REQUIRED_TICKS)
            {
                self.state = CaptureSync::OutOfSync;
            } else if self.good >= REQUIRED_TICKS {
                self.state = CaptureSync::InSync;
            }
        }
    }
}

/// At most three attempts per outage; confirmed good ticks restore the budget.
#[derive(Default)]
pub(crate) struct Recovery {
    pub(crate) attempts: u32,
    pub(crate) total: u32,
    next_attempt: Duration,
}

impl Recovery {
    pub(crate) fn due(&mut self, now: Duration, sync: &SyncMonitor) -> bool {
        if sync.state() == CaptureSync::InSync && sync.good >= REQUIRED_TICKS {
            self.attempts = 0;
            self.next_attempt = Duration::ZERO;
        }
        if sync.state() != CaptureSync::OutOfSync || self.attempts >= 3 || now < self.next_attempt {
            return false;
        }
        self.attempts += 1;
        self.total += 1;
        self.next_attempt = now + Duration::from_secs(1 << self.attempts);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivery_clock_ignores_consumer_pauses_and_starts_with_a_frame() {
        let mut sync = SyncMonitor::new(3_000_000);
        sync.delivery(Duration::from_secs(5), 0);
        assert_eq!(sync.state(), CaptureSync::InSync);
        sync.delivery(Duration::from_secs(6), 1);
        sync.delivery(Duration::from_secs(8), 100);
        assert_eq!(sync.state(), CaptureSync::InSync);
        sync.delivery(Duration::from_secs(10), 100);
        assert_eq!(sync.state(), CaptureSync::OutOfSync);
    }

    #[test]
    fn split_requires_fifteen_consecutive_ticks_and_uses_six_camera_median() {
        let mut sync = SyncMonitor::new(3_000_000);
        for tick in 0..30 {
            for camera in (0..6).rev() {
                sync.frame(
                    camera,
                    tick * 33_333_333 + if camera == 3 { 5_000_000 } else { 0 },
                );
            }
            assert_eq!(
                sync.state(),
                if tick < 14 {
                    CaptureSync::InSync
                } else {
                    CaptureSync::OutOfSync
                }
            );
        }
    }

    #[test]
    fn balanced_split_cannot_hide_inside_the_median_tolerance() {
        let mut sync = SyncMonitor::new(3_000_000);
        for tick in 0..15 {
            for camera in 0..6 {
                sync.frame(
                    camera,
                    tick * PERIOD_NS + if camera >= 3 { 5_000_000 } else { 0 },
                );
            }
        }
        assert_eq!(sync.state(), CaptureSync::OutOfSync);
    }

    #[derive(Default)]
    struct FakeTrigger {
        calls: Vec<&'static str>,
    }
    impl FakeTrigger {
        fn stop(&mut self) -> Result<(), String> {
            self.calls.push("stop");
            Ok(())
        }
        fn drain(&mut self) -> Result<(), String> {
            self.calls.push("drain");
            Ok(())
        }
        fn start(&mut self) -> Result<(), String> {
            self.calls.push("start");
            Ok(())
        }
    }

    #[test]
    fn recovery_stops_drains_starts_and_requires_fresh_good_ticks() {
        let mut sync = SyncMonitor::new(3_000_000);
        let mut recovery = Recovery::default();
        let mut driver = FakeTrigger::default();
        for tick in 0..15 {
            for c in 0..6 {
                sync.frame(c, tick * PERIOD_NS + if c == 3 { 5_000_000 } else { 0 });
            }
        }
        assert!(recovery.due(Duration::ZERO, &sync));
        driver.stop().unwrap();
        driver.drain().unwrap();
        driver.start().unwrap();
        sync.restart();
        assert_eq!(driver.calls, ["stop", "drain", "start"]);
        assert_eq!(sync.state(), CaptureSync::Resyncing);
        for tick in 15..29 {
            for c in 0..6 {
                sync.frame(c, tick * PERIOD_NS);
            }
        }
        assert_eq!(sync.state(), CaptureSync::Resyncing);
        for c in 0..6 {
            sync.frame(c, 29 * PERIOD_NS);
        }
        assert_eq!(sync.state(), CaptureSync::InSync);
        assert_eq!(recovery.attempts, 1);
    }

    #[test]
    fn failed_relock_has_backoff_and_a_three_attempt_budget() {
        let mut sync = SyncMonitor::new(3_000_000);
        let mut recovery = Recovery::default();
        let mut driver = FakeTrigger::default();
        for second in 0..10 {
            sync.state = CaptureSync::OutOfSync;
            if recovery.due(Duration::from_secs(second), &sync) {
                driver.stop().unwrap();
                driver.drain().unwrap();
                driver.start().unwrap();
                sync.restart();
            }
        }
        assert_eq!(recovery.attempts, 3);
        assert_eq!(driver.calls.len(), 9);
        for tick in 0..15 { for camera in 0..NUM_CAMERAS { sync.frame(camera, tick * PERIOD_NS); } }
        assert!(!recovery.due(Duration::from_secs(20), &sync));
        assert_eq!(recovery.attempts, 0);
        sync.missing();
        assert!(recovery.due(Duration::from_secs(20), &sync));
        assert_eq!(recovery.total, 4);
        assert_eq!(sync.state(), CaptureSync::OutOfSync);
    }

    #[test]
    fn one_gap_or_short_split_does_not_resync() {
        let mut sync = SyncMonitor::new(3_000_000);
        for tick in 0..60 {
            for camera in 0..6 {
                if tick == 10 && camera == 0 {
                    continue;
                }
                sync.frame(
                    camera,
                    tick * 33_333_333
                        + if camera == 5 && tick < 14 {
                            -5_300_000
                        } else {
                            0
                        },
                );
            }
        }
        assert_eq!(sync.state(), CaptureSync::InSync);
    }
}
