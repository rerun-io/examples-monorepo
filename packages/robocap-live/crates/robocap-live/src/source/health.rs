//! Shared health lets the scheduler report a warning even while capture drains.
use super::CaptureSync;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Live capture counters, flattened into the final summary.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct CaptureSnapshot {
    /// Current six-camera timestamp health.
    pub capture_sync: CaptureSync,
    /// Total trigger restart attempts; each outage allows three attempts.
    pub capture_resyncs: u32,
    /// Camera frames discarded during recovery, across driver and source queues.
    pub capture_discarded_frames: u64,
    /// Wall time spent out of sync or waiting for confirmation after recovery.
    pub capture_out_of_sync_s: f64,
    /// Complete six-camera framesets per elapsed capture second, including outages.
    pub complete_frameset_hz: f64,
}

#[derive(Default)]
struct Health {
    snapshot: CaptureSnapshot,
    since: Option<Instant>,
}

impl Health {
    fn close_outage(&mut self) {
        if let Some(since) = self.since.take() {
            self.snapshot.capture_out_of_sync_s += since.elapsed().as_secs_f64();
        }
    }
}

/// Shared live health. This does not own devices or camera images.
#[derive(Clone, Default)]
pub struct CaptureHealth(Arc<Mutex<Health>>);

impl CaptureHealth {
    /// Read cumulative counters, including the duration of a current outage.
    pub fn snapshot(&self) -> CaptureSnapshot {
        let health = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let mut snapshot = health.snapshot;
        snapshot.capture_out_of_sync_s += health.since.map_or(0.0, |t| t.elapsed().as_secs_f64());
        snapshot
    }

    pub(super) fn discarded(&self, count: u64) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).snapshot.capture_discarded_frames += count;
    }

    /// Freeze outage timing once capture has stopped, before consumers drain.
    pub(super) fn finish(&self) {
        let mut health = self.0.lock().unwrap_or_else(|e| e.into_inner());
        health.close_outage();
    }

    pub(super) fn update(&self, state: CaptureSync, resyncs: u32, complete_hz: f64) {
        let mut health = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if state == CaptureSync::InSync {
            health.close_outage();
        } else {
            health.since.get_or_insert_with(Instant::now);
        }
        health.snapshot.capture_sync = state;
        health.snapshot.capture_resyncs = resyncs;
        health.snapshot.complete_frameset_hz = complete_hz;
    }
}
