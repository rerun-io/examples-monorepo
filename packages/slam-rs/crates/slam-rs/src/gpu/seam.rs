//! Host-side counters for the per-frameset GPU seam.
//!
//! The device timestamps say every kernel of a two-camera MIO10 frameset is
//! 0.44 ms of GPU time while the frontend spends 1.63 ms of host time, so the
//! question these answer is where the rest goes: how many launches, uploads and
//! synchronising reads a frameset makes, and how long the host sits in each.
//! The answer that shaped D77 was **five reads, 1.51 ms**, against 0.15 ms in
//! every upload and nothing measurable in the launches; D78 took the count to
//! two by letting one stage's download carry another's buffers, so the reads
//! here are counted by whoever *issued* them, not by whose data they hold.
//!
//! Meters observe only the calling producer thread. Snapshots do not reset them;
//! the isolated timing test formats their delta. Queue control lives in submission.

use std::cell::Cell;
use std::thread::LocalKey;
use std::time::Instant;

/// Calls and total host nanoseconds for one operation on one producer thread.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Measurement {
    /// Number of operations.
    pub calls: u64,
    /// Total host duration in nanoseconds.
    pub nanos: u64,
}

impl Measurement {
    fn delta(self, start: Self) -> Self {
        Self {
            calls: self.calls - start.calls,
            nanos: self.nanos - start.nanos,
        }
    }
}

/// Observational meter; its storage belongs to the calling producer thread.
#[derive(Debug)]
pub struct Meter {
    value: &'static LocalKey<Cell<Measurement>>,
}

impl Meter {
    /// Run an operation and observe its duration without controlling submission.
    pub fn measure<T>(&self, body: impl FnOnce() -> T) -> T {
        let mark = Instant::now();
        let out = body();
        let nanos = mark.elapsed().as_nanos() as u64;
        self.value.with(|value| {
            let old = value.get();
            value.set(Measurement {
                calls: old.calls + 1,
                nanos: old.nanos + nanos,
            });
        });
        out
    }

    /// Count an operation without timing it.
    pub fn count(&self) {
        self.value.with(|value| {
            let old = value.get();
            value.set(Measurement {
                calls: old.calls + 1,
                ..old
            });
        });
    }

    fn snapshot(&self) -> Measurement {
        self.value.with(Cell::get)
    }
}

thread_local! {
    static LAUNCH_VALUE: Cell<Measurement> = const { Cell::new(Measurement { calls: 0, nanos: 0 }) };
    static UPLOAD_VALUE: Cell<Measurement> = const { Cell::new(Measurement { calls: 0, nanos: 0 }) };
    static READ_TRACK_VALUE: Cell<Measurement> = const { Cell::new(Measurement { calls: 0, nanos: 0 }) };
    static READ_DETECT_VALUE: Cell<Measurement> = const { Cell::new(Measurement { calls: 0, nanos: 0 }) };
}

/// Kernel launches. Counted only: the enqueue is inside its stage's own timer
/// and measured under a millisecond for all thirty-two of a frameset together.
pub static LAUNCH: Meter = Meter {
    value: &LAUNCH_VALUE,
};
/// `create_from_slice`: a logical allocation and a host-to-device write.
pub static UPLOAD: Meter = Meter {
    value: &UPLOAD_VALUE,
};
/// The tracker batch's one download, which since D78 also carries whatever the
/// corner scanner staged on the [`super::ReadRelay`] — so on the device lane
/// this is where a frameset's cell keys are counted too.
pub static READ_TRACK: Meter = Meter {
    value: &READ_TRACK_VALUE,
};
/// A download the corner scanner made itself: the band path's candidate image,
/// and the cell keys of a frameset no tracker read carried — the first frameset
/// of a run, and a scanner with no relay wired.
pub static READ_DETECT: Meter = Meter {
    value: &READ_DETECT_VALUE,
};

/// All seam observations on the current producer thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    /// Kernel launches.
    pub launch: Measurement,
    /// Host-to-device uploads.
    pub upload: Measurement,
    /// Downloads issued by the tracker.
    pub read_track: Measurement,
    /// Downloads issued by the detector.
    pub read_detect: Measurement,
}

impl Snapshot {
    /// Subtract an earlier snapshot from the same producer thread.
    #[must_use]
    pub fn delta(self, start: Self) -> Self {
        Self {
            launch: self.launch.delta(start.launch),
            upload: self.upload.delta(start.upload),
            read_track: self.read_track.delta(start.read_track),
            read_detect: self.read_detect.delta(start.read_detect),
        }
    }
}

/// Observe this producer thread without changing any counters or reservations.
#[must_use]
pub fn snapshot() -> Snapshot {
    Snapshot {
        launch: LAUNCH.snapshot(),
        upload: UPLOAD.snapshot(),
        read_track: READ_TRACK.snapshot(),
        read_detect: READ_DETECT.snapshot(),
    }
}

thread_local! {
    static PEAK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn queue_reserved(tasks: usize) {
    PEAK.with(|peak| peak.set(peak.get().max(tasks)));
}

/// Largest reserved task count on any device on this producer thread.
pub fn queue_peak() -> usize {
    PEAK.with(|peak| peak.get())
}

/// Start a new queue measurement; does not change outstanding reservations.
pub fn reset_queue_peak() {
    PEAK.with(|peak| peak.set(0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_deltas_are_non_destructive_and_thread_local() {
        let start = snapshot();
        LAUNCH.count();
        UPLOAD.measure(|| ());
        std::thread::spawn(|| {
            LAUNCH.count();
            READ_TRACK.count();
        })
        .join()
        .unwrap_or_else(|_| panic!("meter producer panicked"));
        let end = snapshot();
        assert_eq!(snapshot(), end);
        let delta = end.delta(start);
        assert_eq!(delta.launch.calls, 1);
        assert_eq!(delta.upload.calls, 1);
        assert_eq!(delta.read_track.calls, 0);
        assert_eq!(delta.read_detect.calls, 0);
        LAUNCH.count();
        assert_eq!(snapshot().delta(start).launch.calls, 2);
    }
}
