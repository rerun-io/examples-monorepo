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
//! Relaxed atomics and one `Instant` pair per upload or read: about 0.6 µs a
//! frameset against the 1600 it measures. `tests/gpu_seam_bench.rs` is the rig
//! that prints them.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// How many times an operation ran, and the host time inside it.
#[derive(Debug)]
pub struct Meter {
    calls: AtomicU64,
    nanos: AtomicU64,
}

impl Meter {
    /// A meter at zero.
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        }
    }

    /// Run `body`, adding one call and the host time it took.
    pub fn measure<T>(&self, body: impl FnOnce() -> T) -> T {
        let mark: Instant = Instant::now();
        let out: T = body();
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.nanos
            .fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    }

    /// Count one call whose host time is not worth an `Instant` pair.
    pub fn count(&self) {
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Calls and their total host nanoseconds.
    #[must_use]
    pub fn read(&self) -> (u64, u64) {
        (
            self.calls.load(Ordering::Relaxed),
            self.nanos.load(Ordering::Relaxed),
        )
    }

    /// Back to zero, so a caller can bracket a measured run.
    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.nanos.store(0, Ordering::Relaxed);
    }
}

/// Kernel launches. Counted only: the enqueue is inside its stage's own timer
/// and measured under a millisecond for all thirty-two of a frameset together.
pub static LAUNCH: Meter = Meter::new();
/// `create_from_slice`: a logical allocation and a host-to-device write.
pub static UPLOAD: Meter = Meter::new();
/// The tracker batch's one download, which since D78 also carries whatever the
/// corner scanner staged on the [`super::ReadRelay`] — so on the device lane
/// this is where a frameset's cell keys are counted too.
pub static READ_TRACK: Meter = Meter::new();
/// A download the corner scanner made itself: the band path's candidate image,
/// and the cell keys of a frameset no tracker read carried — the first frameset
/// of a run, and a scanner with no relay wired.
pub static READ_DETECT: Meter = Meter::new();

/// Count one launch.
pub fn launch<R: cubecl::prelude::Runtime>(client: &cubecl::prelude::ComputeClient<R>) {
    super::reserve(client, 1);
    LAUNCH.count();
}

/// Every counter back to zero.
pub fn reset() {
    for meter in [&LAUNCH, &UPLOAD, &READ_TRACK, &READ_DETECT] {
        meter.reset();
    }
}

/// The whole seam over `framesets`, per frameset, on one line.
#[must_use]
pub fn line(framesets: u64) -> String {
    let scale: f64 = framesets.max(1) as f64;
    let (launches, _) = LAUNCH.read();
    let (uploads, upload_ns) = UPLOAD.read();
    let (track_reads, track_ns) = READ_TRACK.read();
    let (detect_reads, detect_ns) = READ_DETECT.read();
    format!(
        "per frameset: {:.2} launches, {:.2} uploads ({:.3} ms), {:.2} reads ({:.3} ms) \
         = {:.2} tracker ({:.3} ms) + {:.2} detector ({:.3} ms)",
        launches as f64 / scale,
        uploads as f64 / scale,
        upload_ns as f64 / scale / 1e6,
        (track_reads + detect_reads) as f64 / scale,
        (track_ns + detect_ns) as f64 / scale / 1e6,
        track_reads as f64 / scale,
        track_ns as f64 / scale / 1e6,
        detect_reads as f64 / scale,
        detect_ns as f64 / scale / 1e6,
    )
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

/// Reserve and measure the one task that uploads a host slice.
pub(super) fn upload<R: cubecl::prelude::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    bytes: &[u8],
) -> cubecl::server::Handle {
    super::reserve(client, 1);
    UPLOAD.measure(|| client.create_from_slice(bytes))
}
