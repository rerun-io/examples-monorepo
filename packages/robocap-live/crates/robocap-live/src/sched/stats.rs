//! Per-stage timings and counters shared by the stages, and their summaries.

use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;

use super::queue::lock;

/// Pipeline stages with timings; the discriminant indexes [`STAGES`] and the per-stage timing arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Stage {
    /// Frameset arrivals (count only).
    Source,
    /// Small images.
    Downsample,
    /// `Vio::track`.
    Slam,
    /// Hands waiting for a pose.
    PoseWait,
    /// `HandTracking::step`.
    Hands,
    /// Output sinks.
    Output,
    /// Source emission -> output done.
    EndToEnd,
    /// Of `Slam`: the frontend.
    SlamFrontend,
    /// Of `Slam`: the LM optimisation.
    SlamOptimize,
    /// Of `Slam`: the marginalisation.
    SlamMarginalize,
    /// `Slam` on keyframe framesets only.
    SlamKeyframe,
    /// Of `SlamFrontend`: pyramids.
    SlamPyramid,
    /// Of `SlamFrontend`: FAST detection.
    SlamDetect,
    /// Of `SlamFrontend`: temporal KLT.
    SlamTrack,
    /// Of `SlamFrontend`: stereo matching.
    SlamStereo,
    /// Final pending estimator work, excluded from the track-call rate.
    SlamFlush,
}

/// Every stage with its name in the run summary (`stages`, `stage_fps`), in declaration order.
pub(super) const STAGES: [(Stage, &str); 16] = [
    (Stage::Source, "source"),
    (Stage::Downsample, "downsample"),
    (Stage::Slam, "slam"),
    (Stage::PoseWait, "pose_wait"),
    (Stage::Hands, "hands"),
    (Stage::Output, "output"),
    (Stage::EndToEnd, "end_to_end"),
    (Stage::SlamFrontend, "slam_frontend"),
    (Stage::SlamOptimize, "slam_optimize"),
    (Stage::SlamMarginalize, "slam_marginalize"),
    (Stage::SlamKeyframe, "slam_keyframe"),
    (Stage::SlamPyramid, "slam_pyramid"),
    (Stage::SlamDetect, "slam_detect"),
    (Stage::SlamTrack, "slam_track"),
    (Stage::SlamStereo, "slam_stereo"),
    (Stage::SlamFlush, "slam_flush"),
];

// A stage's row in STAGES is its discriminant (checked at compile time).
const _: () = {
    let mut row = 0;
    while row < STAGES.len() {
        assert!(STAGES[row].0 as usize == row, "STAGES lists the stages in declaration order");
        row += 1;
    }
};

/// Mean, p50, p95 and max of a set of milliseconds.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Summary {
    /// Samples.
    pub count: usize,
    /// Mean, ms.
    pub mean: f64,
    /// Median, ms.
    pub p50: f64,
    /// 95th percentile, ms.
    pub p95: f64,
    /// Largest, ms.
    pub max: f64,
}

impl Summary {
    /// Summarise `values` (empty -> zeros).
    pub fn of(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
        Self { count: values.len(), mean: values.iter().sum::<f64>() / values.len() as f64, p50: at(0.5), p95: at(0.95), max: at(1.0) }
    }
}

/// The run's event counters (they go into [`super::RunSummary`] as they are).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Counters {
    /// IMU samples from the source.
    pub imu: u64,
    /// SLAM steps with a visually supported pose.
    pub slam_ok: u64,
    /// SLAM step status counts.
    pub slam_status: std::collections::BTreeMap<&'static str, u64>,
    /// Framesets the rate cap skipped.
    pub slam_rate_skipped: u64,
    /// Framesets without all four SLAM cameras.
    pub slam_missing_cameras: u64,
    /// Oldest live IMU samples evicted during a prolonged outage; replay never evicts.
    #[serde(skip_serializing_if = "is_zero")]
    pub slam_imu_dropped: u64,
    /// Framesets the IMU did not cover in time.
    pub slam_imu_timeouts: u64,
    /// IMU samples SLAM dropped because they did not follow the previous one.
    pub slam_imu_unordered: u64,
    /// Track calls that supplied a queued next-frame hint (the GPU may discard a stale hint).
    pub slam_lookahead: u64,
    /// Accepted first frames buffered without a pose, including after a reset or flush.
    pub slam_buffered: u64,
    /// slam-rs errors (each one resets).
    pub slam_failures: u64,
    /// Estimator restarts (gaps and failures).
    pub slam_resets: u64,
    /// Hands-stage errors.
    pub hands_errors: u64,
    /// Network (NPU) failures inside hands steps; each drops the networks and rebuilds them.
    pub nets_failures: u64,
    /// Successful network rebuilds.
    pub nets_recreated: u64,
    /// Framesets whose hands were skipped while no networks were available.
    pub hands_without_nets: u64,
    /// Framesets the busy hands stage could not take, sent to the output without hands (realtime).
    pub hands_bypassed: u64,
    /// Output items older than one already written (a stalled hands step finishing late), dropped.
    pub output_late: u64,
    /// Output sink errors.
    pub output_errors: u64,
}

#[derive(Default)]
pub(super) struct StatsInner {
    pub(super) slam_lane: Option<crate::slam::SlamLane>,
    pub(super) slam_frontend_lag: Option<bool>,
    pub(super) slam_threads: Option<usize>,
    pub(super) window: [Vec<f64>; STAGES.len()],
    pub(super) total: [Vec<f64>; STAGES.len()],
    pub(super) counters: Counters,
    pub(super) imu_window: u64,
    pub(super) slam_ok_window: u64,
    pub(super) first_frameset: Option<Instant>,
    pub(super) last_frameset: Option<Instant>,
}

/// Counters and timings shared by the stages.
#[derive(Default)]
pub(crate) struct Stats {
    inner: Mutex<StatsInner>,
}

impl Stats {
    /// Record one frameset through `stage`, taking `ms`.
    pub fn record(&self, stage: Stage, ms: f64) {
        let mut inner = lock(&self.inner);
        let index = stage as usize;
        inner.window[index].push(ms);
        inner.total[index].push(ms);
        if stage == Stage::Source {
            let now = Instant::now();
            inner.first_frameset.get_or_insert(now);
            inner.last_frameset = Some(now);
        }
    }

    pub(super) fn with<R>(&self, f: impl FnOnce(&mut StatsInner) -> R) -> R {
        f(&mut lock(&self.inner))
    }
}

fn is_zero(value: &u64) -> bool { *value == 0 }
