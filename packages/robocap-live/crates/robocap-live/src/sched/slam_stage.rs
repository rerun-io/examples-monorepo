//! The SLAM stage: rate selection, IMU coverage, estimator calls and pose publication.
//!
//! - **Lane.** `--slam-lane gpu|cpu` (GPU on aarch64, CPU elsewhere; a GPU that fails to start falls back to CPU with a
//!   warning). GPU defaults to one worker and `port.frontend_lag=true`, CPU to two workers and no lag; `--slam-set
//!   port.frontend_lag=..` overrides either, also after a fallback. The startup line and the summary's `slam_lane`,
//!   `slam_frontend_lag` and `slam_threads` report what ran (null when SLAM is off or replays reference poses).
//! - **Lag.** With lag the first accepted frameset buffers without a pose and each later call returns the previous one's. A
//!   result is matched to its pending input by time and keeps that input's index through rate skips and drops. A queued next
//!   frameset is an optional GPU lookahead hint; the stage never waits for one. EOF and a normal stop flush the last result
//!   before the pose store closes.
//! - **Worlds.** A new world starts only on backwards input time (checked before rate selection), a gap of more than
//!   [`crate::slam::RESET_GAP_NS`] between accepted inputs, or an estimator error. A boundary first flushes the old world's
//!   pending pose, and each reset prints one stderr line with its count and cause. Pose progress uses monotonic source
//!   indices, so rewound times never reuse an old world's poses.
//! - **IMU.** A frame is tracked once a sample strictly after its time has arrived. Live mode waits up to the IMU timeout,
//!   lossless mode for samples or EOF; an uncovered frame is skipped and counted in `slam_imu_timeouts` (no reset). After
//!   backwards time the old world's queued IMU tail is dropped.
//! - **Live and lossless.** Live: progress advances when SLAM consumes or skips an input, and hands take the newest usable
//!   pose at or before their time within `--hands-wait-ms`, never waiting a frameset for a lagged pose. Lossless: progress
//!   waits for each accepted input's pose (published or flushed), so hands and output use that frame's own estimate; a skip
//!   (rate, input, or a missing reference) holds the last pose and first flushes any pending estimate, so the bounded queues
//!   never stall.
//! - **Timings.** A pose's index, time, support counts and keyframe flag describe the returned estimate; `compute_ms` describes
//!   the call that produced it, so with lag the frontend timers belong to the next input. Estimator and frontend overlap: do
//!   not add their times. The summary's `slam` counts track calls (the buffered first one too), `slam_flush` the drain work,
//!   and `slam_buffered` / `slam_lookahead` the buffering and the hints.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::downsample::SmallImages;
use crate::frame::SLAM_CAMERAS;
use crate::slam::{RateSelector, SlamConfig, SlamEstimator, SlamLane, SlamPose, SlamStatus};
use kornia_staging_sensors::imu::CombinedImuSample;

use super::{SchedError, Shared, Stage, stage_error};

/// The SLAM stage (A76): rate selection, IMU coverage, `Vio::track`, and the poses published for hands and output.
pub(super) fn slam_loop(
    shared: &Shared,
    imu: &super::queue::StageQueue<CombinedImuSample>,
    config: SlamConfig,
    lossless: bool,
    imu_wait: Duration,
) -> Result<(), SchedError> {
    let mut slam = SlamEstimator::new(&config).map_err(|e| stage_error("slam", e))?;
    eprintln!(
        "robocap-live: SLAM lane {:?}, frontend_lag={}, threads={}",
        slam.lane(),
        slam.frontend_lag(),
        slam.threads()
    );
    shared.stats.with(|s| {
        s.slam_lane = Some(slam.lane());
        s.slam_frontend_lag = Some(slam.frontend_lag());
        s.slam_threads = Some(slam.threads());
    });
    let mut selector = RateSelector::new(config.hz, config.rate_tolerance_ns);
    let mut last_t: Option<i64> = None;
    let mut imu_rewind: Option<i64> = None;
    let mut imu_open = true;
    let mut dropped_imu = 0;
    let push = |slam: &mut SlamEstimator, sample: CombinedImuSample| {
        slam.push_imu(&sample).map_err(|e| stage_error("slam", e))
    };
    let (stats, poses) = (&shared.stats, &shared.poses);
    while let Some(item) = shared.slam.pop() {
        let (index, t) = (item.frameset.index, item.frameset.timestamp_ns);
        let backwards = last_t.is_some_and(|last| t < last);
        if !backwards && !selector.due(t) {
            stats.with(|s| s.counters.slam_rate_skipped += 1);
            skip_frame(shared, &mut slam, lossless, index)?;
            continue;
        }
        if let Some(last) = last_t.filter(|&last| backwards || t - last > crate::slam::RESET_GAP_NS)
        {
            flush_slam(shared, &mut slam)?;
            if backwards {
                restart_slam(
                    shared,
                    &mut slam,
                    index,
                    format_args!("backwards timestamp: {last} -> {t}"),
                )?;
                // Discard the previous world's queued IMU tail until timestamps return to the earlier range.
                // The new world's first sample may already be just after this frame, which supplies coverage.
                imu_rewind = Some(last);
            } else {
                restart_slam(
                    shared,
                    &mut slam,
                    index,
                    format_args!("input gap {:.3} ms", (t - last) as f64 / 1e6),
                )?;
            }
            last_t = None;
            selector = RateSelector::new(config.hz, config.rate_tolerance_ns);
        }
        let Some(images) = slam_images(&item.small) else {
            stats.with(|s| s.counters.slam_missing_cameras += 1);
            skip_frame(shared, &mut slam, lossless, index)?;
            continue;
        };
        let dropped = imu.dropped();
        if dropped != dropped_imu {
            flush_slam(shared, &mut slam)?;
            restart_slam(shared, &mut slam, index, format_args!("IMU retention overflow: {} samples", dropped - dropped_imu))?;
            dropped_imu = dropped;
        }
        let deadline = Instant::now() + imu_wait;
        // Read only through coverage. Draining the channel ahead of time would lose future IMU on a reset.
        while !slam.imu_covers(t) && imu_open {
            let next = if lossless {
                imu.pop().ok_or(mpsc::RecvTimeoutError::Disconnected)
            } else {
                imu.recv_timeout(deadline.saturating_duration_since(Instant::now()))
            };
            match next {
                Ok(sample) => {
                    if imu_rewind.is_some_and(|previous_t| sample.timestamp_ns > previous_t) {
                        continue;
                    }
                    imu_rewind = None;
                    push(&mut slam, sample)?;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => imu_open = false,
            }
        }
        if !slam.imu_covers(t) {
            stats.with(|s| s.counters.slam_imu_timeouts += 1);
            skip_frame(shared, &mut slam, lossless, index)?;
            continue;
        }
        selector.selected(t);
        last_t = Some(t);
        // Snapshot only an already queued, selected, complete frameset. No wait and no IMU requirement for a hint.
        let next = if slam.lane() == SlamLane::Gpu {
            shared.slam.peek_matching(|next| {
                let next_t = next.frameset.timestamp_ns;
                selector.due(next_t) && next_t > t && next_t - t <= crate::slam::RESET_GAP_NS
            })
        } else {
            None
        };
        let lookahead = next
            .as_ref()
            .and_then(|next| Some((next.frameset.timestamp_ns, slam_images(&next.small)?)));
        if lookahead.is_some() {
            stats.with(|s| s.counters.slam_lookahead += 1);
        }
        let pose = match slam.track_with_lookahead(index, t, images, lookahead) {
            Ok(pose) => pose,
            Err(error) => {
                stats.with(|s| s.counters.slam_failures += 1);
                restart_slam(
                    shared,
                    &mut slam,
                    index,
                    format_args!("estimator error at frameset {index}: {error}"),
                )?;
                Some(SlamPose {
                    resets: slam.resets,
                    ..SlamPose::untracked(index, t, SlamStatus::Failed)
                })
            }
        };
        let (compute_ms, stages) = slam.last_call();
        stats.record(Stage::Slam, compute_ms);
        if compute_ms > 0.0 {
            stats.record(Stage::SlamFrontend, stages.frontend_ms);
            stats.record(Stage::SlamOptimize, stages.optimize_ms);
            stats.record(Stage::SlamMarginalize, stages.marginalize_ms);
            stats.record(Stage::SlamPyramid, stages.pyramid_ms);
            stats.record(Stage::SlamDetect, stages.detect_ms);
            stats.record(Stage::SlamTrack, stages.track_ms);
            stats.record(Stage::SlamStereo, stages.stereo_ms);
            if stages.keyframe {
                stats.record(Stage::SlamKeyframe, compute_ms);
            }
        }
        stats.with(|s| {
            s.counters.slam_imu_unordered = slam.imu_unordered;
            s.counters.slam_buffered = slam.buffered;
        });
        if let Some(pose) = pose {
            publish_slam(shared, pose);
        }
        poses.consumed(index, slam.pending_index());
    }
    flush_slam(shared, &mut slam)?;
    Ok(())
}

fn slam_images(small: &SmallImages) -> Option<[&kornia_image::Image<u8, 1>; 4]> {
    let [left_front, right_front, left, right] = SLAM_CAMERAS.map(|c| small[c].as_deref());
    Some([left_front?, right_front?, left?, right?])
}

fn publish_slam(shared: &Shared, pose: SlamPose) {
    shared.stats.with(|s| {
        *s.counters
            .slam_status
            .entry(pose.status.as_str())
            .or_default() += 1;
        if pose.ok {
            s.counters.slam_ok += 1;
            s.slam_ok_window += 1;
        }
    });
    shared.poses.publish(pose);
}

fn restart_slam(
    shared: &Shared,
    slam: &mut SlamEstimator,
    fallback_index: u64,
    cause: std::fmt::Arguments<'_>,
) -> Result<(), SchedError> {
    let reset_at = slam
        .pending_index()
        .map_or(fallback_index, |pending| pending.min(fallback_index));
    slam.reset().map_err(|e| stage_error("slam", e))?;
    shared.poses.reset(reset_at);
    shared.stats.with(|s| s.counters.slam_resets += 1);
    eprintln!(
        "robocap-live: SLAM reset {}: {}",
        slam.resets,
        cause.to_string().replace(['\r', '\n'], " ")
    );
    Ok(())
}

fn skip_frame(
    shared: &Shared,
    slam: &mut SlamEstimator,
    lossless: bool,
    index: u64,
) -> Result<(), SchedError> {
    if lossless {
        flush_slam(shared, slam)?;
    }
    shared.poses.progress(index);
    Ok(())
}

/// Before a lossless skip, drain the older accepted frame: otherwise bounded downstream queues can prevent the next
/// selected frame from ever reaching SLAM. Also used on EOF/stop, before closing the pose store.
fn flush_slam(shared: &Shared, slam: &mut SlamEstimator) -> Result<(), SchedError> {
    if let Some(pending_index) = slam.pending_index() {
        match slam.flush() {
            Ok(Some(pose)) => {
                shared.stats.record(Stage::SlamFlush, slam.last_call().0);
                publish_slam(shared, pose);
            }
            Ok(None) => {}
            Err(error) => {
                shared.stats.with(|s| s.counters.slam_failures += 1);
                restart_slam(
                    shared,
                    slam,
                    pending_index,
                    format_args!("estimator error during flush: {error}"),
                )?;
            }
        }
    }
    Ok(())
}
