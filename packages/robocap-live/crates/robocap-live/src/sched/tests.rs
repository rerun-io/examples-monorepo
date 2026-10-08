//! The pipeline's tests: queue policies end to end, missing cameras, network failures, stage start failures, the record.

use super::*;
use std::sync::mpsc;
use crate::hands::HandOutput;
use crate::nets::{DetNetRaw, KeyNetRaw, NetFrame, NetsError};
use crate::source::replay::{ReplayConfig, ReplaySource, read_reference_poses};

/// Observe the public output seam: frameset identity, assigned pose identity and whether it was flushed.
#[derive(Clone, Copy, Debug)]
struct PoseRow {
    index: u64,
    t_ns: i64,
    pose: Option<SlamPose>,
}

struct PoseSink {
    rows: Arc<Mutex<Vec<PoseRow>>>,
    stop_at: Option<(u64, Arc<AtomicBool>)>,
}

impl FramesetSink for PoseSink {
    fn frameset(&mut self, record: &OutputRecord<'_>) -> Result<(), SinkError> {
        lock(&self.rows).push(PoseRow { index: record.frameset.index, t_ns: record.frameset.timestamp_ns, pose: record.pose.copied() });
        if let Some((index, stop)) = &self.stop_at
            && record.frameset.index >= *index {
                stop.store(true, Ordering::Relaxed);
            }
        Ok(())
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

#[test]
fn lossless_lag_publishes_each_frames_own_pose_including_eof_and_stop()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-lag-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 18)?;
    // Paced replay with blocking queues proves that the stop path drains the final accepted frameset too.
    for (lag, stop_early) in [(false, false), (true, false), (true, true)] {
        let stop = Arc::new(AtomicBool::new(false));
        let source = ReplaySource::open(
            &dir,
            ReplayConfig {
                realtime: stop_early,
                ..Default::default()
            },
            stop.clone(),
        )?;
        let (summary, rows) = lossless_slam_run(source, lag, 30.0, stop, stop_early.then_some(5))?;
        assert!(rows.len() >= 6);
        if !stop_early {
            assert_eq!(rows.len(), 18);
        }
        for row in rows.iter() {
            assert_eq!(
                row.pose.map(|p| (p.index, p.t_ns)),
                Some((row.index, row.t_ns)),
                "lag={lag}, stop={stop_early}: {row:?}"
            );
        }
        assert_eq!(summary.counters.slam_failures, 0);
        assert_eq!(summary.slam_lane, Some(SlamLane::Cpu));
        assert_eq!(summary.slam_frontend_lag, Some(lag));
        assert_eq!(summary.slam_threads, Some(2));
        assert_eq!(
            summary.counters.slam_lookahead, 0,
            "CPU does not consume lookahead hints"
        );
        assert_eq!(summary.counters.slam_buffered, u64::from(lag));
        assert_eq!(
            summary.counters.slam_status.values().sum::<u64>() as usize,
            summary.stages["slam"].count
        );
        assert_eq!(summary.stages["slam_flush"].count, usize::from(lag));
        if lag {
            assert_eq!(
                rows.last()
                    .and_then(|r| r.pose)
                    .map(|p| p.stages.frontend_ms),
                Some(0.0)
            );
        }
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn lossless_lag_with_sparse_selection_and_missing_cameras_does_not_stall()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!(
        "robocap-live-sched-lag-skips-{}",
        std::process::id()
    ));
    crate::source::replay::tests::write_test_dump_with(&dir, 18, &|index, camera| {
        index == 1 && camera == 4
    })?;
    for hz in [30.0, 1.0] {
        let stop = Arc::new(AtomicBool::new(false));
        let source = ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(lossless_slam_run(source, true, hz, stop, None));
        });
        let (summary, rows) = rx.recv_timeout(Duration::from_secs(10))??;
        assert_eq!(summary.framesets, 18);
        assert_eq!(rows.len(), 18);
        assert_eq!(rows[0].pose.map(|p| p.index), Some(0));
        assert_eq!(
            rows[17].pose.map(|p| p.index),
            Some(if hz == 30.0 { 17 } else { 0 })
        );
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

struct EditedReplay<F>(ReplaySource, F);

impl<F: FnMut(SourceEvent) -> Option<SourceEvent> + Send> FrameSource for EditedReplay<F> {
    fn rig(&self) -> &crate::frame::Rig {
        self.0.rig()
    }
    fn next_event(&mut self) -> Result<Option<SourceEvent>, SourceError> {
        while let Some(event) = self.0.next_event()? {
            if let Some(event) = (self.1)(event) {
                return Ok(Some(event));
            }
        }
        Ok(None)
    }
}

fn lossless_slam_run(
    source: impl FrameSource + 'static,
    lag: bool,
    hz: f64,
    stop: Arc<AtomicBool>,
    stop_at: Option<u64>,
) -> Result<(RunSummary, Vec<PoseRow>), SchedError> {
    let hands = HandsStage {
        tracker: Box::new(EchoTracker),
        nets: Box::new(NoNets),
        nets_factory: None,
    };
    let mut config = test_config(
        true,
        SlamMode::On,
        ReferencePoses::new(Vec::new(), 0, None),
        hands,
    );
    config.slam.hz = hz;
    config.slam.overrides.push(
        crate::slam::parse_override(&format!("port.frontend_lag={lag}")).expect("boolean override"),
    );
    let rows = Arc::new(Mutex::new(Vec::new()));
    let sink = PoseSink {
        rows: rows.clone(),
        stop_at: stop_at.map(|index| (index, stop.clone())),
    };
    let summary = run(Box::new(source), config, vec![Box::new(sink)], stop)?;
    let rows = lock(&rows).clone();
    Ok((summary, rows))
}

#[test]
fn time_jumps_keep_or_reset_the_world_and_flush_pending_poses()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!(
        "robocap-live-sched-time-jump-{}",
        std::process::id()
    ));
    crate::source::replay::tests::write_test_dump(&dir, 18)?;
    // The preceding frame is 33,333,333 ns before the first shifted frame: test exactly 3 s as well as either side.
    for (jump_ns, resets) in [
        (500_000_000, 0),
        (2_966_666_667, 0),
        (3_000_000_000, 1),
        (-1_000_000_000, 1),
    ] {
        for lag in [false, true] {
            let stop = Arc::new(AtomicBool::new(false));
            let source = EditedReplay(
                ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?,
                move |mut event| {
                    let t = match &mut event {
                        SourceEvent::Imu(sample) => &mut sample.timestamp_ns,
                        SourceEvent::Frameset(frame) => &mut frame.timestamp_ns,
                    };
                    if *t >= 1_299_999_997 {
                        *t += jump_ns;
                    }
                    Some(event)
                },
            );
            let (summary, rows) = lossless_slam_run(source, lag, 30.0, stop, None)?;
            assert_eq!(
                (
                    summary.counters.slam_resets,
                    summary.counters.slam_failures,
                    summary.counters.slam_imu_timeouts
                ),
                (resets, 0, 0)
            );
            assert_eq!(rows.len(), 18);
            for row in rows.iter() {
                assert_eq!(
                    row.pose.map(|p| (p.index, p.t_ns)),
                    Some((row.index, row.t_ns)),
                    "jump={jump_ns} lag={lag}: {row:?}"
                );
                assert_eq!(
                    row.pose.map(|p| p.resets),
                    Some(if row.index >= 9 { resets } else { 0 }),
                    "{row:?}"
                );
            }
        }
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn frame_gaps_up_to_three_seconds_with_continuous_imu_keep_the_world()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!(
        "robocap-live-sched-frame-gap-{}",
        std::process::id()
    ));
    crate::source::replay::tests::write_test_dump(&dir, 105)?;
    for next_index in [18, 33, 54, 98] {
        let stop = Arc::new(AtomicBool::new(false));
        let source = EditedReplay(
            ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?,
            move |event| {
                if matches!(&event, SourceEvent::Frameset(frame) if (9..next_index).contains(&frame.index))
                {
                    None
                } else {
                    Some(event)
                }
            },
        );
        let (summary, rows) = lossless_slam_run(source, true, 30.0, stop, None)?;
        assert_eq!(
            (
                summary.counters.slam_resets,
                summary.counters.slam_failures,
                summary.counters.slam_imu_timeouts
            ),
            (0, 0, 0)
        );
        assert_eq!(rows.len(), 105 - (next_index as usize - 9));
        for row in rows.iter() {
            assert_eq!(
                row.pose.map(|p| (p.index, p.t_ns, p.resets)),
                Some((row.index, row.t_ns, 0)),
                "{row:?}"
            );
        }
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn replay_loops_still_start_new_worlds() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-loop-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 18)?;
    let stop = Arc::new(AtomicBool::new(false));
    // Stop at the first IMU sample after frameset 23, not from the sink: a stop between a frameset and the sample that covers
    // it leaves that frameset without IMU, which the SLAM stage counts as a timeout.
    let (source_stop, mut last_t) = (stop.clone(), None::<i64>);
    let looping = ReplaySource::open(
        &dir,
        ReplayConfig {
            looping: true,
            ..Default::default()
        },
        stop.clone(),
    )?;
    let source = EditedReplay(looping, move |event| {
        match &event {
            SourceEvent::Frameset(frame) if frame.index == 23 => last_t = Some(frame.timestamp_ns),
            SourceEvent::Imu(sample) if last_t.is_some_and(|t| sample.timestamp_ns > t) => {
                source_stop.store(true, Ordering::Relaxed)
            }
            _ => {}
        }
        Some(event)
    });
    let (summary, rows) = lossless_slam_run(source, true, 30.0, stop, None)?;
    assert!(summary.counters.slam_resets >= 1);
    assert_eq!(rows.last().map(|row| row.index), Some(23));
    assert_eq!(
        (
            summary.counters.slam_failures,
            summary.counters.slam_imu_timeouts
        ),
        (0, 0)
    );
    for row in rows.iter() {
        assert_eq!(
            row.pose.map(|p| (p.index, p.t_ns, p.resets)),
            Some((row.index, row.t_ns, row.index / 18)),
            "{row:?}"
        );
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn absent_imu_coverage_skips_frames_without_resetting() -> Result<(), Box<dyn std::error::Error>> {
    let dir =
        std::env::temp_dir().join(format!("robocap-live-sched-no-imu-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 3)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = EditedReplay(
        ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?,
        |event| {
            if matches!(event, SourceEvent::Imu(_)) {
                None
            } else {
                Some(event)
            }
        },
    );
    let (summary, rows) = lossless_slam_run(source, false, 30.0, stop, None)?;
    assert_eq!(
        (
            summary.counters.slam_resets,
            summary.counters.slam_failures,
            summary.counters.slam_imu_timeouts
        ),
        (0, 0, 3)
    );
    assert_eq!(summary.stages["slam"].count, 0);
    assert!(rows.iter().all(|row| row.pose.is_none()));
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn repeated_estimator_errors_reset_each_failed_world() -> Result<(), Box<dyn std::error::Error>> {
    let dir =
        std::env::temp_dir().join(format!("robocap-live-sched-errors-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 18)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = EditedReplay(
        ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?,
        |mut event| {
            if let SourceEvent::Frameset(frame) = &mut event {
                frame.timestamp_ns = 1_000_000_000;
            }
            Some(event)
        },
    );
    let (summary, rows) = lossless_slam_run(source, true, 0.0, stop, None)?;
    assert_eq!(
        (
            summary.counters.slam_resets,
            summary.counters.slam_failures,
            summary.counters.slam_imu_timeouts
        ),
        (9, 9, 0)
    );
    assert_eq!(rows.len(), 18);
    for row in rows.iter().filter(|row| row.index % 2 == 1) {
        assert_eq!(
            row.pose.map(|p| (p.index, p.status, p.resets)),
            Some((row.index, SlamStatus::Failed, row.index.div_ceil(2)))
        );
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

/// A test pipeline: no pinning, no frequency hints, no per-second lines.
fn test_config(
    lossless: bool,
    slam_mode: SlamMode,
    reference: ReferencePoses,
    hands: HandsStage,
) -> PipelineConfig {
    PipelineConfig {
        lossless,
        slam_mode,
        slam: SlamConfig {
            lane: SlamLane::Cpu,
            frontend_threads: Some(2),
            ..Default::default()
        },
        reference: Some(reference),
        hands: Some(hands),
        hands_wait: Duration::from_millis(5),
        imu_wait: Duration::from_millis(20),
        ..PipelineConfig::default()
    }
}

struct CountingSink(Arc<Mutex<Vec<(u64, bool, bool)>>>);

impl FramesetSink for CountingSink {
    fn frameset(&mut self, record: &OutputRecord<'_>) -> Result<(), SinkError> {
        let small_ok = record
            .small
            .iter()
            .flatten()
            .all(|s| s.size() == crate::frame::SMALL_SIZE)
            && record
                .small
                .iter()
                .zip(record.frameset.cameras.iter())
                .all(|(small, full)| small.is_some() == full.is_some());
        lock(&self.0).push((
            record.frameset.index,
            record.pose.is_some_and(|p| p.ok),
            small_ok && record.hands.is_some(),
        ));
        Ok(())
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

struct NoNets;
impl HandNets for NoNets {
    fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        Ok(frames
            .iter()
            .map(|_| DetNetRaw {
                center: [[0.0; 2]; 2],
                radius: [0.0; 2],
                presence_logit: [-10.0; 2],
            })
            .collect())
    }
    fn keynet(&mut self, _: &[&[f32]], _: &[[f32; 63]]) -> Result<Vec<KeyNetRaw>, NetsError> {
        Ok(Vec::new())
    }
    fn describe(&self) -> String {
        "none".into()
    }
}

struct EchoTracker;
impl HandTracking for EchoTracker {
    fn step(
        &mut self,
        inputs: &HandInputs<'_>,
        world_from_rig: &Isometry3<f64>,
        _: &mut dyn HandNets,
    ) -> Result<HandFrameResult, HandsError> {
        let mut result = HandFrameResult {
            scale: 1.0,
            ..Default::default()
        };
        result.hands[0] = HandOutput {
            tracked: true,
            landmarks_world: Some([[world_from_rig.translation.x, inputs.index as f64, 0.0]; 21]),
            ..Default::default()
        };
        Ok(result)
    }
}

/// A backend whose DetNet fails on some calls (an RKNN run error) and stalls once (an NPU timeout).
struct FailingNets {
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl HandNets for FailingNets {
    fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call == 3 {
            thread::sleep(Duration::from_millis(500));
        }
        if call % 4 == 2 {
            return Err(NetsError::Run {
                net: "detnet",
                message: format!("fake rknn_run failure on call {call}"),
            });
        }
        Ok(frames
            .iter()
            .map(|_| DetNetRaw {
                center: [[0.0; 2]; 2],
                radius: [0.0; 2],
                presence_logit: [-10.0; 2],
            })
            .collect())
    }
    fn keynet(&mut self, _: &[&[f32]], _: &[[f32; 63]]) -> Result<Vec<KeyNetRaw>, NetsError> {
        Ok(Vec::new())
    }
    fn describe(&self) -> String {
        "failing".into()
    }
}

/// Runs DetNet on camera 0 every step and passes its errors on, as the real tracker does.
struct DetNetTracker;
impl HandTracking for DetNetTracker {
    fn step(
        &mut self,
        inputs: &HandInputs<'_>,
        _: &Isometry3<f64>,
        nets: &mut dyn HandNets,
    ) -> Result<HandFrameResult, HandsError> {
        let frame = vec![0u8; crate::nets::DETNET_WIDTH * crate::nets::DETNET_HEIGHT];
        if inputs.small[0].is_some() {
            nets.detnet(&[NetFrame {
                pixels: &frame,
                top: 0,
            }])?;
        }
        Ok(HandFrameResult {
            scale: 1.0,
            detnet_camera: Some(0),
            ..Default::default()
        })
    }
}

/// The NPU-failure policy: a failing or stalling network costs that frameset's hands only; the networks are rebuilt, the
/// output keeps receiving framesets (around the stall), and the run ends normally.
#[test]
fn network_failures_skip_hands_rebuild_the_networks_and_keep_the_output_going()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-npu-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump_with(&dir, 24, &|_, camera| camera != 0)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = ReplaySource::open(
        &dir,
        ReplayConfig {
            realtime: true,
            preload: true,
            ..ReplayConfig::default()
        },
        stop.clone(),
    )?;
    let reference = ReferencePoses::new(read_reference_poses(&dir)?, source.first_t_ns(), None);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let rebuilt = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let factory: NetsFactory = {
        let (calls, rebuilt) = (calls.clone(), rebuilt.clone());
        Box::new(move || {
            rebuilt.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FailingNets {
                calls: calls.clone(),
            }) as Box<dyn HandNets>)
        })
    };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hands = HandsStage {
        tracker: Box::new(DetNetTracker),
        nets: Box::new(FailingNets {
            calls: calls.clone(),
        }),
        nets_factory: Some(factory),
    };
    let config = PipelineConfig {
        hands_wait: Duration::ZERO,
        downsample_threads: 1,
        ..test_config(false, SlamMode::Reference, reference, hands)
    };
    let sinks: Vec<Box<dyn FramesetSink>> = vec![Box::new(CountingSink(seen.clone()))];
    let summary = run(Box::new(source), config, sinks, stop)?;
    assert!(summary.counters.nets_failures >= 1, "{summary:?}");
    assert!(
        summary.counters.nets_recreated >= 1
            && rebuilt.load(Ordering::SeqCst) as u64 == summary.counters.nets_recreated
    );
    assert!(
        summary.counters.hands_bypassed >= 1,
        "framesets went around the 500 ms stall: {summary:?}"
    );
    assert_eq!(
        summary.hands_dropped, summary.counters.hands_bypassed,
        "every frameset hands could not take went to the output"
    );
    let seen = lock(&seen);
    let indices: Vec<u64> = seen.iter().map(|(index, _, _)| *index).collect();
    assert!(
        indices.windows(2).all(|pair| pair[0] < pair[1]),
        "the output stays in order: {indices:?}"
    );
    assert!(
        seen.iter().all(|(_, pose_ok, _)| *pose_ok),
        "poses keep flowing to the output"
    );
    assert!(
        indices.len() >= 21,
        "the output keeps going through the failures and the stall: {indices:?}"
    );
    drop(seen);
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Framesets missing camera 0 (the last ones of s66-full) or other cameras must flow through every stage, in every SLAM mode,
/// and the run must end (a lossless s66-full replay hung there).
#[test]
fn framesets_missing_cameras_flow_through_and_the_run_ends()
-> Result<(), Box<dyn std::error::Error>> {
    let dir =
        std::env::temp_dir().join(format!("robocap-live-sched-missing-{}", std::process::id()));
    let missing = |index: u64, camera: usize| {
        (index >= 4 && camera == 0) || (index == 2 && [1, 4, 5].contains(&camera)) || index == 3
    };
    crate::source::replay::tests::write_test_dump_with(&dir, 7, &missing)?;
    for (slam_mode, realtime) in [
        (SlamMode::Reference, false),
        (SlamMode::On, false),
        (SlamMode::Off, false),
        (SlamMode::On, true),
    ] {
        let stop = Arc::new(AtomicBool::new(false));
        let source = ReplaySource::open(
            &dir,
            ReplayConfig {
                realtime,
                ..ReplayConfig::default()
            },
            stop.clone(),
        )?;
        let reference = ReferencePoses::new(read_reference_poses(&dir)?, source.first_t_ns(), None);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hands = HandsStage {
            tracker: Box::new(EchoTracker),
            nets: Box::new(NoNets),
            nets_factory: None,
        };
        let config = test_config(!realtime, slam_mode, reference, hands);
        let sinks: Vec<Box<dyn FramesetSink>> = vec![Box::new(CountingSink(seen.clone()))];
        let (done_tx, done_rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = done_tx.send(
                run(Box::new(source), config, sinks, stop)
                    .map(|summary| summary.framesets)
                    .map_err(|e| e.to_string()),
            );
        });
        let framesets = done_rx
            .recv_timeout(Duration::from_secs(60))
            .map_err(|_| format!("{slam_mode:?} realtime={realtime}: the run did not end"))??;
        assert_eq!(framesets, 7, "{slam_mode:?}");
        let indices: Vec<u64> = lock(&seen).iter().map(|(index, _, _)| *index).collect();
        assert_eq!(
            indices,
            (0..7).collect::<Vec<_>>(),
            "{slam_mode:?} realtime={realtime}: every frameset reaches the output"
        );
    }
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Panics on its third frameset.
struct PanickingTracker(u64);
impl HandTracking for PanickingTracker {
    fn step(
        &mut self,
        _: &HandInputs<'_>,
        _: &Isometry3<f64>,
        _: &mut dyn HandNets,
    ) -> Result<HandFrameResult, HandsError> {
        self.0 += 1;
        assert!(self.0 < 3, "test panic in the hands stage");
        Ok(HandFrameResult::default())
    }
}

/// A stage that panics stops the run with its error; the lossless queues it no longer drains must not leave the source and the
/// downsample stage blocked.
#[test]
fn a_stage_that_panics_ends_the_run_with_its_error() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-panic-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 24)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?;
    let reference = ReferencePoses::new(read_reference_poses(&dir)?, source.first_t_ns(), None);
    let hands = HandsStage {
        tracker: Box::new(PanickingTracker(0)),
        nets: Box::new(NoNets),
        nets_factory: None,
    };
    let config = test_config(true, SlamMode::Reference, reference, hands);
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = done_tx.send(
            run(Box::new(source), config, Vec::new(), stop)
                .map(|_| ())
                .map_err(|e| e.to_string()),
        );
    });
    let result = done_rx
        .recv_timeout(Duration::from_secs(30))
        .map_err(|_| "the run did not end")?;
    assert!(
        result
            .as_ref()
            .is_err_and(|e| e.contains("rl-hands") && e.contains("test panic")),
        "{result:?}"
    );
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A stage that fails before its loop (here: SLAM cannot pin to a core that does not exist) ends the run with its error; the
/// lossless queue it never drains must not leave the other stages blocked.
#[test]
fn a_stage_that_cannot_start_ends_the_run_with_its_error() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-pin-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 12)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?;
    let reference = ReferencePoses::new(read_reference_poses(&dir)?, source.first_t_ns(), None);
    let hands = HandsStage {
        tracker: Box::new(EchoTracker),
        nets: Box::new(NoNets),
        nets_factory: None,
    };
    let config = PipelineConfig {
        cpus_big: Some(vec![100_000]),
        ..test_config(true, SlamMode::On, reference, hands)
    };
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = done_tx.send(
            run(Box::new(source), config, Vec::new(), stop)
                .map(|_| ())
                .map_err(|e| e.to_string()),
        );
    });
    let result = done_rx
        .recv_timeout(Duration::from_secs(30))
        .map_err(|_| "the run did not end")?;
    assert!(
        result.as_ref().is_err_and(|e| e.contains("affinity")),
        "{result:?}"
    );
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn a_lossless_replay_with_reference_poses_reaches_every_stage_and_the_record()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 5)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?;
    let reference = ReferencePoses::new(read_reference_poses(&dir)?, source.first_t_ns(), None);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let record_path = dir.join("record.jsonl");
    let hands = HandsStage {
        tracker: Box::new(EchoTracker),
        nets: Box::new(NoNets),
        nets_factory: None,
    };
    let config = PipelineConfig {
        hands_wait: Duration::from_millis(10),
        imu_wait: Duration::from_millis(50),
        ..test_config(true, SlamMode::Reference, reference, hands)
    };
    let sinks: Vec<Box<dyn FramesetSink>> = vec![
        Box::new(CountingSink(seen.clone())),
        Box::new(RecordWriter::create(&record_path)?),
    ];
    let summary = run(Box::new(source), config, sinks, stop)?;
    assert_eq!(summary.framesets, 5);
    assert_eq!(
        *lock(&seen),
        (0..5).map(|i| (i, true, true)).collect::<Vec<_>>()
    );
    let lines: Vec<serde_json::Value> = std::fs::read_to_string(&record_path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    assert_eq!(lines.len(), 5);
    assert_eq!(
        lines[3]["world_from_rig"][3], 3.0,
        "reference translation x = index"
    );
    assert_eq!(
        lines[3]["hands"][0]["landmarks"][0][0], 3.0,
        "hands got the pose of their own frameset"
    );
    assert_eq!(lines[3]["slam_status"], "reference");
    assert_eq!(lines[1]["index"], 1);
    assert_eq!(lines[1]["slam_index"], 1);
    assert_eq!(
        lines[1]["slam_t_ns"], lines[1]["t_ns"],
        "the record states whose pose it used"
    );
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn runtime_camera_count_is_rejected_before_pipeline_indexing()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-count-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 1)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = EditedReplay(
        ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?,
        |mut event| {
            if let SourceEvent::Frameset(frame) = &mut event {
                frame.cameras.pop();
            }
            Some(event)
        },
    );
    assert!(matches!(
        lossless_slam_run(source, false, 30.0, stop, None),
        Err(SchedError::Source(SourceError::Frame(
            crate::frame::FrameError::Invalid(_)
        )))
    ));
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn missing_slam_cameras_reset_the_world_before_a_complete_frame_returns()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-split-gap-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump_with(&dir, 100, &|index, camera| index > 0 && camera == 4)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?;
    let (summary, rows) = lossless_slam_run(source, true, 30.0, stop, None)?;
    assert_eq!(summary.counters.slam_missing_cameras, 99);
    assert_eq!(summary.counters.slam_resets, 1, "the gap reset must run during the split");
    assert_eq!(rows.len(), 100);
    assert!(rows.last().unwrap().pose.is_none(), "old-world poses must not survive input loss");
    std::fs::remove_dir_all(dir)?;
    Ok(())
}
