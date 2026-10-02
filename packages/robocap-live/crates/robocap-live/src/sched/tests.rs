//! The pipeline's tests: queue policies end to end, missing cameras, network failures, stage start failures, the record.

use super::*;
use crate::hands::HandOutput;
use crate::nets::{DetNetRaw, KeyNetRaw, NetFrame, NetsError};
use crate::source::replay::{ReplayConfig, ReplaySource, read_reference_poses};

/// A test pipeline: no pinning, no frequency hints, no per-second lines.
fn test_config(lossless: bool, slam_mode: SlamMode, reference: ReferencePoses, hands: HandsStage) -> PipelineConfig {
    PipelineConfig {
        lossless,
        slam_mode,
        slam: SlamConfig::default(),
        reference: Some(reference),
        hands: Some(hands),
        hands_wait: Duration::from_millis(5),
        imu_wait: Duration::from_millis(20),
        small_cameras: None,
        downsample_threads: 2,
        cpus_big: None,
        cpus_little: None,
        cpus_downsample: None,
        cpus_hands: None,
        slam_uclamp_min: None,
        hands_uclamp_min: None,
        duration: None,
        print_every_second: false,
    }
}

struct CountingSink(Arc<Mutex<Vec<(u64, bool, bool)>>>);

impl FramesetSink for CountingSink {
    fn frameset(&mut self, record: &OutputRecord<'_>) -> Result<(), SinkError> {
        let small_ok = record.small.iter().flatten().all(|s| s.size() == crate::frame::SMALL_SIZE)
            && record.small.iter().zip(record.frameset.cameras.iter()).all(|(small, full)| small.is_some() == full.is_some());
        lock(&self.0).push((record.frameset.index, record.pose.is_some_and(|p| p.ok), small_ok && record.hands.is_some()));
        Ok(())
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

struct NoNets;
impl HandNets for NoNets {
    fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        Ok(frames.iter().map(|_| DetNetRaw { center: [[0.0; 2]; 2], radius: [0.0; 2], presence_logit: [-10.0; 2] }).collect())
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
    fn step(&mut self, inputs: &HandInputs<'_>, world_from_rig: &Isometry3<f64>, _: &mut dyn HandNets) -> Result<HandFrameResult, HandsError> {
        let mut result = HandFrameResult { scale: 1.0, ..Default::default() };
        result.hands[0] = HandOutput { tracked: true, landmarks_world: Some([[world_from_rig.translation.x, inputs.index as f64, 0.0]; 21]), ..Default::default() };
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
            return Err(NetsError::Run { net: "detnet", message: format!("fake rknn_run failure on call {call}") });
        }
        Ok(frames.iter().map(|_| DetNetRaw { center: [[0.0; 2]; 2], radius: [0.0; 2], presence_logit: [-10.0; 2] }).collect())
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
    fn step(&mut self, inputs: &HandInputs<'_>, _: &Isometry3<f64>, nets: &mut dyn HandNets) -> Result<HandFrameResult, HandsError> {
        let frame = vec![0u8; crate::nets::DETNET_WIDTH * crate::nets::DETNET_HEIGHT];
        if inputs.small[0].is_some() {
            nets.detnet(&[NetFrame { pixels: &frame, top: 0 }])?;
        }
        Ok(HandFrameResult { scale: 1.0, detnet_camera: Some(0), ..Default::default() })
    }
}

/// The NPU-failure policy: a failing or stalling network costs that frameset's hands only; the networks are rebuilt, the
/// output keeps receiving framesets (around the stall), and the run ends normally.
#[test]
fn network_failures_skip_hands_rebuild_the_networks_and_keep_the_output_going() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-npu-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump_with(&dir, 24, &|_, camera| camera != 0)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = ReplaySource::open(&dir, ReplayConfig { realtime: true, preload: true, ..ReplayConfig::default() }, stop.clone())?;
    let reference = ReferencePoses::new(read_reference_poses(&dir)?, source.first_t_ns(), None);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let rebuilt = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let factory: NetsFactory = {
        let (calls, rebuilt) = (calls.clone(), rebuilt.clone());
        Box::new(move || {
            rebuilt.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FailingNets { calls: calls.clone() }) as Box<dyn HandNets>)
        })
    };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hands = HandsStage { tracker: Box::new(DetNetTracker), nets: Box::new(FailingNets { calls: calls.clone() }), nets_factory: Some(factory) };
    let config = PipelineConfig {
        hands_wait: Duration::ZERO,
        downsample_threads: 1,
        ..test_config(false, SlamMode::Reference, reference, hands)
    };
    let sinks: Vec<Box<dyn FramesetSink>> = vec![Box::new(CountingSink(seen.clone()))];
    let summary = run(Box::new(source), config, sinks, stop)?;
    assert!(summary.counters.nets_failures >= 1, "{summary:?}");
    assert!(summary.counters.nets_recreated >= 1 && rebuilt.load(Ordering::SeqCst) as u64 == summary.counters.nets_recreated);
    assert!(summary.counters.hands_bypassed >= 1, "framesets went around the 500 ms stall: {summary:?}");
    assert_eq!(summary.hands_dropped, summary.counters.hands_bypassed, "every frameset hands could not take went to the output");
    let seen = lock(&seen);
    let indices: Vec<u64> = seen.iter().map(|(index, _, _)| *index).collect();
    assert!(indices.windows(2).all(|pair| pair[0] < pair[1]), "the output stays in order: {indices:?}");
    assert!(seen.iter().all(|(_, pose_ok, _)| *pose_ok), "poses keep flowing to the output");
    assert!(indices.len() >= 21, "the output keeps going through the failures and the stall: {indices:?}");
    drop(seen);
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Framesets missing camera 0 (the last ones of s66-full) or other cameras must flow through every stage, in every SLAM mode,
/// and the run must end (a lossless s66-full replay hung there).
#[test]
fn framesets_missing_cameras_flow_through_and_the_run_ends() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-missing-{}", std::process::id()));
    let missing = |index: u64, camera: usize| (index >= 4 && camera == 0) || (index == 2 && [1, 4, 5].contains(&camera)) || index == 3;
    crate::source::replay::tests::write_test_dump_with(&dir, 7, &missing)?;
    for (slam_mode, realtime) in [(SlamMode::Reference, false), (SlamMode::On, false), (SlamMode::Off, false), (SlamMode::On, true)] {
        let stop = Arc::new(AtomicBool::new(false));
        let source = ReplaySource::open(&dir, ReplayConfig { realtime, ..ReplayConfig::default() }, stop.clone())?;
        let reference = ReferencePoses::new(read_reference_poses(&dir)?, source.first_t_ns(), None);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hands = HandsStage { tracker: Box::new(EchoTracker), nets: Box::new(NoNets), nets_factory: None };
        let config = test_config(!realtime, slam_mode, reference, hands);
        let sinks: Vec<Box<dyn FramesetSink>> = vec![Box::new(CountingSink(seen.clone()))];
        let (done_tx, done_rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = done_tx.send(run(Box::new(source), config, sinks, stop).map(|summary| summary.framesets).map_err(|e| e.to_string()));
        });
        let framesets = done_rx.recv_timeout(Duration::from_secs(60)).map_err(|_| format!("{slam_mode:?} realtime={realtime}: the run did not end"))??;
        assert_eq!(framesets, 7, "{slam_mode:?}");
        let indices: Vec<u64> = lock(&seen).iter().map(|(index, _, _)| *index).collect();
        assert_eq!(indices, (0..7).collect::<Vec<_>>(), "{slam_mode:?} realtime={realtime}: every frameset reaches the output");
    }
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Panics on its third frameset.
struct PanickingTracker(u64);
impl HandTracking for PanickingTracker {
    fn step(&mut self, _: &HandInputs<'_>, _: &Isometry3<f64>, _: &mut dyn HandNets) -> Result<HandFrameResult, HandsError> {
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
    let hands = HandsStage { tracker: Box::new(PanickingTracker(0)), nets: Box::new(NoNets), nets_factory: None };
    let config = test_config(true, SlamMode::Reference, reference, hands);
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = done_tx.send(run(Box::new(source), config, Vec::new(), stop).map(|_| ()).map_err(|e| e.to_string()));
    });
    let result = done_rx.recv_timeout(Duration::from_secs(30)).map_err(|_| "the run did not end")?;
    assert!(result.as_ref().is_err_and(|e| e.contains("rl-hands") && e.contains("test panic")), "{result:?}");
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A stage that fails before its loop (here: SLAM cannot pin to a core that does not exist) ends the run with its error; the
/// lossless queue it never drains must not leave the other stages blocked.
#[test]
fn a_stage_that_cannot_start_ends_the_run_with_its_error() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-pin-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 12)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?;
    let reference = ReferencePoses::new(read_reference_poses(&dir)?, source.first_t_ns(), None);
    let hands = HandsStage { tracker: Box::new(EchoTracker), nets: Box::new(NoNets), nets_factory: None };
    let config = PipelineConfig { cpus_big: Some(vec![100_000]), ..test_config(true, SlamMode::On, reference, hands) };
    let (done_tx, done_rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = done_tx.send(run(Box::new(source), config, Vec::new(), stop).map(|_| ()).map_err(|e| e.to_string()));
    });
    let result = done_rx.recv_timeout(Duration::from_secs(30)).map_err(|_| "the run did not end")?;
    assert!(result.as_ref().is_err_and(|e| e.contains("affinity")), "{result:?}");
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn a_lossless_replay_with_reference_poses_reaches_every_stage_and_the_record() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("robocap-live-sched-{}", std::process::id()));
    crate::source::replay::tests::write_test_dump(&dir, 5)?;
    let stop = Arc::new(AtomicBool::new(false));
    let source = ReplaySource::open(&dir, ReplayConfig::default(), stop.clone())?;
    let reference = ReferencePoses::new(read_reference_poses(&dir)?, source.first_t_ns(), None);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let record_path = dir.join("record.jsonl");
    let hands = HandsStage { tracker: Box::new(EchoTracker), nets: Box::new(NoNets), nets_factory: None };
    let config = PipelineConfig {
        hands_wait: Duration::from_millis(10),
        imu_wait: Duration::from_millis(50),
        ..test_config(true, SlamMode::Reference, reference, hands)
    };
    let sinks: Vec<Box<dyn FramesetSink>> = vec![Box::new(CountingSink(seen.clone())), Box::new(RecordWriter::create(&record_path)?)];
    let summary = run(Box::new(source), config, sinks, stop)?;
    assert_eq!(summary.framesets, 5);
    assert_eq!(*lock(&seen), (0..5).map(|i| (i, true, true)).collect::<Vec<_>>());
    let lines: Vec<serde_json::Value> =
        std::fs::read_to_string(&record_path)?.lines().map(serde_json::from_str).collect::<Result<_, _>>()?;
    assert_eq!(lines.len(), 5);
    assert_eq!(lines[3]["world_from_rig"][3], 3.0, "reference translation x = index");
    assert_eq!(lines[3]["hands"][0]["landmarks"][0][0], 3.0, "hands got the pose of their own frameset");
    assert_eq!(lines[3]["slam_status"], "reference");
    assert_eq!(lines[1]["index"], 1);
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
