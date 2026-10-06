//! Golden test of the hand tracker against handtrack's Python `Tracker` (ROBUST_TRACKER_CONFIG, native fit).
//!
//! `tools/golden_tracker.py` ran the Python tracker on a synthetic six-camera RoboCap-like scene, driven by recorded fake DetNet
//! and KeyNet outputs (`tests/data/tracker/{detections,estimates}.bin`), and recorded every frame. This test drives the Rust
//! tracker with the same outputs and checks, frame by frame: the DetNet camera, every KeyNet request (camera, hand, acquisition
//! or tracked, circle), the internal and reported state of each hand, θ(t) and the reported landmarks; then the live scale
//! calibration (`ScaleMode::Auto` over the same 60 frames) against Python's `calibrate_scale`, and `calibrate_scale` alone on
//! Python's own observations.
#![allow(clippy::needless_range_loop)] // parallel arrays of poses are clearer indexed

use std::sync::{Arc, Mutex};

use kornia_image::{Image, ImageSize};
use kornia_staging_sensors::{CameraFrame, CaptureMeta};
use robocap_live::frame::Luma;
use robocap_live::frame::NUM_CAMERAS;
use robocap_live::hands::model::GenericHandModel;
use robocap_live::hands::scale::{CalibrationConfig, calibrate_scale};
use robocap_live::hands::tracker::{Tracker, TrackerConfig};
use robocap_live::hands::{HandFrameResult, HandInputs, HandsConfig, ScaleMode};

#[path = "common/tracker_fixtures.rs"]
mod fixtures;
use fixtures::{CallLog, Golden, NoNets, Request, RunRecord, TablePerception, golden};

struct Run {
    results: Vec<HandFrameResult>,
    poses: Vec<[Option<[f64; 34]>; 2]>,
    log: Arc<Mutex<CallLog>>,
    tracker: Tracker,
}

fn run(
    golden: &Arc<Golden>,
    scale: ScaleMode,
    frames: usize,
) -> Result<Run, Box<dyn std::error::Error>> {
    run_with(golden, scale, frames, false)
}

/// [`run`] with ROBUST_TRACKER_CONFIG and DetNet on every camera (`detnet_all_cameras`, one group) or round robin (a group per
/// camera).
fn run_with(
    golden: &Arc<Golden>,
    scale: ScaleMode,
    frames: usize,
    detnet_all_cameras: bool,
) -> Result<Run, Box<dyn std::error::Error>> {
    let log = Arc::new(Mutex::new(CallLog::default()));
    let perception = Box::new(TablePerception {
        golden: golden.clone(),
        log: log.clone(),
    });
    let hands = HandsConfig {
        scale,
        cameras: (0..NUM_CAMERAS).collect(),
        max_views: 2,
        detnet_cameras: None,
        detnet_groups: if detnet_all_cameras { 1 } else { NUM_CAMERAS },
        scale_wait: false,
        acquire_threads: 4,
    };
    let mut tracker = Tracker::new(
        &golden.record.rig,
        &hands,
        TrackerConfig::robust(),
        perception,
    )?;
    let image: Luma = Arc::new(Image::new(
        ImageSize {
            width: 1,
            height: 1,
        },
        vec![0u8],
    )?);
    let images: [Option<&Luma>; NUM_CAMERAS] = [Some(&image); NUM_CAMERAS];
    let camera_frame = CameraFrame {
        meta: CaptureMeta::default(),
        full: image.clone(),
    };
    let mut nets = NoNets;
    let mut results = Vec::new();
    let mut poses = Vec::new();
    for frame in 0..frames {
        log.lock().map_err(|_| "log poisoned")?.frame = frame;
        let inputs = HandInputs {
            turned_180: [false; NUM_CAMERAS],
            index: frame as u64,
            t_ns: frame as i64 * 33_333_333,
            full: [Some(&camera_frame); NUM_CAMERAS],
            small: images,
        };
        results.push(tracker.track(&inputs, &golden.isometry(frame), &mut nets)?);
        poses.push(std::array::from_fn(|side| {
            tracker.pose(side).map(|pose| {
                let mut values = [0.0; 34];
                for i in 0..9 {
                    values[i] = pose.rotation[(i / 3, i % 3)];
                }
                for i in 0..3 {
                    values[9 + i] = pose.translation[i];
                }
                for i in 0..22 {
                    values[12 + i] = pose.angles[i];
                }
                values
            })
        }));
    }
    Ok(Run {
        results,
        poses,
        log,
        tracker,
    })
}

/// Largest deviations seen over a run, for the report.
#[derive(Debug, Default)]
struct Deviations {
    translation_m: f64,
    rotation: f64,
    angle_rad: f64,
    landmark_m: f64,
    circle_px: f64,
}

fn compare(name: &str, golden: &RunRecord, run: &Run) -> Result<Deviations, String> {
    let log = run.log.lock().map_err(|_| "log poisoned")?;
    let mut worst = Deviations::default();
    for (index, (expected, result)) in golden.frames.iter().zip(&run.results).enumerate() {
        let frame = expected.frame;
        assert_eq!(frame, index);
        let context = format!("{name} frame {frame}");
        // Python marks a DetNet pass over every camera with the camera count.
        let detnet = match result.detnet_camera {
            None => -1,
            Some(_) if expected.detnet_camera == NUM_CAMERAS as i64 => NUM_CAMERAS as i64,
            Some(c) => c as i64,
        };
        if detnet != expected.detnet_camera {
            return Err(format!(
                "{context}: DetNet camera {detnet}, Python {}",
                expected.detnet_camera
            ));
        }
        let calls: Vec<&Vec<Request>> = log
            .keynet
            .iter()
            .filter(|(f, _)| *f == frame)
            .map(|(_, r)| r)
            .collect();
        if calls.len() != expected.keynet_calls.len() {
            return Err(format!(
                "{context}: {} KeyNet calls, Python {}",
                calls.len(),
                expected.keynet_calls.len()
            ));
        }
        for (call, want) in calls.iter().zip(&expected.keynet_calls) {
            let got: Vec<(usize, usize, bool)> = call
                .iter()
                .map(|r| (r.camera, r.side, r.acquisition))
                .collect();
            let want_views: Vec<(usize, usize, bool)> = want
                .iter()
                .map(|r| (r.camera, r.side, r.acquisition))
                .collect();
            if got != want_views {
                return Err(format!(
                    "{context}: KeyNet views {got:?}, Python {want_views:?}"
                ));
            }
            for (r, w) in call.iter().zip(want) {
                for k in 0..3 {
                    worst.circle_px = worst.circle_px.max((r.circle[k] - w.circle[k]).abs());
                }
            }
        }
        for side in 0..2 {
            let hand = &result.hands[side];
            if hand.tracked != expected.tracked[side] || hand.reported != expected.reported[side] {
                return Err(format!(
                    "{context} hand {side}: tracked {} reported {}, Python {} {}",
                    hand.tracked, hand.reported, expected.tracked[side], expected.reported[side]
                ));
            }
            if let (Some(pose), Some(want)) = (&run.poses[index][side], &expected.poses[side]) {
                for i in 0..9 {
                    worst.rotation = worst.rotation.max((pose[i] - want.rotation[i]).abs());
                }
                for i in 0..3 {
                    worst.translation_m = worst
                        .translation_m
                        .max((pose[9 + i] - want.translation[i]).abs());
                }
                for i in 0..22 {
                    worst.angle_rad = worst
                        .angle_rad
                        .max((pose[12 + i] - want.joint_angles[i]).abs());
                }
            }
            if expected.reported[side] {
                let views: Vec<usize> = hand.fitted_views().map(|v| v.camera).collect();
                if views != expected.view_cameras[side] {
                    return Err(format!(
                        "{context} hand {side}: fit views {views:?}, Python {:?}",
                        expected.view_cameras[side]
                    ));
                }
                let (Some(landmarks), Some(want)) =
                    (&hand.landmarks_world, &expected.landmarks[side])
                else {
                    return Err(format!("{context} hand {side}: reported without landmarks"));
                };
                for (p, q) in landmarks.iter().zip(want) {
                    for k in 0..3 {
                        worst.landmark_m = worst.landmark_m.max((p[k] - q[k]).abs());
                    }
                }
            }
        }
    }
    Ok(worst)
}

/// Pose tolerances. The same float32 inputs reach handfit's f64 solve on both sides and both return float32 poses, so θ(t) agrees
/// to a float32 ulp or two (measured: translation <= 3e-8 m, rotation 6e-8, joint angles 1.8e-7 rad, landmarks 3e-7 m). What
/// differs: the planning pose (projections, the extrapolation's SVD) is float32 torch in Python and f64 here, landmarks are
/// skinned in float32 torch there, and the headset pose comes back from an isometry. The bounds leave ~5x headroom.
const MAX_TRANSLATION_M: f64 = 2e-7;
const MAX_LANDMARK_M: f64 = 2e-6;
const MAX_ROTATION: f64 = 5e-7;
const MAX_ANGLE_RAD: f64 = 1e-6;
/// Circles (measured 5e-4 px): OpenCV's float32 `minEnclosingCircle` pads its radius by 1e-4; float32 torch projections.
const MAX_CIRCLE_PX: f64 = 3e-3;

fn check(deviations: &Deviations) {
    eprintln!("{deviations:?}");
    assert!(
        deviations.translation_m < MAX_TRANSLATION_M,
        "{deviations:?}"
    );
    assert!(deviations.landmark_m < MAX_LANDMARK_M, "{deviations:?}");
    assert!(deviations.rotation < MAX_ROTATION, "{deviations:?}");
    assert!(deviations.angle_rad < MAX_ANGLE_RAD, "{deviations:?}");
    assert!(deviations.circle_px < MAX_CIRCLE_PX, "{deviations:?}");
}

#[test]
fn tracking_at_the_calibrated_scale_matches_python() -> Result<(), Box<dyn std::error::Error>> {
    let golden = Arc::new(golden()?);
    let phi = golden.record.tracking_run.phi;
    let frames = golden.record.frames;
    let run = run(&golden, ScaleMode::Fixed(phi), frames)?;
    check(&compare("tracking", &golden.record.tracking_run, &run)?);
    Ok(())
}

#[test]
fn detnet_on_all_cameras_matches_python() -> Result<(), Box<dyn std::error::Error>> {
    let golden = Arc::new(golden()?);
    let run = run_with(
        &golden,
        ScaleMode::Fixed(golden.record.all_cameras_run.phi),
        golden.record.frames,
        true,
    )?;
    check(&compare(
        "all cameras",
        &golden.record.all_cameras_run,
        &run,
    )?);
    // DetNet ran on all six cameras on every frame with an untracked hand.
    let log = run.log.lock().map_err(|_| "log poisoned")?;
    for frame in golden
        .record
        .all_cameras_run
        .frames
        .iter()
        .filter(|f| f.detnet_camera >= 0)
    {
        let cameras: Vec<usize> = log
            .detnet
            .iter()
            .filter(|(f, _)| *f == frame.frame)
            .map(|(_, c)| *c)
            .collect();
        assert_eq!(
            cameras,
            (0..NUM_CAMERAS).collect::<Vec<_>>(),
            "frame {}",
            frame.frame
        );
    }
    Ok(())
}

#[test]
fn live_calibration_matches_python_calibrate_unknown_hand() -> Result<(), Box<dyn std::error::Error>>
{
    let golden = Arc::new(golden()?);
    let calibration_frames = golden.record.calibration_frames;
    // Frames 0..59 lie within the first 2 s; the solve starts on frame 60.
    let seconds = calibration_frames as f64 * 33_333_333.0 / 1e9;
    let mut run = run(&golden, ScaleMode::Auto { seconds }, calibration_frames + 1)?;
    let deviations = compare("calibration", &golden.record.calibration_run, &run)?;
    check(&deviations);
    assert!(
        run.results
            .iter()
            .take(calibration_frames)
            .all(|r| r.scale == 1.0 && !r.scale_final)
    );
    run.tracker.finish_scale();
    let outcome = run
        .tracker
        .scale_outcome()
        .cloned()
        .ok_or("no calibration outcome")?;
    eprintln!(
        "Rust phi {:.6} from {} blocks in {:.3} s; Python {:.6} from {}",
        outcome.phi,
        outcome.blocks,
        outcome.solve_s,
        golden.record.calibration.phi_raw,
        golden.record.calibration.blocks
    );
    assert_eq!(outcome.blocks, golden.record.calibration.blocks);
    // Measured: 0.920668 vs 0.920667 (same 25 iterations); handtrack's Jacobian is float32 central differences, ours analytic.
    assert!(
        (outcome.phi - golden.record.calibration.phi_raw).abs() < 1e-5,
        "{outcome:?}"
    );
    assert!((run.tracker.phi() - outcome.phi).abs() < 1e-15);
    Ok(())
}

#[test]
fn calibrate_scale_on_pythons_observations_matches_python() -> Result<(), Box<dyn std::error::Error>>
{
    let golden = golden()?;
    let blocks = golden.calibration_blocks()?;
    let generic = GenericHandModel::load()?;
    let begin = std::time::Instant::now();
    let calibration = calibrate_scale(generic.model(), &blocks, &CalibrationConfig::default())?;
    let expected = &golden.record.calibration;
    eprintln!(
        "Rust phi {:.6} ({} iterations, {}, e_2d {:.1}, {:.3} s); Python {:.6} ({} iterations, {}, e_2d {:.1})",
        calibration.phi,
        calibration.iterations,
        calibration.termination.as_str(),
        calibration.e_2d,
        begin.elapsed().as_secs_f64(),
        expected.phi_raw,
        expected.iterations,
        expected.termination,
        expected.e_2d
    );
    assert_eq!(calibration.blocks, expected.blocks);
    assert!((calibration.phi - expected.phi_raw).abs() < 1e-5);
    assert!((calibration.e_2d - expected.e_2d).abs() < 1e-4 * expected.e_2d);
    assert_eq!(calibration.termination.as_str(), expected.termination);
    Ok(())
}
