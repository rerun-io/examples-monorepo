//! The tracker end to end through the real perception: DetNet letterbox + decode (`hands::detect`) and the perspective-crop
//! KeyNet estimator (`hands::estimator::PerspectiveKeyNet`: crop planning, crop sampling from 1920x1080 frames, heatmap and
//! distance decode, back-projection through the fisheye and the letterbox), with fake networks whose outputs are rendered
//! from the ground truth: DetNet's circle from the projected keypoints, KeyNet's heatmaps from the keypoints projected into
//! each planned crop (handtrack `labels/heatmaps.py::render_heatmaps` / `render_distance`, ported in `common/tracker_fixtures.rs`).
//!
//! The scene: the golden test's six-camera RoboCap-like rig (1920x1080 KB4), a still headset, two still hands 8 % smaller
//! than the generic hand.

use std::sync::{Arc, Mutex};

use kornia_image::Image;
use kornia_staging_sensors::{CameraFrame, CaptureMeta};
use nalgebra::Isometry3;
use robocap_live::frame::Luma;
use robocap_live::frame::{FULL_SIZE, NUM_CAMERAS, SMALL_SIZE};
use robocap_live::hands::estimator::PerspectiveKeyNet;
use robocap_live::hands::tracker::{Tracker, TrackerConfig};
use robocap_live::hands::{HandFrameResult, HandInputs, HandsConfig, LEFT, RIGHT, ScaleMode};
use robocap_live::nets::NUM_LANDMARKS;

#[path = "common/tracker_fixtures.rs"]
mod fixtures;
use fixtures::{Queue, QueuedNets, RenderedPerception, TRUE_PHI, Truth, golden, scene_landmarks};

struct Scene {
    tracker: Tracker,
    truth: Arc<Truth>,
    queue: Arc<Mutex<Queue>>,
    full: CameraFrame,
    small: Luma,
    next_frame: usize,
}

fn scene(scale: ScaleMode) -> Result<Scene, Box<dyn std::error::Error>> {
    scene_with(scale, false)
}

fn scene_with(scale: ScaleMode, scale_wait: bool) -> Result<Scene, Box<dyn std::error::Error>> {
    let rig = golden()?.record.rig;
    let truth = Arc::new(Truth::new(&rig, scene_landmarks(false)?)?);
    let queue = Arc::new(Mutex::new(Queue::default()));
    let phi = match scale {
        ScaleMode::Fixed(phi) => phi,
        ScaleMode::Auto { .. } => 1.0,
    };
    let perception = RenderedPerception {
        truth: truth.clone(),
        estimator: PerspectiveKeyNet::new(&rig, phi)?,
        queue: queue.clone(),
    };
    let hands = HandsConfig {
        scale,
        cameras: (0..NUM_CAMERAS).collect(),
        max_views: 2,
        scale_wait,
        ..HandsConfig::default()
    };
    let tracker = Tracker::new(&rig, &hands, TrackerConfig::robust(), Box::new(perception))?;
    let full = CameraFrame {
        meta: CaptureMeta::default(),
        full: Arc::new(Image::from_size_val(FULL_SIZE, 0u8)?),
    };
    let small: Luma = Arc::new(Image::from_size_val(SMALL_SIZE, 0u8)?);
    Ok(Scene {
        tracker,
        truth,
        queue,
        full,
        small,
        next_frame: 0,
    })
}

impl Scene {
    fn run(&mut self, frames: usize) -> Result<Vec<HandFrameResult>, Box<dyn std::error::Error>> {
        let mut nets = QueuedNets {
            queue: self.queue.clone(),
        };
        let mut results = Vec::new();
        for _ in 0..frames {
            let frame = self.next_frame;
            self.next_frame += 1;
            let full: [Option<&CameraFrame>; NUM_CAMERAS] = [Some(&self.full); NUM_CAMERAS];
            let small: [Option<&Luma>; NUM_CAMERAS] = [Some(&self.small); NUM_CAMERAS];
            let inputs = HandInputs {
                turned_180: [false; NUM_CAMERAS],
                index: frame as u64,
                t_ns: frame as i64 * 33_333_333,
                full,
                small,
            };
            results.push(
                self.tracker
                    .track(&inputs, &Isometry3::identity(), &mut nets)?,
            );
        }
        Ok(results)
    }

    fn error_mm(&self, result: &HandFrameResult, side: usize) -> f64 {
        result.hands[side]
            .landmarks_world
            .map_or(f64::INFINITY, |landmarks| {
                landmarks
                    .iter()
                    .zip(&self.truth.landmarks[side])
                    .map(|(p, q)| {
                        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                            .sqrt()
                            * 1000.0
                    })
                    .sum::<f64>()
                    / NUM_LANDMARKS as f64
            })
    }
}

#[test]
fn the_tracker_acquires_and_tracks_a_still_scene_through_the_perspective_estimator()
-> Result<(), Box<dyn std::error::Error>> {
    let mut scene = scene(ScaleMode::Fixed(TRUE_PHI))?;
    let results = scene.run(10)?;
    let last = &results[9];
    for side in [LEFT, RIGHT] {
        let error = scene.error_mm(last, side);
        eprintln!(
            "hand {side}: mean landmark error {error:.3} mm, views {:?}",
            last.hands[side]
                .fitted_views()
                .map(|v| v.camera)
                .collect::<Vec<_>>()
        );
        assert!(
            last.hands[side].reported,
            "hand {side} not reported: {:?}",
            results
                .iter()
                .map(|r| r.hands[side].tracked)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            last.hands[side].fitted_views().count(),
            2,
            "hand {side} in stereo"
        );
        assert!(error < 1.0, "hand {side}: {error} mm");
        // Each KeyNet view names the perspective crop it sampled, aimed at the hand it found.
        for view in &last.hands[side].keynet_views {
            let crop = view.crop.ok_or("a KeyNet view without its crop camera")?;
            assert!(
                crop.usable(),
                "hand {side} camera {}: unusable crop",
                view.camera
            );
        }
    }
    let queue = scene.queue.lock().map_err(|_| "poisoned")?;
    assert!(
        queue.detnet.is_empty() && queue.keynet.is_empty(),
        "every rendered output was consumed"
    );
    eprintln!(
        "DetNet frames {}, KeyNet crops {}",
        queue.detnet_calls, queue.keynet_crops
    );
    Ok(())
}

#[test]
fn scale_wait_switches_on_the_frameset_that_starts_the_solve()
-> Result<(), Box<dyn std::error::Error>> {
    let mut scene = scene_with(ScaleMode::Auto { seconds: 0.49 }, true)?;
    let results = scene.run(17)?;
    assert!(
        results[..15]
            .iter()
            .all(|r| r.scale == 1.0 && !r.scale_final)
    );
    assert!(
        results[15].scale_final && (results[15].scale - TRUE_PHI).abs() < 5e-3,
        "frameset 15 reports the new scale"
    );
    Ok(())
}

#[test]
fn live_calibration_finds_the_hand_scale_through_the_perspective_estimator()
-> Result<(), Box<dyn std::error::Error>> {
    let mut scene = scene(ScaleMode::Auto { seconds: 0.49 })?; // frames 0..14 (t < 490 ms)
    let first = scene.run(15)?;
    assert!(first.iter().all(|r| r.scale == 1.0 && !r.scale_final));
    scene.run(1)?; // frame 15 starts the solve on its thread
    scene.tracker.finish_scale();
    let outcome = scene
        .tracker
        .scale_outcome()
        .cloned()
        .ok_or("no calibration")?;
    eprintln!(
        "calibrated phi {:.4} from {} blocks in {:.3} s ({})",
        outcome.phi, outcome.blocks, outcome.solve_s, outcome.note
    );
    assert!((outcome.phi - TRUE_PHI).abs() < 5e-3, "{outcome:?}");
    let after = scene.run(5)?;
    let last = &after[4];
    assert!(last.scale_final && (last.scale - outcome.phi).abs() < 1e-12);
    for side in [LEFT, RIGHT] {
        let error = scene.error_mm(last, side);
        eprintln!("hand {side}: mean landmark error {error:.3} mm after the switch");
        assert!(error < 2.0, "hand {side}: {error} mm");
    }
    Ok(())
}
