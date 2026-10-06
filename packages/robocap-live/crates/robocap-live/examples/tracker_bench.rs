//! Hand tracker step timings, self-contained so it runs on the cap as one binary.
//!
//! 1. `tracker`: the golden scene (six 1920x1080 KB4 cameras, 100 frames, both hands, drops and re-acquisitions) driven by the
//!    recorded estimator outputs: the tracker's own cost per step (planning, gates, handfit fits), no networks, no crops.
//! 2. `crops`: a still two-hand scene through the real DetNet decode + perspective-crop estimator (crop planning, bilinear
//!    sampling of 96x96 crops from 1920x1080 frames, heatmap decode), the networks replaced by ground-truth renders: the whole
//!    CPU cost of a hands step besides the NPU.
//! 3. `calibration`: the live scale calibration's solve on the golden scene's 98 stereo observations, and on 600 (10 s of two
//!    hands at 30 fps, what `ScaleMode::Auto { seconds: 10 }` collects).
//!
//! Usage: `tracker_bench [repeats] [--nets <models dir>]` (default 10 repeats). With `--nets` the scene benches also run the real
//! RKNN networks (`/usr/lib/librknnrt.so`) on every DetNet frame and KeyNet crop before answering with the
//! renders, so the step pays the NPU's time. Pin it with `taskset -c 0-3` (A55) or `taskset -c 4-7` (A76).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use kornia_image::Image;
use kornia_staging_sensors::{CameraFrame, CaptureMeta};
use nalgebra::Isometry3;
use robocap_live::frame::Luma;
use robocap_live::frame::{FULL_SIZE, NUM_CAMERAS, SMALL_SIZE};
use robocap_live::hands::estimator::PerspectiveKeyNet;
use robocap_live::hands::model::GenericHandModel;
use robocap_live::hands::scale::{CalibrationBlock, CalibrationConfig, calibrate_scale};
use robocap_live::hands::tracker::{Tracker, TrackerConfig};
use robocap_live::hands::{HandFrameResult, HandInputs, HandsConfig, LEFT, RIGHT, ScaleMode};
use robocap_live::nets::{DetNetRaw, HandNets, KeyNetRaw, NUM_LANDMARKS, NetFrame, NetsError};
use robocap_live::sched::Summary;

#[path = "../tests/common/tracker_fixtures.rs"]
mod fixtures;
use fixtures::{
    CallLog, Error, Golden, NoNets, Queue, QueuedNets, RenderedPerception, TRUE_PHI,
    TablePerception, Truth, golden, scene_landmarks,
};

/// p50 / p95 / max / mean of a sample, milliseconds.
fn summary(values: &[f64]) -> String {
    if values.is_empty() {
        return "n 0".into();
    }
    let s = Summary::of(values);
    format!(
        "n {:>4}  mean {:>7.3}  p50 {:>7.3}  p95 {:>7.3}  max {:>7.3} ms",
        s.count, s.mean, s.p50, s.p95, s.max
    )
}

fn bench_tracker(golden: &Arc<Golden>, repeats: usize, all_cameras: bool) -> Result<(), Error> {
    let image: Luma = Arc::new(Image::new(
        kornia_image::ImageSize {
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
    let (mut tracking, mut acquiring, mut fit_tracking, mut fit_acquiring, mut all) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..repeats {
        let log = Arc::new(Mutex::new(CallLog::default()));
        let perception = Box::new(TablePerception {
            golden: golden.clone(),
            log: log.clone(),
        });
        let hands = HandsConfig {
            scale: ScaleMode::Fixed(golden.record.tracking_run.phi),
            cameras: (0..NUM_CAMERAS).collect(),
            max_views: 2,
            detnet_groups: if all_cameras { 1 } else { NUM_CAMERAS },
            ..HandsConfig::default()
        };
        let mut tracker = Tracker::new(
            &golden.record.rig,
            &hands,
            TrackerConfig::default(),
            perception,
        )?;
        for f in 0..golden.record.frames {
            log.lock().map_err(|_| "poisoned")?.frame = f;
            let was_tracked = [tracker.is_tracked(LEFT), tracker.is_tracked(RIGHT)];
            let inputs = HandInputs {
                turned_180: [false; NUM_CAMERAS],
                index: f as u64,
                t_ns: f as i64 * 33_333_333,
                full: [Some(&camera_frame); NUM_CAMERAS],
                small: images,
            };
            let begin = Instant::now();
            let result: HandFrameResult =
                tracker.track(&inputs, &golden.isometry(f), &mut NoNets)?;
            let ms = begin.elapsed().as_secs_f64() * 1e3;
            all.push(ms);
            if result.detnet_camera.is_some()
                && result
                    .hands
                    .iter()
                    .zip(was_tracked)
                    .any(|(h, was)| h.tracked && !was)
            {
                acquiring.push(ms);
                fit_acquiring.push(result.timings.fit_ms);
            } else if was_tracked.iter().all(|t| *t) && result.hands.iter().all(|h| h.tracked) {
                tracking.push(ms);
                fit_tracking.push(result.timings.fit_ms);
            }
        }
    }
    println!(
        "tracker (recorded estimator outputs, no nets, no crops), DetNet {}, {} frames x {repeats}:",
        if all_cameras {
            "on all cameras"
        } else {
            "round robin"
        },
        golden.record.frames
    );
    println!("  every step               {}", summary(&all));
    println!("  two hands tracked        {}", summary(&tracking));
    println!("    of which handfit fits  {}", summary(&fit_tracking));
    println!("  steps with an acquisition {}", summary(&acquiring));
    println!("    of which handfit fits  {}", summary(&fit_acquiring));
    Ok(())
}

/// Networks for the scene benches: the ground-truth renders queued by [`RenderedPerception`], optionally after running the real
/// networks on the same inputs (their outputs are discarded: black frames carry no hands), so a step pays the real NPU time and
/// still tracks the scene.
struct SceneNets {
    queue: Arc<Mutex<Queue>>,
    real: Option<Box<dyn HandNets>>,
}

impl HandNets for SceneNets {
    fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        if let Some(real) = self.real.as_mut() {
            real.detnet(frames)?;
        }
        QueuedNets {
            queue: self.queue.clone(),
        }
        .detnet(frames)
    }
    fn keynet(
        &mut self,
        crops: &[&[f32]],
        keypoints: &[[f32; 3 * NUM_LANDMARKS]],
    ) -> Result<Vec<KeyNetRaw>, NetsError> {
        if let Some(real) = self.real.as_mut() {
            real.keynet(crops, keypoints)?;
        }
        QueuedNets {
            queue: self.queue.clone(),
        }
        .keynet(crops, keypoints)
    }
    fn describe(&self) -> String {
        self.real.as_ref().map_or_else(
            || "renders".into(),
            |r| format!("renders after {}", r.describe()),
        )
    }
}

fn bench_scene(
    golden: &Golden,
    repeats: usize,
    (all_cameras, groups): (bool, usize),
    right_hidden: bool,
    real: &mut Option<Box<dyn HandNets>>,
) -> Result<(), Error> {
    let truth = Arc::new(Truth::new(
        &golden.record.rig,
        scene_landmarks(right_hidden)?,
    )?);
    let full = CameraFrame {
        meta: CaptureMeta::default(),
        full: Arc::new(Image::from_size_val(FULL_SIZE, 0u8)?),
    };
    let small: Luma = Arc::new(Image::from_size_val(SMALL_SIZE, 0u8)?);
    let (mut steps, mut detnet, mut crops, mut keynet, mut fits, mut rest) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut acquisitions = Vec::new();
    for _ in 0..repeats.max(1) {
        let queue = Arc::new(Mutex::new(Queue::default()));
        let estimator = PerspectiveKeyNet::new(&golden.record.rig, TRUE_PHI)?;
        let perception = RenderedPerception {
            truth: truth.clone(),
            estimator,
            queue: queue.clone(),
        };
        let hands = HandsConfig {
            scale: ScaleMode::Fixed(TRUE_PHI),
            cameras: (0..NUM_CAMERAS).collect(),
            max_views: 2,
            detnet_groups: if all_cameras { groups } else { NUM_CAMERAS },
            ..HandsConfig::default()
        };
        let mut tracker = Tracker::new(
            &golden.record.rig,
            &hands,
            TrackerConfig::default(),
            Box::new(perception),
        )?;
        let mut nets = SceneNets {
            queue,
            real: real.take(),
        };
        for f in 0..40 {
            let inputs = HandInputs {
                turned_180: [false; NUM_CAMERAS],
                index: f,
                t_ns: f as i64 * 33_333_333,
                full: [Some(&full); NUM_CAMERAS],
                small: [Some(&small); NUM_CAMERAS],
            };
            let was_tracked = tracker.is_tracked(LEFT);
            let begin = Instant::now();
            let result = tracker.track(&inputs, &Isometry3::identity(), &mut nets)?;
            let ms = begin.elapsed().as_secs_f64() * 1e3;
            if !was_tracked && result.hands[LEFT].tracked {
                acquisitions.push(ms);
            }
            let steady =
                result.hands[LEFT].reported && (right_hidden || result.hands[RIGHT].reported);
            if f >= 10 && steady {
                steps.push(ms);
                detnet.push(result.timings.detnet_ms);
                crops.push(result.timings.crops_ms);
                keynet.push(result.timings.keynet_ms);
                fits.push(result.timings.fit_ms);
                rest.push(result.timings.tracker_ms);
            }
        }
        *real = nets.real.take();
    }
    let nets_name = real
        .as_ref()
        .map_or("ground-truth renders (no NPU)".to_string(), |r| {
            format!("{} + renders", r.describe())
        });
    println!(
        "still scene through the perspective-crop estimator, {}, DetNet {}, nets: {nets_name}",
        if right_hidden {
            "left hand tracked, right hand hidden (DetNet every frameset)"
        } else {
            "both hands tracked (4 crops)"
        },
        match (all_cameras, groups) {
            (false, _) => "round robin".to_string(),
            (true, 1) => "on all 6 cameras".to_string(),
            (true, g) => format!("on all cameras in {g} alternating groups"),
        },
    );
    println!("  step                    {}", summary(&steps));
    println!("    DetNet (letterbox+net) {}", summary(&detnet));
    println!("    crops (plan + sample)  {}", summary(&crops));
    println!("    KeyNet + decode        {}", summary(&keynet));
    println!("    handfit fits           {}", summary(&fits));
    println!("    tracker bookkeeping    {}", summary(&rest));
    println!("  acquisition steps       {}", summary(&acquisitions));
    Ok(())
}

fn bench_calibration(golden: &Golden, repeats: usize) -> Result<(), Error> {
    let blocks = golden.calibration_blocks()?;
    let generic = GenericHandModel::load()?;
    for count in [blocks.len(), 600] {
        let set: Vec<CalibrationBlock> = blocks.iter().cycle().take(count).cloned().collect();
        let mut times = Vec::new();
        let mut last = None;
        for _ in 0..repeats.clamp(1, 5) {
            let begin = Instant::now();
            last = Some(calibrate_scale(
                generic.model(),
                &set,
                &CalibrationConfig::default(),
            )?);
            times.push(begin.elapsed().as_secs_f64() * 1e3);
        }
        let calibration = last.ok_or("no run")?;
        println!(
            "calibration solve, {count} observations: {}  (phi {:.5}, Python {:.5}, {} iterations, {})",
            summary(&times),
            calibration.phi,
            golden.record.calibration.phi,
            calibration.iterations,
            calibration.termination.as_str()
        );
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut repeats: usize = 10;
    let mut models: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--nets" => {
                models = Some(
                    args.get(index + 1)
                        .ok_or("--nets needs the models directory")?
                        .clone(),
                );
                index += 1;
            }
            value => repeats = value.parse()?,
        }
        index += 1;
    }
    let golden = Arc::new(golden()?);
    let mut real: Option<Box<dyn HandNets>> = match &models {
        Some(dir) => Some(Box::new(robocap_live::nets::rknn::RknnNets::open(dir)?)),
        None => None,
    };
    for all_cameras in [false, true] {
        bench_tracker(&golden, repeats, all_cameras)?;
    }
    for right_hidden in [false, true] {
        for mode in [(false, 1), (true, 1), (true, 2)] {
            bench_scene(&golden, repeats.div_ceil(5), mode, right_hidden, &mut real)?;
        }
    }
    bench_calibration(&golden, repeats)?;
    Ok(())
}
