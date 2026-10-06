//! `FrameToFrameOpticalFlow`: detection, matching, tracking and keypoint ids.
//!
//! Moved out of `src/frontend/flow.rs` (S25): they drive the public API only —
//! `FrameToFrameOpticalFlow::new`, `frame()`, `cell_counts`, `essential`,
//! `detection_grid` — and here they can take the synthetic rig and its frames
//! from `tests/common` instead of keeping a second copy of each. The
//! redetection gate, the refused inputs and the rollback of a failed frame have
//! their own files (`flow_redetection.rs`, `flow_refusals.rs`,
//! `flow_rollback.rs`); the fixtures they share are in `tests/common/flow.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use kornia_image::Image;
use kornia_staging_imgproc::features::{BandRequest, CornerScan, DetectError, FastCorner};
use kornia_staging_imgproc::features::{CellGrid, MaskRect, Masks};
use nalgebra::{Matrix4, Vector3};
use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::frontend::flow::*;
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::CpuPatchTracker;
use slam_rs::lie::{Se3, So3};
use slam_rs::pyramid::CpuPyramidBuilder;
use slam_rs::types::KeypointId;

mod common;

use common::flow::frontend;
use common::{
    FLOW_HEIGHT as HEIGHT, FLOW_WIDTH as WIDTH, dotted_image, flow_config as config,
    flow_rig as rig,
};

#[test]
fn the_first_frame_detects_on_camera_zero_and_matches_into_camera_one() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    let frame: &FlowFrame = flow
        .process_frame(1_000, &images, &PosePrediction::default(), &[])
        .unwrap();

    assert_eq!(frame.t_ns, Some(1_000));
    assert_eq!(frame.cameras.len(), 2);
    assert!(!frame.cameras[0].is_empty(), "camera 0 detected nothing");
    assert!(
        !frame.cameras[1].is_empty(),
        "nothing matched into camera 1 out of {} on camera 0",
        frame.cameras[0].len()
    );
    // Every keypoint of camera 1 that came from the match shares camera 0's id.
    let shared: usize = frame.cameras[1]
        .ids
        .iter()
        .filter(|id| frame.cameras[0].get(**id).is_some())
        .count();
    assert!(shared > 0, "camera 1 shares no id with camera 0");
}

/// `last_keypoint_id` is the global landmark id space (
/// ): ids are handed out in detection order and never reused.
#[test]
fn keypoint_ids_are_one_monotonic_space() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let after_first: u64 = flow.last_keypoint_id();
    assert!(after_first > 0);
    for camera in &flow.frame().cameras {
        assert!(camera.ids.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(camera.ids.iter().all(|id| id.0 < after_first));
    }

    let moved: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];
    flow.process_frame(1, &moved, &PosePrediction::default(), &[])
        .unwrap();
    assert!(flow.last_keypoint_id() >= after_first);
    assert_eq!(flow.frame_counter(), 2);
}

/// The watermark says which of the committed frame's keypoints are new, and
/// it belongs to the committed frame: a refused frameset must not move it,
/// or the frame it still describes would read as all-old keypoints.
#[test]
fn the_keypoint_watermark_describes_the_committed_frame() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    assert_eq!(flow.last_keypoint_id_before_frame(), 0);

    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    // Everything the first frameset detected is new, so the watermark is
    // still the empty id space it started from.
    assert_eq!(flow.last_keypoint_id_before_frame(), 0);
    let after_first: u64 = flow.last_keypoint_id();

    let moved: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];
    flow.process_frame(1, &moved, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(flow.last_keypoint_id_before_frame(), after_first);

    // A frameset that does not move the clock forward is refused whole.
    assert!(
        flow.process_frame(1, &moved, &PosePrediction::default(), &[])
            .is_err()
    );
    assert_eq!(flow.last_keypoint_id_before_frame(), after_first);
}

/// `updateCellCounts` / `addKeypoint` / `removeKeypoint` keep
/// `cells` equal to the number of keypoints in each grid cell — except where
/// `addKeypoints` deliberately double-counts, which cannot happen on camera 0.
#[test]
fn camera_zero_cell_counts_match_its_keypoints() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();

    let grid: CellGrid = flow.occupancy_grid();
    let mut expected: Vec<i32> = vec![0; grid.rows * grid.columns];
    for index in 0..flow.frame().cameras[0].len() {
        let translation = flow.frame().cameras[0].transforms.translation(index);
        let (row, column) = grid.cell_of(translation.x, translation.y);
        expected[row * grid.columns + column] += 1;
    }
    assert_eq!(flow.cell_counts(0), &expected[..]);
    // One point per cell is the shipped budget, so no cell may exceed it.
    assert!(flow.cell_counts(0).iter().all(|count| *count <= 1));
}

#[test]
fn tracking_carries_keypoints_across_a_shifted_frame() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let first: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    let before: Vec<KeypointId> = flow.frame().cameras[0].ids.clone();

    let second: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];
    flow.process_frame(1, &second, &PosePrediction::default(), &[])
        .unwrap();
    let survived: usize = flow.frame().cameras[0]
        .ids
        .iter()
        .filter(|id| before.contains(id))
        .count();
    assert!(
        survived * 2 >= before.len(),
        "only {survived} of {} keypoints survived a one-pixel shift",
        before.len()
    );
    // A surviving keypoint moved by about one pixel, and no further.
    for id in flow.frame().cameras[0].ids.clone() {
        if !before.contains(&id) {
            continue;
        }
        let moved: AffineCompact2f = flow.frame().cameras[0].get(id).unwrap();
        assert!(moved.translation.x.is_finite());
    }
}

#[test]
fn one_thread_and_four_threads_produce_the_same_frame() {
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    let moved: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];

    let mut single: FrameToFrameOpticalFlow<Pattern51> = frontend(
        2,
        FrontendOptions {
            threads: 1,
            ..FrontendOptions::default()
        },
    );
    let mut wide: FrameToFrameOpticalFlow<Pattern51> = frontend(
        2,
        FrontendOptions {
            threads: 4,
            ..FrontendOptions::default()
        },
    );
    for (t_ns, frameset) in [(0i64, &images), (1, &moved)] {
        single
            .process_frame(t_ns, frameset, &PosePrediction::default(), &[])
            .unwrap();
        wide.process_frame(t_ns, frameset, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(single.frame(), wide.frame());
    }
    assert_eq!(single.last_keypoint_id(), wide.last_keypoint_id());
}

#[test]
fn four_cpu_cameras_produce_the_same_frames_and_ids_at_one_and_four_threads() {
    let images: [Image<u16, 1>; 4] = std::array::from_fn(|camera| {
        if camera == 0 {
            slam_rs::image::zeros(WIDTH, HEIGHT).unwrap()
        } else {
            dotted_image(0)
        }
    });
    let moved: [Image<u16, 1>; 4] = std::array::from_fn(|camera| {
        if camera == 0 {
            slam_rs::image::zeros(WIDTH, HEIGHT).unwrap()
        } else {
            dotted_image(1)
        }
    });

    let mut calibration = rig(4);
    for (camera, pose) in calibration.t_i_c.iter_mut().enumerate() {
        pose.translation.x = camera as f64 * 4.0;
    }
    let make = |threads| {
        FrameToFrameOpticalFlow::new(
            VioConfig {
                optical_flow_detection_nonoverlap: true,
                ..config()
            },
            &calibration,
            FrontendOptions {
                threads,
                ..FrontendOptions::default()
            },
        )
        .unwrap()
    };
    let mut single: FrameToFrameOpticalFlow<Pattern51> = make(1);
    let mut wide: FrameToFrameOpticalFlow<Pattern51> = make(4);
    for (t_ns, frameset) in [(0i64, &images), (1, &moved)] {
        single
            .process_frame(t_ns, frameset, &PosePrediction::default(), &[])
            .unwrap();
        wide.process_frame(t_ns, frameset, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(single.frame(), wide.frame());
        for camera in &single.frame().cameras[1..] {
            assert!(
                camera
                    .ids
                    .iter()
                    .any(|id| single.frame().cameras[0].get(*id).is_none()),
                "the side camera must detect points of its own"
            );
        }
    }
    assert_eq!(single.last_keypoint_id(), wide.last_keypoint_id());
}

#[test]
fn two_runs_of_the_same_input_produce_the_same_frame() {
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    let moved: [Image<u16, 1>; 2] = [dotted_image(1), dotted_image(1)];
    let mut frames: Vec<FlowFrame> = Vec::new();
    for _ in 0..2 {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        flow.process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap();
        flow.process_frame(1, &moved, &PosePrediction::default(), &[])
            .unwrap();
        frames.push(flow.frame().clone());
    }
    assert_eq!(frames[0], frames[1]);
}

/// The serial adapter takes PatchTracker's default submission path and rebuilds
/// every source template. It is independent of the CPU batch/cache path.
#[test]
fn cached_batches_match_rebuilding_through_masks_losses_and_redetection() {
    for threads in [1, 4] {
        let mut rebuilt = common::flow::failing_frontend(usize::MAX);
        let mut cached: FrameToFrameOpticalFlow<Pattern51> = frontend(
            2,
            FrontendOptions {
                threads,
                ..FrontendOptions::default()
            },
        );
        for step in 0..30 {
            let images = [
                dotted_image(step % 5),
                if step % 7 == 3 {
                    slam_rs::image::zeros(WIDTH, HEIGHT).unwrap()
                } else {
                    dotted_image((step + 1) % 5)
                },
            ];
            let masks = [
                Masks::default(),
                Masks {
                    masks: if step % 4 == 1 {
                        vec![MaskRect {
                            x: 20.0,
                            y: 20.0,
                            w: 70.0,
                            h: 100.0,
                        }]
                    } else {
                        Vec::new()
                    },
                },
            ];
            let prediction = PosePrediction {
                t_w_i_current: Se3::new(
                    So3::exp(&Vector3::new(0.0, 0.001, 0.0)),
                    Vector3::new(0.001, 0.0, 0.0),
                ),
                ..PosePrediction::default()
            };
            rebuilt
                .process_frame(i64::from(step), &images, &prediction, &masks)
                .unwrap();
            cached
                .process_frame(i64::from(step), &images, &prediction, &masks)
                .unwrap();
            assert_eq!(
                cached.frame(),
                rebuilt.frame(),
                "frame {step}, threads {threads}"
            );
            assert_eq!(cached.last_keypoint_id(), rebuilt.last_keypoint_id());
            for camera in 0..2 {
                assert_eq!(cached.cell_counts(camera), rebuilt.cell_counts(camera));
            }
        }
    }
}

/// Single-camera rigs skip stereo passes (trap 17).
#[test]
fn a_single_camera_rig_detects_and_skips_matching() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(1, FrontendOptions::default());
    let images: [Image<u16, 1>; 1] = [dotted_image(0)];
    let frame: &FlowFrame = flow
        .process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(frame.cameras.len(), 1);
    assert!(!frame.cameras[0].is_empty());
}

/// Each camera's epipolar constraint comes from its own pose.
#[test]
fn the_essential_matrix_is_per_camera() {
    let translated = frontend(3, FrontendOptions::default());
    assert_eq!(translated.essential(0), Matrix4::zeros());
    // Translation is normalized, so collinear baselines have the same constraint.
    assert_eq!(translated.essential(1), translated.essential(2));
    let mut rotated = rig(3);
    rotated.t_i_c[2] = Se3::new(
        So3::exp(&Vector3::new(0.0, 0.4, 0.0)),
        Vector3::new(0.10, 0.0, 0.0),
    );
    let flow =
        FrameToFrameOpticalFlow::<Pattern51>::new(config(), &rotated, FrontendOptions::default())
            .unwrap();
    assert_eq!(flow.essential(1), translated.essential(1));
    assert_ne!(flow.essential(2), flow.essential(1));
}

/// An optional clock makes negative timestamps ordinary values.
/// Two identical frames at negative times must track instead of resetting ids.
#[test]
fn identical_frames_at_negative_timestamps_keep_their_ids() {
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    let mut shared_per_start: Vec<usize> = Vec::new();
    for start in [-2_000_000_000i64, -2, 0] {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        let first: Vec<KeypointId> = flow
            .process_frame(start, &images, &PosePrediction::default(), &[])
            .unwrap()
            .cameras[0]
            .ids
            .clone();
        assert!(!first.is_empty());
        assert_eq!(flow.t_ns(), Some(start));
        let second: &FlowFrame = flow
            .process_frame(start + 1, &images, &PosePrediction::default(), &[])
            .unwrap();
        let shared: usize = second.cameras[0]
            .ids
            .iter()
            .filter(|id| first.contains(id))
            .count();
        assert!(
            shared > 0,
            "no id survived an identical frameset at t = {start}"
        );
        shared_per_start.push(shared);
    }
    // Identical input, so the negative starts keep exactly what t = 0 keeps.
    assert!(
        shared_per_start.windows(2).all(|pair| pair[0] == pair[1]),
        "negative timestamps tracked differently from zero: {shared_per_start:?}"
    );
}

/// a mask covering the frame leaves nothing to detect.
#[test]
fn a_mask_over_the_whole_frame_suppresses_detection() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [Image<u16, 1>; 2] = [dotted_image(0), dotted_image(0)];
    let masks: Vec<Masks> = vec![
        Masks {
            masks: vec![MaskRect {
                x: 0.0,
                y: 0.0,
                w: WIDTH as f32,
                h: HEIGHT as f32,
            }],
        };
        2
    ];
    let frame: &FlowFrame = flow
        .process_frame(0, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(frame.cameras[0].is_empty());
    assert!(frame.cameras[1].is_empty());
}

/// The keypoint budget is enforced where the keypoints are created, so a
/// frame is never produced that the tracker cannot then carry.
#[test]
fn the_keypoint_budget_is_never_exceeded() {
    for max_keypoints in [1usize, 3, 7] {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(
            2,
            FrontendOptions {
                max_keypoints,
                ..FrontendOptions::default()
            },
        );
        for step in 0..3 {
            let images: [Image<u16, 1>; 2] = [dotted_image(step), dotted_image(step)];
            let frame: &FlowFrame = flow
                .process_frame(step.into(), &images, &PosePrediction::default(), &[])
                .unwrap();
            for camera in &frame.cameras {
                assert!(
                    camera.len() <= max_keypoints,
                    "{} keypoints against a budget of {max_keypoints}",
                    camera.len()
                );
            }
        }
        assert!(!flow.frame().cameras[0].is_empty());
    }
}

/// Every camera is detected on **its own** grid,
/// even though the occupancy matrix keeps camera 0's shape.
///
/// The review's probe: with 50-pixel cells a 200x200 camera starts at 0 and a
/// 240x240 camera at 20, and those two grids disagree about a corner near
/// (210, 80) — camera 0's grid stops at 150 + 50 = 200, so the port would
/// have missed it entirely if it had imposed camera 0's geometry.
#[test]
fn each_camera_is_detected_on_its_own_grid() {
    let mut mixed: Calibration<f64> = rig(2);
    mixed.resolution[1] = [240, 240];
    let flow: FrameToFrameOpticalFlow<Pattern51> =
        FrameToFrameOpticalFlow::new(config(), &mixed, FrontendOptions::default()).unwrap();

    assert_eq!(flow.detection_grid(0).x_start, 0);
    assert_eq!(flow.detection_grid(1).x_start, 20);
    // The occupancy matrix follows camera 0, which is what `cells` is shaped
    // from; both grids happen to need the same number of columns here.
    assert_eq!(flow.occupancy_grid(), flow.detection_grid(0));
    assert_eq!(flow.detection_grid(1).rows, flow.occupancy_grid().rows);

    assert!(flow.detection_grid(1).contains(210.0, 80.0));
    assert!(!flow.detection_grid(0).contains(210.0, 80.0));

    // A real grid is far under `MAX_CELLS`: 200x200 on 50-pixel cells is 5x5.
    assert_eq!(
        flow.occupancy_grid().rows * flow.occupancy_grid().columns,
        25
    );
}

/// A mixed-resolution rig still runs end to end, and camera 1's keypoints
/// come from its own image.
#[test]
fn a_mixed_resolution_rig_runs() {
    let mut mixed: Calibration<f64> = rig(2);
    mixed.resolution[1] = [240, 240];
    let mut flow: FrameToFrameOpticalFlow<Pattern51> =
        FrameToFrameOpticalFlow::new(config(), &mixed, FrontendOptions::default()).unwrap();

    let wide: Image<u16, 1> = {
        let mut image: Image<u16, 1> = slam_rs::image::zeros(240, 240).unwrap();
        let source: Image<u16, 1> = dotted_image(0);
        for y in 0..240 {
            for x in 0..240 {
                let value: u16 = source
                    .get_pixel(x % WIDTH, y % HEIGHT, 0)
                    .copied()
                    .ok()
                    .unwrap_or(0);
                image.set_pixel(x, y, 0, value).unwrap();
            }
        }
        image
    };
    let images: [Image<u16, 1>; 2] = [dotted_image(0), wide];
    let frame: &FlowFrame = flow
        .process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert!(!frame.cameras[0].is_empty());
    for index in 0..frame.cameras[1].len() {
        let position = frame.cameras[1].transforms.translation(index);
        assert!(position.x >= 0.0 && position.x < 240.0);
        assert!(position.y >= 0.0 && position.y < 240.0);
    }
}

/// A scanner whose empty result differs from the textured images' CPU corners.
#[derive(Debug, Clone)]
struct EmptyScan {
    independent: bool,
    cameras: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
}

impl CornerScan for EmptyScan {
    type Error = DetectError;
    fn fork(&self) -> Option<Box<dyn CornerScan<Error = DetectError>>> {
        self.independent
            .then(|| Box::new(self.clone()) as Box<dyn CornerScan<Error = DetectError>>)
    }

    fn scan(&mut self, camera: usize, _image: &Image<u16, 1>) -> Result<(), DetectError> {
        self.cameras.lock().unwrap().push(camera);
        Ok(())
    }

    fn band(&mut self, _request: BandRequest) -> Result<&[FastCorner], DetectError> {
        Ok(&[])
    }
}

#[test]
fn four_cameras_use_the_selected_scanner_at_one_and_four_threads() {
    let mut expected = None;
    for (threads, independent) in [(1, false), (4, false), (1, true), (4, true)] {
        let mut calibration = rig(4);
        for (camera, pose) in calibration.t_i_c.iter_mut().enumerate() {
            // Side cameras have unmasked cells outside camera 0's view.
            pose.translation.x = camera as f64 * 4.0;
        }
        let config = VioConfig {
            optical_flow_detection_nonoverlap: true,
            ..config()
        };
        let options = FrontendOptions {
            threads,
            ..FrontendOptions::default()
        };
        let pool = WorkPool::new(threads).unwrap();
        let tracker = CpuPatchTracker::<Pattern51>::new(
            options.max_keypoints,
            config.optical_flow_levels as usize + 1,
            config.optical_flow_max_iterations as usize,
            config.optical_flow_max_recovered_dist2,
            pool.clone(),
        )
        .unwrap();
        let cameras = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut flow = FrameToFrameOpticalFlow::<Pattern51, _>::with_stages(
            config,
            &calibration,
            options,
            slam_rs::frontend::stages::CpuStages::new(
                CpuPyramidBuilder::new(),
                tracker,
                kornia_staging_imgproc::features::DetectorScratch::with_scanner(Box::new(
                    EmptyScan {
                        independent,
                        cameras: cameras.clone(),
                    },
                )),
            )
            .unwrap(),
            pool,
        )
        .unwrap();
        let mut frames = Vec::new();
        for shift in 0..2 {
            let images = std::array::from_fn::<_, 4, _>(|_| dotted_image(shift));
            flow.process_frame(i64::from(shift), &images, &PosePrediction::default(), &[])
                .unwrap();
            frames.push(flow.frame().clone());
        }
        let mut calls = cameras.lock().unwrap().clone();
        for frameset in calls.chunks_mut(4) {
            frameset.sort_unstable();
        }
        assert_eq!(calls, vec![0, 1, 2, 3, 0, 1, 2, 3]);
        assert!(
            frames
                .iter()
                .all(|frame| frame.cameras.iter().all(|camera| camera.is_empty()))
        );
        let actual = (frames, flow.last_keypoint_id());
        match &expected {
            None => expected = Some(actual),
            Some(expected) => assert_eq!(&actual, expected),
        }
    }
}
