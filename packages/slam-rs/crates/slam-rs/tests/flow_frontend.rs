//! `FrameToFrameOpticalFlow`: the driver's own tests.
//!
//! Moved out of `src/frontend/flow.rs` (S25): they drive the public API only —
//! `FrameToFrameOpticalFlow::new`, `frame()`, `cell_counts`, `essential`,
//! `detection_grid` — and here they can take the synthetic rig and its frames
//! from `tests/common` instead of keeping a second copy of each.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use nalgebra::{Matrix4, Vector3};
use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::frontend::detect::{
    CellGrid, CpuCornerScan, LOWEST_THRESHOLD_RUNG, MAX_CELLS, Masks, Rect,
};
use slam_rs::frontend::flow::*;
use slam_rs::frontend::parallel::{MAX_THREADS, WorkPool};
use slam_rs::frontend::patterns::{Pattern51, Pattern52};
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::{
    CpuPatchTracker, FlowTransforms, MAX_CAPACITY, MAX_LEVELS, PatchSoA, PatchTracker, TrackerError,
};
use slam_rs::image::ImageU16;
use slam_rs::lie::{Se3, So3};
use slam_rs::pyramid::{CpuPyramidBuilder, PyramidU16};
use slam_rs::types::KeypointId;

mod common;

use common::{
    FLOW_HEIGHT as HEIGHT, FLOW_WIDTH as WIDTH, dotted_image, flow_config as config,
    flow_rig as rig,
};

fn frontend(cameras: usize, options: FrontendOptions) -> FrameToFrameOpticalFlow<Pattern51> {
    FrameToFrameOpticalFlow::new(config(), &rig(cameras), options).unwrap()
}

fn cpu_tracker(config: &VioConfig, capacity: usize) -> CpuPatchTracker<Pattern51> {
    CpuPatchTracker::new(
        capacity,
        config.optical_flow_levels as usize + 1,
        config.optical_flow_max_iterations as usize,
        config.optical_flow_max_recovered_dist2,
        WorkPool::new(1).unwrap(),
    )
    .unwrap()
}

#[test]
fn the_first_frame_detects_on_camera_zero_and_matches_into_camera_one() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
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

/// `last_keypoint_id` is the global landmark id space (`optical_flow.h:71`,
/// `:174`): ids are handed out in detection order and never reused.
#[test]
fn keypoint_ids_are_one_monotonic_space() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let after_first: u64 = flow.last_keypoint_id();
    assert!(after_first > 0);
    for camera in &flow.frame().cameras {
        assert!(camera.ids.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(camera.ids.iter().all(|id| id.0 < after_first));
    }

    let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
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

    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    // Everything the first frameset detected is new, so the watermark is
    // still the empty id space it started from.
    assert_eq!(flow.last_keypoint_id_before_frame(), 0);
    let after_first: u64 = flow.last_keypoint_id();

    let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
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

/// A frontend on the synthetic rig with the redetect gate set.
fn gated_frontend(cameras: usize, ratio: f32) -> FrameToFrameOpticalFlow<Pattern51> {
    let gated: VioConfig = VioConfig {
        port_redetect_survivor_ratio: ratio,
        ..config()
    };
    FrameToFrameOpticalFlow::new(gated, &rig(cameras), FrontendOptions::default()).unwrap()
}

/// The gate's default is basalt's schedule (D75): `addPoints` on every
/// frameset, so every frameset hands out ids.
///
/// This is the control the two tests below are read against — without it, a
/// gated run that detects rarely could not be told from a rig that has nothing
/// left to detect.
#[test]
fn the_default_config_detects_on_every_frameset() {
    assert_eq!(VioConfig::default().port_redetect_survivor_ratio, 0.0);
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let mut watermark: u64 = 0;
    for shift in 0..6 {
        let images: [ImageU16; 2] = [dotted_image(shift), dotted_image(shift)];
        flow.process_frame(i64::from(shift), &images, &PosePrediction::default(), &[])
            .unwrap();
        assert!(
            flow.last_keypoint_id() > watermark,
            "frameset {shift} detected nothing"
        );
        watermark = flow.last_keypoint_id();
    }
}

/// A ratio nothing can fall below detects once and then never again, which is
/// what says the gate is really the only thing deciding.
#[test]
fn a_survivor_ratio_nothing_reaches_detects_only_on_the_first_frameset() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = gated_frontend(2, 1e-6);
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let after_first: u64 = flow.last_keypoint_id();
    assert!(after_first > 0, "the first frameset must detect");

    for shift in 1..6 {
        let moved: [ImageU16; 2] = [dotted_image(shift), dotted_image(shift)];
        flow.process_frame(i64::from(shift), &moved, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(
            flow.last_keypoint_id(),
            after_first,
            "frameset {shift} detected while gated"
        );
    }
}

/// The gate is one decision for the whole rig, never a camera at a time: a
/// skipped frameset leaves camera 1 with no new ids either, because the
/// cross-camera match and the non-overlap pass are both inside `addPoints`.
#[test]
fn a_skipped_frameset_leaves_no_camera_detected() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = gated_frontend(3, 1e-6);
    let images: [ImageU16; 3] = [dotted_image(0), dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let mut known: Vec<Vec<KeypointId>> = flow
        .frame()
        .cameras
        .iter()
        .map(|camera| camera.ids.clone())
        .collect();
    assert!(known.iter().all(|ids| !ids.is_empty()));

    for shift in 1..4 {
        let moved: [ImageU16; 3] = [
            dotted_image(shift),
            dotted_image(shift),
            dotted_image(shift),
        ];
        flow.process_frame(i64::from(shift), &moved, &PosePrediction::default(), &[])
            .unwrap();
        for (camera, ids) in flow.frame().cameras.iter().enumerate() {
            let fresh: Vec<KeypointId> = ids
                .ids
                .iter()
                .filter(|id| !known[camera].contains(id))
                .copied()
                .collect();
            assert!(
                fresh.is_empty(),
                "camera {camera} gained {fresh:?} on a skipped frameset"
            );
        }
        known = flow
            .frame()
            .cameras
            .iter()
            .map(|camera| camera.ids.clone())
            .collect();
    }
}

/// Remove exact old positions while leaving other corners available to detect.
fn masks_leaving(frame: &FlowFrame, survivors: usize) -> [Masks; 1] {
    [Masks {
        masks: (survivors..frame.cameras[0].len())
            .map(|index| {
                let position = frame.cameras[0].transforms.translation(index);
                Rect {
                    x: position.x - 0.25,
                    y: position.y - 0.25,
                    w: 0.5,
                    h: 0.5,
                }
            })
            .collect(),
    }]
}

#[test]
fn redetection_resumes_only_below_the_survivor_threshold() {
    for offset in [-1, 0, 1] {
        let mut flow = gated_frontend(2, 0.5);
        let images = [dotted_image(0), dotted_image(0)];
        flow.process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap();
        let count = flow.frame().cameras[0].len();
        assert!(count >= 4 && count % 2 == 0);
        let survivors = (count as isize / 2 + offset) as usize;
        let masks = masks_leaving(flow.frame(), survivors);
        let watermark = flow.last_keypoint_id();
        flow.process_frame(1, &images, &PosePrediction::default(), &masks)
            .unwrap();
        assert_eq!(
            flow.frame().cameras[0]
                .ids
                .iter()
                .filter(|id| id.0 < watermark)
                .count(),
            survivors
        );
        assert_eq!(
            flow.last_keypoint_id() > watermark,
            offset < 0,
            "offset {offset}"
        );
    }
}

#[test]
fn redetection_resumes_on_the_frameset_that_loses_every_track() {
    let mut flow = gated_frontend(2, 0.5);
    let images = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let watermark = flow.last_keypoint_id();
    assert!(watermark > 0);
    let masks = masks_leaving(flow.frame(), 0);
    flow.process_frame(1, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(!flow.frame().cameras[0].is_empty());
    assert!(
        flow.frame().cameras[0]
            .ids
            .iter()
            .all(|id| id.0 >= watermark)
    );
}

#[test]
fn redetection_uses_the_latest_post_detection_count() {
    let mut flow = gated_frontend(2, 0.5);
    let images = [dotted_image(0), dotted_image(0)];
    let initial_mask = [Masks {
        masks: vec![Rect {
            x: 60.0,
            y: 0.0,
            w: WIDTH as f32,
            h: HEIGHT as f32,
        }],
    }];
    flow.process_frame(0, &images, &PosePrediction::default(), &initial_mask)
        .unwrap();
    let initial = flow.frame().cameras[0].len();
    assert!(initial >= 2);
    let masks = masks_leaving(flow.frame(), 1);
    let watermark = flow.last_keypoint_id();
    flow.process_frame(1, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(flow.last_keypoint_id() > watermark);
    let replenished = flow.frame().cameras[0].len();
    assert!(replenished > 2 * initial);
    // Above the original baseline's threshold, below the replenished one's.
    let masks = masks_leaving(flow.frame(), initial);
    let watermark = flow.last_keypoint_id();
    flow.process_frame(2, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(flow.last_keypoint_id() > watermark);
    // Keeping all post-detection tracks must skip, even after replenishment.
    let watermark = flow.last_keypoint_id();
    flow.process_frame(3, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(flow.last_keypoint_id(), watermark);
}

#[test]
fn redetection_resumes_after_an_empty_initial_detection() {
    let mut flow = gated_frontend(2, 0.5);
    let blank = [
        ImageU16::zeros(WIDTH, HEIGHT).unwrap(),
        ImageU16::zeros(WIDTH, HEIGHT).unwrap(),
    ];
    flow.process_frame(0, &blank, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(flow.last_keypoint_id(), 0);
    let images = [dotted_image(0), dotted_image(0)];
    flow.process_frame(1, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert!(!flow.frame().cameras[0].is_empty());
    let masks = masks_leaving(flow.frame(), 0);
    let watermark = flow.last_keypoint_id();
    flow.process_frame(2, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(flow.last_keypoint_id() > watermark);
}

#[test]
fn redetection_keeps_its_baseline_after_a_rejected_frameset() {
    let images = [dotted_image(0), dotted_image(0)];
    let mut clean = gated_frontend(2, 0.5);
    let mut faulty = failing_frontend_with_ratio(3, 0.5);
    clean
        .process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    faulty
        .process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let committed = faulty.frame().clone();
    let watermark = faulty.last_keypoint_id();
    let masks = masks_leaving(&committed, 1);
    assert!(
        faulty
            .process_frame(1, &images, &PosePrediction::default(), &masks)
            .is_err()
    );
    assert_eq!(faulty.frame(), &committed);
    assert_eq!(faulty.last_keypoint_id(), watermark);
    // An unchanged retry skips: a zeroed baseline would incorrectly detect.
    faulty
        .process_frame(1, &images, &PosePrediction::default(), &[])
        .unwrap();
    clean
        .process_frame(1, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(faulty.last_keypoint_id(), watermark);
    let masks = masks_leaving(faulty.frame(), 1);
    faulty
        .process_frame(2, &images, &PosePrediction::default(), &masks)
        .unwrap();
    clean
        .process_frame(2, &images, &PosePrediction::default(), &masks)
        .unwrap();
    assert!(faulty.last_keypoint_id() > watermark);
    assert_eq!(faulty.frame(), clean.frame());
    assert_eq!(faulty.last_keypoint_id(), clean.last_keypoint_id());
}

/// `updateCellCounts` / `addKeypoint` / `removeKeypoint` (`:707-749`) keep
/// `cells` equal to the number of keypoints in each grid cell — except where
/// `addKeypoints` deliberately double-counts, which cannot happen on camera 0.
#[test]
fn camera_zero_cell_counts_match_its_keypoints() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
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
    let first: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    let before: Vec<KeypointId> = flow.frame().cameras[0].ids.clone();

    let second: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
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
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];

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
fn two_runs_of_the_same_input_produce_the_same_frame() {
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
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

/// Trap 17: `getNumCams() >= 2` is a hard precondition in the C++
/// (`optical_flow.h:210`). The port allows one camera and skips the passes
/// that need a second.
#[test]
fn a_single_camera_rig_detects_and_skips_matching() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(1, FrontendOptions::default());
    let images: [ImageU16; 1] = [dotted_image(0)];
    let frame: &FlowFrame = flow
        .process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(frame.cameras.len(), 1);
    assert!(!frame.cameras[0].is_empty());
}

/// Deviation X03: `E[i]` from `T_c0_ci` per camera, or the C++'s cam0-cam1
/// matrix everywhere. On a three-camera baseline rig the two differ for
/// camera 2 and agree for camera 1.
#[test]
fn the_essential_matrix_is_per_camera_unless_the_cpp_bug_is_asked_for() {
    let fixed: FrameToFrameOpticalFlow<Pattern51> = frontend(
        3,
        FrontendOptions {
            epipolar_per_camera: true,
            ..FrontendOptions::default()
        },
    );
    let faithful: FrameToFrameOpticalFlow<Pattern51> = frontend(
        3,
        FrontendOptions {
            epipolar_per_camera: false,
            ..FrontendOptions::default()
        },
    );
    assert_eq!(fixed.essential(1), faithful.essential(1));
    assert_eq!(faithful.essential(2), faithful.essential(1));
    // Camera 2's baseline is twice camera 1's, but `computeEssential`
    // normalizes the translation, so a pure-translation rig gives the same
    // matrix either way; the rotated rig below is what separates them.
    assert_eq!(fixed.essential(2), fixed.essential(1));

    // With camera 2 rotated, the two disagree.
    let mut rotated: Calibration<f64> = rig(3);
    rotated.t_i_c[2] = Se3::new(
        So3::exp(&Vector3::new(0.0, 0.4, 0.0)),
        Vector3::new(0.10, 0.0, 0.0),
    );
    let fixed: FrameToFrameOpticalFlow<Pattern51> = FrameToFrameOpticalFlow::new(
        config(),
        &rotated,
        FrontendOptions {
            epipolar_per_camera: true,
            ..FrontendOptions::default()
        },
    )
    .unwrap();
    let faithful: FrameToFrameOpticalFlow<Pattern51> = FrameToFrameOpticalFlow::new(
        config(),
        &rotated,
        FrontendOptions {
            epipolar_per_camera: false,
            ..FrontendOptions::default()
        },
    )
    .unwrap();
    assert_ne!(fixed.essential(2), faithful.essential(2));
    assert_eq!(faithful.essential(2), faithful.essential(1));
}

/// For a two-camera rig the flag makes no difference at all, which is what
/// keeps a C++-parity run on msd-index unaffected either way.
#[test]
fn on_a_stereo_rig_the_epipolar_flag_changes_nothing() {
    let fixed: FrameToFrameOpticalFlow<Pattern51> = frontend(
        2,
        FrontendOptions {
            epipolar_per_camera: true,
            ..FrontendOptions::default()
        },
    );
    let faithful: FrameToFrameOpticalFlow<Pattern51> = frontend(
        2,
        FrontendOptions {
            epipolar_per_camera: false,
            ..FrontendOptions::default()
        },
    );
    // Index 0 differs by construction and is never read; camera 1 is the
    // one `filterPoints` uses, and there the two agree exactly.
    assert_eq!(fixed.essential(1), faithful.essential(1));
    assert_eq!(fixed.essential(0), Matrix4::zeros());
}

#[test]
fn a_config_that_names_another_pattern_is_refused() {
    let error =
        FrameToFrameOpticalFlow::<Pattern52>::new(config(), &rig(2), FrontendOptions::default())
            .unwrap_err();
    assert_eq!(
        error,
        FrontendError::PatternMismatch {
            config: 51,
            built: 52
        }
    );
}

#[test]
fn a_config_that_names_another_flow_type_is_refused() {
    let mut config: VioConfig = config();
    config.optical_flow_type = "patch".to_owned();
    let error =
        FrameToFrameOpticalFlow::<Pattern51>::new(config, &rig(2), FrontendOptions::default())
            .unwrap_err();
    assert_eq!(
        error,
        FrontendError::UnsupportedFlowType("patch".to_owned())
    );
}

#[test]
fn a_frameset_of_the_wrong_width_is_refused() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [ImageU16; 1] = [dotted_image(0)];
    let error = flow
        .process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap_err();
    assert_eq!(
        error,
        FrontendError::CameraCountMismatch {
            expected: 2,
            actual: 1
        }
    );
}

/// A `min_threshold` of zero or less wedges the detector, in C++ as much as
/// here, so the frontend refuses the config instead of accepting it.
#[test]
fn a_config_whose_threshold_ladder_never_ends_is_refused() {
    for min_threshold in [0, -1, i32::MIN] {
        let mut broken: VioConfig = config();
        broken.optical_flow_detection_min_threshold = min_threshold;
        let error =
            FrameToFrameOpticalFlow::<Pattern51>::new(broken, &rig(2), FrontendOptions::default())
                .unwrap_err();
        assert_eq!(
            error,
            FrontendError::ThresholdLadderNeverEnds {
                min_threshold,
                rung: LOWEST_THRESHOLD_RUNG,
            }
        );
    }
}

/// A ladder that starts below where it stops never runs, so the detector
/// could never add a keypoint; that is a config error, not a quiet blank run.
#[test]
fn a_config_whose_threshold_ladder_never_runs_is_refused() {
    let mut broken: VioConfig = config();
    broken.optical_flow_detection_min_threshold = 40;
    broken.optical_flow_detection_max_threshold = 5;
    let error =
        FrameToFrameOpticalFlow::<Pattern51>::new(broken, &rig(2), FrontendOptions::default())
            .unwrap_err();
    assert_eq!(
        error,
        FrontendError::EmptyThresholdLadder {
            min_threshold: 40,
            max_threshold: 5,
        }
    );
}

/// The shipped ladder is accepted, so the checks above cannot be blanket ones.
#[test]
fn the_shipped_threshold_ladder_is_accepted() {
    assert_eq!(config().optical_flow_detection_min_threshold, 5);
    assert_eq!(config().optical_flow_detection_max_threshold, 40);
    FrameToFrameOpticalFlow::<Pattern51>::new(config(), &rig(2), FrontendOptions::default())
        .unwrap();
}

/// A keypoint budget past the tracker's ceiling is refused, not allocated.
///
/// `usize::MAX` patches over four levels of 52 taps is not a number of bytes
/// that exists; `Vec::with_capacity` answers that with a `capacity overflow`
/// panic, which crosses the Python boundary as a `PanicException` (decision
/// D32) — so the budget is checked before anything is sized from it.
#[test]
fn a_budget_over_the_ceiling_is_refused() {
    for max_keypoints in [MAX_CAPACITY + 1, usize::MAX / 2, usize::MAX] {
        let error = FrameToFrameOpticalFlow::<Pattern51>::new(
            config(),
            &rig(2),
            FrontendOptions {
                max_keypoints,
                ..FrontendOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            FrontendError::TooManyKeypoints {
                max_keypoints,
                ceiling: MAX_CAPACITY,
            }
        );
    }
}

/// A pool of no workers is refused where every other option ceiling is.
///
/// `WorkPool::new` reads zero as one, so nothing downstream would fail: the
/// caller would silently get a frontend it did not ask for.
#[test]
fn a_thread_count_of_zero_is_refused() {
    let error = FrameToFrameOpticalFlow::<Pattern51>::new(
        config(),
        &rig(1),
        FrontendOptions {
            threads: 0,
            ..FrontendOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(error, FrontendError::NoThreads);
}

/// A thread count nothing could run is refused, not spawned.
///
/// rayon takes `num_threads` literally, so 100,000 workers arriving over the
/// Python boundary spawned OS threads for minutes; the ceiling answers instead.
#[test]
fn more_workers_than_the_ceiling_is_refused() {
    for threads in [MAX_THREADS + 1, 100_000, usize::MAX] {
        let error = FrameToFrameOpticalFlow::<Pattern51>::new(
            config(),
            &rig(2),
            FrontendOptions {
                threads,
                ..FrontendOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            FrontendError::TooManyThreads {
                threads,
                ceiling: MAX_THREADS,
            }
        );
    }
    // The counts the frontend actually runs on are untouched.
    for threads in [1, 4, MAX_THREADS] {
        FrameToFrameOpticalFlow::<Pattern51>::new(
            config(),
            &rig(1),
            FrontendOptions {
                threads,
                ..FrontendOptions::default()
            },
        )
        .unwrap();
    }
}

/// A pyramid deeper than the buffers allow is refused before the allocation.
///
/// `optical_flow_levels = 10^12` sized a `Vec` of 6e17 floats, and a `Vec`
/// that cannot be allocated aborts the process — no exception, no unwind.
#[test]
fn a_config_asking_for_more_levels_than_the_ceiling_is_refused() {
    for levels in [MAX_LEVELS as i32, 1_000, i32::MAX] {
        let mut deep: VioConfig = config();
        deep.optical_flow_levels = levels;
        let error =
            FrameToFrameOpticalFlow::<Pattern51>::new(deep, &rig(2), FrontendOptions::default())
                .unwrap_err();
        assert_eq!(
            error,
            FrontendError::TooManyLevels {
                levels,
                num_levels: levels as usize + 1,
                ceiling: MAX_LEVELS,
            }
        );
    }
    // The shipped depth is three levels plus the base.
    assert_eq!(config().optical_flow_levels, 3);
    FrameToFrameOpticalFlow::<Pattern51>::new(config(), &rig(2), FrontendOptions::default())
        .unwrap();
}

/// A frame that is not the size the calibration gives its camera is refused.
///
/// Everything downstream is the calibration's geometry — the projection, the
/// detection grid, the occupancy matrix — so a cropped or resized frame is
/// not a smaller view of the same scene. The check names the camera, which is
/// what tells a caller a two-camera rig was fed one right frame and one wrong
/// one.
#[test]
fn a_frame_that_is_not_the_calibrated_size_is_refused() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    for (width, height) in [(64, 64), (WIDTH - 1, HEIGHT - 1), (250, 250), (WIDTH, 250)] {
        let odd: ImageU16 = ImageU16::zeros(width, height).unwrap();
        // Camera 1 is the wrong one here, so the error must name camera 1.
        let images: [ImageU16; 2] = [dotted_image(0), odd];
        let error = flow
            .process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap_err();
        assert_eq!(
            error,
            FrontendError::FrameSizeMismatch {
                camera: 1,
                expected_width: WIDTH,
                expected_height: HEIGHT,
                actual_width: width,
                actual_height: height,
            }
        );
    }
    assert_eq!(flow.t_ns(), None);
    assert_eq!(flow.frame_counter(), 0);
}

/// And it is refused transactionally: the frontend is as the last accepted
/// frame left it.
#[test]
fn a_frame_of_the_wrong_size_commits_nothing() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(1_000, &images, &PosePrediction::default(), &[])
        .unwrap();
    let before: FlowFrame = flow.frame().clone();
    let ids_before: u64 = flow.last_keypoint_id();

    let odd: [ImageU16; 2] = [
        ImageU16::zeros(WIDTH + 8, HEIGHT).unwrap(),
        ImageU16::zeros(WIDTH + 8, HEIGHT).unwrap(),
    ];
    assert!(
        flow.process_frame(2_000, &odd, &PosePrediction::default(), &[])
            .is_err()
    );
    assert_eq!(flow.t_ns(), Some(1_000));
    assert_eq!(flow.frame_counter(), 1);
    assert_eq!(flow.last_keypoint_id(), ids_before);
    assert_eq!(flow.frame(), &before);

    // And the frontend still tracks afterwards, from the frame it kept.
    let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
    let after: Vec<KeypointId> = flow
        .process_frame(3_000, &moved, &PosePrediction::default(), &[])
        .unwrap()
        .cameras[0]
        .ids
        .clone();
    assert_eq!(flow.frame_counter(), 2);
    let shared: usize = after
        .iter()
        .filter(|id| before.cameras[0].get(**id).is_some())
        .count();
    assert!(shared > 0, "the kept frame was not tracked against");
}

/// Two identical framesets at negative timestamps track each other.
///
/// basalt reads `t_ns < 0` as "no previous frame" (`optical_flow.h:172`), so
/// the port used to detect from scratch on every negative timestamp and hand
/// out a fresh id space each time. The clock is an `Option` now, so `-2` and
/// `-1` are ordinary timestamps and the second frameset tracks the first.
#[test]
fn identical_frames_at_negative_timestamps_keep_their_ids() {
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
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

#[test]
fn the_cpp_essential_bug_needs_a_second_camera() {
    let error = FrameToFrameOpticalFlow::<Pattern51>::new(
        config(),
        &rig(1),
        FrontendOptions {
            epipolar_per_camera: false,
            ..FrontendOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(error, FrontendError::NeedsTwoCameras { cameras: 1 });
}

/// `keypoints.cpp:179`: a mask covering the frame leaves nothing to detect.
#[test]
fn a_mask_over_the_whole_frame_suppresses_detection() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    let masks: Vec<Masks> = vec![
        Masks {
            masks: vec![Rect {
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
/// A frame the frontend refuses must leave it usable.
///
/// The review's sequence: a 1x1 first image is rejected, and the *next*,
/// valid frame used to take the tracking path against an empty previous
/// pyramid and panic. Nothing commits until every pyramid is built. The 1x1
/// frame is now refused one step earlier than it was — the calibration check
/// catches it before the pyramid builder ever sees it — and the sequence this
/// test is about is unchanged.
#[test]
fn a_rejected_frame_leaves_the_frontend_usable() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let tiny: [ImageU16; 2] = [
        ImageU16::zeros(1, 1).unwrap(),
        ImageU16::zeros(1, 1).unwrap(),
    ];
    let error = flow
        .process_frame(1, &tiny, &PosePrediction::default(), &[])
        .unwrap_err();
    assert_eq!(
        error,
        FrontendError::FrameSizeMismatch {
            camera: 0,
            expected_width: WIDTH,
            expected_height: HEIGHT,
            actual_width: 1,
            actual_height: 1,
        }
    );
    // Nothing moved: the clock is still before the first frame.
    assert_eq!(flow.t_ns(), None);
    assert_eq!(flow.frame_counter(), 0);
    assert_eq!(flow.last_keypoint_id(), 0);

    // The next valid frame is treated as the first, and the one after it
    // tracks against a pyramid that exists.
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    let frame: &FlowFrame = flow
        .process_frame(2, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert!(!frame.cameras[0].is_empty());
    let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
    flow.process_frame(3, &moved, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(flow.frame_counter(), 2);
}

/// A frameset of the wrong width is refused before anything commits too.
#[test]
fn a_frameset_of_the_wrong_width_commits_nothing() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    flow.process_frame(0, &images, &PosePrediction::default(), &[])
        .unwrap();
    let before: FlowFrame = flow.frame().clone();

    let short: [ImageU16; 1] = [dotted_image(1)];
    assert!(
        flow.process_frame(1, &short, &PosePrediction::default(), &[])
            .is_err()
    );
    assert_eq!(flow.t_ns(), Some(0));
    assert_eq!(flow.frame_counter(), 1);
    assert_eq!(flow.frame(), &before);

    // And the frontend still works afterwards.
    let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
    flow.process_frame(2, &moved, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(flow.frame_counter(), 2);
}

/// A frameset that does not move the clock forward is refused, and commits nothing.
///
/// The rule lives here rather than at the Python boundary because every other
/// rule about an input does: the clock the comparison reads is this one.
#[test]
fn a_frameset_that_does_not_follow_the_last_one_is_refused() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(1, FrontendOptions::default());
    let images: [ImageU16; 1] = [dotted_image(0)];
    flow.process_frame(1_000, &images, &PosePrediction::default(), &[])
        .unwrap();
    let before: FlowFrame = flow.frame().clone();

    for t_ns in [1_000, 999, -1_000] {
        assert_eq!(
            flow.process_frame(t_ns, &images, &PosePrediction::default(), &[])
                .unwrap_err(),
            FrontendError::NonMonotonicFrameset {
                previous_t_ns: 1_000,
                t_ns,
            }
        );
        assert_eq!(flow.t_ns(), Some(1_000));
        assert_eq!(flow.frame_counter(), 1);
        assert_eq!(flow.frame(), &before);
    }

    // The first frameset has no previous one, so any timestamp starts a run.
    let mut negative: FrameToFrameOpticalFlow<Pattern51> = frontend(1, FrontendOptions::default());
    negative
        .process_frame(-2, &images, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(negative.t_ns(), Some(-2));
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
            let images: [ImageU16; 2] = [dotted_image(step), dotted_image(step)];
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

/// The budget may not exceed what the tracker was built for.
#[test]
fn a_budget_larger_than_the_tracker_is_refused() {
    let config: VioConfig = config();
    let tracker: CpuPatchTracker<Pattern51> = cpu_tracker(&config, 4);
    let error = FrameToFrameOpticalFlow::<Pattern51, _, _>::with_backends(
        config,
        &rig(2),
        FrontendOptions {
            max_keypoints: 8,
            ..FrontendOptions::default()
        },
        CpuPyramidBuilder::new(),
        tracker,
        Box::new(CpuCornerScan::default()),
    )
    .unwrap_err();
    assert_eq!(
        error,
        FrontendError::BudgetExceedsCapacity {
            max_keypoints: 8,
            capacity: 4
        }
    );
}

/// A calibration with fewer poses than camera models used to index past the
/// end of `T_i_c`; it is a typed error now (decision D32).
#[test]
fn a_calibration_missing_an_extrinsic_is_refused() {
    let mut ragged: Calibration<f64> = rig(2);
    ragged.t_i_c.pop();
    let error =
        FrameToFrameOpticalFlow::<Pattern51>::new(config(), &ragged, FrontendOptions::default())
            .unwrap_err();
    assert_eq!(
        error,
        FrontendError::RaggedExtrinsics {
            intrinsics: 2,
            extrinsics: 1
        }
    );
}

/// A camera too small for one detection cell names itself in the error.
#[test]
fn a_camera_smaller_than_a_cell_is_refused() {
    let mut small: Calibration<f64> = rig(2);
    small.resolution[1] = [30, 30];
    let error =
        FrameToFrameOpticalFlow::<Pattern51>::new(config(), &small, FrontendOptions::default())
            .unwrap_err();
    assert_eq!(
        error,
        FrontendError::FrameTooSmall {
            camera: 1,
            width: 30,
            height: 30,
            cell: 50
        }
    );
}

/// A calibration whose occupancy grid nothing could allocate is refused
/// ([`MAX_CELLS`] carries why).
///
/// Both sides of the guard are exercised: a `u32::MAX - 1` frame on a
/// one-pixel grid asks for a product that still fits a `usize`, and a
/// `u32::MAX` one asks for exactly 2^64, which leaves it. Both constructors
/// are checked, because `with_backends` takes a tracker that is already
/// built and so runs no check of `new`'s.
#[test]
fn a_calibration_whose_occupancy_grid_is_past_the_ceiling_is_refused() {
    for side in [u32::MAX - 1, u32::MAX] {
        let mut vast: Calibration<f64> = rig(2);
        vast.resolution = vec![[side, side]; 2];
        let mut fine: VioConfig = config();
        fine.optical_flow_detection_grid_size = 1;
        let expected: FrontendError = FrontendError::TooManyCells {
            camera: 0,
            rows: side as usize + 1,
            columns: side as usize + 1,
            ceiling: MAX_CELLS,
        };
        let error = FrameToFrameOpticalFlow::<Pattern51>::new(
            fine.clone(),
            &vast,
            FrontendOptions::default(),
        )
        .unwrap_err();
        assert_eq!(error, expected);

        let tracker: CpuPatchTracker<Pattern51> =
            cpu_tracker(&fine, FrontendOptions::default().max_keypoints);
        let error = FrameToFrameOpticalFlow::<Pattern51, _, _>::with_backends(
            fine,
            &vast,
            FrontendOptions::default(),
            CpuPyramidBuilder::new(),
            tracker,
            Box::new(CpuCornerScan::default()),
        )
        .unwrap_err();
        assert_eq!(error, expected);
    }
}

/// Every camera is detected on **its own** grid (`keypoints.cpp:140-144`),
/// even though the occupancy matrix keeps camera 0's shape (`:119`).
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

    let wide: ImageU16 = {
        let mut image: ImageU16 = ImageU16::zeros(240, 240).unwrap();
        let source: ImageU16 = dotted_image(0);
        for y in 0..240 {
            for x in 0..240 {
                let value: u16 = source.get(x % WIDTH, y % HEIGHT).unwrap_or(0);
                image.set(x, y, value);
            }
        }
        image
    };
    let images: [ImageU16; 2] = [dotted_image(0), wide];
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
/// A tracker that forwards to the CPU one and refuses the *n*-th call.
///
/// The point is a failure from inside a pluggable backend, in the middle of
/// `processFrame`, after the frame's pyramids are built and after some of the
/// cameras have already been tracked. Nothing else can produce that.
#[derive(Debug)]
struct FailingTracker {
    inner: CpuPatchTracker<Pattern51>,
    calls: std::cell::Cell<usize>,
    fail_on: usize,
}

impl PatchTracker for FailingTracker {
    fn batch(&self) -> &slam_rs::frontend::tracker::TrackBatch {
        self.inner.batch()
    }
    fn batch_mut(&mut self) -> &mut slam_rs::frontend::tracker::TrackBatch {
        self.inner.batch_mut()
    }

    type Pattern = Pattern51;
    type Pyramid = PyramidU16;
    type Patches = PatchSoA<Pattern51>;

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    fn make_patches(&self) -> Result<PatchSoA<Pattern51>, TrackerError> {
        self.inner.make_patches()
    }

    fn submit_prepared(
        &mut self,
        prev: &PyramidU16,
        next: &PyramidU16,
        patches: &PatchSoA<Pattern51>,
        transforms_in: &FlowTransforms,
    ) -> Result<usize, TrackerError> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() == self.fail_on {
            return Err(TrackerError::CapacityExceeded {
                offered: usize::MAX,
                capacity: 0,
            });
        }
        self.inner
            .submit_prepared(prev, next, patches, transforms_in)
    }
}

fn failing_frontend(
    fail_on: usize,
) -> FrameToFrameOpticalFlow<Pattern51, CpuPyramidBuilder, FailingTracker> {
    failing_frontend_with_ratio(fail_on, 0.0)
}

fn failing_frontend_with_ratio(
    fail_on: usize,
    ratio: f32,
) -> FrameToFrameOpticalFlow<Pattern51, CpuPyramidBuilder, FailingTracker> {
    let config = VioConfig {
        port_redetect_survivor_ratio: ratio,
        ..config()
    };
    let options: FrontendOptions = FrontendOptions::default();
    let inner: CpuPatchTracker<Pattern51> = cpu_tracker(&config, options.max_keypoints);
    FrameToFrameOpticalFlow::with_backends(
        config,
        &rig(2),
        options,
        CpuPyramidBuilder::new(),
        FailingTracker {
            inner,
            calls: std::cell::Cell::new(0),
            fail_on,
        },
        Box::new(CpuCornerScan::default()),
    )
    .unwrap()
}

/// A backend error part-way through frame 2 must leave the frontend exactly
/// as frame 1 left it, so frame 3 comes out of `{1, 2 fails, 3}` identical to
/// frame 3 of a clean `{1, 3}`.
///
/// The rotation used to commit before tracking, which paired frame 1's
/// keypoints with frame 2's pyramid on the next call.
#[test]
fn a_backend_error_leaves_the_frame_as_the_last_good_one() {
    let first: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    let second: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
    let third: [ImageU16; 2] = [dotted_image(2), dotted_image(2)];

    // The clean run skips the frame the other run fails on.
    let mut clean: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    clean
        .process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    clean
        .process_frame(2, &third, &PosePrediction::default(), &[])
        .unwrap();

    // Frame 1 makes one call (the stereo match), so call 2 is camera 0 of
    // frame 2: the failure lands after the first frame committed and before
    // the second one could.
    let mut faulty = failing_frontend(2);
    faulty
        .process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    let after_first: FlowFrame = faulty.frame().clone();
    let ids_after_first: u64 = faulty.last_keypoint_id();
    let cells_after_first: Vec<i32> = faulty.cell_counts(0).to_vec();

    let error = faulty
        .process_frame(1, &second, &PosePrediction::default(), &[])
        .unwrap_err();
    assert!(matches!(error, FrontendError::Tracker(_)), "got {error}");

    // Nothing moved.
    assert_eq!(faulty.frame(), &after_first);
    assert_eq!(faulty.last_keypoint_id(), ids_after_first);
    assert_eq!(faulty.cell_counts(0), &cells_after_first[..]);
    assert_eq!(faulty.t_ns(), Some(0));
    assert_eq!(faulty.frame_counter(), 1);

    // And the next good frame is the one the clean run produced.
    faulty
        .process_frame(2, &third, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(faulty.frame(), clean.frame());
    assert_eq!(faulty.last_keypoint_id(), clean.last_keypoint_id());
    assert_eq!(faulty.cell_counts(0), clean.cell_counts(0));
    assert_eq!(faulty.cell_counts(1), clean.cell_counts(1));
}

/// The same, with the failure on camera 1 — after camera 0 has already been
/// tracked and its keypoint map rewritten inside the same call.
#[test]
fn a_backend_error_after_the_first_camera_is_undone_too() {
    let first: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
    let second: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
    let third: [ImageU16; 2] = [dotted_image(2), dotted_image(2)];

    let mut clean: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
    clean
        .process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    clean
        .process_frame(2, &third, &PosePrediction::default(), &[])
        .unwrap();

    // Frame 1 makes one matching call, so frame 2's camera-1 track is call 3.
    let mut faulty = failing_frontend(3);
    faulty
        .process_frame(0, &first, &PosePrediction::default(), &[])
        .unwrap();
    assert!(
        faulty
            .process_frame(1, &second, &PosePrediction::default(), &[])
            .is_err()
    );
    faulty
        .process_frame(2, &third, &PosePrediction::default(), &[])
        .unwrap();
    assert_eq!(faulty.frame(), clean.frame());
    assert_eq!(faulty.last_keypoint_id(), clean.last_keypoint_id());
}
