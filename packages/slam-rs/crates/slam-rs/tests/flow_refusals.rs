//! `FrameToFrameOpticalFlow`: the configurations, options, rigs and framesets
//! the frontend refuses, on the synthetic rig of `tests/common`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use kornia_image::Image;
use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::frontend::detect::{CpuCornerScan, LOWEST_THRESHOLD_RUNG, MAX_CELLS};
use slam_rs::frontend::flow::*;
use slam_rs::frontend::parallel::{MAX_THREADS, WorkPool};
use slam_rs::frontend::patterns::{Pattern51, Pattern52};
use slam_rs::frontend::tracker::{CpuPatchTracker, MAX_CAPACITY, MAX_LEVELS};
use slam_rs::pyramid::CpuPyramidBuilder;

mod common;

use common::flow::{cpu_tracker, frontend};
use common::{
    FLOW_HEIGHT as HEIGHT, FLOW_WIDTH as WIDTH, dotted_image, flow_config as config,
    flow_rig as rig,
};

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
    let images: [Image<u16, 1>; 1] = [dotted_image(0)];
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

/// Refuse non-positive detector minima to prevent an infinite halving loop.
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
        let odd: Image<u16, 1> = slam_rs::image::zeros(width, height).unwrap();
        // Camera 1 is the wrong one here, so the error must name camera 1.
        let images: [Image<u16, 1>; 2] = [dotted_image(0), odd];
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

/// A frameset that does not move the clock forward is refused, and commits nothing.
///
/// The rule lives here rather than at the Python boundary because every other
/// rule about an input does: the clock the comparison reads is this one.
#[test]
fn a_frameset_that_does_not_follow_the_last_one_is_refused() {
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(1, FrontendOptions::default());
    let images: [Image<u16, 1>; 1] = [dotted_image(0)];
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

/// The budget may not exceed what the tracker was built for.
#[test]
fn a_budget_larger_than_the_tracker_is_refused() {
    let config: VioConfig = config();
    let tracker: CpuPatchTracker<Pattern51> = cpu_tracker(&config, 4);
    let error = FrameToFrameOpticalFlow::<Pattern51, _>::with_stages(
        config,
        &rig(2),
        FrontendOptions {
            max_keypoints: 8,
            ..FrontendOptions::default()
        },
        slam_rs::frontend::stages::CpuStages::new(
            CpuPyramidBuilder::new(),
            tracker,
            slam_rs::frontend::detect::DetectorScratch::with_scanner(Box::new(
                CpuCornerScan::default(),
            )),
        )
        .unwrap(),
        WorkPool::new(1).unwrap(),
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
/// are checked, because `with_stages` takes a tracker that is already
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
        let error = FrameToFrameOpticalFlow::<Pattern51, _>::with_stages(
            fine,
            &vast,
            FrontendOptions::default(),
            slam_rs::frontend::stages::CpuStages::new(
                CpuPyramidBuilder::new(),
                tracker,
                slam_rs::frontend::detect::DetectorScratch::with_scanner(Box::new(
                    CpuCornerScan::default(),
                )),
            )
            .unwrap(),
            WorkPool::new(1).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error, expected);
    }
}
