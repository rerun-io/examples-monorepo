//! The V1 gate for the optical-flow frontend: the Rust KLT against basalt's own.
//!
//! ## Why the gate is a *seeded* comparison
//!
//! basalt detects corners with `cv::FAST` and the port detects them with
//! kornia-rs (decision D09, trap 2). The two are different implementations with
//! different response definitions, so they select different corners inside a
//! grid cell that offers more than the budget, and a keypoint-for-keypoint
//! comparison of a full run would measure the detectors, not the tracker.
//!
//! So the gate isolates the tracker. For each pair of consecutive framesets it
//! takes the **C++'s own keypoints at frame `t`** — ids, positions and the linear
//! part of each warp — feeds them to [`CpuPatchTracker`] as the source patches,
//! and asks how many end up within half a pixel of where the C++ put the same id
//! at `t + 1`. Anything the C++ tracked and the port dropped counts against the
//! fraction too. The detector overlap is measured separately and reported, never
//! gated.
//!
//! ## What the fixtures are
//!
//! `tests/fixtures/flow/frames/` holds the first three framesets of the smoke
//! reference segment `msd-index__MIO_others__MIO10_short_2_panorama` as 8-bit
//! PGMs, exactly the bytes `slam_rs.catalog_feed` produces on the frozen decode
//! path, and `tests/fixtures/flow/dumps/` holds what the C++ frontend made of
//! **eight** of them. Only three framesets are committed because a 960x960 pair
//! is 1.8 MB and downscaling would change the algorithm; point
//! `SLAM_RS_FLOW_FRAMES_DIR` at a directory holding all eight (the fork's
//! `tools/dump_flow.cpp` reads the same layout) to run the whole set.
//!
//! The C++ side is `basalt_dump_flow` on the fork's `slam-rs-reference` branch,
//! driven with one TBB worker, no IMU and therefore an identity `T_c1_c2` — the
//! `first_state_arrived == false` path of `processingLoop`. The Rust side runs
//! with [`PosePrediction::default`], which is the same two identities.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use nalgebra::{Matrix2, Vector2};
use serde::Deserialize;

use slam_rs::calib::Calibration;
use slam_rs::camera::RigCamera;
use slam_rs::config::VioConfig;
use slam_rs::frontend::flow::{
    FlowFrame, FrameToFrameOpticalFlow, FrontendOptions, PosePrediction, project_between_cams,
};
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::frontend::se2::AffineCompact2f;
use slam_rs::frontend::tracker::{
    CpuPatchTracker, FlowResult, FlowTransforms, PatchSoA, PatchTracker, PointsSoA, SourcePatches,
};
use slam_rs::image::ImageU16;
use slam_rs::lie::Se3;
use slam_rs::pyramid::PyramidU16;

mod common;

/// How close a tracked keypoint must land to the C++'s, in pixels.
const POSITION_TOLERANCE_PX: f32 = 0.5;

/// The fraction of the C++'s tracks the port must reproduce.
const SEEDED_TRACKING_GATE: f64 = 0.95;

/// Framesets `tools/dump_flow.cpp` dumped, and the most the PGM directory can
/// hold.
const DUMPED_FRAMESETS: usize = 8;

/// One keypoint of one camera, as `tools/dump_flow.cpp` writes it.
#[derive(Debug, Clone, Deserialize)]
struct DumpKeypoint {
    id: u64,
    x: f32,
    y: f32,
    /// `[m00, m01, m10, m11]` of `AffineCompact2f::linear()`.
    linear: [f32; 4],
    response: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct DumpCamera {
    camera: usize,
    keypoints: Vec<DumpKeypoint>,
}

#[derive(Debug, Clone, Deserialize)]
struct DumpGrid {
    cell: usize,
    x_start: usize,
    y_start: usize,
    columns: usize,
    rows: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct DumpFrame {
    frame: usize,
    t_ns: i64,
    last_keypoint_id: u64,
    grid: DumpGrid,
    cameras: Vec<DumpCamera>,
}

impl DumpFrame {
    fn keypoints(&self, camera: usize) -> &[DumpKeypoint] {
        &self.cameras[camera].keypoints
    }

    fn by_id(&self, camera: usize) -> BTreeMap<u64, &DumpKeypoint> {
        self.keypoints(camera)
            .iter()
            .map(|keypoint| (keypoint.id, keypoint))
            .collect()
    }
}

/// Where the PGMs live: the checked-in three, or a directory holding all eight.
fn frames_dir() -> PathBuf {
    match std::env::var_os("SLAM_RS_FLOW_FRAMES_DIR") {
        Some(path) => PathBuf::from(path),
        None => common::fixtures().join("flow/frames"),
    }
}

/// How many consecutive framesets the PGMs on disk cover.
fn available_framesets() -> usize {
    common::available_framesets(&frames_dir(), 2, DUMPED_FRAMESETS)
}

fn read_dump(frame: usize) -> DumpFrame {
    let path: PathBuf = common::fixtures().join(format!("flow/dumps/frame_{frame:03}.json"));
    let text: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("cannot parse {}: {error}", path.display()))
}

/// A PGM frameset image, widened the way basalt widens a camera source.
fn read_pgm(frame: usize, camera: usize) -> ImageU16 {
    let pgm: common::Pgm = common::read_pgm(&frames_dir(), frame, camera);
    ImageU16::from_u8_strided(&pgm.pixels, pgm.width, pgm.height, pgm.width).unwrap()
}

/// How many of the C++'s own tracks the port reproduces, for one camera and one
/// frameset pair.
struct SeededResult {
    /// Ids the C++ carried from `t` to `t + 1`.
    expected: usize,
    /// Of those, how many the port put within [`POSITION_TOLERANCE_PX`].
    matched: usize,
    /// The largest distance among the ones that did match, for the report.
    worst_matched_px: f32,
}

fn seeded_tracking(
    config: &VioConfig,
    cameras: &[RigCamera<f32>],
    camera: usize,
    previous: &DumpFrame,
    current: &DumpFrame,
) -> SeededResult {
    let levels: usize = config.optical_flow_levels as usize;
    let previous_pyramid: PyramidU16 =
        common::pyramid_of(&read_pgm(previous.frame, camera), levels);
    let current_pyramid: PyramidU16 = common::pyramid_of(&read_pgm(current.frame, camera), levels);

    let source: &[DumpKeypoint] = previous.keypoints(camera);
    let target: BTreeMap<u64, &DumpKeypoint> = current.by_id(camera);

    // The driver's guess for a frame-to-frame track: `projectBetweenCams` through
    // `depth_guess` with `T_c1_c2 = identity`, which is the round trip
    // `project(unproject(pixel))` and not quite the identity on a fisheye
    // (`frame_to_frame_optical_flow.h:335-342`, decision D39).
    let identity: Se3<f32> = Se3::identity();
    let depth: f32 = config.optical_flow_matching_default_depth;

    let mut positions: PointsSoA = PointsSoA::with_capacity(source.len());
    let mut guesses: FlowTransforms = FlowTransforms::with_capacity(source.len());
    for keypoint in source {
        let position: Vector2<f32> = Vector2::new(keypoint.x, keypoint.y);
        positions.push(position);
        guesses.push(&AffineCompact2f {
            linear: Matrix2::new(
                keypoint.linear[0],
                keypoint.linear[1],
                keypoint.linear[2],
                keypoint.linear[3],
            ),
            translation: project_between_cams(cameras, &position, depth, &identity, camera, camera)
                .1,
        });
    }

    let mut patches: PatchSoA<Pattern51> = PatchSoA::new(source.len().max(1), levels + 1).unwrap();
    patches.build(&previous_pyramid, &positions, None).unwrap();
    let mut tracker: CpuPatchTracker<Pattern51> = CpuPatchTracker::new(
        source.len().max(1),
        levels + 1,
        config.optical_flow_max_iterations as usize,
        config.optical_flow_max_recovered_dist2,
        WorkPool::new(1).unwrap(),
    )
    .unwrap();
    let mut result: FlowResult = FlowResult::with_capacity(source.len().max(1));
    tracker
        .track(
            &previous_pyramid,
            &current_pyramid,
            &patches,
            &guesses,
            &mut result,
        )
        .unwrap();

    let mut expected: usize = 0;
    let mut matched: usize = 0;
    let mut worst_matched_px: f32 = 0.0;
    for (index, keypoint) in source.iter().enumerate() {
        let Some(after) = target.get(&keypoint.id) else {
            continue;
        };
        expected += 1;
        if !result.is_valid(index) {
            continue;
        }
        let tracked: Vector2<f32> = result.transform(index).translation;
        let distance: f32 = (tracked - Vector2::new(after.x, after.y)).norm();
        if distance <= POSITION_TOLERANCE_PX {
            matched += 1;
            worst_matched_px = worst_matched_px.max(distance);
        }
    }
    SeededResult {
        expected,
        matched,
        worst_matched_px,
    }
}

/// **The V1 gate.** Seeded with the C++'s keypoints, the port's tracker must
/// reproduce at least 95% of the C++'s own tracks to within half a pixel.
#[test]
fn seeded_tracking_reproduces_the_cpp_tracker() {
    let config: VioConfig = common::config();
    let calibration: Calibration<f64> = common::calibration();
    let cameras: Vec<RigCamera<f32>> = RigCamera::from_calibration(&calibration.cast()).unwrap();

    let framesets: usize = available_framesets();
    assert!(
        framesets >= 3,
        "expected at least the three checked-in framesets, found {framesets}"
    );

    let mut totals: Vec<(usize, usize)> = vec![(0, 0); cameras.len()];
    let mut worst_px: f32 = 0.0;
    for frame in 1..framesets {
        let previous: DumpFrame = read_dump(frame - 1);
        let current: DumpFrame = read_dump(frame);
        assert_eq!(previous.frame, frame - 1);
        assert_eq!(current.frame, frame);

        for (camera, total) in totals.iter_mut().enumerate() {
            let seeded: SeededResult =
                seeded_tracking(&config, &cameras, camera, &previous, &current);
            total.0 += seeded.expected;
            total.1 += seeded.matched;
            worst_px = worst_px.max(seeded.worst_matched_px);
        }
    }

    let mut overall_expected: usize = 0;
    let mut overall_matched: usize = 0;
    for (camera, (expected, matched)) in totals.iter().enumerate() {
        overall_expected += expected;
        overall_matched += matched;
        let fraction: f64 = *matched as f64 / *expected as f64;
        println!(
            "seeded tracking, camera {camera}: {matched}/{expected} = {:.2}% within {POSITION_TOLERANCE_PX} px",
            fraction * 100.0
        );
    }
    let fraction: f64 = overall_matched as f64 / overall_expected as f64;
    println!(
        "seeded tracking, {} frameset pairs of {framesets}: {overall_matched}/{overall_expected} = {:.2}%, worst matched {worst_px:.4} px",
        framesets - 1,
        fraction * 100.0
    );

    assert!(
        overall_expected > 100,
        "only {overall_expected} tracks to compare"
    );
    assert!(
        fraction >= SEEDED_TRACKING_GATE,
        "seeded tracking reproduced {:.2}% of the C++'s tracks, gate is {:.0}%",
        fraction * 100.0,
        SEEDED_TRACKING_GATE * 100.0
    );
}

/// Reported, never gated: how much the two detectors agree, and how much the
/// two occupancy grids do.
///
/// Both sides now run the canonical FAST-9 `cornerScore` and OpenCV's
/// strictly-greater 3x3 suppression, so what is left is the cell walk itself:
/// `cv::FAST` sweeps a whole cell with one three-row score ring, and
/// `std::sort` (`keypoints.cpp:166`) is not stable while the sort here is. This
/// number says how much of the divergence in a full run comes from the detector
/// rather than the tracker; it is printed with `--nocapture` and only asserted
/// to be non-trivial. The two reports share one pipeline run because a
/// 960x960 frameset through an unoptimised build is the test's whole cost.
#[test]
fn detector_overlap_is_reported() {
    let config: VioConfig = common::config();
    let budget: i32 = config.optical_flow_detection_num_points_cell;
    let calibration: Calibration<f64> = common::calibration();
    let mut flow: FrameToFrameOpticalFlow<Pattern51> = FrameToFrameOpticalFlow::new(
        config,
        &calibration,
        FrontendOptions {
            threads: 1,
            ..FrontendOptions::default()
        },
    )
    .unwrap();

    let framesets: usize = available_framesets();
    let mut report: Vec<String> = Vec::new();
    let mut any_overlap: bool = false;
    let mut shared_responses: usize = 0;

    for frame in 0..framesets {
        let dump: DumpFrame = read_dump(frame);
        let images: Vec<ImageU16> = (0..2).map(|camera| read_pgm(frame, camera)).collect();
        let produced: &FlowFrame = flow
            .process_frame(dump.t_ns, &images, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(produced.t_ns, Some(dump.t_ns));

        for camera in 0..2 {
            let expected: &[DumpKeypoint] = dump.keypoints(camera);
            let ours: Vec<Vector2<f32>> = (0..produced.cameras[camera].len())
                .map(|index| produced.cameras[camera].transforms.translation(index))
                .collect();

            // Where the two detectors chose the same pixel, they must agree on
            // its `cornerScore` to the integer: OpenCV returns
            // `max(a0, -b0) - 1` (`fast_score.cpp`) and the port subtracts the
            // same one from kornia's score. A response the C++ dump records as
            // -1 is a keypoint `addKeypoint` was given no response for, and is
            // skipped.
            for keypoint in expected {
                if keypoint.response < 0.0 {
                    continue;
                }
                let Some(index) = (0..produced.cameras[camera].len()).find(|index| {
                    produced.cameras[camera].transforms.translation(*index)
                        == Vector2::new(keypoint.x, keypoint.y)
                }) else {
                    continue;
                };
                let ours: f32 = produced.cameras[camera].responses[index];
                if ours < 0.0 {
                    continue;
                }
                assert_eq!(
                    ours, keypoint.response,
                    "frame {frame} camera {camera}: the two FAST scores at ({}, {}) differ",
                    keypoint.x, keypoint.y
                );
                shared_responses += 1;
            }
            // A C++ keypoint counts as reproduced when some Rust keypoint sits
            // within one pixel of it, whatever id either side gave it.
            let near: usize = expected
                .iter()
                .filter(|keypoint| {
                    ours.iter().any(|translation| {
                        (translation - Vector2::new(keypoint.x, keypoint.y)).norm() <= 1.0
                    })
                })
                .count();
            any_overlap |= near > 0;
            report.push(format!(
                "frame {frame} camera {camera}: cpp {} rust {} overlapping {near} ({:.1}%)",
                expected.len(),
                ours.len(),
                100.0 * near as f64 / expected.len().max(1) as f64
            ));
        }
    }

    // The occupancy the two sides derive from their own keypoints, on the last
    // frameset the loop above left in the frontend.
    let dump: DumpFrame = read_dump(framesets - 1);
    let grid = flow.occupancy_grid();
    let mut cpp: Vec<i32> = vec![0; grid.rows * grid.columns];
    for keypoint in dump.keypoints(0) {
        if !grid.contains(keypoint.x, keypoint.y) {
            continue;
        }
        let (row, column) = grid.cell_of(keypoint.x, keypoint.y);
        cpp[row * grid.columns + column] += 1;
    }
    let ours: &[i32] = flow.cell_counts(0);
    assert_eq!(ours.len(), cpp.len());
    let occupied_cpp: usize = cpp.iter().filter(|count| **count > 0).count();
    let occupied_rust: usize = ours.iter().filter(|count| **count > 0).count();
    let both: usize = cpp
        .iter()
        .zip(ours.iter())
        .filter(|(a, b)| **a > 0 && **b > 0)
        .count();
    report.push(format!(
        "camera 0 cell occupancy: cpp {occupied_cpp} cells, rust {occupied_rust} cells, both {both}"
    ));
    report.push(format!(
        "FAST scores compared at the same pixel and equal: {shared_responses}"
    ));

    for line in &report {
        println!("{line}");
    }
    assert!(
        any_overlap,
        "the two detectors agree on nothing at all, which is a bug rather than a delta"
    );
    assert!(
        shared_responses > 100,
        "only {shared_responses} responses were comparable, which is too few to mean anything"
    );
    assert!(
        ours.iter().all(|count| *count <= budget),
        "a cell holds more than the per-cell budget"
    );
    assert!(both * 2 >= occupied_cpp, "the two grids barely overlap");
}

/// The grid geometry is not a detector question: both sides must centre the same
/// cells over camera 0 (`keypoints.cpp:140-144`). On this rig every camera is
/// 960x960, so the per-camera detection grid and the shared occupancy grid are
/// the same thing.
#[test]
fn the_detection_grid_matches_the_cpp() {
    let config: VioConfig = common::config();
    let calibration: Calibration<f64> = common::calibration();
    let flow: FrameToFrameOpticalFlow<Pattern51> =
        FrameToFrameOpticalFlow::new(config, &calibration, FrontendOptions::default()).unwrap();
    let dump: DumpFrame = read_dump(0);

    assert_eq!(flow.occupancy_grid().cell, dump.grid.cell);
    assert_eq!(flow.occupancy_grid().x_start, dump.grid.x_start);
    assert_eq!(flow.occupancy_grid().y_start, dump.grid.y_start);
    assert_eq!(flow.occupancy_grid().columns, dump.grid.columns);
    assert_eq!(flow.occupancy_grid().rows, dump.grid.rows);
}

/// A dump fixture that has silently changed shape would make the gate meaningless.
#[test]
fn the_fixtures_are_the_ones_the_gate_was_written_for() {
    for frame in 0..8 {
        let dump: DumpFrame = read_dump(frame);
        assert_eq!(dump.frame, frame);
        assert_eq!(dump.cameras.len(), 2);
        for (index, camera) in dump.cameras.iter().enumerate() {
            assert_eq!(camera.camera, index);
            assert!(!camera.keypoints.is_empty());
            for keypoint in &camera.keypoints {
                assert!(keypoint.id < dump.last_keypoint_id);
                assert!(keypoint.x.is_finite() && keypoint.y.is_finite());
                assert!(keypoint.response.is_finite());
            }
        }
        // Ids never repeat inside one camera and never run backwards in time.
        let ids: Vec<u64> = dump.keypoints(0).iter().map(|k| k.id).collect();
        assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    }
    assert_eq!(read_dump(0).t_ns, 0);
    assert_eq!(read_dump(1).t_ns, 18_507_000);
}
