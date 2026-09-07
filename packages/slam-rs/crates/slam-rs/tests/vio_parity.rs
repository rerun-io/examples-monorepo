//! The V3 gate: the whole VIO, Rust frontend and Rust estimator, against
//! basalt's per-frame poses.
//!
//! `vio_oracle.rs` isolates the backend by replaying the C++'s own keypoints.
//! This file does not: it runs [`Vio::track`] on the PGMs and lets the port's
//! frontend choose the keypoints, which is what a caller gets. The two
//! frontends select different corners inside a grid cell that offers more than
//! the budget, because basalt detects with `cv::FAST` and the port with
//! kornia-rs (D09, D45), so 3 to 4.5 % of the keypoints differ — and a keypoint
//! set that differs by one id changes which landmarks get triangulated, which
//! changes the window. So this is a **tolerance** gate on the trajectory, not a
//! parity gate on the numbers, and the tolerances below are the measurement
//! plus a margin.
//!
//! ## What it needs, and why it skips
//!
//! The 60 framesets are 107 MB of PGM and are not committed; point
//! `SLAM_RS_VIO_FRAMES_DIR` at a directory holding
//! `frame_<NNN>_cam<C>.pgm` (`tools/dump_flow.cpp`'s layout, which
//! `slam_rs.catalog_feed` writes on the frozen gray8 decode path) and the test
//! runs, and it has to hold **all** 60: a prefix is refused rather than
//! measured, because a short run cannot show a late divergence. Without the
//! variable the test prints why and passes — the three framesets
//! `flow/frames/` does carry are too few to reach `opt_started`, which needs
//! five states.
//!
//! Timestamps and the IMU window come from the committed fixtures — the
//! oracle's `flow[].t_ns` and `vio/imu.json` — so the directory only has to
//! hold pixels.
//!
//! ## What is measured
//!
//! Per frameset, against the C++ `float` run's newest state: the position
//! difference in metres and the rotation difference in degrees, both absolute,
//! plus the distance travelled so the numbers can be read in context. The
//! estimator runs `f32`, the precision basalt ships (Q07). On the smoke segment
//! all 60 framesets reach `Tracking` and the worst differences are 6.14e-4 m
//! and 0.219° over 1.49 cm of travel — the segment is a panorama, so the
//! rotation is the number with meaning.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use nalgebra::Vector3;

use slam_rs::frontend::flow::FrontendOptions;
use slam_rs::lie::So3;
use slam_rs::{ImageView, Vio, VioResult, VioStatus};

mod common;
use common::{IMU, ORACLE, OracleFrame, OracleState, Pgm, run_named};

/// How far the port's rig position may sit from basalt's, in metres.
///
/// Measured over the 60 framesets: worst 6.14e-4 m. The segment is a panorama —
/// 1.49 cm of translation over 1.09 s — so this is not a drift ratio, it is how
/// far two nearly-stationary trajectories part; five times the measurement.
const POSITION_TOLERANCE_M: f64 = 3e-3;

/// How far the port's rig orientation may sit from basalt's, in degrees.
///
/// Measured worst: 0.219°, and on a panorama this is the number that matters.
/// It is two orders above the backend-only gate's 8.4e-5 relative rotation
/// because the two frontends pick different corners (D45) and a different
/// keypoint set gives a different set of triangulated landmarks; four times
/// the measurement.
const ROTATION_TOLERANCE_DEG: f64 = 1.0;

/// Framesets `tests/fixtures/flow/frames/` carries.
const COMMITTED_FRAMESETS: usize = 3;

/// The last of them, `ORACLE.flow[2].t_ns`.
const LAST_COMMITTED_T_NS: i64 = 37_012_000;

/// How far the IMU is pushed for the committed run: one frameset interval past
/// [`LAST_COMMITTED_T_NS`].
///
/// It has to reach *past* the last frameset. `:330-336` closes the last
/// preintegration by re-stamping the first sample after the frameset, so a
/// queue that stops on the frameset leaves the interval short and `track`
/// reports `NeedMoreImu` for it.
const COMMITTED_IMU_HORIZON_NS: i64 = 40_000_000;

/// The pipeline both tests drive: the fixture's config and calibration, one
/// frontend thread so the reduction shape is fixed (D31).
fn pipeline() -> Vio<f32> {
    Vio::new(
        common::config(),
        common::calibration(),
        FrontendOptions {
            threads: 1,
            ..FrontendOptions::default()
        },
    )
    .unwrap()
}

/// A read PGM as the byte view `Vio::track` takes.
fn view(pgm: &Pgm) -> ImageView<'_> {
    ImageView {
        width: pgm.width,
        height: pgm.height,
        stride: pgm.width,
        data: &pgm.pixels,
    }
}

/// The three committed framesets, and the IMU that covers them.
///
/// The whole pipeline needs real pixels: `Vio::track` runs the KLT before the
/// estimator sees anything, and a constant frame detects no corners. Three is
/// what `tests/fixtures/flow/frames/` carries, so this cannot reach
/// `opt_started` (five states) — that is what the sixty-frameset gate above
/// is for.
fn drive_the_committed_framesets(vio: &mut Vio<f32>) -> Vec<VioResult> {
    let directory: PathBuf = common::fixtures().join("flow/frames");
    let cameras: usize = common::calibration().t_i_c.len();
    let framesets: usize = common::available_framesets(&directory, cameras, COMMITTED_FRAMESETS);
    assert_eq!(framesets, COMMITTED_FRAMESETS, "{}", directory.display());

    for row in IMU
        .iter()
        .take_while(|row| row.t_ns <= COMMITTED_IMU_HORIZON_NS)
    {
        vio.push_imu(row.t_ns, row.gyro, row.accel).unwrap();
    }
    ORACLE
        .flow
        .iter()
        .take(framesets)
        .enumerate()
        .map(|(frame, flow)| {
            let rasters: Vec<Pgm> = (0..cameras)
                .map(|camera| common::read_pgm(&directory, frame, camera))
                .collect();
            let views: Vec<ImageView<'_>> = rasters.iter().map(view).collect();
            vio.track(flow.t_ns, &views).unwrap()
        })
        .collect()
}

/// The frontend and the estimator are wired together, and a repeat run is
/// bit-identical (D17) — `vio_oracle.rs`'s determinism gate replays the C++'s
/// keypoints, so it cannot see the frontend.
///
/// In the default lane: three framesets of 960x960 KLT twice over is 0.31 s
/// with the optimized test profile (`Cargo.toml`), where it was 14 s of the
/// suite's 15 s budget unoptimized, which is why it used to be `#[ignore]`d.
#[test]
fn the_whole_pipeline_tracks_and_repeats_bit_identically() {
    let mut vio: Vio<f32> = pipeline();
    let results: Vec<VioResult> = drive_the_committed_framesets(&mut vio);

    assert_eq!(
        results.iter().map(|r| r.status).collect::<Vec<VioStatus>>(),
        vec![VioStatus::Tracking; COMMITTED_FRAMESETS]
    );
    assert_eq!(
        results.iter().map(|r| r.t_ns).collect::<Vec<i64>>(),
        ORACLE.flow[..COMMITTED_FRAMESETS]
            .iter()
            .map(|flow| flow.t_ns)
            .collect::<Vec<i64>>()
    );
    let estimator = vio.estimator();
    assert_eq!(estimator.snapshot().states.len(), COMMITTED_FRAMESETS);
    assert_eq!(estimator.last_state_t_ns(), LAST_COMMITTED_T_NS);
    // The first frameset is always a keyframe (`sqrt_keypoint_vio.cpp:61`)
    // and three framesets cannot reach `opt_started` (`:1207`).
    assert_eq!(estimator.kf_ids().collect::<Vec<i64>>(), vec![0]);
    assert!(!estimator.optimization_started());
    assert!(
        vio.last_stats()
            .is_some_and(|stats| stats.num_landmarks > 0)
    );

    assert_eq!(
        drive_the_committed_framesets(&mut pipeline()),
        results,
        "a repeat run is not bit-identical"
    );
}

/// Every field of the pipeline as one number.
///
/// `Vio` derives `Debug`, so this reads all of them — the frontend's pyramids,
/// clock, counter, cells and keypoints, both IMU buffers and their popped
/// samples, the published state and depth guess, the whole estimator window.
/// 85 MB of text, so it is hashed rather than kept.
///
/// The one field that cannot be compared across runs is `FrameStats::timings`,
/// which `StageTimings`'s own doc calls wall-clock. It is the last field of the
/// last field, so cutting the text at it drops the clock and nothing else; the
/// assertion says so.
fn fingerprint(vio: &Vio<f32>) -> u64 {
    use std::hash::{Hash, Hasher};

    let text: String = format!("{vio:?}");
    let cut: usize = text.rfind("StageTimings {").unwrap_or(text.len());
    assert!(
        text.len() - cut < 256,
        "the wall-clock block is no longer the tail of the Debug output"
    );
    let mut hasher: std::collections::hash_map::DefaultHasher = Default::default();
    text[..cut].hash(&mut hasher);
    hasher.finish()
}

/// The fixture's IMU rows up to and including `horizon`, from `next`; returns
/// where it stopped.
fn push_imu_through(vio: &mut Vio<f32>, next: usize, horizon: i64) -> usize {
    let mut index: usize = next;
    while index < IMU.len() && IMU[index].t_ns <= horizon {
        let row = &IMU[index];
        vio.push_imu(row.t_ns, row.gyro, row.accel).unwrap();
        index += 1;
    }
    index
}

/// A frameset the IMU does not yet cover leaves the **whole** pipeline
/// untouched, so pushing the samples and tracking the same frameset again gives
/// what a run that had them all along gives (D17: no arrival order may reach
/// the trajectory).
///
/// This is the property `VioStatus::NeedMoreImu` promises and the one the
/// frontend threatens: `track` runs the KLT before the estimator sees anything,
/// and the KLT swaps the pyramids, advances `t_ns` and the frame counter and
/// eats the frontend's own IMU buffer. A coverage test placed after all that
/// would make the retry track the frameset against itself, and the second and
/// third framesets below — refused *after* initialization — would each lose a
/// preintegration interval as well.
#[test]
fn a_refused_frameset_is_retried_bit_identically() {
    let directory: PathBuf = common::fixtures().join("flow/frames");
    let cameras: usize = common::calibration().t_i_c.len();
    let rasters: Vec<Vec<Pgm>> = (0..COMMITTED_FRAMESETS)
        .map(|frame| {
            (0..cameras)
                .map(|camera| common::read_pgm(&directory, frame, camera))
                .collect()
        })
        .collect();

    // The IMU arrives first: the reference run.
    let mut ahead: Vio<f32> = pipeline();
    push_imu_through(&mut ahead, 0, COMMITTED_IMU_HORIZON_NS);
    let wanted: Vec<VioResult> = (0..COMMITTED_FRAMESETS)
        .map(|frame| {
            let views: Vec<ImageView<'_>> = rasters[frame].iter().map(view).collect();
            ahead.track(ORACLE.flow[frame].t_ns, &views).unwrap()
        })
        .collect();

    // Every frameset arrives before the samples that cover it, is refused, and
    // is tracked again once they land.
    let mut behind: Vio<f32> = pipeline();
    let mut next: usize = 0;
    let mut got: Vec<VioResult> = Vec::new();
    for frame in 0..COMMITTED_FRAMESETS {
        let t_ns: i64 = ORACLE.flow[frame].t_ns;
        let views: Vec<ImageView<'_>> = rasters[frame].iter().map(view).collect();

        let before: u64 = fingerprint(&behind);
        let refused: VioResult = behind.track(t_ns, &views).unwrap();
        assert_eq!(
            refused.status,
            VioStatus::NeedMoreImu,
            "frame {frame} was not refused, so the retry it is asked to model never happens"
        );
        assert_eq!(
            fingerprint(&behind),
            before,
            "frame {frame}: the refused frameset moved the pipeline"
        );

        // Up to the next frameset: past this one, so `:330-336` can close its
        // preintegration, and not past the next, so the next is refused too.
        // The last horizon is the reference run's, so both runs end holding the
        // same samples.
        let horizon: i64 = if frame + 1 < COMMITTED_FRAMESETS {
            ORACLE.flow[frame + 1].t_ns
        } else {
            COMMITTED_IMU_HORIZON_NS
        };
        next = push_imu_through(&mut behind, next, horizon);
        got.push(behind.track(t_ns, &views).unwrap());
    }

    assert_eq!(got, wanted, "the retried run took a different trajectory");
    assert_eq!(
        fingerprint(&behind),
        fingerprint(&ahead),
        "the two runs agree on the poses but not on the rest of the pipeline"
    );
}

/// The whole pipeline over **all 60** framesets, reporting the per-frame
/// difference from basalt's own trajectory.
///
/// Every frameset has to reach `Tracking` and every pose is compared: a gate
/// that skipped the ones that did not would pass on a single pose while the
/// port lost tracking on the other 59.
#[test]
fn the_whole_vio_follows_the_cpp_trajectory() {
    let Some(directory) = std::env::var_os("SLAM_RS_VIO_FRAMES_DIR").map(PathBuf::from) else {
        println!(
            "skipped: set SLAM_RS_VIO_FRAMES_DIR to a directory of frame_<NNN>_cam<C>.pgm; the \
             three committed framesets cannot reach opt_started, which needs five states"
        );
        return;
    };

    let run = run_named(&ORACLE, "float");
    let cameras: usize = common::calibration().t_i_c.len();
    let available: usize = common::available_framesets(&directory, cameras, ORACLE.flow.len());
    assert_eq!(
        available,
        ORACLE.flow.len(),
        "{} holds {available} of the {} framesets the fixture describes; a short directory would \
         make this gate pass on a prefix",
        directory.display(),
        ORACLE.flow.len()
    );

    let mut vio: Vio<f32> = pipeline();
    for row in IMU.iter() {
        vio.push_imu(row.t_ns, row.gyro, row.accel).unwrap();
    }

    let mut worst_position: f64 = 0.0;
    let mut worst_rotation_deg: f64 = 0.0;
    let mut travelled: f64 = 0.0;
    let mut previous: Option<Vector3<f64>> = None;
    let mut compared: usize = 0;

    for (frame, flow) in ORACLE.flow.iter().take(available).enumerate() {
        let rasters: Vec<Pgm> = (0..cameras)
            .map(|camera| common::read_pgm(&directory, frame, camera))
            .collect();
        let views: Vec<ImageView<'_>> = rasters.iter().map(view).collect();
        let result = vio.track(flow.t_ns, &views).unwrap();
        assert_eq!(result.t_ns, flow.t_ns, "frame {frame}: timestamp");
        // No frameset may legitimately miss `Tracking` here, and skipping one
        // would let the tolerances below pass on a single pose. The whole IMU
        // is pushed above and its last sample (1_091_997_574 ns) is past the
        // last frameset (1_091_905_000 ns), so every frameset is covered and
        // none can report `NeedMoreImu`; and the estimator initialises inside
        // the same `process_frame` that measures (`:263-296`), so frameset 0 is
        // `Tracking` too and `NotInitialised` is unreachable from `Vio`. The
        // C++ dump agrees: all 60 of its `frames` carry a state, `frames[0]`
        // included.
        assert_eq!(
            result.status,
            VioStatus::Tracking,
            "frame {frame} at {} ns lost tracking",
            flow.t_ns
        );

        // basalt's newest state at this frameset. `last_state_t_ns` is the
        // frameset itself once `measure` has run, so this is the same pose the
        // port reports.
        let expected: &OracleFrame = &run.frames[frame];
        let newest: &OracleState = expected
            .states
            .iter()
            .find(|state| state.t_ns == expected.last_state_t_ns)
            .unwrap_or_else(|| panic!("frame {frame}: the newest state is not in the window"));

        let pose: [f64; 7] = result.world_from_rig;
        let position: Vector3<f64> = Vector3::new(pose[0], pose[1], pose[2]);
        if let Some(before) = previous {
            travelled += (position - before).norm();
        }
        previous = Some(position);

        let offset: f64 = (position - Vector3::from(newest.t)).norm();
        // The relative rotation's angle, from Sophus's own log.
        let mine: So3<f64> = So3::from_quaternion_xyzw(pose[3], pose[4], pose[5], pose[6]).unwrap();
        let theirs: So3<f64> =
            So3::from_quaternion_xyzw(newest.q[0], newest.q[1], newest.q[2], newest.q[3]).unwrap();
        let angle_deg: f64 = (mine.inverse() * theirs).log().norm().to_degrees();

        worst_position = worst_position.max(offset);
        worst_rotation_deg = worst_rotation_deg.max(angle_deg);
        compared += 1;
    }

    println!(
        "whole-VIO parity over {compared} of {available} framesets: worst position \
         {worst_position:.4e} m, worst rotation {worst_rotation_deg:.4e} deg, {travelled:.4} m \
         travelled"
    );
    assert_eq!(
        compared,
        ORACLE.flow.len(),
        "every frameset has to be compared, not just the ones that tracked"
    );
    assert!(
        worst_position <= POSITION_TOLERANCE_M,
        "the trajectory drifted {worst_position:.4e} m from basalt's over {travelled:.4} m"
    );
    assert!(
        worst_rotation_deg <= ROTATION_TOLERANCE_DEG,
        "the orientation drifted {worst_rotation_deg:.4e} deg from basalt's"
    );
}


