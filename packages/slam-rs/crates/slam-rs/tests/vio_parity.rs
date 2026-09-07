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
//! runs. Without it the test prints why and passes: the three framesets
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

use std::path::{Path, PathBuf};

use nalgebra::Vector3;
use serde::Deserialize;

use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::frontend::flow::FrontendOptions;
use slam_rs::lie::So3;
use slam_rs::{ImageView, Vio, VioStatus};

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

/// Fixture shapes: only the fields this file reads.
#[derive(Debug, Deserialize)]
struct Oracle {
    runs: Vec<OracleRun>,
    flow: Vec<OracleFlow>,
}

#[derive(Debug, Deserialize)]
struct OracleRun {
    scalar: String,
    frames: Vec<OracleFrame>,
}

#[derive(Debug, Deserialize)]
struct OracleFlow {
    t_ns: i64,
}

#[derive(Debug, Deserialize)]
struct OracleFrame {
    last_state_t_ns: i64,
    states: Vec<OracleState>,
}

#[derive(Debug, Deserialize)]
struct OracleState {
    t_ns: i64,
    q: [f64; 4],
    t: [f64; 3],
}

#[derive(Debug, Deserialize)]
struct ImuFixture {
    imu: Vec<ImuRow>,
}

#[derive(Debug, Deserialize)]
struct ImuRow {
    t_ns: i64,
    gyro: [f64; 3],
    accel: [f64; 3],
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// An 8-bit binary PGM as its raw raster, the bytes `ImageView` wants.
fn read_pgm(directory: &Path, frame: usize, camera: usize) -> (usize, usize, Vec<u8>) {
    let path: PathBuf = directory.join(format!("frame_{frame:03}_cam{camera}.pgm"));
    let bytes: Vec<u8> = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));

    // "P5\n<w> <h>\n255\n" then the raster; the writer emits exactly that.
    let mut fields: Vec<usize> = Vec::new();
    let mut cursor: usize = 2;
    while fields.len() < 3 {
        while bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let start: usize = cursor;
        while !bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        fields.push(
            std::str::from_utf8(&bytes[start..cursor])
                .unwrap()
                .parse()
                .unwrap(),
        );
    }
    cursor += 1;
    assert_eq!(fields[2], 255, "{} is not an 8-bit PGM", path.display());
    (fields[0], fields[1], bytes[cursor..].to_vec())
}

/// The whole pipeline over as many framesets as the directory holds, reporting
/// the per-frame difference from basalt's own trajectory.
#[test]
fn the_whole_vio_follows_the_cpp_trajectory() {
    let Some(directory) = std::env::var_os("SLAM_RS_VIO_FRAMES_DIR").map(PathBuf::from) else {
        println!(
            "skipped: set SLAM_RS_VIO_FRAMES_DIR to a directory of frame_<NNN>_cam<C>.pgm; the \
             three committed framesets cannot reach opt_started, which needs five states"
        );
        return;
    };

    let text: String = std::fs::read_to_string(fixtures().join("vio/vio_oracle.json")).unwrap();
    let oracle: Oracle = serde_json::from_str(&text).unwrap();
    let run: &OracleRun = oracle
        .runs
        .iter()
        .find(|run| run.scalar == "float")
        .expect("the fixture has no float run");
    let text: String = std::fs::read_to_string(fixtures().join("vio/imu.json")).unwrap();
    let imu: ImuFixture = serde_json::from_str(&text).unwrap();
    let config: VioConfig = VioConfig::from_json_str(
        &std::fs::read_to_string(fixtures().join("msdmi_config.json")).unwrap(),
    )
    .unwrap();
    let calibration: Calibration<f64> = Calibration::from_json_str(
        &std::fs::read_to_string(fixtures().join("msdmi_calib.json")).unwrap(),
    )
    .unwrap();
    let cameras: usize = calibration.t_i_c.len();

    let available: usize = (0..oracle.flow.len())
        .take_while(|frame| {
            (0..cameras).all(|camera| {
                directory
                    .join(format!("frame_{frame:03}_cam{camera}.pgm"))
                    .exists()
            })
        })
        .count();
    assert!(
        available > 4,
        "{} holds {available} framesets; five states have to accumulate before the estimator \
         optimizes at all",
        directory.display()
    );

    let mut vio: Vio<f32> = Vio::new(
        config,
        calibration,
        FrontendOptions {
            threads: 1,
            ..FrontendOptions::default()
        },
    )
    .unwrap();
    for row in &imu.imu {
        vio.push_imu(row.t_ns, row.gyro, row.accel).unwrap();
    }

    let mut worst_position: f64 = 0.0;
    let mut worst_rotation_deg: f64 = 0.0;
    let mut travelled: f64 = 0.0;
    let mut previous: Option<Vector3<f64>> = None;
    let mut compared: usize = 0;

    for (frame, flow) in oracle.flow.iter().take(available).enumerate() {
        let rasters: Vec<(usize, usize, Vec<u8>)> = (0..cameras)
            .map(|camera| read_pgm(&directory, frame, camera))
            .collect();
        let views: Vec<ImageView<'_>> = rasters
            .iter()
            .map(|(width, height, data)| ImageView {
                width: *width,
                height: *height,
                stride: *width,
                data,
            })
            .collect();
        let result = vio.track(flow.t_ns, &views).unwrap();
        assert_eq!(result.t_ns, flow.t_ns, "frame {frame}: timestamp");
        if result.status != VioStatus::Tracking {
            continue;
        }

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
    assert!(compared > 0, "no frameset reached Tracking");
    assert!(
        worst_position <= POSITION_TOLERANCE_M,
        "the trajectory drifted {worst_position:.4e} m from basalt's over {travelled:.4} m"
    );
    assert!(
        worst_rotation_deg <= ROTATION_TOLERANCE_DEG,
        "the orientation drifted {worst_rotation_deg:.4e} deg from basalt's"
    );
}
