//! Optional whole-clip replay harness.
//! Read images, IMU samples and calibration from a local clip dump and write
//! trajectory CSV on the absolute device clock. Accuracy is evaluated outside
//! this Rust test against ground truth.
//!
//! Set `SLAM_RS_CLIP_DIR`, `SLAM_RS_CLIP_SCALAR`, `SLAM_RS_CLIP_OUT` and optionally
//! `SLAM_RS_CLIP_STATS`, then run `cargo test --release --test full_clip -- --nocapture`.
//! Without a clip directory the harness reports that it did not replay a clip.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use slam_rs::calib::Calibration;
use slam_rs::frontend::flow::FrontendOptions;
use slam_rs::lie::LieScalar;
use slam_rs::{ImageView, Vio, VioResult, VioStatus};

mod common;
use common::{Clip, ClipScalar, Pgm, config_for, read_pgm};

/// One inertial sample of `imu.csv`.
struct ImuRow {
    t_ns: i64,
    gyro: [f64; 3],
    accel: [f64; 3],
}

/// `imu.csv`, whose values are written by Python's `repr` and so parse back exactly.
fn read_imu(path: &Path) -> Vec<ImuRow> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(fields.len(), 7, "{line}");
            let value = |index: usize| fields[index].parse::<f64>().unwrap();
            ImuRow {
                t_ns: fields[0].parse().unwrap(),
                gyro: [value(1), value(2), value(3)],
                accel: [value(4), value(5), value(6)],
            }
        })
        .collect()
}

/// Replay one scalar lane and write trajectory and optional per-frame decisions.
/// IMU input may be preloaded or delivered per frame to check arrival-order independence.
fn replay<S: LieScalar>(
    clip: &Clip,
    directory: &Path,
    imu: &[ImuRow],
    calibration: &Path,
    streamed: bool,
    out: &Path,
    stats: Option<PathBuf>,
) {
    let mut vio: Vio<S> = Vio::new(
        config_for(&clip.dataset_name),
        Calibration::from_json_str(&std::fs::read_to_string(calibration).unwrap()).unwrap(),
        FrontendOptions {
            threads: 1,
            ..FrontendOptions::default()
        },
    )
    .unwrap();
    // Select preloaded IMU or per-frame coverage including the first later sample.
    // Both delivery modes should produce the same trajectory.
    let mut cursor: usize = 0;
    if !streamed {
        for row in imu {
            vio.push_imu(row.t_ns, row.gyro, row.accel).unwrap();
        }
        cursor = imu.len();
    }

    let mut poses: String = String::from("#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n");
    let mut decisions: String = String::from(
        "#frame,t_ns,status,took_keyframe,keyframe_vote,frames_after_kf,keyframes,num_landmarks,\
         num_observations,opt_started,lm_steps,last_lambda,last_error_after,termination,marginalized\n",
    );
    let started: Instant = Instant::now();
    let mut tracked: usize = 0;
    for (frame, &t_ns) in clip.frame_t_ns.iter().enumerate() {
        while cursor < imu.len() {
            let row: &ImuRow = &imu[cursor];
            vio.push_imu(row.t_ns, row.gyro, row.accel).unwrap();
            cursor += 1;
            if row.t_ns > t_ns {
                break;
            }
        }
        let rasters: Vec<Pgm> = (0..clip.num_cameras)
            .map(|camera| read_pgm(directory, frame, camera))
            .collect();
        let views: Vec<ImageView<'_>> = rasters
            .iter()
            .map(|pgm| ImageView {
                width: pgm.width,
                height: pgm.height,
                stride: pgm.width,
                data: &pgm.pixels,
            })
            .collect();
        let result: VioResult = vio.track(t_ns, &views).unwrap();
        assert_eq!(result.t_ns, t_ns, "frame {frame}: timestamp");
        if result.status == VioStatus::Tracking {
            tracked += 1;
            let pose: [f64; 7] = result.world_from_rig;
            writeln!(
                poses,
                "{},{:?},{:?},{:?},{:?},{:?},{:?},{:?}",
                t_ns + clip.capture_start_time_ns,
                pose[0],
                pose[1],
                pose[2],
                pose[6],
                pose[3],
                pose[4],
                pose[5]
            )
            .unwrap();
        }
        if stats.is_some() {
            let frame_stats = vio.last_stats().unwrap();
            writeln!(
                decisions,
                "{frame},{},{:?},{},{},{},{},{},{},{},{},{:?},{:?},{:?},{}",
                t_ns + clip.capture_start_time_ns,
                result.status,
                frame_stats.took_keyframe,
                frame_stats.keyframe_vote,
                frame_stats.frames_after_kf,
                frame_stats.kf_ids.len(),
                frame_stats.num_landmarks,
                frame_stats.num_observations,
                frame_stats.opt_started,
                frame_stats.lm.len(),
                frame_stats.lm.last().map(|step| step.lambda),
                frame_stats.lm.last().map(|step| step.error_after),
                frame_stats.termination,
                frame_stats.marginalization.is_some(),
            )
            .unwrap();
        }
        if (frame + 1) % 200 == 0 {
            println!(
                "{}/{} framesets, {tracked} tracked, {:.1} s",
                frame + 1,
                clip.framesets,
                started.elapsed().as_secs_f64()
            );
        }
    }

    std::fs::write(out, poses).unwrap();
    if let Some(path) = stats {
        std::fs::write(path, decisions).unwrap();
    }
    println!(
        "{}: {tracked}/{} framesets tracked in {:.1} s -> {}",
        clip.segment_id,
        clip.framesets,
        started.elapsed().as_secs_f64(),
        out.display()
    );
}

/// The whole clip, in the precision [`ClipScalar::from_env`] selects.
#[test]
fn the_whole_clip_replays_into_a_trajectory_csv() {
    let Some(directory) = std::env::var_os("SLAM_RS_CLIP_DIR").map(PathBuf::from) else {
        println!(
            "skipped: set SLAM_RS_CLIP_DIR to a directory written by tests/tools/dump_clip.py; a \
             clip is gigabytes of PGM and is not committed"
        );
        return;
    };
    let clip: Clip = Clip::read(&directory);
    assert_eq!(clip.frame_t_ns.len(), clip.framesets);
    let imu: Vec<ImuRow> = read_imu(&directory.join("imu.csv"));
    assert_eq!(imu.len(), clip.imu_samples);

    let scalar: ClipScalar = ClipScalar::from_env(ClipScalar::F32);
    let out: PathBuf = std::env::var_os("SLAM_RS_CLIP_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| directory.join(format!("slam_rs_{}.csv", scalar.rust_name())));
    let stats: Option<PathBuf> = std::env::var_os("SLAM_RS_CLIP_STATS").map(PathBuf::from);
    // Use the clip calibration unless overridden. Catalog f32 statics and calibration
    // JSON doubles can differ, so the selected calibration is part of the input.
    let calibration: PathBuf = directory
        .join(std::env::var("SLAM_RS_CLIP_CALIB").unwrap_or_else(|_| "calib.json".to_string()));
    println!(
        "{} ({} calibration from {}, {} cameras, {} framesets, {} inertial samples), {}",
        clip.segment_id,
        clip.calibration_source,
        calibration.display(),
        clip.num_cameras,
        clip.framesets,
        clip.imu_samples,
        scalar.rust_name()
    );
    let streamed: bool = std::env::var("SLAM_RS_CLIP_IMU").is_ok_and(|mode| mode == "streamed");
    match scalar {
        ClipScalar::F32 => {
            replay::<f32>(&clip, &directory, &imu, &calibration, streamed, &out, stats);
        }
        ClipScalar::F64 => {
            replay::<f64>(&clip, &directory, &imu, &calibration, streamed, &out, stats);
        }
    }
}
