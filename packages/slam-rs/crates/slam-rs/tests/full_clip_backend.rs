//! The estimator alone, over a **whole** clip, against basalt's own window.
//!
//! `vio_oracle.rs` is this comparison over the sixty committed framesets, and it
//! asserts: every integer decision identical, `f64` agreeing to 3.1e-10. This
//! file asks the question sixty framesets cannot answer — whether that identity
//! survives four thousand of them — so it takes the same oracle dump produced
//! for a whole clip. In `double` it is D60's identity lane and asserts
//! [`F64_IDENTITY_M`] with no integer break; in `float` it measures and reports,
//! because the f32 pair is 3.3 mm apart with the LM step count breaking, which
//! is basalt's own conditioning and not a fault (S15 §2.3). The dump is
//!
//! ```bash
//! build/basalt_vio_oracle <clip-dir> <clip-dir>/calib.json data/msd/msdmi_config.json \
//!   <clip-dir>/vio_oracle_full.json 4095
//! ```
//!
//! where `<clip-dir>` is what `tests/tools/dump_clip.py` wrote, so both sides
//! read the same pixels, the same inertial samples and the same calibration.
//!
//! ## Why this and not `full_clip.rs`
//!
//! `full_clip.rs` runs the port's own frontend, which picks different corners
//! from `cv::FAST` inside a cell that offers more than the budget (D09, D45).
//! That difference is real, it is not a rounding difference, and over thousands
//! of framesets it decides which landmarks exist. Replaying the C++'s own
//! `OpticalFlowResult` removes it, and what is left is the estimator: the same
//! visual input, the same inertial input, the same calibration, the same config.
//! Whatever separates the two runs here is the backend or its arithmetic, and
//! nothing else.
//!
//! ```bash
//! SLAM_RS_CLIP_DIR=<clip-dir> \
//! SLAM_RS_ORACLE_JSON=<clip-dir>/vio_oracle_full.json \
//! SLAM_RS_ORACLE_SCALAR=double \
//! SLAM_RS_ORACLE_OUT=<clip-dir>/slam_rs_backend_f64.csv \
//!   cargo test --release --test full_clip_backend -- --nocapture
//! ```
//!
//! Without `SLAM_RS_ORACLE_JSON` the test prints why and passes, so the default
//! suite stays fast: the gated dump is `MIO07`'s, 174 MB for its 4,095
//! framesets, and it is a **bundle** artifact rather than a repository one. It
//! sits at `<clip-dir>/vio_oracle_full.json` on the host that produced it
//! beside the clip `dump_clip.py` wrote, and both are rebuilt with the two
//! commands above when that directory is gone.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nalgebra::Vector3;
use serde::Deserialize;

use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::estimator::{FrameOutcome, SqrtKeypointVio};
use slam_rs::imu::ImuSample;
use slam_rs::lie::{LieScalar, So3};

mod common;
use common::{Oracle, OracleFlow, OracleFrame, OracleRun, OracleState};

/// How far the f64 backend may sit from basalt's over a whole clip (D60's identity lane).
///
/// Measured over all 4,095 framesets of `MIO07`: 7.2e-12 m worst, with every
/// integer decision identical (S15 §2.3). This is that number rounded up, so the
/// lane fails on a real difference and never on rounding — `MGO14`'s dump does
/// fail it, at frameset 905, which is the difference C76 named and left open.
const F64_IDENTITY_M: f64 = 1e-11;

/// The clip's inertial window in the fork's own shape, which is what
/// `dump_clip.py` writes beside the pixels for the oracle to read.
#[derive(Debug, Deserialize)]
struct ImuFixture {
    imu: Vec<common::ImuRow>,
}

/// Where one frameset's window sat, so the two implementations can be compared
/// pose by pose after the fact.
struct Divergence {
    /// Position difference of the newest state, metres.
    position_m: f64,
    /// Rotation difference of the newest state, degrees.
    rotation_deg: f64,
}

/// The C++'s newest state at this frameset.
fn newest(frame: &OracleFrame) -> &OracleState {
    frame
        .states
        .iter()
        .find(|state| state.t_ns == frame.last_state_t_ns)
        .unwrap_or_else(|| {
            panic!(
                "frame {}: the newest state is not in the window",
                frame.frame
            )
        })
}

/// Replay the whole flow stream and report where the two windows parted.
///
/// The numbers come back for the caller to gate: this is the measurement the report
/// quotes, and a long clip has to finish to produce it. The integer decisions
/// are counted the same way — the first break is what matters, not a panic at it.
fn compare<S: LieScalar>(
    oracle: &Oracle,
    run: &OracleRun,
    clip: &Path,
    out: Option<PathBuf>,
) -> (f64, usize) {
    let described: common::Clip = common::Clip::read(clip);
    // The oracle's frameset clock is the clip's `video_time`; every basalt CSV,
    // `gt.csv` included, is on the absolute device clock.
    let start_time_ns: i64 = described.capture_start_time_ns;
    let config: VioConfig = common::config_for(&described.dataset_name);
    let calibration: Calibration<f64> =
        Calibration::from_json_str(&std::fs::read_to_string(clip.join("calib.json")).unwrap())
            .unwrap();
    let imu: ImuFixture =
        serde_json::from_str(&std::fs::read_to_string(clip.join("imu.json")).unwrap()).unwrap();

    let mut estimator: SqrtKeypointVio<S> =
        SqrtKeypointVio::with_default_gravity(calibration.cast(), config).unwrap();
    for row in &imu.imu {
        estimator.push_imu(ImuSample {
            t_ns: row.t_ns,
            gyro: Vector3::from(row.gyro),
            accel: Vector3::from(row.accel),
        });
    }

    let mut poses: String = String::from("#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n");
    let mut worst: Divergence = Divergence {
        position_m: 0.0,
        rotation_deg: 0.0,
    };
    let mut worst_frame: usize = 0;
    let mut square_sum: f64 = 0.0;
    let mut first_past_mm: Option<(usize, i64, f64)> = None;
    let mut integer_breaks: BTreeMap<&'static str, (usize, String)> = BTreeMap::new();
    // The frameset `SLAM_RS_ORACLE_TRACE_FRAME` names gets both LM trails
    // printed, which is where a bug hunt starts when the two part company.
    let traced: Option<usize> = std::env::var("SLAM_RS_ORACLE_TRACE_FRAME")
        .ok()
        .and_then(|value| value.parse().ok());
    let started: Instant = Instant::now();

    for (index, expected) in run.frames.iter().enumerate() {
        let flow: &OracleFlow = oracle
            .flow
            .get(index)
            .unwrap_or_else(|| panic!("no flow stream for frame {index}"));
        assert_eq!(flow.t_ns, expected.t_ns, "frame {index}: timestamp");
        let FrameOutcome::Measured(stats) = estimator
            .process_frame(common::observations(flow))
            .unwrap_or_else(|error| panic!("frame {index} ({}) refused: {error}", expected.t_ns))
        else {
            panic!("frame {index} ({}) reported NeedMoreImu", expected.t_ns);
        };

        // The integer decisions the sixty-frameset gate asserts. Recorded, not
        // asserted: the first break is the answer, and the run has to reach the
        // end to say whether there is a second.
        let mut record = |what: &'static str, got: String, want: String| {
            if got != want {
                let entry = integer_breaks.entry(what).or_insert_with(|| {
                    (
                        0,
                        format!(
                            "first at frameset {index} (t={}): rust {got} vs c++ {want}",
                            expected.t_ns
                        ),
                    )
                });
                entry.0 += 1;
            }
        };
        record(
            "last_state_t_ns",
            estimator.last_state_t_ns().to_string(),
            expected.last_state_t_ns.to_string(),
        );
        record(
            "kf_ids",
            format!("{:?}", stats.kf_ids),
            format!("{:?}", expected.kf_ids),
        );
        record(
            "num_landmarks",
            stats.num_landmarks.to_string(),
            expected.num_landmarks.to_string(),
        );
        record(
            "num_observations",
            stats.num_observations.to_string(),
            expected.num_observations.to_string(),
        );
        record(
            "frames_after_kf",
            stats.frames_after_kf.to_string(),
            expected.frames_after_kf.to_string(),
        );
        record(
            "lm_steps",
            stats.lm.len().to_string(),
            expected.lm.len().to_string(),
        );
        // The marginalization schedule: which frames left the window, in which
        // set, and how the prior's indices split. A divergence that starts here
        // is a different question from one that starts in the arithmetic.
        record(
            "marg_fired",
            stats.marginalization.is_some().to_string(),
            expected.marg.is_some().to_string(),
        );
        if let (Some(got), Some(want)) = (&stats.marginalization, &expected.marg) {
            record(
                "marg_schedule",
                format!(
                    "{} {} {:?} {:?} {:?} {:?}",
                    got.states_to_remove,
                    got.last_state_to_marg,
                    got.poses_to_marg,
                    got.states_to_marg_all,
                    got.states_to_marg_vel_bias,
                    got.kfs_to_marg
                ),
                format!(
                    "{} {} {:?} {:?} {:?} {:?}",
                    want.states_to_remove,
                    want.last_state_to_marg,
                    want.poses_to_marg,
                    want.states_to_marg_all,
                    want.states_to_marg_vel_bias,
                    want.kfs_to_marg
                ),
            );
            record(
                "marg_indices",
                format!("{} {}", got.kept_indices, got.marg_indices),
                format!("{} {}", want.idx_to_keep, want.idx_to_marg),
            );
        }

        // Around the traced frameset, the prior is the one quantity carried
        // from the frame before, so its digest says whether a divergence
        // arrived with the marginalization or was made here.
        if traced.is_some_and(|frame| index + 5 >= frame && index <= frame + 5) {
            println!(
                "frameset {index} (t={}): prior rust |H|_F {:.10e} |b| {:.10e} {}x{}, c++ {:.10e} \
                 {:.10e} {}x{}; marg {}",
                expected.t_ns,
                common::frobenius(estimator.marg_data().h.iter().copied()),
                common::frobenius(estimator.marg_data().b.iter().copied()),
                estimator.marg_data().h.nrows(),
                estimator.marg_data().h.ncols(),
                expected.marg_digest.h_frobenius,
                expected.marg_digest.b_norm,
                expected.marg_digest.rows,
                expected.marg_digest.cols,
                expected.marg.is_some(),
            );
        }
        if traced == Some(index) {
            println!("frameset {index} (t={}), rust LM trail:", expected.t_ns);
            for step in &stats.lm {
                println!(
                    "  it={} j={} lambda={:?} error_before={:?} error_after={:?} l_diff={:?} \
                     f_diff={:?} accepted={} valid={} solves={}",
                    step.iteration,
                    step.backtrack,
                    step.lambda,
                    step.error_before,
                    step.error_after,
                    step.l_diff,
                    step.f_diff,
                    step.accepted,
                    step.step_is_valid,
                    step.solve_attempts
                );
            }
            println!("frameset {index}, c++ LM trail:");
            for step in &expected.lm {
                println!(
                    "  it={} j={} lambda={} error_before={} error_after={} l_diff={} f_diff={} \
                     accepted={} valid={} solves={}",
                    step.it,
                    step.backtrack,
                    step.lambda,
                    step.error_before,
                    step.error_after,
                    step.l_diff,
                    step.f_diff,
                    step.step_is_successful,
                    step.step_is_valid,
                    step.solve_attempts
                );
            }
            println!(
                "frameset {index}: rust keyframes {:?} landmarks {} observations {} \
                 frames_after_kf {}; c++ keyframes {:?} landmarks {} observations {} \
                 frames_after_kf {}",
                stats.kf_ids,
                stats.num_landmarks,
                stats.num_observations,
                stats.frames_after_kf,
                expected.kf_ids,
                expected.num_landmarks,
                expected.num_observations,
                expected.frames_after_kf
            );
        }
        let state = estimator
            .snapshot()
            .states
            .into_iter()
            .find(|state| state.t_ns == estimator.last_state_t_ns())
            .expect("the newest state is not in the window");
        let want: &OracleState = newest(expected);
        let position: Vector3<f64> = Vector3::new(
            state.t_w_i.translation.x.to_f64(),
            state.t_w_i.translation.y.to_f64(),
            state.t_w_i.translation.z.to_f64(),
        );
        let quaternion: [S; 4] = state.t_w_i.rotation.quaternion_xyzw();
        let mine: So3<f64> = So3::from_quaternion_xyzw(
            quaternion[0].to_f64(),
            quaternion[1].to_f64(),
            quaternion[2].to_f64(),
            quaternion[3].to_f64(),
        )
        .unwrap();
        let theirs: So3<f64> =
            So3::from_quaternion_xyzw(want.q[0], want.q[1], want.q[2], want.q[3]).unwrap();
        let offset: f64 = (position - Vector3::from(want.t)).norm();
        let angle: f64 = (mine.inverse() * theirs).log().norm().to_degrees();
        square_sum += offset * offset;
        if offset > worst.position_m {
            worst.position_m = offset;
            worst_frame = index;
        }
        worst.rotation_deg = worst.rotation_deg.max(angle);
        if offset > 1e-3 && first_past_mm.is_none() {
            first_past_mm = Some((index, expected.t_ns, offset));
        }
        writeln!(
            poses,
            "{},{:?},{:?},{:?},{:?},{:?},{:?},{:?}",
            expected.t_ns + start_time_ns,
            position.x,
            position.y,
            position.z,
            quaternion[3].to_f64(),
            quaternion[0].to_f64(),
            quaternion[1].to_f64(),
            quaternion[2].to_f64()
        )
        .unwrap();
        if (index + 1) % 200 == 0 {
            println!(
                "{}/{} framesets, here {offset:.3e} m, worst {:.3e} m, {} integer breaks, {:.1} s",
                index + 1,
                run.frames.len(),
                worst.position_m,
                integer_breaks
                    .values()
                    .map(|(count, _)| count)
                    .sum::<usize>(),
                started.elapsed().as_secs_f64()
            );
        }
    }

    let frames: f64 = run.frames.len() as f64;
    println!(
        "backend replay over {} framesets ({}): position rmse {:.4e} m, worst {:.4e} m at frameset \
         {worst_frame}, worst rotation {:.4e} deg, {:.1} s",
        run.frames.len(),
        run.scalar,
        (square_sum / frames).sqrt(),
        worst.position_m,
        worst.rotation_deg,
        started.elapsed().as_secs_f64()
    );
    match first_past_mm {
        Some((frame, t_ns, offset)) => println!(
            "first past 1 mm: frameset {frame} (t={t_ns}), {:.4} mm",
            offset * 1e3
        ),
        None => println!("never past 1 mm"),
    }
    if integer_breaks.is_empty() {
        println!(
            "integer decisions: no break in {} framesets",
            run.frames.len()
        );
    } else {
        println!(
            "integer decisions, per kind, over {} framesets:",
            run.frames.len()
        );
        for (what, (count, first)) in &integer_breaks {
            println!("  {what}: {count} frameset(s), {first}");
        }
    }
    if let Some(path) = out {
        std::fs::write(&path, poses).unwrap();
        println!("trajectory -> {}", path.display());
    }
    (
        worst.position_m,
        integer_breaks.values().map(|(count, _)| count).sum(),
    )
}

/// The whole clip's backend replay, in the precision
/// [`common::ClipScalar::from_env`] selects.
#[test]
fn the_whole_clip_backend_follows_the_cpp() {
    let Some(dump) = std::env::var_os("SLAM_RS_ORACLE_JSON").map(PathBuf::from) else {
        println!(
            "skipped: set SLAM_RS_ORACLE_JSON to a whole-clip basalt_vio_oracle dump and \
             SLAM_RS_CLIP_DIR to the clip it was produced from"
        );
        return;
    };
    let clip: PathBuf = std::env::var_os("SLAM_RS_CLIP_DIR")
        .map(PathBuf::from)
        .expect("SLAM_RS_CLIP_DIR must name the clip the dump was produced from");
    let started: Instant = Instant::now();
    let oracle: Oracle =
        serde_json::from_str(&std::fs::read_to_string(&dump).unwrap()).expect("oracle shape");
    println!(
        "{} parsed in {:.1} s: {} runs, {} framesets of flow",
        dump.display(),
        started.elapsed().as_secs_f64(),
        oracle.runs.len(),
        oracle.flow.len()
    );
    let scalar: common::ClipScalar = common::ClipScalar::from_env(common::ClipScalar::F64);
    let run: &OracleRun = common::run_named(&oracle, scalar.cpp_name());
    let out: Option<PathBuf> = std::env::var_os("SLAM_RS_ORACLE_OUT").map(PathBuf::from);
    let (worst_m, integer_breaks): (f64, usize) = match scalar {
        common::ClipScalar::F64 => compare::<f64>(&oracle, run, &clip, out),
        common::ClipScalar::F32 => compare::<f32>(&oracle, run, &clip, out),
    };
    if scalar == common::ClipScalar::F64 {
        assert!(
            worst_m <= F64_IDENTITY_M,
            "the backend left basalt's by {worst_m:.4e} m over {} framesets, past {F64_IDENTITY_M:.0e}",
            run.frames.len()
        );
        assert_eq!(
            integer_breaks,
            0,
            "the two windows took different discrete decisions over {} framesets",
            run.frames.len()
        );
    }
}
