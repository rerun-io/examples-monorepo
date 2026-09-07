//! The V2 gate for the sliding-window driver: the Rust estimator against
//! basalt's own, on the C++'s own visual input.
//!
//! ## Why the C++'s flow stream and not the port's
//!
//! The two frontends disagree on 3 to 4.5 % of keypoints because they use
//! different corner detectors (D09, D45), and a keypoint set that differs by one
//! id changes which landmarks get triangulated, which changes the window, which
//! changes every number after it. So this file isolates the **backend**: it
//! replays `tests/fixtures/vio/vio_oracle.json`'s `flow` stream — the
//! `OpticalFlowResult` the C++ backend was actually fed, ids, translations and
//! the 2x2 linear parts — into the Rust estimator and compares per frame, in
//! both precisions. `tests/vio_parity.rs` is the other half: the whole pipeline,
//! Rust frontend included, as a tolerance gate.
//!
//! ## What is asserted exactly and what is a tolerance
//!
//! Every **integer decision** is identical in both precisions and asserted as
//! such: the keyframe vote, `kf_ids`, `ltkfs`, `num_points_kf`, the landmark and
//! observation counts, the preintegrated-interval count, which frames are states
//! and which are poses, every fixed-linearization flag, the marginalization
//! schedule (which frames in which set, and the index split) and the prior's
//! `AbsOrderMap`. Over the 60 framesets the `f32` run breaks none of them.
//!
//! The **floating** comparisons are relative and the tolerances are the measured
//! agreement plus a margin; see the constants below for the numbers this run
//! produced.
//!
//! ## The one thing `f32` cannot reproduce: the LM accept decision
//!
//! In `f64` the LM trail is exact — the same step count, the same accept/reject
//! sequence, every number to 3.1e-10. In `f32` it is not, and cannot be. Two
//! facts, both basalt's, put the accept test below the noise floor:
//!
//! * The reduced system's right-hand side is formed by cancellation. Its
//!   velocity and bias rows are mathematically zero (they measure 1e-11 to
//!   1e-18 in `f64`), and its pose rows lose three to four significant digits:
//!   over the eight LM steps of frameset 4, C++'s own `float` `b` differs from
//!   its `double` `b` by 4.7e-4 to 5.8e-1 in relative 2-norm.
//! * The damped normal matrix `H + max(diag(H)·λ, λ_min)` has a condition number
//!   of 1.6e7 to 4.6e8 on those same eight steps, so `cond · eps(f32)` is 1.9 to
//!   56: the single-precision `LDLT` solve has no guaranteed significant digit.
//!
//! The port's own algebra is exact where the precision allows it — measured
//! against the fork's `hb_trace` hook on frameset 4: the dense `H` agrees with
//! Eigen's to 7.9e-17 in `f64` (one ulp) and to 1.0e-8 … 7.8e-8 in `f32` (at
//! most 0.65 `f32` eps), `b` to 3.1e-9 and `inc` to 2.5e-11 in `f64`. In `f32`
//! the port's `inc` differs from C++'s by 7.0e-5 … 9.7e-3 while C++'s differs
//! from its own `double` by 7.6e-5 … 2.5e-2 — the port is as close to Eigen as
//! Eigen is to the truth, and closer in four of the eight steps.
//!
//! What that does to the accept test: `f_diff = error_total − after_error_total`
//! is a difference of two ~1.2e3 quantities whose true value is ~1e-3, i.e. ten
//! `f32` ulps of the terms. Over the 60 framesets the two trails part company on
//! 21 of them, and on every one of those the deciding step's `f_diff` is between
//! −16.5 and +30.4 ulps of `error_before`. So the gate is: the trails agree up
//! to the first step whose accept-or-converge test is decided inside
//! [`F32_ACCEPT_NOISE_ULPS`], and the poses stay inside
//! [`POSE_TOLERANCE_F32`] regardless — measured worst over the 60 framesets:
//! rotation 8.4e-5, translation 9.9e-5, velocity 1.9e-4, bias 1.5e-3.
//!
//! ## The fixture, and the default window
//!
//! `basalt_vio_oracle` on the fork's `slam-rs-reference` branch, over the first
//! 60 framesets of the smoke reference segment
//! `msd-index__MIO_others__MIO10_short_2_panorama` with
//! `tests/fixtures/msdmi_config.json` and `msdmi_calib.json`, driving the real
//! threaded pipeline in lockstep with one TBB worker. The IMU window is
//! `tests/fixtures/vio/imu.json`, the same samples on the same clock. The PGM
//! frames are **not** committed (107 MB); only the frontend needs them, which is
//! why `vio_parity.rs` skips without them and this file does not need them at
//! all.
//!
//! All 60 framesets take about 55 s per run in a debug build, four runs of
//! which would be most of the Rust suite's budget, so the default replays
//! [`DEFAULT_FRAMESETS`] and `SLAM_RS_VIO_ORACLE_FULL=1` replays all 60.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nalgebra::Vector2;
use serde::Deserialize;

use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::estimator::{
    FlowObservations, FrameOutcome, FrameStats, LmIteration, SqrtKeypointVio,
};
use slam_rs::imu::ImuSample;
use slam_rs::lie::LieScalar;
use slam_rs::types::{FrameId, KeypointId};

// ── the window, and the tolerances measured on this fixture ───────────────

/// Framesets the fixture covers.
const ORACLE_FRAMESETS: usize = 60;

/// Framesets the default suite replays. Twelve reaches `opt_started` (frameset
/// 4), eight marginalizations, the second keyframe (7), both keyframe
/// demotions (4 and 9), both prior growth steps and the first `f32` LM-trail
/// divergence (9). The keyframe *eviction* loop first fires at frameset 51, so
/// only the full window covers it — plus the unit tests in
/// `src/estimator/schedule.rs`.
const DEFAULT_FRAMESETS: usize = 12;

/// Relative agreement on every pose, velocity and bias coefficient of the
/// window, in `f64`. Measured worst over the 60 framesets: rotation 9.8e-13,
/// translation 2.0e-13, velocity 6.8e-13, bias 2.6e-11.
const POSE_TOLERANCE_F64: f64 = 2e-10;
/// The same in `f32`. Measured worst: rotation 8.4e-5, translation 9.9e-5,
/// velocity 1.9e-4, bias 1.5e-3.
const POSE_TOLERANCE_F32: f64 = 3e-3;
/// Relative agreement on the LM error terms, `l_diff` and `lambda`, in `f64`.
/// Measured worst: 3.1e-10 (`error_before`).
const ERROR_TOLERANCE_F64: f64 = 2e-9;
/// The same in `f32`, over the trail prefix the two runs share, for the
/// quantities that are sums of well-conditioned terms: the reprojection cost,
/// the IMU and bias costs, the step's infinity norm and the prior's `H`.
/// Measured worst over the 60 framesets: 2.2e-3 (`imu_error`).
const ERROR_TOLERANCE_F32: f64 = 1e-2;
/// Relative agreement on the marginalization prior's Frobenius digest.
/// Measured worst in `f64`: 6.8e-15 on `H`, 1.1e-10 on `b`.
const PRIOR_TOLERANCE_F64: f64 = 2e-9;
/// Relative agreement in `f32` on the quantities the prior's cancelling half
/// drives. `marg_data.b` is `−H·delta` plus a residue three orders smaller
/// (8.30 against 0.12 at frameset 4), so it carries the accumulated `f32` error
/// of every increment the frozen blocks absorbed; `marg_prior_error` is its
/// bilinear form, `error_before` is that plus a reprojection cost two orders
/// smaller (1.2e3 against 1.3e2), and `l_diff` and `lambda` follow from the
/// gain ratio. Measured worst: 0.145 (`lambda`, whose Nielsen update cubes a
/// ratio whose numerator is a noise-level `f_diff`).
const CANCELLING_TOLERANCE_F32: f64 = 3e-1;

/// How far an accept-or-converge decision may be inside the `f32` noise floor
/// before the two LM trails are allowed to part company, in units of
/// `f32::EPSILON · |error_before|`.
///
/// The survey over the 60 framesets found 21 framesets where they do; the
/// widest deciding `f_diff` was 30.4 ulps (frameset 43) and the narrowest
/// −16.5 (frameset 58). 64 is that with room, and still two orders below the
/// 1e-3-scale decrease the `f64` run sees at those steps.
const F32_ACCEPT_NOISE_ULPS: f64 = 64.0;

/// Whether `SLAM_RS_VIO_ORACLE_FULL` asked for all 60 framesets.
fn framesets() -> usize {
    if std::env::var_os("SLAM_RS_VIO_ORACLE_FULL").is_some() {
        ORACLE_FRAMESETS
    } else {
        DEFAULT_FRAMESETS
    }
}

// ── the fixture's shape ───────────────────────────────────────────────────

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
    cameras: Vec<Vec<OraclePoint>>,
}

#[derive(Debug, Deserialize)]
struct OraclePoint {
    id: u64,
    x: f32,
    y: f32,
    #[allow(
        dead_code,
        reason = "the estimator reads only the translation; the linear part is dumped so a reader can see the whole warp"
    )]
    linear: [f32; 4],
}

#[derive(Debug, Deserialize)]
struct OracleFrame {
    frame: usize,
    t_ns: i64,
    states: Vec<OracleState>,
    poses: Vec<OraclePose>,
    kf_ids: Vec<i64>,
    ltkfs: Vec<i64>,
    num_points_kf: Vec<(i64, i64)>,
    last_state_t_ns: i64,
    take_kf: bool,
    frames_after_kf: i32,
    opt_started: bool,
    num_landmarks: usize,
    num_observations: usize,
    num_imu_meas: usize,
    marg_order: Vec<(i64, usize, usize)>,
    marg_digest: OracleDigest,
    marg: Option<OracleMarg>,
    lm: Vec<OracleLm>,
}

#[derive(Debug, Deserialize)]
struct OracleState {
    t_ns: i64,
    q: [f64; 4],
    t: [f64; 3],
    vel: [f64; 3],
    bg: [f64; 3],
    ba: [f64; 3],
    linearized: bool,
}

#[derive(Debug, Deserialize)]
struct OraclePose {
    t_ns: i64,
    q: [f64; 4],
    t: [f64; 3],
    linearized: bool,
}

#[derive(Debug, Deserialize)]
struct OracleDigest {
    rows: usize,
    cols: usize,
    h_frobenius: f64,
    b_norm: f64,
}

#[derive(Debug, Deserialize)]
struct OracleMarg {
    states_to_remove: usize,
    last_state_to_marg: i64,
    poses_to_marg: Vec<i64>,
    states_to_marg_all: Vec<i64>,
    states_to_marg_vel_bias: Vec<i64>,
    kfs_to_marg: Vec<i64>,
    idx_to_keep: usize,
    idx_to_marg: usize,
}

#[derive(Debug, Deserialize)]
struct OracleLm {
    it: i32,
    backtrack: i32,
    error_before: f64,
    error_after: f64,
    vision_error: f64,
    imu_error: f64,
    bg_error: f64,
    ba_error: f64,
    marg_prior_error: f64,
    l_diff: f64,
    f_diff: f64,
    lambda: f64,
    step_norminf: f64,
    solve_attempts: u32,
    step_is_valid: bool,
    step_is_successful: bool,
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

// ── loading ───────────────────────────────────────────────────────────────

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn oracle() -> Oracle {
    let path: PathBuf = fixtures().join("vio/vio_oracle.json");
    let text: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    serde_json::from_str(&text).expect("vio_oracle.json does not match the expected shape")
}

fn imu_window() -> Vec<ImuSample> {
    let path: PathBuf = fixtures().join("vio/imu.json");
    let text: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let fixture: ImuFixture =
        serde_json::from_str(&text).expect("imu.json does not match the expected shape");
    fixture
        .imu
        .into_iter()
        .map(|row| ImuSample {
            t_ns: row.t_ns,
            gyro: nalgebra::Vector3::new(row.gyro[0], row.gyro[1], row.gyro[2]),
            accel: nalgebra::Vector3::new(row.accel[0], row.accel[1], row.accel[2]),
        })
        .collect()
}

fn config() -> VioConfig {
    let text: String = std::fs::read_to_string(fixtures().join("msdmi_config.json")).unwrap();
    VioConfig::from_json_str(&text).unwrap()
}

fn calibration() -> Calibration<f64> {
    let text: String = std::fs::read_to_string(fixtures().join("msdmi_calib.json")).unwrap();
    Calibration::from_json_str(&text).unwrap()
}

fn observations(flow: &OracleFlow) -> Arc<FlowObservations> {
    let mut out: FlowObservations = FlowObservations::new(flow.t_ns, flow.cameras.len());
    for (cam_id, points) in flow.cameras.iter().enumerate() {
        let Some(slot) = out.cameras.get_mut(cam_id) else {
            continue;
        };
        for point in points {
            slot.insert(KeypointId(point.id), Vector2::new(point.x, point.y));
        }
    }
    Arc::new(out)
}

// ── the comparison ────────────────────────────────────────────────────────

/// `|a − b| / max(1, |b|)`, so a quantity near zero is compared absolutely and
/// a large one relatively. Both non-finite is agreement; one non-finite is not.
fn relative(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    if !a.is_finite() || !b.is_finite() {
        return f64::INFINITY;
    }
    (a - b).abs() / b.abs().max(1.0)
}

/// The worst relative difference seen, per quantity, so the report can quote
/// measurements rather than the tolerance it happened to pass at.
#[derive(Debug, Clone, Copy, Default)]
struct Worst {
    rotation: f64,
    translation: f64,
    velocity: f64,
    bias: f64,
    error_before: f64,
    error_after: f64,
    vision_error: f64,
    imu_error: f64,
    bias_error: f64,
    marg_prior_error: f64,
    l_diff: f64,
    lambda: f64,
    step_norminf: f64,
    prior_h: f64,
    prior_b: f64,
}

impl Worst {
    fn take(slot: &mut f64, a: f64, b: f64) {
        let value: f64 = relative(a, b);
        if value > *slot {
            *slot = value;
        }
    }
}

/// How much of the LM trail the precision can be held to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LmGate {
    /// Every step, exactly: the same count and the same accept/reject sequence.
    Exact,
    /// The trail may part company at a step whose accept-or-converge test is
    /// decided inside [`F32_ACCEPT_NOISE_ULPS`]; see the module header.
    NoiseFloor,
}

/// Drive the window and return the worst relative difference per quantity,
/// having asserted every integer decision on the way.
fn compare<S: LieScalar>(oracle: &Oracle, run: &OracleRun, gate: LmGate) -> Worst {
    let mut estimator: SqrtKeypointVio<S> =
        SqrtKeypointVio::with_default_gravity(calibration().cast(), config()).unwrap();
    for sample in imu_window() {
        estimator.push_imu(sample);
    }

    let mut diverged: Vec<i64> = Vec::new();
    let mut worst: Worst = Worst::default();
    for (index, expected) in run.frames.iter().take(framesets()).enumerate() {
        let flow: &OracleFlow = oracle
            .flow
            .get(index)
            .unwrap_or_else(|| panic!("no flow stream for frame {index}"));
        assert_eq!(flow.t_ns, expected.t_ns, "frame {index}: timestamp");

        let outcome: FrameOutcome<S> = estimator
            .process_frame(observations(flow))
            .unwrap_or_else(|error| panic!("frame {index} ({}) refused: {error}", expected.t_ns));
        let FrameOutcome::Measured(stats) = outcome else {
            panic!("frame {index} ({}) reported NeedMoreImu", expected.t_ns);
        };

        let where_: String = format!("{} frame {index} (t={})", run.scalar, expected.t_ns);

        // ── the integer decisions ─────────────────────────────────────────
        assert_eq!(
            estimator.last_state_t_ns(),
            expected.last_state_t_ns,
            "{where_}: last_state_t_ns"
        );
        assert_eq!(
            estimator.kf_ids().collect::<Vec<FrameId>>(),
            expected.kf_ids,
            "{where_}: kf_ids"
        );
        assert_eq!(
            estimator.ltkfs().collect::<Vec<FrameId>>(),
            expected.ltkfs,
            "{where_}: ltkfs"
        );
        let num_points_kf: Vec<(i64, i64)> = estimator
            .num_points_kf()
            .iter()
            .map(|(t, count)| (*t, i64::try_from(*count).unwrap()))
            .collect();
        assert_eq!(
            num_points_kf, expected.num_points_kf,
            "{where_}: num_points_kf"
        );
        assert_eq!(
            stats.num_landmarks, expected.num_landmarks,
            "{where_}: landmark count"
        );
        assert_eq!(
            stats.num_observations, expected.num_observations,
            "{where_}: observation count"
        );
        assert_eq!(
            estimator.imu_meas().len(),
            expected.num_imu_meas,
            "{where_}: preintegrated intervals"
        );
        assert_eq!(
            estimator.optimization_started(),
            expected.opt_started,
            "{where_}: opt_started"
        );

        // The window itself: the same frames, as the same kind of block, with
        // the same fixed-linearization flags.
        let states: Vec<i64> = estimator
            .snapshot()
            .states
            .iter()
            .map(|state| state.t_ns)
            .collect();
        assert_eq!(
            states,
            expected.states.iter().map(|s| s.t_ns).collect::<Vec<i64>>(),
            "{where_}: frame_states"
        );
        let poses: Vec<i64> = estimator
            .snapshot()
            .poses
            .iter()
            .map(|pose| pose.t_ns)
            .collect();
        assert_eq!(
            poses,
            expected.poses.iter().map(|p| p.t_ns).collect::<Vec<i64>>(),
            "{where_}: frame_poses"
        );

        // The LM trail: in `f64` the same number of steps and the same
        // accept/reject sequence, the strictest integer statement in the file;
        // in `f32` the same up to the first decision inside the noise floor.
        let shared: usize = lm_prefix(&where_, &stats.lm, &expected.lm, gate);
        if shared < stats.lm.len().max(expected.lm.len()) {
            diverged.push(expected.t_ns);
        }
        for (step, want) in stats.lm.iter().zip(expected.lm.iter()).take(shared) {
            let step_where: String = format!("{where_} it={} j={}", want.it, want.backtrack);
            assert_eq!(step.iteration, want.it, "{step_where}: iteration index");
            assert_eq!(
                step.backtrack, want.backtrack,
                "{step_where}: backtrack index"
            );
            assert_eq!(
                step.accepted, want.step_is_successful,
                "{step_where}: accepted"
            );
            assert_eq!(
                step.step_is_valid, want.step_is_valid,
                "{step_where}: valid"
            );
            assert_eq!(
                step.solve_attempts, want.solve_attempts,
                "{step_where}: damped solves"
            );
        }

        // The marginalization schedule.
        match (&stats.marginalization, &expected.marg) {
            (None, None) => {}
            (Some(got), Some(want)) => {
                assert_eq!(
                    got.states_to_remove, want.states_to_remove,
                    "{where_}: states_to_remove"
                );
                assert_eq!(
                    got.last_state_to_marg, want.last_state_to_marg,
                    "{where_}: last_state_to_marg"
                );
                assert_eq!(
                    got.poses_to_marg, want.poses_to_marg,
                    "{where_}: poses_to_marg"
                );
                assert_eq!(
                    got.states_to_marg_all, want.states_to_marg_all,
                    "{where_}: states_to_marg_all"
                );
                assert_eq!(
                    got.states_to_marg_vel_bias, want.states_to_marg_vel_bias,
                    "{where_}: states_to_marg_vel_bias"
                );
                assert_eq!(got.kfs_to_marg, want.kfs_to_marg, "{where_}: kfs_to_marg");
                assert_eq!(got.kept_indices, want.idx_to_keep, "{where_}: idx_to_keep");
                assert_eq!(got.marg_indices, want.idx_to_marg, "{where_}: idx_to_marg");
            }
            (got, want) => panic!(
                "{where_}: marginalization fired on one side only (rust {}, c++ {})",
                got.is_some(),
                want.is_some()
            ),
        }

        // The prior's ordering, which is the one thing every later frame's
        // assertion in `optimize` depends on.
        let order: Vec<(i64, usize, usize)> = estimator.marg_data().order.iter().collect();
        assert_eq!(order, expected.marg_order, "{where_}: marg_data.order");
        assert_eq!(
            (
                estimator.marg_data().h.nrows(),
                estimator.marg_data().h.ncols()
            ),
            (expected.marg_digest.rows, expected.marg_digest.cols),
            "{where_}: prior shape"
        );

        // ── the floating agreement ────────────────────────────────────────
        for (state, want) in estimator
            .snapshot()
            .states
            .iter()
            .zip(expected.states.iter())
        {
            let q: [S; 4] = state.t_w_i.rotation.quaternion_xyzw();
            for (got, want) in q.iter().zip(want.q.iter()) {
                Worst::take(&mut worst.rotation, got.to_f64(), *want);
            }
            for i in 0..3 {
                Worst::take(
                    &mut worst.translation,
                    state.t_w_i.translation[i].to_f64(),
                    want.t[i],
                );
                if let Some(vel) = state.vel_w_i {
                    Worst::take(&mut worst.velocity, vel[i].to_f64(), want.vel[i]);
                }
                if let (Some(bg), Some(ba)) = (state.bias_gyro, state.bias_accel) {
                    Worst::take(&mut worst.bias, bg[i].to_f64(), want.bg[i]);
                    Worst::take(&mut worst.bias, ba[i].to_f64(), want.ba[i]);
                }
            }
            assert_eq!(
                state.linearized, want.linearized,
                "{where_}: linearized flag of state {}",
                want.t_ns
            );
        }
        for (pose, want) in estimator.snapshot().poses.iter().zip(expected.poses.iter()) {
            let q: [S; 4] = pose.t_w_i.rotation.quaternion_xyzw();
            for (got, want) in q.iter().zip(want.q.iter()) {
                Worst::take(&mut worst.rotation, got.to_f64(), *want);
            }
            for i in 0..3 {
                Worst::take(
                    &mut worst.translation,
                    pose.t_w_i.translation[i].to_f64(),
                    want.t[i],
                );
            }
            assert_eq!(
                pose.linearized, want.linearized,
                "{where_}: linearized flag of pose {}",
                want.t_ns
            );
        }

        for (step, want) in stats.lm.iter().zip(expected.lm.iter()).take(shared) {
            Worst::take(
                &mut worst.error_before,
                step.error_before.to_f64(),
                want.error_before,
            );
            Worst::take(
                &mut worst.error_after,
                step.error_after.to_f64(),
                want.error_after,
            );
            Worst::take(
                &mut worst.vision_error,
                step.vision_error.to_f64(),
                want.vision_error,
            );
            Worst::take(
                &mut worst.imu_error,
                step.imu_error.to_f64(),
                want.imu_error,
            );
            Worst::take(
                &mut worst.bias_error,
                step.bias_gyro_error.to_f64(),
                want.bg_error,
            );
            Worst::take(
                &mut worst.bias_error,
                step.bias_accel_error.to_f64(),
                want.ba_error,
            );
            Worst::take(
                &mut worst.marg_prior_error,
                step.marg_prior_error.to_f64(),
                want.marg_prior_error,
            );
            Worst::take(&mut worst.l_diff, step.l_diff.to_f64(), want.l_diff);
            Worst::take(&mut worst.lambda, step.lambda.to_f64(), want.lambda);
            Worst::take(
                &mut worst.step_norminf,
                step.step_norminf.to_f64(),
                want.step_norminf,
            );
            let _ = want.f_diff;
        }

        Worst::take(
            &mut worst.prior_h,
            frobenius(&estimator.marg_data().h),
            expected.marg_digest.h_frobenius,
        );
        Worst::take(
            &mut worst.prior_b,
            frobenius_vec(&estimator.marg_data().b),
            expected.marg_digest.b_norm,
        );

        assert_eq!(
            expected.frame, index,
            "the fixture's frames are out of order"
        );
        let _ = (expected.take_kf, expected.frames_after_kf);
    }
    if !diverged.is_empty() {
        println!(
            "{}: LM trail parted company on {} of {} framesets, all inside the noise floor: {diverged:?}",
            run.scalar,
            diverged.len(),
            framesets()
        );
    }
    worst
}

/// How many leading LM steps the two trails share, and — when they do not share
/// all of them — the proof that the step they part on was decided inside the
/// `f32` noise floor.
///
/// The first differing step is the one whose accept-or-converge test went the
/// other way, so it is the step whose `f_diff` has to be at the noise floor;
/// when one trail is shorter the other side terminated on `:1566`'s
/// convergence test, and the last step it ran is the one that decided it.
fn lm_prefix<S: LieScalar>(
    where_: &str,
    got: &[LmIteration<S>],
    want: &[OracleLm],
    gate: LmGate,
) -> usize {
    let sequence = |steps: &[LmIteration<S>]| -> Vec<(i32, i32, bool)> {
        steps
            .iter()
            .map(|step| (step.iteration, step.backtrack, step.accepted))
            .collect()
    };
    let expected: Vec<(i32, i32, bool)> = want
        .iter()
        .map(|step| (step.it, step.backtrack, step.step_is_successful))
        .collect();
    let mine: Vec<(i32, i32, bool)> = sequence(got);
    if mine == expected {
        return got.len();
    }
    let at: usize = mine
        .iter()
        .zip(expected.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| mine.len().min(expected.len()));
    assert_eq!(
        gate,
        LmGate::NoiseFloor,
        "{where_}: LM trail diverged at step {at} (rust {mine:?} vs c++ {expected:?})"
    );

    // The last step each side actually ran up to and including `at`.
    let ours: &LmIteration<S> = &got[at.min(got.len() - 1)];
    let theirs: &OracleLm = &want[at.min(want.len() - 1)];
    let floor: f64 = f64::from(f32::EPSILON) * ours.error_before.to_f64().abs();
    let ulps = |f_diff: f64| -> f64 { f_diff / floor };
    assert!(
        ulps(ours.f_diff.to_f64()).abs() <= F32_ACCEPT_NOISE_ULPS
            && ulps(theirs.f_diff).abs() <= F32_ACCEPT_NOISE_ULPS,
        "{where_}: the LM trails parted at step {at} on a decision outside the f32 noise floor \
         (rust f_diff {:.4e} = {:.2} ulp, c++ {:.4e} = {:.2} ulp, one ulp of error_before {:.4e} \
         is {floor:.4e}); rust {mine:?} vs c++ {expected:?}",
        ours.f_diff.to_f64(),
        ulps(ours.f_diff.to_f64()),
        theirs.f_diff,
        ulps(theirs.f_diff),
        ours.error_before.to_f64(),
    );
    at
}

fn frobenius<S: LieScalar>(m: &nalgebra::DMatrix<S>) -> f64 {
    m.iter()
        .map(|v| v.to_f64() * v.to_f64())
        .sum::<f64>()
        .sqrt()
}

fn frobenius_vec<S: LieScalar>(v: &nalgebra::DVector<S>) -> f64 {
    v.iter()
        .map(|x| x.to_f64() * x.to_f64())
        .sum::<f64>()
        .sqrt()
}

fn run_named<'a>(oracle: &'a Oracle, scalar: &str) -> &'a OracleRun {
    oracle
        .runs
        .iter()
        .find(|run| run.scalar == scalar)
        .unwrap_or_else(|| panic!("the fixture has no {scalar} run"))
}

// ── the gates ─────────────────────────────────────────────────────────────

#[test]
fn the_double_window_follows_the_cpp() {
    let oracle: Oracle = oracle();
    let worst: Worst = compare::<f64>(&oracle, run_named(&oracle, "double"), LmGate::Exact);
    println!("f64 worst relative difference: {worst:#?}");

    assert!(
        worst.rotation <= POSE_TOLERANCE_F64 && worst.translation <= POSE_TOLERANCE_F64,
        "f64 pose drifted: {worst:#?}"
    );
    assert!(
        worst.velocity <= POSE_TOLERANCE_F64 && worst.bias <= POSE_TOLERANCE_F64,
        "f64 velocity or bias drifted: {worst:#?}"
    );
    assert!(
        worst.error_before <= ERROR_TOLERANCE_F64
            && worst.error_after <= ERROR_TOLERANCE_F64
            && worst.vision_error <= ERROR_TOLERANCE_F64
            && worst.imu_error <= ERROR_TOLERANCE_F64
            && worst.bias_error <= ERROR_TOLERANCE_F64
            && worst.marg_prior_error <= ERROR_TOLERANCE_F64
            && worst.l_diff <= ERROR_TOLERANCE_F64
            && worst.lambda <= ERROR_TOLERANCE_F64
            && worst.step_norminf <= ERROR_TOLERANCE_F64,
        "f64 LM trail drifted: {worst:#?}"
    );
    assert!(
        worst.prior_h <= PRIOR_TOLERANCE_F64 && worst.prior_b <= PRIOR_TOLERANCE_F64,
        "f64 prior drifted: {worst:#?}"
    );
}

/// The `f32` lane, which is the precision basalt ships (Q07).
///
/// Every integer decision is still exact; the LM trail is held to the noise
/// floor and the poses to the measured tolerance. See the module header for why
/// the accept test cannot be reproduced and what was measured.
#[test]
fn the_float_window_follows_the_cpp() {
    let oracle: Oracle = oracle();
    let worst: Worst = compare::<f32>(&oracle, run_named(&oracle, "float"), LmGate::NoiseFloor);
    println!("f32 worst relative difference: {worst:#?}");

    assert!(
        worst.rotation <= POSE_TOLERANCE_F32 && worst.translation <= POSE_TOLERANCE_F32,
        "f32 pose drifted: {worst:#?}"
    );
    assert!(
        worst.velocity <= POSE_TOLERANCE_F32 && worst.bias <= POSE_TOLERANCE_F32,
        "f32 velocity or bias drifted: {worst:#?}"
    );
    assert!(
        worst.vision_error <= ERROR_TOLERANCE_F32
            && worst.imu_error <= ERROR_TOLERANCE_F32
            && worst.bias_error <= ERROR_TOLERANCE_F32
            && worst.step_norminf <= ERROR_TOLERANCE_F32
            && worst.prior_h <= ERROR_TOLERANCE_F32,
        "f32 LM trail drifted: {worst:#?}"
    );
    assert!(
        worst.error_before <= CANCELLING_TOLERANCE_F32
            && worst.error_after <= CANCELLING_TOLERANCE_F32
            && worst.marg_prior_error <= CANCELLING_TOLERANCE_F32
            && worst.l_diff <= CANCELLING_TOLERANCE_F32
            && worst.lambda <= CANCELLING_TOLERANCE_F32
            && worst.prior_b <= CANCELLING_TOLERANCE_F32,
        "f32 prior-driven quantities drifted: {worst:#?}"
    );
}

/// Two runs over the same input are bit-identical (D17).
///
/// Not "close": every coefficient of every pose and every LM number is compared
/// with `==`, because nothing in Offline mode may let arrival order, a hash seed
/// or a thread schedule reach a decision.
#[test]
fn a_repeat_run_is_bit_identical() {
    let oracle: Oracle = oracle();
    let flow: Vec<Arc<FlowObservations>> = oracle
        .flow
        .iter()
        .take(framesets())
        .map(observations)
        .collect();

    let first: Vec<Trace> = drive(&flow);
    let second: Vec<Trace> = drive(&flow);
    assert_eq!(first.len(), flow.len());
    assert_eq!(first, second, "a repeat run differed");
}

/// What the determinism test compares: everything a caller can observe except
/// the wall-clock timings, which are measurements and not decisions.
#[derive(Debug, Clone, PartialEq)]
struct Trace {
    t_ns: i64,
    kf_ids: Vec<FrameId>,
    ltkfs: Vec<FrameId>,
    landmarks: usize,
    observations: usize,
    poses: Vec<(i64, [f32; 4], [f32; 3])>,
    lm: Vec<(i32, i32, f32, f32, f32, bool)>,
    prior: Vec<f32>,
}

fn drive(flow: &[Arc<FlowObservations>]) -> Vec<Trace> {
    let mut estimator: SqrtKeypointVio<f32> =
        SqrtKeypointVio::with_default_gravity(calibration().cast(), config()).unwrap();
    for sample in imu_window() {
        estimator.push_imu(sample);
    }
    let mut traces: Vec<Trace> = Vec::new();
    for frame in flow {
        let outcome: FrameOutcome<f32> = estimator.process_frame(Arc::clone(frame)).unwrap();
        let FrameOutcome::Measured(stats) = outcome else {
            panic!("NeedMoreImu at {}", frame.t_ns);
        };
        let stats: FrameStats<f32> = *stats;
        let snapshot = estimator.snapshot();
        traces.push(Trace {
            t_ns: stats.t_ns,
            kf_ids: stats.kf_ids,
            ltkfs: stats.ltkfs,
            landmarks: stats.num_landmarks,
            observations: stats.num_observations,
            poses: snapshot
                .states
                .iter()
                .chain(snapshot.poses.iter())
                .map(|state| {
                    (
                        state.t_ns,
                        state.t_w_i.rotation.quaternion_xyzw(),
                        state.t_w_i.translation.into(),
                    )
                })
                .collect(),
            lm: stats
                .lm
                .iter()
                .map(|step| {
                    (
                        step.iteration,
                        step.backtrack,
                        step.error_after,
                        step.l_diff,
                        step.lambda,
                        step.accepted,
                    )
                })
                .collect(),
            prior: estimator.marg_data().h.iter().copied().collect(),
        });
    }
    traces
}

/// The window stays inside the budget basalt actually enforces, and the index
/// sets the schedule produces are a partition of the ordering.
///
/// A property rather than a fixture comparison: it holds for every frame of the
/// run, not only the ones the oracle covers, and it is the invariant a schedule
/// bug breaks first.
///
/// The keyframe budget is **lazy**, which is the one thing to get right here.
/// `sqrt_keypoint_vio.cpp:767` runs the eviction loop only while
/// `!states_to_marg_vel_bias.empty()`, and that set holds the keyframes leaving
/// the *state* window this step — so a frame that has just been voted a
/// keyframe cannot be evicted while it is still a state, and `kf_ids` sits one
/// over `max_kfs` until it is demoted to a pose block. On this fixture that is
/// framesets 49-50 and 56-57 (eight keyframes against `vio_max_kfs = 7`), and
/// the C++ oracle shows exactly the same eight. The loop's postcondition is
/// therefore `kf_ids ≤ max_kfs || states_to_marg_vel_bias.is_empty()`, and the
/// overshoot is at most one because `vio_min_frames_after_kf = 5` puts
/// keyframes six framesets apart while a state leaves the window after
/// `vio_max_states = 3`.
#[test]
fn the_window_stays_inside_its_budget() {
    let oracle: Oracle = oracle();
    let config: VioConfig = config();
    let max_states: usize = usize::try_from(config.vio_max_states).unwrap();
    let max_kfs: usize = usize::try_from(config.vio_max_kfs).unwrap();

    let mut estimator: SqrtKeypointVio<f64> =
        SqrtKeypointVio::with_default_gravity(calibration().cast(), config).unwrap();
    for sample in imu_window() {
        estimator.push_imu(sample);
    }
    for flow in oracle.flow.iter().take(framesets()) {
        let FrameOutcome::Measured(stats) = estimator.process_frame(observations(flow)).unwrap()
        else {
            panic!("NeedMoreImu at {}", flow.t_ns);
        };
        let ltkfs: usize = stats.ltkfs.len();
        if stats.opt_started {
            assert!(
                estimator.ba.frame_states.len() <= max_states,
                "frame {}: {} states exceed the budget",
                flow.t_ns,
                estimator.ba.frame_states.len()
            );
            let could_evict: bool = stats
                .marginalization
                .as_ref()
                .is_some_and(|marg| !marg.states_to_marg_vel_bias.is_empty());
            let budget: usize = if could_evict { max_kfs } else { max_kfs + 1 };
            assert!(
                stats.kf_ids.len() <= budget,
                "frame {}: {} keyframes exceed the budget of {budget} (a keyframe state {} \
                 demoted this step)",
                flow.t_ns,
                stats.kf_ids.len(),
                if could_evict { "was" } else { "was not" }
            );
            assert!(
                estimator.ba.frame_poses.len() <= ltkfs + max_kfs,
                "frame {}: {} poses exceed the budget",
                flow.t_ns,
                estimator.ba.frame_poses.len()
            );
        }

        if let Some(marg) = &stats.marginalization {
            // The four sets are disjoint, and every frame in them was in the
            // window when the schedule was built.
            let mut seen: BTreeSet<i64> = BTreeSet::new();
            for id in marg
                .states_to_marg_all
                .iter()
                .chain(marg.states_to_marg_vel_bias.iter())
            {
                assert!(seen.insert(*id), "frame {} listed twice", *id);
            }
            assert!(
                !marg.states_to_marg_all.contains(&marg.last_state_to_marg),
                "the frozen state cannot also be removed"
            );
            for id in &marg.kfs_to_marg {
                assert!(
                    marg.poses_to_marg.contains(id),
                    "evicted keyframe {id} is not in poses_to_marg"
                );
            }
            // The index split covers the whole ordering exactly once.
            assert_eq!(
                marg.kept_indices + marg.marg_indices,
                marg.ordering_size,
                "the index split does not cover the {}-wide ordering: kept {} marg {}",
                marg.ordering_size,
                marg.kept_indices,
                marg.marg_indices
            );
            // What survives is the new prior, whose width is what the kept
            // indices amount to only when no rank was lost in the flat QR.
            assert!(
                marg.prior_order
                    .iter()
                    .map(|(_, _, size)| size)
                    .sum::<usize>()
                    == marg.kept_indices,
                "the new ordering is {} wide but {} indices were kept",
                marg.prior_order
                    .iter()
                    .map(|(_, _, size)| size)
                    .sum::<usize>(),
                marg.kept_indices
            );
        }
    }

    // Every keyframe ever created keeps its entry, exactly as basalt never
    // erases `num_points_kf` (`:218`, the small leak trap 15 warns about).
    let counts: &BTreeMap<FrameId, usize> = estimator.num_points_kf();
    assert!(
        counts.len() >= estimator.kf_ids().count(),
        "num_points_kf lost an entry"
    );
}
