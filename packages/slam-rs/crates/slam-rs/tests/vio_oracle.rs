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
//! Every **integer decision** must be identical, in both precisions: the
//! keyframe vote, `kf_ids`, `ltkfs`, `num_points_kf`, the landmark and
//! observation counts, the LM iteration count and its accept/reject sequence,
//! the marginalization schedule (which frames in which set) and the prior's
//! `AbsOrderMap`. Those are what a divergence shows up in first, and none of
//! them is allowed to drift.
//!
//! The **floating** comparisons are relative and the tolerances are the measured
//! agreement plus a margin; see the constants below for the numbers this run
//! produced.
//!
//! ## The fixture
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

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nalgebra::Vector2;
use serde::Deserialize;

use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::estimator::{FlowObservations, FrameOutcome, FrameStats, SqrtKeypointVio};
use slam_rs::imu::ImuSample;
use slam_rs::lie::LieScalar;
use slam_rs::types::{FrameId, KeypointId};

// ── the tolerances, measured on this fixture ──────────────────────────────

/// Relative agreement on every pose, velocity and bias coefficient of the
/// window, in `f64`.
const POSE_TOLERANCE_F64: f64 = 2e-10;
/// The same in `f32`.
const POSE_TOLERANCE_F32: f64 = 3e-3;
/// Relative agreement on the LM error terms, `l_diff` and `lambda`, in `f64`.
const ERROR_TOLERANCE_F64: f64 = 2e-9;
/// The same in `f32`.
const ERROR_TOLERANCE_F32: f64 = 5e-3;
/// Relative agreement on the marginalization prior's Frobenius digest.
const PRIOR_TOLERANCE_F64: f64 = 2e-9;
/// The same in `f32`.
const PRIOR_TOLERANCE_F32: f64 = 5e-3;

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

/// Drive the whole 60-frame window and return the worst relative difference per
/// quantity, having asserted every integer decision on the way.
fn compare<S: LieScalar>(oracle: &Oracle, run: &OracleRun) -> Worst {
    let mut estimator: SqrtKeypointVio<S> =
        SqrtKeypointVio::with_default_gravity(calibration().cast(), config()).unwrap();
    for sample in imu_window() {
        estimator.push_imu(sample);
    }

    let mut worst: Worst = Worst::default();
    for (index, expected) in run.frames.iter().enumerate() {
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

        // The LM trail: the same number of steps and the same accept/reject
        // sequence, which is the strictest integer statement in the file.
        assert_eq!(
            stats.lm.len(),
            expected.lm.len(),
            "{where_}: LM step count (rust {:?} vs c++ {:?})",
            stats
                .lm
                .iter()
                .map(|step| (step.iteration, step.accepted))
                .collect::<Vec<(i32, bool)>>(),
            expected
                .lm
                .iter()
                .map(|step| (step.it, step.step_is_successful))
                .collect::<Vec<(i32, bool)>>()
        );
        for (step, want) in stats.lm.iter().zip(expected.lm.iter()) {
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
            for i in 0..4 {
                Worst::take(&mut worst.rotation, q[i].to_f64(), want.q[i]);
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
            for i in 0..4 {
                Worst::take(&mut worst.rotation, q[i].to_f64(), want.q[i]);
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

        for (step, want) in stats.lm.iter().zip(expected.lm.iter()) {
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
    worst
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
    let worst: Worst = compare::<f64>(&oracle, run_named(&oracle, "double"));
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

#[test]
fn the_float_window_follows_the_cpp() {
    let oracle: Oracle = oracle();
    let worst: Worst = compare::<f32>(&oracle, run_named(&oracle, "float"));
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
        worst.error_before <= ERROR_TOLERANCE_F32
            && worst.error_after <= ERROR_TOLERANCE_F32
            && worst.vision_error <= ERROR_TOLERANCE_F32
            && worst.imu_error <= ERROR_TOLERANCE_F32
            && worst.bias_error <= ERROR_TOLERANCE_F32
            && worst.marg_prior_error <= ERROR_TOLERANCE_F32
            && worst.l_diff <= ERROR_TOLERANCE_F32
            && worst.lambda <= ERROR_TOLERANCE_F32
            && worst.step_norminf <= ERROR_TOLERANCE_F32,
        "f32 LM trail drifted: {worst:#?}"
    );
    assert!(
        worst.prior_h <= PRIOR_TOLERANCE_F32 && worst.prior_b <= PRIOR_TOLERANCE_F32,
        "f32 prior drifted: {worst:#?}"
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
    let flow: Vec<Arc<FlowObservations>> = oracle.flow.iter().map(observations).collect();

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

/// The window never exceeds the configured budget, and the index sets the
/// schedule produces are a partition of the ordering.
///
/// A property rather than a fixture comparison: it holds for every frame of the
/// run, not only the sixty the oracle covers, and it is the invariant a schedule
/// bug breaks first.
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
    for flow in &oracle.flow {
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
            assert!(
                stats.kf_ids.len() <= max_kfs,
                "frame {}: {} keyframes exceed the budget",
                flow.t_ns,
                stats.kf_ids.len()
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
