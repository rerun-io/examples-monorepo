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
//! and which are poses, every fixed-linearization flag, `frames_after_kf`, the
//! marginalization schedule (which frames in which set, and the index split)
//! and the prior's `AbsOrderMap`. Over the 60 framesets the `f32` run breaks
//! none of them. The vote itself is asserted through its effects — `kf_ids` and
//! `num_points_kf` gain an entry exactly when it fires — because the fixture's
//! `take_kf` is dumped *after* `measure` consumed it and is therefore always
//! false; the field stays in the fixture for a reader and is not read here.
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
//! rotation 8.4e-5, translation 7.8e-5, velocity 2.3e-4, bias 9.3e-4.
//!
//! Which framesets diverge is itself chaotic: it is 23 of the 60 as this ships,
//! and changing one reduction order inside the prior's error moved the set
//! (21 framesets, a different 21) without moving a single integer decision.
//!
//! ## The fixture, and the default window
//!
//! `basalt_vio_oracle` on the fork's `slam-rs-reference` branch, over the first
//! 60 framesets of the smoke reference segment
//! `msd-index__MIO_others__MIO10_short_2_panorama` with
//! `configs/msdmi_config.json` and `tests/fixtures/msdmi_calib.json`, driving the real
//! threaded pipeline in lockstep with one TBB worker. The IMU window is
//! `tests/fixtures/vio/imu.json`, the same samples on the same clock. The PGM
//! frames are **not** committed (107 MB); only the frontend needs them, which is
//! why `vio_parity.rs` skips without them and this file does not need them at
//! all.
//!
//! Every lane replays all 60 framesets, which is 2.6 s for the three tests
//! here. It used to be 272 s, and the default was ten framesets with
//! `SLAM_RS_VIO_ORACLE_FULL=1` for the rest; optimizing the test profile
//! (`Cargo.toml`) removed the reason for the split, and with it the one gap
//! the short lane left — the keyframe eviction loop, which first fires at
//! frameset 51.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, LazyLock};

use nalgebra::Vector3;

use slam_rs::config::VioConfig;
use slam_rs::estimator::{
    FlowObservations, FrameOutcome, FrameStats, FrameUpdateOutcome, LmIteration, SqrtKeypointVio,
    WindowSnapshot,
};
use slam_rs::imu::ImuSample;
use slam_rs::lie::LieScalar;
use slam_rs::types::FrameId;

mod common;
use common::{IMU, ORACLE, OracleFlow, OracleLm, OracleRun, run_named};

// ── the window, and the tolerances measured on this fixture ───────────────

/// Framesets the fixture covers, and the number every lane replays.
///
/// The window they walk through: `opt_started` at frameset 4, the second
/// keyframe at 7, both keyframe demotions at 4 and 9, the prior growing from
/// 15x15 to 16x21 at 4 and to 22x27 at 9 — one row wider on each in the `f32`
/// lane, for the reason the prior-shape assertion in `compare` gives — the
/// first `f32` LM-trail divergence at 9, and the keyframe eviction loop's first
/// firing at 51.
const ORACLE_FRAMESETS: usize = 60;

/// Relative agreement on every pose, velocity and bias coefficient of the
/// window, in `f64`. Measured worst over the 60 framesets: rotation 7.9e-13,
/// translation 3.2e-13, velocity 1.1e-12, bias 2.1e-11 — each about 1.6x what
/// it was before S33 routed `So3` through `kornia-algebra`, and still two
/// orders inside the constant.
const POSE_TOLERANCE_F64: f64 = 2e-10;
/// The same in `f32`. Measured worst: rotation 5.4e-5, translation 7.4e-5,
/// velocity 1.9e-4, bias 1.2e-3 — three of the four smaller after S33 and the
/// fourth 1.3x larger, which is what a noise floor does under a re-rounding.
const POSE_TOLERANCE_F32: f64 = 3e-3;
/// Relative agreement in `f64` on the LM error terms, `l_diff`, `lambda` and
/// the marginalization prior's Frobenius digest — the stricter lane needs no
/// split between them. Measured worst: 3.3e-10 (`error_before`), and on the
/// digest 3.1e-15 on `H` and 7.9e-11 on `b`.
const ERROR_TOLERANCE_F64: f64 = 2e-9;
/// The same in `f32`, over the trail prefix the two runs share, for the
/// quantities one well-conditioned formula computes from the current window:
/// the reprojection cost, the IMU and bias costs and the step's infinity norm.
/// Measured worst over the 60 framesets: 1.8e-3 (`imu_error`); the constant is
/// 5.5 times that.
const ERROR_TOLERANCE_F32: f64 = 1e-2;
/// Relative agreement in `f32` on the quantities that carry the prior's
/// accumulated history rather than the current window.
///
/// `marg_data.b` is `−H·delta` plus a residue three orders smaller (8.30
/// against 0.12 at frameset 4), so it carries the accumulated `f32` error of
/// every increment the frozen blocks absorbed; `marg_prior_error` is its
/// bilinear form; `error_before` is that plus a reprojection cost two orders
/// smaller (1.2e3 against 1.3e2); `l_diff` and `lambda` follow from the gain
/// ratio; and `marg_data.H` comes out of a rank-revealing QR of a window that
/// has already diverged.
///
/// These are chaos-limited, not physics: changing one reduction order inside
/// the prior's error moved `‖H‖_F` from 5.0e-6 to 3.3e-4 and `lambda` from
/// 0.145 to 0.016 without moving any decision. The constant is 3.4 times the
/// worse of the two measurements (0.145, `lambda` — whose Nielsen update cubes
/// a ratio whose numerator is a noise-level `f_diff`); on the current tree the
/// worst in the group is 0.155, `lambda` again, so it is a margin, not a bound
/// anything derives.
const CANCELLING_TOLERANCE_F32: f64 = 5e-1;

/// How far an accept-or-converge decision may be inside the `f32` noise floor
/// before the two LM trails are allowed to part company, in units of
/// `f32::EPSILON · |error_before|`.
///
/// The survey over the 60 framesets found 23 framesets where they do, 21 before
/// one reduction order inside the prior's error changed and 20 after S33 changed
/// another; over those runs the widest deciding `f_diff` was 30.4 ulps and the
/// narrowest −16.5. 64 is
/// that with room, and still two orders below the 1e-3-scale decrease the
/// `f64` run sees at those steps.
const F32_ACCEPT_NOISE_ULPS: f64 = 64.0;

/// The row norm below which a marginalization prior's row is a null direction
/// rather than a constraint, used only where the `f32` lane keeps a row the C++
/// discards (see the prior-shape assertion in `compare`).
///
/// Measured over both lanes and all 120 framesets: the surplus rows are 8.8e-4
/// and 9.2e-4, the smallest non-zero row either lane keeps anywhere is 0.35, and
/// the largest is 1.3e4. The constant sits an order above the first group and
/// 35 times below the second.
const MARG_NULL_ROW_NORM: f64 = 1e-2;

/// What `configs/profiles/fast.json` gives the frame update, so the lane the
/// test drives is the lane the benchmark runs (D76).
const FRAME_UPDATE_STEPS: i32 = 5;

// ── loading ───────────────────────────────────────────────────────────────

static CONFIG: LazyLock<VioConfig> = LazyLock::new(common::config);

/// The estimator every lane starts from: the fixture's calibration and config,
/// with the whole IMU window already pushed.
fn window<S: LieScalar>(config: VioConfig) -> SqrtKeypointVio<S> {
    let mut estimator: SqrtKeypointVio<S> =
        SqrtKeypointVio::with_default_gravity(common::calibration().cast(), config).unwrap();
    for row in IMU.iter() {
        estimator.push_imu(ImuSample {
            t_ns: row.t_ns,
            gyro: Vector3::from(row.gyro),
            accel: Vector3::from(row.accel),
        });
    }
    estimator
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
fn compare<S: LieScalar>(run: &OracleRun, gate: LmGate) -> Worst {
    let mut estimator: SqrtKeypointVio<S> = window(CONFIG.clone());
    let max_states: usize = usize::try_from(CONFIG.vio_max_states).unwrap();
    let max_kfs: usize = usize::try_from(CONFIG.vio_max_kfs).unwrap();

    let mut diverged: Vec<i64> = Vec::new();
    let mut worst: Worst = Worst::default();
    for (index, expected) in run.frames.iter().take(ORACLE_FRAMESETS).enumerate() {
        let flow: &OracleFlow = ORACLE
            .flow
            .get(index)
            .unwrap_or_else(|| panic!("no flow stream for frame {index}"));
        assert_eq!(flow.t_ns, expected.t_ns, "frame {index}: timestamp");

        let outcome: FrameOutcome<S> = estimator
            .process_frame(common::observations(flow))
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
        assert_eq!(stats.kf_ids, expected.kf_ids, "{where_}: kf_ids");
        assert_eq!(stats.ltkfs, expected.ltkfs, "{where_}: ltkfs");
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
            stats.opt_started, expected.opt_started,
            "{where_}: opt_started"
        );
        assert_eq!(
            stats.frames_after_kf, expected.frames_after_kf,
            "{where_}: frames_after_kf, the keyframe vote's rate limiter"
        );

        // The window itself: the same frames, as the same kind of block, with
        // the same fixed-linearization flags. One snapshot per frameset — it
        // clones the window and re-derives every landmark's world position.
        let snapshot: WindowSnapshot<S> = estimator.snapshot();
        let states: Vec<i64> = snapshot.states.iter().map(|state| state.t_ns).collect();
        assert_eq!(
            states,
            expected.states.iter().map(|s| s.t_ns).collect::<Vec<i64>>(),
            "{where_}: frame_states"
        );
        let poses: Vec<i64> = snapshot.poses.iter().map(|pose| pose.t_ns).collect();
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
        let prior: &nalgebra::DMatrix<S> = &estimator.marg_data().h;
        assert_eq!(
            prior.ncols(),
            expected.marg_digest.cols,
            "{where_}: prior columns"
        );
        // The row count is the *rank* the marginalizing QR found — how many
        // columns cleared `|beta| > sqrt(epsilon)` (`marg/helper.rs`) — and in
        // `f32` that threshold is 3.4e-4 with a gauge-null direction sitting on
        // it. S33 routed `So3`'s point action through `kornia-algebra`, whose
        // association differs from Sophus's in the last bit of every rotated
        // point, and on seven of the 60 `f32` framesets (4 through 10) the
        // direction now clears the threshold: 17x21 where the C++ has 16x21,
        // then 23x27 where it has 22x27. `f64` is unchanged on all 60.
        //
        // What the exact assertion was standing in for is that the prior
        // constrains the same subspace, so that is what is asserted: no rank is
        // *lost*, and any surplus row is a null direction rather than a
        // constraint. Measured over both lanes and all 120 framesets, the
        // surplus rows have norm 8.8e-4 and 9.2e-4, against 0.35 for the
        // smallest non-zero row either lane keeps anywhere and 1.3e4 for the
        // largest — three and a half orders of separation, which is why
        // [`MARG_NULL_ROW_NORM`] can sit between them. (Exactly-zero rows are a
        // third thing and stay: the 15x15 seed prior has five of them, and
        // there `rows` equals the C++'s, so nothing below is asked of it.)
        assert!(
            (expected.marg_digest.rows..=expected.marg_digest.rows + 1).contains(&prior.nrows()),
            "{where_}: prior rows {} outside the C++'s {} plus the one threshold flip",
            prior.nrows(),
            expected.marg_digest.rows
        );
        let surplus: usize = prior.nrows() - expected.marg_digest.rows;
        if surplus > 0 {
            let mut norms: Vec<f64> = (0..prior.nrows())
                .map(|i| {
                    prior
                        .row(i)
                        .iter()
                        .fold(S::zero(), |acc, v| acc + *v * *v)
                        .sqrt()
                        .to_f64()
                })
                .collect();
            norms.sort_by(|a, b| a.partial_cmp(b).expect("a finite prior"));
            for (i, norm) in norms.iter().take(surplus).enumerate() {
                assert!(
                    *norm < MARG_NULL_ROW_NORM,
                    "{where_}: surplus prior row {i} has norm {norm:.3e}, a constraint rather \
                     than a null direction"
                );
            }
        }

        // ── the floating agreement ────────────────────────────────────────
        for (state, want) in snapshot.states.iter().zip(expected.states.iter()) {
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
                if let Some(vel_bias) = state.vel_bias {
                    Worst::take(
                        &mut worst.velocity,
                        vel_bias.vel_w_i[i].to_f64(),
                        want.vel[i],
                    );
                    Worst::take(&mut worst.bias, vel_bias.bias_gyro[i].to_f64(), want.bg[i]);
                    Worst::take(&mut worst.bias, vel_bias.bias_accel[i].to_f64(), want.ba[i]);
                }
            }
            assert_eq!(
                state.linearized, want.linearized,
                "{where_}: linearized flag of state {}",
                want.t_ns
            );
        }
        for (pose, want) in snapshot.poses.iter().zip(expected.poses.iter()) {
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
        }

        Worst::take(
            &mut worst.prior_h,
            common::frobenius(estimator.marg_data().h.iter().copied()),
            expected.marg_digest.h_frobenius,
        );
        Worst::take(
            &mut worst.prior_b,
            common::frobenius(estimator.marg_data().b.iter().copied()),
            expected.marg_digest.b_norm,
        );

        assert_eq!(
            expected.frame, index,
            "the fixture's frames are out of order"
        );

        // ── the window's own budget ───────────────────────────────────────
        //
        // A property rather than a fixture comparison: the window stays inside
        // the budget basalt actually enforces, and the index sets the schedule
        // produces are a partition of the ordering. It holds on every frameset
        // of either lane, which is why it rides this replay instead of one of
        // its own.
        //
        // The keyframe budget is **lazy**, which is the one thing to get right
        // here. `sqrt_keypoint_vio.cpp:767` runs the eviction loop only while
        // `!states_to_marg_vel_bias.empty()`, and that set holds the keyframes
        // leaving the *state* window this step — so a frame that has just been
        // voted a keyframe cannot be evicted while it is still a state, and
        // `kf_ids` sits one over `max_kfs` until it is demoted to a pose
        // block. On this fixture that is framesets 49-50 and 56-57 (eight
        // keyframes against `vio_max_kfs = 7`), and the C++ oracle shows
        // exactly the same eight. The loop's postcondition is therefore
        // `kf_ids <= max_kfs || states_to_marg_vel_bias.is_empty()`, and the
        // overshoot is at most one because `vio_min_frames_after_kf = 5` puts
        // keyframes six framesets apart while a state leaves the window after
        // `vio_max_states = 3`.
        if stats.opt_started {
            assert!(
                snapshot.states.len() <= max_states,
                "frame {}: {} states exceed the budget",
                flow.t_ns,
                snapshot.states.len()
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
                snapshot.poses.len() <= stats.ltkfs.len() + max_kfs,
                "frame {}: {} poses exceed the budget",
                flow.t_ns,
                snapshot.poses.len()
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
            let prior_width: usize = marg.prior_order.iter().map(|(_, _, size)| size).sum();
            assert_eq!(
                prior_width, marg.kept_indices,
                "the new ordering is {prior_width} wide but {} indices were kept",
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
    if !diverged.is_empty() {
        println!(
            "{}: LM trail parted company on {} of {} framesets, all inside the noise floor: {diverged:?}",
            run.scalar,
            diverged.len(),
            ORACLE_FRAMESETS
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

    // The last step each side actually ran up to and including `at`. Either
    // trail can be empty when the other is not — that is itself a divergence —
    // so this reports it rather than underflowing on `len() - 1`.
    let ours: &LmIteration<S> = got
        .get(at)
        .or_else(|| got.last())
        .unwrap_or_else(|| panic!("{where_}: no rust LM step at all, c++ ran {}", want.len()));
    let theirs: &OracleLm = want
        .get(at)
        .or_else(|| want.last())
        .unwrap_or_else(|| panic!("{where_}: no c++ LM step at all, rust ran {}", got.len()));
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

// ── the gates ─────────────────────────────────────────────────────────────

/// Both lanes' agreement, in the three groups the `f32` lane has to
/// distinguish (the module header says why).
///
/// * `pose` — the window's poses, velocities and biases.
/// * `window_local` — the quantities one well-conditioned formula computes
///   from the current window: the reprojection cost, the IMU and bias costs
///   and the step's infinity norm.
/// * `history` — the quantities that carry the prior's accumulated history:
///   the prior's own digest, the errors that include the prior's bilinear
///   form, and the damping that follows from their gain ratio.
///
/// In `f64` the last two groups share one tolerance, so the split costs the
/// stricter lane nothing.
fn assert_worst(worst: &Worst, label: &str, pose: f64, window_local: f64, history: f64) {
    assert!(
        worst.rotation <= pose && worst.translation <= pose,
        "{label} pose drifted: {worst:#?}"
    );
    assert!(
        worst.velocity <= pose && worst.bias <= pose,
        "{label} velocity or bias drifted: {worst:#?}"
    );
    assert!(
        worst.vision_error <= window_local
            && worst.imu_error <= window_local
            && worst.bias_error <= window_local
            && worst.step_norminf <= window_local,
        "{label} LM trail drifted: {worst:#?}"
    );
    assert!(
        worst.error_before <= history
            && worst.error_after <= history
            && worst.marg_prior_error <= history
            && worst.l_diff <= history
            && worst.lambda <= history
            && worst.prior_h <= history
            && worst.prior_b <= history,
        "{label} prior-driven quantities drifted: {worst:#?}"
    );
}

#[test]
fn the_double_window_follows_the_cpp() {
    let worst: Worst = compare::<f64>(run_named(&ORACLE, "double"), LmGate::Exact);
    println!("f64 worst relative difference: {worst:#?}");
    assert_worst(
        &worst,
        "f64",
        POSE_TOLERANCE_F64,
        ERROR_TOLERANCE_F64,
        ERROR_TOLERANCE_F64,
    );
}

/// The `f32` lane, which is the precision basalt ships (Q07).
///
/// Every integer decision is still exact; the LM trail is held to the noise
/// floor and the poses to the measured tolerance. See the module header for why
/// the accept test cannot be reproduced and what was measured.
#[test]
fn the_float_window_follows_the_cpp() {
    let worst: Worst = compare::<f32>(run_named(&ORACLE, "float"), LmGate::NoiseFloor);
    println!("f32 worst relative difference: {worst:#?}");
    assert_worst(
        &worst,
        "f32",
        POSE_TOLERANCE_F32,
        ERROR_TOLERANCE_F32,
        CANCELLING_TOLERANCE_F32,
    );
}

/// Two runs over the same input are bit-identical (D17).
///
/// Not "close": every coefficient of every pose and every LM number is compared
/// with `==`, because nothing in Offline mode may let arrival order, a hash seed
/// or a thread schedule reach a decision.
#[test]
fn a_repeat_run_is_bit_identical() {
    let flow: Vec<Arc<FlowObservations>> = ORACLE
        .flow
        .iter()
        .take(ORACLE_FRAMESETS)
        .map(common::observations)
        .collect();

    let first: Vec<Trace> = drive(&flow);
    let second: Vec<Trace> = drive(&flow);
    assert_eq!(first.len(), flow.len());
    assert_eq!(first, second, "a repeat run differed");
}

/// D76: with `port.frame_update_max_iterations` on, a frameset that took no
/// keyframe solves its own state and nothing else.
///
/// Three things are asserted, and the first is the one that matters: the
/// schedule is still a pure function of the data, so two runs are bit-identical
/// coefficient for coefficient. Then which branch each frameset took, read off
/// `FrameStats::frame_update` rather than inferred from the length of its LM
/// trail: `Taken` on every frameset the knob is eligible for, `NotAttempted`
/// through the warmup and on every keyframe. The work the taken framesets cost
/// is a separate assertion, because a cap the trail happens to respect is not
/// evidence about which solver ran. What the schedule costs in accuracy is not
/// a unit test's question: it is the A/B harness's, against ground truth on a
/// whole clip.
#[test]
fn the_frame_update_lane_is_deterministic_and_takes_its_branch() {
    let flow: Vec<Arc<FlowObservations>> = ORACLE
        .flow
        .iter()
        .take(ORACLE_FRAMESETS)
        .map(common::observations)
        .collect();

    let mut config: VioConfig = CONFIG.clone();
    config.port_frame_update_max_iterations = FRAME_UPDATE_STEPS;
    let first: Vec<Trace> = drive_with(config.clone(), &flow);
    let second: Vec<Trace> = drive_with(config, &flow);
    assert_eq!(first, second, "the frame update lane is not deterministic");

    let joint: Vec<Trace> = drive(&flow);
    // The frameset that starts the optimizer runs the joint solve on both lanes:
    // `opt_started` is still false when the gate is read. It is the first with a
    // trail at all, and every frameset after it is the frame update's to decline
    // or take.
    let warmup: usize = joint
        .iter()
        .position(|trace| !trace.lm.is_empty())
        .expect("no frameset reached the optimizer");
    let mut updated: usize = 0;
    for (index, (trace, reference)) in first.iter().zip(joint.iter()).enumerate() {
        // The two lanes agree on which framesets are keyframes over this
        // segment, which is what lets the comparison below be per frameset.
        assert_eq!(
            trace.took_keyframe, reference.took_keyframe,
            "{}",
            trace.t_ns
        );
        // The gate is `knob > 0 && !took_keyframe && opt_started`, and
        // `opt_started` is still false on the frameset that starts the
        // optimizer, so everything up to and including it runs the joint solve
        // on both lanes.
        if index <= warmup || trace.took_keyframe {
            assert_eq!(
                trace.frame_update,
                FrameUpdateOutcome::NotAttempted,
                "frameset {} is warmup or a keyframe and still attempted the update",
                trace.t_ns
            );
            continue;
        }
        assert_eq!(
            trace.frame_update,
            FrameUpdateOutcome::Taken,
            "frameset {} did not take the frame update",
            trace.t_ns
        );
        updated += 1;
    }
    assert!(
        updated > 40,
        "only {updated} framesets took the frame update"
    );

    // The work budget, which is a different question from which solver ran: a
    // taken frameset stays inside the knob's inclusive cap.
    let cap: usize = usize::try_from(FRAME_UPDATE_STEPS + 1).unwrap();
    for trace in first
        .iter()
        .filter(|trace| trace.frame_update == FrameUpdateOutcome::Taken)
    {
        assert!(
            trace.lm.len() <= cap,
            "frameset {} took {} steps against a cap of {cap}",
            trace.t_ns,
            trace.lm.len()
        );
    }

    let steps = |lane: &[Trace]| -> usize {
        lane.iter()
            .skip(warmup + 1)
            .filter(|trace| !trace.took_keyframe)
            .map(|trace| trace.lm.len())
            .sum()
    };
    let joint_steps: usize = steps(&joint);
    let update_steps: usize = steps(&first);
    println!(
        "{updated} framesets took the frame update: {update_steps} steps against the joint solve's {joint_steps}"
    );
    assert!(
        update_steps * 2 < joint_steps,
        "the frame update took {update_steps} steps against the joint solve's {joint_steps}"
    );
}

/// What the determinism test compares: everything a caller can observe except
/// the wall-clock timings, which are measurements and not decisions.
#[derive(Debug, Clone, PartialEq)]
struct Trace {
    t_ns: i64,
    took_keyframe: bool,
    /// What D76's frame update did with this frameset, as the estimator states
    /// it: a decision, so a repeat run has to make the same one.
    frame_update: FrameUpdateOutcome,
    kf_ids: Vec<FrameId>,
    ltkfs: Vec<FrameId>,
    landmarks: usize,
    observations: usize,
    poses: Vec<(i64, [f32; 4], [f32; 3])>,
    lm: Vec<(i32, i32, f32, f32, f32, bool)>,
    prior: Vec<f32>,
}

fn drive(flow: &[Arc<FlowObservations>]) -> Vec<Trace> {
    drive_with(CONFIG.clone(), flow)
}

fn drive_with(config: VioConfig, flow: &[Arc<FlowObservations>]) -> Vec<Trace> {
    let mut estimator: SqrtKeypointVio<f32> = window(config);
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
            took_keyframe: stats.took_keyframe,
            frame_update: stats.frame_update,
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
