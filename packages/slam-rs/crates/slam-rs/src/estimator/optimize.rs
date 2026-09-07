//! `SqrtKeypointVioEstimator::optimize()` (`sqrt_keypoint_vio.cpp:1201-1639`).
//!
//! A Levenberg–Marquardt loop with an inner backtracking loop, tabulated line
//! by line in papers-part2 §12.7. Five things about it are basalt's and not the
//! textbook's, and all five are reproduced here:
//!
//! * **`lambda` is reset every frame** to `vio_lm_lambda_initial` (`:1249`,
//!   D11), so the trust region has no memory across framesets.
//! * **The damping is `lambda · diag(H)` with a floor**, not `lambda·I`
//!   (`:1415`, D10).
//! * **The 7-iteration budget is shared with backtracking**: `it++` fires on a
//!   rejection too (`:1592`, D12), so a frame that backtracks three times gets
//!   four real linearizations.
//! * **The increment is negated** before back-substitution (`:1450`, D13),
//!   because `ba_utils.h:117` computes `π(x) − z` where the paper writes
//!   `z − π(x)`.
//! * **No Jacobian scaling and no landmark or pose damping** (D9/D34): the four
//!   calls are commented out in the shipped source, and damping enters only
//!   through the reduced system's diagonal. `backSubstitute` still calls
//!   `setLandmarkDamping(0)` on itself, so the undo path runs as a no-op.
//!
//! The accept test compares the true cost decrease with the linearized model's,
//! and the model's includes the landmarks' own gain — it is positive at
//! `inc = 0` (pr10-linearize.md finding 1). Nothing here "fixes" that.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, DVector, Vector3};

use super::{EstimatorError, FrameStats, SqrtKeypointVio, VEE_FACTOR, duration_ns};
use crate::imu::{ImuLinData, IntegratedImuMeasurement, Matrix9};
use crate::lie::{LieScalar, eigen_maxi};
use crate::linearize::{ImuInput, LinearizationAbsQR, LinearizationInputs, LinearizationOptions};
use crate::marg::eigen_ldlt::EigenLdlt;
use crate::types::{
    AbsOrderMap, FrameId, POSE_SIZE, POSE_VEL_BIAS_SIZE, PoseVelBiasState, PoseVelBiasStateWithLin,
    Vector9, Vector15,
};

/// `max_num_iter` for the damped solve (`:1408`).
const MAX_SOLVE_ATTEMPTS: u32 = 3;

/// The two hard-coded convergence constants of `:1566`, which are **not**
/// config fields.
const FUNCTION_TOLERANCE: f64 = 1e-6;
/// See [`FUNCTION_TOLERANCE`].
const STEP_TOLERANCE: f64 = 1e-4;

/// `H.diagonal().segment<POSE_SIZE>(idx).array() = 1e20` (`:1400`), the value
/// `vio_fix_long_term_keyframes` pins a long-term keyframe's rows with.
const FIXED_KEYFRAME_WEIGHT: f64 = 1e20;

/// Why the LM loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmTermination {
    /// `opt_started` was false and `frame_states.size() <= 4`, so no
    /// linearization ran at all (`:1207`).
    NotStarted,
    /// `(f_diff > 0 && f_diff < 1e-6) || step_norminf < 1e-4` (`:1565-1568`).
    Converged,
    /// The iteration budget ran out without converging (`:1283`).
    MaxIterations,
    /// `lambda > max_lambda` after a rejection (`:1595-1598`).
    MaxDamping,
}

/// One step of the LM loop, accepted or rejected.
///
/// The five error components are split because their sum is what the accept
/// test compares and a parity gap has to be attributable: the marg-prior term
/// deliberately drops `½rᵀr` and can be negative (`ba_base.cpp:452-455`, D20),
/// so a port that folds it into one number cannot tell which term drifted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmIteration<S: LieScalar> {
    /// `it` when this step ran (`:1283`).
    pub iteration: i32,
    /// `j`, the backtracking index inside this linearization (`:1350`).
    pub backtrack: i32,
    /// `error_total` from `linearizeProblem` (`:1297`).
    pub error_before: S,
    /// `after_error_total` (`:1500`).
    pub error_after: S,
    /// `computeError`'s reprojection cost after the increment (`:1487`).
    pub vision_error: S,
    /// `computeImuError`'s `imu_error` (`:1492`).
    pub imu_error: S,
    /// `computeImuError`'s `bg_error`.
    pub bias_gyro_error: S,
    /// `computeImuError`'s `ba_error`.
    pub bias_accel_error: S,
    /// `computeMargPriorError` after the increment (`:1488`).
    pub marg_prior_error: S,
    /// `l_diff` from `backSubstitute` (`:1454`).
    pub l_diff: S,
    /// `f_diff = error_total − after_error_total` (`:1509`).
    pub f_diff: S,
    /// `relative_decrease = f_diff / l_diff` (`:1511`).
    pub relative_decrease: S,
    /// `lambda` the damped solve used.
    pub lambda: S,
    /// `step_norminf = inc.array().abs().maxCoeff()` (`:1477`).
    pub step_norminf: S,
    /// How many times the damped LDLT was retried (`:1410-1430`); more than one
    /// means the first increment was not finite.
    pub solve_attempts: u32,
    /// `step_is_valid = l_diff > 0` (`:1528`).
    pub step_is_valid: bool,
    /// `step_is_successful` (`:1529`), i.e. whether the increment was kept.
    pub accepted: bool,
}

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// `optimize()` (`:1201-1639`).
    ///
    /// # Errors
    ///
    /// [`EstimatorError::PriorOrderMismatch`] where C++ asserts the window
    /// agrees with the prior (`:1227`, `:1237`),
    /// [`EstimatorError::NumericallyInvalid`] where it prints "did not expect
    /// numerical failure during linearization" and fails the frame
    /// (`:1300-1303`), and the linearization's own errors.
    pub(super) fn optimize(&mut self, stats: &mut FrameStats<S>) -> Result<(), EstimatorError> {
        // `:1207`: five states have to accumulate before the first
        // optimization.
        if !self.opt_started && self.ba.frame_states.len() <= 4 {
            return Ok(());
        }
        self.opt_started = true;
        stats.opt_started = true;

        let imu_lin: ImuLinData<S> = self.imu_lin_data();
        // The nine estimator members the C++ reads, as disjoint field borrows:
        // the linearizer needs `ba` mutably while `marg_data` and `imu_meas`
        // are borrowed into its inputs.
        let Self {
            ref mut ba,
            ref mut damping,
            ref marg_data,
            ref imu_meas,
            ref ltkfs,
            ref config,
            ..
        } = *self;

        // `:1221-1242`: poses first, then states, both in ascending timestamp
        // order, and each entry checked against the prior's. C++ reads the prior
        // with `.at()` for the poses (an out-of-range throw when it disagrees) and
        // guards the states with `aom.items < marg_data.order.size()`, because the
        // newest states are not in the prior yet.
        //
        // This is deliberately not `marg::window::build_absolute_ordering`: that
        // one is `:726-763`, which walks the same two maps but stops at
        // `last_state_to_marg` and returns the marginalization's own split. basalt
        // writes the two loops out twice for the same reason, and merging them
        // would mean one function with two payloads and two stopping rules.
        let mut aom: AbsOrderMap = AbsOrderMap::new();
        for frame_id in ba.frame_poses.keys().copied() {
            let index: usize = aom.push(frame_id, POSE_SIZE)?;
            let found: (usize, usize) = (index, POSE_SIZE);
            if marg_data.order.get(frame_id) != Some(found) {
                return Err(EstimatorError::PriorOrderMismatch {
                    frame_id,
                    expected: marg_data.order.get(frame_id),
                    found,
                });
            }
        }
        for frame_id in ba.frame_states.keys().copied() {
            let checked: bool = aom.items() < marg_data.order.items();
            let index: usize = aom.push(frame_id, POSE_VEL_BIAS_SIZE)?;
            let found: (usize, usize) = (index, POSE_VEL_BIAS_SIZE);
            if checked && marg_data.order.get(frame_id) != Some(found) {
                return Err(EstimatorError::PriorOrderMismatch {
                    frame_id,
                    expected: marg_data.order.get(frame_id),
                    found,
                });
            }
        }

        // `:1249`, D11.
        damping.lambda = S::from_literal(config.vio_lm_lambda_initial);

        // `:1266-1267`: every interval, with no `aom` filter — unlike
        // `marginalize`, which keeps only the intervals both of whose ends are in
        // the ordering.
        let imu_input: ImuInput<'_, S> = ImuInput {
            lin_data: imu_lin,
            measurements: imu_meas.iter().map(|(t, meas)| (*t, meas)).collect(),
        };
        let fixed_kfs: BTreeSet<FrameId> = if config.vio_fix_long_term_keyframes {
            ltkfs.clone()
        } else {
            BTreeSet::new()
        };
        let inputs: LinearizationInputs<'_, S> = LinearizationInputs {
            marg: Some(marg_data),
            imu: Some(&imu_input),
            used_frames: None,
            lost_landmarks: None,
            fixed_frames: Some(&fixed_kfs),
        };

        // `:1268-1274`: one linearizer for the whole frame; the outer loop
        // re-linearizes into it rather than rebuilding it.
        let mut lqr: LinearizationAbsQR<S> =
            LinearizationAbsQR::new(ba, &aom, LinearizationOptions::default(), &inputs)?;

        let gyro_bias_weight: Vector3<S> = imu_lin.gyro_bias_weight_sqrt.map(|w| w * w);
        let accel_bias_weight: Vector3<S> = imu_lin.accel_bias_weight_sqrt.map(|w| w * w);

        let mut it: i32 = 0;
        let mut termination: Option<LmTermination> = None;

        // `:1283`.
        while it <= config.vio_max_iterations && termination.is_none() {
            let mark: std::time::Instant = std::time::Instant::now();
            // `:1297-1303`.
            let (error_total, numerically_valid) = lqr.linearize_problem(ba, &inputs)?;
            if !numerically_valid {
                return Err(EstimatorError::NumericallyInvalid { t_ns: stats.t_ns });
            }
            // `:1320`.
            lqr.perform_qr()?;
            stats.timings.linearize_ns += duration_ns(mark);

            // `:1350`: the inner loop shares `it` with the outer one.
            let mut backtrack: i32 = 0;
            while it <= config.vio_max_iterations && termination.is_none() {
                let mark: std::time::Instant = std::time::Instant::now();
                // `:1393`.
                let (mut h, mut b) = lqr.get_dense_h_b(ba, &inputs)?;

                // `:1395-1406`.
                if config.vio_fix_long_term_keyframes {
                    let weight: S = S::from_literal(FIXED_KEYFRAME_WEIGHT);
                    for t_ns in ltkfs {
                        let Some((idx, size)) = aom.get(*t_ns) else {
                            // `:1397-1399`: C++ prints "[UNEXPECTED]" and skips.
                            continue;
                        };
                        for row in idx..(idx + size).min(h.nrows()) {
                            for col in 0..h.ncols() {
                                h[(row, col)] = S::zero();
                            }
                            b[row] = S::zero();
                        }
                        for row in idx..(idx + POSE_SIZE).min(h.nrows()) {
                            h[(row, row)] = weight;
                        }
                    }
                }

                // `:1408-1430`: up to three damped solves, escalating `lambda` on a
                // non-finite increment.
                let size: usize = h.nrows();
                let mut solve_attempts: u32 = 0;
                let lambda_used: S = damping.lambda;
                // `MAX_SOLVE_ATTEMPTS` is three, so the first solve always happens
                // and the increment never needs a placeholder value.
                let (mut inc, inc_valid): (DVector<S>, bool) = loop {
                    // `:1415-1417`. `cwiseMax` is `numext::maxi`, so a NaN on the
                    // left survives where `f32::max` would drop it.
                    let mut h_copy: DMatrix<S> = h.clone();
                    for i in 0..size {
                        let damped: S = eigen_maxi(h[(i, i)] * damping.lambda, damping.min_lambda);
                        h_copy[(i, i)] += damped;
                    }
                    // `:1419-1420`.
                    let inc: DVector<S> = EigenLdlt::new(h_copy).solve_vec(&b);
                    solve_attempts += 1;
                    if inc.iter().all(|v| v.is_finite()) {
                        break (inc, true);
                    }
                    damping.lambda = damping.lambda_vee * damping.lambda;
                    damping.lambda_vee *= S::from_literal(VEE_FACTOR);
                    if solve_attempts >= MAX_SOLVE_ATTEMPTS {
                        break (inc, false);
                    }
                };
                // `:1432`: C++ warns and carries on with the non-finite increment.
                if !inc_valid {
                    log::warn!(
                        "frame {} ns: increment still not finite after {MAX_SOLVE_ATTEMPTS} damped solves",
                        stats.t_ns
                    );
                }
                stats.timings.solver_ns += duration_ns(mark);

                // `:1443`.
                ba.backup();

                // `:1447-1454`, D13: negate, then back-substitute.
                let mark: std::time::Instant = std::time::Instant::now();
                inc = -inc;
                let l_diff: S = lqr.back_substitute(ba, &inputs, &inc)?;
                stats.timings.back_substitution_ns += duration_ns(mark);

                // `:1466-1474`.
                for (frame_id, state) in &mut ba.frame_poses {
                    let Some((idx, _)) = aom.get(*frame_id) else {
                        continue;
                    };
                    let step: nalgebra::SVector<S, POSE_SIZE> =
                        nalgebra::SVector::from_iterator(inc.rows(idx, POSE_SIZE).iter().copied());
                    state.apply_inc(&step);
                }
                for (frame_id, state) in &mut ba.frame_states {
                    let Some((idx, _)) = aom.get(*frame_id) else {
                        continue;
                    };
                    let step: Vector15<S> =
                        Vector15::from_iterator(inc.rows(idx, POSE_VEL_BIAS_SIZE).iter().copied());
                    state.apply_inc(&step);
                }

                // `:1477`: `inc.array().abs().maxCoeff()`. `maxCoeff` folds with
                // `numext::maxi`, which is order-independent for finite values, so a
                // sequential fold is the same number; with a non-finite increment
                // the fold order can matter and this one is left to right.
                let mut step_norminf: S = S::zero();
                for value in inc.iter() {
                    step_norminf = eigen_maxi(step_norminf, value.abs());
                }

                // `:1484-1497`: the true cost at the new state.
                let mark: std::time::Instant = std::time::Instant::now();
                let (vision_error, _) = ba.compute_error(None, S::zero())?;
                let marg_prior_error: S = ba.compute_marg_prior_error(marg_data)?;
                let (imu_error, bias_gyro_error, bias_accel_error) = compute_imu_error(
                    &aom,
                    &ba.frame_states,
                    imu_meas,
                    &gyro_bias_weight,
                    &accel_bias_weight,
                    &imu_lin.g,
                );
                // `:1495`: `vision += ((imu + bg) + ba)`, in that association.
                let vision_and_inertial: S =
                    vision_error + ((imu_error + bias_gyro_error) + bias_accel_error);
                stats.timings.error_ns += duration_ns(mark);

                // `:1500`.
                let error_after: S = vision_and_inertial + marg_prior_error;
                // `:1509-1511`.
                let f_diff: S = error_total - error_after;
                let relative_decrease: S = f_diff / l_diff;
                // `:1528-1529`.
                let step_is_valid: bool = l_diff > S::zero();
                let accepted: bool = step_is_valid && relative_decrease > S::zero();

                stats.lm.push(LmIteration {
                    iteration: it,
                    backtrack,
                    error_before: error_total,
                    error_after,
                    vision_error,
                    imu_error,
                    bias_gyro_error,
                    bias_accel_error,
                    marg_prior_error,
                    l_diff,
                    f_diff,
                    relative_decrease,
                    lambda: lambda_used,
                    step_norminf,
                    solve_attempts,
                    step_is_valid,
                    accepted,
                });

                if accepted {
                    // `:1557-1562`: Nielsen's update. `std::pow<Scalar>(x, 3)`
                    // deduces the exponent as `int`, so `__promote_2<Scalar, int>`
                    // is `double` in both instantiations and the power and the
                    // `1 −` happen in `double` before narrowing back.
                    let x: S = S::from_literal(2.0) * relative_decrease - S::one();
                    let gain: S = S::from_literal(1.0 - x.to_f64().powf(3.0));
                    let floor: S = S::one() / S::from_literal(3.0);
                    damping.lambda *= eigen_maxi(floor, gain);
                    damping.lambda = eigen_maxi(damping.min_lambda, damping.lambda);
                    damping.lambda_vee = S::from_literal(VEE_FACTOR);
                    it += 1;

                    // `:1565-1568`, both constants hard-coded in C++ too.
                    if (f_diff > S::zero() && f_diff < S::from_literal(FUNCTION_TOLERANCE))
                        || step_norminf < S::from_literal(STEP_TOLERANCE)
                    {
                        termination = Some(LmTermination::Converged);
                    }
                    // `:1571`: leave the inner loop and re-linearize.
                    break;
                }

                // `:1585-1598`.
                damping.lambda = damping.lambda_vee * damping.lambda;
                damping.lambda_vee *= S::from_literal(VEE_FACTOR);
                ba.restore();
                it += 1;
                backtrack += 1;
                if damping.lambda > damping.max_lambda {
                    termination = Some(LmTermination::MaxDamping);
                }
            }
        }

        stats.termination = termination.unwrap_or(LmTermination::MaxIterations);
        Ok(())
    }
}

/// `ScBundleAdjustmentBase::computeImuError` (`sc_ba_base.cpp:657-704`), which
/// the QR path calls even though everything else about it is Schur-complement
/// machinery (`sqrt_keypoint_vio.cpp:1490-1492`).
///
/// Three sums: the whitened preintegration residual, and one random-walk term
/// per bias. Intervals of zero length and intervals whose two ends are not both
/// in the ordering are skipped (`:667`, `:672`).
///
/// **What is not Eigen's order.** `res.transpose() * cov_inv * res` is a
/// `1×9 · 9×9 · 9×1` chain that Eigen dispatches through `gemv`, and the two
/// bias terms are `1×3 · diag · 3×1` coefficient-based products. The bias terms
/// are ported exactly — Eigen's small coefficient-based product is the left fold
/// written below (`ProductEvaluators.h`, `etor_product_coeff_impl` recurses on
/// `UnrollingIndex − 1` and adds the last term) — while the 9-dimensional chain
/// is a plain pair of left folds where Eigen's kernel splits by packet. The
/// oracle reports the three components separately for exactly this reason.
fn compute_imu_error<S: LieScalar>(
    aom: &AbsOrderMap,
    states: &BTreeMap<FrameId, PoseVelBiasStateWithLin<S>>,
    imu_meas: &BTreeMap<i64, IntegratedImuMeasurement<S>>,
    gyro_bias_weight: &Vector3<S>,
    accel_bias_weight: &Vector3<S>,
    g: &Vector3<S>,
) -> (S, S, S) {
    let mut imu_error: S = S::zero();
    let mut bg_error: S = S::zero();
    let mut ba_error: S = S::zero();

    for meas in imu_meas.values() {
        if meas.get_dt_ns() == 0 {
            continue;
        }
        let start_t: i64 = meas.get_start_t_ns();
        let end_t: i64 = start_t + meas.get_dt_ns();
        if !aom.contains(start_t) || !aom.contains(end_t) {
            continue;
        }
        let (Some(start_state), Some(end_state)) = (states.get(&start_t), states.get(&end_t))
        else {
            // C++ uses `.at()` here; the `aom` test above already guarantees the
            // frames exist as *some* block, and a pose-only block cannot carry a
            // bias, so a miss means this interval has no IMU factor.
            continue;
        };

        let start: &PoseVelBiasState<S> = start_state.state();
        let end: &PoseVelBiasState<S> = end_state.state();
        let res: Vector9<S> = meas.residual(
            &start.pose_vel_state(),
            g,
            &end.pose_vel_state(),
            &start.bias_gyro,
            &start.bias_accel,
        );
        let cov_inv: Matrix9<S> = meas.get_cov_inv();
        // `(0.5 · resᵀ) · cov_inv · res`, left to right as Eigen's expression
        // tree evaluates it.
        let mut row: [S; 9] = [S::zero(); 9];
        for (j, slot) in row.iter_mut().enumerate() {
            let mut acc: S = S::zero();
            for i in 0..9 {
                acc += res[i] * cov_inv[(i, j)];
            }
            *slot = acc;
        }
        let mut quadratic: S = S::zero();
        for (j, value) in row.iter().enumerate() {
            quadratic += *value * res[j];
        }
        imu_error += S::from_literal(0.5) * quadratic;

        // `:688`: `dt` in seconds, formed as `int64 · Scalar(1e-9)`.
        let dt: S = S::from_literal(meas.get_dt_ns() as f64) * S::from_literal(1e-9);
        let res_bg: Vector3<S> = start.bias_gyro - end.bias_gyro;
        let gyro_dt: Vector3<S> = gyro_bias_weight / dt;
        let mut bg: S = S::zero();
        for i in 0..3 {
            bg += (res_bg[i] * gyro_dt[i]) * res_bg[i];
        }
        bg_error += S::from_literal(0.5) * bg;

        let res_ba: Vector3<S> = start.bias_accel - end.bias_accel;
        let accel_dt: Vector3<S> = accel_bias_weight / dt;
        let mut ab: S = S::zero();
        for i in 0..3 {
            ab += (res_ba[i] * accel_dt[i]) * res_ba[i];
        }
        ba_error += S::from_literal(0.5) * ab;
    }

    (imu_error, bg_error, ba_error)
}
