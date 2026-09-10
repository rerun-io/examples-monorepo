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
//!   `setLandmarkDamping(0)` on itself; here there is nothing to undo, because
//!   the three damping rows are never written.
//!
//! The accept test compares the true cost decrease with the linearized model's,
//! and the model's includes the landmarks' own gain — it is positive at
//! `inc = 0` (pr10-linearize.md finding 1). Nothing here "fixes" that.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector, Vector3};

use super::{
    EstimatorError, LmDamping, SqrtKeypointVio, StageTimings, VEE_FACTOR, fixed_keyframes,
};
use crate::duration_ns;
use crate::eigen::ldlt::EigenLdlt;
use crate::imu::{ImuLinData, IntegratedImuMeasurement, Matrix9};
use crate::lie::{LieScalar, eigen_maxi};
use crate::linearize::{
    DenseHbWorkspace, ImuInput, LinearizationAbsQR, LinearizationInputs, LinearizationOptions,
};
use crate::types::{
    AbsOrderMap, FrameId, POSE_SIZE, POSE_VEL_BIAS_SIZE, PoseVelBiasState, PoseVelBiasStateWithLin,
    Vector9, Vector15,
};

/// `max_num_iter` for the damped solve (`:1408`).
const MAX_SOLVE_ATTEMPTS: u32 = 3;

/// Everything one `optimize` call works in that outlives the call.
///
/// The estimator holds one and hands it to the loop, so the buffers survive
/// from frame to frame; they are reset or fully overwritten before each read,
/// which is what makes holding them a change of allocation and not of
/// arithmetic. Kept as a struct rather than as loose fields so
/// [`super::SqrtKeypointVio`] has one thing to name and the destructuring at
/// the top of [`SqrtKeypointVio::optimize`] stays one line.
#[derive(Debug, Clone)]
pub(super) struct OptimizeScratch<S: LieScalar> {
    /// The dense reduction's accumulator, subtree partials and leaf transpose.
    pub(super) dense: DenseHbWorkspace<S>,
    /// The damped solve's factorization, its working copy and the increment.
    pub(super) solve: EigenLdlt<S>,
    /// The increment [`damped_solve`] writes and the loop then negates.
    pub(super) increment: DVector<S>,
}

impl<S: LieScalar> Default for OptimizeScratch<S> {
    /// Empty buffers, sized on the first call. Written out rather than derived:
    /// `#[derive(Default)]` would demand `S: Default`, which `LieScalar` does
    /// not.
    fn default() -> Self {
        Self {
            dense: DenseHbWorkspace::default(),
            solve: EigenLdlt::empty(),
            increment: DVector::zeros(0),
        }
    }
}

/// The two hard-coded convergence constants of `:1566`, which are **not**
/// config fields.
pub(super) const FUNCTION_TOLERANCE: f64 = 1e-6;
/// See [`FUNCTION_TOLERANCE`].
pub(super) const STEP_TOLERANCE: f64 = 1e-4;

/// `H.diagonal().segment<POSE_SIZE>(idx).array() = 1e20` (`:1400`), the value
/// `vio_fix_long_term_keyframes` pins a long-term keyframe's rows with.
const FIXED_KEYFRAME_WEIGHT: f64 = 1e20;

/// What one solve leaves behind: the LM trail, why it stopped, and the stages
/// it timed.
///
/// Named because two solves return it — the window's [`SqrtKeypointVio::optimize`]
/// and the frame update of D76 — and `measure` takes whichever ran.
pub(super) type SolveOutcome<S> = (Vec<LmIteration<S>>, LmTermination, StageTimings);

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
    /// `lambda` as the retry loop left it (`:1571`), which is what
    /// `sqrt_keypoint_vio.h:209` calls "the value the damped solve used": the
    /// escalated one after a non-finite increment, not the value the iteration
    /// started with. Every attempt escalates on failure, the last one included,
    /// so three failures record a `lambda` no attempt used — the fork's number
    /// all the same.
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
    /// `optimize()` (`:1201-1639`), returning the LM trail, why it stopped and
    /// the stages it timed. Whether it ran at all is `self.opt_started`, which
    /// it sets.
    ///
    /// # Errors
    ///
    /// [`EstimatorError::PriorOrderMismatch`] where C++ asserts the window
    /// agrees with the prior (`:1227`, `:1237`),
    /// [`EstimatorError::NumericallyInvalid`] where it prints "did not expect
    /// numerical failure during linearization" and fails the frame
    /// (`:1300-1303`), [`EstimatorError::FrameNotInOrdering`] where `:1468`
    /// and `:1472` read the ordering with `.at()`, and the linearization's own
    /// errors.
    pub(super) fn optimize(&mut self, t_ns: i64) -> Result<SolveOutcome<S>, EstimatorError> {
        let mut lm: Vec<LmIteration<S>> = Vec::new();
        let mut timings: StageTimings = StageTimings::default();
        // `:1207`: five states have to accumulate before the first
        // optimization.
        if !self.opt_started && self.ba.frame_states.len() <= 4 {
            return Ok((lm, LmTermination::NotStarted, timings));
        }
        self.opt_started = true;

        let imu_lin: ImuLinData<S> = self.imu_lin_data();
        // The nine estimator members the C++ reads, as disjoint field borrows:
        // the linearizer needs `ba` mutably while `marg_data` and `imu_meas`
        // are borrowed into its inputs.
        let Self {
            ref mut ba,
            ref mut damping,
            ref mut scratch,
            ref marg_data,
            ref imu_meas,
            ref ltkfs,
            ref config,
            ..
        } = *self;
        let OptimizeScratch {
            ref mut dense,
            ref mut solve,
            ref mut increment,
        } = *scratch;

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
        let inputs: LinearizationInputs<'_, S> = LinearizationInputs {
            marg: Some(marg_data),
            imu: Some(&imu_input),
            used_frames: None,
            lost_landmarks: None,
            // `:1258`: the same set `marginalize` builds at `:924`.
            fixed_frames: fixed_keyframes(config, ltkfs),
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
                return Err(EstimatorError::NumericallyInvalid { t_ns });
            }
            // `:1320`.
            lqr.perform_qr()?;
            timings.linearize_ns += duration_ns(mark);

            // `:1350`: the inner loop shares `it` with the outer one.
            let mut backtrack: i32 = 0;
            while it <= config.vio_max_iterations && termination.is_none() {
                let mark: std::time::Instant = std::time::Instant::now();
                // `:1393`.
                let (h, b) = lqr.get_dense_h_b_into(ba, &inputs, dense)?;
                // The reduced system is the ordering's: every `(idx, size)` in
                // `aom` is a block of it, and every frame of the two maps has
                // an entry, because `aom` was built from those maps above and
                // nothing since has added or removed a frame. The three loops
                // below index it under that invariant.
                debug_assert_eq!(h.nrows(), aom.total_size());

                // `:1395-1406`.
                if config.vio_fix_long_term_keyframes {
                    let weight: S = S::from_literal(FIXED_KEYFRAME_WEIGHT);
                    for t_ns in ltkfs {
                        let Some((idx, size)) = aom.get(*t_ns) else {
                            // `:1397-1399`: C++ prints "[UNEXPECTED]" and skips.
                            log::warn!(
                                "[UNEXPECTED] long-term keyframe {t_ns} ns is not in the ordering"
                            );
                            continue;
                        };
                        for row in idx..(idx + size) {
                            for col in 0..h.ncols() {
                                h[(row, col)] = S::zero();
                            }
                            b[row] = S::zero();
                        }
                        for row in idx..(idx + POSE_SIZE) {
                            h[(row, row)] = weight;
                        }
                    }
                }

                // `:1408-1430`.
                let (inc_valid, solve_attempts): (bool, u32) =
                    damped_solve(h, b, damping, solve, increment);
                // `:1432`: C++ warns and carries on with the non-finite increment.
                if !inc_valid {
                    log::warn!(
                        "frame {t_ns} ns: increment still not finite after {MAX_SOLVE_ATTEMPTS} damped solves"
                    );
                }
                timings.solver_ns += duration_ns(mark);

                // `:1443`.
                ba.backup();

                // `:1447-1454`, D13: negate, then back-substitute.
                let mark: std::time::Instant = std::time::Instant::now();
                increment.neg_mut();
                let inc: &DVector<S> = increment;
                let l_diff: S = lqr.back_substitute(ba, &inputs, inc)?;
                timings.back_substitution_ns += duration_ns(mark);

                // `:1466-1474`.
                for (frame_id, state) in &mut ba.frame_poses {
                    let Some((idx, _)) = aom.get(*frame_id) else {
                        return Err(EstimatorError::FrameNotInOrdering {
                            frame_id: *frame_id,
                        });
                    };
                    let step: nalgebra::SVector<S, POSE_SIZE> =
                        nalgebra::SVector::from_iterator(inc.rows(idx, POSE_SIZE).iter().copied());
                    state.apply_inc(&step);
                }
                for (frame_id, state) in &mut ba.frame_states {
                    let Some((idx, _)) = aom.get(*frame_id) else {
                        return Err(EstimatorError::FrameNotInOrdering {
                            frame_id: *frame_id,
                        });
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
                )?;
                // `:1495`: `vision += ((imu + bg) + ba)`, in that association.
                let vision_and_inertial: S =
                    vision_error + ((imu_error + bias_gyro_error) + bias_accel_error);
                timings.error_ns += duration_ns(mark);

                // `:1500`.
                let error_after: S = vision_and_inertial + marg_prior_error;
                // `:1509-1511`.
                let f_diff: S = error_total - error_after;
                let relative_decrease: S = f_diff / l_diff;
                // `:1528-1529`.
                let step_is_valid: bool = l_diff > S::zero();
                let accepted: bool = step_is_valid && relative_decrease > S::zero();

                lm.push(LmIteration {
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
                    // `:1571`: the fork reads `lambda` here, after the retry
                    // loop, so a non-finite first solve is recorded with the
                    // escalated value the next one would use — "the value the
                    // damped solve used" (`sqrt_keypoint_vio.h:209`). Nothing
                    // between the loop and this push moves it.
                    lambda: damping.lambda,
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

        Ok((
            lm,
            termination.unwrap_or(LmTermination::MaxIterations),
            timings,
        ))
    }
}

/// `:1408-1430`: up to three damped LDLT solves, escalating `lambda` on a
/// non-finite increment.
///
/// Returns the increment, whether it is finite, and how many attempts it took.
/// The `lambda` the LM trace records is `damping.lambda` **after** this returns
/// (`:1571`, `sqrt_keypoint_vio.h:209`), which is why the escalation is left in
/// `damping` rather than restored: every failed attempt raises it, the last one
/// included, so three failures leave a `lambda` no attempt used and C++ records
/// that one too.
pub(super) fn damped_solve<S: LieScalar>(
    h: &DMatrix<S>,
    b: &DVector<S>,
    damping: &mut LmDamping<S>,
    ldlt: &mut EigenLdlt<S>,
    inc: &mut DVector<S>,
) -> (bool, u32) {
    let size: usize = h.nrows();
    let mut solve_attempts: u32 = 0;
    // `MAX_SOLVE_ATTEMPTS` is three, so the first solve always happens and the
    // increment never needs a placeholder value.
    loop {
        // `:1415-1417`. `cwiseMax` is `numext::maxi`, so a NaN on the left
        // survives where `f32::max` would drop it.
        //
        // Eigen factorizes in place over a copy of `H`; the copy is the
        // factorization's own working buffer, written here rather than cloned,
        // so a frame's three attempts share one `87x87` allocation instead of
        // taking a fresh one each.
        let copy: &mut DMatrix<S> = ldlt.working_copy(size);
        copy.copy_from(h);
        for i in 0..size {
            let damped: S = eigen_maxi(h[(i, i)] * damping.lambda, damping.min_lambda);
            copy[(i, i)] += damped;
        }
        // `:1419-1420`.
        ldlt.factor();
        ldlt.solve_vec_into(b, inc);
        solve_attempts += 1;
        if inc.iter().all(|v| v.is_finite()) {
            return (true, solve_attempts);
        }
        damping.lambda = damping.lambda_vee * damping.lambda;
        damping.lambda_vee *= S::from_literal(VEE_FACTOR);
        if solve_attempts >= MAX_SOLVE_ATTEMPTS {
            return (false, solve_attempts);
        }
    }
}

/// `ScBundleAdjustmentBase::computeImuError` (`sc_ba_base.cpp:657-704`), which
/// the QR path calls even though everything else about it is Schur-complement
/// machinery (`sqrt_keypoint_vio.cpp:1490-1492`).
///
/// Three sums: the whitened preintegration residual, and one random-walk term
/// per bias. Intervals of zero length and intervals whose two ends are not both
/// in the ordering are skipped (`:667`, `:672`); an interval that is in the
/// ordering but has no state is [`EstimatorError::ImuFactorStateMissing`],
/// where C++ throws out of `states.at()`.
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
) -> Result<(S, S, S), EstimatorError> {
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
            // `:671-672` are `.at()` calls: both endpoints are in `aom`, so a
            // miss in `frame_states` is an invariant break, and skipping the
            // factor would drop its residual from the true cost and change
            // which LM step is accepted (D32).
            return Err(EstimatorError::ImuFactorStateMissing {
                start_t_ns: start_t,
                end_t_ns: end_t,
                missing_t_ns: if states.contains_key(&start_t) {
                    end_t
                } else {
                    start_t
                },
            });
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

    Ok((imu_error, bg_error, ba_error))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::imu::ImuSample;
    use crate::lie::Se3;
    use crate::types::PoseVelBiasState;

    /// `lambda`, `min_lambda`, `max_lambda`, `lambda_vee` as `optimize` starts a
    /// frame: `lambda_vee` is [`VEE_FACTOR`], as `:1249-1251` resets it.
    fn damping(lambda: f64, min_lambda: f64) -> LmDamping<f64> {
        LmDamping {
            lambda,
            min_lambda,
            max_lambda: 1e6,
            lambda_vee: VEE_FACTOR,
        }
    }

    /// A single finite solve leaves `lambda` alone, so the value the trace
    /// records is the one the iteration started with — the oracle's case, and
    /// the reason the fixture cannot see the retry bug.
    #[test]
    fn a_finite_solve_records_the_lambda_it_used() {
        let h: DMatrix<f64> = DMatrix::identity(3, 3);
        let b: DVector<f64> = DVector::from_element(3, 1.0);
        let mut lm: LmDamping<f64> = damping(1e-4, 1e-32);
        let mut ldlt: EigenLdlt<f64> = EigenLdlt::empty();
        let mut inc: DVector<f64> = DVector::zeros(0);
        let (valid, attempts) = damped_solve(&h, &b, &mut lm, &mut ldlt, &mut inc);
        assert!(valid);
        assert_eq!(attempts, 1);
        assert!(inc.iter().all(|v| v.is_finite()));
        assert_eq!(lm.lambda, 1e-4);
        assert_eq!(lm.lambda_vee, VEE_FACTOR);
    }

    /// A system the damping cannot rescue: every attempt fails, so `lambda` is
    /// escalated three times (`:1426-1427` runs after the third failure too)
    /// and the trace records `lambda · 2 · 4 · 8` — the value the port captured
    /// **before** the loop until this fix, where the fork reads it after
    /// (`:1571`, `sqrt_keypoint_vio.h:209`).
    #[test]
    fn a_solve_that_never_becomes_finite_records_the_escalated_lambda() {
        let h: DMatrix<f64> = DMatrix::identity(3, 3);
        // Damping only touches the diagonal, so a non-finite right-hand side is
        // a non-finite increment at every `lambda`.
        let b: DVector<f64> = DVector::from_vec(vec![1.0, f64::NAN, 1.0]);
        let mut lm: LmDamping<f64> = damping(1e-4, 1e-32);

        let mut ldlt: EigenLdlt<f64> = EigenLdlt::empty();
        let mut inc: DVector<f64> = DVector::zeros(0);
        let (valid, attempts) = damped_solve(&h, &b, &mut lm, &mut ldlt, &mut inc);
        assert!(!valid, "a NaN right-hand side has to fail: {inc:?}");
        assert_eq!(attempts, MAX_SOLVE_ATTEMPTS);
        assert_eq!(lm.lambda, 1e-4 * 64.0);
        assert_eq!(lm.lambda_vee, 16.0);
    }

    /// The two-attempt case: the increment of the 1x1 system `b / (H + H·λ)`
    /// overflows `f32` at the first `lambda` and fits at the second, so the
    /// trace records the doubled value, not the one the iteration started with.
    ///
    /// `1e10 / (1e-30 · 21) = 4.8e38` is past `f32::MAX` (3.4e38);
    /// `1e10 / (1e-30 · 41) = 2.4e38` is inside it.
    #[test]
    fn a_retried_solve_records_the_lambda_of_the_attempt_that_worked() {
        let h: DMatrix<f32> = DMatrix::from_element(1, 1, 1e-30);
        let b: DVector<f32> = DVector::from_element(1, 1e10);
        let mut lm: LmDamping<f32> = LmDamping {
            lambda: 20.0,
            min_lambda: 0.0,
            max_lambda: 1e6,
            lambda_vee: VEE_FACTOR as f32,
        };

        let mut ldlt: EigenLdlt<f32> = EigenLdlt::empty();
        let mut inc: DVector<f32> = DVector::zeros(0);
        let (valid, attempts) = damped_solve(&h, &b, &mut lm, &mut ldlt, &mut inc);
        assert!(valid);
        assert_eq!(attempts, 2);
        assert!(inc[0].is_finite());
        assert_eq!(lm.lambda, 40.0);
        assert_eq!(lm.lambda_vee, 4.0);
    }

    /// `computeImuError` reads both endpoints with `.at()` (`sc_ba_base.cpp:671-672`).
    /// Skipping a factor whose state is gone understates the true cost and can
    /// flip the LM accept test, so the port refuses instead.
    ///
    /// Only a test can build this: `measure` inserts the state and the
    /// preintegration together, and `marginalize` removes a frame from the
    /// ordering and the window in the same pass, so no ported path leaves an
    /// interval in `aom` without its state.
    #[test]
    fn an_imu_factor_whose_state_left_the_window_is_refused() {
        let mut aom: AbsOrderMap = AbsOrderMap::new();
        aom.push(0, POSE_VEL_BIAS_SIZE).unwrap();
        aom.push(100, POSE_VEL_BIAS_SIZE).unwrap();

        let zero: Vector3<f64> = Vector3::zeros();
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(0, &zero, &zero);
        let cov: Vector3<f64> = Vector3::from_element(1e-6);
        meas.integrate(
            &ImuSample {
                t_ns: 100,
                gyro: zero,
                accel: Vector3::new(0.0, 0.0, 9.81),
            },
            &cov,
            &cov,
        )
        .unwrap();
        let imu_meas: BTreeMap<i64, IntegratedImuMeasurement<f64>> = BTreeMap::from([(0, meas)]);

        let state: PoseVelBiasStateWithLin<f64> = PoseVelBiasStateWithLin::new(
            PoseVelBiasState::new(0, Se3::identity(), zero, zero, zero),
            false,
        );
        // The end state is missing; the start is not.
        let states: BTreeMap<FrameId, PoseVelBiasStateWithLin<f64>> = BTreeMap::from([(0, state)]);

        let weight: Vector3<f64> = Vector3::from_element(1.0);
        let refused: EstimatorError = compute_imu_error(
            &aom,
            &states,
            &imu_meas,
            &weight,
            &weight,
            &Vector3::new(0.0, 0.0, -9.81),
        )
        .unwrap_err();
        assert_eq!(
            refused,
            EstimatorError::ImuFactorStateMissing {
                start_t_ns: 0,
                end_t_ns: 100,
                missing_t_ns: 100,
            }
        );

        // With both endpoints present the same call is the three sums again.
        let both: BTreeMap<FrameId, PoseVelBiasStateWithLin<f64>> = BTreeMap::from([
            (0, states[&0]),
            (
                100,
                PoseVelBiasStateWithLin::new(
                    PoseVelBiasState::new(100, Se3::identity(), zero, zero, zero),
                    false,
                ),
            ),
        ]);
        assert!(
            compute_imu_error(
                &aom,
                &both,
                &imu_meas,
                &weight,
                &weight,
                &Vector3::new(0.0, 0.0, -9.81),
            )
            .is_ok()
        );
    }
}
