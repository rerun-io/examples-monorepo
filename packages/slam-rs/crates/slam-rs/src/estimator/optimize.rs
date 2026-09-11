//! Levenberg-Marquardt optimization with an inner backtracking loop.
//!
//! Lambda resets each frame (D11). Damping is `lambda · diag(H)` with a floor
//! (D10), and rejected trials consume the same iteration budget as accepted ones
//! (D12). The increment is negated before back-substitution because residuals
//! are `π(x) − z` (D13). There is no Jacobian scaling or landmark/pose damping;
//! only the reduced system's diagonal is damped (D34, D68).
//! The acceptance test compares true cost decrease with predicted decrease,
//! including the eliminated landmarks' own gain, which can be positive at zero
//! pose increment.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector, Vector3};

use super::{
    EstimatorError, LmDamping, SqrtKeypointVio, StageTimings, fixed_keyframes, lm_converged,
};
use crate::duration_ns;
use crate::imu::{ImuLinData, IntegratedImuMeasurement, Matrix9};
use crate::lie::{LieScalar, eigen_maxi};
use crate::linearize::{
    DenseHbWorkspace, ImuInput, LinearizationAbsQR, LinearizationInputs, LinearizationOptions,
};
use crate::types::{
    AbsOrderMap, FrameId, POSE_SIZE, POSE_VEL_BIAS_SIZE, PoseVelBiasState, PoseVelBiasStateWithLin,
    Vector9, Vector15,
};

/// `max_num_iter` for the damped solve.
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
    /// The dense accumulator and per-landmark transpose scratch.
    pub(super) dense: DenseHbWorkspace<S>,
    /// Reused double-precision storage for the scaled, damped normal matrix.
    pub(super) solve: DMatrix<f64>,
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
            solve: DMatrix::zeros(0, 0),
            increment: DVector::zeros(0),
        }
    }
}

/// The two hard-coded convergence constants, which are **not**
/// config fields.
pub(super) const FUNCTION_TOLERANCE: f64 = 1e-6;
/// See [`FUNCTION_TOLERANCE`].
pub(super) const STEP_TOLERANCE: f64 = 1e-4;

/// `H.diagonal().segment<POSE_SIZE>(idx).array() = 1e20`, the value
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
    /// linearization ran at all.
    NotStarted,
    /// `(f_diff > 0 && f_diff < 1e-6) || step_norminf < 1e-4`.
    Converged,
    /// The iteration budget ran out without converging.
    MaxIterations,
    /// `lambda > max_lambda` after a rejection.
    MaxDamping,
}

/// One accepted or rejected LM step.
/// Separate error components make changes in the objective attributable. The
/// prior omits `½rᵀr` and can be negative (D20), so its term remains visible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmIteration<S: LieScalar> {
    /// `it` when this step ran.
    pub iteration: i32,
    /// `j`, the backtracking index inside this linearization.
    pub backtrack: i32,
    /// `error_total` from `linearizeProblem`.
    pub error_before: S,
    /// `after_error_total`.
    pub error_after: S,
    /// `computeError`'s reprojection cost after the increment.
    pub vision_error: S,
    /// `computeImuError`'s `imu_error`.
    pub imu_error: S,
    /// `computeImuError`'s `bg_error`.
    pub bias_gyro_error: S,
    /// `computeImuError`'s `ba_error`.
    pub bias_accel_error: S,
    /// `computeMargPriorError` after the increment.
    pub marg_prior_error: S,
    /// `l_diff` from `backSubstitute`.
    pub l_diff: S,
    /// `f_diff = error_total − after_error_total`.
    pub f_diff: S,
    /// `relative_decrease = f_diff / l_diff`.
    pub relative_decrease: S,
    /// Lambda after the retry loop. Every failure escalates it, including the last
    /// attempt, so three failures record a value that no attempt used.
    pub lambda: S,
    /// `step_norminf = inc.array().abs().maxCoeff()`.
    pub step_norminf: S,
    /// How many times the damped LDLT was retried; more than one
    /// means the first increment was not finite.
    pub solve_attempts: u32,
    /// `step_is_valid = l_diff > 0`.
    pub step_is_valid: bool,
    /// `step_is_successful`, i.e. whether the increment was kept.
    pub accepted: bool,
}

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// Optimize and return the LM trail, stop reason and stage durations.
    /// `self.opt_started` records whether optimization ran.
    ///
    /// # Errors
    /// Returns typed errors for prior-order mismatch, invalid linearization,
    /// missing ordering entries and failures from the linearizer.
    pub(super) fn optimize(&mut self, t_ns: i64) -> Result<SolveOutcome<S>, EstimatorError> {
        let mut lm: Vec<LmIteration<S>> = Vec::new();
        let mut timings: StageTimings = StageTimings::default();
        // five states have to accumulate before the first
        // optimization.
        if !self.opt_started && self.ba.frame_states.len() <= 4 {
            return Ok((lm, LmTermination::NotStarted, timings));
        }
        self.opt_started = true;

        let imu_lin: ImuLinData<S> = self.imu_lin_data();
        // Borrow estimator fields separately so the linearizer can mutate `ba` while
        // borrowing the prior and IMU inputs.
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

        // Order poses before states, each by timestamp, and check the prior prefix.
        // This differs from marginalization ordering, which stops at
        // `last_state_to_marg` and returns its own keep/marginalize split.
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

        // D11.
        damping.lambda = S::from_literal(config.vio_lm_lambda_initial);

        // every interval, with no `aom` filter — unlike
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
            // Fix the same long-term keyframes as marginalization.
            fixed_frames: fixed_keyframes(config, ltkfs),
        };

        // one linearizer for the whole frame; the outer loop
        // re-linearizes into it rather than rebuilding it.
        let mut lqr: LinearizationAbsQR<S> =
            LinearizationAbsQR::new(ba, &aom, LinearizationOptions::default(), &inputs)?;

        let gyro_bias_weight: Vector3<S> = imu_lin.gyro_bias_weight_sqrt.map(|w| w * w);
        let accel_bias_weight: Vector3<S> = imu_lin.accel_bias_weight_sqrt.map(|w| w * w);

        let mut it: i32 = 0;
        let mut termination: Option<LmTermination> = None;

        while it <= config.vio_max_iterations && termination.is_none() {
            let mark: std::time::Instant = std::time::Instant::now();
            let (error_total, numerically_valid) = lqr.linearize_problem(ba, &inputs)?;
            if !numerically_valid {
                return Err(EstimatorError::NumericallyInvalid { t_ns });
            }
            lqr.perform_qr()?;
            timings.linearize_ns += duration_ns(mark);

            // A rejected trial restores the state and leaves the factors unchanged.
            // Assemble once; retries change only damping and solve scratch.
            let mark: std::time::Instant = std::time::Instant::now();
            let (h, b) = lqr.get_dense_h_b_into(ba, &inputs, dense)?;
            // The reduced system is the ordering's: every `(idx, size)` in
            // `aom` is a block of it, and every frame of the two maps has
            // an entry, because `aom` was built from those maps above and
            // nothing since has added or removed a frame. The three loops
            // below index it under that invariant.
            debug_assert_eq!(h.nrows(), aom.total_size());

            if config.vio_fix_long_term_keyframes {
                let weight: S = S::from_literal(FIXED_KEYFRAME_WEIGHT);
                for t_ns in ltkfs {
                    let Some((idx, size)) = aom.get(*t_ns) else {
                        // Skip the unexpected missing entry.
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

            timings.solver_ns += duration_ns(mark);

            // The inner loop shares `it` with the outer one.
            let mut backtrack: i32 = 0;
            while it <= config.vio_max_iterations && termination.is_none() {
                let mark: std::time::Instant = std::time::Instant::now();
                let (inc_valid, solve_attempts): (bool, u32) =
                    damped_solve(h, b, damping, solve, increment);
                // Continue with the non-finite increment after exhausting retries.
                if !inc_valid {
                    log::warn!(
                        "frame {t_ns} ns: increment still not finite after {MAX_SOLVE_ATTEMPTS} damped solves"
                    );
                }
                timings.solver_ns += duration_ns(mark);

                ba.backup();

                // D13: negate, then back-substitute.
                let mark: std::time::Instant = std::time::Instant::now();
                increment.neg_mut();
                let inc: &DVector<S> = increment;
                let l_diff: S = lqr.back_substitute(ba, &inputs, inc)?;
                timings.back_substitution_ns += duration_ns(mark);

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

                // Fold absolute increment coefficients left to right. Order can matter when
                // an increment contains non-finite values.
                let mut step_norminf: S = S::zero();
                for value in inc.iter() {
                    step_norminf = eigen_maxi(step_norminf, value.abs());
                }

                // the true cost at the new state.
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
                // `vision += ((imu + bg) + ba)`, in that association.
                let vision_and_inertial: S =
                    vision_error + ((imu_error + bias_gyro_error) + bias_accel_error);
                timings.error_ns += duration_ns(mark);

                let error_after: S = vision_and_inertial + marg_prior_error;
                let f_diff: S = error_total - error_after;
                let relative_decrease: S = f_diff / l_diff;
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
                    // Record lambda after retries, including any escalation after the final failure.
                    lambda: damping.lambda,
                    step_norminf,
                    solve_attempts,
                    step_is_valid,
                    accepted,
                });

                if accepted {
                    damping.accept(relative_decrease);
                    it += 1;

                    if lm_converged(f_diff, step_norminf) {
                        termination = Some(LmTermination::Converged);
                    }
                    // leave the inner loop and re-linearize.
                    break;
                }

                damping.escalate();
                ba.restore();
                it += 1;
                backtrack += 1;
                if damping.exhausted() {
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

/// Solve the positively damped normal equations, with up to three attempts.
/// Factor in f64 so small damping terms survive addition to f32 input values.
/// Symmetric diagonal scaling handles the different units and fixed-pose weights.
/// A failed attempt escalates lambda, including the final failed attempt.
/// The caller negates the returned increment before applying it.
pub(super) fn damped_solve<S: LieScalar>(
    h: &DMatrix<S>,
    b: &DVector<S>,
    damping: &mut LmDamping<S>,
    working: &mut DMatrix<f64>,
    inc: &mut DVector<S>,
) -> (bool, u32) {
    let size: usize = h.nrows();
    let mut solve_attempts: u32 = 0;
    // `MAX_SOLVE_ATTEMPTS` is three, so the first solve always happens and the
    // increment never needs a placeholder value.
    loop {
        // Preserve NaNs in the diagonal so invalid inputs fail the finite check.
        if working.nrows() != size {
            working.resize_mut(size, size, 0.0);
        }
        for (dst, src) in working.iter_mut().zip(h.iter()) {
            *dst = src.to_f64();
        }
        for i in 0..size {
            let damped = eigen_maxi(h[(i, i)] * damping.lambda, damping.min_lambda);
            working[(i, i)] += damped.to_f64();
        }
        // Floating-point normal matrices need not stay positive definite when
        // the damping is below one ulp. Pivoted LU also handles that case.
        // Solve (D^-1 H D^-1) y = D^-1 b, then recover x = D^-1 y.
        // Keep the scaled input buffer; the library owns one factor copy.
        let scales = DVector::from_iterator(
            size,
            (0..size).map(|i| {
                let scale = working[(i, i)].abs().sqrt();
                if scale > 0.0 { scale } else { 1.0 }
            }),
        );
        for col in 0..size {
            for row in 0..size {
                working[(row, col)] /= scales[row] * scales[col];
            }
        }
        let factor = nalgebra::linalg::FullPivLU::new(working.clone());
        if inc.nrows() != size {
            inc.resize_vertically_mut(size, S::zero());
        }
        let mut solution = b.map(|value| value.to_f64());
        solution.component_div_assign(&scales);
        if factor.solve_mut(&mut solution) {
            for i in 0..size {
                inc[i] = S::from_literal(solution[i] / scales[i]);
            }
        } else {
            inc.fill(S::from_literal(f64::NAN));
        }
        solve_attempts += 1;
        if inc.iter().all(|v| v.is_finite()) {
            return (true, solve_attempts);
        }
        damping.escalate();
        if solve_attempts >= MAX_SOLVE_ATTEMPTS {
            return (false, solve_attempts);
        }
    }
}

/// IMU cost: whitened preintegration plus one random-walk term per bias.
/// Skip zero-length intervals and intervals not fully in the ordering. A missing
/// state for an ordered endpoint returns [`EstimatorError::ImuFactorStateMissing`].
/// Quadratic forms use fixed-order residual folds.
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
            //  are `.at()` calls: both endpoints are in `aom`, so a
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
        // Compute half the residual quadratic form.
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

        // `dt` in seconds, formed as `int64 · Scalar(1e-9)`.
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
    use crate::estimator::VEE_FACTOR;
    use crate::imu::ImuSample;
    use crate::lie::Se3;
    use crate::types::PoseVelBiasState;

    /// `lambda`, `min_lambda`, `max_lambda`, `lambda_vee` as `optimize` starts a
    /// frame: `lambda_vee` is [`VEE_FACTOR`], as resets it.
    fn damping(lambda: f64, min_lambda: f64) -> LmDamping<f64> {
        LmDamping {
            lambda,
            min_lambda,
            max_lambda: 1e6,
            lambda_vee: VEE_FACTOR,
        }
    }

    /// A finite first solve leaves the recorded lambda at its initial value.
    #[test]
    fn a_finite_solve_records_the_lambda_it_used() {
        let h: DMatrix<f64> = DMatrix::identity(3, 3);
        let b: DVector<f64> = DVector::from_element(3, 1.0);
        let mut lm: LmDamping<f64> = damping(1e-4, 1e-32);
        let mut ldlt: DMatrix<f64> = DMatrix::zeros(0, 0);
        let mut inc: DVector<f64> = DVector::zeros(0);
        let (valid, attempts) = damped_solve(&h, &b, &mut lm, &mut ldlt, &mut inc);
        assert!(valid);
        assert_eq!(attempts, 1);
        assert!(inc.iter().all(|v| v.is_finite()));
        assert_eq!(lm.lambda, 1e-4);
        assert_eq!(lm.lambda_vee, VEE_FACTOR);
    }

    #[test]
    fn a_rounded_indefinite_system_still_has_a_finite_solve() {
        // Rounding can leave a tiny negative eigenvalue in a normal matrix.
        // Positive damping below one ulp does not guarantee a positive factor.
        let off = 1.0f32 + f32::EPSILON;
        let h = DMatrix::from_row_slice(2, 2, &[1.0, off, off, 1.0]);
        let b = DVector::from_element(2, 1.0f32);
        let mut lm = LmDamping {
            lambda: 1e-9,
            min_lambda: 1e-9,
            max_lambda: 1e6,
            lambda_vee: 2.0,
        };
        let mut working = DMatrix::zeros(0, 0);
        let mut inc = DVector::zeros(0);
        let (valid, attempts) = damped_solve(&h, &b, &mut lm, &mut working, &mut inc);
        assert!(valid);
        assert_eq!(attempts, 1);
        assert!((h * inc - b).norm() < 1e-6);
    }

    #[test]
    fn damping_below_f32_precision_still_regularizes_a_singular_system() {
        let h = DMatrix::from_element(2, 2, 1.0f32);
        let b = DVector::from_vec(vec![1.0f32, -1.0]);
        let mut lm = LmDamping {
            lambda: 1e-9,
            min_lambda: 1e-9,
            max_lambda: 1e6,
            lambda_vee: 2.0,
        };
        let mut working = DMatrix::zeros(0, 0);
        let mut inc = DVector::zeros(0);
        let (valid, attempts) = damped_solve(&h, &b, &mut lm, &mut working, &mut inc);
        assert!(valid);
        assert_eq!(attempts, 1);
        // Check in f64: rounding the damped matrix back to f32 removes its rank.
        let mut damped = h.map(f64::from);
        damped[(0, 0)] += f64::from(lm.lambda);
        damped[(1, 1)] += f64::from(lm.lambda);
        assert!((damped * inc.map(f64::from) - b.map(f64::from)).norm() < 1e-6);
    }

    proptest::proptest! {
        #[test]
        fn damped_system_has_a_small_residual(
            values in proptest::collection::vec(-1.0f64..1.0, 36),
            rhs in proptest::collection::vec(-1.0f64..1.0, 6),
            exponent in -10i32..0,
        ) {
            let g = DMatrix::from_row_slice(6, 6, &values);
            let h = g.transpose() * g * 10.0f64.powi(exponent);
            let b = DVector::from_vec(rhs);
            let mut lm = damping(1e-3, 1e-8);
            let mut solver = DMatrix::zeros(0, 0);
            let mut inc = DVector::zeros(0);
            let (valid, _) = damped_solve(&h, &b, &mut lm, &mut solver, &mut inc);
            proptest::prop_assert!(valid);
            let mut damped = h.clone();
            for i in 0..6 {
                damped[(i, i)] += (h[(i, i)] * lm.lambda).max(lm.min_lambda);
            }
            // The caller negates inc to solve H x = -b.
            proptest::prop_assert!((damped * -inc + &b).norm() < 1e-8 * (1.0 + b.norm()));
        }
    }

    /// Three failed attempts escalate lambda by `2 · 4 · 8`, including after the
    /// last failure. The trace must record that final value.
    #[test]
    fn a_solve_that_never_becomes_finite_records_the_escalated_lambda() {
        let h: DMatrix<f64> = DMatrix::identity(3, 3);
        // Damping only touches the diagonal, so a non-finite right-hand side is
        // a non-finite increment at every `lambda`.
        let b: DVector<f64> = DVector::from_vec(vec![1.0, f64::NAN, 1.0]);
        let mut lm: LmDamping<f64> = damping(1e-4, 1e-32);

        let mut ldlt: DMatrix<f64> = DMatrix::zeros(0, 0);
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

        let mut ldlt: DMatrix<f64> = DMatrix::zeros(0, 0);
        let mut inc: DVector<f32> = DVector::zeros(0);
        let (valid, attempts) = damped_solve(&h, &b, &mut lm, &mut ldlt, &mut inc);
        assert!(valid);
        assert_eq!(attempts, 2);
        assert!(inc[0].is_finite());
        assert_eq!(lm.lambda, 40.0);
        assert_eq!(lm.lambda_vee, 4.0);
    }

    /// `computeImuError` reads both endpoints with `.at()`.
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
