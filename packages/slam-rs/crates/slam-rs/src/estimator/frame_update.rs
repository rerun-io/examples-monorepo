//! The non-keyframe frame update (D76): the two newest states, against fixed
//! landmarks and their IMU factors.
//!
//! basalt has no counterpart. It is a fixed-lag smoother and re-solves the whole
//! sliding window on every frameset ([`SqrtKeypointVio::optimize`]); cuVSLAM
//! instead solves only the newest pose against constant landmarks per frame
//! (`libs/pnp/multicam_pnp.cpp`) and runs bundle adjustment at keyframes, and
//! this is that schedule's cheap half. It runs only when
//! `port.frame_update_max_iterations` is above zero and only on a frameset that
//! took no keyframe; at the knob's `0` default nothing here is reached and the
//! estimator is basalt's, frame for frame.
//!
//! Everything numeric is borrowed rather than restated: the residual and its
//! pose Jacobian are [`linearize_point`] and [`compute_rel_pose`], the robust
//! weight is the landmark block's own [`compute_error_weight`], the IMU factor is
//! [`ImuBlock::linearize`], and the damped solve is [`damped_solve`] — the same
//! Eigen LDLT the window solve runs. There is one reprojection model in the
//! crate.
//!
//! ## What is free, what is held, and why the prior is not here
//!
//! The free block is the **two newest states**, 30 unknowns. Two states and not
//! one because of where the marginalization freezes: at frameset `k` the window
//! holds states `k−2, k−1, k`, `last_state_to_marg` is `k−1`, and
//! `marginalize` calls `setLinTrue` on it at the end of the same call. So `k−1`
//! is the state whose linearization point is about to become the FEJ prior's,
//! and solving only `k` would freeze it having had exactly one 15-dof update.
//! Measured, that is the whole accuracy cost of this lever: on MIO10 the
//! newest-state-only variant scores 1.698 cm against 1.447 and on MGO09 0.960
//! against 0.757. cuVSLAM's per-frame solver is two-state for the same reason
//! (`libs/imu/soft_inertial_pnp.cpp`, a 30x30).
//!
//! Everything older is held: landmarks, their host keyframes, and `k−2` — which
//! this very frameset marginalizes away. The marginalization prior orders the
//! keyframe poses and `k−2` and nothing else — it can only hold blocks frozen at
//! a linearization point, `computeDelta` refuses any other (`ba_base.cpp:294`),
//! and `k−1` is frozen only at the end of this call — so the prior's cost does
//! not depend on either free state and contributes neither a Jacobian nor a
//! gradient. [`SqrtKeypointVio::frame_update`] checks that per frameset and
//! hands it back to the joint solve if the prior ever does order one of them.
//!
//! ## The loop
//!
//! The window loop's shape, constant for constant: `lambda` reset to
//! `vio_lm_lambda_initial` every frame (D11), `lambda·diag(H)` damping with the
//! same floor (D10), the budget shared with backtracking (D12), the increment
//! negated before it is applied (D13), Nielsen's update on an accept and the
//! same hard-coded `1e-6`/`1e-4` convergence pair. One difference, and it is a
//! simplification the window cannot make: with nothing eliminated, the model's
//! predicted decrease is `−(inc·b + ½ incᵀ H inc)` in closed form, which is what
//! `backSubstitute` accumulates block by block when there are no landmark
//! columns to substitute back.

use nalgebra::{DMatrix, DVector, Matrix2x6, Matrix4, Matrix6, Vector2, Vector6};

use super::optimize::{
    FUNCTION_TOLERANCE, LmIteration, LmTermination, STEP_TOLERANCE, SolveOutcome, damped_solve,
};
use super::{EstimatorError, SqrtKeypointVio, StageTimings, VEE_FACTOR};
use crate::ba_base::{BundleAdjustmentBase, LinearizePointOut, compute_rel_pose, linearize_point};
use crate::duration_ns;
use crate::eigen::ldlt::EigenLdlt;
use crate::imu::{ImuBlock, ImuLinData, IntegratedImuMeasurement};
use crate::lie::{LieScalar, Se3, eigen_maxi};
use crate::linearize::{LandmarkBlockOptions, LinearizeError, compute_error_weight};
use crate::types::{
    FrameId, POSE_SIZE, POSE_VEL_BIAS_SIZE, PoseStateWithLin, PoseVelBiasStateWithLin, TimeCamId,
    Vector15,
};

/// The two states an IMU factor spans, stacked, which is what
/// [`ImuBlock::add_dense_h_b`] writes into — and, because the free block is the
/// same two states, the size of the frame update's own system.
const FREE_SIZE: usize = 2 * POSE_VEL_BIAS_SIZE;

/// Where the older of the two free states sits in the system.
const PREVIOUS_OFFSET: usize = 0;

/// Where the newest state sits.
const CURRENT_OFFSET: usize = POSE_VEL_BIAS_SIZE;

/// Everything one [`SqrtKeypointVio::frame_update`] works in that outlives the
/// call.
///
/// Same reasoning as [`super::optimize::OptimizeScratch`] (D73): the buffers are
/// reset or fully overwritten before each read, so holding them changes the
/// allocation and not the arithmetic, and the loop runs two or three times a
/// frame on six framesets in seven.
#[derive(Debug, Clone)]
pub(super) struct FrameUpdateScratch<S: LieScalar> {
    /// The normal equations at the point the loop is standing on.
    h: DMatrix<S>,
    /// See [`Self::h`].
    b: DVector<S>,
    /// The normal equations at the trial point, which become [`Self::h`] and
    /// [`Self::b`] when the step is accepted.
    h_trial: DMatrix<S>,
    /// See [`Self::h_trial`].
    b_trial: DVector<S>,
    /// One IMU factor's own 30x30 system, before it is folded into [`Self::h`]:
    /// whole for the factor between the two free states, trailing corner only
    /// for the one that reaches back to the held state.
    imu_h: DMatrix<S>,
    /// See [`Self::imu_h`].
    imu_b: DVector<S>,
    /// The damped solve's factorization and working copy.
    solve: EigenLdlt<S>,
    /// The increment [`damped_solve`] writes and the loop then negates.
    increment: DVector<S>,
}

impl<S: LieScalar> Default for FrameUpdateScratch<S> {
    /// Buffers at their final size: unlike the window's, this system's shape is
    /// a compile-time constant.
    fn default() -> Self {
        Self {
            h: DMatrix::zeros(FREE_SIZE, FREE_SIZE),
            b: DVector::zeros(FREE_SIZE),
            h_trial: DMatrix::zeros(FREE_SIZE, FREE_SIZE),
            b_trial: DVector::zeros(FREE_SIZE),
            imu_h: DMatrix::zeros(FREE_SIZE, FREE_SIZE),
            imu_b: DVector::zeros(FREE_SIZE),
            solve: EigenLdlt::empty(),
            increment: DVector::zeros(FREE_SIZE),
        }
    }
}

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// Solve the two newest states against fixed landmarks and their IMU factors.
    ///
    /// `Ok(None)` means the frameset is not one this can serve — a window too
    /// short, an IMU factor missing, a prior that orders a state this would move,
    /// or a long-term keyframe among them — and the caller must run the joint
    /// solve instead. Every one of those tests reads the window alone, so a
    /// replay repeats the choice.
    ///
    /// # Errors
    ///
    /// [`EstimatorError::BundleAdjustment`] where a pose the residual needs is
    /// not in the window, and [`EstimatorError::Linearize`] where the camera id
    /// of an observation is not in the rig.
    pub(super) fn frame_update(
        &mut self,
        t_ns: i64,
    ) -> Result<Option<SolveOutcome<S>>, EstimatorError> {
        // The newest state has to be this frameset's, and the window has to hold
        // the two states behind it: one to free with it and one to hold the pair
        // against.
        if self.ba.frame_states.keys().next_back() != Some(&t_ns) {
            return Ok(None);
        }
        let mut older = self.ba.frame_states.keys().rev().skip(1).copied();
        let (Some(prev_t_ns), Some(held_t_ns)) = (older.next(), older.next()) else {
            return Ok(None);
        };
        // Both preintegrated intervals, each joining the states it is named for.
        let joins = |start: i64, end: i64| -> bool {
            self.imu_meas
                .get(&start)
                .is_some_and(|meas| start.checked_add(meas.get_dt_ns()) == Some(end))
        };
        if !joins(held_t_ns, prev_t_ns) || !joins(prev_t_ns, t_ns) {
            return Ok(None);
        }
        // The prior is a quadratic in blocks frozen at a linearization point.
        // Neither free state is one — the newest is appended unfrozen and
        // `marginalize` freezes its predecessor only at the end of this very
        // call — so the prior cannot order either, and its gradient here is
        // zero. If it ever does order one, this solve would be wrong.
        if self.marg_data.order.get(t_ns).is_some() || self.marg_data.order.get(prev_t_ns).is_some()
        {
            return Ok(None);
        }
        // A long-term keyframe's pose is pinned in both of the window's
        // linearizations (`fixed_keyframes`); freeing one here would not be.
        if self.ltkfs.contains(&prev_t_ns) || self.ltkfs.contains(&t_ns) {
            return Ok(None);
        }

        let mut lm: Vec<LmIteration<S>> = Vec::new();
        let mut timings: StageTimings = StageTimings::default();
        let imu_lin: ImuLinData<S> = self.imu_lin_data();
        let options: LandmarkBlockOptions<S> = LandmarkBlockOptions {
            huber_parameter: self.ba.huber_thresh,
            obs_std_dev: self.ba.obs_std_dev,
            ..LandmarkBlockOptions::default()
        };

        let Self {
            ref mut ba,
            ref mut damping,
            ref mut frame_scratch,
            ref imu_meas,
            ref config,
            ..
        } = *self;
        let FrameUpdateScratch {
            ref mut h,
            ref mut b,
            ref mut h_trial,
            ref mut b_trial,
            ref mut imu_h,
            ref mut imu_b,
            ref mut solve,
            ref mut increment,
        } = *frame_scratch;
        let (Some(held_meas), Some(prev_meas)) =
            (imu_meas.get(&held_t_ns), imu_meas.get(&prev_t_ns))
        else {
            // Unreachable: the same lookups succeeded above and nothing since
            // has touched `imu_meas`.
            return Ok(None);
        };

        // `:1249`, D11: the trust region has no memory across framesets.
        damping.lambda = S::from_literal(config.vio_lm_lambda_initial);

        let mark: std::time::Instant = std::time::Instant::now();
        let states: FreeStates = FreeStates {
            held_t_ns,
            prev_t_ns,
            t_ns,
        };
        let (mut error_total, _): (S, S) = linearize_states(
            ba, held_meas, prev_meas, &imu_lin, &states, &options, h, b, imu_h, imu_b,
        )?;
        timings.linearize_ns += duration_ns(mark);

        let mut it: i32 = 0;
        let mut termination: Option<LmTermination> = None;
        let mut backtrack: i32 = 0;
        while it <= config.port_frame_update_max_iterations && termination.is_none() {
            let mark: std::time::Instant = std::time::Instant::now();
            let (inc_valid, solve_attempts): (bool, u32) =
                damped_solve(h, b, damping, solve, increment);
            if !inc_valid {
                log::warn!(
                    "frame {t_ns} ns: the frame update's increment is still not finite after {solve_attempts} damped solves"
                );
            }
            timings.solver_ns += duration_ns(mark);

            // D13: negate, then apply.
            increment.neg_mut();
            let inc: &DVector<S> = increment;

            // `−(inc·b + ½ incᵀ H inc)`, the model's predicted decrease. With no
            // eliminated block this is `backSubstitute`'s sum in closed form.
            let mut linear: S = S::zero();
            let mut quadratic: S = S::zero();
            for i in 0..FREE_SIZE {
                linear += inc[i] * b[i];
                for j in 0..FREE_SIZE {
                    quadratic += inc[i] * h[(i, j)] * inc[j];
                }
            }
            let l_diff: S = -(linear + S::from_literal(0.5) * quadratic);

            for (frame_id, offset) in [(prev_t_ns, PREVIOUS_OFFSET), (t_ns, CURRENT_OFFSET)] {
                let Some(state) = ba.frame_states.get_mut(&frame_id) else {
                    // Unreachable: the keys came from this map and nothing
                    // removes one inside the loop.
                    return Err(EstimatorError::PreviousStateMissing { t_ns: frame_id });
                };
                let step: Vector15<S> =
                    Vector15::from_iterator(inc.rows(offset, POSE_VEL_BIAS_SIZE).iter().copied());
                state.backup();
                state.apply_inc(&step);
            }

            // `:1477`, folded left to right like the window's.
            let mut step_norminf: S = S::zero();
            for value in inc.iter() {
                step_norminf = eigen_maxi(step_norminf, value.abs());
            }

            let mark: std::time::Instant = std::time::Instant::now();
            let (error_after, imu_after): (S, S) = linearize_states(
                ba, held_meas, prev_meas, &imu_lin, &states, &options, h_trial, b_trial, imu_h,
                imu_b,
            )?;
            timings.error_ns += duration_ns(mark);

            let f_diff: S = error_total - error_after;
            let relative_decrease: S = f_diff / l_diff;
            let step_is_valid: bool = l_diff > S::zero();
            let accepted: bool = step_is_valid && relative_decrease > S::zero();

            lm.push(LmIteration {
                iteration: it,
                backtrack,
                error_before: error_total,
                error_after,
                // The frame update's objective is two terms, not five: it has
                // no prior, and `ImuBlock` returns the preintegration and the
                // two bias random walks already summed.
                vision_error: error_after - imu_after,
                imu_error: imu_after,
                bias_gyro_error: S::zero(),
                bias_accel_error: S::zero(),
                marg_prior_error: S::zero(),
                l_diff,
                f_diff,
                relative_decrease,
                lambda: damping.lambda,
                step_norminf,
                solve_attempts,
                step_is_valid,
                accepted,
            });

            if accepted {
                // `:1557-1562`: Nielsen's update, in `double` as C++ deduces it.
                let x: S = S::from_literal(2.0) * relative_decrease - S::one();
                let gain: S = S::from_literal(1.0 - x.to_f64().powf(3.0));
                let floor: S = S::one() / S::from_literal(3.0);
                damping.lambda *= eigen_maxi(floor, gain);
                damping.lambda = eigen_maxi(damping.min_lambda, damping.lambda);
                damping.lambda_vee = S::from_literal(VEE_FACTOR);
                it += 1;
                backtrack = 0;

                // The trial point is where the loop now stands, and its
                // linearization is already built.
                std::mem::swap(h, h_trial);
                std::mem::swap(b, b_trial);
                error_total = error_after;

                // `:1565-1568`, both constants hard-coded in C++ too.
                if (f_diff > S::zero() && f_diff < S::from_literal(FUNCTION_TOLERANCE))
                    || step_norminf < S::from_literal(STEP_TOLERANCE)
                {
                    termination = Some(LmTermination::Converged);
                }
                continue;
            }

            // `:1585-1598`.
            damping.lambda = damping.lambda_vee * damping.lambda;
            damping.lambda_vee *= S::from_literal(VEE_FACTOR);
            for frame_id in [prev_t_ns, t_ns] {
                let Some(state) = ba.frame_states.get_mut(&frame_id) else {
                    // Unreachable, as above.
                    return Err(EstimatorError::PreviousStateMissing { t_ns: frame_id });
                };
                state.restore();
            }
            it += 1;
            backtrack += 1;
            if damping.lambda > damping.max_lambda {
                termination = Some(LmTermination::MaxDamping);
            }
        }

        Ok(Some((
            lm,
            termination.unwrap_or(LmTermination::MaxIterations),
            timings,
        )))
    }
}

/// The three states one frame update reads: the two it frees and the one behind
/// them that anchors the pair.
#[derive(Debug, Clone, Copy)]
struct FreeStates {
    /// `k−2`: held, and marginalized away at the end of this frameset.
    held_t_ns: FrameId,
    /// `k−1`: free, and frozen into the FEJ prior at the end of this frameset.
    prev_t_ns: FrameId,
    /// `k`: free, and this frameset's output pose.
    t_ns: FrameId,
}

/// Fill `h` and `b` with the two free states' normal equations at their current
/// values, and return `(the cost there, the IMU factors' share of it)`.
///
/// Three factor groups and nothing else: every observation either free frameset
/// filed on a landmark the window hosts, the IMU factor between them, and the
/// IMU factor reaching back to the held state. All three are whitened exactly as
/// the window whitens them, so the two objectives are the same function of the
/// same states.
///
/// The 30 unknowns are the two states' own, `k−1` first: pose in 0-5, velocity
/// in 6-8, gyro bias in 9-11 and accel bias in 12-14 of each
/// (`PoseVelBiasState::apply_inc`).
#[expect(clippy::too_many_arguments, reason = "every buffer is the caller's")]
fn linearize_states<S: LieScalar>(
    ba: &BundleAdjustmentBase<S>,
    held_meas: &IntegratedImuMeasurement<S>,
    prev_meas: &IntegratedImuMeasurement<S>,
    imu_lin: &ImuLinData<S>,
    states: &FreeStates,
    options: &LandmarkBlockOptions<S>,
    h: &mut DMatrix<S>,
    b: &mut DVector<S>,
    imu_h: &mut DMatrix<S>,
    imu_b: &mut DVector<S>,
) -> Result<(S, S), EstimatorError> {
    h.fill(S::zero());
    b.fill(S::zero());
    let mut error: S = S::zero();

    // Each free frameset's pose is one state's, so it is resolved once rather
    // than per observation.
    let free: [(FrameId, usize); 2] = [
        (states.prev_t_ns, PREVIOUS_OFFSET),
        (states.t_ns, CURRENT_OFFSET),
    ];
    let poses: [PoseStateWithLin<S>; 2] = [
        ba.get_pose_state_with_lin(states.prev_t_ns)?,
        ba.get_pose_state_with_lin(states.t_ns)?,
    ];
    let cameras = ba.cameras();

    for lm in ba.lmdb.landmarks() {
        let tcid_h: TimeCamId = lm.host_kf_id;
        // A landmark hosted by one of the free framesets moves with it, so that
        // observation carries a second pose block. Every other host is held and
        // its Jacobian is not formed at all.
        let host: Option<usize> = free
            .iter()
            .position(|(frame_id, _)| *frame_id == tcid_h.frame_id);
        for (target, (target_t_ns, target_offset)) in free.iter().enumerate() {
            let state_t: &PoseStateWithLin<S> = &poses[target];
            for cam_id in 0..cameras.len() {
                let tcid_t: TimeCamId = TimeCamId::new(*target_t_ns, cam_id);
                let Some(kpt_obs) = lm.obs.get(&tcid_t) else {
                    continue;
                };
                let camera = cameras.get(cam_id).ok_or(LinearizeError::UnknownCamera {
                    cam_id,
                    camera_count: cameras.len(),
                })?;

                // `linearization_abs_qr.cpp:207-241`: the Jacobians at the
                // linearization point, the value at the current state when
                // either end is frozen. A landmark observed in its own image has
                // no pose Jacobian at all (`:235-239`), which is what an
                // identity relative pose means.
                let mut d_rel_d_h: Matrix6<S> = Matrix6::zeros();
                let mut d_rel_d_t: Matrix6<S> = Matrix6::zeros();
                let t_t_h: Matrix4<S> =
                    if tcid_h == tcid_t {
                        Matrix4::identity()
                    } else {
                        let state_h: PoseStateWithLin<S> =
                            ba.get_pose_state_with_lin(tcid_h.frame_id)?;
                        let t_i_c_h: &Se3<S> = ba.calib.t_i_c.get(tcid_h.cam_id).ok_or(
                            LinearizeError::UnknownCamera {
                                cam_id: tcid_h.cam_id,
                                camera_count: ba.calib.t_i_c.len(),
                            },
                        )?;
                        let t_i_c_t: &Se3<S> =
                            ba.calib
                                .t_i_c
                                .get(cam_id)
                                .ok_or(LinearizeError::UnknownCamera {
                                    cam_id,
                                    camera_count: ba.calib.t_i_c.len(),
                                })?;
                        let mut rel: Se3<S> = compute_rel_pose(
                            state_h.pose_lin(),
                            t_i_c_h,
                            state_t.pose_lin(),
                            t_i_c_t,
                            host.map(|_| &mut d_rel_d_h),
                            Some(&mut d_rel_d_t),
                        );
                        if state_h.is_linearized() || state_t.is_linearized() {
                            rel = compute_rel_pose(
                                state_h.pose(),
                                t_i_c_h,
                                state_t.pose(),
                                t_i_c_t,
                                None,
                                None,
                            );
                        }
                        rel.matrix()
                    };

                let mut res: Vector2<S> = Vector2::zeros();
                let mut d_res_d_xi: Matrix2x6<S> = Matrix2x6::zeros();
                let valid: bool = linearize_point(
                    kpt_obs,
                    lm,
                    &t_t_h,
                    camera,
                    &mut res,
                    &mut LinearizePointOut {
                        d_res_d_xi: Some(&mut d_res_d_xi),
                        d_res_d_p: None,
                        proj: None,
                    },
                );
                // `landmark_block_abs_dynamic.hpp:152`.
                if options.use_valid_projections_only && !valid {
                    continue;
                }
                // `:153-163`: zeroed, never fatal.
                if !d_res_d_xi.iter().all(|v| v.to_f64().is_finite()) {
                    log::warn!(
                        "d_res_d_xi is not valid in the frame update, lm = {:?}",
                        lm.id
                    );
                    d_res_d_xi.fill(S::zero());
                }

                // `:168-179`, with the landmark's own column dropped: it is a
                // constant here, and so is every pose block outside `free`.
                let res_squared: S = res[0] * res[0] + res[1] * res[1];
                let (weighted_error, weight) = compute_error_weight(res_squared, options);
                let sqrt_weight: S = weight.sqrt() / options.obs_std_dev;
                error += weighted_error / (options.obs_std_dev * options.obs_std_dev);

                d_res_d_xi *= sqrt_weight;
                let residual: Vector2<S> = res * sqrt_weight;
                let mut blocks: [(usize, Matrix2x6<S>); 2] = [
                    (*target_offset, d_res_d_xi * d_rel_d_t),
                    (0, Matrix2x6::zeros()),
                ];
                let mut count: usize = 1;
                if let Some(index) = host {
                    blocks[1] = (free[index].1, d_res_d_xi * d_rel_d_h);
                    count = 2;
                }
                // Both blocks are accumulated, and a pair that lands on the same
                // offset accumulates twice — which is what makes a landmark
                // hosted and seen in the same free frameset come out right
                // (`:177-179`).
                for (row_offset, row_block) in blocks.iter().take(count) {
                    let jtr: Vector6<S> = row_block.transpose() * residual;
                    for i in 0..POSE_SIZE {
                        b[row_offset + i] += jtr[i];
                    }
                    for (column_offset, column_block) in blocks.iter().take(count) {
                        let jtj: Matrix6<S> = row_block.transpose() * column_block;
                        for i in 0..POSE_SIZE {
                            for j in 0..POSE_SIZE {
                                h[(row_offset + i, column_offset + j)] += jtj[(i, j)];
                            }
                        }
                    }
                }
            }
        }
    }

    // The IMU factor over `(k−2, k−1]`, with `k−2` held: its 30x30 system's
    // trailing corner is `k−1`'s, which is what deleting a fixed variable's rows
    // and columns comes to. Then the factor over `(k−1, k]`, both of whose ends
    // are free, whole.
    let state = |frame_id: FrameId,
                 missing: FrameId|
     -> Result<&PoseVelBiasStateWithLin<S>, EstimatorError> {
        ba.frame_states
            .get(&frame_id)
            .ok_or(EstimatorError::ImuFactorStateMissing {
                start_t_ns: states.held_t_ns,
                end_t_ns: states.t_ns,
                missing_t_ns: missing,
            })
    };
    let held: &PoseVelBiasStateWithLin<S> = state(states.held_t_ns, states.held_t_ns)?;
    let previous: &PoseVelBiasStateWithLin<S> = state(states.prev_t_ns, states.prev_t_ns)?;
    let current: &PoseVelBiasStateWithLin<S> = state(states.t_ns, states.t_ns)?;

    let mut imu_error: S = S::zero();
    for (meas, start, end, folded) in [
        (held_meas, held, previous, POSE_VEL_BIAS_SIZE),
        (prev_meas, previous, current, 0),
    ] {
        let block: ImuBlock<S> = ImuBlock::linearize(meas, imu_lin, start, end);
        imu_h.fill(S::zero());
        imu_b.fill(S::zero());
        block.add_dense_h_b(PREVIOUS_OFFSET, CURRENT_OFFSET, imu_h, imu_b);
        let size: usize = FREE_SIZE - folded;
        for i in 0..size {
            for j in 0..size {
                h[(i, j)] += imu_h[(folded + i, folded + j)];
            }
            b[i] += imu_b[folded + i];
        }
        imu_error += block.error;
    }
    error += imu_error;

    Ok((error, imu_error))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use nalgebra::{Vector3, Vector4};

    use super::*;
    use crate::calib::Calibration;
    use crate::camera::CameraEnum;
    use crate::config::VioConfig;
    use crate::imu::{ImuNoise, ImuSample};
    use crate::landmark::{Landmark, StereographicParam};
    use crate::lie::So3;
    use crate::types::{LandmarkId, PoseVelBiasState};

    const CALIB: &str = include_str!("../../tests/fixtures/msdmi_calib.json");
    const CONFIG: &str = include_str!("../../../../configs/msdmi_config.json");

    /// The host keyframe, then the three states: one held and two free.
    const HOST_T_NS: i64 = 0;
    /// See [`HOST_T_NS`].
    const HELD_T_NS: i64 = 20_000_000;
    /// See [`HOST_T_NS`].
    const PREV_T_NS: i64 = 40_000_000;
    /// See [`HOST_T_NS`].
    const CURRENT_T_NS: i64 = 60_000_000;

    /// A window whose IMU factors and whose observations all point at one
    /// trajectory.
    ///
    /// The host keyframe sits at the origin and hosts twelve landmarks two metres
    /// out. Each state is *defined* as the preintegration's prediction from the
    /// one before it, so both IMU residuals are zero along it, and every pixel is
    /// a landmark projected through the state that sees it, so the reprojection
    /// residuals are zero too. The global minimum is therefore the truth at cost
    /// zero, which is what makes this a known answer rather than a regression
    /// baseline.
    ///
    /// Returns the estimator with both free states left **at** the truth, and the
    /// two true states beside it.
    fn a_window(iterations: i32) -> (SqrtKeypointVio<f64>, [PoseVelBiasState<f64>; 2]) {
        let mut config: VioConfig = VioConfig::from_json_str(CONFIG).unwrap();
        config.port_frame_update_max_iterations = iterations;
        let calibration: Calibration<f64> = Calibration::from_json_str(CALIB).unwrap();
        let gravity: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81);
        let mut vio: SqrtKeypointVio<f64> =
            SqrtKeypointVio::new(gravity, calibration, config).unwrap();

        // The host keyframe, frozen as a marginalized pose block is.
        vio.kf_ids.insert(HOST_T_NS);
        vio.ba.frame_poses.insert(
            HOST_T_NS,
            PoseStateWithLin::new(HOST_T_NS, Se3::identity(), true),
        );

        // The oldest state, half a metre along `x` and looking the same way. It
        // is the one the pair is anchored against and the one this frameset
        // marginalizes away.
        let zero: Vector3<f64> = Vector3::zeros();
        let mut truth: PoseVelBiasState<f64> = PoseVelBiasState::new(
            HELD_T_NS,
            Se3::new(So3::identity(), Vector3::new(0.5, 0.0, 0.0)),
            Vector3::new(0.1, 0.0, 0.0),
            zero,
            zero,
        );
        vio.ba
            .frame_states
            .insert(HELD_T_NS, PoseVelBiasStateWithLin::new(truth, false));

        // Two preintegrated intervals, each folded sample by sample at the rig's
        // own rate and with the rig's own noise, exactly as `process_frame` folds
        // them: a single wide step leaves the covariance stiff enough that the
        // whitened system has no significant digits left in f64 and the loop
        // stalls six orders above the minimum. Gravity is cancelled, so the
        // motion is the previous velocity plus a small turn.
        let noise: ImuNoise<f64> = ImuNoise::from_calibration(&vio.ba.calib);
        let step_ns: i64 = (1e9 / vio.ba.calib.imu_update_rate) as i64;
        let mut free: [PoseVelBiasState<f64>; 2] = [truth; 2];
        for (index, (start_t_ns, end_t_ns)) in [(HELD_T_NS, PREV_T_NS), (PREV_T_NS, CURRENT_T_NS)]
            .into_iter()
            .enumerate()
        {
            let mut meas: IntegratedImuMeasurement<f64> =
                IntegratedImuMeasurement::new(start_t_ns, &truth.bias_gyro, &truth.bias_accel);
            let mut t_ns: i64 = start_t_ns + step_ns;
            while t_ns <= end_t_ns {
                meas.integrate(
                    &ImuSample {
                        t_ns,
                        gyro: Vector3::new(0.05, -0.03, 0.02),
                        accel: Vector3::new(0.0, 0.0, 9.81),
                    },
                    &noise.accel_cov,
                    &noise.gyro_cov,
                )
                .unwrap();
                t_ns += step_ns;
            }
            let predicted = meas.predict_state(&truth.pose_vel_state(), &gravity);
            truth = PoseVelBiasState::new(
                end_t_ns,
                predicted.t_w_i,
                predicted.vel_w_i,
                truth.bias_gyro,
                truth.bias_accel,
            );
            free[index] = truth;
            vio.imu_meas.insert(start_t_ns, meas);
            vio.ba
                .frame_states
                .insert(end_t_ns, PoseVelBiasStateWithLin::new(truth, false));
        }
        vio.last_state_t_ns = CURRENT_T_NS;
        vio.opt_started = true;

        // Twelve landmarks in a grid two metres in front of the host camera, each
        // observed by every camera of both free framesets that can see it — at
        // the pixel the truth projects it to, which is what makes the truth a
        // zero-cost point.
        let cameras: Vec<CameraEnum<f64>> = vio.ba.cameras().to_vec();
        let host: TimeCamId = TimeCamId::new(HOST_T_NS, 0);
        let mut next_id: u64 = 0;
        for row in -1..=1_i32 {
            for column in -2..=1_i32 {
                let point: Vector4<f64> =
                    Vector4::new(f64::from(column) * 0.3, f64::from(row) * 0.3, 2.0, 0.0);
                let landmark: Landmark<f64> = Landmark::new(
                    LandmarkId(next_id),
                    host,
                    StereographicParam::project(&point),
                    1.0 / crate::eigen::norm3(point[0], point[1], point[2]),
                );
                next_id += 1;

                let mut filed: bool = false;
                for state in free {
                    for (cam_id, camera) in cameras.iter().enumerate() {
                        let rel: Se3<f64> = compute_rel_pose(
                            &Se3::identity(),
                            &vio.ba.calib.t_i_c[host.cam_id],
                            &state.t_w_i,
                            &vio.ba.calib.t_i_c[cam_id],
                            None,
                            None,
                        );
                        let mut pixel: Vector2<f64> = Vector2::zeros();
                        let visible: bool = linearize_point(
                            &Vector2::zeros(),
                            &landmark,
                            &rel.matrix(),
                            camera,
                            &mut pixel,
                            &mut LinearizePointOut::default(),
                        );
                        if !visible {
                            continue;
                        }
                        if !filed {
                            vio.ba.lmdb.add_landmark(landmark.id, &landmark);
                            filed = true;
                        }
                        vio.ba
                            .lmdb
                            .add_observation(TimeCamId::new(state.t_ns, cam_id), landmark.id, pixel)
                            .unwrap();
                    }
                }
            }
        }
        assert!(
            vio.ba.lmdb.num_observations() >= 24,
            "the fixture has to give both poses something to see: {}",
            vio.ba.lmdb.num_observations()
        );

        (vio, free)
    }

    /// A small twist, velocity and bias step off the truth, applied to both free
    /// states.
    fn push_off(vio: &mut SqrtKeypointVio<f64>, scale: f64) {
        let mut perturbation: Vector15<f64> = Vector15::zeros();
        perturbation
            .fixed_rows_mut::<3>(0)
            .copy_from(&(Vector3::new(0.02, -0.015, 0.01) * scale));
        perturbation
            .fixed_rows_mut::<3>(3)
            .copy_from(&(Vector3::new(0.004, 0.006, -0.003) * scale));
        perturbation
            .fixed_rows_mut::<3>(6)
            .copy_from(&(Vector3::new(0.05, -0.04, 0.03) * scale));
        for frame_id in [PREV_T_NS, CURRENT_T_NS] {
            vio.ba
                .frame_states
                .get_mut(&frame_id)
                .unwrap()
                .apply_inc(&perturbation);
        }
    }

    /// How far a recovered state may sit from the truth: the fixture's minimum is
    /// exact, so this is convergence and not agreement. Measured from the
    /// perturbation above, which converges in three LM steps
    /// (2.0e6 -> 2.3 -> 5.4e-6 -> 1.7e-12): 1.3e-10 m, 4.4e-12 rad and
    /// 5.0e-9 m/s, worst over the two free states.
    const CONVERGENCE_TOLERANCE: f64 = 1e-6;

    /// What the cost at the recovered point may be, against 2.0e6 at the
    /// perturbation. Measured: 1.7e-12.
    const COST_TOLERANCE: f64 = 1e-9;

    /// The known answer: two states pushed off a zero-cost minimum come back to
    /// it.
    ///
    /// All three factor groups agree at the truth by construction, so the frame
    /// update has one thing to find and the assertion is against that value
    /// rather than against a recorded run.
    #[test]
    fn the_frame_update_recovers_states_pushed_off_a_zero_cost_minimum() {
        let (mut vio, truth) = a_window(5);
        push_off(&mut vio, 1.0);

        let (lm, _, _) = vio.frame_update(CURRENT_T_NS).unwrap().unwrap();
        assert!(!lm.is_empty(), "the loop has to take at least one step");
        let last = lm.last().unwrap();
        assert!(
            last.error_after < COST_TOLERANCE,
            "the minimum is exact, so the cost has to reach it: {}",
            last.error_after
        );
        assert!(
            lm.iter().all(|step| step.accepted),
            "a quadratic with an exact minimum should not need a backtrack"
        );

        for (frame_id, expected) in [(PREV_T_NS, truth[0]), (CURRENT_T_NS, truth[1])] {
            let recovered: &PoseVelBiasState<f64> = vio.ba.frame_states[&frame_id].state();
            let position: f64 = (recovered.t_w_i.translation - expected.t_w_i.translation).norm();
            let rotation: f64 = (recovered.t_w_i.rotation * expected.t_w_i.rotation.inverse())
                .log()
                .norm();
            let velocity: f64 = (recovered.vel_w_i - expected.vel_w_i).norm();
            assert!(position < CONVERGENCE_TOLERANCE, "{frame_id}: {position}");
            assert!(rotation < CONVERGENCE_TOLERANCE, "{frame_id}: {rotation}");
            assert!(velocity < CONVERGENCE_TOLERANCE, "{frame_id}: {velocity}");
        }
    }

    /// The step cap is the knob's, and it counts accepted and backtracked steps
    /// together as `vio_max_iterations` does (D12).
    #[test]
    fn the_knob_caps_the_steps() {
        for cap in 1..=3_i32 {
            let (mut vio, _) = a_window(cap);
            push_off(&mut vio, 10.0);
            let (lm, _, _) = vio.frame_update(CURRENT_T_NS).unwrap().unwrap();
            assert!(
                i32::try_from(lm.len()).unwrap() <= cap + 1,
                "cap {cap} took {} steps",
                lm.len()
            );
        }
    }

    /// A frameset the frame update cannot serve goes back to the joint solve
    /// rather than being solved wrong, and every test reads the window alone.
    #[test]
    fn a_frameset_the_frame_update_cannot_serve_is_declined() {
        // No preintegration reaching back to the held state, and none joining
        // the two free ones.
        for missing in [HELD_T_NS, PREV_T_NS] {
            let (mut vio, _) = a_window(2);
            vio.imu_meas.remove(&missing);
            assert!(vio.frame_update(CURRENT_T_NS).unwrap().is_none());
        }

        // A frameset that is not the newest state.
        let (mut vio, _) = a_window(2);
        assert!(vio.frame_update(PREV_T_NS).unwrap().is_none());

        // Only two states, so there is nothing to anchor the pair against.
        let (mut vio, _) = a_window(2);
        vio.ba.frame_states.remove(&HELD_T_NS);
        assert!(vio.frame_update(CURRENT_T_NS).unwrap().is_none());

        // A prior that orders a state this would move: its gradient would not be
        // zero. Either one is enough.
        for ordered in [PREV_T_NS, CURRENT_T_NS] {
            let (mut vio, _) = a_window(2);
            vio.marg_data
                .order
                .push(ordered, POSE_VEL_BIAS_SIZE)
                .unwrap();
            assert!(vio.frame_update(CURRENT_T_NS).unwrap().is_none());
        }

        // A long-term keyframe among the free states, whose pose both of the
        // window's own linearizations pin.
        for pinned in [PREV_T_NS, CURRENT_T_NS] {
            let (mut vio, _) = a_window(2);
            vio.ltkfs.insert(pinned);
            assert!(vio.frame_update(CURRENT_T_NS).unwrap().is_none());
        }
    }

    /// A landmark hosted by one of the free framesets carries a second pose
    /// block, and dropping it would leave the residual's own host moving with no
    /// Jacobian to say so.
    ///
    /// It happens whenever the frameset before this one took a keyframe, which is
    /// one frameset in 7.65 on MIO10.
    #[test]
    fn a_landmark_hosted_by_a_free_frameset_moves_with_it() {
        let (mut vio, truth) = a_window(5);
        // Re-host the first landmark on the older free frameset, keeping the
        // direction it would have had from there.
        let id: LandmarkId = vio.ba.lmdb.landmarks()[0].id;
        let observed: Vector2<f64> =
            vio.ba.lmdb.get_landmark(id).unwrap().obs[&TimeCamId::new(PREV_T_NS, 0)];
        let rehosted: Landmark<f64> = Landmark::new(
            id,
            TimeCamId::new(PREV_T_NS, 0),
            vio.ba.lmdb.get_landmark(id).unwrap().direction,
            vio.ba.lmdb.get_landmark(id).unwrap().inv_dist,
        );
        vio.ba.lmdb.add_landmark(id, &rehosted);
        vio.ba
            .lmdb
            .add_observation(TimeCamId::new(PREV_T_NS, 0), id, observed)
            .unwrap();
        push_off(&mut vio, 1.0);

        let (lm, _, _) = vio.frame_update(CURRENT_T_NS).unwrap().unwrap();
        // The re-hosted landmark's direction is the host camera's, not this one's,
        // so the fixture is no longer a zero-cost minimum; what has to hold is
        // that the loop still descends and both states stay finite.
        assert!(lm.last().unwrap().error_after < lm[0].error_before);
        for (frame_id, expected) in [(PREV_T_NS, truth[0]), (CURRENT_T_NS, truth[1])] {
            let recovered: &PoseVelBiasState<f64> = vio.ba.frame_states[&frame_id].state();
            assert!(recovered.t_w_i.translation.iter().all(|v| v.is_finite()));
            assert!(
                (recovered.t_w_i.translation - expected.t_w_i.translation).norm() < 0.1,
                "{frame_id} left the neighbourhood"
            );
        }
    }

    /// Offline mode lets nothing but the data reach a decision: the same window
    /// solved twice gives the same states, coefficient for coefficient.
    #[test]
    fn a_repeat_frame_update_is_bit_identical() {
        /// The states the solve left behind and the trail it took to get there.
        type Solved = ([PoseVelBiasState<f64>; 2], Vec<(i32, f64, f64, bool)>);
        let solve = || -> Solved {
            let (mut vio, _) = a_window(3);
            push_off(&mut vio, 1.0);
            let (lm, _, _) = vio.frame_update(CURRENT_T_NS).unwrap().unwrap();
            (
                [
                    *vio.ba.frame_states[&PREV_T_NS].state(),
                    *vio.ba.frame_states[&CURRENT_T_NS].state(),
                ],
                lm.iter()
                    .map(|step| (step.iteration, step.error_after, step.l_diff, step.accepted))
                    .collect(),
            )
        };
        assert_eq!(solve(), solve());
    }
}
