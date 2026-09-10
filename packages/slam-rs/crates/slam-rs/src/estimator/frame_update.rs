//! The non-keyframe frame update (D76): the newest state alone, against fixed
//! landmarks and its IMU factor.
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
//! ## What is held, and why the prior is not here
//!
//! Landmarks, their host keyframes and every older state are constants, so the
//! only free block is the newest state's 15 unknowns. The marginalization prior
//! covers only blocks frozen at a linearization point — `computeDelta` refuses
//! any other (`ba_base.cpp:294`) — and the newest state is appended unfrozen, so
//! the prior's cost does not depend on the one variable this solves for and
//! contributes neither a Jacobian nor a gradient. [`SqrtKeypointVio::frame_update`]
//! checks that per frame and hands the frameset back to the joint solve if the
//! prior ever does order it.
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
use crate::linearize::{LandmarkBlockOptions, compute_error_weight};
use crate::types::{
    FrameId, POSE_SIZE, POSE_VEL_BIAS_SIZE, PoseVelBiasStateWithLin, TimeCamId, Vector15,
};

/// The two states an IMU factor spans, stacked, which is what
/// [`ImuBlock::add_dense_h_b`] writes into.
const IMU_BLOCK_SIZE: usize = 2 * POSE_VEL_BIAS_SIZE;

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
    /// The IMU factor's own 30x30 system, of which the trailing 15x15 corner is
    /// the newest state's.
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
            h: DMatrix::zeros(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE),
            b: DVector::zeros(POSE_VEL_BIAS_SIZE),
            h_trial: DMatrix::zeros(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE),
            b_trial: DVector::zeros(POSE_VEL_BIAS_SIZE),
            imu_h: DMatrix::zeros(IMU_BLOCK_SIZE, IMU_BLOCK_SIZE),
            imu_b: DVector::zeros(IMU_BLOCK_SIZE),
            solve: EigenLdlt::empty(),
            increment: DVector::zeros(POSE_VEL_BIAS_SIZE),
        }
    }
}

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// Solve the newest state against fixed landmarks and its IMU factor.
    ///
    /// `Ok(None)` means the frameset is not one this can serve — no previous
    /// state, no IMU factor joining it to this one, or a prior that orders the
    /// newest state — and the caller must run the joint solve instead. Every one
    /// of those tests reads the window alone, so a replay repeats the choice.
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
        // The newest state has to be this frameset's, and it has to have a
        // predecessor and the preintegration that joins them.
        if self.ba.frame_states.keys().next_back() != Some(&t_ns) {
            return Ok(None);
        }
        let Some(prev_t_ns) = self.ba.frame_states.keys().rev().nth(1).copied() else {
            return Ok(None);
        };
        let Some(meas) = self.imu_meas.get(&prev_t_ns) else {
            return Ok(None);
        };
        if prev_t_ns.checked_add(meas.get_dt_ns()) != Some(t_ns) {
            return Ok(None);
        }
        // The prior is a quadratic in blocks frozen at a linearization point and
        // the newest state is not one, so it cannot order it; if it ever does,
        // its gradient is not zero here and this solve would be wrong.
        if self.marg_data.order.get(t_ns).is_some() {
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
        let Some(meas) = imu_meas.get(&prev_t_ns) else {
            // Unreachable: the same lookup succeeded above and nothing since has
            // touched `imu_meas`.
            return Ok(None);
        };

        // `:1249`, D11: the trust region has no memory across framesets.
        damping.lambda = S::from_literal(config.vio_lm_lambda_initial);

        let mark: std::time::Instant = std::time::Instant::now();
        let (mut error_total, _): (S, S) = linearize_state(
            ba, meas, &imu_lin, prev_t_ns, t_ns, &options, h, b, imu_h, imu_b,
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
            let inc: Vector15<S> = Vector15::from_iterator(increment.iter().copied());

            // `−(inc·b + ½ incᵀ H inc)`, the model's predicted decrease. With no
            // eliminated block this is `backSubstitute`'s sum in closed form.
            let mut linear: S = S::zero();
            let mut quadratic: S = S::zero();
            for i in 0..POSE_VEL_BIAS_SIZE {
                linear += inc[i] * b[i];
                for j in 0..POSE_VEL_BIAS_SIZE {
                    quadratic += inc[i] * h[(i, j)] * inc[j];
                }
            }
            let l_diff: S = -(linear + S::from_literal(0.5) * quadratic);

            let Some(state) = ba.frame_states.get_mut(&t_ns) else {
                // Unreachable: the key came from this map and nothing removes
                // one inside the loop.
                return Err(EstimatorError::PreviousStateMissing { t_ns });
            };
            state.backup();
            state.apply_inc(&inc);

            // `:1477`, folded left to right like the window's.
            let mut step_norminf: S = S::zero();
            for value in inc.iter() {
                step_norminf = eigen_maxi(step_norminf, value.abs());
            }

            let mark: std::time::Instant = std::time::Instant::now();
            let (error_after, imu_after): (S, S) = linearize_state(
                ba, meas, &imu_lin, prev_t_ns, t_ns, &options, h_trial, b_trial, imu_h, imu_b,
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
            let Some(state) = ba.frame_states.get_mut(&t_ns) else {
                // Unreachable, as above.
                return Err(EstimatorError::PreviousStateMissing { t_ns });
            };
            state.restore();
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

/// Fill `h` and `b` with the newest state's normal equations at its current
/// value, and return `(the cost there, the IMU factor's share of it)`.
///
/// Two factor groups and nothing else: every observation the frameset filed on a
/// landmark the window hosts, and the IMU factor from the previous state. Both
/// are whitened exactly as the window whitens them, so the two objectives are
/// the same function of the same state.
///
/// The 15 unknowns are the state's own: pose in 0-5, velocity in 6-8, gyro bias
/// in 9-11 and accel bias in 12-14 (`PoseVelBiasState::apply_inc`).
#[expect(clippy::too_many_arguments, reason = "every buffer is the caller's")]
fn linearize_state<S: LieScalar>(
    ba: &BundleAdjustmentBase<S>,
    meas: &IntegratedImuMeasurement<S>,
    imu_lin: &ImuLinData<S>,
    prev_t_ns: FrameId,
    t_ns: FrameId,
    options: &LandmarkBlockOptions<S>,
    h: &mut DMatrix<S>,
    b: &mut DVector<S>,
    imu_h: &mut DMatrix<S>,
    imu_b: &mut DVector<S>,
) -> Result<(S, S), EstimatorError> {
    h.fill(S::zero());
    b.fill(S::zero());
    let mut error: S = S::zero();

    // The target's pose is one state's, so it is resolved once rather than per
    // observation. It is never frozen — only `last_state_to_marg` is, and that
    // is never the newest state — but the general form is kept so the two
    // linearizations cannot drift apart.
    let state_t = ba.get_pose_state_with_lin(t_ns)?;
    let cameras = ba.cameras();

    for lm in ba.lmdb.landmarks() {
        for cam_id in 0..cameras.len() {
            let tcid_t: TimeCamId = TimeCamId::new(t_ns, cam_id);
            let Some(kpt_obs) = lm.obs.get(&tcid_t) else {
                continue;
            };
            let tcid_h: TimeCamId = lm.host_kf_id;
            let camera = cameras
                .get(cam_id)
                .ok_or(crate::linearize::LinearizeError::UnknownCamera {
                    cam_id,
                    camera_count: cameras.len(),
                })?;

            // `linearization_abs_qr.cpp:207-241`: the Jacobian at the
            // linearization point, the value at the current state when either
            // end is frozen. A landmark hosted by this frameset has no pose
            // Jacobian at all (`:235-239`), which is what an identity relative
            // pose means.
            let (t_t_h, d_rel_d_t): (Matrix4<S>, Matrix6<S>) = if tcid_h == tcid_t {
                (Matrix4::identity(), Matrix6::zeros())
            } else {
                let state_h = ba.get_pose_state_with_lin(tcid_h.frame_id)?;
                let t_i_c_h: &Se3<S> = ba.calib.t_i_c.get(tcid_h.cam_id).ok_or(
                    crate::linearize::LinearizeError::UnknownCamera {
                        cam_id: tcid_h.cam_id,
                        camera_count: ba.calib.t_i_c.len(),
                    },
                )?;
                let t_i_c_t: &Se3<S> = ba.calib.t_i_c.get(cam_id).ok_or(
                    crate::linearize::LinearizeError::UnknownCamera {
                        cam_id,
                        camera_count: ba.calib.t_i_c.len(),
                    },
                )?;
                let mut d_rel_d_t: Matrix6<S> = Matrix6::zeros();
                let mut rel: Se3<S> = compute_rel_pose(
                    state_h.pose_lin(),
                    t_i_c_h,
                    state_t.pose_lin(),
                    t_i_c_t,
                    None,
                    Some(&mut d_rel_d_t),
                );
                if state_h.is_linearized() || state_t.is_linearized() {
                    rel = compute_rel_pose(state_h.pose(), t_i_c_h, state_t.pose(), t_i_c_t, None, None);
                }
                (rel.matrix(), d_rel_d_t)
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
                log::warn!("d_res_d_xi is not valid in the frame update, lm = {:?}", lm.id);
                d_res_d_xi.fill(S::zero());
            }

            // `:168-179`, with the host block dropped: the landmark and its host
            // are constants here.
            let res_squared: S = res[0] * res[0] + res[1] * res[1];
            let (weighted_error, weight) = compute_error_weight(res_squared, options);
            let sqrt_weight: S = weight.sqrt() / options.obs_std_dev;
            error += weighted_error / (options.obs_std_dev * options.obs_std_dev);

            d_res_d_xi *= sqrt_weight;
            let jacobian: Matrix2x6<S> = d_res_d_xi * d_rel_d_t;
            let residual: Vector2<S> = res * sqrt_weight;
            let jtj: Matrix6<S> = jacobian.transpose() * jacobian;
            let jtr: Vector6<S> = jacobian.transpose() * residual;
            for i in 0..POSE_SIZE {
                for j in 0..POSE_SIZE {
                    h[(i, j)] += jtj[(i, j)];
                }
                b[i] += jtr[i];
            }
        }
    }

    // The IMU factor over `(prev, t_ns]`, with the previous state held: its
    // 30x30 system's trailing corner is the newest state's, which is what
    // deleting a fixed variable's rows and columns comes to.
    let start_state: &PoseVelBiasStateWithLin<S> = ba.frame_states.get(&prev_t_ns).ok_or(
        EstimatorError::ImuFactorStateMissing {
            start_t_ns: prev_t_ns,
            end_t_ns: t_ns,
            missing_t_ns: prev_t_ns,
        },
    )?;
    let end_state: &PoseVelBiasStateWithLin<S> = ba.frame_states.get(&t_ns).ok_or(
        EstimatorError::ImuFactorStateMissing {
            start_t_ns: prev_t_ns,
            end_t_ns: t_ns,
            missing_t_ns: t_ns,
        },
    )?;
    let block: ImuBlock<S> = ImuBlock::linearize(meas, imu_lin, start_state, end_state);
    imu_h.fill(S::zero());
    imu_b.fill(S::zero());
    block.add_dense_h_b(0, POSE_VEL_BIAS_SIZE, imu_h, imu_b);
    for i in 0..POSE_VEL_BIAS_SIZE {
        for j in 0..POSE_VEL_BIAS_SIZE {
            h[(i, j)] += imu_h[(POSE_VEL_BIAS_SIZE + i, POSE_VEL_BIAS_SIZE + j)];
        }
        b[i] += imu_b[POSE_VEL_BIAS_SIZE + i];
    }
    error += block.error;

    Ok((error, block.error))
}
