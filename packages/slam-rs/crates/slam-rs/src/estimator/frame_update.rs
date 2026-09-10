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

/// Which precondition sent a frameset back to the joint solve.
///
/// [`SqrtKeypointVio::frame_update`] serves only a frameset whose newest state
/// is genuinely joined to its predecessor by the preintegration the joint solve
/// itself would use, and only while the prior leaves that state free. Naming
/// each refusal is what makes "the update never engaged on this clip" a
/// measurement rather than a guess: the variant travels out through
/// [`FrameUpdateOutcome`] on every [`super::FrameStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameUpdateDecline {
    /// The newest state in the window is not this frameset's.
    NotNewest,
    /// The newest state has no predecessor in the window.
    NoPredecessor,
    /// No preintegration in `imu_meas` starts at the predecessor.
    NoImuFactor,
    /// The preintegration that starts at the predecessor does not end at this
    /// frameset, so it is not the factor that joins the two.
    ImuIntervalGap,
    /// The marginalization prior orders the newest state, so its gradient here
    /// is not zero and a solve that omits it would be wrong.
    PriorOrdersState,
}

/// What the frame update did with one frameset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameUpdateOutcome {
    /// The knob is off, the frameset took a keyframe, or the warmup still owns
    /// the window: [`SqrtKeypointVio::frame_update`] was never called.
    NotAttempted,
    /// The frame update solved this frameset.
    Taken,
    /// A precondition refused it and the joint solve ran instead.
    Declined(FrameUpdateDecline),
}

/// Either the solve, or the precondition that refused the frameset.
pub(super) type FrameUpdateResult<S> = Result<SolveOutcome<S>, FrameUpdateDecline>;

impl FrameUpdateOutcome {
    /// A stable name per outcome, for the Python snapshot and for logs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotAttempted => "not_attempted",
            Self::Taken => "taken",
            Self::Declined(FrameUpdateDecline::NotNewest) => "declined_not_newest",
            Self::Declined(FrameUpdateDecline::NoPredecessor) => "declined_no_predecessor",
            Self::Declined(FrameUpdateDecline::NoImuFactor) => "declined_no_imu_factor",
            Self::Declined(FrameUpdateDecline::ImuIntervalGap) => "declined_imu_interval_gap",
            Self::Declined(FrameUpdateDecline::PriorOrdersState) => "declined_prior_orders_state",
        }
    }
}

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
    /// `Err(FrameUpdateDecline)` means the frameset is not one this can serve —
    /// no previous state, no IMU factor joining it to this one, or a prior that
    /// orders the newest state — and the caller must run the joint solve
    /// instead. Every one of those tests reads the window alone, so a replay
    /// repeats the choice, and the variant is reported per frameset so a clip
    /// where the update never engages says which precondition refused it.
    ///
    /// # Errors
    ///
    /// [`EstimatorError::BundleAdjustment`] where a pose the residual needs is
    /// not in the window, and [`EstimatorError::Linearize`] where the camera id
    /// of an observation is not in the rig.
    pub(super) fn frame_update(
        &mut self,
        t_ns: i64,
    ) -> Result<FrameUpdateResult<S>, EstimatorError> {
        // The newest state has to be this frameset's, and it has to have a
        // predecessor and the preintegration that joins them.
        if self.ba.frame_states.keys().next_back() != Some(&t_ns) {
            return Ok(Err(FrameUpdateDecline::NotNewest));
        }
        let Some(prev_t_ns) = self.ba.frame_states.keys().rev().nth(1).copied() else {
            return Ok(Err(FrameUpdateDecline::NoPredecessor));
        };
        let Some(meas) = self.imu_meas.get(&prev_t_ns) else {
            return Ok(Err(FrameUpdateDecline::NoImuFactor));
        };
        if prev_t_ns.checked_add(meas.get_dt_ns()) != Some(t_ns) {
            return Ok(Err(FrameUpdateDecline::ImuIntervalGap));
        }
        // The prior is a quadratic in blocks frozen at a linearization point and
        // the newest state is not one, so it cannot order it; if it ever does,
        // its gradient is not zero here and this solve would be wrong.
        if self.marg_data.order.get(t_ns).is_some() {
            return Ok(Err(FrameUpdateDecline::PriorOrdersState));
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
            return Ok(Err(FrameUpdateDecline::NoImuFactor));
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

        Ok(Ok((
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
            let camera =
                cameras
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
                    rel = compute_rel_pose(
                        state_h.pose(),
                        t_i_c_h,
                        state_t.pose(),
                        t_i_c_t,
                        None,
                        None,
                    );
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
                log::warn!(
                    "d_res_d_xi is not valid in the frame update, lm = {:?}",
                    lm.id
                );
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
    let start_state: &PoseVelBiasStateWithLin<S> =
        ba.frame_states
            .get(&prev_t_ns)
            .ok_or(EstimatorError::ImuFactorStateMissing {
                start_t_ns: prev_t_ns,
                end_t_ns: t_ns,
                missing_t_ns: prev_t_ns,
            })?;
    let end_state: &PoseVelBiasStateWithLin<S> =
        ba.frame_states
            .get(&t_ns)
            .ok_or(EstimatorError::ImuFactorStateMissing {
                start_t_ns: prev_t_ns,
                end_t_ns: t_ns,
                missing_t_ns: t_ns,
            })?;
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
    use crate::types::{LandmarkId, PoseStateWithLin, PoseVelBiasState};

    const CALIB: &str = include_str!("../../tests/fixtures/msdmi_calib.json");
    const CONFIG: &str = include_str!("../../../../configs/msdmi_config.json");
    /// The four-camera MGO rig, which is what MGO09 replays.
    const CALIB_4: &str = include_str!("../../tests/fixtures/msdmg_calib.json");
    /// See [`CALIB_4`].
    const CONFIG_4: &str = include_str!("../../../../configs/msdmg_config.json");

    /// The host keyframe, the previous state and the newest state.
    const HOST_T_NS: i64 = 0;
    /// See [`HOST_T_NS`].
    const PREV_T_NS: i64 = 20_000_000;
    /// See [`HOST_T_NS`].
    const CURRENT_T_NS: i64 = 40_000_000;

    /// A window whose IMU factor and whose observations both point at one pose.
    ///
    /// The host keyframe sits at the origin and hosts twelve landmarks two
    /// metres out. The truth is *defined* as the preintegration's prediction from
    /// the previous state, so the IMU residual is zero there, and every pixel is
    /// the projection of a landmark through that same pose, so the reprojection
    /// residual is zero there too. The global minimum is therefore the truth at
    /// cost zero, which is what makes this a known answer rather than a
    /// regression baseline.
    ///
    /// Returns the estimator with the newest state left **at** the truth, and the
    /// truth beside it.
    fn a_window(iterations: i32) -> (SqrtKeypointVio<f64>, PoseVelBiasState<f64>) {
        a_window_on(CALIB, CONFIG, iterations)
    }

    /// [`a_window`] on a named rig, so the same known answer can be asked of a
    /// two-camera and a four-camera window.
    fn a_window_on(
        calib: &str,
        config: &str,
        iterations: i32,
    ) -> (SqrtKeypointVio<f64>, PoseVelBiasState<f64>) {
        let mut config: VioConfig = VioConfig::from_json_str(config).unwrap();
        config.port_frame_update_max_iterations = iterations;
        let calibration: Calibration<f64> = Calibration::from_json_str(calib).unwrap();
        let gravity: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81);
        let mut vio: SqrtKeypointVio<f64> =
            SqrtKeypointVio::new(gravity, calibration, config).unwrap();

        // The host keyframe, frozen as a marginalized pose block is.
        vio.kf_ids.insert(HOST_T_NS);
        vio.ba.frame_poses.insert(
            HOST_T_NS,
            PoseStateWithLin::new(HOST_T_NS, Se3::identity(), true),
        );

        // The previous state, half a metre along `x` and looking the same way.
        let zero: Vector3<f64> = Vector3::zeros();
        let previous: PoseVelBiasState<f64> = PoseVelBiasState::new(
            PREV_T_NS,
            Se3::new(So3::identity(), Vector3::new(0.5, 0.0, 0.0)),
            Vector3::new(0.1, 0.0, 0.0),
            zero,
            zero,
        );
        vio.ba
            .frame_states
            .insert(PREV_T_NS, PoseVelBiasStateWithLin::new(previous, false));

        // One preintegrated interval over the 20 ms between them, folded sample
        // by sample at the rig's own rate and with the rig's own noise, exactly
        // as `process_frame` folds it: a single wide step leaves the covariance
        // stiff enough that the whitened problem has no significant digits left.
        // Gravity is cancelled, so the motion is the previous velocity plus a
        // small turn.
        let noise: ImuNoise<f64> = ImuNoise::from_calibration(&vio.ba.calib);
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(PREV_T_NS, &zero, &zero);
        let step_ns: i64 = (1e9 / vio.ba.calib.imu_update_rate) as i64;
        let mut t_ns: i64 = PREV_T_NS + step_ns;
        while t_ns <= CURRENT_T_NS {
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
        let predicted = meas.predict_state(&previous.pose_vel_state(), &gravity);
        let truth: PoseVelBiasState<f64> = PoseVelBiasState::new(
            CURRENT_T_NS,
            predicted.t_w_i,
            predicted.vel_w_i,
            previous.bias_gyro,
            previous.bias_accel,
        );
        vio.imu_meas.insert(PREV_T_NS, meas);
        vio.ba
            .frame_states
            .insert(CURRENT_T_NS, PoseVelBiasStateWithLin::new(truth, false));
        vio.last_state_t_ns = CURRENT_T_NS;
        vio.opt_started = true;

        // Twelve landmarks in a grid two metres in front of the host camera,
        // each observed by every camera that can see it — at the pixel the truth
        // projects it to, which is what makes the truth a zero-cost point.
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
                for (cam_id, camera) in cameras.iter().enumerate() {
                    let rel: Se3<f64> = compute_rel_pose(
                        &Se3::identity(),
                        &vio.ba.calib.t_i_c[host.cam_id],
                        &truth.t_w_i,
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
                        .add_observation(TimeCamId::new(CURRENT_T_NS, cam_id), landmark.id, pixel)
                        .unwrap();
                }
            }
        }
        assert!(
            vio.ba.lmdb.num_observations() >= 12,
            "the fixture has to give the pose something to see: {}",
            vio.ba.lmdb.num_observations()
        );

        (vio, truth)
    }

    /// How far the recovered state may sit from the truth: the fixture's minimum
    /// is exact, so this is convergence and not agreement. Measured from the
    /// perturbation below, which converges in three LM steps
    /// (2.0e6 -> 9.1e-2 -> 5.3e-9 -> 3.6e-17): 2.1e-13 m, 3.6e-15 rad and
    /// 1.9e-11 m/s.
    const CONVERGENCE_TOLERANCE: f64 = 1e-9;

    /// The known answer: a state pushed off a zero-cost minimum comes back to it.
    ///
    /// Both factor groups agree at the truth by construction, so the frame update
    /// has one thing to find and the assertion is against that value rather than
    /// against a recorded run.
    #[test]
    fn the_frame_update_recovers_a_state_pushed_off_a_zero_cost_minimum() {
        let (mut vio, truth) = a_window(5);
        let mut perturbation: Vector15<f64> = Vector15::zeros();
        perturbation
            .fixed_rows_mut::<3>(0)
            .copy_from(&Vector3::new(0.02, -0.015, 0.01));
        perturbation
            .fixed_rows_mut::<3>(3)
            .copy_from(&Vector3::new(0.004, 0.006, -0.003));
        perturbation
            .fixed_rows_mut::<3>(6)
            .copy_from(&Vector3::new(0.05, -0.04, 0.03));
        vio.ba
            .frame_states
            .get_mut(&CURRENT_T_NS)
            .unwrap()
            .apply_inc(&perturbation);

        let (lm, _, _) = vio.frame_update(CURRENT_T_NS).unwrap().unwrap();
        assert!(!lm.is_empty(), "the loop has to take at least one step");
        let last = lm.last().unwrap();
        assert!(
            last.error_after < 1e-12,
            "the minimum is exact, so the cost has to reach it: {}",
            last.error_after
        );
        assert!(
            lm.iter().all(|step| step.accepted),
            "a quadratic with an exact minimum should not need a backtrack"
        );

        let recovered: &PoseVelBiasState<f64> = vio.ba.frame_states[&CURRENT_T_NS].state();
        let position: f64 = (recovered.t_w_i.translation - truth.t_w_i.translation).norm();
        let rotation: f64 = (recovered.t_w_i.rotation * truth.t_w_i.rotation.inverse())
            .log()
            .norm();
        let velocity: f64 = (recovered.vel_w_i - truth.vel_w_i).norm();
        assert!(position < CONVERGENCE_TOLERANCE, "position {position}");
        assert!(rotation < CONVERGENCE_TOLERANCE, "rotation {rotation}");
        assert!(velocity < CONVERGENCE_TOLERANCE, "velocity {velocity}");
    }

    /// The step cap is the knob's, and it counts accepted and backtracked steps
    /// together as `vio_max_iterations` does (D12).
    #[test]
    fn the_knob_caps_the_steps() {
        for cap in 1..=3_i32 {
            let (mut vio, _) = a_window(cap);
            let mut perturbation: Vector15<f64> = Vector15::zeros();
            perturbation
                .fixed_rows_mut::<3>(0)
                .copy_from(&Vector3::new(0.2, -0.15, 0.1));
            vio.ba
                .frame_states
                .get_mut(&CURRENT_T_NS)
                .unwrap()
                .apply_inc(&perturbation);
            let (lm, _, _) = vio.frame_update(CURRENT_T_NS).unwrap().unwrap();
            assert!(
                i32::try_from(lm.len()).unwrap() <= cap + 1,
                "cap {cap} took {} steps",
                lm.len()
            );
        }
    }

    /// A frameset the frame update cannot serve goes back to the joint solve
    /// rather than being solved wrong, and each refusal names the precondition
    /// that made it.
    ///
    /// The names are the load-bearing part: they are what
    /// [`FrameUpdateOutcome`] carries out to `FrameStats` and to the Python
    /// snapshot, so "the update never engaged on this clip" can be answered with
    /// a count per precondition instead of a guess. Every test reads the window
    /// alone, so a replay repeats the choice.
    #[test]
    fn every_refusal_names_its_precondition() {
        // No preintegration joining the two states.
        let (mut vio, _) = a_window(2);
        vio.imu_meas.clear();
        assert_eq!(
            vio.frame_update(CURRENT_T_NS).unwrap().unwrap_err(),
            FrameUpdateDecline::NoImuFactor
        );

        // A preintegration that starts at the predecessor but ends short of this
        // frameset: it is not the factor the joint solve would use, and the
        // interval it covers is not the one being solved over. One IMU sample of
        // an interval is enough to see it.
        let (mut vio, _) = a_window(2);
        let short: IntegratedImuMeasurement<f64> = {
            let noise = crate::imu::ImuNoise::from_calibration(&vio.ba.calib);
            let zero: Vector3<f64> = Vector3::zeros();
            let mut meas = IntegratedImuMeasurement::new(PREV_T_NS, &zero, &zero);
            let step_ns: i64 = (1e9 / vio.ba.calib.imu_update_rate) as i64;
            let mut t_ns: i64 = PREV_T_NS + step_ns;
            while t_ns <= CURRENT_T_NS - step_ns {
                meas.integrate(
                    &ImuSample {
                        t_ns,
                        gyro: Vector3::zeros(),
                        accel: Vector3::new(0.0, 0.0, 9.81),
                    },
                    &noise.accel_cov,
                    &noise.gyro_cov,
                )
                .unwrap();
                t_ns += step_ns;
            }
            meas
        };
        assert_eq!(
            PREV_T_NS + short.get_dt_ns(),
            CURRENT_T_NS - (1e9 / vio.ba.calib.imu_update_rate) as i64,
            "the fixture has to end one IMU sample short of the frameset"
        );
        vio.imu_meas.insert(PREV_T_NS, short);
        assert_eq!(
            vio.frame_update(CURRENT_T_NS).unwrap().unwrap_err(),
            FrameUpdateDecline::ImuIntervalGap
        );

        // A frameset that is not the newest state.
        let (mut vio, _) = a_window(2);
        assert_eq!(
            vio.frame_update(PREV_T_NS).unwrap().unwrap_err(),
            FrameUpdateDecline::NotNewest
        );

        // No previous state to hold.
        let (mut vio, _) = a_window(2);
        vio.ba.frame_states.remove(&PREV_T_NS);
        assert_eq!(
            vio.frame_update(CURRENT_T_NS).unwrap().unwrap_err(),
            FrameUpdateDecline::NoPredecessor
        );

        // A prior that orders the newest state: its gradient would not be zero.
        let (mut vio, _) = a_window(2);
        vio.marg_data
            .order
            .push(CURRENT_T_NS, POSE_VEL_BIAS_SIZE)
            .unwrap();
        assert_eq!(
            vio.frame_update(CURRENT_T_NS).unwrap().unwrap_err(),
            FrameUpdateDecline::PriorOrdersState
        );
    }

    /// The four-camera rig: the held landmarks' Jacobians w.r.t. the newest state
    /// are right for a non-host camera too.
    ///
    /// Every landmark here is hosted by camera 0 of the host keyframe and is
    /// observed by whichever of the four cameras can see it, so the relative pose
    /// each residual is formed through carries a different `T_i_c` on the target
    /// side. Get one of those extrinsics or its `d_rel_d_t` wrong and the
    /// perturbed state cannot return to a cost of zero, because the wrong
    /// Jacobian points somewhere else. This is the check behind the MGO09 reading
    /// in D76: the four-camera drift is the schedule, not a defect in the update's
    /// per-camera algebra.
    #[test]
    fn the_frame_update_recovers_on_a_four_camera_rig() {
        let (mut vio, truth) = a_window_on(CALIB_4, CONFIG_4, 5);
        assert_eq!(vio.ba.cameras().len(), 4);
        let observing: usize = (0..4)
            .filter(|cam_id| {
                vio.ba
                    .lmdb
                    .landmarks()
                    .iter()
                    .any(|lm| lm.obs.contains_key(&TimeCamId::new(CURRENT_T_NS, *cam_id)))
            })
            .count();
        assert!(
            observing >= 2,
            "the fixture has to exercise a non-host camera: {observing} of 4 observe"
        );

        let mut perturbation: Vector15<f64> = Vector15::zeros();
        perturbation
            .fixed_rows_mut::<3>(0)
            .copy_from(&Vector3::new(0.02, -0.015, 0.01));
        perturbation
            .fixed_rows_mut::<3>(3)
            .copy_from(&Vector3::new(0.004, 0.006, -0.003));
        vio.ba
            .frame_states
            .get_mut(&CURRENT_T_NS)
            .unwrap()
            .apply_inc(&perturbation);

        let (lm, _, _) = vio.frame_update(CURRENT_T_NS).unwrap().unwrap();
        let last = lm.last().unwrap();
        assert!(
            last.error_after < 1e-12,
            "the minimum is exact on four cameras too: {}",
            last.error_after
        );
        let recovered: &PoseVelBiasState<f64> = vio.ba.frame_states[&CURRENT_T_NS].state();
        let position: f64 = (recovered.t_w_i.translation - truth.t_w_i.translation).norm();
        let rotation: f64 = (recovered.t_w_i.rotation * truth.t_w_i.rotation.inverse())
            .log()
            .norm();
        assert!(position < CONVERGENCE_TOLERANCE, "position {position}");
        assert!(rotation < CONVERGENCE_TOLERANCE, "rotation {rotation}");
    }

    /// Offline mode lets nothing but the data reach a decision: the same window
    /// solved twice gives the same state, coefficient for coefficient.
    #[test]
    fn a_repeat_frame_update_is_bit_identical() {
        /// The state the solve left behind and the trail it took to get there.
        type Solved = (PoseVelBiasState<f64>, Vec<(i32, f64, f64, bool)>);
        let solve = || -> Solved {
            let (mut vio, _) = a_window(3);
            let mut perturbation: Vector15<f64> = Vector15::zeros();
            perturbation
                .fixed_rows_mut::<3>(0)
                .copy_from(&Vector3::new(0.02, -0.015, 0.01));
            vio.ba
                .frame_states
                .get_mut(&CURRENT_T_NS)
                .unwrap()
                .apply_inc(&perturbation);
            let (lm, _, _) = vio.frame_update(CURRENT_T_NS).unwrap().unwrap();
            (
                *vio.ba.frame_states[&CURRENT_T_NS].state(),
                lm.iter()
                    .map(|step| (step.iteration, step.error_after, step.l_diff, step.accepted))
                    .collect(),
            )
        };
        assert_eq!(solve(), solve());
    }
}
