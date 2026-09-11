//! IMU preintegration, covariance, bias Jacobians and whitened residuals.
//!
//! The estimator uses midpoint rotation `ΔR exp(½ Δt ω)`, retains `½ a Δt²`
//! in position, and orders the state `[translation, rotation, velocity]` (D12).
//! The full state appends gyro and accelerometer biases. Gyro-bias correction
//! is pre-multiplied, requiring right inverse Jacobians for state blocks and a
//! left inverse Jacobian for the gyro-bias block. Position residuals include
//! `−v_i Δt`.
//!
//! Covariance whitening uses guarded pivoted LDLT. Tiny or negative pivots receive
//! zero weight; singular inputs yield a congruence generalized inverse, not
//! necessarily a Moore-Penrose inverse. The 9x9 factor is recomputed per request
//! to avoid interior mutability. Pivot selection precedes column updates.
//!
//! Callers request all Jacobians or none. Prediction returns a fresh state with
//! an updated timestamp, and gravity initialization uses a cross-product axis.
//! Non-monotonic samples return typed errors (D32). Accelerometer bias cannot
//! change delta rotation because its input Jacobian has zero rotation rows and
//! the transition's rotation rows are `[0 I 0]`; tests enforce this identity.

use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3};

use crate::calib::Calibration;
use crate::lie::{
    LieScalar, So3, c, left_jacobian_inv_so3, right_jacobian_inv_so3, right_jacobian_so3,
};
use crate::types::{
    POSE_VEL_BIAS_SIZE, POSE_VEL_SIZE, PoseVelBiasStateWithLin, PoseVelState, Vector9, Vector15,
};

/// Where the gyroscope bias starts inside a 15-vector state block: the `+9` of
const BIAS_GYRO_OFFSET: usize = POSE_VEL_SIZE;
/// Accelerometer bias starts at offset 12 in the full state.
const BIAS_ACCEL_OFFSET: usize = POSE_VEL_SIZE + 3;

/// A `Vector3<f64>` in the measurement's scalar type.
#[inline]
fn cast3<S: LieScalar>(v: &Vector3<f64>) -> Vector3<S> {
    Vector3::new(c::<S>(v.x), c::<S>(v.y), c::<S>(v.z))
}

/// The 9x9 blocks: covariance, state transition, residual Jacobians.
pub type Matrix9<S> = SMatrix<S, POSE_VEL_SIZE, POSE_VEL_SIZE>;
/// The 9x3 blocks: noise input and bias Jacobians.
pub type Matrix9x3<S> = SMatrix<S, POSE_VEL_SIZE, 3>;
/// The 9x6 residual Jacobian with respect to both biases at once.
pub type Matrix9x6<S> = SMatrix<S, POSE_VEL_SIZE, 6>;
/// The IMU block's Jacobian: 15 rows over the two 15-column states.
pub type Matrix15x30<S> = SMatrix<S, POSE_VEL_BIAS_SIZE, { 2 * POSE_VEL_BIAS_SIZE }>;

/// Gravity in the world frame.
pub fn gravity<S: LieScalar>() -> Vector3<S> {
    Vector3::new(S::zero(), S::zero(), c::<S>(-9.81))
}

/// Timestamped gyroscope and accelerometer reading stored in f64.
/// Samples have static calibration applied. Linearization-point bias correction
/// occurs during integration.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ImuSample {
    /// Timestamp of the sample, in nanoseconds.
    pub t_ns: i64,
    /// Angular velocity in the rig frame, rad/s.
    pub gyro: Vector3<f64>,
    /// Specific force in the rig frame, m/s².
    pub accel: Vector3<f64>,
}

/// Diagonal noise covariances: square continuous-time densities after multiplying
/// by `sqrt(imu_update_rate)` to obtain discrete-time densities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuNoise<S: LieScalar> {
    /// Diagonal of the accelerometer noise covariance.
    pub accel_cov: Vector3<S>,
    /// Diagonal of the gyroscope noise covariance.
    pub gyro_cov: Vector3<S>,
}

impl<S: LieScalar> ImuNoise<S> {
    /// The noise the estimator builds from a calibration
    pub fn from_calibration(calib: &Calibration<S>) -> Self {
        Self {
            accel_cov: calib
                .discrete_time_accel_noise_std()
                .map(|value: S| value * value),
            gyro_cov: calib
                .discrete_time_gyro_noise_std()
                .map(|value: S| value * value),
        }
    }
}

/// Typed preintegration failures; invalid data must not panic (D32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ImuError {
    /// A sample fails to strictly follow the integrated interval. Duplicate timestamps
    /// would give zero duration and a singular measurement.
    #[error("imu sample at {t_ns} ns does not follow the integrated state at {previous_t_ns} ns")]
    NonMonotonicSample {
        /// End of what has been integrated so far, relative to the start time.
        previous_t_ns: i64,
        /// Timestamp of the rejected sample, on the same clock.
        t_ns: i64,
    },
    /// Two frame timestamps that are equal or go backwards.
    ///
    ///  asserts both of these separately, and its
    /// message says a zero time delta "leads to invalid IMU integration".
    #[error("frame interval [{t0_ns}, {t1_ns}] ns is empty or reversed")]
    NonMonotonicFrames {
        /// Timestamp of the previous frame.
        t0_ns: i64,
        /// Timestamp of the current frame.
        t1_ns: i64,
    },
    /// The requested interval starts at a different time from the measurement.
    #[error("interval starts at {t0_ns} ns but the measurement starts at {start_t_ns} ns")]
    StartTimeMismatch {
        /// Start time the measurement was constructed with.
        start_t_ns: i64,
        /// Start of the interval the caller asked for.
        t0_ns: i64,
    },
    /// No sample follows the frame to close the integration interval.
    #[error("no imu sample after {t1_ns} ns to close the interval with")]
    MissingSampleAfterFrame {
        /// Timestamp the interval had to reach.
        t1_ns: i64,
    },
    /// A timestamp difference does not fit in an `i64`.
    #[error("timestamps {a_ns} and {b_ns} ns are too far apart to subtract")]
    TimestampOverflow {
        /// Left operand.
        a_ns: i64,
        /// Right operand.
        b_ns: i64,
    },
}

/// The three Jacobians of one propagation step
/// (`F`, `A` and `G` at ).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropagationJacobians<S: LieScalar> {
    /// `F = ∂next/∂curr`, Paper 1 Eq. (13)'s `J_f^s`.
    pub d_next_d_curr: Matrix9<S>,
    /// `A = ∂next/∂accel`, Paper 1 Eq. (13)'s `J_f^a`.
    pub d_next_d_accel: Matrix9x3<S>,
    /// `G = ∂next/∂gyro`, Paper 1 Eq. (13)'s `J_f^g`.
    pub d_next_d_gyro: Matrix9x3<S>,
}

/// The residual's Jacobians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuResidualJacobians<S: LieScalar> {
    /// `∂r/∂state0`.
    pub d_res_d_state0: Matrix9<S>,
    /// `∂r/∂state1`.
    pub d_res_d_state1: Matrix9<S>,
    /// Bias Jacobian: gyro bias in columns 0–2, accelerometer bias in columns 3–5.
    /// These occupy offsets 9 and 12 of the full `[t, R, v, b_g, b_a]` state.
    pub d_res_d_bias: Matrix9x6<S>,
}

/// A pseudo-measurement from consecutive IMU samples.
/// Delta time is elapsed nanoseconds from [`Self::get_start_t_ns`], not absolute time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegratedImuMeasurement<S: LieScalar> {
    start_t_ns: i64,
    delta_state: PoseVelState<S>,
    cov: Matrix9<S>,
    d_state_d_ba: Matrix9x3<S>,
    d_state_d_bg: Matrix9x3<S>,
    bias_gyro_lin: Vector3<S>,
    bias_accel_lin: Vector3<S>,
}

impl<S: LieScalar> Default for IntegratedImuMeasurement<S> {
    /// everything zero, start time zero.
    fn default() -> Self {
        Self::new(0, &Vector3::zeros(), &Vector3::zeros())
    }
}

/// One calibrated sample as the two producer loops pop them: `(t_ns, gyro,
/// accel)`, the shape `accumulate_to` carries its pending sample in.
pub type Popped<S> = (i64, Vector3<S>, Vector3<S>);

impl<S: LieScalar> IntegratedImuMeasurement<S> {
    /// An empty measurement starting at `start_t_ns`, linearized about the two
    /// biases.
    pub fn new(start_t_ns: i64, bias_gyro: &Vector3<S>, bias_accel: &Vector3<S>) -> Self {
        Self {
            start_t_ns,
            delta_state: PoseVelState::default(),
            cov: Matrix9::zeros(),
            d_state_d_ba: Matrix9x3::zeros(),
            d_state_d_bg: Matrix9x3::zeros(),
            bias_gyro_lin: *bias_gyro,
            bias_accel_lin: *bias_accel,
        }
    }

    /// Propagate one state with a bias-corrected sample and compute all three Jacobians.
    /// The timestamp is relative to the measurement start; biases are already removed.
    pub fn propagate_state(
        curr_state: &PoseVelState<S>,
        t_ns: i64,
        accel: &Vector3<S>,
        gyro: &Vector3<S>,
    ) -> Result<(PoseVelState<S>, PropagationJacobians<S>), ImuError> {
        // Refuse duplicate or reordered samples before they create a singular measurement.
        if t_ns <= curr_state.t_ns {
            return Err(ImuError::NonMonotonicSample {
                previous_t_ns: curr_state.t_ns,
                t_ns,
            });
        }
        let dt_ns: i64 = t_ns
            .checked_sub(curr_state.t_ns)
            .ok_or(ImuError::TimestampOverflow {
                a_ns: t_ns,
                b_ns: curr_state.t_ns,
            })?;
        let dt: S = c::<S>(dt_ns as f64) * c::<S>(1e-9); // `:82-83`

        // Deviation 1: the acceleration is rotated by the *midpoint* rotation.
        let r_w_i_new_2: So3<S> =
            curr_state.t_w_i.rotation * So3::exp(&(*gyro * (c::<S>(0.5) * dt))); // `:85`
        let rr_w_i_new_2: Matrix3<S> = r_w_i_new_2.matrix(); // `:86`
        let accel_world: Vector3<S> = rr_w_i_new_2 * accel; // `:88`

        let mut next_state: PoseVelState<S> = PoseVelState {
            t_ns, // `:90`
            t_w_i: curr_state.t_w_i,
            vel_w_i: curr_state.vel_w_i + accel_world * dt, // `:92`
        };
        next_state.t_w_i.rotation = curr_state.t_w_i.rotation * So3::exp(&(*gyro * dt)); // `:91`
        // deviation 2: the quadratic term stays.
        next_state.t_w_i.translation = curr_state.t_w_i.translation
            + curr_state.vel_w_i * dt
            + accel_world * c::<S>(0.5) * dt * dt;

        // Only the *diagonal* of the (0,6) block is set, so the port
        // writes three entries rather than a scaled identity.
        let mut d_next_d_curr: Matrix9<S> = Matrix9::identity();
        for i in 0..3 {
            d_next_d_curr[(i, 6 + i)] = dt;
        }
        let hat_accel_dt: Matrix3<S> = So3::hat(&(-accel_world * dt));
        d_next_d_curr
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&hat_accel_dt);
        d_next_d_curr
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(hat_accel_dt * dt * c::<S>(0.5)));

        let mut d_next_d_accel: Matrix9x3<S> = Matrix9x3::zeros();
        d_next_d_accel
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(rr_w_i_new_2 * c::<S>(0.5) * dt * dt));
        d_next_d_accel
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(rr_w_i_new_2 * dt));

        let mut d_next_d_gyro: Matrix9x3<S> = Matrix9x3::zeros();
        let jr: Matrix3<S> = right_jacobian_so3(&(*gyro * dt));
        let jr2: Matrix3<S> = right_jacobian_so3(&(*gyro * (c::<S>(0.5) * dt)));
        d_next_d_gyro
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(next_state.t_w_i.rotation.matrix() * jr * dt));
        let d_vel_d_gyro: Matrix3<S> = hat_accel_dt * rr_w_i_new_2 * jr2 * c::<S>(0.5) * dt;
        d_next_d_gyro
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&d_vel_d_gyro);
        d_next_d_gyro
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(d_vel_d_gyro * (c::<S>(0.5) * dt)));

        Ok((
            next_state,
            PropagationJacobians {
                d_next_d_curr,
                d_next_d_accel,
                d_next_d_gyro,
            },
        ))
    }

    /// Fold one sample into the measurement.
    ///
    /// `accel_cov` and `gyro_cov` are the diagonals of the discrete-time noise
    /// covariances. Nothing here allocates: every matrix is a fixed-size
    /// `SMatrix` on the stack.
    pub fn integrate(
        &mut self,
        data: &ImuSample,
        accel_cov: &Vector3<S>,
        gyro_cov: &Vector3<S>,
    ) -> Result<(), ImuError> {
        self.integrate_calibrated(
            data.t_ns,
            &cast3::<S>(&data.accel),
            &cast3::<S>(&data.gyro),
            accel_cov,
            gyro_cov,
        )
    }

    /// The same fold, with the sample already in `Scalar`.
    ///
    /// The backend needs this: `popFromImuDataQueue` casts to `Scalar`
    /// **before** `calib_accel_bias.getCalibrated` runs
    /// so the static bias
    /// calibration of an `f32` estimator happens in `f32`. Calibrating in `f64`
    /// and casting afterwards is a different number.
    ///
    /// # Errors
    ///
    /// [`ImuError`] as [`Self::integrate`].
    pub fn integrate_calibrated(
        &mut self,
        sample_t_ns: i64,
        accel: &Vector3<S>,
        gyro: &Vector3<S>,
        accel_cov: &Vector3<S>,
        gyro_cov: &Vector3<S>,
    ) -> Result<(), ImuError> {
        // relative time, bias removed at the linearization point.
        let t_ns: i64 =
            sample_t_ns
                .checked_sub(self.start_t_ns)
                .ok_or(ImuError::TimestampOverflow {
                    a_ns: sample_t_ns,
                    b_ns: self.start_t_ns,
                })?;
        let accel: Vector3<S> = accel - self.bias_accel_lin;
        let gyro: Vector3<S> = gyro - self.bias_gyro_lin;

        let (new_state, j): (PoseVelState<S>, PropagationJacobians<S>) =
            Self::propagate_state(&self.delta_state, t_ns, &accel, &gyro)?; // `:158`

        self.delta_state = new_state; // `:160`
        // Paper 1 Eq. (21).
        self.cov = j.d_next_d_curr * self.cov * j.d_next_d_curr.transpose()
            + j.d_next_d_accel * Matrix3::from_diagonal(accel_cov) * j.d_next_d_accel.transpose()
            + j.d_next_d_gyro * Matrix3::from_diagonal(gyro_cov) * j.d_next_d_gyro.transpose();
        // Paper 1 Eqs. (15)-(16).
        self.d_state_d_ba = -j.d_next_d_accel + j.d_next_d_curr * self.d_state_d_ba;
        self.d_state_d_bg = -j.d_next_d_gyro + j.d_next_d_curr * self.d_state_d_bg;
        Ok(())
    }

    /// Accumulate samples over one frame interval.
    /// Skip samples at or before `skip_past_ns`, then integrate through `until_ns`.
    /// If the interval remains short, retime the first later sample to the boundary
    /// and integrate its own values, without interpolation. Return that pending
    /// sample at its original timestamp for the next frame.
    ///
    /// Both frontend and estimator preintegrators use this loop (D24), with their
    /// own sample source and noise model. Passing pending state by value leaves
    /// the caller free to mutate its queue through `pop`.
    ///
    /// # Errors
    /// Refuse reversed intervals, a mismatched start time, missing closing samples
    /// or any input rejected by [`Self::integrate_calibrated`].
    pub fn accumulate_to(
        &mut self,
        pending: Option<Popped<S>>,
        mut pop: impl FnMut() -> Option<Popped<S>>,
        skip_past_ns: i64,
        until_ns: i64,
        noise: &ImuNoise<S>,
    ) -> Result<Option<Popped<S>>, ImuError> {
        // the frame gap is what the measurement integrates over, so
        // an empty or reversed one has no measurement.
        if until_ns <= skip_past_ns {
            return Err(ImuError::NonMonotonicFrames {
                t0_ns: skip_past_ns,
                t1_ns: until_ns,
            });
        }
        //  builds the measurement with `prev_frame->t_ns` and
        // skips past that same variable. Any other origin times every sample
        // against a start the measurement does not have.
        if skip_past_ns != self.start_t_ns {
            return Err(ImuError::StartTimeMismatch {
                start_t_ns: self.start_t_ns,
                t0_ns: skip_past_ns,
            });
        }

        let mut pending: Option<Popped<S>> = pending.or_else(&mut pop);

        // discard everything at or before the previous frameset.
        while let Some((t_ns, _, _)) = pending {
            if t_ns > skip_past_ns {
                break;
            }
            pending = pop();
        }
        // integrate everything up to and including the frameset.
        while let Some((t_ns, gyro, accel)) = pending {
            if t_ns > until_ns {
                break;
            }
            self.integrate_calibrated(t_ns, &accel, &gyro, &noise.accel_cov, &noise.gyro_cov)?;
            pending = pop();
        }
        // A measurement stopping before the frame is invalid; require a closing sample.
        if self.start_t_ns + self.get_dt_ns() < until_ns {
            let Some((_, gyro, accel)) = pending else {
                return Err(ImuError::MissingSampleAfterFrame { t1_ns: until_ns });
            };
            self.integrate_calibrated(until_ns, &accel, &gyro, &noise.accel_cov, &noise.gyro_cov)?;
        }
        Ok(pending)
    }

    /// Predict the state at the end of the interval.
    ///
    /// **Unchecked precondition:** `state0` is the state at
    /// [`Self::get_start_t_ns`]. The delta this applies is relative to that
    /// start, so a `state0` from any other time advances the pose and the
    /// velocity over an interval it did not begin, and the timestamp it reports
    /// is `state0.t_ns` plus the delta **saturated** — at `i64::MAX` the clock
    /// stands still while the motion still applies. Both callers hold the
    /// precondition by construction and neither reads the timestamp: the
    /// estimator files the predicted state under the frameset's own `t_ns`
    ///  and the frontend takes only the pose
    /// Checking it would make this
    /// fallible on the LM path for a value nothing there reads, which is the
    /// review follow-up rather than this fix round.
    pub fn predict_state(&self, state0: &PoseVelState<S>, g: &Vector3<S>) -> PoseVelState<S> {
        let dt: S = c::<S>(self.delta_state.t_ns as f64) * c::<S>(1e-9); // `:175`
        let mut state1: PoseVelState<S> = PoseVelState {
            // Advance the predicted timestamp by the integrated duration.
            t_ns: state0.t_ns.saturating_add(self.delta_state.t_ns),
            t_w_i: state0.t_w_i,
            vel_w_i: state0.vel_w_i + *g * dt + state0.t_w_i.rotation * self.delta_state.vel_w_i,
        };
        state1.t_w_i.rotation = state0.t_w_i.rotation * self.delta_state.t_w_i.rotation; // `:177`
        state1.t_w_i.translation = state0.t_w_i.translation
            + state0.vel_w_i * dt
            + *g * c::<S>(0.5) * dt * dt
            + state0.t_w_i.rotation * self.delta_state.t_w_i.translation;
        state1
    }

    /// The 9-vector residual between two states.
    ///
    /// Ordered `[translation(3), rotation(3), velocity(3)]` (deviation 3).
    pub fn residual(
        &self,
        state0: &PoseVelState<S>,
        g: &Vector3<S>,
        state1: &PoseVelState<S>,
        curr_bg: &Vector3<S>,
        curr_ba: &Vector3<S>,
    ) -> Vector9<S> {
        self.residual_parts(state0, g, state1, curr_bg, curr_ba).0
    }

    /// [`Self::residual`] plus the three Jacobians.
    pub fn residual_with_jacobians(
        &self,
        state0: &PoseVelState<S>,
        g: &Vector3<S>,
        state1: &PoseVelState<S>,
        curr_bg: &Vector3<S>,
        curr_ba: &Vector3<S>,
    ) -> (Vector9<S>, ImuResidualJacobians<S>) {
        let (res, r0_inv, tmp, tmp2, dt): (Vector9<S>, Matrix3<S>, Vector3<S>, Vector3<S>, S) =
            self.residual_parts(state0, g, state1, curr_bg, curr_ba);

        let res_rot: Vector3<S> = res.fixed_rows::<3>(3).into_owned();
        let j: Matrix3<S> = right_jacobian_inv_so3(&res_rot); // `:228`

        let mut d_res_d_state0: Matrix9<S> = Matrix9::zeros();
        d_res_d_state0
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&-r0_inv);
        d_res_d_state0
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(So3::hat(&tmp) * r0_inv));
        d_res_d_state0
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(j * r0_inv));
        d_res_d_state0
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(So3::hat(&tmp2) * r0_inv));
        d_res_d_state0
            .fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(-r0_inv * dt));
        d_res_d_state0
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&-r0_inv);

        let mut d_res_d_state1: Matrix9<S> = Matrix9::zeros();
        d_res_d_state1
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&r0_inv);
        d_res_d_state1
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(-j * r0_inv));
        d_res_d_state1
            .fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&r0_inv);

        // Gyro-bias columns start as `-d_state_d_bg`, then replace the rotation rows
        // with the rotation correction Jacobian, without that minus sign.
        let mut d_res_d_bias: Matrix9x6<S> = Matrix9x6::zeros();
        d_res_d_bias
            .fixed_view_mut::<9, 3>(0, 0)
            .copy_from(&-self.d_state_d_bg);
        let j_left: Matrix3<S> = left_jacobian_inv_so3(&res_rot); // `:256`
        d_res_d_bias
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(j_left * self.d_state_d_bg.fixed_view::<3, 3>(3, 0))); // `:257`
        d_res_d_bias
            .fixed_view_mut::<9, 3>(0, 3)
            .copy_from(&-self.d_state_d_ba); // `:250`

        (
            res,
            ImuResidualJacobians {
                d_res_d_state0,
                d_res_d_state1,
                d_res_d_bias,
            },
        )
    }

    /// The residual and the four intermediates its Jacobians reuse:
    /// `R0_inv`, `tmp`, `tmp2` and
    /// `dt`.
    fn residual_parts(
        &self,
        state0: &PoseVelState<S>,
        g: &Vector3<S>,
        state1: &PoseVelState<S>,
        curr_bg: &Vector3<S>,
        curr_ba: &Vector3<S>,
    ) -> (Vector9<S>, Matrix3<S>, Vector3<S>, Vector3<S>, S) {
        let dt: S = c::<S>(self.delta_state.t_ns as f64) * c::<S>(1e-9); // `:203`

        // Paper 1 Eq. (17).
        let bg_diff: Vector9<S> = self.d_state_d_bg * (curr_bg - self.bias_gyro_lin);
        let ba_diff: Vector9<S> = self.d_state_d_ba * (curr_ba - self.bias_accel_lin);

        let r0_inv: Matrix3<S> = state0.t_w_i.rotation.inverse().matrix(); // `:213`
        // The `− vel * dt` term is the one Paper 1's Eq. (20) omits.
        let tmp: Vector3<S> = r0_inv
            * (state1.t_w_i.translation
                - state0.t_w_i.translation
                - state0.vel_w_i * dt
                - *g * c::<S>(0.5) * dt * dt);
        let tmp2: Vector3<S> = r0_inv * (state1.vel_w_i - state0.vel_w_i - *g * dt);

        let mut res: Vector9<S> = Vector9::zeros();
        res.fixed_rows_mut::<3>(0).copy_from(
            &(tmp
                - (self.delta_state.t_w_i.translation
                    + bg_diff.fixed_rows::<3>(0)
                    + ba_diff.fixed_rows::<3>(0))),
        );
        // deviation 4: the gyro-bias correction is pre-multiplied.
        res.fixed_rows_mut::<3>(3).copy_from(
            &(So3::exp(&bg_diff.fixed_rows::<3>(3).into_owned())
                * self.delta_state.t_w_i.rotation
                * state1.t_w_i.rotation.inverse()
                * state0.t_w_i.rotation)
                .log(),
        );
        res.fixed_rows_mut::<3>(6).copy_from(
            &(tmp2
                - (self.delta_state.vel_w_i
                    + bg_diff.fixed_rows::<3>(6)
                    + ba_diff.fixed_rows::<3>(6))),
        );

        (res, r0_inv, tmp, tmp2, dt)
    }

    /// Elapsed time of the measurement, in nanoseconds.
    pub fn get_dt_ns(&self) -> i64 {
        self.delta_state.t_ns
    }

    /// Start time of the measurement, in nanoseconds.
    pub fn get_start_t_ns(&self) -> i64 {
        self.start_t_ns
    }

    /// The preintegrated delta state.
    pub fn get_delta_state(&self) -> &PoseVelState<S> {
        &self.delta_state
    }

    /// The measurement covariance.
    pub fn get_cov(&self) -> &Matrix9<S> {
        &self.cov
    }

    /// Jacobian of the delta state with respect to the accelerometer bias
    pub fn get_d_state_d_ba(&self) -> &Matrix9x3<S> {
        &self.d_state_d_ba
    }

    /// Jacobian of the delta state with respect to the gyroscope bias
    pub fn get_d_state_d_bg(&self) -> &Matrix9x3<S> {
        &self.d_state_d_bg
    }

    /// Whitening factor `M = D⁺½ L⁻¹ P` for `P cov Pᵀ = L D Lᵀ`.
    /// For SPD covariance, `Mᵀ M = cov⁻¹`. Rows with a pivot below the smallest
    /// positive normal value are zeroed so degenerate covariance contributes
    /// finite, zero weight in those factor coordinates. The singular result is
    /// a generalized inverse through congruence, not a Moore-Penrose inverse.
    pub fn get_cov_inv_sqrt(&self) -> Matrix9<S> {
        let mut mat: Matrix9<S> = self.cov;
        let transpositions: [usize; POSE_VEL_SIZE] = ldlt_in_place(&mut mat);

        // Apply the pivot permutation to the identity before solving with L.
        let mut m: Matrix9<S> = Matrix9::identity();
        for (k, pivot) in transpositions.iter().copied().enumerate() {
            if pivot != k {
                m.swap_rows(k, pivot);
            }
        }
        // `matrixL()` is a *unit* lower triangular view, so the stored
        // diagonal (which holds D) is not read here.
        for i in 0..POSE_VEL_SIZE {
            for k in 0..i {
                let factor: S = mat[(i, k)];
                for column in 0..POSE_VEL_SIZE {
                    let above: S = m[(k, column)];
                    m[(i, column)] -= factor * above;
                }
            }
        }
        // The comparison is against
        // `std::numeric_limits<Scalar>::min()`, so a *negative* pivot — what a
        // rank-deficient covariance actually produces, see [`ldlt_in_place`] —
        // zeroes its row rather than taking the square root of a negative.
        for i in 0..POSE_VEL_SIZE {
            let scale: S = if mat[(i, i)] < S::min_positive() {
                S::zero()
            } else {
                S::one() / mat[(i, i)].sqrt()
            };
            for column in 0..POSE_VEL_SIZE {
                m[(i, column)] *= scale;
            }
        }
        m
    }

    /// The inverse covariance, `Mᵀ M`.
    pub fn get_cov_inv(&self) -> Matrix9<S> {
        let m: Matrix9<S> = self.get_cov_inv_sqrt();
        m.transpose() * m
    }
}

/// Fires the near-antiparallel warning in [`gravity_from_first_accel`] once per
/// process, not once per frame.
static ANTIPARALLEL_WARNING: std::sync::Once = std::sync::Once::new();

/// The initial orientation from one accelerometer sample
///
/// Align measured specific force with world +Z. Only roll and pitch are
/// observable. Near antiparallel inputs use a cross-product axis orthogonal
/// to both directions; for exactly antiparallel inputs an arbitrary orthogonal
/// axis resolves the unobservable yaw. Zero and non-finite samples return identity.
pub fn gravity_from_first_accel<S: LieScalar>(accel: &Vector3<S>) -> So3<S> {
    let norm: S = accel.norm();
    if !norm.is_finite() || norm <= S::zero() {
        return So3::identity();
    }
    let v0: Vector3<S> = accel / norm; // `:693`
    let v1: Vector3<S> = Vector3::new(S::zero(), S::zero(), S::one());
    let dot: S = v1.dot(&v0); // `:695`

    if dot < c::<S>(-1.0) + S::eigen_dummy_precision() {
        // Report this ambiguous initial orientation once per process.
        ANTIPARALLEL_WARNING.call_once(|| {
            log::warn!(
                "gravity initialisation took the near-antiparallel branch: the rig started \
                 within milliradians of upside down, and the initial yaw and up to \
                 2*sqrt(2*eps) rad of roll and pitch differ from basalt's Eigen JacobiSVD"
            );
        });
        let clamped: S = dot.max(c::<S>(-1.0)); // `:708`
        let axis: Vector3<S> = axis_orthogonal_to_both(&v0, &v1);
        let w2: S = (S::one() + clamped) * c::<S>(0.5); // `:714`
        let vector_scale: S = (S::one() - w2).sqrt(); // `:716`
        let quaternion = nalgebra::Quaternion::new(
            w2.sqrt(),
            axis.x * vector_scale,
            axis.y * vector_scale,
            axis.z * vector_scale,
        );
        // Normalize to remove rounding in the squared quaternion norm.
        return So3::from_unit_quaternion(nalgebra::UnitQuaternion::new_normalize(quaternion));
    }

    let axis: Vector3<S> = v0.cross(&v1);
    let s: S = ((S::one() + dot) * c::<S>(2.0)).sqrt();
    let inv_s: S = S::one() / s;
    let quaternion = nalgebra::Quaternion::new(
        s * c::<S>(0.5),
        axis.x * inv_s,
        axis.y * inv_s,
        axis.z * inv_s,
    );
    So3::from_unit_quaternion(nalgebra::UnitQuaternion::new_normalize(quaternion))
}

/// A unit vector orthogonal to both inputs, using their cross product.
/// For parallel inputs, project the least-aligned canonical axis off v1.
fn axis_orthogonal_to_both<S: LieScalar>(v0: &Vector3<S>, v1: &Vector3<S>) -> Vector3<S> {
    let cross: Vector3<S> = v0.cross(v1);
    let norm: S = cross.norm();
    if norm > S::zero() {
        return cross / norm;
    }
    let basis: Vector3<S> = if v1.x.abs() <= v1.y.abs() && v1.x.abs() <= v1.z.abs() {
        Vector3::new(S::one(), S::zero(), S::zero())
    } else if v1.y.abs() <= v1.z.abs() {
        Vector3::new(S::zero(), S::one(), S::zero())
    } else {
        Vector3::new(S::zero(), S::zero(), S::one())
    };
    let projected: Vector3<S> = basis - v1 * basis.dot(v1);
    let projected_norm: S = projected.norm();
    if projected_norm > S::zero() {
        projected / projected_norm
    } else {
        basis
    }
}

/// Factor a symmetric 9x9 in place as `P A Pᵀ = L D Lᵀ`.
/// Store D on the diagonal and unit-lower L below it. The upper triangle is
/// unused. Apply the returned row swaps in ascending order to construct P.
/// Select pivots from the original diagonal before updating the column.
/// Whitening zeros negative and subnormal pivots instead of taking their roots.
fn ldlt_in_place<S: LieScalar>(mat: &mut Matrix9<S>) -> [usize; POSE_VEL_SIZE] {
    let size: usize = POSE_VEL_SIZE;
    let mut transpositions: [usize; POSE_VEL_SIZE] = [0; POSE_VEL_SIZE];

    for k in 0..size {
        // `maxCoeff` keeps the *first* maximum, so the comparison
        // has to be strict.
        let mut pivot: usize = k;
        for i in (k + 1)..size {
            if mat[(i, i)].abs() > mat[(pivot, pivot)].abs() {
                pivot = i;
            }
        }
        transpositions[k] = pivot;

        if pivot != k {
            // a symmetric swap written to keep only the lower
            // triangle valid, which is all the rest of the algorithm reads.
            for column in 0..k {
                let swapped: S = mat[(k, column)];
                mat[(k, column)] = mat[(pivot, column)];
                mat[(pivot, column)] = swapped;
            }
            for row in (pivot + 1)..size {
                let swapped: S = mat[(row, k)];
                mat[(row, k)] = mat[(row, pivot)];
                mat[(row, pivot)] = swapped;
            }
            let swapped: S = mat[(k, k)];
            mat[(k, k)] = mat[(pivot, pivot)];
            mat[(pivot, pivot)] = swapped;
            for i in (k + 1)..pivot {
                let swapped: S = mat[(i, k)];
                mat[(i, k)] = mat[(pivot, i)];
                mat[(pivot, i)] = swapped;
            }
        }

        // the delayed update. Column `k` is brought up to date from
        // the columns already factorized; the trailing diagonal is not touched.
        let rs: usize = size - k - 1;
        if k > 0 {
            let mut temp: [S; POSE_VEL_SIZE] = [S::zero(); POSE_VEL_SIZE];
            for (j, entry) in temp.iter_mut().enumerate().take(k) {
                *entry = mat[(j, j)] * mat[(k, j)]; // `:336`
            }
            let mut diagonal: S = S::zero();
            for (j, entry) in temp.iter().enumerate().take(k) {
                diagonal += mat[(k, j)] * *entry;
            }
            mat[(k, k)] -= diagonal; // `:337`
            if rs > 0 {
                for i in (k + 1)..size {
                    let mut sum: S = S::zero();
                    for (j, entry) in temp.iter().enumerate().take(k) {
                        sum += mat[(i, j)] * *entry;
                    }
                    mat[(i, k)] -= sum; // `:338`
                }
            }
        }

        // Guard exact zero to avoid division by zero in the column update.
        let real_akk: S = mat[(k, k)];
        let pivot_is_valid: bool = real_akk.abs() > S::zero();

        if k == 0 && !pivot_is_valid {
            // the whole diagonal is zero, so there is nothing left
            // to do but fill in the identity transpositions. The empty
            // measurement takes this branch and whitens to zero.
            for (j, entry) in transpositions.iter_mut().enumerate() {
                *entry = j;
            }
            return transpositions;
        }

        if rs > 0 && pivot_is_valid {
            for i in (k + 1)..size {
                mat[(i, k)] /= real_akk;
            }
        }
    }
    transpositions
}

/// Linearization inputs for an IMU factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuLinData<S: LieScalar> {
    /// Gravity in the world frame.
    pub g: Vector3<S>,
    /// `1 / gyro_bias_std`, the square-root weight of the gyro random walk
    pub gyro_bias_weight_sqrt: Vector3<S>,
    /// `1 / accel_bias_std`, the same for the accelerometer.
    pub accel_bias_weight_sqrt: Vector3<S>,
}

/// An IMU factor with 15 whitened rows over two 15-column states.
/// Rows 0–8 are preintegration, 9–11 gyro-bias random walk, and 12–14
/// accelerometer-bias random walk. Start-state columns precede end-state columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuBlock<S: LieScalar> {
    /// `Jp`.
    pub jp: Matrix15x30<S>,
    /// `r`.
    pub r: Vector15<S>,
    /// What `linearizeImu` returns: `imu_error + bg_error + ba_error`
    pub error: S,
}

impl<S: LieScalar> ImuBlock<S> {
    /// `ImuBlock::linearizeImu`.
    ///
    /// The residual is evaluated at the *linearized* states together with its
    /// Jacobians and then, if either state has a frozen linearization
    /// point, re-evaluated at the current states for its **value only**
    /// Skipping that second evaluation is trap 7 of the architecture
    /// dossier: it drifts rather than fails.
    pub fn linearize(
        meas: &IntegratedImuMeasurement<S>,
        lin_data: &ImuLinData<S>,
        start_state: &PoseVelBiasStateWithLin<S>,
        end_state: &PoseVelBiasStateWithLin<S>,
    ) -> Self {
        let start_idx: usize = 0;
        let end_idx: usize = POSE_VEL_BIAS_SIZE;

        let start_lin = start_state.state_lin();
        let end_lin = end_state.state_lin();
        let (mut res, jacobians): (Vector9<S>, ImuResidualJacobians<S>) = meas
            .residual_with_jacobians(
                &start_lin.pose_vel_state(),
                &lin_data.g,
                &end_lin.pose_vel_state(),
                &start_lin.bias_gyro,
                &start_lin.bias_accel,
            );

        if start_state.is_linearized() || end_state.is_linearized() {
            let start = start_state.state();
            let end = end_state.state();
            res = meas.residual(
                &start.pose_vel_state(),
                &lin_data.g,
                &end.pose_vel_state(),
                &start.bias_gyro,
                &start.bias_accel,
            );
        }

        let sqrt_cov_inv: Matrix9<S> = meas.get_cov_inv_sqrt();
        let mut jp: Matrix15x30<S> = Matrix15x30::zeros();
        let mut r: Vector15<S> = Vector15::zeros();

        let imu_error: S = c::<S>(0.5) * (sqrt_cov_inv * res).norm_squared(); // `:51`

        jp.fixed_view_mut::<9, 9>(0, start_idx)
            .copy_from(&(sqrt_cov_inv * jacobians.d_res_d_state0)); // `:54`
        jp.fixed_view_mut::<9, 9>(0, end_idx)
            .copy_from(&(sqrt_cov_inv * jacobians.d_res_d_state1)); // `:55`
        jp.fixed_view_mut::<9, 6>(0, start_idx + BIAS_GYRO_OFFSET)
            .copy_from(&(sqrt_cov_inv * jacobians.d_res_d_bias)); // `:57-58`
        r.fixed_rows_mut::<9>(0).copy_from(&(sqrt_cov_inv * res)); // `:60`

        let dt: S = c::<S>(meas.get_dt_ns() as f64) * c::<S>(1e-9); // `:63`
        let sqrt_dt: S = dt.sqrt();

        let gyro_bias_weight_dt: Vector3<S> = lin_data.gyro_bias_weight_sqrt / sqrt_dt;
        let res_bg: Vector3<S> = start_state.state().bias_gyro - end_state.state().bias_gyro;
        jp.fixed_view_mut::<3, 3>(9, start_idx + BIAS_GYRO_OFFSET)
            .copy_from(&Matrix3::from_diagonal(&gyro_bias_weight_dt));
        jp.fixed_view_mut::<3, 3>(9, end_idx + BIAS_GYRO_OFFSET)
            .copy_from(&Matrix3::from_diagonal(&-gyro_bias_weight_dt));
        let weighted_bg: Vector3<S> = gyro_bias_weight_dt.component_mul(&res_bg);
        r.fixed_rows_mut::<3>(9).copy_from(&weighted_bg);
        let bg_error: S = c::<S>(0.5) * weighted_bg.norm_squared();

        let accel_bias_weight_dt: Vector3<S> = lin_data.accel_bias_weight_sqrt / sqrt_dt;
        let res_ba: Vector3<S> = start_state.state().bias_accel - end_state.state().bias_accel;
        jp.fixed_view_mut::<3, 3>(12, start_idx + BIAS_ACCEL_OFFSET)
            .copy_from(&Matrix3::from_diagonal(&accel_bias_weight_dt));
        jp.fixed_view_mut::<3, 3>(12, end_idx + BIAS_ACCEL_OFFSET)
            .copy_from(&Matrix3::from_diagonal(&-accel_bias_weight_dt));
        let weighted_ba: Vector3<S> = accel_bias_weight_dt.component_mul(&res_ba);
        r.fixed_rows_mut::<3>(12).copy_from(&weighted_ba);
        let ba_error: S = c::<S>(0.5) * weighted_ba.norm_squared();

        Self {
            jp,
            r,
            error: imu_error + bg_error + ba_error,
        }
    }

    /// Scatter `JᵀJ` and `Jᵀr` at the supplied start/end state offsets.
    /// Out-of-range offsets are ignored instead of panicking (D32).
    pub fn add_dense_h_b(
        &self,
        start_idx: usize,
        end_idx: usize,
        h: &mut DMatrix<S>,
        b: &mut DVector<S>,
    ) {
        let size: usize = POSE_VEL_BIAS_SIZE;
        // The offsets come from an `AbsOrderMap` the caller owns, so the sum has
        // to be checked before it is compared: `usize::MAX + 15` wraps to a
        // small number in release and panics in debug (decision D32).
        let Some(needed) = start_idx.max(end_idx).checked_add(size) else {
            return;
        };
        if h.nrows() < needed || h.ncols() < needed || b.nrows() < needed {
            return;
        }
        let full_h: SMatrix<S, { 2 * POSE_VEL_BIAS_SIZE }, { 2 * POSE_VEL_BIAS_SIZE }> =
            self.jp.transpose() * self.jp;
        let full_b: SMatrix<S, { 2 * POSE_VEL_BIAS_SIZE }, 1> = self.jp.transpose() * self.r;

        for (block_row, row_offset) in [(0, start_idx), (size, end_idx)] {
            for (block_column, column_offset) in [(0, start_idx), (size, end_idx)] {
                for i in 0..size {
                    for j in 0..size {
                        h[(row_offset + i, column_offset + j)] +=
                            full_h[(block_row + i, block_column + j)];
                    }
                }
            }
            for i in 0..size {
                b[row_offset + i] += full_b[block_row + i];
            }
        }
    }

    /// Scatter the 15 whitened rows into the stacked square-root system,
    /// `add_dense_Q2Jp_Q2r`.
    ///
    /// `row_start_idx` is where this interval's rows begin; the driver advances
    /// it by `POSE_VEL_BIAS_SIZE` per interval
    /// Both column blocks are **added**,
    /// not assigned, which matters only if two intervals were ever
    /// given the same rows.
    ///
    /// Out-of-range offsets are ignored rather than panicking (decision D32),
    /// exactly as [`Self::add_dense_h_b`] does.
    pub fn add_dense_q2jp_q2r(
        &self,
        start_idx: usize,
        end_idx: usize,
        row_start_idx: usize,
        q2jp: &mut DMatrix<S>,
        q2r: &mut DVector<S>,
    ) {
        let size: usize = POSE_VEL_BIAS_SIZE;
        let Some(col_end) = start_idx.max(end_idx).checked_add(size) else {
            return;
        };
        let Some(row_end) = row_start_idx.checked_add(size) else {
            return;
        };
        if q2jp.ncols() < col_end || q2jp.nrows() < row_end || q2r.nrows() < row_end {
            return;
        }
        for (block_col, col_offset) in [(0, start_idx), (size, end_idx)] {
            for i in 0..size {
                for j in 0..size {
                    q2jp[(row_start_idx + i, col_offset + j)] += self.jp[(i, block_col + j)];
                }
            }
        }
        for i in 0..size {
            q2r[row_start_idx + i] += self.r[i];
        }
    }

    /// This factor's share of the model cost change, `backSubstitute`
    ///
    /// There is nothing to back-substitute — the IMU block has no eliminated
    /// variables — so the whole method is the `l_diff` accumulation
    /// `l_diff -= (J inc)ᵀ (0.5 (J inc) + r)` over the two states' slices of the
    /// increment.
    pub fn back_substitute(
        &self,
        start_idx: usize,
        end_idx: usize,
        pose_inc: &DVector<S>,
        l_diff: &mut S,
    ) {
        let size: usize = POSE_VEL_BIAS_SIZE;
        let fits = |offset: usize| {
            offset
                .checked_add(size)
                .is_some_and(|end| end <= pose_inc.nrows())
        };
        if !fits(start_idx) || !fits(end_idx) {
            return;
        }
        // `pose_inc_reduced` : the start state's block, then the
        // end state's.
        let mut reduced: SMatrix<S, { 2 * POSE_VEL_BIAS_SIZE }, 1> = SMatrix::zeros();
        for i in 0..size {
            reduced[i] = pose_inc[start_idx + i];
            reduced[size + i] = pose_inc[end_idx + i];
        }
        let jinc: SMatrix<S, POSE_VEL_BIAS_SIZE, 1> = self.jp * reduced;
        let mut diff: S = S::zero();
        for i in 0..size {
            diff += jinc[i] * (S::from_literal(0.5) * jinc[i] + self.r[i]);
        }
        *l_diff -= diff;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::collections::HashMap;

    use super::*;
    use crate::lie::Se3;
    use crate::types::{AbsOrderMap, POSE_SIZE, PoseVelBiasState};
    use approx::assert_abs_diff_eq;
    use nalgebra::SVector;
    use proptest::prelude::*;

    // Test support uses a smooth analytic trajectory with closed-form pose, velocity,
    // acceleration and body angular velocity, plus a seeded xorshift generator.
    // This keeps tests deterministic without a spline dependency.

    /// xorshift64*, so every ported test runs the same numbers each time.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x: u64 = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// Uniform value in `[-1, 1]`.
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }

        /// Uniform on `[low, high]`.
        fn range(&mut self, low: f64, high: f64) -> f64 {
            low + (self.uniform() + 1.0) * 0.5 * (high - low)
        }

        /// Three independent uniform values in `[-1, 1]`.
        fn vector3(&mut self) -> Vector3<f64> {
            Vector3::new(self.uniform(), self.uniform(), self.uniform())
        }

        /// `Sophus::Vector6d::Random()`.
        fn vector6(&mut self) -> SVector<f64, 6> {
            SVector::<f64, 6>::from_fn(|_, _| self.uniform())
        }
    }

    /// Smooth analytic trajectory: sinusoidal position gives exact velocity and
    /// acceleration. For `R = exp(phi(t))`, body angular velocity is `J_r(phi) phi_dot`.
    /// Low frequencies keep midpoint integration error bounded over the 20-second test.
    struct Trajectory {
        pos_amp: Vector3<f64>,
        pos_freq: Vector3<f64>,
        pos_phase: Vector3<f64>,
        rot_amp: Vector3<f64>,
        rot_freq: Vector3<f64>,
        rot_phase: Vector3<f64>,
    }

    impl Trajectory {
        fn new(rng: &mut Rng) -> Self {
            Self {
                pos_amp: Vector3::from_fn(|_, _| rng.range(0.8, 1.5)),
                pos_freq: Vector3::from_fn(|_, _| rng.range(0.15, 0.35)),
                pos_phase: Vector3::from_fn(|_, _| rng.range(0.0, std::f64::consts::TAU)),
                rot_amp: Vector3::from_fn(|_, _| rng.range(0.15, 0.30)),
                rot_freq: Vector3::from_fn(|_, _| rng.range(0.15, 0.30)),
                rot_phase: Vector3::from_fn(|_, _| rng.range(0.0, std::f64::consts::TAU)),
            }
        }

        fn seconds(t_ns: i64) -> f64 {
            t_ns as f64 * 1e-9
        }

        fn rotation_vector(&self, t: f64) -> Vector3<f64> {
            Vector3::from_fn(|i, _| {
                self.rot_amp[i] * (self.rot_freq[i] * t + self.rot_phase[i]).sin()
            })
        }

        fn rotation_vector_dot(&self, t: f64) -> Vector3<f64> {
            Vector3::from_fn(|i, _| {
                self.rot_amp[i]
                    * self.rot_freq[i]
                    * (self.rot_freq[i] * t + self.rot_phase[i]).cos()
            })
        }

        fn pose(&self, t_ns: i64) -> Se3<f64> {
            let t: f64 = Self::seconds(t_ns);
            Se3::new(
                So3::exp(&self.rotation_vector(t)),
                Vector3::from_fn(|i, _| {
                    self.pos_amp[i] * (self.pos_freq[i] * t + self.pos_phase[i]).sin()
                }),
            )
        }

        fn trans_vel_world(&self, t_ns: i64) -> Vector3<f64> {
            let t: f64 = Self::seconds(t_ns);
            Vector3::from_fn(|i, _| {
                self.pos_amp[i]
                    * self.pos_freq[i]
                    * (self.pos_freq[i] * t + self.pos_phase[i]).cos()
            })
        }

        fn trans_accel_world(&self, t_ns: i64) -> Vector3<f64> {
            let t: f64 = Self::seconds(t_ns);
            Vector3::from_fn(|i, _| {
                -self.pos_amp[i]
                    * self.pos_freq[i]
                    * self.pos_freq[i]
                    * (self.pos_freq[i] * t + self.pos_phase[i]).sin()
            })
        }

        /// `omega_body = J_r(phi) phi_dot` for `R(t) = exp(phi(t))`.
        fn rot_vel_body(&self, t_ns: i64) -> Vector3<f64> {
            let t: f64 = Self::seconds(t_ns);
            right_jacobian_so3(&self.rotation_vector(t)) * self.rotation_vector_dot(t)
        }

        /// Body-frame specific force and angular velocity sampled at the interval midpoint.
        fn sample(&self, t_ns: i64, dt_ns: i64) -> ImuSample {
            let pose: Se3<f64> = self.pose(t_ns);
            ImuSample {
                t_ns: t_ns + dt_ns / 2,
                gyro: self.rot_vel_body(t_ns),
                accel: pose.rotation.inverse() * (self.trans_accel_world(t_ns) - gravity::<f64>()),
            }
        }
    }

    /// Central differences at zero with epsilon `1e-8` and norm tolerance `1e-3`.
    /// Zero comparison uses the absolute norm; relative comparison scales by the
    /// smaller input norm.
    fn test_jacobian<const R: usize, const C: usize>(
        name: &str,
        ja: &SMatrix<f64, R, C>,
        func: impl Fn(&SVector<f64, C>) -> SVector<f64, R>,
        eps: f64,
        max_norm: f64,
    ) {
        let mut jn: SMatrix<f64, R, C> = SMatrix::zeros();
        for i in 0..C {
            let mut inc: SVector<f64, C> = SVector::zeros();
            inc[i] = eps;
            let fpe: SVector<f64, R> = func(&inc);
            let fme: SVector<f64, R> = func(&(-inc));
            jn.set_column(i, &((fpe - fme) / (2.0 * eps)));
        }

        assert!(
            ja.iter().all(|v: &f64| v.is_finite()),
            "{name}: Ja not finite\n{ja}"
        );
        assert!(
            jn.iter().all(|v: &f64| v.is_finite()),
            "{name}: Jn not finite\n{jn}"
        );

        let difference: f64 = (jn - ja).norm();
        if jn.norm() <= max_norm && ja.norm() <= max_norm {
            assert!(
                difference <= max_norm,
                "{name}: Ja not equal to Jn (diff norm {difference})\nJa\n{ja}\nJn\n{jn}"
            );
        } else {
            let bound: f64 = max_norm * jn.norm().min(ja.norm());
            assert!(
                difference <= bound,
                "{name}: Ja not equal to Jn (diff norm {difference} > {bound})\nJa\n{ja}\nJn\n{jn}"
            );
        }
    }

    /// `TestConstants<double>::epsilon`.
    const DEFAULT_EPS: f64 = 1e-8;
    /// `TestConstants<double>::max_norm`.
    const DEFAULT_MAX_NORM: f64 = 1e-3;
    const ACCEL_STD_DEV: f64 = 0.23;
    const GYRO_STD_DEV: f64 = 0.0027;

    fn noise_from_std_dev() -> ImuNoise<f64> {
        ImuNoise {
            accel_cov: Vector3::repeat(ACCEL_STD_DEV * ACCEL_STD_DEV),
            gyro_cov: Vector3::repeat(GYRO_STD_DEV * GYRO_STD_DEV),
        }
    }

    /// Relative matrix comparison scaled by the smaller input norm.
    fn is_approx(a: &Vector3<f64>, b: &Vector3<f64>, precision: f64) -> bool {
        (a - b).norm() <= precision * a.norm().min(b.norm())
    }

    // Preintegration invariants include the first covariance step, symmetric positive
    // semidefiniteness and the whitening identity. These directly check covariance
    // behavior without a large Monte Carlo run.

    /// `ImuPreintegrationTestCase.PredictTestGT`.
    ///
    /// 2 000 samples over 20 s, then `predictState` from the true state at 0 has
    /// to land on the true state at the end: velocity and translation within a
    /// relative `1e-4`, orientation within `1e-6` rad.
    #[test]
    fn predict_state_reaches_the_ground_truth_state() {
        let mut rng: Rng = Rng::new(0x5eed_0001);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());

        let state0: PoseVelState<f64> =
            PoseVelState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));

        let dt_ns: i64 = 10_000_000;
        let ones: Vector3<f64> = Vector3::repeat(1.0);
        let mut t_ns: i64 = dt_ns / 2;
        while t_ns < 20_000_000_000 {
            meas.integrate(&trajectory.sample(t_ns, dt_ns), &ones, &ones)
                .unwrap();
            t_ns += dt_ns;
        }

        let end_t_ns: i64 = meas.get_dt_ns();
        let state1: PoseVelState<f64> = meas.predict_state(&state0, &gravity::<f64>());
        let gt_pose: Se3<f64> = trajectory.pose(end_t_ns);
        let gt_vel: Vector3<f64> = trajectory.trans_vel_world(end_t_ns);

        assert!(
            is_approx(&gt_vel, &state1.vel_w_i, 1e-4),
            "vel_gt {gt_vel} vel {}",
            state1.vel_w_i
        );
        let angular_distance: f64 = gt_pose
            .rotation
            .quaternion()
            .angle_to(state1.t_w_i.rotation.quaternion());
        assert!(
            angular_distance <= 1e-6,
            "angular distance {angular_distance}"
        );
        assert!(
            is_approx(&gt_pose.translation, &state1.t_w_i.translation, 1e-4),
            "p_gt {} p {}",
            gt_pose.translation,
            state1.t_w_i.translation
        );
    }

    /// Compare transition and noise-input Jacobians with central differences at each step.
    #[test]
    fn propagate_state_jacobians_match_finite_differences() {
        let mut rng: Rng = Rng::new(0x5eed_0002);
        let trajectory: Trajectory = Trajectory::new(&mut rng);

        let dt_ns: i64 = 10_000_000;
        let mut t_ns: i64 = dt_ns / 2;
        while t_ns < 2_000_000_000 {
            let sample: ImuSample = trajectory.sample(t_ns, dt_ns);
            let curr_t_ns: i64 = t_ns - dt_ns / 2;
            let curr_state: PoseVelState<f64> = PoseVelState::new(
                curr_t_ns,
                trajectory.pose(curr_t_ns),
                trajectory.trans_vel_world(curr_t_ns),
            );
            let accel: Vector3<f64> = sample.accel;
            let gyro: Vector3<f64> = sample.gyro;
            let (next_state, j) = IntegratedImuMeasurement::<f64>::propagate_state(
                &curr_state,
                sample.t_ns,
                &accel,
                &gyro,
            )
            .unwrap();

            test_jacobian(
                "F_TEST",
                &j.d_next_d_curr,
                |x: &Vector9<f64>| {
                    let mut perturbed: PoseVelState<f64> = curr_state;
                    perturbed.apply_inc(x);
                    let (next, _) = IntegratedImuMeasurement::<f64>::propagate_state(
                        &perturbed,
                        sample.t_ns,
                        &accel,
                        &gyro,
                    )
                    .unwrap();
                    next_state.diff(&next)
                },
                DEFAULT_EPS,
                DEFAULT_MAX_NORM,
            );

            test_jacobian(
                "A_TEST",
                &j.d_next_d_accel,
                |x: &Vector3<f64>| {
                    let (next, _) = IntegratedImuMeasurement::<f64>::propagate_state(
                        &curr_state,
                        sample.t_ns,
                        &(accel + x),
                        &gyro,
                    )
                    .unwrap();
                    next_state.diff(&next)
                },
                DEFAULT_EPS,
                DEFAULT_MAX_NORM,
            );

            test_jacobian(
                "G_TEST",
                &j.d_next_d_gyro,
                |x: &Vector3<f64>| {
                    let (next, _) = IntegratedImuMeasurement::<f64>::propagate_state(
                        &curr_state,
                        sample.t_ns,
                        &accel,
                        &(gyro + x),
                    )
                    .unwrap();
                    next_state.diff(&next)
                },
                1e-8,
                DEFAULT_MAX_NORM,
            );

            t_ns += dt_ns;
        }
    }

    /// `ImuPreintegrationTestCase.ResidualTest`.
    ///
    /// The residual at the true end state is zero to `1e-6` per coefficient, and
    /// the four Jacobians match central differences at a perturbed end state.
    #[test]
    fn residual_vanishes_at_the_truth_and_its_jacobians_match() {
        let mut rng: Rng = Rng::new(0x5eed_0003);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;

        let mut meas: IntegratedImuMeasurement<f64> = IntegratedImuMeasurement::new(0, &bg, &ba);
        let state0: PoseVelState<f64> =
            PoseVelState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));

        let dt_ns: i64 = 10_000_000;
        let ones: Vector3<f64> = Vector3::repeat(1.0);
        let mut t_ns: i64 = dt_ns / 2;
        while t_ns < 100_000_000 {
            let mut sample: ImuSample = trajectory.sample(t_ns, dt_ns);
            sample.accel += ba;
            sample.gyro += bg;
            meas.integrate(&sample, &ones, &ones).unwrap();
            t_ns += dt_ns;
        }

        let end_t_ns: i64 = meas.get_dt_ns();
        let g: Vector3<f64> = gravity::<f64>();
        let state1_gt: PoseVelState<f64> = PoseVelState::new(
            end_t_ns,
            trajectory.pose(end_t_ns),
            trajectory.trans_vel_world(end_t_ns),
        );
        let res_gt: Vector9<f64> = meas.residual(&state0, &g, &state1_gt, &bg, &ba);
        assert!(res_gt.amax() <= 1e-6, "res_gt {}", res_gt.transpose());

        let state1: PoseVelState<f64> = PoseVelState::new(
            end_t_ns,
            trajectory.pose(end_t_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
            trajectory.trans_vel_world(end_t_ns) + rng.vector3() / 10.0,
        );
        let (_, jacobians) = meas.residual_with_jacobians(&state0, &g, &state1, &bg, &ba);

        test_jacobian(
            "d_res_d_state0",
            &jacobians.d_res_d_state0,
            |x: &Vector9<f64>| {
                let mut perturbed: PoseVelState<f64> = state0;
                perturbed.apply_inc(x);
                meas.residual(&perturbed, &g, &state1, &bg, &ba)
            },
            DEFAULT_EPS,
            DEFAULT_MAX_NORM,
        );
        test_jacobian(
            "d_res_d_state1",
            &jacobians.d_res_d_state1,
            |x: &Vector9<f64>| {
                let mut perturbed: PoseVelState<f64> = state1;
                perturbed.apply_inc(x);
                meas.residual(&state0, &g, &perturbed, &bg, &ba)
            },
            DEFAULT_EPS,
            DEFAULT_MAX_NORM,
        );
        test_jacobian(
            "d_res_d_bg",
            &jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 0).into_owned(),
            |x: &Vector3<f64>| meas.residual(&state0, &g, &state1, &(bg + x), &ba),
            DEFAULT_EPS,
            DEFAULT_MAX_NORM,
        );
        test_jacobian(
            "d_res_d_ba",
            &jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 3).into_owned(),
            |x: &Vector3<f64>| meas.residual(&state0, &g, &state1, &bg, &(ba + x)),
            DEFAULT_EPS,
            DEFAULT_MAX_NORM,
        );
    }

    /// The samples `BiasTest` and `ResidualBiasTest` share
    /// 100 samples over 1 s with both
    /// biases added in.
    fn biased_samples(
        trajectory: &Trajectory,
        bg: &Vector3<f64>,
        ba: &Vector3<f64>,
    ) -> Vec<ImuSample> {
        let dt_ns: i64 = 10_000_000;
        let mut samples: Vec<ImuSample> = Vec::new();
        let mut t_ns: i64 = dt_ns / 2;
        while t_ns < 1_000_000_000 {
            let mut sample: ImuSample = trajectory.sample(t_ns, dt_ns);
            sample.accel += ba;
            sample.gyro += bg;
            samples.push(sample);
            t_ns += dt_ns;
        }
        samples
    }

    fn integrate_all(
        start_t_ns: i64,
        bg: &Vector3<f64>,
        ba: &Vector3<f64>,
        samples: &[ImuSample],
    ) -> IntegratedImuMeasurement<f64> {
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(start_t_ns, bg, ba);
        let ones: Vector3<f64> = Vector3::repeat(1.0);
        for sample in samples {
            meas.integrate(sample, &ones, &ones).unwrap();
        }
        meas
    }

    /// `ImuPreintegrationTestCase.BiasTest`.
    ///
    /// The two bias Jacobians against re-integrating the same samples about a
    /// perturbed linearization point, compared through `PoseVelState::diff`.
    #[test]
    fn bias_jacobians_match_reintegration_at_a_perturbed_bias() {
        let mut rng: Rng = Rng::new(0x5eed_0004);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;
        let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);

        let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
        let delta_state: PoseVelState<f64> = *meas.get_delta_state();

        test_jacobian(
            "d_state_d_bg",
            meas.get_d_state_d_bg(),
            |x: &Vector3<f64>| {
                let perturbed: IntegratedImuMeasurement<f64> =
                    integrate_all(0, &(bg + x), &ba, &samples);
                delta_state.diff(perturbed.get_delta_state())
            },
            DEFAULT_EPS,
            DEFAULT_MAX_NORM,
        );
        test_jacobian(
            "d_state_d_ba",
            meas.get_d_state_d_ba(),
            |x: &Vector3<f64>| {
                let perturbed: IntegratedImuMeasurement<f64> =
                    integrate_all(0, &bg, &(ba + x), &samples);
                delta_state.diff(perturbed.get_delta_state())
            },
            DEFAULT_EPS,
            DEFAULT_MAX_NORM,
        );
    }

    /// Bias correction must agree with reintegration about the shifted bias within
    /// relative `1e-4`. Compare both bias Jacobians through reintegration, with the
    /// gyro Jacobian's norm tolerance `1e-2`.
    #[test]
    fn residual_bias_correction_agrees_with_reintegration() {
        let mut rng: Rng = Rng::new(0x5eed_0005);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;
        let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
        let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);

        let g: Vector3<f64> = gravity::<f64>();
        let state0: PoseVelState<f64> =
            PoseVelState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));
        let end_t_ns: i64 = meas.get_dt_ns();
        let state1: PoseVelState<f64> = PoseVelState::new(
            end_t_ns,
            trajectory.pose(end_t_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
            trajectory.trans_vel_world(end_t_ns) + rng.vector3() / 10.0,
        );

        let bg_test: Vector3<f64> = bg + rng.vector3() / 1000.0;
        let ba_test: Vector3<f64> = ba + rng.vector3() / 100.0;
        let (res, jacobians) =
            meas.residual_with_jacobians(&state0, &g, &state1, &bg_test, &ba_test);

        let reintegrated: IntegratedImuMeasurement<f64> =
            integrate_all(0, &bg_test, &ba_test, &samples);
        let res1: Vector9<f64> = reintegrated.residual(&state0, &g, &state1, &bg_test, &ba_test);
        assert!(
            (res - res1).norm() <= 1e-4 * res.norm().min(res1.norm()),
            "res {}\nres1 {}",
            res.transpose(),
            res1.transpose()
        );

        test_jacobian(
            "d_res_d_ba",
            &jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 3).into_owned(),
            |x: &Vector3<f64>| {
                let perturbed: IntegratedImuMeasurement<f64> =
                    integrate_all(0, &bg_test, &(ba_test + x), &samples);
                perturbed.residual(&state0, &g, &state1, &bg_test, &(ba_test + x))
            },
            DEFAULT_EPS,
            DEFAULT_MAX_NORM,
        );
        test_jacobian(
            "d_res_d_bg",
            &jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 0).into_owned(),
            |x: &Vector3<f64>| {
                let perturbed: IntegratedImuMeasurement<f64> =
                    integrate_all(0, &(bg_test + x), &ba_test, &samples);
                perturbed.residual(&state0, &g, &state1, &(bg_test + x), &ba_test)
            },
            1e-8,
            1e-2,
        );
    }

    // IMU factor invariants.

    /// `ScBundleAdjustmentBase::checkNullspace`,
    /// restricted to the pose-velocity-bias blocks the two IMU tests use — the
    /// pose-only branch belongs to the visual factors.
    ///
    /// Returns `xHx + xb` for six global directions plus one random direction: a
    /// shift of every position along x, y and z, and a rotation of every pose
    /// *and velocity* about the position centroid in roll, pitch and yaw. For a
    /// VIO problem only yaw and the three translations are unobservable, so only
    /// entries 0, 1, 2 and 5 are expected to vanish.
    fn check_nullspace(
        h: &DMatrix<f64>,
        b: &DVector<f64>,
        order: &AbsOrderMap,
        frame_states: &HashMap<i64, PoseVelBiasStateWithLin<f64>>,
        rng: &mut Rng,
    ) -> [f64; 7] {
        let size: usize = order.total_size();
        let mut increments: [DVector<f64>; 6] = std::array::from_fn(|_| DVector::zeros(size));

        // the centroid the rotations turn about.
        let mut mean_trans: Vector3<f64> = Vector3::zeros();
        for (frame_id, _, _) in order.iter() {
            mean_trans += frame_states[&frame_id].state_lin().t_w_i.translation;
        }
        mean_trans /= order.items() as f64;

        let eps: f64 = 0.01; // `:541`
        for (frame_id, offset, _) in order.iter() {
            let state: &PoseVelBiasState<f64> = frame_states[&frame_id].state_lin();
            for axis in 0..3 {
                increments[axis][offset + axis] = eps; // `:545-547`
                increments[3 + axis][offset + 3 + axis] = eps; // `:548-550`
            }

            //  and : the rotation increments also move the
            // translations (about the centroid) and the velocities.
            let j: Matrix3<f64> = -So3::hat(&(state.t_w_i.translation - mean_trans)) * eps;
            let j_vel: Matrix3<f64> = -So3::hat(&state.vel_w_i) * eps;
            for axis in 0..3 {
                for row in 0..3 {
                    increments[3 + axis][offset + row] = j[(row, axis)];
                    increments[3 + axis][offset + POSE_SIZE + row] = j_vel[(row, axis)];
                }
            }
        }

        let mut result: [f64; 7] = [0.0; 7];
        for (index, increment) in increments.iter().enumerate() {
            let unit: DVector<f64> = increment / increment.norm(); // `:589-594`
            result[index] = unit.dot(&(h * &unit)) + unit.dot(b);
        }
        let random: DVector<f64> = DVector::from_fn(size, |_, _| rng.uniform());
        let random: DVector<f64> = &random / random.norm(); // `:601-603`
        result[6] = random.dot(&(h * &random)) + random.dot(b);
        result
    }

    /// `ScBundleAdjustmentBase::computeImuError`,
    /// for the measurements the two IMU tests hold, summed into one number.
    ///
    /// Note `gyro_bias_weight / dt` here against `gyro_bias_weight_sqrt / sqrt(dt)`
    /// in the block : the caller passes the squared weight.
    fn compute_imu_error(
        order: &AbsOrderMap,
        states: &HashMap<i64, PoseVelBiasStateWithLin<f64>>,
        measurements: &[IntegratedImuMeasurement<f64>],
        gyro_bias_weight: &Vector3<f64>,
        accel_bias_weight: &Vector3<f64>,
        g: &Vector3<f64>,
    ) -> f64 {
        let mut total: f64 = 0.0;
        for meas in measurements {
            if meas.get_dt_ns() == 0 {
                continue;
            }
            let start_t: i64 = meas.get_start_t_ns();
            let end_t: i64 = start_t + meas.get_dt_ns();
            if !order.contains(start_t) || !order.contains(end_t) {
                continue;
            }
            let start: &PoseVelBiasState<f64> = states[&start_t].state();
            let end: &PoseVelBiasState<f64> = states[&end_t].state();
            let res: Vector9<f64> = meas.residual(
                &start.pose_vel_state(),
                g,
                &end.pose_vel_state(),
                &start.bias_gyro,
                &start.bias_accel,
            );
            total += 0.5 * res.dot(&(meas.get_cov_inv() * res));

            let dt: f64 = meas.get_dt_ns() as f64 * 1e-9;
            let res_bg: Vector3<f64> = start.bias_gyro - end.bias_gyro;
            total += 0.5 * res_bg.dot(&(gyro_bias_weight / dt).component_mul(&res_bg));
            let res_ba: Vector3<f64> = start.bias_accel - end.bias_accel;
            total += 0.5 * res_ba.dot(&(accel_bias_weight / dt).component_mul(&res_ba));
        }
        total
    }

    /// Noisy samples over `[from_ns, to_ns)`, as both nullspace tests build them
    fn noisy_samples(
        trajectory: &Trajectory,
        bg: &Vector3<f64>,
        ba: &Vector3<f64>,
        from_ns: i64,
        to_ns: i64,
        rng: &mut Rng,
    ) -> Vec<ImuSample> {
        let dt_ns: i64 = 10_000_000;
        let mut samples: Vec<ImuSample> = Vec::new();
        let mut t_ns: i64 = from_ns;
        while t_ns < to_ns {
            let mut sample: ImuSample = trajectory.sample(t_ns, dt_ns);
            sample.accel += ba + rng.vector3() * ACCEL_STD_DEV;
            sample.gyro += bg + rng.vector3() * GYRO_STD_DEV;
            samples.push(sample);
            t_ns += dt_ns;
        }
        samples
    }

    fn integrate_noisy(
        start_t_ns: i64,
        bg: &Vector3<f64>,
        ba: &Vector3<f64>,
        samples: &[ImuSample],
    ) -> IntegratedImuMeasurement<f64> {
        let noise: ImuNoise<f64> = noise_from_std_dev();
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(start_t_ns, bg, ba);
        for sample in samples {
            meas.integrate(sample, &noise.accel_cov, &noise.gyro_cov)
                .unwrap();
        }
        meas
    }

    /// both weights are `1e3`.
    fn lin_data() -> ImuLinData<f64> {
        ImuLinData {
            g: gravity::<f64>(),
            gyro_bias_weight_sqrt: Vector3::repeat(1e3),
            accel_bias_weight_sqrt: Vector3::repeat(1e3),
        }
    }

    /// `VioTestSuite.ImuNullspace2Test`.
    ///
    /// One IMU factor between two full states: the block's `H` and `b` have to
    /// reproduce the error change to `2e-2` for ten small random increments, and
    /// the three global translations and the yaw rotation have to lie in the
    /// nullspace of `H` and `b` to `1e-8` and `1e-6`.
    #[test]
    fn imu_nullspace_2() {
        let mut rng: Rng = Rng::new(0x5eed_0006);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;

        let samples: Vec<ImuSample> =
            noisy_samples(&trajectory, &bg, &ba, 5_000_000, 100_000_000, &mut rng);
        let meas: IntegratedImuMeasurement<f64> = integrate_noisy(0, &bg, &ba, &samples);

        let state0: PoseVelBiasState<f64> =
            PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
        let end_t_ns: i64 = meas.get_dt_ns();
        let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
            end_t_ns,
            trajectory.pose(end_t_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
            trajectory.trans_vel_world(end_t_ns) + rng.vector3() / 10.0,
            bg,
            ba,
        );

        let mut frame_states: HashMap<i64, PoseVelBiasStateWithLin<f64>> = HashMap::new();
        frame_states.insert(0, PoseVelBiasStateWithLin::new(state0, false));
        frame_states.insert(end_t_ns, PoseVelBiasStateWithLin::new(state1, false));

        let mut order: AbsOrderMap = AbsOrderMap::new();
        order.push(0, POSE_VEL_BIAS_SIZE).unwrap();
        order.push(end_t_ns, POSE_VEL_BIAS_SIZE).unwrap();
        let size: usize = order.total_size();

        let ild: ImuLinData<f64> = lin_data();
        let block: ImuBlock<f64> =
            ImuBlock::linearize(&meas, &ild, &frame_states[&0], &frame_states[&end_t_ns]);
        let mut h: DMatrix<f64> = DMatrix::zeros(size, size);
        let mut b: DVector<f64> = DVector::zeros(size);
        block.add_dense_h_b(0, POSE_VEL_BIAS_SIZE, &mut h, &mut b);
        let e0: f64 = block.error;

        // the quadratic model has to predict the error change.
        let gyro_weight: Vector3<f64> = ild.gyro_bias_weight_sqrt.map(|v: f64| v * v);
        let accel_weight: Vector3<f64> = ild.accel_bias_weight_sqrt.map(|v: f64| v * v);
        for _ in 0..10 {
            let raw: DVector<f64> = DVector::from_fn(size, |_, _| rng.uniform());
            let inc: DVector<f64> = &raw / raw.norm() / 10_000.0;

            let mut moved: HashMap<i64, PoseVelBiasStateWithLin<f64>> = frame_states.clone();
            let mut inc0: Vector15<f64> = Vector15::zeros();
            let mut inc1: Vector15<f64> = Vector15::zeros();
            for i in 0..POSE_VEL_BIAS_SIZE {
                inc0[i] = inc[i];
                inc1[i] = inc[POSE_VEL_BIAS_SIZE + i];
            }
            moved.get_mut(&0).unwrap().apply_inc(&inc0);
            moved.get_mut(&end_t_ns).unwrap().apply_inc(&inc1);

            let e1: f64 = compute_imu_error(
                &order,
                &moved,
                std::slice::from_ref(&meas),
                &gyro_weight,
                &accel_weight,
                &ild.g,
            ) - e0;
            let e2: f64 = 0.5 * inc.dot(&(&h * &inc)) + inc.dot(&b);
            assert!((e1 - e2).abs() <= 2e-2, "e1 {e1} e2 {e2}");
        }

        let null_res: [f64; 7] = check_nullspace(&h, &b, &order, &frame_states, &mut rng);
        assert!(null_res[0].abs() <= 1e-8, "x {}", null_res[0]);
        assert!(null_res[1].abs() <= 1e-8, "y {}", null_res[1]);
        assert!(null_res[2].abs() <= 1e-8, "z {}", null_res[2]);
        assert!(null_res[5].abs() <= 1e-6, "yaw {}", null_res[5]);
        // Gravity makes roll and pitch observable; require real information in those
        // and random directions so an all-zero Hessian cannot pass vacuously.
        assert!(null_res[3].abs() > 1.0, "roll {}", null_res[3]);
        assert!(null_res[4].abs() > 1.0, "pitch {}", null_res[4]);
        assert!(null_res[6].abs() > 1.0, "random {}", null_res[6]);
    }

    /// `VioTestSuite.ImuNullspace3Test`.
    ///
    /// Two consecutive IMU factors over three states; the same four directions
    /// have to stay in the nullspace once both blocks are accumulated.
    #[test]
    fn imu_nullspace_3() {
        let mut rng: Rng = Rng::new(0x5eed_0007);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;

        let samples1: Vec<ImuSample> =
            noisy_samples(&trajectory, &bg, &ba, 5_000_000, 1_000_000_000, &mut rng);
        let meas1: IntegratedImuMeasurement<f64> = integrate_noisy(0, &bg, &ba, &samples1);
        let t1_ns: i64 = meas1.get_dt_ns();

        let samples2: Vec<ImuSample> = noisy_samples(
            &trajectory,
            &bg,
            &ba,
            t1_ns + 5_000_000,
            2_000_000_000,
            &mut rng,
        );
        let meas2: IntegratedImuMeasurement<f64> = integrate_noisy(t1_ns, &bg, &ba, &samples2);
        let t2_ns: i64 = t1_ns + meas2.get_dt_ns();

        let state0: PoseVelBiasState<f64> =
            PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
        let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
            t1_ns,
            trajectory.pose(t1_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
            trajectory.trans_vel_world(t1_ns) + rng.vector3() / 10.0,
            bg,
            ba,
        );
        let state2: PoseVelBiasState<f64> = PoseVelBiasState::new(
            t2_ns,
            trajectory.pose(t2_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
            trajectory.trans_vel_world(t2_ns) + rng.vector3() / 10.0,
            bg,
            ba,
        );

        let mut frame_states: HashMap<i64, PoseVelBiasStateWithLin<f64>> = HashMap::new();
        frame_states.insert(0, PoseVelBiasStateWithLin::new(state0, false));
        frame_states.insert(t1_ns, PoseVelBiasStateWithLin::new(state1, false));
        frame_states.insert(t2_ns, PoseVelBiasStateWithLin::new(state2, false));

        let mut order: AbsOrderMap = AbsOrderMap::new();
        order.push(0, POSE_VEL_BIAS_SIZE).unwrap();
        order.push(t1_ns, POSE_VEL_BIAS_SIZE).unwrap();
        order.push(t2_ns, POSE_VEL_BIAS_SIZE).unwrap();
        let size: usize = order.total_size();

        let ild: ImuLinData<f64> = lin_data();
        let mut h: DMatrix<f64> = DMatrix::zeros(size, size);
        let mut b: DVector<f64> = DVector::zeros(size);
        ImuBlock::linearize(&meas1, &ild, &frame_states[&0], &frame_states[&t1_ns]).add_dense_h_b(
            0,
            POSE_VEL_BIAS_SIZE,
            &mut h,
            &mut b,
        );
        ImuBlock::linearize(&meas2, &ild, &frame_states[&t1_ns], &frame_states[&t2_ns])
            .add_dense_h_b(POSE_VEL_BIAS_SIZE, 2 * POSE_VEL_BIAS_SIZE, &mut h, &mut b);

        let null_res: [f64; 7] = check_nullspace(&h, &b, &order, &frame_states, &mut rng);
        assert!(null_res[0].abs() <= 1e-8, "x {}", null_res[0]);
        assert!(null_res[1].abs() <= 1e-8, "y {}", null_res[1]);
        assert!(null_res[2].abs() <= 1e-8, "z {}", null_res[2]);
        assert!(null_res[5].abs() <= 1e-6, "yaw {}", null_res[5]);
        // Gravity makes roll and pitch observable; require real information in those
        // and random directions so an all-zero Hessian cannot pass vacuously.
        assert!(null_res[3].abs() > 1.0, "roll {}", null_res[3]);
        assert!(null_res[4].abs() > 1.0, "pitch {}", null_res[4]);
        assert!(null_res[6].abs() > 1.0, "random {}", null_res[6]);
    }

    // ── Port-specific tests ─────────────────────────────────────────────

    /// Accelerometer bias cannot move delta rotation: its input Jacobian has zero
    /// rotation rows and the transition's rotation rows are `[0 I 0]`.
    #[test]
    fn the_accel_bias_jacobian_never_touches_rotation() {
        let mut rng: Rng = Rng::new(0x5eed_0008);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;
        let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
        let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
        assert_eq!(
            meas.get_d_state_d_ba()
                .fixed_view::<3, 3>(3, 0)
                .into_owned(),
            Matrix3::zeros()
        );
    }

    /// The empty measurement is exactly zero everywhere, and its square-root
    /// inverse covariance is zero rather than infinite — the pseudo-inverse
    /// branch.
    #[test]
    fn an_empty_measurement_has_a_zero_pseudo_inverse() {
        let meas: IntegratedImuMeasurement<f64> = IntegratedImuMeasurement::default();
        assert_eq!(meas.get_dt_ns(), 0);
        assert_eq!(*meas.get_cov(), Matrix9::zeros());
        assert_eq!(meas.get_cov_inv_sqrt(), Matrix9::zeros());
        assert_eq!(meas.get_cov_inv(), Matrix9::zeros());
    }

    /// `F`, `A` and `G` for a rig whose gyroscope reads exactly zero, written
    /// out here from Paper 1 Eq. (13) instead of being taken from
    /// [`IntegratedImuMeasurement::propagate_state`].
    ///
    /// With no rotation the delta rotation stays the identity, so `accel_world`
    /// is the measurement itself and `rightJacobianSO3(0)` is the identity: the
    /// three Jacobians are the *same* at every step, which turns the covariance
    /// recurrence into a closed-form sum. Being written twice is the point —
    /// a check that rebuilds the expected covariance out of the implementation's
    /// own `F` cannot see a wrong `F`.
    fn constant_jacobians(
        dt: f64,
        accel: &Vector3<f64>,
    ) -> (Matrix9<f64>, Matrix9x3<f64>, Matrix9x3<f64>) {
        let hat: Matrix3<f64> = So3::hat(&(-accel * dt));

        let mut f: Matrix9<f64> = Matrix9::identity();
        f.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(Matrix3::identity() * dt));
        f.fixed_view_mut::<3, 3>(6, 3).copy_from(&hat);
        f.fixed_view_mut::<3, 3>(0, 3).copy_from(&(hat * dt * 0.5));

        let mut a: Matrix9x3<f64> = Matrix9x3::zeros();
        a.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(Matrix3::identity() * 0.5 * dt * dt));
        a.fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(Matrix3::identity() * dt));

        let mut g: Matrix9x3<f64> = Matrix9x3::zeros();
        g.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(Matrix3::identity() * dt));
        let d_vel_d_gyro: Matrix3<f64> = hat * 0.5 * dt;
        g.fixed_view_mut::<3, 3>(6, 0).copy_from(&d_vel_d_gyro);
        g.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(d_vel_d_gyro * 0.5 * dt));

        (f, a, g)
    }

    /// The relative Frobenius distance between two matrices.
    fn relative_distance<const R: usize, const C: usize>(
        got: &SMatrix<f64, R, C>,
        want: &SMatrix<f64, R, C>,
    ) -> f64 {
        let scale: f64 = want.norm().max(f64::MIN_POSITIVE);
        (got - want).norm() / scale
    }

    proptest! {
        /// The covariance and both bias Jacobians against the closed forms of
        /// their recurrences, evaluated with an independently written `F`, `A`
        /// and `G`.
        ///
        /// `cov_n = Σ_{k<n} F^k Q (F^k)ᵀ` with `Q = A Σa Aᵀ + G Σg Gᵀ`
        /// `d_state_d_ba_n = -Σ_{k<n} F^k A` and
        /// `d_state_d_bg_n = -Σ_{k<n} F^k G`. One step, thirty
        /// steps and sub-millisecond intervals all fall out of the ranges.
        #[test]
        fn the_covariance_recurrence_matches_its_closed_form(
            steps in 1usize..30,
            dt_ns in 50_000i64..20_000_000,
            ax in -12.0f64..12.0,
            ay in -12.0f64..12.0,
            az in -12.0f64..12.0,
        ) {
            let noise: ImuNoise<f64> = noise_from_std_dev();
            let accel: Vector3<f64> = Vector3::new(ax, ay, az);
            let dt: f64 = dt_ns as f64 * 1e-9;

            let mut meas: IntegratedImuMeasurement<f64> =
                IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
            for step in 1..=steps as i64 {
                meas.integrate(
                    &ImuSample { t_ns: step * dt_ns, gyro: Vector3::zeros(), accel },
                    &noise.accel_cov,
                    &noise.gyro_cov,
                )?;
            }

            let (f, a, g) = constant_jacobians(dt, &accel);
            let q: Matrix9<f64> = a * Matrix3::from_diagonal(&noise.accel_cov) * a.transpose()
                + g * Matrix3::from_diagonal(&noise.gyro_cov) * g.transpose();

            let mut power: Matrix9<f64> = Matrix9::identity();
            let mut cov: Matrix9<f64> = Matrix9::zeros();
            let mut d_ba: Matrix9x3<f64> = Matrix9x3::zeros();
            let mut d_bg: Matrix9x3<f64> = Matrix9x3::zeros();
            for _ in 0..steps {
                cov += power * q * power.transpose();
                d_ba -= power * a;
                d_bg -= power * g;
                power *= f;
            }

            prop_assert!(
                relative_distance(meas.get_cov(), &cov) <= 1e-9,
                "cov off by {} relative",
                relative_distance(meas.get_cov(), &cov)
            );
            prop_assert!(
                relative_distance(meas.get_d_state_d_ba(), &d_ba) <= 1e-12,
                "d_state_d_ba off by {} relative",
                relative_distance(meas.get_d_state_d_ba(), &d_ba)
            );
            prop_assert!(
                relative_distance(meas.get_d_state_d_bg(), &d_bg) <= 1e-12,
                "d_state_d_bg off by {} relative",
                relative_distance(meas.get_d_state_d_bg(), &d_bg)
            );
        }
    }

    /// The same closed form in `f32`, at one fixed configuration.
    ///
    /// The recurrence and the explicit sum are different orders of the same
    /// arithmetic, so the tolerance is the `f32` accumulation of 20 steps, not
    /// the agreement of two exact quantities.
    #[test]
    fn the_covariance_recurrence_matches_its_closed_form_in_float() {
        let accel: Vector3<f64> = Vector3::new(0.35, -1.25, 9.75);
        let dt_ns: i64 = 2_500_000;
        let steps: i64 = 20;
        let noise: ImuNoise<f32> = ImuNoise {
            accel_cov: Vector3::repeat((ACCEL_STD_DEV * ACCEL_STD_DEV) as f32),
            gyro_cov: Vector3::repeat((GYRO_STD_DEV * GYRO_STD_DEV) as f32),
        };

        let mut meas: IntegratedImuMeasurement<f32> =
            IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
        for step in 1..=steps {
            meas.integrate(
                &ImuSample {
                    t_ns: step * dt_ns,
                    gyro: Vector3::zeros(),
                    accel,
                },
                &noise.accel_cov,
                &noise.gyro_cov,
            )
            .unwrap();
        }

        let dt: f64 = dt_ns as f64 * 1e-9;
        let (f, a, g) = constant_jacobians(dt, &accel);
        let accel_cov: Vector3<f64> = Vector3::repeat(f64::from(noise.accel_cov.x));
        let gyro_cov: Vector3<f64> = Vector3::repeat(f64::from(noise.gyro_cov.x));
        let q: Matrix9<f64> = a * Matrix3::from_diagonal(&accel_cov) * a.transpose()
            + g * Matrix3::from_diagonal(&gyro_cov) * g.transpose();

        let mut power: Matrix9<f64> = Matrix9::identity();
        let mut cov: Matrix9<f64> = Matrix9::zeros();
        for _ in 0..steps {
            cov += power * q * power.transpose();
            power *= f;
        }

        let got: Matrix9<f64> = meas.get_cov().map(f64::from);
        assert!(
            relative_distance(&got, &cov) <= 1e-5,
            "cov off by {} relative",
            relative_distance(&got, &cov)
        );
    }

    /// A duplicate or reordered sample is rejected and nothing is integrated.
    #[test]
    fn duplicate_and_reordered_samples_are_rejected() {
        let ones: Vector3<f64> = Vector3::repeat(1.0);
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(1_000, &Vector3::zeros(), &Vector3::zeros());
        let sample = |t_ns: i64| ImuSample {
            t_ns,
            gyro: Vector3::new(0.01, 0.0, 0.0),
            accel: Vector3::new(0.0, 0.0, 9.81),
        };

        meas.integrate(&sample(2_000), &ones, &ones).unwrap();
        assert_eq!(meas.get_dt_ns(), 1_000);
        assert_eq!(
            meas.integrate(&sample(2_000), &ones, &ones),
            Err(ImuError::NonMonotonicSample {
                previous_t_ns: 1_000,
                t_ns: 1_000
            })
        );
        assert_eq!(
            meas.integrate(&sample(1_500), &ones, &ones),
            Err(ImuError::NonMonotonicSample {
                previous_t_ns: 1_000,
                t_ns: 500
            })
        );
        // The measurement is unchanged after both refusals.
        assert_eq!(meas.get_dt_ns(), 1_000);

        // A sample at exactly the start time would be a zero-length step.
        let mut fresh: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(1_000, &Vector3::zeros(), &Vector3::zeros());
        assert_eq!(
            fresh.integrate(&sample(1_000), &ones, &ones),
            Err(ImuError::NonMonotonicSample {
                previous_t_ns: 0,
                t_ns: 0
            })
        );
    }

    /// The five ways the shared accumulation loop refuses a malformed interval,
    /// and the closing step that must still happen when a sample does follow.
    ///
    /// `integrate_between` carried these cases and RC9 deleted it with its
    /// tests, leaving the two live producers driving a loop that accepted an
    /// empty interval, an interval starting somewhere else, and an interval it
    /// could not close — the last as `Ok` with a measurement shorter than the
    /// frame gap. Both producers precheck a sample strictly after the frame
    /// ( through `imu_covers_frame`, and
    /// `Vio::track`'s own coverage test), so none of this can fire on the
    /// shipped path; a public method promising to close the interval exactly
    /// must say so anyway (D32).
    #[test]
    fn accumulate_to_rejects_bad_intervals() {
        let noise: ImuNoise<f64> = noise_from_std_dev();
        let sample =
            |t_ns: i64| -> Popped<f64> { (t_ns, Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81)) };
        let meas = || IntegratedImuMeasurement::<f64>::new(0, &Vector3::zeros(), &Vector3::zeros());
        // The queue both producers pop from, as a closure over a list.
        let feed = |samples: Vec<Popped<f64>>| {
            let mut samples = samples.into_iter();
            move || samples.next()
        };

        // An empty interval: asserts it, because a zero time delta
        // "leads to invalid IMU integration".
        assert_eq!(
            meas().accumulate_to(None, feed(vec![sample(1)]), 0, 0, &noise),
            Err(ImuError::NonMonotonicFrames { t0_ns: 0, t1_ns: 0 })
        );
        // An interval that does not start where the measurement was built:
        // every sample would be timed against the wrong origin.
        assert_eq!(
            meas().accumulate_to(None, feed(vec![sample(1)]), 5, 10, &noise),
            Err(ImuError::StartTimeMismatch {
                start_t_ns: 0,
                t0_ns: 5
            })
        );
        // A duplicate and a reordered sample, both from the per-sample step.
        assert_eq!(
            meas().accumulate_to(None, feed(vec![sample(2), sample(2)]), 0, 10, &noise),
            Err(ImuError::NonMonotonicSample {
                previous_t_ns: 2,
                t_ns: 2
            })
        );
        assert_eq!(
            meas().accumulate_to(None, feed(vec![sample(3), sample(1)]), 0, 10, &noise),
            Err(ImuError::NonMonotonicSample {
                previous_t_ns: 3,
                t_ns: 1
            })
        );
        // Nothing after the frame to close the interval with.
        assert_eq!(
            meas().accumulate_to(None, feed(vec![sample(2), sample(4)]), 0, 10, &noise),
            Err(ImuError::MissingSampleAfterFrame { t1_ns: 10 })
        );

        // The sample that does follow closes the interval exactly on the frame
        // and comes back out at its own time.
        let mut closed: IntegratedImuMeasurement<f64> = meas();
        let pending: Option<Popped<f64>> = closed
            .accumulate_to(
                None,
                feed(vec![sample(2), sample(4), sample(12)]),
                0,
                10,
                &noise,
            )
            .unwrap();
        assert_eq!(pending.map(|(t_ns, _, _)| t_ns), Some(12));
        assert_eq!(closed.get_dt_ns(), 10);
    }

    /// `Quaternion::FromTwoVectors(accel, UnitZ)`
    /// rotates the measured specific force onto the world `+Z` axis, so gravity
    /// lands along `-Z`.
    #[test]
    fn gravity_init_aligns_the_accelerometer_with_world_up() {
        let mut rng: Rng = Rng::new(0x5eed_0009);
        for _ in 0..64 {
            let accel: Vector3<f64> = rng.vector3() * 9.81;
            if accel.norm() < 1e-3 {
                continue;
            }
            let rotation: So3<f64> = gravity_from_first_accel(&accel);
            let up: Vector3<f64> = rotation * (accel / accel.norm());
            assert_abs_diff_eq!(up, Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-12);
            // The rig-frame gravity is minus the measured specific force.
            let g_body: Vector3<f64> = rotation.inverse() * gravity::<f64>();
            assert_abs_diff_eq!(g_body.normalize(), -accel.normalize(), epsilon = 1e-12);
        }
    }

    /// Both degenerate directions: a sample already along `+Z` gives the
    /// identity, and the anti-parallel sample still lands on `+Z`.
    #[test]
    fn gravity_init_handles_the_degenerate_directions() {
        let up: So3<f64> = gravity_from_first_accel(&Vector3::new(0.0, 0.0, 9.81));
        assert_abs_diff_eq!(up.log(), Vector3::zeros(), epsilon = 1e-12);

        let down: So3<f64> = gravity_from_first_accel(&Vector3::new(0.0, 0.0, -9.81));
        assert_abs_diff_eq!(
            down * Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 0.0, 1.0),
            epsilon = 1e-12
        );

        // Nothing to align: the identity, not a NaN.
        assert_eq!(
            gravity_from_first_accel(&Vector3::<f64>::zeros()),
            So3::identity()
        );
        assert_eq!(
            gravity_from_first_accel(&Vector3::new(f64::NAN, 0.0, 0.0)),
            So3::identity()
        );
    }

    /// `gravity_from_first_accel` recovers the orientation of a rig at rest for
    /// any roll and pitch, up to the yaw it cannot see.
    #[test]
    fn gravity_init_recovers_roll_and_pitch() {
        let mut rng: Rng = Rng::new(0x5eed_000b);
        let g: Vector3<f64> = gravity::<f64>();
        for _ in 0..64 {
            let truth: So3<f64> = So3::exp(&(rng.vector3() * 1.2));
            // What a rig at rest measures: minus gravity, in the rig frame.
            let accel: Vector3<f64> = truth.inverse() * (-g);
            let estimate: So3<f64> = gravity_from_first_accel(&accel);
            // Both map the measured direction onto world up, so the two differ
            // by a rotation about the world z axis: yaw only.
            let residual: Vector3<f64> = (estimate * truth.inverse()).log();
            assert_abs_diff_eq!(residual.x, 0.0, epsilon = 1e-9);
            assert_abs_diff_eq!(residual.y, 0.0, epsilon = 1e-9);
        }
    }

    /// The same 100-sample integration in `f32` and `f64` agrees to `1e-4`
    /// (decision D05: the measurement is generic and both are instantiated).
    #[test]
    fn f32_agrees_with_f64_over_a_hundred_samples() {
        let mut rng: Rng = Rng::new(0x5eed_000a);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;
        let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
        assert_eq!(samples.len(), 100);

        let wide: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);

        let mut narrow: IntegratedImuMeasurement<f32> = IntegratedImuMeasurement::new(
            0,
            &Vector3::new(bg.x as f32, bg.y as f32, bg.z as f32),
            &Vector3::new(ba.x as f32, ba.y as f32, ba.z as f32),
        );
        let ones: Vector3<f32> = Vector3::repeat(1.0);
        for sample in &samples {
            narrow.integrate(sample, &ones, &ones).unwrap();
        }

        assert_eq!(narrow.get_dt_ns(), wide.get_dt_ns());
        let wide_delta: &PoseVelState<f64> = wide.get_delta_state();
        let narrow_delta: &PoseVelState<f32> = narrow.get_delta_state();
        for axis in 0..3 {
            assert!(
                (f64::from(narrow_delta.t_w_i.translation[axis])
                    - wide_delta.t_w_i.translation[axis])
                    .abs()
                    <= 1e-4
            );
            assert!(
                (f64::from(narrow_delta.vel_w_i[axis]) - wide_delta.vel_w_i[axis]).abs() <= 1e-4
            );
        }
        let narrow_log: Vector3<f32> = narrow_delta.t_w_i.rotation.log();
        let wide_log: Vector3<f64> = wide_delta.t_w_i.rotation.log();
        for axis in 0..3 {
            assert!((f64::from(narrow_log[axis]) - wide_log[axis]).abs() <= 1e-4);
        }
    }

    /// The 9x6 bias Jacobian carries the gyro block in columns 0-2 and the accel
    /// block in 3-5, which is what the estimator's `start_idx + 9` and
    /// `start_idx + 12` column offsets mean.
    #[test]
    fn the_bias_jacobian_columns_are_gyro_then_accel() {
        let mut rng: Rng = Rng::new(0x5eed_000c);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;
        let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
        let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);

        let g: Vector3<f64> = gravity::<f64>();
        let state0: PoseVelState<f64> =
            PoseVelState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));
        let end_t_ns: i64 = meas.get_dt_ns();
        let state1: PoseVelState<f64> = PoseVelState::new(
            end_t_ns,
            trajectory.pose(end_t_ns),
            trajectory.trans_vel_world(end_t_ns),
        );
        let (res, jacobians) = meas.residual_with_jacobians(&state0, &g, &state1, &bg, &ba);

        // The accel half is exactly `-d_state_d_ba`.
        assert_eq!(
            jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 3).into_owned(),
            -meas.get_d_state_d_ba()
        );
        // The gyro half is `-d_state_d_bg` with its rotation rows replaced by
        // `+ leftJacobianInv(res_rot) * d_state_d_bg(3,0)`.
        let gyro_half: Matrix9x3<f64> =
            jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 0).into_owned();
        assert_eq!(
            gyro_half.fixed_view::<3, 3>(0, 0).into_owned(),
            -meas.get_d_state_d_bg().fixed_view::<3, 3>(0, 0)
        );
        assert_eq!(
            gyro_half.fixed_view::<3, 3>(6, 0).into_owned(),
            -meas.get_d_state_d_bg().fixed_view::<3, 3>(6, 0)
        );
        let res_rot: Vector3<f64> = res.fixed_rows::<3>(3).into_owned();
        assert_abs_diff_eq!(
            gyro_half.fixed_view::<3, 3>(3, 0).into_owned(),
            left_jacobian_inv_so3(&res_rot) * meas.get_d_state_d_bg().fixed_view::<3, 3>(3, 0),
            epsilon = 1e-15
        );
    }

    /// An out-of-range block offset is ignored, and the *check itself* does not
    /// overflow: `usize::MAX + 15` panics in debug and wraps to a small,
    /// accepted number in release (decision D32).
    #[test]
    fn the_block_ignores_offsets_that_do_not_fit() {
        let mut rng: Rng = Rng::new(0x5eed_000e);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;
        let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
        let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
        let end_t_ns: i64 = meas.get_dt_ns();

        let state0: PoseVelBiasState<f64> =
            PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
        let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
            end_t_ns,
            trajectory.pose(end_t_ns),
            trajectory.trans_vel_world(end_t_ns),
            bg,
            ba,
        );
        let block: ImuBlock<f64> = ImuBlock::linearize(
            &meas,
            &lin_data(),
            &PoseVelBiasStateWithLin::new(state0, false),
            &PoseVelBiasStateWithLin::new(state1, false),
        );

        let size: usize = 2 * POSE_VEL_BIAS_SIZE;
        for (start_idx, end_idx) in [
            (usize::MAX, 0),
            (0, usize::MAX),
            (usize::MAX, usize::MAX),
            (size, 0),
            (0, size),
        ] {
            let mut h: DMatrix<f64> = DMatrix::zeros(size, size);
            let mut b: DVector<f64> = DVector::zeros(size);
            block.add_dense_h_b(start_idx, end_idx, &mut h, &mut b);
            assert_eq!(h.norm(), 0.0, "({start_idx}, {end_idx}) wrote to H");
            assert_eq!(b.norm(), 0.0, "({start_idx}, {end_idx}) wrote to b");
        }

        // The offsets that do fit still work.
        let mut h: DMatrix<f64> = DMatrix::zeros(size, size);
        let mut b: DVector<f64> = DVector::zeros(size);
        block.add_dense_h_b(0, POSE_VEL_BIAS_SIZE, &mut h, &mut b);
        assert!(h.norm() > 0.0);
    }

    /// The square-root and the squared form of the same factor agree:
    /// `Q2Jpᵀ Q2Jp = H` and `Q2Jpᵀ Q2r = b`.
    ///
    /// `addJp_diag2` must likewise be the squared column norms of
    /// the same scattered Jacobian, and `backSubstitute` the model
    /// cost change of the same `J` and `r`.
    #[test]
    fn the_imu_block_exports_agree_with_each_other() {
        let mut rng: Rng = Rng::new(0x5eed_00e1);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;
        let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
        let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
        let end_t_ns: i64 = meas.get_dt_ns();
        let state0: PoseVelBiasState<f64> =
            PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
        let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
            end_t_ns,
            trajectory.pose(end_t_ns),
            trajectory.trans_vel_world(end_t_ns),
            bg + rng.vector3() / 1000.0,
            ba + rng.vector3() / 1000.0,
        );
        let block: ImuBlock<f64> = ImuBlock::linearize(
            &meas,
            &lin_data(),
            &PoseVelBiasStateWithLin::new(state0, false),
            &PoseVelBiasStateWithLin::new(state1, false),
        );

        // Two states, side by side, plus a third that nothing touches — so the
        // scatter really is checked against the offsets it was given.
        let size: usize = POSE_VEL_BIAS_SIZE;
        let total: usize = 3 * size;
        let (start_idx, end_idx) = (size, 2 * size);

        let mut q2jp: DMatrix<f64> = DMatrix::zeros(size, total);
        let mut q2r: DVector<f64> = DVector::zeros(size);
        block.add_dense_q2jp_q2r(start_idx, end_idx, 0, &mut q2jp, &mut q2r);

        let mut h: DMatrix<f64> = DMatrix::zeros(total, total);
        let mut b: DVector<f64> = DVector::zeros(total);
        block.add_dense_h_b(start_idx, end_idx, &mut h, &mut b);

        let h_sqrt: DMatrix<f64> = q2jp.transpose() * &q2jp;
        let b_sqrt: DVector<f64> = q2jp.transpose() * &q2r;
        assert_abs_diff_eq!(h_sqrt, h, epsilon = 1e-9 * h.norm());
        assert_abs_diff_eq!(b_sqrt, b, epsilon = 1e-9 * b.norm().max(1.0));

        // The first state's block is untouched.
        assert_eq!(q2jp.columns(0, size).norm(), 0.0);

        // `backSubstitute`: the model cost change of the scattered system.
        let inc: DVector<f64> =
            DVector::from_iterator(total, (0..total).map(|_| rng.uniform() / 100.0));
        let mut l_diff: f64 = 0.0;
        block.back_substitute(start_idx, end_idx, &inc, &mut l_diff);
        let jinc: DVector<f64> = &q2jp * &inc;
        let jinc: DVector<f64> = DVector::from_column_slice(jinc.as_slice());
        let want: f64 = -(jinc.transpose() * (0.5 * &jinc + &q2r))[(0, 0)];
        assert_abs_diff_eq!(l_diff, want, epsilon = 1e-12 * want.abs().max(1.0));

        // Offsets that do not fit are ignored, not a panic (decision D32).
        let mut l_diff: f64 = 0.0;
        block.back_substitute(0, total, &inc, &mut l_diff);
        assert_eq!(l_diff, 0.0);
    }

    /// A frozen linearization point changes only the residual *value*, not the
    /// Jacobians (trap 7 of the architecture dossier).
    #[test]
    fn the_block_re_evaluates_the_residual_at_a_linearized_state() {
        let mut rng: Rng = Rng::new(0x5eed_000d);
        let trajectory: Trajectory = Trajectory::new(&mut rng);
        let bg: Vector3<f64> = rng.vector3() / 100.0;
        let ba: Vector3<f64> = rng.vector3() / 10.0;
        let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
        let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
        let end_t_ns: i64 = meas.get_dt_ns();

        let state0: PoseVelBiasState<f64> =
            PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
        let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
            end_t_ns,
            trajectory.pose(end_t_ns),
            trajectory.trans_vel_world(end_t_ns),
            bg,
            ba,
        );
        let ild: ImuLinData<f64> = lin_data();

        let plain0: PoseVelBiasStateWithLin<f64> = PoseVelBiasStateWithLin::new(state0, false);
        let plain1: PoseVelBiasStateWithLin<f64> = PoseVelBiasStateWithLin::new(state1, false);
        let reference: ImuBlock<f64> = ImuBlock::linearize(&meas, &ild, &plain0, &plain1);

        let mut frozen0: PoseVelBiasStateWithLin<f64> = PoseVelBiasStateWithLin::new(state0, false);
        frozen0.set_linearized().unwrap();
        let mut inc: Vector15<f64> = Vector15::zeros();
        inc.fixed_rows_mut::<3>(0).copy_from(&Vector3::repeat(0.01));
        frozen0.apply_inc(&inc);

        let frozen: ImuBlock<f64> = ImuBlock::linearize(&meas, &ild, &frozen0, &plain1);
        // Same linearization point, so the Jacobian is untouched...
        assert_eq!(frozen.jp, reference.jp);
        // but the residual moved with the state.
        assert!((frozen.r - reference.r).norm() > 1e-6);
    }

    // ── Properties ──────────────────────────────────────────────────────

    fn zero_motion_measurement(
        steps: usize,
        dt_ns: i64,
        accel: Vector3<f64>,
    ) -> IntegratedImuMeasurement<f64> {
        let noise: ImuNoise<f64> = noise_from_std_dev();
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
        for step in 1..=steps as i64 {
            meas.integrate(
                &ImuSample {
                    t_ns: step * dt_ns,
                    gyro: Vector3::zeros(),
                    accel,
                },
                &noise.accel_cov,
                &noise.gyro_cov,
            )
            .unwrap();
        }
        meas
    }

    proptest! {
        /// Gravity-only accelerometer, zero gyro, zero bias: the delta state has
        /// no rotation, its velocity is `-g T` and its position `-½ g T²`, so
        /// `predict_state` leaves a rig that started at rest exactly where it
        /// was, and the residual against that prediction vanishes.
        #[test]
        fn a_rig_at_rest_stays_at_rest(steps in 1usize..40, dt_ns in 1_000_000i64..20_000_000) {
            let g: Vector3<f64> = gravity::<f64>();
            let meas: IntegratedImuMeasurement<f64> = zero_motion_measurement(steps, dt_ns, -g);
            let total: f64 = meas.get_dt_ns() as f64 * 1e-9;

            let delta: &PoseVelState<f64> = meas.get_delta_state();
            prop_assert!(delta.t_w_i.rotation.log().norm() <= 1e-15);
            prop_assert!((delta.vel_w_i - (-g * total)).norm() <= 1e-9);
            prop_assert!((delta.t_w_i.translation - (-g * 0.5 * total * total)).norm() <= 1e-9);

            let state0: PoseVelState<f64> = PoseVelState::default();
            let state1: PoseVelState<f64> = meas.predict_state(&state0, &g);
            prop_assert!(state1.t_w_i.translation.norm() <= 1e-9);
            prop_assert!(state1.vel_w_i.norm() <= 1e-9);
            prop_assert!(state1.t_w_i.rotation.log().norm() <= 1e-15);
            prop_assert_eq!(state1.t_ns, meas.get_dt_ns());

            let res = meas.residual(&state0, &g, &state1, &Vector3::zeros(), &Vector3::zeros());
            prop_assert!(res.amax() <= 1e-9);
        }

        /// A free-falling rig — zero specific force — accumulates nothing, and
        /// the prediction is pure gravity: `v = g T`, `p = ½ g T²`.
        #[test]
        fn free_fall_is_pure_gravity(steps in 1usize..40, dt_ns in 1_000_000i64..20_000_000) {
            let g: Vector3<f64> = gravity::<f64>();
            let meas: IntegratedImuMeasurement<f64> =
                zero_motion_measurement(steps, dt_ns, Vector3::zeros());
            let total: f64 = meas.get_dt_ns() as f64 * 1e-9;

            let delta: &PoseVelState<f64> = meas.get_delta_state();
            prop_assert_eq!(delta.vel_w_i, Vector3::zeros());
            prop_assert_eq!(delta.t_w_i.translation, Vector3::zeros());

            let state1: PoseVelState<f64> = meas.predict_state(&PoseVelState::default(), &g);
            prop_assert!((state1.vel_w_i - g * total).norm() <= 1e-12);
            prop_assert!((state1.t_w_i.translation - g * 0.5 * total * total).norm() <= 1e-12);
        }

        #[test]
        fn random_spd_covariance_is_inverted(values in prop::collection::vec(-1.0f64..1.0, 81)) {
            let g = Matrix9::from_row_slice(&values);
            let a = g.transpose() * g + Matrix9::identity();
            let mut meas = IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
            meas.cov = a;
            let inverse = meas.get_cov_inv();
            let reference = a.try_inverse().unwrap();
            prop_assert!((inverse - reference).norm() < 1e-10 * (1.0 + reference.norm()));
        }

        #[test]
        fn rank_deficient_gram_covariance_has_a_generalized_inverse(
            weights in prop::array::uniform4(0.25f64..4.0), shift in 0usize..9,
        ) {
            // Four independent rows and scaled duplicate columns. This tests
            // oblique null directions without rounding GᵀG into full rank.
            let mut g = Matrix9::zeros();
            for i in 0..4 {
                g[(i, i)] = weights[i];
                g[(i, i + 4)] = 2.0 * weights[i];
            }
            g.swap_columns(0, shift);
            let a = g.transpose() * g;
            let mut meas = IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
            meas.cov = a;
            let inverse = meas.get_cov_inv();
            prop_assert!(inverse.iter().all(|v| v.is_finite()));
            prop_assert!((inverse - inverse.transpose()).norm() < 1e-12 * (1.0 + inverse.norm()));
            prop_assert!((a * inverse * a - a).norm() < 1e-10 * (1.0 + a.norm()));
        }

        /// The covariance stays symmetric and positive semi-definite, and the
        /// square-root inverse really is one: `(MᵀM) cov = I`.
        #[test]
        fn the_covariance_is_symmetric_psd_and_its_factor_inverts_it(
            steps in 2usize..25,
            dt_ns in 1_000_000i64..10_000_000,
            gyro_scale in 0.0f64..1.0,
        ) {
            let noise: ImuNoise<f64> = noise_from_std_dev();
            let mut meas: IntegratedImuMeasurement<f64> =
                IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
            for step in 1..=steps as i64 {
                meas.integrate(
                    &ImuSample {
                        t_ns: step * dt_ns,
                        gyro: Vector3::new(0.3, -0.2, 0.5) * gyro_scale,
                        accel: Vector3::new(0.4, -0.3, 9.7),
                    },
                    &noise.accel_cov,
                    &noise.gyro_cov,
                )?;
            }

            let cov: Matrix9<f64> = *meas.get_cov();
            let asymmetry: f64 = (cov - cov.transpose()).amax();
            prop_assert!(asymmetry <= 1e-12 * cov.amax().max(1.0), "asymmetry {}", asymmetry);

            let smallest: f64 = cov.symmetric_eigenvalues().min();
            prop_assert!(smallest >= -1e-9 * cov.amax(), "smallest eigenvalue {}", smallest);

            let m: Matrix9<f64> = meas.get_cov_inv_sqrt();
            let deviation: f64 = (m.transpose() * m * cov - Matrix9::identity()).amax();
            prop_assert!(deviation <= 1e-6, "MᵀM cov - I = {}", deviation);
        }

        /// A duplicate or backwards timestamp is always refused, whatever the
        /// start time, and the measurement does not move.
        #[test]
        fn out_of_order_samples_are_always_refused(
            start_t_ns in -1_000_000_000i64..1_000_000_000,
            step_ns in 1i64..10_000_000,
            back_ns in 0i64..10_000_000,
        ) {
            let ones: Vector3<f64> = Vector3::repeat(1.0);
            let mut meas: IntegratedImuMeasurement<f64> =
                IntegratedImuMeasurement::new(start_t_ns, &Vector3::zeros(), &Vector3::zeros());
            let sample = |t_ns: i64| ImuSample {
                t_ns,
                gyro: Vector3::new(0.02, 0.0, -0.01),
                accel: Vector3::new(0.0, 0.0, 9.81),
            };
            meas.integrate(&sample(start_t_ns + step_ns), &ones, &ones)?;
            let before: IntegratedImuMeasurement<f64> = meas;

            let result = meas.integrate(&sample(start_t_ns + step_ns - back_ns), &ones, &ones);
            prop_assert_eq!(
                result,
                Err(ImuError::NonMonotonicSample {
                    previous_t_ns: step_ns,
                    t_ns: step_ns - back_ns
                })
            );
            prop_assert_eq!(meas, before);
        }
    }
}
