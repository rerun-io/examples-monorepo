//! IMU integration, prediction, bias Jacobians and covariance whitening.

use super::{Matrix9, Matrix9x3, Matrix9x6};
use crate::calib::Calibration;
use crate::ldlt::ldlt_in_place;
use crate::lie::{
    LieScalar, So3, c, left_jacobian_inv_so3, right_jacobian_inv_so3, right_jacobian_so3,
};
use crate::types::{POSE_VEL_SIZE, PoseVelState, Vector9};
use nalgebra::{Matrix3, Vector3};

/// A `Vector3<f64>` in the measurement's scalar type.
#[inline]
fn cast3<S: LieScalar>(v: &Vector3<f64>) -> Vector3<S> {
    Vector3::new(c::<S>(v.x), c::<S>(v.y), c::<S>(v.z))
}

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

#[cfg(test)]
mod tests;
