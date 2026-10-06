//! IMU integration, prediction, bias Jacobians and covariance whitening.

use super::{CombinedImuSample, NavState};
use kornia_staging_algebra::lie::Rotation3;
use nalgebra::{SMatrix, SVector};
type Matrix9<S> = SMatrix<S, 9, 9>;
type Matrix9x3<S> = SMatrix<S, 9, 3>;
type Matrix9x6<S> = SMatrix<S, 9, 6>;
type Vector9<S> = SVector<S, 9>;
const POSE_VEL_SIZE: usize = 9;
use kornia_staging_algebra::lie::{
    left_jacobian_inv_so3, right_jacobian_inv_so3, right_jacobian_so3,
};
use kornia_staging_algebra::linalg::ldlt::ldlt_in_place;
use kornia_staging_algebra::Scalar;
use nalgebra::{Matrix3, Vector3};

// Keep result validation out of the large propagation kernel to limit register pressure.
#[inline(never)]
fn all_finite<S: Scalar>(
    state: &NavState<S>,
    a: &Matrix9<S>,
    b: &Matrix9x3<S>,
    c: &Matrix9x3<S>,
) -> bool {
    if !state
        .t_w_i
        .translation
        .as_slice()
        .iter()
        .chain(state.vel_w_i.as_slice())
        .chain(state.t_w_i.rotation.quaternion().coords.as_slice())
        .all(|value| value.is_finite())
    {
        return false;
    }
    a.as_slice()
        .iter()
        .chain(b.as_slice())
        .chain(c.as_slice())
        .fold(true, |finite, value| finite & value.is_finite())
}

use crate::SensorError;

/// Validated discrete diagonal IMU noise covariances.
/// Construct once from calibration and reuse for each sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuNoise<S: Scalar> {
    accel_cov: Vector3<S>,
    gyro_cov: Vector3<S>,
}

impl<S: Scalar> ImuNoise<S> {
    /// Validate finite, nonnegative diagonal covariances.
    ///
    /// # Errors
    /// Returns an invalid configuration error for a nonfinite or negative entry.
    pub fn new(accel_cov: Vector3<S>, gyro_cov: Vector3<S>) -> Result<Self, SensorError> {
        if !accel_cov
            .iter()
            .chain(gyro_cov.iter())
            .all(|v| v.is_finite() && *v >= S::zero())
        {
            return Err(SensorError::InvalidNoise);
        }
        Ok(Self {
            accel_cov,
            gyro_cov,
        })
    }
}

/// The three Jacobians of one propagation step
/// (`F`, `A` and `G`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropagationJacobians<S: Scalar> {
    /// `F = ∂next/∂curr`, `J_f^s`.
    pub d_next_d_curr: Matrix9<S>,
    /// `A = ∂next/∂accel`, `J_f^a`.
    pub d_next_d_accel: Matrix9x3<S>,
    /// `G = ∂next/∂gyro`, `J_f^g`.
    pub d_next_d_gyro: Matrix9x3<S>,
}

/// The residual's Jacobians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuResidualJacobians<S: Scalar> {
    /// `∂r/∂state0`.
    pub d_res_d_state0: Matrix9<S>,
    /// `∂r/∂state1`.
    pub d_res_d_state1: Matrix9<S>,
    /// Bias Jacobian: gyro bias in columns 0–2, accelerometer bias in columns 3–5.
    /// These occupy offsets 9 and 12 of the full `[t, R, v, b_g, b_a]` state.
    pub d_res_d_bias: Matrix9x6<S>,
}

/// Residual and shared intermediates for its state Jacobians.
struct ResidualParts<S: Scalar> {
    residual: Vector9<S>,
    inverse_rotation: Matrix3<S>,
    position_delta: Vector3<S>,
    velocity_delta: Vector3<S>,
    dt: S,
}

/// A pseudo-measurement from consecutive IMU samples.
/// Delta time is elapsed nanoseconds from [`Self::start_timestamp_ns`], not absolute time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegratedImuMeasurement<S: Scalar> {
    start_timestamp_ns: i64,
    delta_state: NavState<S>,
    cov: Matrix9<S>,
    d_state_d_ba: Matrix9x3<S>,
    d_state_d_bg: Matrix9x3<S>,
    bias_gyro_lin: Vector3<S>,
    bias_accel_lin: Vector3<S>,
}

impl<S: Scalar> Default for IntegratedImuMeasurement<S> {
    /// everything zero, start time zero.
    fn default() -> Self {
        Self::new(0, &Vector3::zeros(), &Vector3::zeros())
    }
}

impl<S: Scalar> IntegratedImuMeasurement<S> {
    /// An empty measurement starting at `start_timestamp_ns`, linearized about the two
    /// biases.
    /// ```
    /// use kornia_staging_sensors::imu::{CombinedImuSample, IntegratedImuMeasurement};
    /// use nalgebra::Vector3;
    /// let mut measurement = IntegratedImuMeasurement::<f64>::new(10, &Vector3::zeros(), &Vector3::zeros());
    /// measurement.integrate(&CombinedImuSample {
    ///     timestamp_ns: 100_000_010, accel: [0.0, 0.0, 2.0].into(), gyro: [0.0; 3].into(),
    /// }, &kornia_staging_sensors::imu::ImuNoise::new(Vector3::zeros(), Vector3::zeros()).unwrap())?;
    /// assert!((measurement.delta_state().vel_w_i.z - 0.2).abs() < 1e-12);
    /// assert!((measurement.delta_state().t_w_i.translation.z - 0.01).abs() < 1e-12);
    /// # Ok::<(), kornia_staging_sensors::SensorError>(())
    /// ```
    pub fn new(start_timestamp_ns: i64, bias_gyro: &Vector3<S>, bias_accel: &Vector3<S>) -> Self {
        Self {
            start_timestamp_ns,
            delta_state: NavState::default(),
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
        curr_state: &NavState<S>,
        timestamp_ns: i64,
        accel: &Vector3<S>,
        gyro: &Vector3<S>,
    ) -> Result<(NavState<S>, PropagationJacobians<S>), SensorError> {
        if !accel
            .as_slice()
            .iter()
            .chain(gyro.as_slice())
            .chain(curr_state.t_w_i.translation.as_slice().iter())
            .chain(curr_state.vel_w_i.as_slice().iter())
            .chain(
                curr_state
                    .t_w_i
                    .rotation
                    .quaternion()
                    .coords
                    .as_slice()
                    .iter(),
            )
            .all(|v| v.is_finite())
        {
            return Err(SensorError::NonFiniteInput);
        }
        let (next_state, j) = Self::propagate_step(curr_state, timestamp_ns, accel, gyro)?;
        let finite = all_finite(
            &next_state,
            &j.d_next_d_curr,
            &j.d_next_d_accel,
            &j.d_next_d_gyro,
        );
        if !finite {
            return Err(SensorError::NonFiniteResult);
        }
        Ok((next_state, j))
    }

    // The caller owns finite-value validation. Integration validates the final
    // covariance and bias Jacobians, which also expose non-finite step Jacobians.
    fn propagate_step(
        curr_state: &NavState<S>,
        timestamp_ns: i64,
        accel: &Vector3<S>,
        gyro: &Vector3<S>,
    ) -> Result<(NavState<S>, PropagationJacobians<S>), SensorError> {
        // Refuse duplicate or reordered samples before they create a singular measurement.
        if timestamp_ns <= curr_state.timestamp_ns {
            return Err(SensorError::NonMonotonicSample {
                previous_timestamp_ns: curr_state.timestamp_ns,
                timestamp_ns,
            });
        }
        let dt_ns: i64 = timestamp_ns.checked_sub(curr_state.timestamp_ns).ok_or(
            SensorError::TimestampOverflow {
                a_ns: timestamp_ns,
                b_ns: curr_state.timestamp_ns,
            },
        )?;
        let dt: S = S::from_literal(dt_ns as f64) * S::from_literal(1e-9);

        // Deviation 1: the acceleration is rotated by the *midpoint* rotation.
        let r_w_i_new_2: Rotation3<S> =
            curr_state.t_w_i.rotation * Rotation3::exp(&(*gyro * (S::from_literal(0.5) * dt)));
        let rr_w_i_new_2: Matrix3<S> = r_w_i_new_2.matrix();
        let accel_world: Vector3<S> = rr_w_i_new_2 * accel;

        let mut next_state: NavState<S> = NavState {
            timestamp_ns,
            t_w_i: curr_state.t_w_i,
            vel_w_i: curr_state.vel_w_i + accel_world * dt,
        };
        next_state.t_w_i.rotation = curr_state.t_w_i.rotation * Rotation3::exp(&(*gyro * dt));
        // deviation 2: the quadratic term stays.
        next_state.t_w_i.translation = curr_state.t_w_i.translation
            + curr_state.vel_w_i * dt
            + accel_world * S::from_literal(0.5) * dt * dt;

        // Only the *diagonal* of the (0,6) block is set, so the port
        // writes three entries rather than a scaled identity.
        let mut d_next_d_curr: Matrix9<S> = Matrix9::identity();
        for i in 0..3 {
            d_next_d_curr[(i, 6 + i)] = dt;
        }
        let hat_accel_dt: Matrix3<S> = Rotation3::hat(&(-accel_world * dt));
        d_next_d_curr
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&hat_accel_dt);
        d_next_d_curr
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(hat_accel_dt * dt * S::from_literal(0.5)));

        let mut d_next_d_accel: Matrix9x3<S> = Matrix9x3::zeros();
        d_next_d_accel
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(rr_w_i_new_2 * S::from_literal(0.5) * dt * dt));
        d_next_d_accel
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(rr_w_i_new_2 * dt));

        let mut d_next_d_gyro: Matrix9x3<S> = Matrix9x3::zeros();
        let jr: Matrix3<S> = right_jacobian_so3(&(*gyro * dt));
        let jr2: Matrix3<S> = right_jacobian_so3(&(*gyro * (S::from_literal(0.5) * dt)));
        d_next_d_gyro
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(next_state.t_w_i.rotation.matrix() * jr * dt));
        let d_vel_d_gyro: Matrix3<S> =
            hat_accel_dt * rr_w_i_new_2 * jr2 * S::from_literal(0.5) * dt;
        d_next_d_gyro
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&d_vel_d_gyro);
        d_next_d_gyro
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(d_vel_d_gyro * (S::from_literal(0.5) * dt)));

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
        data: &CombinedImuSample,
        noise: &ImuNoise<S>,
    ) -> Result<(), SensorError> {
        self.integrate_calibrated(
            data.timestamp_ns,
            &Vector3::new(
                S::from_literal(data.accel.x),
                S::from_literal(data.accel.y),
                S::from_literal(data.accel.z),
            ),
            &Vector3::new(
                S::from_literal(data.gyro.x),
                S::from_literal(data.gyro.y),
                S::from_literal(data.gyro.z),
            ),
            noise,
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
    /// [`SensorError`] as [`Self::integrate`].
    #[allow(clippy::op_ref)] // Reuse each Jacobian across covariance and bias products.
    pub fn integrate_calibrated(
        &mut self,
        sample_timestamp_ns: i64,
        accel: &Vector3<S>,
        gyro: &Vector3<S>,
        noise: &ImuNoise<S>,
    ) -> Result<(), SensorError> {
        // relative time, bias removed at the linearization point.
        let timestamp_ns: i64 = sample_timestamp_ns
            .checked_sub(self.start_timestamp_ns)
            .ok_or(SensorError::TimestampOverflow {
                a_ns: sample_timestamp_ns,
                b_ns: self.start_timestamp_ns,
            })?;
        let accel: Vector3<S> = accel - self.bias_accel_lin;
        let gyro: Vector3<S> = gyro - self.bias_gyro_lin;

        if !accel
            .as_slice()
            .iter()
            .chain(gyro.as_slice())
            .all(|v| v.is_finite())
        {
            return Err(SensorError::NonFiniteInput);
        }
        // delta_state is private and only committed after the result checks below.
        let (new_state, j): (NavState<S>, PropagationJacobians<S>) =
            Self::propagate_step(&self.delta_state, timestamp_ns, &accel, &gyro)?;

        // Propagate covariance and add accelerometer and gyroscope noise.
        let cov = &j.d_next_d_curr * &self.cov * j.d_next_d_curr.transpose()
            + &j.d_next_d_accel
                * Matrix3::from_diagonal(&noise.accel_cov)
                * j.d_next_d_accel.transpose()
            + &j.d_next_d_gyro
                * Matrix3::from_diagonal(&noise.gyro_cov)
                * j.d_next_d_gyro.transpose();
        // Bias perturbations propagate through the same state transition.
        let d_state_d_ba = -j.d_next_d_accel + &j.d_next_d_curr * &self.d_state_d_ba;
        let d_state_d_bg = -j.d_next_d_gyro + &j.d_next_d_curr * &self.d_state_d_bg;
        let finite = all_finite(&new_state, &cov, &d_state_d_ba, &d_state_d_bg);
        if !finite {
            return Err(SensorError::NonFiniteResult);
        }
        self.delta_state = new_state;
        self.cov = cov;
        self.d_state_d_ba = d_state_d_ba;
        self.d_state_d_bg = d_state_d_bg;
        Ok(())
    }

    /// Predict the state at the end of the interval.
    ///
    /// `state0` must describe the state at [`Self::start_timestamp_ns`].
    /// This method applies the full integrated motion and adds the elapsed
    /// nanoseconds to `state0.timestamp_ns` with saturation at the `i64` limits.
    pub fn predict_state(&self, state0: &NavState<S>, g: &Vector3<S>) -> NavState<S> {
        let dt: S = S::from_literal(self.delta_state.timestamp_ns as f64) * S::from_literal(1e-9);
        let mut state1: NavState<S> = NavState {
            // Advance the predicted timestamp by the integrated duration.
            timestamp_ns: state0
                .timestamp_ns
                .saturating_add(self.delta_state.timestamp_ns),
            t_w_i: state0.t_w_i,
            vel_w_i: state0.vel_w_i + *g * dt + state0.t_w_i.rotation * self.delta_state.vel_w_i,
        };
        state1.t_w_i.rotation = state0.t_w_i.rotation * self.delta_state.t_w_i.rotation;
        state1.t_w_i.translation = state0.t_w_i.translation
            + state0.vel_w_i * dt
            + *g * S::from_literal(0.5) * dt * dt
            + state0.t_w_i.rotation * self.delta_state.t_w_i.translation;
        state1
    }

    /// The 9-vector residual between two states.
    ///
    /// Ordered `[translation(3), rotation(3), velocity(3)]` (deviation 3).
    pub fn residual(
        &self,
        state0: &NavState<S>,
        g: &Vector3<S>,
        state1: &NavState<S>,
        curr_bg: &Vector3<S>,
        curr_ba: &Vector3<S>,
    ) -> Vector9<S> {
        self.residual_parts(state0, g, state1, curr_bg, curr_ba)
            .residual
    }

    /// [`Self::residual`] plus the three Jacobians.
    pub fn residual_with_jacobians(
        &self,
        state0: &NavState<S>,
        g: &Vector3<S>,
        state1: &NavState<S>,
        curr_bg: &Vector3<S>,
        curr_ba: &Vector3<S>,
    ) -> (Vector9<S>, ImuResidualJacobians<S>) {
        let ResidualParts {
            residual: res,
            inverse_rotation: r0_inv,
            position_delta: tmp,
            velocity_delta: tmp2,
            dt,
        } = self.residual_parts(state0, g, state1, curr_bg, curr_ba);

        let res_rot: Vector3<S> = res.fixed_rows::<3>(3).into_owned();
        let j: Matrix3<S> = right_jacobian_inv_so3(&res_rot);

        let mut d_res_d_state0: Matrix9<S> = Matrix9::zeros();
        d_res_d_state0
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&-r0_inv);
        d_res_d_state0
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(Rotation3::hat(&tmp) * r0_inv));
        d_res_d_state0
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(j * r0_inv));
        d_res_d_state0
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(Rotation3::hat(&tmp2) * r0_inv));
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
        let j_left: Matrix3<S> = left_jacobian_inv_so3(&res_rot);
        d_res_d_bias
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(j_left * self.d_state_d_bg.fixed_view::<3, 3>(3, 0)));
        d_res_d_bias
            .fixed_view_mut::<9, 3>(0, 3)
            .copy_from(&-self.d_state_d_ba);

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
        state0: &NavState<S>,
        g: &Vector3<S>,
        state1: &NavState<S>,
        curr_bg: &Vector3<S>,
        curr_ba: &Vector3<S>,
    ) -> ResidualParts<S> {
        let dt: S = S::from_literal(self.delta_state.timestamp_ns as f64) * S::from_literal(1e-9);

        // Correct the preintegrated state for the current bias displacement.
        let bg_diff: Vector9<S> = self.d_state_d_bg * (curr_bg - self.bias_gyro_lin);
        let ba_diff: Vector9<S> = self.d_state_d_ba * (curr_ba - self.bias_accel_lin);

        let r0_inv: Matrix3<S> = state0.t_w_i.rotation.inverse().matrix();
        // Subtract initial velocity and gravity before rotating the position residual.
        let tmp: Vector3<S> = r0_inv
            * (state1.t_w_i.translation
                - state0.t_w_i.translation
                - state0.vel_w_i * dt
                - *g * S::from_literal(0.5) * dt * dt);
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
            &(Rotation3::exp(&bg_diff.fixed_rows::<3>(3).into_owned())
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

        ResidualParts {
            residual: res,
            inverse_rotation: r0_inv,
            position_delta: tmp,
            velocity_delta: tmp2,
            dt,
        }
    }

    /// Elapsed time of the measurement, in nanoseconds.
    pub fn dt_ns(&self) -> i64 {
        self.delta_state.timestamp_ns
    }

    /// Start time of the measurement, in nanoseconds.
    pub fn start_timestamp_ns(&self) -> i64 {
        self.start_timestamp_ns
    }

    /// The preintegrated delta state.
    pub fn delta_state(&self) -> &NavState<S> {
        &self.delta_state
    }

    /// The measurement covariance.
    pub fn cov(&self) -> &Matrix9<S> {
        &self.cov
    }

    /// Jacobian of the delta state with respect to the accelerometer bias
    pub fn d_state_d_bias_accel(&self) -> &Matrix9x3<S> {
        &self.d_state_d_ba
    }

    /// Jacobian of the delta state with respect to the gyroscope bias
    pub fn d_state_d_bias_gyro(&self) -> &Matrix9x3<S> {
        &self.d_state_d_bg
    }

    /// Whitening factor `M = D⁺½ L⁻¹ P` for `P cov Pᵀ = L D Lᵀ`.
    /// For SPD covariance, `Mᵀ M = cov⁻¹`. Rows with a pivot below the smallest
    /// positive normal value are zeroed so degenerate covariance contributes
    /// finite, zero weight in those factor coordinates. The singular result is
    /// a generalized inverse through congruence, not a Moore-Penrose inverse.
    pub fn cov_inv_sqrt(&self) -> Matrix9<S> {
        let mut mat: Matrix9<S> = self.cov;
        let transpositions: [usize; POSE_VEL_SIZE] = ldlt_in_place(&mut mat.data.0);

        // Apply the pivot permutation to the identity before solving with L.
        let mut m: Matrix9<S> = Matrix9::identity();
        kornia_staging_algebra::linalg::ldlt::ldlt_forward_in_place(
            &mat.data.0,
            &transpositions,
            &mut m.data.0,
        );
        // The comparison is against
        // `std::numeric_limits<Scalar>::min()`, so a *negative* pivot — what a
        // rank-deficient covariance actually produces, see [`ldlt_in_place`] —
        // zeroes its row rather than taking the square root of a negative.
        for i in 0..POSE_VEL_SIZE {
            let scale: S = if mat[(i, i)] < S::MIN_POSITIVE_NORMAL {
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
    pub fn cov_inv(&self) -> Matrix9<S> {
        let m: Matrix9<S> = self.cov_inv_sqrt();
        m.transpose() * m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    #[test]
    fn noise_is_checked_before_integration() {
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            assert_eq!(
                ImuNoise::new(Vector3::repeat(bad), Vector3::zeros()),
                Err(SensorError::InvalidNoise)
            );
            assert_eq!(
                ImuNoise::new(Vector3::zeros(), Vector3::repeat(bad)),
                Err(SensorError::InvalidNoise)
            );
        }
        assert!(ImuNoise::new(Vector3::<f64>::zeros(), Vector3::zeros()).is_ok());
    }

    #[test]
    fn finite_scan_covers_every_lane_and_remainder() {
        fn check<S: Scalar>(max: S, subnormal: S, signaling_nan: S) {
            let finite = [
                S::zero(),
                -S::zero(),
                S::one(),
                -S::one(),
                max,
                -max,
                subnormal,
                -subnormal,
            ];
            let mut values: Vec<S> = finite.into_iter().cycle().take(145).collect();
            let check_values = |values: &[S], expected: bool| {
                let mut state = NavState::default();
                state.t_w_i.translation = Vector3::from_column_slice(&values[..3]);
                state.vel_w_i = Vector3::from_column_slice(&values[3..6]);
                // Validation inspects coefficients, without performing rotation arithmetic.
                state.t_w_i.rotation =
                    Rotation3::from_kornia_quaternion(values[6..10].try_into().unwrap());
                let a = Matrix9::from_column_slice(&values[10..91]);
                let b = Matrix9x3::from_column_slice(&values[91..118]);
                let c = Matrix9x3::from_column_slice(&values[118..]);
                assert_eq!(all_finite(&state, &a, &b, &c), expected);
            };
            check_values(&values, true);
            for i in 0..values.len() {
                let original = values[i];
                for invalid in [
                    S::from_literal(f64::NAN),
                    S::from_literal(f64::INFINITY),
                    S::from_literal(f64::NEG_INFINITY),
                    signaling_nan,
                    -signaling_nan,
                ] {
                    values[i] = invalid;
                    check_values(&values, false);
                }
                values[i] = original;
            }
        }
        check(f32::MAX, f32::from_bits(1), f32::from_bits(0x7f80_0001));
        check(
            f64::MAX,
            f64::from_bits(1),
            f64::from_bits(0x7ff0_0000_0000_0001),
        );
    }

    #[test]
    fn zero_noise_does_not_hide_overflow_in_step_jacobians() {
        fn check<S: Scalar>(max: S) {
            let accel = Vector3::repeat(max / S::from_literal(8.0));
            let zero = Vector3::zeros();
            let (state, j) = IntegratedImuMeasurement::<S>::propagate_step(
                &NavState::default(),
                4_000_000_000,
                &accel,
                &zero,
            )
            .unwrap();
            assert!(state
                .t_w_i
                .translation
                .iter()
                .chain(state.vel_w_i.iter())
                .all(|v| v.is_finite()));
            assert!(!j.d_next_d_curr.iter().all(|v| v.is_finite()));
            assert_eq!(
                IntegratedImuMeasurement::<S>::propagate_state(
                    &NavState::default(),
                    4_000_000_000,
                    &accel,
                    &zero,
                ),
                Err(SensorError::NonFiniteResult)
            );
            let mut measurement = IntegratedImuMeasurement::<S>::default();
            let before = measurement;
            assert_eq!(
                measurement.integrate_calibrated(
                    4_000_000_000,
                    &accel,
                    &zero,
                    &crate::imu::ImuNoise::new(zero, zero).unwrap()
                ),
                Err(SensorError::NonFiniteResult)
            );
            assert_eq!(measurement, before);
        }
        check(f32::MAX);
        check(f64::MAX);
    }

    #[test]
    fn invalid_propagation_and_finite_overflow_leave_state_unchanged() {
        assert!(matches!(
            IntegratedImuMeasurement::<f64>::propagate_state(
                &NavState::default(),
                1,
                &Vector3::repeat(f64::NAN),
                &Vector3::zeros(),
            ),
            Err(SensorError::NonFiniteInput)
        ));
        let mut measurement =
            IntegratedImuMeasurement::<f64>::new(0, &Vector3::zeros(), &Vector3::zeros());
        let before = measurement;
        let result = measurement.integrate(
            &CombinedImuSample {
                timestamp_ns: 2_000_000_000,
                accel: [f64::MAX; 3].into(),
                gyro: [0.0; 3].into(),
            },
            &crate::imu::ImuNoise::new(Vector3::repeat(1.0), Vector3::repeat(1.0)).unwrap(),
        );
        assert_eq!(result, Err(SensorError::NonFiniteResult));
        assert_eq!(measurement, before);
    }

    proptest! {
        #[test]
        fn random_spd_covariance_is_inverted(values in prop::collection::vec(-1.0f64..1.0, 81)) {
            let g = Matrix9::from_row_slice(&values);
            let a = g.transpose() * g + Matrix9::identity();
            let mut meas = IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
            meas.cov = a;
            let inverse = meas.cov_inv();
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
            let inverse = meas.cov_inv();
            prop_assert!(inverse.iter().all(|v| v.is_finite()));
            prop_assert!((inverse - inverse.transpose()).norm() < 1e-12 * (1.0 + inverse.norm()));
            prop_assert!((a * inverse * a - a).norm() < 1e-10 * (1.0 + a.norm()));
        }

    }
}

#[cfg(test)]
mod numerical_tests;
#[cfg(test)]
mod properties;
#[cfg(test)]
mod test_support;
