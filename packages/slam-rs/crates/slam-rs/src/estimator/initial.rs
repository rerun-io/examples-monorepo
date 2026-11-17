//! A first state solved outside the window: the inertial bootstrap.
//!
//! The stereo start takes gravity from one accelerometer sample, sets the
//! velocity to zero and needs landmarks triangulated inside the first
//! frameset. A rig without stereo overlap that starts moving has none of
//! these; the inertial bootstrap (`kornia_staging_slam::init::inertial`)
//! solves the first state from a short window instead, and the estimator
//! starts from it here. Nothing else changes: the same init block runs on
//! the first frameset, without the stereo-landmark gate, and the first
//! landmarks come from temporal triangulation at the next keyframe.
use super::{EstimatorError, SqrtKeypointVio};
use crate::calib::Calibration;
use crate::config::VioConfig;
use crate::lie::Se3;
use kornia_staging_algebra::Scalar;
use nalgebra::Vector3;

/// The solved state at the first frameset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialState<S: Scalar> {
    /// The first frameset's timestamp; the first `process_frame` must carry it.
    pub t_ns: i64,
    /// World-from-IMU pose: gravity-aligned rotation (world z up), zero translation.
    pub t_w_i: Se3<S>,
    /// IMU velocity in the world frame, m/s.
    pub vel_w_i: Vector3<S>,
    /// Gyroscope bias, rad/s, on top of the static calibration.
    pub bias_gyro: Vector3<S>,
    /// Accelerometer bias, m/s^2, on top of the static calibration.
    pub bias_accel: Vector3<S>,
    /// One-sigma of the velocity prior, m/s, per axis.
    pub velocity_sigma: S,
}

impl<S: Scalar> SqrtKeypointVio<S> {
    /// An estimator that starts from a solved first state.
    ///
    /// As [`Self::new`], plus: the first state is `initial` instead of the
    /// stereo start, the bias priors are centred on its biases, and the
    /// initial square-root prior gains `1 / velocity_sigma` on the velocity
    /// (indices 6 to 8), which the stereo start leaves free. The pose prior
    /// is the stereo start's.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], and [`EstimatorError::InitialVelocitySigma`].
    pub fn with_initial_state(
        g: Vector3<S>,
        calibration: Calibration<S>,
        config: VioConfig,
        initial: InitialState<S>,
    ) -> Result<Self, EstimatorError> {
        let sigma: f64 = initial.velocity_sigma.to_f64();
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(EstimatorError::InitialVelocitySigma(sigma));
        }
        let mut estimator: Self = Self::new(g, calibration, config)?;
        let velocity_sqrt_weight: S = S::one() / initial.velocity_sigma;
        for i in 0..3 {
            estimator.marg_data.h[(6 + i, 6 + i)] = velocity_sqrt_weight;
        }
        estimator.initial_bias_gyro = initial.bias_gyro;
        estimator.initial_bias_accel = initial.bias_accel;
        estimator.initial_state = Some(initial);
        Ok(estimator)
    }

    /// The solved first state this estimator starts from, if any.
    pub fn initial_state(&self) -> Option<&InitialState<S>> {
        self.initial_state.as_ref()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::estimator::{FlowObservations, FrameOutcome};
    use crate::lie::So3;
    use kornia_staging_sensors::imu::CombinedImuSample;
    use std::sync::Arc;

    const MSDMI_CONFIG: &str = include_str!("../../../../configs/msdmi_config.json");
    const MSDMI_CALIB: &str = include_str!("../../tests/fixtures/msdmi_calib.json");

    fn initial(t_ns: i64) -> InitialState<f64> {
        InitialState {
            t_ns,
            t_w_i: Se3::new(So3::exp(&Vector3::new(0.1, -0.2, 0.3)), Vector3::zeros()),
            vel_w_i: Vector3::new(1.2, -0.3, 0.05),
            bias_gyro: Vector3::new(0.001, -0.002, 0.003),
            bias_accel: Vector3::new(0.02, 0.0, -0.01),
            velocity_sigma: 0.05,
        }
    }

    fn estimator(state: InitialState<f64>) -> Result<SqrtKeypointVio<f64>, EstimatorError> {
        SqrtKeypointVio::with_initial_state(
            crate::imu::gravity(9.81),
            Calibration::from_json_str(MSDMI_CALIB).unwrap(),
            VioConfig::from_json_str(MSDMI_CONFIG).unwrap(),
            state,
        )
    }

    fn still_imu(estimator: &mut SqrtKeypointVio<f64>) {
        for i in -10..=100 {
            estimator.push_imu(CombinedImuSample {
                timestamp_ns: i * 1_000_000,
                gyro: [0.0, 0.0, 0.0].into(),
                accel: [0.0, 0.0, 9.81].into(),
            });
        }
    }

    /// The velocity block of the initial prior carries `1 / sigma`; the rest
    /// is the stereo start's prior.
    #[test]
    fn velocity_prior_joins_the_stereo_prior() {
        let stereo = SqrtKeypointVio::<f64>::with_default_gravity(
            Calibration::from_json_str(MSDMI_CALIB).unwrap(),
            VioConfig::from_json_str(MSDMI_CONFIG).unwrap(),
        )
        .unwrap();
        let solved = estimator(initial(0)).unwrap();
        for i in 0..15 {
            for j in 0..15 {
                let expected: f64 = if i == j && (6..9).contains(&i) {
                    20.0
                } else {
                    stereo.marg_data.h[(i, j)]
                };
                assert_eq!(solved.marg_data.h[(i, j)], expected, "({i}, {j})");
            }
        }
        assert_eq!(
            stereo.marg_data.h[(6, 6)],
            0.0,
            "the stereo start leaves velocity free"
        );
        assert!(stereo.initial_state().is_none());
    }

    /// The first frameset starts from the solved state with no landmarks at
    /// all: the stereo gate does not apply.
    #[test]
    fn first_frameset_takes_the_solved_state_without_landmarks() {
        let state = initial(50_000_000);
        let mut vio = estimator(state).unwrap();
        still_imu(&mut vio);
        let frame = Arc::new(FlowObservations::new(50_000_000, 2));
        let outcome = vio.process_frame(frame, None).unwrap();
        assert!(matches!(outcome, FrameOutcome::Measured(_)));
        let first = vio.state().unwrap();
        assert_eq!(first.t_w_i, state.t_w_i);
        assert_eq!(first.vel_w_i, state.vel_w_i);
        assert_eq!(first.bias_gyro, state.bias_gyro);
        assert_eq!(first.bias_accel, state.bias_accel);
    }

    #[test]
    fn a_different_first_frameset_is_refused() {
        let mut vio = estimator(initial(50_000_000)).unwrap();
        still_imu(&mut vio);
        let frame = Arc::new(FlowObservations::new(60_000_000, 2));
        assert_eq!(
            vio.process_frame(frame, None).err(),
            Some(EstimatorError::InitialStateTime {
                expected_t_ns: 50_000_000,
                t_ns: 60_000_000
            })
        );
    }

    #[test]
    fn a_bad_velocity_sigma_is_refused() {
        for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let state = InitialState {
                velocity_sigma: sigma,
                ..initial(0)
            };
            assert!(matches!(
                estimator(state).err(),
                Some(EstimatorError::InitialVelocitySigma(_))
            ));
        }
    }
}
