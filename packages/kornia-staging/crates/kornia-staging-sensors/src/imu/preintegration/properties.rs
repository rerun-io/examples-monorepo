//! Whitening and bias properties of the staged numerical kernel.
use super::test_support::{ACCEL_STD_DEV, GYRO_STD_DEV};
use super::{CombinedImuSample, IntegratedImuMeasurement};
use nalgebra::{SVector, Vector3};
use proptest::prelude::*;

fn integrate(
    samples: &[CombinedImuSample],
    bg: Vector3<f64>,
    ba: Vector3<f64>,
) -> IntegratedImuMeasurement<f64> {
    let mut measurement = IntegratedImuMeasurement::new(0, &bg, &ba);
    for sample in samples {
        measurement
            .integrate(
                sample,
                &crate::imu::ImuNoise::new(
                    Vector3::repeat(ACCEL_STD_DEV.powi(2)),
                    Vector3::repeat(GYRO_STD_DEV.powi(2)),
                )
                .unwrap(),
            )
            .unwrap();
    }
    measurement
}

fn sample(k: i64, dt_ns: i64) -> CombinedImuSample {
    CombinedImuSample {
        timestamp_ns: (k + 1) * dt_ns,
        accel: kornia_algebra::Vec3F64::new(
            (k % 7) as f64 * 0.25 - 0.75,
            (k % 5) as f64 * 0.5 - 1.0,
            9.8125 + (k % 3) as f64 * 0.0625,
        ),
        gyro: kornia_algebra::Vec3F64::new(
            (k % 11) as f64 * 0.015625 - 0.078125,
            (k % 13) as f64 * 0.0078125 - 0.046875,
            (k % 9) as f64 * 0.03125 - 0.125,
        ),
    }
}

#[test]
fn the_whitening_is_idempotent_on_the_observable_directions() {
    for count in [2, 10, 100] {
        let samples: Vec<_> = (0..count).map(|k| sample(k, 5_000_000)).collect();
        let measurement = integrate(&samples, Vector3::zeros(), Vector3::zeros());
        let m = measurement.cov_inv_sqrt();
        let product = m * measurement.cov() * m.transpose();
        assert!((product * product - product).amax() < 1e-6);
    }
}

#[test]
fn a_rank_deficient_covariance_whitens_to_zero_position_weight() {
    let sample = CombinedImuSample {
        timestamp_ns: 5_000_000,
        gyro: kornia_algebra::Vec3F64::ZERO,
        accel: kornia_algebra::Vec3F64::ZERO,
    };
    let measurement = integrate(&[sample], Vector3::zeros(), Vector3::zeros());
    let m = measurement.cov_inv_sqrt();
    let information = measurement.cov_inv();
    for axis in 0..3 {
        assert_eq!(information[(axis, axis)], 0.0);
        assert_eq!(m.row(axis + 6).norm(), 0.0);
    }
    // A single constant sample gives rotation/velocity standard deviation sigma * dt.
    assert!((m[(3, 3)] - 1.0 / (GYRO_STD_DEV * 0.005)).abs() < 1e-6);
    assert!((m[(0, 6)] - 1.0 / (ACCEL_STD_DEV * 0.005)).abs() < 1e-9);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn preintegration_and_bias_jacobians_obey_their_contract(
        readings in prop::collection::vec((1_000_000i64..10_000_000, prop::array::uniform3(-5.0f64..5.0), prop::array::uniform3(-30.0f64..30.0)), 2..30)
    ) {
        let mut time = 0;
        let samples: Vec<_> = readings.into_iter().map(|(dt, gyro, accel)| {
            time += dt;
            CombinedImuSample { timestamp_ns: time, gyro: gyro.into(), accel: accel.into() }
        }).collect();
        let measurement = integrate(&samples, Vector3::zeros(), Vector3::zeros());
        let state = measurement.delta_state();
        prop_assert!(state.diff(state).iter().all(|v| v.is_finite()));
        let covariance = measurement.cov();
        prop_assert!(covariance.iter().all(|v| v.is_finite()));
        prop_assert!((covariance - covariance.transpose()).amax() < 1e-12);
        prop_assert!(covariance.symmetric_eigen().eigenvalues.min() >= -1e-12);
        let m = measurement.cov_inv_sqrt();
        let product = m * covariance * m.transpose();
        prop_assert!((product * product - product).amax() < 1e-6);
        // Central differences in the delta state's tangent coordinates, step 1e-5,
        // absolute tolerance 1e-7 (position, angle and velocity per bias unit).
        let epsilon = 1e-5;
        for gyro in [false, true] {
            let analytic = if gyro { measurement.d_state_d_bias_gyro() } else { measurement.d_state_d_bias_accel() };
            for axis in 0..3 {
                let mut shift = Vector3::zeros();
                shift[axis] = epsilon;
                let (plus, minus) = if gyro {
                    (integrate(&samples, shift, Vector3::zeros()), integrate(&samples, -shift, Vector3::zeros()))
                } else {
                    (integrate(&samples, Vector3::zeros(), shift), integrate(&samples, Vector3::zeros(), -shift))
                };
                let numeric: SVector<f64, 9> = (state.diff(plus.delta_state()) - state.diff(minus.delta_state())) / (2.0 * epsilon);
                prop_assert!((numeric - analytic.column(axis)).amax() < 1e-7);
            }
        }
    }

}
