//! Self-contained IMU preintegration and gravity alignment properties.
#![allow(clippy::unwrap_used)]

use nalgebra::{SVector, Vector3};
use proptest::prelude::*;
use slam_rs::imu::{ImuSample, IntegratedImuMeasurement, gravity_from_first_accel};

const ACCEL_STD_DEV: f64 = 0.23;
const GYRO_STD_DEV: f64 = 0.0027;

fn integrate(
    samples: &[ImuSample],
    bg: Vector3<f64>,
    ba: Vector3<f64>,
) -> IntegratedImuMeasurement<f64> {
    let mut measurement = IntegratedImuMeasurement::new(0, &bg, &ba);
    for sample in samples {
        measurement
            .integrate(
                sample,
                &Vector3::repeat(ACCEL_STD_DEV.powi(2)),
                &Vector3::repeat(GYRO_STD_DEV.powi(2)),
            )
            .unwrap();
    }
    measurement
}

fn sample(k: i64, dt_ns: i64) -> ImuSample {
    ImuSample {
        t_ns: (k + 1) * dt_ns,
        accel: Vector3::new(
            (k % 7) as f64 * 0.25 - 0.75,
            (k % 5) as f64 * 0.5 - 1.0,
            9.8125 + (k % 3) as f64 * 0.0625,
        ),
        gyro: Vector3::new(
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
        let m = measurement.get_cov_inv_sqrt();
        let product = m * measurement.get_cov() * m.transpose();
        assert!((product * product - product).amax() < 1e-6);
    }
}

#[test]
fn a_rank_deficient_covariance_whitens_to_zero_position_weight() {
    let sample = ImuSample {
        t_ns: 5_000_000,
        gyro: Vector3::zeros(),
        accel: Vector3::zeros(),
    };
    let measurement = integrate(&[sample], Vector3::zeros(), Vector3::zeros());
    let m = measurement.get_cov_inv_sqrt();
    let information = measurement.get_cov_inv();
    for axis in 0..3 {
        assert_eq!(information[(axis, axis)], 0.0);
        assert_eq!(m.row(axis + 6).norm(), 0.0);
    }
    // A single constant sample gives rotation/velocity standard deviation sigma * dt.
    assert!((m[(3, 3)] - 1.0 / (GYRO_STD_DEV * 0.005)).abs() < 1e-6);
    assert!((m[(0, 6)] - 1.0 / (ACCEL_STD_DEV * 0.005)).abs() < 1e-9);
}

#[test]
fn gravity_init_deviation_stays_within_its_bound() {
    for tilt in [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3] {
        let a = Vector3::new(tilt, -tilt, -1.0).normalize();
        assert!(
            (gravity_from_first_accel(&a) * a - Vector3::z()).norm() <= 2.0 * (2e-12_f64).sqrt()
        );
        let a = a.map(|v| v as f32);
        assert!(
            (gravity_from_first_accel(&a) * a - Vector3::z()).norm() <= 2.0 * (2e-5_f32).sqrt()
        );
    }
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
            ImuSample { t_ns: time, gyro: Vector3::from(gyro), accel: Vector3::from(accel) }
        }).collect();
        let measurement = integrate(&samples, Vector3::zeros(), Vector3::zeros());
        let state = measurement.get_delta_state();
        prop_assert!(state.diff(state).iter().all(|v| v.is_finite()));
        let covariance = measurement.get_cov();
        prop_assert!(covariance.iter().all(|v| v.is_finite()));
        prop_assert!((covariance - covariance.transpose()).amax() < 1e-12);
        prop_assert!(covariance.symmetric_eigen().eigenvalues.min() >= -1e-12);
        let m = measurement.get_cov_inv_sqrt();
        let product = m * covariance * m.transpose();
        prop_assert!((product * product - product).amax() < 1e-6);
        // Central differences in the delta state's tangent coordinates, step 1e-5,
        // absolute tolerance 1e-7 (position, angle and velocity per bias unit).
        let epsilon = 1e-5;
        for gyro in [false, true] {
            let analytic = if gyro { measurement.get_d_state_d_bg() } else { measurement.get_d_state_d_ba() };
            for axis in 0..3 {
                let mut shift = Vector3::zeros();
                shift[axis] = epsilon;
                let (plus, minus) = if gyro {
                    (integrate(&samples, shift, Vector3::zeros()), integrate(&samples, -shift, Vector3::zeros()))
                } else {
                    (integrate(&samples, Vector3::zeros(), shift), integrate(&samples, Vector3::zeros(), -shift))
                };
                let numeric: SVector<f64, 9> = (state.diff(plus.get_delta_state()) - state.diff(minus.get_delta_state())) / (2.0 * epsilon);
                prop_assert!((numeric - analytic.column(axis)).amax() < 1e-7);
            }
        }
    }

    #[test]
    fn a_rotation_maps_one_direction_to_the_other(
        a in prop::array::uniform3(-20.0f64..20.0),
        b in prop::array::uniform3(-20.0f64..20.0),
        antiparallel in any::<bool>()
    ) {
        let a = Vector3::from(a);
        let b = if antiparallel { -a } else { Vector3::from(b) };
        prop_assume!(a.norm() > 0.01 && b.norm() > 0.01);
        // The public API aligns to +Z. Compose two such rotations to test arbitrary pairs.
        let rotation = gravity_from_first_accel(&b).inverse() * gravity_from_first_accel(&a);
        prop_assert!((rotation * a.normalize() - b.normalize()).norm() < 6e-6);
    }
}
