#![allow(clippy::unwrap_used)]

use crate::lie::Se3;
use crate::types::PoseVelBiasState;
type Vector9<S> = SVector<S, 9>;
use super::*;
use approx::assert_abs_diff_eq;
use kornia_staging_sensors::imu::CombinedImuSample;
use nalgebra::{Matrix3, SVector};
use proptest::prelude::*;

// Share numerical fixtures without exposing a production test-support API.
#[path = "../../../../../kornia-staging/crates/kornia-staging-sensors/src/imu/preintegration/test_support.rs"]
mod test_support;
use test_support::{ACCEL_STD_DEV, GYRO_STD_DEV, Rng, Trajectory, biased_samples, integrate_all};

fn noise_from_std_dev() -> ImuNoise<f64> {
    ImuNoise::new(
        Vector3::repeat(ACCEL_STD_DEV * ACCEL_STD_DEV),
        Vector3::repeat(GYRO_STD_DEV * GYRO_STD_DEV),
    )
    .unwrap()
}

/// A duplicate or reordered sample is rejected and nothing is integrated.
#[test]
fn duplicate_and_reordered_samples_are_rejected() {
    let ones: Vector3<f64> = Vector3::repeat(1.0);
    let mut meas: IntegratedImuMeasurement<f64> =
        IntegratedImuMeasurement::new(1_000, &Vector3::zeros(), &Vector3::zeros());
    let sample = |t_ns: i64| CombinedImuSample {
        timestamp_ns: t_ns,
        gyro: kornia_algebra::Vec3F64::new(0.01, 0.0, 0.0),
        accel: kornia_algebra::Vec3F64::new(0.0, 0.0, 9.81),
    };

    meas.integrate(
        &sample(2_000),
        &kornia_staging_sensors::imu::ImuNoise::new(ones, ones).unwrap(),
    )
    .unwrap();
    assert_eq!(meas.dt_ns(), 1_000);
    assert_eq!(
        meas.integrate(
            &sample(2_000),
            &kornia_staging_sensors::imu::ImuNoise::new(ones, ones).unwrap()
        ),
        Err(SensorError::NonMonotonicSample {
            previous_timestamp_ns: 1_000,
            timestamp_ns: 1_000
        })
    );
    assert_eq!(
        meas.integrate(
            &sample(1_500),
            &kornia_staging_sensors::imu::ImuNoise::new(ones, ones).unwrap()
        ),
        Err(SensorError::NonMonotonicSample {
            previous_timestamp_ns: 1_000,
            timestamp_ns: 500
        })
    );
    // The measurement is unchanged after both refusals.
    assert_eq!(meas.dt_ns(), 1_000);

    // A sample at exactly the start time would be a zero-length step.
    let mut fresh: IntegratedImuMeasurement<f64> =
        IntegratedImuMeasurement::new(1_000, &Vector3::zeros(), &Vector3::zeros());
    assert_eq!(
        fresh.integrate(
            &sample(1_000),
            &kornia_staging_sensors::imu::ImuNoise::new(ones, ones).unwrap()
        ),
        Err(SensorError::NonMonotonicSample {
            previous_timestamp_ns: 0,
            timestamp_ns: 0
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
        accumulate_to(&mut meas(), None, feed(vec![sample(1)]), 0, 0, &noise),
        Err(AccumulateError::NonMonotonicFrames { t0_ns: 0, t1_ns: 0 })
    );
    // An interval that does not start where the measurement was built:
    // every sample would be timed against the wrong origin.
    assert_eq!(
        accumulate_to(&mut meas(), None, feed(vec![sample(1)]), 5, 10, &noise),
        Err(AccumulateError::StartTimeMismatch {
            start_timestamp_ns: 0,
            t0_ns: 5
        })
    );
    // A duplicate and a reordered sample, both from the per-sample step.
    assert_eq!(
        accumulate_to(
            &mut meas(),
            None,
            feed(vec![sample(2), sample(2)]),
            0,
            10,
            &noise
        ),
        Err(AccumulateError::Sensor(SensorError::NonMonotonicSample {
            previous_timestamp_ns: 2,
            timestamp_ns: 2
        }))
    );
    assert_eq!(
        accumulate_to(
            &mut meas(),
            None,
            feed(vec![sample(3), sample(1)]),
            0,
            10,
            &noise
        ),
        Err(AccumulateError::Sensor(SensorError::NonMonotonicSample {
            previous_timestamp_ns: 3,
            timestamp_ns: 1
        }))
    );
    // Nothing after the frame to close the interval with.
    assert_eq!(
        accumulate_to(
            &mut meas(),
            None,
            feed(vec![sample(2), sample(4)]),
            0,
            10,
            &noise
        ),
        Err(AccumulateError::MissingSampleAfterFrame { t1_ns: 10 })
    );

    // The sample that does follow closes the interval exactly on the frame
    // and comes back out at its own time.
    let mut closed: IntegratedImuMeasurement<f64> = meas();
    let pending: Option<Popped<f64>> = accumulate_to(
        &mut closed,
        None,
        feed(vec![sample(2), sample(4), sample(12)]),
        0,
        10,
        &noise,
    )
    .unwrap();
    assert_eq!(pending.map(|(t_ns, _, _)| t_ns), Some(12));
    assert_eq!(closed.dt_ns(), 10);
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

proptest! {
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
        let sample = |t_ns: i64| CombinedImuSample {
            timestamp_ns: t_ns,
            gyro: kornia_algebra::Vec3F64::new(0.02, 0.0, -0.01),
            accel: kornia_algebra::Vec3F64::new(0.0, 0.0, 9.81),
        };
        meas.integrate(&sample(start_t_ns + step_ns), &kornia_staging_sensors::imu::ImuNoise::new(ones, ones).unwrap())?;
        let before: IntegratedImuMeasurement<f64> = meas;

        let result = meas.integrate(&sample(start_t_ns + step_ns - back_ns), &kornia_staging_sensors::imu::ImuNoise::new(ones, ones).unwrap());
        prop_assert_eq!(
            result,
            Err(SensorError::NonMonotonicSample {
                previous_timestamp_ns: step_ns,
                timestamp_ns: step_ns - back_ns
            })
        );
        prop_assert_eq!(meas, before);
    }
}

mod factor;

mod properties;

#[test]
fn queue_rejects_nonfinite_input_before_changing_measurement() {
    let mut measurement =
        IntegratedImuMeasurement::<f64>::new(0, &Vector3::zeros(), &Vector3::zeros());
    let before = measurement;
    let result = accumulate_to(
        &mut measurement,
        Some((1, Vector3::repeat(f64::NAN), Vector3::zeros())),
        || Some((3, Vector3::zeros(), Vector3::zeros())),
        0,
        2,
        &ImuNoise::new(Vector3::repeat(1.0), Vector3::repeat(1.0)).unwrap(),
    );
    assert_eq!(result, Err(AccumulateError::Sensor(SensorError::NonFiniteInput)));
    assert_eq!(measurement, before);
}
