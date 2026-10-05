#![allow(clippy::unwrap_used)]

use super::*;
use crate::config::VioConfig;

const CALIB: &str = include_str!("../../tests/fixtures/msdmi_calib.json");
const CONFIG: &str = include_str!("../../../../configs/msdmi_config.json");

/// The fixture rig and config, `f32` as the shipped lane runs.
fn estimator() -> SqrtKeypointVio<f32> {
    SqrtKeypointVio::with_default_gravity(
        Calibration::<f64>::from_json_str(CALIB).unwrap().cast(),
        VioConfig::from_json_str(CONFIG).unwrap(),
    )
    .unwrap()
}

/// A sample the initialization can take a gravity direction from.
fn sample(t_ns: i64) -> ImuSample {
    ImuSample {
        t_ns,
        gyro: Vector3::zeros(),
        accel: Vector3::new(0.0, 0.0, 9.81),
    }
}

/// Check rig list lengths and all scalar domains at the boundary (D32).
/// Negative prior weights or IMU rates produce NaNs under square root; zero bias
/// or observation deviations produce infinities. Valid shipped calibrations
/// cannot exercise these failures, so explicit invalid inputs are required.
/// The probes use values exactly representable in f32 before widening to f64.
#[test]
fn a_scalar_outside_its_domain_is_refused_before_the_estimator_exists() {
    let refuse_config = |mutate: &dyn Fn(&mut VioConfig)| -> EstimatorError {
        let mut config: VioConfig = VioConfig::from_json_str(CONFIG).unwrap();
        mutate(&mut config);
        SqrtKeypointVio::<f32>::with_default_gravity(
            Calibration::<f64>::from_json_str(CALIB).unwrap().cast(),
            config,
        )
        .unwrap_err()
    };
    let refuse_calibration = |mutate: &dyn Fn(&mut Calibration<f64>)| -> EstimatorError {
        let mut calibration: Calibration<f64> = Calibration::from_json_str(CALIB).unwrap();
        mutate(&mut calibration);
        SqrtKeypointVio::<f32>::with_default_gravity(
            calibration.cast(),
            VioConfig::from_json_str(CONFIG).unwrap(),
        )
        .unwrap_err()
    };

    // The shipped pair, which every domain accepts.
    estimator();

    // The three initial prior weights, whose square roots the prior is.
    assert_eq!(
        refuse_config(&|config| config.vio_init_pose_weight = -1.0),
        EstimatorError::NegativeScalar {
            field: "vio_init_pose_weight",
            value: -1.0,
        }
    );
    assert_eq!(
        refuse_config(&|config| config.vio_init_ba_weight = f64::NEG_INFINITY),
        EstimatorError::NegativeScalar {
            field: "vio_init_ba_weight",
            value: f64::NEG_INFINITY,
        }
    );
    assert_eq!(
        refuse_config(&|config| config.vio_init_bg_weight = f64::INFINITY),
        EstimatorError::NegativeScalar {
            field: "vio_init_bg_weight",
            value: f64::INFINITY,
        }
    );

    // The reprojection cost's two scalars.
    assert_eq!(
        refuse_config(&|config| config.vio_obs_std_dev = 0.0),
        EstimatorError::NonPositiveScalar {
            field: "vio_obs_std_dev",
            value: 0.0,
        }
    );
    assert_eq!(
        refuse_config(&|config| config.vio_obs_huber_thresh = -1.0),
        EstimatorError::NonPositiveScalar {
            field: "vio_obs_huber_thresh",
            value: -1.0,
        }
    );

    // The damping: three positive bounds, and they must not cross.
    assert_eq!(
        refuse_config(&|config| config.vio_lm_lambda_initial = f64::INFINITY),
        EstimatorError::NonPositiveScalar {
            field: "vio_lm_lambda_initial",
            value: f64::INFINITY,
        }
    );
    assert_eq!(
        refuse_config(&|config| config.vio_lm_lambda_min = 0.0),
        EstimatorError::NonPositiveScalar {
            field: "vio_lm_lambda_min",
            value: 0.0,
        }
    );
    assert_eq!(
        refuse_config(&|config| {
            config.vio_lm_lambda_min = 1.0;
            config.vio_lm_lambda_max = 0.5;
        }),
        EstimatorError::DampingRangeReversed { min: 1.0, max: 0.5 }
    );

    // The IMU rate and the four deviations, one probe each.
    assert_eq!(
        refuse_calibration(&|calibration| calibration.imu_update_rate = -200.0),
        EstimatorError::NonPositiveScalar {
            field: "imu_update_rate",
            value: -200.0,
        }
    );
    assert_eq!(
        refuse_calibration(&|calibration| calibration.gyro_bias_std.y = 0.0),
        EstimatorError::NonPositiveScalar {
            field: "gyro_bias_std",
            value: 0.0,
        }
    );
    assert_eq!(
        refuse_calibration(&|calibration| calibration.accel_bias_std.z = -0.5),
        EstimatorError::NonPositiveScalar {
            field: "accel_bias_std",
            value: -0.5,
        }
    );
    assert_eq!(
        refuse_calibration(&|calibration| calibration.accel_noise_std.x = f64::INFINITY),
        EstimatorError::NonPositiveScalar {
            field: "accel_noise_std",
            value: f64::INFINITY,
        }
    );
    // A NaN cannot be compared with `assert_eq!`; it is refused as well.
    assert!(matches!(
        refuse_calibration(&|calibration| calibration.gyro_noise_std.x = f64::NAN),
        EstimatorError::NonPositiveScalar {
            field: "gyro_noise_std",
            ..
        }
    ));
}

#[test]
fn a_rig_whose_intrinsics_and_extrinsics_disagree_is_refused() {
    let config: VioConfig = VioConfig::from_json_str(CONFIG).unwrap();
    let mut calibration: Calibration<f64> = Calibration::from_json_str(CALIB).unwrap();
    calibration.intrinsics.pop();
    let refused: EstimatorError =
        SqrtKeypointVio::new(Vector3::new(0.0, 0.0, -9.81), calibration, config).unwrap_err();
    assert!(
        matches!(
            refused,
            EstimatorError::CameraCountMismatch {
                expected: 2,
                actual: 1
            }
        ),
        "two extrinsics and one intrinsic is not a rig, got {refused:?}"
    );
}

/// Compare IMU timestamps with the newest accepted sample, including `pending`.
/// Comparing only with an empty queue would let an older sample enter behind it
/// and reverse timestamps during integration.
#[test]
fn an_imu_sample_behind_the_pending_one_is_dropped() {
    let mut estimator: SqrtKeypointVio<f32> = estimator();
    estimator.push_imu(sample(10));
    // One frameset before the sample: the initialization pops it into
    // `pending` and leaves the queue empty, which is the whole setup.
    estimator
        .process_frame(Arc::new(FlowObservations::new(5, 2)))
        .unwrap();
    assert!(estimator.imu_queue.is_empty());
    assert_eq!(estimator.pending.map(|(t_ns, _, _)| t_ns), Some(10));

    estimator.push_imu(sample(7));
    assert!(
        estimator.imu_queue.is_empty(),
        "a sample older than the pending one was accepted behind it"
    );
    assert_eq!(estimator.newest_imu_t_ns, Some(10));

    // A sample that does follow the pending one is still accepted.
    estimator.push_imu(sample(12));
    assert_eq!(
        estimator.imu_queue.back().map(|s| s.t_ns),
        Some(12),
        "the ordering check rejected a sample that does follow"
    );
}

/// A missing host pixel must return an error instead of silently skipping a landmark.
/// Production `measure` builds both maps from the same frameset. This test
/// constructs inconsistent input explicitly to check that invariant (D32).
#[test]
fn an_unconnected_keypoint_missing_from_its_own_frameset_is_refused() {
    let mut estimator: SqrtKeypointVio<f32> = estimator();
    // This triangulation-error probe needs a host pose. A blank frameset no
    // longer initializes a world, so supply that precondition directly.
    estimator.ba.frame_states.insert(
        5,
        PoseVelBiasStateWithLin::new(
            PoseVelBiasState::new(
                5,
                Se3::identity(),
                Vector3::zeros(),
                Vector3::zeros(),
                Vector3::zeros(),
            ),
            true,
        ),
    );

    let frame: FlowObservations = FlowObservations::new(5, 2);
    let unconnected: Vec<BTreeSet<KeypointId>> =
        vec![BTreeSet::from([KeypointId(7)]), BTreeSet::new()];
    assert_eq!(
        estimator
            .triangulate_unconnected(&frame, &unconnected)
            .unwrap_err(),
        EstimatorError::UnconnectedKeypointMissing {
            cam_id: 0,
            kpt_id: KeypointId(7),
        }
    );

    // The same call over an id the frameset does carry gets as far as the
    // triangulation, which one view cannot satisfy: no landmark, no error.
    let mut carried: FlowObservations = FlowObservations::new(5, 2);
    carried.cameras[0].insert(KeypointId(7), Vector2::new(480.0, 480.0));
    assert_eq!(
        estimator
            .triangulate_unconnected(&carried, &unconnected)
            .unwrap(),
        0
    );
}
