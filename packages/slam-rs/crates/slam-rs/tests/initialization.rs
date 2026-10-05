//! A world needs metric stereo structure before it can publish a state.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use nalgebra::{Matrix2x4, Vector2, Vector3, Vector4};
use slam_rs::camera::CameraEnum;
use slam_rs::estimator::{FlowObservations, FrameOutcome, SqrtKeypointVio};
use slam_rs::imu::ImuSample;
use slam_rs::lie::LieScalar;
use slam_rs::types::KeypointId;
use slam_rs::{Backend, ImageView, Vio, VioStatus};

mod common;

fn stereo(t_ns: i64, count: usize) -> Arc<FlowObservations> {
    let calibration = common::calibration();
    let mut frame = FlowObservations::new(t_ns, calibration.t_i_c.len());
    for id in 0..count {
        let point = calibration.t_i_c[0]
            * Vector3::new(
                (id % 5) as f64 * 0.15 - 0.3,
                (id / 5) as f64 * 0.15 - 0.3,
                3.0,
            );
        for (cam, pixels) in frame.cameras.iter_mut().enumerate() {
            let p = calibration.t_i_c[cam].inverse() * point;
            let model = CameraEnum::from_model(&calibration.intrinsics[cam]).unwrap();
            let mut pixel = Vector2::zeros();
            assert!(model.project_with_jacobian(
                &Vector4::new(p.x, p.y, p.z, 1.0),
                &mut pixel,
                &mut Matrix2x4::zeros(),
            ));
            pixels.insert(KeypointId(id as u64), pixel.cast());
        }
    }
    Arc::new(frame)
}

#[test]
fn nine_landmarks_wait_and_ten_start_with_fresh_imu_and_prior() {
    check_threshold::<f32>();
    check_threshold::<f64>();
}

fn check_threshold<S: LieScalar>() {
    let make = || {
        SqrtKeypointVio::<S>::with_default_gravity(common::calibration().cast(), common::config())
            .unwrap()
    };
    let mut waiting = make();
    let mut fresh = make();
    for n in 0..=30 {
        let sample = ImuSample {
            t_ns: n * 5_000_000,
            gyro: Vector3::zeros(),
            accel: if n < 10 {
                Vector3::new(2.0, 0.0, 9.0)
            } else {
                Vector3::new(0.0, 0.0, 9.81)
            },
        };
        waiting.push_imu(sample);
        fresh.push_imu(sample);
    }
    for (t, count) in [(0, 0), (20_000_000, 9)] {
        assert_eq!(
            waiting.process_frame(stereo(t, count)).unwrap(),
            FrameOutcome::NoVisualFeatures
        );
        assert!(
            !waiting.is_initialized(),
            "{count} landmarks must not start a world"
        );
        assert!(waiting.state().is_none());
        assert!(waiting.imu_meas().is_empty());
        assert!(waiting.num_points_kf().is_empty());
    }
    for t in [
        60_000_000,
        80_000_000,
        100_000_000,
        120_000_000,
        140_000_000,
    ] {
        let FrameOutcome::Measured(mut actual) = waiting.process_frame(stereo(t, 10)).unwrap()
        else {
            panic!("ten landmarks must start")
        };
        let FrameOutcome::Measured(mut expected) = fresh.process_frame(stereo(t, 10)).unwrap()
        else {
            panic!("fresh start must work")
        };
        actual.timings = Default::default();
        expected.timings = Default::default();
        assert_eq!(actual, expected);
        assert_eq!(waiting.state(), fresh.state());
        assert_eq!(waiting.marg_data(), fresh.marg_data());
    }
}

#[test]
fn frontend_keeps_tracking_while_lag_waits_for_stereo() {
    check_pipeline(Backend::Cpu);
}

fn check_pipeline(backend: Backend) {
    let directory = common::fixtures().join("flow/frames");
    let stereo: Vec<_> = (0..2)
        .map(|camera| common::read_pgm(&directory, 0, camera))
        .collect();
    let blank = vec![0; stereo[0].pixels.len()];
    for lag in [false, true] {
        for deferred in [false, true] {
            let mut config = common::config();
            config.port_frontend_lag = lag;
            config.port_keyframe_solve_deferred = deferred;
            config.port_frame_update_max_iterations = 5;

            let mut vio = Vio::<f32>::with_backend(
                config,
                common::calibration(),
                Default::default(),
                backend,
            )
            .unwrap();
            for n in 0..100 {
                vio.push_imu(n * 5_000_000, [0.0; 3], [0.0, 0.0, 9.81])
                    .unwrap();
            }
            let mut results = Vec::new();
            for index in 0..10 {
                let views_for = |index| {
                    stereo
                        .iter()
                        .map(|image| ImageView {
                            width: image.width,
                            height: image.height,
                            stride: image.width,
                            data: if index < 3 { &blank } else { &image.pixels },
                        })
                        .collect::<Vec<_>>()
                };
                let views = views_for(index);
                let t_ns = index * 20_000_000;
                let result = vio.track(t_ns, &views).unwrap();
                assert_eq!(vio.frontend().frame_counter(), index as u64 + 1);
                if result.status != VioStatus::Buffered {
                    results.push(result);
                }
                if index < 3 {
                    assert!(vio.estimator().state().is_none());
                    assert!(!vio.estimator().has_deferred_keyframe());
                }
            }
            results.extend(vio.flush().unwrap());
            assert!(vio.flush().unwrap().is_none());
            assert_eq!(results.len(), 10);
            for (index, result) in results.iter().enumerate() {
                assert_eq!(result.t_ns, index as i64 * 20_000_000);
                assert_eq!(
                    result.status,
                    if index < 3 {
                        VioStatus::NoVisualFeatures
                    } else {
                        VioStatus::Tracking
                    }
                );
            }
            assert!(vio.last_stats().unwrap().opt_started);
        }
    }
}

#[test]
fn flushing_an_unstarted_world_reports_no_pose_once() {
    let mut config = common::config();
    config.port_frontend_lag = true;
    let mut vio = Vio::<f32>::new(config, common::calibration(), Default::default()).unwrap();
    vio.push_imu(10, [0.0; 3], [0.0, 0.0, 9.81]).unwrap();
    let blank = vec![0; 960 * 960];
    let views = [ImageView {
        width: 960,
        height: 960,
        stride: 960,
        data: &blank,
    }; 2];
    assert_eq!(vio.track(0, &views).unwrap().status, VioStatus::Buffered);
    let result = vio.flush().unwrap().unwrap();
    assert_eq!(result.status, VioStatus::NoVisualFeatures);
    assert!(vio.estimator().state().is_none());
    assert!(vio.flush().unwrap().is_none());
}
