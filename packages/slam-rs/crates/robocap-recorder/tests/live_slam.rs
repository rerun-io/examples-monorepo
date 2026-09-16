#![cfg(feature = "live-slam")]
use anyhow::Result;
use robocap_recorder::{ImuChannel, LiveSlam, LiveSlamOptions, SlamInput};
use slam_rs::{
    calib::{Calibration, CameraModel, PinholeParams},
    config::VioConfig,
    lie::{Se3, So3},
};

#[test]
fn a_known_stationary_textured_rig_stays_at_its_origin() -> Result<()> {
    let largest = stationary_probe([0.0; 3], [0.0, 0.0, 9.81], false)?;
    assert!(largest < 0.02, "stationary rig moved {largest} m");
    Ok(())
}

#[test]
#[ignore = "diagnostic comparison, not an acceptance gate for unknown bias"]
fn stationary_bias_probe() -> Result<()> {
    let tilted = stationary_probe(
        [-0.0239258, 0.0017384, -0.0047842],
        [-0.282325, -2.02408, -9.62928],
        false,
    )?;
    eprintln!("Cap B mean IMU, known synthetic geometry: {tilted} m");
    for correct_bias in [false, true] {
        let largest =
            stationary_probe([-0.024, 0.002, -0.005], [0.03, -0.02, 9.845], correct_bias)?;
        eprintln!("bias corrected={correct_bias}: max displacement {largest} m");
        if correct_bias {
            assert!(largest < 0.02, "known bias correction failed: {largest}");
        }
    }
    Ok(())
}

fn stationary_probe(gyro: [f64; 3], accel: [f64; 3], correct_bias: bool) -> Result<f64> {
    let mut calibration = Calibration::<f64>::from_json_str(include_str!(
        "../../../configs/robocap_calib_downscale3.json"
    ))?;
    for camera in 0..4 {
        calibration.t_i_c[camera] =
            Se3::new(So3::identity(), [camera as f64 * 0.08, 0.0, 0.0].into());
        calibration.intrinsics[camera] = CameraModel::Pinhole(PinholeParams {
            fx: 300.0,
            fy: 300.0,
            cx: 320.0,
            cy: 180.0,
        });
    }
    if correct_bias {
        calibration.calib_gyro_bias.params[..3].copy_from_slice(&gyro);
        calibration.calib_accel_bias.params[..3].copy_from_slice(&[
            accel[0],
            accel[1],
            accel[2] - 9.81,
        ]);
    }
    let config = VioConfig::from_json_str(include_str!("../../../configs/msdmo_config.json"))?;
    let mut slam = LiveSlam::with_configuration(calibration, config, LiveSlamOptions::default())?;
    let images: Vec<Vec<u8>> = (0..4)
        .map(|camera| {
            (0..640 * 360)
                .map(|i| {
                    // One fixed textured plane at z=3m: 0.08m stereo baseline gives
                    // exactly 8 pixels of disparity at fx=300. This renderer uses no
                    // projection or pose function from the estimator under test.
                    let x = i % 640 + camera * 8;
                    let y = i / 640;
                    let hash = ((x / 5) as u32).wrapping_mul(73856093)
                        ^ ((y / 5) as u32).wrapping_mul(19349663);
                    (40 + hash % 180) as u8
                })
                .collect()
        })
        .collect();
    let mut poses = Vec::new();
    for tick in 0..=1200_i64 {
        let t = 1_000_000_000 + tick * 5_000_000;
        for (channel, xyz) in [(ImuChannel::Gyro, gyro), (ImuChannel::Accel, accel)] {
            if let Some(report) = slam.push(
                SlamInput::Imu {
                    channel,
                    timestamp_ns: t,
                    xyz,
                },
                t,
            )? && let Some(pose) = report.pose
            {
                poses.push(pose);
            }
        }
        if tick % 14 == 7 {
            for (camera, pixels) in images.iter().enumerate() {
                if let Some(report) = slam.push(
                    SlamInput::Frame {
                        camera,
                        timestamp_ns: t - 1_000_000,
                        pixels: pixels.clone(),
                    },
                    t,
                )? && let Some(pose) = report.pose
                {
                    poses.push(pose);
                }
            }
        }
    }
    assert!(
        poses.len() > 60,
        "insufficient visually supported poses: {}",
        poses.len()
    );
    let largest = poses
        .iter()
        .map(|pose| pose[..3].iter().map(|v| v * v).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "known stationary rig: {} poses, max translation {largest} m",
        poses.len()
    );
    Ok(largest)
}

#[test]
fn stationary_dark_input_reports_missing_visual_support_without_fabricating_a_pose() -> Result<()> {
    let mut slam = LiveSlam::cap_a_fast_profile(LiveSlamOptions::default())?;
    let mut reports = Vec::new();
    for tick in 0..=240_i64 {
        let t = 1_000_000_000 + tick * 5_000_000;
        for (channel, xyz) in [
            (ImuChannel::Gyro, [0.0; 3]),
            (ImuChannel::Accel, [0.0, 0.0, 9.81]),
        ] {
            if let Some(report) = slam.push(
                SlamInput::Imu {
                    channel,
                    timestamp_ns: t,
                    xyz,
                },
                t,
            )? {
                reports.push(report);
            }
        }
        if tick % 14 == 7 {
            for camera in 0..4 {
                if let Some(report) = slam.push(
                    SlamInput::Frame {
                        camera,
                        timestamp_ns: t - 1_000_000,
                        pixels: vec![0; 640 * 360],
                    },
                    t,
                )? {
                    reports.push(report);
                }
            }
        }
    }
    assert!(
        reports.len() >= 12,
        "live estimator did not process the supplied frames"
    );
    assert!(reports.iter().all(|report| report.pose.is_none()
        && report.status == robocap_recorder::SlamStatus::NoVisualFeatures));
    assert!(
        reports
            .windows(2)
            .all(|pair| pair[1].timestamp_ns > pair[0].timestamp_ns)
    );
    Ok(())
}

#[test]
fn plain_statuses_keep_the_existing_json_strings() -> Result<()> {
    use robocap_recorder::SlamStatus;
    for (status, name) in [
        (SlamStatus::WaitingForImu, "waiting_for_imu"),
        (SlamStatus::NoVisualFeatures, "no_visual_features"),
        (SlamStatus::TrackingProvisional, "tracking_provisional"),
    ] {
        let json = format!("\"{name}\"");
        assert_eq!(serde_json::to_string(&status)?, json);
        assert_eq!(serde_json::from_str::<SlamStatus>(&json)?, status);
        assert_eq!(status.to_string(), name);
    }
    assert_eq!(
        SlamStatus::Failed("test".into()).to_string(),
        "failed: test"
    );
    Ok(())
}
