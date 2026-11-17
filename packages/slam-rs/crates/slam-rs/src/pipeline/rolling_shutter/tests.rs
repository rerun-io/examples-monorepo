#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::calib::{BasaltCamera, PinholeParams};
use crate::types::KeypointId;
use nalgebra::Vector2;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FOCAL_PX: f64 = 500.0;
const READOUT_S: f64 = 0.02;

/// Two pinhole cameras with the principal point at the image centre; camera 1 has `readouts[1]`.
fn calibration(readouts: [Option<f64>; 2]) -> Calibration<f64> {
    let mut calibration =
        Calibration::from_json_str(include_str!("../../../tests/fixtures/msdmg_calib.json"))
            .unwrap();
    let pinhole = BasaltCamera::Pinhole(PinholeParams {
        fx: FOCAL_PX,
        fy: FOCAL_PX,
        cx: 0.5 * f64::from(WIDTH - 1),
        cy: 0.5 * f64::from(HEIGHT - 1),
    });
    calibration.intrinsics = vec![pinhole; 2];
    calibration.resolution = vec![[WIDTH, HEIGHT]; 2];
    calibration.readout_s = readouts.to_vec();
    calibration.vignette.clear();
    calibration
}

fn shutter(readouts: [Option<f64>; 2]) -> RollingShutter {
    RollingShutter::for_calibration(&calibration(readouts))
        .unwrap()
        .expect("a rig with a readout has a correction")
}

/// One keypoint per camera at `pixel`.
fn keypoints(t_ns: i64, pixel: [f32; 2]) -> FlowObservations {
    let mut observations = FlowObservations::new(t_ns, 2);
    for (camera, slot) in observations.cameras.iter_mut().enumerate() {
        slot.insert(KeypointId(camera as u64 + 1), Vector2::from(pixel));
    }
    observations
}

/// A camera with no readout, or a zero one, gets no correction: the estimator sees the frontend's positions.
#[test]
fn a_rig_without_a_readout_has_no_correction() {
    for readouts in [[None, None], [Some(0.0), Some(0.0)], [None, Some(0.0)]] {
        assert!(
            RollingShutter::for_calibration(&calibration(readouts))
                .unwrap()
                .is_none(),
            "{readouts:?}"
        );
    }
    let mut no_readouts = calibration([None, None]);
    no_readouts.readout_s.clear();
    assert!(
        RollingShutter::for_calibration(&no_readouts)
            .unwrap()
            .is_none()
    );
}

/// Row `v` is read `readout * (v - (H - 1) / 2) / H` after the frame time: a turn about the camera's `y` axis
/// leaves the middle row where it is and moves a keypoint on row 0 by `f * tan(omega * dt)` columns (and its row by
/// the `1 / cos` of that turn), with the sign of the readout. A camera without a readout keeps its keypoints.
#[test]
fn each_row_moves_by_the_turn_over_its_own_read_time() {
    let omega: f64 = 1.0;
    let cx: f32 = 0.5 * (WIDTH - 1) as f32;
    let middle: f32 = 0.5 * (HEIGHT - 1) as f32;
    let t_ns: i64 = 1_000_000_000;
    for readout_s in [READOUT_S, -READOUT_S] {
        let mut rolling = shutter([Some(readout_s), None]);
        // The gyro is in the IMU frame: a turn about camera 0's `y` axis.
        let rate_imu: Vector3<f64> =
            calibration([None, None]).t_i_c[0].rotation * Vector3::new(0.0, omega, 0.0);
        rolling.push_gyro(t_ns - 1_000_000, rate_imu);
        rolling.push_gyro(t_ns + 1_000_000, rate_imu);

        let mut at_middle = keypoints(t_ns, [cx, middle]);
        rolling.correct(&mut at_middle, None);
        let moved = at_middle.cameras[0][&KeypointId(1)];
        assert!(
            (moved - Vector2::new(cx, middle)).norm() < 1e-3,
            "{moved:?}"
        );

        let mut at_top = keypoints(t_ns, [cx, 0.0]);
        let raw = at_top.clone();
        rolling.correct(&mut at_top, None);
        let dt: f64 = readout_s * (0.0 - f64::from(middle)) / f64::from(HEIGHT);
        let expected: [f64; 2] = [
            f64::from(cx) + FOCAL_PX * (omega * dt).tan(),
            f64::from(middle) * (1.0 - 1.0 / (omega * dt).cos()),
        ];
        let moved = at_top.cameras[0][&KeypointId(1)];
        assert!(
            (f64::from(moved.x) - expected[0]).abs() < 1e-3
                && (f64::from(moved.y) - expected[1]).abs() < 1e-3,
            "readout {readout_s}: {moved:?}, expected {expected:?}"
        );
        assert!((f64::from(moved.x) - f64::from(cx)).abs() > 4.0);
        assert_eq!(
            at_top.cameras[1], raw.cameras[1],
            "camera 1 states no readout"
        );
        assert_eq!(rolling.counts(), (2, 0));
    }
}

/// The rate of a frame is the gyro interpolated at the frame time, less the bias: two samples 1 ms apart
/// straddle the frame, and the correction is the one the interpolated rate gives. A frame the gyro does not
/// cover on both sides stays raw and is counted.
#[test]
fn the_rate_is_the_interpolated_gyro_at_the_frame_time_less_the_bias() {
    let mut rolling = shutter([Some(READOUT_S); 2]);
    let t_ns: i64 = 1_000_000_000;
    let mut observations = keypoints(t_ns, [100.0, 50.0]);
    let raw = observations.clone();
    rolling.correct(&mut observations, None);
    assert_eq!(observations, raw, "no gyro sample, no correction");
    assert_eq!(rolling.counts(), (0, 1));
    let (a, b) = (Vector3::new(0.4, -0.8, 0.3), Vector3::new(1.0, 0.2, -0.5));
    rolling.push_gyro(t_ns - 250_000, a);
    rolling.correct(&mut observations, None);
    assert_eq!(
        observations, raw,
        "a gyro that stops before the frame is no coverage"
    );
    assert_eq!(rolling.counts(), (0, 2));
    rolling.push_gyro(t_ns + 750_000, b);
    let bias = Vector3::new(0.01, -0.02, 0.03);
    rolling.correct(&mut observations, Some(bias));
    assert_eq!(rolling.counts(), (1, 2));
    let rate = a * 0.75 + b * 0.25 - bias;
    let rig = calibration([Some(READOUT_S); 2]);
    for camera in 0..2 {
        let expected = rolling.cameras[camera]
            .as_ref()
            .unwrap()
            .correct(
                [100.0, 50.0],
                &(rig.t_i_c[camera].rotation.inverse() * rate),
            )
            .unwrap();
        let got = observations.cameras[camera][&KeypointId(camera as u64 + 1)];
        assert!(
            (f64::from(got.x) - expected[0]).abs() < 1e-3
                && (f64::from(got.y) - expected[1]).abs() < 1e-3,
            "camera {camera}: {got:?} vs {expected:?}"
        );
        assert!(
            (got - Vector2::new(100.0, 50.0)).norm() > 0.1,
            "camera {camera} moved"
        );
    }
}

/// Per-frame pushes, window-sized pushes and one push of the whole window give identical corrected
/// observations: the buffer is trimmed by the frames consumed, not by the newest sample.
#[test]
fn the_correction_does_not_depend_on_how_the_imu_is_batched() {
    let frame_ns: i64 = 33_000_000;
    let frames: Vec<i64> = (1..=40).map(|k| k * frame_ns).collect();
    let samples: Vec<(i64, Vector3<f64>)> = (0..=1_400)
        .map(|k| {
            let t = k * 1_000_000;
            let s = t as f64 * 1e-9;
            (
                t,
                Vector3::new((3.0 * s).sin(), 0.5 * (5.0 * s).cos(), 0.2 * s),
            )
        })
        .collect();
    // When each frame is corrected, all samples up to this time have been pushed.
    let schedules: [&dyn Fn(i64) -> i64; 3] = [
        &|t| t + 1_000_000,
        &|t| (t / 300_000_000 + 1) * 300_000_000,
        &|_| i64::MAX,
    ];
    let runs: Vec<Vec<FlowObservations>> = schedules
        .iter()
        .map(|pushed_until| {
            let mut rolling = shutter([Some(READOUT_S); 2]);
            let mut next = 0;
            frames
                .iter()
                .map(|&t| {
                    while next < samples.len() && samples[next].0 <= pushed_until(t) {
                        rolling.push_gyro(samples[next].0, samples[next].1);
                        next += 1;
                    }
                    let mut observations = keypoints(t, [100.0, 50.0]);
                    rolling.correct(&mut observations, None);
                    assert_eq!(rolling.counts().1, 0, "frame {t} is covered");
                    observations
                })
                .collect()
        })
        .collect();
    assert!(
        runs[0]
            != frames
                .iter()
                .map(|&t| keypoints(t, [100.0, 50.0]))
                .collect::<Vec<_>>()
    );
    assert_eq!(runs[0], runs[1], "window-sized pushes");
    assert_eq!(runs[0], runs[2], "one whole-window push");
}

/// The read time of a row is its fraction of the calibrated image height times the readout, not an absolute
/// sensor row: the same rig at half resolution (focal and rows halved, pixel centres mapped by
/// `(v + 0.5) / 2 - 0.5`) moves the same keypoint to the same place.
#[test]
fn the_read_time_is_the_rows_fraction_of_the_calibrated_height() {
    let half = |value: f64| (value + 0.5) / 2.0 - 0.5;
    let mut half_rig = calibration([Some(READOUT_S); 2]);
    half_rig.intrinsics = vec![
        BasaltCamera::Pinhole(PinholeParams {
            fx: 0.5 * FOCAL_PX,
            fy: 0.5 * FOCAL_PX,
            cx: half(0.5 * f64::from(WIDTH - 1)),
            cy: half(0.5 * f64::from(HEIGHT - 1)),
        });
        2
    ];
    half_rig.resolution = vec![[WIDTH / 2, HEIGHT / 2]; 2];
    let full: RollingShutter = shutter([Some(READOUT_S); 2]);
    let halved: RollingShutter = RollingShutter::for_calibration(&half_rig).unwrap().unwrap();
    let omega: Vector3<f64> = Vector3::new(0.3, 1.2, -0.4);
    for pixel in [[100.0, 0.0], [500.0, 120.0], [320.0, 479.0]] {
        let at_full = full.cameras[0]
            .as_ref()
            .unwrap()
            .correct(pixel, &omega)
            .unwrap();
        let at_half = halved.cameras[0]
            .as_ref()
            .unwrap()
            .correct(pixel.map(half), &omega)
            .unwrap();
        assert!((at_full[0] - pixel[0]).abs() > 1.0, "{pixel:?} moves");
        for axis in 0..2 {
            assert!(
                (half(at_full[axis]) - at_half[axis]).abs() < 1e-9,
                "{pixel:?}: full {at_full:?}, half {at_half:?}"
            );
        }
    }
}
