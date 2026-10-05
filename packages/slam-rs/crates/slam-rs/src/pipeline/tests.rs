#![allow(clippy::unwrap_used)]

use super::*;
use crate::VERSION;
use proptest::prelude::*;

fn image(bytes: &[u8], width: usize, height: usize) -> ImageView<'_> {
    ImageView {
        width,
        height,
        stride: width,
        data: bytes,
    }
}

/// MSDMI configuration named by the package manifest.
const MSDMI_CONFIG: &str = include_str!("../../../../configs/msdmi_config.json");

fn pipeline() -> Vio<f32> {
    let directory: std::path::PathBuf =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let config: config::VioConfig = config::VioConfig::from_json_str(MSDMI_CONFIG).unwrap();
    let calibration: calib::Calibration<f64> = calib::Calibration::from_json_str(
        &std::fs::read_to_string(directory.join("msdmi_calib.json")).unwrap(),
    )
    .unwrap();
    Vio::new(
        config,
        calibration,
        frontend::flow::FrontendOptions {
            threads: 1,
            ..frontend::flow::FrontendOptions::default()
        },
    )
    .unwrap()
}

/// Insufficient IMU coverage must return [`VioStatus::NeedMoreImu`] before the
/// frontend advances pyramids, clock or ids. Otherwise a retry tracks the frame
/// against itself. This test checks that nothing moved (D17).
#[test]
fn a_frameset_ahead_of_the_imu_needs_more_imu() {
    let mut vio: Vio<f32> = pipeline();
    let blank: Vec<u8> = vec![0; 960 * 960];
    let views: [ImageView<'_>; 2] = [image(&blank, 960, 960), image(&blank, 960, 960)];
    let before: String = format!("{vio:?}");
    let result: VioResult = vio.track(1_000, &views).unwrap();
    assert_eq!(result.status, VioStatus::NeedMoreImu);
    assert!(!vio.estimator().is_initialized());
    assert!(result.pose.is_none());
    assert_eq!(vio.frontend().frame_counter(), 0);
    // `None` rather than a sentinel: the frontend has seen no frameset.
    assert_eq!(vio.frontend().t_ns(), None);
    assert_eq!(
        format!("{vio:?}"),
        before,
        "the refused frameset moved a field"
    );
}

/// `vio_enforce_realtime` drops framesets, which Offline mode cannot do
/// without letting arrival order reach a decision, so it is refused at
/// construction rather than silently ignored.
#[test]
fn realtime_frame_dropping_is_refused() {
    let directory: std::path::PathBuf =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut config: config::VioConfig = config::VioConfig::from_json_str(MSDMI_CONFIG).unwrap();
    config.vio_enforce_realtime = true;
    let calibration: calib::Calibration<f64> = calib::Calibration::from_json_str(
        &std::fs::read_to_string(directory.join("msdmi_calib.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        Vio::<f32>::new(
            config,
            calibration,
            frontend::flow::FrontendOptions::default()
        )
        .err(),
        Some(VioError::Estimator(
            estimator::EstimatorError::EnforceRealtime
        ))
    );
}

#[test]
fn version_is_the_crate_version() {
    assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
}

#[test]
fn repeated_imu_timestamps_are_rejected() {
    let mut vio: Vio<f32> = pipeline();
    vio.push_imu(5, [0.0; 3], [0.0; 3]).unwrap();
    assert_eq!(
        vio.push_imu(5, [0.0; 3], [0.0; 3]),
        Err(VioError::NonMonotonicImu {
            previous_t_ns: 5,
            t_ns: 5
        })
    );
}

/// A component that is not a number is bad input, not a value to integrate:
/// it reaches both preintegrators and there is nothing that undoes it (D32).
#[test]
fn a_non_finite_imu_sample_is_rejected() {
    let mut vio: Vio<f32> = pipeline();
    assert_eq!(
        vio.push_imu(5, [0.0, f64::NAN, 0.0], [0.0; 3]),
        Err(VioError::NonFiniteImu {
            t_ns: 5,
            field: "gyro"
        })
    );
    assert_eq!(
        vio.push_imu(5, [0.0; 3], [f64::NEG_INFINITY, 0.0, 0.0]),
        Err(VioError::NonFiniteImu {
            t_ns: 5,
            field: "accel"
        })
    );
    // The guard is the only thing that moved, so the same timestamp is
    // still the one the next sample has to take.
    assert!(vio.push_imu(5, [0.0; 3], [0.0; 3]).is_ok());
}

#[test]
fn a_frameset_of_the_wrong_width_is_rejected() {
    let mut vio: Vio<f32> = pipeline();
    let pixels: Vec<u8> = vec![0; 16];
    assert_eq!(
        vio.track(0, &[image(&pixels, 4, 4)]),
        Err(VioError::CameraCountMismatch {
            expected: 2,
            actual: 1
        })
    );
}

/// The geometry of the buffer is checked before a pixel is read, so an
/// under-long or a narrow-stride frameset is a typed refusal rather than the
/// out-of-range read the widening would otherwise make (D32).
#[test]
fn a_short_buffer_is_rejected() {
    let mut vio: Vio<f32> = pipeline();
    let pixels: Vec<u8> = vec![0; 8];
    let short: [ImageView<'_>; 2] = [image(&pixels, 4, 4), image(&pixels, 4, 4)];
    assert_eq!(
        vio.track(0, &short),
        Err(VioError::ShortImage {
            index: 0,
            height: 4,
            stride: 4,
            len: 8
        })
    );
    let narrow: [ImageView<'_>; 2] = [
        ImageView {
            width: 4,
            height: 2,
            stride: 2,
            data: &pixels,
        },
        image(&pixels, 4, 2),
    ];
    assert_eq!(
        vio.track(0, &narrow),
        Err(VioError::StrideTooSmall {
            index: 0,
            width: 4,
            stride: 2,
        })
    );
}

#[test]
fn an_image_whose_size_overflows_is_rejected() {
    let mut vio: Vio<f32> = pipeline();
    let huge: [ImageView<'_>; 2] = [
        ImageView {
            width: 1,
            height: 2,
            stride: 1 << 63,
            data: &[],
        },
        image(&[], 0, 0),
    ];
    assert_eq!(
        vio.track(0, &huge),
        Err(VioError::ImageSizeOverflow {
            index: 0,
            height: 2,
            stride: 1 << 63,
        })
    );
}

proptest! {
    // The pipeline reads two JSON fixtures and builds a frontend per case,
    // so the case count is cut to what the ordering guard needs.
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Whatever the first timestamp is, a second one that does not strictly
    /// follow it is rejected, and the accepted state does not move.
    #[test]
    fn non_monotonic_imu_is_always_rejected(first in -1_000_000i64..1_000_000, back in 0i64..1_000_000) {
        let mut vio: Vio<f32> = pipeline();
        vio.push_imu(first, [0.0; 3], [0.0; 3]).unwrap();
        prop_assert_eq!(
            vio.push_imu(first - back, [0.0; 3], [0.0; 3]),
            Err(VioError::NonMonotonicImu { previous_t_ns: first, t_ns: first - back })
        );
        // The guard is the only thing that moved, so the next in-order
        // sample is still accepted.
        prop_assert!(vio.push_imu(first + 1, [0.0; 3], [0.0; 3]).is_ok());
    }
}
