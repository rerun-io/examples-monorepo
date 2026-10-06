//! Shipped-calibration round trips and the homogeneous SLAM point-Jacobian boundary.
#![allow(clippy::unwrap_used, clippy::excessive_precision)]
use kornia_staging_algebra::Scalar;
use kornia_staging_3d::camera::{CameraModelKind, Pinhole};
use nalgebra::{Matrix2x4, Vector2, Vector4};
use proptest::prelude::*;
use serde::{Serialize, de::DeserializeOwned};
use slam_rs::calib::Calibration;
use slam_rs::camera::{SlamCamera, RigCamera};
mod common;

trait TestConstants: Scalar + Serialize + DeserializeOwned {}
impl TestConstants for f32 {}
impl TestConstants for f64 {}

fn basalt_pinholes<S: TestConstants>() -> Vec<SlamCamera<S>> {
    [
        [
            460.76484651566468,
            459.4051018049483,
            365.8937161309615,
            249.33499869752445,
        ],
        [
            191.14799816648748,
            191.13150946585135,
            254.95857715233118,
            256.8815466235898,
        ],
    ]
    .map(|params| SlamCamera {
        inner: CameraModelKind::Pinhole(Pinhole::new(params.map(S::from_literal)).unwrap()),
    })
    .to_vec()
}

fn shipped<S: TestConstants>(name: &str, radius: f64) -> Vec<(SlamCamera<S>, RigCamera<S>, f64)> {
    let calibration: Calibration<S> =
        Calibration::from_json_str(common::calibration_text(name)).unwrap();
    RigCamera::from_calibration(&calibration)
        .unwrap()
        .into_iter()
        .map(|rig| (rig.model, rig, radius))
        .collect()
}
fn shipped_kb4<S: TestConstants>() -> Vec<(SlamCamera<S>, RigCamera<S>, f64)> {
    shipped("msdmi", MSDMI_SAFE_RADIUS)
        .into_iter()
        .chain(shipped("robocap", ROBOCAP_SAFE_RADIUS))
        .collect()
}
fn shipped_radtan8<S: TestConstants>() -> Vec<(SlamCamera<S>, RigCamera<S>, f64)> {
    shipped("msdmg", MSDMG_SAFE_RADIUS)
}
fn f32_pixel_bound(value: f64, principal_point: f64) -> f64 {
    8.0 * f64::from(f32::EPSILON) * (value.abs() + principal_point.abs())
}
fn on_sensor<S: Scalar>(
    rig: &RigCamera<S>,
    safe_radius: f64,
) -> impl Fn(&Vector2<S>) -> bool + use<'_, S> {
    move |proj: &Vector2<S>| {
        let centre: Vector2<S> = Vector2::new(
            S::from_literal(f64::from(rig.width()) / 2.0),
            S::from_literal(f64::from(rig.height()) / 2.0),
        );
        rig.in_bounds(proj, S::zero()) && (proj - centre).norm() <= S::from_literal(safe_radius)
    }
}

/// `optical_flow_image_safe_radius` per shipped calibration.
const MSDMI_SAFE_RADIUS: f64 = 472.0;
const MSDMG_SAFE_RADIUS: f64 = 340.0;
const ROBOCAP_SAFE_RADIUS: f64 = 388.0;

/// Within safe radii, shipped calibrations should invert accurately except for
/// the explicit regression cases below. Bearing errors affect both epipolar
/// prediction and filtering, even when projected pixels still look plausible.
#[test]
fn the_round_trip_is_exact_inside_the_safe_radius() {
    let mut worst_by_camera: Vec<(String, f64)> = Vec::new();
    for (label, text, safe_radius) in [
        (
            "msdmi",
            common::calibration_text("msdmi"),
            MSDMI_SAFE_RADIUS,
        ),
        (
            "msdmg",
            common::calibration_text("msdmg"),
            MSDMG_SAFE_RADIUS,
        ),
        (
            "robocap",
            common::calibration_text("robocap"),
            ROBOCAP_SAFE_RADIUS,
        ),
    ] {
        let calibration: Calibration<f64> = Calibration::from_json_str(text).unwrap();
        for (index, rig) in RigCamera::from_calibration(&calibration)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            let domain = on_sensor(&rig, safe_radius);
            let mut worst: f64 = 0.0;
            for x in -20..=20 {
                for y in -20..=20 {
                    for z in 1..=8 {
                        let point: Vector4<f64> = Vector4::new(
                            f64::from(x) / 4.0,
                            f64::from(y) / 4.0,
                            f64::from(z) / 2.0,
                            1.0,
                        );
                        let mut proj: Vector2<f64> = Vector2::zeros();
                        if !rig.model.project_point(&point, &mut proj, None) || !domain(&proj) {
                            continue;
                        }
                        let mut bearing: Vector4<f64> = Vector4::zeros();
                        rig.model.unproject(&proj, &mut bearing);
                        let mut expected: Vector4<f64> = Vector4::zeros();
                        expected
                            .fixed_rows_mut::<3>(0)
                            .copy_from(&point.fixed_rows::<3>(0).normalize());
                        worst = worst.max((bearing - expected).norm());
                    }
                }
            }
            worst_by_camera.push((format!("{label} cam{index}"), worst));
        }
    }

    for (name, worst) in &worst_by_camera {
        // msd-g2 cam2 is the exception: see the test below.
        let bound: f64 = if name == "msdmg cam2" { 0.2 } else { 1e-8 };
        assert!(
            *worst <= bound,
            "{name}: worst round-trip bearing error {worst:e} exceeds {bound:e}"
        );
    }
    assert_eq!(worst_by_camera.len(), 10);
}

/// msd-g2 cam2 has multiple Brown pre-images inside its declared valid radius.
/// A converged inverse must reproject even when it selects another pre-image.
#[test]
fn msd_g2_cam2_robust_inverse_inside_the_safe_radius() {
    let calibration: Calibration<f64> =
        Calibration::from_json_str(common::calibration_text("msdmg")).unwrap();
    let rig: RigCamera<f64> = RigCamera::from_calibration(&calibration).unwrap()[2];

    let point: Vector4<f64> = Vector4::new(-9.1, 7.6, 4.25, 1.0);
    let mut proj: Vector2<f64> = Vector2::zeros();
    assert!(rig.model.project_point(&point, &mut proj, None));
    assert!(on_sensor(&rig, MSDMG_SAFE_RADIUS)(&proj));

    let mut bearing: Vector4<f64> = Vector4::zeros();
    assert!(rig.model.unproject(&proj, &mut bearing));
    let mut expected: Vector4<f64> = Vector4::zeros();
    expected
        .fixed_rows_mut::<3>(0)
        .copy_from(&point.fixed_rows::<3>(0).normalize());
    // This Brown calibration has multiple roots inside rpmax. The inverse
    // selects the root reached from the normalized pixel, and must reproject.
    let mut reprojected = Vector2::zeros();
    assert!(rig.model.project_point(&bearing, &mut reprojected, None));
    assert!((reprojected - proj).norm() < 1e-6);
    assert!((bearing - expected).norm() > 0.12);
}

/// Pin the selected off-axis unprojection result where fixed Newton iterations
/// leave a bearing error, independently of the ordinary sensor-domain sweep.
#[test]
fn robocap_front_left_robust_inverse_outside_the_safe_radius() {
    let calibration: Calibration<f64> =
        Calibration::from_json_str(common::calibration_text("robocap")).unwrap();
    let rig: RigCamera<f64> = RigCamera::from_calibration(&calibration).unwrap()[0];

    let point: Vector4<f64> = Vector4::new(-9.0, -4.3, 1.5, 1.0);
    let mut proj: Vector2<f64> = Vector2::zeros();
    assert!(rig.model.project_point(&point, &mut proj, None));
    assert!(rig.in_bounds(&proj, 0.0));
    assert!(!on_sensor(&rig, ROBOCAP_SAFE_RADIUS)(&proj));

    let mut bearing: Vector4<f64> = Vector4::zeros();
    assert!(rig.model.unproject(&proj, &mut bearing));
    assert!(bearing.iter().all(|value| value.is_finite()));
    assert!((bearing.norm() - 1.0).abs() < 1e-12);
    let mut reprojected = Vector2::zeros();
    assert!(rig.model.project_point(&bearing, &mut reprojected, None));
    // The robust inverse also round-trips this pixel outside the old safe radius.
    assert!((reprojected - proj).norm() < 1e-6);

    // The recovered bearing points in the original direction.
    let mut expected: Vector4<f64> = Vector4::zeros();
    expected
        .fixed_rows_mut::<3>(0)
        .copy_from(&point.fixed_rows::<3>(0).normalize());
    assert!(bearing.dot(&expected) > 1.0 - 1e-12);
}

/// The case that made the old bound flaky, pinned.
///
/// A fresh proptest seed found `(-1.1009091, 0.27731937, 0.52727574)` on
/// msd-index cam0: the `f64` pixel is -32.09515 and the `f32` one -32.09503, so
/// the two differ by 1.2e-4 px. The old bound floored at 1e-4 px, which that
/// point clears by 20 percent — not because `f32` is unusually bad there, but
/// because a pixel near the principal point is a small difference of two numbers
/// around 500, and 1.2e-4 is two ulps of 500. The bound now says so.
#[test]
fn a_pixel_near_the_principal_point_still_agrees_between_the_precisions() {
    let camera64: SlamCamera<f64> = shipped_kb4::<f64>()[0].0;
    let camera32: SlamCamera<f32> = shipped_kb4::<f32>()[0].0;

    let (x, y, z): (f32, f32, f32) = (-1.1009091, 0.27731937, 0.52727574);
    let mut proj64: Vector2<f64> = Vector2::zeros();
    let mut proj32: Vector2<f32> = Vector2::zeros();
    assert!(camera64.project_point(
        &Vector4::new(f64::from(x), f64::from(y), f64::from(z), 1.0),
        &mut proj64,
        None
    ));
    assert!(camera32.project_point(&Vector4::new(x, y, z, 1.0), &mut proj32, None));

    let principal_point: [f64; 4] = camera64.focal_and_principal_point();
    let difference: f64 = (proj64[0] - f64::from(proj32[0])).abs();
    assert!(
        difference > 1e-4,
        "this case is only a regression while it exceeds the old floor: {difference:e}"
    );
    assert!(
        difference < f32_pixel_bound(proj64[0], principal_point[2]),
        "f64 {} f32 {}",
        proj64[0],
        proj32[0]
    );
    // And the reason: the sum that produced it works at the scale of the
    // principal point, some fifteen times the pixel itself.
    assert!(proj64[0].abs() < 33.0);
    assert!(principal_point[2] > 469.0);
}

// ─── properties ───────────────────────────────────────────────────────────

proptest! {
    /// Unprojecting a projection returns the direction of the point.
    #[test]
    fn unproject_inverts_project(
        x in -3.0f64..3.0,
        y in -3.0f64..3.0,
        z in 0.3f64..8.0,
        camera_index in 0usize..3,
    ) {
        let cameras: [(SlamCamera<f64>, RigCamera<f64>, f64); 3] = [
            (
                basalt_pinholes::<f64>()[0],
                RigCamera {
                    model: basalt_pinholes::<f64>()[0],
                    // `PinholeCamera::getTestResolutions()`.
                    resolution: [752, 480],
                },
                376.0,
            ),
            (
                shipped_kb4::<f64>()[0].0,
                shipped_kb4::<f64>()[0].1,
                MSDMI_SAFE_RADIUS,
            ),
            (
                shipped_radtan8::<f64>()[0].0,
                shipped_radtan8::<f64>()[0].1,
                MSDMG_SAFE_RADIUS,
            ),
        ];
        let (camera, rig, safe_radius): (SlamCamera<f64>, RigCamera<f64>, f64) = cameras[camera_index];
        let point: Vector4<f64> = Vector4::new(x, y, z, 1.0);

        let mut proj: Vector2<f64> = Vector2::zeros();
        prop_assume!(camera.project_point(&point, &mut proj, None));
        prop_assume!(on_sensor(&rig, safe_radius)(&proj));

        let mut bearing: Vector4<f64> = Vector4::zeros();
        prop_assert!(camera.unproject(&proj, &mut bearing));

        let mut expected: Vector4<f64> = Vector4::zeros();
        expected.fixed_rows_mut::<3>(0).copy_from(&point.fixed_rows::<3>(0).normalize());
        prop_assert!(
            (bearing - expected).norm() < 1e-8,
            "bearing {} expected {}", bearing.transpose(), expected.transpose()
        );
        prop_assert!((bearing.norm() - 1.0).abs() < 1e-12);
        prop_assert_eq!(bearing[3], 0.0);
    }

    /// A point behind the camera is rejected, and the pixel it writes is still a
    /// number: the caller reads the flag, but nothing downstream sees a NaN.
    #[test]
    fn points_behind_the_camera_are_rejected(
        x in -3.0f64..3.0,
        y in -3.0f64..3.0,
        z in -8.0f64..-0.1,
    ) {
        // kb4 accepts points behind its own plane whenever the radius is large
        // so the negative-z rejection is
        // only meaningful near the optical axis for that model.
        let pinhole: SlamCamera<f64> = basalt_pinholes::<f64>()[0];
        let radtan8: SlamCamera<f64> = shipped_radtan8::<f64>()[0].0;
        let kb4: SlamCamera<f64> = shipped_kb4::<f64>()[0].0;

        let point: Vector4<f64> = Vector4::new(x, y, z, 1.0);
        let axial: Vector4<f64> = Vector4::new(0.0, 0.0, z, 1.0);
        let mut proj: Vector2<f64> = Vector2::zeros();

        prop_assert!(!pinhole.project_point(&point, &mut proj, None));
        prop_assert!(proj.iter().all(|value| value.is_finite()));
        prop_assert!(!radtan8.project_point(&point, &mut proj, None));
        prop_assert!(proj.iter().all(|value| value.is_finite()));
        prop_assert!(!kb4.project_point(&axial, &mut proj, None));
        prop_assert!(proj.iter().all(|value| value.is_finite()));
    }

    /// The `f32` instantiation projects where the `f64` one does, to within
    /// [`f32_pixel_bound`], for all three models.
    ///
    /// Measured over a 481 x 481 x 20 grid of the drawn domain, the worst
    /// difference sits at 0.11 of the bound for pinhole, 0.37 for kb4 and 0.44
    /// for pinhole-radtan8 — the rational distortion is the least accurate of
    /// the three in `f32`, but only by a factor of four, not the order of
    /// magnitude a separate bound for it once implied.
    #[test]
    fn f32_and_f64_agree_on_in_domain_points(
        x in -1.2f32..1.2,
        y in -1.2f32..1.2,
        z in 0.5f32..5.0,
    ) {
        // Drawn in f32 and widened, so both runs see exactly the same point.
        let point64: Vector4<f64> = Vector4::new(f64::from(x), f64::from(y), f64::from(z), 1.0);
        let point32: Vector4<f32> = Vector4::new(x, y, z, 1.0);

        let mut proj64: Vector2<f64> = Vector2::zeros();
        let mut proj32: Vector2<f32> = Vector2::zeros();

        for (camera64, camera32) in [
            (
                basalt_pinholes::<f64>()[0],
                basalt_pinholes::<f32>()[0],
            ),
            (
                shipped_kb4::<f64>()[0].0,
                shipped_kb4::<f32>()[0].0,
            ),
            (
                shipped_radtan8::<f64>()[0].0,
                shipped_radtan8::<f32>()[0].0,
            ),
        ] {
            prop_assume!(camera64.project_point(&point64, &mut proj64, None));
            prop_assert!(camera32.project_point(&point32, &mut proj32, None));
            let principal_point: [f64; 4] = camera64.focal_and_principal_point();
            for axis in 0..2 {
                let bound: f64 = f32_pixel_bound(proj64[axis], principal_point[2 + axis]);
                prop_assert!(
                    (proj64[axis] - f64::from(proj32[axis])).abs() < bound,
                    "{} axis {axis}: f64 {} f32 {} (bound {bound:e})",
                    camera64.name(), proj64[axis], proj32[axis]
                );
            }
        }
    }
}

#[test]
fn homogeneous_point_jacobians_match_finite_differences() {
    for (camera, _, _) in shipped::<f64>("msdmg", MSDMG_SAFE_RADIUS)
        .into_iter()
        .chain(shipped_kb4::<f64>())
        .chain(basalt_pinholes::<f64>().into_iter().map(|camera| {
            (
                camera,
                RigCamera {
                    model: camera,
                    resolution: [752, 480],
                },
                376.0,
            )
        }))
    {
        for point in [
            Vector4::new(0.1, -0.2, 1.3, 1.0),
            Vector4::new(-0.4, 0.3, 2.1, 0.2),
        ] {
            let mut pixel = Vector2::zeros();
            let mut jacobian = Matrix2x4::zeros();
            assert!(camera.project_point(&point, &mut pixel, Some(&mut jacobian)));
            for col in 0..4 {
                let mut plus = point;
                let mut minus = point;
                plus[col] += 1e-6;
                minus[col] -= 1e-6;
                let mut a = Vector2::zeros();
                let mut b = Vector2::zeros();
                assert!(camera.project_point(&plus, &mut a, None));
                assert!(camera.project_point(&minus, &mut b, None));
                assert!(((a - b) / 2e-6 - jacobian.column(col)).norm() < 1e-5);
            }
        }
    }
}
