//! `test_camera.cpp` ported: project/unproject round trips and analytic
//! Jacobians against central finite differences, at `f64` and `f32`.
//!
//! The C++ file is `thirdparty/basalt-headers/test/src/test_camera.cpp` and its
//! finite-difference helper is `test/include/test_utils.h:22-61`. Both are
//! reproduced here down to the tolerances (`test_utils.h:10-20`) and the
//! comparison rule Eigen's `isApprox`/`isZero` implement, so a failure here
//! means what it means in the C++ suite: the analytic Jacobian is not the
//! derivative of the projection beside it.
//!
//! Two families of test run:
//!
//! * **basalt's own test cameras over basalt's own grid**, a literal port. The
//!   radtan8 one needs the `rpmax` that `computeRpmax()` estimates, which this
//!   port does not implement; the number is read from the C++ oracle fixture
//!   instead (`camera_oracle.json`, `radtan8_odyssey_computed_rpmax`).
//! * **the shipped intrinsics** — msdmi (2 kb4), msdmg (4 pinhole-radtan8, each
//!   with its own `rpmax`), robocap (4 kb4) — restricted to points that land on
//!   the sensor. That restriction is not a fudge: at 90 degrees off axis a real
//!   fisheye projects a thousand pixels outside a 960-pixel image, and there
//!   `unproject`'s three Newton steps (`kannala_brandt_camera4.hpp:359`) do not
//!   converge, so neither the round trip nor the unprojection Jacobian holds —
//!   in basalt either. `camera_oracle.rs` covers those points instead, by
//!   agreeing with the C++ number for number.
//!
//! No shipped calibration is a pinhole (EuRoC's is double sphere), so the
//! pinhole cameras are basalt's own test projections (`pinhole_camera.hpp:287`).

#![allow(clippy::unwrap_used)]
// Several constants below are basalt's own literals or a C++ `%.17g` printout,
// carried over exactly even where an f64 does not need every figure. Keeping the
// printout verbatim is what makes them evidence.
#![allow(clippy::excessive_precision)]

use nalgebra::{Matrix2x4, Matrix4x2, SMatrix, SVector, Vector2, Vector4};
use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use slam_rs::calib::Calibration;
use slam_rs::camera::{
    Camera, CameraEnum, KannalaBrandt4, Pinhole, PinholeRadtan8, RigCamera, UnprojectJacobians,
};
use slam_rs::lie::LieScalar;

mod common;

/// `computeRpmax()` for the Odyssey+ test intrinsics, taken from the C++ oracle
/// fixture (`camera_oracle.json`, camera `radtan8_odyssey_computed_rpmax`).
/// basalt's own test camera is constructed with it
/// (`pinhole_radtan8_camera.hpp:102`, `test_camera.cpp:306`).
const ODYSSEY_COMPUTED_RPMAX: f64 = 2.5927503282280915;

/// `TestConstants<Scalar>` (`test/include/test_utils.h:10-20`), plus the serde
/// bounds the calibration reader needs so a sweep can be written once and run at
/// both precisions.
trait TestConstants: LieScalar + Serialize + DeserializeOwned {
    /// Central difference step.
    fn fd_epsilon() -> Self;
    /// Relative tolerance for the comparison.
    fn fd_max_norm() -> Self;
}

impl TestConstants for f64 {
    fn fd_epsilon() -> Self {
        1e-8
    }
    fn fd_max_norm() -> Self {
        1e-3
    }
}

impl TestConstants for f32 {
    fn fd_epsilon() -> Self {
        1e-2
    }
    fn fd_max_norm() -> Self {
        1e-2
    }
}

/// `Eigen::DenseBase::isZero(prec)`: every coefficient is within `prec` of zero.
fn is_zero<S: LieScalar, const R: usize, const C: usize>(m: &SMatrix<S, R, C>, prec: S) -> bool {
    m.iter().all(|value| value.abs() <= prec)
}

/// `Eigen::DenseBase::isApprox(other, prec)`:
/// `(a - b).norm() <= prec * min(a.norm(), b.norm())`.
fn is_approx<S: LieScalar, const R: usize, const C: usize>(
    a: &SMatrix<S, R, C>,
    b: &SMatrix<S, R, C>,
    prec: S,
) -> bool {
    (a - b).norm() <= prec * a.norm().min(b.norm())
}

/// How far apart the `f32` and `f64` projections of one point may be, in pixels.
///
/// Not a constant, and not relative to the pixel either. Every projection ends
/// in `u = fx * m + cx`, and near the principal point those two terms cancel:
/// at `u = -32` on the msd-index calibration the summands are `-501` and `469`,
/// so the last rounding step is an ulp of **501**, not of 32. The bound is
/// therefore eight ulps of `|u| + |c|`, the magnitude the addition actually
/// works at.
///
/// Eight rather than one: `f32` rounds at every step of the Horner chain, the
/// `atan2` promotion and the divisions before it, and the point of this test is
/// that `f32` stays far below the half pixel the KLT tracker cares about — not
/// that it is correctly rounded. At a 960-pixel image edge the bound is 1.3e-3
/// px; at the principal point it is 4.5e-4 px.
fn f32_pixel_bound(value: f64, principal_point: f64) -> f64 {
    8.0 * f64::from(f32::EPSILON) * (value.abs() + principal_point.abs())
}

/// `test_jacobian` (`test/include/test_utils.h:22-61`), with `x0` always zero as
/// every call site in `test_camera.cpp` passes `…::Zero()`.
fn assert_jacobian<S: TestConstants, const R: usize, const C: usize>(
    name: &str,
    analytic: &SMatrix<S, R, C>,
    f: impl Fn(&SVector<S, C>) -> SVector<S, R>,
) {
    let eps: S = S::fd_epsilon();
    let max_norm: S = S::fd_max_norm();

    let mut numeric: SMatrix<S, R, C> = SMatrix::<S, R, C>::zeros();
    for column in 0..C {
        let mut inc: SVector<S, C> = SVector::<S, C>::zeros();
        inc[column] = eps;
        let plus: SVector<S, R> = f(&inc);
        let minus: SVector<S, R> = f(&(-inc));
        numeric.set_column(column, &(plus - minus));
    }
    numeric /= eps + eps;

    assert!(
        analytic.iter().all(|value| value.is_finite()),
        "{name}: analytic Jacobian is not finite:\n{analytic}"
    );
    assert!(
        numeric.iter().all(|value| value.is_finite()),
        "{name}: numeric Jacobian is not finite:\n{numeric}"
    );

    let agrees: bool = if is_zero(&numeric, max_norm) && is_zero(analytic, max_norm) {
        is_zero(&(numeric - analytic), max_norm)
    } else {
        is_approx(&numeric, analytic, max_norm)
    };
    assert!(
        agrees,
        "{name}: analytic and numeric disagree (diff norm {})\nanalytic:\n{analytic}numeric:\n{numeric}",
        (numeric - analytic).norm()
    );
}

/// Which Jacobians a sweep checks.
///
/// `d_proj_d_param` is the calibration optimizer's, and calibration runs in
/// double; the estimator only ever asks for `d_proj_d_p3d`. On the msd-g2
/// intrinsics the parameter columns for `k4, k5, k6` reach 1e4 near the edge of
/// the valid radius, where neither a 1e-2 nor a 1e-3 central difference in `f32`
/// is a derivative any more — so that combination is checked in `f64` and
/// against the C++ fixture, not by finite differences in `f32`.
#[derive(Clone, Copy, PartialEq)]
enum Check {
    /// `d_proj_d_p3d` only.
    Point,
    /// Both Jacobians.
    PointAndParam,
}

/// `testProjectJacobian` (`test_camera.cpp:40-90`): the grid is
/// `x, y in -10..=10`, `z in -1..=5`, homogeneous `w = 1`.
///
/// `domain` narrows the sweep to the pixels a camera can actually produce; the
/// literal port passes `everywhere`.
fn sweep_project_jacobians<S, const N: usize, Cam>(
    camera: &Cam,
    domain: impl Fn(&Vector2<S>) -> bool,
    check: Check,
) where
    S: TestConstants,
    Cam: Camera<S, Params = SVector<S, N>, ParamJacobian = SMatrix<S, 2, N>>,
{
    for x in -10..=10 {
        for y in -10..=10 {
            for z in -1..=5 {
                let point: Vector4<S> = Vector4::new(
                    S::from_literal(f64::from(x)),
                    S::from_literal(f64::from(y)),
                    S::from_literal(f64::from(z)),
                    S::one(),
                );

                let mut proj: Vector2<S> = Vector2::zeros();
                let mut d_proj_d_p3d: Matrix2x4<S> = Matrix2x4::zeros();
                let mut d_proj_d_param: SMatrix<S, 2, N> = SMatrix::zeros();
                let valid: bool = camera.project_with_jacobians(
                    &point,
                    &mut proj,
                    Some(&mut d_proj_d_p3d),
                    Some(&mut d_proj_d_param),
                );
                if !valid || !domain(&proj) {
                    continue;
                }

                assert_jacobian("d_r_d_p", &d_proj_d_p3d, |inc: &Vector4<S>| {
                    let mut res: Vector2<S> = Vector2::zeros();
                    camera.project(&(point + inc), &mut res);
                    res
                });

                if check == Check::PointAndParam {
                    assert_jacobian("d_r_d_param", &d_proj_d_param, |inc: &SVector<S, N>| {
                        let mut perturbed: Cam = *camera;
                        perturbed.apply_inc(inc);
                        let mut res: Vector2<S> = Vector2::zeros();
                        perturbed.project(&point, &mut res);
                        res
                    });
                }
            }
        }
    }
}

/// `testProjectUnproject` (`test_camera.cpp:158-187`): the unprojected bearing is
/// the normalized point, to `epsilonSqrt`. The homogeneous coordinate is
/// `0.23424` on the way in and zero on the way back, which is the C++'s own way
/// of checking that projection ignores it and unprojection zeroes it.
fn sweep_project_unproject<S: TestConstants, Cam: Camera<S>>(
    camera: &Cam,
    domain: impl Fn(&Vector2<S>) -> bool,
) {
    for x in -10..=10 {
        for y in -10..=10 {
            for z in 0..=5 {
                let point: Vector4<S> = Vector4::new(
                    S::from_literal(f64::from(x)),
                    S::from_literal(f64::from(y)),
                    S::from_literal(f64::from(z)),
                    S::from_literal(0.23424),
                );
                let mut normalized: Vector4<S> = Vector4::zeros();
                normalized
                    .fixed_rows_mut::<3>(0)
                    .copy_from(&point.fixed_rows::<3>(0).normalize());

                let mut proj: Vector2<S> = Vector2::zeros();
                if !camera.project(&point, &mut proj) || !domain(&proj) {
                    continue;
                }

                let mut bearing: Vector4<S> = Vector4::zeros();
                camera.unproject(&proj, &mut bearing);
                assert!(
                    is_approx(&normalized, &bearing, S::sophus_epsilon_sqrt()),
                    "normalized {} unprojected {}",
                    normalized.transpose(),
                    bearing.transpose()
                );
            }
        }
    }
}

/// `testUnprojectJacobians` (`test_camera.cpp:190-241`).
fn sweep_unproject_jacobians<S, const N: usize, Cam>(
    camera: &Cam,
    domain: impl Fn(&Vector2<S>) -> bool,
) where
    S: TestConstants,
    Cam: UnprojectJacobians<S, Params = SVector<S, N>, UnprojectParamJacobian = SMatrix<S, 4, N>>,
{
    for x in -10..=10 {
        for y in -10..=10 {
            for z in 0..=5 {
                let point: Vector4<S> = Vector4::new(
                    S::from_literal(f64::from(x)),
                    S::from_literal(f64::from(y)),
                    S::from_literal(f64::from(z)),
                    S::zero(),
                );
                let mut proj: Vector2<S> = Vector2::zeros();
                if !camera.project(&point, &mut proj) || !domain(&proj) {
                    continue;
                }

                let mut bearing: Vector4<S> = Vector4::zeros();
                let mut d_p3d_d_proj: Matrix4x2<S> = Matrix4x2::zeros();
                let mut d_p3d_d_param: SMatrix<S, 4, N> = SMatrix::zeros();
                camera.unproject_with_jacobians(
                    &proj,
                    &mut bearing,
                    Some(&mut d_p3d_d_proj),
                    Some(&mut d_p3d_d_param),
                );

                assert_jacobian("d_r_d_p", &d_p3d_d_proj, |inc: &Vector2<S>| {
                    let mut res: Vector4<S> = Vector4::zeros();
                    camera.unproject(&(proj + inc), &mut res);
                    res
                });

                assert_jacobian("d_r_d_param", &d_p3d_d_param, |inc: &SVector<S, N>| {
                    let mut perturbed: Cam = *camera;
                    perturbed.apply_inc(inc);
                    let mut res: Vector4<S> = Vector4::zeros();
                    perturbed.unproject(&proj, &mut res);
                    res
                });
            }
        }
    }
}

/// `PinholeCamera::getTestProjections()` (`pinhole_camera.hpp:281-295`): EuRoC
/// and TUM VI 512.
fn basalt_pinholes<S: TestConstants>() -> Vec<Pinhole<S>> {
    vec![
        Pinhole::new(SVector::<S, 4>::from([
            S::from_literal(460.76484651566468),
            S::from_literal(459.4051018049483),
            S::from_literal(365.8937161309615),
            S::from_literal(249.33499869752445),
        ])),
        Pinhole::new(SVector::<S, 4>::from([
            S::from_literal(191.14799816648748),
            S::from_literal(191.13150946585135),
            S::from_literal(254.95857715233118),
            S::from_literal(256.8815466235898),
        ])),
    ]
}

/// `KannalaBrandtCamera4::getTestProjections()` (`kannala_brandt_camera4.hpp:487-495`).
fn basalt_kb4<S: TestConstants>() -> KannalaBrandt4<S> {
    KannalaBrandt4::new(SVector::<S, 8>::from([
        S::from_literal(379.045),
        S::from_literal(379.008),
        S::from_literal(505.512),
        S::from_literal(509.969),
        S::from_literal(0.00693023),
        S::from_literal(-0.0013828),
        S::from_literal(-0.000272596),
        S::from_literal(-0.000452646),
    ]))
}

/// `PinholeRadtan8Camera::getTestProjections()` (`pinhole_radtan8_camera.hpp:705-718`),
/// the Odyssey+, with the radius `computeRpmax()` estimates for it.
fn basalt_radtan8<S: TestConstants>() -> PinholeRadtan8<S> {
    PinholeRadtan8::new(
        SVector::<S, 12>::from([
            S::from_literal(269.0600776672363),
            S::from_literal(269.1679859161377),
            S::from_literal(324.3333053588867),
            S::from_literal(245.22674560546875),
            S::from_literal(0.6257319450378418),
            S::from_literal(0.46612036228179932),
            S::from_literal(-0.00018502399325370789),
            S::from_literal(-4.2882973502855748e-5),
            S::from_literal(0.0041795829311013222),
            S::from_literal(0.89431935548782349),
            S::from_literal(0.54253977537155151),
            S::from_literal(0.0662121474742889),
        ]),
        S::from_literal(ODYSSEY_COMPUTED_RPMAX),
    )
}

/// Every kb4 in the shipped calibrations, with the resolution of its images.
fn shipped_kb4<S: TestConstants>() -> Vec<(KannalaBrandt4<S>, RigCamera<S>, f64)> {
    let mut cameras: Vec<(KannalaBrandt4<S>, RigCamera<S>, f64)> = Vec::new();
    for (text, safe_radius) in [
        (common::calibration_text("msdmi"), MSDMI_SAFE_RADIUS),
        (common::calibration_text("robocap"), ROBOCAP_SAFE_RADIUS),
    ] {
        let calibration: Calibration<S> = Calibration::from_json_str(text).unwrap();
        for rig in RigCamera::from_calibration(&calibration).unwrap() {
            match rig.model {
                CameraEnum::Kb4(camera) => cameras.push((camera, rig, safe_radius)),
                other => panic!("expected kb4, got {}", other.name()),
            }
        }
    }
    cameras
}

/// The four msd-g2 cameras, each with the `rpmax` its calibration carries.
fn shipped_radtan8<S: TestConstants>() -> Vec<(PinholeRadtan8<S>, RigCamera<S>, f64)> {
    let calibration: Calibration<S> =
        Calibration::from_json_str(common::calibration_text("msdmg")).unwrap();
    RigCamera::from_calibration(&calibration)
        .unwrap()
        .into_iter()
        .map(|rig| match rig.model {
            CameraEnum::PinholeRadtan8(camera) => (camera, rig, MSDMG_SAFE_RADIUS),
            other => panic!("expected pinhole-radtan8, got {}", other.name()),
        })
        .collect()
}

/// The whole grid, as the C++ tests use it.
fn everywhere<S: LieScalar>(_proj: &Vector2<S>) -> bool {
    true
}

/// basalt's own operational domain for a keypoint: inside the image, and within
/// `optical_flow_image_safe_radius` of the image centre
/// (`frame_to_frame_optical_flow.h:508-511`, which masks the black corners of a
/// fisheye). The radius is a config field, 472 for msd-index, 340 for msd-g2 and
/// 388 for the Odyssey config RoboCap runs
/// (`configs/msdmi_config.json`, `msdmg_config.json`, `msdmo_config.json`).
fn on_sensor<S: LieScalar>(
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

// ─── basalt's own cameras, basalt's own grid ──────────────────────────────

#[test]
fn pinhole_project_jacobians() {
    for camera in basalt_pinholes::<f64>() {
        sweep_project_jacobians(&camera, everywhere, Check::PointAndParam);
    }
    for camera in basalt_pinholes::<f32>() {
        sweep_project_jacobians(&camera, everywhere, Check::PointAndParam);
    }
}

#[test]
fn kb4_project_jacobians() {
    sweep_project_jacobians(&basalt_kb4::<f64>(), everywhere, Check::PointAndParam);
    sweep_project_jacobians(&basalt_kb4::<f32>(), everywhere, Check::PointAndParam);
}

#[test]
fn radtan8_project_jacobians() {
    sweep_project_jacobians(&basalt_radtan8::<f64>(), everywhere, Check::PointAndParam);
    sweep_project_jacobians(&basalt_radtan8::<f32>(), everywhere, Check::PointAndParam);
}

#[test]
fn pinhole_project_unproject() {
    for camera in basalt_pinholes::<f64>() {
        sweep_project_unproject(&camera, everywhere);
    }
    for camera in basalt_pinholes::<f32>() {
        sweep_project_unproject(&camera, everywhere);
    }
}

#[test]
fn kb4_project_unproject() {
    sweep_project_unproject(&basalt_kb4::<f64>(), everywhere);
    sweep_project_unproject(&basalt_kb4::<f32>(), everywhere);
}

#[test]
fn radtan8_project_unproject() {
    sweep_project_unproject(&basalt_radtan8::<f64>(), everywhere);
    sweep_project_unproject(&basalt_radtan8::<f32>(), everywhere);
}

#[test]
fn pinhole_unproject_jacobians() {
    for camera in basalt_pinholes::<f64>() {
        sweep_unproject_jacobians(&camera, everywhere);
    }
    for camera in basalt_pinholes::<f32>() {
        sweep_unproject_jacobians(&camera, everywhere);
    }
}

/// `f64` only, as in the C++: `KannalaBrandtUnprojectJacobiansFloat` is
/// commented out (`test_camera.cpp:401-403`).
#[test]
fn kb4_unproject_jacobians() {
    sweep_unproject_jacobians(&basalt_kb4::<f64>(), everywhere);
}

// ─── the shipped intrinsics, on the sensor ────────────────────────────────

#[test]
fn shipped_kb4_project_jacobians() {
    for (camera, rig, safe_radius) in shipped_kb4::<f64>() {
        sweep_project_jacobians(&camera, on_sensor(&rig, safe_radius), Check::PointAndParam);
    }
    for (camera, rig, safe_radius) in shipped_kb4::<f32>() {
        sweep_project_jacobians(&camera, on_sensor(&rig, safe_radius), Check::PointAndParam);
    }
}

#[test]
fn shipped_radtan8_project_jacobians() {
    for (camera, rig, safe_radius) in shipped_radtan8::<f64>() {
        sweep_project_jacobians(&camera, on_sensor(&rig, safe_radius), Check::PointAndParam);
    }
    for (camera, rig, safe_radius) in shipped_radtan8::<f32>() {
        sweep_project_jacobians(&camera, on_sensor(&rig, safe_radius), Check::Point);
    }
}

#[test]
fn shipped_cameras_project_unproject() {
    for (camera, rig, safe_radius) in shipped_kb4::<f64>() {
        sweep_project_unproject(&camera, on_sensor(&rig, safe_radius));
    }
    for (camera, rig, safe_radius) in shipped_radtan8::<f64>() {
        sweep_project_unproject(&camera, on_sensor(&rig, safe_radius));
    }
}

#[test]
fn shipped_kb4_unproject_jacobians() {
    for (camera, rig, safe_radius) in shipped_kb4::<f64>() {
        sweep_unproject_jacobians(&camera, on_sensor(&rig, safe_radius));
    }
}

// ─── what unprojection actually delivers on real fisheye calibrations ─────

/// Inside basalt's safe radius the shipped calibrations invert to eleven
/// digits — with one exception, which is pinned below.
///
/// This is the property the frontend depends on: `unproject` builds the
/// epipolar guess and filters matches, and a bearing that is off by a degree is
/// a false match. The grid is 41 x 41 x 8 points per camera, coarse enough to
/// run in milliseconds and dense enough to cover the disc.
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
                        if !rig.model.project(&point, &mut proj) || !domain(&proj) {
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

/// msd-g2 cam2 does not invert inside the safe radius.
///
/// The cause is `unproject`'s five Newton steps on a distortion whose radial
/// numerator and denominator both change sign (`k2 = -0.46`, `k5 = -0.59`): the
/// iteration lands on a different pre-image. The pixel is 337 px from the image
/// centre, inside the 340 px safe radius, so nothing in the frontend masks it.
/// It is recorded here because the estimator stage has to decide whether to
/// widen the iteration.
#[test]
fn msd_g2_cam2_does_not_invert_inside_the_safe_radius() {
    let calibration: Calibration<f64> =
        Calibration::from_json_str(common::calibration_text("msdmg")).unwrap();
    let rig: RigCamera<f64> = RigCamera::from_calibration(&calibration).unwrap()[2];

    let point: Vector4<f64> = Vector4::new(-9.1, 7.6, 4.25, 1.0);
    let mut proj: Vector2<f64> = Vector2::zeros();
    assert!(rig.model.project(&point, &mut proj));
    assert!(on_sensor(&rig, MSDMG_SAFE_RADIUS)(&proj));

    let mut bearing: Vector4<f64> = Vector4::zeros();
    assert!(rig.model.unproject(&proj, &mut bearing));
    let mut expected: Vector4<f64> = Vector4::zeros();
    expected
        .fixed_rows_mut::<3>(0)
        .copy_from(&point.fixed_rows::<3>(0).normalize());
    assert!((bearing - expected).norm() > 0.12);
}

/// Outside the safe radius a wide kb4 can invert to a bearing pointing the other
/// way, and again basalt agrees digit for digit. RoboCap cam1 at 511 px from the
/// image centre is 123 px beyond the 388 px radius the config masks with, so the
/// frontend never asks; the estimator stage must keep it that way.
#[test]
fn robocap_cam1_inverts_backwards_outside_the_safe_radius() {
    let calibration: Calibration<f64> =
        Calibration::from_json_str(common::calibration_text("robocap")).unwrap();
    let rig: RigCamera<f64> = RigCamera::from_calibration(&calibration).unwrap()[1];

    let point: Vector4<f64> = Vector4::new(-9.0, -4.3, 1.5, 1.0);
    let mut proj: Vector2<f64> = Vector2::zeros();
    assert!(rig.model.project(&point, &mut proj));
    assert!(rig.in_bounds(&proj, 0.0));
    assert!(!on_sensor(&rig, ROBOCAP_SAFE_RADIUS)(&proj));

    let mut bearing: Vector4<f64> = Vector4::zeros();
    assert!(rig.model.unproject(&proj, &mut bearing));
    let cpp: Vector4<f64> = Vector4::new(
        0.88995195205308286,
        0.42519926598091740,
        -0.16489726270073851,
        0.0,
    );
    assert!(
        (bearing - cpp).norm() < 1e-12,
        "bearing {}",
        bearing.transpose()
    );

    // Not merely inaccurate: the bearing points back the way it came.
    let mut expected: Vector4<f64> = Vector4::zeros();
    expected
        .fixed_rows_mut::<3>(0)
        .copy_from(&point.fixed_rows::<3>(0).normalize());
    assert!(bearing.dot(&expected) < -0.9);
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
    let camera64: CameraEnum<f64> = CameraEnum::Kb4(shipped_kb4::<f64>()[0].0);
    let camera32: CameraEnum<f32> = CameraEnum::Kb4(shipped_kb4::<f32>()[0].0);

    let (x, y, z): (f32, f32, f32) = (-1.1009091, 0.27731937, 0.52727574);
    let mut proj64: Vector2<f64> = Vector2::zeros();
    let mut proj32: Vector2<f32> = Vector2::zeros();
    assert!(camera64.project(
        &Vector4::new(f64::from(x), f64::from(y), f64::from(z), 1.0),
        &mut proj64
    ));
    assert!(camera32.project(&Vector4::new(x, y, z, 1.0), &mut proj32));

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
        let cameras: [(CameraEnum<f64>, RigCamera<f64>, f64); 3] = [
            (
                CameraEnum::Pinhole(basalt_pinholes::<f64>()[0]),
                RigCamera {
                    model: CameraEnum::Pinhole(basalt_pinholes::<f64>()[0]),
                    // `PinholeCamera::getTestResolutions()` (`pinhole_camera.hpp:301`).
                    resolution: [752, 480],
                },
                376.0,
            ),
            (
                CameraEnum::Kb4(shipped_kb4::<f64>()[0].0),
                shipped_kb4::<f64>()[0].1,
                MSDMI_SAFE_RADIUS,
            ),
            (
                CameraEnum::PinholeRadtan8(shipped_radtan8::<f64>()[0].0),
                shipped_radtan8::<f64>()[0].1,
                MSDMG_SAFE_RADIUS,
            ),
        ];
        let (camera, rig, safe_radius): (CameraEnum<f64>, RigCamera<f64>, f64) = cameras[camera_index];
        let point: Vector4<f64> = Vector4::new(x, y, z, 1.0);

        let mut proj: Vector2<f64> = Vector2::zeros();
        prop_assume!(camera.project(&point, &mut proj));
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
        // (`kannala_brandt_camera4.hpp:152`), so the negative-z rejection is
        // only meaningful near the optical axis for that model.
        let pinhole: CameraEnum<f64> = CameraEnum::Pinhole(basalt_pinholes::<f64>()[0]);
        let radtan8: CameraEnum<f64> = CameraEnum::PinholeRadtan8(shipped_radtan8::<f64>()[0].0);
        let kb4: CameraEnum<f64> = CameraEnum::Kb4(shipped_kb4::<f64>()[0].0);

        let point: Vector4<f64> = Vector4::new(x, y, z, 1.0);
        let axial: Vector4<f64> = Vector4::new(0.0, 0.0, z, 1.0);
        let mut proj: Vector2<f64> = Vector2::zeros();

        prop_assert!(!pinhole.project(&point, &mut proj));
        prop_assert!(proj.iter().all(|value| value.is_finite()));
        prop_assert!(!radtan8.project(&point, &mut proj));
        prop_assert!(proj.iter().all(|value| value.is_finite()));
        prop_assert!(!kb4.project(&axial, &mut proj));
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
                CameraEnum::Pinhole(basalt_pinholes::<f64>()[0]),
                CameraEnum::Pinhole(basalt_pinholes::<f32>()[0]),
            ),
            (
                CameraEnum::Kb4(shipped_kb4::<f64>()[0].0),
                CameraEnum::Kb4(shipped_kb4::<f32>()[0].0),
            ),
            (
                CameraEnum::PinholeRadtan8(shipped_radtan8::<f64>()[0].0),
                CameraEnum::PinholeRadtan8(shipped_radtan8::<f32>()[0].0),
            ),
        ] {
            prop_assume!(camera64.project(&point64, &mut proj64));
            prop_assert!(camera32.project(&point32, &mut proj32));
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
