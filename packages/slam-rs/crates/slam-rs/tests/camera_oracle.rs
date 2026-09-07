//! The port against basalt's own C++, number for number.
//!
//! `fixtures/camera_oracle.json` is the output of `tools/camera_oracle.cpp` on
//! the fork's `slam-rs-reference` branch (target `basalt_camera_oracle`), built
//! and run out of tree against the basalt headers, since the monorepo never
//! compiles C++ (decision D15). It holds, for nine cameras and thirty points
//! each, everything the C++ returns:
//! validity, pixel, `d_proj_d_p3d`, `d_proj_d_param`, the unprojected bearing
//! and, where the model has one, `d_p3d_d_proj`.
//!
//! This is the load-bearing test of the stage. Finite differences (see
//! `camera_jacobians.rs`) prove the analytic Jacobian is the derivative of
//! *this* implementation; only the fixture proves the implementation is
//! basalt's. Both external oracles were rejected: `sophus_sensor` 0.15 and
//! `apex-camera-models` 0.3 each pull nalgebra 0.33 beside the workspace's 0.35
//! (`cargo tree -i nalgebra` on a probe crate), which the brief rules out — and
//! neither would have been bit-exact anyway.
//!
//! The tolerance is `1e-15` relative, not zero: Rust and C++ evaluate the same
//! expressions in the same order but are free to contract `a * b + c` into an
//! FMA differently.

#![allow(clippy::unwrap_used)]

use nalgebra::{Matrix2x4, Matrix4x2, SMatrix, SVector, Vector2, Vector4};
use serde::Deserialize;
use slam_rs::calib::Calibration;
use slam_rs::camera::{
    Camera, CameraEnum, KannalaBrandt4, Pinhole, PinholeRadtan8, UnprojectJacobians,
};

const ORACLE: &str = include_str!("fixtures/camera_oracle.json");
const MSDMI: &str = include_str!("fixtures/msdmi_calib.json");
const MSDMG: &str = include_str!("fixtures/msdmg_calib.json");
const ROBOCAP: &str = include_str!("fixtures/robocap-basalt-calib.json");

/// Agreement with the C++ number, per coefficient, relative to `max(|want|, 1)`.
const TOLERANCE: f64 = 1e-15;

/// The same for unprojections, which run a Newton iteration: on the one shipped
/// calibration whose solve is ill-conditioned (msd-g2 cam2) a last-bit
/// difference in the residual moves the third iterate by 1.2e-15, which is three
/// orders of magnitude tighter than anything downstream can see and still not
/// bit-equality.
const UNPROJECT_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Deserialize)]
struct OracleCamera {
    name: String,
    model: String,
    rpmax: f64,
    params: Vec<f64>,
    points: Vec<OraclePoint>,
}

#[derive(Debug, Deserialize)]
struct OraclePoint {
    p: [f64; 4],
    valid: bool,
    proj: [f64; 2],
    d_proj_d_p3d: Vec<f64>,
    d_proj_d_param: Vec<f64>,
    unproject_valid: bool,
    unproject: [f64; 4],
    d_p3d_d_proj: Option<Vec<f64>>,
}

fn oracle() -> Vec<OracleCamera> {
    serde_json::from_str(ORACLE).unwrap()
}

fn camera(name: &str) -> OracleCamera {
    oracle()
        .into_iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("no oracle camera named {name}"))
}

/// Every coefficient within `TOLERANCE` relative to the C++ value.
#[track_caller]
fn assert_within(tolerance: f64, what: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale: f64 = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance * scale,
            "{what}[{index}]: got {got}, basalt says {want}"
        );
    }
}

/// Every coefficient within [`TOLERANCE`] of the C++ value.
#[track_caller]
fn assert_matches(what: &str, actual: &[f64], expected: &[f64]) {
    assert_within(TOLERANCE, what, actual, expected);
}

/// Row-major readout of a Jacobian, the layout the fixture writes.
fn row_major<const R: usize, const C: usize>(m: &SMatrix<f64, R, C>) -> Vec<f64> {
    let mut values: Vec<f64> = Vec::with_capacity(R * C);
    for row in 0..R {
        for column in 0..C {
            values.push(m[(row, column)]);
        }
    }
    values
}

/// Replay one oracle camera through the port.
fn check<const N: usize, Cam>(entry: &OracleCamera, camera: &Cam)
where
    Cam: Camera<f64, Params = SVector<f64, N>, ParamJacobian = SMatrix<f64, 2, N>>,
{
    assert_eq!(camera.name(), entry.model);
    assert_matches(
        &format!("{}: params", entry.name),
        camera.params().as_slice(),
        &entry.params,
    );

    for (index, point) in entry.points.iter().enumerate() {
        let what = format!("{} point {index}", entry.name);
        let p: Vector4<f64> = Vector4::from_column_slice(&point.p);

        let mut proj: Vector2<f64> = Vector2::zeros();
        let mut d_proj_d_p3d: Matrix2x4<f64> = Matrix2x4::zeros();
        let mut d_proj_d_param: SMatrix<f64, 2, N> = SMatrix::zeros();
        let valid: bool = camera.project_with_jacobians(
            &p,
            &mut proj,
            Some(&mut d_proj_d_p3d),
            Some(&mut d_proj_d_param),
        );

        assert_eq!(valid, point.valid, "{what}: validity");
        assert_matches(&format!("{what}: proj"), proj.as_slice(), &point.proj);
        assert_matches(
            &format!("{what}: d_proj_d_p3d"),
            &row_major(&d_proj_d_p3d),
            &point.d_proj_d_p3d,
        );
        assert_matches(
            &format!("{what}: d_proj_d_param"),
            &row_major(&d_proj_d_param),
            &point.d_proj_d_param,
        );

        let mut bearing: Vector4<f64> = Vector4::zeros();
        let unproject_valid: bool = camera.unproject(&proj, &mut bearing);
        assert_eq!(
            unproject_valid, point.unproject_valid,
            "{what}: unprojection validity"
        );
        assert_within(
            UNPROJECT_TOLERANCE,
            &format!("{what}: unproject"),
            bearing.as_slice(),
            &point.unproject,
        );
    }
}

/// The same, for the two models whose unprojection basalt differentiates.
fn check_unproject_jacobian<const N: usize, Cam>(entry: &OracleCamera, camera: &Cam)
where
    Cam: UnprojectJacobians<f64, Params = SVector<f64, N>, ParamJacobian = SMatrix<f64, 2, N>>,
{
    for (index, point) in entry.points.iter().enumerate() {
        let expected: &Vec<f64> = point
            .d_p3d_d_proj
            .as_ref()
            .unwrap_or_else(|| panic!("{} has no unprojection Jacobian", entry.name));
        let proj: Vector2<f64> = Vector2::from_column_slice(&point.proj);
        let mut bearing: Vector4<f64> = Vector4::zeros();
        let mut d_p3d_d_proj: Matrix4x2<f64> = Matrix4x2::zeros();
        camera.unproject_with_jacobians(&proj, &mut bearing, Some(&mut d_p3d_d_proj), None);
        assert_matches(
            &format!("{} point {index}: d_p3d_d_proj", entry.name),
            &row_major(&d_p3d_d_proj),
            expected,
        );
    }
}

#[test]
fn pinhole_matches_the_cpp() {
    let entry: OracleCamera = camera("pinhole_euroc");
    let model: Pinhole<f64> = Pinhole::new(SVector::<f64, 4>::from_column_slice(&entry.params));
    check(&entry, &model);
    check_unproject_jacobian(&entry, &model);
}

#[test]
fn kb4_matches_the_cpp() {
    for name in [
        "kb4_basalt_test",
        "kb4_msdmi_cam0",
        "kb4_robocap_cam0",
        "kb4_robocap_cam1",
    ] {
        let entry: OracleCamera = camera(name);
        let model: KannalaBrandt4<f64> =
            KannalaBrandt4::new(SVector::<f64, 8>::from_column_slice(&entry.params));
        check(&entry, &model);
        check_unproject_jacobian(&entry, &model);
    }
}

#[test]
fn radtan8_matches_the_cpp() {
    for name in [
        "radtan8_msdmg_cam0",
        "radtan8_msdmg_cam2",
        "radtan8_odyssey_no_rpmax",
        "radtan8_odyssey_computed_rpmax",
    ] {
        let entry: OracleCamera = camera(name);
        let model: PinholeRadtan8<f64> = PinholeRadtan8::new(
            SVector::<f64, 12>::from_column_slice(&entry.params),
            entry.rpmax,
        );
        check(&entry, &model);
    }
}

/// The generator hard-codes its intrinsics; these assertions are what stop a
/// typo in `tools/camera_oracle.cpp` from quietly weakening the oracle.
#[test]
fn the_oracle_cameras_are_the_shipped_calibrations() {
    let pairs: [(&str, &str, usize); 5] = [
        ("kb4_msdmi_cam0", MSDMI, 0),
        ("kb4_robocap_cam0", ROBOCAP, 0),
        ("kb4_robocap_cam1", ROBOCAP, 1),
        ("radtan8_msdmg_cam0", MSDMG, 0),
        ("radtan8_msdmg_cam2", MSDMG, 2),
    ];
    for (name, text, index) in pairs {
        let entry: OracleCamera = camera(name);
        let calibration: Calibration<f64> = Calibration::from_json_str(text).unwrap();
        let model: CameraEnum<f64> =
            CameraEnum::from_model(&calibration.intrinsics[index]).unwrap();
        let params: Vec<f64> = match model {
            CameraEnum::Pinhole(cam) => cam.params().as_slice().to_vec(),
            CameraEnum::Kb4(cam) => cam.params().as_slice().to_vec(),
            CameraEnum::PinholeRadtan8(cam) => {
                assert_eq!(cam.rpmax(), entry.rpmax, "{name}: rpmax");
                cam.params().as_slice().to_vec()
            }
        };
        assert_eq!(params, entry.params, "{name}: parameters");
    }
}

/// `f32` reproduces the C++ `f64` pixel to well under a hundredth of a pixel
/// wherever the point lands on the sensor, so the frontend's precision costs
/// nothing the KLT tracker can see.
///
/// "On the sensor" is `|u - cx| <= cx` and `|v - cy| <= cy`, which is basalt's
/// own stand-in for an image extent when it has no resolution to hand
/// (`pinhole_radtan8_camera.hpp:196-200`: `w = 2 cx`, `h = 2 cy`). The bound
/// matters: at the extreme fixture points, several image widths outside any
/// sensor, `f32` and `f64` part company by 0.22 px, and no camera in the
/// reference set can see there.
#[test]
fn the_f32_instantiation_tracks_the_cpp_numbers() {
    let mut worst: f64 = 0.0;
    for entry in oracle() {
        let cx: f64 = entry.params[2];
        let cy: f64 = entry.params[3];
        for point in &entry.points {
            if !point.valid {
                continue;
            }
            let on_sensor: bool =
                (point.proj[0] - cx).abs() <= cx && (point.proj[1] - cy).abs() <= cy;
            if !on_sensor {
                continue;
            }
            let p: Vector4<f32> = Vector4::new(
                point.p[0] as f32,
                point.p[1] as f32,
                point.p[2] as f32,
                point.p[3] as f32,
            );
            let mut proj: Vector2<f32> = Vector2::zeros();
            let model: CameraEnum<f32> = match entry.model.as_str() {
                "pinhole" => CameraEnum::Pinhole(Pinhole::new(SVector::<f32, 4>::from_iterator(
                    entry.params.iter().map(|value| *value as f32),
                ))),
                "kb4" => CameraEnum::Kb4(KannalaBrandt4::new(SVector::<f32, 8>::from_iterator(
                    entry.params.iter().map(|value| *value as f32),
                ))),
                "pinhole-radtan8" => CameraEnum::PinholeRadtan8(PinholeRadtan8::new(
                    SVector::<f32, 12>::from_iterator(
                        entry.params.iter().map(|value| *value as f32),
                    ),
                    entry.rpmax as f32,
                )),
                other => panic!("unexpected model {other}"),
            };
            assert!(model.project(&p, &mut proj));
            for axis in 0..2 {
                worst = worst.max((point.proj[axis] - f64::from(proj[axis])).abs());
            }
        }
    }
    assert!(worst < 1e-2, "worst f32-vs-C++ pixel error {worst}");
}
