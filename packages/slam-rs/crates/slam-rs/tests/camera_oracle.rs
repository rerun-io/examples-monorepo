//! The port against basalt's own C++, number for number.
//!
//! `fixtures/camera_oracle.json` is the output of `tools/camera_oracle.cpp` on
//! the fork's `slam-rs-reference` branch (target `basalt_camera_oracle`), built
//! and run out of tree against the basalt headers, since the monorepo never
//! compiles C++ (decision D15). It holds two sections:
//!
//! * `cameras` — ten cameras in `f64` and the same ten in `f32`, thirty points
//!   each: validity, pixel, the unprojected bearing, and in the `f64` pass also
//!   `d_proj_d_p3d`, `d_proj_d_param` and `d_p3d_d_proj` where the model has one.
//! * `probes` — pixels handed straight to `unproject`, including one where the
//!   radtan8 Newton solve meets a **singular** Jacobian. Coefficients the C++
//!   returns as NaN are written as JSON `null`.
//!
//! This is the load-bearing test of the stage. Finite differences (see
//! `camera_jacobians.rs`) prove the analytic Jacobian is the derivative of
//! *this* implementation; only the fixture proves the implementation is
//! basalt's. Both external oracles were rejected: `sophus_sensor` 0.15 and
//! `apex-camera-models` 0.3 each pull nalgebra 0.33 beside the workspace's 0.35
//! (`cargo tree -i nalgebra` on a probe crate), which the brief rules out — and
//! neither would have been bit-exact anyway.
//!
//! The `f64` tolerance is `1e-15` relative, not zero: Rust and C++ evaluate the
//! same expressions in the same order but are free to contract `a * b + c` into
//! an FMA differently. The `f32` pass asks for **exact equality on x86-64**,
//! which is where the fixture was produced and where the two agree bit for bit;
//! see [`F32_TOLERANCE`] for the band the other targets get.

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

/// What the `f32` pass allows: exact on `x86_64`, two ULP anywhere else.
///
/// The fixture is a dump the C++ produced on `x86_64`, so exact equality there
/// is a statement about the port. Elsewhere it is a statement about a second
/// toolchain's rounding: on `aarch64-apple-darwin` one of the three hundred
/// points comes back 1.5e-8 out on a bearing of 0.236 — one ULP at `f32` — from
/// an FMA contracted differently in the Kannala-Brandt Newton step, while the
/// `f64` instantiation and the other four oracle tests pass. Bit parity off the
/// reference host was never the bar (D58, D60: "we never expected bit parity,
/// just close enough"), and two ULP is four orders of magnitude tighter than the
/// centimetre the trajectory gate reads.
///
/// The band is relative to [`assert_within`]'s `max(|want|, 1)` scale, so below
/// unit magnitude it is looser than two ULP of the value itself.
const F32_TOLERANCE: f64 = if cfg!(target_arch = "x86_64") {
    0.0
} else {
    2.0 * f32::EPSILON as f64
};

#[derive(Debug, Deserialize)]
struct Oracle {
    cameras: Vec<OracleCamera>,
    probes: Vec<OracleProbe>,
}

#[derive(Debug, Deserialize)]
struct OracleCamera {
    name: String,
    model: String,
    scalar: String,
    rpmax: f64,
    params: Vec<f64>,
    points: Vec<OraclePoint>,
}

/// `None` in any of these arrays means the C++ returned a non-finite number.
#[derive(Debug, Deserialize)]
struct OraclePoint {
    p: [f64; 4],
    valid: bool,
    proj: Vec<Option<f64>>,
    d_proj_d_p3d: Option<Vec<Option<f64>>>,
    d_proj_d_param: Option<Vec<Option<f64>>>,
    unproject_valid: bool,
    unproject: Vec<Option<f64>>,
    d_p3d_d_proj: Option<Vec<Option<f64>>>,
}

#[derive(Debug, Deserialize)]
struct OracleProbe {
    name: String,
    model: String,
    scalar: String,
    rpmax: f64,
    params: Vec<f64>,
    pixel: [f64; 2],
    valid: bool,
    unproject: Vec<Option<f64>>,
}

fn oracle() -> Oracle {
    serde_json::from_str(ORACLE).unwrap()
}

fn camera(name: &str) -> OracleCamera {
    oracle()
        .cameras
        .into_iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("no oracle camera named {name}"))
}

/// Every coefficient within `tolerance` of the C++ value, and non-finite exactly
/// where the C++ is non-finite.
#[track_caller]
fn assert_within(tolerance: f64, what: &str, actual: &[f64], expected: &[Option<f64>]) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        match want {
            Some(want) => {
                let scale: f64 = want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tolerance * scale,
                    "{what}[{index}]: got {got}, basalt says {want}"
                );
            }
            None => assert!(
                !got.is_finite(),
                "{what}[{index}]: got {got}, basalt says a non-finite number"
            ),
        }
    }
}

/// Every coefficient within [`TOLERANCE`] of the C++ value.
#[track_caller]
fn assert_matches(what: &str, actual: &[f64], expected: &[Option<f64>]) {
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

/// Replay one `f64` oracle camera through the port.
fn check<const N: usize, Cam>(entry: &OracleCamera, camera: &Cam)
where
    Cam: Camera<f64, Params = SVector<f64, N>, ParamJacobian = SMatrix<f64, 2, N>>,
{
    assert_eq!(entry.scalar, "f64");
    assert_eq!(camera.name(), entry.model);
    assert_eq!(camera.params().as_slice(), entry.params.as_slice());

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
            point.d_proj_d_p3d.as_ref().unwrap(),
        );
        assert_matches(
            &format!("{what}: d_proj_d_param"),
            &row_major(&d_proj_d_param),
            point.d_proj_d_param.as_ref().unwrap(),
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
        let expected: &Vec<Option<f64>> = point
            .d_p3d_d_proj
            .as_ref()
            .unwrap_or_else(|| panic!("{} has no unprojection Jacobian", entry.name));
        let proj: Vector2<f64> = Vector2::new(point.proj[0].unwrap(), point.proj[1].unwrap());
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

/// The variant the fixture entry describes, at whichever scalar it was produced
/// in. `f64 -> f32` on the parameters is exact: they were printed from the
/// `float` the C++ held.
fn model_f32(entry_model: &str, params: &[f64], rpmax: f64) -> CameraEnum<f32> {
    let cast = |value: &f64| *value as f32;
    match entry_model {
        "pinhole" => CameraEnum::Pinhole(Pinhole::new(SVector::<f32, 4>::from_iterator(
            params.iter().map(cast),
        ))),
        "kb4" => CameraEnum::Kb4(KannalaBrandt4::new(SVector::<f32, 8>::from_iterator(
            params.iter().map(cast),
        ))),
        "pinhole-radtan8" => CameraEnum::PinholeRadtan8(PinholeRadtan8::new(
            SVector::<f32, 12>::from_iterator(params.iter().map(cast)),
            rpmax as f32,
        )),
        other => panic!("unexpected model {other}"),
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
        "radtan8_singular",
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

/// The `f32` instantiation reproduces the C++ `float` build pixel and bearing on
/// all ten cameras: exactly on `x86_64`, inside [`F32_TOLERANCE`] elsewhere.
///
/// The bearing is the point of this test. Getting the pixel right leaves the
/// unprojection free to drift: it took reproducing Eigen's 2x2 inverse
/// (one reciprocal of the determinant, then a multiply per cofactor,
/// `eigen/Eigen/src/LU/InverseImpl.h:66-83`) rather than nalgebra's
/// divide-each-coefficient to bring msd-g2 cam2 back into line, and only a
/// bearing comparison can see that.
#[test]
fn the_f32_instantiation_matches_the_cpp() {
    let mut checked: usize = 0;
    for entry in oracle().cameras.iter().filter(|e| e.scalar == "f32") {
        let model: CameraEnum<f32> = model_f32(&entry.model, &entry.params, entry.rpmax);
        for (index, point) in entry.points.iter().enumerate() {
            let what = format!("{} point {index}", entry.name);
            let p: Vector4<f32> = Vector4::new(
                point.p[0] as f32,
                point.p[1] as f32,
                point.p[2] as f32,
                point.p[3] as f32,
            );

            let mut proj: Vector2<f32> = Vector2::zeros();
            assert_eq!(
                model.project(&p, &mut proj),
                point.valid,
                "{what}: validity"
            );
            let widened: Vec<f64> = proj.iter().map(|value| f64::from(*value)).collect();
            assert_within(
                F32_TOLERANCE,
                &format!("{what}: proj"),
                &widened,
                &point.proj,
            );

            let mut bearing: Vector4<f32> = Vector4::zeros();
            assert_eq!(
                model.unproject(&proj, &mut bearing),
                point.unproject_valid,
                "{what}: unprojection validity"
            );
            let widened: Vec<f64> = bearing.iter().map(|value| f64::from(*value)).collect();
            assert_within(
                F32_TOLERANCE,
                &format!("{what}: unproject"),
                &widened,
                &point.unproject,
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 300);
}

/// Pixels handed straight to `unproject`, including the singular Newton case.
///
/// `radtan8_singular` is `fx = fy = 100`, `cx = 320`, `cy = 240`, `k4 = 1`: the
/// distortion is `xp / (1 + rp^2)`, whose derivative vanishes at `xp = 1`. Pixel
/// (420, 240) asks for `xp'' = 1`, which that map never reaches — it peaks at
/// 0.5 — so Newton walks onto the singular point. The C++ divides by a zero
/// determinant, produces NaNs and **rejects** the pixel; so does the port. An
/// earlier version of this module stopped the iteration instead and returned
/// success with a bearing that reprojected 50 px away.
#[test]
fn the_probe_pixels_match_the_cpp() {
    let probes: Vec<OracleProbe> = oracle().probes;
    assert_eq!(probes.len(), 6);
    for probe in &probes {
        let pixel64: Vector2<f64> = Vector2::new(probe.pixel[0], probe.pixel[1]);
        let (valid, bearing): (bool, Vec<f64>) = if probe.scalar == "f64" {
            let model: PinholeRadtan8<f64> = PinholeRadtan8::new(
                SVector::<f64, 12>::from_column_slice(&probe.params),
                probe.rpmax,
            );
            let mut bearing: Vector4<f64> = Vector4::zeros();
            let valid: bool = model.unproject(&pixel64, &mut bearing);
            (valid, bearing.iter().copied().collect())
        } else {
            let model: CameraEnum<f32> = model_f32(&probe.model, &probe.params, probe.rpmax);
            let pixel32: Vector2<f32> = Vector2::new(probe.pixel[0] as f32, probe.pixel[1] as f32);
            let mut bearing: Vector4<f32> = Vector4::zeros();
            let valid: bool = model.unproject(&pixel32, &mut bearing);
            (
                valid,
                bearing.iter().map(|value| f64::from(*value)).collect(),
            )
        };

        assert_eq!(valid, probe.valid, "{}: validity", probe.name);
        let tolerance: f64 = if probe.scalar == "f64" {
            UNPROJECT_TOLERANCE
        } else {
            0.0
        };
        assert_within(
            tolerance,
            &format!("{}: unproject", probe.name),
            &bearing,
            &probe.unproject,
        );
    }
}
