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
//! see [`F32_ULP_LIMIT`] for the band the other targets get.

#![allow(clippy::unwrap_used)]

use nalgebra::{Matrix2x4, Matrix4x2, SMatrix, SVector, Vector2, Vector4};
use serde::Deserialize;
use slam_rs::calib::Calibration;
use slam_rs::camera::{
    Camera, CameraEnum, KannalaBrandt4, Pinhole, PinholeRadtan8, UnprojectJacobians,
};

mod common;

const ORACLE: &str = include_str!("fixtures/camera_oracle.json");

/// Agreement with the C++ number, per coefficient, relative to `max(|want|, 1)`.
const TOLERANCE: f64 = 1e-15;

/// The same for unprojections, which run a Newton iteration: on the one shipped
/// calibration whose solve is ill-conditioned (msd-g2 cam2) a last-bit
/// difference in the residual moves the third iterate by 1.2e-15, which is three
/// orders of magnitude tighter than anything downstream can see and still not
/// bit-equality.
const UNPROJECT_TOLERANCE: f64 = 1e-12;

/// What the `f32` pass allows, counted in ULP: bit-exact on `x86_64`, two ULP
/// anywhere else.
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
/// ULP and not an absolute band, because an absolute band is not the limit it
/// claims to be: `2 * f32::EPSILON` on [`assert_within`]'s `max(|want|, 1)`
/// scale is 2.4e-7, which at that 0.236 bearing is sixteen ULP and near zero is
/// unbounded in ULP. [`ulp_check`] is the comparison, and
/// [`the_f32_comparator_counts_ulps`] is the test of it.
const F32_ULP_LIMIT: u64 = if cfg!(target_arch = "x86_64") { 0 } else { 2 };

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

/// Where a finite `f32` sits in the ordered sequence of every `f32`, as an
/// integer: neighbouring representable numbers are one apart, `-0.0` and `+0.0`
/// are one point, and a pair straddling zero counts through it.
fn ordinal(value: f32) -> i64 {
    let bits: u32 = value.to_bits();
    if bits >> 31 == 0 {
        i64::from(bits)
    } else {
        -i64::from(bits & 0x7fff_ffff)
    }
}

/// How many representable `f32` values apart two numbers are — their ULP
/// distance. A non-finite number on either side is `u64::MAX`, which no limit
/// this oracle runs with accepts.
fn ulp_distance(got: f32, want: f32) -> u64 {
    if !got.is_finite() || !want.is_finite() {
        return u64::MAX;
    }
    (ordinal(got) - ordinal(want)).unsigned_abs()
}

/// Every coefficient within `limit` ULP of the C++ `float`, and non-finite
/// exactly where the C++ is non-finite.
///
/// A value rather than an assertion so the comparison itself can be tested; the
/// message names the ULP distance, which is the number a limit is read against.
fn ulp_check(
    limit: u64,
    what: &str,
    actual: &[f32],
    expected: &[Option<f64>],
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{what}: {} coefficients against basalt's {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        match want {
            // Exact: the fixture prints the C++ `float`, so the `f64` it parses
            // as is that `float` widened and narrows back to it.
            Some(want) => {
                let want: f32 = *want as f32;
                let distance: u64 = ulp_distance(*got, want);
                if distance > limit {
                    return Err(format!(
                        "{what}[{index}]: got {got}, basalt says {want}, {distance} ULP apart, limit {limit}"
                    ));
                }
            }
            None => {
                if got.is_finite() {
                    return Err(format!(
                        "{what}[{index}]: got {got}, basalt says a non-finite number"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// [`ulp_check`] as an assertion.
#[track_caller]
fn assert_within_ulps(limit: u64, what: &str, actual: &[f32], expected: &[Option<f64>]) {
    if let Err(message) = ulp_check(limit, what, actual, expected) {
        panic!("{message}");
    }
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
            &common::row_major(&d_proj_d_p3d),
            point.d_proj_d_p3d.as_ref().unwrap(),
        );
        assert_matches(
            &format!("{what}: d_proj_d_param"),
            &common::row_major(&d_proj_d_param),
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
            &common::row_major(&d_p3d_d_proj),
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
        ("kb4_msdmi_cam0", common::calibration_text("msdmi"), 0),
        ("kb4_robocap_cam0", common::calibration_text("robocap"), 0),
        ("kb4_robocap_cam1", common::calibration_text("robocap"), 1),
        ("radtan8_msdmg_cam0", common::calibration_text("msdmg"), 0),
        ("radtan8_msdmg_cam2", common::calibration_text("msdmg"), 2),
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
/// all ten cameras: bit for bit on `x86_64`, within [`F32_ULP_LIMIT`] elsewhere.
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
            assert_within_ulps(
                F32_ULP_LIMIT,
                &format!("{what}: proj"),
                proj.as_slice(),
                &point.proj,
            );

            let mut bearing: Vector4<f32> = Vector4::zeros();
            assert_eq!(
                model.unproject(&proj, &mut bearing),
                point.unproject_valid,
                "{what}: unprojection validity"
            );
            assert_within_ulps(
                F32_ULP_LIMIT,
                &format!("{what}: unproject"),
                bearing.as_slice(),
                &point.unproject,
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 300);
}

/// The comparison the `f32` pass makes, on values whose ULP distance is known.
///
/// The band this replaced was `2 * f32::EPSILON` scaled by `max(|want|, 1)`: at
/// the documented 0.236 bearing that is 2.4e-7 against a 1.5e-8 ULP — sixteen of
/// them — and at 1e-6 it is five orders of magnitude looser still, so a real
/// three-ULP regression passed off the reference host. One and two ULP are what
/// the Mac's own difference is inside; three is a regression, on either side of
/// zero and at any magnitude.
#[test]
fn the_f32_comparator_counts_ulps() {
    // The limit the oracle pass runs with: bit-exact where the fixture was
    // produced, two ULP on a second toolchain.
    #[cfg(target_arch = "x86_64")]
    assert_eq!(F32_ULP_LIMIT, 0);
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(F32_ULP_LIMIT, 2);

    for anchor in [0.236_f32, -0.236, 1e-6, -1e-6, 1.0, -1.0] {
        let want: Vec<Option<f64>> = vec![Some(f64::from(anchor))];
        assert!(
            ulp_check(2, "same", &[anchor], &want).is_ok(),
            "{anchor} against itself"
        );
        for steps in [1_i64, -1, 2, -2] {
            let moved: f32 = step_ulps(anchor, steps);
            assert_eq!(
                ulp_distance(moved, anchor),
                steps.unsigned_abs(),
                "{anchor} moved {steps}"
            );
            assert!(
                ulp_check(2, "inside", &[moved], &want).is_ok(),
                "{anchor} moved {steps} ULP"
            );
            // The x86-64 limit is bit equality, so one ULP already misses it.
            assert!(
                ulp_check(0, "exact", &[moved], &want).is_err(),
                "{anchor} moved {steps} ULP, limit 0"
            );
        }
        for steps in [3_i64, -3] {
            let moved: f32 = step_ulps(anchor, steps);
            let refused: String = ulp_check(2, "outside", &[moved], &want).unwrap_err();
            assert!(
                refused.contains("3 ULP apart, limit 2"),
                "{anchor} moved {steps} ULP: {refused}"
            );
        }
    }

    // Zero: the two signed zeros are one number, and a pair straddling zero
    // counts through it rather than through the whole negative range.
    assert_eq!(ulp_distance(0.0, -0.0), 0);
    assert!(ulp_check(0, "signed zero", &[-0.0], &[Some(0.0)]).is_ok());
    assert_eq!(ulp_distance(step_ulps(0.0, 1), step_ulps(0.0, -1)), 2);
    assert!(
        ulp_check(
            2,
            "across zero",
            &[step_ulps(0.0, 1)],
            &[Some(f64::from(step_ulps(0.0, -1)))]
        )
        .is_ok()
    );
    assert!(
        ulp_check(
            2,
            "across zero",
            &[step_ulps(0.0, 2)],
            &[Some(f64::from(step_ulps(0.0, -2)))]
        )
        .is_err()
    );

    // A non-finite number is not a near miss, in either direction, and is
    // accepted only where the C++ is non-finite too.
    assert!(ulp_check(2, "not a number", &[f32::NAN], &[Some(0.236)]).is_err());
    assert!(ulp_check(2, "infinite", &[f32::INFINITY], &[Some(0.236)]).is_err());
    assert!(ulp_check(2, "finite", &[0.236], &[None]).is_err());
    assert!(ulp_check(0, "both non-finite", &[f32::NAN], &[None]).is_ok());
}

/// `value` moved `steps` representable `f32` values along the number line, the
/// inverse of [`ordinal`]: how the case above builds a number a known ULP away.
fn step_ulps(value: f32, steps: i64) -> f32 {
    let moved: i64 = ordinal(value) + steps;
    if moved >= 0 {
        f32::from_bits(moved as u32)
    } else {
        f32::from_bits((-moved) as u32 | 0x8000_0000)
    }
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
