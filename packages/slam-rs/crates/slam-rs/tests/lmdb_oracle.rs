//! The landmark parameterisation, the reprojection residual and triangulation
//! against basalt's own C++, number for number.
//!
//! `fixtures/lmdb/lmdb_oracle.json` is the output of `tools/lmdb_oracle.cpp` on
//! the fork's `slam-rs-reference` branch (target `basalt_lmdb_oracle`), built
//! and run out of tree against the basalt sources, since the monorepo never
//! compiles C++ (decision D15). It holds three sections, each emitted twice —
//! once in `f64`, once in `f32`:
//!
//! * `stereographic` — twelve 3-D points through `StereographicParam::project`
//!   with its 2x4 Jacobian, and the resulting chart coordinate back through
//!   `unproject` with its 4x2 Jacobian. The last three points are on the
//!   equator and at the projection point, where `norm` degenerates.
//! * `linearize_point` — five configurations per camera for the two shipped
//!   reference models (msd-index cam0, kb4; msd-g2 cam0, pinhole-radtan8):
//!   a zero residual, a displaced observation, a landmark at infinity
//!   (`inv_dist = 1e-7`), a close one (`inv_dist = 2.5`), and one behind the
//!   camera. The inputs are plain numbers — the direction, the inverse distance,
//!   the 4x4 `T_t_h` and the observation — so nothing has to be reconstructed
//!   through a group operation and the comparison is on the residual path alone.
//! * `triangulate` — ten DLT cases, four of them deliberately on basalt's
//!   `0 < inv_dist < 3` acceptance gate (`sqrt_keypoint_vio.cpp:534`): a point
//!   exactly 1/3 m away, one just inside, one just outside, and a sub-millimetre
//!   baseline. The fixture records the `accepted` flag as well as the vector, so
//!   the port is checked on the decision, not only on the numbers.
//!
//! This is the load-bearing test of the stage. The finite-difference tests in
//! `src/ba_base.rs` prove the analytic Jacobians are the derivative of *this*
//! implementation; only the fixture proves the implementation is basalt's.
//!
//! Tolerances. The `f64` pass asks for `1e-12` relative, and the `f32` pass for
//! **exact equality** on the stereographic chart and on `linearize_point`, where
//! the port and the C++ evaluate the same expressions in the same order.
//! `triangulate` is the exception in both precisions: it runs a whole Jacobi SVD
//! sweep, and the port's is a step-for-step reimplementation of Eigen's rather
//! than Eigen itself, so one rotation applied in a different grouping moves the
//! null vector by an ulp. It is bit-equal on nine of the ten cases in `f64` and
//! within `1e-15` on the tenth, and within `1e-7` in `f32`. What must not move is
//! basalt's `accepted` decision, and that is asserted exactly, on four cases
//! placed on the gate on purpose.
//!
//! Two ulp-level findings came out of this fixture and changed the port:
//! `So3 * Vector3` now sums Sophus's three terms in Sophus's order rather than
//! nalgebra's (`lie.rs`), and every `head<3>().norm()` on the residual path sums
//! in Eigen's `a0 + (a1 + a2)` unroller order (`landmark::eigen_norm3`). Without
//! the first, `triangulate` was an ulp off in `f64`; without the second, `proj[2]`
//! for a landmark at `inv_dist = 1e-7` was an ulp off in `f32`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use nalgebra::{
    Matrix2x3, Matrix2x4, Matrix2x6, Matrix4, Matrix4x2, Quaternion, UnitQuaternion, Vector2,
    Vector3, Vector4,
};
use serde::Deserialize;
use slam_rs::ba_base::{LinearizePointOut, linearize_point, triangulate};
use slam_rs::calib::Calibration;
use slam_rs::camera::{CameraEnum, KannalaBrandt4, PinholeRadtan8};
use slam_rs::landmark::{Landmark, StereographicParam};
use slam_rs::lie::{LieScalar, Se3, So3};
use slam_rs::types::{LandmarkId, TimeCamId};

const ORACLE: &str = include_str!("fixtures/lmdb/lmdb_oracle.json");
const MSDMI: &str = include_str!("fixtures/msdmi_calib.json");
const MSDMG: &str = include_str!("fixtures/msdmg_calib.json");

/// Agreement with the C++ number, per coefficient, relative to `max(|want|, 1)`.
const TOLERANCE: f64 = 1e-12;

/// The same for the DLT, whose Jacobi sweep is a reimplementation rather than a
/// call into Eigen; see the module docs. Nine of the ten cases are bit-equal;
/// the `rotated` one differs by one ulp in the first coefficient.
const TRIANGULATE_TOLERANCE_F64: f64 = 1e-15;

/// And in `f32`, where the sweep accumulates its rotations in single precision.
const TRIANGULATE_TOLERANCE_F32: f64 = 1e-7;

#[derive(Debug, Deserialize)]
struct Oracle {
    stereographic: Vec<OracleStereographic>,
    linearize_point: Vec<OracleLinearize>,
    triangulate: Vec<OracleTriangulate>,
}

#[derive(Debug, Deserialize)]
struct OracleStereographic {
    scalar: String,
    p3d: Vec<f64>,
    project: Vec<Option<f64>>,
    d_project_d_p3d: Vec<Option<f64>>,
    unproject: Vec<Option<f64>>,
    d_unproject_d_proj: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize)]
struct OracleLinearize {
    name: String,
    model: String,
    scalar: String,
    params: Vec<f64>,
    direction: Vec<f64>,
    inv_dist: f64,
    #[serde(rename = "T_t_h")]
    t_t_h: Vec<f64>,
    kpt_obs: Vec<f64>,
    valid: bool,
    res: Vec<Option<f64>>,
    d_res_d_xi: Vec<Option<f64>>,
    d_res_d_p: Vec<Option<f64>>,
    proj: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize)]
struct OracleTriangulate {
    name: String,
    scalar: String,
    f0: Vec<f64>,
    f1: Vec<f64>,
    #[serde(rename = "T_0_1_quaternion_xyzw")]
    quaternion_xyzw: Vec<f64>,
    #[serde(rename = "T_0_1_translation")]
    translation: Vec<f64>,
    p2: Vec<f64>,
    result: Vec<Option<f64>>,
    accepted: bool,
}

fn oracle() -> Oracle {
    serde_json::from_str(ORACLE).unwrap()
}

/// Compare one coefficient against the C++ number.
///
/// A `None` is what the generator writes for a coefficient the C++ produced as
/// NaN or infinity — JSON has no such literal — and the port must be non-finite
/// there too. Otherwise `tolerance == 0.0` means bit equality, which is what the
/// `f32` passes ask for, and any other bound is relative to `max(|want|, 1)`, so
/// a coefficient that is legitimately zero is compared absolutely.
fn close(label: &str, got: f64, want: Option<f64>, tolerance: f64) {
    let Some(want) = want else {
        assert!(!got.is_finite(), "{label}: got {got}, want non-finite");
        return;
    };
    if tolerance == 0.0 {
        assert_eq!(got, want, "{label}: not bit-equal");
        return;
    }
    let bound: f64 = tolerance * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= bound,
        "{label}: got {got}, want {want} (bound {bound})"
    );
}

fn close_all<S: LieScalar>(label: &str, got: &[S], want: &[Option<f64>], tolerance: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        close(&format!("{label}[{i}]"), a.to_f64(), *b, tolerance);
    }
}

/// The same for an input vector, which is always finite.
fn close_all_finite<S: LieScalar>(label: &str, got: &[S], want: &[f64], tolerance: f64) {
    let want: Vec<Option<f64>> = want.iter().map(|v| Some(*v)).collect();
    close_all(label, got, &want, tolerance);
}

/// Bit-for-bit equality, which `assert_eq!` cannot express for NaN.
fn same_bits<S: LieScalar>(a: &[S], b: &[S]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_f64().to_bits() == y.to_f64().to_bits())
}

/// A coefficient the fixture records as finite.
fn finite(values: &[Option<f64>], i: usize) -> f64 {
    values[i].expect("a finite coefficient")
}

/// `f32` in the fixture is printed as the exact decimal of its own bit pattern,
/// so reading it back as `f64` and narrowing is lossless.
fn scalar<S: LieScalar>(value: f64) -> S {
    S::from_literal(value)
}

fn vector2<S: LieScalar>(values: &[f64]) -> Vector2<S> {
    Vector2::new(scalar(values[0]), scalar(values[1]))
}

fn vector3<S: LieScalar>(values: &[f64]) -> Vector3<S> {
    Vector3::new(scalar(values[0]), scalar(values[1]), scalar(values[2]))
}

fn vector4<S: LieScalar>(values: &[f64]) -> Vector4<S> {
    Vector4::new(
        scalar(values[0]),
        scalar(values[1]),
        scalar(values[2]),
        scalar(values[3]),
    )
}

/// The 4x4 the fixture stores row major.
fn matrix4<S: LieScalar>(values: &[f64]) -> Matrix4<S> {
    let mut m: Matrix4<S> = Matrix4::zeros();
    for r in 0..4 {
        for c in 0..4 {
            m[(r, c)] = scalar(values[r * 4 + c]);
        }
    }
    m
}

fn row_major<S: LieScalar, const R: usize, const C: usize>(
    m: &nalgebra::SMatrix<S, R, C>,
) -> Vec<S> {
    let mut out: Vec<S> = Vec::with_capacity(R * C);
    for r in 0..R {
        for c in 0..C {
            out.push(m[(r, c)]);
        }
    }
    out
}

// ─── the stereographic chart ───────────────────────────────────────────────

fn check_stereographic<S: LieScalar>(entries: &[&OracleStereographic], tolerance: f64) {
    assert!(!entries.is_empty());
    for entry in entries {
        let p: Vector4<S> = vector4(&entry.p3d);
        let label: String = format!("stereographic {} {:?}", entry.scalar, entry.p3d);

        let mut d_project: Matrix2x4<S> = Matrix2x4::zeros();
        let proj: Vector2<S> = StereographicParam::project_with_jacobian(&p, &mut d_project);
        close_all(
            &format!("{label} project"),
            proj.as_slice(),
            &entry.project,
            tolerance,
        );
        close_all(
            &format!("{label} d_project_d_p3d"),
            &row_major(&d_project),
            &entry.d_project_d_p3d,
            tolerance,
        );
        // `project` without the Jacobian returns the same pixel — bit for bit,
        // including the NaNs at the projection point, where `assert_eq!` would
        // compare two NaNs unequal.
        assert!(same_bits(
            StereographicParam::project(&p).as_slice(),
            proj.as_slice()
        ));

        let mut d_unproject: Matrix4x2<S> = Matrix4x2::zeros();
        let bearing: Vector4<S> =
            StereographicParam::unproject_with_jacobian(&proj, &mut d_unproject);
        close_all(
            &format!("{label} unproject"),
            bearing.as_slice(),
            &entry.unproject,
            tolerance,
        );
        close_all(
            &format!("{label} d_unproject_d_proj"),
            &row_major(&d_unproject),
            &entry.d_unproject_d_proj,
            tolerance,
        );
        assert!(same_bits(
            StereographicParam::unproject(&proj).as_slice(),
            bearing.as_slice()
        ));
    }
}

#[test]
fn the_stereographic_chart_matches_the_cpp_in_double() {
    let oracle: Oracle = oracle();
    let entries: Vec<&OracleStereographic> = oracle
        .stereographic
        .iter()
        .filter(|e| e.scalar == "f64")
        .collect();
    assert_eq!(entries.len(), 12);
    check_stereographic::<f64>(&entries, TOLERANCE);
}

/// The `f32` pass asks for **exact** equality: `project` and `unproject` are
/// half a dozen multiplications and one square root, and the port evaluates them
/// in the C++'s order.
#[test]
fn the_stereographic_chart_matches_the_cpp_in_float() {
    let oracle: Oracle = oracle();
    let entries: Vec<&OracleStereographic> = oracle
        .stereographic
        .iter()
        .filter(|e| e.scalar == "f32")
        .collect();
    assert_eq!(entries.len(), 12);
    check_stereographic::<f32>(&entries, 0.0);
}

/// The projection point and the equator, where `norm = p[2] + |p|` is zero or
/// the whole chart is at infinity, are in the fixture on purpose: whatever
/// non-finite coefficients the C++ produces there, the port produces the same.
#[test]
fn the_degenerate_directions_are_covered() {
    let oracle: Oracle = oracle();
    let degenerate: usize = oracle
        .stereographic
        .iter()
        .filter(|e| e.project.iter().any(Option::is_none) || e.p3d[2] <= 0.0)
        .count();
    assert!(degenerate >= 6, "{degenerate} degenerate probes");
}

// ─── the reprojection residual ─────────────────────────────────────────────

fn camera_from<S: LieScalar>(entry: &OracleLinearize) -> CameraEnum<S> {
    match entry.model.as_str() {
        "kb4" => {
            let mut param: nalgebra::SVector<S, 8> = nalgebra::SVector::zeros();
            for (i, value) in entry.params.iter().enumerate() {
                param[i] = scalar(*value);
            }
            CameraEnum::Kb4(KannalaBrandt4::new(param))
        }
        "pinhole-radtan8" => {
            let mut param: nalgebra::SVector<S, 12> = nalgebra::SVector::zeros();
            for (i, value) in entry.params.iter().enumerate() {
                param[i] = scalar(*value);
            }
            // The fixture's cameras carry no `distortion_valid_radius`, so
            // basalt's default `rpmax = 0` applies.
            CameraEnum::PinholeRadtan8(PinholeRadtan8::new(param, S::zero()))
        }
        other => panic!("unexpected model {other}"),
    }
}

fn check_linearize<S: LieScalar>(entries: &[&OracleLinearize], tolerance: f64) {
    assert!(!entries.is_empty());
    for entry in entries {
        let label: String = format!(
            "linearize_point {} {} inv_dist {}",
            entry.name, entry.scalar, entry.inv_dist
        );
        let cam: CameraEnum<S> = camera_from(entry);
        let kpt_pos: Landmark<S> = Landmark::new(
            LandmarkId(0),
            TimeCamId::new(0, 0),
            vector2(&entry.direction),
            scalar(entry.inv_dist),
        );
        let t_t_h: Matrix4<S> = matrix4(&entry.t_t_h);
        let kpt_obs: Vector2<S> = vector2(&entry.kpt_obs);

        let mut res: Vector2<S> = Vector2::zeros();
        let mut d_res_d_xi: Matrix2x6<S> = Matrix2x6::zeros();
        let mut d_res_d_p: Matrix2x3<S> = Matrix2x3::zeros();
        let mut proj: Vector4<S> = Vector4::zeros();
        let valid: bool = linearize_point(
            &kpt_obs,
            &kpt_pos,
            &t_t_h,
            &cam,
            &mut res,
            &mut LinearizePointOut {
                d_res_d_xi: Some(&mut d_res_d_xi),
                d_res_d_p: Some(&mut d_res_d_p),
                proj: Some(&mut proj),
            },
        );

        assert_eq!(valid, entry.valid, "{label}: validity");
        // The C++ leaves whatever the camera model wrote in `res` on the invalid
        // path and returns before subtracting the observation (`ba_utils.h:100-111`);
        // the port does the same, so the fixture pins that too.
        close_all(
            &format!("{label} res"),
            res.as_slice(),
            &entry.res,
            tolerance,
        );
        if !valid {
            continue;
        }
        close_all(
            &format!("{label} d_res_d_xi"),
            &row_major(&d_res_d_xi),
            &entry.d_res_d_xi,
            tolerance,
        );
        close_all(
            &format!("{label} d_res_d_p"),
            &row_major(&d_res_d_p),
            &entry.d_res_d_p,
            tolerance,
        );
        // The fourth slot of `proj` is never written by `linearizePoint`
        // (`ba_utils.h:113-116`); the caller overwrites it with the landmark id.
        close_all(
            &format!("{label} proj"),
            &proj.as_slice()[..3],
            &entry.proj[..3],
            tolerance,
        );
    }
}

#[test]
fn linearize_point_matches_the_cpp_in_double() {
    let oracle: Oracle = oracle();
    let entries: Vec<&OracleLinearize> = oracle
        .linearize_point
        .iter()
        .filter(|e| e.scalar == "f64")
        .collect();
    assert_eq!(entries.len(), 10);
    check_linearize::<f64>(&entries, TOLERANCE);
}

/// Exact in `f32`: the residual path is the stereographic unprojection, one 4x4
/// matrix-vector product and the camera model, and all three already match the
/// C++ bit for bit in single precision (`camera_oracle.rs`).
#[test]
fn linearize_point_matches_the_cpp_in_float() {
    let oracle: Oracle = oracle();
    let entries: Vec<&OracleLinearize> = oracle
        .linearize_point
        .iter()
        .filter(|e| e.scalar == "f32")
        .collect();
    assert_eq!(entries.len(), 10);
    check_linearize::<f32>(&entries, 0.0);
}

/// Both reference calibrations really are the cameras the oracle was built with,
/// so a typo in the generator fails a test instead of weakening the fixture.
#[test]
fn the_oracle_cameras_are_the_shipped_calibrations() {
    let oracle: Oracle = oracle();
    let msdmi: Calibration<f64> = Calibration::from_json_str(MSDMI).unwrap();
    let msdmg: Calibration<f64> = Calibration::from_json_str(MSDMG).unwrap();
    for entry in &oracle.linearize_point {
        let want: Vec<f64> = match entry.name.as_str() {
            "kb4_msdmi_cam0" => msdmi.intrinsics[0].params(),
            "radtan8_msdmg_cam0" => msdmg.intrinsics[0].params(),
            other => panic!("unexpected camera {other}"),
        };
        match entry.scalar.as_str() {
            // The `f64` pass carries the calibration's own doubles.
            "f64" => assert_eq!(entry.params, want, "{}", entry.name),
            // The `f32` pass carries them narrowed, which is what a calibration
            // read in single precision would hold.
            _ => {
                let narrowed: Vec<f64> = want.iter().map(|v| f64::from(*v as f32)).collect();
                assert_eq!(entry.params, narrowed, "{}", entry.name);
            }
        }
    }
}

/// The residual really is `pi(...) - z`, basalt's flipped sign
/// (`ba_utils.h:117`, papers-part2 §13 D13), and not the paper's `z - pi(...)`.
#[test]
fn the_residual_sign_is_basalts() {
    let oracle: Oracle = oracle();
    let displaced: &OracleLinearize = oracle
        .linearize_point
        .iter()
        .find(|e| e.scalar == "f64" && e.valid && finite(&e.res, 0) != 0.0)
        .unwrap();
    // The generator built the observation as `projection + [offset, -offset]`,
    // so `pi - z` is `[-offset, offset]`: the first component is negative when
    // the observation was pushed to the right.
    assert!(finite(&displaced.res, 0) < 0.0 && finite(&displaced.res, 1) > 0.0);
    // And the projection itself, which `proj` carries, is the observation plus
    // the residual.
    for i in 0..2 {
        let sum: f64 = displaced.kpt_obs[i] + finite(&displaced.res, i);
        assert!((sum - finite(&displaced.proj, i)).abs() <= 1e-9 * sum.abs().max(1.0));
    }
}

// ─── triangulation ─────────────────────────────────────────────────────────

fn pose_from<S: LieScalar>(entry: &OracleTriangulate) -> Se3<S> {
    // Sophus keeps a normalised quaternion, and the fixture prints its exact
    // coefficients; wrapping them unchanged reproduces the C++ pose bit for bit,
    // where re-normalising could move the last ulp.
    let q: Quaternion<S> = Quaternion::new(
        scalar(entry.quaternion_xyzw[3]),
        scalar(entry.quaternion_xyzw[0]),
        scalar(entry.quaternion_xyzw[1]),
        scalar(entry.quaternion_xyzw[2]),
    );
    Se3::new(
        So3::from_unit_quaternion(UnitQuaternion::new_unchecked(q)),
        vector3(&entry.translation),
    )
}

fn check_triangulate<S: LieScalar>(entries: &[&OracleTriangulate], tolerance: f64) {
    assert!(!entries.is_empty());
    for entry in entries {
        let label: String = format!("triangulate {} {}", entry.name, entry.scalar);
        let t_0_1: Se3<S> = pose_from(entry);

        // The reconstruction is exact: the 3x4 the DLT builds from is the one
        // the C++ built from. Without this the comparison below would not
        // separate a pose that differs from a DLT that differs.
        let p2: nalgebra::Matrix3x4<S> = t_0_1.inverse().matrix3x4();
        close_all_finite(&format!("{label} p2"), &row_major(&p2), &entry.p2, 0.0);

        let f0: Vector3<S> = vector3(&entry.f0);
        let f1: Vector3<S> = vector3(&entry.f1);
        let got: Vector4<S> = triangulate(&f0, &f1, &t_0_1);
        close_all(
            &format!("{label} result"),
            got.as_slice(),
            &entry.result,
            tolerance,
        );

        // basalt's acceptance gate (`sqrt_keypoint_vio.cpp:534`). This is the
        // decision the port must reproduce exactly, whatever the last ulps do.
        let accepted: bool = got.iter().all(|v| v.to_f64().is_finite())
            && got[3] > S::zero()
            && got[3] < S::from_literal(3.0);
        assert_eq!(accepted, entry.accepted, "{label}: acceptance");
    }
}

#[test]
fn triangulate_matches_the_cpp_in_double() {
    let oracle: Oracle = oracle();
    let entries: Vec<&OracleTriangulate> = oracle
        .triangulate
        .iter()
        .filter(|e| e.scalar == "f64")
        .collect();
    assert_eq!(entries.len(), 10);
    check_triangulate::<f64>(&entries, TRIANGULATE_TOLERANCE_F64);
}

#[test]
fn triangulate_matches_the_cpp_in_float() {
    let oracle: Oracle = oracle();
    let entries: Vec<&OracleTriangulate> = oracle
        .triangulate
        .iter()
        .filter(|e| e.scalar == "f32")
        .collect();
    assert_eq!(entries.len(), 10);
    check_triangulate::<f32>(&entries, TRIANGULATE_TOLERANCE_F32);
}

/// The three cases that sit on basalt's `inv_dist < 3` gate really do straddle
/// it, in both precisions — otherwise the acceptance assertions above would be
/// checking nothing.
#[test]
fn the_borderline_cases_straddle_the_acceptance_gate() {
    let oracle: Oracle = oracle();
    for precision in ["f64", "f32"] {
        let by_name = |name: &str| -> &OracleTriangulate {
            oracle
                .triangulate
                .iter()
                .find(|e| e.scalar == precision && e.name == name)
                .unwrap()
        };
        assert!(!by_name("at_the_gate").accepted, "{precision} at_the_gate");
        assert!(
            by_name("just_inside_the_gate").accepted,
            "{precision} inside"
        );
        assert!(
            !by_name("just_outside_the_gate").accepted,
            "{precision} outside"
        );
        // A point 1/3 m away lands within a few ulps of the threshold itself.
        assert!((finite(&by_name("at_the_gate").result, 3) - 3.0).abs() < 1e-6);
        // And the far cases are the other end of the gate: an inverse distance
        // close to zero, which basalt also rejects only at exactly zero.
        assert!(finite(&by_name("nearly_parallel").result, 3) > 0.0);
        assert!(finite(&by_name("nearly_parallel").result, 3) < 1e-3);
    }
}

/// The sign flip at `ba_base.h:113` fires: a point behind the first camera comes
/// back with a bearing that still points at it, not away from it.
#[test]
fn the_bearing_sign_flip_fires() {
    let oracle: Oracle = oracle();
    let behind: &OracleTriangulate = oracle
        .triangulate
        .iter()
        .find(|e| e.scalar == "f64" && e.name == "behind")
        .unwrap();
    let dot: f64 = (0..3)
        .map(|i| behind.f0[i] * finite(&behind.result, i))
        .sum();
    assert!(dot > 0.0, "the result points along f0");
    assert!(
        finite(&behind.result, 2) < 0.0,
        "and f0 itself points backwards"
    );
}
