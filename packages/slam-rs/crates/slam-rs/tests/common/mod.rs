//! What more than one integration test needs.
//!
//! Integration tests are separate crates, so a `tests/common/mod.rs` declared
//! with `mod common;` is the only way to share code between them. Each test
//! binary compiles its own copy and uses part of it, which is why the module
//! allows dead code: the alternative is a `cfg` per item per test.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use nalgebra::{DMatrix, DVector, Vector3, Vector6};
use serde::Deserialize;
use slam_rs::calib::{Calibration, CameraModel, Kb4Params};
use slam_rs::config::VioConfig;
use slam_rs::lie::{LieScalar, Se3};

const MSDMI: &str = include_str!("../fixtures/msdmi_calib.json");

// ── the fixture directory and the two files every VIO lane reads ───────────

/// `crates/slam-rs/tests/fixtures`.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The MSDMI config, which every VIO lane and the whole-pipeline test share.
pub fn config() -> VioConfig {
    VioConfig::from_json_str(
        &std::fs::read_to_string(fixtures().join("msdmi_config.json")).unwrap(),
    )
    .unwrap()
}

/// The MSDMI calibration, always `f64`: the estimator casts it to its own
/// scalar, so a lane that runs `f32` still reads the file's doubles.
pub fn calibration() -> Calibration<f64> {
    Calibration::from_json_str(
        &std::fs::read_to_string(fixtures().join("msdmi_calib.json")).unwrap(),
    )
    .unwrap()
}

// ── the PGM framesets ─────────────────────────────────────────────

/// One camera's image as it sits in a PGM: the raw 8-bit raster and its shape.
///
/// The raster, not an `ImageU16`: `flow_parity` widens it the way basalt's
/// camera source does, while `vio_parity` hands the bytes straight to
/// [`slam_rs::ImageView`], and one of those wrapping the other is the only
/// difference between them.
pub struct Pgm {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// `frame_<NNN>_cam<C>.pgm` under `directory`, in `tools/dump_flow.cpp`'s
/// layout.
pub fn read_pgm(directory: &Path, frame: usize, camera: usize) -> Pgm {
    let path: PathBuf = directory.join(format!("frame_{frame:03}_cam{camera}.pgm"));
    let bytes: Vec<u8> = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));

    // "P5\n<w> <h>\n255\n" then the raster; the writer emits exactly that.
    let mut fields: Vec<usize> = Vec::new();
    let mut cursor: usize = 2;
    while fields.len() < 3 {
        while bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let start: usize = cursor;
        while !bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        fields.push(
            std::str::from_utf8(&bytes[start..cursor])
                .unwrap()
                .parse()
                .unwrap(),
        );
    }
    cursor += 1;
    assert_eq!(fields[2], 255, "{} is not an 8-bit PGM", path.display());
    Pgm {
        width: fields[0],
        height: fields[1],
        pixels: bytes[cursor..].to_vec(),
    }
}

/// How many consecutive framesets `directory` covers, up to `limit`: a
/// frameset counts only when every camera's PGM is there.
pub fn available_framesets(directory: &Path, cameras: usize, limit: usize) -> usize {
    (0..limit)
        .take_while(|frame| {
            (0..cameras).all(|camera| {
                directory
                    .join(format!("frame_{frame:03}_cam{camera}.pgm"))
                    .exists()
            })
        })
        .count()
}

// ── the VIO oracle fixture ─────────────────────────────────────

/// `tools/vio_oracle.cpp`'s dump: one run per precision, plus the C++
/// frontend's keypoints per frameset. Both VIO lanes read it, so the shape
/// lives here even though `vio_parity` reads only part of it.
#[derive(Debug, Deserialize)]
pub struct Oracle {
    pub runs: Vec<OracleRun>,
    pub flow: Vec<OracleFlow>,
}

#[derive(Debug, Deserialize)]
pub struct OracleRun {
    pub scalar: String,
    pub frames: Vec<OracleFrame>,
}

#[derive(Debug, Deserialize)]
pub struct OracleFlow {
    pub t_ns: i64,
    pub cameras: Vec<Vec<OraclePoint>>,
}

/// One tracked keypoint of the C++ frontend's `OpticalFlowResult`.
///
/// The fixture also carries the warp's four `linear` coefficients so a reader
/// can see the whole `AffineCompact2f`; the estimator reads only the
/// translation, so they are not deserialized.
#[derive(Debug, Deserialize)]
pub struct OraclePoint {
    pub id: u64,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Deserialize)]
pub struct OracleFrame {
    pub frame: usize,
    pub t_ns: i64,
    pub states: Vec<OracleState>,
    pub poses: Vec<OraclePose>,
    pub kf_ids: Vec<i64>,
    pub ltkfs: Vec<i64>,
    pub num_points_kf: Vec<(i64, i64)>,
    pub last_state_t_ns: i64,
    pub frames_after_kf: i32,
    pub opt_started: bool,
    pub num_landmarks: usize,
    pub num_observations: usize,
    pub num_imu_meas: usize,
    pub marg_order: Vec<(i64, usize, usize)>,
    pub marg_digest: OracleDigest,
    pub marg: Option<OracleMarg>,
    pub lm: Vec<OracleLm>,
}

#[derive(Debug, Deserialize)]
pub struct OracleState {
    pub t_ns: i64,
    pub q: [f64; 4],
    pub t: [f64; 3],
    pub vel: [f64; 3],
    pub bg: [f64; 3],
    pub ba: [f64; 3],
    pub linearized: bool,
}

#[derive(Debug, Deserialize)]
pub struct OraclePose {
    pub t_ns: i64,
    pub q: [f64; 4],
    pub t: [f64; 3],
    pub linearized: bool,
}

#[derive(Debug, Deserialize)]
pub struct OracleDigest {
    pub rows: usize,
    pub cols: usize,
    pub h_frobenius: f64,
    pub b_norm: f64,
}

#[derive(Debug, Deserialize)]
pub struct OracleMarg {
    pub states_to_remove: usize,
    pub last_state_to_marg: i64,
    pub poses_to_marg: Vec<i64>,
    pub states_to_marg_all: Vec<i64>,
    pub states_to_marg_vel_bias: Vec<i64>,
    pub kfs_to_marg: Vec<i64>,
    pub idx_to_keep: usize,
    pub idx_to_marg: usize,
}

#[derive(Debug, Deserialize)]
pub struct OracleLm {
    pub it: i32,
    pub backtrack: i32,
    pub error_before: f64,
    pub error_after: f64,
    pub vision_error: f64,
    pub imu_error: f64,
    pub bg_error: f64,
    pub ba_error: f64,
    pub marg_prior_error: f64,
    pub l_diff: f64,
    pub f_diff: f64,
    pub lambda: f64,
    pub step_norminf: f64,
    pub solve_attempts: u32,
    pub step_is_valid: bool,
    pub step_is_successful: bool,
}

/// One uncalibrated IMU sample of `vio/imu.json`, as the fixture writes it.
#[derive(Debug, Deserialize)]
pub struct ImuRow {
    pub t_ns: i64,
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
}

#[derive(Debug, Deserialize)]
struct ImuFixture {
    imu: Vec<ImuRow>,
}

/// The 1.55 MB oracle, parsed once per test binary: five lanes read it and
/// `serde_json` is otherwise the slowest thing in them.
pub static ORACLE: LazyLock<Oracle> = LazyLock::new(|| {
    let path: PathBuf = fixtures().join("vio/vio_oracle.json");
    let text: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    serde_json::from_str(&text).expect("vio_oracle.json does not match the expected shape")
});

/// The 1,077 uncalibrated samples of the window, in capture order.
pub static IMU: LazyLock<Vec<ImuRow>> = LazyLock::new(|| {
    let path: PathBuf = fixtures().join("vio/imu.json");
    let text: String = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let fixture: ImuFixture =
        serde_json::from_str(&text).expect("imu.json does not match the expected shape");
    fixture.imu
});

/// The run of one precision, `"double"` or `"float"`.
pub fn run_named<'a>(oracle: &'a Oracle, scalar: &str) -> &'a OracleRun {
    oracle
        .runs
        .iter()
        .find(|run| run.scalar == scalar)
        .unwrap_or_else(|| panic!("the fixture has no {scalar} run"))
}

/// `KannalaBrandtCamera4<Scalar>::getTestProjections()[0]`
/// (`basalt-headers/include/basalt/camera/kannala_brandt_camera4.hpp:487-495`),
/// which is what `test_linearization.cpp:19` puts in both camera slots.
pub const KB4_TEST_PROJECTION: [f64; 8] = [
    379.045,
    379.008,
    505.512,
    509.969,
    0.00693023,
    -0.0013828,
    -0.000272596,
    -0.000452646,
];

/// xorshift64*, standing in for Eigen's `Random()`.
///
/// A Rust test that flakes is worse than one that is merely differently
/// arbitrary, so nothing here draws from the system generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x: u64 = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on `[-1, 1]`, like `Eigen::Matrix::Random()`.
    pub fn symmetric(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    pub fn vector3(&mut self) -> Vector3<f64> {
        Vector3::new(self.symmetric(), self.symmetric(), self.symmetric())
    }

    pub fn vector6(&mut self) -> Vector6<f64> {
        Vector6::from_iterator((0..6).map(|_| self.symmetric()))
    }
}

/// The calibration `get_vo_estimator` builds (`test_linearization.cpp:15-22`):
/// two camera-to-IMU transforms that are small perturbations of identity, and
/// [`KB4_TEST_PROJECTION`] in both camera slots.
///
/// The file supplies only the fields the estimator never touches; everything
/// read is overwritten here.
pub fn test_calibration(rng: &mut Rng) -> Calibration<f64> {
    let mut calib: Calibration<f64> = Calibration::from_json_str(MSDMI).unwrap();
    calib.t_i_c = (0..2)
        .map(|_| Se3::<f64>::exp_decoupled(&(rng.vector6() / 100.0)))
        .collect();
    let p: [f64; 8] = KB4_TEST_PROJECTION;
    calib.intrinsics = vec![
        CameraModel::Kb4(Kb4Params {
            fx: p[0],
            fy: p[1],
            cx: p[2],
            cy: p[3],
            k1: p[4],
            k2: p[5],
            k3: p[6],
            k4: p[7],
        });
        2
    ];
    calib
}

/// `H_kk − H_km H_mm⁻¹ H_mk` and `b_k − H_km H_mm⁻¹ b_m`, written out
/// independently of anything under test.
///
/// The reference the square-root marginalization is checked against: the QR of
/// `marginalizeHelperSqrtToSqrt` never forms `JᵀJ`, so squaring its output and
/// comparing with this is the same argument `VoMargSqrtLinearizationTest` makes
/// about the linearization (`test_linearization.cpp:379-388`), one level up.
pub fn dense_schur(
    h: &DMatrix<f64>,
    b: &DVector<f64>,
    keep: &[usize],
    marg: &[usize],
) -> (DMatrix<f64>, DVector<f64>) {
    let k: usize = keep.len();
    let m: usize = marg.len();
    let h_kk: DMatrix<f64> = DMatrix::from_fn(k, k, |i, j| h[(keep[i], keep[j])]);
    let h_km: DMatrix<f64> = DMatrix::from_fn(k, m, |i, j| h[(keep[i], marg[j])]);
    let h_mk: DMatrix<f64> = DMatrix::from_fn(m, k, |i, j| h[(marg[i], keep[j])]);
    let h_mm: DMatrix<f64> = DMatrix::from_fn(m, m, |i, j| h[(marg[i], marg[j])]);
    let b_k: DVector<f64> = DVector::from_fn(k, |i, _| b[keep[i]]);
    let b_m: DVector<f64> = DVector::from_fn(m, |i, _| b[marg[i]]);
    let h_mm_inv: DMatrix<f64> = h_mm.try_inverse().expect("the marginalized block inverts");
    let cross: DMatrix<f64> = &h_km * &h_mm_inv;
    (h_kk - &cross * h_mk, b_k - &cross * b_m)
}

/// Relative comparison against a C++ dump, tracking the worst case seen.
///
/// **Scale.** Every coefficient of an array is compared against the *array's*
/// largest magnitude, not against itself. That is not laziness: after a
/// Householder reflection the sub-diagonal entries of the landmark columns are
/// zero in exact arithmetic and pure cancellation in floating point, so in `f32`
/// C++ leaves `4.3e-5` where the port leaves `6.5e-3` — both of them noise on a
/// block whose live coefficients are in the hundreds. basalt's own tests compare
/// `(H_a - H_b).norm()` for the same reason (`test_linearization.cpp:148-157`).
pub struct Compare {
    tolerance: f64,
    pub worst: f64,
    pub worst_what: String,
}

impl Compare {
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            worst: 0.0,
            worst_what: String::from("(nothing compared)"),
        }
    }

    /// One coefficient against a scale the caller chose.
    pub fn close_scaled(&mut self, got: f64, want: f64, scale: f64, what: &str) {
        let scale: f64 = scale.max(1.0);
        let relative: f64 = (got - want).abs() / scale;
        if relative > self.worst {
            self.worst = relative;
            self.worst_what = format!("{what}: got {got:.9e}, want {want:.9e}");
        }
        assert!(
            relative <= self.tolerance,
            "{what}: got {got:.17e}, want {want:.17e}, relative {relative:.3e} > {:.1e}",
            self.tolerance
        );
    }

    /// One scalar, against its own magnitude.
    pub fn close(&mut self, got: f64, want: f64, what: &str) {
        self.close_scaled(got, want, want.abs(), what);
    }

    pub fn close_slice<S: LieScalar>(&mut self, got: &[S], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale: f64 = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            self.close_scaled(g.to_f64(), *w, scale, &format!("{what}[{i}]"));
        }
    }

    /// A row-major matrix from the fixture against a column-major `DMatrix`.
    pub fn close_matrix<S: LieScalar>(
        &mut self,
        got: &DMatrix<S>,
        want: &[f64],
        rows: usize,
        cols: usize,
        what: &str,
    ) {
        assert_eq!(got.nrows(), rows, "{what}: rows");
        assert_eq!(got.ncols(), cols, "{what}: cols");
        assert_eq!(want.len(), rows * cols, "{what}: fixture size");
        let scale: f64 = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for r in 0..rows {
            for col in 0..cols {
                self.close_scaled(
                    got[(r, col)].to_f64(),
                    want[r * cols + col],
                    scale,
                    &format!("{what}[{r},{col}]"),
                );
            }
        }
    }
}
