//! What more than one integration test needs.
//!
//! Integration tests are separate crates, so a `tests/common/mod.rs` declared
//! with `mod common;` is the only way to share code between them. Each test
//! binary compiles its own copy and uses part of it, which is why the module
//! allows dead code: the alternative is a `cfg` per item per test.

#![allow(dead_code)]

use nalgebra::{DMatrix, DVector, Vector3, Vector6};
use slam_rs::calib::{Calibration, CameraModel, Kb4Params};
use slam_rs::lie::{LieScalar, Se3};

const MSDMI: &str = include_str!("../fixtures/msdmi_calib.json");

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
