//! IMU preintegration against basalt's own C++, number for number.
//!
//! `fixtures/imu/imu_oracle.json` is the output of `tools/imu_oracle.cpp` on the
//! fork's `slam-rs-reference` branch (target `basalt_imu_oracle`), built and run
//! out of tree against the basalt headers, since the monorepo never compiles C++
//! (decision D15). For six sample runs it holds the delta state, `get_cov()`,
//! `get_sqrt_cov_inv()`, `get_cov_inv()`, both bias Jacobians and the LDLT
//! itself — `transpositionsP()`, `vectorD()` and `matrixL()` — plus, for ten
//! accelerometer readings, what `Quaternion::FromTwoVectors(accel, UnitZ)`
//! returns.
//!
//! This is the load-bearing test of the stage, and the covariance half of it
//! cannot be replaced by anything internal. Finite differences and the
//! closed-form recurrence in `imu.rs` prove the port is self-consistent; only
//! the fixture proves it is basalt.
//!
//! The whitening is where that matters. `compute_sqrt_cov_inv`
//! (`preintegration.h:305-321`) delegates the whole question of what a
//! *singular* covariance whitens to to `Eigen::LDLT`, whose delayed column
//! updates make it pivot on the un-updated diagonal. On the `singular` run
//! below — one 5 ms sample with zero gyro and accelerometer, where position and
//! velocity are perfectly correlated — Eigen eliminates velocity first and the
//! three position pivots come out as `-1.6e-27`, which basalt's
//! `vectorD()[i] < numeric_limits<Scalar>::min()` test zeroes. A textbook
//! right-looking LDLT pivots the other way, leaves a tiny *positive* pivot, and
//! puts an information weight of `6.2e26` on a direction the measurement says
//! nothing about.
//!
//! The tolerance is relative, not zero: Rust and C++ evaluate the same
//! expressions in the same order but are free to contract `a * b + c` into an
//! FMA differently.

#![allow(clippy::unwrap_used)]
// The two whitening constants below are a C++ `%.17g` printout, carried over
// verbatim even where an f64 does not need every figure. Keeping the printout is
// what makes them evidence.
#![allow(clippy::excessive_precision)]

use nalgebra::{SMatrix, Vector3};
use serde::Deserialize;
use slam_rs::imu::{ImuSample, IntegratedImuMeasurement, Matrix9, gravity_from_first_accel};
use slam_rs::lie::{LieScalar, So3};

const ORACLE: &str = include_str!("fixtures/imu/imu_oracle.json");

/// Agreement with the C++ number, per coefficient, relative to `max(|want|, 1)`.
const TOLERANCE: f64 = 1e-14;

/// The same for `f32`, where the C++ and the port accumulate 100 steps in single
/// precision and the last two bits of each step compound.
const TOLERANCE_F32: f64 = 2e-5;

/// `sqrt_cov_inv` inverts the covariance, so a relative input difference comes
/// out multiplied by the condition number — 1e5 on the long run.
const WHITENING_TOLERANCE: f64 = 1e-7;

/// The same in `f32`. The 100-step covariance has a condition number of 7 736
/// (`cov.eigenvalues()` on the fixture), and `7736 * f32::EPSILON` is 9.2e-4, so
/// this is the conditioning of the problem rather than a slack tolerance.
const WHITENING_TOLERANCE_F32: f64 = 1e-3;

#[derive(Debug, Deserialize)]
struct OracleRun {
    name: String,
    scalar: String,
    count: usize,
    sample_dt_ns: i64,
    zero_samples: bool,
    dt_ns: i64,
    delta_translation: Vec<f64>,
    delta_quaternion_xyzw: Vec<f64>,
    delta_velocity: Vec<f64>,
    cov: Vec<f64>,
    ldlt_transpositions: Vec<f64>,
    ldlt_d: Vec<f64>,
    ldlt_l: Vec<f64>,
    sqrt_cov_inv: Vec<f64>,
    cov_inv: Vec<f64>,
    d_state_d_ba: Vec<f64>,
    d_state_d_bg: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct OracleTwoVectors {
    scalar: String,
    accel: Vec<f64>,
    quaternion_xyzw: Vec<f64>,
    rotated: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Oracle {
    runs: Vec<OracleRun>,
    two_vectors: Vec<OracleTwoVectors>,
}

/// `sample()` from `tools/imu_oracle.cpp`: built from the index alone, out of
/// small exact binary fractions, so both languages get the same bits without the
/// inputs having to travel through the fixture.
fn sample(k: i64, dt_ns: i64) -> ImuSample {
    ImuSample {
        t_ns: (k + 1) * dt_ns,
        accel: Vector3::new(
            (k % 7) as f64 * 0.25 - 0.75,
            (k % 5) as f64 * 0.5 - 1.0,
            9.8125 + (k % 3) as f64 * 0.0625,
        ),
        gyro: Vector3::new(
            (k % 11) as f64 * 0.015625 - 0.078125,
            (k % 13) as f64 * 0.0078125 - 0.046875,
            (k % 9) as f64 * 0.03125 - 0.125,
        ),
    }
}

const ACCEL_STD_DEV: f64 = 0.23;
const GYRO_STD_DEV: f64 = 0.0027;

fn integrate<S: LieScalar>(run: &OracleRun) -> IntegratedImuMeasurement<S> {
    let accel_cov: Vector3<S> =
        Vector3::repeat(S::from_literal(ACCEL_STD_DEV) * S::from_literal(ACCEL_STD_DEV));
    let gyro_cov: Vector3<S> =
        Vector3::repeat(S::from_literal(GYRO_STD_DEV) * S::from_literal(GYRO_STD_DEV));
    let mut meas: IntegratedImuMeasurement<S> =
        IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
    for k in 0..run.count as i64 {
        let mut data: ImuSample = sample(k, run.sample_dt_ns);
        if run.zero_samples {
            data.accel = Vector3::zeros();
            data.gyro = Vector3::zeros();
        }
        meas.integrate(&data, &accel_cov, &gyro_cov).unwrap();
    }
    meas
}

/// Every coefficient within `tolerance` of the C++ number, relative to
/// `max(|want|, 1)`.
fn assert_close(name: &str, run: &str, got: &[f64], want: &[f64], tolerance: f64) {
    assert_eq!(got.len(), want.len(), "{run}/{name}: length");
    for (index, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let scale: f64 = w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tolerance * scale,
            "{run}/{name}[{index}]: got {g:.17e}, want {w:.17e}, \
             difference {:.3e} over {:.3e}",
            (g - w).abs(),
            tolerance * scale
        );
    }
}

fn row_major<const R: usize, const C: usize, S: LieScalar>(m: &SMatrix<S, R, C>) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(R * C);
    for r in 0..R {
        for c in 0..C {
            out.push(m[(r, c)].to_f64());
        }
    }
    out
}

fn check_run<S: LieScalar>(
    run: &OracleRun,
    tolerance: f64,
    whitening_tolerance: f64,
    compare_factor: bool,
) {
    let meas: IntegratedImuMeasurement<S> = integrate::<S>(run);
    let name: &str = &run.name;

    assert_eq!(meas.get_dt_ns(), run.dt_ns, "{name}: dt_ns");

    let delta = meas.get_delta_state();
    assert_close(
        "delta_translation",
        name,
        &delta
            .t_w_i
            .translation
            .iter()
            .map(|v: &S| v.to_f64())
            .collect::<Vec<f64>>(),
        &run.delta_translation,
        tolerance,
    );
    assert_close(
        "delta_quaternion_xyzw",
        name,
        &delta
            .t_w_i
            .rotation
            .quaternion_xyzw()
            .iter()
            .map(|v: &S| v.to_f64())
            .collect::<Vec<f64>>(),
        &run.delta_quaternion_xyzw,
        tolerance,
    );
    assert_close(
        "delta_velocity",
        name,
        &delta
            .vel_w_i
            .iter()
            .map(|v: &S| v.to_f64())
            .collect::<Vec<f64>>(),
        &run.delta_velocity,
        tolerance,
    );

    assert_close("cov", name, &row_major(meas.get_cov()), &run.cov, tolerance);
    assert_close(
        "d_state_d_ba",
        name,
        &row_major(meas.get_d_state_d_ba()),
        &run.d_state_d_ba,
        tolerance,
    );
    assert_close(
        "d_state_d_bg",
        name,
        &row_major(meas.get_d_state_d_bg()),
        &run.d_state_d_bg,
        tolerance,
    );

    // `cov_inv` is `MᵀM` and does not depend on the order the LDLT eliminated
    // the directions in, so it is compared on every run.
    assert_close(
        "cov_inv",
        name,
        &row_major(&meas.get_cov_inv()),
        &run.cov_inv,
        whitening_tolerance,
    );

    if !compare_factor {
        return;
    }
    assert_close(
        "sqrt_cov_inv",
        name,
        &row_major(&meas.get_cov_inv_sqrt()),
        &run.sqrt_cov_inv,
        whitening_tolerance,
    );
    // A zero in the C++ whitening is a rank-deficient direction, and it has to
    // be an exact zero here too: a relative tolerance against zero would hide
    // exactly the 1e26 weight this fixture exists to catch.
    let got_whitening: Vec<f64> = row_major(&meas.get_cov_inv_sqrt());
    for (index, want) in run.sqrt_cov_inv.iter().enumerate() {
        if *want == 0.0 {
            assert_eq!(
                got_whitening[index], 0.0,
                "{name}/sqrt_cov_inv[{index}]: C++ is exactly zero, port is {}",
                got_whitening[index]
            );
        }
    }
}

/// The delta state, covariance and bias Jacobians agree with the C++ on every
/// run, at both precisions.
#[test]
fn the_measurement_matches_the_cpp() {
    let oracle: Oracle = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(oracle.runs.len(), 7);
    for run in &oracle.runs {
        match run.scalar.as_str() {
            // `full_rank_100_f32` is the one run whose pivot order the two
            // languages disagree on: its two largest velocity variances differ
            // by one `f32` ulp (1.323982e-4 against 1.323981e-4) and the
            // ulp-level difference in the covariance flips which one
            // `maxCoeff` sees first. The factor is then a row permutation of
            // the C++ one, and only `MᵀM` is comparable. Every rank-deficient
            // direction — the case that matters — lands in a run where the
            // order is unambiguous.
            "double" => check_run::<f64>(run, TOLERANCE, WHITENING_TOLERANCE, true),
            "float" => check_run::<f32>(
                run,
                TOLERANCE_F32,
                WHITENING_TOLERANCE_F32,
                run.name != "full_rank_100_f32",
            ),
            other => panic!("unknown scalar {other}"),
        }
    }
}

/// The LDLT itself — pivot order, `D` and `L` — matches Eigen's, which is what
/// makes the whitening of a rank-deficient covariance reproducible.
///
/// `crates/slam-rs/src/imu.rs`'s `ldlt_in_place` is private, so the factors are
/// recovered here from the public `get_cov_inv_sqrt`: `M = D^{-1/2} L^{-1} P`,
/// hence `M cov Mᵀ` is the identity on the directions with a valid pivot and
/// zero on the rest. That, plus the exact-zero check above, pins everything the
/// factorization decides.
#[test]
fn the_whitening_is_idempotent_on_the_observable_directions() {
    let oracle: Oracle = serde_json::from_str(ORACLE).unwrap();
    for run in oracle.runs.iter().filter(|r| r.scalar == "double") {
        let meas: IntegratedImuMeasurement<f64> = integrate::<f64>(run);
        let m: Matrix9<f64> = meas.get_cov_inv_sqrt();
        let product: Matrix9<f64> = m * meas.get_cov() * m.transpose();

        // The C++ `vectorD` says which directions survive; a row of `M` is zero
        // exactly where the pivot did not.
        let rank: usize = run
            .ldlt_d
            .iter()
            .filter(|d| **d >= f64::MIN_POSITIVE)
            .count();
        let mut identity_rows: usize = 0;
        for i in 0..9 {
            let row_norm: f64 = m.row(i).norm();
            if row_norm == 0.0 {
                assert!(
                    product.row(i).norm() <= 1e-9,
                    "{}: row {i} whitens a rank-deficient direction to {}",
                    run.name,
                    product.row(i).norm()
                );
            } else {
                identity_rows += 1;
                assert!(
                    (product[(i, i)] - 1.0).abs() <= 1e-6,
                    "{}: M cov Mᵀ[{i},{i}] = {}",
                    run.name,
                    product[(i, i)]
                );
            }
        }
        assert_eq!(identity_rows, rank, "{}: rank", run.name);

        // And the pivot order and factor are the C++ ones, read back through the
        // relation `P cov Pᵀ = L D Lᵀ` the fixture records.
        let l: Matrix9<f64> = Matrix9::from_row_iterator(run.ldlt_l.iter().copied());
        let d: Matrix9<f64> = Matrix9::from_diagonal(&nalgebra::SVector::<f64, 9>::from_iterator(
            run.ldlt_d.iter().copied(),
        ));
        let mut permuted: Matrix9<f64> = *meas.get_cov();
        for (k, pivot) in run.ldlt_transpositions.iter().enumerate() {
            let pivot: usize = *pivot as usize;
            if pivot != k {
                permuted.swap_rows(k, pivot);
                permuted.swap_columns(k, pivot);
            }
        }
        let reconstructed: Matrix9<f64> = l * d * l.transpose();
        assert!(
            (permuted - reconstructed).amax() <= 1e-14 * permuted.amax().max(1.0),
            "{}: P cov Pᵀ != L D Lᵀ",
            run.name
        );
    }
}

/// The `singular` run, spelled out: this is the case the review caught.
#[test]
fn a_rank_deficient_covariance_whitens_to_zero_position_weight() {
    let oracle: Oracle = serde_json::from_str(ORACLE).unwrap();
    let run: &OracleRun = oracle
        .runs
        .iter()
        .find(|r| r.name == "singular_one_zero_sample")
        .unwrap();

    // Eigen eliminates the three velocity directions first, on their larger
    // *original* variance, and leaves the three position pivots negative.
    assert_eq!(
        run.ldlt_transpositions,
        vec![6.0, 7.0, 8.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    for pivot in &run.ldlt_d[6..9] {
        assert!(*pivot < 0.0, "C++ pivot {pivot} is not negative");
    }

    let meas: IntegratedImuMeasurement<f64> = integrate::<f64>(run);
    let m: Matrix9<f64> = meas.get_cov_inv_sqrt();
    let information: Matrix9<f64> = meas.get_cov_inv();

    // Position carries no information at all, in C++ and here.
    for axis in 0..3 {
        assert_eq!(information[(axis, axis)], 0.0);
        assert_eq!(m.row(axis + 6).norm(), 0.0);
    }
    // Rotation and velocity carry theirs, exactly.
    assert!((m[(3, 3)] - 74074.074074074078).abs() <= 1e-6);
    assert!((m[(0, 6)] - 869.56521739130437).abs() <= 1e-9);
}

fn check_two_vectors<S: LieScalar>(entry: &OracleTwoVectors, tolerance: f64) -> f64 {
    let accel: Vector3<S> = Vector3::new(
        S::from_literal(entry.accel[0]),
        S::from_literal(entry.accel[1]),
        S::from_literal(entry.accel[2]),
    );
    let rotation: So3<S> = gravity_from_first_accel(&accel);
    let rotated: Vector3<S> = rotation * (accel / accel.norm());
    let want: Vector3<f64> = Vector3::new(entry.rotated[0], entry.rotated[1], entry.rotated[2]);
    let got: Vector3<f64> =
        Vector3::new(rotated.x.to_f64(), rotated.y.to_f64(), rotated.z.to_f64());
    // How far each side leaves the reading from world up.
    let up: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
    let cpp_error: f64 = (want - up).norm();
    let rust_error: f64 = (got - up).norm();
    assert!(
        rust_error <= tolerance,
        "accel {:?} ({}): the port leaves the reading {rust_error} from +Z, \
         over {tolerance}",
        entry.accel,
        entry.scalar
    );
    cpp_error
}

/// `Quaternion::FromTwoVectors(accel, UnitZ)` — the gravity initialisation —
/// agrees with the C++ quaternion in `f64`, on both branches.
#[test]
fn gravity_init_matches_the_cpp_in_double() {
    let oracle: Oracle = serde_json::from_str(ORACLE).unwrap();
    let mut checked: usize = 0;
    for entry in oracle.two_vectors.iter().filter(|e| e.scalar == "double") {
        let accel: Vector3<f64> = Vector3::new(entry.accel[0], entry.accel[1], entry.accel[2]);
        let rotation: So3<f64> = gravity_from_first_accel(&accel);
        let got: [f64; 4] = rotation.quaternion_xyzw();
        // Eigen returns the coefficients as computed and does not renormalize,
        // so on the near-antiparallel branch its quaternion is up to 1.6e-11
        // off unit length — `V.col(2)` comes out of a `JacobiSVD` on a nearly
        // rank-deficient 2x3. The port normalizes; after dividing the C++
        // coefficients by their own norm the two agree to the last bit.
        let want_norm: f64 = entry
            .quaternion_xyzw
            .iter()
            .map(|v: &f64| v * v)
            .sum::<f64>()
            .sqrt();
        // A quaternion and its negation are the same rotation; Eigen's sign is
        // whatever `setFromTwoVectors` built, and the port builds the same one.
        let sign: f64 = if got[3] * entry.quaternion_xyzw[3] < 0.0 {
            -1.0
        } else {
            1.0
        };
        for (i, (coefficient, cpp)) in got.iter().zip(entry.quaternion_xyzw.iter()).enumerate() {
            let want: f64 = cpp / want_norm;
            assert!(
                (sign * coefficient - want).abs() <= 1e-15,
                "accel {:?}: coefficient {i} is {} against C++ {want}",
                entry.accel,
                sign * coefficient
            );
        }
        check_two_vectors::<f64>(entry, 1e-12);
        checked += 1;
    }
    assert_eq!(checked, 6);
}

/// In `f32` the port always lands on `+Z`; Eigen does not.
///
/// The near-antiparallel branch takes its axis from `V.col(2)` of a 2x3
/// `JacobiSVD`, whose sign is arbitrary. On `accel = (0.01, -0.005, -9.81)` in
/// `f32` Eigen picks the opposite sign to the exact null vector and misses `+Z`
/// by 2.2e-3 rad; the port takes `normalize(v0 × v1)` and hits it. This test
/// records that difference rather than reproducing it — see the note on
/// `gravity_from_first_accel`.
#[test]
fn gravity_init_is_at_least_as_good_as_the_cpp_in_float() {
    let oracle: Oracle = serde_json::from_str(ORACLE).unwrap();
    let mut worst_cpp: f64 = 0.0;
    let mut checked: usize = 0;
    for entry in oracle.two_vectors.iter().filter(|e| e.scalar == "float") {
        // 1e-4 is the `f32` rounding floor of this construction: the port's own
        // worst case here is 4.8e-5 and Eigen's good cases sit at 2.5e-5.
        worst_cpp = worst_cpp.max(check_two_vectors::<f32>(entry, 1e-4));
        checked += 1;
    }
    assert_eq!(checked, 4);
    // Eigen's own worst case here is the 2.2e-3 rad one; if a future Eigen
    // fixes its SVD sign this bound drops and the note above can go.
    assert!(
        (2.2e-3..2.3e-3).contains(&worst_cpp),
        "the C++ worst-case tilt moved to {worst_cpp}"
    );
}
