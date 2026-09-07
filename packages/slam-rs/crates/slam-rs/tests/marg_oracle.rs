//! The marginalization helper and the complete orthogonal decomposition
//! against basalt's own C++, coefficient for coefficient.
//!
//! `fixtures/marg/marg_oracle.json` is the output of `tools/marg_oracle.cpp` on
//! the fork's `slam-rs-reference` branch (target `basalt_marg_oracle`), built
//! and run out of tree against the basalt sources, since the monorepo never
//! compiles C++ (decision D15).
//!
//! Nine problems, each emitted in `f64` and in `f32`:
//!
//! | name | shape | what it adds |
//! |---|---|---|
//! | `generic` | 10x6, marg 2 | a well-conditioned baseline |
//! | `rank_def_least_squares` | 10x6, `J.col(1) = J.col(4)` | **basalt's own `test_qr.cpp:38-83`**, index sets included |
//! | `rank_def_in_marg` | 12x6, two dependent marginalized columns | the deficiency inside the block being eliminated |
//! | `zero_column` | 9x5, a kept column of exact zeros | a direction nothing observes |
//! | `threshold_exact` | 8x4, first marg column a spike of exactly `sqrt(eps)` | **on** the rank threshold |
//! | `threshold_below` | the same problem, spike one ulp lower | just under it |
//! | `threshold_above` | the same problem, spike one ulp higher | just over it |
//! | `wide_window` | 40x27, marg = one 6-dof pose + a 9-dof vel/bias tail | past the LDLT panel width, in the window's real shape |
//! | `marg_exhausts_rank` | 3x5, marg 3 | the degenerate shape where the C++ reads out of range |
//!
//! Each case carries the problem as plain numbers — `j`, `r`, and `sq_h`/`sq_b`
//! as **Eigen** formed `JᵀJ` and `Jᵀr`, so the two squared routines start from
//! the same bytes the C++ used rather than from a Rust re-multiplication — the
//! index split, the three routines' reduced `(H, b)`, and the complete
//! orthogonal decomposition's rank and minimum-norm solution for the original
//! problem, the marginalized block's pseudo-inverse, and each reduced system.
//!
//! **Tolerances, and what they measure.** The flat QR of
//! `marginalizeHelperSqrtToSqrt` is Eigen's `makeHouseholder` and
//! `applyHouseholderOnTheLeft` in Eigen's order, so what is left is the
//! association inside Eigen's `gemv` kernel, the residue decision D50 already
//! measured for the landmark blocks. The two squared routines add a
//! pseudo-inverse and two `gemm`s on top of that. The constants below are the
//! measured agreement, not an aspiration; see the report for the exact figures.
//!
//! **What must be exact.** The rank decisions: `sqrt_to_sqrt`'s
//! `|beta| > sqrt(epsilon)` test (`marg_helper.cpp:301`) and the complete
//! orthogonal decomposition's `rank()` (`ColPivHouseholderQR.h:261-268`). Those
//! are integer outcomes and are asserted as equalities.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use slam_rs::lie::LieScalar;
use slam_rs::marg::{
    Cod, MargHelper, ReducedSystem, marginalize_helper_sq_to_sq, marginalize_helper_sq_to_sqrt,
    marginalize_helper_sqrt_to_sqrt,
};

// ─── the fixture ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Oracle {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    scalar: String,
    rows: usize,
    cols: usize,
    idx_to_keep: Vec<usize>,
    idx_to_marg: Vec<usize>,
    j: Vec<f64>,
    r: Vec<f64>,
    sq_h: Vec<f64>,
    sq_b: Vec<f64>,
    rank_threshold: f64,
    #[serde(default)]
    beta_probe: Option<f64>,
    #[serde(default)]
    beta_probe_accepted: Option<bool>,
    full_rank: usize,
    full_solution: Vec<f64>,
    h_mm_rank: usize,
    h_mm_size: usize,
    keep_size: usize,
    h_mm: Vec<f64>,
    h_mm_pinv: Vec<f64>,
    sqrt_to_sqrt_out_of_range: bool,
    #[serde(default)]
    sqrt_to_sqrt: Option<Reduced>,
    sq_to_sqrt: Reduced,
    sq_to_sqrt_squared: Reduced,
    sq_to_sq: Reduced,
}

#[derive(Debug, Deserialize)]
struct Reduced {
    rows: usize,
    cols: usize,
    h: Vec<f64>,
    b: Vec<f64>,
    rank: usize,
    solution: Vec<f64>,
}

fn load() -> Oracle {
    let text: &str = include_str!("fixtures/marg/marg_oracle.json");
    serde_json::from_str(text).expect("marg_oracle.json parses")
}

// ─── comparison ────────────────────────────────────────────────────────────

/// The largest magnitude in a slice, floored at one.
///
/// Comparisons scale by the array's own largest coefficient rather than by each
/// entry: the sub-diagonal entries a Householder QR drives to zero have no
/// scale of their own, and basalt's own tests compare `(H_a - H_b).norm()` for
/// exactly that reason (`test_linearization.cpp:148-157`).
fn scale_of(values: &[f64]) -> f64 {
    values.iter().fold(1.0f64, |acc, v| acc.max(v.abs()))
}

struct Worst {
    value: f64,
    at: String,
}

impl Worst {
    fn new() -> Self {
        Self {
            value: 0.0,
            at: String::new(),
        }
    }

    fn note(&mut self, diff: f64, at: impl FnOnce() -> String) {
        if diff > self.value {
            self.value = diff;
            self.at = at();
        }
    }
}

/// Compare a Rust matrix with the C++ row-major dump.
fn compare<S: LieScalar>(
    got: &DMatrix<S>,
    want: &[f64],
    rows: usize,
    cols: usize,
    tol: f64,
    label: &str,
    worst: &mut Worst,
) {
    assert_eq!(got.nrows(), rows, "{label}: row count");
    assert_eq!(got.ncols(), cols, "{label}: column count");
    let scale: f64 = scale_of(want);
    for i in 0..rows {
        for j in 0..cols {
            let expected: f64 = want[i * cols + j];
            let actual: f64 = got[(i, j)].to_f64();
            let diff: f64 = (actual - expected).abs() / scale;
            worst.note(diff, || format!("{label}[{i},{j}]"));
            assert!(
                diff <= tol,
                "{label}[{i},{j}]: {actual} vs {expected}, relative {diff:e} > {tol:e}"
            );
        }
    }
}

fn compare_vec<S: LieScalar>(
    got: &DVector<S>,
    want: &[f64],
    tol: f64,
    label: &str,
    worst: &mut Worst,
) {
    assert_eq!(got.nrows(), want.len(), "{label}: length");
    let scale: f64 = scale_of(want);
    for (i, expected) in want.iter().enumerate() {
        let actual: f64 = got[i].to_f64();
        let diff: f64 = (actual - expected).abs() / scale;
        worst.note(diff, || format!("{label}[{i}]"));
        assert!(
            diff <= tol,
            "{label}[{i}]: {actual} vs {expected}, relative {diff:e} > {tol:e}"
        );
    }
}

/// How close the port has to be, per quantity.
///
/// One number covers everything except `marginalizeHelperSqToSqrt`'s residual,
/// which needs its own; see [`Tolerances::sq_to_sqrt_b`].
#[derive(Debug, Clone, Copy)]
struct Tolerances {
    /// Every reduced matrix, every residual, the pseudo-inverse and the
    /// minimum-norm solutions: two to three ulps of the array's own scale.
    general: f64,
    /// `marginalizeHelperSqToSqrt`'s `r_m` when the reduced Hessian is rank
    /// deficient (`marg_helper.cpp:223-231`).
    ///
    /// The last LDLT pivot of a deficient `marg_H` is cancellation noise —
    /// `4.4e-16` against a matrix whose other eigenvalues are order 1 — and
    /// `sqrt` of it is `2.1e-8`, comfortably above the `sqrt(min())` floor at
    /// `:229`, so basalt **divides**. The entry that comes out is noise over
    /// noise, and the two sides agree on it to `2.5e-9` rather than to an ulp.
    /// Nothing consumes it on its own: the estimator only ever sees
    /// `J_mᵀ J_m` and `J_mᵀ r_m`, and those are checked against the squared
    /// routine's own output at [`Tolerances::general`] in
    /// [`check_case`] — `J_mᵀ r_m` agrees **exactly** on this case.
    sq_to_sqrt_b: f64,
}

// ─── rebuilding one case in the target scalar ──────────────────────────────

/// The `f64` coefficients the C++ printed, narrowed to the case's scalar.
///
/// `%.17g` of a `float` is the exact decimal of its own bit pattern, so the
/// narrowing round-trips: the Rust side runs on exactly the matrix the C++ ran
/// on, which is what lets the threshold cases test a branch rather than a
/// tolerance.
fn matrix_of<S: LieScalar>(values: &[f64], rows: usize, cols: usize) -> DMatrix<S> {
    DMatrix::from_fn(rows, cols, |i, j| S::from_literal(values[i * cols + j]))
}

fn vector_of<S: LieScalar>(values: &[f64]) -> DVector<S> {
    DVector::from_iterator(values.len(), values.iter().map(|v| S::from_literal(*v)))
}

fn index_set(values: &[usize]) -> BTreeSet<usize> {
    values.iter().copied().collect()
}

// ─── the tests ─────────────────────────────────────────────────────────────

/// Everything one case asserts, in one scalar.
fn check_case<S: LieScalar>(case: &Case, tols: Tolerances, worst: &mut Worst) {
    let tol: f64 = tols.general;
    let label = |what: &str| format!("{} {} {what}", case.name, case.scalar);
    let keep: BTreeSet<usize> = index_set(&case.idx_to_keep);
    let marg: BTreeSet<usize> = index_set(&case.idx_to_marg);

    let j: DMatrix<S> = matrix_of::<S>(&case.j, case.rows, case.cols);
    let r: DVector<S> = vector_of::<S>(&case.r);
    let sq_h: DMatrix<S> = matrix_of::<S>(&case.sq_h, case.cols, case.cols);
    let sq_b: DVector<S> = vector_of::<S>(&case.sq_b);

    // The rank threshold is a property of the scalar, not of the data
    // (`marg_helper.cpp:284`), and both sides must agree on it exactly.
    assert_eq!(
        S::default_epsilon().sqrt().to_f64(),
        case.rank_threshold,
        "{}: sqrt(epsilon)",
        label("rank_threshold")
    );

    // `marginalizeHelperSqrtToSqrt` (`:247-327`).
    if let Some(want) = &case.sqrt_to_sqrt {
        let got: ReducedSystem<S> =
            marginalize_helper_sqrt_to_sqrt(j.clone(), r.clone(), &keep, &marg).unwrap();
        compare(
            &got.h,
            &want.h,
            want.rows,
            want.cols,
            tol,
            &label("sqrt_to_sqrt.h"),
            worst,
        );
        compare_vec(&got.b, &want.b, tol, &label("sqrt_to_sqrt.b"), worst);
        // The rank the QR reached is visible in the shape, and is an integer:
        // no tolerance applies.
        assert_eq!(
            got.h.nrows(),
            want.rows,
            "{}: rows the QR left",
            label("sqrt_to_sqrt")
        );
        // The reduced system's own rank, through the ported decomposition.
        assert_eq!(
            Cod::new(&got.h).rank(),
            want.rank,
            "{}: rank of the reduced system",
            label("sqrt_to_sqrt")
        );
        let solution: DVector<S> = Cod::new(&got.h).solve_vec(&got.b).unwrap();
        compare_vec(
            &solution,
            &want.solution,
            tol,
            &label("sqrt_to_sqrt.solution"),
            worst,
        );
    } else {
        // The C++ reads out of range on this shape; the port must not.
        assert!(case.sqrt_to_sqrt_out_of_range);
        let got: ReducedSystem<S> =
            marginalize_helper_sqrt_to_sqrt(j.clone(), r.clone(), &keep, &marg).unwrap();
        assert_eq!(
            got.h.nrows(),
            1,
            "{}: the clamped row",
            label("sqrt_to_sqrt")
        );
        assert_eq!(got.h.ncols(), case.keep_size);
        for v in got.h.iter() {
            assert_eq!(
                *v,
                S::zero(),
                "{}: nothing constrains the kept variables",
                label("sqrt_to_sqrt")
            );
        }
        assert_eq!(got.b[0], S::zero());
    }

    // The marginalized block's pseudo-inverse, which is what the two squared
    // routines consume (`:99-100`, `:184-185`).
    let h_mm: DMatrix<S> = matrix_of::<S>(&case.h_mm, case.h_mm_size, case.h_mm_size);
    let cod: Cod<S> = Cod::new(&h_mm);
    assert_eq!(
        cod.rank(),
        case.h_mm_rank,
        "{}: rank of the marginalized block",
        label("h_mm")
    );
    compare(
        &cod.pseudo_inverse(),
        &case.h_mm_pinv,
        case.h_mm_size,
        case.h_mm_size,
        tol,
        &label("h_mm_pinv"),
        worst,
    );

    // `marginalizeHelperSqToSqrt` (`:120-244`).
    {
        let got: ReducedSystem<S> =
            marginalize_helper_sq_to_sqrt(sq_h.clone(), sq_b.clone(), &keep, &marg).unwrap();
        let want: &Reduced = &case.sq_to_sqrt;
        compare(
            &got.h,
            &want.h,
            want.rows,
            want.cols,
            tol,
            &label("sq_to_sqrt.h"),
            worst,
        );
        compare_vec(
            &got.b,
            &want.b,
            tols.sq_to_sqrt_b,
            &label("sq_to_sqrt.b"),
            worst,
        );

        // What the estimator actually consumes: the square root squared. The
        // rank-deficient row that the coefficient comparison above has to be
        // lenient about contributes nothing here, which is the point.
        let squared: DMatrix<S> = got.h.transpose() * &got.h;
        let squared_b: DVector<S> = got.h.transpose() * &got.b;
        compare(
            &squared,
            &case.sq_to_sq.h,
            case.sq_to_sq.rows,
            case.sq_to_sq.cols,
            tol,
            &label("sq_to_sqrt.h^T h"),
            worst,
        );
        compare_vec(
            &squared_b,
            &case.sq_to_sq.b,
            tol,
            &label("sq_to_sqrt.h^T b"),
            worst,
        );
    }

    // `marginalizeHelperSqToSq` (`:42-117`).
    {
        let got: ReducedSystem<S> =
            marginalize_helper_sq_to_sq(sq_h.clone(), sq_b.clone(), &keep, &marg).unwrap();
        let want: &Reduced = &case.sq_to_sq;
        compare(
            &got.h,
            &want.h,
            want.rows,
            want.cols,
            tol,
            &label("sq_to_sq.h"),
            worst,
        );
        compare_vec(&got.b, &want.b, tol, &label("sq_to_sq.b"), worst);
        assert_eq!(
            Cod::new(&got.h).rank(),
            want.rank,
            "{}: rank of the reduced system",
            label("sq_to_sq")
        );
    }

    // The whole problem's minimum-norm least-squares solution, the
    // `original_solution` of `test_qr.cpp:48`.
    let full: Cod<S> = Cod::new(&j);
    assert_eq!(full.rank(), case.full_rank, "{}: rank", label("full"));
    compare_vec(
        &full.solve_vec(&r).unwrap(),
        &case.full_solution,
        tol,
        &label("full_solution"),
        worst,
    );
}

#[test]
fn the_helper_matches_the_cpp_in_double() {
    let oracle: Oracle = load();
    let mut worst: Worst = Worst::new();
    let mut seen: usize = 0;
    for case in &oracle.cases {
        if case.scalar != "f64" {
            continue;
        }
        check_case::<f64>(
            case,
            Tolerances {
                general: 1e-14,
                sq_to_sqrt_b: 1e-8,
            },
            &mut worst,
        );
        seen += 1;
    }
    assert_eq!(seen, 9, "every double case ran");
    println!(
        "worst f64 relative difference {:e} at {}",
        worst.value, worst.at
    );
}

#[test]
fn the_helper_matches_the_cpp_in_float() {
    let oracle: Oracle = load();
    let mut worst: Worst = Worst::new();
    let mut seen: usize = 0;
    for case in &oracle.cases {
        if case.scalar != "f32" {
            continue;
        }
        check_case::<f32>(
            case,
            Tolerances {
                general: 5e-6,
                sq_to_sqrt_b: 5e-6,
            },
            &mut worst,
        );
        seen += 1;
    }
    assert_eq!(seen, 9, "every float case ran");
    println!(
        "worst f32 relative difference {:e} at {}",
        worst.value, worst.at
    );
}

/// The rank threshold decides these three cases and nothing else does.
///
/// `threshold_exact`, `threshold_below` and `threshold_above` are the *same*
/// matrix apart from the height of a single spike in the first marginalized
/// column, placed exactly on `sqrt(epsilon)`, one ulp below it and one ulp
/// above. `marg_helper.cpp:301` tests `std::abs(beta) > rank_threshold`, a
/// strict inequality, so:
///
/// * `exact` and `below` are rejected — the column is zeroed, the rank does not
///   advance, no reflection is applied — and must produce a **bit-identical**
///   reduced system;
/// * `above` is accepted and must produce a different one.
///
/// The port has to make the same call on the same bits or two runs of the
/// estimator diverge from a single ulp.
#[test]
fn the_rank_threshold_decision_matches_the_cpp() {
    let oracle: Oracle = load();
    let mut checked: usize = 0;
    for scalar in ["f64", "f32"] {
        let case_named = |name: &str| -> &Case {
            oracle
                .cases
                .iter()
                .find(|c| c.name == name && c.scalar == scalar)
                .expect("threshold case present")
        };
        let exact: &Case = case_named("threshold_exact");
        let below: &Case = case_named("threshold_below");
        let above: &Case = case_named("threshold_above");

        for case in [exact, below, above] {
            let beta: f64 = case.beta_probe.expect("threshold case carries beta");
            let accepted: bool = case.beta_probe_accepted.expect("and its decision");
            // The port's own comparison, on the same bits, in the same scalar.
            let ours: bool = if scalar == "f64" {
                beta.abs() > f64::EPSILON.sqrt()
            } else {
                let beta32: f32 = beta as f32;
                beta32.abs() > f32::EPSILON.sqrt()
            };
            assert_eq!(
                ours, accepted,
                "{} {}: |beta| = {beta:e} against sqrt(eps) = {:e}",
                case.name, scalar, case.rank_threshold
            );
            checked += 1;
        }
        assert!(!exact.beta_probe_accepted.unwrap(), "equality is not >");
        assert!(!below.beta_probe_accepted.unwrap());
        assert!(above.beta_probe_accepted.unwrap());

        // And the outputs agree with that: rejecting leaves the kept columns
        // untouched, so the two rejected cases coincide exactly.
        let e: &Reduced = exact.sqrt_to_sqrt.as_ref().unwrap();
        let bl: &Reduced = below.sqrt_to_sqrt.as_ref().unwrap();
        let ab: &Reduced = above.sqrt_to_sqrt.as_ref().unwrap();
        assert_eq!(e.h, bl.h, "{scalar}: on and below the threshold coincide");
        assert_eq!(e.b, bl.b);
        assert_ne!(e.h, ab.h, "{scalar}: above the threshold differs");

        // The port reproduces that same relation, computed rather than read.
        if scalar == "f64" {
            let run = |case: &Case| -> ReducedSystem<f64> {
                marginalize_helper_sqrt_to_sqrt(
                    matrix_of::<f64>(&case.j, case.rows, case.cols),
                    vector_of::<f64>(&case.r),
                    &index_set(&case.idx_to_keep),
                    &index_set(&case.idx_to_marg),
                )
                .unwrap()
            };
            let (ours_e, ours_b, ours_a) = (run(exact), run(below), run(above));
            assert_eq!(
                ours_e.h, ours_b.h,
                "the port coincides on the threshold too"
            );
            assert_eq!(ours_e.b, ours_b.b);
            assert_ne!(ours_e.h, ours_a.h);
        }
    }
    assert_eq!(checked, 6);
}

/// `test/src/test_qr.cpp`'s `RankDefLeastSquares` (`:38-83`), ported.
///
/// basalt builds a rank-deficient `10x6` least-squares problem
/// (`J.col(1) = J.col(4)`), marginalizes `{0, 1}` three different ways, and
/// asserts the three minimum-norm solutions over the kept `{2, 3, 4, 5}` agree:
///
/// ```text
/// EXPECT_TRUE(sol_qr.isApprox(sol_sc));         // :80
/// EXPECT_TRUE(sol_qr.isApprox(sol_sqrt_sc2));   // :81
/// ```
///
/// `isApprox` is a **relative** comparison whose default precision is
/// `NumTraits<Scalar>::dummy_precision()` — `1e-12` in double, `1e-5` in float
/// (`DenseBase.h:351-352` for the default, `NumTraits.h:236,241` for the
/// values) — and whose test is
/// `‖a − b‖² ≤ prec² · min(‖a‖², ‖b‖²)` (`Fuzzy.h:23-27`), i.e.
/// `‖a − b‖ ≤ prec · min(‖a‖, ‖b‖)`. basalt passes no precision
/// (`test_qr.cpp:80-81`), so that is the tolerance, and it is what is used
/// here — on the port's own three solutions rather than on the fixture's.
///
/// It is about 15,000 times tighter than the `sqrt(epsilon)` this test used
/// before the S7 review.
#[test]
fn rank_def_least_squares() {
    let oracle: Oracle = load();
    let case: &Case = oracle
        .cases
        .iter()
        .find(|c| c.name == "rank_def_least_squares" && c.scalar == "f64")
        .expect("basalt's own case is in the fixture");

    let keep: BTreeSet<usize> = index_set(&case.idx_to_keep);
    let marg: BTreeSet<usize> = index_set(&case.idx_to_marg);
    assert_eq!(case.idx_to_marg, vec![0, 1], "basalt's own index sets");
    assert_eq!(case.idx_to_keep, vec![2, 3, 4, 5]);

    let j: DMatrix<f64> = matrix_of::<f64>(&case.j, case.rows, case.cols);
    let r: DVector<f64> = vector_of::<f64>(&case.r);
    let sq_h: DMatrix<f64> = matrix_of::<f64>(&case.sq_h, case.cols, case.cols);
    let sq_b: DVector<f64> = vector_of::<f64>(&case.sq_b);

    // `:46-51`: the whole problem is rank deficient, and the decomposition
    // says so.
    let full: Cod<f64> = Cod::new(&j);
    assert_eq!(full.rank(), 5, "one dependent column of six");
    assert_eq!(full.rank(), case.full_rank);

    // `:63-78`, the QR version.
    let qr: ReducedSystem<f64> = MargHelper::sqrt_to_sqrt(j, r, &keep, &marg).unwrap();
    let sol_qr: DVector<f64> = Cod::new(&qr.h).solve_vec(&qr.b).unwrap();

    // `:46-61`, the SC version.
    let sc: ReducedSystem<f64> =
        MargHelper::sq_to_sq(sq_h.clone(), sq_b.clone(), &keep, &marg).unwrap();
    let sol_sc: DVector<f64> = Cod::new(&sc.h).solve_vec(&sc.b).unwrap();

    // `:23-44`, the square-root SC version, and its squared form.
    let sqrt_sc: ReducedSystem<f64> = MargHelper::sq_to_sqrt(sq_h, sq_b, &keep, &marg).unwrap();
    let squared: DMatrix<f64> = sqrt_sc.h.transpose() * &sqrt_sc.h;
    let squared_b: DVector<f64> = sqrt_sc.h.transpose() * &sqrt_sc.b;
    let sol_sqrt_sc2: DVector<f64> = Cod::new(&squared).solve_vec(&squared_b).unwrap();

    // `:80-81`, with Eigen's `isApprox` default precision.
    let prec: f64 = <f64 as LieScalar>::eigen_dummy_precision();
    assert_eq!(prec, 1e-12, "NumTraits<double>::dummy_precision()");
    let is_approx = |a: &DVector<f64>, b: &DVector<f64>| -> bool {
        (a - b).norm() <= prec * a.norm().min(b.norm())
    };
    assert!(
        is_approx(&sol_qr, &sol_sc),
        "sol_qr {sol_qr:?} vs sol_sc {sol_sc:?}"
    );
    assert!(
        is_approx(&sol_qr, &sol_sqrt_sc2),
        "sol_qr {sol_qr:?} vs sol_sqrt_sc2 {sol_sqrt_sc2:?}"
    );

    // And each agrees with what the C++ printed for it.
    let mut worst: Worst = Worst::new();
    compare_vec(
        &sol_qr,
        &case.sqrt_to_sqrt.as_ref().unwrap().solution,
        1e-13,
        "sol_qr",
        &mut worst,
    );
    compare_vec(
        &sol_sc,
        &case.sq_to_sq.solution,
        1e-13,
        "sol_sc",
        &mut worst,
    );
    compare_vec(
        &sol_sqrt_sc2,
        &case.sq_to_sqrt_squared.solution,
        1e-13,
        "sol_sqrt_sc2",
        &mut worst,
    );
}
