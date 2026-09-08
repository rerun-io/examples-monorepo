//! The marginalization helper against basalt's own C++, coefficient for
//! coefficient.
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
//! Each case carries the problem as plain numbers — `j`, `r`, the index split
//! and the reduced `(H, b)`. The fixture also carries the two squared routines'
//! output, the complete orthogonal decomposition's rank, pseudo-inverse and
//! minimum-norm solutions; nothing reads those entries any more, because the
//! squared form and the decomposition it needed went with D68. The fixture is
//! left as the C++ emitted it.
//!
//! **Tolerances, and what they measure.** The flat QR of
//! `marginalizeHelperSqrtToSqrt` is Eigen's `makeHouseholder` and
//! `applyHouseholderOnTheLeft` in Eigen's order, so what is left is the
//! association inside Eigen's `gemv` kernel, the residue decision D50 already
//! measured for the landmark blocks. The constant below is the measured
//! agreement, not an aspiration; see the report for the exact figures.
//!
//! **What must be exact.** The rank decision: `|beta| > sqrt(epsilon)`
//! (`marg_helper.cpp:301`). That is an integer outcome and is asserted as an
//! equality.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::sync::LazyLock;

use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use slam_rs::lie::LieScalar;
use slam_rs::marg::{ReducedSystem, marginalize_helper_sqrt_to_sqrt};

mod common;
use common::Compare;

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
    rank_threshold: f64,
    #[serde(default)]
    beta_probe: Option<f64>,
    #[serde(default)]
    beta_probe_accepted: Option<bool>,
    keep_size: usize,
    sqrt_to_sqrt_out_of_range: bool,
    #[serde(default)]
    sqrt_to_sqrt: Option<Reduced>,
}

#[derive(Debug, Deserialize)]
struct Reduced {
    rows: usize,
    cols: usize,
    h: Vec<f64>,
    b: Vec<f64>,
}

/// The fixture, parsed once for the whole binary.
static ORACLE: LazyLock<Oracle> = LazyLock::new(|| {
    let text: &str = include_str!("fixtures/marg/marg_oracle.json");
    serde_json::from_str(text).expect("marg_oracle.json parses")
});

// ─── comparison ────────────────────────────────────────────────────────────

/// How close the port has to be.
#[derive(Debug, Clone, Copy)]
struct Tolerances {
    /// The reduced matrix and its residual: two to three ulps of the array's
    /// own scale.
    general: f64,
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

/// Everything one case asserts, in one scalar. Every case adds to the same
/// worst-case record.
fn check_case<S: LieScalar>(case: &Case, general: &mut Compare) {
    let label = |what: &str| format!("{} {} {what}", case.name, case.scalar);
    let keep: BTreeSet<usize> = index_set(&case.idx_to_keep);
    let marg: BTreeSet<usize> = index_set(&case.idx_to_marg);

    let j: DMatrix<S> = matrix_of::<S>(&case.j, case.rows, case.cols);
    let r: DVector<S> = vector_of::<S>(&case.r);
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
        general.close_matrix(
            &got.h,
            &want.h,
            want.rows,
            want.cols,
            &label("sqrt_to_sqrt.h"),
        );
        general.close_slice(got.b.as_slice(), &want.b, &label("sqrt_to_sqrt.b"));
        // The rank the QR reached is visible in the shape, and is an integer:
        // no tolerance applies.
        assert_eq!(
            got.h.nrows(),
            want.rows,
            "{}: rows the QR left",
            label("sqrt_to_sqrt")
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
}

/// Every case of one precision, and the worst relative difference over all of
/// them.
fn run_all<S: LieScalar>(scalar: &str, tols: Tolerances) {
    let mut general: Compare = Compare::new(tols.general);
    let mut seen: usize = 0;
    for case in ORACLE.cases.iter().filter(|c| c.scalar == scalar) {
        check_case::<S>(case, &mut general);
        seen += 1;
    }
    assert_eq!(seen, 9, "every {scalar} case ran");
    println!(
        "worst {scalar} relative difference {:e} at {}",
        general.worst, general.worst_what
    );
}

#[test]
fn the_helper_matches_the_cpp_in_double() {
    run_all::<f64>("f64", Tolerances { general: 1e-14 });
}

#[test]
fn the_helper_matches_the_cpp_in_float() {
    run_all::<f32>("f32", Tolerances { general: 5e-6 });
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
    let oracle: &Oracle = &ORACLE;
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

        // The three cases sit where their names say, in C++'s own numbers: the
        // fixture prints both `|beta|` and `sqrt(epsilon)`, so this reads the
        // C++ against itself and says nothing about the port.
        for (case, ordering) in [
            (exact, std::cmp::Ordering::Equal),
            (below, std::cmp::Ordering::Less),
            (above, std::cmp::Ordering::Greater),
        ] {
            let beta: f64 = case.beta_probe.expect("threshold case carries beta");
            assert_eq!(
                beta.abs().total_cmp(&case.rank_threshold),
                ordering,
                "{} {scalar}: |beta| = {beta:e} against sqrt(eps) = {:e}",
                case.name,
                case.rank_threshold
            );
            checked += 1;
        }

        // C++'s own decision on those numbers: equality is not greater-than,
        // so the case built *on* the threshold is rejected.
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

        // The port reproduces that same relation, computed rather than read:
        // its own reduction, its own `|beta|`, its own branch. Running it in
        // the case's own scalar is what makes this the port and not a second
        // copy of `helper.rs`'s comparison.
        match scalar {
            "f64" => reproduce_threshold_relation::<f64>(exact, below, above, scalar),
            _ => reproduce_threshold_relation::<f32>(exact, below, above, scalar),
        }
    }
    assert_eq!(checked, 6);
}

/// The port's own flat QR on the three threshold cases: the two rejected ones
/// must coincide and the accepted one must differ.
fn reproduce_threshold_relation<S: LieScalar>(
    exact: &Case,
    below: &Case,
    above: &Case,
    scalar: &str,
) {
    let run = |case: &Case| -> ReducedSystem<S> {
        marginalize_helper_sqrt_to_sqrt(
            matrix_of::<S>(&case.j, case.rows, case.cols),
            vector_of::<S>(&case.r),
            &index_set(&case.idx_to_keep),
            &index_set(&case.idx_to_marg),
        )
        .unwrap()
    };
    let (ours_e, ours_b, ours_a) = (run(exact), run(below), run(above));
    assert_eq!(
        ours_e.h, ours_b.h,
        "{scalar}: the port coincides on the threshold too"
    );
    assert_eq!(ours_e.b, ours_b.b, "{scalar}");
    assert_ne!(ours_e.h, ours_a.h, "{scalar}: above the threshold differs");
}
