//! The association of `tbb::parallel_deterministic_reduce`, against the fork's
//! own TBB.
//!
//! `fixtures/linearize/tbb_reduce_oracle.json` is the output of
//! `tools/tbb_reduce_probe.cpp` on the fork's `slam-rs-reference` branch
//! (target `basalt_tbb_reduce_probe`), linked against the same oneTBB the
//! basalt binaries use.
//!
//! basalt reduces at four sites in `src/linearization/linearization_abs_qr.cpp`
//! (`:262`, `:307`, `:354`, `:550`), all of them over
//! `tbb::blocked_range<size_t>(0, n)` built with the two-argument constructor,
//! i.e. **grainsize 1**. With `simple_partitioner` — which is what makes
//! `parallel_deterministic_reduce` deterministic — the range splits while
//! `grainsize < size()`, `blocked_range` cuts at `begin + (end - begin) / 2`,
//! and bodies join left-then-right. The result is a balanced tree that depends
//! only on `n`.
//!
//! Decision D31 said these four sums needed a fixed order. It did not say which
//! one, and the port's first answer — a left fold — was the wrong one: on
//! `f32 [2²⁴, 1, 1, 1]` the tree gives `16777218` and a left fold `16777216`.
//! This fixture pins the right one. 90 of its 120 cases are inputs where the
//! two disagree, and the probe records that TBB's answer does not move between
//! one thread and the machine's default, which is the property the whole
//! deterministic lane rests on.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde::Deserialize;
use slam_rs::linearize::deterministic_reduce_scalar_for_tests;

mod common;

const ORACLE: &str = include_str!("fixtures/linearize/tbb_reduce_oracle.json");

#[derive(Debug, Deserialize)]
struct Oracle {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    pattern: u32,
    n: usize,
    values: Vec<f64>,
    tbb: f64,
    tbb_one_thread: f64,
    left_fold: f64,
    distinguishes: bool,
}

#[test]
fn the_reduction_reproduces_tbbs_association_bit_for_bit() {
    let oracle: Oracle = serde_json::from_str(ORACLE).expect("the TBB fixture parses");
    assert_eq!(oracle.cases.len(), 120);

    let mut distinguishing: usize = 0;
    let mut fold_disagreements: usize = 0;
    for case in &oracle.cases {
        assert_eq!(case.values.len(), case.n);

        // The probe's own evidence that the answer is thread-independent.
        assert_eq!(
            (case.tbb as f32).to_bits(),
            (case.tbb_one_thread as f32).to_bits(),
            "pattern {} n {}: TBB moved between one thread and the default",
            case.pattern,
            case.n
        );

        let values: Vec<f32> = case.values.iter().map(|v| *v as f32).collect();
        let got: f32 =
            deterministic_reduce_scalar_for_tests::<f32>(case.n, &mut |i, acc| acc + values[i]);
        assert_eq!(
            got.to_bits(),
            (case.tbb as f32).to_bits(),
            "pattern {} n {}: got {got}, TBB {}",
            case.pattern,
            case.n,
            case.tbb
        );

        // And the left fold the port used to do, so the fixture is evidence
        // that the two really are different and not a distinction on paper.
        let fold: f32 = values.iter().fold(0.0f32, |acc, v| acc + *v);
        assert_eq!(
            fold.to_bits(),
            (case.left_fold as f32).to_bits(),
            "pattern {} n {}: the fixture's left fold is not a left fold",
            case.pattern,
            case.n
        );
        if case.distinguishes {
            distinguishing += 1;
            assert_ne!(got.to_bits(), fold.to_bits());
            fold_disagreements += 1;
        }
    }

    assert_eq!(
        distinguishing, 90,
        "the fixture's discriminating case count"
    );
    assert_eq!(fold_disagreements, 90);
}

/// The case that shows the tree is not a fold, spelled out:
/// `f32 [2²⁴, 1, 1, 1]`, where `2²⁴ + 1` is not representable.
#[test]
fn the_four_element_tree_is_not_a_left_fold() {
    let values: [f32; 4] = [16_777_216.0, 1.0, 1.0, 1.0];
    let tree: f32 = deterministic_reduce_scalar_for_tests::<f32>(4, &mut |i, acc| acc + values[i]);
    let fold: f32 = values.iter().fold(0.0f32, |acc, v| acc + *v);
    assert_eq!(tree, 16_777_218.0, "(x0 + x1) + (x2 + x3)");
    assert_eq!(fold, 16_777_216.0, "((x0 + x1) + x2) + x3");
}
