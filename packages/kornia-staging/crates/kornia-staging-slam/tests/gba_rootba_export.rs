//! Global BA on RootSLAM's exported GBA problems (rootba `bal` input), read at
//! test time. Assets: `<runs>/<SEQ>/map/{map,pose_graph}.json` and the
//! `output.<SEQ>.{balpgf,balpgd}.log` of the reproduction (rootba `bal` in f32 and
//! f64; default runs root `/mnt/nas/ssd-8tb/scratch/rootslam/runs`, override
//! `GBA_EXPORT_RUNS`), and the RootSLAM Basalt calibrations (default
//! `~/0Dev/forks/rootslam/data`, override `GBA_EXPORT_CALIB`). The golden and
//! bench lanes are `#[ignore]`; run explicitly, a missing asset fails the test
//! and names the asset. `pixi run -e kornia-staging --frozen kornia-staging-gba-golden`
//! runs the goldens (`cargo test --release -p kornia-staging-slam --test
//! gba_rootba_export -- --ignored golden`).
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::time::Instant;

use kornia_staging_slam::sqrt_ba::{solve_global, GbaCamera, GbaOptions, GbaReport};

#[path = "support/rootba_export.rs"]
mod rootba_export;
use rootba_export::{load, rootba_run, DoubleSphere, Export, RootbaRun};

fn run(export: &Export, threads: usize) -> (GbaReport, f64) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let start = Instant::now();
    let (mut centred, centroid) = export.problem.centred::<f32>();
    let report = solve_global(
        &mut centred,
        &export.rig,
        &GbaOptions::default(),
        Some(&pool),
    )
    .unwrap();
    let mut solved = export.problem.clone();
    solved.uncentre_from(&centred, &centroid);
    std::hint::black_box(&solved);
    (report, start.elapsed().as_secs_f64())
}

fn summary(seq: &str, threads: usize, report: &GbaReport, wall: f64) -> String {
    let sum = |f: fn(&kornia_staging_slam::sqrt_ba::GbaStageSeconds) -> f64| {
        report.iterations.iter().map(|i| f(&i.seconds)).sum::<f64>()
    };
    let pcg_s = sum(|s| s.pcg);
    let its = report.pcg_iterations();
    format!(
        "{seq} threads={threads} wall_s={wall:.3} stage1_s={:.3} stage2_s={:.3} pcg_s={pcg_s:.3} cg_its={its} cg_ms_per_it={:.3} backsub_s={:.3} eval_s={:.3} lm_attempts={} accepted={} cost {:.4e} -> {:.4e} valid={} termination={:?} dropped_rows={} nonfinite_lm={} precond_fallbacks={}",
        sum(|s| s.stage1),
        sum(|s| s.stage2),
        1e3 * pcg_s / its.max(1) as f64,
        sum(|s| s.back_substitution),
        sum(|s| s.evaluation),
        report.iterations.len(),
        report.iterations.iter().filter(|i| i.accepted).count(),
        report.initial_cost.error,
        report.final_cost.error,
        report.final_cost.valid,
        report.termination,
        report.dropped_rows,
        report.nonfinite_landmark_steps,
        report.preconditioner_fallbacks,
    )
}

/// Our run, rootba's f32 (`balpgf`) and f64 (`balpgd`) runs of one sequence.
fn golden(seq: &str) -> (GbaReport, RootbaRun, RootbaRun) {
    let export = load(seq);
    let (f32_run, f64_run) = (rootba_run(seq, "balpgf"), rootba_run(seq, "balpgd"));
    let (report, wall) = run(&export, 4);
    eprintln!(
        "{}\nrootba bal f32 {f32_run:?}\nrootba bal f64 {f64_run:?}",
        summary(seq, 4, &report, wall)
    );
    // The band: the final cost reaches rootba's (f32 and f64) within 0.5 %,
    // and is not suspiciously lower.
    let ours = report.final_cost.error;
    for rootba in [f32_run.final_cost, f64_run.final_cost] {
        assert!(ours <= rootba * 1.005, "{seq}: {ours} vs rootba {rootba}");
        assert!(
            ours >= rootba * 0.95,
            "{seq}: suspiciously low: {ours} vs {rootba}"
        );
    }
    assert!(ours < report.initial_cost.error);
    assert_eq!(report.nonfinite_landmark_steps, 0);
    assert_eq!(report.dropped_rows, 0);
    (report, f32_run, f64_run)
}

/// The band on MGO07 (pinhole-radtan rig, every LM step accepted):
/// cost band, LM attempts and the CG iteration total within 15 % of rootba f32's.
#[test]
#[ignore = "reads RootSLAM's exported MGO07 GBA problem from the NAS"]
fn golden_mgo07_matches_rootba() {
    let (report, rootba, _) = golden("MGO07");
    let accepted = report.iterations.iter().filter(|i| i.accepted).count();
    assert_eq!(report.iterations.len(), rootba.accepted + rootba.rejected);
    assert!(
        accepted.abs_diff(rootba.accepted) <= 1,
        "accepted {accepted} vs rootba {rootba:?}"
    );
    let cg = report.pcg_iterations() as f64;
    let band = cg / rootba.cg_iterations as f64;
    assert!(
        (0.85..=1.15).contains(&band),
        "CG iterations {cg} vs rootba {rootba:?}"
    );
}

/// The band on TM1 (double-sphere rig, test-only camera): the cost band, and
/// rejected LM steps that occur and are handled. A rejected attempt leaves the
/// state, so the cost of the next attempt is measured from the same start;
/// lambda grows after it; the accepted costs fall.
#[test]
#[ignore = "reads RootSLAM's exported TM1 GBA problem from the NAS"]
fn golden_tm1_matches_rootba_with_rejected_steps() {
    let (report, rootba, _) = golden("TM1");
    assert!(
        rootba.rejected > 0,
        "rootba's TM1 run rejects steps: {rootba:?}"
    );
    let rejected = report.iterations.iter().filter(|i| !i.accepted).count();
    assert!(rejected > 0, "no rejected LM step on TM1");
    let mut cost = report.initial_cost.error;
    for (i, it) in report.iterations.iter().enumerate() {
        let after = it.cost_after.expect("every TM1 step is finite").error;
        if it.accepted {
            assert!(after < cost, "attempt {i}: accepted {after} >= {cost}");
            cost = after;
        } else {
            // Rejected: the cost rose, or the model predicted no decrease.
            assert!(
                after >= cost || it.model_decrease <= 0.0,
                "attempt {i}: rejected a predicted decrease {after} < {cost}"
            );
            if let Some(next) = report.iterations.get(i + 1) {
                assert!(next.lambda > it.lambda, "attempt {i}: lambda did not grow");
            }
        }
    }
    assert_eq!(cost, report.final_cost.error);
}

/// CG ms per iteration and totals. `GBA_BENCH_SEQS` (default `MGO07,TM1`) and
/// `GBA_BENCH_THREADS` (default `2,8`); pin with `taskset`.
#[test]
#[ignore = "benchmark on RootSLAM's exported GBA problems from the NAS"]
fn bench_exported_problems() {
    let seqs = std::env::var("GBA_BENCH_SEQS").unwrap_or_else(|_| "MGO07,TM1".into());
    let threads = std::env::var("GBA_BENCH_THREADS").unwrap_or_else(|_| "2,8".into());
    for seq in seqs.split(',') {
        let export = load(seq);
        for t in threads.split(',').map(|t| t.parse::<usize>().unwrap()) {
            let (report, wall) = run(&export, t);
            println!(
                "BENCH {} load {}",
                summary(seq, t, &report, wall),
                std::fs::read_to_string("/proc/loadavg")
                    .unwrap_or_default()
                    .trim()
            );
        }
    }
}

/// The test-only double-sphere Jacobian against central differences (TM1's
/// camera 0 intrinsics), in f64 arithmetic on the f32 model's output.
#[test]
fn double_sphere_jacobian_matches_central_differences() {
    let cam = DoubleSphere {
        f: [158.286, 158.274, 254.961, 256.889],
        xi: -0.172,
        alpha: 0.593,
    };
    let mut checked = 0;
    for p in [
        [0.1f32, -0.2, 1.0],
        [0.8, 0.3, 0.5],
        [-1.2, 0.4, 0.3],
        [0.3, -0.9, -0.1],
    ] {
        let mut j = [[0.0f32; 3]; 2];
        let (_, valid) = cam.project(p, Some(&mut j));
        assert!(valid, "{p:?}");
        let h = 1e-3f32;
        for k in 0..3 {
            let (mut plus, mut minus) = (p, p);
            plus[k] += h;
            minus[k] -= h;
            let (a, _) = cam.project(plus, None);
            let (b, _) = cam.project(minus, None);
            for r in 0..2 {
                let numeric = (f64::from(a[r]) - f64::from(b[r])) / (2.0 * f64::from(h));
                let analytic = f64::from(j[r][k]);
                assert!(
                    (numeric - analytic).abs() < 1e-2 * (1.0 + analytic.abs()),
                    "{p:?} row {r} col {k}: {numeric} vs {analytic}"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 12);
}
