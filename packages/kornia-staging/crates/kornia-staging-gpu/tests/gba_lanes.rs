//! The S4 band and S5 speed table of DESIGN §7.4 (PR e1): the global BA on
//! RootSLAM's exported problems, CPU lane against the device lane.
//!
//! `#[ignore]`: it reads the NAS assets that `kornia-staging-slam`'s
//! `gba_rootba_export` reads (same loader). Run explicitly, in release, under
//! the GPU lock:
//! `flock /tmp/gpu-5090.lock cargo test --release -p kornia-staging-gpu --features wgpu --test gba_lanes -- --ignored --nocapture`.
//! `GBA_LANES_SEQS` (default `MGO05,MGO07,MGO13,TM1,TM2,MIPB02`),
//! `GBA_LANES_RUNS` (lanes in order, default `cpu,gpu,cpu,gpu,cpu64`),
//! `GBA_LANES_THREADS` (CPU lane threads, default 2). One `LANE` line per run,
//! one `BAND` line per sequence.
#![cfg(feature = "wgpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::time::Instant;

use kornia_staging_gpu::linalg::{probe_pcg, GpuPcg};
use kornia_staging_gpu::runtime::gpu_client;
use kornia_staging_slam::sqrt_ba::{
    solve_global_with, GbaOptions, GbaProblem, GbaReport, GbaStageSeconds,
};

#[path = "../../kornia-staging-slam/tests/support/rootba_export.rs"]
mod rootba_export;
use rootba_export::{load, Export};

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
}

/// One run: the report, wall seconds, the uncentred solution and the device's peak bytes.
struct Run {
    lane: String,
    report: GbaReport,
    wall: f64,
    solved: GbaProblem<f64>,
    device_bytes: usize,
    load_before: String,
}

fn run(export: &Export, lane: &str, threads: usize) -> Run {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let options = GbaOptions::default();
    let load_before = loadavg();
    let mut solved = export.problem.clone();
    let start = Instant::now();
    let (report, device_bytes) = if lane == "cpu64" {
        let (mut centred, centroid) = export.problem.centred::<f64>();
        let report =
            solve_global_with(&mut centred, &export.rig64, &options, Some(&pool), None).unwrap();
        solved.uncentre_from(&centred, &centroid);
        (report, 0)
    } else {
        let mut device = (lane == "gpu").then(|| GpuPcg::new(gpu_client().unwrap()));
        let (mut centred, centroid) = export.problem.centred::<f32>();
        let report = solve_global_with(
            &mut centred,
            &export.rig,
            &options,
            Some(&pool),
            device
                .as_mut()
                .map(|d| d as &mut dyn kornia_staging_slam::sqrt_ba::PcgLane<f32>),
        )
        .unwrap();
        solved.uncentre_from(&centred, &centroid);
        (report, device.map_or(0, |d| d.peak_device_bytes()))
    };
    Run {
        lane: lane.to_string(),
        report,
        wall: start.elapsed().as_secs_f64(),
        solved,
        device_bytes,
        load_before,
    }
}

fn line(seq: &str, threads: usize, r: &Run) -> String {
    let sum = |f: fn(&GbaStageSeconds) -> f64| {
        r.report
            .iterations
            .iter()
            .map(|i| f(&i.seconds))
            .sum::<f64>()
    };
    let (pcg, upload) = (sum(|s| s.pcg), sum(|s| s.pcg_upload));
    let its = r.report.pcg_iterations();
    format!(
        "LANE seq={seq} lane={} threads={threads} wall_s={:.3} stage1_s={:.3} stage2_s={:.3} upload_s={upload:.3} \
         cg_s={:.3} cg_its={its} cg_ms_per_it={:.3} backsub_s={:.3} eval_s={:.3} syncs={} lm={} accepted={} \
         cost={:.6e} termination={:?} dropped_rows={} device_mb={:.1} load_before=[{}] load_after=[{}]",
        r.lane,
        r.wall,
        sum(|s| s.stage1),
        sum(|s| s.stage2),
        pcg - upload,
        1e3 * (pcg - upload) / its.max(1) as f64,
        sum(|s| s.back_substitution),
        sum(|s| s.evaluation),
        r.report
            .iterations
            .iter()
            .map(|i| i.pcg_syncs)
            .sum::<usize>(),
        r.report.iterations.len(),
        r.report.iterations.iter().filter(|i| i.accepted).count(),
        r.report.final_cost.error,
        r.report.termination,
        r.report.dropped_rows,
        r.device_bytes as f64 / 1e6,
        r.load_before,
        loadavg(),
    )
}

/// Keyframe position RMS between two solutions, millimetres (no alignment:
/// both start from one state).
fn pose_rms_mm(a: &GbaProblem<f64>, b: &GbaProblem<f64>) -> f64 {
    let sum: f64 = a
        .imu_from_world
        .iter()
        .zip(&b.imu_from_world)
        .map(|(x, y)| (x.inverse().translation - y.inverse().translation).norm_squared())
        .sum();
    1e3 * (sum / a.imu_from_world.len() as f64).sqrt()
}

/// The same RMS after the rigid transform that best maps `a`'s keyframe
/// positions onto `b`'s (Kabsch): the free gauge taken out, as the product's
/// `remove_gauge` does.
fn aligned_rms_mm(a: &GbaProblem<f64>, b: &GbaProblem<f64>) -> f64 {
    use nalgebra::{Matrix3, Vector3};
    let pos = |p: &GbaProblem<f64>| -> Vec<Vector3<f64>> {
        p.imu_from_world
            .iter()
            .map(|t| t.inverse().translation)
            .collect()
    };
    let (pa, pb) = (pos(a), pos(b));
    let n = pa.len() as f64;
    let (ca, cb) = (
        pa.iter().sum::<Vector3<f64>>() / n,
        pb.iter().sum::<Vector3<f64>>() / n,
    );
    let mut h = Matrix3::zeros();
    for (x, y) in pa.iter().zip(&pb) {
        h += (x - ca) * (y - cb).transpose();
    }
    let svd = h.svd(true, true);
    let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (v_t.transpose() * u.transpose()).determinant().signum();
    let r = v_t.transpose() * d * u.transpose();
    let sum: f64 = pa
        .iter()
        .zip(&pb)
        .map(|(x, y)| (r * (x - ca) + cb - y).norm_squared())
        .sum();
    1e3 * (sum / n).sqrt()
}

#[test]
#[ignore = "reads RootSLAM's exported GBA problems from the NAS; a benchmark"]
fn band_and_speed_on_exported_problems() {
    let seqs = std::env::var("GBA_LANES_SEQS")
        .unwrap_or_else(|_| "MGO05,MGO07,MGO13,TM1,TM2,MIPB02".into());
    let lanes = std::env::var("GBA_LANES_RUNS").unwrap_or_else(|_| "cpu,gpu,cpu,gpu,cpu64".into());
    let threads: usize = std::env::var("GBA_LANES_THREADS")
        .ok()
        .map_or(2, |t| t.parse().unwrap());
    // Bring the device up and compile the kernels before the first timed run,
    // as the product's startup probe does.
    probe_pcg(&gpu_client().unwrap()).unwrap();
    for seq in seqs.split(',') {
        let export = load(seq);
        let runs: Vec<Run> = lanes
            .split(',')
            .map(|lane| {
                let r = run(&export, lane, threads);
                println!("{}", line(seq, threads, &r));
                r
            })
            .collect();
        let first = |lane: &str| runs.iter().find(|r| r.lane == lane);
        let cost_rel = |a: &Run, b: &Run| {
            (a.report.final_cost.error - b.report.final_cost.error) / b.report.final_cost.error
        };
        let mut band = format!("BAND seq={seq}");
        for (a, b) in [("gpu", "cpu"), ("gpu", "cpu64"), ("cpu", "cpu64")] {
            if let (Some(x), Some(y)) = (first(a), first(b)) {
                band += &format!(
                    " {a}_vs_{b}: cost_rel={:+.3e} kf_rms_mm={:.4} kf_rms_aligned_mm={:.4}",
                    cost_rel(x, y),
                    pose_rms_mm(&x.solved, &y.solved),
                    aligned_rms_mm(&x.solved, &y.solved)
                );
            }
        }
        if let (Some(x), Some(y)) = (runs.iter().filter(|r| r.lane == "gpu").nth(1), first("gpu")) {
            band += &format!(" gpu_runs_bit_equal={}", x.solved == y.solved);
        }
        println!("{band}");
    }
}
