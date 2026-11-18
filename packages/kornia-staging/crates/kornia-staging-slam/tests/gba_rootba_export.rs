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
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

use kornia_staging_3d::camera::{formats::BasaltCamera, CameraModelKind};
use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use kornia_staging_slam::sqrt_ba::{
    solve_global, GbaCamera, GbaLandmark, GbaObservation, GbaOptions, GbaProblem, GbaReport,
    GbaRigCamera, RelPoseConstraint,
};
use nalgebra::{UnitQuaternion, Vector3};
use serde::Deserialize;

// RootSLAM's map export and Basalt's calibration: third-party formats, read
// partially (unknown fields allowed).

/// `map/map.json`.
#[derive(Deserialize)]
struct MapJson {
    keyframes: Vec<KeyframeJson>,
    landmarks: Vec<LandmarkJson>,
    observations: Vec<ObservationJson>,
}

#[derive(Deserialize)]
struct KeyframeJson {
    id: u64,
    /// `[qw, qx, qy, qz, tx, ty, tz]`.
    #[serde(rename = "T_w_i")]
    world_from_imu: [f64; 7],
}

#[derive(Deserialize)]
struct LandmarkJson {
    id: u64,
    p_w: [f64; 3],
}

#[derive(Deserialize)]
struct ObservationJson {
    lm_id: u64,
    kf_id: u64,
    cam_id: usize,
    pos: [f64; 2],
}

/// `map/pose_graph.json`.
#[derive(Deserialize)]
struct PoseGraphJson {
    edges: Vec<EdgeJson>,
}

#[derive(Deserialize)]
struct EdgeJson {
    kf_id_a: u64,
    kf_id_b: u64,
    /// `[qw, qx, qy, qz, tx, ty, tz]`.
    #[serde(rename = "T_a_b")]
    a_from_b: [f64; 7],
}

/// Basalt calibration file: `{"value0": {...}}`.
#[derive(Deserialize)]
struct CalibrationFile {
    value0: CalibrationJson,
}

#[derive(Deserialize)]
struct CalibrationJson {
    #[serde(rename = "T_imu_cam")]
    imu_from_camera: Vec<ImuFromCameraJson>,
    /// Basalt camera records; `ds` is read here, the rest by [`BasaltCamera`].
    intrinsics: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct ImuFromCameraJson {
    px: f64,
    py: f64,
    pz: f64,
    qx: f64,
    qy: f64,
    qz: f64,
    qw: f64,
}

#[derive(Deserialize)]
struct DoubleSphereJson {
    intrinsics: DoubleSphereIntrinsicsJson,
}

#[derive(Deserialize)]
struct DoubleSphereIntrinsicsJson {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    xi: f64,
    alpha: f64,
}

/// Basalt `ds` (double sphere), absent from the staged camera set.
#[derive(Debug, Clone, Copy)]
struct DoubleSphere {
    f: [f32; 4],
    xi: f32,
    alpha: f32,
}
impl GbaCamera<f32> for DoubleSphere {
    fn project(&self, p: [f32; 3], jacobian: Option<&mut [[f32; 3]; 2]>) -> ([f32; 2], bool) {
        let [fx, fy, cx, cy] = self.f;
        let (xi, alpha) = (self.xi, self.alpha);
        let [x, y, z] = p;
        let d1 = (x * x + y * y + z * z).sqrt();
        let k = xi * d1 + z;
        let d2 = (x * x + y * y + k * k).sqrt();
        let norm = alpha * d2 + (1.0 - alpha) * k;
        let pixel = [fx * x / norm + cx, fy * y / norm + cy];
        let w1 = if alpha <= 0.5 {
            alpha / (1.0 - alpha)
        } else {
            (1.0 - alpha) / alpha
        };
        let w2 = (w1 + xi) / (2.0 * w1 * xi + xi * xi + 1.0).sqrt();
        let valid = z > -w2 * d1;
        if let Some(j) = jacobian {
            let dk = [xi * x / d1, xi * y / d1, xi * z / d1 + 1.0];
            let pv = [x, y, 0.0];
            let mut dn = [0.0f32; 3];
            for i in 0..3 {
                let dd2 = (pv[i] + k * dk[i]) / d2;
                dn[i] = alpha * dd2 + (1.0 - alpha) * dk[i];
            }
            for i in 0..3 {
                let e = |c: usize| if c == i { 1.0 } else { 0.0 };
                j[0][i] = fx * (e(0) / norm - x * dn[i] / (norm * norm));
                j[1][i] = fy * (e(1) / norm - y * dn[i] / (norm * norm));
            }
        }
        (pixel, valid)
    }
}

#[derive(Debug, Clone, Copy)]
enum ExportCamera {
    Staged(CameraModelKind<f32>),
    Ds(DoubleSphere),
}

impl GbaCamera<f32> for ExportCamera {
    fn project(&self, p: [f32; 3], jacobian: Option<&mut [[f32; 3]; 2]>) -> ([f32; 2], bool) {
        match self {
            Self::Staged(c) => GbaCamera::project(c, p, jacobian),
            Self::Ds(c) => c.project(p, jacobian),
        }
    }
}

fn runs_root() -> PathBuf {
    std::env::var_os("GBA_EXPORT_RUNS").map_or_else(
        || PathBuf::from("/mnt/nas/ssd-8tb/scratch/rootslam/runs"),
        PathBuf::from,
    )
}

fn calib_path(seq: &str) -> PathBuf {
    let root = std::env::var_os("GBA_EXPORT_CALIB").map_or_else(
        || PathBuf::from(std::env::var("HOME").unwrap()).join("0Dev/forks/rootslam/data"),
        PathBuf::from,
    );
    match seq {
        s if s.starts_with("MG") => root.join("msd/msdmg_calib.json"),
        s if s.starts_with("MI") => root.join("msd/msdmi_calib.json"),
        s if s.starts_with("MO") => root.join("msd/msdmo_calib.json"),
        s if s.starts_with('T') => root.join("tum/tumvi_512_ds_calib.json"),
        _ => root.join("euroc/euroc_ds_calib.json"),
    }
}

/// `[qw, qx, qy, qz, tx, ty, tz]`, as rootba reads it.
fn pose7(a: &[f64; 7]) -> RigidTransform<f64> {
    let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(a[0], a[1], a[2], a[3]));
    RigidTransform::new(
        Rotation3::from_unit_quaternion(q),
        Vector3::new(a[4], a[5], a[6]),
    )
}

/// The asset's text; panics naming the asset when it is absent (the lanes that
/// read assets are `#[ignore]`, so running them is an explicit request).
fn read_asset(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("asset {} is absent or unreadable: {e}", path.display()))
}

fn parse<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    serde_json::from_str(&read_asset(path))
        .unwrap_or_else(|e| panic!("asset {} does not parse: {e}", path.display()))
}

/// What rootba `bal` reports in `output.<SEQ>.<variant>.log`.
#[derive(Debug, Clone, Copy)]
struct RootbaRun {
    final_cost: f64,
    accepted: usize,
    rejected: usize,
    cg_iterations: usize,
}

fn rootba_run(seq: &str, variant: &str) -> RootbaRun {
    let path = runs_root()
        .join(seq)
        .join(format!("output.{seq}.{variant}.log"));
    let log = read_asset(&path);
    let final_cost = log
        .lines()
        .find(|l| l.starts_with("Final Cost: error valid:"))
        .and_then(|l| l.split_whitespace().nth(4)?.parse().ok())
        .unwrap_or_else(|| panic!("{} has no 'Final Cost' line", path.display()));
    let attempts = log
        .lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("[Success]") || l.starts_with("[Reject]"));
    let (mut accepted, mut rejected, mut cg_iterations) = (0, 0, 0);
    for line in attempts {
        if line.starts_with("[Success]") {
            accepted += 1;
        } else {
            rejected += 1;
        }
        cg_iterations += line
            .split("cg_iter: ")
            .nth(1)
            .and_then(|t| t.split(',').next()?.parse::<usize>().ok())
            .unwrap_or_else(|| panic!("{}: no cg_iter in {line}", path.display()));
    }
    RootbaRun {
        final_cost,
        accepted,
        rejected,
        cg_iterations,
    }
}

struct Export {
    problem: GbaProblem<f64>,
    rig: Vec<GbaRigCamera<f32, ExportCamera>>,
}

fn load(seq: &str) -> Export {
    let dir = runs_root().join(seq).join("map");
    let map: MapJson = parse(&dir.join("map.json"));
    let graph: PoseGraphJson = parse(&dir.join("pose_graph.json"));
    let calib = parse::<CalibrationFile>(&calib_path(seq)).value0;

    let kf_index: HashMap<u64, usize> = map
        .keyframes
        .iter()
        .enumerate()
        .map(|(i, kf)| (kf.id, i))
        .collect();
    let imu_from_world = map
        .keyframes
        .iter()
        .map(|kf| pose7(&kf.world_from_imu).inverse())
        .collect();
    let lm_index: HashMap<u64, usize> = map
        .landmarks
        .iter()
        .enumerate()
        .map(|(i, lm)| (lm.id, i))
        .collect();
    let mut landmarks: Vec<GbaLandmark<f64>> = map
        .landmarks
        .iter()
        .map(|lm| GbaLandmark {
            position: Vector3::from(lm.p_w),
            observations: Vec::new(),
        })
        .collect();
    for obs in &map.observations {
        landmarks[lm_index[&obs.lm_id]]
            .observations
            .push(GbaObservation {
                keyframe: kf_index[&obs.kf_id],
                camera: obs.cam_id,
                pixel: obs.pos,
            });
    }
    let rel_poses = graph
        .edges
        .iter()
        .filter_map(|e| {
            let a = *kf_index.get(&e.kf_id_a)?;
            let b = *kf_index.get(&e.kf_id_b)?;
            (a != b).then(|| RelPoseConstraint::with_sigmas(a, b, pose7(&e.a_from_b), 0.05, 0.01))
        })
        .collect();
    let rig = calib
        .intrinsics
        .iter()
        .zip(&calib.imu_from_camera)
        .map(|(intr, t)| {
            let model = if intr["camera_type"] == "ds" {
                let p = DoubleSphereJson::deserialize(intr).unwrap().intrinsics;
                ExportCamera::Ds(DoubleSphere {
                    f: [p.fx, p.fy, p.cx, p.cy].map(|v| v as f32),
                    xi: p.xi as f32,
                    alpha: p.alpha as f32,
                })
            } else {
                let basalt = BasaltCamera::<f32>::deserialize(intr).unwrap();
                ExportCamera::Staged(CameraModelKind::try_from(&basalt).unwrap())
            };
            let q =
                UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(t.qw, t.qx, t.qy, t.qz));
            let imu_from_camera = RigidTransform::new(
                Rotation3::from_unit_quaternion(q),
                Vector3::new(t.px, t.py, t.pz),
            );
            GbaRigCamera {
                model,
                imu_from_camera: imu_from_camera.cast(),
            }
        })
        .collect();
    let mut problem = GbaProblem {
        imu_from_world,
        landmarks,
        rel_poses,
    };
    let landmarks = problem.landmarks.len();
    let dropped = problem.drop_sparse_observations(10);
    eprintln!(
        "{seq}: {} keyframes, {} landmarks, {} observations, {} pose-graph rows (dropped {} obs, {} landmarks)",
        problem.imu_from_world.len(),
        problem.landmarks.len(),
        problem.landmarks.iter().map(|l| l.observations.len()).sum::<usize>(),
        problem.rel_poses.len(),
        dropped.observations,
        landmarks - dropped.kept_landmarks.len()
    );
    Export { problem, rig }
}

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
