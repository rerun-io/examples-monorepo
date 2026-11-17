//! Global BA on RootSLAM's exported GBA problems (rootba `bal` input), read at
//! test time. Assets: `<runs>/<SEQ>/map/{map,pose_graph}.json` and the
//! `output.<SEQ>.balpgf.log` of the reproduction (default runs root
//! `/mnt/nas/ssd-8tb/scratch/rootslam/runs`, override `GBA_EXPORT_RUNS`), and the
//! RootSLAM Basalt calibrations (default `~/0Dev/forks/rootslam/data`, override
//! `GBA_EXPORT_CALIB`). Both lanes are `#[ignore]`:
//! `cargo test --release -p kornia-staging-slam --test gba_rootba_export -- --ignored --nocapture`.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::{collections::HashMap, path::PathBuf, time::Instant};

use kornia_staging_3d::camera::{formats::BasaltCamera, CameraModelKind};
use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use kornia_staging_slam::sqrt_ba::{
    solve_global, GbaCamera, GbaLandmark, GbaObservation, GbaOptions, GbaProblem, GbaReport,
    GbaRigCamera, RelPoseConstraint,
};
use nalgebra::{UnitQuaternion, Vector3};
use serde_json::Value;

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

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

/// `[qw, qx, qy, qz, tx, ty, tz]`, as rootba reads it.
fn pose7(v: &Value) -> RigidTransform<f64> {
    let a: Vec<f64> = v.as_array().unwrap().iter().map(f).collect();
    let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(a[0], a[1], a[2], a[3]));
    RigidTransform::new(
        Rotation3::from_unit_quaternion(q),
        Vector3::new(a[4], a[5], a[6]),
    )
}

struct Export {
    problem: GbaProblem<f64>,
    rig: Vec<GbaRigCamera<f32, ExportCamera>>,
    rootba_final_cost: Option<f64>,
}

/// `None` (with a message naming the asset) when an input is missing.
fn load(seq: &str) -> Option<Export> {
    let dir = runs_root().join(seq);
    let calib = calib_path(seq);
    for path in [
        dir.join("map/map.json"),
        dir.join("map/pose_graph.json"),
        calib.clone(),
    ] {
        if !path.exists() {
            eprintln!("SKIP {seq}: asset {} is absent", path.display());
            return None;
        }
    }
    let read = |p: PathBuf| -> Value {
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    };
    let map = read(dir.join("map/map.json"));
    let graph = read(dir.join("map/pose_graph.json"));
    let calib = read(calib)["value0"].clone();

    let mut kf_index = HashMap::new();
    let imu_from_world = map["keyframes"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, kf)| {
            kf_index.insert(kf["id"].as_u64().unwrap(), i);
            pose7(&kf["T_w_i"]).inverse()
        })
        .collect::<Vec<_>>();
    let mut lm_index = HashMap::new();
    let mut landmarks: Vec<GbaLandmark<f64>> = map["landmarks"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, lm)| {
            lm_index.insert(lm["id"].as_u64().unwrap(), i);
            let p: Vec<f64> = lm["p_w"].as_array().unwrap().iter().map(f).collect();
            GbaLandmark {
                position: Vector3::new(p[0], p[1], p[2]),
                observations: Vec::new(),
            }
        })
        .collect();
    for obs in map["observations"].as_array().unwrap() {
        let pos = obs["pos"].as_array().unwrap();
        landmarks[lm_index[&obs["lm_id"].as_u64().unwrap()]]
            .observations
            .push(GbaObservation {
                keyframe: kf_index[&obs["kf_id"].as_u64().unwrap()],
                camera: obs["cam_id"].as_u64().unwrap() as usize,
                pixel: [f(&pos[0]), f(&pos[1])],
            });
    }
    let rel_poses = graph["edges"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| {
            let a = *kf_index.get(&e["kf_id_a"].as_u64()?)?;
            let b = *kf_index.get(&e["kf_id_b"].as_u64()?)?;
            (a != b).then(|| RelPoseConstraint::with_sigmas(a, b, pose7(&e["T_a_b"]), 0.05, 0.01))
        })
        .collect();
    let rig = calib["intrinsics"]
        .as_array()
        .unwrap()
        .iter()
        .zip(calib["T_imu_cam"].as_array().unwrap())
        .map(|(intr, t)| {
            let model = if intr["camera_type"] == "ds" {
                let p = &intr["intrinsics"];
                ExportCamera::Ds(DoubleSphere {
                    f: ["fx", "fy", "cx", "cy"].map(|k| f(&p[k]) as f32),
                    xi: f(&p["xi"]) as f32,
                    alpha: f(&p["alpha"]) as f32,
                })
            } else {
                let basalt: BasaltCamera<f32> = serde_json::from_value(intr.clone()).unwrap();
                ExportCamera::Staged(CameraModelKind::try_from(&basalt).unwrap())
            };
            let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                f(&t["qw"]),
                f(&t["qx"]),
                f(&t["qy"]),
                f(&t["qz"]),
            ));
            let imu_from_camera = RigidTransform::new(
                Rotation3::from_unit_quaternion(q),
                Vector3::new(f(&t["px"]), f(&t["py"]), f(&t["pz"])),
            );
            GbaRigCamera {
                model,
                imu_from_camera: imu_from_camera.cast(),
            }
        })
        .collect();
    let rootba_final_cost = std::fs::read_to_string(dir.join(format!("output.{seq}.balpgf.log")))
        .ok()
        .and_then(|log| {
            let line = log
                .lines()
                .find(|l| l.starts_with("Final Cost: error valid:"))?;
            line.split_whitespace().nth(4)?.parse().ok()
        });
    let mut problem = GbaProblem {
        imu_from_world,
        landmarks,
        rel_poses,
    };
    let (obs, lms) = problem.drop_sparse_observations(10);
    eprintln!(
        "{seq}: {} keyframes, {} landmarks, {} observations, {} pose-graph rows (dropped {obs} obs, {lms} landmarks)",
        problem.imu_from_world.len(),
        problem.landmarks.len(),
        problem.landmarks.iter().map(|l| l.observations.len()).sum::<usize>(),
        problem.rel_poses.len()
    );
    Some(Export {
        problem,
        rig,
        rootba_final_cost,
    })
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

/// DESIGN §7.4 / §10 d1: our f32 GBA on RootSLAM's exported MGO07 problem reaches
/// rootba `bal`'s final cost (balpgf, same options) within 0.5 %.
#[test]
#[ignore = "reads RootSLAM's exported MGO07 GBA problem from the NAS"]
fn mgo07_reaches_rootba_final_cost() {
    let Some(export) = load("MGO07") else { return };
    let Some(rootba) = export.rootba_final_cost else {
        eprintln!("SKIP: output.MGO07.balpgf.log (rootba's final cost) is absent");
        return;
    };
    let (report, wall) = run(&export, 4);
    eprintln!(
        "{}\nrootba bal final cost {rootba:.4e}",
        summary("MGO07", 4, &report, wall)
    );
    assert!(
        report.final_cost.error <= rootba * 1.005,
        "{} vs rootba {rootba}",
        report.final_cost.error
    );
    assert!(
        report.final_cost.error >= rootba * 0.95,
        "suspiciously low: {} vs {rootba}",
        report.final_cost.error
    );
    assert_eq!(report.nonfinite_landmark_steps, 0);
}

/// CG ms per iteration and totals. `GBA_BENCH_SEQS` (default `MGO07,TM1`) and
/// `GBA_BENCH_THREADS` (default `2,8`); pin with `taskset`.
#[test]
#[ignore = "benchmark on RootSLAM's exported GBA problems from the NAS"]
fn bench_exported_problems() {
    let seqs = std::env::var("GBA_BENCH_SEQS").unwrap_or_else(|_| "MGO07,TM1".into());
    let threads = std::env::var("GBA_BENCH_THREADS").unwrap_or_else(|_| "2,8".into());
    for seq in seqs.split(',') {
        let Some(export) = load(seq) else { continue };
        for t in threads.split(',').map(|t| t.parse::<usize>().unwrap()) {
            let (report, wall) = run(&export, t);
            println!(
                "BENCH {} rootba_final={:?}",
                summary(seq, t, &report, wall),
                export.rootba_final_cost
            );
        }
    }
}
