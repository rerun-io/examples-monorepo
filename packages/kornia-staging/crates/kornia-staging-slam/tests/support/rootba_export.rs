//! RootSLAM's exported GBA problems (rootba `bal` input) and rootba's logs,
//! read at test time; shared by the CPU golden (`gba_rootba_export.rs`) and the
//! device lane's band and speed lanes (`kornia-staging-gpu/tests/gba_lanes.rs`)
//! and a consumer's driver golden, each through `#[path]`.
//! Assets: `<runs>/<SEQ>/map/{map,pose_graph}.json` and
//! `output.<SEQ>.{balpgf,balpgd}.log` (default runs root
//! `/mnt/nas/ssd-8tb/scratch/rootslam/runs`, override `GBA_EXPORT_RUNS`), and
//! the RootSLAM Basalt calibrations (default `~/0Dev/forks/rootslam/data`,
//! override `GBA_EXPORT_CALIB`). A missing asset panics and names it.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use kornia_staging_3d::camera::{CameraModelKind, formats::BasaltCamera};
use kornia_staging_algebra::{
    Scalar,
    lie::{RigidTransform, Rotation3},
};
use kornia_staging_slam::sqrt_ba::{
    GbaCamera, GbaLandmark, GbaObservation, GbaProblem, GbaRigCamera, RelPoseConstraint,
};
use nalgebra::{UnitQuaternion, Vector3};
use serde::Deserialize;

// RootSLAM's map export and Basalt's calibration: third-party formats, read
// partially (unknown fields allowed).

/// `map/map.json`.
#[derive(Deserialize)]
pub struct MapJson {
    pub keyframes: Vec<KeyframeJson>,
    pub landmarks: Vec<LandmarkJson>,
    pub observations: Vec<ObservationJson>,
}

#[derive(Deserialize)]
pub struct KeyframeJson {
    pub id: u64,
    /// `[qw, qx, qy, qz, tx, ty, tz]`.
    #[serde(rename = "T_w_i")]
    pub world_from_imu: [f64; 7],
}

#[derive(Deserialize)]
pub struct LandmarkJson {
    pub id: u64,
    pub p_w: [f64; 3],
}

#[derive(Deserialize)]
pub struct ObservationJson {
    pub lm_id: u64,
    pub kf_id: u64,
    pub cam_id: usize,
    pub pos: [f64; 2],
}

/// `map/pose_graph.json`.
#[derive(Deserialize)]
pub struct PoseGraphJson {
    pub edges: Vec<EdgeJson>,
}

#[derive(Deserialize)]
pub struct EdgeJson {
    pub kf_id_a: u64,
    pub kf_id_b: u64,
    /// `[qw, qx, qy, qz, tx, ty, tz]`.
    #[serde(rename = "T_a_b")]
    pub a_from_b: [f64; 7],
}

/// Basalt calibration file: `{"value0": {...}}`.
#[derive(Deserialize)]
pub struct CalibrationFile {
    value0: CalibrationJson,
}

#[derive(Deserialize)]
pub struct CalibrationJson {
    #[serde(rename = "T_imu_cam")]
    imu_from_camera: Vec<ImuFromCameraJson>,
    /// Basalt camera records; `ds` is read here, the rest by [`BasaltCamera`].
    intrinsics: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct ImuFromCameraJson {
    px: f64,
    py: f64,
    pz: f64,
    qx: f64,
    qy: f64,
    qz: f64,
    qw: f64,
}

#[derive(Deserialize)]
pub struct DoubleSphereJson {
    intrinsics: DoubleSphereIntrinsicsJson,
}

#[derive(Deserialize)]
pub struct DoubleSphereIntrinsicsJson {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    xi: f64,
    alpha: f64,
}

/// Basalt `ds` (double sphere), absent from the staged camera set.
#[derive(Debug, Clone, Copy)]
pub struct DoubleSphere<S> {
    pub f: [S; 4],
    pub xi: S,
    pub alpha: S,
}
impl<S: Scalar> GbaCamera<S> for DoubleSphere<S> {
    fn project(&self, p: [S; 3], jacobian: Option<&mut [[S; 3]; 2]>) -> ([S; 2], bool) {
        let [fx, fy, cx, cy] = self.f;
        let (xi, alpha) = (self.xi, self.alpha);
        let (one, two, half) = (S::one(), S::from_literal(2.0), S::from_literal(0.5));
        let [x, y, z] = p;
        let d1 = (x * x + y * y + z * z).sqrt();
        let k = xi * d1 + z;
        let d2 = (x * x + y * y + k * k).sqrt();
        let norm = alpha * d2 + (one - alpha) * k;
        let pixel = [fx * x / norm + cx, fy * y / norm + cy];
        let w1 = if alpha <= half {
            alpha / (one - alpha)
        } else {
            (one - alpha) / alpha
        };
        let w2 = (w1 + xi) / (two * w1 * xi + xi * xi + one).sqrt();
        let valid = z > -w2 * d1;
        if let Some(j) = jacobian {
            let dk = [xi * x / d1, xi * y / d1, xi * z / d1 + one];
            let pv = [x, y, S::zero()];
            let mut dn = [S::zero(); 3];
            for i in 0..3 {
                let dd2 = (pv[i] + k * dk[i]) / d2;
                dn[i] = alpha * dd2 + (one - alpha) * dk[i];
            }
            for i in 0..3 {
                let e = |c: usize| if c == i { one } else { S::zero() };
                j[0][i] = fx * (e(0) / norm - x * dn[i] / (norm * norm));
                j[1][i] = fy * (e(1) / norm - y * dn[i] / (norm * norm));
            }
        }
        (pixel, valid)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExportCamera<S: Scalar> {
    Staged(CameraModelKind<S>),
    Ds(DoubleSphere<S>),
}

impl<S: Scalar> GbaCamera<S> for ExportCamera<S> {
    fn project(&self, p: [S; 3], jacobian: Option<&mut [[S; 3]; 2]>) -> ([S; 2], bool) {
        match self {
            Self::Staged(c) => GbaCamera::project(c, p, jacobian),
            Self::Ds(c) => c.project(p, jacobian),
        }
    }
}

pub fn runs_root() -> PathBuf {
    std::env::var_os("GBA_EXPORT_RUNS").map_or_else(
        || PathBuf::from("/mnt/nas/ssd-8tb/scratch/rootslam/runs"),
        PathBuf::from,
    )
}

pub fn calib_path(seq: &str) -> PathBuf {
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
pub fn pose7(a: &[f64; 7]) -> RigidTransform<f64> {
    let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(a[0], a[1], a[2], a[3]));
    RigidTransform::new(
        Rotation3::from_unit_quaternion(q),
        Vector3::new(a[4], a[5], a[6]),
    )
}

/// The asset's text; panics naming the asset when it is absent (the lanes that
/// read assets are `#[ignore]`, so running them is an explicit request).
pub fn read_asset(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("asset {} is absent or unreadable: {e}", path.display()))
}

pub fn parse<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    serde_json::from_str(&read_asset(path))
        .unwrap_or_else(|e| panic!("asset {} does not parse: {e}", path.display()))
}

/// What rootba `bal` reports in `output.<SEQ>.<variant>.log`.
#[derive(Debug, Clone, Copy)]
pub struct RootbaRun {
    pub final_cost: f64,
    pub accepted: usize,
    pub rejected: usize,
    pub cg_iterations: usize,
}

pub fn rootba_run(seq: &str, variant: &str) -> RootbaRun {
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

pub struct Export {
    pub problem: GbaProblem<f64>,
    pub rig: Vec<GbaRigCamera<f32, ExportCamera<f32>>>,
    pub rig64: Vec<GbaRigCamera<f64, ExportCamera<f64>>>,
}

fn export_rig<S: Scalar + for<'de> Deserialize<'de>>(
    calib: &CalibrationJson,
) -> Vec<GbaRigCamera<S, ExportCamera<S>>> {
    calib
        .intrinsics
        .iter()
        .zip(&calib.imu_from_camera)
        .map(|(intr, t)| {
            let model = if intr["camera_type"] == "ds" {
                let p = DoubleSphereJson::deserialize(intr).unwrap().intrinsics;
                ExportCamera::Ds(DoubleSphere {
                    f: [p.fx, p.fy, p.cx, p.cy].map(S::from_literal),
                    xi: S::from_literal(p.xi),
                    alpha: S::from_literal(p.alpha),
                })
            } else {
                let basalt = BasaltCamera::<S>::deserialize(intr).unwrap();
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
        .collect()
}

pub fn load(seq: &str) -> Export {
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
    let (rig, rig64) = (export_rig(&calib), export_rig(&calib));
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
        problem
            .landmarks
            .iter()
            .map(|l| l.observations.len())
            .sum::<usize>(),
        problem.rel_poses.len(),
        dropped.observations,
        landmarks - dropped.kept_landmarks.len()
    );
    Export {
        problem,
        rig,
        rig64,
    }
}
