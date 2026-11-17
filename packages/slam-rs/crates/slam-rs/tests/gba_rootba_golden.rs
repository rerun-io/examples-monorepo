//! The global BA driver on RootSLAM's exported MGO07 and MGO13 maps, against
//! rootba `bal` (DESIGN §7.4, PR d2). The map goes through the same path as
//! `Vio::finish_map`: a [`KeyframeMap`] with host-relative landmarks, then
//! `map_problem`, `solve` (f64 centring, f32) and `apply`, without
//! `remove_gauge`: rootba holds no pose and moves no gauge either, so the
//! keyframes are compared with the solver's output before the re-alignment.
//!
//! Assets, read at test time as third-party formats: `<runs>/<seq>/map/{map,
//! pose_graph}.json` and `map/optimized_tracking_pg_float.csv`, and the
//! `output.<seq>.{balpgf,balpgd}.log` of the reproduction (default runs root
//! `/mnt/nas/ssd-8tb/scratch/rootslam/runs`, override `GBA_EXPORT_RUNS`), and
//! RootSLAM's Basalt calibration `msd/msdmg_calib.json` (default
//! `~/0Dev/forks/rootslam/data`, override `GBA_EXPORT_CALIB`). `#[ignore]`: run
//! them with `pixi run -e slam-rs-dev slam-rs-gba-golden`; a missing asset then
//! fails and names the asset.

#![cfg(feature = "backend")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, UnitQuaternion, Vector2, Vector3};
use serde::Deserialize;
use slam_rs::backend::gba::{GbaRig, apply, map_problem, solve};
use slam_rs::backend::map::host_relative;
use slam_rs::backend::{KeyframeMap, PoseGraphEdge, PoseGraphEdgeKind};
use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::estimator::StampLandmark;
use slam_rs::lie::{Se3, So3};
use slam_rs::types::{LandmarkId, TimeCamId};

#[derive(Deserialize)]
struct MapJson {
    keyframes: Vec<KeyframeJson>,
    landmarks: Vec<LandmarkJson>,
    observations: Vec<ObservationJson>,
}

#[derive(Deserialize)]
struct KeyframeJson {
    id: i64,
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
    kf_id: i64,
    cam_id: usize,
    pos: [f64; 2],
}

#[derive(Deserialize)]
struct PoseGraphJson {
    edges: Vec<EdgeJson>,
}

#[derive(Deserialize)]
struct EdgeJson {
    kf_id_a: i64,
    kf_id_b: i64,
}

fn runs_root() -> PathBuf {
    std::env::var_os("GBA_EXPORT_RUNS").map_or_else(
        || PathBuf::from("/mnt/nas/ssd-8tb/scratch/rootslam/runs"),
        PathBuf::from,
    )
}

fn calib_path() -> PathBuf {
    std::env::var_os("GBA_EXPORT_CALIB")
        .map_or_else(
            || PathBuf::from(std::env::var("HOME").unwrap()).join("0Dev/forks/rootslam/data"),
            PathBuf::from,
        )
        .join("msd/msdmg_calib.json")
}

fn read_asset(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("asset {} is absent or unreadable: {e}", path.display()))
}

fn pose7(a: &[f64; 7]) -> Se3<f64> {
    let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(a[0], a[1], a[2], a[3]));
    Se3::new(So3::from_unit_quaternion(q), Vector3::new(a[4], a[5], a[6]))
}

/// `Final Cost: error valid: <v>` of a rootba log.
fn final_cost(path: &Path) -> f64 {
    read_asset(path)
        .lines()
        .find(|l| l.starts_with("Final Cost: error valid:"))
        .and_then(|l| l.split_whitespace().nth(4)?.parse().ok())
        .unwrap_or_else(|| panic!("{} has no 'Final Cost' line", path.display()))
}

/// Keyframe positions of a EuRoC trajectory CSV (rootba's optimized output).
fn positions(path: &Path) -> BTreeMap<i64, Vector3<f64>> {
    read_asset(path)
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let v: Vec<f64> = l.split(',').map(|x| x.trim().parse().unwrap()).collect();
            (v[0] as i64, Vector3::new(v[1], v[2], v[3]))
        })
        .collect()
}

/// RMS of `a - b` after the rigid alignment (Kabsch) of `a` onto `b`.
fn aligned_rms(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().sum::<Vector3<f64>>() / n,
        b.iter().sum::<Vector3<f64>>() / n,
    );
    let mut h = Matrix3::zeros();
    for (x, y) in a.iter().zip(b) {
        h += (x - ma) * (y - mb).transpose();
    }
    let svd = h.svd(true, true);
    let (u, v_t) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (v_t.transpose() * u.transpose()).determinant().signum();
    let r = v_t.transpose() * d * u.transpose();
    (a.iter()
        .zip(b)
        .map(|(x, y)| (r * (x - ma) + mb - y).norm_squared())
        .sum::<f64>()
        / n)
        .sqrt()
}

fn rms(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> f64 {
    (a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm_squared())
        .sum::<f64>()
        / a.len() as f64)
        .sqrt()
}

#[test]
#[ignore = "reads RootSLAM's exported MGO07 map and rootba's runs from the NAS"]
fn golden_mgo07_driver_matches_rootba_bal_cost_and_keyframe_poses() {
    golden("MGO07");
}

/// MGO13 has 34 keyframes under `min_obs_per_kf`: rootba drops them and their
/// pose-graph rows, we keep those rows (B22). Our cost ends at or below `bal`'s,
/// and we keep every keyframe rootba keeps; the poses are not compared.
#[test]
#[ignore = "reads RootSLAM's exported MGO13 map and rootba's runs from the NAS"]
fn golden_mgo13_driver_keeps_sparse_keyframes_and_ends_at_or_below_rootba_bal() {
    golden("MGO13");
}

fn golden(seq: &str) {
    let run = runs_root().join(seq);
    let export: MapJson = serde_json::from_str(&read_asset(&run.join("map/map.json"))).unwrap();
    let graph: PoseGraphJson =
        serde_json::from_str(&read_asset(&run.join("map/pose_graph.json"))).unwrap();
    let calibration: Calibration<f64> =
        Calibration::from_json_str(&read_asset(&calib_path())).unwrap();
    let models = calibration
        .intrinsics
        .iter()
        .map(|m| m.to_staged().unwrap())
        .collect();
    let rig = GbaRig::new(models, &calibration.t_i_c).unwrap();
    let t_i_c = &calibration.t_i_c;

    // The export as our map: host-relative landmarks, hosted by their first
    // observation (converted once), keyframe observations.
    let mut map = KeyframeMap::new();
    for kf in &export.keyframes {
        map.upsert_keyframe(kf.id, pose7(&kf.world_from_imu), None);
    }
    let mut obs: HashMap<u64, BTreeMap<TimeCamId, Vector2<f32>>> = HashMap::new();
    for o in &export.observations {
        obs.entry(o.lm_id).or_default().insert(
            TimeCamId::new(o.kf_id, o.cam_id),
            Vector2::new(o.pos[0] as f32, o.pos[1] as f32),
        );
    }
    let landmarks: Vec<StampLandmark> = export
        .landmarks
        .iter()
        .filter_map(|lm| {
            let obs = obs.remove(&lm.id)?;
            let host = *obs.keys().next()?;
            let pose = map.keyframe(host.frame_id)?.pose * t_i_c[host.cam_id];
            let (direction, inv_dist) = host_relative(&pose, &Vector3::from(lm.p_w))?;
            Some(StampLandmark {
                id: LandmarkId(lm.id),
                host,
                direction,
                inv_dist,
                obs,
            })
        })
        .collect();
    map.merge_landmarks(&landmarks);
    let edges: Vec<PoseGraphEdge> = graph
        .edges
        .iter()
        .map(|e| PoseGraphEdge {
            from: e.kf_id_a,
            to: e.kf_id_b,
            kind: PoseGraphEdgeKind::SpanningTree,
        })
        .collect();

    let config = VioConfig::default();
    let input = map_problem(&map, &edges, t_i_c, &config);
    let (solved, report) = solve(&input, &rig, &config).unwrap();
    apply(&mut map, &input, &solved, t_i_c);
    let ours = report.final_cost.error;
    let (bal_f32, bal_f64) = (
        final_cost(&run.join(format!("output.{seq}.balpgf.log"))),
        final_cost(&run.join(format!("output.{seq}.balpgd.log"))),
    );
    eprintln!(
        "{seq} driver: {:?}; cost {:.4e} -> {ours:.4e} in {} attempts, {} CG; rootba bal f32 {bal_f32:.4e}, f64 {bal_f64:.4e}",
        input.counts,
        report.initial_cost.error,
        report.iterations.len(),
        report.pcg_iterations()
    );
    // d1's band: rootba's final cost (f32 and f64) within 0.5 %, not suspiciously
    // lower. With sparse keyframes, B22 keeps rows rootba drops: at or below bal.
    let sparse = input.counts.keyframes_without_reprojection;
    for bal in [bal_f32, bal_f64] {
        let upper = if sparse > 0 { 1.0 } else { 1.005 };
        assert!(
            ours <= bal * upper && ours >= bal * 0.95,
            "ours {ours} vs bal {bal} ({sparse} sparse keyframes)"
        );
    }

    // Keyframe poses against rootba f32's optimized keyframes (DESIGN §7.4: RMS
    // within 1 mm on MG). Neither run holds a pose constant and both start from
    // the same poses, so the gauge barely floats: the raw RMS is within 1 mm of
    // the aligned one. rootba keeps the keyframes with at least
    // `min_obs_per_kf` observations in map.json (its filter_kf); we keep all.
    let bal_poses = positions(&run.join("map/optimized_tracking_pg_float.csv"));
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (id, keyframe) in map.keyframes() {
        if let Some(p) = bal_poses.get(id) {
            a.push(keyframe.pose.translation);
            b.push(*p);
        }
    }
    let mut per_keyframe: HashMap<i64, usize> = HashMap::new();
    for o in &export.observations {
        *per_keyframe.entry(o.kf_id).or_default() += 1;
    }
    let kept = export
        .keyframes
        .iter()
        .filter(|kf| per_keyframe.get(&kf.id).copied().unwrap_or(0) >= config.gba_min_obs_per_kf)
        .count();
    assert_eq!(
        a.len(),
        kept,
        "{} of {} keyframes in rootba's output, {kept} with >= {} observations",
        a.len(),
        export.keyframes.len(),
        config.gba_min_obs_per_kf
    );
    let (raw, aligned) = (rms(&a, &b), aligned_rms(&a, &b));
    eprintln!(
        "{seq} keyframes vs rootba bal f32: {} compared, RMS raw {:.3} mm, SE3-aligned {:.3} mm",
        a.len(),
        1e3 * raw,
        1e3 * aligned
    );
    // With sparse keyframes rootba also drops their pose-graph rows, so it
    // solves another problem (MGO13: 38 mm apart): the band needs equal problems.
    if sparse == 0 {
        assert!(aligned < 1e-3, "aligned keyframe RMS {aligned} m");
        assert!(
            raw - aligned < 1e-3,
            "the gauge floated: raw {raw} m vs aligned {aligned} m"
        );
    }
}
