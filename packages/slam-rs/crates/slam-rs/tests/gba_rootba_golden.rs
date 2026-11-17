//! The global BA driver on RootSLAM's exported MGO07 map, against rootba `bal`
//! (DESIGN §7.4, PR d2). The map goes through the same path as
//! `Vio::finish_map`: a [`KeyframeMap`] with host-relative landmarks, then
//! `map_problem`, `solve` (f64 centring, f32) and `apply`.
//!
//! Assets, read at test time as third-party formats: `<runs>/MGO07/map/{map,
//! pose_graph}.json` and `map/optimized_tracking_pg_float.csv`, and the
//! `output.MGO07.{balpgf,balpgd}.log` of the reproduction (default runs root
//! `/mnt/nas/ssd-8tb/scratch/rootslam/runs`, override `GBA_EXPORT_RUNS`), and
//! RootSLAM's Basalt calibration `msd/msdmg_calib.json` (default
//! `~/0Dev/forks/rootslam/data`, override `GBA_EXPORT_CALIB`). `#[ignore]`: run
//! it with `--ignored`; a missing asset then fails and names the asset.

#![cfg(feature = "backend")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, UnitQuaternion, Vector2, Vector3};
use serde::Deserialize;
use slam_rs::backend::gba::{GbaRig, GbaSettings, apply, host_relative, map_problem, solve};
use slam_rs::backend::{KeyframeMap, PoseGraphEdge, PoseGraphEdgeKind};
use slam_rs::calib::Calibration;
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
    let run = runs_root().join("MGO07");
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
    let rig = GbaRig::new(models, calibration.t_i_c.clone()).unwrap();

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
            let pose = map.keyframe(host.frame_id)?.pose * rig.t_i_c[host.cam_id];
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

    let settings = GbaSettings::default();
    let input = map_problem(&map, &edges, &rig.t_i_c, &settings);
    let solved = solve(&input, &rig, &settings).unwrap();
    apply(&mut map, &input, &solved.problem, &rig.t_i_c);
    let ours = solved.report.final_cost.error;
    let (bal_f32, bal_f64) = (
        final_cost(&run.join("output.MGO07.balpgf.log")),
        final_cost(&run.join("output.MGO07.balpgd.log")),
    );
    eprintln!(
        "MGO07 driver: {:?}; cost {:.4e} -> {ours:.4e} in {} attempts, {} CG; rootba bal f32 {bal_f32:.4e}, f64 {bal_f64:.4e}",
        input.counts,
        solved.report.initial_cost.error,
        solved.report.iterations.len(),
        solved.report.pcg_iterations()
    );
    // d1's band: rootba's final cost (f32 and f64) within 0.5 %, not suspiciously lower.
    for bal in [bal_f32, bal_f64] {
        assert!(
            ours <= bal * 1.005 && ours >= bal * 0.95,
            "ours {ours} vs bal {bal}"
        );
    }

    // Keyframe poses against rootba f32's optimized keyframes (DESIGN §7.4: RMS
    // within 1 mm on MG). Neither run holds a pose constant and both start from
    // the same poses, so the gauge barely floats: the raw RMS is within 1 mm of
    // the aligned one.
    let bal_poses = positions(&run.join("map/optimized_tracking_pg_float.csv"));
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (id, keyframe) in map.keyframes() {
        if let Some(p) = bal_poses.get(id) {
            a.push(keyframe.pose.translation);
            b.push(*p);
        }
    }
    assert!(
        a.len() + 5 >= export.keyframes.len(),
        "{} of {} keyframes in rootba's output",
        a.len(),
        export.keyframes.len()
    );
    let (raw, aligned) = (rms(&a, &b), aligned_rms(&a, &b));
    eprintln!(
        "MGO07 keyframes vs rootba bal f32: {} compared, RMS raw {:.3} mm, SE3-aligned {:.3} mm",
        a.len(),
        1e3 * raw,
        1e3 * aligned
    );
    assert!(aligned < 1e-3, "aligned keyframe RMS {aligned} m");
    assert!(
        raw - aligned < 1e-3,
        "the gauge floated: raw {raw} m vs aligned {aligned} m"
    );
}
