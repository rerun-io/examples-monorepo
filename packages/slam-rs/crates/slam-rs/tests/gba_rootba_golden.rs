//! The global BA driver on RootSLAM's exported MGO07 and MGO13 maps, against
//! rootba `bal` (DESIGN §7.4, PR d2). The map goes through the same path as
//! `Vio::finish_map`: a [`KeyframeMap`] with host-relative landmarks, then
//! `map_problem` and `solve` (f64 centring, f32), without `remove_gauge`:
//! rootba holds no pose and moves no gauge either, so the keyframes are
//! compared with the solver's output before the re-alignment.
//!
//! The export, its assets and rootba's logs are read by kornia-staging-slam's
//! `tests/support/rootba_export.rs`, shared with the solver golden; this file
//! adds `map/optimized_tracking_pg_float.csv` (rootba's optimized keyframes).
//! `#[ignore]`: run them with `pixi run -e slam-rs-dev slam-rs-gba-golden`; a
//! missing asset then fails and names the asset.

#![cfg(feature = "backend")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use nalgebra::{Matrix3, Vector2, Vector3};
use slam_rs::backend::gba::{GbaRig, map_problem, solve};
use slam_rs::backend::map::host_relative;
use slam_rs::backend::{KeyframeMap, PoseGraphEdge, PoseGraphEdgeKind};
use slam_rs::calib::Calibration;
use slam_rs::config::VioConfig;
use slam_rs::estimator::StampLandmark;
use slam_rs::types::{LandmarkId, TimeCamId};

#[path = "../../../../kornia-staging/crates/kornia-staging-slam/tests/support/rootba_export.rs"]
mod rootba_export;
use rootba_export::{
    MapJson, PoseGraphJson, calib_path, parse, pose7, read_asset, rootba_run, runs_root,
};

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

/// The rigid transform (`R`, `t`) that best maps `a` onto `b` (Kabsch:
/// upstream `umeyama_f64` without scale).
fn rigid_fit(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> (Matrix3<f64>, Vector3<f64>) {
    let points = |p: &[Vector3<f64>]| -> Vec<kornia_algebra::Vec3F64> {
        p.iter()
            .map(|v| kornia_algebra::Vec3F64::new(v.x, v.y, v.z))
            .collect()
    };
    let (r, t, _) =
        kornia_algebra::linalg::rigid::umeyama_f64(&points(a), &points(b), false).unwrap();
    (
        Matrix3::from_column_slice(&<[f64; 9]>::from(r)),
        Vector3::new(t.x, t.y, t.z),
    )
}

/// RMS of `a - b` after the rigid alignment (Kabsch) of `a` onto `b`.
fn aligned_rms(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> f64 {
    let (r, t) = rigid_fit(a, b);
    let sq: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (r * x + t - y).norm_squared())
        .sum();
    (sq / a.len() as f64).sqrt()
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
    let export: MapJson = parse(&run.join("map/map.json"));
    let graph: PoseGraphJson = parse(&run.join("map/pose_graph.json"));
    let calibration: Calibration<f64> =
        Calibration::from_json_str(&read_asset(&calib_path(seq))).unwrap();
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
        map.upsert_keyframe(kf.id as i64, pose7(&kf.world_from_imu), None);
    }
    let mut obs: HashMap<u64, BTreeMap<TimeCamId, Vector2<f32>>> = HashMap::new();
    for o in &export.observations {
        obs.entry(o.lm_id).or_default().insert(
            TimeCamId::new(o.kf_id as i64, o.cam_id),
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
            from: e.kf_id_a as i64,
            to: e.kf_id_b as i64,
            kind: PoseGraphEdgeKind::SpanningTree,
        })
        .collect();

    let config = VioConfig::default();
    let input = map_problem(&map, &edges, t_i_c, &config);
    let (solved, report) = solve(&input, &rig, &config, slam_rs::Backend::Cpu).unwrap();
    let ours = report.final_cost.error;
    let (bal_f32, bal_f64) = (
        rootba_run(seq, "balpgf").final_cost,
        rootba_run(seq, "balpgd").final_cost,
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
    for (id, imu_from_world) in input.keyframes.iter().zip(&solved.imu_from_world) {
        if let Some(p) = bal_poses.get(id) {
            a.push(imu_from_world.inverse().translation);
            b.push(*p);
        }
    }
    let mut per_keyframe: HashMap<u64, usize> = HashMap::new();
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
