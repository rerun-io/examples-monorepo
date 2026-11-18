//! The map -> problem -> map driver on synthetic maps (DESIGN §3.1, PR d2).

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

use kornia_staging_3d::camera::{CameraModelKind, Pinhole};
use nalgebra::{Matrix3, UnitQuaternion, Vector2, Vector3};

use super::*;
use crate::backend::{MapConfig, PoseGraphEdge, PoseGraphEdgeKind};
use crate::estimator::StampLandmark;
use crate::landmark::StereographicParam;
use crate::lie::So3;
use crate::types::{LandmarkId, TimeCamId};

const CAMERAS: usize = 2;

fn wobble(i: usize, k: usize) -> f64 {
    (((i * 7919 + k * 104729) % 9973) as f64 * 0.618_033_988_7).sin()
}

fn t_i_c() -> Vec<Se3<f64>> {
    (0..CAMERAS)
        .map(|c| {
            Se3::new(
                So3::exp(&Vector3::new(0.02 * c as f64, 0.15 * c as f64 - 0.1, 0.01)),
                Vector3::new(0.08 * c as f64 - 0.05, 0.01 * c as f64, 0.02),
            )
        })
        .collect()
}

fn rig() -> GbaRig {
    let model: CameraModelKind<f64> =
        CameraModelKind::Pinhole(Pinhole::new([300.0, 300.0, 320.0, 240.0]).unwrap());
    GbaRig::new(vec![model; CAMERAS], &t_i_c()).unwrap()
}

/// `T_w_i` of keyframe `k` on a circle of radius 5 m around `centre`, looking in.
fn truth_pose(k: usize, centre: Vector3<f64>) -> Se3<f64> {
    let th = 0.25 * k as f64;
    let (s, c) = th.sin_cos();
    let m = Matrix3::from_columns(&[
        Vector3::new(-s, c, 0.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(-c, -s, 0.0),
    ]);
    Se3::new(
        So3::from_unit_quaternion(UnitQuaternion::from_matrix(&m)),
        centre + Vector3::new(5.0 * c, 5.0 * s, 0.3 * (k as f64).sin()),
    )
}

fn truth_point(l: usize, centre: Vector3<f64>) -> Vector3<f64> {
    centre + Vector3::new(1.5 * wobble(l, 1), 1.5 * wobble(l, 2), wobble(l, 3))
}

fn config() -> MapConfig {
    MapConfig {
        enable_culling: false,
        max_redundancy_score: 0.9,
        redundancy_exclude_hosted: false,
        max_keyframes: 1500,
        batch_cull_size: 30,
        high_covisibility: 1_000_000,
    }
}

/// A map of `n_kf` keyframes (ids 1..) and `n_lm` landmarks around `centre`.
/// Poses drift by `pose_noise` metres (and `pose_noise / 10` radians of yaw)
/// per keyframe, landmarks carry `point_noise` metres, pixels `pixel_noise`. The keyframes form a
/// chain in the covisibility graph. Returns the backend and the true `T_w_i`.
fn synthetic_map(
    n_kf: usize,
    n_lm: usize,
    centre: Vector3<f64>,
    pose_noise: f64,
    point_noise: f64,
    pixel_noise: f64,
) -> (MapBackend, Vec<Se3<f64>>) {
    let rig = rig();
    let truth: Vec<Se3<f64>> = (0..n_kf).map(|k| truth_pose(k, centre)).collect();
    let noisy: Vec<Se3<f64>> = truth
        .iter()
        .enumerate()
        .map(|(k, pose)| {
            // VIO-like error: a drift that grows along the sequence (yaw and
            // translation), plus a little white noise.
            let drift = k as f64 * pose_noise;
            let d = Vector3::new(drift, 0.5 * drift, 0.0)
                + Vector3::new(wobble(k, 21), wobble(k, 22), wobble(k, 23)) * 0.05 * pose_noise;
            let r = Vector3::new(0.0, 0.0, 0.1 * drift)
                + Vector3::new(wobble(k, 24), wobble(k, 25), wobble(k, 26)) * 0.01 * pose_noise;
            Se3::new(So3::exp(&r) * pose.rotation, pose.translation + d)
        })
        .collect();
    let mut backend = MapBackend::new(config(), vec![[640, 480]; CAMERAS])
        .with_extrinsics(t_i_c())
        .with_gba_rig(Some(rig.clone()));
    for (k, pose) in noisy.iter().enumerate() {
        backend.map.upsert_keyframe(k as i64 + 1, *pose, None);
    }
    let mut landmarks: Vec<StampLandmark> = Vec::new();
    for l in 0..n_lm {
        let p = truth_point(l, centre);
        let mut obs = BTreeMap::new();
        for (k, pose) in truth.iter().enumerate() {
            for (c, cam) in rig.cameras.iter().enumerate() {
                if (k + c + l) % 3 == 0 {
                    continue;
                }
                let p_c = (*pose * cam.imu_from_camera).inverse() * p;
                if let Ok(px) = cam.model.project([p_c.x, p_c.y, p_c.z])
                    && (0.0..640.0).contains(&px[0])
                    && (0.0..480.0).contains(&px[1])
                {
                    let pixel = Vector2::new(
                        (px[0] + pixel_noise * wobble(l, 10 * k + c)) as f32,
                        (px[1] + pixel_noise * wobble(l, 10 * k + c + 5)) as f32,
                    );
                    obs.insert(TimeCamId::new(k as i64 + 1, c), pixel);
                }
            }
        }
        let Some(&host) = obs.keys().next() else {
            continue;
        };
        let noisy_p = p + Vector3::new(wobble(l, 31), wobble(l, 32), wobble(l, 33)) * point_noise;
        let host_pose = noisy[(host.frame_id - 1) as usize] * t_i_c()[host.cam_id];
        let (direction, inv_dist) = host_relative(&host_pose, &noisy_p).unwrap();
        landmarks.push(StampLandmark {
            id: LandmarkId(l as u64),
            host,
            direction,
            inv_dist,
            obs,
        });
    }
    backend.map.merge_landmarks(&landmarks);
    for k in 0..n_kf as i64 {
        let shared: BTreeMap<i64, u32> = (k > 0).then_some((k, 100)).into_iter().collect();
        backend.graph.insert_keyframe(k + 1, &shared);
    }
    (backend, truth)
}

/// RMS of the keyframe positions after the rigid alignment (Kabsch) of `est`
/// onto `truth`: the GBA fixes no pose, so its gauge floats.
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

fn aligned_rms(est: &[Se3<f64>], truth: &[Se3<f64>]) -> f64 {
    let (e, t): (Vec<Vector3<f64>>, Vec<Vector3<f64>>) = (
        est.iter().map(|p| p.translation).collect(),
        truth.iter().map(|p| p.translation).collect(),
    );
    let (r, shift) = rigid_fit(&e, &t);
    let sq: f64 = e
        .iter()
        .zip(&t)
        .map(|(e, t)| (r * e + shift - t).norm_squared())
        .sum();
    (sq / e.len() as f64).sqrt()
}

fn chain_edges(n_kf: usize) -> Vec<PoseGraphEdge> {
    (1..n_kf as i64)
        .map(|k| PoseGraphEdge {
            from: k,
            to: k + 1,
            kind: PoseGraphEdgeKind::SpanningTree,
        })
        .collect()
}

/// World-z tilt, degrees, of the rigid fit (Kabsch on positions) of `est`
/// onto `reference`: how far a solve turned the map out of the input's gravity
/// frame.
fn tilt_deg(est: &[Se3<f64>], reference: &[Se3<f64>]) -> f64 {
    let positions =
        |poses: &[Se3<f64>]| -> Vec<Vector3<f64>> { poses.iter().map(|p| p.translation).collect() };
    let (r, _) = rigid_fit(&positions(est), &positions(reference));
    (r * Vector3::z()).angle(&Vector3::z()).to_degrees()
}

/// The gauge (DESIGN §3.1, d2 decision): no pose is held constant in the
/// solve, so the result is brought back onto the input poses by one rigid
/// transform. A drifted map keeps the input's world z.
#[test]
fn a_drifted_map_keeps_the_input_world_z() {
    let (mut backend, _) = synthetic_map(16, 300, Vector3::new(30.0, -20.0, 1.0), 0.01, 0.03, 0.3);
    let before: Vec<Se3<f64>> = backend.map().keyframes().values().map(|k| k.pose).collect();
    let result = backend.finish(true, &VioConfig::default()).unwrap();
    let tilt = tilt_deg(&result.keyframe_poses, &before);
    assert!(tilt < 0.01, "the GBA tilted the map by {tilt} deg");
}

/// A map with no error anywhere: the GBA moves no pose (beyond f32 rounding)
/// and keeps world z.
#[test]
fn a_zero_error_map_keeps_every_pose_and_world_z() {
    let (mut backend, _) = synthetic_map(12, 200, Vector3::new(30.0, -20.0, 1.0), 0.0, 0.0, 0.0);
    let before: Vec<Se3<f64>> = backend.map().keyframes().values().map(|k| k.pose).collect();
    let result = backend.finish(true, &VioConfig::default()).unwrap();
    assert!(tilt_deg(&result.keyframe_poses, &before) < 0.01);
    for (a, b) in result.keyframe_poses.iter().zip(&before) {
        assert!(
            (a.translation - b.translation).norm() < 1e-5,
            "{a:?} vs {b:?}"
        );
        assert!((a.rotation.inverse() * b.rotation).log().norm() < 1e-5);
    }
}

#[test]
fn host_relative_landmarks_become_world_points_once_and_round_trip() {
    let (backend, _) = synthetic_map(6, 40, Vector3::zeros(), 0.0, 0.0, 0.3);
    let input = map_problem(backend.map(), &[], &backend.t_i_c, &VioConfig::default());
    assert!(!input.landmarks.is_empty());
    for (id, landmark) in input.landmarks.iter().zip(&input.problem.landmarks) {
        let stored = backend.map().landmark(*id).unwrap();
        let host = backend.map().keyframe(stored.host.frame_id).unwrap().pose;
        let bearing = StereographicParam::<f64>::unproject(&stored.direction);
        let p_c = Vector3::new(bearing[0], bearing[1], bearing[2]) / stored.inv_dist;
        let expected = host * backend.t_i_c[stored.host.cam_id] * p_c;
        assert!((landmark.position - expected).norm() < 1e-12);
        // And back, against the same host pose.
        let (direction, inv_dist) = host_relative(
            &(host * backend.t_i_c[stored.host.cam_id]),
            &landmark.position,
        )
        .unwrap();
        assert!((direction - stored.direction).norm() < 1e-12);
        assert!((inv_dist - stored.inv_dist).abs() < 1e-12);
    }
}

/// B22 and the RootSLAM export rule: every keyframe that has a row stays a
/// variable; one under `min_obs_per_kf` keeps only its pose-graph rows; a
/// landmark needs two observations and a finite position.
#[test]
fn sparse_keyframes_keep_their_pose_graph_rows_and_landmarks_need_two_finite_observations() {
    let (mut backend, _) = synthetic_map(6, 60, Vector3::zeros(), 0.0, 0.0, 0.3);
    // Keyframe 7 sees three landmarks only; landmark 1000 is at infinity.
    backend
        .map
        .upsert_keyframe(7, truth_pose(6, Vector3::zeros()), None);
    let extra: Vec<StampLandmark> = (0..3)
        .map(|l| StampLandmark {
            id: LandmarkId(500 + l),
            host: TimeCamId::new(6, 0),
            direction: Vector2::new(0.01 * l as f64, 0.0),
            inv_dist: 0.3,
            obs: [
                (TimeCamId::new(6, 0), Vector2::new(300.0, 200.0)),
                (TimeCamId::new(7, 1), Vector2::new(310.0, 220.0)),
            ]
            .into_iter()
            .collect(),
        })
        .chain([StampLandmark {
            id: LandmarkId(1000),
            host: TimeCamId::new(1, 0),
            direction: Vector2::new(0.0, 0.0),
            inv_dist: 0.0,
            obs: [
                (TimeCamId::new(1, 0), Vector2::new(320.0, 240.0)),
                (TimeCamId::new(2, 0), Vector2::new(321.0, 240.0)),
            ]
            .into_iter()
            .collect(),
        }])
        .collect();
    backend.map.merge_landmarks(&extra);
    let mut edges = chain_edges(6);
    edges.push(PoseGraphEdge {
        from: 6,
        to: 7,
        kind: PoseGraphEdgeKind::SpanningTree,
    });
    let input = map_problem(backend.map(), &edges, &backend.t_i_c, &VioConfig::default());
    assert_eq!(input.keyframes, (1..=7).collect::<Vec<i64>>());
    let k7 = input.keyframes.iter().position(|&id| id == 7).unwrap();
    assert!(
        input
            .problem
            .landmarks
            .iter()
            .flat_map(|l| &l.observations)
            .all(|o| o.keyframe != k7),
        "keyframe 7 has 3 observations: it keeps no reprojection row"
    );
    assert!(
        input
            .problem
            .rel_poses
            .iter()
            .any(|c| c.frame_b == k7 || c.frame_a == k7),
        "keyframe 7 keeps its pose-graph row"
    );
    assert_eq!(input.counts.keyframes_without_reprojection, 1);
    assert_eq!(input.counts.dropped_observations, 3);
    assert!(
        !input.landmarks.contains(&LandmarkId(1000)),
        "a landmark at infinity is left out"
    );
    assert_eq!(input.counts.nonfinite_landmarks, 1);
    for id in 500..503 {
        // One observation left after keyframe 7's went: dropped.
        assert!(!input.landmarks.contains(&LandmarkId(id)));
    }
    for landmark in &input.problem.landmarks {
        assert!(landmark.observations.len() >= 2);
        assert!(landmark.position.iter().all(|v| v.is_finite()));
    }
    assert_eq!(input.counts.landmarks, input.problem.landmarks.len());
    assert_eq!(input.counts.pose_graph_rows, 6);
}

/// B21 in rootba's residual form: one row per graph edge, measured from the
/// map's own poses (zero residual at the start), sigmas 0.05 m / 0.01 rad.
#[test]
fn pose_graph_rows_are_the_graph_edges_measured_from_the_input_poses() {
    let (backend, _) = synthetic_map(5, 30, Vector3::zeros(), 0.02, 0.0, 0.3);
    let mut edges = chain_edges(5);
    edges.push(PoseGraphEdge {
        from: 1,
        to: 4,
        kind: PoseGraphEdgeKind::LoopClosure,
    });
    // An edge to a keyframe the map does not hold is skipped.
    edges.push(PoseGraphEdge {
        from: 2,
        to: 99,
        kind: PoseGraphEdgeKind::HighCovisibility,
    });
    let input = map_problem(backend.map(), &edges, &backend.t_i_c, &VioConfig::default());
    assert_eq!(input.problem.rel_poses.len(), 5);
    for c in &input.problem.rel_poses {
        let a = backend
            .map()
            .keyframe(input.keyframes[c.frame_a])
            .unwrap()
            .pose;
        let b = backend
            .map()
            .keyframe(input.keyframes[c.frame_b])
            .unwrap()
            .pose;
        let expected = a.inverse() * b;
        assert!((c.a_from_b.translation - expected.translation).norm() < 1e-12);
        assert!(
            (c.a_from_b.rotation.inverse() * expected.rotation)
                .log()
                .norm()
                < 1e-12
        );
        let res = kornia_staging_slam::sqrt_ba::relative_pose_error(
            &c.a_from_b,
            &input.problem.imu_from_world[c.frame_a],
            &input.problem.imu_from_world[c.frame_b],
            None,
            None,
        );
        assert!(res.norm() < 1e-9);
        for i in 0..3 {
            assert!((c.sqrt_info[(i, i)] - 20.0).abs() < 1e-12);
            assert!((c.sqrt_info[(i + 3, i + 3)] - 100.0).abs() < 1e-12);
        }
    }
}

/// The LM settings are rootba `bal`'s: Huber 1 px, 20 attempts, no pose held
/// constant (DESIGN §3.1), and the PCG limits from the config.
#[test]
fn the_options_are_rootba_bal_with_no_fixed_pose() {
    let options = gba_options(&VioConfig::default());
    assert_eq!(options.huber_px, 1.0);
    assert_eq!(options.max_iterations, 20);
    assert!(options.fixed_keyframes.is_empty());
    assert_eq!(options.pcg.max_iterations, 500);
    assert_eq!(options.pcg.eta, 0.1);
}

/// End to end, f32 and f64, far from the origin: a drifted map comes back
/// toward the truth (the pose-graph rows, measured from the drifted poses,
/// hold part of the drift: the band is a ratio), and f32 lands on f64 within
/// a millimetre, which only f64 centring before the cast allows at 10 km.
/// ERROR_VALID counts no row behind a camera here.
#[test]
fn a_drifted_synthetic_map_recovers_in_f32_and_f64_far_from_the_origin() {
    let centre = Vector3::new(1e4, -2e3, 5.0);
    let mut solved: Vec<Vec<Se3<f64>>> = Vec::new();
    for use_double in [false, true] {
        let (mut backend, truth) = synthetic_map(16, 300, centre, 0.01, 0.03, 0.3);
        let before: Vec<Se3<f64>> = backend.map().keyframes().values().map(|k| k.pose).collect();
        let config = VioConfig {
            gba_use_double: use_double,
            ..VioConfig::default()
        };
        let result = backend.finish(true, &config).unwrap();
        let gba = result.gba.as_ref().unwrap();
        assert!(gba.report.iterations.len() <= 20);
        assert!(gba.report.final_cost.error < 0.01 * gba.report.initial_cost.error);
        assert_eq!(gba.report.final_cost.invalid, 0);
        assert_eq!(result.keyframe_poses_vio, before);
        let after: Vec<Se3<f64>> = backend.map().keyframes().values().map(|k| k.pose).collect();
        assert_eq!(result.keyframe_poses, after);
        let (rms_before, rms_after) = (aligned_rms(&before, &truth), aligned_rms(&after, &truth));
        assert!(rms_before > 0.04, "{rms_before}");
        assert!(
            rms_after < 0.35 * rms_before,
            "use_double {use_double}: {rms_before} -> {rms_after}"
        );
        solved.push(after);
    }
    let gap: f64 = (solved[0]
        .iter()
        .zip(&solved[1])
        .map(|(a, b)| (a.translation - b.translation).norm_squared())
        .sum::<f64>()
        / solved[0].len() as f64)
        .sqrt();
    assert!(gap < 1e-3, "f32 vs f64 keyframe RMS {gap} m");
}

/// B11: the solution is applied as a delta on the pose the map holds now, and
/// a solved landmark moves with its host the same way.
#[test]
fn the_result_is_applied_as_a_delta_on_the_current_pose() {
    let (mut backend, _) = synthetic_map(6, 60, Vector3::zeros(), 0.02, 0.01, 0.3);
    let config = VioConfig::default();
    let edges = chain_edges(6);
    let input = map_problem(backend.map(), &edges, &backend.t_i_c, &config);
    let (solved, _) = solve(&input, backend.gba_rig.as_ref().unwrap(), &config).unwrap();
    // The map moved keyframe 1 (a host) after the problem was built.
    let moved = Se3::new(
        So3::exp(&Vector3::new(0.0, 0.01, 0.0)),
        Vector3::new(0.1, 0.0, 0.0),
    );
    let now = moved * backend.map().keyframe(1).unwrap().pose;
    backend.map.upsert_keyframe(1, now, None);
    apply(&mut backend.map, &input, &solved, &backend.t_i_c);
    let k = input.keyframes.iter().position(|&id| id == 1).unwrap();
    let t_gba = solved.imu_from_world[k].inverse();
    let t_input = input.problem.imu_from_world[k].inverse();
    let expected = t_gba * t_input.inverse() * now;
    let got = backend.map().keyframe(1).unwrap().pose;
    assert!((got.translation - expected.translation).norm() < 1e-9);
    assert!((got.rotation.inverse() * expected.rotation).log().norm() < 1e-9);
    let stored_world = |id: LandmarkId| {
        let stored = backend.map().landmark(id).unwrap();
        let host = backend.map().keyframe(stored.host.frame_id).unwrap().pose
            * backend.t_i_c[stored.host.cam_id];
        let bearing = StereographicParam::<f64>::unproject(&stored.direction);
        (
            stored.host.frame_id,
            host * (Vector3::new(bearing[0], bearing[1], bearing[2]) / stored.inv_dist),
        )
    };
    let mut on_moved_host = 0;
    for (&id, solved_lm) in input.landmarks.iter().zip(&solved.landmarks) {
        let (host, p_w) = stored_world(id);
        // A landmark of an unmoved host sits at its solved world position; one
        // of keyframe 1 moved with it, by the same delta as the pose.
        let expected = if host == 1 {
            on_moved_host += 1;
            got * t_gba.inverse() * solved_lm.position
        } else {
            solved_lm.position
        };
        assert!(
            (p_w - expected).norm() < 1e-9,
            "landmark {id:?} of host {host}"
        );
    }
    assert!(on_moved_host > 0);
}

/// The gauge: a solution the free gauge turned and shifted (a 2 deg tilt) is
/// brought back onto the input poses, landmarks with it, and the cost does
/// not change.
#[test]
fn a_tilted_solution_is_rotated_back_and_keeps_its_cost() {
    let (backend, _) = synthetic_map(8, 120, Vector3::new(30.0, -20.0, 1.0), 0.0, 0.0, 0.3);
    let config = VioConfig::default();
    let input = map_problem(backend.map(), &chain_edges(8), &backend.t_i_c, &config);
    let tilt = Se3::new(
        So3::exp(&Vector3::new(2f64.to_radians(), -1f64.to_radians(), 0.3)),
        Vector3::new(0.5, -0.2, 0.1),
    );
    let mut tilted = input.problem.clone();
    for pose in &mut tilted.imu_from_world {
        *pose = *pose * tilt.inverse();
    }
    for landmark in &mut tilted.landmarks {
        landmark.position = tilt * landmark.position;
    }
    let cost = |problem: &GbaProblem<f64>| {
        let options = GbaOptions {
            max_iterations: 0,
            ..gba_options(&config)
        };
        solve_global(&mut problem.clone(), &rig().cameras, &options, None)
            .unwrap()
            .initial_cost
            .error
    };
    let before = cost(&tilted);
    let gauge = remove_gauge(&input.problem, &mut tilted);
    assert!((gauge * tilt).log().norm() < 1e-9, "{gauge:?}");
    for (a, b) in tilted
        .imu_from_world
        .iter()
        .zip(&input.problem.imu_from_world)
    {
        assert!((a.inverse() * *b).log().norm() < 1e-9);
    }
    for (a, b) in tilted.landmarks.iter().zip(&input.problem.landmarks) {
        assert!((a.position - b.position).norm() < 1e-9);
    }
    let after = cost(&tilted);
    assert!(
        (after - before).abs() <= 1e-9 * before,
        "{before} -> {after}"
    );
    assert!((after - cost(&input.problem)).abs() <= 1e-9 * before);
}

/// `finish(false)` copies the map and changes nothing.
#[test]
fn finish_without_the_gba_returns_the_map_unchanged() {
    let (mut backend, _) = synthetic_map(6, 60, Vector3::zeros(), 0.02, 0.01, 0.3);
    let map_before = backend.map().clone();
    let result = backend.finish(false, &VioConfig::default()).unwrap();
    assert!(result.gba.is_none());
    assert_eq!(backend.map(), &map_before);
    assert_eq!(result.keyframe_poses, result.keyframe_poses_vio);
    assert_eq!(result.keyframe_t_ns, (1..=6).collect::<Vec<i64>>());
}

/// The scalars the deep page plots: per LM attempt the cost, lambda, CG
/// iterations and stage times, and the wall of each driver stage.
#[test]
fn the_report_carries_the_iteration_scalars_and_stage_walls() {
    let (mut backend, _) = synthetic_map(8, 120, Vector3::zeros(), 0.02, 0.01, 0.3);
    let result = backend.finish(true, &VioConfig::default()).unwrap();
    let gba = result.gba.unwrap();
    assert!(!gba.report.iterations.is_empty() && gba.report.iterations.len() <= 20);
    assert!(
        gba.report
            .iterations
            .iter()
            .all(|i| i.pcg.iterations > 0 && i.lambda > 0.0)
    );
    assert!(gba.report.iterations.iter().any(|i| i.accepted));
    assert!(gba.build_s > 0.0 && gba.solve_s > 0.0 && gba.apply_s > 0.0);
    assert_eq!(gba.threads, 2);
    assert_eq!(gba.counts.keyframes, 8);
    assert_eq!(gba.counts.pose_graph_rows, 7);
}

/// A map without the rig's camera models cannot run the GBA, and says so.
#[test]
fn the_gba_needs_the_rig() {
    let (backend, _) = synthetic_map(4, 30, Vector3::zeros(), 0.0, 0.0, 0.3);
    let mut backend = backend.with_gba_rig(None);
    assert!(backend.finish(false, &VioConfig::default()).is_ok());
    assert!(matches!(
        backend.finish(true, &VioConfig::default()),
        Err(GbaDriverError::NoRig)
    ));
}
