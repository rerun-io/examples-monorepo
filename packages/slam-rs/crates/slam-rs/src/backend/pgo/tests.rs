//! The PGO's graph, gauge and delta apply on hand-built maps (PR c2).

#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{Matrix6, Vector2, Vector3, Vector6};

use super::*;
use crate::estimator::StampLandmark;
use crate::lie::So3;
use crate::types::{LandmarkId, TimeCamId};

fn at(x: f64, yaw: f64) -> Se3<f64> {
    Se3::new(
        So3::exp(&Vector3::new(0.0, 0.0, yaw)),
        Vector3::new(x, 0.1 * x, 0.0),
    )
}

/// Keyframes `1..=n` at `poses[i - 1]`; each shares `chain` landmarks with the
/// one before (the tree) and `skip` with the one two before.
fn chain_map(poses: &[Se3<f64>], chain: u32, skip: u32) -> (KeyframeMap, CovisibilityGraph) {
    let mut map = KeyframeMap::new();
    let mut graph = CovisibilityGraph::new();
    for (i, pose) in poses.iter().enumerate() {
        let id = i as FrameId + 1;
        map.upsert_keyframe(id, *pose, None);
        let shared: BTreeMap<FrameId, u32> = [(id - 1, chain), (id - 2, skip)]
            .into_iter()
            .filter(|&(other, weight)| other >= 1 && weight > 0)
            .collect();
        graph.insert_keyframe(id, &shared);
    }
    (map, graph)
}

fn loop_config() -> LoopClosingConfig {
    LoopClosingConfig::default()
}

fn parity_config() -> LoopClosingConfig {
    LoopClosingConfig {
        pgo_edge_sigmas: [1.0, 1.0],
        pgo_loop_edges_first: false,
        ..LoopClosingConfig::default()
    }
}

#[test]
fn the_graph_takes_tree_past_loop_and_strong_edges_then_the_loop_measured_from_the_map() {
    let poses: Vec<Se3<f64>> = (0..6).map(|i| at(i as f64, 0.1 * i as f64)).collect();
    // 1-2-3-4-5-6 tree at weight 100; skips at 40 (below 50) except 4-6 at 80.
    let (map, mut graph) = chain_map(&poses, 100, 40);
    graph.increment_edge(4, 6, 40);
    graph.add_loop_closure(1, 4);
    let corrected = at(0.2, 0.05);
    let active = BTreeSet::from([5, 6]);
    for (config, sqrt_info, loop_first) in [
        (
            loop_config(),
            Matrix6::from_diagonal(&Vector6::new(20.0, 20.0, 20.0, 100.0, 100.0, 100.0)),
            true,
        ),
        (parity_config(), Matrix6::identity(), false),
    ] {
        let graph = PoseGraph::build(&map, &graph, 50, &active, (6, &[1, 2], &corrected), &config);
        assert_eq!(graph.ids, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(graph.copy, poses);
        assert_eq!(graph.loop_edges, 2);
        let pairs: Vec<(usize, usize)> = graph.edges.iter().map(|e| (e.from, e.to)).collect();
        // Tree (parent -> child), the past loop 1-4, the strong 4-6; then the loop.
        let map_pairs = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3), (3, 5)];
        let loop_pairs = vec![(0, 5), (1, 5)];
        let expected: Vec<(usize, usize)> = if loop_first {
            loop_pairs.iter().chain(&map_pairs).copied().collect()
        } else {
            map_pairs.iter().chain(&loop_pairs).copied().collect()
        };
        assert_eq!(pairs, expected);
        for edge in &graph.edges {
            let to_pose = if (edge.from, edge.to) == (0, 5) || (edge.from, edge.to) == (1, 5) {
                corrected
            } else {
                poses[edge.to]
            };
            assert_eq!(edge.from_t_to, poses[edge.from].inverse() * to_pose);
            assert_eq!(edge.sqrt_information, sqrt_info);
        }
    }
}

#[test]
fn the_gauge_is_the_window_and_the_newest_keyframe_and_they_stay_bit_identical() {
    let poses: Vec<Se3<f64>> = (0..8).map(|i| at(i as f64, 0.02 * i as f64)).collect();
    let (map, graph) = chain_map(&poses, 100, 0);
    // The window lags: 5 and 6 are in it, 8 is the newest map keyframe.
    let active = BTreeSet::from([5, 6]);
    let graph = PoseGraph::build(
        &map,
        &graph,
        50,
        &active,
        (7, &[1], &at(0.5, 0.3)),
        &loop_config(),
    );
    assert_eq!(
        graph.fixed,
        vec![false, false, false, false, true, true, false, true]
    );
    let (optimized, report) = graph.solve().unwrap();
    assert!(report.final_cost < report.initial_cost);
    for (i, (pose, copy)) in optimized.iter().zip(&graph.copy).enumerate() {
        if graph.fixed[i] {
            assert_eq!(pose, copy, "keyframe {} is in the gauge", graph.ids[i]);
        }
    }
    assert_ne!(optimized[0], graph.copy[0], "the loop moves the old map");
}

#[test]
fn the_delta_keeps_a_window_write_made_after_the_copy_and_skips_new_and_culled_keyframes() {
    // B11: RootSLAM writes the stale optimized copy back.
    let poses: Vec<Se3<f64>> = (0..5).map(|i| at(i as f64, 0.05 * i as f64)).collect();
    let (mut map, graph) = chain_map(&poses, 100, 0);
    let active = BTreeSet::from([5]);
    let pose_graph = PoseGraph::build(
        &map,
        &graph,
        50,
        &active,
        (5, &[1], &at(0.3, 0.2)),
        &loop_config(),
    );
    let (optimized, _) = pose_graph.solve().unwrap();
    // While the PGO ran: keyframe 3 was rewritten, 2 culled, 6 added.
    let now_3 = at(2.5, 0.4);
    map.upsert_keyframe(3, now_3, None);
    map.remove_keyframe(2);
    map.upsert_keyframe(6, at(9.0, 0.0), None);
    let moved = apply_delta(&mut map, &pose_graph, &optimized);
    assert_eq!(moved, vec![1, 3, 4]);
    assert_eq!(
        map.keyframe(1).unwrap().pose,
        optimized[0],
        "unchanged since the copy"
    );
    let expected_3 = optimized[2] * pose_graph.copy[2].inverse() * now_3;
    assert_eq!(map.keyframe(3).unwrap().pose, expected_3);
    assert_ne!(map.keyframe(3).unwrap().pose, optimized[2]);
    assert!(map.keyframe(2).is_none(), "a culled keyframe stays culled");
    assert_eq!(map.keyframe(5).unwrap().pose, poses[4], "the gauge");
    assert_eq!(
        map.keyframe(6).unwrap().pose,
        at(9.0, 0.0),
        "not in the graph"
    );
}

#[test]
fn landmarks_move_with_their_host_keyframe() {
    let poses: Vec<Se3<f64>> = (0..4).map(|i| at(i as f64, 0.1 * i as f64)).collect();
    let (mut map, graph) = chain_map(&poses, 100, 0);
    map.merge_landmarks(&[StampLandmark {
        id: LandmarkId(7),
        host: TimeCamId::new(1, 0),
        direction: Vector2::new(0.1, -0.2),
        inv_dist: 0.5,
        obs: [
            (TimeCamId::new(1, 0), Vector2::new(1.0, 1.0)),
            (TimeCamId::new(2, 0), Vector2::new(2.0, 2.0)),
        ]
        .into_iter()
        .collect(),
    }]);
    let t_i_c = [at(0.05, 0.3)];
    let before = map.landmark_world(LandmarkId(7), &t_i_c).unwrap();
    let pose_graph = PoseGraph::build(
        &map,
        &graph,
        50,
        &BTreeSet::from([4]),
        (4, &[1], &at(0.4, 0.1)),
        &loop_config(),
    );
    let (optimized, _) = pose_graph.solve().unwrap();
    apply_delta(&mut map, &pose_graph, &optimized);
    let after = map.landmark_world(LandmarkId(7), &t_i_c).unwrap();
    let expected = optimized[0] * pose_graph.copy[0].inverse() * before;
    assert!((after - expected).norm() < 1e-12, "{after} vs {expected}");
    assert!((after - before).norm() > 1e-3, "the host moved");
}

/// Keyframes on a circle, tracked with a constant yaw and scale drift per
/// step; the last one sees the first again.
fn drifting_ring(n: usize) -> (Vec<Se3<f64>>, Vec<Se3<f64>>) {
    let truth: Vec<Se3<f64>> = (0..n)
        .map(|i| {
            let angle = std::f64::consts::TAU * i as f64 / n as f64;
            Se3::new(
                So3::exp(&Vector3::new(0.0, 0.0, angle + std::f64::consts::FRAC_PI_2)),
                Vector3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 0.0),
            )
        })
        .collect();
    let step_error = Se3::new(
        So3::exp(&Vector3::new(0.0, 0.0, 0.01)),
        Vector3::new(0.01, 0.0, 0.0),
    );
    let mut tracked: Vec<Se3<f64>> = vec![truth[0]];
    for i in 1..n {
        let odometry = truth[i - 1].inverse() * truth[i];
        tracked.push(tracked[i - 1] * odometry * step_error);
    }
    (truth, tracked)
}

#[test]
fn a_loop_recovers_a_synthetic_drift() {
    let n = 40;
    let (truth, tracked) = drifting_ring(n);
    let (mut map, graph) = chain_map(&tracked, 100, 60);
    let current = n as FrameId;
    let island: Vec<FrameId> = vec![1, 2, 3];
    // PnP against keyframe 1, which has not drifted: the true pose.
    let corrected = tracked[0] * truth[0].inverse() * truth[n - 1];
    let active = BTreeSet::from([current - 2, current - 1, current]);
    // Errors in the gauge's frame: the current keyframe is where the map says.
    let align = truth[n - 1] * tracked[n - 1].inverse();
    let worst = |map: &KeyframeMap| -> f64 {
        (0..n)
            .map(|i| {
                let pose = map.keyframe(i as FrameId + 1).unwrap().pose;
                ((align * pose).translation - truth[i].translation).norm()
            })
            .fold(0.0, f64::max)
    };
    let before = worst(&map);
    let pose_graph = PoseGraph::build(
        &map,
        &graph,
        50,
        &active,
        (current, &island, &corrected),
        &loop_config(),
    );
    let (optimized, report) = pose_graph.solve().unwrap();
    apply_delta(&mut map, &pose_graph, &optimized);
    let after = worst(&map);
    assert!(before > 1.0, "{before}");
    assert!(after < 0.15 * before, "{before} -> {after}, {report:?}");
}
