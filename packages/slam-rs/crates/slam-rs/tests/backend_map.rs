//! The persistent keyframe map on the moving-ceiling rig (DESIGN a1).
//!
//! The invariant: the raw VIO is bit-identical with the back end on or off,
//! synchronous and deferred (D84) keyframes alike, because the window never
//! reads the map. And the map holds what the window handed it: more keyframes
//! than the window, the window keyframes still active, solved poses, and landmarks with two or more observations.

#![cfg(feature = "backend")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use slam_rs as api;
#[path = "common/deferred.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::*;

fn pipeline_with_map(deferred: bool, enabled: bool) -> api::Vio<f32> {
    let mut config = config(deferred);
    config.backend_enabled = enabled;
    // More keyframes and a short keyframe budget, so
    // keyframes leave the window and the map outgrows it.
    config.vio_new_kf_keypoints_thresh = 0.9;
    config.vio_max_kfs = 3;
    api::Vio::new(
        config,
        calibration(),
        api::frontend::flow::FrontendOptions {
            threads: 2,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn the_map_never_moves_a_vio_pose_and_keeps_every_keyframe() {
    let frames: Vec<[Vec<u8>; 4]> = (0..FRAMES)
        .map(|index| {
            let t_s = index as f64 * FRAME_NS as f64 / 1e9;
            std::array::from_fn(|camera| render(t_s, camera))
        })
        .collect();

    for deferred in [false, true] {
        let mut off = pipeline_with_map(deferred, false);
        let mut on = pipeline_with_map(deferred, true);
        assert!(off.map_backend().is_none());
        let without = run(&frames, &mut off);
        let with = run(&frames, &mut on);
        assert_eq!(
            with.results, without.results,
            "the back end moved a pose (deferred = {deferred})"
        );
        assert_eq!(with.solves, without.solves);

        on.finish_deferred_keyframe().unwrap();
        let map_backend = on.map_backend().unwrap();
        let map = map_backend.map();
        let stats = map_backend.stats();
        let window: Vec<i64> = on.estimator().kf_ids().collect();
        eprintln!("deferred = {deferred}: {stats:?}, window {window:?}");
        // The map is a superset of the window, and active = the window.
        assert!(stats.keyframes > window.len(), "{stats:?}");
        assert_eq!(
            map_backend
                .active_keyframes()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            window
        );
        // Window keyframes carry a solved pose: the one of the last keyframe
        // solve, which later framesets may refine a little.
        let snapshot = on.estimator().snapshot();
        for id in &window {
            let solved = snapshot
                .states
                .iter()
                .chain(snapshot.poses.iter())
                .find(|state| state.t_ns == *id)
                .map(|state| state.t_w_i.cast::<f64>())
                .unwrap();
            let stored = map.keyframe(*id).unwrap().pose;
            assert!((stored.translation - solved.translation).norm() < 1e-3);
        }
        assert!(stats.landmarks > 50, "{stats:?}");
        assert!(stats.observations >= 2 * stats.landmarks);
        assert!(
            map.landmarks()
                .values()
                .all(|landmark| landmark.obs.len() >= 2)
        );
        // The graph spans the map with one tree.
        assert_eq!(map_backend.graph().len(), stats.keyframes);
        assert_eq!(
            map_backend.graph().root(),
            map.keyframes().keys().next().copied()
        );
        // Every keyframe kept its keypoints, and nothing else image-sized.
        assert!(
            map.keyframes()
                .values()
                .all(|keyframe| keyframe.keypoints.len() == 4)
        );
        assert!(map.retained_bytes() < stats.keyframes * 64 * 1024);
    }
}
