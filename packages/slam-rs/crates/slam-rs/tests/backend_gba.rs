//! `Vio::finish_map` on the moving-ceiling rig (DESIGN §8, PR d2).
//!
//! The global BA runs on the map at the end of a sequence. It never touches the
//! window: the raw VIO poses are the ones a run without the back end gives, and
//! the window is the same before and after. Without the GBA, `finish_map`
//! returns the map as it is. With it, the map's keyframes move, at most 20 LM
//! attempts run, the iteration scalars come back, and Offline and Realtime give
//! the same result.

#![cfg(feature = "backend")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use slam_rs as api;
#[path = "common/deferred.rs"]
#[allow(dead_code)]
mod fixture;
use api::config::BackendMode;
use fixture::*;

fn pipeline_with_map(mode: Option<BackendMode>) -> api::Vio<f32> {
    let mut config = config(false);
    config.backend_enabled = mode.is_some();
    config.backend_mode = mode.unwrap_or_default();
    // Keyframes leave the window, so the map holds more than it.
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
fn finish_map_runs_the_gba_on_the_map_and_never_on_the_window() {
    let frames: Vec<[Vec<u8>; 4]> = (0..FRAMES)
        .map(|index| {
            let t_s = index as f64 * FRAME_NS as f64 / 1e9;
            std::array::from_fn(|camera| render(t_s, camera))
        })
        .collect();
    let mut off = pipeline_with_map(None);
    let mut on = pipeline_with_map(Some(BackendMode::Offline));
    let mut realtime = pipeline_with_map(Some(BackendMode::Realtime));
    let without = run(&frames, &mut off);
    let with = run(&frames, &mut on);
    let threaded = run(&frames, &mut realtime);
    assert_eq!(with.results, without.results);
    assert_eq!(threaded.results, without.results);
    assert!(off.finish_map(true).unwrap().is_none(), "no map, no result");

    let window = format!("{:?}", on.estimator());
    let snapshot = on.map_snapshot().unwrap();
    let plain = on.finish_map(false).unwrap().unwrap();
    assert!(plain.gba.is_none());
    assert_eq!(plain.keyframe_t_ns, snapshot.keyframe_t_ns);
    assert_eq!(plain.keyframe_poses, snapshot.keyframe_poses);
    assert_eq!(plain.keyframe_poses_vio, snapshot.keyframe_poses);
    assert_eq!(
        on.map_snapshot().unwrap(),
        snapshot,
        "finish_map(false) changed the map"
    );

    let result = on.finish_map(true).unwrap().unwrap();
    let gba = result.gba.as_ref().unwrap();
    assert!(
        (1..=20).contains(&gba.report.iterations.len()),
        "{:?}",
        gba.report.termination
    );
    assert!(gba.report.final_cost.error <= gba.report.initial_cost.error);
    assert!(gba.report.iterations.iter().any(|i| i.accepted));
    assert!(gba.counts.keyframes > 3 && gba.counts.landmarks > 0 && gba.counts.pose_graph_rows > 0);
    assert_eq!(gba.threads, 2);
    assert_eq!(result.keyframe_poses_vio, snapshot.keyframe_poses);
    assert_ne!(
        result.keyframe_poses, result.keyframe_poses_vio,
        "the GBA moved no keyframe"
    );
    // The map holds the GBA's poses; the window is untouched.
    assert_eq!(
        on.map_snapshot().unwrap().keyframe_poses,
        result.keyframe_poses
    );
    assert_eq!(
        format!("{:?}", on.estimator()),
        window,
        "finish_map changed the window"
    );

    // Realtime builds the same map, so its GBA gives the same result, and its
    // posted view shows it.
    realtime.finish_deferred_keyframe().unwrap();
    let threaded_result = realtime.finish_map(true).unwrap().unwrap();
    assert_eq!(threaded_result.keyframe_poses, result.keyframe_poses);
    assert_eq!(
        realtime.map_snapshot().unwrap().keyframe_poses,
        result.keyframe_poses
    );
}
