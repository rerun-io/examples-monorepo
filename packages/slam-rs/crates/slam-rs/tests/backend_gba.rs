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
    pipeline_on(mode, api::Backend::Cpu)
}

/// The CPU frontend, with the global BA's PCG on `gba`.
fn pipeline_on(mode: Option<BackendMode>, gba: api::Backend) -> api::Vio<f32> {
    let mut config = config(false);
    config.backend_enabled = mode.is_some();
    config.backend_mode = mode.unwrap_or_default();
    // Keyframes leave the window, so the map holds more than it.
    config.vio_new_kf_keypoints_thresh = 0.9;
    config.vio_max_kfs = 3;
    api::Vio::with_backends(
        config,
        calibration(),
        api::frontend::flow::FrontendOptions {
            threads: 2,
            ..Default::default()
        },
        api::Backends {
            frontend: api::Backend::Cpu,
            gba,
        },
    )
    .unwrap()
}

fn frames() -> Vec<[Vec<u8>; 4]> {
    (0..FRAMES)
        .map(|index| {
            let t_s = index as f64 * FRAME_NS as f64 / 1e9;
            std::array::from_fn(|camera| render(t_s, camera))
        })
        .collect()
}

/// The GBA records its lane; a build without the GPU lane refuses it by name
/// instead of running the CPU.
#[test]
fn the_gba_records_its_lane() {
    let frames = frames();
    let mut cpu = pipeline_on(Some(BackendMode::Offline), api::Backend::Cpu);
    run(&frames, &mut cpu);
    let result = cpu.finish_map(true).unwrap();
    assert_eq!(result.gba.unwrap().lane, api::Backend::Cpu);
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        let mut gpu = pipeline_on(Some(BackendMode::Offline), api::Backend::Gpu);
        run(&frames, &mut gpu);
        let failed = gpu.finish_map(true).unwrap();
        let error = failed.gba_failure.unwrap().to_string();
        assert!(error.contains("GPU lane cannot start"), "{error}");
        assert!(failed.gba.is_none());
        assert_eq!(failed.keyframe_poses, failed.keyframe_poses_before_gba);
    }
}

/// The device lane lands in the CPU lane's band on the same map (DESIGN §7.4).
#[cfg(feature = "gpu-wgpu")]
#[test]
fn the_gpu_lane_lands_in_the_cpu_band() {
    let frames = frames();
    let mut cpu = pipeline_on(Some(BackendMode::Offline), api::Backend::Cpu);
    let mut gpu = pipeline_on(Some(BackendMode::Offline), api::Backend::Gpu);
    run(&frames, &mut cpu);
    run(&frames, &mut gpu);
    let (cpu, gpu) = (cpu.finish_map(true).unwrap(), gpu.finish_map(true).unwrap());
    let (c, g) = (cpu.gba.unwrap(), gpu.gba.unwrap());
    assert_eq!(g.lane, api::Backend::Gpu);
    let (cc, gc) = (c.report.final_cost.error, g.report.final_cost.error);
    assert!((gc - cc).abs() <= 1e-3 * cc.max(1e-9), "{gc} vs {cc}");
    for (a, b) in cpu.keyframe_poses.iter().zip(&gpu.keyframe_poses) {
        assert!((a.translation - b.translation).norm() < 1e-3);
    }
}

#[test]
fn finish_map_runs_the_gba_on_the_map_and_never_on_the_window() {
    let frames = frames();
    let mut off = pipeline_with_map(None);
    let mut on = pipeline_with_map(Some(BackendMode::Offline));
    let mut realtime = pipeline_with_map(Some(BackendMode::Realtime));
    let without = run(&frames, &mut off);
    let with = run(&frames, &mut on);
    let threaded = run(&frames, &mut realtime);
    assert_eq!(with.results, without.results);
    assert_eq!(threaded.results, without.results);
    assert!(off.finish_map(true).is_none(), "no map, no result");

    let window = format!("{:?}", on.estimator());
    let snapshot = on.map_snapshot().unwrap();
    let plain = on.finish_map(false).unwrap();
    assert!(plain.gba.is_none());
    assert_eq!(plain.keyframe_t_ns, snapshot.keyframe_t_ns);
    assert_eq!(plain.keyframe_poses, snapshot.keyframe_poses);
    assert_eq!(plain.keyframe_poses_before_gba, snapshot.keyframe_poses);
    assert_eq!(
        on.map_snapshot().unwrap(),
        snapshot,
        "finish_map(false) changed the map"
    );

    let result = on.finish_map(true).unwrap();
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
    assert_eq!(result.keyframe_poses_before_gba, snapshot.keyframe_poses);
    assert_ne!(
        result.keyframe_poses, result.keyframe_poses_before_gba,
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
    let threaded_result = realtime.finish_map(true).unwrap();
    assert_eq!(threaded_result.keyframe_poses, result.keyframe_poses);
    assert_eq!(
        realtime.map_snapshot().unwrap().keyframe_poses,
        result.keyframe_poses
    );
}

/// The final trajectory (DESIGN §1, §2; PR k0): the global BA on the map after
/// loops and PGO, then the raw VIO re-anchored on the GBA's keyframes. Without
/// the GBA it is f1's +LC final exactly, and the raw VIO never moves.
#[test]
fn the_final_trajectory_is_the_raw_vio_reanchored_on_the_gba_keyframes() {
    let frames = frames();
    let mut off = pipeline_with_map(None);
    let mut on = pipeline_with_map(Some(BackendMode::Offline));
    let without = run(&frames, &mut off);
    let with = run(&frames, &mut on);
    assert_eq!(with.results, without.results, "the raw VIO moved");
    let raw: Vec<(i64, [f64; 7])> = with
        .results
        .iter()
        .filter_map(|result| Some((result.t_ns, result.pose?.world_from_rig)))
        .collect();
    assert!(raw.len() > 10);

    let lc_final = on.map_snapshot().unwrap().reanchor(&raw).unwrap();
    let plain = on.finish_map(false).unwrap();
    assert_eq!(
        plain.reanchor(&raw).unwrap(),
        lc_final,
        "run_gba=false is not f1's +LC final"
    );

    let result = on.finish_map(true).unwrap();
    assert!(result.gba.is_some());
    let keyframes: Vec<(i64, api::lie::Se3<f64>)> = result
        .keyframe_t_ns
        .iter()
        .copied()
        .zip(result.keyframe_poses.iter().copied())
        .collect();
    let expected = api::backend::reanchor(
        &raw.iter()
            .map(|&(t_ns, pose)| (t_ns, se3(&pose)))
            .collect::<Vec<_>>(),
        &keyframes,
    )
    .unwrap();
    let gba_final = result.reanchor(&raw).unwrap();
    assert_eq!(gba_final.len(), raw.len());
    for (ours, theirs) in gba_final.iter().zip(&expected) {
        assert_eq!(*ours, pose_array(theirs));
    }
    assert_ne!(gba_final, lc_final, "the GBA moved no frame");
    // Each keyframe lands on its GBA pose.
    for (t_ns, pose) in &keyframes {
        let row = raw.binary_search_by_key(t_ns, |&(t, _)| t).unwrap();
        assert_eq!(gba_final[row], pose_array(pose));
    }
    assert_eq!(
        on.map_snapshot().unwrap().reanchor(&raw).unwrap(),
        gba_final,
        "the map after finish_map gives the same final"
    );
}

fn se3(pose: &[f64; 7]) -> api::lie::Se3<f64> {
    api::lie::Se3::new(
        api::lie::So3::from_quaternion_xyzw(pose[3], pose[4], pose[5], pose[6]).unwrap(),
        nalgebra::Vector3::new(pose[0], pose[1], pose[2]),
    )
}

fn pose_array(pose: &api::lie::Se3<f64>) -> [f64; 7] {
    let [x, y, z, w] = pose.rotation.quaternion_xyzw();
    [
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        x,
        y,
        z,
        w,
    ]
}
