//! D84: `port.keyframe_solve_deferred` on a synthetic moving rig.
//!
//! Four pinhole cameras 8 cm apart look up at a textured ceiling 3 m above the
//! rig, which slides sideways from rest and back (no rotation) with a matching
//! IMU. The same frames run with the keyframe solve synchronous and deferred:
//! the deferred run must defer keyframes, report each deferred solve on the
//! next frameset, repeat itself exactly, and stay as close to the true path as
//! the synchronous run.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use slam_rs as api;
#[path = "common/deferred.rs"]
mod fixture;
use fixture::*;

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

#[test]
fn frontend_lag_and_deferred_keyframes_drain_in_order() {
    let mut config = config(true);
    config.port_frontend_lag = true;
    let mut vio = api::Vio::<f32>::new(
        config,
        calibration(),
        api::frontend::flow::FrontendOptions {
            threads: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let t0 = 1_000_000_000;
    let mut imu_t = t0 - 100_000_000;
    let mut deferred_t = None;
    let mut solved = 0;
    let mut results = Vec::new();
    for index in 0..FRAMES {
        let t_ns = t0 + index * FRAME_NS;
        while imu_t <= t_ns + 5_000_000 {
            let t_s = ((imu_t - t0) as f64 / 1e9).max(0.0);
            let accel = acceleration(t_s) + nalgebra::Vector3::new(0.0, 0.0, 9.81);
            vio.push_imu(imu_t, [0.0; 3], accel.into()).unwrap();
            imu_t += 5_000_000;
        }
        let frames: [Vec<u8>; 4] =
            std::array::from_fn(|camera| render((t_ns - t0) as f64 / 1e9, camera));
        let views: Vec<_> = frames
            .iter()
            .map(|data| api::ImageView {
                data,
                width: 640,
                height: 360,
                stride: 640,
            })
            .collect();
        let result = vio.track(t_ns, &views).unwrap();
        if index == 0 {
            assert_eq!(result.status, api::VioStatus::Buffered);
            continue;
        }
        assert_eq!(result.t_ns, t_ns - FRAME_NS);
        assert_eq!(
            vio.last_deferred_keyframe().map(|stats| stats.t_ns),
            deferred_t
        );
        solved += usize::from(vio.last_deferred_keyframe().is_some());
        let stats = vio.last_stats().unwrap();
        deferred_t = stats.keyframe_deferred.then_some(result.t_ns);
        results.push(result);
    }
    results.push(vio.flush().unwrap().unwrap());
    assert_eq!(results.len(), FRAMES as usize);
    assert_eq!(results.last().unwrap().t_ns, t0 + (FRAMES - 1) * FRAME_NS);
    assert!(solved >= 3, "only {solved} deferred solves");
    assert!(!vio.estimator().has_deferred_keyframe());
    assert_eq!(vio.flush().unwrap(), None);
    for result in results {
        let truth = position((result.t_ns - t0) as f64 / 1e9);
        let estimate = nalgebra::Vector3::from_row_slice(&result.pose.unwrap().world_from_rig[..3]);
        assert!((estimate - truth).norm() < 0.10);
    }
}

#[test]
fn a_deferred_keyframe_solve_is_reported_next_deterministic_and_as_accurate() {
    let frames: Vec<[Vec<u8>; 4]> = (0..FRAMES)
        .map(|index| {
            let t_s = index as f64 * FRAME_NS as f64 / 1e9;
            std::array::from_fn(|camera| render(t_s, camera))
        })
        .collect();

    let synchronous = run(&frames, &mut pipeline(false));
    let deferred = run(&frames, &mut pipeline(true));
    let again = run(&frames, &mut pipeline(true));

    assert_eq!(synchronous.deferred_keyframes, 0);
    assert!(
        deferred.deferred_keyframes >= 3,
        "only {} keyframes were deferred",
        deferred.deferred_keyframes
    );
    assert!(
        deferred.reported_solves + 1 >= deferred.deferred_keyframes,
        "a deferred solve went unreported"
    );
    assert_eq!(deferred.solves, again.solves);
    assert_eq!(
        deferred.results, again.results,
        "the deferred lane is not deterministic"
    );

    let (sync_rms, deferred_rms) = (rms(&synchronous.errors_m), rms(&deferred.errors_m));
    let worst = deferred.errors_m.iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "{} keyframes deferred, {} solves reported; position error rms: synchronous {:.4} m, deferred {:.4} m (worst {:.4} m)",
        deferred.deferred_keyframes, deferred.reported_solves, sync_rms, deferred_rms, worst
    );
    assert!(
        sync_rms < 0.05,
        "the synchronous run is off the true path: {sync_rms} m rms"
    );
    assert!(
        worst < 0.10,
        "the deferred run left the true path by {worst} m"
    );
    assert!(
        deferred_rms <= 1.5 * sync_rms + 0.005,
        "deferred {deferred_rms} m rms vs synchronous {sync_rms} m"
    );
}
