//! Capability validation runs once per process, however many pipelines are built.
#![cfg(feature = "gpu-wgpu")]
#![allow(clippy::unwrap_used)]

use slam_rs::config::BackendMode;
use slam_rs::{
    Backend, BackendRequest, Backends, GbaRequest, Vio, device_probes_run, gba_probes_run,
    probe_back_end, select_backend,
};

mod common;

/// The selector and the GPU frontend share one cached validation: the first
/// request runs the storage and subgroup probes once, the GBA lane runs its
/// one round trip (counted apart), and neither repeated selection nor repeated
/// GPU pipeline construction runs another of either.
#[test]
fn repeated_selection_and_construction_probe_the_device_once() {
    let gpu = Backends {
        frontend: Backend::Gpu,
        gba: Backend::Gpu,
    };
    let backends: Backends = select_backend(BackendRequest::Auto, GbaRequest::Follow).unwrap();
    assert_eq!(
        backends, gpu,
        "this test needs a GPU that passes the probes and the GBA round trip"
    );
    probe_back_end().unwrap();
    let after_first: usize = device_probes_run();
    assert!(
        (1..=2).contains(&after_first),
        "one validation is one storage and one subgroup probe, ran {after_first}"
    );
    assert_eq!(gba_probes_run(), 1, "the GBA round trip runs once");
    for _ in 0..3 {
        assert_eq!(
            select_backend(BackendRequest::Auto, GbaRequest::Follow).unwrap(),
            gpu
        );
        assert_eq!(
            select_backend(BackendRequest::Gpu, GbaRequest::Gpu).unwrap(),
            gpu
        );
        // A pipeline built on a named frontend (the RoboCap paths) with the
        // map on: its GBA follows the frontend, from the cached round trip.
        for (frontend, gba) in [(Backend::Gpu, Backend::Gpu), (Backend::Cpu, Backend::Cpu)] {
            let vio: Vio<f32> = Vio::with_backend(
                common::config().with_backend(Some(BackendMode::Offline)),
                common::calibration(),
                Default::default(),
                frontend,
            )
            .unwrap();
            assert_eq!((vio.backend(), vio.gba_lane()), (frontend, gba));
        }
    }
    assert_eq!(
        device_probes_run(),
        after_first,
        "pipeline construction probed the device again"
    );
    assert_eq!(
        gba_probes_run(),
        1,
        "pipeline construction ran another round trip"
    );
}
