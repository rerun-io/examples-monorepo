//! Capability validation runs once per process, however many pipelines are built.
#![cfg(feature = "gpu-wgpu")]
#![allow(clippy::unwrap_used)]

use slam_rs::{
    Backend, BackendRequest, Backends, GbaRequest, Vio, device_probes_run, probe_back_end,
    select_backend,
};

mod common;

/// The selector and the GPU frontend share one cached validation: the first
/// request runs the storage and subgroup probes once, the GBA lane adds its one
/// round trip, and neither repeated selection nor repeated GPU pipeline
/// construction runs another.
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
        (1..=3).contains(&after_first),
        "one validation is one storage, one subgroup and one GBA probe, ran {after_first}"
    );
    for _ in 0..3 {
        assert_eq!(
            select_backend(BackendRequest::Auto, GbaRequest::Follow).unwrap(),
            gpu
        );
        assert_eq!(
            select_backend(BackendRequest::Gpu, GbaRequest::Gpu).unwrap(),
            gpu
        );
        let vio: Vio<f32> = Vio::with_backend(
            common::config(),
            common::calibration(),
            Default::default(),
            Backend::Gpu,
        )
        .unwrap();
        assert_eq!(vio.backend(), Backend::Gpu);
    }
    assert_eq!(
        device_probes_run(),
        after_first,
        "pipeline construction probed the device again"
    );
}
