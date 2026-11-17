//! Capability validation runs once per process, however many pipelines are built.
#![cfg(feature = "gpu-wgpu")]
#![allow(clippy::unwrap_used)]

use slam_rs::{Backend, BackendRequest, Vio, device_probes_run, select_backend};

mod common;

/// The selector and the GPU frontend share one cached validation: the first
/// request runs the storage and subgroup probes once, and neither repeated
/// selection nor repeated GPU pipeline construction runs another.
#[test]
fn repeated_selection_and_construction_probe_the_device_once() {
    let backend: Backend = select_backend(BackendRequest::Auto).unwrap();
    assert_eq!(
        backend,
        Backend::Gpu,
        "this test needs a GPU that passes the probes"
    );
    let after_first: usize = device_probes_run();
    assert!(
        (1..=2).contains(&after_first),
        "one validation is one storage and one subgroup probe, ran {after_first}"
    );
    for _ in 0..3 {
        assert_eq!(select_backend(BackendRequest::Auto).unwrap(), Backend::Gpu);
        assert_eq!(select_backend(BackendRequest::Gpu).unwrap(), Backend::Gpu);
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
