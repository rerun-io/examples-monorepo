//! The GBA lane when the device's round trip disagrees with the host (DESIGN
//! §7.3): the real selector over the real probe, whose device result a
//! test-only hook turns into zeros, as a runtime that runs no kernel would.
//! One process: the hook stays armed for the whole binary.
#![cfg(all(feature = "gpu-wgpu", feature = "gpu-test-support"))]
#![allow(clippy::unwrap_used)]

use slam_rs::config::BackendMode;
use slam_rs::{Backend, BackendRequest, Backends, GbaRequest, Vio, gba_probes_run, select_backend};

mod common;

#[test]
fn a_disagreeing_round_trip_moves_only_the_gba_to_the_cpu() {
    kornia_staging_gpu::linalg::corrupt_round_trips();
    for _ in 0..3 {
        assert_eq!(
            select_backend(BackendRequest::Auto, GbaRequest::Follow).unwrap(),
            Backends {
                frontend: Backend::Gpu,
                gba: Backend::Cpu
            },
            "this test needs a GPU that passes the frontend probes"
        );
    }
    let error = select_backend(BackendRequest::Auto, GbaRequest::Gpu).unwrap_err();
    assert!(
        error.to_string().contains("disagrees with the CPU lane"),
        "{error}"
    );
    // A pipeline on a named GPU frontend with the map on takes the same verdict.
    let vio: Vio<f32> = Vio::with_backend(
        common::config().with_backend(Some(BackendMode::Offline)),
        common::calibration(),
        Default::default(),
        Backend::Gpu,
    )
    .unwrap();
    assert_eq!(
        (vio.backend(), vio.gba_lane()),
        (Backend::Gpu, Backend::Cpu)
    );
    assert_eq!(gba_probes_run(), 1, "the verdict is cached: one round trip");
}
