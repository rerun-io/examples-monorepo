//! Visual-inertial odometry core consuming grayscale images and IMU samples.
//! Python catalog, evaluation and logging live in `slam_rs`; bindings live in
//! `slam-rs-py`. The CLI currently provides only `version`.
//! [`Vio`] buffers IMU samples and processes frames synchronously, returning final
//! values with deterministic replay (D17). GPU frontend support is feature-gated.

mod pipeline;
pub use pipeline::{
    Backend, FrontendLane, FrontendTimings, ImageView, OverlapTimings, PreparedTrack, Vio,
    VioError, VioPose, VioResult, VioStatus, check_imu_sample,
};

pub mod area;
pub mod ba_base;
pub mod calib;
pub mod camera;
pub mod catalog_timing;
pub mod config;
pub mod estimator;
pub mod frontend;
#[cfg(feature = "gpu-core")]
pub mod gpu;
mod ldlt;
pub(crate) mod qr;

// `gpu-core` is the kernels and the seam; the runtime comes from `gpu-wgpu`. Enabled on its own there would be no client to build one on, and
// the failure would be a wall of missing items rather than a sentence.
#[cfg(all(feature = "gpu-core", not(feature = "gpu-wgpu")))]
compile_error!(
    "feature `gpu-core` carries the CubeCL kernels but no runtime: enable \
     `gpu-wgpu` for the wgpu runtime"
);
pub mod image;
pub mod imu;
pub mod landmark;
pub mod lie;
pub mod linearize;
pub mod marg;
pub mod pyramid;
pub mod replay;
pub mod types;

/// Version of the core, as declared in `crates/slam-rs/Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Which GPU runtime this build's frontend carries, or `None` for the CPU-only
/// default.
///
/// Nothing a caller can pass through `gpu: bool` names the runtime a build
/// carries. This is that fact, and it is read on the Python side
/// (`_core.gpu_backend`) to name the lane a fleet row was measured on.
///
/// [`gpu::BACKEND_NAME`] is the same name; this wrapper is what a build without
/// the feature can still answer.
#[cfg(feature = "gpu-core")]
pub const GPU_BACKEND: Option<&str> = Some(gpu::BACKEND_NAME);

/// Which GPU runtime this build's frontend carries: none, this being the
/// off-by-default CPU port the fleet installs.
#[cfg(not(feature = "gpu-core"))]
pub const GPU_BACKEND: Option<&str> = None;

/// Elapsed nanoseconds, saturating rather than panicking on an absurd clock.
///
/// The one place a stage mark is taken: the estimator's six
/// ([`estimator::StageTimings`]) and the frontend's
/// ([`frontend::flow::FlowTimings`]) are the same measurement of different work.
pub(crate) fn duration_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
