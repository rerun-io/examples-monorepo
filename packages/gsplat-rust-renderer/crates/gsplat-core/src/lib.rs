//! GPU-resident Gaussian splat rendering. Algorithms follow Brush 1388f74c.
//!
//! Devices need the WebGPU compute limits in `device_caps`. `wgpu::Features::SUBGROUP` is optional:
//! where the device enables it, the scan and sort kernels use subgroup operations; otherwise they
//! use a workgroup-memory variant with the same results (`shader/scan_portable.wgsl`).
//! On the web the subgroup modules get `enable subgroups;` prepended (naga 30 rejects it natively).

mod camera;
mod device_caps;
pub use device_caps::{NO_SUBGROUPS_ENV, check_adapter, compute_limits, required_features};
mod gpu;
mod kernels;
mod lens;
pub mod native;
mod output;
mod primitives;
mod renderer;
mod scene;
mod shader;
pub mod sog;
mod sog_gpu;
mod timing;
pub mod trigs;
mod types;
mod view;
/// The glam of every public math type here (the workspace glam the CLI shares with Brush cameras),
/// for callers on another glam.
pub use glam;
pub use lens::CameraModel;
pub use output::Target;
pub use renderer::Renderer;
pub use scene::Scene;
pub use timing::{QUERY_COUNT, STAGE_NAMES, stage_ms};
pub use types::{Camera, FrameStats, HexelVariant, Hexels, RenderMode, RenderOptions, Splats};
pub use view::ViewState;

/// Input, capability, or readback failure at the renderer boundary.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "gsplat-core requires eight storage buffers, three storage textures, 256 threads and 10 KiB workgroup memory"
    )]
    Capabilities,

    #[error("invalid splat render input: {0}")]
    Input(&'static str),

    #[error("view busy: {0}")]
    Busy(&'static str),

    #[error(
        "GPU buffer requires {required} bytes, device allows {limit}; no splats were truncated"
    )]
    Capacity { required: u64, limit: u64 },

    #[error("intersection count exceeds u32 capacity; no truncated frame was rendered")]
    IntersectionOverflow,

    #[error("GPU count readback: {0}")]
    Readback(#[from] wgpu::BufferAsyncError),

    #[error("GPU mapped range: {0}")]
    MappedRange(#[from] wgpu::MapRangeError),

    #[error("GPU device poll: {0}")]
    Poll(#[from] wgpu::PollError),

    #[error("GPU feedback callback disconnected")]
    Disconnected(#[from] std::sync::mpsc::TryRecvError),

    #[error("invalid GPU stage timestamps: {0:?}")]
    InvalidTimestamps(Vec<u64>),

    #[error(transparent)]
    Sog(#[from] sog::Error),

    #[error("too many splats: {0}")]
    TooManySplats(#[from] std::num::TryFromIntError),
}

#[cfg(test)]
mod hexel_tests;
#[cfg(test)]
mod primitive_tests;

#[cfg(any(test, feature = "testing"))]
pub mod test_utils;
