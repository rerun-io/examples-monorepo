//! GPU-resident Gaussian splat rendering. Algorithms follow Brush 1388f74c.
//!
//! Native devices must enable `wgpu::Features::SUBGROUP` and WebGPU compute limits.
//! A future web build needs the WebGPU SUBGROUP mapping from at least Brush's wgpu
//! fork commit `4db81837f`, and `enable subgroups;` prepended on WebGPU only.
//! Naga 30 rejects that directive; no browser path is implemented here.

mod camera;
mod device_caps;
pub use device_caps::{check_adapter, compute_limits, required_features};
mod gpu;
mod kernels;
mod lens;
pub mod native;
mod output;
mod primitives;
mod renderer;
mod scene;
mod shader;
mod timing;
mod types;
mod view;
pub use lens::CameraModel;
pub use output::Target;
pub use renderer::Renderer;
pub use scene::Scene;
pub use timing::{QUERY_COUNT, STAGE_NAMES, stage_ms};
pub use types::{Camera, FrameStats, RenderMode, RenderOptions, Splats};
pub use view::ViewState;

/// Input, capability, or readback failure at the renderer boundary.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "gsplat-core requires SUBGROUP, eight storage buffers, two storage textures, 256 threads and 10 KiB workgroup memory"
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

    #[error("too many splats: {0}")]
    TooManySplats(#[from] std::num::TryFromIntError),
}

#[cfg(test)]
mod primitive_tests;

#[cfg(any(test, feature = "testing"))]
pub mod test_utils;
