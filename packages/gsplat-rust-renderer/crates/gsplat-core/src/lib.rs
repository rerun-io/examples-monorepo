//! GPU-resident Gaussian splat rendering. Algorithms follow Brush 1388f74c.
//!
//! Native devices must enable `wgpu::Features::SUBGROUP` and WebGPU compute limits.
//! A future web build needs the WebGPU SUBGROUP mapping from at least Brush's wgpu
//! fork commit `4db81837f`, and `enable subgroups;` prepended on WebGPU only.
//! Naga 30 rejects that directive; no browser path is implemented here.

mod camera;
mod gpu;
mod kernels;
mod primitives;
mod renderer;
mod scene;
mod types;
mod view;
pub use camera::CameraModel;
pub use renderer::Renderer;
pub use scene::Scene;
pub use types::{Camera, FrameStats, RenderMode, RenderOptions, Splats, Target};
pub use view::ViewState;

/// Input, capability, or readback failure at the renderer boundary.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "gsplat-core requires enabled SUBGROUP, eight storage buffers, 256 compute threads and 10 KiB workgroup memory; request these when creating the wgpu device"
    )]
    Capabilities,
    #[error("invalid splat render input: {0}")]
    Input(&'static str),
    #[error(
        "GPU buffer requires {required} bytes, device allows {limit}; no splats were truncated"
    )]
    Capacity { required: u64, limit: u64 },
    #[error("intersection count exceeds u32 capacity; no truncated frame was rendered")]
    IntersectionOverflow,
    #[error("GPU count readback: {0}")]
    Readback(String),
}

#[cfg(test)]
mod primitive_tests;
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_utils;
