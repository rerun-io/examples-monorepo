//! GPU-resident Gaussian splat rendering. Algorithms follow Brush 1388f74c.

mod gpu;
mod primitives;
mod renderer;
mod types;
pub use renderer::Renderer;
pub use types::{Camera, Capabilities, FrameStats, RenderOptions, Splats, Target};

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
    #[error("GPU count readback: {0}")]
    Readback(String),
}

#[cfg(test)]
mod primitive_tests;
