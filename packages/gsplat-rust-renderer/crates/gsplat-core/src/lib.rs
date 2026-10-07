//! GPU-resident Gaussian splat rendering. Algorithms follow Brush 1388f74c.
//!
//! Native devices must enable `wgpu::Features::SUBGROUP` and WebGPU compute limits.
//! A future web build needs the WebGPU SUBGROUP mapping from at least Brush's wgpu
//! fork commit `4db81837f`, and `enable subgroups;` prepended on WebGPU only.
//! Naga 30 rejects that directive; no browser path is implemented here.

// Optional depth raster: 256 * (9 splat floats + 1 depth float), plus four shared scalars.
pub(crate) const REQUIRED_WORKGROUP_STORAGE_BYTES: u32 = 10_256;

mod camera;
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
pub use timing::{STAGE_NAMES, stage_queries};
pub use types::{Camera, FrameStats, RenderMode, RenderOptions, Splats};
pub use view::ViewState;

/// Input, capability, or readback failure at the renderer boundary.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "gsplat-core requires SUBGROUP, eight storage buffers, 256 threads and 10 KiB workgroup memory"
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
    Readback(String),
}

#[cfg(test)]
mod primitive_tests;
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_utils;

/// Validate the enabled compute capabilities before creating pipelines.
pub fn check_adapter(features: wgpu::Features, limits: &wgpu::Limits) -> Result<(), Error> {
    if !features.contains(wgpu::Features::SUBGROUP)
        || limits.max_storage_buffers_per_shader_stage < 8
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_storage_size < REQUIRED_WORKGROUP_STORAGE_BYTES
    {
        return Err(Error::Capabilities);
    }
    Ok(())
}
/// Subgroups are mandatory; timing is enabled only when supported.
pub fn required_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    wgpu::Features::SUBGROUP | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY)
}
