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
    Readback(String),
}

#[cfg(test)]
mod primitive_tests;
#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod test_utils;

fn required_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        max_storage_textures_per_shader_stage: 2,
        max_compute_workgroup_size_y: 1,
        max_compute_workgroup_size_z: 1,
        max_compute_workgroups_per_dimension: 65_535,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_storage_size: REQUIRED_WORKGROUP_STORAGE_BYTES,
        ..wgpu::Limits::downlevel_webgl2_defaults()
    }
}
/// Validate the enabled compute capabilities before creating pipelines.
pub fn check_adapter(features: wgpu::Features, limits: &wgpu::Limits) -> Result<(), Error> {
    if !features.contains(wgpu::Features::SUBGROUP) || !required_limits().check_limits(limits) {
        return Err(Error::Capabilities);
    }
    Ok(())
}
/// Subgroups are mandatory; timing is enabled only when supported.
pub fn required_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    wgpu::Features::SUBGROUP | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY)
}

/// Add only the core's compute requirements to a caller's renderer limits.
/// Storage binding capacity follows the adapter so a large SH buffer can fit.
pub fn compute_limits(available: &wgpu::Limits, mut requested: wgpu::Limits) -> wgpu::Limits {
    requested = requested
        .or_better_values_from(&required_limits())
        .or_worse_values_from(available);
    requested.max_storage_buffer_binding_size = available.max_storage_buffer_binding_size;
    requested.max_buffer_size = requested
        .max_buffer_size
        .max(available.max_storage_buffer_binding_size)
        .min(available.max_buffer_size);
    requested
}

#[cfg(test)]
mod capability_tests {
    #[test]
    fn startup_reports_missing_subgroups() {
        let error =
            super::check_adapter(wgpu::Features::empty(), &wgpu::Limits::default()).unwrap_err();
        assert!(error.to_string().contains("SUBGROUP"));
        assert!(super::check_adapter(wgpu::Features::SUBGROUP, &wgpu::Limits::default()).is_ok());
        let mut failures = 0;
        super::required_limits().check_limits_with_fail_fn(
            &wgpu::Limits::downlevel_webgl2_defaults(),
            false,
            |_, _, _| failures += 1,
        );
        assert_eq!(failures, 8);
    }
    #[test]
    fn depth_raster_requires_two_storage_textures() {
        let limits = wgpu::Limits {
            max_storage_textures_per_shader_stage: 1,
            ..wgpu::Limits::default()
        };
        assert!(super::check_adapter(wgpu::Features::SUBGROUP, &limits).is_err());
    }
    #[test]
    fn compute_limits_preserve_renderer_limits_and_fit_large_bindings() {
        let available = wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 2 << 30,
            ..wgpu::Limits::default()
        };
        let requested = wgpu::Limits::downlevel_webgl2_defaults();
        let limits = super::compute_limits(&available, requested.clone());
        assert_eq!(limits.max_buffer_size, 1 << 30);
        assert_eq!(limits.max_storage_buffer_binding_size, 1 << 30);
        assert_eq!(limits.max_bind_groups, requested.max_bind_groups);
        assert_eq!(
            limits.max_texture_dimension_2d,
            requested.max_texture_dimension_2d
        );
        assert_eq!(
            limits.max_uniform_buffers_per_shader_stage,
            requested.max_uniform_buffers_per_shader_stage
        );
        assert_eq!(limits.max_storage_buffers_per_shader_stage, 8);
        assert_eq!(limits.max_storage_textures_per_shader_stage, 2);
        super::check_adapter(wgpu::Features::SUBGROUP, &limits).unwrap();
        assert!(limits.check_limits(&available));
        let weak = wgpu::Limits::downlevel_webgl2_defaults();
        assert!(super::compute_limits(&weak, weak.clone()).check_limits(&weak));
    }
}
