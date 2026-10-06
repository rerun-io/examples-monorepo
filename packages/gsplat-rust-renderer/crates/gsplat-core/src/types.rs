//! Public input layout and render configuration. Camera coordinates are right/down/forward.
use glam::{Quat, UVec2, Vec2, Vec3};

/// Brush-compatible f32 parameters; no activation or precision conversion occurs on upload.
#[derive(Clone, Debug, Default)]
pub struct Splats {
    /// Mean xyz, quaternion wxyz, log-scale xyz, tightly packed (40 bytes).
    pub transforms: Vec<[f32; 10]>,
    pub raw_opacities: Vec<f32>,
    /// Splat-major, then coefficient-major RGB (12 bytes per coefficient).
    pub sh_coefficients: Vec<[f32; 3]>,
    pub sh_degree: u32,
    pub min_scale: Option<Vec<f32>>,
}

/// Camera pose is camera-to-world. Principal point is normalized by image size.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
    pub fov_x: f64,
    pub fov_y: f64,
    pub center_uv: Vec2,
    pub size: UVec2,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderOptions {
    /// Applied during rasterization; alpha remains accumulated splat coverage.
    pub background: Vec3,
}

/// Caller-owned GPU output. Buffers require STORAGE; texture requires STORAGE_BINDING.
pub enum Target<'a> {
    /// Four f32 lanes per pixel, unclipped RGB, for parity and HDR composition.
    Float(&'a wgpu::Buffer),
    /// One packed RGBA8 u32 per pixel, Brush's truncating quantization.
    Packed(&'a wgpu::Buffer),
    /// rgba8unorm storage texture, for viewer composition.
    Texture(&'a wgpu::TextureView),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub visible: u32,
    pub intersections: u32,
    pub intersection_capacity: u32,
    pub overflow_events: u32,
}

/// Enabled-device capabilities, validated again by Renderer::new.
#[derive(Clone, Copy, Debug)]
pub struct Capabilities {
    pub max_storage_buffer_bytes: u64,
}
impl Capabilities {
    pub fn from_device(device: &wgpu::Device) -> Result<Self, crate::Error> {
        let limits = device.limits();
        if !device.features().contains(wgpu::Features::SUBGROUP)
            || limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_workgroup_storage_size < 10_240
        {
            return Err(crate::Error::Capabilities);
        }
        Ok(Self {
            max_storage_buffer_bytes: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
        })
    }
}
