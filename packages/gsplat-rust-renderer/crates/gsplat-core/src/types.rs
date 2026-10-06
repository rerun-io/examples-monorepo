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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub model: crate::CameraModel,
    pub position: Vec3,
    pub rotation: Quat,
    pub fov_x: f64,
    pub fov_y: f64,
    pub center_uv: Vec2,
    pub size: UVec2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderMode {
    #[default]
    Default,
    Mip,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOptions {
    /// Per-view override; None uses the shared scene's mode.
    pub render_mode: Option<RenderMode>,
    /// Instance affine transform; SH directions remain in local coordinates.
    pub world_from_local: glam::Affine3A,
    /// Applied during rasterization; alpha remains accumulated splat coverage.
    pub background: Vec3,
    /// Positive multiplier, applied as a log-scale offset before the 3D floor.
    pub splat_scale: f32,
}
impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            render_mode: None,
            world_from_local: glam::Affine3A::IDENTITY,
            background: Vec3::ZERO,
            splat_scale: 1.0,
        }
    }
}

/// Caller-owned GPU output. Buffers require STORAGE; texture requires STORAGE_BINDING.
pub enum Target<'a> {
    /// Four f32 lanes per pixel, unclipped RGB, for parity and HDR composition.
    Float(&'a wgpu::Buffer),
    /// One packed RGBA8 u32 per pixel, Brush's truncating quantization.
    Packed(&'a wgpu::Buffer),
    /// rgba8unorm storage texture, for viewer composition.
    Texture(&'a wgpu::TextureView),
    /// Viewer-only color plus alpha-weighted expected camera depth (positive Z), r32float.
    TextureDepth {
        color: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub visible: u32,
    pub intersections: u32,
    pub intersection_capacity: u32,
    pub overflow_events: u32,
    /// The previous allocation was too small; submit this view again after feedback.
    pub needs_rerender: bool,
}
