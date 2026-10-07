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
    /// Resolved mode for this render.
    pub render_mode: RenderMode,
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
            render_mode: RenderMode::Default,
            world_from_local: glam::Affine3A::IDENTITY,
            background: Vec3::ZERO,
            splat_scale: 1.0,
        }
    }
}

/// Caller-owned GPU output. Buffers require STORAGE; texture requires STORAGE_BINDING.
#[derive(Clone, PartialEq)]
pub enum Target {
    /// Four f32 lanes per pixel, unclipped RGB, for parity and HDR composition.
    Float(wgpu::Buffer),
    /// One packed RGBA8 u32 per pixel, Brush's truncating quantization.
    Packed(wgpu::Buffer),
    /// rgba8unorm storage texture, for viewer composition.
    Texture(wgpu::TextureView),
    /// Viewer-only color plus alpha-weighted expected camera depth (positive Z), r32float.
    TextureDepth {
        color: wgpu::TextureView,
        depth: wgpu::TextureView,
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

impl Camera {
    pub fn validate(&self) -> Result<(), crate::Error> {
        if self.size.x == 0
            || self.size.y == 0
            || !self.position.is_finite()
            || !self.rotation.is_finite()
            || !self.center_uv.is_finite()
            || !(self.fov_x > 0.0 && self.fov_x < std::f64::consts::TAU)
            || !(self.fov_y > 0.0 && self.fov_y < std::f64::consts::TAU)
            || !self.model.coefficients().iter().all(|x| x.is_finite())
        {
            return Err(crate::Error::Input("invalid camera"));
        }
        Ok(())
    }
}
impl RenderOptions {
    pub fn validate(&self) -> Result<(), crate::Error> {
        if !self.world_from_local.is_finite()
            || !self.world_from_local.inverse().is_finite()
            || !self.background.is_finite()
            || !self.splat_scale.is_finite()
            || self.splat_scale <= 0.0
        {
            return Err(crate::Error::Input("invalid render options"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
pub(crate) enum RasterKind {
    Float,
    Packed,
    Texture,
    TextureDepth,
}
impl RasterKind {
    pub fn binding(self) -> u32 {
        match self {
            Self::Float => 4,
            Self::Packed => 5,
            Self::Texture | Self::TextureDepth => 6,
        }
    }
}
impl Target {
    pub(crate) fn layout(&self, size: UVec2) -> Result<RasterKind, crate::Error> {
        let texture = |view: &wgpu::TextureView, format| {
            let t = view.texture();
            if t.width() != size.x
                || t.height() != size.y
                || t.depth_or_array_layers() != 1
                || t.mip_level_count() != 1
                || t.sample_count() != 1
                || t.dimension() != wgpu::TextureDimension::D2
                || t.format() != format
                || !t.usage().contains(wgpu::TextureUsages::STORAGE_BINDING)
            {
                return Err(crate::Error::Input(
                    "target requires a matching single-mip storage texture",
                ));
            }
            Ok(())
        };
        match self {
            Self::Float(buffer) | Self::Packed(buffer) => {
                let float = matches!(self, Self::Float(_));
                let bytes = u64::from(size.x) * u64::from(size.y);
                let bytes = bytes
                    .checked_mul(if float { 16 } else { 4 })
                    .ok_or(crate::Error::Input("target size overflow"))?;
                if buffer.size() < bytes || !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
                    return Err(crate::Error::Input(
                        "target buffer is too small or lacks STORAGE usage",
                    ));
                }
                Ok(if float {
                    RasterKind::Float
                } else {
                    RasterKind::Packed
                })
            }
            Self::Texture(color) => {
                texture(color, wgpu::TextureFormat::Rgba8Unorm)?;
                Ok(RasterKind::Texture)
            }
            Self::TextureDepth { color, depth } => {
                texture(color, wgpu::TextureFormat::Rgba8Unorm)?;
                texture(depth, wgpu::TextureFormat::R32Float)?;
                Ok(RasterKind::TextureDepth)
            }
        }
    }
    pub(crate) fn resource(&self) -> wgpu::BindingResource<'_> {
        match self {
            Self::Float(buffer) | Self::Packed(buffer) => buffer.as_entire_binding(),
            Self::Texture(color) | Self::TextureDepth { color, .. } => {
                wgpu::BindingResource::TextureView(color)
            }
        }
    }
}
