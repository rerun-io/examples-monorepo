//! Local float/packed outputs and viewer texture target layouts.
use glam::UVec2;

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
