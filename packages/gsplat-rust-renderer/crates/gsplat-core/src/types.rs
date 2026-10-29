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

/// Hexel appearance: spatial RGB (V1), plus a linear view term (V2) or SH degrees 1-3 (V3).
/// `Image` hexels are 2D Image-GS-style fits in pixel space, blended top-10 and normalized.
/// The discriminants are the `SceneInfo.variant` codes the shaders test (`HEXEL_V1`, `HEXEL_V2`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HexelVariant {
    #[default]
    V1 = 1,
    V2 = 2,
    V3 = 3,
    Image = 4,
}

impl HexelVariant {
    /// f32 values per hexel. 3D: centre, wxyz rotation, two scales, opacity, sharpness,
    /// base colour and its u/v gradients (20), then 9 (V2) or 45 (V3) view coefficients.
    /// Image: centre and scales in pixels, angle, sharpness, colour and its x/y gradients (15).
    pub const fn stride(self) -> usize {
        match self {
            Self::V1 => 20,
            Self::V2 => 29,
            Self::V3 => 65,
            Self::Image => 15,
        }
    }

    /// Bytes of one projected hexel, matching `Hexel` and `ImageHexel` in the shaders.
    pub(crate) const fn projected_bytes(self) -> u64 {
        match self {
            Self::Image => 64,
            Self::V1 | Self::V2 | Self::V3 => 84,
        }
    }
}

/// Activated planar hexagons (honeycomb-surfels/hexels), `variant.stride()` floats per hexel.
#[derive(Clone, Debug, Default)]
pub struct Hexels {
    pub variant: HexelVariant,
    pub params: Vec<f32>,
}

impl Hexels {
    pub fn len(&self) -> usize {
        self.params.len() / self.variant.stride()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Largest GPU binding for this many hexels: the parameters or the projected hexels.
    pub fn gpu_bytes(count: usize, variant: HexelVariant) -> u64 {
        count as u64 * (variant.stride() as u64 * 4).max(variant.projected_bytes())
    }
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
            || !self.rotation.is_normalized()
            || !self.center_uv.is_finite()
            || !(self.fov_x > 0.0 && self.fov_x < std::f64::consts::TAU)
            || !(self.fov_y > 0.0 && self.fov_y < std::f64::consts::TAU)
            || !self.model.coefficients().iter().all(|x| x.is_finite())
        {
            return Err(crate::Error::Input("invalid camera"));
        }
        if matches!(
            self.model,
            crate::CameraModel::Pinhole | crate::CameraModel::RadialTangential8 { .. }
        ) && (self.fov_x >= std::f64::consts::PI || self.fov_y >= std::f64::consts::PI)
        {
            return Err(crate::Error::Input(
                "forward lens field of view must be below pi",
            ));
        }
        let focal = self.focal();
        if !focal.is_finite() || focal.x <= 0.0 || focal.y <= 0.0 {
            return Err(crate::Error::Input(
                "camera focal lengths must be finite and positive",
            ));
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

impl Splats {
    /// Largest GPU buffer needed for a cloud of this size and SH degree.
    pub fn gpu_bytes(count: usize, degree: u32) -> u64 {
        count as u64 * (12 * (u64::from(degree) + 1).pow(2)).max(40)
    }
}

#[cfg(test)]
mod tests {
    use super::{HexelVariant, Hexels};

    #[test]
    fn hexel_preflight_counts_the_projected_binding_when_it_is_larger() {
        // 1.65 M V1 hexels: 132,000,000 input bytes fit a 128 MiB binding, 84-byte projections do not.
        assert_eq!(Hexels::gpu_bytes(1_650_000, HexelVariant::V1), 138_600_000);
        assert!(Hexels::gpu_bytes(1_650_000, HexelVariant::V1) > 128 << 20);
        // Image hexels: 60 input bytes, 64 projected bytes.
        assert_eq!(Hexels::gpu_bytes(1, HexelVariant::Image), 64);
        // V3 inputs (260 bytes) stay the larger buffer.
        assert_eq!(Hexels::gpu_bytes(2, HexelVariant::V3), 520);
    }
}
