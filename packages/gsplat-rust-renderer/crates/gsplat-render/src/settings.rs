//! Render controls shared by the standalone CLI and benchmark.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    #[default]
    Auto,
    Default,
    Mip,
}
#[derive(Clone, Copy, Debug, clap::Args, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderSettings {
    #[arg(long, value_enum, default_value = "auto")]
    pub render_mode: Mode,
    #[arg(long, default_value_t = 1.0)]
    pub splat_scale: f32,
    #[arg(long)]
    pub min_scale: Option<f32>,
    #[arg(long, default_value_t = 1_048_576)]
    pub initial_capacity: u32,
}
impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            render_mode: Mode::Auto,
            splat_scale: 1.0,
            min_scale: None,
            initial_capacity: 1_048_576,
        }
    }
}
impl RenderSettings {
    pub fn validate(&self) -> crate::Result<()> {
        if !self.splat_scale.is_finite()
            || self.splat_scale <= 0.0
            || self.min_scale.is_some_and(|s| !s.is_finite() || s < 0.0)
        {
            return Err(crate::Error::Invalid(
                "invalid splat scale or 3D scale floor".into(),
            ));
        }
        Ok(())
    }
    pub fn mode(&self, metadata: gsplat_core::RenderMode) -> gsplat_core::RenderMode {
        match self.render_mode {
            Mode::Auto => metadata,
            Mode::Default => gsplat_core::RenderMode::Default,
            Mode::Mip => gsplat_core::RenderMode::Mip,
        }
    }
    pub fn options(&self) -> gsplat_core::RenderOptions {
        gsplat_core::RenderOptions {
            splat_scale: self.splat_scale,
            ..Default::default()
        }
    }
}
