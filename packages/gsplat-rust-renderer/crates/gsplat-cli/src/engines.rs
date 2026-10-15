//! Loading and pixel transfers are separate from the synchronized speed lane.
use crate::PlyScene;
use crate::camera::CameraSpec;
use crate::settings::RenderSettings;
use anyhow::Result;
use glam::Vec3;
use serde::Serialize;
use std::path::Path;
mod archetype;
mod brush;
mod native;
pub use archetype::archetype_splats;
pub use brush::Brush;
pub use native::Native;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Implementation {
    Brush,
    Ours,
    Native,
}

/// Exactly one initialized renderer; parity source choice is independent of backend choice.
pub enum Engine {
    Brush(Box<Brush>),
    Ours(Box<crate::StandaloneRenderer>),
    Native(Box<Native>),
}

impl Engine {
    pub async fn new(
        kind: Implementation,
        scene: &PlyScene,
        settings: &RenderSettings,
        camera: &CameraSpec,
        ply: &Path,
        archetype: Option<&Path>,
    ) -> Result<Self> {
        settings.validate()?;
        if matches!(kind, Implementation::Native)
            && (settings.mode(scene.mode) != gsplat_core::RenderMode::Default
                || settings.splat_scale != 1.0
                || settings.min_scale.is_some())
        {
            return Err(crate::Unsupported(
                "render-mode, scale, or floor controls for this reference renderer".into(),
            )
            .into());
        }
        match kind {
            Implementation::Brush => Ok(Self::Brush(Box::new(Brush::new(scene, settings).await))),
            Implementation::Native => Ok(Self::Native(Box::new(
                Native::new(ply, scene.data.num_splats()).await?,
            ))),
            Implementation::Ours => Ok(Self::Ours(Box::new(
                ours(
                    scene,
                    glam::uvec2(camera.width, camera.height),
                    settings,
                    archetype,
                    Vec3::ZERO,
                )
                .await?,
            ))),
        }
    }
    pub fn output_format(&self) -> &'static str {
        match self {
            Self::Brush(_) => "Brush Packed",
            Self::Ours(_) => "Packed RGBA8",
            Self::Native(_) => "Rerun RGBA8UnormSrgb/MSAA4, opaque black background",
        }
    }

    pub async fn capture(&mut self, camera: &CameraSpec, parity: bool) -> Result<Vec<f32>> {
        self.capture_next_frame();
        self.render(camera, parity).await?;
        self.finish()?;
        self.read_rgba_f32().await
    }
    pub fn capture_next_frame(&mut self) {
        if let Self::Native(r) = self {
            r.capture_next_frame();
        }
    }
    pub async fn render(&mut self, camera: &CameraSpec, parity: bool) -> Result<Counts> {
        match self {
            Self::Brush(r) => r.render(camera, parity).await,
            Self::Ours(r) => {
                let stats = r.render(
                    camera,
                    if parity {
                        crate::Output::Float
                    } else {
                        crate::Output::Packed
                    },
                )?;
                Ok(Counts {
                    visible: Some(stats.visible),
                    intersections: Some(stats.intersections),
                    overflow_events: Some(stats.overflow_events),
                })
            }
            Self::Native(r) => r.render(camera, parity),
        }
    }
    pub fn finish(&self) -> Result<()> {
        match self {
            Self::Brush(r) => r.finish(),
            Self::Ours(_) => Ok(()),
            Self::Native(r) => r.finish(),
        }
    }
    pub async fn read_rgba_f32(&mut self) -> Result<Vec<f32>> {
        match self {
            Self::Brush(r) => r.read_rgba_f32().await,
            Self::Ours(r) => r.read_rgba(),
            Self::Native(r) => r.read_rgba_f32().await,
        }
    }
    pub fn adapter(&self) -> Adapter {
        match self {
            Self::Brush(r) => r.adapter(),
            Self::Ours(r) => r.adapter_info().clone().into(),
            Self::Native(r) => r.adapter(),
        }
    }
    pub async fn stages(&mut self, camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
        if self.adapter().backend == "Metal" {
            return Ok(None);
        }
        match self {
            Self::Brush(r) => r.stages(camera).await,
            Self::Ours(r) => r.stages(camera),
            Self::Native(_) => Ok(None),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Adapter {
    pub name: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
}

impl From<wgpu::AdapterInfo> for Adapter {
    fn from(i: wgpu::AdapterInfo) -> Self {
        Self {
            name: i.name,
            backend: format!("{:?}", i.backend),
            driver: i.driver,
            driver_info: i.driver_info,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Counts {
    pub visible: Option<u32>,
    pub intersections: Option<u32>,
    #[serde(default)]
    pub overflow_events: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct StageTiming {
    pub name: String,
    pub ms: f64,
}

impl crate::StandaloneRenderer {
    pub fn stages(&mut self, camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
        Ok(self.stage_ms(camera)?.map(|times| {
            gsplat_core::STAGE_NAMES
                .into_iter()
                .zip(times)
                .map(|(name, ms)| StageTiming {
                    name: name.into(),
                    ms,
                })
                .collect()
        }))
    }
}

/// Construct the shared renderer with the benchmark's recorded controls.
pub async fn ours(
    scene: &PlyScene,
    size: glam::UVec2,
    settings: &RenderSettings,
    archetype: Option<&Path>,
    background: Vec3,
) -> Result<crate::StandaloneRenderer> {
    let mut splats = if let Some(path) = archetype {
        archetype_splats(path)?
    } else {
        crate::raw_splats(&scene.data)?
    };
    if splats.transforms.len() != scene.data.num_splats() {
        anyhow::bail!("PLY loaders disagree on count");
    }
    if let Some(floor) = settings.min_scale {
        splats.min_scale = Some(vec![floor; scene.data.num_splats()]);
    }
    crate::StandaloneRenderer::new(
        &splats,
        gsplat_core::RenderOptions {
            background,
            ..settings.options(scene.mode)
        },
        size,
        settings.initial_capacity,
    )
    .await
}

#[cfg(test)]
mod coverage;
