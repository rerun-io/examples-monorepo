//! Loading and pixel transfers are separate from the synchronized speed lane.
use crate::Result;
use crate::camera::CameraSpec;
use crate::settings::RenderSettings;
use anyhow::Context as _;
use brush_render::{
    TextureMode,
    gaussian_splats::{SplatRenderMode, Splats},
};
use burn::tensor::{Device, Tensor};
use glam::Vec3;
use serde::Serialize;
use std::path::Path;
mod archetype;
mod native;
pub use archetype::archetype_splats;
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
    Ours(Box<crate::Renderer>),
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
            Self::Native(r) => r.render(camera, parity).await,
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

pub use crate::PlyScene;

pub struct Brush {
    splats: Splats,
    device: Device,
    adapter: Adapter,
    output: Option<Tensor<3>>,
    packed: bool,
    splat_scale: f32,
}
impl Brush {
    pub async fn new(scene: &PlyScene, settings: &RenderSettings) -> Self {
        use burn::backend::wgpu::{
            RuntimeOptions, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async,
        };
        static ADAPTER: tokio::sync::OnceCell<Adapter> = tokio::sync::OnceCell::const_new();
        let adapter = ADAPTER
            .get_or_init(|| async {
                let setup = init_setup_async::<AutoGraphicsApi>(
                    &WgpuDevice::default(),
                    RuntimeOptions {
                        tasks_max: 64,
                        memory_config: burn::backend::wgpu::MemoryConfiguration::ExclusivePages,
                    },
                )
                .await;
                setup.adapter.get_info().into()
            })
            .await
            .clone();
        let device = burn::backend::wgpu::WgpuDevice::default().into();
        let mode = match settings.mode(scene.mode) {
            gsplat_core::RenderMode::Default => SplatRenderMode::Default,
            gsplat_core::RenderMode::Mip => SplatRenderMode::Mip,
        };
        let mut splats = scene.data.clone().into_splats(&device, mode);
        if let Some(floor) = settings.min_scale {
            splats = splats.with_min_scale(Tensor::from_data(
                burn::tensor::TensorData::new(
                    vec![floor; scene.data.num_splats()],
                    [scene.data.num_splats()],
                ),
                &device,
            ));
        }
        Self {
            splats,
            splat_scale: settings.splat_scale,
            device,
            adapter,
            output: None,
            packed: false,
        }
    }

    pub async fn render(&mut self, camera: &CameraSpec, parity: bool) -> Result<Counts> {
        let (output, aux) = brush_render::render_splats(
            self.splats.clone(),
            &camera.brush_camera(),
            glam::uvec2(camera.width, camera.height),
            Vec3::ZERO,
            Some(self.splat_scale),
            if parity {
                TextureMode::Float
            } else {
                TextureMode::Packed
            },
        )
        .await;
        self.output = Some(output);
        self.packed = !parity;
        Ok(Counts {
            visible: Some(aux.num_visible),
            intersections: Some(aux.num_intersections),
            overflow_events: Some(0),
        })
    }
    pub fn finish(&self) -> Result<()> {
        self.device.sync().map_err(|e| anyhow::anyhow!("{e}"))
    }
    pub async fn read_rgba_f32(&mut self) -> Result<Vec<f32>> {
        let values = self
            .output
            .as_ref()
            .context("render before readback")?
            .clone()
            .into_data_async()
            .await?
            .try_into_vec::<f32>()?;
        Ok(if self.packed {
            values
                .into_iter()
                .flat_map(|v| v.to_bits().to_le_bytes().map(|x| x as f32 / 255.0))
                .collect()
        } else {
            values
        })
    }
    pub fn adapter(&self) -> Adapter {
        self.adapter.clone()
    }
}

/// Construct the shared renderer with the benchmark's recorded controls.
pub async fn ours(
    scene: &PlyScene,
    size: glam::UVec2,
    settings: &RenderSettings,
    archetype: Option<&Path>,
    background: Vec3,
) -> Result<crate::Renderer> {
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
    crate::Renderer::new(
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

struct Decoded {
    centers: Vec<[f32; 3]>,
    scales: Vec<[f32; 3]>,
    quaternions: Vec<[f32; 4]>,
    colors: Vec<u32>,
    sh: Vec<[[half::f16; 3]; 15]>,
    degree: u32,
}
impl Decoded {
    fn native(&self) -> gsplat_core::native::NativeSplats<'_> {
        gsplat_core::native::NativeSplats {
            centers: &self.centers,
            scales: &self.scales,
            quaternions: &self.quaternions,
            colors: &self.colors,
            sh: &self.sh,
            degree: self.degree,
        }
    }
}
fn column<C: re_types_core::FromArrow>(
    batch: &Option<re_sdk_types::SerializedComponentBatch>,
) -> Result<Vec<C>> {
    batch
        .as_ref()
        .map(|b| C::from_arrow(&b.array))
        .transpose()
        .map(|v| v.unwrap_or_default())
        .map_err(|e| anyhow::anyhow!("{e}"))
}
fn decode(native: &re_sdk_types::archetypes::GaussianSplats3D) -> Result<Decoded> {
    use re_sdk_types::components as c;
    if native.centers.is_none() {
        anyhow::bail!("native splats have no centers");
    }
    Ok(Decoded {
        centers: column::<c::Position3D>(&native.centers)?
            .into_iter()
            .map(|v| v.0.0)
            .collect(),
        scales: column::<c::Scale3D>(&native.scales)?
            .into_iter()
            .map(|v| v.0.0)
            .collect(),
        quaternions: column::<c::RotationQuat>(&native.quaternions)?
            .into_iter()
            .map(|v| v.0.0)
            .collect(),
        colors: column::<c::Color>(&native.colors)?
            .into_iter()
            .map(|v| u32::from_be_bytes(v.to_array()))
            .collect(),
        sh: column::<c::SphericalHarmonics3Rgb>(&native.sh_coefficients)?
            .into_iter()
            .map(|v| v.0.0)
            .collect(),
        degree: column::<c::SphericalHarmonicsDegree>(&native.spherical_harmonics_degree)?
            .first()
            .map_or(3, |v| v.0.0),
    })
}
