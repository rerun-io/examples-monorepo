//! Loading and pixel transfers are separate from the synchronized speed lane.
use crate::camera::CameraSpec;
use crate::settings::RenderSettings;
use crate::{Error, Result, gpu};
use brush_render::{
    TextureMode,
    gaussian_splats::{SplatRenderMode, Splats},
};
use burn::tensor::{Device, Tensor};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::path::Path;
mod archetype;
mod native;
pub use archetype::archetype_splats;
pub use native::Native;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
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
            return Err(Error::Unsupported(
                "render-mode, scale, or floor controls for this reference renderer".into(),
            ));
        }
        match kind {
            Implementation::Brush => Ok(Self::Brush(Box::new(Brush::new(scene, settings).await))),
            Implementation::Native => Ok(Self::Native(Box::new(
                Native::new(ply, scene.data.num_splats()).await?,
            ))),
            Implementation::Ours => {
                let renderer = if let Some(path) = archetype {
                    let mut splats = archetype_splats(path)?;
                    if splats.transforms.len() != scene.data.num_splats() {
                        return Err(Error::Invalid("PLY loaders disagree on count".into()));
                    }
                    splats.min_scale = settings
                        .min_scale
                        .map(|floor| vec![floor; splats.transforms.len()]);
                    crate::Renderer::new(
                        &splats,
                        settings.options(scene.mode),
                        glam::uvec2(camera.width, camera.height),
                        settings.initial_capacity,
                    )
                    .await?
                } else {
                    ours(scene, camera.width, camera.height, settings).await?
                };
                Ok(Self::Ours(Box::new(renderer)))
            }
        }
    }
    pub fn output_format(&self) -> &'static str {
        match self {
            Self::Brush(_) => "Brush Packed",
            Self::Ours(_) => "Packed RGBA8",
            Self::Native(_) => "Rerun RGBA8UnormSrgb/MSAA4, opaque black background",
        }
    }
}
impl RenderEngine for Engine {
    fn capture_next_frame(&mut self) {
        match self {
            Self::Brush(r) => r.capture_next_frame(),
            Self::Ours(r) => r.capture_next_frame(),
            Self::Native(r) => r.capture_next_frame(),
        }
    }
    async fn render(&mut self, camera: &CameraSpec, parity: bool) -> Result<Counts> {
        match self {
            Self::Brush(r) => RenderEngine::render(r.as_mut(), camera, parity).await,
            Self::Ours(r) => RenderEngine::render(r.as_mut(), camera, parity).await,
            Self::Native(r) => RenderEngine::render(r.as_mut(), camera, parity).await,
        }
    }
    fn finish(&self) -> Result<()> {
        match self {
            Self::Brush(r) => RenderEngine::finish(r.as_ref()),
            Self::Ours(r) => RenderEngine::finish(r.as_ref()),
            Self::Native(r) => RenderEngine::finish(r.as_ref()),
        }
    }
    async fn read_rgba_f32(&mut self) -> Result<Vec<f32>> {
        match self {
            Self::Brush(r) => r.read_rgba_f32().await,
            Self::Ours(r) => r.read_rgba_f32().await,
            Self::Native(r) => r.read_rgba_f32().await,
        }
    }
    fn adapter(&self) -> Adapter {
        match self {
            Self::Brush(r) => RenderEngine::adapter(r.as_ref()),
            Self::Ours(r) => RenderEngine::adapter(r.as_ref()),
            Self::Native(r) => RenderEngine::adapter(r.as_ref()),
        }
    }
    async fn stages(&mut self, camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
        match self {
            Self::Brush(r) => r.stages(camera).await,
            Self::Ours(r) => r.stages(camera).await,
            Self::Native(r) => r.stages(camera).await,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub visible: Option<u32>,
    pub intersections: Option<u32>,
    #[serde(default)]
    pub overflow_events: Option<u32>,
}

#[allow(async_fn_in_trait)]
pub trait RenderEngine {
    /// Encode and submit one camera. Parity selects float for compute/Brush
    /// and transparent RGBA8 for native; false selects the timed output.
    async fn render(&mut self, camera: &CameraSpec, parity: bool) -> Result<Counts>;
    /// Request capture of the next frame without changing its render settings.
    fn capture_next_frame(&mut self) {}
    /// Wait for completion without reading pixels.
    fn finish(&self) -> Result<()>;
    async fn read_rgba_f32(&mut self) -> Result<Vec<f32>>;
    fn adapter(&self) -> Adapter;
    async fn stages(&mut self, _camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
        Ok(None)
    }
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
}
impl RenderEngine for Brush {
    async fn render(&mut self, camera: &CameraSpec, parity: bool) -> Result<Counts> {
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
    fn finish(&self) -> Result<()> {
        self.device.sync().map_err(gpu)
    }
    async fn read_rgba_f32(&mut self) -> Result<Vec<f32>> {
        let values = self
            .output
            .as_ref()
            .ok_or_else(|| Error::Invalid("render before readback".into()))?
            .clone()
            .into_data_async()
            .await
            .map_err(gpu)?
            .try_into_vec::<f32>()
            .map_err(gpu)?;
        Ok(if self.packed {
            values
                .into_iter()
                .flat_map(|v| v.to_bits().to_le_bytes().map(|x| x as f32 / 255.0))
                .collect()
        } else {
            values
        })
    }
    async fn stages(&mut self, camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
        use burn::cubecl::{Device as CubeDevice, wgpu::WgpuDevice};
        use burn::tensor::TimingMethod;
        self.finish()?;
        let client = CubeDevice::Wgpu(WgpuDevice::default()).client();
        let window = client.profile_start().map_err(gpu)?;
        if let Err(error) = self.render(camera, false).await {
            client.profile_abandon(window);
            return Err(error);
        }
        let duration = client.profile_end(window).map_err(gpu)?;
        if duration.timing_method() != TimingMethod::Device {
            return Err(Error::Unsupported(
                "Brush device timestamps unavailable".into(),
            ));
        }
        let ticks = duration
            .resolve()
            .await
            .ok_or_else(|| Error::Gpu("empty Brush profile window".into()))?;
        Ok(Some(vec![StageTiming {
            name: "Brush GPU device window".into(),
            ms: ticks.duration().as_secs_f64() * 1000.0,
        }]))
    }
    fn adapter(&self) -> Adapter {
        self.adapter.clone()
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageTiming {
    pub name: String,
    pub ms: f64,
}
impl RenderEngine for crate::Renderer {
    async fn stages(&mut self, camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
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

    async fn render(&mut self, camera: &CameraSpec, parity: bool) -> Result<Counts> {
        let stats = crate::Renderer::render(
            self,
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
    fn finish(&self) -> Result<()> {
        Ok(())
    }
    async fn read_rgba_f32(&mut self) -> Result<Vec<f32>> {
        self.read_rgba()
    }
    fn adapter(&self) -> Adapter {
        self.adapter_info().clone().into()
    }
}

/// Construct the shared renderer with the benchmark's recorded controls.
pub async fn ours(
    scene: &PlyScene,
    width: u32,
    height: u32,
    settings: &RenderSettings,
) -> Result<crate::Renderer> {
    let mut splats = crate::raw_splats(&scene.data)?;
    if let Some(floor) = settings.min_scale {
        splats.min_scale = Some(vec![floor; scene.data.num_splats()]);
    }
    crate::Renderer::new(
        &splats,
        settings.options(scene.mode),
        glam::UVec2::new(width, height),
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
fn column<C: re_types_core::FromArrow>(
    batch: &Option<re_sdk_types::SerializedComponentBatch>,
) -> Result<Vec<C>> {
    batch
        .as_ref()
        .map(|b| C::from_arrow(&b.array))
        .transpose()
        .map(|v| v.unwrap_or_default())
        .map_err(|e| Error::Invalid(e.to_string()))
}
fn decode(native: &re_sdk_types::archetypes::GaussianSplats3D) -> Result<Decoded> {
    use re_sdk_types::components as c;
    if native.centers.is_none() {
        return Err(Error::Invalid("native splats have no centers".into()));
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
