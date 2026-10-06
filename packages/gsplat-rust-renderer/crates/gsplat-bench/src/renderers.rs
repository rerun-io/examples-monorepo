//! Loading and pixel transfers are separate from the synchronized speed lane.
use crate::settings::RenderSettings;
use crate::{Error, Result, camera::CameraSpec, gpu};
use brush_render::{
    TextureMode,
    gaussian_splats::{SplatRenderMode, Splats},
};
use burn::tensor::{Device, Tensor};
use glam::Vec3;
use serde::{Deserialize, Serialize};
mod archetype;
mod native;
pub use archetype::archetype_splats;
pub use native::Native;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Implementation {
    Brush,
    Ours,
    OursArchetype,
    Native,
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
    /// Encode and submit one camera; retain output only when parity is requested.
    async fn render(&mut self, camera: &CameraSpec, parity: bool) -> Result<Counts>;
    /// Wait for completion without reading pixels.
    fn finish(&self) -> Result<()>;
    async fn read_rgba_f32(&mut self) -> Result<Vec<f32>>;
    fn adapter(&self) -> Adapter;
    async fn stages(&mut self, _camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
        Ok(None)
    }
}

pub use gsplat_render::Scene;

pub struct Brush {
    splats: Splats,
    device: Device,
    adapter: Adapter,
    output: Option<Tensor<3>>,
    packed: bool,
    splat_scale: f32,
}
impl Brush {
    pub async fn new(scene: &Scene, settings: &RenderSettings) -> Self {
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
impl RenderEngine for gsplat_render::Renderer {
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
        let stats = gsplat_render::Renderer::render(
            self,
            camera,
            if parity {
                gsplat_render::Output::Float
            } else {
                gsplat_render::Output::Packed
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
        Ok(self.read_rgba(gsplat_render::Output::Float)?)
    }
    fn adapter(&self) -> Adapter {
        self.adapter.clone().into()
    }
}

/// Construct the shared renderer with the benchmark's recorded controls.
pub async fn ours(
    scene: &Scene,
    width: u32,
    height: u32,
    settings: &RenderSettings,
) -> Result<gsplat_render::Renderer> {
    let mut splats = gsplat_render::raw_splats(&scene.data)?;
    if let Some(floor) = settings.min_scale {
        splats.min_scale = Some(vec![floor; scene.data.num_splats()]);
    }
    let mut renderer = gsplat_render::Renderer::new(
        &splats,
        settings.mode(scene.mode),
        glam::UVec2::new(width, height),
        settings.initial_capacity,
    )
    .await?;
    renderer.options = settings.options();
    Ok(renderer)
}

#[cfg(test)]
mod coverage;
