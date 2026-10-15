//! Brush renderer oracle backend.
use super::{Adapter, Counts, StageTiming};
use crate::{PlyScene, camera::CameraSpec, settings::RenderSettings};
use anyhow::{Context as _, Result};
use brush_render::{
    TextureMode,
    gaussian_splats::{SplatRenderMode, Splats},
};
use burn::tensor::{Device, Tensor};
use glam::Vec3;

pub struct Brush {
    pub(super) splats: Splats,
    pub(super) device: Device,
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
    pub async fn stages(&mut self, camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
        use burn::cubecl::{Device as CubeDevice, wgpu::WgpuDevice};
        use burn::tensor::TimingMethod;
        self.finish()?;
        let client = CubeDevice::Wgpu(WgpuDevice::default()).client();
        let window = client.profile_start()?;
        if let Err(error) = self.render(camera, false).await {
            client.profile_abandon(window);
            return Err(error);
        }
        let duration = client.profile_end(window)?;
        if duration.timing_method() != TimingMethod::Device {
            return Err(crate::Unsupported("Brush device timestamps unavailable".into()).into());
        }
        let ticks = duration
            .resolve()
            .await
            .context("empty Brush profile window")?;
        Ok(Some(vec![StageTiming {
            name: "Brush GPU device window".into(),
            ms: ticks.duration().as_secs_f64() * 1000.0,
        }]))
    }
    pub fn adapter(&self) -> Adapter {
        self.adapter.clone()
    }
}
