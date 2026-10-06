//! Loading and pixel transfers are separate from the synchronized speed lane.
use crate::settings::RenderSettings;
use crate::{
    Error, Result,
    camera::{CameraModel, CameraSpec, opengl_to_opencv},
    gpu, wait,
};
use brush_render::{
    TextureMode,
    gaussian_splats::{SplatRenderMode, Splats},
};
use burn::tensor::{Device, Tensor};
use glam::{Mat4, Vec2, Vec3};
use gsplat_lib::gsplat_core::{
    CameraApproximation, GpuContext, GpuRenderResources, GpuRenderer, RenderGaussianCloud,
    RenderShCoefficients,
};
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
    OursOld,
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

pub struct Old {
    ctx: GpuContext,
    renderer: GpuRenderer,
    resources: GpuRenderResources,
    width: u32,
    height: u32,
}
impl Old {
    pub fn new(scene: &Scene, width: u32, height: u32) -> Result<Self> {
        let data = &scene.data;
        let n = data.num_splats();
        let means = data
            .means
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| Vec3::from_array(*v))
            .collect();
        let rotations = data
            .rotations
            .clone()
            .unwrap_or_else(|| [1.0, 0.0, 0.0, 0.0].repeat(n));
        let rotations = rotations
            .as_chunks::<4>()
            .0
            .iter()
            .map(|q| {
                gsplat_lib::gsplat_core::normalize_quat_or_identity(glam::Quat::from_xyzw(
                    q[1], q[2], q[3], q[0],
                ))
            })
            .collect();
        let scales = data
            .log_scales
            .clone()
            .unwrap_or_else(|| vec![-4.0; n * 3])
            .as_chunks::<3>()
            .0
            .iter()
            .map(|s| Vec3::new(s[0].exp(), s[1].exp(), s[2].exp()).max(Vec3::splat(1e-6)))
            .collect();
        let opacities = data
            .raw_opacities
            .clone()
            .unwrap_or_else(|| vec![0.0; n])
            .iter()
            .map(|v| (1.0 / (1.0 + (-v).exp())).clamp(0.0, 1.0))
            .collect();
        let coefficients = data.sh_coeffs.clone().unwrap_or_else(|| vec![0.5; n * 3]);
        let count = coefficients.len() / n / 3;
        let colors = coefficients
            .chunks_exact(count * 3)
            .map(|c| {
                std::array::from_fn(|i| (c[i] * gsplat_lib::gsplat_core::SH_C0 + 0.5).max(0.0))
            })
            .collect();
        let cloud = RenderGaussianCloud::from_raw(
            means,
            rotations,
            scales,
            opacities,
            colors,
            Some(RenderShCoefficients {
                coeffs_per_channel: count,
                coefficients: coefficients.into(),
            }),
        );
        let ctx = GpuContext::new().map_err(gpu)?;
        let renderer = GpuRenderer::new(&ctx.device);
        let resources = GpuRenderResources::new(
            &ctx.device,
            &renderer,
            &cloud,
            Vec2::new(width as f32, height as f32),
        );
        Ok(Self {
            ctx,
            renderer,
            resources,
            width,
            height,
        })
    }
    fn camera(&self, c: &CameraSpec) -> Result<CameraApproximation> {
        if !matches!(c.model, CameraModel::Pinhole) {
            return Err(Error::Unsupported("ours-old lens model".into()));
        }
        if (c.width, c.height) != (self.width, self.height) {
            return Err(Error::Invalid("renderer resolution changed".into()));
        }
        let view = opengl_to_opencv(c.pose()).inverse();
        let mut projection = Mat4::perspective_rh(c.vertical_fov(), c.aspect(), 0.01, 10000.0);
        projection.z_axis.x = 1.0 - 2.0 * c.cx / c.width as f32;
        projection.z_axis.y = 2.0 * c.cy / c.height as f32 - 1.0;
        Ok(CameraApproximation {
            view_from_world: glam::Affine3A::from_mat4(view),
            projection_from_view: projection,
            world_position: c.pose().w_axis.truncate(),
            viewport_size_px: Vec2::new(c.width as f32, c.height as f32),
            near_plane: 0.01,
        })
    }
    /// A separate diagnostic frame; no query resolution/readback enters lane 2.
    pub fn stage_ms(&self, camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
        if !self
            .ctx
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return Ok(None);
        }
        let query = self.ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("stages"),
            ty: wgpu::QueryType::Timestamp,
            count: (2 * gsplat_lib::gsplat_core::gpu_renderer::STAGE_NAMES.len()) as u32,
        });
        self.resources.render_gpu(
            &self.ctx,
            &self.renderer,
            &self.camera(camera)?,
            Some(&query),
        );
        let resolve = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamps"),
            size: (16 * gsplat_lib::gsplat_core::gpu_renderer::STAGE_NAMES.len()) as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp readback"),
            size: (16 * gsplat_lib::gsplat_core::gpu_renderer::STAGE_NAMES.len()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.ctx.device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(&query, 0..query.count(), &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, resolve.size());
        self.ctx.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.finish()?;
        rx.recv().map_err(gpu)?.map_err(gpu)?;
        let mapped = read.slice(..).get_mapped_range().map_err(gpu)?;
        let timestamps: &[u64] = bytemuck::cast_slice(&mapped);
        let period = self.ctx.queue.get_timestamp_period() as f64 / 1e6;
        let names = gsplat_lib::gsplat_core::gpu_renderer::STAGE_NAMES;
        Ok(Some(
            names
                .iter()
                .enumerate()
                .map(|(i, name)| StageTiming {
                    name: (*name).into(),
                    ms: timestamps[2 * i + 1].saturating_sub(timestamps[2 * i]) as f64 * period,
                })
                .collect(),
        ))
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageTiming {
    pub name: String,
    pub ms: f64,
}
impl RenderEngine for Old {
    async fn stages(&mut self, camera: &CameraSpec) -> Result<Option<Vec<StageTiming>>> {
        self.stage_ms(camera)
    }
    async fn render(&mut self, c: &CameraSpec, _parity: bool) -> Result<Counts> {
        self.resources
            .render_gpu(&self.ctx, &self.renderer, &self.camera(c)?, None);
        Ok(Counts::default())
    }
    fn finish(&self) -> Result<()> {
        wait(&self.ctx.device)
    }
    async fn read_rgba_f32(&mut self) -> Result<Vec<f32>> {
        self.resources
            .check_intersection_capacity(&self.ctx)
            .map_err(Error::Invalid)?;
        Ok(self
            .resources
            .read_output(&self.ctx)
            .pixels
            .into_iter()
            .map(|x| x as f32 / 255.0)
            .collect())
    }
    fn adapter(&self) -> Adapter {
        self.ctx.adapter_info.clone().into()
    }
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
