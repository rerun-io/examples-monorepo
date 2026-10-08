//! PLY metadata and raw f32 uploads at the application boundary.
use crate::{Result, camera::CameraSpec};
use anyhow::Context as _;
use brush_serde::import::SplatData;
use glam::{UVec2, Vec3};
use gsplat_core::{RenderMode, RenderOptions, Splats, Target};
use std::path::Path;

pub struct PlyScene {
    pub data: SplatData,
    pub mode: RenderMode,
    pub center: Vec3,
    pub extent: f32,
}
impl PlyScene {
    pub async fn load(path: &Path) -> Result<Self> {
        let loaded =
            brush_serde::import::load_splat_from_ply(tokio::fs::File::open(path).await?, None)
                .await?;
        if loaded.data.num_splats() == 0 {
            anyhow::bail!("empty splat cloud");
        }
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for p in loaded.data.means.as_chunks::<3>().0 {
            let p = Vec3::from_array(*p);
            if !p.is_finite() {
                anyhow::bail!("nonfinite splat center");
            }
            min = min.min(p);
            max = max.max(p);
        }
        Ok(Self {
            data: loaded.data,
            mode: if matches!(
                loaded.meta.render_mode,
                Some(brush_render::gaussian_splats::SplatRenderMode::Mip)
            ) {
                RenderMode::Mip
            } else {
                RenderMode::Default
            },
            center: (min + max) * 0.5,
            extent: (max - min).length() * 0.5,
        })
    }
}

/// Preserve Brush's raw parameter defaults and coefficient order; no half conversion.
pub fn raw_splats(data: &SplatData) -> Result<Splats> {
    let n = data.num_splats();
    if n == 0
        || data.means.len() != n * 3
        || data.rotations.as_ref().is_some_and(|x| x.len() != n * 4)
        || data.log_scales.as_ref().is_some_and(|x| x.len() != n * 3)
        || data.raw_opacities.as_ref().is_some_and(|x| x.len() != n)
    {
        anyhow::bail!("inconsistent PLY parameter lengths");
    }
    let sh = data.sh_coeffs.clone().unwrap_or_else(|| vec![0.5; n * 3]);
    let count = sh.len() / n / 3;
    let root = (count as f64).sqrt() as usize;
    if root == 0 || root > 5 || root * root != count || sh.len() != n * count * 3 {
        anyhow::bail!("invalid SH coefficient dimensions");
    }
    let transforms = (0..n)
        .map(|i| {
            let mut row = [0.0; 10];
            row[..3].copy_from_slice(&data.means[i * 3..i * 3 + 3]);
            row[3..7].copy_from_slice(
                data.rotations
                    .as_ref()
                    .map_or(&[1.0, 0.0, 0.0, 0.0], |v| &v[i * 4..i * 4 + 4]),
            );
            row[7..].copy_from_slice(
                data.log_scales
                    .as_ref()
                    .map_or(&[-4.0; 3], |v| &v[i * 3..i * 3 + 3]),
            );
            row
        })
        .collect();
    Ok(Splats {
        transforms,
        raw_opacities: data.raw_opacities.clone().unwrap_or_else(|| vec![0.0; n]),
        sh_coefficients: sh.as_chunks::<3>().0.to_vec(),
        sh_degree: root as u32 - 1,
        min_scale: None,
    })
}

/// Persistent upload and targets; rendering completion is separate from pixel transfer.
#[derive(Clone, Copy, Debug)]
pub enum Output {
    Float,
    Packed,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::AdapterInfo,
    core: gsplat_core::Renderer,
    view: gsplat_core::ViewState,
    options: RenderOptions,
    float: wgpu::Buffer,
    packed: wgpu::Buffer,
    size: UVec2,
    last_output: Option<Output>,
}
impl Renderer {
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter
    }
    pub async fn new(
        splats: &Splats,
        options: RenderOptions,
        size: UVec2,
        initial_capacity: u32,
    ) -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;
        gsplat_core::check_adapter(adapter.features(), &adapter.limits())?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: gsplat_core::required_features(&adapter),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await?;
        let core = gsplat_core::Renderer::new(&device)?;
        let scene = core.upload(splats)?;
        let view = core.create_view(&scene, initial_capacity)?;
        let bytes = u64::from(size.x) * u64::from(size.y) * 16;
        if bytes == 0 || bytes > device.limits().max_storage_buffer_binding_size {
            anyhow::bail!("output dimensions exceed GPU buffer limits");
        }
        let float = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("float target"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let packed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("packed target"),
            size: bytes / 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(Self {
            device,
            queue,
            adapter: adapter.get_info(),
            core,
            view,
            options,
            float,
            packed,
            size,
            last_output: None,
        })
    }
    pub fn render(
        &mut self,
        camera: &CameraSpec,
        output: Output,
    ) -> Result<gsplat_core::FrameStats> {
        camera.validate()?;
        if UVec2::new(camera.width, camera.height) != self.size {
            anyhow::bail!("renderer resolution changed");
        }
        loop {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.core.render(
                &self.queue,
                &mut encoder,
                &mut self.view,
                &camera.core_camera(),
                &self.options,
                match output {
                    Output::Float => Target::Float(self.float.clone()),
                    Output::Packed => Target::Packed(self.packed.clone()),
                },
            )?;
            self.queue.submit([encoder.finish()]);
            self.finish()?;
            let feedback = self
                .view
                .poll_feedback()?
                .context("completed frame has no count feedback")?;
            if !feedback.needs_rerender {
                self.last_output = Some(output);
                return Ok(feedback);
            }
        }
    }
    pub fn finish(&self) -> Result<()> {
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }
    /// Read the target written by the last successful render.
    pub fn read_rgba(&self) -> Result<Vec<f32>> {
        let output = self.last_output.context("render before readback")?;
        let target = if matches!(output, Output::Float) {
            &self.float
        } else {
            &self.packed
        };
        let encoder = self.device.create_command_encoder(&Default::default());
        let data = self.read_buffer(encoder, target)?;
        Ok(if matches!(output, Output::Float) {
            bytemuck::cast_slice(&data).to_vec()
        } else {
            data.iter().map(|x| f32::from(*x) / 255.0).collect()
        })
    }
    fn read_buffer(
        &self,
        mut encoder: wgpu::CommandEncoder,
        target: &wgpu::Buffer,
    ) -> Result<wgpu::BufferView> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gsplat readback"),
            size: target.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(target, 0, &staging, 0, target.size());
        let (tx, rx) = std::sync::mpsc::channel();
        encoder.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
            let _ = tx.send(result);
        });
        self.queue.submit([encoder.finish()]);
        self.finish()?;
        rx.recv()??;
        staging
            .get_mapped_range(..)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}
