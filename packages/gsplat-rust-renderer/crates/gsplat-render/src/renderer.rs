//! PLY metadata and raw f32 uploads at the application boundary.
use crate::{Error, Result, camera::CameraSpec};
use brush_serde::import::SplatData;
use glam::{UVec2, Vec3};
use gsplat_core::{RenderMode, RenderOptions, Splats, Target};
use std::path::Path;

pub struct Scene {
    pub data: SplatData,
    pub mode: RenderMode,
    pub center: Vec3,
    pub extent: f32,
}
impl Scene {
    pub async fn load(path: &Path) -> Result<Self> {
        let loaded =
            brush_serde::import::load_splat_from_ply(tokio::fs::File::open(path).await?, None)
                .await
                .map_err(|e| Error::Invalid(e.to_string()))?;
        if loaded.data.num_splats() == 0 {
            return Err(Error::Invalid("empty splat cloud".into()));
        }
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for p in loaded.data.means.as_chunks::<3>().0 {
            let p = Vec3::from_array(*p);
            if !p.is_finite() {
                return Err(Error::Invalid("nonfinite splat center".into()));
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
        return Err(Error::Invalid("inconsistent PLY parameter lengths".into()));
    }
    let sh = data.sh_coeffs.clone().unwrap_or_else(|| vec![0.5; n * 3]);
    let count = sh.len() / n / 3;
    let root = (count as f64).sqrt() as usize;
    if root == 0 || root > 5 || root * root != count || sh.len() != n * count * 3 {
        return Err(Error::Invalid("invalid SH coefficient dimensions".into()));
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
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::AdapterInfo,
    pub core: gsplat_core::Renderer,
    pub view: gsplat_core::ViewState,
    pub options: RenderOptions,
    float: wgpu::Buffer,
    packed: wgpu::Buffer,
    size: UVec2,
}
impl Renderer {
    pub async fn new(
        splats: &Splats,
        mode: RenderMode,
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
            .await
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let features = adapter.features();
        if !features.contains(wgpu::Features::SUBGROUP) {
            return Err(gsplat_core::Error::Capabilities.into());
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP
                    | (features & wgpu::Features::TIMESTAMP_QUERY),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: adapter
                        .limits()
                        .max_storage_buffer_binding_size,
                    max_buffer_size: adapter.limits().max_buffer_size,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let core = gsplat_core::Renderer::new(&device, &queue)?;
        let scene = core.upload(splats, mode)?;
        let view = core.create_view(&scene, initial_capacity)?;
        let bytes = u64::from(size.x) * u64::from(size.y) * 16;
        if bytes == 0 || bytes > device.limits().max_storage_buffer_binding_size {
            return Err(Error::Invalid(
                "output dimensions exceed GPU buffer limits".into(),
            ));
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
            options: RenderOptions::default(),
            float,
            packed,
            size,
        })
    }
    pub fn render(
        &mut self,
        camera: &CameraSpec,
        output: Output,
    ) -> Result<gsplat_core::FrameStats> {
        camera.validate()?;
        if UVec2::new(camera.width, camera.height) != self.size {
            return Err(Error::Invalid("renderer resolution changed".into()));
        }
        loop {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.core.render(
                &mut encoder,
                &mut self.view,
                &camera.core_camera(),
                &self.options,
                match output {
                    Output::Float => Target::Float(&self.float),
                    Output::Packed => Target::Packed(&self.packed),
                },
            )?;
            self.queue.submit([encoder.finish()]);
            self.finish()?;
            let feedback = self
                .view
                .poll_feedback()?
                .ok_or_else(|| Error::Gpu("completed frame has no count feedback".into()))?;
            if !feedback.needs_rerender {
                return Ok(feedback);
            }
        }
    }
    pub fn finish(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(e.to_string()))?;
        Ok(())
    }
    pub fn read_rgba(&self, output: Output) -> Result<Vec<f32>> {
        let target = if matches!(output, Output::Float) {
            &self.float
        } else {
            &self.packed
        };
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel readback"),
            size: target.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(target, 0, &staging, 0, target.size());
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.finish()?;
        rx.recv()
            .map_err(|e| Error::Gpu(e.to_string()))?
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let data = staging
            .get_mapped_range(..)
            .map_err(|e| Error::Gpu(e.to_string()))?;
        Ok(if matches!(output, Output::Float) {
            bytemuck::cast_slice(&data).to_vec()
        } else {
            data.iter().map(|x| f32::from(*x) / 255.0).collect()
        })
    }
}
