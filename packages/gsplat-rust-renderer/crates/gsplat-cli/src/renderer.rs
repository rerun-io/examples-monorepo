//! PLY metadata and raw f32 uploads at the application boundary.
use crate::camera::CameraSpec;
use anyhow::Context as _;
use anyhow::Result;
use brush_serde::import::SplatData;
use glam::{UVec2, Vec3};
use gsplat_core::{Hexels, RenderMode, RenderOptions, Splats, Target};
use std::path::Path;

/// A splat or hexel PLY. For hexels, `data` holds only the centres used for framing.
pub struct PlyScene {
    pub data: SplatData,
    pub hexels: Option<Hexels>,
    /// Image size of a 2D hexel fit.
    pub image_size: Option<UVec2>,
    pub mode: RenderMode,
    pub center: Vec3,
    pub extent: f32,
}

impl PlyScene {
    pub async fn load(path: &Path) -> Result<Self> {
        let bytes = tokio::fs::read(path).await?;
        let mut image_size = None;
        let (data, hexels, mode) = if crate::hexel_ply::is_hexel_ply(&bytes) {
            let parsed = crate::hexel_ply::parse(&bytes)?;
            let hexels = parsed.hexels;
            image_size = parsed.image_size;
            // 2D centres sit on the z = 0 plane.
            let planar = hexels.variant == gsplat_core::HexelVariant::Image;
            let means = hexels
                .params
                .chunks_exact(hexels.variant.stride())
                .flat_map(|h| [h[0], h[1], if planar { 0.0 } else { h[2] }])
                .collect();
            let data = SplatData {
                means,
                rotations: None,
                log_scales: None,
                sh_coeffs: None,
                raw_opacities: None,
            };
            (data, Some(hexels), RenderMode::Default)
        } else {
            let loaded =
                brush_serde::import::load_splat_from_ply(std::io::Cursor::new(bytes), None).await?;
            let mode = if matches!(
                loaded.meta.render_mode,
                Some(brush_render::gaussian_splats::SplatRenderMode::Mip)
            ) {
                RenderMode::Mip
            } else {
                RenderMode::Default
            };
            (loaded.data, None, mode)
        };
        if data.num_splats() == 0 {
            anyhow::bail!("empty splat cloud");
        }
        let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for p in data.means.as_chunks::<3>().0 {
            let p = Vec3::from_array(*p);
            if !p.is_finite() {
                anyhow::bail!("nonfinite splat center");
            }
            min = min.min(p);
            max = max.max(p);
        }
        Ok(Self {
            data,
            hexels,
            image_size,
            mode,
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
    let sh = data.sh_coeffs.as_deref();
    let count = sh.map_or(1, |sh| sh.len() / n / 3);
    let root = (count as f64).sqrt() as usize;
    if root == 0
        || root > 5
        || root * root != count
        || sh.is_some_and(|sh| sh.len() != n * count * 3)
    {
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
        sh_coefficients: sh.map_or_else(|| vec![[0.5; 3]; n], |sh| sh.as_chunks::<3>().0.to_vec()),
        sh_degree: root as u32 - 1,
        min_scale: None,
    })
}

/// Persistent upload and targets; rendering completion is separate from pixel transfer.
#[derive(Clone, Copy, Debug)]
pub enum Output {
    Float,
    Packed,
    /// Float colour plus depth and normal buffers (hexel scenes).
    Surface,
}

/// The primitives a standalone renderer uploads.
#[derive(Clone, Copy)]
pub enum Primitives<'a> {
    Splats(&'a Splats),
    Hexels(&'a Hexels),
}

/// Per-pixel surface outputs: expected and median depth, then the world normal (w = 0).
pub struct Surface {
    pub depth: Vec<[f32; 2]>,
    pub normal: Vec<[f32; 4]>,
}

pub struct StandaloneRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::AdapterInfo,
    core: gsplat_core::Renderer,
    view: gsplat_core::ViewState,
    options: RenderOptions,
    float: wgpu::Buffer,
    packed: wgpu::Buffer,
    surface: Option<(wgpu::Buffer, wgpu::Buffer)>,
    size: UVec2,
    last_output: Option<Output>,
}

impl StandaloneRenderer {
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter
    }
    /// GPU stage durations from a separate packed-output diagnostic frame.
    /// Pixel transfers and timestamp resolution never enter the benchmark loop.
    pub fn stage_ms(&mut self, camera: &CameraSpec) -> Result<Option<[f64; 8]>> {
        if self.adapter.backend == wgpu::Backend::Metal
            || !self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return Ok(None);
        }
        let query = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gsplat stage times"),
            ty: wgpu::QueryType::Timestamp,
            count: gsplat_core::QUERY_COUNT,
        });
        self.view.set_timestamp_queries(Some(query.clone()))?;
        let rendered = self.render(camera, Output::Packed);
        self.view.set_timestamp_queries(None)?;
        rendered?;
        // Metal counter samples must complete before a later blit resolves them.
        // This extra synchronization is outside the measured frame-time loop.
        self.finish()?;
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stage timestamps"),
            size: u64::from(gsplat_core::QUERY_COUNT) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(&query, 0..gsplat_core::QUERY_COUNT, &resolve, 0);
        let data = self.read_buffer(encoder, &resolve)?;
        let ticks: &[u64] = bytemuck::cast_slice(&data);
        Ok(Some(gsplat_core::stage_ms(
            ticks,
            self.queue.get_timestamp_period(),
        )?))
    }
    pub async fn new(
        primitives: Primitives<'_>,
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
        let scene = match primitives {
            Primitives::Splats(splats) => core.upload(splats)?,
            Primitives::Hexels(hexels) => core.upload_hexels(hexels)?,
        };
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
            surface: None,
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
        if matches!(output, Output::Surface) && self.surface.is_none() {
            let pixels = self.float.size() / 16;
            let buffer = |label, bytes| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
            self.surface = Some((
                buffer("depth target", pixels * 8),
                buffer("normal target", pixels * 16),
            ));
        }
        loop {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let target = match (output, &self.surface) {
                (Output::Float, _) => Target::Float(self.float.clone()),
                (Output::Packed, _) => Target::Packed(self.packed.clone()),
                (Output::Surface, Some((depth, normal))) => Target::Surface {
                    color: self.float.clone(),
                    depth: depth.clone(),
                    normal: normal.clone(),
                },
                (Output::Surface, None) => unreachable!("surface buffers are allocated above"),
            };
            self.core.render(
                &self.queue,
                &mut encoder,
                &mut self.view,
                &camera.core_camera(),
                &self.options,
                target,
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
    /// Throughput mode: submit a float frame without waiting for it, keeping up to three frames in
    /// flight as the viewer does. Fails if a completed frame overflowed its intersection capacity,
    /// so warm up with [`Self::render`] first. [`Self::finish_submitted`] waits for the rest.
    pub fn submit_float(&mut self, camera: &CameraSpec) -> Result<()> {
        camera.validate()?;
        if UVec2::new(camera.width, camera.height) != self.size {
            anyhow::bail!("renderer resolution changed");
        }
        loop {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            match self.core.render(
                &self.queue,
                &mut encoder,
                &mut self.view,
                &camera.core_camera(),
                &self.options,
                Target::Float(self.float.clone()),
            ) {
                Ok(()) => {
                    self.queue.submit([encoder.finish()]);
                    self.last_output = Some(Output::Float);
                    return self.check_submitted();
                }
                Err(gsplat_core::Error::Busy(_)) => {
                    self.finish()?;
                    self.check_submitted()?;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    /// Wait for every submitted frame and check its feedback.
    pub fn finish_submitted(&mut self) -> Result<()> {
        self.finish()?;
        while self.view.has_pending_frames() {
            self.check_submitted()?;
        }
        Ok(())
    }
    fn check_submitted(&mut self) -> Result<()> {
        if let Some(stats) = self.view.poll_feedback()? {
            anyhow::ensure!(
                !stats.needs_rerender,
                "a submitted frame overflowed its intersection capacity; warm up with render() first"
            );
        }
        Ok(())
    }
    /// Read the target written by the last successful render.
    pub fn read_rgba(&self) -> Result<Vec<f32>> {
        let output = self.last_output.context("render before readback")?;
        let packed = matches!(output, Output::Packed);
        let target = if packed { &self.packed } else { &self.float };
        let encoder = self.device.create_command_encoder(&Default::default());
        let data = self.read_buffer(encoder, target)?;
        Ok(if packed {
            data.iter().map(|x| f32::from(*x) / 255.0).collect()
        } else {
            bytemuck::cast_slice(&data).to_vec()
        })
    }
    /// Read the depth and normal targets of the last `Output::Surface` render.
    pub fn read_surface(&self) -> Result<Surface> {
        let (Some(Output::Surface), Some((depth, normal))) = (self.last_output, &self.surface)
        else {
            anyhow::bail!("render a surface output before reading it");
        };
        let read = |buffer| {
            let encoder = self.device.create_command_encoder(&Default::default());
            self.read_buffer(encoder, buffer)
        };
        Ok(Surface {
            depth: bytemuck::cast_slice(&read(depth)?).to_vec(),
            normal: bytemuck::cast_slice(&read(normal)?).to_vec(),
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
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        encoder.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
            tx.send(result).ok();
        });
        self.queue.submit([encoder.finish()]);
        self.finish()?;
        rx.recv()??;
        staging
            .get_mapped_range(..)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Preserve the scored float values; conversion to PNG belongs only at export.
pub fn parity_image(pixels: Vec<f32>, width: u32, height: u32) -> Result<image::DynamicImage> {
    if pixels.len() != width as usize * height as usize * 4 || !pixels.iter().all(|v| v.is_finite())
    {
        anyhow::bail!("invalid finite RGBA dimensions");
    }
    Ok(image::DynamicImage::ImageRgba32F(
        image::Rgba32FImage::from_raw(width, height, pixels).expect("validated dimensions"),
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn evidence_preserves_scored_hdr_and_alpha() {
        let image = super::parity_image(vec![1.17, 0.5, -0.2, 0.1], 1, 1).unwrap();
        assert_eq!(image.to_rgba32f().as_raw(), &[1.17, 0.5, -0.2, 0.1]);
        assert!(super::parity_image(vec![f32::NAN; 4], 1, 1).is_err());
    }
}
