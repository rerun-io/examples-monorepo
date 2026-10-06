//! Opt-in headless frame timing. Includes UI, compute, composite, and GPU completion;
//! excludes screenshot readback. GSPLAT_VIEWER_PROBE names the output JSON.
use re_viewer::external::{eframe, egui};
use serde::Serialize;
use std::time::Instant;

pub fn observe(ctx: &egui::Context, camera: gsplat_core::Camera) {
    ctx.data_mut(|data| data.insert_temp(egui::Id::new("gsplat submitted camera"), camera));
}

#[derive(Serialize)]
struct Frame {
    ms: f64,
    camera: CameraSpec,
}
/// Matches gsplat-render's strict CameraSpec input, allowing identical camera replay.
#[derive(Serialize)]
struct CameraSpec {
    world_from_camera: [[f32; 4]; 4],
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    model: &'static str,
}
#[derive(Serialize)]
struct Report<'a> {
    source_sha: String,
    adapter: String,
    boundary: &'static str,
    warmup_frames: usize,
    warmup_seconds: f64,
    frames: &'a [Frame],
}

pub struct FrameProbe {
    path: std::path::PathBuf,
    state: eframe::egui_wgpu::RenderState,
    target: wgpu::Texture,
    warmup: usize,
    warmup_ms: f64,
    frames: Vec<Frame>,
    measured_ms: f64,
    started: Instant,
}
impl FrameProbe {
    pub fn new(
        path: std::path::PathBuf,
        state: eframe::egui_wgpu::RenderState,
        size: egui::Vec2,
    ) -> Self {
        let target = state.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless frame probe"),
            size: wgpu::Extent3d {
                width: size.x as u32,
                height: size.y as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: state.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self {
            path,
            state,
            target,
            warmup: 0,
            warmup_ms: 0.0,
            frames: Vec::new(),
            measured_ms: 0.0,
            started: Instant::now(),
        }
    }
    pub fn begin_frame(&mut self, ctx: &egui::Context) {
        ctx.data_mut(|data| {
            data.remove::<gsplat_core::Camera>(egui::Id::new("gsplat submitted camera"))
        });
        self.started = Instant::now();
    }
    pub fn screenshot_path(&self) -> std::path::PathBuf {
        self.path.with_extension("png")
    }
    /// Paint the full UI into a persistent offscreen target, then wait for the GPU.
    pub fn end_frame(
        &mut self,
        ctx: &egui::Context,
        output: &egui::FullOutput,
    ) -> anyhow::Result<bool> {
        let Some(camera) = ctx.data(|data| {
            data.get_temp::<gsplat_core::Camera>(egui::Id::new("gsplat submitted camera"))
        }) else {
            return Ok(false);
        };
        if camera.size != glam::uvec2(1920, 1080) {
            anyhow::ensure!(
                self.frames.is_empty(),
                "measured viewport changed: {:?}",
                camera.size
            );
            return Ok(false); // Initial blueprint/layout is still settling.
        }
        let screen = eframe::egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.target.width(), self.target.height()],
            pixels_per_point: ctx.pixels_per_point(),
        };
        let shapes = ctx.tessellate(output.shapes.clone(), ctx.pixels_per_point());
        let mut renderer = self.state.renderer.write();
        let mut encoder = self
            .state
            .device
            .create_command_encoder(&Default::default());
        let buffers = renderer.update_buffers(
            &self.state.device,
            &self.state.queue,
            &mut encoder,
            &shapes,
            &screen,
        );
        let view = self.target.create_view(&Default::default());
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("headless frame probe composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            renderer.render(&mut pass, &shapes, &screen);
        }
        self.state
            .queue
            .submit(buffers.into_iter().chain([encoder.finish()]));
        self.state
            .device
            .poll(wgpu::PollType::wait_indefinitely())?;
        let ms = self.started.elapsed().as_secs_f64() * 1000.0;
        if self.warmup < 300 || self.warmup_ms < 5_000.0 {
            self.warmup += 1;
            self.warmup_ms += ms;
            return Ok(false);
        }
        let focal = camera.focal();
        self.frames.push(Frame {
            ms,
            camera: CameraSpec {
                world_from_camera: glam::Mat4::from_rotation_translation(
                    camera.rotation,
                    camera.position,
                )
                .transpose()
                .to_cols_array_2d(),
                width: camera.size.x,
                height: camera.size.y,
                fx: focal.x,
                fy: focal.y,
                cx: camera.center_uv.x * camera.size.x as f32,
                cy: camera.center_uv.y * camera.size.y as f32,
                model: "Pinhole",
            },
        });
        self.measured_ms += ms;
        if self.frames.len() < 300 || self.measured_ms < 10_000.0 {
            return Ok(false);
        }
        let report = Report {
            source_sha: std::env::var("GSPLAT_SOURCE_SHA").unwrap_or_else(|_| "unrecorded".into()),
            adapter: self.state.adapter.get_info().name,
            boundary: "headless UI step + compute + Rerun composite + egui paint + GPU completion; no pixel readback; no idle wait",
            warmup_frames: self.warmup,
            warmup_seconds: self.warmup_ms / 1000.0,
            frames: &self.frames,
        };
        std::fs::write(&self.path, serde_json::to_vec_pretty(&report)?)?;
        std::fs::write(
            self.path.with_extension("cameras.json"),
            serde_json::to_vec_pretty(&self.frames.iter().map(|f| &f.camera).collect::<Vec<_>>())?,
        )?;
        Ok(true)
    }
}
