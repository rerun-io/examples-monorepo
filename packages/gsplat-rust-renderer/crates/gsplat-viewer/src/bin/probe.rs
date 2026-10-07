//! Opt-in headless frame timing. Includes UI, compute, composite, and GPU completion;
//! excludes screenshot readback. The production viewer has no probe hooks.
#[path = "../application.rs"]
pub mod application;

use gsplat_render::camera::CameraSpec;
use re_viewer::external::{eframe, egui};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct Frame {
    ms: f64,
    camera: CameraSpec,
}
#[derive(Serialize)]
struct Report<'a> {
    source_sha: String,
    adapter: String,
    boundary: &'static str,
    warmup_frames: usize,
    warmup_seconds: f64,
    minimum_camera_delta: f32,
    msaa_samples: u32,
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
    require_motion: bool,
}
impl FrameProbe {
    pub fn new(
        path: std::path::PathBuf,
        state: eframe::egui_wgpu::RenderState,
        size: egui::Vec2,
        require_motion: bool,
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
            require_motion,
        }
    }
    fn begin_frame(&mut self) {
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
        camera: gsplat_core::Camera,
        msaa_samples: u32,
    ) -> anyhow::Result<bool> {
        if let Some(first) = self.frames.first() {
            anyhow::ensure!(
                camera.size == glam::uvec2(first.camera.width, first.camera.height),
                "measured viewport changed"
            );
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
        if self.warmup < 300 || self.warmup_ms < 5_000.0 || !self.warmup.is_multiple_of(300) {
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
                model: camera.model,
            },
        });
        self.measured_ms += ms;
        if self.frames.len() < 300
            || self.measured_ms < 10_000.0
            || !self.frames.len().is_multiple_of(300)
        {
            return Ok(false);
        }
        let minimum_camera_delta = self
            .frames
            .windows(2)
            .map(|pair| {
                pair[0]
                    .camera
                    .world_from_camera
                    .iter()
                    .flatten()
                    .zip(pair[1].camera.world_from_camera.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
            })
            .fold(f32::INFINITY, f32::min);
        anyhow::ensure!(
            !self.require_motion || minimum_camera_delta > 1e-5,
            "camera held during a moving-view measurement"
        );
        let report = Report {
            source_sha: std::env::var("GSPLAT_SOURCE_SHA").unwrap_or_else(|_| "unrecorded".into()),
            adapter: self.state.adapter.get_info().name,
            boundary: "headless UI step + compute + Rerun composite + egui paint + GPU completion; no pixel readback; no idle wait",
            warmup_frames: self.warmup,
            warmup_seconds: self.warmup_ms / 1000.0,
            minimum_camera_delta,
            msaa_samples,
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

use clap::Parser as _;
use re_sdk_types::View as _;
use re_viewer_context::{IdentifiedViewSystem, ViewContextSystem};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

#[derive(clap::Parser)]
struct Cli {
    rrd: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "1920x1080", value_parser = application::parse_window_size)]
    window_size: egui::Vec2,
    #[arg(long, default_value_t = 0)]
    port: u16,
    /// Request a smaller storage binding limit to exercise the stock-renderer fallback.
    #[arg(long)]
    storage_binding_limit: Option<u64>,
    /// Reject a timing run if any adjacent measured cameras are identical.
    #[arg(long)]
    require_motion: bool,
}
struct ProbeView {
    camera: gsplat_core::Camera,
    msaa_samples: u32,
}
static CAMERAS: Mutex<Vec<ProbeView>> = Mutex::new(Vec::new());
#[derive(Default)]
struct ProbeEye;
impl IdentifiedViewSystem for ProbeEye {
    fn identifier() -> re_viewer_context::ViewSystemIdentifier {
        "ProbeEye".into()
    }
}
impl ViewContextSystem for ProbeEye {
    fn execute(
        &mut self,
        ctx: &re_viewer_context::ViewContext<'_>,
        _: &re_viewer_context::MissingChunkReporter,
        query: &re_viewer_context::ViewQuery<'_>,
        _: &re_viewer_context::ViewContextSystemOncePerFrameResult,
    ) {
        if let Some(camera) = gsplat_viewer::visualizer::camera_from_view(ctx, query) {
            CAMERAS.lock().expect("probe camera").push(ProbeView {
                camera,
                msaa_samples: re_renderer::ViewBuilder::main_target_default_msaa_state(
                    ctx.render_ctx().render_config(),
                    false,
                )
                .count,
            });
        }
    }
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    anyhow::ensure!(
        cli.rrd.is_file(),
        "recording does not exist: {}",
        cli.rrd.display()
    );
    if let Some(parent) = cli.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    re_log::setup_logging();
    let (rx, _server) = re_grpc_server::spawn_with_recv(
        std::net::SocketAddr::from(([127, 0, 0, 1], cli.port)),
        re_grpc_server::ServerOptions::default(),
        re_grpc_server::shutdown::never(),
    );
    let state = Arc::new(OnceLock::new());
    let setup_state = state.clone();
    let token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();
    let mut setup = application::compute_wgpu_setup();
    if let Some(limit) = cli.storage_binding_limit
        && let eframe::egui_wgpu::WgpuSetup::CreateNew(config) = &mut setup
    {
        let original = config.device_descriptor.clone();
        config.device_descriptor = Arc::new(move |adapter| {
            let mut descriptor = original(adapter);
            descriptor.required_limits.max_storage_buffer_binding_size =
                limit.min(descriptor.required_limits.max_storage_buffer_binding_size);
            descriptor
        });
    }
    let mut harness = egui_kittest::Harness::<re_viewer::App>::builder()
        .with_size(cli.window_size)
        .with_step_dt(1.0 / 60.0)
        .wgpu_setup(setup)
        .build_eframe(move |cc| {
            let _ = setup_state.set(cc.wgpu_render_state.clone().expect("probe device"));
            let mut app = application::create_app(
                cc,
                token,
                re_viewer::AppEnvironment::Custom("Gaussian splat frame probe".into()),
                re_viewer::StartupOptions {
                    persist_state: false,
                    hide_welcome_screen: true,
                    ..Default::default()
                },
                rx,
                Some(cli.rrd),
            )
            .expect("probe viewer");
            app.app_options_mut().show_notification_toasts = false;
            app.extend_view_class(
                re_sdk_types::blueprint::views::Spatial3DView::identifier(),
                |registrator| registrator.register_context_system::<ProbeEye>(),
            )
            .expect("probe eye system");
            app
        });
    let mut probe = FrameProbe::new(
        cli.out,
        state.get().expect("created device").clone(),
        cli.window_size,
        cli.require_motion,
    );
    let started = Instant::now();
    loop {
        CAMERAS.lock().expect("probe cameras").clear();
        probe.begin_frame();
        harness.step();
        let cameras = std::mem::take(&mut *CAMERAS.lock().expect("probe cameras"));
        anyhow::ensure!(cameras.len() <= 1, "the frame probe requires one 3D view");
        let Some(view) = cameras.first() else {
            anyhow::ensure!(
                started.elapsed().as_secs() < 120,
                "no camera became available from the view state"
            );
            continue;
        };
        if probe.end_frame(
            &harness.ctx,
            harness.output(),
            view.camera,
            view.msaa_samples,
        )? {
            harness
                .render()
                .map_err(|error| anyhow::anyhow!(error))?
                .save(probe.screenshot_path())?;
            return Ok(());
        }
    }
}
