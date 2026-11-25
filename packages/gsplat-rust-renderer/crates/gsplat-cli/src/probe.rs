//! Opt-in headless frame timing. Includes UI, compute, composite, and GPU completion;
//! excludes screenshot readback. The production viewer keeps one hook for it: a process-wide count of
//! encoded splat frames (`gsplat_viewer::compute_encoded_frames`).
//! With `--frames-in-flight` the probe queues frames like a swapchain instead of waiting for each one,
//! so `encoded_frames` shows how many UI frames encoded a new splat image.
//! Splat frames per second: `encoded_frames / measured_wall_seconds` (both cover the measured frames).
use gsplat_viewer::application;

use gsplat_cli::camera::CameraSpec;
use re_viewer::external::{eframe, egui, parking_lot::Mutex};
use serde::Serialize;
use std::time::Instant;

/// Warm-up and measurement end on whole orbits of the kit recordings (300 frames at the 1/60 s step).
const ORBIT_FRAMES: usize = 300;
const WARMUP_MS: f64 = 10_000.0;
const MEASURE_MS: f64 = 10_000.0;

#[derive(Serialize)]
struct Frame {
    ms: f64,
    camera: CameraSpec,
}
#[derive(Serialize)]
struct Report<'a> {
    #[serde(flatten)]
    provenance: gsplat_cli::Provenance,
    adapter: String,
    boundary: String,
    warmup_frames: usize,
    warmup_seconds: f64,
    minimum_camera_delta: f32,
    msaa_samples: u32,
    frames_in_flight: usize,
    /// Splat frames encoded during the measured frames, summed over every compute view and entity of
    /// the process. With one splat entity, `encoded_frames / frames.len()` is the fraction of UI frames
    /// that encode a new splat image (the others show the previous one). A frame counts when it is
    /// encoded, also one that overflows its intersection capacity and keeps the previous image.
    encoded_frames: u64,
    /// Wall time from the end of the last warmup frame to the end of the last measured frame; unlike
    /// the sum of `frames[].ms` it includes the time between frames.
    measured_wall_seconds: f64,
    /// Compute-renderer GPU bytes at the last frame: shared scene uploads, per-view scratch and targets.
    gpu_bytes_scene: u64,
    gpu_bytes_views: u64,
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
    frames_in_flight: usize,
    submitted: std::collections::VecDeque<wgpu::SubmissionIndex>,
    encoded_at_start: u64,
    measured_from: Instant,
    splats_seen: bool,
}
impl FrameProbe {
    pub fn new(
        path: std::path::PathBuf,
        state: eframe::egui_wgpu::RenderState,
        size: egui::Vec2,
        require_motion: bool,
        frames_in_flight: usize,
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
            frames_in_flight,
            submitted: std::collections::VecDeque::new(),
            encoded_at_start: 0,
            measured_from: Instant::now(),
            splats_seen: false,
        }
    }
    /// Paint the full UI into a persistent offscreen target, then wait for the GPU to finish all but
    /// `frames_in_flight` frames.
    pub fn end_frame(
        &mut self,
        ctx: &egui::Context,
        output: &egui::FullOutput,
        camera: gsplat_core::Camera,
        msaa_samples: u32,
        gpu_bytes: [u64; 2],
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
        self.submitted.push_back(
            self.state
                .queue
                .submit(buffers.into_iter().chain([encoder.finish()])),
        );
        while self.submitted.len() > self.frames_in_flight {
            self.state.device.poll(wgpu::PollType::Wait {
                submission_index: self.submitted.pop_front(),
                timeout: None,
            })?;
        }
        let ms = self.started.elapsed().as_secs_f64() * 1000.0;
        // Read after egui's paint callbacks, which encode this frame's splats.
        let encoded_frames = gsplat_viewer::compute_encoded_frames();
        // The warm-up clock restarts at the first splat frame: a large scene loads for seconds with cheap
        // UI-only frames, and a desktop GPU needs that long under load to settle its clock.
        if !self.splats_seen && encoded_frames > 0 {
            self.splats_seen = true;
            self.warmup_ms = 0.0;
            // A load longer than the warm-up: the frames measured so far drew no splats.
            self.warmup += self.frames.len();
            self.frames.clear();
            self.measured_ms = 0.0;
        }
        if self.warmup < ORBIT_FRAMES
            || self.warmup_ms < WARMUP_MS
            || !self.warmup.is_multiple_of(ORBIT_FRAMES)
        {
            self.warmup += 1;
            self.warmup_ms += ms;
            self.encoded_at_start = encoded_frames;
            self.measured_from = Instant::now();
            return Ok(false);
        }
        self.frames.push(Frame {
            ms,
            camera: CameraSpec::from_core(camera),
        });
        self.measured_ms += ms;
        if self.frames.len() < ORBIT_FRAMES
            || self.measured_ms < MEASURE_MS
            || !self.frames.len().is_multiple_of(ORBIT_FRAMES)
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
        anyhow::ensure!(
            self.splats_seen,
            "no splat frame was encoded: the measured frames drew only the UI"
        );
        // With frames in flight a frame waits for an older submission, so its ms is the steady-state frame
        // interval, not one frame's latency.
        let completion = if self.frames_in_flight == 0 {
            String::new()
        } else {
            format!(
                " of the frame {} submissions back (swapchain-like)",
                self.frames_in_flight
            )
        };
        let report = Report {
            provenance: gsplat_cli::Provenance::capture(),
            adapter: self.state.adapter.get_info().name,
            boundary: format!(
                "headless UI step + compute + Rerun composite + egui paint + GPU completion{completion}; no pixel readback; no idle wait"
            ),
            warmup_frames: self.warmup,
            warmup_seconds: self.warmup_ms / 1000.0,
            minimum_camera_delta,
            msaa_samples,
            frames_in_flight: self.frames_in_flight,
            encoded_frames: encoded_frames - self.encoded_at_start,
            measured_wall_seconds: self.measured_from.elapsed().as_secs_f64(),
            gpu_bytes_scene: gpu_bytes[0],
            gpu_bytes_views: gpu_bytes[1],
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

use re_sdk_types::View as _;
use re_viewer_context::{IdentifiedViewSystem, ViewContextSystem};
use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};

#[derive(clap::Args)]
pub struct Args {
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
    /// Queue up to this many frames before waiting for the oldest, like a swapchain; 0 waits for every frame.
    #[arg(long, default_value_t = 0)]
    frames_in_flight: usize,
}
struct ProbeView {
    camera: gsplat_core::Camera,
    msaa_samples: u32,
    gpu_bytes: [u64; 2],
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
        if let Some(camera) = gsplat_viewer::camera_from_view(ctx, query) {
            CAMERAS.lock().push(ProbeView {
                camera,
                msaa_samples: re_renderer::ViewBuilder::main_target_default_msaa_state(
                    ctx.render_ctx().render_config(),
                    false,
                )
                .count,
                gpu_bytes: gsplat_viewer::compute_gpu_bytes(ctx),
            });
        }
    }
}
pub fn run(cli: Args) -> anyhow::Result<()> {
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
        re_grpc_server::ServerListener::bind(std::net::SocketAddr::from((
            [127, 0, 0, 1],
            cli.port,
        )))?,
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
            setup_state
                .set(cc.wgpu_render_state.clone().expect("probe device"))
                .ok();
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
        cli.frames_in_flight,
    );
    let started = Instant::now();
    loop {
        CAMERAS.lock().clear();
        probe.started = Instant::now();
        harness.step();
        let cameras = std::mem::take(&mut *CAMERAS.lock());
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
            view.gpu_bytes,
        )? {
            harness
                .render()
                .map_err(|error| anyhow::anyhow!(error))?
                .save(probe.path.with_extension("png"))?;
            return Ok(());
        }
    }
}
