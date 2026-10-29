//! Stock Rerun viewer plus one extra Gaussian splat visualizer.
//!
//! Queries native `GaussianSplats3D` through `ComputeGaussianSplats3D` and composites
//! gsplat-core output in `Spatial3DView`. Supports live gRPC, saved recordings,
//! and headless screenshots with the same registration path.

use crate::visualizer;
use re_viewer::external::{
    parking_lot, re_log, re_log_channel, re_memory, re_renderer, re_sdk_types, re_viewer_context,
};

use clap::Parser as _;
use parking_lot::{Condvar, Mutex};
use re_sdk_types::View as _;
use re_viewer::external::{eframe, egui};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const VIEWER_NAME: &str = "Gaussian Splats Viewer";

const GRPC_PORT: u16 = 9876;

const DEFAULT_HEADLESS_SIZE: egui::Vec2 = egui::vec2(1600.0, 900.0);

/// Run the SDK-compatible custom viewer binary.
pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.print_version {
        println!("rerun {}", re_viewer::build_info().version);
        return Ok(());
    }

    re_log::setup_logging();
    re_crash_handler::install_crash_handlers(re_viewer::build_info());

    let grpc_addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, cli.port));
    re_log::info!(
        "Listening for Rerun logs on rerun+http://127.0.0.1:{}/proxy",
        cli.port
    );
    let (grpc_rx, _grpc_server_handle) = re_grpc_server::spawn_with_recv(
        grpc_addr,
        re_grpc_server::ServerOptions::default(),
        re_grpc_server::shutdown::never(),
    );
    let memory_limit = cli
        .memory_limit
        .as_deref()
        .map(re_memory::MemoryLimit::parse)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let rrd_path = cli.rrd_path;

    let main_thread_token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();
    let app_env = re_viewer::AppEnvironment::Custom(VIEWER_NAME.to_owned());
    let startup_options = re_viewer::StartupOptions {
        persist_state: false,
        hide_welcome_screen: cli.hide_welcome_screen,
        ..Default::default()
    };

    if cli.headless {
        return run_headless(cli.window_size, move |cc| {
            let mut app = create_app(
                cc,
                main_thread_token,
                app_env,
                startup_options,
                grpc_rx,
                rrd_path,
            )?;
            if let Some(limit) = memory_limit {
                app.app_options_mut().memory_limit = limit;
            }
            Ok(app)
        });
    }

    eframe::run_native(
        "Rerun Viewer",
        native_options(),
        Box::new(move |cc| {
            let mut viewer = create_app(
                cc,
                main_thread_token,
                app_env,
                startup_options,
                grpc_rx,
                rrd_path,
            )?;
            if let Some(limit) = memory_limit {
                viewer.app_options_mut().memory_limit = limit;
            }
            Ok(Box::new(viewer))
        }),
    )
    .map_err(|err| anyhow::anyhow!(err))
}

/// Build a stock viewer with compute splat systems registered.
pub fn create_app(
    cc: &eframe::CreationContext<'_>,
    main_thread_token: re_viewer::MainThreadToken,
    app_env: re_viewer::AppEnvironment,
    startup_options: re_viewer::StartupOptions,
    grpc_rx: re_log_channel::LogReceiver,
    rrd_path: Option<PathBuf>,
) -> anyhow::Result<re_viewer::App> {
    re_viewer::customize_eframe_and_setup_renderer(cc)?;

    let mut viewer = re_viewer::App::new(
        main_thread_token,
        re_viewer::build_info(),
        app_env,
        startup_options,
        cc,
        None, // No custom connection registry
        re_viewer::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()?,
    );

    viewer
        .with_render_ctx_mut(|ctx| {
            ctx.renderers_mut()
                .register::<crate::renderer::GaussianRenderer>();
        })
        .ok_or_else(|| {
            anyhow::anyhow!("the compute splat visualizer needs a wgpu render context")
        })?;

    viewer.extend_view_class(
        re_sdk_types::blueprint::views::Spatial3DView::identifier(),
        register_splat_system,
    )?;

    viewer.add_log_receiver(grpc_rx);
    if let Some(rrd_path) = rrd_path {
        viewer.open_url_or_file(&rrd_path.to_string_lossy());
    }

    Ok(viewer)
}

// Copy of re_viewer::run_headless_app because 0.38.1 cannot accept our WgpuSetup.
fn run_headless(
    window_size: Option<egui::Vec2>,
    app_creator: impl FnOnce(&eframe::CreationContext<'_>) -> anyhow::Result<re_viewer::App>,
) -> anyhow::Result<()> {
    let size = window_size.unwrap_or(DEFAULT_HEADLESS_SIZE);

    let repaint_signal: Arc<(Mutex<Option<Instant>>, Condvar)> =
        Arc::new((Mutex::new(None), Condvar::new()));

    let mut harness = {
        let repaint_signal = repaint_signal.clone();
        egui_kittest::Harness::<re_viewer::App>::builder()
            .with_size(size)
            .with_step_dt(1.0 / 60.0)
            .wgpu_setup(compute_wgpu_setup())
            .build_eframe(move |cc| {
                let repaint_signal = repaint_signal.clone();
                cc.egui_ctx.set_request_repaint_callback(move |info| {
                    let (lock, cvar) = &*repaint_signal;
                    let deadline = Instant::now() + info.delay;
                    let mut earliest = lock.lock();
                    if earliest.is_none_or(|current| deadline < current) {
                        *earliest = Some(deadline);
                        cvar.notify_all();
                    }
                });
                let mut app = app_creator(cc)
                    .unwrap_or_else(|err| panic!("failed to create headless viewer app: {err}"));
                app.app_options_mut().show_notification_toasts = false;
                app
            })
    };

    re_log::info!("Headless viewer running at {}x{}.", size.x, size.y);

    // Wake idle viewers once per second so close and connection changes get serviced.
    const IDLE_TICK: Duration = Duration::from_secs(1);
    // Headless has no vsync; cap immediate repaints at about 60 frames per second.
    const MIN_FRAME_TIME: Duration = Duration::from_millis(16);
    loop {
        let frame_start = Instant::now();
        harness.step();
        if has_pending_close(&harness) {
            re_log::info!("Headless viewer received close request, shutting down.");
            return Ok(());
        }

        let (lock, cvar) = &*repaint_signal;
        let idle_deadline = Instant::now() + IDLE_TICK;
        let frame_floor = frame_start + MIN_FRAME_TIME;
        let mut requested = lock.lock();
        loop {
            let deadline = requested
                .map_or(idle_deadline, |deadline| deadline.min(idle_deadline))
                .max(frame_floor);
            let now = Instant::now();
            if deadline <= now {
                break;
            }
            cvar.wait_for(&mut requested, deadline - now);
        }
        *requested = None;
    }
}

fn has_pending_close(harness: &egui_kittest::Harness<'_, re_viewer::App>) -> bool {
    harness
        .output()
        .viewport_output
        .values()
        .flat_map(|v| v.commands.iter())
        .any(|cmd| matches!(cmd, egui::ViewportCommand::Close))
}

#[derive(Clone, Debug, clap::Parser)]
#[command(disable_version_flag = true)]
struct Cli {
    #[arg(long, default_value_t = GRPC_PORT)]
    port: u16,
    #[arg(long = "version", short = 'V')]
    print_version: bool,
    #[arg(long)]
    headless: bool,
    #[arg(long, value_parser = parse_window_size)]
    window_size: Option<egui::Vec2>,
    #[arg(long)]
    hide_welcome_screen: bool,
    rrd_path: Option<PathBuf>,
    #[arg(long = "memory-limit")]
    memory_limit: Option<String>,

    /// Accepted for compatibility with the Rerun SDK spawn path; this viewer does not host a store server.
    #[arg(long = "server-memory-limit", hide = true)]
    _server_memory_limit: Option<String>,

    /// Accepted for compatibility with the Rerun SDK spawn path.
    #[arg(long = "expect-data-soon", hide = true)]
    _expect_data_soon: bool,
}

/// Parse positive logical viewport dimensions as `WIDTHxHEIGHT`.
pub fn parse_window_size(value: &str) -> anyhow::Result<egui::Vec2> {
    let (width, height) = value
        .split_once(['x', 'X'])
        .ok_or_else(|| anyhow::anyhow!("invalid window size '{value}': expected WIDTHxHEIGHT"))?;
    let width: f32 = width
        .trim()
        .parse()
        .map_err(|err| anyhow::anyhow!("invalid window width '{width}': {err}"))?;
    let height: f32 = height
        .trim()
        .parse()
        .map_err(|err| anyhow::anyhow!("invalid window height '{height}': {err}"))?;
    if !(width > 0.0 && height > 0.0) {
        anyhow::bail!("invalid window size '{value}': dimensions must be positive");
    }
    Ok(egui::vec2(width, height))
}

/// Request compute capabilities when the adapter supports the splat kernels.
pub fn compute_wgpu_setup() -> eframe::egui_wgpu::WgpuSetup {
    eframe::egui_wgpu::WgpuSetup::CreateNew(eframe::egui_wgpu::WgpuSetupCreateNew {
        instance_descriptor: re_renderer::device_caps::instance_descriptor(None),
        native_adapter_selector: Some(Arc::new(move |adapters, surface| {
            re_renderer::device_caps::select_adapter(
                adapters,
                re_renderer::device_caps::instance_descriptor(None).backends,
                surface,
            )
        })),
        device_descriptor: Arc::new(|adapter| {
            let mut descriptor =
                re_renderer::device_caps::DeviceCaps::from_adapter_without_validation(adapter)
                    .device_descriptor();
            descriptor.label = Some("Gaussian compute viewer");
            let available = adapter.limits();
            if gsplat_core::check_adapter(adapter.features(), &available).is_ok() {
                descriptor.required_features |= wgpu::Features::SUBGROUP;
                descriptor.required_limits =
                    gsplat_core::compute_limits(&available, descriptor.required_limits);
            }
            descriptor.memory_hints = wgpu::MemoryHints::MemoryUsage;
            descriptor
        }),
        ..eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle()
    })
}

fn native_options() -> eframe::NativeOptions {
    let mut native_options = re_viewer::native::eframe_options(None);
    native_options.wgpu_options.wgpu_setup = compute_wgpu_setup();
    native_options
}

fn register_splat_system(
    registrator: &mut re_viewer_context::ViewSystemRegistrator<'_>,
) -> Result<(), re_viewer_context::ViewClassRegistryError> {
    crate::bounds::register(registrator);
    registrator.register_context_system::<crate::selection::AutomaticComputeSelection>()?;
    registrator.register_visualizer::<visualizer::GaussianSplatVisualizer>()?;
    registrator.register_visualizer::<crate::hexels::HexelVisualizer>()
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn startup_arguments_and_sdk_flags() {
        let cli = Cli::try_parse_from([
            "viewer",
            "--headless",
            "--port=4321",
            "--window-size",
            "800x600",
            "--expect-data-soon",
            "training.rrd",
        ])
        .unwrap();
        assert_eq!(cli.rrd_path, Some(PathBuf::from("training.rrd")));
        assert_eq!(cli.window_size, Some(egui::vec2(800.0, 600.0)));
        assert_eq!(cli.port, 4321);
        assert!(cli.headless);
    }

    #[test]
    fn invalid_arguments_fail_clearly() {
        for args in [
            vec!["viewer", "one.rrd", "two.rrd"],
            vec!["viewer", "--future-output", "value.rrd"],
            vec!["viewer", "--window-size=0x1"],
        ] {
            assert!(Cli::try_parse_from(args).is_err());
        }
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use re_viewer::external::re_view_spatial;
    use re_viewer_context::{ViewClass as _, ViewClassRegistry};

    #[test]
    fn custom_spatial_view_keeps_both_explicit_choices() {
        let mut registry = ViewClassRegistry::default();
        let reflection = re_sdk_types::reflection::reflection();
        let options = Default::default();
        let mut fallbacks = Default::default();
        registry
            .add_class::<re_view_spatial::SpatialView3D>(reflection, &options, &mut fallbacks)
            .unwrap();
        registry
            .extend_class(
                re_view_spatial::SpatialView3D::identifier(),
                reflection,
                &options,
                &mut fallbacks,
                register_splat_system,
            )
            .unwrap();
        let visualizers =
            registry.new_visualizer_collection(re_view_spatial::SpatialView3D::identifier());
        let mut splats: Vec<_> = visualizers
            .iter_with_identifiers()
            .map(|(id, _)| id.to_string())
            .filter(|id| id.contains("GaussianSplats3D"))
            .collect();
        splats.sort();
        assert_eq!(splats, ["ComputeGaussianSplats3D", "GaussianSplats3D"]);
        assert!(
            visualizers
                .iter_with_identifiers()
                .any(|(id, _)| id.as_str() == crate::selection::COMPUTE_HEXELS)
        );
        assert!(
            visualizers.iter_with_identifiers().count() > 10,
            "Other spatial visualizers remain registered"
        );
    }
}
