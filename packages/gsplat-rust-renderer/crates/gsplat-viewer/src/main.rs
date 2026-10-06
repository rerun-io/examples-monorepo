//! Stock Rerun viewer plus one extra Gaussian splat visualizer.
//!
//! Queries native GaussianSplats3D through ComputeGaussianSplats3D and composites
//! gsplat-core output in Spatial3DView. Supports live gRPC, saved recordings,
//! and headless screenshots with the same registration path.

use gsplat_viewer::gaussian_visualizer;

use clap::Parser as _;
use re_sdk_types::View as _;
use re_viewer::external::{eframe, egui};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Name shown in the viewer title bar and "About" dialog.
const VIEWER_NAME: &str = "Gaussian Splats Viewer";

/// Default TCP port for the gRPC server that receives Rerun log data.
/// This matches Rerun's default `rr.connect_grpc()` target.
const GRPC_PORT: u16 = 9876;

/// Default headless viewport size in logical points (used when `--headless`
/// is given without `--window-size WxH`).
const DEFAULT_HEADLESS_SIZE: egui::Vec2 = egui::vec2(1600.0, 900.0);

/// Entry point.  We use `#[tokio::main]` because the gRPC server needs an
/// async runtime, while the viewer itself runs on the main thread (required
/// by most windowing systems).
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = parse_cli()?;
    if cli.print_version {
        println!("rerun {}", re_viewer::build_info().version);
        return Ok(());
    }

    // Standard Rerun logging and crash-handler setup.
    re_log::setup_logging();
    re_crash_handler::install_crash_handlers(re_viewer::build_info());

    // ── Start the gRPC server ─────────────────────────────────────────────
    // This spawns a background task that listens for incoming Rerun log
    // messages.  `grpc_rx` is a channel receiver that feeds those messages
    // into the viewer's data store.
    //
    // Keep the returned server handle bound for the whole `main` scope (both
    // the windowed and headless run loops block until shutdown) so the
    // server-side message proxy stays alive.
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
    let rrd_path = cli.rrd_path;

    // ── Create the viewer application ─────────────────────────────────────
    // `MainThreadToken` is a safety marker that proves we're on the main
    // thread, which is required by the native windowing backend.
    let main_thread_token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();
    let app_env = re_viewer::AppEnvironment::Custom(VIEWER_NAME.to_owned());
    let startup_options = re_viewer::StartupOptions {
        // Don't persist viewer state between runs — each launch starts fresh.
        persist_state: false,
        hide_welcome_screen: cli.hide_welcome_screen,
        ..Default::default()
    };

    // ── Headless mode ─────────────────────────────────────────────────────
    // No OS window: the viewer is driven by an `egui_kittest` harness instead
    // of `eframe::run_native`.  The gRPC server keeps running, so SDK clients
    // (including `ViewerClient.save_screenshot`) work the same way.
    if cli.headless {
        return run_headless(cli.window_size, move |cc| {
            create_app(
                cc,
                main_thread_token,
                app_env,
                startup_options,
                grpc_rx,
                rrd_path,
            )
        });
    }

    // ── Launch the native window ──────────────────────────────────────────
    // `eframe::run_native` opens the OS window and hands control to the
    // viewer's render loop.  The closure receives `cc` (creation context)
    // which provides access to the wgpu device and egui setup.
    eframe::run_native(
        "Rerun Viewer",
        native_options(),
        Box::new(move |cc| {
            let viewer = create_app(
                cc,
                main_thread_token,
                app_env,
                startup_options,
                grpc_rx,
                rrd_path,
            )?;
            Ok(Box::new(viewer))
        }),
    )
    .map_err(|err| anyhow::anyhow!(err))
}

/// Create the customized Rerun viewer `App`.
///
/// Shared by the windowed (`eframe::run_native`) and headless
/// (`egui_kittest` harness) paths so both register the exact same gRPC
/// receiver and custom visualizer.
fn create_app(
    cc: &eframe::CreationContext<'_>,
    main_thread_token: re_viewer::MainThreadToken,
    app_env: re_viewer::AppEnvironment,
    startup_options: re_viewer::StartupOptions,
    grpc_rx: re_log_channel::LogReceiver,
    rrd_path: Option<PathBuf>,
) -> anyhow::Result<re_viewer::App> {
    // Let Rerun set up its custom wgpu renderer (re_renderer) and
    // egui integration before we create the App.
    re_viewer::customize_eframe_and_setup_renderer(cc)?;

    let mut viewer = re_viewer::App::new(
        main_thread_token,
        re_viewer::build_info(),
        app_env,
        startup_options,
        cc,
        None, // No custom connection registry
        re_viewer::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()
            .expect("tokio runtime should exist"),
    );

    // Rerun 0.38 requires registration before the first renderer() lookup.
    viewer.with_render_ctx_mut(|ctx| {
        ctx.renderers_mut()
            .register::<gsplat_viewer::gaussian_renderer::GaussianRenderer>();
    });

    // Keep both explicit choices; the context system installs process-local defaults.
    viewer.extend_view_class(
        re_sdk_types::blueprint::views::Spatial3DView::identifier(),
        register_splat_system,
    )?;

    // Wire up live logs and positional files only after the custom visualizer
    // is registered, so the first activated blueprint can resolve ComputeGaussianSplats3D.
    viewer.add_log_receiver(grpc_rx);
    if let Some(rrd_path) = rrd_path {
        // Rerun's normal file-opening route both ingests the stores and selects
        // the recording. Feeding decoded messages through add_log_receiver left
        // the app on the catalog page with no active recording.
        viewer.open_url_or_file(&rrd_path.to_string_lossy());
    }

    Ok(viewer)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Headless mode
// ═══════════════════════════════════════════════════════════════════════════════
//
// Drive the public test harness with the same compute-capable device as the window.

/// Run the viewer in headless mode.
///
/// Instead of opening a real OS window via `eframe::run_native`, this drives
/// the viewer through an `egui_kittest` harness backed by wgpu, repeatedly
/// calling `step()`.  The gRPC server keeps running in the background just
/// like in the normal viewer, so SDK clients (including `save_screenshot`)
/// work the same way.
///
/// Blocks until the viewer receives a close request or the process is killed.
fn run_headless(
    window_size: Option<egui::Vec2>,
    app_creator: impl FnOnce(&eframe::CreationContext<'_>) -> anyhow::Result<re_viewer::App>,
) -> anyhow::Result<()> {
    let size = window_size.unwrap_or(DEFAULT_HEADLESS_SIZE);

    // Signal flipped to `true` whenever something calls `ctx.request_repaint()`.
    // The headless loop uses this to wake up early instead of waiting the full
    // 1s idle tick — keeps animations and incoming gRPC data feeling snappy
    // while still letting an idle viewer sleep most of the time.
    let repaint_signal: Arc<(Mutex<bool>, Condvar)> = Arc::new((Mutex::new(false), Condvar::new()));
    #[cfg(feature = "probe")]
    let probe_state = Arc::new(std::sync::OnceLock::new());

    let mut harness = {
        let repaint_signal = repaint_signal.clone();
        #[cfg(feature = "probe")]
        let probe_state = probe_state.clone();
        egui_kittest::Harness::<re_viewer::App>::builder()
            .with_size(size)
            .with_step_dt(1.0 / 60.0)
            .wgpu_setup(full_limits_wgpu_setup())
            .build_eframe(move |cc| {
                #[cfg(feature = "probe")]
                let _ = probe_state.set(cc.wgpu_render_state.clone().expect("wgpu viewer"));
                let repaint_signal = repaint_signal.clone();
                cc.egui_ctx.set_request_repaint_callback(move |_info| {
                    let (lock, cvar) = &*repaint_signal;
                    *lock.lock().expect("repaint signal mutex poisoned") = true;
                    cvar.notify_all();
                });
                // Creation failures (renderer setup, visualizer registration)
                // are fatal in a non-interactive headless run — abort loudly.
                let mut app = app_creator(cc)
                    .unwrap_or_else(|err| panic!("failed to create headless viewer app: {err}"));
                // Keep logs/reports available without transient overlays in pixel captures.
                app.app_options_mut().show_notification_toasts = false;
                app
            })
    };

    re_log::info!("Headless viewer running at {}x{}.", size.x, size.y);

    let idle_timeout = Duration::from_secs(1);
    #[cfg(feature = "probe")]
    let mut probe = std::env::var_os("GSPLAT_VIEWER_PROBE").map(|path| {
        gsplat_viewer::frame_probe::FrameProbe::new(
            path.into(),
            probe_state.get().expect("created device").clone(),
            size,
        )
    });
    loop {
        #[cfg(feature = "probe")]
        if let Some(probe) = &mut probe {
            probe.begin_frame(&harness.ctx);
        }
        harness.step();
        #[cfg(feature = "probe")]
        if let Some(probe) = &mut probe
            && probe.end_frame(&harness.ctx, harness.output())?
        {
            // Pixel evidence is captured only after the last timed sample.
            harness
                .render()
                .map_err(|err| anyhow::anyhow!(err))?
                .save(probe.screenshot_path())?;
            return Ok(());
        }

        if has_pending_close(&harness) {
            re_log::info!("Headless viewer received close request, shutting down.");
            return Ok(());
        }

        let (lock, cvar) = &*repaint_signal;
        let mut signaled = lock.lock().expect("repaint signal mutex poisoned");
        if !*signaled {
            // Spurious wakeups just cause one extra (cheap) frame — harmless.
            signaled = cvar
                .wait_timeout(signaled, idle_timeout)
                .expect("repaint signal mutex poisoned")
                .0;
        }
        *signaled = false;
    }
}

/// Detect `ViewportCommand::Close` in this frame's viewport output.
///
/// `UICommand::Quit` (and the Ctrl-C handler) ultimately send
/// `ViewportCommand::Close`.  In a normal `eframe::run_native` setup the
/// windowing backend consumes that and exits the event loop. The harness
/// handles screenshot commands; this outer loop must still honor `Close`.
fn has_pending_close(harness: &egui_kittest::Harness<'_, re_viewer::App>) -> bool {
    harness
        .output()
        .viewport_output
        .values()
        .flat_map(|v| v.commands.iter())
        .any(|cmd| matches!(cmd, egui::ViewportCommand::Close))
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLI parsing
// ═══════════════════════════════════════════════════════════════════════════════

/// Viewer startup options. Unknown flags fail with a useful CLI error.
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
    // SDK spawn compatibility. Rerun manages these defaults internally.
    #[arg(long = "memory-limit", hide = true)]
    _memory_limit: Option<String>,
    #[arg(long = "server-memory-limit", hide = true)]
    _server_memory_limit: Option<String>,
    #[arg(long = "expect-data-soon", hide = true)]
    _expect_data_soon: bool,
}
fn parse_cli() -> anyhow::Result<Cli> {
    Ok(Cli::parse())
}

/// Parse a `WIDTHxHEIGHT` string (e.g. `1600x900`) into an `egui::Vec2`.
fn parse_window_size(value: &str) -> anyhow::Result<egui::Vec2> {
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

// ═══════════════════════════════════════════════════════════════════════════════
// Window and GPU configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the wgpu setup shared by the windowed and headless paths.
///
/// Key choices:
/// - **Adapter selection** — delegates to `re_renderer` so we pick the same
///   GPU that Rerun's internal renderer expects.
/// - **Device limits** — we request the WebGPU compute limits so compute
///   shaders (storage buffers, workgroup sizes) aren't artificially capped.
///   This is why the headless path can't use `re_viewer`'s own wgpu options:
///   those request zero compute limits.
fn validate_compute_adapter(
    features: re_renderer::external::wgpu::Features,
    limits: &re_renderer::external::wgpu::Limits,
) -> Result<(), String> {
    use re_renderer::external::wgpu;
    if !features.contains(wgpu::Features::SUBGROUP) {
        return Err("ComputeGaussianSplats3D requires wgpu Features::SUBGROUP".into());
    }
    let required = wgpu::Limits {
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_buffer_size: limits.max_buffer_size,
        ..Default::default()
    };
    if !required.check_limits(limits) {
        return Err("ComputeGaussianSplats3D requires WebGPU compute limits (8 storage buffers, 256 threads per workgroup)".into());
    }
    Ok(())
}

fn full_limits_wgpu_setup() -> eframe::egui_wgpu::WgpuSetup {
    eframe::egui_wgpu::WgpuSetup::CreateNew(eframe::egui_wgpu::WgpuSetupCreateNew {
        // Use Rerun's preferred wgpu instance descriptor (Vulkan on
        // Linux, Metal on macOS).
        instance_descriptor: re_renderer::device_caps::instance_descriptor(None),
        native_adapter_selector: Some(Arc::new(move |adapters, surface| {
            let adapter = re_renderer::device_caps::select_adapter(
                adapters,
                re_renderer::device_caps::instance_descriptor(None).backends,
                surface,
            )?;
            validate_compute_adapter(adapter.features(), &adapter.limits())
                .map_err(|error| format!("{}: {error}", adapter.get_info().name))?;
            Ok(adapter)
        })),
        device_descriptor: Arc::new(|adapter| re_renderer::external::wgpu::DeviceDescriptor {
            label: Some("gsplat-rust-renderer device"),
            required_features: gsplat_core::required_features(adapter),
            required_limits: gsplat_core::required_limits(adapter),
            memory_hints: re_renderer::external::wgpu::MemoryHints::MemoryUsage,
            trace: re_renderer::external::wgpu::Trace::Off,
            experimental_features: Default::default(),
        }),
        ..eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle()
    })
}

/// Build the `eframe::NativeOptions` that configure the OS window and the
/// wgpu (WebGPU) graphics backend.
///
/// Key choices:
/// - **VSync** (`AutoVsync`) — prevents tearing and saves power.
/// - **GPU setup** — see [`full_limits_wgpu_setup`].
fn native_options() -> eframe::NativeOptions {
    let mut native_options = re_viewer::native::eframe_options(None);
    native_options.wgpu_options = eframe::egui_wgpu::WgpuConfiguration {
        surface: eframe::egui_wgpu::SurfaceConfig {
            present_mode: re_renderer::external::wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: None,
        },
        on_surface_status: Arc::new(|status| {
            // On non-Windows platforms, an "Outdated" surface just means the
            // window was resized — recreate the surface and carry on.
            if matches!(
                status,
                re_renderer::external::wgpu::CurrentSurfaceTexture::Outdated
            ) && !cfg!(target_os = "windows")
            {
                eframe::egui_wgpu::SurfaceErrorAction::RecreateSurface
            } else {
                eframe::egui_wgpu::SurfaceErrorAction::SkipFrame
            }
        }),
        wgpu_setup: full_limits_wgpu_setup(),
    };
    native_options
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
mod device_tests {
    use re_renderer::external::wgpu;

    #[test]
    fn startup_reports_missing_subgroups() {
        let error =
            super::validate_compute_adapter(wgpu::Features::empty(), &wgpu::Limits::default())
                .unwrap_err();
        assert!(error.contains("SUBGROUP"));
        assert!(
            super::validate_compute_adapter(wgpu::Features::SUBGROUP, &wgpu::Limits::default())
                .is_ok()
        );
    }
}

fn register_splat_system(
    registrator: &mut re_viewer_context::ViewSystemRegistrator<'_>,
) -> Result<(), re_viewer_context::ViewClassRegistryError> {
    gsplat_viewer::bounds::register(registrator);
    registrator
        .register_context_system::<gsplat_viewer::automatic_selection::AutomaticSplatSelection>()?;
    registrator.register_visualizer::<gaussian_visualizer::GaussianSplatVisualizer>()
}

#[cfg(test)]
mod automatic_selection_tests {
    use super::*;
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
            visualizers.iter_with_identifiers().count() > 10,
            "Other spatial visualizers remain registered"
        );
    }
}
