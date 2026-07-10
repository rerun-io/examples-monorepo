//! Stock Rerun viewer plus one extra Gaussian splat visualizer.
//!
//! # Architecture
//!
//! This binary is a lightly customized Rerun viewer.  It does two things on top
//! of the stock viewer:
//!
//! 1. **Starts a gRPC server** on `127.0.0.1:9876` that accepts standard Rerun
//!    log messages.  Any Python process that calls `rr.connect_grpc()` will send
//!    component data here.
//!
//! 2. **Registers a custom `Gaussians3D` visualizer** on the built-in
//!    `Spatial3DView`.  When the data store contains entities matching the
//!    upstream `Gaussians3D` contract (`centers`, and optionally `scales`,
//!    `quaternions`, `colors`, `sh_coefficients`, `show_spherical_harmonics`),
//!    the custom visualizer takes over rendering using a GPU-accelerated
//!    Gaussian splatting pipeline instead of the stock point-cloud renderer.
//!
//! Everything else (UI, blueprint, timeline, selection, etc.) is inherited from
//! the stock Rerun viewer unchanged.
//!
//! # Usage
//!
//! ```bash
//! # Terminal 1 – launch the viewer:
//! cargo run --release --bin gsplat-rust-renderer
//!
//! # Headless variant (no OS window; screenshots via ViewerClient.save_screenshot):
//! cargo run --release --bin gsplat-rust-renderer -- --headless
//!
//! # Terminal 2 – send Gaussian splat data from Python:
//! python tools/log_gaussian_ply.py --rr-config.connect
//! ```

use gsplat_lib::gaussian_visualizer;

use re_sdk_types::View as _;
use re_viewer::external::{eframe, egui};
use std::ffi::OsString;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::num::ParseIntError;
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

    // ── Register the custom Gaussian splat visualizer ─────────────────
    // `extend_view_class` adds our visualizer to the existing
    // Spatial3DView.  Any entity that matches the Gaussians3D
    // archetype will be rendered by our custom GPU pipeline instead
    // of the stock point-cloud renderer.
    viewer.extend_view_class(
        re_sdk_types::blueprint::views::Spatial3DView::identifier(),
        |registrator| {
            registrator.register_visualizer::<gaussian_visualizer::GaussianSplatVisualizer>()?;
            Ok(())
        },
    )?;

    // Wire up live logs and positional files only after the custom visualizer
    // is registered, so the first activated blueprint can resolve Gaussians3D.
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
// Vendored from Rerun's `re_viewer/src/headless.rs`, with one key change:
// the harness uses OUR full-limits wgpu device (see `full_limits_wgpu_setup`)
// instead of `re_viewer`'s crate-private `wgpu_options()`.  Rerun's headless
// device requests downlevel-defaults limits with **zero compute limits**, which
// cannot create the Gaussian splatting compute pipelines.

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

    let mut harness = {
        let repaint_signal = repaint_signal.clone();
        egui_kittest::Harness::<re_viewer::App>::builder()
            .with_size(size)
            .wgpu_setup(full_limits_wgpu_setup())
            .build_eframe(move |cc| {
                let repaint_signal = repaint_signal.clone();
                cc.egui_ctx.set_request_repaint_callback(move |_info| {
                    let (lock, cvar) = &*repaint_signal;
                    *lock.lock().expect("repaint signal mutex poisoned") = true;
                    cvar.notify_all();
                });
                // Creation failures (renderer setup, visualizer registration)
                // are fatal in a non-interactive headless run — abort loudly.
                app_creator(cc)
                    .unwrap_or_else(|err| panic!("failed to create headless viewer app: {err}"))
            })
    };

    re_log::info!("Headless viewer running at {}x{}.", size.x, size.y);

    let idle_timeout = Duration::from_secs(1);
    loop {
        harness.step();
        handle_pending_screenshots(&mut harness);

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
/// windowing backend consumes that and exits the event loop.  `kittest`
/// ignores viewport commands, so we have to detect `Close` here and break
/// out of the headless loop ourselves.
fn has_pending_close(harness: &egui_kittest::Harness<'_, re_viewer::App>) -> bool {
    harness
        .output()
        .viewport_output
        .values()
        .flat_map(|v| v.commands.iter())
        .any(|cmd| matches!(cmd, egui::ViewportCommand::Close))
}

/// Bridge `egui::ViewportCommand::Screenshot` requests through `kittest`'s
/// offscreen renderer.
///
/// In a normal `eframe::run_native` setup, the windowing backend captures the
/// framebuffer after a screenshot command and emits an
/// `egui::Event::Screenshot` that the viewer's `App` listens for.  `kittest`
/// doesn't process viewport commands itself, so we have to do that translation
/// here, otherwise `save_screenshot` requests would be silently dropped.
fn handle_pending_screenshots(harness: &mut egui_kittest::Harness<'_, re_viewer::App>) {
    let pending: Vec<egui::UserData> = harness
        .output()
        .viewport_output
        .values()
        .flat_map(|v| v.commands.iter())
        .filter_map(|cmd| match cmd {
            egui::ViewportCommand::Screenshot(user_data) => Some(user_data.clone()),
            _ => None,
        })
        .collect();

    if pending.is_empty() {
        return;
    }

    let rgba = match harness.render() {
        Ok(rgba) => rgba,
        Err(err) => {
            re_log::error!("Failed to render headless screenshot: {err}");
            return;
        }
    };
    let size = [rgba.width() as usize, rgba.height() as usize];
    let pixels = rgba.into_raw();
    let color_image = Arc::new(egui::ColorImage::from_rgba_premultiplied(size, &pixels));

    for user_data in pending {
        harness.event(egui::Event::Screenshot {
            viewport_id: egui::ViewportId::ROOT,
            user_data,
            image: color_image.clone(),
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLI parsing
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimal command-line options.  We only need `--port`, `--headless`,
/// `--window-size`, `--hide-welcome-screen` and `--version`; everything else
/// (memory limits, etc.) is silently ignored so the binary can be used as a
/// drop-in replacement for `rerun`.
#[derive(Clone, Debug)]
struct Cli {
    /// TCP port for the gRPC server.
    port: u16,
    /// If true, print the version string and exit.
    print_version: bool,
    /// Run without an OS window (screenshots via `ViewerClient.save_screenshot`).
    headless: bool,
    /// Viewport size in logical points, parsed from `--window-size WxH`.
    window_size: Option<egui::Vec2>,
    /// Hide the welcome screen (passed by `rr.spawn` / `ViewerClient.spawn`).
    hide_welcome_screen: bool,
    /// Optional recording loaded into the custom viewer at startup.
    rrd_path: Option<PathBuf>,
}

fn parse_cli() -> anyhow::Result<Cli> {
    parse_cli_from(std::env::args_os().skip(1))
}

fn parse_cli_from(mut args: impl Iterator<Item = OsString>) -> anyhow::Result<Cli> {
    let mut cli = Cli {
        port: GRPC_PORT,
        print_version: false,
        headless: false,
        window_size: None,
        hide_welcome_screen: false,
        rrd_path: None,
    };

    while let Some(arg) = args.next() {
        if arg == "--version" || arg == "-V" {
            cli.print_version = true;
            continue;
        }

        if arg == "--port" {
            let value = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing value for --port"))?;
            cli.port = parse_port(&value)?;
            continue;
        }

        if let Some(value) = arg.to_str().and_then(|arg| arg.strip_prefix("--port=")) {
            cli.port = parse_port_str(value)?;
            continue;
        }

        if arg == "--headless" {
            cli.headless = true;
            continue;
        }

        if arg == "--window-size" {
            let value = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing value for --window-size"))?;
            let value = value
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("non-utf8 window size value"))?;
            cli.window_size = Some(parse_window_size(value)?);
            continue;
        }

        if let Some(value) = arg
            .to_str()
            .and_then(|arg| arg.strip_prefix("--window-size="))
        {
            cli.window_size = Some(parse_window_size(value)?);
            continue;
        }

        if arg == "--hide-welcome-screen" {
            cli.hide_welcome_screen = true;
            continue;
        }

        // Silently ignore flags that the stock `rerun` binary accepts so
        // this binary can be used as a drop-in replacement.
        if arg == "--memory-limit" || arg == "--server-memory-limit" {
            let _ = args.next();
            continue;
        }

        if arg == "--expect-data-soon" {
            continue;
        }

        if arg.to_str().is_some_and(|arg| {
            arg.starts_with("--memory-limit=") || arg.starts_with("--server-memory-limit=")
        }) {
            continue;
        }

        let path = PathBuf::from(&arg);
        if path.extension().is_some_and(|extension| extension == "rrd") {
            cli.rrd_path = Some(path);
        }
    }

    Ok(cli)
}

fn parse_port(value: &OsString) -> anyhow::Result<u16> {
    let value = value
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-utf8 port value"))?;
    parse_port_str(value)
}

fn parse_port_str(value: &str) -> anyhow::Result<u16> {
    value
        .parse::<u16>()
        .map_err(|err: ParseIntError| anyhow::anyhow!("invalid port '{value}': {err}"))
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
/// - **Device limits** — we request the adapter's full limits so compute
///   shaders (storage buffers, workgroup sizes) aren't artificially capped.
///   This is why the headless path can't use `re_viewer`'s own wgpu options:
///   those request zero compute limits.
fn full_limits_wgpu_setup() -> eframe::egui_wgpu::WgpuSetup {
    eframe::egui_wgpu::WgpuSetup::CreateNew(eframe::egui_wgpu::WgpuSetupCreateNew {
        // Use Rerun's preferred wgpu instance descriptor (Vulkan on
        // Linux, Metal on macOS).
        instance_descriptor: re_renderer::device_caps::instance_descriptor(None),
        native_adapter_selector: Some(Arc::new(move |adapters, surface| {
            re_renderer::device_caps::select_adapter(
                adapters,
                re_renderer::device_caps::instance_descriptor(None).backends,
                surface,
            )
        })),
        device_descriptor: Arc::new(|adapter| re_renderer::external::wgpu::DeviceDescriptor {
            label: Some("gsplat-rust-renderer device"),
            // Request all features the adapter supports, except
            // MAPPABLE_PRIMARY_BUFFERS which isn't needed and can
            // cause issues on some drivers.
            required_features: adapter
                .features()
                .difference(re_renderer::external::wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            // Use the adapter's full limits so our compute shaders
            // aren't restricted by the default (very conservative)
            // wgpu limits.
            required_limits: adapter.limits(),
            memory_hints: re_renderer::external::wgpu::MemoryHints::MemoryUsage,
            trace: re_renderer::external::wgpu::Trace::Off,
            experimental_features: unsafe {
                re_renderer::external::wgpu::ExperimentalFeatures::enabled()
            },
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
    fn positional_rrd_is_kept_for_startup_loading() {
        let cli = parse_cli_from(
            [
                OsString::from("--headless"),
                OsString::from("--port"),
                OsString::from("4321"),
                OsString::from("training.rrd"),
            ]
            .into_iter(),
        )
        .unwrap();

        assert_eq!(cli.rrd_path, Some(std::path::PathBuf::from("training.rrd")));
    }
}
