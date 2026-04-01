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
//! 2. **Registers a custom `GaussianSplats3D` visualizer** on the built-in
//!    `Spatial3DView`.  When the data store contains entities with the right
//!    component contract (centers, quaternions, scales, opacities, colors, and
//!    optionally SH coefficients), the custom visualizer takes over rendering
//!    using a GPU-accelerated Gaussian splatting pipeline instead of the stock
//!    point-cloud renderer.
//!
//! Everything else (UI, blueprint, timeline, selection, etc.) is inherited from
//! the stock Rerun viewer unchanged.
//!
//! # Usage
//!
//! ```bash
//! # Terminal 1 – launch the viewer:
//! cargo run --release
//!
//! # Terminal 2 – send Gaussian splat data from Python:
//! python tools/log_gaussian_ply.py --rr-config.connect
//! ```

mod gaussian_renderer;
mod gaussian_visualizer;

use re_sdk_types::View as _;
use re_viewer::external::eframe;
use std::ffi::OsString;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::num::ParseIntError;
use std::sync::Arc;

/// Name shown in the viewer title bar and "About" dialog.
const VIEWER_NAME: &str = "Gaussian Splats Viewer";

/// Default TCP port for the gRPC server that receives Rerun log data.
/// This matches Rerun's default `rr.connect_grpc()` target.
const GRPC_PORT: u16 = 9876;

/// Entry point.  We use `#[tokio::main]` because the gRPC server needs an
/// async runtime, while the viewer itself runs on the main thread (required
/// by most windowing systems).
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = parse_cli()?;
    if cli.print_version {
        println!("rerun-gs-viewer {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    // Standard Rerun logging and crash-handler setup.
    re_log::setup_logging();
    re_crash_handler::install_crash_handlers(re_viewer::build_info());

    // ── Start the gRPC server ─────────────────────────────────────────────
    // This spawns a background task that listens for incoming Rerun log
    // messages.  `grpc_rx` is a channel receiver that feeds those messages
    // into the viewer's data store.
    let grpc_addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, cli.port));
    re_log::info!(
        "Listening for Rerun logs on rerun+http://127.0.0.1:{}/proxy",
        cli.port
    );
    let grpc_rx = re_grpc_server::spawn_with_recv(
        grpc_addr,
        re_grpc_server::ServerOptions::default(),
        re_grpc_server::shutdown::never(),
    );

    // ── Create the viewer application ─────────────────────────────────────
    // `MainThreadToken` is a safety marker that proves we're on the main
    // thread, which is required by the native windowing backend.
    let main_thread_token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();
    let app_env = re_viewer::AppEnvironment::Custom(VIEWER_NAME.to_owned());
    let startup_options = re_viewer::StartupOptions {
        // Don't persist viewer state between runs — each launch starts fresh.
        persist_state: false,
        ..Default::default()
    };

    // ── Launch the native window ──────────────────────────────────────────
    // `eframe::run_native` opens the OS window and hands control to the
    // viewer's render loop.  The closure receives `cc` (creation context)
    // which provides access to the wgpu device and egui setup.
    eframe::run_native(
        "Rerun Viewer",
        native_options(),
        Box::new(move |cc| {
            // Let Rerun set up its custom wgpu renderer (re_renderer) and
            // egui integration before we create the App.
            re_viewer::customize_eframe_and_setup_renderer(cc)?;

            let mut viewer = re_viewer::App::new(
                main_thread_token,
                re_viewer::build_info(),
                app_env,
                startup_options,
                cc,
                None, // No custom StoreHub
                re_viewer::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()
                    .expect("tokio runtime should exist"),
            );

            // Wire up the gRPC channel so incoming log messages appear in
            // the viewer's data store automatically.
            viewer.add_log_receiver(grpc_rx);

            // ── Register the custom Gaussian splat visualizer ─────────
            // `extend_view_class` adds our visualizer to the existing
            // Spatial3DView.  Any entity that matches the GaussianSplats3D
            // archetype will be rendered by our custom GPU pipeline instead
            // of the stock point-cloud renderer.
            viewer.extend_view_class(
                re_sdk_types::blueprint::views::Spatial3DView::identifier(),
                |registrator| {
                    registrator
                        .register_visualizer::<gaussian_visualizer::GaussianSplatVisualizer>()?;
                    Ok(())
                },
            )?;

            Ok(Box::new(viewer))
        }),
    )
    .map_err(|err| anyhow::anyhow!(err))
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLI parsing
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimal command-line options.  We only need `--port` and `--version`;
/// everything else (memory limits, etc.) is silently ignored so the binary
/// can be used as a drop-in replacement for `rerun`.
#[derive(Clone, Copy, Debug)]
struct Cli {
    /// TCP port for the gRPC server.
    port: u16,
    /// If true, print the version string and exit.
    print_version: bool,
}

fn parse_cli() -> anyhow::Result<Cli> {
    let mut cli = Cli {
        port: GRPC_PORT,
        print_version: false,
    };
    let mut args = std::env::args_os().skip(1);

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

        // Silently ignore flags that the stock `rerun` binary accepts so
        // this binary can be used as a drop-in replacement.
        if arg == "--memory-limit" || arg == "--server-memory-limit" {
            let _ = args.next();
            continue;
        }

        if arg == "--expect-data-soon" || arg == "--hide-welcome-screen" {
            continue;
        }

        if arg.to_str().is_some_and(|arg| {
            arg.starts_with("--memory-limit=") || arg.starts_with("--server-memory-limit=")
        }) {
            continue;
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

// ═══════════════════════════════════════════════════════════════════════════════
// Window and GPU configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the `eframe::NativeOptions` that configure the OS window and the
/// wgpu (WebGPU) graphics backend.
///
/// Key choices:
/// - **VSync** (`AutoVsync`) — prevents tearing and saves power.
/// - **Adapter selection** — delegates to `re_renderer` so we pick the same
///   GPU that Rerun's internal renderer expects.
/// - **Device limits** — we request the adapter's full limits so compute
///   shaders (storage buffers, workgroup sizes) aren't artificially capped.
fn native_options() -> eframe::NativeOptions {
    let mut native_options = re_viewer::native::eframe_options(None);
    native_options.wgpu_options = eframe::egui_wgpu::WgpuConfiguration {
        present_mode: re_renderer::external::wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: None,
        on_surface_error: Arc::new(|err| {
            // On non-Windows platforms, an "Outdated" surface just means the
            // window was resized — recreate the surface and carry on.
            if err == re_renderer::external::wgpu::SurfaceError::Outdated
                && !cfg!(target_os = "windows")
            {
                eframe::egui_wgpu::SurfaceErrorAction::RecreateSurface
            } else {
                eframe::egui_wgpu::SurfaceErrorAction::SkipFrame
            }
        }),
        wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(
            eframe::egui_wgpu::WgpuSetupCreateNew {
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
                device_descriptor: Arc::new(|adapter| {
                    re_renderer::external::wgpu::DeviceDescriptor {
                        label: Some("gsplat-rust-renderer device"),
                        // Request all features the adapter supports, except
                        // MAPPABLE_PRIMARY_BUFFERS which isn't needed and can
                        // cause issues on some drivers.
                        required_features: adapter.features().difference(
                            re_renderer::external::wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                        ),
                        // Use the adapter's full limits so our compute shaders
                        // aren't restricted by the default (very conservative)
                        // wgpu limits.
                        required_limits: adapter.limits(),
                        memory_hints: re_renderer::external::wgpu::MemoryHints::MemoryUsage,
                        trace: re_renderer::external::wgpu::Trace::Off,
                        experimental_features: unsafe {
                            re_renderer::external::wgpu::ExperimentalFeatures::enabled()
                        },
                    }
                }),
                ..Default::default()
            },
        ),
    };
    native_options
}
