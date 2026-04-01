//! Stock Rerun viewer plus one extra Gaussian splat visualizer.
//!
//! This binary does exactly two non-standard things:
//! - listen for normal Rerun gRPC logs on `127.0.0.1:9876`
//! - register one custom Gaussian splat visualizer on the built-in `Spatial3DView`

mod gaussian_renderer;
mod gaussian_visualizer;

use re_sdk_types::View as _;
use re_viewer::external::eframe;
use std::ffi::OsString;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::num::ParseIntError;
use std::sync::Arc;

const VIEWER_NAME: &str = "Gaussian Splats Viewer";
const GRPC_PORT: u16 = 9876;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = parse_cli()?;
    if cli.print_version {
        println!("rerun-gs-viewer {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    re_log::setup_logging();
    re_crash_handler::install_crash_handlers(re_viewer::build_info());

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

    let main_thread_token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();
    let app_env = re_viewer::AppEnvironment::Custom(VIEWER_NAME.to_owned());
    let startup_options = re_viewer::StartupOptions {
        persist_state: false,
        ..Default::default()
    };

    eframe::run_native(
        "Rerun Viewer",
        native_options(),
        Box::new(move |cc| {
            re_viewer::customize_eframe_and_setup_renderer(cc)?;

            let mut viewer = re_viewer::App::new(
                main_thread_token,
                re_viewer::build_info(),
                app_env,
                startup_options,
                cc,
                None,
                re_viewer::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()
                    .expect("tokio runtime should exist"),
            );
            viewer.add_log_receiver(grpc_rx);
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

#[derive(Clone, Copy, Debug)]
struct Cli {
    port: u16,
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

fn native_options() -> eframe::NativeOptions {
    let mut native_options = re_viewer::native::eframe_options(None);
    native_options.wgpu_options = eframe::egui_wgpu::WgpuConfiguration {
        present_mode: re_renderer::external::wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: None,
        on_surface_error: Arc::new(|err| {
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
                        label: Some("rerun-simple-gs device"),
                        required_features: adapter.features().difference(
                            re_renderer::external::wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
                        ),
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
