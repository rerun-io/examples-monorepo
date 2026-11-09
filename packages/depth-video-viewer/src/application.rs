//! Stock Rerun viewer plus the `DepthVideoStream` visualizer in the 2D and 3D spatial views.
//!
//! Supports live gRPC, saved recordings and headless screenshots with the same registration path.

use crate::visualizer::DepthVideoStreamVisualizer;
use re_viewer::external::{egui, re_log, re_memory, re_sdk_types, re_viewer_context};

use clap::Parser as _;
use re_sdk_types::View as _;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::path::PathBuf;

const VIEWER_NAME: &str = "Depth Video Viewer";

const GRPC_PORT: u16 = 9876;

/// Run the SDK-compatible custom viewer binary.
pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.print_version {
        println!("rerun {}", build_info().version);
        return Ok(());
    }

    re_log::setup_logging();
    re_crash_handler::install_crash_handlers(build_info());

    let grpc_addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, cli.port));
    re_log::info!(
        "Listening for Rerun logs on rerun+http://127.0.0.1:{}/proxy",
        cli.port
    );
    let (grpc_rx, _grpc_server_handle) = re_grpc_server::spawn_with_recv(
        re_grpc_server::ServerListener::bind(grpc_addr)?,
        re_grpc_server::ServerOptions::default(),
        re_grpc_server::shutdown::never(),
    );
    let memory_limit = cli
        .memory_limit
        .as_deref()
        .map(re_memory::MemoryLimit::parse)
        .transpose()
        .map_err(anyhow::Error::msg)?;

    let main_thread_token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();
    let startup_options = re_viewer::StartupOptions {
        persist_state: false,
        hide_welcome_screen: cli.hide_welcome_screen,
        ..Default::default()
    };
    let rrd_path = cli.rrd_path;
    let create = move |cc: &re_viewer::external::eframe::CreationContext<'_>| {
        let mut app = re_viewer::App::new(
            main_thread_token,
            build_info(),
            re_viewer::AppEnvironment::Custom(VIEWER_NAME.to_owned()),
            startup_options,
            cc,
            None, // No custom connection registry
            re_viewer::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()?,
        );
        for view in [
            re_sdk_types::blueprint::views::Spatial2DView::identifier(),
            re_sdk_types::blueprint::views::Spatial3DView::identifier(),
        ] {
            app.extend_view_class(view, register_depth_video)?;
        }
        if let Some(limit) = memory_limit {
            app.app_options_mut().memory_limit = limit;
        }
        app.add_log_receiver(grpc_rx);
        if let Some(rrd_path) = &rrd_path {
            app.open_url_or_file(&rrd_path.to_string_lossy());
        }
        anyhow::Ok(app)
    };

    if cli.headless {
        re_viewer::run_headless_app(
            Box::new(move |cc| {
                let mut app = create(cc)
                    .unwrap_or_else(|err| panic!("failed to create headless viewer app: {err}"));
                app.app_options_mut().show_notification_toasts = false;
                app
            }),
            None,
            cli.window_size,
        )
    } else {
        re_viewer::run_native_app(
            main_thread_token,
            Box::new(move |cc| Ok(Box::new(create(cc)?))),
            None,
        )
    }
    .map_err(|err| anyhow::anyhow!("{err}"))
}

/// The Rerun release this viewer embeds; 0.42 removed `re_viewer::build_info()`.
/// `re_build_info` versions in lockstep with `re_viewer`, so `CrateVersion::LOCAL` is Rerun's.
fn build_info() -> re_build_info::BuildInfo {
    re_build_info::BuildInfo {
        version: re_build_info::CrateVersion::LOCAL,
        ..re_build_info::cargo_build_info!()
    }
}

fn register_depth_video(
    registrator: &mut re_viewer_context::ViewSystemRegistrator<'_>,
) -> Result<(), re_viewer_context::ViewClassRegistryError> {
    registrator.register_visualizer::<DepthVideoStreamVisualizer>()
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

#[cfg(test)]
mod tests {
    use super::*;
    use re_viewer::external::{parking_lot, re_view_spatial};
    use re_viewer_context::{IdentifiedViewSystem as _, ViewClass as _, ViewClassRegistry};
    use std::sync::Arc;

    #[test]
    fn startup_arguments_and_sdk_flags() {
        let cli = Cli::try_parse_from([
            "viewer",
            "--headless",
            "--port=4321",
            "--window-size",
            "800x600",
            "--expect-data-soon",
            "depth.rrd",
        ])
        .unwrap();
        assert_eq!(cli.rrd_path, Some(PathBuf::from("depth.rrd")));
        assert_eq!(cli.window_size, Some(egui::vec2(800.0, 600.0)));
        assert_eq!(cli.port, 4321);
        assert!(cli.headless);
        assert!(Cli::try_parse_from(["viewer", "--window-size=0x1"]).is_err());
    }

    /// Both spatial views keep their built-in visualizers and gain `DepthVideoStream`.
    #[test]
    fn both_spatial_views_get_the_visualizer() {
        let mut registry = ViewClassRegistry::default();
        let reflection = Arc::new(parking_lot::RwLock::new(
            re_sdk_types::reflection::reflection().clone(),
        ));
        let options = Default::default();
        let mut fallbacks = Default::default();
        registry
            .add_class::<re_view_spatial::SpatialView2D>(&reflection, &options, &mut fallbacks)
            .unwrap();
        registry
            .add_class::<re_view_spatial::SpatialView3D>(&reflection, &options, &mut fallbacks)
            .unwrap();
        for view in [
            re_view_spatial::SpatialView2D::identifier(),
            re_view_spatial::SpatialView3D::identifier(),
        ] {
            registry
                .extend_class(
                    view,
                    &reflection,
                    &options,
                    &mut fallbacks,
                    register_depth_video,
                )
                .unwrap();
            let visualizers = registry.new_visualizer_collection(view);
            let ids: Vec<_> = visualizers
                .iter_with_identifiers()
                .map(|(id, _)| id)
                .collect();
            assert!(ids.contains(&DepthVideoStreamVisualizer::identifier()));
            assert!(ids.len() > 10, "built-in spatial visualizers remain");
        }
    }
}
