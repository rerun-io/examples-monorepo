//! The headless viewer App that `gsplat probe` measures.
//!
//! The stock Rerun viewer with the compute splat systems registered. The canonical viewer (packages/canonical-viewer)
//! is the viewer people run; this module goes when the probe moves there.

use re_viewer::external::{eframe, egui, re_log_channel};
use std::path::PathBuf;

/// The Rerun release this viewer embeds; 0.42 removed `re_viewer::build_info()`.
/// `re_build_info` versions in lockstep with `re_viewer`, so `CrateVersion::LOCAL` is Rerun's.
fn build_info() -> re_build_info::BuildInfo {
    re_build_info::BuildInfo {
        version: re_build_info::CrateVersion::LOCAL,
        ..re_build_info::cargo_build_info!()
    }
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
        build_info(),
        app_env,
        startup_options,
        cc,
        None, // No custom connection registry
        re_viewer::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()?,
    );

    crate::setup::register_compute_splats(&mut viewer)?;

    viewer.add_log_receiver(grpc_rx);
    if let Some(rrd_path) = rrd_path {
        viewer.open_url_or_file(&rrd_path.to_string_lossy());
    }

    Ok(viewer)
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

pub use crate::setup::compute_wgpu_setup;

#[cfg(test)]
mod selection_tests {
    use re_viewer::external::parking_lot;
    use re_viewer::external::{re_sdk_types, re_view_spatial, re_viewer_context};
    use re_viewer_context::{ViewClass as _, ViewClassRegistry};
    use std::sync::Arc;

    #[test]
    fn custom_spatial_view_keeps_both_explicit_choices() {
        let mut registry = ViewClassRegistry::default();
        let reflection = Arc::new(parking_lot::RwLock::new(
            re_sdk_types::reflection::reflection().clone(),
        ));
        let options = Default::default();
        let mut fallbacks = Default::default();
        registry
            .add_class::<re_view_spatial::SpatialView3D>(&reflection, &options, &mut fallbacks)
            .unwrap();
        registry
            .extend_class(
                re_view_spatial::SpatialView3D::identifier(),
                &reflection,
                &options,
                &mut fallbacks,
                crate::setup::register_splat_system,
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
