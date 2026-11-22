//! Registration of the compute splat renderer and visualizers, shared by the native and web viewers.

use crate::visualizer;
use re_sdk_types::View as _;
use re_viewer::external::{eframe, re_renderer, re_sdk_types, re_viewer_context};
use std::sync::Arc;

/// Register the compute splat renderer and the `Spatial3DView` visualizers on a stock viewer.
pub fn register_compute_splats(viewer: &mut re_viewer::App) -> anyhow::Result<()> {
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
    Ok(())
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
        device_descriptor: compute_device_descriptor(gsplat_core::subgroups_allowed()),
        ..eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle()
    })
}

/// Rerun's device descriptor plus the limits the splat kernels need, when the adapter has them.
///
/// `SUBGROUP` is requested when the adapter offers it and `allow_subgroups` is set; without it the
/// splat kernels take their workgroup-memory variant (the web viewer's `?force_no_subgroups=1`).
pub fn compute_device_descriptor(
    allow_subgroups: bool,
) -> Arc<dyn Fn(&wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> + Send + Sync> {
    Arc::new(move |adapter| {
        let mut descriptor =
            re_renderer::device_caps::DeviceCaps::from_adapter_without_validation(adapter)
                .device_descriptor();
        descriptor.label = Some("Gaussian compute viewer");
        let available = adapter.limits();
        if gsplat_core::check_adapter(&available).is_ok() {
            if allow_subgroups {
                descriptor.required_features |= adapter.features() & wgpu::Features::SUBGROUP;
            }
            if crate::profile::requested() {
                descriptor.required_features |=
                    adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
            }
            descriptor.required_limits =
                gsplat_core::compute_limits(&available, descriptor.required_limits);
        }
        descriptor.memory_hints = wgpu::MemoryHints::MemoryUsage;
        descriptor
    })
}

pub(crate) fn register_splat_system(
    registrator: &mut re_viewer_context::ViewSystemRegistrator<'_>,
) -> Result<(), re_viewer_context::ViewClassRegistryError> {
    crate::bounds::register(registrator);
    // A new view default or override of the render mode starts at the mode used without one.
    registrator.register_fallback_provider(visualizer::render_mode_descriptor().component, |_| {
        re_sdk_types::components::Text::from("default")
    });
    registrator.register_context_system::<crate::selection::AutomaticComputeSelection>()?;
    registrator.register_visualizer::<visualizer::GaussianSplatVisualizer>()?;
    registrator.register_visualizer::<crate::hexels::HexelVisualizer>()
}
