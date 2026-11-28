//! Registration of the compute splat renderer and visualizers, shared by the canonical viewer ([`EXTENSION`]) and the
//! gsplat native and web viewers.

use crate::visualizer;
use canonical_viewer_extension::Extension;
use re_sdk_types::View as _;
use re_viewer::external::{re_sdk_types, re_viewer_context};

/// The gsplat extension of the canonical viewer: the compute splat renderer, its visualizers and its device needs.
pub const EXTENSION: Extension = Extension {
    name: "gsplat",
    device_needs: Some(|offered_features, offered_limits, wanted| {
        // The device is requested before any App exists, so the page flags are read here.
        #[cfg(target_arch = "wasm32")]
        crate::web::enable_profile_from_page();
        apply_compute_needs(
            offered_features,
            offered_limits,
            subgroups_allowed(),
            wanted,
        );
    }),
    // Default framing on splat and hexel bounds (crate::bounds), which Rerun's spatial views cannot see.
    replaces_fallbacks: &[
        ("3D", "EyeControls3D:position"),
        ("3D", "EyeControls3D:look_target"),
    ],
    register: register_compute_splats,
};

/// `GSPLAT_NO_SUBGROUPS=1` withholds subgroups natively; on the web, the page flag `force_no_subgroups=1` does.
fn subgroups_allowed() -> bool {
    #[cfg(target_arch = "wasm32")]
    return !crate::web::query_flag("force_no_subgroups");
    #[cfg(not(target_arch = "wasm32"))]
    return gsplat_core::subgroups_allowed();
}

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

/// Add the splat kernels' features and limits to `descriptor`, out of what the adapter offers.
fn apply_compute_needs(
    offered_features: &wgpu::Features,
    offered_limits: &wgpu::Limits,
    allow_subgroups: bool,
    descriptor: &mut wgpu::DeviceDescriptor<'static>,
) {
    if gsplat_core::check_adapter(offered_limits).is_ok() {
        if allow_subgroups {
            descriptor.required_features |= *offered_features & wgpu::Features::SUBGROUP;
        }
        if crate::profile::requested() {
            descriptor.required_features |= *offered_features & wgpu::Features::TIMESTAMP_QUERY;
        }
        descriptor.required_limits =
            gsplat_core::compute_limits(offered_limits, descriptor.required_limits.clone());
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use re_sdk_types::blueprint::{archetypes::EyeControls3D, views::Spatial3DView};

    #[test]
    fn replaced_fallbacks_name_what_bounds_registers() {
        let view = Spatial3DView::identifier();
        let expected = [
            EyeControls3D::descriptor_position().component,
            EyeControls3D::descriptor_look_target().component,
        ]
        .map(|component| (view.as_str().to_owned(), component.as_str().to_owned()));
        let declared = EXTENSION
            .replaces_fallbacks
            .iter()
            .map(|(view, component)| ((*view).to_owned(), (*component).to_owned()));
        assert!(declared.eq(expected));
    }
}
