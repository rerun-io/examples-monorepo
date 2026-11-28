//! Rerun viewer extension that draws depth stored as AV1 10-bit monochrome video (`DepthVideoStream`)
//! in the built-in 2D and 3D spatial views.
//!
//! The archetype, decoder and visualizer are out-of-tree scaffolding for checking depth as video end to end;
//! they are deleted once `DepthVideoStream` ships in Rerun.
mod archetype;
mod decoder;
mod visualizer;

use canonical_viewer_extension::Extension;
use re_sdk_types::View as _;
use re_viewer::external::{re_sdk_types, re_viewer_context};

/// The depth-video extension of the canonical viewer: the `DepthVideoStream` visualizer in the 2D and 3D views.
pub const EXTENSION: Extension = Extension {
    name: "depth-video",
    device_needs: None,
    replaces_fallbacks: &[],
    register: register_depth_video,
};

/// Add the `DepthVideoStream` visualizer to `Spatial2DView` and `Spatial3DView`.
fn register_depth_video(app: &mut re_viewer::App) -> anyhow::Result<()> {
    for view in [
        re_sdk_types::blueprint::views::Spatial2DView::identifier(),
        re_sdk_types::blueprint::views::Spatial3DView::identifier(),
    ] {
        app.extend_view_class(view, register_visualizer)?;
    }
    Ok(())
}

fn register_visualizer(
    registrator: &mut re_viewer_context::ViewSystemRegistrator<'_>,
) -> Result<(), re_viewer_context::ViewClassRegistryError> {
    registrator.register_visualizer::<visualizer::DepthVideoStreamVisualizer>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_viewer::external::{parking_lot, re_view_spatial};
    use re_viewer_context::{IdentifiedViewSystem as _, ViewClass as _, ViewClassRegistry};
    use std::sync::Arc;

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
                    register_visualizer,
                )
                .unwrap();
            let visualizers = registry.new_visualizer_collection(view);
            let ids: Vec<_> = visualizers
                .iter_with_identifiers()
                .map(|(id, _)| id)
                .collect();
            assert!(ids.contains(&visualizer::DepthVideoStreamVisualizer::identifier()));
            assert!(ids.len() > 10, "built-in spatial visualizers remain");
        }
    }
}
