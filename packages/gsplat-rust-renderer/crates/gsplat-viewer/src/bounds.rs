//! Bounds for default/reset framing through public view fallback providers.
use crate::cache::GpuCache;
use glam::Vec3;
use re_sdk_types::{blueprint::archetypes::EyeControls3D, components::Position3D};
use re_viewer::external::{re_log, re_renderer, re_sdk_types, re_view_spatial, re_viewer_context};
use re_viewer_context::{QueryContext, ViewStateExt as _, ViewSystemRegistrator};

pub(crate) fn register(registry: &mut ViewSystemRegistrator<'_>) {
    registry.register_fallback_provider(EyeControls3D::descriptor_look_target().component, |ctx| {
        let center = scene_bounds(ctx).center();
        Position3D::from(if center.is_finite() {
            center
        } else {
            Vec3::ZERO
        })
    });
    registry.register_fallback_provider(EyeControls3D::descriptor_position().component, |ctx| {
        let Ok(state) = ctx
            .view_state()
            .downcast_ref::<re_view_spatial::SpatialViewState>()
        else {
            re_log::error_once!("Eye position fallback queried outside a spatial view");
            return Position3D::ZERO;
        };
        let bounds = scene_bounds(ctx);
        let mut center = bounds.center();
        if !center.is_finite() {
            center = Vec3::ZERO;
        }
        let mut radius = 1.5 * bounds.half_size().length();
        if !radius.is_finite() || radius == 0.0 {
            radius = 1.0;
        }
        // EyeController is private in 0.38.1; MIN_ORBIT_DISTANCE is twice its 0.01 near plane.
        radius = radius.max(0.02);
        let coordinates = state.state_3d.scene_view_coordinates;
        let right = coordinates.right().map(Vec3::from).unwrap_or(Vec3::X);
        let forward = coordinates.forward().map(Vec3::from).unwrap_or(Vec3::Y);
        let up = coordinates.up().map(Vec3::from).unwrap_or(Vec3::Z);
        let direction = (0.75 * up.cross(right) + 0.25 * right - 0.25 * up)
            .try_normalize()
            .unwrap_or(forward);
        Position3D::from(center - radius * direction)
    });
}

fn scene_bounds(ctx: &QueryContext<'_>) -> macaw::BoundingBox {
    let splats = ctx
        .store_ctx()
        .memoizer::<GpuCache, _>(|cache| cache.view_bounds(ctx.view_ctx.view_id));
    let native = ctx
        .view_state()
        .downcast_ref::<re_view_spatial::SpatialViewState>()
        .ok()
        .map(|s| s.bounding_boxes.region_of_interest_current)
        .filter(|b| b.min.is_finite() && b.max.is_finite());
    match (splats, native) {
        (Some(a), Some(b)) => a.union(b),
        (Some(b), _) | (_, Some(b)) => b,
        _ => macaw::BoundingBox::nothing(),
    }
}

/// Match the stock splat visualizer: ignore nearly transparent and nonfinite centers.
pub(crate) fn from_splats(
    cloud: &gsplat_core::native::NativeSplats<'_>,
) -> re_renderer::RobustBounds {
    visible_bounds(cloud.centers, |i| {
        gsplat_core::native::at_or_last(cloud.colors, i, 0xFFFFFFFF) as u8 >= 5
    })
}

/// Hexels: same rule on the activated opacity.
pub(crate) fn from_hexels(
    hexels: &gsplat_core::native::NativeHexels<'_>,
) -> re_renderer::RobustBounds {
    visible_bounds(hexels.centers, |i| {
        gsplat_core::native::at_or_last(hexels.opacities, i, 1.0) * 255.0 >= 5.0
    })
}

fn visible_bounds(
    centers: &[[f32; 3]],
    visible: impl Fn(usize) -> bool,
) -> re_renderer::RobustBounds {
    re_tracing::profile_scope!("bounds");
    let centers: Vec<_> = centers
        .iter()
        .enumerate()
        .filter_map(|(i, center)| {
            let center = Vec3::from_array(*center);
            (visible(i) && center.is_finite()).then_some(center)
        })
        .collect();
    re_renderer::RobustBounds::from_points(&centers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_follow_tiny_and_negative_instance_scales() {
        let transform = glam::Affine3A::from_scale_rotation_translation(
            Vec3::new(-0.0001, 0.0001, 0.0001),
            glam::Quat::IDENTITY,
            Vec3::X,
        );
        let bounds =
            macaw::BoundingBox::from_min_max(Vec3::ZERO, Vec3::ONE).transform_affine3(&transform);
        assert!(bounds.min.abs_diff_eq(Vec3::new(0.9999, 0.0, 0.0), 1e-7));
        assert!(bounds.max.abs_diff_eq(Vec3::new(1.0, 0.0001, 0.0001), 1e-7));
    }
}
