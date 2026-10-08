//! Bounds for default/reset framing through public view fallback providers.
use crate::cache::GpuCache;
use glam::Vec3;
use re_sdk_types::{blueprint::archetypes::EyeControls3D, components::Position3D};
use re_viewer_context::{QueryContext, ViewStateExt as _, ViewSystemRegistrator};

pub fn register(registry: &mut ViewSystemRegistrator<'_>) {
    registry.register_fallback_provider(EyeControls3D::descriptor_look_target().component, |ctx| {
        let [min, max] = scene_bounds(ctx);
        Position3D::from((min + max) * 0.5)
    });
    registry.register_fallback_provider(EyeControls3D::descriptor_position().component, |ctx| {
        let [min, max] = scene_bounds(ctx);
        let center = (min + max) * 0.5;
        let state = ctx
            .view_state()
            .downcast_ref::<re_view_spatial::SpatialViewState>()
            .ok();
        let up = state
            .and_then(|s| s.state_3d.scene_view_coordinates.up())
            .map(Vec3::from)
            .unwrap_or(Vec3::Z);
        let right = state
            .and_then(|s| s.state_3d.scene_view_coordinates.right())
            .map(Vec3::from)
            .unwrap_or(Vec3::X);
        let direction = (right - 3.0 * up.cross(right) + up).normalize_or(Vec3::Z);
        Position3D::from(center + direction * (max - min).length().max(1.0) * 1.5)
    });
}
fn scene_bounds(ctx: &QueryContext<'_>) -> [Vec3; 2] {
    let splats = ctx
        .store_ctx()
        .memoizer::<GpuCache, _>(|cache| cache.view_bounds(ctx.view_ctx.view_id));
    let native = ctx
        .view_state()
        .downcast_ref::<re_view_spatial::SpatialViewState>()
        .ok()
        .map(|s| [s.bounding_boxes.current.min, s.bounding_boxes.current.max])
        .filter(|[min, max]| min.is_finite() && max.is_finite() && min.cmple(*max).all());
    match (splats, native) {
        (Some(a), Some(b)) => [a[0].min(b[0]), a[1].max(b[1])],
        (Some(b), _) | (_, Some(b)) => b,
        _ => [Vec3::splat(-0.5), Vec3::splat(0.5)],
    }
}

/// Include the visible three-sigma radius when framing compute or fallback splats.
pub(crate) fn from_splats(cloud: &gsplat_core::native::NativeSplats<'_>) -> [Vec3; 2] {
    let mut bounds = [Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)];
    for (i, center) in cloud.centers.iter().enumerate() {
        let center = Vec3::from_array(*center);
        let radius = Vec3::from_array(gsplat_core::native::at_or_last(cloud.scales, i, [0.01; 3]))
            .abs()
            .max_element()
            * 3.0;
        bounds[0] = bounds[0].min(center - radius);
        bounds[1] = bounds[1].max(center + radius);
    }
    bounds
}

/// Transform all eight corners; negative scale and rotation are supported.
pub(crate) fn transformed(bounds: [Vec3; 2], transform: glam::Affine3A) -> [Vec3; 2] {
    let mut result = [Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)];
    for bits in 0..8 {
        let p = transform.transform_point3(Vec3::new(
            bounds[bits & 1].x,
            bounds[(bits >> 1) & 1].y,
            bounds[(bits >> 2) & 1].z,
        ));
        result[0] = result[0].min(p);
        result[1] = result[1].max(p);
    }
    result
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
        let [min, max] = transformed([Vec3::ZERO, Vec3::ONE], transform);
        assert!(min.abs_diff_eq(Vec3::new(0.9999, 0.0, 0.0), 1e-7));
        assert!(max.abs_diff_eq(Vec3::new(1.0, 0.0001, 0.0001), 1e-7));
    }
}
