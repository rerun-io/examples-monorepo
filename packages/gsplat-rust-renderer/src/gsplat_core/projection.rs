//! Gaussian projection from 3D world space to 2D screen space.
//!
//! Implements the CPU-side projection used by both the software rasterizer and
//! the Rerun viewer's CPU fallback path.  The math matches `gaussian_project.wgsl`
//! and the [Brush](https://github.com/ArthurBrussee/brush) renderer.

use glam::{Mat3, Quat, Vec2, Vec3};

use super::constants::{BRUSH_VISIBILITY_ALPHA_THRESHOLD, MIN_RADIUS_PX, SIGMA_COVERAGE};
use super::covariance::{
    brush_bbox_extent_px, brush_covariance_in_pixels, compensate_covariance_px,
    pixel_covariance_to_ndc, regularize_covariance,
};
use super::types::{CameraApproximation, PreparedSplat, ProjectedGaussian};

/// Normalize a quaternion, returning the identity quaternion if the input is
/// near-zero (degenerate).
pub fn normalize_quat_or_identity(quat: Quat) -> Quat {
    if quat.length_squared() > 1e-12 {
        quat.normalize()
    } else {
        Quat::IDENTITY
    }
}

/// Project a single 3D Gaussian onto the screen using the CPU fallback path.
///
/// This implements the **Brush-style local linearization**: build the 3D
/// covariance matrix from rotation + scale, project it into pixel space via
/// the camera Jacobian, derive a conservative 2D bounding box, then convert
/// to the NDC representation used by the instanced-quad shader.
///
/// Returns `None` if the splat is invisible (behind camera, too small, off-screen, etc.).
pub fn project_gaussian_to_ndc(
    camera: &CameraApproximation,
    mean_world: Vec3,
    quat_world: Quat,
    scales_world: Vec3,
    opacity: f32,
) -> Option<ProjectedGaussian> {
    let mean_view = camera.view_from_world.transform_point3(mean_world);
    let camera_depth = -mean_view.z;
    if !camera_depth.is_finite() || camera_depth <= camera.near_plane {
        return None;
    }

    let opacity = opacity.clamp(0.0, 1.0);
    if opacity < BRUSH_VISIBILITY_ALPHA_THRESHOLD {
        return None;
    }

    // ── Compute pixel-space mean ──────────────────────────────────────
    let viewport_size_px = camera.viewport_size_px.max(Vec2::ONE);
    let pixel_center = viewport_size_px * 0.5;
    // Extract focal lengths from the projection matrix (in NDC units), then
    // scale to pixels.
    let focal_ndc = Vec2::new(
        camera.projection_from_view.x_axis.x,
        camera.projection_from_view.y_axis.y,
    );
    let focal_px = focal_ndc * pixel_center;
    let mean_camera = Vec3::new(mean_view.x, mean_view.y, camera_depth);
    let mean_px = focal_px * mean_camera.truncate() / camera_depth + pixel_center;
    if !mean_px.is_finite() {
        return None;
    }

    // ── Build 3D covariance and project to 2D ─────────────────────────
    // Covariance = R * diag(s²) * Rᵀ  where R is the rotation matrix and
    // s is the per-axis scale vector.
    let rotation = Mat3::from_quat(normalize_quat_or_identity(quat_world));
    let scale_diag = Mat3::from_diagonal(scales_world.max(Vec3::splat(1e-6)).powf(2.0));
    let covariance_world = rotation * scale_diag * rotation.transpose();
    // Project the 3D covariance into pixel space using the camera Jacobian,
    // then add a small blur to prevent aliasing.
    let covariance_px = compensate_covariance_px(brush_covariance_in_pixels(
        camera,
        covariance_world,
        mean_camera,
        focal_px,
        pixel_center,
    ));

    // ── Compute screen-space extent ───────────────────────────────────
    // `power_threshold` determines how far from the center we extend the
    // bounding box.  It's derived from opacity so that nearly-transparent
    // splats get smaller footprints.
    let power_threshold = (opacity * 255.0).ln();
    if !power_threshold.is_finite() || power_threshold <= 0.0 {
        return None;
    }

    let extent_px = brush_bbox_extent_px(covariance_px, power_threshold) * (SIGMA_COVERAGE / 3.0);
    let px_radius = extent_px.max_element();
    if !px_radius.is_finite() || px_radius < MIN_RADIUS_PX {
        return None;
    }

    // Reject splats whose bounding box is entirely off-screen.
    if mean_px.x + extent_px.x <= 0.0
        || mean_px.x - extent_px.x >= viewport_size_px.x
        || mean_px.y + extent_px.y <= 0.0
        || mean_px.y - extent_px.y >= viewport_size_px.y
    {
        return None;
    }

    // ── Convert to NDC for the instanced-quad shader ──────────────────
    let ndc_per_pixel = Vec2::new(2.0 / viewport_size_px.x, 2.0 / viewport_size_px.y);
    let center_ndc = (mean_px - pixel_center) * ndc_per_pixel;
    let covariance_ndc =
        regularize_covariance(pixel_covariance_to_ndc(covariance_px, viewport_size_px));
    let extent_ndc = extent_px * ndc_per_pixel;
    let clip = camera.projection_from_view * mean_view.extend(1.0);
    let inv_cov = covariance_ndc.inverse();

    Some(ProjectedGaussian {
        center_ndc,
        ndc_depth: clip.z / clip.w,
        inv_cov_ndc_xx_xy_yy: [inv_cov.x_axis.x, inv_cov.y_axis.x, inv_cov.y_axis.y],
        radius_ndc: extent_ndc.max_element(),
    })
}

/// Combine a projected Gaussian with its final color and opacity into a
/// [`PreparedSplat`] ready for GPU upload.
pub fn build_prepared_splat(
    projected: ProjectedGaussian,
    color_rgb: [f32; 3],
    opacity: f32,
) -> PreparedSplat {
    PreparedSplat {
        center_ndc: projected.center_ndc,
        ndc_depth: projected.ndc_depth,
        inv_cov_ndc_xx_xy_yy: projected.inv_cov_ndc_xx_xy_yy,
        radius_ndc: projected.radius_ndc,
        color_rgb,
        opacity,
    }
}
