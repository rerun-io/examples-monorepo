//! 2D covariance math for Gaussian projection.
//!
//! Projects a 3D Gaussian covariance matrix into 2D pixel space using the
//! Brush-style local linearization (camera Jacobian).  These functions are
//! shared between the CPU software rasterizer and the GPU compute shaders
//! (which implement the same math in WGSL).

use glam::{Affine3A, Mat2, Mat3, Vec2, Vec3};

use super::constants::BRUSH_COVARIANCE_BLUR_PX;
use super::types::CameraApproximation;

/// Ensure a 2×2 covariance matrix is positive semi-definite by clamping
/// diagonal elements to a small positive minimum and symmetrizing the
/// off-diagonal.
pub fn regularize_covariance(covariance: Mat2) -> Mat2 {
    let off_diagonal = 0.5 * (covariance.x_axis.y + covariance.y_axis.x);
    Mat2::from_cols(
        Vec2::new(covariance.x_axis.x.max(1e-8), off_diagonal),
        Vec2::new(off_diagonal, covariance.y_axis.y.max(1e-8)),
    )
}

/// Extract the 3×3 linear (rotation+scale) part from an affine transform.
pub fn mat3_from_affine(affine: Affine3A) -> Mat3 {
    Mat3::from_cols(
        affine.matrix3.x_axis.into(),
        affine.matrix3.y_axis.into(),
        affine.matrix3.z_axis.into(),
    )
}

/// Project a 3D covariance matrix into 2D pixel space using the Brush-style
/// local linearization.
///
/// The math: transform the world-space covariance into view space, compute
/// the camera Jacobian (partial derivatives of the pixel projection at the
/// splat's mean), and apply the Jacobian to get a 2D covariance in pixels.
///
/// `J * Σ_view * Jᵀ` where J is the 2×3 Jacobian and Σ_view is the 3×3
/// view-space covariance.
pub fn brush_covariance_in_pixels(
    camera: &CameraApproximation,
    covariance_world: Mat3,
    mean_camera: Vec3,
    focal_px: Vec2,
    pixel_center: Vec2,
) -> Mat2 {
    // Transform covariance from world to view space: Σ_view = V * Σ_world * Vᵀ
    let view_linear = mat3_from_affine(camera.view_from_world);
    let covariance_view = view_linear * covariance_world * view_linear.transpose();
    // Compute the Jacobian rows of the pixel projection at the splat mean.
    let [row0, row1] = brush_camera_jacobian_rows(
        mean_camera,
        focal_px,
        camera.viewport_size_px.max(Vec2::ONE),
        pixel_center,
    );
    // Apply the Jacobian: Σ_px = J * Σ_view * Jᵀ
    let cov_times_row0 = covariance_view * row0;
    let cov_times_row1 = covariance_view * row1;
    regularize_covariance(Mat2::from_cols(
        Vec2::new(row0.dot(cov_times_row0), row0.dot(cov_times_row1)),
        Vec2::new(row1.dot(cov_times_row0), row1.dot(cov_times_row1)),
    ))
}

/// Compute the two rows of the camera projection Jacobian at a given view-space
/// point.  This is the derivative of `pixel_position(x, y, z)` with respect to
/// the view-space coordinates, evaluated at `mean_camera`.
///
/// The UV coordinates are clamped to a slightly larger-than-viewport range to
/// prevent numerical issues for splats at the very edge of the screen.
pub fn brush_camera_jacobian_rows(
    mean_camera: Vec3,
    focal_px: Vec2,
    viewport_size_px: Vec2,
    pixel_center: Vec2,
) -> [Vec3; 2] {
    let lims_pos = (1.15 * viewport_size_px - pixel_center) / focal_px.max(Vec2::splat(1e-6));
    let lims_neg = (-0.15 * viewport_size_px - pixel_center) / focal_px.max(Vec2::splat(1e-6));
    let uv = mean_camera.truncate() / mean_camera.z.max(1e-6);
    let uv_clipped = uv.clamp(lims_neg, lims_pos);
    let duv_dxy = focal_px / mean_camera.z.max(1e-6);
    [
        Vec3::new(duv_dxy.x, 0.0, -duv_dxy.x * uv_clipped.x),
        Vec3::new(0.0, duv_dxy.y, -duv_dxy.y * uv_clipped.y),
    ]
}

/// Add a small anti-aliasing blur to the projected covariance.
/// This prevents infinitely sharp splats from creating Moiré patterns.
pub fn compensate_covariance_px(covariance_px: Mat2) -> Mat2 {
    let mut compensated = regularize_covariance(covariance_px);
    compensated.x_axis.x += BRUSH_COVARIANCE_BLUR_PX;
    compensated.y_axis.y += BRUSH_COVARIANCE_BLUR_PX;
    regularize_covariance(compensated)
}

/// Compute the axis-aligned bounding box extent (in pixels) of a 2D Gaussian
/// given its covariance and a power threshold.
///
/// The extent is `sqrt(2 * threshold * variance)` along each axis.
pub fn brush_bbox_extent_px(covariance_px: Mat2, power_threshold: f32) -> Vec2 {
    Vec2::new(
        (2.0 * power_threshold * covariance_px.x_axis.x)
            .max(0.0)
            .sqrt(),
        (2.0 * power_threshold * covariance_px.y_axis.y)
            .max(0.0)
            .sqrt(),
    )
}

/// Scale a pixel-space covariance matrix to NDC (normalized device coordinates).
/// NDC ranges from `[-1, 1]` across the viewport, so the scale factor is
/// `2 / viewport_size` along each axis.
pub fn pixel_covariance_to_ndc(covariance_px: Mat2, viewport_size_px: Vec2) -> Mat2 {
    let scale = Vec2::new(
        2.0 / viewport_size_px.x.max(1.0),
        2.0 / viewport_size_px.y.max(1.0),
    );
    let xx = covariance_px.x_axis.x * scale.x * scale.x;
    let xy = covariance_px.x_axis.y * scale.x * scale.y;
    let yy = covariance_px.y_axis.y * scale.y * scale.y;
    Mat2::from_cols(Vec2::new(xx, xy), Vec2::new(xy, yy))
}
