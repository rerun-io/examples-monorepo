//! Camera construction utilities.
//!
//! Provides constructors for [`CameraApproximation`] from different sources:
//! look-at parameters, NeRF transform matrices, or fallback bounds.

use glam::{Affine3A, Mat4, Vec2, Vec3};

use super::types::CameraApproximation;

/// Construct a synthetic camera that frames the cloud's bounding box.
/// Used before the user has interacted with the 3D camera.
pub fn fallback_camera(bounds: Option<(Vec3, Vec3)>) -> CameraApproximation {
    let (center, extent) = bounds
        .map(|(min, max)| (0.5 * (min + max), max - min))
        .unwrap_or((Vec3::ZERO, Vec3::ONE));
    let distance = extent.length().max(1.0) * 1.5;
    make_camera_approximation(
        center + Vec3::new(distance, distance * 0.5, distance),
        center,
        Vec3::Y,
        55.0_f32.to_radians(),
        Vec2::new(1600.0, 900.0),
        0.01,
    )
}

/// Helper to build a [`CameraApproximation`] from look-at parameters.
pub fn make_camera_approximation(
    world_position: Vec3,
    look_at: Vec3,
    up: Vec3,
    vertical_fov: f32,
    viewport_size_px: Vec2,
    near_plane: f32,
) -> CameraApproximation {
    let view_from_world_mat = Mat4::look_at_rh(world_position, look_at, up);
    let aspect_ratio = (viewport_size_px.x / viewport_size_px.y.max(1.0)).max(1e-4);
    let projection_from_view =
        Mat4::perspective_infinite_rh(vertical_fov, aspect_ratio, near_plane);
    CameraApproximation {
        view_from_world: Affine3A::from_mat4(view_from_world_mat),
        projection_from_view,
        world_position,
        viewport_size_px,
        near_plane,
    }
}

/// Build a [`CameraApproximation`] from a NeRF-style camera-to-world transform.
///
/// NeRF transforms JSON stores 4×4 camera-to-world (c2w) matrices in OpenGL
/// convention (Y up, -Z forward).  This function inverts to get world-to-camera,
/// and derives the projection matrix from the horizontal FOV + image dimensions.
///
/// # Arguments
///
/// * `camera_to_world` — 4×4 transform from camera local space to world space
/// * `camera_angle_x` — horizontal field of view in radians
/// * `width` — image width in pixels
/// * `height` — image height in pixels
pub fn camera_from_nerf_transform(
    camera_to_world: Mat4,
    camera_angle_x: f32,
    width: u32,
    height: u32,
) -> CameraApproximation {
    let world_to_camera = camera_to_world.inverse();
    let view_from_world = Affine3A::from_mat4(world_to_camera);
    let world_position = camera_to_world.col(3).truncate();

    let aspect = width as f32 / height as f32;
    // camera_angle_x is horizontal FOV; derive vertical FOV from aspect ratio.
    let vertical_fov = 2.0 * ((camera_angle_x / 2.0).tan() / aspect).atan();
    let near_plane = 0.01;
    let viewport_size_px = Vec2::new(width as f32, height as f32);
    let projection_from_view = Mat4::perspective_infinite_rh(vertical_fov, aspect, near_plane);

    CameraApproximation {
        view_from_world,
        projection_from_view,
        world_position,
        viewport_size_px,
        near_plane,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_camera_has_finite_fields() {
        let cam = fallback_camera(Some((Vec3::ZERO, Vec3::ONE)));
        assert!(cam.world_position.is_finite());
        assert!(cam.viewport_size_px.x > 0.0);
        assert!(cam.near_plane > 0.0);
    }

    #[test]
    fn nerf_identity_camera_looks_down_neg_z() {
        // Identity c2w means camera is at origin, looking down -Z.
        let cam = camera_from_nerf_transform(Mat4::IDENTITY, 0.6911, 800, 800);
        assert!((cam.world_position - Vec3::ZERO).length() < 1e-6);
        assert!(cam.viewport_size_px.x == 800.0);
        assert!(cam.viewport_size_px.y == 800.0);
    }
}
