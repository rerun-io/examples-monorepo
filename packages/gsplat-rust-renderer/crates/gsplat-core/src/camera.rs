//! Convert a renderer view (right/up/back) to the core camera (right/down/forward).
use crate::{Camera, CameraModel};
use glam::{Affine3A, Quat, UVec2, Vec2};

impl Camera {
    /// Camera-to-world pose from a Rerun-style RUB perspective view.
    pub fn from_view(world_from_view: Affine3A, fov_y: f64, size: UVec2) -> Self {
        let world_from_rdf =
            world_from_view * Affine3A::from_quat(Quat::from_xyzw(1.0, 0.0, 0.0, 0.0));
        let (_, rotation, position) = world_from_rdf.to_scale_rotation_translation();
        Self {
            model: CameraModel::Pinhole,
            position,
            rotation,
            fov_x: 2.0 * ((fov_y * 0.5).tan() * f64::from(size.x) / f64::from(size.y)).atan(),
            fov_y,
            center_uv: Vec2::splat(0.5),
            size,
        }
    }
    /// Transform world positions to the camera's OpenCV coordinate frame.
    pub fn world_to_local(&self) -> glam::Mat4 {
        glam::Mat4::from(
            glam::Affine3A::from_rotation_translation(self.rotation, self.position).inverse(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn view_pose_preserves_projection_and_converts_rub_to_rdf() {
        let world_from_view = Affine3A::from_rotation_translation(
            Quat::from_rotation_y(0.7),
            Vec3::new(3.0, -2.0, 1.0),
        );
        let camera = Camera::from_view(world_from_view, 1.0, UVec2::new(1920, 1080));
        camera.validate().unwrap();
        let world = world_from_view.transform_point3(Vec3::new(1.0, 2.0, -3.0));
        let rdf = camera.world_to_local().transform_point3(world);
        assert!((rdf - Vec3::new(1.0, -2.0, 3.0)).length() < 1e-5);
        assert!((camera.position - Vec3::from(world_from_view.translation)).length() < 1e-5);
        assert!((camera.focal().x - camera.focal().y).abs() < 1e-3);
        assert_eq!(camera.center_uv, Vec2::splat(0.5));
    }
}
