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
    pub fn view_from_world(&self) -> glam::Mat4 {
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
        let rdf = camera.view_from_world().transform_point3(world);
        assert!((rdf - Vec3::new(1.0, -2.0, 3.0)).length() < 1e-5);
        assert!((camera.position - Vec3::from(world_from_view.translation)).length() < 1e-5);
        assert!((camera.focal().x - camera.focal().y).abs() < 1e-3);
        assert_eq!(camera.center_uv, Vec2::splat(0.5));
    }
    #[test]
    fn camera_validation_rejects_non_unit_rotation() {
        let mut camera = crate::test_utils::pinhole_camera(64);
        camera.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 2.0);
        assert!(camera.validate().is_err());
        camera.rotation = Quat::from_rotation_y(0.7);
        assert!(camera.validate().is_ok());
    }

    #[test]
    fn forward_lenses_reject_fov_at_or_above_pi() {
        for model in [
            CameraModel::Pinhole,
            CameraModel::RadialTangential8 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                k4: 0.0,
                k5: 0.0,
                k6: 0.0,
                p1: 0.0,
                p2: 0.0,
            },
        ] {
            for fov in [std::f64::consts::PI, 4.0] {
                let mut camera = crate::test_utils::pinhole_camera(64);
                camera.model = model;
                camera.fov_x = fov;
                assert!(camera.validate().is_err());
                camera.fov_x = 1.0;
                camera.fov_y = fov;
                assert!(camera.validate().is_err());
            }
        }
        let mut camera = crate::test_utils::pinhole_camera(64);
        camera.model = CameraModel::KannalaBrandt4 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        };
        camera.fov_x = 4.0;
        camera.fov_y = 4.0;
        assert!(camera.validate().is_ok());
    }

    #[test]
    fn camera_validation_rejects_non_positive_or_non_finite_focal() {
        let mut camera = crate::test_utils::pinhole_camera(64);
        camera.model = CameraModel::KannalaBrandt4 {
            k1: -1.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        };
        for fov in [2.0, 3.0] {
            // Zero, then negative projected radius.
            camera.fov_x = fov;
            assert!(camera.validate().is_err());
        }
        camera.model = CameraModel::Pinhole;
        camera.fov_x = f64::MIN_POSITIVE; // f64 focal overflows when narrowed to f32.
        assert!(camera.validate().is_err());
    }
}
