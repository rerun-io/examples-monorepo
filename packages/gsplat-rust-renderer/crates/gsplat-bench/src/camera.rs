//! Camera paths generated only for benchmark measurements.
use glam::Mat4;
use gsplat_render::camera::CameraSpec;

pub fn orbit(
    center: glam::Vec3,
    radius: glam::Vec2,
    elevation: f32,
    count: usize,
    template: &CameraSpec,
    up: Option<glam::Vec3>,
) -> Vec<CameraSpec> {
    let basis = up.map(|up| {
        let rotation = glam::Quat::from_rotation_arc(glam::Vec3::Z, up.normalize());
        Mat4::from_translation(center) * Mat4::from_quat(rotation) * Mat4::from_translation(-center)
    });
    (0..count)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / count as f32;
            let position =
                center + glam::Vec3::new(radius.x * angle.cos(), radius.y * angle.sin(), elevation);
            let pose = Mat4::look_at_lh(position, center, -glam::Vec3::Z).inverse();
            let pose = basis.map_or(pose, |basis| basis * pose);
            CameraSpec {
                world_from_camera: pose.transpose().to_cols_array_2d(),
                ..template.clone()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gsplat_core::CameraModel;
    #[test]
    fn orbit_looks_at_center_and_closes() {
        let template = CameraSpec {
            world_from_camera: Mat4::IDENTITY.to_cols_array_2d(),
            width: 640,
            height: 480,
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            model: CameraModel::Pinhole,
        };
        let center = glam::Vec3::new(1.0, 2.0, 3.0);
        let path = orbit(center, glam::Vec2::new(4.0, 2.0), 1.0, 4, &template, None);
        assert_eq!(path.len(), 4);
        for camera in &path {
            assert!(
                camera
                    .project(center)
                    .abs_diff_eq(glam::Vec2::new(320.0, 240.0), 1e-3)
            );
        }
        assert!(
            path[0]
                .pose()
                .w_axis
                .truncate()
                .abs_diff_eq(center + glam::Vec3::new(4.0, 0.0, 1.0), 1e-5)
        );
    }
}
