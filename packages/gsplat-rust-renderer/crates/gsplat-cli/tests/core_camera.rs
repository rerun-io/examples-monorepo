//! Camera arithmetic against the pinned Brush oracle.
use brush_render::kernels::camera_model::{
    CameraModel as BrushModel, kannala_brandt_4::KannalaBrandt4Params,
    radial_tangential_8::RadialTangential8Params,
};
use glam::{Vec2, Vec4};
use gsplat_core::{Camera, CameraModel};

#[test]
fn world_to_camera_matches_brush_for_oblique_poses() {
    for angle in [0.1, 0.37, 0.81, 1.43] {
        let camera = Camera {
            position: glam::Vec3::new(1.234, -2.567, 0.392),
            rotation: glam::Quat::from_euler(glam::EulerRot::XYZ, angle, -0.71, 1.2),
            fov_x: 0.8,
            fov_y: 0.6,
            center_uv: Vec2::new(0.48, 0.52),
            size: glam::uvec2(1280, 720),
            model: CameraModel::Pinhole,
        };
        let brush = brush_render::camera::Camera::new(
            camera.position,
            camera.rotation,
            camera.fov_x,
            camera.fov_y,
            camera.center_uv,
            BrushModel::Pinhole,
        );
        assert_eq!(
            camera.world_to_local(),
            glam::Mat4::from(brush.world_to_local())
        );
    }
}

#[test]
fn fold_culling_matches_brush_for_phone_ultrawide() {
    let [k1, k2, k3, k4] = [0.278781, 0.464781, -0.148871, -0.150010];
    let ours = CameraModel::KannalaBrandt4 { k1, k2, k3, k4 };
    let brush = BrushModel::KannalaBrandt4(KannalaBrandt4Params { k1, k2, k3, k4 });
    assert!((ours.max_render_theta().to_degrees() - 65.0).abs() < 0.1);
    assert!(
        (ours.max_render_theta() - brush_render::camera::max_render_theta(&brush)).abs() < 1e-12
    );
    for fov in [0.5, 1.0, 1.8, 2.1] {
        assert_eq!(
            ours.fov_to_focal(fov, 1280),
            brush_render::camera::fov_to_focal(fov, 1280, &brush)
        );
    }
}

#[test]
fn off_center_radial_jacobian_clamp_matches_brush() {
    let [k1, k2, k3, k4, k5, k6, p1, p2] = [0.15, 0.08, 0.02, 0.03, 0.0, 0.0, 0.001, -0.002];
    let model = BrushModel::RadialTangential8(RadialTangential8Params {
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        p1,
        p2,
    });
    let ours = Camera {
        model: CameraModel::RadialTangential8 {
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            p1,
            p2,
        },
        size: glam::uvec2(1920, 1080),
        position: glam::Vec3::ZERO,
        rotation: glam::Quat::IDENTITY,
        fov_x: 1.8,
        fov_y: 1.2,
        center_uv: Vec2::new(0.48, 0.52),
    };
    let brush = brush_render::camera::Camera::new(
        ours.position,
        ours.rotation,
        ours.fov_x,
        ours.fov_y,
        ours.center_uv,
        model,
    );
    let reference = brush_render::camera::calculate_jacobian_clamp_limits(
        ours.size,
        brush.build_pinhole_params(ours.size),
        model,
    );
    let (limits, radius) = ours.clamp_limits();
    assert_eq!(
        limits,
        Vec4::new(
            reference.lim_neg_x,
            reference.lim_neg_y,
            reference.lim_pos_x,
            reference.lim_pos_y
        )
    );
    assert_eq!(radius, reference.lim_r);
}
