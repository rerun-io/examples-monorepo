//! Tests of `gsplat_core::camera_spec` (re-exported as `gsplat_cli::camera`) with the CLI's projection helper.
use crate::camera::*;
use crate::common::project;
use glam::Mat4;
use glam::Vec3;
use gsplat_core::CameraModel;
#[test]
fn nerf_axis_conversion_round_trip() {
    let pose = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    let cv = opengl_to_opencv(pose);
    assert_eq!(cv.transform_point3(Vec3::Z), Vec3::new(1.0, 2.0, 2.0));
    assert_eq!(cv.transform_vector3(Vec3::Y), -Vec3::Y);
    assert!(opengl_to_opencv(cv).abs_diff_eq(pose, 1e-6));
}
#[test]
fn colmap_pose_inverts_world_to_camera() {
    let world = colmap_to_world([1.0, 0.0, 0.0, 0.0], [-1.0, -2.0, -3.0]);
    assert_eq!(world.transform_point3(Vec3::ZERO), Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn known_point_projects_to_expected_pixel() {
    let cam = CameraSpec {
        world_from_camera: Mat4::IDENTITY.to_cols_array_2d(),
        width: 640,
        height: 480,
        fx: 200.0,
        fy: 300.0,
        cx: 310.0,
        cy: 220.0,
        model: CameraModel::Pinhole,
    };
    assert_eq!(
        project(&cam, glam::Vec3::new(1.0, 2.0, 10.0)),
        glam::Vec2::new(330.0, 280.0)
    );
}

#[test]
fn rejects_nonrigid_pose_and_zero_focal_length() {
    let mut cam = CameraSpec::from_nerf(Mat4::IDENTITY, 1.0, 800, 600);
    assert!(cam.validate().is_ok());
    cam.fx = 0.0;
    assert!(cam.validate().is_err());
    cam.fx = 200.0;
    cam.world_from_camera[0][0] = 2.0;
    assert!(cam.validate().is_err());
}
#[test]
fn resize_preserves_normalized_projection() {
    let cam = CameraSpec::from_nerf(Mat4::IDENTITY, 1.0, 800, 600);
    let p = glam::Vec3::new(0.2, 0.1, -1.0);
    assert!(project(&cam.resized(400, 300), p).abs_diff_eq(project(&cam, p) * 0.5, 1e-5));
}

#[test]
fn lens_models_round_trip_and_keep_principal_point() {
    let models = [
        CameraModel::Pinhole,
        CameraModel::KannalaBrandt4 {
            k1: 0.1,
            k2: 0.02,
            k3: 0.003,
            k4: 0.0004,
        },
        CameraModel::RadialTangential8 {
            k1: 0.1,
            k2: 0.02,
            k3: 0.003,
            k4: 0.004,
            k5: 0.0005,
            k6: 0.0006,
            p1: 0.001,
            p2: 0.002,
        },
        CameraModel::ThinPrismFisheye {
            k1: 0.1,
            k2: 0.02,
            k3: 0.003,
            k4: 0.0004,
            p1: 0.001,
            p2: 0.002,
            sx1: 0.005,
            sy1: 0.006,
        },
    ];
    let expected = [
        [397.0, 133.0],
        [392.019, 137.35837],
        [399.638_1, 130.86668],
        [392.535, 137.76437],
    ];
    for (model, expected) in models.into_iter().zip(expected) {
        let mut camera = CameraSpec::from_nerf(Mat4::IDENTITY, 1.0, 640, 480);
        camera.model = model;
        camera.fx = 240.0;
        camera.fy = 280.0;
        camera.cx = 301.0;
        camera.cy = 217.0;
        let text = serde_json::to_string(&camera).unwrap();
        let decoded: CameraSpec = serde_json::from_str(&text).unwrap();
        assert_eq!(serde_json::to_string(&decoded).unwrap(), text);
        assert!(
            project(&decoded, glam::Vec3::new(0.4, 0.3, -1.0))
                .abs_diff_eq(glam::Vec2::from_array(expected), 1e-3)
        );
        let brush = decoded.brush_camera();
        assert!(
            brush
                .focal(glam::uvec2(640, 480))
                .abs_diff_eq(glam::Vec2::new(camera.fx, camera.fy), 1e-3)
        );
    }
}
#[test]
fn colmap_rotation_round_trip_and_projection() {
    let rotation = glam::Quat::from_rotation_z(0.4);
    let translation = glam::Vec3::new(1.0, 2.0, 3.0);
    let world = colmap_to_world(
        [rotation.w, rotation.x, rotation.y, rotation.z],
        translation.to_array(),
    );
    assert!(
        (Mat4::from_rotation_translation(rotation, translation) * world)
            .abs_diff_eq(Mat4::IDENTITY, 1e-5)
    );
    let mut camera = CameraSpec::from_nerf(Mat4::IDENTITY, 1.0, 640, 480);
    camera.world_from_camera = world.transpose().to_cols_array_2d();
    assert!(
        project(&camera, world.transform_point3(glam::Vec3::Z))
            .abs_diff_eq(glam::Vec2::new(320.0, 240.0), 1e-3)
    );
}
#[test]
fn unknown_camera_fields_are_rejected() {
    let c = CameraSpec::from_nerf(Mat4::IDENTITY, 1.0, 640, 480);
    let text = serde_json::to_string(&c)
        .unwrap()
        .replacen('{', "{\"typo\":1,", 1);
    assert!(serde_json::from_str::<CameraSpec>(&text).is_err());
}
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
        assert!(project(camera, center).abs_diff_eq(glam::Vec2::new(320.0, 240.0), 1e-3));
    }
    assert!(
        path[0]
            .pose()
            .w_axis
            .truncate()
            .abs_diff_eq(center + glam::Vec3::new(4.0, 0.0, 1.0), 1e-5)
    );
}
