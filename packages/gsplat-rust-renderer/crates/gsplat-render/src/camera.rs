//! Camera conventions and deterministic camera paths.
use glam::Mat4;
pub fn opengl_to_opencv(pose: Mat4) -> Mat4 {
    pose * Mat4::from_scale(glam::Vec3::new(1.0, -1.0, -1.0))
}
pub fn colmap_to_world(q: [f32; 4], t: [f32; 3]) -> Mat4 {
    Mat4::from_rotation_translation(
        glam::Quat::from_xyzw(q[1], q[2], q[3], q[0]).normalize(),
        glam::Vec3::from_array(t),
    )
    .inverse()
}

pub use gsplat_core::CameraModel;

/// OpenCV camera axes: +x right, +y down, +z forward. The rigid pose is a
/// row-major world-from-camera matrix; fx/fy/cx/cy are in image pixels.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraSpec {
    pub world_from_camera: [[f32; 4]; 4],
    pub width: u32,
    pub height: u32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub model: CameraModel,
}

#[derive(Debug, thiserror::Error)]
#[error("invalid camera: {0}")]
pub struct CameraError(pub &'static str);

pub use crate::camera_files::{CameraFrame, load_frames};

impl CameraSpec {
    pub fn pose(&self) -> Mat4 {
        Mat4::from_cols_array_2d(&self.world_from_camera).transpose()
    }
    pub fn project(&self, point: glam::Vec3) -> glam::Vec2 {
        let p = self.pose().inverse().transform_point3(point);
        let x = p.x / p.z;
        let y = p.y / p.z;
        let r2 = x * x + y * y;
        let kb = |k1: f32, k2: f32, k3: f32, k4: f32| {
            let r = r2.sqrt();
            if r < 1e-8 {
                return 1.0;
            }
            let theta = r.atan();
            let t2 = theta * theta;
            theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) / r
        };
        let (radial, p1, p2, sx, sy) = match self.model {
            CameraModel::Pinhole => (1.0, 0.0, 0.0, 0.0, 0.0),
            CameraModel::KannalaBrandt4 { k1, k2, k3, k4 } => {
                (kb(k1, k2, k3, k4), 0.0, 0.0, 0.0, 0.0)
            }
            CameraModel::RadialTangential8 {
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                p1,
                p2,
            } => (
                (1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))),
                p1,
                p2,
                0.0,
                0.0,
            ),
            CameraModel::ThinPrismFisheye {
                k1,
                k2,
                k3,
                k4,
                p1,
                p2,
                sx1,
                sy1,
            } => (kb(k1, k2, k3, k4), p1, p2, sx1, sy1),
        };
        glam::Vec2::new(
            self.fx * (radial * x + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + sx * r2) + self.cx,
            self.fy * (radial * y + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y) + sy * r2) + self.cy,
        )
    }

    pub fn validate(&self) -> Result<(), CameraError> {
        if self.width == 0
            || self.height == 0
            || self.fx <= 0.0
            || self.fy <= 0.0
            || ![self.fx, self.fy, self.cx, self.cy]
                .iter()
                .all(|v| v.is_finite())
        {
            return Err(CameraError(
                "nonpositive dimensions/focal length or nonfinite intrinsics",
            ));
        }
        let pose = self.pose();
        let rotation = glam::Mat3::from_mat4(pose);
        if !pose.is_finite()
            || !pose.row(3).abs_diff_eq(glam::Vec4::W, 1e-5)
            || !(rotation.transpose() * rotation).abs_diff_eq(glam::Mat3::IDENTITY, 1e-3)
            || rotation.determinant() < 0.0
        {
            return Err(CameraError("world_from_camera must be a finite rigid pose"));
        }
        let coefficients = self.model.coefficients();
        if !coefficients.iter().all(|v| v.is_finite()) {
            return Err(CameraError("nonfinite lens coefficients"));
        }
        Ok(())
    }
    pub fn from_nerf(pose: Mat4, angle_x: f32, width: u32, height: u32) -> Self {
        let focal = width as f32 / (2.0 * (angle_x * 0.5).tan());
        Self {
            world_from_camera: opengl_to_opencv(pose).transpose().to_cols_array_2d(),
            width,
            height,
            fx: focal,
            fy: focal,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            model: CameraModel::Pinhole,
        }
    }
    pub fn resized(&self, width: u32, height: u32) -> Self {
        let sx = width as f32 / self.width as f32;
        let sy = height as f32 / self.height as f32;
        Self {
            width,
            height,
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: self.cx * sx,
            cy: self.cy * sy,
            ..self.clone()
        }
    }

    /// Convert all lens coefficients in Brush's documented parameter order.
    pub fn brush_camera(&self) -> brush_render::camera::Camera {
        use brush_render::kernels::camera_model::{
            CameraModel as B, kannala_brandt_4::KannalaBrandt4Params as K,
            radial_tangential_8::RadialTangential8Params as R,
            thin_prism_fisheye::ThinPrismFisheyeParams as T,
        };
        let model = match self.model {
            CameraModel::Pinhole => B::Pinhole,
            CameraModel::KannalaBrandt4 { k1, k2, k3, k4 } => {
                B::KannalaBrandt4(K { k1, k2, k3, k4 })
            }
            CameraModel::RadialTangential8 {
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                p1,
                p2,
            } => B::RadialTangential8(R {
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                p1,
                p2,
            }),
            CameraModel::ThinPrismFisheye {
                k1,
                k2,
                k3,
                k4,
                p1,
                p2,
                sx1,
                sy1,
            } => B::ThinPrismFisheye(T {
                kb4: K { k1, k2, k3, k4 },
                p1,
                p2,
                sx1,
                sy1,
            }),
        };
        let pose = self.pose();
        brush_render::camera::Camera::new(
            pose.w_axis.truncate(),
            glam::Quat::from_mat3(&glam::Mat3::from_mat4(pose)),
            brush_render::camera::focal_to_fov(self.fx as f64, self.width, &model),
            brush_render::camera::focal_to_fov(self.fy as f64, self.height, &model),
            glam::Vec2::new(self.cx / self.width as f32, self.cy / self.height as f32),
            model,
        )
    }

    pub fn core_camera(&self) -> gsplat_core::Camera {
        let camera = self.brush_camera();
        gsplat_core::Camera {
            position: camera.position,
            rotation: camera.rotation,
            fov_x: camera.fov_x,
            fov_y: camera.fov_y,
            center_uv: camera.center_uv,
            size: glam::UVec2::new(self.width, self.height),
            model: self.model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
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
            cam.project(glam::Vec3::new(1.0, 2.0, 10.0)),
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
        assert!(
            cam.resized(400, 300)
                .project(p)
                .abs_diff_eq(cam.project(p) * 0.5, 1e-5)
        );
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
                decoded
                    .project(glam::Vec3::new(0.4, 0.3, -1.0))
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
            camera
                .project(world.transform_point3(glam::Vec3::Z))
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
}
