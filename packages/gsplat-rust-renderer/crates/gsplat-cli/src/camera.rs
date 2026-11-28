//! Camera conventions and deterministic camera paths (`gsplat_core::camera_spec`), plus the Brush conversions that need
//! Brush and so stay in the CLI.
pub use gsplat_core::camera_spec::*;

use gsplat_core::CameraModel;

/// Conversions of a [`CameraSpec`] that go through Brush's camera model.
pub trait BrushCamera {
    /// Convert all lens coefficients in Brush's documented parameter order.
    fn brush_camera(&self) -> brush_render::camera::Camera;
    /// The renderer's camera, with Brush's focal-to-field-of-view convention.
    fn core_camera(&self) -> gsplat_core::Camera;
}

impl BrushCamera for CameraSpec {
    fn brush_camera(&self) -> brush_render::camera::Camera {
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

    fn core_camera(&self) -> gsplat_core::Camera {
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
