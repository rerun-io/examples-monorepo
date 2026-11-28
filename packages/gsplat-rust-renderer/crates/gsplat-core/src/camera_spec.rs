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

use crate::CameraModel;

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

impl CameraSpec {
    pub fn pose(&self) -> Mat4 {
        Mat4::from_cols_array_2d(&self.world_from_camera).transpose()
    }
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.width == 0
            || self.height == 0
            || self.fx <= 0.0
            || self.fy <= 0.0
            || ![self.fx, self.fy, self.cx, self.cy]
                .iter()
                .all(|v| v.is_finite())
        {
            anyhow::bail!("nonpositive dimensions/focal length or nonfinite intrinsics");
        }
        let pose = self.pose();
        let rotation = glam::Mat3::from_mat4(pose);
        if !pose.is_finite()
            || !pose.row(3).abs_diff_eq(glam::Vec4::W, 1e-5)
            || !(rotation.transpose() * rotation).abs_diff_eq(glam::Mat3::IDENTITY, 1e-3)
            || rotation.determinant() < 0.0
        {
            anyhow::bail!("world_from_camera must be a finite rigid pose");
        }
        let coefficients = self.model.coefficients();
        if !coefficients.iter().all(|v| v.is_finite()) {
            anyhow::bail!("nonfinite lens coefficients");
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

    pub fn from_core(camera: crate::Camera) -> Self {
        let focal = camera.focal();
        Self {
            world_from_camera: Mat4::from_rotation_translation(camera.rotation, camera.position)
                .transpose()
                .to_cols_array_2d(),
            width: camera.size.x,
            height: camera.size.y,
            fx: focal.x,
            fy: focal.y,
            cx: camera.center_uv.x * camera.size.x as f32,
            cy: camera.center_uv.y * camera.size.y as f32,
            model: camera.model,
        }
    }
}

pub fn orbit(
    center: glam::Vec3,
    radius: glam::Vec2,
    elevation: f32,
    count: usize,
    template: &CameraSpec,
    up: Option<glam::Vec3>,
) -> Vec<CameraSpec> {
    let basis = up.map(|up| {
        let rotation = glam::Quat::from_rotation_arc(glam::Vec3::Z, up.normalize_or(glam::Vec3::Z));
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
