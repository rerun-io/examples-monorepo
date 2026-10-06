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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CameraModel {
    Pinhole,
    KannalaBrandt4 {
        k1: f32,
        k2: f32,
        k3: f32,
        k4: f32,
    },
    RadialTangential8 {
        k1: f32,
        k2: f32,
        k3: f32,
        k4: f32,
        k5: f32,
        k6: f32,
        p1: f32,
        p2: f32,
    },
    ThinPrismFisheye {
        k1: f32,
        k2: f32,
        k3: f32,
        k4: f32,
        p1: f32,
        p2: f32,
        sx1: f32,
        sy1: f32,
    },
}

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

pub fn orbit(
    center: glam::Vec3,
    radius: glam::Vec2,
    elevation: f32,
    count: usize,
    template: &CameraSpec,
) -> Vec<CameraSpec> {
    (0..count)
        .map(|i| {
            let angle = std::f32::consts::TAU * i as f32 / count as f32;
            let position =
                center + glam::Vec3::new(radius.x * angle.cos(), radius.y * angle.sin(), elevation);
            let pose = Mat4::look_at_lh(position, center, -glam::Vec3::Z).inverse();
            CameraSpec {
                world_from_camera: pose.transpose().to_cols_array_2d(),
                ..template.clone()
            }
        })
        .collect()
}

// Third-party NeRF records intentionally allow extra metadata.
#[derive(serde::Deserialize)]
struct NerfTransforms {
    camera_angle_x: f32,
    frames: Vec<NerfFrame>,
}
#[derive(serde::Deserialize)]
struct NerfFrame {
    file_path: String,
    transform_matrix: [[f32; 4]; 4],
}

/// Load NeRF transforms and infer each source resolution from its image.
pub fn test_views(path: &std::path::Path) -> crate::Result<Vec<CameraSpec>> {
    let document: NerfTransforms = serde_json::from_reader(std::fs::File::open(path)?)?;
    let root = path.parent().unwrap_or(std::path::Path::new("."));
    let mut cameras = Vec::new();
    for frame in document.frames {
        let mut image = root.join(frame.file_path);
        if image.extension().is_none() {
            image.set_extension("png");
        }
        let (w, h) = image::image_dimensions(image)?;
        let camera = CameraSpec::from_nerf(
            Mat4::from_cols_array_2d(&frame.transform_matrix).transpose(),
            document.camera_angle_x,
            w,
            h,
        );
        cameras.push(camera);
    }
    if cameras.is_empty() {
        return Err(crate::Error::Invalid("empty camera path".into()));
    }
    Ok(cameras)
}

impl CameraSpec {
    pub fn vertical_fov(&self) -> f32 {
        2.0 * (self.height as f32 / (2.0 * self.fy)).atan()
    }
    pub fn aspect(&self) -> f32 {
        self.width as f32 * self.fy / (self.height as f32 * self.fx)
    }
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
    pub fn validate(&self) -> crate::Result<()> {
        if self.width == 0
            || self.height == 0
            || self.fx <= 0.0
            || self.fy <= 0.0
            || ![self.fx, self.fy, self.cx, self.cy]
                .iter()
                .all(|v| v.is_finite())
        {
            return Err(crate::Error::Invalid(
                "nonpositive dimensions/focal length or nonfinite intrinsics".into(),
            ));
        }
        let pose = self.pose();
        let rotation = glam::Mat3::from_mat4(pose);
        if !pose.is_finite()
            || !pose.row(3).abs_diff_eq(glam::Vec4::W, 1e-5)
            || !(rotation.transpose() * rotation).abs_diff_eq(glam::Mat3::IDENTITY, 1e-3)
            || rotation.determinant() < 0.0
        {
            return Err(crate::Error::Invalid(
                "world_from_camera must be a finite rigid pose".into(),
            ));
        }
        let coefficients = match self.model {
            CameraModel::Pinhole => [0.0; 8],
            CameraModel::KannalaBrandt4 { k1, k2, k3, k4 } => [k1, k2, k3, k4, 0.0, 0.0, 0.0, 0.0],
            CameraModel::RadialTangential8 {
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                p1,
                p2,
            } => [k1, k2, k3, k4, k5, k6, p1, p2],
            CameraModel::ThinPrismFisheye {
                k1,
                k2,
                k3,
                k4,
                p1,
                p2,
                sx1,
                sy1,
            } => [k1, k2, k3, k4, p1, p2, sx1, sy1],
        };
        if !coefficients.iter().all(|v| v.is_finite()) {
            return Err(crate::Error::Invalid("nonfinite lens coefficients".into()));
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
}

/// Explicit CLI camera paths; malformed prefixes are never guessed as filenames.
#[derive(Debug, Clone)]
pub enum CameraPath {
    Orbit(usize),
    TestViews(std::path::PathBuf),
    Specs(std::path::PathBuf),
    Colmap(std::path::PathBuf),
}
impl std::str::FromStr for CameraPath {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "held" {
            return Ok(Self::Orbit(1));
        }
        let (kind, path) = value
            .split_once(':')
            .ok_or("path must be orbit:N, held, test-views:FILE, specs:FILE, or colmap:DIR")?;
        if path.is_empty() {
            return Err("camera path payload is empty".into());
        }
        match kind {
            "orbit" => {
                let n = path
                    .parse::<usize>()
                    .map_err(|_| "orbit count must be a positive integer")?;
                if n == 0 {
                    Err("orbit count must be positive".into())
                } else {
                    Ok(Self::Orbit(n))
                }
            }
            "test-views" => Ok(Self::TestViews(path.into())),
            "specs" => Ok(Self::Specs(path.into())),
            "colmap" => Ok(Self::Colmap(path.into())),
            _ => Err(format!("unknown camera path prefix {kind:?}")),
        }
    }
}

/// Read a COLMAP binary sparse model, in image-id order. Distortion follows
/// COLMAP's parameter order; unsupported FOV models fail explicitly.
pub async fn colmap_views(path: &std::path::Path) -> crate::Result<Vec<CameraSpec>> {
    use colmap_reader::ColmapCameraModel as M;
    use tokio::io::BufReader;
    let intrinsics = colmap_reader::read_cameras(
        BufReader::new(tokio::fs::File::open(path.join("cameras.bin")).await?),
        true,
    )
    .await?;
    let mut images = colmap_reader::read_images(
        BufReader::new(tokio::fs::File::open(path.join("images.bin")).await?),
        true,
        false,
    )
    .await?;
    images.sort_by_key(|i| i.id);
    images
        .into_iter()
        .map(|image| {
            let c = intrinsics
                .iter()
                .find(|c| c.id == image.camera_id)
                .ok_or_else(|| {
                    crate::Error::Invalid(format!("missing COLMAP camera {}", image.camera_id))
                })?;
            let p: Vec<f32> = c.params.iter().map(|p| *p as f32).collect();
            let model = match c.model {
                M::SimplePinhole | M::Pinhole => CameraModel::Pinhole,
                M::SimpleRadial | M::Radial => CameraModel::RadialTangential8 {
                    k1: p[3],
                    k2: if matches!(c.model, M::Radial) {
                        p[4]
                    } else {
                        0.0
                    },
                    k3: 0.0,
                    k4: 0.0,
                    k5: 0.0,
                    k6: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                },
                M::OpenCV | M::FullOpenCV => CameraModel::RadialTangential8 {
                    k1: p[4],
                    k2: p[5],
                    p1: p[6],
                    p2: p[7],
                    k3: *p.get(8).unwrap_or(&0.0),
                    k4: *p.get(9).unwrap_or(&0.0),
                    k5: *p.get(10).unwrap_or(&0.0),
                    k6: *p.get(11).unwrap_or(&0.0),
                },
                M::OpenCvFishEye => CameraModel::KannalaBrandt4 {
                    k1: p[4],
                    k2: p[5],
                    k3: p[6],
                    k4: p[7],
                },
                M::SimpleRadialFisheye | M::RadialFisheye => CameraModel::KannalaBrandt4 {
                    k1: p[3],
                    k2: *p.get(4).unwrap_or(&0.0),
                    k3: 0.0,
                    k4: 0.0,
                },
                M::ThinPrismFisheye => CameraModel::ThinPrismFisheye {
                    k1: p[4],
                    k2: p[5],
                    p1: p[6],
                    p2: p[7],
                    k3: p[8],
                    k4: p[9],
                    sx1: p[10],
                    sy1: p[11],
                },
                M::Fov => return Err(crate::Error::Unsupported("COLMAP FOV camera".into())),
            };
            let q = image.quat;
            let pose = colmap_to_world([q.w, q.x, q.y, q.z], image.tvec.to_array());
            let (fx, fy) = c.focal();
            let center = c.principal_point();
            Ok(CameraSpec {
                world_from_camera: pose.transpose().to_cols_array_2d(),
                width: c.width as u32,
                height: c.height as u32,
                fx: fx as f32,
                fy: fy as f32,
                cx: center.x,
                cy: center.y,
                model,
            })
        })
        .collect()
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
        let path = orbit(center, glam::Vec2::new(4.0, 2.0), 1.0, 4, &template);
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
    #[test]
    fn explicit_camera_path_syntax() {
        assert!(matches!(
            "orbit:300".parse::<CameraPath>().unwrap(),
            CameraPath::Orbit(300)
        ));
        for invalid in ["orbit300", "orbit:0", "orbit:-1", "colmap:", "testview:x"] {
            assert!(invalid.parse::<CameraPath>().is_err(), "{invalid}");
        }
    }
    #[test]
    fn nerf_path_loads_image_size_and_axis_convention() {
        let directory =
            std::env::temp_dir().join(format!("gsplat-camera-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        image::RgbImage::new(32, 24)
            .save(directory.join("frame.png"))
            .unwrap();
        let source = directory.join("transforms_test.json");
        std::fs::write(&source,r#"{"camera_angle_x":1.0,"unrelated":true,"frames":[{"file_path":"frame","transform_matrix":[[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]}]}"#).unwrap();
        let views = test_views(&source).unwrap();
        assert_eq!((views[0].width, views[0].height), (32, 24));
        assert_eq!(
            views[0].pose().w_axis.truncate(),
            glam::Vec3::new(1.0, 2.0, 3.0)
        );
        assert_eq!(views[0].pose().z_axis.truncate(), glam::Vec3::NEG_Z);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
