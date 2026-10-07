use gsplat_core::CameraModel;

pub fn project(camera: &crate::camera::CameraSpec, point: glam::Vec3) -> glam::Vec2 {
    let p = camera.pose().inverse().transform_point3(point);
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
    let (radial, p1, p2, sx, sy) = match camera.model {
        CameraModel::Pinhole => (1.0, 0.0, 0.0, 0.0, 0.0),
        CameraModel::KannalaBrandt4 { k1, k2, k3, k4 } => (kb(k1, k2, k3, k4), 0.0, 0.0, 0.0, 0.0),
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
        camera.fx * (radial * x + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + sx * r2) + camera.cx,
        camera.fy * (radial * y + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y) + sy * r2) + camera.cy,
    )
}
