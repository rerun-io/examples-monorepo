//! Lens math ported from Brush camera.rs at 1388f74c.
use crate::Camera;
use glam::{Vec2, Vec4};
use std::f64::consts::PI;

/// Distortion coefficient order matches Brush. Fisheye models use angular projection.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub enum CameraModel {
    #[default]
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

impl CameraModel {
    pub(crate) fn kind(self) -> u32 {
        match self {
            Self::Pinhole => 0,
            Self::KannalaBrandt4 { .. } => 1,
            Self::RadialTangential8 { .. } => 2,
            Self::ThinPrismFisheye { .. } => 3,
        }
    }
    pub fn coefficients(self) -> [f32; 8] {
        match self {
            Self::Pinhole => [0.0; 8],
            Self::KannalaBrandt4 { k1, k2, k3, k4 } => [k1, k2, k3, k4, 0.0, 0.0, 0.0, 0.0],
            Self::RadialTangential8 {
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                p1,
                p2,
            } => [k1, k2, k3, k4, k5, k6, p1, p2],
            Self::ThinPrismFisheye {
                k1,
                k2,
                k3,
                k4,
                p1,
                p2,
                sx1,
                sy1,
            } => [k1, k2, k3, k4, p1, p2, sx1, sy1],
        }
    }
    fn kb4(self, theta: f64) -> f64 {
        let p = self.coefficients().map(f64::from);
        let t2 = theta * theta;
        let t3 = t2 * theta;
        let t5 = t3 * t2;
        let t7 = t5 * t2;
        let t9 = t7 * t2;
        theta + p[0] * t3 + p[1] * t5 + p[2] * t7 + p[3] * t9
    }
    fn radial(self, r: f64) -> f64 {
        let p = self.coefficients().map(f64::from);
        let r2 = r * r;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        (1.0 + p[0] * r2 + p[1] * r4 + p[2] * r6) / (1.0 + p[3] * r2 + p[4] * r4 + p[5] * r6)
    }
    fn undistort_radius(self, distorted: f64) -> f64 {
        let mut r = distorted;
        for _ in 0..30 {
            let factor = self.radial(r);
            if factor.abs() < 1e-12 {
                break;
            }
            let next = distorted / factor;
            if (next - r).abs() < 1e-12 {
                r = next;
                break;
            }
            r = next;
        }
        r
    }
    fn corner_radius(self, distorted: f64) -> Option<f64> {
        let mut lo = 0.0;
        let mut hi = None;
        for i in 1..=4096 {
            let r = f64::from(i) * 16.0 / 4096.0;
            if r * self.radial(r) >= distorted {
                hi = Some(r);
                break;
            }
            lo = r;
        }
        let mut hi = hi?;
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if mid * self.radial(mid) < distorted {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-12 {
                break;
            }
        }
        Some(0.5 * (lo + hi))
    }
    /// First fold of the KB4 angular polynomial, or pi when it is monotonic.
    pub fn max_render_theta(self) -> f64 {
        if matches!(self, Self::Pinhole | Self::RadialTangential8 { .. }) {
            return PI;
        }
        let p = self.coefficients().map(f64::from);
        let derivative = |theta: f64| {
            let t2 = theta * theta;
            let t4 = t2 * t2;
            let t6 = t4 * t2;
            let t8 = t4 * t4;
            1.0 + 3.0 * p[0] * t2 + 5.0 * p[1] * t4 + 7.0 * p[2] * t6 + 9.0 * p[3] * t8
        };
        let mut lo = 0.0;
        let mut hi = PI;
        let mut found = false;
        for i in 1..=128 {
            let theta = f64::from(i) * PI / 128.0;
            if derivative(theta) <= 0.0 {
                hi = theta;
                found = true;
                break;
            }
            lo = theta;
        }
        if !found {
            return PI;
        }
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if derivative(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-12 {
                break;
            }
        }
        0.5 * (lo + hi)
    }
    /// Focal length corresponding to the lens's angular field of view.
    pub fn fov_to_focal(self, fov: f64, pixels: u32) -> f64 {
        let theta = fov * 0.5;
        let projected = match self {
            Self::Pinhole => theta.tan(),
            Self::RadialTangential8 { .. } => {
                let r = theta.tan();
                r * self.radial(r)
            }
            _ => self.kb4(theta),
        };
        f64::from(pixels) * 0.5 / projected
    }
}
impl Camera {
    pub fn focal(&self) -> Vec2 {
        Vec2::new(
            self.model.fov_to_focal(self.fov_x, self.size.x) as f32,
            self.model.fov_to_focal(self.fov_y, self.size.y) as f32,
        )
    }
    pub fn clamp_limits(&self) -> (Vec4, f32) {
        let focal = self.focal();
        let center = self.center_uv * self.size.as_vec2();
        let lo = (-0.15 * self.size.as_vec2() - center) / focal;
        let hi = (1.15 * self.size.as_vec2() - center) / focal;
        match self.model {
            CameraModel::Pinhole => (Vec4::new(lo.x, lo.y, hi.x, hi.y), f32::MAX),
            CameraModel::RadialTangential8 { .. } => {
                let inverse =
                    |x: f32| (self.model.undistort_radius(f64::from(x.abs())) as f32) * x.signum();
                let corner = f64::from(hi.x.abs().max(lo.x.abs()))
                    .hypot(f64::from(hi.y.abs().max(lo.y.abs())));
                (
                    Vec4::new(inverse(lo.x), inverse(lo.y), inverse(hi.x), inverse(hi.y)),
                    self.model
                        .corner_radius(corner)
                        .map_or(f32::MAX, |r| r as f32),
                )
            }
            _ => (Vec4::ZERO, f32::MAX),
        }
    }
    pub(crate) fn half_max_render_fov(&self) -> f32 {
        (((self.fov_x as f32).hypot(self.fov_y as f32) * 1.05)
            .min(2.0 * std::f32::consts::PI - 1e-6)
            * 0.5)
            .min(self.model.max_render_theta() as f32)
    }
}
