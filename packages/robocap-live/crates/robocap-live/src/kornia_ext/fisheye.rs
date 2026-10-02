//! Kannala-Brandt (KB4) unprojection on the lens model's physical branch.
//!
//! kornia-3d's `FisheyeCamera::unproject` runs Newton from `theta = theta_d`. Real calibrations often have a distortion
//! polynomial `theta_d(theta) = theta + k1 theta^3 + k2 theta^5 + k3 theta^7 + k4 theta^9` that peaks before 90 degrees (RoboCap's
//! right_front camera: at ~84 degrees). Near the image edge `theta_d` then exceeds that peak's `theta`, Newton starts on the far,
//! decreasing branch and returns a ray up to 7 degrees off whose projection is the same pixel. [`MonotonicFisheye`] finds the
//! peak once and solves on `[0, theta_max]` (safeguarded Newton), saturating at the peak where no inverse exists. Target
//! upstream: a fix to kornia-3d `camera::fisheye`.

use kornia_3d::camera::FisheyeCamera;
use kornia_algebra::{Vec2F64, Vec3F64};

/// A KB4 camera with its monotonic range: unprojection never leaves the physical branch.
#[derive(Clone, Debug)]
pub struct MonotonicFisheye {
    /// The kornia-3d camera (projection is unchanged).
    pub camera: FisheyeCamera,
    /// The largest angle (radians, at most pi) up to which `theta_d` increases.
    pub theta_max: f64,
}

fn distortion(camera: &FisheyeCamera, theta: f64) -> (f64, f64) {
    let t2 = theta * theta;
    let theta_d = theta * (1.0 + t2 * (camera.k1 + t2 * (camera.k2 + t2 * (camera.k3 + t2 * camera.k4))));
    let slope = 1.0 + t2 * (3.0 * camera.k1 + t2 * (5.0 * camera.k2 + t2 * (7.0 * camera.k3 + t2 * 9.0 * camera.k4)));
    (theta_d, slope)
}

impl MonotonicFisheye {
    /// Wrap `camera`, finding where its distortion polynomial stops increasing (a 1 mrad scan, then bisection).
    ///
    /// # Arguments
    ///
    /// * `camera` - The KB4 camera.
    ///
    /// # Returns
    ///
    /// The camera with `theta_max` set (pi when the polynomial increases everywhere on `[0, pi]`).
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::camera::FisheyeCamera;
    /// use robocap_live::kornia_ext::fisheye::MonotonicFisheye;
    /// let ideal = FisheyeCamera { fx: 600.0, fy: 600.0, cx: 960.0, cy: 540.0, k1: 0.0, k2: 0.0, k3: 0.0, k4: 0.0 };
    /// assert_eq!(MonotonicFisheye::new(ideal).theta_max, std::f64::consts::PI);
    /// ```
    pub fn new(camera: FisheyeCamera) -> Self {
        let step = 1e-3;
        let mut theta_max = std::f64::consts::PI;
        let mut previous = 0.0;
        let mut theta = step;
        while theta <= std::f64::consts::PI {
            if distortion(&camera, theta).1 <= 0.0 {
                let (mut low, mut high) = (previous, theta);
                for _ in 0..60 {
                    let middle = 0.5 * (low + high);
                    if distortion(&camera, middle).1 > 0.0 {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
                theta_max = low;
                break;
            }
            previous = theta;
            theta += step;
        }
        Self { camera, theta_max }
    }

    /// The unit camera-frame ray through `pixel`, with `theta` in `[0, theta_max]`.
    ///
    /// # Arguments
    ///
    /// * `pixel` - Pixel coordinates.
    ///
    /// # Returns
    ///
    /// The unit ray; pixels beyond the polynomial's peak get the ray at `theta_max` in their direction.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::camera::FisheyeCamera;
    /// use kornia_algebra::{Vec2F64, Vec3F64};
    /// use robocap_live::kornia_ext::fisheye::MonotonicFisheye;
    /// let lens = MonotonicFisheye::new(FisheyeCamera { fx: 600.0, fy: 600.0, cx: 960.0, cy: 540.0, k1: 0.05, k2: -0.01, k3: 0.0, k4: 0.0 });
    /// let ray = lens.unproject(&Vec2F64::new(1500.0, 700.0));
    /// let (pixel, _) = lens.camera.project(&ray).unwrap();
    /// assert!((pixel.x - 1500.0).abs() < 1e-9 && (pixel.y - 700.0).abs() < 1e-9);
    /// ```
    pub fn unproject(&self, pixel: &Vec2F64) -> Vec3F64 {
        let camera = &self.camera;
        let mx = (pixel.x - camera.cx) / camera.fx;
        let my = (pixel.y - camera.cy) / camera.fy;
        let theta_d = (mx * mx + my * my).sqrt();
        if theta_d < 1e-12 {
            return Vec3F64::new(0.0, 0.0, 1.0);
        }
        let (peak, _) = distortion(camera, self.theta_max);
        let theta = if theta_d >= peak {
            self.theta_max
        } else {
            // theta_d(theta) increases on [low, high]: Newton steps, bisecting whenever a step leaves the bracket.
            let (mut low, mut high) = (0.0f64, self.theta_max);
            let mut theta = theta_d.min(self.theta_max);
            for _ in 0..50 {
                let (value, slope) = distortion(camera, theta);
                let residual = value - theta_d;
                if residual.abs() < 1e-15 {
                    break;
                }
                if residual > 0.0 {
                    high = theta;
                } else {
                    low = theta;
                }
                let newton = theta - residual / slope;
                theta = if slope > 0.0 && newton > low && newton < high { newton } else { 0.5 * (low + high) };
            }
            theta
        };
        let (sin_theta, cos_theta) = theta.sin_cos();
        Vec3F64::new(sin_theta * mx / theta_d, sin_theta * my / theta_d, cos_theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RoboCap s66 right_front (cam_01), whose polynomial peaks at ~84 degrees.
    fn right_front() -> FisheyeCamera {
        FisheyeCamera {
            fx: 630.6917724609375,
            fy: 628.777587890625,
            cx: 946.6721801757812,
            cy: 539.53125,
            k1: 0.07725944370031357,
            k2: -0.06258341670036316,
            k3: 0.08006518334150314,
            k4: -0.02879575826227665,
        }
    }

    #[test]
    fn the_peak_is_found_and_edge_pixels_stay_on_the_physical_branch() {
        let lens = MonotonicFisheye::new(right_front());
        assert!((lens.theta_max.to_degrees() - 84.0).abs() < 1.5, "{}", lens.theta_max.to_degrees());
        let pixel = Vec2F64::new(100.0, 80.0);
        let ours = lens.unproject(&pixel);
        let theirs = lens.camera.unproject(&pixel);
        // Both project onto the pixel, but kornia's lies on the far branch.
        for ray in [ours, theirs] {
            let (back, _) = lens.camera.project(&ray).unwrap_or((Vec2F64::new(f64::NAN, f64::NAN), 0.0));
            assert!((back.x - pixel.x).abs() < 1e-6 && (back.y - pixel.y).abs() < 1e-6);
        }
        assert!(ours.z.acos() < lens.theta_max && theirs.z.acos() > lens.theta_max);
    }

    #[test]
    fn interior_pixels_match_kornia_and_corners_saturate() {
        let lens = MonotonicFisheye::new(right_front());
        for (x, y) in [(946.0, 540.0), (1200.0, 300.0), (300.0, 900.0), (1700.0, 100.0)] {
            let pixel = Vec2F64::new(x, y);
            let (a, b) = (lens.unproject(&pixel), lens.camera.unproject(&pixel));
            assert!((a - b).length() < 1e-9, "{x} {y}");
        }
        let corner = lens.unproject(&Vec2F64::new(0.0, 0.0));
        assert!((corner.z.acos() - lens.theta_max).abs() < 1e-12);
    }
}
