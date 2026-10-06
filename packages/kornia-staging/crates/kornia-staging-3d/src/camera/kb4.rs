use super::*;

/// Kannala–Brandt camera with four angular radial terms.
///
/// Parameters are immutable so the cached physical branch cannot become stale.
/// Reconstruct after a calibration increment. The f32 lane evaluates `atan2` in
/// f64 before rounding, as the SLAM reference does.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "[S;8]", into = "[S;8]"))]
pub struct KannalaBrandt4<S: Scalar> {
    pub(crate) param: [S; 8],
    theta_max: S,
    r_max: S,
}
impl<S: Scalar> KannalaBrandt4<S> {
    #[inline]
    fn project_core(
        &self,
        [x, y, z]: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut [[S; 8]; 2]>,
        r: S,
        theta: S,
    ) -> [S; 2] {
        let [fx, fy, cx, cy, k1, k2, k3, k4] = self.param;
        let r2 = x * x + y * y;
        if r == S::zero() {
            let pinhole = Pinhole {
                param: [fx, fy, cx, cy],
            };
            let mut head = [[S::zero(); 4]; 2];
            let pixel = pinhole.project_unchecked_with_jacobians(
                [x, y, z],
                jp,
                if ji.is_some() { Some(&mut head) } else { None },
            );
            if let Some(j) = ji {
                *j = [[S::zero(); 8]; 2];
                for row in 0..2 {
                    j[row][..4].copy_from_slice(&head[row]);
                }
            }
            return pixel;
        }
        let t2 = theta * theta;
        // Preserve the SLAM Horner evaluation order on the projection hot path.
        let rt = ((((k4 * t2 + k3) * t2 + k2) * t2 + k1) * t2 + S::one()) * theta;
        let mx = x * rt / r;
        let my = y * rt / r;
        if let Some(j) = jp {
            let slope = (((c::<S>(9.0) * k4 * t2 + c::<S>(7.0) * k3) * t2 + c::<S>(5.0) * k2) * t2
                + c::<S>(3.0) * k1)
                * t2
                + S::one();
            let tmp = z * z + r2;
            let tx = x / r * z / tmp;
            let ty = y / r * z / tmp;
            let tz = -r / tmp;
            *j = [
                [
                    fx * (rt * r + x * r * slope * tx - x * x * rt / r) / r2,
                    fx * x * (slope * ty * r - y * rt / r) / r2,
                    fx * x * slope * tz / r,
                ],
                [
                    fy * y * (slope * tx * r - x * rt / r) / r2,
                    fy * (rt * r + y * r * slope * ty - y * y * rt / r) / r2,
                    fy * y * slope * tz / r,
                ],
            ];
        }
        if let Some(j) = ji {
            *j = [[S::zero(); 8]; 2];
            j[0][0] = mx;
            j[0][2] = S::one();
            j[1][1] = my;
            j[1][3] = S::one();
            j[0][4] = fx * x * theta * t2 / r;
            j[1][4] = fy * y * theta * t2 / r;
            for column in 5..8 {
                j[0][column] = j[0][column - 1] * t2;
                j[1][column] = j[1][column - 1] * t2;
            }
        }
        [fx * mx + cx, fy * my + cy]
    }
    #[cold]
    fn project_tiny(&self, point: [S; 3]) -> Result<[S; 2], ProjectionReject> {
        // Subnormal squared radii can round enough that |x/r| exceeds one.
        // Check the computed pixel instead of relying on the cached radial bound.
        self.project_with_jacobians(point, None, None)
    }
    /// Build a KB4 camera and locate the first radial derivative sign change.
    /// # Arguments
    /// * `param` - `[fx,fy,cx,cy,k1,k2,k3,k4]`.
    /// # Errors
    /// Rejects non-finite parameters, nonpositive focals, or an overflowing branch.
    pub fn new(param: [S; 8]) -> Result<Self, InvalidCalibration> {
        validate_parameters(&param)?;
        let theta_max = angular::peak(&param[4..]);
        let r_max = angular::radial(theta_max, &param[4..]).0;
        if !(r_max >= S::zero() && r_max < S::largest().sqrt())
            || !(param[0] * r_max + param[2].abs()).is_finite()
            || !(param[1] * r_max + param[3].abs()).is_finite()
        {
            return Err(InvalidCalibration);
        }
        Ok(Self {
            theta_max,
            r_max,
            param,
        })
    }
    /// Intrinsics in constructor order.
    pub fn params(&self) -> [S; 8] {
        self.param
    }
    /// Endpoint of the increasing physical branch, in radians, at most pi.
    pub fn max_angle(&self) -> S {
        self.theta_max
    }
    /// Unit ray at the physical branch endpoint in a pixel's azimuth.
    ///
    /// Callers with a saturation policy may use this after `OutsideDomain`.
    /// # Errors
    /// Rejects invalid input, an undefined central azimuth, or a branch ending at pi.
    pub fn boundary_bearing(&self, pixel: [S; 2]) -> Result<[S; 3], UnprojectError> {
        let [x, y] = normalized_pixel(pixel, &self.param)?;
        let r = (x * x + y * y).sqrt();
        if !r.is_finite() {
            return Err(UnprojectError::NonFinite);
        }
        if r == S::zero() || self.theta_max >= S::pi() {
            return Err(UnprojectError::OutsideDomain);
        }
        let (sin, cos) = self.theta_max.sin_cos();
        Ok([sin * x / r, sin * y / r, cos])
    }
}
impl<S: Scalar> CameraModel<S> for KannalaBrandt4<S> {
    type IntrinsicJacobian = [[S; 8]; 2];
    type UnprojectIntrinsicJacobian = [[S; 8]; 3];
    #[inline(always)]
    fn project(&self, [x, y, z]: [S; 3]) -> Result<[S; 2], ProjectionReject> {
        let r2 = x * x + y * y;
        if !(r2 <= S::largest() && z.abs() <= S::largest()) {
            return Err(ProjectionReject::NonFinite);
        }
        if r2 < S::MIN_POSITIVE_NORMAL {
            return self.project_tiny([x, y, z]);
        }
        let r = r2.sqrt();
        let theta = c::<S>(r.to_f64().atan2(z.to_f64()));
        if theta > self.theta_max {
            return Err(ProjectionReject::OutsideDomain);
        }
        let [fx, fy, cx, cy, k1, k2, k3, k4] = self.param;
        let t2 = theta * theta;
        let rt = ((((k4 * t2 + k3) * t2 + k2) * t2 + k1) * t2 + S::one()) * theta;
        Ok([fx * (x * rt / r) + cx, fy * (y * rt / r) + cy])
    }
    #[inline]
    fn project_with_jacobians(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> Result<[S; 2], ProjectionReject> {
        let (pixel, status) = self.project_with_status(point, jp, ji);
        status.map(|()| pixel)
    }
    #[inline]
    fn project_unchecked_with_jacobians(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> [S; 2] {
        let r = (point[0] * point[0] + point[1] * point[1]).sqrt();
        let theta = c::<S>(r.to_f64().atan2(point[2].to_f64()));
        self.project_core(point, jp, ji, r, theta)
    }
    #[inline]
    fn project_with_status(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> ([S; 2], Result<(), ProjectionReject>) {
        let [x, y, z] = point;
        let r2 = x * x + y * y;
        let r = r2.sqrt();
        let theta = c::<S>(r.to_f64().atan2(z.to_f64()));
        let status = if !(r2 <= S::largest() && z.abs() <= S::largest()) {
            Err(ProjectionReject::NonFinite)
        } else if r == S::zero() && z <= S::zero() {
            Err(ProjectionReject::BelowMinDepth)
        } else if theta > self.theta_max {
            Err(ProjectionReject::OutsideDomain)
        } else {
            Ok(())
        };
        let pixel = self.project_core(point, jp, ji, r, theta);
        (
            pixel,
            status.and_then(|()| checked_pixel(pixel).map(|_| ())),
        )
    }
    fn unproject(&self, pixel: [S; 2]) -> Result<[S; 3], UnprojectError> {
        let [x, y] = normalized_pixel(pixel, &self.param)?;
        let r = (x * x + y * y).sqrt();
        if r == S::zero() {
            return Ok([S::zero(), S::zero(), S::one()]);
        }
        let theta = angular::invert(r, &self.param[4..], self.theta_max, self.r_max)?;
        // At pi the azimuth is undefined; do not accept endpoint-tolerance snaps.
        if theta >= S::pi() {
            return Err(UnprojectError::OutsideDomain);
        }
        let (sin, cos) = theta.sin_cos();
        let ray = [sin * x / r, sin * y / r, cos];
        Ok(ray)
    }
    fn unproject_with_jacobians(
        &self,
        pixel: [S; 2],
        jp: Option<&mut [[S; 2]; 3]>,
        ji: Option<&mut Self::UnprojectIntrinsicJacobian>,
    ) -> Result<[S; 3], UnprojectError> {
        unproject_jacobians(self, pixel, jp, ji, Some(&self.param[4..]))
    }
}
#[cfg(feature = "kornia-3d-interop")]
impl TryFrom<kornia_3d::camera::FisheyeCamera> for KannalaBrandt4<f64> {
    type Error = InvalidCalibration;
    fn try_from(v: kornia_3d::camera::FisheyeCamera) -> Result<Self, Self::Error> {
        Self::new([v.fx, v.fy, v.cx, v.cy, v.k1, v.k2, v.k3, v.k4])
    }
}
#[cfg(feature = "kornia-3d-interop")]
impl From<KannalaBrandt4<f64>> for kornia_3d::camera::FisheyeCamera {
    fn from(v: KannalaBrandt4<f64>) -> Self {
        let [fx, fy, cx, cy, k1, k2, k3, k4] = v.param;
        Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            k3,
            k4,
        }
    }
}

impl<S: Scalar> TryFrom<[S; 8]> for KannalaBrandt4<S> {
    type Error = InvalidCalibration;
    fn try_from(p: [S; 8]) -> Result<Self, Self::Error> {
        Self::new(p)
    }
}
impl<S: Scalar> From<KannalaBrandt4<S>> for [S; 8] {
    fn from(v: KannalaBrandt4<S>) -> Self {
        v.params()
    }
}
