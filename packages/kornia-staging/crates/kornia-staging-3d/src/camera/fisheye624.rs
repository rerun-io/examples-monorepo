use super::*;

/// Six angular radial terms followed by tangential and thin-prism distortion.
///
/// Parameter order: `[fx,fy,cx,cy,k1,k2,k3,k4,k5,k6,p_x,p_y,s1,s2,s3,s4]`.
/// Tangential `p_x` (Aria/Fisheye62 p1) acts on x; `p_y` (p2) acts on y.
/// Brown uses the opposite p1/p2 order. Prism `(s1,s2)` acts on x and `(s3,s4)` on y.
/// Aria ties fx=fy. Image bounds and the SDK's visible-pixel circle are caller metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "[S;16]", into = "[S;16]"))]
pub struct Fisheye624<S: Scalar> {
    pub(crate) param: [S; 16],
    theta_max: S,
    r_max: S,
}
impl<S: Scalar> Fisheye624<S> {
    /// Build from the documented sixteen parameters; cache the physical branch.
    /// # Arguments
    /// * `param` - `[fx,fy,cx,cy,k1..k6,p_x,p_y,s1..s4]`.
    /// # Errors
    /// Rejects non-finite parameters, nonpositive focals, or an overflowing branch.
    pub fn new(param: [S; 16]) -> Result<Self, InvalidCalibration> {
        validate_parameters(&param)?;
        let theta_max = angular::peak(&param[4..10]).min(c(std::f64::consts::FRAC_PI_2));
        let r_max = angular::radial(theta_max, &param[4..10]).0;
        if !r_max.is_finite() {
            return Err(InvalidCalibration);
        }
        Ok(Self {
            theta_max,
            r_max,
            param,
        })
    }
    /// Build Fisheye62 by setting all prism coefficients to zero.
    /// # Arguments
    /// * `intrinsics` - `[fx,fy,cx,cy]` in pixels.
    /// * `radial` - Six angular coefficients in increasing power order.
    /// * `tangential_x_y` - `[p_x,p_y]`, opposite Brown's `[p1,p2]` order.
    /// # Errors
    /// Rejects invalid calibration parameters.
    pub fn fisheye62(
        intrinsics: [S; 4],
        radial: [S; 6],
        tangential_x_y: [S; 2],
    ) -> Result<Self, InvalidCalibration> {
        let mut param = [S::zero(); 16];
        param[..4].copy_from_slice(&intrinsics);
        param[4..10].copy_from_slice(&radial);
        param[10..12].copy_from_slice(&tangential_x_y);
        Self::new(param)
    }
    /// Intrinsics in constructor order.
    pub fn params(&self) -> [S; 16] {
        self.param
    }
    /// Endpoint of the increasing angular branch, in radians.
    pub fn max_angle(&self) -> S {
        self.theta_max
    }
    #[inline]
    fn project_core(
        &self,
        [x, y, z]: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut [[S; 16]; 2]>,
    ) -> ([S; 2], S, S) {
        let param = &self.param;
        let [fx, fy, cx, cy] = param[..4].try_into().unwrap();
        let r = (x * x + y * y).sqrt();
        // Preserve the SDK's divide/atan/power-sum order for forward points.
        // The optimizer extension behind the plane uses atan2, like Fisheye62.
        let (theta, a, b, norm) = if z > S::zero() {
            let inv_z = S::one() / z;
            let a = x * inv_z;
            let b = y * inv_z;
            let norm = (a * a + b * b).sqrt();
            (norm.atan(), a, b, norm)
        } else {
            (r.atan2(z), x, y, r)
        };
        let t2 = theta * theta;
        let mut polynomial = S::one();
        let mut power = t2;
        for k in &param[4..10] {
            polynomial += power * *k;
            power *= t2;
        }
        let scale = if norm == S::zero() {
            S::one()
        } else {
            theta / norm
        };
        let q = [polynomial * scale * a, polynomial * scale * b];
        let mut distortion_j = [[S::zero(); 2]; 2];
        let out = Self::distort(
            param,
            q,
            if jp.is_some() || ji.is_some() {
                Some(&mut distortion_j)
            } else {
                None
            },
        );
        if let Some(j) = jp {
            let mut angular_j = [[S::zero(); 3]; 2];
            if r == S::zero() {
                angular_j[0][0] = S::one() / z;
                angular_j[1][1] = S::one() / z;
            } else {
                let rt = theta * polynomial;
                let slope = angular::radial(theta, &param[4..10]).1;
                let norm2 = r * r + z * z;
                let dt = [x / r * z / norm2, y / r * z / norm2, -r / norm2];
                for (row, gradient) in angular_j.iter_mut().enumerate() {
                    let xy = if row == 0 { x } else { y };
                    for col in 0..3 {
                        let dr = if col == 0 {
                            x / r
                        } else if col == 1 {
                            y / r
                        } else {
                            S::zero()
                        };
                        gradient[col] = xy * (slope * dt[col] / r - rt * dr / (r * r));
                        if row == col {
                            gradient[col] += rt / r;
                        }
                    }
                }
            }
            for row in 0..2 {
                let focal = if row == 0 { fx } else { fy };
                for col in 0..3 {
                    j[row][col] = focal
                        * (distortion_j[row][0] * angular_j[0][col]
                            + distortion_j[row][1] * angular_j[1][col]);
                }
            }
        }
        if let Some(j) = ji {
            *j = [[S::zero(); 16]; 2];
            j[0][0] = out[0];
            j[1][1] = out[1];
            j[0][2] = S::one();
            j[1][3] = S::one();
            let mut power = t2;
            let [jx, jy] = &mut *j;
            for (jx, jy) in jx[4..10].iter_mut().zip(&mut jy[4..10]) {
                let dq = [scale * a * power, scale * b * power];
                *jx = fx * (distortion_j[0][0] * dq[0] + distortion_j[0][1] * dq[1]);
                *jy = fy * (distortion_j[1][0] * dq[0] + distortion_j[1][1] * dq[1]);
                power *= t2;
            }
            let [qx, qy] = q;
            let rho = qx * qx + qy * qy;
            let two = c::<S>(2.0);
            j[0][10] = fx * (rho + two * qx * qx);
            j[1][10] = fy * two * qx * qy;
            j[0][11] = fx * two * qx * qy;
            j[1][11] = fy * (rho + two * qy * qy);
            j[0][12] = fx * rho;
            j[0][13] = fx * rho * rho;
            j[1][14] = fy * rho;
            j[1][15] = fy * rho * rho;
        }
        ([fx * out[0] + cx, fy * out[1] + cy], theta, norm)
    }
    fn distort(param: &[S; 16], [x, y]: [S; 2], j: Option<&mut [[S; 2]; 2]>) -> [S; 2] {
        let [px, py, s1, s2, s3, s4] = param[10..].try_into().unwrap();
        let r = x * x + y * y;
        let r4 = r * r;
        let two = c::<S>(2.0);
        let temp = two * (x * px + y * py);
        let u = x + (temp * x + r * px);
        let v = y + (temp * y + r * py);
        if let Some(j) = j {
            let off = two * (x * py + y * px);
            let a = two * (s1 + two * s2 * r);
            let b = two * (s3 + two * s4 * r);
            *j = [
                [
                    S::one() + c::<S>(6.0) * x * px + two * y * py + x * a,
                    off + y * a,
                ],
                [
                    off + x * b,
                    S::one() + c::<S>(6.0) * y * py + two * x * px + y * b,
                ],
            ];
        }
        [u + (s1 * r + s2 * r4), v + (s3 * r + s4 * r4)]
    }
}
impl<S: Scalar> CameraModel<S> for Fisheye624<S> {
    type IntrinsicJacobian = [[S; 16]; 2];
    type UnprojectIntrinsicJacobian = [[S; 16]; 3];
    fn project(&self, point: [S; 3]) -> Result<[S; 2], ProjectionReject> {
        self.project_with_jacobians(point, None, None)
    }
    fn project_with_jacobians(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> Result<[S; 2], ProjectionReject> {
        let (pixel, status) = self.project_with_status(point, jp, ji);
        status.map(|()| pixel)
    }
    fn project_unchecked_with_jacobians(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> [S; 2] {
        self.project_core(point, jp, ji).0
    }
    #[inline]
    fn project_with_status(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> ([S; 2], Result<(), ProjectionReject>) {
        let (pixel, theta, norm) = self.project_core(point, jp, ji);
        let status = if !finite(point) {
            Err(ProjectionReject::NonFinite)
        } else if point[2] <= S::zero() {
            Err(ProjectionReject::BelowMinDepth)
        } else if !norm.is_finite() {
            Err(ProjectionReject::NonFinite)
        } else if theta > self.theta_max {
            Err(ProjectionReject::OutsideDomain)
        } else {
            Ok(())
        };
        (
            pixel,
            status.and_then(|()| checked_pixel(pixel).map(|_| ())),
        )
    }
    fn unproject(&self, pixel: [S; 2]) -> Result<[S; 3], UnprojectError> {
        let target = normalized_pixel(pixel, &self.param)?;
        let [x, y] = newton2(target, |q, j| Self::distort(&self.param, q, Some(j)))?;
        let radius = (x * x + y * y).sqrt();
        if radius == S::zero() {
            return Ok([S::zero(), S::zero(), S::one()]);
        }
        let theta = angular::invert(radius, &self.param[4..10], self.theta_max, self.r_max)?;
        // The horizon is excluded by this model's positive-depth checked domain.
        if theta >= c(std::f64::consts::FRAC_PI_2) {
            return Err(UnprojectError::OutsideDomain);
        }
        let (sin, cos) = theta.sin_cos();
        let ray = [sin * x / radius, sin * y / radius, cos];
        if finite(ray) {
            Ok(ray)
        } else {
            Err(UnprojectError::NonFinite)
        }
    }
    fn unproject_with_jacobians(
        &self,
        pixel: [S; 2],
        jp: Option<&mut [[S; 2]; 3]>,
        ji: Option<&mut Self::UnprojectIntrinsicJacobian>,
    ) -> Result<[S; 3], UnprojectError> {
        unproject_jacobians(self, pixel, jp, ji, Some(&self.param[4..10]))
    }
}

impl<S: Scalar> TryFrom<[S; 16]> for Fisheye624<S> {
    type Error = InvalidCalibration;
    fn try_from(p: [S; 16]) -> Result<Self, Self::Error> {
        Self::new(p)
    }
}
impl<S: Scalar> From<Fisheye624<S>> for [S; 16] {
    fn from(v: Fisheye624<S>) -> Self {
        v.params()
    }
}
