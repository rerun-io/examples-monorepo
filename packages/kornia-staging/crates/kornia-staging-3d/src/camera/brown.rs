use super::*;

/// Rational Brown–Conrady camera with optional prism, sensor tilt and valid radius.
///
/// OpenCV order is `[fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tau_x,tau_y]`.
/// Brown `p1` is the **y-axis** tangential term, `p2` the **x-axis** term;
/// this is the opposite order from Fisheye624. Prism `(s1,s2)` acts on x,
/// `(s3,s4)` on y. Zero padding represents Brown4/5/8/12.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct BrownConrady<S: Scalar> {
    #[cfg_attr(feature = "serde", serde(rename = "parameters"))]
    pub(crate) param: [S; 18],
    pub(crate) valid_radius: Option<S>,
}
impl<S: Scalar> BrownConrady<S> {
    /// Build from the documented OpenCV order and an undistorted normalized radius.
    /// # Arguments
    /// * `param` - Focal/principal head followed by fourteen OpenCV coefficients.
    /// * `valid_radius` - Optional positive radius; independent of intrinsic increments.
    /// # Errors
    /// Rejects non-finite coefficients, nonpositive focals or an invalid radius.
    pub fn new(param: [S; 18], valid_radius: Option<S>) -> Result<Self, InvalidCalibration> {
        validate_parameters(&param)?;
        if valid_radius.is_some_and(|r| !r.is_finite() || r <= S::zero()) {
            return Err(InvalidCalibration);
        }
        Ok(Self {
            param,
            valid_radius,
        })
    }
    /// Intrinsics in OpenCV order.
    pub fn params(&self) -> [S; 18] {
        self.param
    }
    /// Optional radius in the normalized undistorted plane.
    pub fn valid_radius(&self) -> Option<S> {
        self.valid_radius
    }

    #[inline(always)]
    fn distort(
        &self,
        [x, y]: [S; 2],
        jp: Option<&mut [[S; 2]; 2]>,
        ji: Option<&mut [[S; 14]; 2]>,
    ) -> [S; 2] {
        let [_, _, _, _, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty] = self.param;
        let zero = S::zero();
        let one = S::one();
        let two = c::<S>(2.0);
        let r = x * x + y * y;
        let r4 = r * r;
        let r6 = r4 * r;
        let num = one + r * (k1 + r * (k2 + r * k3));
        let den = one + r * (k4 + r * (k5 + r * k6));
        let radial = num / den;
        let dx = two * p1 * x * y + p2 * (r + two * x * x);
        let dy = two * p2 * x * y + p1 * (r + two * y * y);
        let q = [
            x * radial + dx + (s1 * r + s2 * r4),
            y * radial + dy + (s3 * r + s4 * r4),
        ];
        let derivatives = jp.is_some() || ji.is_some();
        let mut local = [[zero; 2]; 2];
        let mut intrinsic = [[zero; 14]; 2];
        if derivatives {
            let dr = ((k1 + two * k2 * r + c::<S>(3.0) * k3 * r4) * den
                - num * (k4 + two * k5 * r + c::<S>(3.0) * k6 * r4))
                / (den * den);
            local = [
                [
                    radial
                        + two * x * x * dr
                        + two * p1 * y
                        + c::<S>(6.0) * p2 * x
                        + two * x * (s1 + two * s2 * r),
                    two * x * y * dr + two * p1 * x + two * p2 * y + two * y * (s1 + two * s2 * r),
                ],
                [
                    two * x * y * dr + two * p1 * x + two * p2 * y + two * x * (s3 + two * s4 * r),
                    radial
                        + two * y * y * dr
                        + c::<S>(6.0) * p1 * y
                        + two * p2 * x
                        + two * y * (s3 + two * s4 * r),
                ],
            ];
        }
        if ji.is_some() {
            // (numerator k1/k2/k3, denominator k4/k5/k6, radius power).
            for (column, denominator_column, power) in [(0, 5, r), (1, 6, r4), (4, 7, r6)] {
                intrinsic[0][column] = x * power / den;
                intrinsic[1][column] = y * power / den;
                intrinsic[0][denominator_column] = -x * num * power / (den * den);
                intrinsic[1][denominator_column] = -y * num * power / (den * den);
            }
            intrinsic[0][2] = two * x * y;
            intrinsic[1][2] = r + two * y * y;
            intrinsic[0][3] = r + two * x * x;
            intrinsic[1][3] = two * x * y;
            intrinsic[0][8] = r;
            intrinsic[0][9] = r4;
            intrinsic[1][10] = r;
            intrinsic[1][11] = r4;
        }
        let mut out = q;
        // M = [[R33,0,-R13],[0,R33,-R23],[0,0,1]] R(ty) R(tx).
        // This simplifies to the triangular homogeneous matrix below.
        if tx != zero || ty != zero || ji.is_some() {
            let (sx, cx) = tx.sin_cos();
            let (sy, cy) = ty.sin_cos();
            let a = cy * q[1] - sy * sx * q[0];
            let d = sy * q[0] - cy * sx * q[1] + cy * cx;
            out = [cx * q[0] / d, a / d];
            let m = [
                [
                    (cx * d - cx * q[0] * sy) / (d * d),
                    cx * q[0] * cy * sx / (d * d),
                ],
                [
                    (-sy * sx * d - a * sy) / (d * d),
                    (cy * d + a * cy * sx) / (d * d),
                ],
            ];
            if derivatives {
                local = std::array::from_fn(|r| {
                    std::array::from_fn(|i| m[r][0] * local[0][i] + m[r][1] * local[1][i])
                });
            }
            if ji.is_some() {
                intrinsic = std::array::from_fn(|r| {
                    std::array::from_fn(|i| m[r][0] * intrinsic[0][i] + m[r][1] * intrinsic[1][i])
                });
                let ddx = -cy * cx * q[1] - cy * sx;
                let ddy = cy * q[0] + sy * sx * q[1] - sy * cx;
                intrinsic[0][12] = (-sx * q[0] * d - cx * q[0] * ddx) / (d * d);
                intrinsic[1][12] = (-sy * cx * q[0] * d - a * ddx) / (d * d);
                intrinsic[0][13] = -cx * q[0] * ddy / (d * d);
                intrinsic[1][13] = ((-sy * q[1] - cy * sx * q[0]) * d - a * ddy) / (d * d);
            }
        }
        if let Some(j) = jp {
            *j = local;
        }
        if let Some(j) = ji {
            *j = intrinsic;
        }
        out
    }
}
impl<S: Scalar> CameraModel<S> for BrownConrady<S> {
    type IntrinsicJacobian = [[S; 18]; 2];
    type UnprojectIntrinsicJacobian = [[S; 18]; 3];
    #[inline(always)]
    fn project(&self, [x, y, z]: [S; 3]) -> Result<[S; 2], ProjectionReject> {
        if !(x.abs() <= S::largest() && y.abs() <= S::largest() && z.abs() <= S::largest()) {
            return Err(ProjectionReject::NonFinite);
        }
        if z < S::sophus_epsilon_sqrt() {
            return Err(ProjectionReject::BelowMinDepth);
        }
        let xy = [x / z, y / z];
        if self
            .valid_radius
            .is_some_and(|r| xy[0] * xy[0] + xy[1] * xy[1] > r * r)
        {
            return Err(ProjectionReject::OutsideDomain);
        }
        let q = self.distort(xy, None, None);
        let pixel = [
            self.param[0] * q[0] + self.param[2],
            self.param[1] * q[1] + self.param[3],
        ];
        if pixel[0].abs() <= S::largest() && pixel[1].abs() <= S::largest() {
            Ok(pixel)
        } else {
            Err(ProjectionReject::NonFinite)
        }
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
    #[inline(always)]
    fn project_unchecked_with_jacobians(
        &self,
        [x, y, z]: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> [S; 2] {
        let [fx, fy, cx, cy] = self.param[..4].try_into().unwrap();
        let xy = [x / z, y / z];
        let mut local = [[S::zero(); 2]; 2];
        let mut intrinsic = [[S::zero(); 14]; 2];
        let q = self.distort(
            xy,
            if jp.is_some() { Some(&mut local) } else { None },
            if ji.is_some() {
                Some(&mut intrinsic)
            } else {
                None
            },
        );
        if let Some(j) = jp {
            for r in 0..2 {
                let f = if r == 0 { fx } else { fy };
                j[r] = [
                    f * local[r][0] / z,
                    f * local[r][1] / z,
                    -f * (local[r][0] * x + local[r][1] * y) / (z * z),
                ];
            }
        }
        if let Some(j) = ji {
            *j = [[S::zero(); 18]; 2];
            j[0][0] = q[0];
            j[0][2] = S::one();
            j[1][1] = q[1];
            j[1][3] = S::one();
            for i in 0..14 {
                j[0][4 + i] = fx * intrinsic[0][i];
                j[1][4 + i] = fy * intrinsic[1][i];
            }
        }
        [fx * q[0] + cx, fy * q[1] + cy]
    }
    #[inline]
    fn project_with_status(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> ([S; 2], Result<(), ProjectionReject>) {
        let [x, y, z] = point;
        let status = if !finite(point) {
            Err(ProjectionReject::NonFinite)
        } else if z < S::sophus_epsilon_sqrt() {
            Err(ProjectionReject::BelowMinDepth)
        } else if self
            .valid_radius
            .is_some_and(|r| (x / z) * (x / z) + (y / z) * (y / z) > r * r)
        {
            Err(ProjectionReject::OutsideDomain)
        } else {
            Ok(())
        };
        let pixel = self.project_unchecked_with_jacobians(point, jp, ji);
        (
            pixel,
            status.and_then(|()| checked_pixel(pixel).map(|_| ())),
        )
    }
    fn unproject(&self, pixel: [S; 2]) -> Result<[S; 3], UnprojectError> {
        let target = normalized_pixel(pixel, &self.param)?;
        let xy = newton2(target, |xy, j| self.distort(xy, Some(j), None))?;
        if self
            .valid_radius
            .is_some_and(|r| xy[0] * xy[0] + xy[1] * xy[1] > r * r)
        {
            return Err(UnprojectError::OutsideDomain);
        }
        let n = xy[0].hypot(xy[1]).hypot(S::one());
        Ok([xy[0] / n, xy[1] / n, S::one() / n])
    }
    fn unproject_with_jacobians(
        &self,
        pixel: [S; 2],
        jp: Option<&mut [[S; 2]; 3]>,
        ji: Option<&mut Self::UnprojectIntrinsicJacobian>,
    ) -> Result<[S; 3], UnprojectError> {
        unproject_jacobians(self, pixel, jp, ji, None)
    }
}

#[cfg(feature = "serde")]
impl<'de, S: Scalar + serde::Deserialize<'de>> serde::Deserialize<'de> for BrownConrady<S> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Parameters<S> {
            parameters: [S; 18],
            valid_radius: Option<S>,
        }
        let p = Parameters::deserialize(deserializer)?;
        Self::new(p.parameters, p.valid_radius).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "kornia-3d-interop")]
impl TryFrom<kornia_3d::camera::PinholeCamera> for BrownConrady<f64> {
    type Error = InvalidCalibration;
    fn try_from(v: kornia_3d::camera::PinholeCamera) -> Result<Self, Self::Error> {
        let mut param = [0.0; 18];
        param[..8].copy_from_slice(&[v.fx, v.fy, v.cx, v.cy, v.k1, v.k2, v.p1, v.p2]);
        Self::new(param, None)
    }
}
#[cfg(feature = "kornia-3d-interop")]
impl TryFrom<BrownConrady<f64>> for kornia_3d::camera::PinholeCamera {
    type Error = InvalidCalibration;
    fn try_from(v: BrownConrady<f64>) -> Result<Self, Self::Error> {
        if v.valid_radius.is_some() || v.param[8..].iter().any(|x| *x != 0.0) {
            return Err(InvalidCalibration);
        }
        let [fx, fy, cx, cy, k1, k2, p1, p2, ..] = v.param;
        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
        })
    }
}
