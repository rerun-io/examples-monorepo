use super::*;

/// Double sphere camera, `[fx, fy, cx, cy, xi, alpha]`.
///
/// Usenko, Demmel and Cremers, "The Double Sphere Camera Model" (3DV 2018), eqs. 40-47.
/// Projection is valid for `z > -w2 * |p|`; with `alpha > 0.5` the inverse is
/// defined only for `r^2 < 1 / (2 alpha - 1)`. Rear rays inside that cone are valid.
/// Parameters are immutable so the cached domain bound cannot become stale.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "[S;6]", into = "[S;6]"))]
pub struct DoubleSphere<S: Scalar> {
    pub(crate) param: [S; 6],
    w2: S,
}
impl<S: Scalar> DoubleSphere<S> {
    /// Build a double sphere camera.
    /// # Arguments
    /// * `param` - `[fx, fy, cx, cy, xi, alpha]`, `xi` in `[-1, 1]`, `alpha` in `[0, 1]`.
    /// # Errors
    /// Rejects non-finite parameters, nonpositive focals, or `xi`/`alpha` out of range.
    pub fn new(param: [S; 6]) -> Result<Self, InvalidCalibration> {
        validate_parameters(&param)?;
        let [.., xi, alpha] = param;
        if xi.abs() > S::one() || !(S::zero()..=S::one()).contains(&alpha) {
            return Err(InvalidCalibration);
        }
        let w1 = if alpha > c(0.5) {
            (S::one() - alpha) / alpha
        } else {
            alpha / (S::one() - alpha)
        };
        let w2 = (w1 + xi) / (c::<S>(2.0) * w1 * xi + xi * xi + S::one()).sqrt();
        Ok(Self { param, w2 })
    }
    /// Intrinsics in constructor order.
    pub fn params(&self) -> [S; 6] {
        self.param
    }
    /// Domain bound `w2`: a point is valid when `z > -w2 * |p|`.
    pub fn domain_bound(&self) -> S {
        self.w2
    }
}
impl<S: Scalar> CameraModel<S> for DoubleSphere<S> {
    type IntrinsicJacobian = [[S; 6]; 2];
    type UnprojectIntrinsicJacobian = [[S; 6]; 3];
    #[inline]
    fn project(&self, point: [S; 3]) -> Result<[S; 2], ProjectionReject> {
        self.project_with_jacobians(point, None, None)
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
        [x, y, z]: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> [S; 2] {
        let [fx, fy, cx, cy, xi, alpha] = self.param;
        let one = S::one();
        let r2 = x * x + y * y;
        let d1 = (r2 + z * z).sqrt();
        let k = xi * d1 + z;
        let d2 = (r2 + k * k).sqrt();
        let n = alpha * d2 + (one - alpha) * k;
        let (mx, my) = (x / n, y / n);
        let n2 = n * n;
        if let Some(j) = jp {
            let dn = (xi * (one - alpha) / d1 + alpha * (xi * k / d1 + one) / d2) / n2;
            let tt2 = xi * z / d1 + one;
            let dz = ((one - alpha) * tt2 + alpha * k * tt2 / d2) / n2;
            *j = [
                [fx * (one / n - x * x * dn), -fx * x * y * dn, -fx * x * dz],
                [-fy * x * y * dn, fy * (one / n - y * y * dn), -fy * y * dz],
            ];
        }
        if let Some(j) = ji {
            let dxi = (alpha - one - alpha * k / d2) * d1 / n2;
            let dalpha = (k - d2) / n2;
            *j = [
                [mx, S::zero(), one, S::zero(), fx * x * dxi, fx * x * dalpha],
                [S::zero(), my, S::zero(), one, fy * y * dxi, fy * y * dalpha],
            ];
        }
        [fx * mx + cx, fy * my + cy]
    }
    #[inline]
    fn project_with_status(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> ([S; 2], Result<(), ProjectionReject>) {
        let d1 = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
        let status = if !finite(point) || !d1.is_finite() {
            Err(ProjectionReject::NonFinite)
        } else if d1 == S::zero() {
            Err(ProjectionReject::BelowMinDepth)
        } else if point[2] <= -self.w2 * d1 {
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
        let [mx, my] = normalized_pixel(pixel, &self.param)?;
        let [.., xi, alpha] = self.param;
        let one = S::one();
        let r2 = mx * mx + my * my;
        let two_alpha = alpha + alpha - one;
        // The boundary itself is rejected, as in Basalt's `unproject`.
        if alpha > c(0.5) && r2 * two_alpha >= one {
            return Err(UnprojectError::OutsideDomain);
        }
        let mz = (one - alpha * alpha * r2) / (alpha * (one - two_alpha * r2).sqrt() + one - alpha);
        let k = (mz * xi + (mz * mz + (one - xi * xi) * r2).sqrt()) / (mz * mz + r2);
        let ray = [k * mx, k * my, k * mz - xi];
        if !finite(ray) {
            return Err(UnprojectError::NonFinite);
        }
        // Keep the inverse inside the forward projection's domain.
        if ray[2] <= -self.w2 {
            return Err(UnprojectError::OutsideDomain);
        }
        Ok(ray)
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
impl<S: Scalar> TryFrom<[S; 6]> for DoubleSphere<S> {
    type Error = InvalidCalibration;
    fn try_from(p: [S; 6]) -> Result<Self, Self::Error> {
        Self::new(p)
    }
}
impl<S: Scalar> From<DoubleSphere<S>> for [S; 6] {
    fn from(v: DoubleSphere<S>) -> Self {
        v.params()
    }
}
