use super::*;

/// Enhanced unified camera model (EUCM), `[fx, fy, cx, cy, alpha, beta]`.
///
/// Khomutenko, Garcia and Martinet, "An Enhanced Unified Camera Model" (RA-L 2016).
/// With `rho = sqrt(beta (x^2 + y^2) + z^2)`, projection is valid for `z > -w rho`;
/// with `alpha > 0.5` the inverse is defined only for `r^2 < 1 / (beta (2 alpha - 1))`.
/// Parameters are immutable so the cached domain bound cannot become stale.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "[S;6]", into = "[S;6]"))]
pub struct Eucm<S: Scalar> {
    pub(crate) param: [S; 6],
    w: S,
}
impl<S: Scalar> Eucm<S> {
    /// Build an EUCM camera.
    /// # Arguments
    /// * `param` - `[fx, fy, cx, cy, alpha, beta]`, `alpha` in `[0, 1]`, `beta > 0`.
    /// # Errors
    /// Rejects non-finite parameters, nonpositive focals, or `alpha`/`beta` out of range.
    pub fn new(param: [S; 6]) -> Result<Self, InvalidCalibration> {
        validate_parameters(&param)?;
        let [.., alpha, beta] = param;
        if !(S::zero()..=S::one()).contains(&alpha) || beta <= S::zero() {
            return Err(InvalidCalibration);
        }
        let w = if alpha > c(0.5) {
            (S::one() - alpha) / alpha
        } else {
            alpha / (S::one() - alpha)
        };
        Ok(Self { param, w })
    }
    /// Intrinsics in constructor order.
    pub fn params(&self) -> [S; 6] {
        self.param
    }
    /// Domain bound `w`: a point is valid when `z > -w * rho`.
    pub fn domain_bound(&self) -> S {
        self.w
    }
}
impl<S: Scalar> CameraModel<S> for Eucm<S> {
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
        let [fx, fy, cx, cy, alpha, beta] = self.param;
        let one = S::one();
        let r2 = x * x + y * y;
        let rho = (beta * r2 + z * z).sqrt();
        let n = alpha * rho + (one - alpha) * z;
        let (mx, my) = (x / n, y / n);
        if let Some(j) = jp {
            let denom = n * n * rho;
            let (ab, add) = (alpha * beta, n * rho);
            let dz = alpha * z + (one - alpha) * rho;
            *j = [
                [fx * (add - ab * x * x), -fx * ab * x * y, -fx * x * dz],
                [-fy * ab * x * y, fy * (add - ab * y * y), -fy * y * dz],
            ]
            .map(|row| row.map(|v| v / denom));
        }
        if let Some(j) = ji {
            let n2 = n * n;
            let (tx, ty) = (-fx * x / n2, -fy * y / n2);
            let (dalpha, dbeta) = (rho - z, c::<S>(0.5) * alpha * r2 / rho);
            *j = [
                [mx, S::zero(), one, S::zero(), tx * dalpha, tx * dbeta],
                [S::zero(), my, S::zero(), one, ty * dalpha, ty * dbeta],
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
        let [x, y, z] = point;
        let rho = (self.param[5] * (x * x + y * y) + z * z).sqrt();
        let status = if !finite(point) || !rho.is_finite() {
            Err(ProjectionReject::NonFinite)
        } else if rho == S::zero() {
            Err(ProjectionReject::BelowMinDepth)
        } else if z <= -self.w * rho {
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
        let [.., alpha, beta] = self.param;
        let one = S::one();
        let r2 = mx * mx + my * my;
        let bound = (alpha + alpha - one) * beta;
        // The boundary itself is rejected, as in Basalt's `unproject`.
        if alpha > c(0.5) && r2 * bound >= one {
            return Err(UnprojectError::OutsideDomain);
        }
        let mz =
            (one - alpha * alpha * beta * r2) / (alpha * (one - bound * r2).sqrt() + one - alpha);
        let norm = (r2 + mz * mz).sqrt();
        let ray = [mx / norm, my / norm, mz / norm];
        if !finite(ray) {
            return Err(UnprojectError::NonFinite);
        }
        // Keep the inverse inside the forward projection's domain.
        let rho = (beta * (ray[0] * ray[0] + ray[1] * ray[1]) + ray[2] * ray[2]).sqrt();
        if ray[2] <= -self.w * rho {
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
impl<S: Scalar> TryFrom<[S; 6]> for Eucm<S> {
    type Error = InvalidCalibration;
    fn try_from(p: [S; 6]) -> Result<Self, Self::Error> {
        Self::new(p)
    }
}
impl<S: Scalar> From<Eucm<S>> for [S; 6] {
    fn from(v: Eucm<S>) -> Self {
        v.params()
    }
}
