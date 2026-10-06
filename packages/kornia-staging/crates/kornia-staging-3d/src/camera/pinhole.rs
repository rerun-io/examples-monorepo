use super::*;

/// Ideal perspective camera, with `[fx, fy, cx, cy]` parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "[S;4]", into = "[S;4]"))]
pub struct Pinhole<S: Scalar> {
    pub(crate) param: [S; 4],
}
impl<S: Scalar> Pinhole<S> {
    /// Build from focal lengths and principal point, in pixels.
    /// # Arguments
    /// * `param` - `[fx, fy, cx, cy]`; checked inverse requires positive focal lengths.
    /// # Errors
    /// Rejects non-finite parameters, nonpositive focals, or an overflowing branch.
    pub fn new(param: [S; 4]) -> Result<Self, InvalidCalibration> {
        validate_parameters(&param)?;
        Ok(Self { param })
    }
    /// Intrinsics in constructor order.
    pub fn params(&self) -> [S; 4] {
        self.param
    }
}
impl<S: Scalar> CameraModel<S> for Pinhole<S> {
    type IntrinsicJacobian = [[S; 4]; 2];
    type UnprojectIntrinsicJacobian = [[S; 4]; 3];
    #[inline]
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
    #[inline]
    fn project_unchecked_with_jacobians(
        &self,
        [x, y, z]: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> [S; 2] {
        let [fx, fy, cx, cy] = self.param;
        if let Some(j) = jp {
            *j = [
                [fx / z, S::zero(), -fx * x / (z * z)],
                [S::zero(), fy / z, -fy * y / (z * z)],
            ];
        }
        if let Some(j) = ji {
            *j = [
                [x / z, S::zero(), S::one(), S::zero()],
                [S::zero(), y / z, S::zero(), S::one()],
            ];
        }
        [fx * x / z + cx, fy * y / z + cy]
    }
    #[inline]
    fn project_with_status(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> ([S; 2], Result<(), ProjectionReject>) {
        let status = if !finite(point) {
            Err(ProjectionReject::NonFinite)
        } else if point[2] < S::sophus_epsilon_sqrt() {
            Err(ProjectionReject::BelowMinDepth)
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
        let [x, y] = normalized_pixel(pixel, &self.param)?;
        let n = x.hypot(y).hypot(S::one());
        if !n.is_finite() {
            return Err(UnprojectError::NonFinite);
        }
        let ray = [x / n, y / n, S::one() / n];
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
        unproject_jacobians(self, pixel, jp, ji, None)
    }
}
#[cfg(feature = "kornia-3d-interop")]
impl From<Pinhole<f64>> for kornia_3d::camera::PinholeCamera {
    fn from(value: Pinhole<f64>) -> Self {
        let [fx, fy, cx, cy] = value.param;
        Self {
            fx,
            fy,
            cx,
            cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}
impl<S: Scalar> TryFrom<[S; 4]> for Pinhole<S> {
    type Error = InvalidCalibration;
    fn try_from(p: [S; 4]) -> Result<Self, Self::Error> {
        Self::new(p)
    }
}
impl<S: Scalar> From<Pinhole<S>> for [S; 4] {
    fn from(v: Pinhole<S>) -> Self {
        v.params()
    }
}
