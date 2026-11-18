use super::*;

/// Equirectangular (longitude-latitude) camera, `[fx, fy, cx, cy, lon_max]`.
///
/// Camera frame RDF with +Z at the image centre: longitude `atan2(x, z)` grows to the
/// right, latitude `atan2(y, hypot(x, z))` grows downward, `u = fx lon + cx` and
/// `v = fy lat + cy`. `lon_max` (radians, in `(0, pi]`) is the half-window of valid
/// longitudes, so a window narrower than the sphere never wraps. The poles, where the
/// longitude is undefined, are outside the domain. A full `W x H` panorama has
/// `fx = W / 2pi`, `fy = H / pi`, `cx = W / 2 - 0.5`, `cy = H / 2 - 0.5`, `lon_max = pi`;
/// this matches COLMAP's `EQUIRECTANGULAR` (`sensor/models/spherical.h`) minus half a pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "[S;5]", into = "[S;5]"))]
pub struct Equirectangular<S: Scalar> {
    pub(crate) param: [S; 5],
}
impl<S: Scalar> Equirectangular<S> {
    /// Build an equirectangular camera.
    /// # Arguments
    /// * `param` - `[fx, fy, cx, cy, lon_max]`, pixels per radian, pixels, radians.
    /// # Errors
    /// Rejects non-finite parameters, nonpositive focals, or `lon_max` outside `(0, pi]`.
    pub fn new(param: [S; 5]) -> Result<Self, InvalidCalibration> {
        validate_parameters(&param)?;
        if !(param[4] > S::zero() && param[4] <= S::pi()) {
            return Err(InvalidCalibration);
        }
        Ok(Self { param })
    }
    /// The full sphere on a `width x height` panorama, longitude window `[-pi, pi]`.
    /// # Errors
    /// Rejects a zero dimension.
    pub fn panorama(width: u32, height: u32) -> Result<Self, InvalidCalibration> {
        let [w, h] = [width, height].map(|v| c::<S>(f64::from(v)));
        Self::new([
            w / S::two_pi(),
            h / S::pi(),
            w * c(0.5) - c(0.5),
            h * c(0.5) - c(0.5),
            S::pi(),
        ])
    }
    /// One lens of a dual-fisheye rig as a centred `width x height` tile.
    /// # Arguments
    /// * `pixels_per_radian` - Equal angular resolution in longitude and latitude.
    /// * `lon_max` - Longitude half-window, radians, e.g. 96 degrees for a hemisphere plus overlap.
    /// # Errors
    /// Rejects a nonpositive resolution or `lon_max` outside `(0, pi]`.
    pub fn hemisphere_tile(
        width: u32,
        height: u32,
        pixels_per_radian: S,
        lon_max: S,
    ) -> Result<Self, InvalidCalibration> {
        let [w, h] = [width, height].map(|v| c::<S>(f64::from(v)));
        Self::new([
            pixels_per_radian,
            pixels_per_radian,
            w * c(0.5) - c(0.5),
            h * c(0.5) - c(0.5),
            lon_max,
        ])
    }
    /// Intrinsics in constructor order.
    pub fn params(&self) -> [S; 5] {
        self.param
    }
    /// Longitude half-window, radians.
    pub fn lon_max(&self) -> S {
        self.param[4]
    }
}
impl<S: Scalar> CameraModel<S> for Equirectangular<S> {
    type IntrinsicJacobian = [[S; 5]; 2];
    type UnprojectIntrinsicJacobian = [[S; 5]; 3];
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
        let [fx, fy, cx, cy, _] = self.param;
        let rho = x.hypot(z);
        let lon = x.atan2(z);
        let lat = y.atan2(rho);
        if let Some(j) = jp {
            let (rho2, n2) = (rho * rho, rho * rho + y * y);
            *j = [
                [fx * z / rho2, S::zero(), -fx * x / rho2],
                [
                    -fy * x * y / (n2 * rho),
                    fy * rho / n2,
                    -fy * z * y / (n2 * rho),
                ],
            ];
        }
        if let Some(j) = ji {
            let o = S::zero();
            *j = [[lon, o, S::one(), o, o], [o, lat, o, S::one(), o]];
        }
        [fx * lon + cx, fy * lat + cy]
    }
    #[inline]
    fn project_with_status(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> ([S; 2], Result<(), ProjectionReject>) {
        let [x, y, z] = point;
        let rho = x.hypot(z);
        let status = if !finite(point) {
            Err(ProjectionReject::NonFinite)
        } else if rho == S::zero() && y == S::zero() {
            Err(ProjectionReject::BelowMinDepth)
        } else if rho <= S::sophus_epsilon_sqrt() * rho.hypot(y) || x.atan2(z).abs() > self.param[4]
        {
            // Poles have no longitude; the window keeps the image from wrapping.
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
        let [lon, lat] = normalized_pixel(pixel, &self.param)?;
        let (sin_lat, cos_lat) = lat.sin_cos();
        if lon.abs() > self.param[4]
            || lat.abs() >= S::frac_pi_2()
            || cos_lat <= S::sophus_epsilon_sqrt()
        {
            return Err(UnprojectError::OutsideDomain);
        }
        let (sin_lon, cos_lon) = lon.sin_cos();
        Ok([cos_lat * sin_lon, sin_lat, cos_lat * cos_lon])
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
impl<S: Scalar> TryFrom<[S; 5]> for Equirectangular<S> {
    type Error = InvalidCalibration;
    fn try_from(p: [S; 5]) -> Result<Self, Self::Error> {
        Self::new(p)
    }
}
impl<S: Scalar> From<Equirectangular<S>> for [S; 5] {
    fn from(v: Equirectangular<S>) -> Self {
        v.params()
    }
}
