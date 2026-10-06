//! Differentiable cameras in the integer pixel-centre convention.
//!
//! Pixel `(0, 0)` is the centre of the top-left pixel. Bearings are unit length.
//! Matrices are row-major plain arrays. Intrinsic columns follow each constructor.
//! KB4 admits rear rays on its increasing branch; checked Fisheye624 requires
//! positive depth. The negative optical axis has no unique azimuth. Optimizers can use
//! [`CameraModel::project_unchecked`] outside the checked domain.
//!
//! ```
//! use kornia_staging_3d::camera::{CameraModel, Pinhole};
//! let camera = Pinhole::new([400.0, 400.0, 320.0, 240.0]).expect("valid camera calibration");
//! assert_eq!(camera.project([0.0, 0.0, 1.0])?, [320.0, 240.0]);
//! # Ok::<(), kornia_staging_3d::camera::ProjectionReject>(())
//! ```

mod angular;
mod brown;
pub use brown::BrownConrady;
mod fisheye624;
pub use fisheye624::Fisheye624;
mod kb4;
mod pinhole;
pub use kb4::KannalaBrandt4;
pub use pinhole::Pinhole;

use kornia_staging_algebra::Scalar;

#[inline]
pub(crate) fn c<S: Scalar>(value: f64) -> S {
    S::from_literal(value)
}

/// Camera parameters cannot define a finite, positive-focal camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid camera calibration")]
pub struct InvalidCalibration;

fn validate_parameters<S: Scalar>(param: &[S]) -> Result<(), InvalidCalibration> {
    if param.iter().any(|x| !x.is_finite()) || param[0] <= S::zero() || param[1] <= S::zero() {
        return Err(InvalidCalibration);
    }
    Ok(())
}

/// A checked projection cannot produce a pixel.
/// Extends upstream with `OutsideDomain` and `NonFinite`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProjectionReject {
    /// The ray is behind a perspective camera or on the negative optical axis.
    #[error("point is behind the camera or below its minimum depth")]
    BelowMinDepth,
    /// The projection is geometrically invalid (upstream variant).
    #[error("invalid projection")]
    InvalidProjection,
    /// The pixel lies outside caller-supplied image bounds (upstream variant).
    #[error("projection outside image")]
    OutOfImage,
    /// The calibrated radius or increasing angular branch was exceeded.
    #[error("point is outside the valid radius or angle")]
    OutsideDomain,
    /// Input, calibration, or computed pixel is non-finite.
    #[error("non-finite projection")]
    NonFinite,
}

/// A pixel cannot be inverted to a unique physical bearing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum UnprojectError {
    /// Non-finite input or invalid focal lengths.
    #[error("non-finite input or invalid focal length")]
    NonFinite,
    /// Pixel exceeds the physical branch or calibrated radius.
    #[error("pixel is outside the invertible domain")]
    OutsideDomain,
    /// Iteration failed its residual check.
    #[error("inverse distortion did not converge")]
    NoConvergence,
    /// Derivatives cannot be inverted at this point.
    #[error("singular inverse Jacobian")]
    Singular,
}

/// A camera with fixed-size derivatives and unit-bearing inversion.
pub trait CameraModel<S: Scalar>: Copy {
    /// Fixed 2×N intrinsic derivative array (columns in constructor order).
    type IntrinsicJacobian: Copy;
    /// Fixed 3×N inverse intrinsic derivative array.
    type UnprojectIntrinsicJacobian: Copy;
    /// Project a camera-frame point, rejecting invalid geometry.
    ///
    /// # Errors
    /// Returns a typed domain or non-finite rejection; never returns a NaN pixel.
    fn project(&self, point: [S; 3]) -> Result<[S; 2], ProjectionReject>;
    /// Project and differentiate with the same validity checks as `project`.
    /// # Errors
    /// Returns a typed rejection outside the model's checked domain.
    fn project_with_jacobians(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> Result<[S; 2], ProjectionReject>;

    /// Project without domain checks. May produce non-finite values at singularities.
    #[inline]
    fn project_unchecked(&self, point: [S; 3]) -> [S; 2] {
        self.project_unchecked_with_jacobians(point, None, None)
    }
    /// Project without checks and optionally fill analytic row-major derivatives.
    fn project_unchecked_with_jacobians(
        &self,
        point: [S; 3],
        point_jacobian: Option<&mut [[S; 3]; 2]>,
        intrinsic_jacobian: Option<&mut Self::IntrinsicJacobian>,
    ) -> [S; 2];
    /// Project once, retaining the unchecked pixel even when geometry is rejected.
    /// The status has the checked projection's rejection; consumers must inspect it
    /// before using the pixel in downstream geometry. Derivatives are meaningful only on success.
    fn project_with_status(
        &self,
        point: [S; 3],
        jacobian: Option<&mut [[S; 3]; 2]>,
        intrinsic_jacobian: Option<&mut Self::IntrinsicJacobian>,
    ) -> ([S; 2], Result<(), ProjectionReject>);

    /// Invert a pixel into a unit bearing on the physical branch.
    ///
    /// # Errors
    /// Rejects non-finite, out-of-domain, singular or unconverged inputs.
    fn unproject(&self, pixel: [S; 2]) -> Result<[S; 3], UnprojectError>;
    /// Invert a pixel and optionally fill analytic derivatives.
    ///
    /// # Errors
    /// Same errors as `unproject`, plus a singular Jacobian at a branch endpoint.
    fn unproject_with_jacobians(
        &self,
        pixel: [S; 2],
        pixel_jacobian: Option<&mut [[S; 2]; 3]>,
        intrinsic_jacobian: Option<&mut Self::UnprojectIntrinsicJacobian>,
    ) -> Result<[S; 3], UnprojectError>;
    /// Invert onto the z=1 plane, for perspective consumers.
    ///
    /// # Errors
    /// Returns the inverse error, or `OutsideDomain` when the ray is not forward.
    fn unproject_z1(&self, pixel: [S; 2]) -> Result<[S; 3], UnprojectError> {
        let ray = self.unproject(pixel)?;
        if ray[2] <= S::zero() {
            return Err(UnprojectError::OutsideDomain);
        }
        Ok([ray[0] / ray[2], ray[1] / ray[2], S::one()])
    }
}

/// Camera families for heterogeneous rigs. Intrinsic derivatives retain their concrete sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(tag = "model", content = "parameters", rename_all = "snake_case")
)]
pub enum CameraModelKind<S: Scalar> {
    /// Perspective camera.
    Pinhole(Pinhole<S>),
    /// Six-term angular camera with tangential and prism distortion.
    Fisheye624(Fisheye624<S>),
    /// Rational Brown camera, including prism and tilt.
    BrownConrady(BrownConrady<S>),
    /// Four-term angular camera.
    Kb4(KannalaBrandt4<S>),
}
impl<S: Scalar> CameraModelKind<S> {
    /// Project a point through the selected family.
    /// # Errors
    /// Returns the concrete camera's typed rejection.
    pub fn project(&self, point: [S; 3]) -> Result<[S; 2], ProjectionReject> {
        match self {
            Self::Pinhole(v) => v.project(point),
            Self::Fisheye624(v) => v.project(point),
            Self::BrownConrady(v) => v.project(point),
            Self::Kb4(v) => v.project(point),
        }
    }
    /// Project without validity checks.
    pub fn project_unchecked(&self, point: [S; 3]) -> [S; 2] {
        match self {
            Self::Pinhole(v) => v.project_unchecked(point),
            Self::Fisheye624(v) => v.project_unchecked(point),
            Self::BrownConrady(v) => v.project_unchecked(point),
            Self::Kb4(v) => v.project_unchecked(point),
        }
    }
    /// Project once and retain both the pixel and checked-domain status.
    /// A rejected pixel is unchecked and must not enter downstream geometry.
    #[inline]
    pub fn project_with_status(
        &self,
        point: [S; 3],
        jacobian: Option<&mut [[S; 3]; 2]>,
    ) -> ([S; 2], Result<(), ProjectionReject>) {
        match self {
            Self::Pinhole(v) => v.project_with_status(point, jacobian, None),
            Self::Fisheye624(v) => v.project_with_status(point, jacobian, None),
            Self::BrownConrady(v) => v.project_with_status(point, jacobian, None),
            Self::Kb4(v) => v.project_with_status(point, jacobian, None),
        }
    }
    /// Return a unit bearing.
    /// # Errors
    /// Returns the concrete camera's inverse failure.
    pub fn unproject(&self, pixel: [S; 2]) -> Result<[S; 3], UnprojectError> {
        match self {
            Self::Pinhole(v) => v.unproject(pixel),
            Self::Fisheye624(v) => v.unproject(pixel),
            Self::BrownConrady(v) => v.unproject(pixel),
            Self::Kb4(v) => v.unproject(pixel),
        }
    }
}

#[inline]
fn finite<S: Scalar, const N: usize>(values: [S; N]) -> bool {
    values.iter().all(|v| v.is_finite())
}
#[inline]
fn checked_pixel<S: Scalar>(pixel: [S; 2]) -> Result<[S; 2], ProjectionReject> {
    if finite(pixel) {
        Ok(pixel)
    } else {
        Err(ProjectionReject::NonFinite)
    }
}
fn normalized_pixel<S: Scalar>(pixel: [S; 2], intrinsics: &[S]) -> Result<[S; 2], UnprojectError> {
    if !finite(pixel) {
        return Err(UnprojectError::NonFinite);
    }
    let xy = [
        (pixel[0] - intrinsics[2]) / intrinsics[0],
        (pixel[1] - intrinsics[3]) / intrinsics[1],
    ];
    if finite(xy) {
        Ok(xy)
    } else {
        Err(UnprojectError::NonFinite)
    }
}

fn unproject_jacobians<S: Scalar, C, const N: usize>(
    camera: &C,
    pixel: [S; 2],
    jp: Option<&mut [[S; 2]; 3]>,
    ji: Option<&mut [[S; N]; 3]>,
    angular_terms: Option<&[S]>,
) -> Result<[S; 3], UnprojectError>
where
    C: CameraModel<S, IntrinsicJacobian = [[S; N]; 2], UnprojectIntrinsicJacobian = [[S; N]; 3]>,
{
    let ray = camera.unproject(pixel)?;
    if jp.is_some() || ji.is_some() {
        if let Some(terms) = angular_terms {
            let theta = (ray[0] * ray[0] + ray[1] * ray[1]).sqrt().atan2(ray[2]);
            if angular::slope_is_singular(theta, terms) {
                return Err(UnprojectError::Singular);
            }
        }
        let mut forward = [[S::zero(); 3]; 2];
        let mut intrinsic = [[S::zero(); N]; 2];
        camera.project_unchecked_with_jacobians(ray, Some(&mut forward), Some(&mut intrinsic));
        inverse_jacobians(forward, intrinsic, jp, ji)?;
    }
    Ok(ray)
}

// The differential of a homogeneous projection annihilates its unit bearing.
// J^T (J J^T)^-1 is therefore the inverse differential in that bearing's tangent plane.
fn inverse_jacobians<S: Scalar, const N: usize>(
    jp: [[S; 3]; 2],
    ji: [[S; N]; 2],
    pixel: Option<&mut [[S; 2]; 3]>,
    intrinsic: Option<&mut [[S; N]; 3]>,
) -> Result<(), UnprojectError> {
    let dot = |a: [S; 3], b: [S; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (a, b, d) = (dot(jp[0], jp[0]), dot(jp[0], jp[1]), dot(jp[1], jp[1]));
    let det = a * d - b * b;
    if !det.is_finite() || det <= S::zero() {
        return Err(UnprojectError::Singular);
    }
    let ju = std::array::from_fn::<_, 3, _>(|r| {
        [
            (jp[0][r] * d - jp[1][r] * b) / det,
            (jp[1][r] * a - jp[0][r] * b) / det,
        ]
    });
    if let Some(out) = pixel {
        *out = ju;
    }
    if let Some(out) = intrinsic {
        *out = std::array::from_fn(|r| {
            std::array::from_fn(|i| -ju[r][0] * ji[0][i] - ju[r][1] * ji[1][i])
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;

// Damped Newton on the full two-dimensional distortion. Every accepted step
// reduces the residual, and success requires the residual, not the step, to be small.
fn newton2<S: Scalar>(
    target: [S; 2],
    distort: impl Fn([S; 2], &mut [[S; 2]; 2]) -> [S; 2],
) -> Result<[S; 2], UnprojectError> {
    let mut xy = target;
    // Reuse the accepted line-search value and Jacobian on the next iteration.
    let mut j = [[S::zero(); 2]; 2];
    let mut value = distort(xy, &mut j);
    for _ in 0..80 {
        if !finite(value) {
            return Err(UnprojectError::NonFinite);
        }
        let residual = [value[0] - target[0], value[1] - target[1]];
        let norm = residual[0].abs().max(residual[1].abs());
        if norm <= inverse_epsilon::<S>() * (S::one() + target[0].abs().max(target[1].abs())) {
            return Ok(xy);
        }
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if !det.is_finite() || det == S::zero() {
            return Err(UnprojectError::Singular);
        }
        let step = [
            (j[1][1] * residual[0] - j[0][1] * residual[1]) / det,
            (-j[1][0] * residual[0] + j[0][0] * residual[1]) / det,
        ];
        let mut scale = S::one();
        let mut accepted = false;
        for _ in 0..20 {
            let candidate = [xy[0] - scale * step[0], xy[1] - scale * step[1]];
            // Further halvings cannot change a candidate that already rounds to xy.
            if candidate == xy {
                break;
            }
            value = distort(candidate, &mut j);
            let next = (value[0] - target[0])
                .abs()
                .max((value[1] - target[1]).abs());
            if finite(value) && next < norm {
                xy = candidate;
                accepted = true;
                break;
            }
            scale *= c::<S>(0.5);
        }
        if !accepted {
            return Err(UnprojectError::NoConvergence);
        }
    }
    Err(UnprojectError::NoConvergence)
}

/// Residual tolerance for camera inverse solves in normalized coordinates.
/// The sealed scalar set contains only f32 and f64; this policy belongs to cameras.
#[inline]
pub fn inverse_epsilon<S: Scalar>() -> S {
    S::from_literal(if std::mem::size_of::<S>() == std::mem::size_of::<f32>() {
        2e-7
    } else {
        2e-15
    })
}
// SymForce oracles generated by packages/handfit/tools/codegen.py at b8ff66ba.
// The generator now emits hand FK only; these archived camera outputs are test-only.
#[cfg(test)]
#[allow(non_snake_case, clippy::all)]
mod test_oracles {
    pub mod fisheye62;
    pub mod pinhole;
}

