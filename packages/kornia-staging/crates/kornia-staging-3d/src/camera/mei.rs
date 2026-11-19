use super::*;

/// The distortion a [`Mei`] camera applies to the normalized sphere point `m`, `r2 = |m|^2`.
///
/// `V3`: `x_d = m_x R + 2 p1 m_x m_y + p2 (r2 + 2 m_x^2)`, `y_d = m_y R + p1 (r2 + 2 m_y^2) + 2 p2 m_x m_y`,
/// `R = 1 + k1 r2 + k2 r2^2 + k3 r2^3`.
/// `V6`: `x_d = m_x R + (r2 + 2 m_x^2) a + 2 m_x m_y b + d9 r2 + d11 r2^2`,
/// `y_d = m_y R + (r2 + 2 m_y^2) b + 2 m_x m_y a + d10 r2 + d12 r2^2`, `a = d5 + d7 r2`, `b = d6 + d8 r2`,
/// `R = 1 + d0 r2 + .. + d4 r2^5` (5 radial, 4 tangential, 4 thin-prism terms).
/// The names are the Insta360 lens-string versions (Gyroflow `insta360.rs`; video-types `MeiDistortion`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum MeiDistortion<S> {
    /// Three radial terms and Brown tangential `p1 p2`.
    V3 {
        /// Radial `k1 k2 k3`.
        k: [S; 3],
        /// Tangential `p1 p2`.
        p: [S; 2],
    },
    /// `d0..d4` radial, `d5..d8` tangential, `d9..d12` thin prism.
    V6 {
        /// Coefficients `d0..d12`.
        d: [S; 13],
    },
}

impl<S: Scalar> MeiDistortion<S> {
    fn finite(&self) -> bool {
        match self {
            Self::V3 { k, p } => finite(*k) && finite(*p),
            Self::V6 { d } => finite(*d),
        }
    }

    /// Distort `m`, optionally writing `d(x_d, y_d) / d(m_x, m_y)` row-major.
    /// The value's operation order is the factory reader's, so f64 pixels stay bit-identical to it.
    fn apply(&self, [mx, my]: [S; 2], jacobian: Option<&mut [[S; 2]; 2]>) -> [S; 2] {
        let (one, two) = (S::one(), c::<S>(2.0));
        let r2 = mx * mx + my * my;
        let radial: &[S] = match self {
            Self::V3 { k, .. } => k,
            Self::V6 { d } => &d[..5],
        };
        let mut power = one;
        let mut gain = one;
        // d gain / d r2.
        let mut slope = S::zero();
        for (i, k) in radial.iter().enumerate() {
            slope += c::<S>((i + 1) as f64) * *k * power;
            power *= r2;
            gain += *k * power;
        }
        match *self {
            Self::V3 { p: [p1, p2], .. } => {
                if let Some(j) = jacobian {
                    *j = [
                        [
                            gain + two * mx * mx * slope + two * p1 * my + c::<S>(6.0) * p2 * mx,
                            two * mx * my * slope + two * p1 * mx + two * p2 * my,
                        ],
                        [
                            two * mx * my * slope + two * p1 * mx + two * p2 * my,
                            gain + two * my * my * slope + c::<S>(6.0) * p1 * my + two * p2 * mx,
                        ],
                    ];
                }
                [
                    mx * gain + two * p1 * mx * my + p2 * (r2 + two * mx * mx),
                    my * gain + p1 * (r2 + two * my * my) + two * p2 * mx * my,
                ]
            }
            Self::V6 { d } => {
                let (a, b) = (d[5] + d[7] * r2, d[6] + d[8] * r2);
                if let Some(j) = jacobian {
                    let four = c::<S>(4.0);
                    let (sx, sy) = (r2 + two * mx * mx, r2 + two * my * my);
                    let xy = two * mx * my;
                    // Every r2-dependent term differentiates through d r2 = 2 m.
                    let x_r2 = mx * slope + a + sx * d[7] + xy * d[8] + d[9] + two * d[11] * r2;
                    let y_r2 = my * slope + b + sy * d[8] + xy * d[7] + d[10] + two * d[12] * r2;
                    *j = [
                        [
                            gain + four * mx * a + two * my * b + two * mx * x_r2,
                            two * mx * b + two * my * x_r2,
                        ],
                        [
                            two * my * a + two * mx * y_r2,
                            gain + four * my * b + two * mx * a + two * my * y_r2,
                        ],
                    ];
                }
                [
                    mx * gain
                        + (r2 + two * mx * mx) * a
                        + two * mx * my * b
                        + d[9] * r2
                        + d[11] * r2 * r2,
                    my * gain
                        + (r2 + two * my * my) * b
                        + two * mx * my * a
                        + d[10] * r2
                        + d[12] * r2 * r2,
                ]
            }
        }
    }
}

/// Unified (Mei) sphere camera with [`MeiDistortion`], `[fx, fy, cx, cy, xi]` plus `half_field_rad` and the
/// distortion: the Insta360 factory lens model.
///
/// C. Mei and P. Rives, "Single View Point Omnidirectional Camera Calibration from Planar Grids" (ICRA 2007):
/// a ray `P` maps to `m = (X, Y) / (Z + xi |P|)`, then [`MeiDistortion`], then `u = fx x_d + cx`,
/// `v = fy y_d + cy`. A ray more than `half_field_rad` off the optical axis does not project (the image circle
/// ends there), rear rays inside that cone do. The inverse solves the distortion by damped Newton and lifts
/// through the sphere in closed form. Intrinsic derivatives cover `fx fy cx cy xi`, not the distortion
/// coefficients. The device kernel (`kornia_staging_gpu::camera::mei`) evaluates the same rules in f32.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(try_from = "formats::MeiParams<S>", into = "formats::MeiParams<S>")
)]
pub struct Mei<S: Scalar> {
    param: [S; 5],
    half_field_rad: S,
    distortion: MeiDistortion<S>,
    cos_half_field: S,
}

impl<S: Scalar> Mei<S> {
    /// Build a Mei camera.
    /// # Arguments
    /// * `param` - `[fx, fy, cx, cy, xi]`, `xi >= 0`.
    /// * `half_field_rad` - largest angle off the optical axis that projects, in `(0, pi]`; the cone must stay
    ///   inside the sphere's one-to-one region, `cos(half_field_rad) > -xi` for `xi <= 1` (where
    ///   `z + xi |P| > 0`) and `cos(half_field_rad) > -1 / xi` for `xi > 1`.
    /// * `distortion` - the distortion of the normalized sphere point.
    /// # Errors
    /// Rejects non-finite values, nonpositive focals, a negative `xi` or a field outside those bounds.
    pub fn new(
        param: [S; 5],
        half_field_rad: S,
        distortion: MeiDistortion<S>,
    ) -> Result<Self, InvalidCalibration> {
        validate_parameters(&param)?;
        let xi = param[4];
        let cos_half_field = half_field_rad.cos();
        if xi < S::zero()
            || !distortion.finite()
            || !(half_field_rad > S::zero() && half_field_rad <= S::pi())
            || (xi > S::one() && cos_half_field * xi <= -S::one())
            || (xi <= S::one() && cos_half_field <= -xi)
        {
            return Err(InvalidCalibration);
        }
        Ok(Self {
            param,
            half_field_rad,
            distortion,
            cos_half_field,
        })
    }
    /// `[fx, fy, cx, cy, xi]`.
    pub fn params(&self) -> [S; 5] {
        self.param
    }
    /// Largest angle off the optical axis that projects, radians.
    pub fn half_field_rad(&self) -> S {
        self.half_field_rad
    }
    /// The distortion of the normalized sphere point.
    pub fn distortion(&self) -> MeiDistortion<S> {
        self.distortion
    }
    /// `cos(half_field_rad)`: a ray whose `z` is below `|P|` times this lies outside the field.
    pub fn cos_half_field(&self) -> S {
        self.cos_half_field
    }
}

impl<S: Scalar> CameraModel<S> for Mei<S> {
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
    fn project_unchecked_with_jacobians(
        &self,
        [x, y, z]: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> [S; 2] {
        let [fx, fy, cx, cy, xi] = self.param;
        let norm = (x * x + y * y + z * z).sqrt();
        let den = z + xi * norm;
        let m = [x / den, y / den];
        let mut dd = [[S::zero(); 2]; 2];
        let wants = jp.is_some() || ji.is_some();
        let [xd, yd] = self.distortion.apply(m, wants.then_some(&mut dd));
        if let Some(j) = jp {
            // dm/dP = ([I 0] - m (xi P / |P| + e_z)^T) / den.
            let dden = [xi * x / norm, xi * y / norm, xi * z / norm + S::one()];
            let dm: [[S; 3]; 2] = std::array::from_fn(|r| {
                std::array::from_fn(|k| {
                    (if r == k { S::one() } else { S::zero() } - m[r] * dden[k]) / den
                })
            });
            let f = [fx, fy];
            *j = std::array::from_fn(|r| {
                std::array::from_fn(|k| f[r] * (dd[r][0] * dm[0][k] + dd[r][1] * dm[1][k]))
            });
        }
        if let Some(j) = ji {
            // dm/dxi = -m |P| / den.
            let dm = [-m[0] * norm / den, -m[1] * norm / den];
            *j = [
                [
                    xd,
                    S::zero(),
                    S::one(),
                    S::zero(),
                    fx * (dd[0][0] * dm[0] + dd[0][1] * dm[1]),
                ],
                [
                    S::zero(),
                    yd,
                    S::zero(),
                    S::one(),
                    fy * (dd[1][0] * dm[0] + dd[1][1] * dm[1]),
                ],
            ];
        }
        [fx * xd + cx, fy * yd + cy]
    }
    #[inline]
    fn project_with_status(
        &self,
        point: [S; 3],
        jp: Option<&mut [[S; 3]; 2]>,
        ji: Option<&mut Self::IntrinsicJacobian>,
    ) -> ([S; 2], Result<(), ProjectionReject>) {
        let norm = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
        let status = if !finite(point) || !norm.is_finite() {
            Err(ProjectionReject::NonFinite)
        } else if norm == S::zero() {
            Err(ProjectionReject::BelowMinDepth)
        } else if point[2] < norm * self.cos_half_field {
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
        let [mx, my] = newton2(target, |m, j| self.distortion.apply(m, Some(j)))?;
        let xi = self.param[4];
        let one = S::one();
        let r2 = mx * mx + my * my;
        // Closed-form lift onto the unit sphere; the root is real only inside the one-to-one region.
        let disc = one + (one - xi * xi) * r2;
        if disc < S::zero() {
            return Err(UnprojectError::OutsideDomain);
        }
        let lambda = (xi + disc.sqrt()) / (one + r2);
        let ray = [lambda * mx, lambda * my, lambda - xi];
        if !finite(ray) {
            return Err(UnprojectError::NonFinite);
        }
        if ray[2] < self.cos_half_field {
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
