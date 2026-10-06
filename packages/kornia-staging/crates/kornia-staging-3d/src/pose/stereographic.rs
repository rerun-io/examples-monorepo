//! Stereographic unit-bearing coordinates; projection ignores inverse distance.
use kornia_staging_algebra::Scalar;

/// The bearing cannot be represented in this stereographic chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StereographicError {
    /// A spatial coordinate or computed spatial norm is not finite.
    #[error("non-finite spatial direction")]
    NonFinite,
    /// The spatial direction has zero length.
    #[error("zero spatial direction")]
    ZeroDirection,
    /// The direction lies at the chart's south pole (or rounds onto it).
    #[error("south pole is outside the stereographic chart")]
    SouthPole,
}

#[inline]
fn project<S: Scalar>(
    p: [S; 4],
    jacobian: Option<&mut [[S; 2]; 4]>,
) -> Result<[S; 2], StereographicError> {
    let sqrt = nalgebra::Vector3::from([p[0], p[1], p[2]]).norm();
    if !sqrt.is_finite() {
        return Err(StereographicError::NonFinite);
    }
    if sqrt <= S::zero() {
        return Err(StereographicError::ZeroDirection);
    }
    let norm = p[2] + sqrt;
    if norm <= S::zero() {
        return Err(StereographicError::SouthPole);
    }
    let norm_inv = S::one() / norm;
    if let Some(j) = jacobian {
        let tmp = -(norm_inv * norm_inv) / sqrt;
        *j = [
            [norm_inv + p[0] * p[0] * tmp, p[0] * p[1] * tmp],
            [p[0] * p[1] * tmp, norm_inv + p[1] * p[1] * tmp],
            [p[0] * norm * tmp, p[1] * norm * tmp],
            [S::zero(); 2],
        ];
    }
    Ok([p[0] * norm_inv, p[1] * norm_inv])
}
/// Project a homogeneous bearing, ignoring its inverse distance.
/// # Errors
/// Rejects non-finite or zero directions and the south pole.
#[inline]
pub fn stereographic_project<S: Scalar>(point: [S; 4]) -> Result<[S; 2], StereographicError> {
    project(point, None)
}
/// Project a bearing and its column-major 2x4 Jacobian.
/// # Errors
/// Rejects non-finite or zero directions and the south pole.
#[inline]
pub fn stereographic_project_with_jacobian<S: Scalar>(
    point: [S; 4],
) -> Result<([S; 2], [[S; 2]; 4]), StereographicError> {
    let mut j = [[S::zero(); 2]; 4];
    let value = project(point, Some(&mut j))?;
    Ok((value, j))
}
#[inline]
fn unproject<S: Scalar>(p: [S; 2], jacobian: Option<&mut [[S; 4]; 2]>) -> [S; 4] {
    let x2 = p[0] * p[0];
    let y2 = p[1] * p[1];
    let norm_inv = S::from_literal(2.0) / (S::one() + (x2 + y2));
    if let Some(j) = jacobian {
        let norm_inv2 = norm_inv * norm_inv;
        let xy = p[0] * p[1];
        *j = [
            [
                norm_inv - x2 * norm_inv2,
                -xy * norm_inv2,
                -p[0] * norm_inv2,
                S::zero(),
            ],
            [
                -xy * norm_inv2,
                norm_inv - y2 * norm_inv2,
                -p[1] * norm_inv2,
                S::zero(),
            ],
        ];
    }
    [
        p[0] * norm_inv,
        p[1] * norm_inv,
        norm_inv - S::one(),
        S::zero(),
    ]
}
/// Unproject finite chart coordinates to a unit bearing, with zero inverse distance.
#[inline]
pub fn stereographic_unproject<S: Scalar>(point: [S; 2]) -> [S; 4] {
    unproject(point, None)
}
/// Unproject finite chart coordinates with the column-major 4x2 Jacobian.
#[inline]
pub fn stereographic_unproject_with_jacobian<S: Scalar>(point: [S; 2]) -> ([S; 4], [[S; 4]; 2]) {
    let mut j = [[S::zero(); 4]; 2];
    (unproject(point, Some(&mut j)), j)
}
