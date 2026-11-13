//! Row-major small-matrix arithmetic and checked numeric conversions.

use crate::error::TsdfError;

/// `a[0] * b[0] + a[1] * b[1] + a[2] * b[2]`, summed in that order: callers'
/// bytes (mesh normals and their hashes) depend on it.
pub(crate) fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean length, as `dot(vector, vector).sqrt()`.
pub(crate) fn length(vector: [f32; 3]) -> f32 {
    dot(vector, vector).sqrt()
}

/// Rotation row `row` of a row-major 4-by-4 rigid transform.
fn rotation_row(matrix: [f32; 16], row: usize) -> [f32; 3] {
    [matrix[4 * row], matrix[4 * row + 1], matrix[4 * row + 2]]
}

/// Rotation column `column` of a row-major 4-by-4 rigid transform: the
/// image of unit axis `column`.
pub(crate) fn rotation_column(matrix: [f32; 16], column: usize) -> [f32; 3] {
    [matrix[column], matrix[4 + column], matrix[8 + column]]
}

/// Apply the rigid row-major 4-by-4 `matrix` to `point`.
pub(crate) fn transform_point(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|row| dot(rotation_row(matrix, row), point) + matrix[4 * row + 3])
}

/// Map a camera-frame `point` to the world with the transpose of the rigid
/// `camera_from_world`'s rotation.
pub(crate) fn camera_to_world(camera_from_world: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    let shifted = std::array::from_fn(|row| point[row] - camera_from_world[4 * row + 3]);
    std::array::from_fn(|column| dot(rotation_column(camera_from_world, column), shifted))
}

pub(crate) fn multiply_3x3(matrix: [f32; 9], vector: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|row| {
        dot(
            [matrix[3 * row], matrix[3 * row + 1], matrix[3 * row + 2]],
            vector,
        )
    })
}

/// Inverse of a row-major 3-by-3 matrix; `None` when singular or non-finite.
pub(crate) fn invert_3x3(matrix: [f32; 9]) -> Option<[f32; 9]> {
    let determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    Some([
        (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inverse_determinant,
        (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inverse_determinant,
        (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inverse_determinant,
        (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inverse_determinant,
        (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inverse_determinant,
        (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inverse_determinant,
        (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inverse_determinant,
        (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inverse_determinant,
        (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inverse_determinant,
    ])
}

pub(crate) fn add<T: Copy + std::ops::Add<Output = T>>(left: [T; 3], right: [T; 3]) -> [T; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub(crate) fn add_scaled(base: [f32; 3], step: [f32; 3], amount: f32) -> [f32; 3] {
    [
        base[0] + step[0] * amount,
        base[1] + step[1] * amount,
        base[2] + step[2] * amount,
    ]
}

/// `value` truncated toward zero, when finite and inside the `i64` range.
pub(crate) fn checked_trunc_i64(value: f64) -> Option<i64> {
    // -(i64::MIN as f64) is 2^63, the first value past i64::MAX.
    (value.is_finite() && value >= i64::MIN as f64 && value < -(i64::MIN as f64))
        .then_some(value as i64)
}

/// `floor(value)`, when finite and inside the `i64` range.
pub(crate) fn checked_floor_i64(value: f64) -> Option<i64> {
    // Truncation plus a correction avoids the libm floor call.
    let truncated = checked_trunc_i64(value)?;
    Some(if truncated as f64 > value {
        truncated - 1
    } else {
        truncated
    })
}

pub(crate) fn validate_positive(name: &'static str, value: f32) -> Result<(), TsdfError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(TsdfError::InvalidParameter {
            name,
            reason: "must be finite and positive",
            value,
        })
    }
}

pub(crate) fn validate_non_negative(name: &'static str, value: f32) -> Result<(), TsdfError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(TsdfError::InvalidParameter {
            name,
            reason: "must be finite and non-negative",
            value,
        })
    }
}

/// Check that a slice of `got` values covers `dimensions` exactly.
pub(crate) fn validate_length(
    name: &'static str,
    got: usize,
    dimensions: &[usize],
) -> Result<(), TsdfError> {
    let expected = dimensions.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| TsdfError::DimensionOverflow {
                name,
                dimensions: dimensions.to_vec(),
            })
    })?;
    if got == expected {
        Ok(())
    } else {
        Err(TsdfError::ShapeMismatch {
            name,
            expected,
            got,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_floor_matches_floor_inside_the_range() {
        for value in [
            -2.5, -2.0, -1e-9, -0.0, 0.0, 0.5, 3.0, 3.75, -9.2e18, 9.2e18,
        ] {
            assert_eq!(checked_floor_i64(value), Some(f64::floor(value) as i64));
        }
        for value in [f64::NAN, f64::INFINITY, 1e19, -1e19] {
            assert_eq!(checked_floor_i64(value), None);
        }
    }
}
