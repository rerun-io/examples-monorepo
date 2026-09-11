//! Pivoted LDLT for a symmetric 3x3 patch Hessian.
//!
//! The solve zeros rows whose diagonal pivot has magnitude at most the smallest
//! positive normal value. Degenerate patches thus retain finite, zero-weight
//! directions instead of divisions by zero. For singular inputs the result is
//! a generalized inverse through congruence, `Pᵀ L⁻ᵀ D⁺ L⁻¹ P`, not the
//! Moore-Penrose inverse; it need not annihilate the null space.

use nalgebra::Matrix3;

/// Invert a symmetric patch Hessian, reading only its lower triangle.
/// Singular diagonal pivots contribute zero in the factor coordinates.
pub fn ldlt_inverse3(a: &Matrix3<f32>) -> Matrix3<f32> {
    let (mat, transpositions): (Matrix3<f32>, [usize; 3]) = ldlt_decompose3(a);
    let mut result: Matrix3<f32> = Matrix3::identity();
    ldlt_solve3(&mat, &transpositions, &mut result);
    result
}

/// Pack unit-lower L and diagonal D, returning the symmetric pivot sequence.
// A pivot step addresses the permutation and both matrix dimensions.
#[allow(clippy::needless_range_loop)]
fn ldlt_decompose3(a: &Matrix3<f32>) -> (Matrix3<f32>, [usize; 3]) {
    const SIZE: usize = 3;
    let mut mat: Matrix3<f32> = *a;
    let mut transpositions: [usize; SIZE] = [0; SIZE];
    let mut temp: [f32; SIZE] = [0.0; SIZE];

    for k in 0..SIZE {
        // the first index of the maximum, so ties take the earliest row.
        let mut biggest: usize = k;
        for i in (k + 1)..SIZE {
            if mat[(i, i)].abs() > mat[(biggest, biggest)].abs() {
                biggest = i;
            }
        }
        transpositions[k] = biggest;

        if k != biggest {
            for j in 0..k {
                let swap: f32 = mat[(k, j)];
                mat[(k, j)] = mat[(biggest, j)];
                mat[(biggest, j)] = swap;
            }
            for i in (biggest + 1)..SIZE {
                let swap: f32 = mat[(i, k)];
                mat[(i, k)] = mat[(i, biggest)];
                mat[(i, biggest)] = swap;
            }
            let swap: f32 = mat[(k, k)];
            mat[(k, k)] = mat[(biggest, biggest)];
            mat[(biggest, biggest)] = swap;
            for i in (k + 1)..biggest {
                let swap: f32 = mat[(i, k)];
                mat[(i, k)] = mat[(biggest, i)];
                mat[(biggest, i)] = swap;
            }
        }

        if k > 0 {
            for i in 0..k {
                temp[i] = mat[(i, i)] * mat[(k, i)];
            }
            let mut correction: f32 = 0.0;
            for i in 0..k {
                correction += mat[(k, i)] * temp[i];
            }
            mat[(k, k)] -= correction;
            for row in (k + 1)..SIZE {
                let mut update: f32 = 0.0;
                for i in 0..k {
                    update += mat[(row, i)] * temp[i];
                }
                mat[(row, k)] -= update;
            }
        }

        let pivot: f32 = mat[(k, k)];
        let pivot_is_valid: bool = pivot.abs() > 0.0;

        if k == 0 && !pivot_is_valid {
            for (j, transposition) in transpositions.iter_mut().enumerate() {
                *transposition = j;
            }
            return (mat, transpositions);
        }

        if pivot_is_valid {
            for row in (k + 1)..SIZE {
                mat[(row, k)] /= pivot;
            }
        }
    }

    (mat, transpositions)
}

/// Apply the permutation, forward solve, guarded diagonal solve and back solve.
fn ldlt_solve3(mat: &Matrix3<f32>, transpositions: &[usize; 3], rhs_and_result: &mut Matrix3<f32>) {
    const SIZE: usize = 3;

    for (k, target) in transpositions.iter().enumerate() {
        if *target != k {
            rhs_and_result.swap_rows(k, *target);
        }
    }

    for row in 1..SIZE {
        for column in 0..row {
            let factor: f32 = mat[(row, column)];
            for j in 0..SIZE {
                let value: f32 = rhs_and_result[(column, j)];
                rhs_and_result[(row, j)] -= factor * value;
            }
        }
    }

    // is `numeric_limits<RealScalar>::min()`, the smallest positive normal.
    let tolerance: f32 = f32::MIN_POSITIVE;
    for i in 0..SIZE {
        let d: f32 = mat[(i, i)];
        if d.abs() > tolerance {
            for j in 0..SIZE {
                rhs_and_result[(i, j)] /= d;
            }
        } else {
            for j in 0..SIZE {
                rhs_and_result[(i, j)] = 0.0;
            }
        }
    }

    for row in (0..SIZE).rev() {
        for column in (row + 1)..SIZE {
            // `L^T(row, column) == L(column, row)`.
            let factor: f32 = mat[(column, row)];
            for j in 0..SIZE {
                let value: f32 = rhs_and_result[(column, j)];
                rhs_and_result[(row, j)] -= factor * value;
            }
        }
    }

    for k in (0..SIZE).rev() {
        let target: usize = transpositions[k];
        if target != k {
            rhs_and_result.swap_rows(k, target);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use proptest::prelude::*;

    fn assert_is_inverse(a: &Matrix3<f32>, epsilon: f32) {
        let inverse: Matrix3<f32> = ldlt_inverse3(a);
        let product: Matrix3<f32> = a * inverse;
        for row in 0..3 {
            for column in 0..3 {
                let expected: f32 = if row == column { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(product[(row, column)], expected, epsilon = epsilon);
            }
        }
    }

    #[test]
    fn the_identity_inverts_to_itself() {
        assert_eq!(ldlt_inverse3(&Matrix3::identity()), Matrix3::identity());
    }

    #[test]
    fn a_diagonal_matrix_inverts_elementwise() {
        let a: Matrix3<f32> = Matrix3::from_diagonal(&Vector3::new(4.0, 0.5, 100.0));
        let inverse: Matrix3<f32> = ldlt_inverse3(&a);
        assert_abs_diff_eq!(inverse[(0, 0)], 0.25, epsilon = 1e-7);
        assert_abs_diff_eq!(inverse[(1, 1)], 2.0, epsilon = 1e-7);
        assert_abs_diff_eq!(inverse[(2, 2)], 0.01, epsilon = 1e-7);
    }

    /// A pivot has to be chosen, and the transposition undone, or the inverse of
    /// a matrix whose first diagonal entry is not the largest comes out wrong.
    #[test]
    fn pivoting_survives_a_small_leading_diagonal_entry() {
        let a: Matrix3<f32> = Matrix3::new(1e-3, 0.4, 0.2, 0.4, 9.0, 1.0, 0.2, 1.0, 4.0);
        assert_is_inverse(&a, 1e-4);
    }

    /// The reason this module exists: a singular `H` gives zeros, not infinities,
    #[test]
    fn a_rank_deficient_matrix_gives_zeros_not_infinities() {
        let g: Vector3<f32> = Vector3::new(2.0, 1.0, 0.0);
        let a: Matrix3<f32> = g * g.transpose();
        let inverse: Matrix3<f32> = ldlt_inverse3(&a);
        assert!(inverse.iter().all(|value| value.is_finite()));
        assert_eq!(inverse[(2, 2)], 0.0);
    }

    #[test]
    fn an_all_zero_matrix_gives_all_zeros() {
        let inverse: Matrix3<f32> = ldlt_inverse3(&Matrix3::zeros());
        assert_eq!(inverse, Matrix3::zeros());
    }

    proptest! {
        #[test]
        fn a_rank_deficient_gram_has_a_generalized_inverse(
            a in 0.25f32..4.0, b in 0.25f32..4.0, shift in 0usize..3,
        ) {
            // Duplicate scaled columns give a non-coordinate null direction.
            // Powers of two keep that dependency exact in floating point.
            let mut g = Matrix3::new(a, 0.0, 2.0 * a, 0.0, b, 0.0, 0.0, 0.0, 0.0);
            g.swap_columns(0, shift);
            let a = g.transpose() * g;
            let inverse = ldlt_inverse3(&a);
            prop_assert!(inverse.iter().all(|v| v.is_finite()));
            prop_assert!((inverse - inverse.transpose()).norm() < 1e-5 * (1.0 + inverse.norm()));
            prop_assert!((a * inverse * a - a).norm() < 1e-5 * (1.0 + a.norm()));
        }
    }

    proptest! {
        /// For any well-conditioned `J^T J` built from three random rows, the
        /// factorisation inverts it.
        #[test]
        fn a_full_rank_gram_matrix_is_inverted(
            values in prop::array::uniform9(-3.0f32..3.0),
        ) {
            let rows: Matrix3<f32> = Matrix3::from_row_slice(&values);
            let gram: Matrix3<f32> = rows.transpose() * rows + Matrix3::identity();
            let inverse: Matrix3<f32> = ldlt_inverse3(&gram);
            let product: Matrix3<f32> = gram * inverse;
            for row in 0..3 {
                for column in 0..3 {
                    let expected: f32 = if row == column { 1.0 } else { 0.0 };
                    prop_assert!((product[(row, column)] - expected).abs() < 1e-4);
                }
            }
        }
    }
}
