//! Shared pivoted LDLT, reading only the lower triangle.
//! Pivots are selected before delayed column updates. Callers decide how to
//! use negative and tiny diagonal entries when solving or whitening.

use crate::Scalar;

/// Apply the LDLT pivot sequence and unit-lower forward solve to column-major right-hand sides.
/// `factors` and `transpositions` must come from [`ldlt_in_place`]. The diagonal
/// stores D and is not read; callers own the diagonal solve and any back substitution.
///
/// ```
/// use kornia_staging_algebra::linalg::ldlt::{ldlt_in_place, ldlt_forward_in_place};
/// let mut factors = [[4.0_f64, 2.0], [0.0, 3.0]];
/// let pivots = ldlt_in_place(&mut factors);
/// let mut rhs = [[2.0, 4.0]];
/// ldlt_forward_in_place(&factors, &pivots, &mut rhs);
/// assert_eq!(rhs, [[2.0, 3.0]]);
/// ```
#[allow(clippy::needless_range_loop)] // Preserve row/column arithmetic order.
pub fn ldlt_forward_in_place<S: Scalar, const N: usize, const M: usize>(
    factors: &[[S; N]; N],
    transpositions: &[usize; N],
    rhs: &mut [[S; N]; M],
) {
    for (k, &pivot) in transpositions.iter().enumerate() {
        if k != pivot {
            for column in rhs.iter_mut() {
                column.swap(k, pivot);
            }
        }
    }
    for row in 1..N {
        for k in 0..row {
            let factor = factors[k][row];
            for column in rhs.iter_mut() {
                let above = column[k];
                column[row] -= factor * above;
            }
        }
    }
}

/// Pack unit-lower L and diagonal D, returning the symmetric pivot sequence.
///
/// # Arguments
/// * `mat` - A square column-major matrix: `mat[column][row]`. Only its lower
///   triangle is read or written. The upper triangle is left untouched.
///
/// At step `k`, swap rows/columns `k` and `result[k]` to form the permutation.
/// Pivots use the first maximum absolute diagonal, before delayed updates.
/// Zero pivots are left undivided; the caller owns tiny/negative-pivot policy.
/// This is not a general Bunch-Kaufman factorization for indefinite matrices.
///
/// ```
/// use kornia_staging_algebra::linalg::ldlt::ldlt_in_place;
/// let mut columns = [[4.0f64, 2.0], [99.0, 3.0]];
/// assert_eq!(ldlt_in_place(&mut columns), [0, 1]);
/// assert_eq!(columns, [[4.0, 0.5], [99.0, 2.0]]);
/// ```
// A pivot step addresses the permutation and both matrix dimensions.
#[allow(clippy::needless_range_loop)]
pub fn ldlt_in_place<S: Scalar, const SIZE: usize>(mat: &mut [[S; SIZE]; SIZE]) -> [usize; SIZE] {
    let mut transpositions: [usize; SIZE] = [0; SIZE];
    let mut temp: [S; SIZE] = [S::zero(); SIZE];

    for k in 0..SIZE {
        // the first index of the maximum, so ties take the earliest row.
        let mut biggest: usize = k;
        for i in (k + 1)..SIZE {
            if mat[i][i].abs() > mat[biggest][biggest].abs() {
                biggest = i;
            }
        }
        transpositions[k] = biggest;

        if k != biggest {
            for j in 0..k {
                mat[j].swap(k, biggest);
            }
            for i in (biggest + 1)..SIZE {
                let swap: S = mat[k][i];
                mat[k][i] = mat[biggest][i];
                mat[biggest][i] = swap;
            }
            let swap: S = mat[k][k];
            mat[k][k] = mat[biggest][biggest];
            mat[biggest][biggest] = swap;
            for i in (k + 1)..biggest {
                let swap: S = mat[k][i];
                mat[k][i] = mat[i][biggest];
                mat[i][biggest] = swap;
            }
        }

        if k > 0 {
            for i in 0..k {
                temp[i] = mat[i][i] * mat[i][k];
            }
            let mut correction: S = S::zero();
            for i in 0..k {
                correction += mat[i][k] * temp[i];
            }
            mat[k][k] -= correction;
            for row in (k + 1)..SIZE {
                let mut update: S = S::zero();
                for i in 0..k {
                    update += mat[i][row] * temp[i];
                }
                mat[k][row] -= update;
            }
        }

        let pivot: S = mat[k][k];
        let pivot_is_valid: bool = pivot.abs() > S::zero();

        if k == 0 && !pivot_is_valid {
            for (j, transposition) in transpositions.iter_mut().enumerate() {
                *transposition = j;
            }
            return transpositions;
        }

        if pivot_is_valid {
            for row in (k + 1)..SIZE {
                mat[k][row] /= pivot;
            }
        }
    }

    transpositions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_precisions_preserve_first_maximum_ties_and_upper_triangle() {
        let mut wide = [[4.0f64, 2.0], [99.0, 4.0]];
        let mut narrow = [[4.0f32, 2.0], [99.0, 4.0]];
        assert_eq!(ldlt_in_place(&mut wide), [0, 1]);
        assert_eq!(ldlt_in_place(&mut narrow), [0, 1]);
        assert_eq!(wide, [[4.0, 0.5], [99.0, 3.0]]);
        assert_eq!(narrow, [[4.0, 0.5], [99.0, 3.0]]);
    }

    #[test]
    fn pivoted_factors_reconstruct_the_permuted_matrix() {
        let original = nalgebra::Matrix3::<f64>::new(1.0, 0.5, 0.25, 0.5, 4.0, 1.0, 0.25, 1.0, 2.0);
        let mut factors = original;
        let pivots = ldlt_in_place(&mut factors.data.0);
        let mut permuted = original;
        for (k, pivot) in pivots.into_iter().enumerate() {
            permuted.swap_rows(k, pivot);
            permuted.swap_columns(k, pivot);
        }
        assert_eq!(pivots[0], 1);
        let mut lower = nalgebra::Matrix3::<f64>::identity();
        for column in 0..3 {
            for row in column + 1..3 {
                lower[(row, column)] = factors[(row, column)];
            }
        }
        let diagonal = nalgebra::Matrix3::from_diagonal(&factors.diagonal());
        approx::assert_abs_diff_eq!(
            lower * diagonal * lower.transpose(),
            permuted,
            epsilon = 1e-14
        );
    }

    #[test]
    fn zero_and_negative_pivots_remain_caller_policy() {
        let mut zero = [[0.0f64; 3]; 3];
        assert_eq!(ldlt_in_place(&mut zero), [0, 1, 2]);
        assert_eq!(zero, [[0.0; 3]; 3]);
        let mut indefinite = [[-4.0f32, 2.0], [0.0, 0.0]];
        assert_eq!(ldlt_in_place(&mut indefinite), [0, 1]);
        assert_eq!(indefinite, [[-4.0, -0.5], [0.0, 1.0]]);
    }
}
