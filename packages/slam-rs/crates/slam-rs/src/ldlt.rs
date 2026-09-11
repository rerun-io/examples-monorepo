//! Shared pivoted LDLT, reading only the lower triangle.
//! Pivots are selected before delayed column updates. Callers decide how to
//! use negative and tiny diagonal entries when solving or whitening.

use crate::lie::LieScalar;
use nalgebra::SMatrix;

/// Pack unit-lower L and diagonal D, returning the symmetric pivot sequence.
// A pivot step addresses the permutation and both matrix dimensions.
#[allow(clippy::needless_range_loop)]
pub(crate) fn ldlt_in_place<S: LieScalar, const SIZE: usize>(
    mat: &mut SMatrix<S, SIZE, SIZE>,
) -> [usize; SIZE] {
    let mut transpositions: [usize; SIZE] = [0; SIZE];
    let mut temp: [S; SIZE] = [S::zero(); SIZE];

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
                let swap: S = mat[(k, j)];
                mat[(k, j)] = mat[(biggest, j)];
                mat[(biggest, j)] = swap;
            }
            for i in (biggest + 1)..SIZE {
                let swap: S = mat[(i, k)];
                mat[(i, k)] = mat[(i, biggest)];
                mat[(i, biggest)] = swap;
            }
            let swap: S = mat[(k, k)];
            mat[(k, k)] = mat[(biggest, biggest)];
            mat[(biggest, biggest)] = swap;
            for i in (k + 1)..biggest {
                let swap: S = mat[(i, k)];
                mat[(i, k)] = mat[(biggest, i)];
                mat[(biggest, i)] = swap;
            }
        }

        if k > 0 {
            for i in 0..k {
                temp[i] = mat[(i, i)] * mat[(k, i)];
            }
            let mut correction: S = S::zero();
            for i in 0..k {
                correction += mat[(k, i)] * temp[i];
            }
            mat[(k, k)] -= correction;
            for row in (k + 1)..SIZE {
                let mut update: S = S::zero();
                for i in 0..k {
                    update += mat[(row, i)] * temp[i];
                }
                mat[(row, k)] -= update;
            }
        }

        let pivot: S = mat[(k, k)];
        let pivot_is_valid: bool = pivot.abs() > S::zero();

        if k == 0 && !pivot_is_valid {
            for (j, transposition) in transpositions.iter_mut().enumerate() {
                *transposition = j;
            }
            return transpositions;
        }

        if pivot_is_valid {
            for row in (k + 1)..SIZE {
                mat[(row, k)] /= pivot;
            }
        }
    }

    transpositions
}
