//! Eigen's column-pivoted Householder QR and the complete orthogonal
//! decomposition built on top of it.
//!
//! `marginalizeHelperSqToSq` and `marginalizeHelperSqToSqrt` invert the
//! marginalized block with
//! `abs_H.bottomRightCorner(marg_size, marg_size).completeOrthogonalDecomposition().pseudoInverse()`
//! (`marg_helper.cpp:99-100`, `:184-185`), after basalt tried and rejected
//! `ldlt`, `fullPivLu`, `colPivHouseholderQr` and a Jacobi-SVD pseudo-inverse —
//! the last one with a "DO NOT USE!!!" (`:91-96`). The choice is load-bearing:
//! the block is inverted precisely when it may be rank deficient, and the
//! answer on a deficient block is decided entirely by the decomposition's rank
//! threshold. So this is a port of Eigen's algorithm, not a substitution.
//!
//! Ported from the vendored Eigen 3.4:
//!
//! * `ColPivHouseholderQR::computeInPlace` (`Eigen/src/QR/ColPivHouseholderQR.h:488-583`)
//!   — the LAPACK `xGEQP3` norm-downdate with its `sqrt(epsilon)` recompute
//!   threshold, the `bug 941` `m_nonzero_pivots` rule, and `m_maxpivot`;
//! * `ColPivHouseholderQR::rank` (`:261-268`) and the default threshold
//!   `epsilon * diagonalSize` (`:375-381`);
//! * `CompleteOrthogonalDecomposition::computeInPlace` (`Eigen/src/QR/CompleteOrthogonalDecomposition.h:455-500`)
//!   — the `Z` reflectors that zero `R12`, built from the right, with the
//!   column head swaps around each one;
//! * `_solve_impl` (`:544-569`) and `applyZAdjointOnTheLeftInPlace` (`:524-539`);
//! * `pseudoInverse()`, which is `solve(Identity)` (`:640-651`).
//!
//! **Two documented departures, both product-kernel association, neither
//! reaching a rank decision.**
//!
//! 1. Eigen applies `Qᵀ` through a `HouseholderSequence`, which switches to a
//!    *blocked* application once the sequence is at least 48 long and the right
//!    hand side has more than one column (`HouseholderSequence.h:332-356`).
//!    This port always takes the unblocked branch (`:357-370`), which is the
//!    same reflections in the same order with a different summation shape.
//!    A marginalized block of 48 or more columns is possible but not typical:
//!    basalt's shipped window marginalizes at most a few 6- and 15-dof blocks
//!    at a time.
//! 2. The triangular solves are plain back-substitutions where Eigen calls a
//!    blocked kernel. Same statement of the problem, different association —
//!    the residue decision D50 already measured and accepted.
//!
//! Everything that decides *rank* — the pivot search, the norm downdate, the
//! threshold comparisons — is Eigen's arithmetic in Eigen's order.

use nalgebra::{DMatrix, DVector};

use crate::lie::LieScalar;
use crate::linearize::eigen_qr::{
    BlockSpan, apply_householder_on_the_left_block, apply_householder_on_the_right_block,
    make_householder, make_householder_row,
};

/// `Eigen::ColPivHouseholderQR<MatX>`.
#[derive(Debug, Clone)]
pub(crate) struct ColPivHouseholderQr<S: LieScalar> {
    /// `m_qr`: the packed factor, `R` above the diagonal and the Householder
    /// essentials below it.
    qr: DMatrix<S>,
    /// `m_hCoeffs`.
    h_coeffs: Vec<S>,
    /// `m_colsPermutation.indices()`.
    cols_permutation: Vec<usize>,
    /// `m_nonzero_pivots` (`:517`, `:528`).
    nonzero_pivots: usize,
    /// `m_maxpivot` (`:518`, `:547`).
    max_pivot: S,
    rows: usize,
    cols: usize,
}

impl<S: LieScalar> ColPivHouseholderQr<S> {
    /// `computeInPlace()` (`ColPivHouseholderQR.h:488-583`).
    pub(crate) fn new(matrix: &DMatrix<S>) -> Self {
        let mut qr: DMatrix<S> = matrix.clone();
        let rows: usize = qr.nrows();
        let cols: usize = qr.ncols();
        let size: usize = rows.min(cols);

        let mut h_coeffs: Vec<S> = vec![S::zero(); size];
        let mut temp: Vec<S> = vec![S::zero(); cols];
        let mut cols_transpositions: Vec<usize> = vec![0; cols];

        // `:504-511`.
        let mut col_norms_updated: Vec<S> = vec![S::zero(); cols];
        let mut col_norms_direct: Vec<S> = vec![S::zero(); cols];
        for k in 0..cols {
            let mut acc: S = S::zero();
            for i in 0..rows {
                acc += qr[(i, k)] * qr[(i, k)];
            }
            col_norms_direct[k] = acc.sqrt();
            col_norms_updated[k] = col_norms_direct[k];
        }

        // `:513-515`.
        let mut max_norm: S = S::zero();
        for value in &col_norms_updated {
            if *value > max_norm {
                max_norm = *value;
            }
        }
        let scaled: S = max_norm * S::default_epsilon();
        let rows_scalar: S = S::from_literal(rows as f64);
        let threshold_helper: S = if rows == 0 {
            S::zero()
        } else {
            (scaled * scaled) / rows_scalar
        };
        let norm_downdate_threshold: S = S::default_epsilon().sqrt();

        let mut nonzero_pivots: usize = size;
        let mut max_pivot: S = S::zero();
        let mut essential: Vec<S> = vec![S::zero(); rows.saturating_sub(1)];

        for k in 0..size {
            // `:521-524`: the largest *updated* column norm in the tail.
            // `maxCoeff` keeps the first maximum, so the comparison is strict.
            let mut biggest_col_index: usize = k;
            for j in (k + 1)..cols {
                if col_norms_updated[j] > col_norms_updated[biggest_col_index] {
                    biggest_col_index = j;
                }
            }
            let biggest: S = col_norms_updated[biggest_col_index];
            let biggest_col_sq_norm: S = biggest * biggest;

            // `:528`, "bug 941": track the pivots that mean something without
            // stopping, so the original matrix is still reproduced.
            if nonzero_pivots == size
                && biggest_col_sq_norm < threshold_helper * S::from_literal((rows - k) as f64)
            {
                nonzero_pivots = k;
            }

            // `:531-537`.
            cols_transpositions[k] = biggest_col_index;
            if k != biggest_col_index {
                qr.swap_columns(k, biggest_col_index);
                col_norms_updated.swap(k, biggest_col_index);
                col_norms_direct.swap(k, biggest_col_index);
            }

            // `:540-544`.
            let len: usize = rows - k;
            let (tau, beta) = make_householder(&qr, k, k, len, &mut essential);
            h_coeffs[k] = tau;
            for (i, e) in essential.iter().enumerate().take(len - 1) {
                qr[(k + 1 + i, k)] = *e;
            }
            qr[(k, k)] = beta;

            // `:547`.
            if beta.abs() > max_pivot {
                max_pivot = beta.abs();
            }

            // `:550-551`.
            apply_householder_on_the_left_block(
                &mut qr,
                BlockSpan {
                    row_start: k,
                    rows: len,
                    col_start: k + 1,
                    cols: cols - k - 1,
                },
                &essential[..len - 1],
                tau,
                &mut temp[k + 1..],
            );

            // `:554-574`: the LAPACK stable norm downdate.
            for j in (k + 1)..cols {
                if col_norms_updated[j] != S::zero() {
                    let mut t: S = qr[(k, j)].abs() / col_norms_updated[j];
                    t = (S::one() + t) * (S::one() - t);
                    if t < S::zero() {
                        t = S::zero();
                    }
                    let ratio: S = col_norms_updated[j] / col_norms_direct[j];
                    let temp2: S = t * (ratio * ratio);
                    if temp2 <= norm_downdate_threshold {
                        // `:568-569`: recompute directly.
                        let mut acc: S = S::zero();
                        for i in (k + 1)..rows {
                            acc += qr[(i, j)] * qr[(i, j)];
                        }
                        col_norms_direct[j] = acc.sqrt();
                        col_norms_updated[j] = col_norms_direct[j];
                    } else {
                        // `:571`.
                        col_norms_updated[j] *= t.sqrt();
                    }
                }
            }
        }

        // `:577-579`: identity, then `applyTranspositionOnTheRight(k, t[k])`
        // for ascending `k`, which swaps two *indices*.
        let mut cols_permutation: Vec<usize> = (0..cols).collect();
        for (k, &j) in cols_transpositions.iter().enumerate().take(size) {
            if j != k {
                cols_permutation.swap(k, j);
            }
        }

        Self {
            qr,
            h_coeffs,
            cols_permutation,
            nonzero_pivots,
            max_pivot,
            rows,
            cols,
        }
    }

    /// `threshold()` (`:375-381`): the default, `epsilon * diagonalSize`.
    /// basalt never calls `setThreshold`.
    fn threshold(&self) -> S {
        S::default_epsilon() * S::from_literal(self.rows.min(self.cols) as f64)
    }

    /// `rank()` (`:261-268`).
    ///
    /// Recomputed from the *current* diagonal every call, as Eigen does — which
    /// matters because [`Cod`] overwrites that diagonal with `T`'s.
    fn rank(&self) -> usize {
        let premultiplied: S = self.max_pivot.abs() * self.threshold();
        let mut result: usize = 0;
        for i in 0..self.nonzero_pivots {
            if self.qr[(i, i)].abs() > premultiplied {
                result += 1;
            }
        }
        result
    }

    /// `c.applyOnTheLeft(householderQ().setLength(length).adjoint())`.
    ///
    /// The adjoint of `Q = H₀H₁…` applied on the left is `H_{l-1}…H₀`, which the
    /// unblocked branch walks with `actual_k` **ascending**
    /// (`HouseholderSequence.h:359-368`, `m_reverse` set by `adjoint()`), each
    /// reflection acting on rows `k..rows`.
    fn apply_q_adjoint_on_the_left(&self, dst: &mut DMatrix<S>, length: usize) {
        let cols: usize = dst.ncols();
        let mut work: Vec<S> = vec![S::zero(); cols];
        let mut essential: Vec<S> = vec![S::zero(); self.rows.saturating_sub(1)];
        for k in 0..length {
            let len: usize = self.rows - k;
            for (i, slot) in essential.iter_mut().enumerate().take(len.saturating_sub(1)) {
                *slot = self.qr[(k + 1 + i, k)];
            }
            apply_householder_on_the_left_block(
                dst,
                BlockSpan {
                    row_start: k,
                    rows: len,
                    col_start: 0,
                    cols,
                },
                &essential[..len.saturating_sub(1)],
                self.h_coeffs[k],
                &mut work,
            );
        }
    }
}

/// `Eigen::CompleteOrthogonalDecomposition<MatX>`.
#[derive(Debug, Clone)]
pub struct Cod<S: LieScalar> {
    cpqr: ColPivHouseholderQr<S>,
    /// `m_zCoeffs`.
    z_coeffs: Vec<S>,
}

impl<S: LieScalar> Cod<S> {
    /// `CompleteOrthogonalDecomposition(matrix)` then `computeInPlace()`
    /// (`CompleteOrthogonalDecomposition.h:455-500`).
    pub fn new(matrix: &DMatrix<S>) -> Self {
        let mut cpqr: ColPivHouseholderQr<S> = ColPivHouseholderQr::new(matrix);
        let rank: usize = cpqr.rank();
        let cols: usize = cpqr.cols;
        let rows: usize = cpqr.rows;
        let mut z_coeffs: Vec<S> = vec![S::zero(); rows.min(cols)];
        let mut temp: Vec<S> = vec![S::zero(); cols];

        if rank < cols && rank > 0 {
            // `:476-498`: build `Z` from the right, one reflector per rank step,
            // zeroing `R12` row by row from the bottom up.
            let tail: usize = cols - rank + 1;
            let mut essential: Vec<S> = vec![S::zero(); tail.saturating_sub(1)];
            for k in (0..rank).rev() {
                if k != rank - 1 {
                    // `:481`: swap the leading parts of columns `k` and `rank-1`
                    // so the reflector sees `[X(k, k), X(k, rank:n)]`.
                    for i in 0..=k {
                        let swapped: S = cpqr.qr[(i, k)];
                        cpqr.qr[(i, k)] = cpqr.qr[(i, rank - 1)];
                        cpqr.qr[(i, rank - 1)] = swapped;
                    }
                }
                // `:487-488`.
                let (tau, beta) = make_householder_row(&cpqr.qr, k, rank - 1, tail, &mut essential);
                z_coeffs[k] = tau;
                for (i, e) in essential.iter().enumerate().take(tail - 1) {
                    cpqr.qr[(k, rank + i)] = *e;
                }
                cpqr.qr[(k, rank - 1)] = beta;

                if k > 0 {
                    // `:491-492`: apply `Z(k)` to the first `k` rows.
                    apply_householder_on_the_right_block(
                        &mut cpqr.qr,
                        BlockSpan {
                            row_start: 0,
                            rows: k,
                            col_start: rank - 1,
                            cols: tail,
                        },
                        &essential[..tail - 1],
                        tau,
                        &mut temp,
                    );
                }

                if k != rank - 1 {
                    // `:496`: swap back.
                    for i in 0..=k {
                        let swapped: S = cpqr.qr[(i, k)];
                        cpqr.qr[(i, k)] = cpqr.qr[(i, rank - 1)];
                        cpqr.qr[(i, rank - 1)] = swapped;
                    }
                }
            }
        }

        Self { cpqr, z_coeffs }
    }

    /// `rank()` (`:253`).
    pub fn rank(&self) -> usize {
        self.cpqr.rank()
    }

    /// `applyZAdjointOnTheLeftInPlace(rhs)` (`:524-539`).
    fn apply_z_adjoint_on_the_left_in_place(&self, rhs: &mut DMatrix<S>) {
        let cols: usize = self.cpqr.cols;
        let rank: usize = self.rank();
        if rank == 0 {
            return;
        }
        let nrhs: usize = rhs.ncols();
        let mut work: Vec<S> = vec![S::zero(); nrhs.max(cols)];
        let tail: usize = cols - rank;
        let mut essential: Vec<S> = vec![S::zero(); tail];
        for k in 0..rank {
            if k != rank - 1 {
                rhs.swap_rows(k, rank - 1);
            }
            for (i, slot) in essential.iter_mut().enumerate().take(tail) {
                *slot = self.cpqr.qr[(k, rank + i)];
            }
            apply_householder_on_the_left_block(
                rhs,
                BlockSpan {
                    row_start: rank - 1,
                    rows: cols - rank + 1,
                    col_start: 0,
                    cols: nrhs,
                },
                &essential[..tail],
                self.z_coeffs[k],
                &mut work,
            );
            if k != rank - 1 {
                rhs.swap_rows(k, rank - 1);
            }
        }
    }

    /// `solve(rhs)` (`_solve_impl`, `:544-569`): the minimum-norm least-squares
    /// solution.
    pub fn solve(&self, rhs: &DMatrix<S>) -> DMatrix<S> {
        let cols: usize = self.cpqr.cols;
        let nrhs: usize = rhs.ncols();
        let mut dst: DMatrix<S> = DMatrix::zeros(cols, nrhs);
        let rank: usize = self.rank();
        if rank == 0 {
            return dst;
        }

        // `:553-554`: `c = Q* rhs`, with the sequence truncated to `rank`.
        let mut c: DMatrix<S> = rhs.clone();
        self.cpqr.apply_q_adjoint_on_the_left(&mut c, rank);

        // `:557`: `T z = c(1:rank, :)`, upper triangular back substitution.
        for j in 0..nrhs {
            for i in (0..rank).rev() {
                let mut acc: S = c[(i, j)];
                for k in (i + 1)..rank {
                    acc -= self.cpqr.qr[(i, k)] * dst[(k, j)];
                }
                dst[(i, j)] = acc / self.cpqr.qr[(i, i)];
            }
        }

        // `:560-565`: `y = Z* [z; 0]`.
        if rank < cols {
            self.apply_z_adjoint_on_the_left_in_place(&mut dst);
        }

        // `:568`: `x = P y`, i.e. `dst.row(indices[i]) = y.row(i)`.
        let mut permuted: DMatrix<S> = DMatrix::zeros(cols, nrhs);
        for i in 0..cols {
            let target: usize = self.cpqr.cols_permutation[i];
            for j in 0..nrhs {
                permuted[(target, j)] = dst[(i, j)];
            }
        }
        permuted
    }

    /// `pseudoInverse()` (`:640-651`): `solve(Identity(rows, rows))`.
    pub fn pseudo_inverse(&self) -> DMatrix<S> {
        self.solve(&DMatrix::identity(self.cpqr.rows, self.cpqr.rows))
    }

    /// `solve` for a single right-hand side, which is what
    /// `test_qr.cpp`'s `RankDefLeastSquares` asks for.
    pub fn solve_vec(&self, rhs: &DVector<S>) -> DVector<S> {
        let as_matrix: DMatrix<S> = DMatrix::from_iterator(rhs.nrows(), 1, rhs.iter().copied());
        let solved: DMatrix<S> = self.solve(&as_matrix);
        DVector::from_iterator(solved.nrows(), solved.column(0).iter().copied())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use proptest::prelude::*;

    fn matrix_from(values: &[f64], rows: usize, cols: usize) -> DMatrix<f64> {
        let mut m: DMatrix<f64> = DMatrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                m[(r, c)] = values[(r * cols + c) % values.len()];
            }
        }
        m
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// A full-rank square matrix's pseudo-inverse is its inverse.
        #[test]
        fn the_pseudo_inverse_of_a_full_rank_matrix_is_the_inverse(
            values in prop::collection::vec(-2.0f64..2.0, 36..37),
        ) {
            let size: usize = 6;
            let base: DMatrix<f64> = matrix_from(&values, size, size);
            let a: DMatrix<f64> =
                base.transpose() * &base + DMatrix::identity(size, size) * 0.5;
            let cod: Cod<f64> = Cod::new(&a);
            prop_assert_eq!(cod.rank(), size);
            let inv: DMatrix<f64> = cod.pseudo_inverse();
            let product: DMatrix<f64> = &a * &inv;
            for r in 0..size {
                for c in 0..size {
                    let expected: f64 = if r == c { 1.0 } else { 0.0 };
                    prop_assert!((product[(r, c)] - expected).abs() < 1e-9);
                }
            }
        }

        /// On a rank-deficient matrix the four Moore-Penrose conditions hold and
        /// the rank is the true one.
        #[test]
        fn the_pseudo_inverse_of_a_rank_deficient_matrix_is_moore_penrose(
            values in prop::collection::vec(-2.0f64..2.0, 36..37),
        ) {
            let size: usize = 6;
            let mut base: DMatrix<f64> = matrix_from(&values, size, size);
            // Two dependent columns: the true rank is at most 4.
            let col4: DVector<f64> = base.column(4).into_owned();
            base.column_mut(1).copy_from(&col4);
            let col3: DVector<f64> = base.column(3).into_owned();
            base.column_mut(0).copy_from(&col3);
            let a: DMatrix<f64> = base.transpose() * &base;

            let cod: Cod<f64> = Cod::new(&a);
            prop_assert!(cod.rank() <= 4);
            let inv: DMatrix<f64> = cod.pseudo_inverse();
            let apa: DMatrix<f64> = &a * &inv * &a;
            let scale: f64 = a.iter().fold(0.0f64, |acc, v| acc.max(v.abs())).max(1.0);
            for r in 0..size {
                for c in 0..size {
                    prop_assert!((apa[(r, c)] - a[(r, c)]).abs() < 1e-8 * scale);
                }
            }
            let pap: DMatrix<f64> = &inv * &a * &inv;
            let inv_scale: f64 = inv.iter().fold(0.0f64, |acc, v| acc.max(v.abs())).max(1.0);
            for r in 0..size {
                for c in 0..size {
                    prop_assert!((pap[(r, c)] - inv[(r, c)]).abs() < 1e-8 * inv_scale);
                }
            }
        }
    }

    /// The zero matrix has rank zero and `solve` returns zero, the `rank == 0`
    /// early exit of `:547-550`.
    #[test]
    fn a_zero_matrix_solves_to_zero() {
        let cod: Cod<f64> = Cod::new(&DMatrix::zeros(4, 4));
        assert_eq!(cod.rank(), 0);
        let solved: DVector<f64> = cod.solve_vec(&DVector::from_element(4, 1.0));
        assert_eq!(solved, DVector::zeros(4));
    }
}
