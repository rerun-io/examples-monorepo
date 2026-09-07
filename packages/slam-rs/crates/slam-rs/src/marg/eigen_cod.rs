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
//! Ported from the vendored Eigen 5.0.1 (`Eigen/Version:12`):
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
    BlockSpan, ColumnRedux, apply_householder_on_the_left_block,
    apply_householder_on_the_right_block, contiguous_squared_norm, make_householder,
    make_householder_row,
};
use crate::marg::MargError;

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

        // `:504-511`: `m_colNormsDirect(k) = m_qr.col(k).norm()`. `m_qr` is
        // column-major, so the whole column is contiguous and Eigen's
        // vectorised reduction runs; a sequential fold here moves the pivot
        // search and `rank()` (the review's `10x2` problem: Eigen 2, a
        // sequential fold 1, in both precisions).
        let mut col_norms_updated: Vec<S> = vec![S::zero(); cols];
        let mut col_norms_direct: Vec<S> = vec![S::zero(); cols];
        for k in 0..cols {
            col_norms_direct[k] = contiguous_squared_norm(&qr, k, 0, rows).sqrt();
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
        let threshold_helper: S = (scaled * scaled) / S::from_literal(rows as f64);
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
            let (tau, beta) =
                make_householder(&qr, k, k, len, ColumnRedux::Contiguous, &mut essential);
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
                        // `:568-569`: recompute directly, from
                        // `m_qr.col(j).tail(rows - k - 1).norm()` — contiguous
                        // again. The tail is empty when `k + 1 == rows`, which a
                        // wider-than-tall matrix reaches on its last step, and
                        // an empty reduction is zero.
                        col_norms_direct[j] =
                            contiguous_squared_norm(&qr, j, k + 1, rows - k - 1).sqrt();
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
        }
    }

    /// `threshold()` (`:375-381`): the default, `epsilon * diagonalSize`.
    /// basalt never calls `setThreshold`.
    fn threshold(&self) -> S {
        S::default_epsilon() * S::from_literal(self.qr.nrows().min(self.qr.ncols()) as f64)
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
        let rows: usize = self.qr.nrows();
        let cols: usize = dst.ncols();
        let mut work: Vec<S> = vec![S::zero(); cols];
        let mut essential: Vec<S> = vec![S::zero(); rows.saturating_sub(1)];
        for k in 0..length {
            let len: usize = rows - k;
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
        let cols: usize = cpqr.qr.ncols();
        let rows: usize = cpqr.qr.nrows();
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
        let cols: usize = self.cpqr.qr.ncols();
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
    ///
    /// C++ asserts the right-hand side's height —
    /// `derived().rows() == b.rows()` in `SolverBase::solve`
    /// (`Eigen/src/Core/util/../SolverBase.h`), where
    /// `CompleteOrthogonalDecomposition::rows()` is the *factorized matrix's*
    /// row count. The port returns [`MargError::RhsLengthMismatch`] instead: a
    /// short right-hand side used to index past the end of it through the
    /// unchecked [`BlockSpan`] built from `rows` rather than from the argument
    /// (decision D32).
    pub fn solve(&self, rhs: &DMatrix<S>) -> Result<DMatrix<S>, MargError> {
        let rows: usize = self.cpqr.qr.nrows();
        if rhs.nrows() != rows {
            return Err(MargError::RhsLengthMismatch {
                rows,
                rhs: rhs.nrows(),
            });
        }
        Ok(self.solve_with_validated_rhs(rhs))
    }

    /// The body of `_solve_impl` (`:544-569`).
    ///
    /// **Contract: `rhs.nrows() == self.cpqr.qr.nrows()`,** the factorized
    /// matrix's row count. [`Self::solve`] is the checked boundary;
    /// [`Self::pseudo_inverse`] passes an identity of exactly that many rows,
    /// one line away from where it is built.
    fn solve_with_validated_rhs(&self, rhs: &DMatrix<S>) -> DMatrix<S> {
        let cols: usize = self.cpqr.qr.ncols();
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
    ///
    /// Infallible, because the identity it solves against is `rows` tall by
    /// construction — the one dimension [`Self::solve`] would check.
    pub fn pseudo_inverse(&self) -> DMatrix<S> {
        let rows: usize = self.cpqr.qr.nrows();
        self.solve_with_validated_rhs(&DMatrix::identity(rows, rows))
    }

    /// `solve` for a single right-hand side, which is what
    /// `test_qr.cpp`'s `RankDefLeastSquares` asks for.
    pub fn solve_vec(&self, rhs: &DVector<S>) -> Result<DVector<S>, MargError> {
        let as_matrix: DMatrix<S> = DMatrix::from_iterator(rhs.nrows(), 1, rhs.iter().copied());
        let solved: DMatrix<S> = self.solve(&as_matrix)?;
        Ok(DVector::from_iterator(
            solved.nrows(),
            solved.column(0).iter().copied(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    // The probe columns below are the fork's `%.17g` printout, carried over
    // verbatim even where the scalar does not need every figure. Keeping the
    // printout is what makes them evidence.
    #![allow(clippy::excessive_precision)]

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

            // The other two conditions. `APA = A` and `PAP = P` alone hold for
            // any generalized inverse; what singles the Moore-Penrose one out
            // is that both products are *symmetric*, i.e. `AP` and `PA` are the
            // orthogonal projectors onto the column and row spaces.
            let ap: DMatrix<f64> = &a * &inv;
            let pa: DMatrix<f64> = &inv * &a;
            let ap_scale: f64 = ap.iter().fold(0.0f64, |acc, v| acc.max(v.abs())).max(1.0);
            let pa_scale: f64 = pa.iter().fold(0.0f64, |acc, v| acc.max(v.abs())).max(1.0);
            for r in 0..size {
                for c in 0..size {
                    prop_assert!(
                        (ap[(r, c)] - ap[(c, r)]).abs() < 1e-8 * ap_scale,
                        "(AP)ᵀ != AP at ({}, {})", r, c
                    );
                    prop_assert!(
                        (pa[(r, c)] - pa[(c, r)]).abs() < 1e-8 * pa_scale,
                        "(PA)ᵀ != PA at ({}, {})", r, c
                    );
                }
            }
        }
    }

    /// A matrix with no rows has rank zero, where the column norms used to
    /// reduce an empty column by reading its first coefficient.
    #[test]
    fn a_matrix_with_no_rows_has_rank_zero() {
        assert_eq!(Cod::new(&DMatrix::<f64>::zeros(0, 3)).rank(), 0);
        assert_eq!(Cod::new(&DMatrix::<f32>::zeros(0, 3)).rank(), 0);
    }

    /// The zero matrix has rank zero and `solve` returns zero, the `rank == 0`
    /// early exit of `:547-550`.
    #[test]
    fn a_zero_matrix_solves_to_zero() {
        let cod: Cod<f64> = Cod::new(&DMatrix::zeros(4, 4));
        assert_eq!(cod.rank(), 0);
        let solved: DVector<f64> = cod.solve_vec(&DVector::from_element(4, 1.0)).unwrap();
        assert_eq!(solved, DVector::zeros(4));
    }

    /// A right-hand side of the wrong height is a typed error, not an
    /// out-of-range read through the unchecked span of
    /// [`ColPivHouseholderQr::apply_q_adjoint_on_the_left`].
    ///
    /// `Cod::new` on a `2x2` then `solve` on a `1x1` used to index row 1 of a
    /// one-row matrix.
    #[test]
    fn a_right_hand_side_of_the_wrong_height_is_refused() {
        let mut a: DMatrix<f64> = DMatrix::zeros(2, 2);
        a[(0, 0)] = 1.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 1.0;
        let cod: Cod<f64> = Cod::new(&a);
        assert_eq!(cod.rank(), 2);
        assert_eq!(
            cod.solve(&DMatrix::zeros(1, 1)),
            Err(MargError::RhsLengthMismatch { rows: 2, rhs: 1 })
        );
        assert_eq!(
            cod.solve_vec(&DVector::zeros(1)),
            Err(MargError::RhsLengthMismatch { rows: 2, rhs: 1 })
        );
        assert_eq!(
            cod.solve(&DMatrix::zeros(3, 1)),
            Err(MargError::RhsLengthMismatch { rows: 2, rhs: 3 })
        );
        // The right height still works, whatever the width.
        assert!(cod.solve(&DMatrix::zeros(2, 5)).is_ok());
    }

    /// The `10x2` problem the S7 review reproduced: Eigen's rank is **2** and
    /// the sequential column-norm fold the port shipped made it **1**, in both
    /// precisions.
    ///
    /// `a(0, 0) = 1` and rows 1..9 of column 1 hold a vector scaled to just
    /// past `2 * epsilon`, so the second pivot lands within an ulp of
    /// `rank()`'s `maxpivot * epsilon * diagonalSize` threshold and the
    /// reduction order decides it. The two columns and the expected ranks come
    /// from the fork (`tools/marg_norm_probe.cpp`, the "COD problem" section).
    #[test]
    fn the_reviews_rank_boundary_problem_has_eigens_rank() {
        fn check<S: LieScalar>(column: &[f64; 9]) {
            let mut a: DMatrix<S> = DMatrix::zeros(10, 2);
            a[(0, 0)] = S::one();
            for (i, v) in column.iter().enumerate() {
                a[(i + 1, 1)] = S::from_literal(*v);
            }
            assert_eq!(Cod::new(&a).rank(), 2);
        }
        // The `f32` column, scaled in `f32`.
        check::<f32>(&[
            3.8839900184939324e-08,
            8.5545108774454093e-09,
            -1.0945134221174158e-07,
            -7.9057826951611787e-08,
            -1.1235827912514651e-07,
            -1.0663756455642215e-07,
            -8.3013901530648582e-08,
            -6.6877191784442402e-08,
            -4.0892032870942785e-08,
        ]);
        // The `f64` column.
        check::<f64>(&[
            2.1142743579943638e-16,
            -8.1965718710367306e-17,
            -2.4687331390702622e-16,
            -9.5228852078720228e-17,
            1.365723716499534e-16,
            1.1965862760119914e-17,
            -3.6399123055194747e-17,
            2.049020062317018e-16,
            -1.1694185041507896e-16,
        ]);
    }
}
