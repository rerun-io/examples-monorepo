//! Eigen's pivoted LDLT at dynamic size, and the three things
//! `marginalizeHelperSqToSqrt` asks of it.
//!
//! `marg_helper.cpp:202-231` takes the square root of the reduced Hessian
//! through `Eigen::LDLT<Eigen::Ref<MatX>>`, and reads back `vectorD()`,
//! `transpositionsP()`, `matrixU()` and `matrixL().solveInPlace()`. Decision
//! D41 settled that Eigen's LDLT is **not** interchangeable with a textbook
//! pivoted LDLT — the IMU stage found a rank-deficient covariance where the two
//! differ by 26 orders of magnitude — so it is ported statement by statement
//! from the vendored Eigen 3.4
//! (`thirdparty/basalt-headers/thirdparty/eigen/Eigen/src/Cholesky/LDLT.h`),
//! exactly as [`crate::imu`] did at fixed size 9 and `frontend::ldlt` at 3.
//!
//! Two properties decide what a rank-deficient prior becomes, and neither
//! survives a right-looking rewrite:
//!
//! * **The pivot is chosen on the un-updated diagonal** (`LDLT.h:335-339`), so
//!   at step `k` the trailing diagonal still holds the original entries. The
//!   Eigen source says so itself: LDLT "is not rank-revealing" (`:342-344`).
//! * **A dependent direction leaves a tiny pivot of either sign**, and
//!   `marginalizeHelperSqToSqrt` clamps it with `vectorD().array().max(0)`
//!   (`marg_helper.cpp:204`) and then drops the matching `b` entry when the
//!   root is below `sqrt(numeric_limits::min())` (`:229`). Which side of zero
//!   the residue lands on is a property of the precision, not of the algorithm.
//!
//! **What is exact and what is not.** The factorization, the transpositions,
//! `vectorD` and the `matrixU() * P` product are elementary operations in
//! Eigen's own order. The unit-lower solve reproduces Eigen's panel structure
//! (`TriangularSolverVector.h:96-113`, `EIGEN_TUNE_TRIANGULAR_PANEL_WIDTH = 8`)
//! but its trailing update is a plain loop where Eigen calls a blocked `gemv`,
//! so systems wider than 8 carry the same product-kernel association residue
//! decision D50 already accepted for the landmark blocks.

use nalgebra::{DMatrix, DVector};

use crate::lie::LieScalar;

/// Eigen's `EIGEN_TUNE_TRIANGULAR_PANEL_WIDTH` (`Eigen/src/Core/util/Macros.h`).
const TRIANGULAR_PANEL_WIDTH: usize = 8;

/// `Eigen::LDLT<MatX, Lower>` over a dense symmetric matrix.
///
/// Holds the packed factor exactly as Eigen does: `D` on the diagonal, the
/// strict lower triangle of the unit-lower `L` below it, and whatever the input
/// had above it. `P A Pᵀ = L D Lᵀ`.
#[derive(Debug, Clone)]
pub(crate) struct EigenLdlt<S: LieScalar> {
    mat: DMatrix<S>,
    transpositions: Vec<usize>,
}

impl<S: LieScalar> EigenLdlt<S> {
    /// `internal::ldlt_inplace<Lower>::unblocked` (`LDLT.h:280-382`).
    ///
    /// Only the lower triangle of `a` is read, as `LDLT<MatrixType, Lower>`
    /// reads. `a` is consumed because Eigen's is an in-place factorization and
    /// `marg_helper.cpp:202` hands it a `Ref` into `marg_H`, which is dead
    /// afterwards.
    pub(crate) fn new(mut mat: DMatrix<S>) -> Self {
        let size: usize = mat.nrows().min(mat.ncols());
        let mut transpositions: Vec<usize> = vec![0; size];
        // Eigen passes one `temp` workspace through the whole sweep
        // (`m_temporary.resize(size)` at `:496`, then `temp.head(k)` at
        // `:336-338`); step `k` reads and writes its first `k` entries, so
        // hoisting it here changes no read and no write.
        let mut temp: Vec<S> = vec![S::zero(); size];

        for k in 0..size {
            // "Find largest diagonal element" (`:305-307`). `maxCoeff` reports
            // the *first* index of the maximum, so the comparison is strict.
            let mut pivot: usize = k;
            for i in (k + 1)..size {
                if mat[(i, i)].abs() > mat[(pivot, pivot)].abs() {
                    pivot = i;
                }
            }
            transpositions[k] = pivot;

            if pivot != k {
                // `:313-321`: a symmetric swap written so that only the lower
                // triangle stays valid, which is all the rest reads.
                for column in 0..k {
                    let swapped: S = mat[(k, column)];
                    mat[(k, column)] = mat[(pivot, column)];
                    mat[(pivot, column)] = swapped;
                }
                for row in (pivot + 1)..size {
                    let swapped: S = mat[(row, k)];
                    mat[(row, k)] = mat[(row, pivot)];
                    mat[(row, pivot)] = swapped;
                }
                let swapped: S = mat[(k, k)];
                mat[(k, k)] = mat[(pivot, pivot)];
                mat[(pivot, pivot)] = swapped;
                for i in (k + 1)..pivot {
                    let swapped: S = mat[(i, k)];
                    mat[(i, k)] = mat[(pivot, i)];
                    mat[(pivot, i)] = swapped;
                }
            }

            // `:330-339`: the delayed update. Column `k` is brought up to date
            // from the columns already factorized; the trailing diagonal is not
            // touched, which is what makes the pivot choice above the
            // *un-updated* one.
            let rs: usize = size - k - 1;
            if k > 0 {
                let temp: &mut [S] = &mut temp[..k];
                for (j, entry) in temp.iter_mut().enumerate() {
                    *entry = mat[(j, j)] * mat[(k, j)]; // `:336`
                }
                let mut diagonal: S = S::zero();
                for (j, entry) in temp.iter().enumerate() {
                    diagonal += mat[(k, j)] * *entry;
                }
                mat[(k, k)] -= diagonal; // `:337`
                if rs > 0 {
                    for i in (k + 1)..size {
                        let mut sum: S = S::zero();
                        for (j, entry) in temp.iter().enumerate() {
                            sum += mat[(i, j)] * *entry;
                        }
                        mat[(i, k)] -= sum; // `:338`
                    }
                }
            }

            // `:345-346`. Eigen's cutoff is exactly zero, not an epsilon: the
            // guard only prevents an infinity or a NaN (`:341-344`).
            let real_akk: S = mat[(k, k)];
            let pivot_is_valid: bool = real_akk.abs() > S::zero();

            if k == 0 && !pivot_is_valid {
                // `:348-357`: the whole diagonal is zero, so the identity
                // transpositions are all that is left to fill in.
                for (j, entry) in transpositions.iter_mut().enumerate() {
                    *entry = j;
                }
                return Self {
                    mat,
                    transpositions,
                };
            }

            // `:359-360`. Eigen divides; it does not multiply by a reciprocal.
            if rs > 0 && pivot_is_valid {
                for i in (k + 1)..size {
                    mat[(i, k)] /= real_akk;
                }
            }
        }

        Self {
            mat,
            transpositions,
        }
    }

    /// `vectorD()` (`LDLT.h:141`): the diagonal of the packed factor.
    pub(crate) fn vector_d(&self) -> DVector<S> {
        let size: usize = self.transpositions.len();
        DVector::from_iterator(size, (0..size).map(|i| self.mat[(i, i)]))
    }

    /// `transpositionsP() * m` on the left of a dense matrix
    /// (`ProductEvaluators.h:1194-1200`, `Side == OnTheLeft`, not transposed):
    /// swap rows `k` and `t[k]` for **ascending** `k`.
    pub(crate) fn apply_transpositions_left(&self, m: &mut DMatrix<S>) {
        for (k, &j) in self.transpositions.iter().enumerate() {
            if j != k {
                m.swap_rows(k, j);
            }
        }
    }

    /// The same on a vector, `transpositionsP() * marg_b` (`marg_helper.cpp:223`).
    pub(crate) fn apply_transpositions_left_vec(&self, v: &mut DVector<S>) {
        for (k, &j) in self.transpositions.iter().enumerate() {
            if j != k {
                v.swap_rows(k, j);
            }
        }
    }

    /// `matrixU() * m` (`marg_helper.cpp:214`), where `matrixU()` is
    /// `m_matrix.adjoint().triangularView<UnitUpper>()` (`LDLT.h:134`): unit
    /// diagonal, and `U(i, j) = L(j, i)` above it.
    pub(crate) fn matrix_u_times(&self, m: &DMatrix<S>) -> DMatrix<S> {
        let size: usize = self.transpositions.len();
        let cols: usize = m.ncols();
        let mut out: DMatrix<S> = DMatrix::zeros(size, cols);
        for i in 0..size {
            for j in 0..cols {
                // `U(i, i) = 1`, then the strict upper part.
                let mut acc: S = m[(i, j)];
                for k in (i + 1)..size {
                    acc += self.mat[(k, i)] * m[(k, j)];
                }
                out[(i, j)] = acc;
            }
        }
        out
    }

    /// `matrixL().solveInPlace(v)` (`marg_helper.cpp:224`): the unit-lower
    /// forward substitution of `TriangularSolverVector.h:96-113`, `ColMajor`.
    ///
    /// The panel structure is Eigen's — `PanelWidth = 8`, column-oriented
    /// `axpy` inside a panel, then one trailing update per panel. Eigen's
    /// trailing update is a blocked `gemv`; this is a plain column-major loop,
    /// so a system of eight or fewer kept rows is exact and a wider one carries
    /// the product-kernel residue of D50.
    ///
    /// **Contract: `v.nrows() == self.transpositions.len()`**, i.e. `v` is as
    /// long as the factorized matrix is wide. `marg_helper.cpp:224` solves
    /// against the `marg_b` that came out of the same Schur complement as the
    /// `marg_H` this was factorized from, so the two agree by construction; a
    /// shorter `v` would be a caller bug, and clamping it would return a
    /// *partially* solved vector instead.
    pub(crate) fn solve_unit_lower_in_place(&self, v: &mut DVector<S>) {
        debug_assert_eq!(v.nrows(), self.transpositions.len());
        let size: usize = self.transpositions.len();
        let mut pi: usize = 0;
        while pi < size {
            let panel: usize = TRIANGULAR_PANEL_WIDTH.min(size - pi);
            let end_block: usize = pi + panel;
            for k in 0..panel {
                let i: usize = pi + k;
                if v[i] != S::zero() {
                    // `Mode & UnitDiag`, so no division by the diagonal.
                    let r: usize = panel - k - 1;
                    let scale: S = v[i];
                    for s in (i + 1)..(i + 1 + r) {
                        v[s] -= scale * self.mat[(s, i)];
                    }
                }
            }
            // The trailing update: `rhs.tail -= L(endBlock.., panel) * rhs(panel)`.
            for j in pi..end_block {
                let scale: S = v[j];
                if scale != S::zero() {
                    for i in end_block..size {
                        v[i] -= scale * self.mat[(i, j)];
                    }
                }
            }
            pi = end_block;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nalgebra::DMatrix;
    use proptest::prelude::*;

    /// Rebuild `P A Pᵀ = L D Lᵀ` from the packed factor and compare with the
    /// input, which is the identity the whole routine rests on.
    fn reconstruct(ldlt: &EigenLdlt<f64>, size: usize) -> DMatrix<f64> {
        let mut l: DMatrix<f64> = DMatrix::identity(size, size);
        for i in 0..size {
            for j in 0..i {
                l[(i, j)] = ldlt.mat[(i, j)];
            }
        }
        let d: DVector<f64> = ldlt.vector_d();
        let mut ldlt_product: DMatrix<f64> = DMatrix::zeros(size, size);
        for i in 0..size {
            for j in 0..size {
                let mut acc: f64 = 0.0;
                for k in 0..size {
                    acc += l[(i, k)] * d[k] * l[(j, k)];
                }
                ldlt_product[(i, j)] = acc;
            }
        }
        ldlt_product
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// On a positive-definite `A`, `P A Pᵀ = L D Lᵀ`.
        #[test]
        fn the_factorization_reproduces_the_permuted_matrix(
            values in prop::collection::vec(-2.0f64..2.0, 25..26),
        ) {
            let size: usize = 5;
            let mut j: DMatrix<f64> = DMatrix::zeros(size, size);
            for r in 0..size {
                for c in 0..size {
                    j[(r, c)] = values[r * size + c];
                }
            }
            let a: DMatrix<f64> = j.transpose() * &j + DMatrix::identity(size, size);

            let ldlt: EigenLdlt<f64> = EigenLdlt::new(a.clone());
            let mut permuted: DMatrix<f64> = a.clone();
            // `P A Pᵀ`: rows then columns, both ascending.
            for (k, &p) in ldlt.transpositions.iter().enumerate() {
                if p != k {
                    permuted.swap_rows(k, p);
                }
            }
            for (k, &p) in ldlt.transpositions.iter().enumerate() {
                if p != k {
                    permuted.swap_columns(k, p);
                }
            }
            let rebuilt: DMatrix<f64> = reconstruct(&ldlt, size);
            for r in 0..size {
                for c in 0..size {
                    prop_assert!((rebuilt[(r, c)] - permuted[(r, c)]).abs() < 1e-9);
                }
            }
        }

        /// The unit-lower solve inverts a unit-lower multiply, across the
        /// eight-wide panel boundary.
        #[test]
        fn the_unit_lower_solve_inverts_the_multiply(
            values in prop::collection::vec(-2.0f64..2.0, 121..122),
        ) {
            let size: usize = 11;
            let mut j: DMatrix<f64> = DMatrix::zeros(size, size);
            for r in 0..size {
                for c in 0..size {
                    j[(r, c)] = values[r * size + c];
                }
            }
            let a: DMatrix<f64> = j.transpose() * &j + DMatrix::identity(size, size);
            let ldlt: EigenLdlt<f64> = EigenLdlt::new(a);

            let x: DVector<f64> = DVector::from_iterator(size, values.iter().take(size).copied());
            // `L x`, unit diagonal.
            let mut lx: DVector<f64> = x.clone();
            for i in (0..size).rev() {
                let mut acc: f64 = x[i];
                for k in 0..i {
                    acc += ldlt.mat[(i, k)] * x[k];
                }
                lx[i] = acc;
            }
            let mut solved: DVector<f64> = lx;
            ldlt.solve_unit_lower_in_place(&mut solved);
            for i in 0..size {
                prop_assert!((solved[i] - x[i]).abs() < 1e-9);
            }
        }
    }

    /// The all-zero matrix takes the `k == 0 && !pivot_is_valid` branch and
    /// leaves the identity transpositions (`LDLT.h:348-357`).
    #[test]
    fn an_all_zero_matrix_leaves_identity_transpositions() {
        let ldlt: EigenLdlt<f64> = EigenLdlt::new(DMatrix::zeros(4, 4));
        assert_eq!(ldlt.transpositions, vec![0, 1, 2, 3]);
        assert_eq!(ldlt.vector_d(), DVector::zeros(4));
    }
}
