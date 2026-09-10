//! Eigen's pivoted LDLT at dynamic size: the Levenberg-Marquardt step's solve.
//!
//! `sqrt_keypoint_vio.cpp:1415-1430` solves `(H + lambda diag(H)) inc = b`
//! through `Eigen::LDLT<Eigen::Ref<MatX>>`. Decision D41 settled that Eigen's
//! LDLT is **not** interchangeable with a textbook pivoted LDLT — the IMU stage
//! found a rank-deficient covariance where the two differ by 26 orders of
//! magnitude — so it is ported statement by statement from the vendored Eigen
//! 5.0.1 (`Eigen/Version:12`)
//! (`thirdparty/basalt-headers/thirdparty/eigen/Eigen/src/Cholesky/LDLT.h`),
//! exactly as [`crate::imu`] did at fixed size 9 and `frontend::ldlt` at 3.
//!
//! One property of the factorization decides what a rank-deficient system
//! becomes, and it does not survive a right-looking rewrite: **the pivot is
//! chosen on the un-updated diagonal** (`LDLT.h:335-339`), so at step `k` the
//! trailing diagonal still holds the original entries. The Eigen source says so
//! itself: LDLT "is not rank-revealing" (`:342-344`). A dependent direction
//! therefore leaves a tiny pivot of either sign, and which side of zero the
//! residue lands on is a property of the precision, not of the algorithm.
//!
//! **What is exact and what is not.** The factorization, the transpositions,
//! `vectorD` and the `matrixU() * P` product are elementary operations in
//! Eigen's own order. The two triangular solves reproduce Eigen's panel
//! structure (`TriangularSolverVector.h:30-113`,
//! `EIGEN_TUNE_TRIANGULAR_PANEL_WIDTH = 8`); their trailing updates are
//! nalgebra's `gemv`/`gemv_tr` over views into the factor and into the two
//! disjoint halves of the right-hand side. Until S33 they went through a port of
//! Eigen's own `gemv` blocking, because the increment reaches a threshold
//! comparison and D44 wanted the last bit; the panel *structure* is what decides
//! the answer's shape and it is unchanged, the association inside one panel is
//! now the library's.

use nalgebra::{DMatrix, DVector, DVectorView, DVectorViewMut};

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
    /// Eigen's own `m_temporary` (`LDLT.h:496`), held so a reused
    /// factorization allocates nothing.
    temp: Vec<S>,
    /// The trailing update's accumulators, one per row below the pivot.
    accumulator: Vec<S>,
}

impl<S: LieScalar> EigenLdlt<S> {
    /// An unfactorized `0x0`, for a caller that reuses one factorization.
    ///
    /// Every buffer is sized by [`Self::working_copy`], so this allocates
    /// nothing.
    pub(crate) fn empty() -> Self {
        Self {
            mat: DMatrix::zeros(0, 0),
            transpositions: Vec::new(),
            temp: Vec::new(),
            accumulator: Vec::new(),
        }
    }

    /// The working copy, `size x size` and with **undefined contents**, for a
    /// caller that fills it and then calls [`Self::factor`].
    ///
    /// This and `factor` are `new` split in two, so the Levenberg-Marquardt
    /// step can write the damped matrix straight into the buffer the
    /// factorization consumes: `sqrt_keypoint_vio.cpp:1415-1420` copies `H`,
    /// adds `lambda * diag(H)` and factorizes, up to three times per inner
    /// step, and Eigen's factorization is in place.
    pub(crate) fn working_copy(&mut self, size: usize) -> &mut DMatrix<S> {
        if self.mat.nrows() != size || self.mat.ncols() != size {
            self.mat = DMatrix::zeros(size, size);
        }
        &mut self.mat
    }

    /// `internal::ldlt_inplace<Lower>::unblocked` (`LDLT.h:280-382`).
    ///
    /// Only the lower triangle of `a` is read, as `LDLT<MatrixType, Lower>`
    /// reads. `a` is consumed because Eigen's is an in-place factorization and
    /// `marg_helper.cpp:202` hands it a `Ref` into `marg_H`, which is dead
    /// afterwards.
    ///
    /// The shipped Levenberg-Marquardt path does not use it: it reuses one
    /// factorization through [`Self::working_copy`] and [`Self::factor`], which
    /// are this constructor split in two. This spelling is what the tests
    /// below and the ported reference drive.
    #[cfg(test)]
    pub(crate) fn new(mat: DMatrix<S>) -> Self {
        let mut this: Self = Self {
            mat,
            transpositions: Vec::new(),
            temp: Vec::new(),
            accumulator: Vec::new(),
        };
        this.factor();
        this
    }

    /// The sweep itself, over whatever [`Self::working_copy`] left in place.
    pub(crate) fn factor(&mut self) {
        let Self {
            ref mut mat,
            ref mut transpositions,
            ref mut temp,
            ref mut accumulator,
        } = *self;
        let size: usize = mat.nrows().min(mat.ncols());
        transpositions.clear();
        transpositions.resize(size, 0);
        // Eigen passes one `temp` workspace through the whole sweep
        // (`m_temporary.resize(size)` at `:496`, then `temp.head(k)` at
        // `:336-338`); step `k` reads and writes its first `k` entries, so
        // hoisting it here changes no read and no write.
        temp.clear();
        temp.resize(size, S::zero());
        accumulator.clear();
        accumulator.resize(size, S::zero());

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
                    // `:338`, one accumulator per trailing row instead of one
                    // fold per row.
                    //
                    // Eigen's expression is `A(k+1.., 0..k) * temp`, a
                    // matrix-vector product whose every output coefficient sums
                    // over `j`. Row `i`'s additions still happen in ascending
                    // `j` — `j` is the outer loop, so each `acc[i]` is touched
                    // once per `j`, in order — so no sum is reassociated; what
                    // changes is that the coefficients are read down a column
                    // of the column-major factor, contiguously, rather than
                    // across a row at a stride of `nrows`.
                    let column_stride: usize = mat.nrows();
                    let acc: &mut [S] = &mut accumulator[..rs];
                    acc.fill(S::zero());
                    let data: &[S] = mat.as_slice();
                    for (j, entry) in temp.iter().enumerate() {
                        let factor: S = *entry;
                        let base: usize = j * column_stride + k + 1;
                        let trailing: &[S] = &data[base..base + rs];
                        for (slot, value) in acc.iter_mut().zip(trailing.iter()) {
                            *slot += *value * factor;
                        }
                    }
                    for (i, sum) in acc.iter().enumerate() {
                        mat[(k + 1 + i, k)] -= *sum;
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
                return;
            }

            // `:359-360`. Eigen divides; it does not multiply by a reciprocal.
            if rs > 0 && pivot_is_valid {
                for i in (k + 1)..size {
                    mat[(i, k)] /= real_akk;
                }
            }
        }
    }

    /// `transpositionsP() * v` on the left of a vector: swap rows `k` and
    /// `t[k]` for **ascending** `k` (`ProductEvaluators.h:1194-1200`,
    /// `Side == OnTheLeft`, not transposed), which is the first step of
    /// [`Self::solve_vec`].
    pub(crate) fn apply_transpositions_left_vec(&self, v: &mut DVector<S>) {
        for (k, &j) in self.transpositions.iter().enumerate() {
            if j != k {
                v.swap_rows(k, j);
            }
        }
    }

    /// `matrixL().solveInPlace(v)` (`marg_helper.cpp:224`): the unit-lower
    /// forward substitution of `TriangularSolverVector.h:96-113`, `ColMajor`.
    ///
    /// The panel structure is Eigen's — `PanelWidth = 8`, column-oriented
    /// `axpy` inside a panel, then one `general_matrix_vector_product<ColMajor>`
    /// per panel for the trailing rows.
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
            // The trailing update (`TriangularSolverVector.h:104-113`): one
            // `general_matrix_vector_product<ColMajor>` with `alpha = -1` and
            // `beta = 1`, and nalgebra's `gemv` is that product. It walks the
            // columns (`blas_uninit.rs:157-173`): the first column's `axcpy`
            // carries `beta`, so with `beta = 1` it accumulates into `v` like
            // every later column's does. That is the association a
            // `for j { for i { v[i] -= v[j] * L(i, j) } }` loop has — one
            // rounding into `v` per column, in column order — which is what
            // this call is here to keep.
            let r: usize = size - end_block;
            if r > 0 {
                // The `rhs` is the panel just substituted and the `res` the
                // trailing rows: two disjoint halves of `v` at `end_block`, so
                // the split is what lets one call read and write it in place.
                let (substituted, trailing) = v.as_mut_slice().split_at_mut(end_block);
                let rhs: DVectorView<S> = DVectorView::from_slice(&substituted[pi..], panel);
                let mut res: DVectorViewMut<S> = DVectorViewMut::from_slice(trailing, r);
                res.gemv(
                    -S::one(),
                    &self.mat.view((end_block, pi), (r, panel)),
                    &rhs,
                    S::one(),
                );
            }
            pi = end_block;
        }
    }

    /// `matrixL().adjoint().solveInPlace(v)`, the second substitution of
    /// `LDLT::_solve_impl_transposed` (`LDLT.h:534-565`).
    ///
    /// `matrixL()` is a `UnitLower` view of the column-major factor, so its
    /// adjoint is a `UnitUpper` view of a `Transpose`, which Eigen dispatches to
    /// the **row-major** branch of `triangular_solve_vector`
    /// (`TriangularSolverVector.h:30-72`): panels walk backwards from the end,
    /// each takes one row-major `gemv` against everything already solved below
    /// it, and the panel's own rows are then finished by an inner product whose
    /// `.sum()` follows Eigen's dynamic-length reduction tree.
    ///
    /// **Contract: `v.nrows() == self.transpositions.len()`**, as in
    /// [`Self::solve_unit_lower_in_place`] — the two run back to back on the
    /// same vector.
    pub(crate) fn solve_unit_upper_in_place(&self, v: &mut DVector<S>) {
        debug_assert_eq!(v.nrows(), self.transpositions.len());
        let size: usize = self.transpositions.len();
        let mut pi: usize = size;
        while pi > 0 {
            let panel: usize = TRIANGULAR_PANEL_WIDTH.min(pi);
            let start_row: usize = pi - panel;
            // "remaining size" (`:47`): everything already solved, which in the
            // upper triangle sits to the right of the panel.
            let r: usize = size - pi;
            if r > 0 {
                // The `res` is the panel and the `rhs` everything already
                // substituted to its right: two disjoint halves of `v` at `pi`.
                let (panel_rows, substituted) = v.as_mut_slice().split_at_mut(pi);
                let rhs: DVectorView<S> = DVectorView::from_slice(substituted, r);
                let mut res: DVectorViewMut<S> =
                    DVectorViewMut::from_slice(&mut panel_rows[start_row..], panel);
                // `matrixL().adjoint()` is a `Transpose` of the column-major
                // factor, so the block Eigen reads as `(i, j)` is stored at
                // `(pi + j, start_row + i)`: the transposed product of the stored
                // block, which is `gemv_tr`.
                res.gemv_tr(
                    -S::one(),
                    &self.mat.view((pi, start_row), (r, panel)),
                    &rhs,
                    S::one(),
                );
            }
            // `Mode & UnitDiag`, so there is no division by the diagonal, and
            // the panel's last row (`k == 0`) has nothing to its right yet.
            for k in 1..panel {
                let i: usize = pi - k - 1;
                let s: usize = i + 1;
                // `cjLhs.row(i).segment(s, k)` over the transposed view is
                // `mat[(s + t, i)]`, contiguous in the column-major factor, so
                // the inner product is a dot of two column segments and needs no
                // temporary.
                let update: S = self.mat.column(i).rows(s, k).dot(&v.rows(s, k));
                v[i] -= update;
            }
            pi = start_row;
        }
    }

    /// `transpositionsP().transpose() * v`: swap rows `k` and `t[k]` for
    /// **descending** `k`, undoing [`Self::apply_transpositions_left_vec`].
    pub(crate) fn apply_transpositions_transpose_left_vec(&self, v: &mut DVector<S>) {
        for k in (0..self.transpositions.len()).rev() {
            let j: usize = self.transpositions[k];
            if j != k {
                v.swap_rows(k, j);
            }
        }
    }

    /// `LDLT::solve(rhs)` for one right-hand side
    /// (`LDLT.h:_solve_impl_transposed<true>`), which is what the LM step at
    /// `sqrt_keypoint_vio.cpp:1419-1420` calls.
    ///
    /// `P b`, the unit-lower forward substitution, the **pseudo**-inverse of `D`
    /// — a diagonal entry at or below `numeric_limits<Scalar>::min()` zeroes its
    /// row instead of dividing (Eigen's bug 241; the tolerance is deliberately
    /// `min()` rather than an epsilon, because "LDLT is not rank-revealing" and
    /// LAPACK's `xSYTRS` uses zero) — then the unit-upper back substitution and
    /// `Pᵀ`.
    ///
    /// **Contract: `rhs.nrows() == self.transpositions.len()`**. The LM step
    /// factorizes the damped `H` and solves against the `b` that came out of
    /// the same `get_dense_H_b` (`sqrt_keypoint_vio.cpp:1393`, `:1419`), so the
    /// two agree by construction.
    ///
    /// Allocates its result; the shipped path calls [`Self::solve_vec_into`]
    /// over a vector it keeps.
    #[cfg(test)]
    pub(crate) fn solve_vec(&self, rhs: &DVector<S>) -> DVector<S> {
        let mut dst: DVector<S> = DVector::zeros(rhs.nrows());
        self.solve_vec_into(rhs, &mut dst);
        dst
    }

    /// [`Self::solve_vec`] into a vector the caller keeps.
    ///
    /// `dst` is resized when it has to be and overwritten from `rhs` either
    /// way, so its previous contents decide nothing.
    pub(crate) fn solve_vec_into(&self, rhs: &DVector<S>, dst: &mut DVector<S>) {
        debug_assert_eq!(rhs.nrows(), self.transpositions.len());
        if dst.nrows() == rhs.nrows() {
            dst.copy_from(rhs);
        } else {
            *dst = rhs.clone();
        }
        self.apply_transpositions_left_vec(dst);
        self.solve_unit_lower_in_place(dst);

        let tolerance: S = S::min_positive();
        for i in 0..self.transpositions.len() {
            let d: S = self.mat[(i, i)];
            if d.abs() > tolerance {
                dst[i] /= d;
            } else {
                dst[i] = S::zero();
            }
        }

        self.solve_unit_upper_in_place(dst);
        self.apply_transpositions_transpose_left_vec(dst);
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
        let d: DVector<f64> = DVector::from_iterator(size, (0..size).map(|i| ldlt.mat[(i, i)]));
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

        /// The unit-upper solve inverts a unit-upper multiply, the second
        /// substitution of `LDLT::_solve_impl_transposed`. `matrixL()` is a
        /// `UnitLower` view of the packed factor, so its adjoint's coefficient
        /// `(i, j)` for `j > i` is `mat[(j, i)]`.
        #[test]
        fn the_unit_upper_solve_inverts_the_multiply(
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
            // `Lᵀ x`, unit diagonal.
            let mut ltx: DVector<f64> = x.clone();
            for i in 0..size {
                let mut acc: f64 = x[i];
                for k in (i + 1)..size {
                    acc += ldlt.mat[(k, i)] * x[k];
                }
                ltx[i] = acc;
            }
            let mut solved: DVector<f64> = ltx;
            ldlt.solve_unit_upper_in_place(&mut solved);
            for i in 0..size {
                prop_assert!((solved[i] - x[i]).abs() < 1e-9);
            }
        }

        /// `solve_vec` is `A⁻¹ b` end to end — permutation, both substitutions
        /// and the diagonal — on a well-conditioned SPD `A` wider than one
        /// panel.
        #[test]
        fn solve_vec_inverts_a_positive_definite_system(
            values in prop::collection::vec(-2.0f64..2.0, 121..122),
        ) {
            let size: usize = 11;
            let mut j: DMatrix<f64> = DMatrix::zeros(size, size);
            for r in 0..size {
                for c in 0..size {
                    j[(r, c)] = values[r * size + c];
                }
            }
            let a: DMatrix<f64> = j.transpose() * &j + DMatrix::identity(size, size) * 4.0;
            let x: DVector<f64> = DVector::from_iterator(size, values.iter().take(size).copied());
            let b: DVector<f64> = &a * &x;

            let solved: DVector<f64> = EigenLdlt::new(a).solve_vec(&b);
            for i in 0..size {
                prop_assert!(
                    (solved[i] - x[i]).abs() < 1e-7 * (1.0 + x[i].abs()),
                    "coefficient {i}: {} vs {}", solved[i], x[i]
                );
            }
        }
    }

    /// A zero direction in `D` is pseudo-inverted, not divided by: Eigen's bug
    /// 241 zeroes the row when the pivot is at or below
    /// `numeric_limits<Scalar>::min()` (`LDLT.h:_solve_impl_transposed`), so a
    /// singular system comes back with that coefficient zero instead of an
    /// infinity.
    #[test]
    fn solve_vec_zeroes_a_singular_direction() {
        // `diag(4, 0, 9)`: the middle direction has no information at all.
        let mut a: DMatrix<f64> = DMatrix::zeros(3, 3);
        a[(0, 0)] = 4.0;
        a[(2, 2)] = 9.0;
        let b: DVector<f64> = DVector::from_vec(vec![8.0, 5.0, 27.0]);

        let solved: DVector<f64> = EigenLdlt::new(a).solve_vec(&b);
        assert_eq!(solved[1], 0.0, "the singular direction must be zeroed");
        assert!(solved.iter().all(|v| v.is_finite()));
        // The other two directions are still solved exactly.
        assert!((solved[0] - 2.0).abs() < 1e-12);
        assert!((solved[2] - 3.0).abs() < 1e-12);
    }

    /// The all-zero matrix takes the `k == 0 && !pivot_is_valid` branch and
    /// leaves the identity transpositions (`LDLT.h:348-357`).
    #[test]
    fn an_all_zero_matrix_leaves_identity_transpositions() {
        let ldlt: EigenLdlt<f64> = EigenLdlt::new(DMatrix::zeros(4, 4));
        assert_eq!(ldlt.transpositions, vec![0, 1, 2, 3]);
        assert!((0..4).all(|i| ldlt.mat[(i, i)] == 0.0));
    }
}
