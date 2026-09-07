//! Eigen's Householder reflector and Givens rotation, ported coefficient for
//! coefficient over a dense `DMatrix`.
//!
//! These are the two primitives the landmark block is built from
//! (`landmark_block_abs_dynamic.hpp:429-454`, `:216-250`). nalgebra ships both
//! (`linalg::householder::reflection_axis_mut`, `linalg::givens::GivensRotation`)
//! and the ecosystem dossier maps them one-for-one onto the Eigen calls
//! (rust-ecosystem-gaps §4), but neither is arithmetically identical:
//!
//! * `reflection_axis_mut` normalises the axis to unit length and returns the
//!   *signed norm*, i.e. it builds `H = I - 2 v vᵀ` with `‖v‖ = 1`. Eigen's
//!   `makeHouseholder` builds `H = I - tau v vᵀ` with `v = [1, essential]`, and
//!   the essential part comes out of the division `tail / (c0 - beta)`
//!   (`Householder.h:83`). The two are the same reflection in exact arithmetic
//!   and different in the last bits, and the reflection is applied to the whole
//!   block — the pose Jacobians and the residual column — so the difference
//!   propagates into `Q₂ᵀJ_p`, into `H`, and from there into the increment.
//! * `GivensRotation::cancel_y` computes `c = x/‖(x,y)‖`, `s = -y/‖(x,y)‖` from
//!   one hypot; Eigen's real `makeGivens` (`Jacobi.h:205-233`) branches on
//!   `|p| > |q|` and divides the smaller by the larger first, which is both more
//!   accurate and a different rounding.
//!
//! Decision D44 says an elementary operation whose rounding can reach a
//! threshold comparison is ported in Eigen's operation order rather than
//! delegated to nalgebra, and here the whole point of the fixture is that these
//! numbers reach `det(Q1Jl) == 0` and the Levenberg-Marquardt accept/reject
//! test. So: **Eigen's arithmetic, ported; nalgebra's versions are not called.**
//!
//! One reduction order is worth spelling out. `makeHouseholder` needs
//! `tail.squaredNorm()` over a *column* of `storage`, and `storage` is
//! `Eigen::RowMajor` (`landmark_block_abs_dynamic.hpp:530`), so that column has
//! an inner stride of `num_cols` and is not packet-accessible. Eigen therefore
//! takes `DefaultTraversal` with no unrolling (`Redux.h:174-192`) — a strictly
//! sequential left fold — in **both** precisions. That is why the fold below is
//! sequential and why it does not go through [`LieScalar::eigen_redux3`], which
//! covers the contiguous three-coefficient case and is precision-dependent
//! (decision D47).

use nalgebra::{DMatrix, DVector};

use crate::ba_base::JacobiRotation;
use crate::lie::LieScalar;

/// The sub-block a reflection acts on, `storage.block(row_start, col_start,
/// rows, cols)` in Eigen's spelling.
///
/// Four indices travel together through
/// [`apply_householder_on_the_left_block`] and
/// [`apply_householder_on_the_right_block`]; naming them as one value is what
/// keeps those signatures readable.
///
/// **Contract: the span must lie inside the matrix**, and the two functions
/// index without re-checking, because they run once per column of a QR sweep.
/// The checked *public* boundary is [`crate::linearize::reflect_column`], whose
/// `checked_add` and `ReflectionOutOfRange` guard a caller-supplied range
/// (decision D32). Everything else that builds a `BlockSpan` computes a span
/// that is exact by construction, and each one is one line away from the
/// dimension it is exact against:
///
/// * `reflect_column` — the full width, after its own range check;
/// * `marg_helper`'s flat QR — `row_start + rows == q2jp.nrows()` and
///   `col_start + cols == q2jp.ncols()`, over index sets `check_indices` has
///   already validated against the column count;
/// * `ColPivHouseholderQr` — `k + (rows - k)` and `(k + 1) + (cols - k - 1)`;
/// * `Cod` — `(rank - 1) + (cols - rank + 1) == cols`, with `rank <= cols`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockSpan {
    /// First row of the block.
    pub(crate) row_start: usize,
    /// Number of rows.
    pub(crate) rows: usize,
    /// First column of the block.
    pub(crate) col_start: usize,
    /// Number of columns.
    pub(crate) cols: usize,
}

/// `makeHouseholder` (`Householder.h:63-86`), real scalars, over
/// `storage.col(col).segment(start, len)`.
///
/// Writes the `len - 1` coefficients of the essential part into `essential` and
/// returns `(tau, beta)`. `beta` is what the reflected column's first
/// coefficient would be; basalt never uses it (`:448-450` names it and drops
/// it), because [`apply_householder_on_the_left`] recomputes the whole column.
///
/// The `tailSqNorm <= tol` branch (`:76-79`) is the already-reduced column:
/// `tau = 0` makes the reflection the identity, which is what
/// [`apply_householder_on_the_left`] then skips.
pub(crate) fn make_householder<S: LieScalar>(
    storage: &DMatrix<S>,
    col: usize,
    start: usize,
    len: usize,
    essential: &mut [S],
) -> (S, S) {
    let c0: S = storage[(start, col)];

    // `tail.squaredNorm()` (`:72`): a sequential fold, see the module docs. The
    // `size() == 1` guard of `:72` is the empty loop here.
    let mut tail_sq_norm: S = S::zero();
    for i in 1..len {
        let v: S = storage[(start + i, col)];
        tail_sq_norm += v * v;
    }

    // `std::numeric_limits<RealScalar>::min()` (`:74`), the smallest positive
    // normal — not `RealField::min_value()`, which is the most negative finite.
    let tol: S = S::min_positive();

    if tail_sq_norm <= tol {
        // `:77-79`. The imaginary-part test of `:76` is vacuous for real scalars.
        for e in essential.iter_mut().take(len.saturating_sub(1)) {
            *e = S::zero();
        }
        (S::zero(), c0)
    } else {
        // `:81-84`.
        let mut beta: S = (c0 * c0 + tail_sq_norm).sqrt();
        if c0 >= S::zero() {
            beta = -beta;
        }
        let denom: S = c0 - beta;
        for (i, e) in essential.iter_mut().enumerate().take(len - 1) {
            *e = storage[(start + 1 + i, col)] / denom;
        }
        let tau: S = (beta - c0) / beta;
        (tau, beta)
    }
}

/// `applyHouseholderOnTheLeft` (`Householder.h:103-118`), real scalars, over
/// `storage.block(start, 0, len, num_cols)`.
///
/// `work` is the `tmp` row of `:110`, at least `num_cols` long; basalt passes
/// `tempVector1.data()` (`:452`). The reflection is applied to **every** column
/// of the block — the pose Jacobians, the padding, the landmark columns and the
/// residual — which is exactly what leaves `Q₂ᵀJ_p` and `Q₂ᵀr` in place
/// (papers-part2 §12.5).
pub(crate) fn apply_householder_on_the_left<S: LieScalar>(
    storage: &mut DMatrix<S>,
    start: usize,
    len: usize,
    essential: &[S],
    tau: S,
    work: &mut [S],
) {
    // Every column of the block: the full-width special case of
    // [`apply_householder_on_the_left_block`], which is the only
    // implementation so the two cannot drift apart.
    let num_cols: usize = storage.ncols();
    apply_householder_on_the_left_block(
        storage,
        BlockSpan {
            row_start: start,
            rows: len,
            col_start: 0,
            cols: num_cols,
        },
        essential,
        tau,
        work,
    );
}

/// The same reflection over an arbitrary sub-block
/// `storage.block(row_start, col_start, rows, cols)`.
///
/// `applyHouseholderOnTheLeft` is a method on whatever expression it is called
/// on, and the marginalization QR calls it on `Q2Jp.bottomRightCorner(...)`
/// rather than on a full-width block (`marg_helper.cpp:304-305`), so the column
/// range has to be a parameter. [`apply_householder_on_the_left`] is this
/// function over every column, which is what the landmark block wants
/// (`landmark_block_abs_dynamic.hpp:452`).
///
/// `work` is the `tmp` row of `Householder.h:110` and must be at least `cols`
/// long; C++ passes a pointer into a shared scratch vector offset by the
/// column the QR is on (`marg_helper.cpp:305`, `tempData + k + 1`), which is
/// the same storage with a different name.
pub(crate) fn apply_householder_on_the_left_block<S: LieScalar>(
    storage: &mut DMatrix<S>,
    span: BlockSpan,
    essential: &[S],
    tau: S,
    work: &mut [S],
) {
    let BlockSpan {
        row_start,
        rows,
        col_start,
        cols,
    } = span;
    if rows == 1 {
        // `:107-108`.
        let factor: S = S::one() - tau;
        for j in 0..cols {
            storage[(row_start, col_start + j)] *= factor;
        }
        return;
    }
    if tau == S::zero() {
        // `:109`, `is_exactly_zero`.
        return;
    }

    // `tmp.noalias() = essential.adjoint() * bottom` (`:113`): one dot product
    // per column, over the `rows - 1` rows below the first.
    for (j, slot) in work.iter_mut().enumerate().take(cols) {
        let mut acc: S = S::zero();
        for (i, e) in essential.iter().enumerate().take(rows - 1) {
            acc += *e * storage[(row_start + 1 + i, col_start + j)];
        }
        *slot = acc;
    }

    // `tmp += this->row(0)` (`:114`).
    for (j, slot) in work.iter_mut().enumerate().take(cols) {
        *slot += storage[(row_start, col_start + j)];
    }

    // `this->row(0) -= tau * tmp` (`:115`).
    for j in 0..cols {
        storage[(row_start, col_start + j)] -= tau * work[j];
    }

    // `bottom.noalias() -= tau * essential * tmp` (`:116`): the outer product,
    // with the scalar folded into the left factor as C++'s left-associative
    // `*` does.
    for (i, e) in essential.iter().enumerate().take(rows - 1) {
        let scale: S = tau * *e;
        for j in 0..cols {
            storage[(row_start + 1 + i, col_start + j)] -= scale * work[j];
        }
    }
}

/// `applyHouseholderOnTheLeft` on a column vector segment `v.segment(start, len)`.
///
/// The `Q2r` half of the marginalization QR (`marg_helper.cpp:306`) — the same
/// reflection as [`apply_householder_on_the_left_block`] with a single column,
/// where the `tmp` row of `Householder.h:110` collapses to one scalar.
pub(crate) fn apply_householder_on_the_left_vec<S: LieScalar>(
    v: &mut DVector<S>,
    start: usize,
    len: usize,
    essential: &[S],
    tau: S,
) {
    if len == 1 {
        v[start] *= S::one() - tau;
        return;
    }
    if tau == S::zero() {
        return;
    }
    let mut tmp: S = S::zero();
    for (i, e) in essential.iter().enumerate().take(len - 1) {
        tmp += *e * v[start + 1 + i];
    }
    tmp += v[start];
    v[start] -= tau * tmp;
    for (i, e) in essential.iter().enumerate().take(len - 1) {
        v[start + 1 + i] -= (tau * *e) * tmp;
    }
}

/// `makeHouseholder` over a **row** segment `storage.row(row).segment(col_start, len)`.
///
/// Same arithmetic as [`make_householder`], different traversal: the complete
/// orthogonal decomposition builds its `Z` reflectors out of rows
/// (`CompleteOrthogonalDecomposition.h:487`).
pub(crate) fn make_householder_row<S: LieScalar>(
    storage: &DMatrix<S>,
    row: usize,
    col_start: usize,
    len: usize,
    essential: &mut [S],
) -> (S, S) {
    let c0: S = storage[(row, col_start)];
    let mut tail_sq_norm: S = S::zero();
    for i in 1..len {
        let v: S = storage[(row, col_start + i)];
        tail_sq_norm += v * v;
    }
    let tol: S = S::min_positive();
    if tail_sq_norm <= tol {
        for e in essential.iter_mut().take(len.saturating_sub(1)) {
            *e = S::zero();
        }
        (S::zero(), c0)
    } else {
        let mut beta: S = (c0 * c0 + tail_sq_norm).sqrt();
        if c0 >= S::zero() {
            beta = -beta;
        }
        let denom: S = c0 - beta;
        for (i, e) in essential.iter_mut().enumerate().take(len - 1) {
            *e = storage[(row, col_start + 1 + i)] / denom;
        }
        let tau: S = (beta - c0) / beta;
        (tau, beta)
    }
}

/// `applyHouseholderOnTheRight` (`Householder.h:137-150`), real scalars, over
/// `storage.block(row_start, col_start, rows, cols)`.
///
/// The mirror of [`apply_householder_on_the_left_block`]: the reflector acts on
/// the block's **columns**, `tmp` is a column of `rows` entries, and the rank
/// one update is `tau * tmp * essentialᵀ` with the scalar folded into the left
/// factor. Used only by the complete orthogonal decomposition
/// (`CompleteOrthogonalDecomposition.h:491-492`).
pub(crate) fn apply_householder_on_the_right_block<S: LieScalar>(
    storage: &mut DMatrix<S>,
    span: BlockSpan,
    essential: &[S],
    tau: S,
    work: &mut [S],
) {
    let BlockSpan {
        row_start,
        rows,
        col_start,
        cols,
    } = span;
    if cols == 1 {
        // `:139-140`.
        let factor: S = S::one() - tau;
        for i in 0..rows {
            storage[(row_start + i, col_start)] *= factor;
        }
        return;
    }
    if tau == S::zero() {
        // `:141`.
        return;
    }

    // `tmp.noalias() = right * essential` (`:145`).
    for (i, slot) in work.iter_mut().enumerate().take(rows) {
        let mut acc: S = S::zero();
        for (j, e) in essential.iter().enumerate().take(cols - 1) {
            acc += storage[(row_start + i, col_start + 1 + j)] * *e;
        }
        *slot = acc;
    }

    // `tmp += this->col(0)` (`:146`).
    for (i, slot) in work.iter_mut().enumerate().take(rows) {
        *slot += storage[(row_start + i, col_start)];
    }

    // `this->col(0) -= tau * tmp` (`:147`).
    for i in 0..rows {
        storage[(row_start + i, col_start)] -= tau * work[i];
    }

    // `right.noalias() -= tau * tmp * essential.adjoint()` (`:148`).
    for i in 0..rows {
        let scale: S = tau * work[i];
        for (j, e) in essential.iter().enumerate().take(cols - 1) {
            storage[(row_start + i, col_start + 1 + j)] -= scale * *e;
        }
    }
}

/// `JacobiRotation::makeGivens(p, q)`, the real specialization
/// (`Jacobi.h:205-233`).
///
/// The rotation that zeroes `q` against `p`. The branch on `|p| > |q|` always
/// divides the smaller magnitude by the larger, so neither `t` nor `1 + t²`
/// can overflow.
pub(crate) fn make_givens<S: LieScalar>(p: S, q: S) -> JacobiRotation<S> {
    if q == S::zero() {
        // `:212-214`.
        JacobiRotation {
            c: if p < S::zero() { -S::one() } else { S::one() },
            s: S::zero(),
        }
    } else if p == S::zero() {
        // `:215-217`.
        JacobiRotation {
            c: S::zero(),
            s: if q < S::zero() { S::one() } else { -S::one() },
        }
    } else if p.abs() > q.abs() {
        // `:218-224`.
        let t: S = q / p;
        let mut u: S = (S::one() + t * t).sqrt();
        if p < S::zero() {
            u = -u;
        }
        let c: S = S::one() / u;
        JacobiRotation { c, s: -t * c }
    } else {
        // `:225-231`.
        let t: S = p / q;
        let mut u: S = (S::one() + t * t).sqrt();
        if q < S::zero() {
            u = -u;
        }
        let s: S = -S::one() / u;
        JacobiRotation { c: -t * s, s }
    }
}

/// `MatrixBase::applyOnTheLeft(p, q, j)` (`Jacobi.h:261-266`) on two rows of a
/// dense matrix: `apply_rotation_in_the_plane` (`:286-300`) with every `conj`
/// an identity.
pub(crate) fn apply_rotation_on_the_left<S: LieScalar>(
    storage: &mut DMatrix<S>,
    p: usize,
    q: usize,
    rot: JacobiRotation<S>,
) {
    for j in 0..storage.ncols() {
        let x: S = storage[(p, j)];
        let y: S = storage[(q, j)];
        storage[(p, j)] = rot.c * x + rot.s * y;
        storage[(q, j)] = -rot.s * x + rot.c * y;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nalgebra::DMatrix;
    use proptest::prelude::*;

    /// Rebuild `H = I - tau v vᵀ` from what [`make_householder`] returned and
    /// check it against the reflection the block actually applies.
    fn dense_reflection(len: usize, essential: &[S64], tau: f64) -> DMatrix<f64> {
        let mut v: nalgebra::DVector<f64> = nalgebra::DVector::zeros(len);
        v[0] = 1.0;
        for i in 1..len {
            v[i] = essential[i - 1];
        }
        DMatrix::identity(len, len) - v.clone() * v.transpose() * tau
    }

    type S64 = f64;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The reflector zeroes everything below the first coefficient, and the
        /// first coefficient becomes `beta`.
        #[test]
        fn the_householder_reflector_reduces_the_column(
            values in prop::collection::vec(-10.0f64..10.0, 2..8),
        ) {
            let len: usize = values.len();
            let mut storage: DMatrix<f64> = DMatrix::zeros(len, 1);
            for (i, v) in values.iter().enumerate() {
                storage[(i, 0)] = *v;
            }
            let mut essential: Vec<f64> = vec![0.0; len - 1];
            let (tau, beta) = make_householder(&storage, 0, 0, len, &mut essential);
            let h: DMatrix<f64> = dense_reflection(len, &essential, tau);
            let reflected: nalgebra::DVector<f64> =
                h * nalgebra::DVector::from_column_slice(&values);
            prop_assert!((reflected[0] - beta).abs() < 1e-9 * beta.abs().max(1.0));
            for i in 1..len {
                prop_assert!(reflected[i].abs() < 1e-9);
            }
        }

        /// Applying the reflector through [`apply_householder_on_the_left`] is
        /// the same as multiplying by the dense `H`.
        #[test]
        fn applying_the_reflector_matches_the_dense_product(
            values in prop::collection::vec(-10.0f64..10.0, 2..7),
            extra in prop::collection::vec(-5.0f64..5.0, 12..13),
        ) {
            let len: usize = values.len();
            let cols: usize = 4;
            let mut storage: DMatrix<f64> = DMatrix::zeros(len, cols);
            for i in 0..len {
                storage[(i, 0)] = values[i];
                for j in 1..cols {
                    storage[(i, j)] = extra[(i * cols + j) % extra.len()];
                }
            }
            let mut essential: Vec<f64> = vec![0.0; len - 1];
            let (tau, _) = make_householder(&storage, 0, 0, len, &mut essential);
            let dense: DMatrix<f64> = dense_reflection(len, &essential, tau) * storage.clone();
            let mut work: Vec<f64> = vec![0.0; cols];
            apply_householder_on_the_left(&mut storage, 0, len, &essential, tau, &mut work);
            for i in 0..len {
                for j in 0..cols {
                    prop_assert!((storage[(i, j)] - dense[(i, j)]).abs() < 1e-9);
                }
            }
        }

        /// The block form over the whole width is the full-width function.
        ///
        /// [`apply_householder_on_the_left`] delegates to
        /// [`apply_householder_on_the_left_block`], so the landmark block's
        /// proven path and the marginalization QR's share one implementation;
        /// this is what says the delegation changed nothing.
        #[test]
        fn the_block_form_over_the_whole_width_is_the_full_width_form(
            values in prop::collection::vec(-4.0f64..4.0, 30..31),
        ) {
            let (rows, cols): (usize, usize) = (5, 6);
            let build = || -> DMatrix<f64> {
                DMatrix::from_fn(rows, cols, |i, j| values[i * cols + j])
            };
            let mut essential: Vec<f64> = vec![0.0; rows - 1];
            let mut a: DMatrix<f64> = build();
            let (tau, _) = make_householder(&a, 0, 0, rows, &mut essential);

            let mut work: Vec<f64> = vec![0.0; cols];
            apply_householder_on_the_left(&mut a, 0, rows, &essential, tau, &mut work);

            let mut b: DMatrix<f64> = build();
            let mut work: Vec<f64> = vec![0.0; cols];
            apply_householder_on_the_left_block(
                &mut b,
                BlockSpan { row_start: 0, rows, col_start: 0, cols },
                &essential,
                tau,
                &mut work,
            );
            prop_assert_eq!(a, b);
        }

        /// A reflection on a sub-block leaves everything outside the span
        /// untouched, which is what lets the flat QR of `marg_helper.cpp:304`
        /// act on `bottomRightCorner` without disturbing the columns already
        /// reduced.
        #[test]
        fn a_sub_block_reflection_touches_nothing_outside_the_span(
            values in prop::collection::vec(-4.0f64..4.0, 42..43),
        ) {
            let (rows, cols): (usize, usize) = (6, 7);
            let before: DMatrix<f64> =
                DMatrix::from_fn(rows, cols, |i, j| values[i * cols + j]);
            let mut after: DMatrix<f64> = before.clone();
            let span: BlockSpan =
                BlockSpan { row_start: 2, rows: 4, col_start: 3, cols: 4 };

            let mut essential: Vec<f64> = vec![0.0; span.rows - 1];
            let (tau, _) = make_householder(&after, 1, span.row_start, span.rows, &mut essential);
            let mut work: Vec<f64> = vec![0.0; span.cols];
            apply_householder_on_the_left_block(&mut after, span, &essential, tau, &mut work);

            for i in 0..rows {
                for j in 0..cols {
                    let inside: bool = i >= span.row_start
                        && i < span.row_start + span.rows
                        && j >= span.col_start
                        && j < span.col_start + span.cols;
                    if !inside {
                        prop_assert_eq!(after[(i, j)], before[(i, j)], "at ({}, {})", i, j);
                    }
                }
            }
        }

        /// `makeGivens` produces a rotation (`c² + s² = 1`) that cancels `q`.
        ///
        /// Which row it cancels depends on the argument order at the call site,
        /// and basalt's is the reverse of the obvious one: `makeGivens(p, q)`
        /// then `applyOnTheLeft(row_of_q, row_of_p, rot)`
        /// (`landmark_block_abs_dynamic.hpp:435-436`, `:245-246`). With
        /// `apply_rotation_in_the_plane`'s `x' = c x + s y` that puts
        /// `c q + s p` in the first row — the zero — and `-s q + c p = r` in the
        /// second. Getting this backwards leaves the block upper *left*
        /// triangular and every later step wrong.
        #[test]
        fn the_givens_rotation_cancels_the_second_coefficient(p in -10.0f64..10.0, q in -10.0f64..10.0) {
            let rot: JacobiRotation<f64> = make_givens(p, q);
            prop_assert!((rot.c * rot.c + rot.s * rot.s - 1.0).abs() < 1e-12);
            let scale: f64 = p.abs().max(q.abs()).max(1.0);
            let cancelled: f64 = rot.c * q + rot.s * p;
            prop_assert!(cancelled.abs() < 1e-12 * scale, "cancelled {cancelled}");
            // ...and the surviving coefficient is the hypotenuse, up to sign.
            let survivor: f64 = -rot.s * q + rot.c * p;
            prop_assert!((survivor.abs() - (p * p + q * q).sqrt()).abs() < 1e-12 * scale);
        }
    }
}
