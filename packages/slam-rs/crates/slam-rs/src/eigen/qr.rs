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
//! **One reduction order is the whole of decision D47 again, and it is not the
//! same at every call site.** `makeHouseholder` needs `tail.squaredNorm()` over
//! a *column* of the matrix it is called on, and which `Redux.h` traversal that
//! takes depends on the **C++** matrix's storage order:
//!
//! * the landmark block's `storage` is `Eigen::RowMajor`
//!   (`landmark_block_abs_dynamic.hpp:530`), so its columns have an inner stride
//!   of `num_cols`, carry no `PacketAccessBit`, and reduce through
//!   `LinearTraversal` — a strictly sequential left fold (`Redux.h:236-244`) —
//!   in **both** precisions;
//! * basalt's marginalization matrices are plain `Eigen::Matrix<Scalar,
//!   Dynamic, Dynamic>`, which is column-major, so a column segment is
//!   *contiguous* and reduces through `LinearVectorizedTraversal`
//!   (`Redux.h:274-325`): two packet accumulators, `predux` to fold the lanes,
//!   then a scalar tail.
//!
//! The two disagree in the last bits, and the result reaches
//! `|beta| > sqrt(epsilon)` (`marg_helper.cpp:284`) and
//! `ColPivHouseholderQR::rank()`. [`ColumnRedux`] is therefore a parameter of
//! [`make_householder`], named at every call site, and
//! [`contiguous_squared_norm`] is the vectorised order. The measured agreement
//! against the fork's own Eigen, and against every wrong order that was tried,
//! is in the package README under "Marginalization, and the two places a rank
//! decision is load-bearing".

use nalgebra::{DMatrix, DVector};

use super::blas::redux_contiguous;
use super::svd::JacobiRotation;
use crate::lie::LieScalar;

/// The sub-block a reflection acts on, `storage.block(row_start, col_start,
/// rows, cols)` in Eigen's spelling.
///
/// Four indices travel together through
/// [`apply_householder_on_the_left_block`]; naming them as one value is what
/// keeps that signature readable.
///
/// **Contract: the span must lie inside the matrix**, and the two functions
/// index without re-checking, because they run once per column of a QR sweep;
/// a `debug_assert!` in each states the contract without paying for it in
/// release. The checked *public* boundary for a caller-supplied range is
/// [`crate::linearize::reflect_column`], whose `checked_add` and
/// `ReflectionOutOfRange` guard it (decision D32).
///
/// Every other site computes a span that is exact against a dimension one line
/// away from it, with one exception worth naming:
/// `ColPivHouseholderQr::apply_q_adjoint_on_the_left` spans the *factorized
/// matrix's* row count over a right-hand side the caller supplied. That is
/// exact only because [`crate::marg::Cod::solve`] refuses a right-hand side of
/// any other height — and it is the one the S7 review found unguarded.
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

/// Which `Redux.h` traversal a column segment's `squaredNorm()` takes, which is
/// decided by the **C++** matrix's storage order and not by nalgebra's.
///
/// Every `DMatrix` in this port is column-major whatever it stands for, so the
/// distinction cannot be read off the Rust type; it has to be named where the
/// reduction happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ColumnRedux {
    /// The C++ matrix is `Eigen::RowMajor`, so the column has an inner stride of
    /// `cols`, no `PacketAccessBit`, and `LinearTraversal` folds it left to
    /// right (`Redux.h:236-244`). The landmark block
    /// (`landmark_block_abs_dynamic.hpp:530`).
    Strided,
    /// The C++ matrix is column-major, so the segment is contiguous and
    /// `LinearVectorizedTraversal` runs (`Redux.h:274-325`). basalt's
    /// marginalization matrices — `Q2Jp` in `marg_helper.cpp` and `m_qr` in
    /// `ColPivHouseholderQR`.
    Contiguous,
}

/// `squaredNorm()` over the **contiguous** column segment
/// `storage.col(col).segment(start, len)`, in Eigen's order.
///
/// `unaryExpr(squared_norm_functor)` (`Dot.h:24`) reduced by
/// [`crate::eigen::blas::redux_contiguous`], whose doc carries the traversal and
/// the `alignedStart == 0` argument; `Evaluator::SizeAtCompileTime` is
/// `Dynamic`, so the cost is `HugeCost` and the unrolled variants never apply.
/// The shape sweep this order was verified on, and what each wrong order
/// scores on it, are in the package README.
pub(crate) fn contiguous_squared_norm<S: LieScalar>(
    storage: &DMatrix<S>,
    col: usize,
    start: usize,
    len: usize,
) -> S {
    redux_contiguous(len, |i| {
        let v: S = storage[(start + i, col)];
        v * v
    })
}

/// `makeHouseholder` (`Householder.h:63-86`), real scalars, over
/// `storage.col(col).segment(start, len)`.
///
/// Writes the `len - 1` coefficients of the essential part into `essential` and
/// returns `(tau, beta)`. `beta` is what the reflected column's first
/// coefficient would be. basalt's landmark QR names it and drops it
/// (`:448-450`), because [`apply_householder_on_the_left`] recomputes the whole
/// column; `crate::marg::helper` keeps it, and that is the rank test below.
///
/// The `tailSqNorm <= tol` branch (`:76-79`) is the already-reduced column:
/// `tau = 0` makes the reflection the identity, which is what
/// [`apply_householder_on_the_left`] then skips.
///
/// `redux` names which reduction `tail.squaredNorm()` (`:72`) takes; see the
/// module docs and [`ColumnRedux`]. Getting it wrong moves `beta`, and `beta`
/// is what the marginalization QR compares against `sqrt(epsilon)`.
pub(crate) fn make_householder<S: LieScalar>(
    storage: &DMatrix<S>,
    col: usize,
    start: usize,
    len: usize,
    redux: ColumnRedux,
    essential: &mut [S],
) -> (S, S) {
    // `tail.squaredNorm()` (`:72`) over `segment(start + 1, len - 1)`. The
    // `size() == 1` guard of `:72` is the empty fold / zero length here, and
    // `saturating_sub` and the empty-sum rule agree that a tail of no
    // coefficients is zero.
    let tail_sq_norm: S = match redux {
        ColumnRedux::Strided => {
            let mut acc: S = S::zero();
            for i in 1..len {
                let v: S = storage[(start + i, col)];
                acc += v * v;
            }
            acc
        }
        ColumnRedux::Contiguous => {
            contiguous_squared_norm(storage, col, start + 1, len.saturating_sub(1))
        }
    };
    reflector_from_tail(storage[(start, col)], tail_sq_norm, len, essential, |i| {
        storage[(start + 1 + i, col)]
    })
}

/// The rest of `makeHouseholder` once `tail.squaredNorm()` is known
/// (`Householder.h:74-85`): the `numeric_limits::min()` test, the sign of
/// `beta`, the division that makes the essential part, and `tau`.
///
/// `tail` reads coefficient `i` of the tail, which [`make_householder`] takes
/// down a column. C++ has a second entry point that takes it along a row, for
/// the complete orthogonal decomposition's `Z` reflectors
/// (`CompleteOrthogonalDecomposition.h:487`); the port had one too until the
/// squared-form marginalization it served went (D68).
fn reflector_from_tail<S: LieScalar>(
    c0: S,
    tail_sq_norm: S,
    len: usize,
    essential: &mut [S],
    tail: impl Fn(usize) -> S,
) -> (S, S) {
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
            *e = tail(i) / denom;
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
    debug_assert!(
        row_start + rows <= storage.nrows() && col_start + cols <= storage.ncols(),
        "span {span:?} outside a {}x{} matrix",
        storage.nrows(),
        storage.ncols()
    );
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]

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

    /// `applyHouseholderOnTheLeft` (`Householder.h:103-118`) written out over a
    /// nested `Vec`, so it shares no indexing with the code under test.
    fn reference_apply_left(a: &mut [Vec<f64>], span: BlockSpan, essential: &[f64], tau: f64) {
        let (r0, c0): (usize, usize) = (span.row_start, span.col_start);
        if span.rows == 1 {
            // `:107-108`.
            for j in 0..span.cols {
                a[r0][c0 + j] *= 1.0 - tau;
            }
            return;
        }
        if tau == 0.0 {
            // `:109`.
            return;
        }
        // `:113`.
        let mut tmp: Vec<f64> = (0..span.cols)
            .map(|j| {
                let mut acc: f64 = 0.0;
                for (i, e) in essential.iter().enumerate().take(span.rows - 1) {
                    acc += *e * a[r0 + 1 + i][c0 + j];
                }
                acc
            })
            .collect();
        // `:114`.
        for (j, slot) in tmp.iter_mut().enumerate() {
            *slot += a[r0][c0 + j];
        }
        // `:115`.
        for j in 0..span.cols {
            a[r0][c0 + j] -= tau * tmp[j];
        }
        // `:116`.
        for (i, e) in essential.iter().enumerate().take(span.rows - 1) {
            let scale: f64 = tau * *e;
            for j in 0..span.cols {
                a[r0 + 1 + i][c0 + j] -= scale * tmp[j];
            }
        }
    }

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
            let (tau, beta) =
                make_householder(&storage, 0, 0, len, ColumnRedux::Strided, &mut essential);
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
            let (tau, _) =
                make_householder(&storage, 0, 0, len, ColumnRedux::Strided, &mut essential);
            let dense: DMatrix<f64> = dense_reflection(len, &essential, tau) * storage.clone();
            let mut work: Vec<f64> = vec![0.0; cols];
            apply_householder_on_the_left(&mut storage, 0, len, &essential, tau, &mut work);
            for i in 0..len {
                for j in 0..cols {
                    prop_assert!((storage[(i, j)] - dense[(i, j)]).abs() < 1e-9);
                }
            }
        }

        /// [`apply_householder_on_the_left_block`] is `Householder.h:103-118`,
        /// bit for bit, on a sub-block as well as on the whole width.
        ///
        /// The reference is [`reference_apply_left`] — the same four statements
        /// written out over plain row and column indices, with no
        /// [`BlockSpan`] and no shared code — so a span computed from the
        /// wrong bound shows up here rather than in an out-of-range read.
        /// Comparing [`apply_householder_on_the_left`] against
        /// [`apply_householder_on_the_left_block`] would not: the first
        /// delegates to the second. The *independent* check on the arithmetic
        /// itself is `applying_the_reflector_matches_the_dense_product` above,
        /// which multiplies by `I - τ v vᵀ`.
        #[test]
        fn the_block_reflection_is_eigens_four_statements(
            values in prop::collection::vec(-4.0f64..4.0, 56..57),
            row_start in 0usize..3,
            col_start in 0usize..3,
        ) {
            let (rows, cols): (usize, usize) = (7, 8);
            let build = || -> DMatrix<f64> {
                DMatrix::from_fn(rows, cols, |i, j| values[i * cols + j])
            };
            let span: BlockSpan = BlockSpan {
                row_start,
                rows: rows - row_start,
                col_start,
                cols: cols - col_start,
            };

            let mut essential: Vec<f64> = vec![0.0; span.rows.saturating_sub(1)];
            let source: DMatrix<f64> = build();
            let (tau, _) = make_householder(
                &source,
                col_start,
                span.row_start,
                span.rows,
                ColumnRedux::Strided,
                &mut essential,
            );

            let mut got: DMatrix<f64> = build();
            let mut work: Vec<f64> = vec![0.0; span.cols];
            apply_householder_on_the_left_block(&mut got, span, &essential, tau, &mut work);

            let mut want: Vec<Vec<f64>> = (0..rows)
                .map(|i| (0..cols).map(|j| source[(i, j)]).collect())
                .collect();
            reference_apply_left(&mut want, span, &essential, tau);

            for i in 0..rows {
                for j in 0..cols {
                    prop_assert_eq!(got[(i, j)], want[i][j], "at ({}, {})", i, j);
                }
            }

            // ...and the full-width entry point is that block over every
            // column, which is the one thing the delegation has to preserve.
            let full: BlockSpan =
                BlockSpan { row_start, rows: rows - row_start, col_start: 0, cols };
            let mut wide: DMatrix<f64> = build();
            let mut work: Vec<f64> = vec![0.0; cols];
            apply_householder_on_the_left(
                &mut wide, full.row_start, full.rows, &essential, tau, &mut work,
            );
            let mut wide_want: Vec<Vec<f64>> = (0..rows)
                .map(|i| (0..cols).map(|j| source[(i, j)]).collect())
                .collect();
            reference_apply_left(&mut wide_want, full, &essential, tau);
            for i in 0..rows {
                for j in 0..cols {
                    prop_assert_eq!(wide[(i, j)], wide_want[i][j], "wide at ({}, {})", i, j);
                }
            }

            // ...and the vector entry point is that block with one column,
            // which is the shape `marg_helper.cpp:306` reflects `Q2r` on: the
            // reflector of column `col_start`, applied to the residual column.
            let residual: usize = cols - 1;
            let mut narrow: DVector<f64> = DVector::from_fn(rows, |i, _| source[(i, residual)]);
            apply_householder_on_the_left_vec(
                &mut narrow, span.row_start, span.rows, &essential, tau,
            );
            let mut narrow_want: Vec<Vec<f64>> =
                (0..rows).map(|i| vec![source[(i, residual)]]).collect();
            reference_apply_left(
                &mut narrow_want,
                BlockSpan { row_start, rows: span.rows, col_start: 0, cols: 1 },
                &essential,
                tau,
            );
            for i in 0..rows {
                prop_assert_eq!(narrow[i], narrow_want[i][0], "vec at {}", i);
            }
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
            let (tau, _) = make_householder(
                &after,
                1,
                span.row_start,
                span.rows,
                ColumnRedux::Strided,
                &mut essential,
            );
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

    /// One case of [`contiguous_squared_norm`] against the bits Eigen produced
    /// for it, decoded from `column` and compared to `expected`.
    fn assert_contiguous<S: LieScalar>(column: &[S], expected: S) {
        let len: usize = column.len();
        // A wide matrix, so the reduced column is neither the first nor the
        // last: the port must not depend on where in the allocation it sits.
        let mut storage: DMatrix<S> = DMatrix::zeros(len, 3);
        for (i, v) in column.iter().enumerate() {
            storage[(i, 1)] = *v;
        }
        let got: S = contiguous_squared_norm(&storage, 1, 0, len);
        assert_eq!(
            got.to_f64().to_bits(),
            expected.to_f64().to_bits(),
            "len {len}: got {got:?}, Eigen {expected:?}"
        );
    }

    /// [`contiguous_squared_norm`] reproduces Eigen's `squaredNorm()` bit for
    /// bit on four cases, one per branch of `Redux.h:274-325`.
    ///
    /// The numbers are the fork's, printed by `tools/marg_norm_probe.cpp` as
    /// raw bit patterns, so nothing is lost to decimal. Each case is chosen to
    /// **fail** under a plausible wrong order, which is what makes the test say
    /// something: the `f64` cases of length 8 and 9 both reject a
    /// single-accumulator packet loop and the sequential left fold the port
    /// shipped before this, and the `f32` case of length 16 rejects those two
    /// *and* a `predux` that folds the four lanes left to right instead of
    /// pairing them `(a₀+a₂) + (a₁+a₃)`.
    #[test]
    fn the_contiguous_reduction_is_eigens_reduction() {
        // `alignedSize == 0`: one coefficient, nothing to vectorize.
        assert_contiguous::<f64>(
            &[f64::from_bits(0xbfe8_0d2e_9c86_ddda)],
            f64::from_bits(0x3fe2_13cb_58ed_5f5d),
        );
        // Four packets, no tail: the two accumulators and their fold.
        assert_contiguous::<f64>(
            &[
                f64::from_bits(0x3fdf_6dcc_c0d2_463c),
                f64::from_bits(0x3fe1_a542_f880_2596),
                f64::from_bits(0xbf92_ffaa_148d_aa00),
                f64::from_bits(0xbfdc_fbf2_788d_6bc4),
                f64::from_bits(0x3fc0_4612_22f0_f260),
                f64::from_bits(0xbfe7_d130_9531_90a8),
                f64::from_bits(0x3fed_74ea_52b4_3272),
                f64::from_bits(0x3fd8_2eff_ab90_0400),
            ],
            f64::from_bits(0x4002_7ccc_bea8_8ea9),
        );
        // Nine coefficients: four packets and a one-coefficient scalar tail.
        assert_contiguous::<f64>(
            &[
                f64::from_bits(0xbfd6_f147_67c1_07fc),
                f64::from_bits(0x3fe7_3f95_5182_2c26),
                f64::from_bits(0xbfdf_8a84_bbe8_45f4),
                f64::from_bits(0xbfc1_dadc_d961_a750),
                f64::from_bits(0xbfea_9027_5d35_44d2),
                f64::from_bits(0xbfd9_23aa_5c1d_6a98),
                f64::from_bits(0xbfdb_dd99_e6da_f05c),
                f64::from_bits(0x3fd2_a792_6e43_18e4),
                f64::from_bits(0xbfd8_ee16_95fb_94d0),
            ],
            f64::from_bits(0x4001_819b_c3ad_06dd),
        );
        // Four `Packet4f`s: the lane pairing decides this one.
        assert_contiguous::<f32>(
            &[
                f32::from_bits(0x3e85_2fc5),
                f32::from_bits(0x3ec7_c6df),
                f32::from_bits(0xbf4b_e21e),
                f32::from_bits(0x3ec1_4ea3),
                f32::from_bits(0x3f4d_2a52),
                f32::from_bits(0x3dc6_8a92),
                f32::from_bits(0x3d79_1b02),
                f32::from_bits(0x3ea9_b602),
                f32::from_bits(0x3f62_822b),
                f32::from_bits(0xbf0e_1bf2),
                f32::from_bits(0xbf5b_9c9a),
                f32::from_bits(0xbeed_9ec9),
                f32::from_bits(0xbea2_8f3d),
                f32::from_bits(0x3e72_a5ee),
                f32::from_bits(0x3f69_6d43),
                f32::from_bits(0xbf05_df74),
            ],
            f32::from_bits(0x40a2_1e1a),
        );
    }

    /// The whole shape sweep, when the fork's probe has been run.
    ///
    /// `SLAM_RS_MARG_NORM_SWEEP` points at the file
    /// `basalt_marg_norm_probe <file>` writes: one line per case, `scalar rows
    /// cols col start expected-bits value-bits…`, 7,486 cases per precision
    /// over rows 1..100. Unset — the default — this passes without checking
    /// anything, the way the optical-flow parity test treats its frame
    /// directory; the inline cases above cover the branches on every run.
    #[test]
    fn the_contiguous_reduction_reproduces_eigen_over_the_whole_sweep() {
        let Ok(path) = std::env::var("SLAM_RS_MARG_NORM_SWEEP") else {
            return;
        };
        let text: String = std::fs::read_to_string(&path).expect("the sweep file");
        let mut checked: usize = 0;
        for line in text.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert!(fields.len() > 6, "short line: {line}");
            let scalar: &str = fields[0];
            let expected: u64 = u64::from_str_radix(fields[5], 16).expect("hex");
            let values: Vec<u64> = fields[6..]
                .iter()
                .map(|f| u64::from_str_radix(f, 16).expect("hex"))
                .collect();
            match scalar {
                "f64" => {
                    let column: Vec<f64> = values.iter().map(|b| f64::from_bits(*b)).collect();
                    assert_contiguous::<f64>(&column, f64::from_bits(expected));
                }
                "f32" => {
                    let column: Vec<f32> = values
                        .iter()
                        .map(|b| f32::from_bits(u32::try_from(*b).expect("32 bits")))
                        .collect();
                    assert_contiguous::<f32>(
                        &column,
                        f32::from_bits(u32::try_from(expected).expect("32 bits")),
                    );
                }
                other => panic!("unknown scalar {other}"),
            }
            checked += 1;
        }
        assert!(checked > 10_000, "only {checked} cases in {path}");
    }
}
