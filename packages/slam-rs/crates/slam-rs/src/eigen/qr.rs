use nalgebra::{DMatrix, DVector};

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

/// Construct a reflection from a column segment using a plain squared norm.
pub(crate) fn make_householder<S: LieScalar>(
    storage: &DMatrix<S>,
    col: usize,
    start: usize,
    len: usize,
    essential: &mut [S],
) -> (S, S) {
    let tail_sq_norm: S = storage
        .column(col)
        .rows(start + 1, len.saturating_sub(1))
        .norm_squared();
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

    // `bottom` is `rows - 1` consecutive coefficients of each column, and a
    // `DMatrix` is column-major, so it is one contiguous run per column. Both
    // loops below walk it that way — through the slice rather than through
    // `storage[(i, j)]`, which recomputes `j * nrows + i` and bounds-checks it
    // per coefficient.
    let nrows: usize = storage.nrows();
    let tail: usize = rows - 1;
    let essential: &[S] = &essential[..tail.min(essential.len())];

    // `tmp.noalias() = essential.adjoint() * bottom` (`:113`): one dot product
    // per column, over the `rows - 1` rows below the first. The fold stays
    // sequential and stays in this order — it is the one reduction here.
    {
        let data: &[S] = storage.as_slice();
        for (j, slot) in work.iter_mut().enumerate().take(cols) {
            let base: usize = (col_start + j) * nrows + row_start + 1;
            let bottom: &[S] = &data[base..base + tail];
            let mut acc: S = S::zero();
            for (e, value) in essential.iter().zip(bottom.iter()) {
                acc += *e * *value;
            }
            *slot = acc;
        }
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
    //
    // Column outer, row inner. The product is elementwise — every coefficient
    // is one `-= (tau * e_i) * tmp_j` and no coefficient is summed twice — so
    // the loop order is free, and this one is the unit-stride one: `i` inner
    // walks a column's contiguous run, where `j` inner strided by `nrows` and
    // read a fresh cache line per coefficient. `tau * *e` is recomputed per
    // column rather than hoisted into a scratch vector; it is the same product
    // of the same two values, so the coefficient is unchanged.
    {
        let data: &mut [S] = storage.as_mut_slice();
        for (j, &value) in work.iter().enumerate().take(cols) {
            let base: usize = (col_start + j) * nrows + row_start + 1;
            let bottom: &mut [S] = &mut data[base..base + tail];
            for (target, e) in bottom.iter_mut().zip(essential.iter()) {
                *target -= (tau * *e) * value;
            }
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

/// A Jacobi rotation `(c, s)`, `Eigen::JacobiRotation`
/// (`Eigen/src/Jacobi/Jacobi.h:30-84`), real scalars only.
///
/// Only the Givens half of Eigen's type survives: [`make_givens`] builds one and
/// [`apply_rotation_on_the_left`] applies it, which is the rotation basalt's
/// landmark QR uses (`landmark_block_abs_dynamic.hpp:429-439`). The
/// `makeJacobi`/`transpose`/`operator*` half left with the hand-rolled 4x4
/// `JacobiSVD`, now that `crate::ba_base::triangulate` calls
/// [`nalgebra::linalg::SVD`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JacobiRotation<S: LieScalar> {
    /// The cosine.
    pub(crate) c: S,
    /// The sine.
    pub(crate) s: S,
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
                make_householder(&storage, 0, 0, len, &mut essential);
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
                make_householder(&storage, 0, 0, len, &mut essential);
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
}
