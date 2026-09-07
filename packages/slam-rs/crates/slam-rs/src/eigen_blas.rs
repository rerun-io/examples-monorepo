//! Eigen's dense matrix-vector kernels, ported at the arithmetic level.
//!
//! Stage S6 accepted the association inside Eigen's `gemv`/`gemm` as
//! non-exact, because nothing it fed reached a threshold comparison (D50). The
//! sliding-window driver cannot: `Eigen::LDLT::solve` is the only place the LM
//! increment is produced, the increment reaches `inc.array().isFinite().all()`
//! and `step_norminf < 1e-4` (`sqrt_keypoint_vio.cpp:1424`, `:1477`), and every
//! pose in the window is moved by it. So the two kernels the triangular solvers
//! call are ported here coefficient for coefficient.
//!
//! Both are simpler than they look, because the vectorised and the scalar
//! lanes of Eigen's kernels do the *same* arithmetic per output coefficient —
//! the packet blocking only decides which coefficients are computed together,
//! never how a single one is summed. What does differ per precision is the
//! packet width, because it decides how a dot product is split before the
//! horizontal add:
//!
//! * `ColMajor` (`GeneralMatrixVector.h:105-258`) accumulates each output
//!   coefficient from a **fresh zero** over the whole column block and adds
//!   `alpha *` that into the result once. This is not the same as the naive
//!   `for j { for i { res[i] -= lhs(i, j) * rhs[j] } }`, which accumulates into
//!   `res` and rounds every step.
//! * `RowMajor` (`:298-450`) accumulates one packet per output row over
//!   `cols / PacketSize` full blocks, folds the lanes with `predux`, then adds
//!   the scalar tail.
//!
//! The packet widths are the fork's, not this machine's: the build appends
//! conda's `-march=nocona` after `-march=native` and the last wins, so the
//! reference binary is SSE3 — `Packet4f` and `Packet2d`, and no FMA, which is
//! why `pmadd(a, b, c)` below is `a * b + c` with two roundings rather than one
//! (D47 established the same fact for the three-coefficient reductions).

use nalgebra::DMatrix;

use crate::lie::LieScalar;
use crate::linearize::eigen_qr::BlockSpan;

/// `redux_impl<Func, Evaluator, LinearVectorizedTraversal, NoUnrolling>`
/// (`Core/Redux.h:274-325`) over a contiguous, runtime-length expression.
///
/// This is what `.sum()` does to the `cwiseProduct` inside the row-major
/// triangular solve (`TriangularSolverVector.h:66-69`). The expression is a
/// `CwiseBinaryOp`, which carries no `DirectAccessBit`, so `first_aligned`
/// short-circuits to zero (`Core/util/Memory.h`, `first_aligned<Alignment>(const
/// DenseBase&)`) and the split below depends only on the length — not on where
/// the data happens to sit in memory.
pub(crate) fn redux_dynamic<S: LieScalar>(terms: &[S]) -> S {
    let size: usize = terms.len();
    let packet: usize = S::EIGEN_PACKET_SIZE;
    let aligned_size: usize = (size / packet) * packet;
    let aligned_size2: usize = (size / (2 * packet)) * (2 * packet);

    if aligned_size == 0 {
        // "too small to vectorize anything" (`:317-322`): a left fold, and an
        // empty input is zero rather than a read of `coeff(0)`.
        return terms.iter().copied().fold(S::zero(), |acc, x| acc + x);
    }

    // Four lanes because that is the widest packet either scalar has; only the
    // first `packet` of them are read, as in
    // [`crate::linearize::eigen_qr::contiguous_squared_norm`].
    let mut res0: [S; 4] = [S::zero(); 4];
    res0[..packet].copy_from_slice(&terms[..packet]);
    if aligned_size > packet {
        let mut res1: [S; 4] = [S::zero(); 4];
        res1[..packet].copy_from_slice(&terms[packet..2 * packet]);
        let mut index: usize = 2 * packet;
        while index < aligned_size2 {
            for lane in 0..packet {
                res0[lane] += terms[index + lane];
                res1[lane] += terms[index + packet + lane];
            }
            index += 2 * packet;
        }
        for lane in 0..packet {
            res0[lane] += res1[lane];
        }
        if aligned_size > aligned_size2 {
            for lane in 0..packet {
                res0[lane] += terms[aligned_size2 + lane];
            }
        }
    }
    let mut res: S = S::eigen_predux(&res0[..packet]);
    // `alignedStart` is zero, so only the trailing coefficients are left.
    for &term in &terms[aligned_size..] {
        res += term;
    }
    res
}

/// `res += alpha * lhs.block(row0, col0, rows, cols) * rhs`, with Eigen's
/// `ColMajor` association (`GeneralMatrixVector.h:105-258`).
///
/// **Contract: `cols < 128`**, so `GeneralMatrixVector.h:143`'s
/// `cols < 128 ? cols : ...` makes the column block the whole width — one
/// block, each output coefficient one left fold from zero. Every caller here is
/// a triangular panel, at most eight wide. Also `rhs.len() == span.cols` and
/// `res.len() == span.rows`, the shape the span itself names: dropping a
/// coefficient of a shorter `res` would return a partially updated LM
/// increment instead of failing.
pub(crate) fn gemv_col_major_block<S: LieScalar>(
    lhs: &DMatrix<S>,
    span: BlockSpan,
    rhs: &[S],
    res: &mut [S],
    alpha: S,
) {
    let BlockSpan {
        row_start,
        rows,
        col_start,
        cols,
    } = span;
    debug_assert!(cols < 128, "GeneralMatrixVector.h:143 blocks a wider gemv");
    debug_assert_eq!(rhs.len(), cols);
    debug_assert_eq!(res.len(), rows);
    for i in 0..rows {
        let mut acc: S = S::zero();
        for j in 0..cols {
            // `pcj.pmadd(lhs, b0, c)` without FMA: `a * b + c`.
            acc = lhs[(row_start + i, col_start + j)] * rhs[j] + acc;
        }
        res[i] += alpha * acc;
    }
}

/// `res += alpha * lhsᵀ.block(row0, col0, rows, cols) * rhs` read as a
/// `RowMajor` product (`GeneralMatrixVector.h:298-450`).
///
/// The logical coefficient `(i, j)` is `lhs[(col_start + j, row_start + i)]`:
/// this is the shape `matrixL().adjoint()` hands the solver, a triangular view
/// over a `Transpose` of a column-major matrix, which Eigen therefore
/// dispatches to the row-major kernel.
///
/// **Contract: `rhs.len() == span.cols` and `res.len() == span.rows`**, as in
/// [`gemv_col_major_block`].
pub(crate) fn gemv_row_major_of_transpose<S: LieScalar>(
    lhs: &DMatrix<S>,
    span: BlockSpan,
    rhs: &[S],
    res: &mut [S],
    alpha: S,
) {
    let BlockSpan {
        row_start,
        rows,
        col_start,
        cols,
    } = span;
    debug_assert_eq!(rhs.len(), cols);
    debug_assert_eq!(res.len(), rows);
    let packet: usize = S::EIGEN_PACKET_SIZE;
    let full_col_block_end: usize = packet * (cols / packet);
    for i in 0..rows {
        let mut lanes: [S; 4] = [S::zero(); 4];
        let mut j: usize = 0;
        while j < full_col_block_end {
            for lane in 0..packet {
                lanes[lane] =
                    lhs[(col_start + j + lane, row_start + i)] * rhs[j + lane] + lanes[lane];
            }
            j += packet;
        }
        let mut acc: S = S::eigen_predux(&lanes[..packet]);
        for j in full_col_block_end..cols {
            acc += lhs[(col_start + j, row_start + i)] * rhs[j];
        }
        res[i] += alpha * acc;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nalgebra::DMatrix;
    use proptest::prelude::*;

    /// The `f64` splits of `redux_dynamic`, length by length, against the trees
    /// derived from `Redux.h:274-325` by hand.
    #[test]
    fn redux_dynamic_splits_as_eigen_does() {
        let t: Vec<f64> = (1..=7).map(|i| 1.0 / f64::from(i)).collect();
        assert_eq!(redux_dynamic(&t[0..1]), t[0]);
        assert_eq!(redux_dynamic(&t[0..2]), t[0] + t[1]);
        assert_eq!(redux_dynamic(&t[0..3]), (t[0] + t[1]) + t[2]);
        assert_eq!(
            redux_dynamic(&t[0..4]),
            (t[0] + t[2]) + (t[1] + t[3]),
            "two packets fold lane-wise before predux"
        );
        assert_eq!(
            redux_dynamic(&t[0..5]),
            ((t[0] + t[2]) + (t[1] + t[3])) + t[4]
        );
        assert_eq!(
            redux_dynamic(&t[0..6]),
            ((t[0] + t[2]) + t[4]) + ((t[1] + t[3]) + t[5])
        );
        assert_eq!(
            redux_dynamic(&t[0..7]),
            (((t[0] + t[2]) + t[4]) + ((t[1] + t[3]) + t[5])) + t[6]
        );
        assert_eq!(redux_dynamic::<f64>(&[]), 0.0);
    }

    /// The `f32` split: nothing below four coefficients vectorises, and four or
    /// more fold one packet then a scalar tail.
    #[test]
    fn redux_dynamic_needs_four_coefficients_in_f32() {
        let t: Vec<f32> = (1..=6).map(|i| 1.0 / (i as f32)).collect();
        assert_eq!(redux_dynamic(&t[0..3]), (t[0] + t[1]) + t[2]);
        assert_eq!(redux_dynamic(&t[0..4]), (t[0] + t[2]) + (t[1] + t[3]));
        assert_eq!(
            redux_dynamic(&t[0..5]),
            ((t[0] + t[2]) + (t[1] + t[3])) + t[4]
        );
    }

    /// The column-major kernel accumulates from zero, which a naive
    /// `res[i] -= lhs * rhs[j]` loop does not: on this input the two disagree
    /// in `f32`.
    #[test]
    fn the_column_major_kernel_accumulates_from_zero() {
        let tiny: f32 = f32::EPSILON / 2.0;
        let lhs: DMatrix<f32> = DMatrix::from_row_slice(1, 2, &[tiny, tiny]);
        let rhs: [f32; 2] = [1.0, 1.0];
        let mut res: [f32; 1] = [1.0];
        gemv_col_major_block(
            &lhs,
            BlockSpan {
                row_start: 0,
                rows: 1,
                col_start: 0,
                cols: 2,
            },
            &rhs,
            &mut res,
            1.0,
        );
        assert_eq!(res[0], 1.0 + (tiny + tiny), "one rounding into res");

        let mut naive: f32 = 1.0;
        naive += tiny;
        naive += tiny;
        assert_eq!(naive, 1.0, "the naive loop loses both");
    }

    proptest! {
        /// Both kernels compute the mathematical product; the tests above pin
        /// the association, this pins the value.
        #[test]
        fn the_kernels_agree_with_the_dense_product(
            values in prop::collection::vec(-4.0f64..4.0, 5 * 7),
            rhs in prop::collection::vec(-4.0f64..4.0, 7),
        ) {
            let lhs: DMatrix<f64> = DMatrix::from_row_slice(5, 7, &values);
            let expected: Vec<f64> = (0..5)
                .map(|i| (0..7).map(|j| lhs[(i, j)] * rhs[j]).sum::<f64>())
                .collect();

            let mut res: Vec<f64> = vec![0.0; 5];
            gemv_col_major_block(&lhs, BlockSpan { row_start: 0, rows: 5, col_start: 0, cols: 7 }, &rhs, &mut res, 1.0);
            for i in 0..5 {
                prop_assert!((res[i] - expected[i]).abs() <= 1e-12 * (1.0 + expected[i].abs()));
            }

            let transposed: DMatrix<f64> = lhs.transpose();
            let mut res2: Vec<f64> = vec![0.0; 5];
            gemv_row_major_of_transpose(&transposed, BlockSpan { row_start: 0, rows: 5, col_start: 0, cols: 7 }, &rhs, &mut res2, 1.0);
            for i in 0..5 {
                prop_assert!((res2[i] - expected[i]).abs() <= 1e-12 * (1.0 + expected[i].abs()));
            }
        }
    }
}
