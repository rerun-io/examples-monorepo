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

/// Lanes in an SSE packet of `S`: four for `f32`, two for `f64`.
///
/// `unpacket_traits<Packet4f>::size` / `<Packet2d>::size`
/// (`arch/SSE/PacketMath.h:304-320`). Both packets are one 16-byte register,
/// and `unpacket_traits<Packet4f>::half` is `Packet4f` itself, so Eigen's
/// half- and quarter-packet paths are disabled for both scalars and the only
/// widths that occur are these.
pub(crate) fn packet_size<S>() -> usize {
    match size_of::<S>() {
        4 => 4,
        8 => 2,
        // No other `LieScalar` exists; one lane makes the kernels a plain
        // sequential fold rather than a panic (D32).
        _ => 1,
    }
}

/// The sub-block a kernel reads, `lhs.block(row0, col0, rows, cols)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Block {
    /// First row of the block in the enclosing matrix.
    pub row0: usize,
    /// First column of the block.
    pub col0: usize,
    /// Rows in the block, which is the length of `res`.
    pub rows: usize,
    /// Columns in the block, which is the length of `rhs`.
    pub cols: usize,
}

/// `internal::predux` on one SSE packet (`arch/SSE/Reductions.h:206-278`).
///
/// `Packet2d` is one `_mm_unpackhi_pd` and an add, so `a0 + a1`. `Packet4f`
/// is `movehl` then `movehdup`, which pairs the lanes across the halves:
/// `(a0 + a2) + (a1 + a3)`. A left fold over four lanes is a different value in
/// `f32`.
fn predux<S: LieScalar>(lanes: &[S]) -> S {
    match lanes {
        [a0, a1] => *a0 + *a1,
        [a0, a1, a2, a3] => (*a0 + *a2) + (*a1 + *a3),
        // Not reachable for `f32`/`f64`, the only `LieScalar` impls; a left
        // fold keeps the function total rather than panicking (D32).
        rest => rest.iter().copied().fold(S::zero(), |acc, x| acc + x),
    }
}

/// `redux_impl<Func, Evaluator, LinearVectorizedTraversal, NoUnrolling>`
/// (`Core/Redux.h:207-253`) over a contiguous, runtime-length expression.
///
/// This is what `.sum()` does to the `cwiseProduct` inside the row-major
/// triangular solve (`TriangularSolverVector.h:66-69`). The expression is a
/// `CwiseBinaryOp`, which carries no `DirectAccessBit`, so `first_aligned`
/// short-circuits to zero (`Core/util/Memory.h`, `first_aligned<Alignment>(const
/// DenseBase&)`) and the split below depends only on the length — not on where
/// the data happens to sit in memory.
pub(crate) fn redux_dynamic<S: LieScalar>(terms: &[S]) -> S {
    let size: usize = terms.len();
    let packet: usize = packet_size::<S>();
    let aligned_size: usize = (size / packet) * packet;
    let aligned_size2: usize = (size / (2 * packet)) * (2 * packet);

    if aligned_size == 0 {
        // "too small to vectorize anything" (`:246-252`): a left fold, and an
        // empty input is zero rather than a read of `coeff(0)`.
        return terms.iter().copied().fold(S::zero(), |acc, x| acc + x);
    }

    let mut res0: Vec<S> = terms[0..packet].to_vec();
    if aligned_size > packet {
        let mut res1: Vec<S> = terms[packet..2 * packet].to_vec();
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
    let mut res: S = predux(&res0);
    // `alignedStart` is zero, so only the trailing coefficients are left.
    for &term in &terms[aligned_size..] {
        res += term;
    }
    res
}

/// `res += alpha * lhs.block(row0, col0, rows, cols) * rhs`, with Eigen's
/// `ColMajor` association (`GeneralMatrixVector.h:105-258`).
///
/// `block_cols` is `cols` whenever `cols < 128` (`:143`), which every caller
/// here satisfies — the triangular panel is at most eight wide — so there is a
/// single column block and each output coefficient is one left fold from zero.
pub(crate) fn gemv_col_major_block<S: LieScalar>(
    lhs: &DMatrix<S>,
    block: Block,
    rhs: &[S],
    res: &mut [S],
    alpha: S,
) {
    let Block {
        row0,
        col0,
        rows,
        cols,
    } = block;
    let block_cols: usize = if cols < 128 { cols } else { 4 };
    let mut j2: usize = 0;
    while j2 < cols {
        let jend: usize = (j2 + block_cols).min(cols);
        for i in 0..rows {
            let mut acc: S = S::zero();
            for j in j2..jend {
                // `pcj.pmadd(lhs, b0, c)` without FMA: `a * b + c`.
                acc = lhs[(row0 + i, col0 + j)] * rhs[j] + acc;
            }
            if let Some(slot) = res.get_mut(i) {
                *slot += alpha * acc;
            }
        }
        j2 = jend;
    }
}

/// `res += alpha * lhsᵀ.block(row0, col0, rows, cols) * rhs` read as a
/// `RowMajor` product (`GeneralMatrixVector.h:298-450`).
///
/// The logical coefficient `(i, j)` is `lhs[(col0 + j, row0 + i)]`: this is the
/// shape `matrixL().adjoint()` hands the solver, a triangular view over a
/// `Transpose` of a column-major matrix, which Eigen therefore dispatches to
/// the row-major kernel.
pub(crate) fn gemv_row_major_of_transpose<S: LieScalar>(
    lhs: &DMatrix<S>,
    block: Block,
    rhs: &[S],
    res: &mut [S],
    alpha: S,
) {
    let Block {
        row0,
        col0,
        rows,
        cols,
    } = block;
    let packet: usize = packet_size::<S>();
    let full_col_block_end: usize = packet * (cols / packet);
    let mut lanes: Vec<S> = vec![S::zero(); packet];
    for i in 0..rows {
        lanes.fill(S::zero());
        let mut j: usize = 0;
        while j < full_col_block_end {
            for lane in 0..packet {
                lanes[lane] = lhs[(col0 + j + lane, row0 + i)] * rhs[j + lane] + lanes[lane];
            }
            j += packet;
        }
        let mut acc: S = predux(&lanes);
        for j in full_col_block_end..cols {
            acc += lhs[(col0 + j, row0 + i)] * rhs[j];
        }
        if let Some(slot) = res.get_mut(i) {
            *slot += alpha * acc;
        }
    }
}

/// `numext::maxi(a, b)` (`Core/MathFunctions.h`), which is what `cwiseMax`
/// applies coefficient by coefficient.
///
/// `(a < b ? b : a)`, so a NaN on the left survives and `f32::max`'s
/// NaN-suppressing behaviour is wrong here. `sqrt_keypoint_vio.cpp:1415` sends
/// the result straight into the damped diagonal, so a NaN that Eigen keeps and
/// Rust would drop changes whether the solve retries.
pub(crate) fn eigen_maxi<S: LieScalar>(a: S, b: S) -> S {
    if a < b { b } else { a }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nalgebra::DMatrix;
    use proptest::prelude::*;

    #[test]
    fn the_packet_widths_are_the_forks() {
        assert_eq!(packet_size::<f32>(), 4);
        assert_eq!(packet_size::<f64>(), 2);
    }

    /// The two `predux` trees, spelled out on values whose sum is
    /// order-dependent in `f32`: `1 + 2^-24` rounds away against `1` but
    /// survives against `2^-24`.
    #[test]
    fn predux_pairs_the_lanes_across_the_halves() {
        let tiny: f32 = f32::EPSILON / 2.0;
        // (1 + 1) + (tiny + tiny) keeps both tiny terms; a left fold
        // ((1 + 1) + tiny) + tiny loses them.
        assert_eq!(predux(&[1.0f32, tiny, 1.0f32, tiny]), 2.0 + (tiny + tiny));
        assert_eq!(
            [1.0f32, tiny, 1.0f32, tiny]
                .iter()
                .copied()
                .fold(0.0f32, |a, b| a + b),
            2.0
        );
        assert_eq!(predux(&[1.0f64, 2.0f64]), 3.0);
    }

    /// The `f64` splits of `redux_dynamic`, length by length, against the trees
    /// derived from `Redux.h:207-253` by hand.
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
            Block {
                row0: 0,
                col0: 0,
                rows: 1,
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
            gemv_col_major_block(&lhs, Block { row0: 0, col0: 0, rows: 5, cols: 7 }, &rhs, &mut res, 1.0);
            for i in 0..5 {
                prop_assert!((res[i] - expected[i]).abs() <= 1e-12 * (1.0 + expected[i].abs()));
            }

            let transposed: DMatrix<f64> = lhs.transpose();
            let mut res2: Vec<f64> = vec![0.0; 5];
            gemv_row_major_of_transpose(&transposed, Block { row0: 0, col0: 0, rows: 5, cols: 7 }, &rhs, &mut res2, 1.0);
            for i in 0..5 {
                prop_assert!((res2[i] - expected[i]).abs() <= 1e-12 * (1.0 + expected[i].abs()));
            }
        }
    }

    #[test]
    fn eigen_maxi_keeps_a_nan_on_the_left() {
        assert!(eigen_maxi(f64::NAN, 1.0).is_nan());
        assert_eq!(eigen_maxi(1.0f64, f64::NAN), 1.0);
        assert_eq!(f64::NAN.max(1.0), 1.0, "std::f64::max is the other way");
    }
}
