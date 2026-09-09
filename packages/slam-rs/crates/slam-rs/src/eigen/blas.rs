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

use crate::lie::LieScalar;

/// `v.head<3>().norm()` in Eigen's summation order, which differs between the
/// two precisions.
///
/// See [`crate::lie::LieScalar::eigen_redux3`]: `f64` reduces one `Packet2d` and folds the
/// remainder in, `(a + b) + c`; `f32` finds `Packet4f` too wide and falls back
/// to the scalar unroller's `a + (b + c)`. Using one order for both is a
/// one-ulp error, and it reaches a threshold in each precision — the `f32`
/// `proj[2]` of a landmark at `inv_dist = 1e-7`, and the `f64` acceptance gate
/// on a landmark exactly 1/3 m away (decision D44's rule).
#[inline]
pub fn norm3<S: LieScalar>(x: S, y: S, z: S) -> S {
    S::eigen_redux3(x * x, y * y, z * z).sqrt()
}

/// `redux_impl<Func, Evaluator, LinearVectorizedTraversal, NoUnrolling>::run`
/// (`Core/Redux.h:274-325`): the sum of `len` contiguous coefficients in
/// Eigen's order, with `term(i)` supplying coefficient `i`.
///
/// Two of basalt's reductions come through here — the `.sum()` of a
/// `cwiseProduct` (an `InnerProduct` product, or the row-major triangular
/// solve's `TriangularSolverVector.h:66-69`) and the `squaredNorm()` of a
/// contiguous column segment ([`crate::eigen::qr::contiguous_squared_norm`]).
/// They differ only in where the coefficients come from, which is why this
/// takes a closure: the traversal, the two accumulators and the scalar tail are
/// Eigen's and must be written once.
///
/// **`alignedStart` is always zero, and that is a property of the expression,
/// not of the address.** `Redux.h:290` calls
/// `internal::first_default_aligned(xpr)`; every expression that reaches this
/// port is a `CwiseUnaryOp` or `CwiseBinaryOp`, which carry no
/// `DirectAccessBit`, so `DenseCoeffsBase.h:533`'s `ReturnZero` is true and the
/// head split never happens. The fork's `tools/marg_norm_probe.cpp` prints
/// `first_default_aligned(unaryExpr) = 0` from Eigen itself.
///
/// An empty reduction is `Scalar(0)`: `DenseBase::sum()` returns without
/// reducing at all (`Redux.h:489`), which is what makes an empty column and an
/// empty tail well defined rather than a read of `coeff(0)`.
pub(crate) fn redux_contiguous<S: LieScalar>(len: usize, term: impl Fn(usize) -> S) -> S {
    if len == 0 {
        return S::zero();
    }
    let packet: usize = S::EIGEN_PACKET_SIZE;
    // `:291-292` with `alignedStart == 0`.
    let aligned_size2: usize = (len / (2 * packet)) * (2 * packet);
    let aligned_size: usize = (len / packet) * packet;

    if aligned_size == 0 {
        // `:317-322`: "too small to vectorize anything" — `coeff(0)`, then a
        // left fold over the rest.
        let mut res: S = term(0);
        for i in 1..len {
            res += term(i);
        }
        return res;
    }

    // `:296`. Four lanes because that is the widest packet either scalar has;
    // only the first `packet` of them are read.
    let mut packet0: [S; 4] = [S::zero(); 4];
    for (lane, slot) in packet0.iter_mut().enumerate().take(packet) {
        *slot = term(lane);
    }
    if aligned_size > packet {
        // `:297-307`: two accumulators, one for the even packets and one for
        // the odd, folded together at the end.
        let mut packet1: [S; 4] = [S::zero(); 4];
        for (lane, slot) in packet1.iter_mut().enumerate().take(packet) {
            *slot = term(packet + lane);
        }
        let mut index: usize = 2 * packet;
        while index < aligned_size2 {
            for (lane, slot) in packet0.iter_mut().enumerate().take(packet) {
                *slot += term(index + lane);
            }
            for (lane, slot) in packet1.iter_mut().enumerate().take(packet) {
                *slot += term(index + packet + lane);
            }
            index += 2 * packet;
        }
        // `:306`.
        for (lane, slot) in packet0.iter_mut().enumerate().take(packet) {
            *slot += packet1[lane];
        }
        // `:307-308`: one odd packet left over.
        if aligned_size > aligned_size2 {
            for (lane, slot) in packet0.iter_mut().enumerate().take(packet) {
                *slot += term(aligned_size2 + lane);
            }
        }
    }
    // `:310`, then `:314` — the head loop of `:312` is empty because
    // `alignedStart` is zero.
    let mut res: S = S::eigen_predux(&packet0[..packet]);
    for i in aligned_size..len {
        res += term(i);
    }
    res
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// The `f64` splits, length by length, against the trees derived from
    /// `Redux.h:274-325` by hand.
    #[test]
    fn the_reduction_splits_as_eigen_does() {
        let t: Vec<f64> = (1..=7).map(|i| 1.0 / f64::from(i)).collect();
        let sum = |len: usize| -> f64 { redux_contiguous(len, |i| t[i]) };
        assert_eq!(sum(1), t[0]);
        assert_eq!(sum(2), t[0] + t[1]);
        assert_eq!(sum(3), (t[0] + t[1]) + t[2]);
        assert_eq!(
            sum(4),
            (t[0] + t[2]) + (t[1] + t[3]),
            "two packets fold lane-wise before predux"
        );
        assert_eq!(sum(5), ((t[0] + t[2]) + (t[1] + t[3])) + t[4]);
        assert_eq!(sum(6), ((t[0] + t[2]) + t[4]) + ((t[1] + t[3]) + t[5]));
        assert_eq!(
            sum(7),
            (((t[0] + t[2]) + t[4]) + ((t[1] + t[3]) + t[5])) + t[6]
        );
        assert_eq!(redux_contiguous::<f64>(0, |_| 1.0), 0.0);
    }

    /// The `f32` split: nothing below four coefficients vectorises, and four or
    /// more fold one packet then a scalar tail.
    #[test]
    fn the_reduction_needs_four_coefficients_in_f32() {
        let t: Vec<f32> = (1..=6).map(|i| 1.0 / (i as f32)).collect();
        let sum = |len: usize| -> f32 { redux_contiguous(len, |i| t[i]) };
        assert_eq!(sum(3), (t[0] + t[1]) + t[2]);
        assert_eq!(sum(4), (t[0] + t[2]) + (t[1] + t[3]));
        assert_eq!(sum(5), ((t[0] + t[2]) + (t[1] + t[3])) + t[4]);
    }
}
