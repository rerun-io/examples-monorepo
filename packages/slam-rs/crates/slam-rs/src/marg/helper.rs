//! `MargHelper<Scalar>` (`include/basalt/vi_estimator/marg_helper.h`,
//! `src/vi_estimator/marg_helper.cpp`): the three routines that turn a linear
//! system over `keep ∪ marg` into an equivalent one over `keep` alone.
//!
//! | routine | input | output | C++ |
//! |---|---|---|---|
//! | [`marginalize_helper_sqrt_to_sqrt`] | `Q₂J_p`, `Q₂r` | `J_m`, `r_m` | `marg_helper.cpp:247-327` |
//! | [`marginalize_helper_sq_to_sqrt`] | `H`, `b` | `J_m`, `r_m` | `:120-244` |
//! | [`marginalize_helper_sq_to_sq`] | `H`, `b` | `H_m`, `b_m` | `:42-117` |
//!
//! Only the first is on the shipped path (decision D13): `vio_sqrt_marg` is on
//! by default and the linearization is `ABS_QR`, so `marginalize()` takes the
//! `is_lin_sqrt && marg_data.is_sqrt` branch (`sqrt_keypoint_vio.cpp:1071-1073`).
//! The third is reachable with `vio_sqrt_marg = false`; the second needs a
//! square linearization *and* a square-root prior, which no shipped
//! configuration selects, and is here because `test_qr.cpp`'s
//! `RankDefLeastSquares` drives it.
//!
//! **All three consume their input.** C++ takes `MatX&` and ends with
//! `abs_H.resize(0, 0)` (`:115-116`, `:242-243`, `:325-326`); the port takes the
//! matrices **by value**, which says the same thing in a way the compiler
//! checks.
//!
//! **The two permutation conventions are not the same, and the difference is
//! the whole design.** `sqrt_to_sqrt` orders the columns **marg first**
//! (`:259-273`) so that a flat QR sweeping left to right eliminates the
//! marginalized variables before it reaches the kept ones, and the rows below
//! the marginalized rank are exactly the prior. The two square forms order
//! **keep first** (`:52-66`, `:137-151`) because they take a Schur complement,
//! which wants the block to invert in the bottom-right corner.

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector};

use crate::lie::LieScalar;
use crate::linearize::eigen_qr::{
    BlockSpan, ColumnRedux, apply_householder_on_the_left_block, apply_householder_on_the_left_vec,
    make_householder,
};
use crate::marg::MargError;
use crate::marg::eigen_cod::Cod;
use crate::marg::eigen_ldlt::EigenLdlt;

/// What a marginalization helper returns: the reduced system over the kept
/// variables, in whichever form the routine produces.
#[derive(Debug, Clone, PartialEq)]
pub struct ReducedSystem<S: LieScalar> {
    /// `marg_sqrt_H` (`J_m`) or `marg_H`, depending on the routine.
    pub h: DMatrix<S>,
    /// `marg_sqrt_b` (`r_m`) or `marg_b`.
    pub b: DVector<S>,
}

/// Validate the index sets against a system of `total` columns.
///
/// C++ has one assertion, `keep_size + marg_size == abs_H.cols()` (`:47`), and
/// trusts the caller for the rest; an index out of range or shared between the
/// two sets is an out-of-bounds read there. The port checks all three
/// (decision D32).
fn check_indices(
    idx_to_keep: &BTreeSet<usize>,
    idx_to_marg: &BTreeSet<usize>,
    total: usize,
) -> Result<(usize, usize), MargError> {
    let keep_size: usize = idx_to_keep.len();
    let marg_size: usize = idx_to_marg.len();
    if keep_size + marg_size != total {
        return Err(MargError::IndexCountMismatch {
            keep: keep_size,
            marg: marg_size,
            total,
        });
    }
    for index in idx_to_keep.iter().chain(idx_to_marg.iter()) {
        if *index >= total {
            return Err(MargError::IndexOutOfRange {
                index: *index,
                total,
            });
        }
    }
    if let Some(index) = idx_to_keep.intersection(idx_to_marg).next() {
        return Err(MargError::IndexInBothSets { index: *index });
    }
    Ok((keep_size, marg_size))
}

/// `MargHelper::marginalizeHelperSqrtToSqrt` (`marg_helper.cpp:247-327`).
///
/// A rank-revealing Householder QR over the whole stacked `[J_marg | J_keep]`,
/// swept column by column. Every column that produces a reflector with
/// `|beta| > sqrt(epsilon)` consumes one row of rank; a column that does not is
/// zeroed and the rank does not advance. `marg_rank` is the rank reached when
/// the last marginalized column is done (`:316`), and the prior is the block of
/// rows `[marg_rank, total_rank)` against the kept columns (`:322-323`) — the
/// part of the residual the marginalized variables can no longer explain.
///
/// **The rank threshold is `sqrt(numeric_limits<Scalar>::epsilon())` on the raw
/// `beta`** (`:284`, `:301`), not a relative test against the largest pivot.
/// It is absolute, so it depends on the units the problem is scaled in — that
/// is basalt's choice, and reproducing the C++'s decision on a column that
/// lands on it is what makes two runs agree.
///
/// The Householder vector is written into the column and read back from it in
/// C++ (`makeHouseholderInPlace`, then `Q2Jp.col(k).tail(...)` as the essential
/// part, `:299-306`); the port keeps it in a scratch vector, which holds the
/// same coefficients, and zeroes the column exactly where `:313` does.
pub fn marginalize_helper_sqrt_to_sqrt<S: LieScalar>(
    mut q2jp: DMatrix<S>,
    mut q2r: DVector<S>,
    idx_to_keep: &BTreeSet<usize>,
    idx_to_marg: &BTreeSet<usize>,
) -> Result<ReducedSystem<S>, MargError> {
    let rows: usize = q2jp.nrows();
    let cols: usize = q2jp.ncols();
    let (keep_size, marg_size) = check_indices(idx_to_keep, idx_to_marg, cols)?;
    // `:254`.
    if q2r.nrows() != rows {
        return Err(MargError::RhsLengthMismatch {
            rows,
            rhs: q2r.nrows(),
        });
    }

    // `:256-278`: **marg first**, then keep, and `Q2Jp.applyOnTheRight(p)`,
    // which is `new.col(i) = old.col(indices[i])`
    // (`ProductEvaluators.h:1150-1160`, `Side == OnTheRight`).
    let indices: Vec<usize> = idx_to_marg
        .iter()
        .chain(idx_to_keep.iter())
        .copied()
        .collect();
    let permuted: DMatrix<S> = DMatrix::from_fn(rows, cols, |i, j| q2jp[(i, indices[j])]);
    q2jp = permuted;

    // `:280-318`.
    let rank_threshold: S = S::default_epsilon().sqrt();
    let mut marg_rank: usize = 0;
    let mut total_rank: usize = 0;
    // `tempVector.resize(cols + 1)` (`:290`): one scratch for the block and one
    // slot past the end for the right-hand side.
    let mut temp: Vec<S> = vec![S::zero(); cols + 1];
    let mut essential: Vec<S> = vec![S::zero(); rows.saturating_sub(1)];

    for k in 0..cols {
        if total_rank >= rows {
            break;
        }
        let base: usize = total_rank;
        let remaining_rows: usize = rows - base;
        let remaining_cols: usize = cols - k - 1;

        // `:299`, `makeHouseholderInPlace` on `Q2Jp.col(k).tail(remainingRows)`.
        //
        // `Q2Jp` is `MatX`, i.e. column-major, so that segment is contiguous and
        // its `squaredNorm()` takes Eigen's vectorised reduction — not the
        // sequential fold a row-major landmark column takes. The difference
        // reaches the `|beta| > sqrt(epsilon)` test three lines down: on the
        // review's `9x2` problem the sequential fold accepts a column `f32`
        // Eigen rejects and rejects one `f64` Eigen accepts.
        let (h_coeff, beta) = make_householder(
            &q2jp,
            k,
            base,
            remaining_rows,
            ColumnRedux::Contiguous,
            &mut essential,
        );

        if beta.abs() > rank_threshold {
            // `:302`.
            q2jp[(base, k)] = beta;
            // `:304-305`: the reflection on `bottomRightCorner(remainingRows,
            // remainingCols)`, which starts at row `base`, column `k + 1`.
            apply_householder_on_the_left_block(
                &mut q2jp,
                BlockSpan {
                    row_start: base,
                    rows: remaining_rows,
                    col_start: k + 1,
                    cols: remaining_cols,
                },
                &essential[..remaining_rows.saturating_sub(1)],
                h_coeff,
                &mut temp[k + 1..],
            );
            // `:306`: the same reflection on the residual, in lockstep.
            apply_householder_on_the_left_vec(
                &mut q2r,
                base,
                remaining_rows,
                &essential[..remaining_rows.saturating_sub(1)],
                h_coeff,
            );
            total_rank += 1;
        } else {
            // `:309`.
            q2jp[(base, k)] = S::zero();
        }

        // `:313`: overwrite the Householder vector with zeros. `remainingRows`
        // is the value from **before** the rank advanced.
        for i in (base + 1)..rows {
            q2jp[(i, k)] = S::zero();
        }

        // `:316`: `k == marg_size - 1` on a signed index, so with nothing to
        // marginalize C++ compares against `-1` and never fires.
        if k + 1 == marg_size {
            marg_rank = total_rank;
        }
    }

    // `:320-323`.
    let keep_valid_rows: usize = (total_rank - marg_rank).max(1);
    let mut h: DMatrix<S> = DMatrix::zeros(keep_valid_rows, keep_size);
    let mut b: DVector<S> = DVector::zeros(keep_valid_rows);
    for i in 0..keep_valid_rows {
        let row: usize = marg_rank + i;
        // `max(total_rank - marg_rank, 1)` can ask for a row past the end when
        // the marginalized part alone exhausted the rank; C++ reads out of
        // range there, the port leaves the row zero — which is the answer the
        // arithmetic gives: nothing is left to constrain the kept variables.
        if row >= rows {
            break;
        }
        for j in 0..keep_size {
            h[(i, j)] = q2jp[(row, marg_size + j)];
        }
        b[i] = q2r[row];
    }

    Ok(ReducedSystem { h, b })
}

/// `MargHelper::marginalizeHelperSqToSq` (`marg_helper.cpp:42-117`).
///
/// The plain Schur complement, with the marginalized block inverted through
/// Eigen's complete orthogonal decomposition (`:99-100`) — see
/// `crate::marg::eigen_cod` for why that specific decomposition is ported
/// rather than substituted.
///
/// `abs_H` is permuted **keep first** and then overwritten in place: the
/// top-right corner becomes `H_km H_mm⁺` (`:103`), and the reduced system is
/// `H_kk − (H_km H_mm⁺) H_mk`, `b_k − (H_km H_mm⁺) b_m` (`:112-113`).
pub fn marginalize_helper_sq_to_sq<S: LieScalar>(
    abs_h: DMatrix<S>,
    abs_b: DVector<S>,
    idx_to_keep: &BTreeSet<usize>,
    idx_to_marg: &BTreeSet<usize>,
) -> Result<ReducedSystem<S>, MargError> {
    let (h, b) = schur_complement(abs_h, abs_b, idx_to_keep, idx_to_marg)?;
    Ok(ReducedSystem { h, b })
}

/// `MargHelper::marginalizeHelperSqToSqrt` (`marg_helper.cpp:120-244`).
///
/// The same Schur complement as [`marginalize_helper_sq_to_sq`], followed by a
/// square root of the reduced Hessian through Eigen's pivoted LDLT (`:202-231`):
///
/// ```text
/// marg_H  = Pᵀ L D Lᵀ P          so   J_m = sqrt(D) Lᵀ P     (:212-215)
/// marg_b  = J_mᵀ r_m             so   r_m = sqrt(D)⁻¹ L⁻¹ P marg_b   (:223-231)
/// ```
///
/// `sqrt(D)` clamps negative pivots to zero (`:204`), and a root at or below
/// `sqrt(numeric_limits::min())` zeroes the matching entry of `r_m` instead of
/// dividing by it (`:229-230`) — the rank-deficient case, and the reason
/// decision D41 insists on Eigen's own LDLT here.
pub fn marginalize_helper_sq_to_sqrt<S: LieScalar>(
    abs_h: DMatrix<S>,
    abs_b: DVector<S>,
    idx_to_keep: &BTreeSet<usize>,
    idx_to_marg: &BTreeSet<usize>,
) -> Result<ReducedSystem<S>, MargError> {
    let (marg_h, marg_b) = schur_complement(abs_h, abs_b, idx_to_keep, idx_to_marg)?;
    let keep_size: usize = marg_h.nrows();

    // `:202`.
    let ldlt: EigenLdlt<S> = EigenLdlt::new(marg_h);

    // `:204`: `vectorD().array().max(0).sqrt()`.
    let vector_d: DVector<S> = ldlt.vector_d();
    let d_sqrt: DVector<S> =
        DVector::from_iterator(keep_size, vector_d.iter().map(|d| d.max(S::zero()).sqrt()));

    // `:212-215`: `sqrt(D) · U · P`, built right to left.
    let mut h: DMatrix<S> = DMatrix::identity(keep_size, keep_size);
    ldlt.apply_transpositions_left(&mut h);
    h = ldlt.matrix_u_times(&h);
    for i in 0..keep_size {
        for j in 0..keep_size {
            h[(i, j)] = d_sqrt[i] * h[(i, j)];
        }
    }

    // `:223-224`.
    let mut b: DVector<S> = marg_b;
    ldlt.apply_transpositions_left_vec(&mut b);
    ldlt.solve_unit_lower_in_place(&mut b);

    // `:228-231`: negative roots are already clamped, and a root close to zero
    // zeroes `b` rather than dividing by it.
    let floor: S = S::min_positive().sqrt();
    for i in 0..b.nrows() {
        if d_sqrt[i] > floor {
            b[i] /= d_sqrt[i];
        } else {
            b[i] = S::zero();
        }
    }

    Ok(ReducedSystem { h, b })
}

/// The shared body of `:44-113` and `:129-200`, which are the same statements
/// twice over in the C++.
///
/// Returns `(marg_H, marg_b)`, both over the kept variables.
fn schur_complement<S: LieScalar>(
    abs_h: DMatrix<S>,
    abs_b: DVector<S>,
    idx_to_keep: &BTreeSet<usize>,
    idx_to_marg: &BTreeSet<usize>,
) -> Result<(DMatrix<S>, DVector<S>), MargError> {
    let total: usize = abs_h.ncols();
    let (keep_size, marg_size) = check_indices(idx_to_keep, idx_to_marg, total)?;
    if abs_h.nrows() != total {
        return Err(MargError::NotSquare {
            rows: abs_h.nrows(),
            cols: total,
        });
    }
    if abs_b.nrows() != total {
        return Err(MargError::RhsLengthMismatch {
            rows: total,
            rhs: abs_b.nrows(),
        });
    }

    // `:68-74`: `pt = p.transpose()`, `abs_b.applyOnTheLeft(pt)`,
    // `abs_H.applyOnTheLeft(pt)`, `abs_H.applyOnTheRight(p)`. Both work out to
    // `new[i] = old[indices[i]]` on each axis.
    // **keep first**, then marg, both in ascending index order because C++
    // walks a `std::set` (`:52-66`).
    let indices: Vec<usize> = idx_to_keep
        .iter()
        .chain(idx_to_marg.iter())
        .copied()
        .collect();
    let h: DMatrix<S> = DMatrix::from_fn(total, total, |i, j| abs_h[(indices[i], indices[j])]);
    let b: DVector<S> = DVector::from_fn(total, |i, _| abs_b[indices[i]]);

    // `:99-100`.
    let h_mm: DMatrix<S> = h
        .view((keep_size, keep_size), (marg_size, marg_size))
        .into_owned();
    let h_mm_inv: DMatrix<S> = Cod::new(&h_mm).pseudo_inverse();

    // `:103`: `abs_H.topRightCorner(keep_size, marg_size) *= H_mm_inv`.
    let mut h_km_inv: DMatrix<S> = DMatrix::zeros(keep_size, marg_size);
    for i in 0..keep_size {
        for j in 0..marg_size {
            let mut acc: S = S::zero();
            for k in 0..marg_size {
                acc += h[(i, keep_size + k)] * h_mm_inv[(k, j)];
            }
            h_km_inv[(i, j)] = acc;
        }
    }

    // `:109-113`.
    let mut marg_h: DMatrix<S> = DMatrix::zeros(keep_size, keep_size);
    let mut marg_b: DVector<S> = DVector::zeros(keep_size);
    for i in 0..keep_size {
        for j in 0..keep_size {
            let mut acc: S = S::zero();
            for k in 0..marg_size {
                acc += h_km_inv[(i, k)] * h[(keep_size + k, j)];
            }
            marg_h[(i, j)] = h[(i, j)] - acc;
        }
        let mut acc: S = S::zero();
        for k in 0..marg_size {
            acc += h_km_inv[(i, k)] * b[keep_size + k];
        }
        marg_b[i] = b[i] - acc;
    }

    Ok((marg_h, marg_b))
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

    fn index_sets(keep: &[usize], marg: &[usize]) -> (BTreeSet<usize>, BTreeSet<usize>) {
        (
            keep.iter().copied().collect(),
            marg.iter().copied().collect(),
        )
    }

    fn random_jacobian(values: &[f64], rows: usize, cols: usize) -> DMatrix<f64> {
        DMatrix::from_fn(rows, cols, |i, j| values[(i * cols + j) % values.len()])
    }

    /// The dense Schur complement, written out independently of everything
    /// under test, for a full-rank marginalized block.
    fn dense_schur(
        h: &DMatrix<f64>,
        b: &DVector<f64>,
        keep: &[usize],
        marg: &[usize],
    ) -> (DMatrix<f64>, DVector<f64>) {
        let k: usize = keep.len();
        let m: usize = marg.len();
        let h_kk: DMatrix<f64> = DMatrix::from_fn(k, k, |i, j| h[(keep[i], keep[j])]);
        let h_km: DMatrix<f64> = DMatrix::from_fn(k, m, |i, j| h[(keep[i], marg[j])]);
        let h_mk: DMatrix<f64> = DMatrix::from_fn(m, k, |i, j| h[(marg[i], keep[j])]);
        let h_mm: DMatrix<f64> = DMatrix::from_fn(m, m, |i, j| h[(marg[i], marg[j])]);
        let b_k: DVector<f64> = DVector::from_fn(k, |i, _| b[keep[i]]);
        let b_m: DVector<f64> = DVector::from_fn(m, |i, _| b[marg[i]]);
        let h_mm_inv: DMatrix<f64> = h_mm.try_inverse().unwrap();
        let cross: DMatrix<f64> = &h_km * &h_mm_inv;
        (h_kk - &cross * h_mk, b_k - &cross * b_m)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// The square-root QR marginalization and the dense Schur complement of
        /// the same full-rank problem agree, which is basalt's own
        /// `VoMargSqrtLinearizationTest` argument applied to `MargHelper`.
        #[test]
        fn the_qr_marginalization_is_the_schur_complement(
            values in prop::collection::vec(-1.5f64..1.5, 96..97),
        ) {
            let rows: usize = 16;
            let cols: usize = 6;
            let j: DMatrix<f64> = random_jacobian(&values, rows, cols);
            let r: DVector<f64> = DVector::from_fn(rows, |i, _| values[(i * 7 + 3) % values.len()]);

            let keep: [usize; 4] = [2, 3, 4, 5];
            let marg: [usize; 2] = [0, 1];
            let (keep_set, marg_set) = index_sets(&keep, &marg);

            let reduced =
                marginalize_helper_sqrt_to_sqrt(j.clone(), r.clone(), &keep_set, &marg_set)
                    .unwrap();

            let h: DMatrix<f64> = j.transpose() * &j;
            let b: DVector<f64> = j.transpose() * &r;
            let (schur_h, schur_b) = dense_schur(&h, &b, &keep, &marg);

            let got_h: DMatrix<f64> = reduced.h.transpose() * &reduced.h;
            let got_b: DVector<f64> = reduced.h.transpose() * &reduced.b;
            let scale: f64 = schur_h.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
            for i in 0..keep.len() {
                for jj in 0..keep.len() {
                    prop_assert!((got_h[(i, jj)] - schur_h[(i, jj)]).abs() < 1e-9 * scale);
                }
                prop_assert!((got_b[i] - schur_b[i]).abs() < 1e-9 * scale);
            }
        }

        /// The three routines agree on a full-rank problem: the two square-root
        /// forms squared, and the squared form itself.
        #[test]
        fn the_three_routines_agree_on_a_full_rank_problem(
            values in prop::collection::vec(-1.5f64..1.5, 96..97),
        ) {
            let rows: usize = 16;
            let cols: usize = 6;
            let j: DMatrix<f64> = random_jacobian(&values, rows, cols);
            let r: DVector<f64> = DVector::from_fn(rows, |i, _| values[(i * 5 + 1) % values.len()]);
            let h: DMatrix<f64> = j.transpose() * &j;
            let b: DVector<f64> = j.transpose() * &r;

            let (keep_set, marg_set) = index_sets(&[2, 3, 4, 5], &[0, 1]);

            let qr = marginalize_helper_sqrt_to_sqrt(j, r, &keep_set, &marg_set).unwrap();
            let sq_sqrt =
                marginalize_helper_sq_to_sqrt(h.clone(), b.clone(), &keep_set, &marg_set).unwrap();
            let sq = marginalize_helper_sq_to_sq(h, b, &keep_set, &marg_set).unwrap();

            let qr_h: DMatrix<f64> = qr.h.transpose() * &qr.h;
            let qr_b: DVector<f64> = qr.h.transpose() * &qr.b;
            let ss_h: DMatrix<f64> = sq_sqrt.h.transpose() * &sq_sqrt.h;
            let ss_b: DVector<f64> = sq_sqrt.h.transpose() * &sq_sqrt.b;
            let scale: f64 = sq.h.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
            for i in 0..4 {
                for jj in 0..4 {
                    prop_assert!((qr_h[(i, jj)] - sq.h[(i, jj)]).abs() < 1e-8 * scale);
                    prop_assert!((ss_h[(i, jj)] - sq.h[(i, jj)]).abs() < 1e-8 * scale);
                }
                prop_assert!((qr_b[i] - sq.b[i]).abs() < 1e-8 * scale);
                prop_assert!((ss_b[i] - sq.b[i]).abs() < 1e-8 * scale);
            }
        }
    }

    /// The index checks are typed errors, not out-of-range reads.
    #[test]
    fn inconsistent_index_sets_are_refused() {
        let j: DMatrix<f64> = DMatrix::identity(4, 4);
        let r: DVector<f64> = DVector::zeros(4);
        let (keep, marg) = index_sets(&[0, 1], &[2]);
        assert_eq!(
            marginalize_helper_sqrt_to_sqrt(j.clone(), r.clone(), &keep, &marg),
            Err(MargError::IndexCountMismatch {
                keep: 2,
                marg: 1,
                total: 4
            })
        );

        let (keep, marg) = index_sets(&[0, 1, 2], &[2]);
        assert_eq!(
            marginalize_helper_sqrt_to_sqrt(j.clone(), r.clone(), &keep, &marg),
            Err(MargError::IndexInBothSets { index: 2 })
        );

        let (keep, marg) = index_sets(&[0, 1, 9], &[2]);
        assert_eq!(
            marginalize_helper_sqrt_to_sqrt(j.clone(), r.clone(), &keep, &marg),
            Err(MargError::IndexOutOfRange { index: 9, total: 4 })
        );

        let (keep, marg) = index_sets(&[0, 1, 2], &[3]);
        assert_eq!(
            marginalize_helper_sqrt_to_sqrt(j, DVector::zeros(3), &keep, &marg),
            Err(MargError::RhsLengthMismatch { rows: 4, rhs: 3 })
        );
    }

    /// Marginalizing nothing is the identity on the information, which is the
    /// `marg_size == 0` corner where C++ compares `k` against `-1`.
    #[test]
    fn marginalizing_nothing_keeps_the_whole_system() {
        let j: DMatrix<f64> = DMatrix::from_fn(5, 3, |i, jj| ((i * 3 + jj) as f64).sin());
        let r: DVector<f64> = DVector::from_fn(5, |i, _| (i as f64).cos());
        let keep: BTreeSet<usize> = (0..3).collect();
        let marg: BTreeSet<usize> = BTreeSet::new();
        let reduced = marginalize_helper_sqrt_to_sqrt(j.clone(), r.clone(), &keep, &marg).unwrap();
        let got: DMatrix<f64> = reduced.h.transpose() * &reduced.h;
        let want: DMatrix<f64> = j.transpose() * &j;
        for i in 0..3 {
            for jj in 0..3 {
                assert!((got[(i, jj)] - want[(i, jj)]).abs() < 1e-12);
            }
        }
    }

    /// The `9x2` problem the S7 review reproduced, in both precisions.
    ///
    /// Column 0 is a vector scaled so that `|beta|` lands *on*
    /// `sqrt(epsilon)`; column 1 is `e₀`; the residual is all ones; `{0}` is
    /// marginalized and `{1}` kept. The rank decision at
    /// `marg_helper.cpp:301` therefore turns on the last bit of
    /// `tail.squaredNorm()`, and the two reduction orders disagree in *opposite
    /// directions* in the two precisions:
    ///
    /// * `f32` — Eigen gets `|beta| == sqrt(epsilon)` exactly, and the test is
    ///   strictly greater, so the column is **rejected**: `marg_rank` stays 0,
    ///   the kept column's own reflector fills row 0, and the prior is
    ///   `H = [1]`, `b = [1]`. The sequential fold got a larger `|beta|` and
    ///   accepted.
    /// * `f64` — Eigen gets `|beta|` one ulp *above* the threshold and
    ///   **accepts**, so the prior is row 1 of the reduced system,
    ///   `H = [0.98664153958748302]`, `b = [1.3950496069376288]`. The
    ///   sequential fold rejected and returned `[1]`, `[1]`.
    ///
    /// The column and the expected output are the fork's, printed by
    /// `tools/marg_norm_probe.cpp` (the "flat-QR problem" section) from
    /// basalt's own `MargHelper<Scalar>::marginalizeHelperSqrtToSqrt`.
    ///
    /// `H` is asserted **bit for bit** — that is the rank decision, and before
    /// the reduction order was fixed the `f64` case returned `[1]` from the
    /// other branch. `r_m` is asserted to four ulps instead: it carries the
    /// residual through `Q2r.applyHouseholderOnTheLeft`, whose
    /// `essential.adjoint() * bottom` is a one-column product and so goes
    /// through Eigen 5's dedicated inner-product kernel — a **four**-packet
    /// accumulator loop with `pmadd` (`Eigen/src/Core/InnerProduct.h:129-176`),
    /// a different association from the port's fold. That association reaches
    /// no comparison, which is decision D50's accepted residue; it is called
    /// out here because this test is the place where it is visible.
    #[test]
    fn the_reviews_rank_threshold_problem_takes_eigens_branch() {
        fn check<S: LieScalar>(column: &[f64; 9], want_h: S, want_b: S) {
            let mut j: DMatrix<S> = DMatrix::zeros(9, 2);
            for (i, v) in column.iter().enumerate() {
                j[(i, 0)] = S::from_literal(*v);
            }
            j[(0, 1)] = S::one();
            let r: DVector<S> = DVector::from_element(9, S::one());
            let keep: BTreeSet<usize> = [1].into_iter().collect();
            let marg: BTreeSet<usize> = [0].into_iter().collect();
            let reduced = marginalize_helper_sqrt_to_sqrt(j, r, &keep, &marg).unwrap();
            assert_eq!((reduced.h.nrows(), reduced.h.ncols()), (1, 1));
            assert_eq!(
                reduced.h[(0, 0)].to_f64().to_bits(),
                want_h.to_f64().to_bits(),
                "H: got {:?}, C++ {want_h:?}",
                reduced.h[(0, 0)]
            );
            let slack: f64 = 4.0 * want_b.to_f64().abs() * S::default_epsilon().to_f64();
            let gap: f64 = (reduced.b[0].to_f64() - want_b.to_f64()).abs();
            assert!(
                gap <= slack,
                "b: got {:?}, C++ {want_b:?}, off by {gap} with {slack} allowed",
                reduced.b[0]
            );
        }

        // `f32`: Eigen rejects the marginalized column.
        check::<f32>(
            &[
                1.2011454373350716e-06,
                -9.3419708719011396e-05,
                -3.672563616419211e-05,
                4.3773041397798806e-05,
                -0.00020206424233037978,
                -1.0586967619019561e-05,
                -0.00021459527488332242,
                -1.7148460756288841e-05,
                0.00014116837701294571,
            ],
            f32::from_bits(0x3f80_0000),
            f32::from_bits(0x3f80_0000),
        );
        // `f64`: Eigen accepts it.
        check::<f64>(
            &[
                2.4274934831195633e-09,
                5.3465687495424216e-10,
                -6.8407082063443796e-09,
                -4.9411137195442199e-09,
                -7.022391473991677e-09,
                -6.6648469173392786e-09,
                -5.1883683328356522e-09,
                -4.1798241328628197e-09,
                -2.5557516707027757e-09,
            ],
            f64::from_bits(0x3fef_9291_472c_e812),
            f64::from_bits(0x3ff6_521f_8961_842e),
        );
    }

    /// A column of exact zeros never reaches the rank threshold, so the rank
    /// does not advance and the reduced system loses that direction.
    #[test]
    fn an_exactly_singular_column_does_not_advance_the_rank() {
        let mut j: DMatrix<f64> = DMatrix::from_fn(6, 4, |i, jj| ((i * 4 + jj + 1) as f64).sin());
        for i in 0..6 {
            j[(i, 3)] = 0.0;
        }
        let r: DVector<f64> = DVector::from_fn(6, |i, _| (i as f64 + 0.5).cos());
        let keep: BTreeSet<usize> = [2, 3].into_iter().collect();
        let marg: BTreeSet<usize> = [0, 1].into_iter().collect();
        let reduced = marginalize_helper_sqrt_to_sqrt(j, r, &keep, &marg).unwrap();
        // Column 3 of the original is column 1 of the kept block, and it is
        // zero, so the prior says nothing about it.
        for i in 0..reduced.h.nrows() {
            assert_eq!(reduced.h[(i, 1)], 0.0);
        }
    }
}
