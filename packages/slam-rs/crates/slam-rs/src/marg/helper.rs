//! `MargHelper<Scalar>` (`include/basalt/vi_estimator/marg_helper.h`,
//! `src/vi_estimator/marg_helper.cpp`): the routine that turns a linear system
//! over `keep ∪ marg` into an equivalent one over `keep` alone.
//!
//! [`marginalize_helper_sqrt_to_sqrt`] takes `Q₂J_p`, `Q₂r` to `J_m`, `r_m`
//! (`marg_helper.cpp:247-327`), and it is the only one of the C++'s three the
//! port carries: the linearization is `ABS_QR` and `vio_sqrt_marg` is on in
//! every shipped configuration, so `marginalize()` takes the
//! `is_lin_sqrt && marg_data.is_sqrt` branch (`sqrt_keypoint_vio.cpp:1071-1073`)
//! and `SqrtKeypointVio::new` refuses the flag off (D68).
//!
//! **It consumes its input.** C++ takes `MatX&` and ends with
//! `abs_H.resize(0, 0)` (`:325-326`); the port takes the matrices **by value**,
//! which says the same thing in a way the compiler checks.
//!
//! **The columns are permuted marg first** (`:259-273`), so that a flat QR
//! sweeping left to right eliminates the marginalized variables before it
//! reaches the kept ones and the rows below the marginalized rank are exactly
//! the prior. The C++'s two squared forms permuted keep first because they took
//! a Schur complement, which wanted the block to invert in the bottom-right
//! corner.

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector};

use crate::eigen::qr::{
    BlockSpan, apply_householder_on_the_left_block, apply_householder_on_the_left_vec,
    make_householder,
};
use crate::lie::LieScalar;
use crate::marg::MargError;

/// What the marginalization helper returns: the reduced system over the kept
/// variables, as a square-root prior.
#[derive(Debug, Clone, PartialEq)]
pub struct ReducedSystem<S: LieScalar> {
    /// `marg_sqrt_H` (`J_m`).
    pub h: DMatrix<S>,
    /// `marg_sqrt_b` (`r_m`).
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
/// The rank policy uses the absolute threshold `sqrt(epsilon)` on `beta`.
/// It therefore depends on the units of the scaled problem. A rejected
/// column is zeroed without advancing the rank.
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

        // Reflect this column; the resulting diagonal determines its rank.
        let (h_coeff, beta) = make_householder(&q2jp, k, base, remaining_rows, &mut essential);

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
