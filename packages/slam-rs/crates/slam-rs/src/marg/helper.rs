//! Eliminate marginalized variables from a square-root system.
//! [`marginalize_helper_sqrt_to_sqrt`] consumes matrices by value and produces
//! the prior Jacobian and residual. Put marginalized columns first, then sweep
//! QR left to right; rows below their rank constrain only kept variables.
//! Only square-root marginalization is supported (D68).

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector};

use crate::lie::LieScalar;
use crate::marg::MargError;
use crate::qr::{apply_householder_on_the_left, make_householder};

/// What the marginalization helper returns: the reduced system over the kept
/// variables, as a square-root prior.
#[derive(Debug, Clone, PartialEq)]
pub struct ReducedSystem<S: LieScalar> {
    /// `marg_sqrt_H` (`J_m`).
    pub h: DMatrix<S>,
    /// `marg_sqrt_b` (`r_m`).
    pub b: DVector<S>,
}

/// Validate complete, disjoint, in-range keep/marginalize index sets (D32).
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

/// `MargHelper::marginalizeHelperSqrtToSqrt`.
///
/// A rank-revealing Householder QR over the whole stacked `[J_marg | J_keep]`,
/// swept column by column. Every column that produces a reflector with
/// `|beta| > sqrt(epsilon)` consumes one row of rank; a column that does not is
/// zeroed and the rank does not advance. `marg_rank` is the rank reached when
/// the last marginalized column is done, and the prior is the block of
/// rows `[marg_rank, total_rank)` against the kept columns — the
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
    if q2r.nrows() != rows {
        return Err(MargError::RhsLengthMismatch {
            rows,
            rhs: q2r.nrows(),
        });
    }

    // Permute marginalized columns first: new column i is old column indices[i].
    let indices: Vec<usize> = idx_to_marg
        .iter()
        .chain(idx_to_keep.iter())
        .copied()
        .collect();
    let permuted: DMatrix<S> = DMatrix::from_fn(rows, cols, |i, j| q2jp[(i, indices[j])]);
    q2jp = permuted;

    let rank_threshold: S = S::default_epsilon().sqrt();
    let mut marg_rank: usize = 0;
    let mut total_rank: usize = 0;
    // Reuse a full unit-axis buffer for every reflection.
    let mut essential: Vec<S> = vec![S::zero(); rows];

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
            q2jp[(base, k)] = beta;
            // the reflection acts on the trailing block that starts at row
            // `base`, column `k + 1`.
            apply_householder_on_the_left(
                q2jp.view_mut((base, k + 1), (remaining_rows, remaining_cols)),
                &essential[..remaining_rows],
                h_coeff,
            );
            // the same reflection on the residual, in lockstep.
            apply_householder_on_the_left(
                q2r.rows_mut(base, remaining_rows),
                &essential[..remaining_rows],
                h_coeff,
            );
            total_rank += 1;
        } else {
            q2jp[(base, k)] = S::zero();
        }

        // overwrite the Householder vector with zeros. `remainingRows`
        // is the value from **before** the rank advanced.
        for i in (base + 1)..rows {
            q2jp[(i, k)] = S::zero();
        }

        // With no marginalized columns there is no boundary at which to capture marginal rank.
        if k + 1 == marg_size {
            marg_rank = total_rank;
        }
    }

    let keep_valid_rows: usize = (total_rank - marg_rank).max(1);
    let mut h: DMatrix<S> = DMatrix::zeros(keep_valid_rows, keep_size);
    let mut b: DVector<S> = DVector::zeros(keep_valid_rows);
    for i in 0..keep_valid_rows {
        let row: usize = marg_rank + i;
        // If marginalized variables exhaust the rank, leave the minimum output row zero:
        // no remaining constraint acts on kept variables.
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
    // Keep probe constants at their recorded precision.
    #![allow(clippy::excessive_precision)]

    use super::*;
    use proptest::prelude::*;

    // Generate distance from the threshold, so shrinking cannot cross it.
    macro_rules! rank_boundary_properties {
        ($name:ident, $scalar:ty) => {
            proptest! {
                #[test]
                fn $name(distance in 0.01f64..0.49, above in any::<bool>(), residual in -4.0f64..4.0) {
                    let threshold = <$scalar>::EPSILON.sqrt();
                    let pivot = threshold * (1.0 + if above { distance as $scalar } else { -distance as $scalar });
                    // Marginalize one direction and its duplicate. Keep a zero
                    // column, the boundary pivot, and another direction plus its duplicate.
                    let mut j = DMatrix::<$scalar>::zeros(6, 6);
                    j[(0, 0)] = 2.0;
                    j[(0, 1)] = 4.0;
                    j[(1, 3)] = pivot;
                    j[(2, 4)] = 3.0;
                    j[(2, 5)] = 6.0;
                    let r = DVector::from_vec(vec![1.0, residual as $scalar, 2.0, 0.0, 0.0, 0.0]);
                    let reduced = marginalize_helper_sqrt_to_sqrt(j, r, &(2..6).collect(), &(0..2).collect()).unwrap();
                    prop_assert_eq!(reduced.h.nrows(), 1 + usize::from(above));
                    let h = reduced.h.transpose() * &reduced.h;
                    let b = reduced.h.transpose() * &reduced.b;
                    prop_assert_eq!(h[(0, 0)], 0.0);
                    let expected = if above { pivot * pivot } else { 0.0 };
                    prop_assert!((h[(1, 1)] - expected).abs() <= 16.0 * <$scalar>::EPSILON * pivot * pivot);
                    let expected_b = if above { pivot * residual as $scalar } else { 0.0 };
                    prop_assert!((b[1] - expected_b).abs() <= 16.0 * <$scalar>::EPSILON * pivot * (1.0 + residual.abs() as $scalar));
                    for (row, col, value) in [(2, 2, 9.0), (2, 3, 18.0), (3, 2, 18.0), (3, 3, 36.0)] {
                        prop_assert!((h[(row, col)] - value).abs() < 64.0 * <$scalar>::EPSILON * value);
                    }
                    prop_assert!((b[2] - 6.0).abs() < 64.0 * <$scalar>::EPSILON * 6.0);
                    prop_assert!((b[3] - 12.0).abs() < 64.0 * <$scalar>::EPSILON * 12.0);
                    let expected_residual = 4.0 + if above { (residual as $scalar).powi(2) } else { 0.0 };
                    prop_assert!((reduced.b.norm_squared() - expected_residual).abs() < 128.0 * <$scalar>::EPSILON * (1.0 + expected_residual));
                }
            }
        };
    }
    rank_boundary_properties!(rank_boundary_f32, f32);
    rank_boundary_properties!(rank_boundary_f64, f64);

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

        /// Square-root marginalization must equal the dense Schur complement for a full-rank problem.
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

    /// Marginalizing no variables preserves information.
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
