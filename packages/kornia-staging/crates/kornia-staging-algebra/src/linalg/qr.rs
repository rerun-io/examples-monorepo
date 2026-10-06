//! Orthogonal transforms over column-major buffers, with caller-owned scratch.

use crate::Scalar;
use nalgebra::{DVectorView, DVectorViewMut};

/// Build a unit Householder axis with the original two normalization passes.
///
/// # Arguments
/// `column` is a nonempty contiguous column segment; `axis` is caller scratch
/// with at least that many elements. Returns `(active, signed_diagonal)`.
///
/// ```
/// use kornia_staging_algebra::linalg::qr::{make_householder_unchecked, apply_householder_unchecked};
/// let mut column = [3.0f64, 4.0];
/// let mut axis = [0.0; 2];
/// let (active, beta) = make_householder_unchecked(&column, &mut axis);
/// apply_householder_unchecked(&mut column, 2, 1, 2, &axis, active);
/// assert!((column[0] - beta).abs() < 1e-12);
/// assert!(column[1].abs() < 1e-12);
/// ```
///
/// # Panics
/// Panics if `column` is empty or scratch is too short. The caller validates
/// dimensions once at its matrix boundary; this kernel does not repeat validation.
#[inline]
pub fn make_householder_unchecked<S: Scalar>(column: &[S], axis: &mut [S]) -> (bool, S) {
    let len = column.len();
    axis[..len].copy_from_slice(column);
    let mut axis = DVectorViewMut::from_slice(&mut axis[..len], len);
    let squared_norm = axis.norm_squared();
    let norm = squared_norm.sqrt();
    let (modulus, sign) = axis[0].to_exp();
    let signed_norm = sign * norm;
    let factor = (squared_norm + modulus * norm) * S::from_literal(2.0);
    axis[0] += signed_norm;
    if factor != S::zero() {
        axis.unscale_mut(factor.sqrt());
        let _ = axis.normalize_mut();
        (true, -signed_norm)
    } else {
        (false, signed_norm)
    }
}

/// Reflect a strided column-major submatrix on the left, without allocating.
///
/// # Arguments
/// `matrix` starts at the submatrix's top left; `rows` and `cols` select its
/// shape and `stride` is the parent row count. `axis[..rows]` is a unit axis
/// from [`make_householder_unchecked`]. An inactive reflection does no work.
///
/// # Panics
/// Panics on insufficient matrix/axis storage. The caller must establish
/// `stride >= rows` and sufficient storage at its matrix boundary.
#[inline]
pub fn apply_householder_unchecked<S: Scalar>(
    matrix: &mut [S],
    rows: usize,
    cols: usize,
    stride: usize,
    axis: &[S],
    active: bool,
) {
    if active {
        let axis = DVectorView::from_slice(&axis[..rows], rows);
        for col in 0..cols {
            let start = col * stride;
            let mut column = DVectorViewMut::from_slice(&mut matrix[start..start + rows], rows);
            let factor = axis.dotc(&column) * S::from_literal(-2.0);
            column.axpy(factor, &axis, S::one());
        }
    }
}

/// A scaled two-row Givens rotation; coefficients follow `[c, -s; s, c]`.
#[derive(Debug, Clone, Copy)]
pub struct Givens<S: Scalar> {
    c: S,
    s: S,
}

impl<S: Scalar> Givens<S> {
    /// Construct a rotation cancelling `q` from `[p, q]`, scaling before squaring.
    /// Zero inputs produce the identity; non-finite inputs propagate.
    #[inline]
    pub fn cancel_y(p: S, q: S) -> Self {
        let scale = p.abs().max(q.abs());
        if scale == S::zero() {
            return Self {
                c: S::one(),
                s: S::zero(),
            };
        }
        let p = p / scale;
        let q = q / scale;
        if q == S::zero() {
            return Self {
                c: S::one(),
                s: S::zero(),
            };
        }
        let (modulus, sign) = p.to_exp();
        let denom = (modulus * modulus + q * q).sqrt();
        Self {
            c: modulus / denom,
            s: -q / (sign * denom),
        }
    }

    /// Rotate two adjacent rows of a column-major submatrix in place.
    ///
    /// # Arguments
    /// `matrix` starts at the first row, `cols` is its column count, and `stride`
    /// is the parent row count (at least two).
    ///
    /// # Panics
    /// Panics if the buffer cannot hold the selected rows and columns.
    #[inline]
    pub fn apply_unchecked(&self, matrix: &mut [S], cols: usize, stride: usize) {
        for col in 0..cols {
            let start = col * stride;
            let a = matrix[start];
            let b = matrix[start + 1];
            matrix[start] = a * self.c - self.s * b;
            matrix[start + 1] = self.s * a + b * self.c;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use proptest::prelude::*;

    #[test]
    fn transforms_preserve_reference_bits_in_both_precisions() {
        fn check<S: Scalar>() {
            use nalgebra::linalg::{givens::GivensRotation, householder::reflection_axis_mut};
            use nalgebra::{DMatrix, DVector, Reflection, Unit, Vector2};
            for rows in [1, 2, 3, 4, 7, 8, 17, 32] {
                let input = DVector::from_iterator(
                    rows,
                    (0..rows).map(|i| S::from_literal((i as f64 + 0.7).sin())),
                );
                let mut reference_axis = input.clone();
                let (beta, active) = reflection_axis_mut(&mut reference_axis);
                let mut axis = vec![S::zero(); rows];
                let result = make_householder_unchecked(input.as_slice(), &mut axis);
                assert_eq!(result, (active, beta));
                assert_eq!(axis, reference_axis.as_slice());
                let mut reference = DMatrix::from_fn(rows, 5, |r, c| {
                    S::from_literal((r as f64 + c as f64 * 0.3).cos())
                });
                let mut actual = reference.clone();
                if active {
                    Reflection::new(Unit::new_unchecked(reference_axis), S::zero())
                        .reflect(&mut reference);
                }
                apply_householder_unchecked(actual.as_mut_slice(), rows, 5, rows, &axis, active);
                for (a, b) in actual.iter().zip(reference.iter()) {
                    assert_eq!(a.to_f64().to_bits(), b.to_f64().to_bits());
                }
            }
            for (p, q) in [
                (0.0, 0.0),
                (-0.0, 1.0),
                (3.0, 4.0),
                (-3.0, 4.0),
                (1e30, -1e30),
            ] {
                let p = S::from_literal(p);
                let q = S::from_literal(q);
                let scale = p.abs().max(q.abs());
                let rotation = if scale == S::zero() {
                    GivensRotation::identity()
                } else {
                    GivensRotation::cancel_y(&Vector2::new(p / scale, q / scale))
                        .map_or_else(GivensRotation::identity, |(r, _)| r)
                };
                let mut reference = DMatrix::from_row_slice(2, 2, &[p, q, q, p]);
                let mut actual = reference.clone();
                rotation.rotate(&mut reference.fixed_rows_mut::<2>(0));
                Givens::cancel_y(p, q).apply_unchecked(actual.as_mut_slice(), 2, 2);
                for (a, b) in actual.iter().zip(reference.iter()) {
                    assert_eq!(a.to_f64().to_bits(), b.to_f64().to_bits());
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    proptest! {
        #[test]
        fn a_sub_block_reflection_preserves_its_surroundings(
            values in prop::collection::vec(-5.0f64..5.0, 42),
            start in 0usize..3,
        ) {
            let mut a = DMatrix::from_row_slice(7, 6, &values);
            let before = a.clone();
            let mut axis = [0.0; 4];
            let (active, beta) = make_householder_unchecked(&a.as_slice()[14 + start..18 + start], &mut axis);
            apply_householder_unchecked(&mut a.as_mut_slice()[2 * 7 + start..], 4, 3, 7, &axis, active);
            for i in 0..7 {
                for j in 0..6 {
                    if !(start..start + 4).contains(&i) || !(2..5).contains(&j) {
                        prop_assert_eq!(a[(i, j)], before[(i, j)]);
                    }
                }
            }
            prop_assert!((a[(start, 2)] - beta).abs() < 1e-12);
            for i in start + 1..start + 4 {
                prop_assert!(a[(i, 2)].abs() < 1e-12);
            }
            let mut residual = before.column(2).into_owned();
            apply_householder_unchecked(&mut residual.as_mut_slice()[start..], 4, 1, 7, &axis, active);
            prop_assert!((residual - a.column(2)).norm() < 1e-12);
        }

        #[test]
        fn givens_cancels_the_lower_coefficient(p in -100.0f64..100.0, q in -100.0f64..100.0) {
            let rotation = Givens::cancel_y(p, q);
            let mut a = DMatrix::from_row_slice(2, 1, &[p, q]);
            rotation.apply_unchecked(a.as_mut_slice(), 1, 2);
            prop_assert!(a[(1, 0)].abs() < 1e-12);
            prop_assert!((a.norm() - p.hypot(q)).abs() < 1e-12);
        }
    }

    proptest! {
        #[test]
        fn reflections_preserve_norms_and_form_an_orthogonal_q(
            values in prop::collection::vec(-10.0f64..10.0, 24),
        ) {
            let a = DMatrix::from_row_slice(6, 4, &values);
            // Carry an identity beside A to observe the same sequence of reflections.
            let mut augmented = DMatrix::zeros(6, 10);
            augmented.columns_mut(0, 4).copy_from(&a);
            augmented.columns_mut(4, 6).copy_from(&DMatrix::identity(6, 6));
            for col in 0..4 {
                let mut axis = [0.0; 6];
                let (active, _) = make_householder_unchecked(&augmented.as_slice()[col * 6 + col..(col + 1) * 6], &mut axis);
                apply_householder_unchecked(&mut augmented.as_mut_slice()[col..], 6 - col, 10, 6, &axis, active);
            }
            let r = augmented.columns(0, 4);
            let qt = augmented.columns(4, 6);
            prop_assert!((qt * qt.transpose() - DMatrix::identity(6, 6)).norm() < 1e-12);
            prop_assert!((r.norm() - a.norm()).abs() < 1e-12 * (1.0 + a.norm()));
            prop_assert!((qt * &a - r).norm() < 1e-12 * (1.0 + a.norm()));
            for col in 0..4 {
                for row in col + 1..6 {
                    prop_assert!(r[(row, col)].abs() < 1e-12 * (1.0 + a.norm()));
                }
            }
        }
    }

    /// `QRvsLLT` and `QRvsLLTRankDef`, as the assertion they
    /// demonstrate: for a `10 x 6` `J`, the `R` of a Householder QR and the `Lᵀ` of
    /// the Cholesky of `JᵀJ` agree row by row up to a sign.
    ///
    /// The QR is built from this port's own crate Householder primitives, so
    /// the test is on the port, not on nalgebra.
    #[test]
    fn householder_qr_matches_the_cholesky_of_the_normal_equations() {
        for (case, rank_deficient) in [("full rank", false), ("rank deficient", true)] {
            let mut seed: u64 = 0xfeed_1235;
            let mut j: DMatrix<f64> = DMatrix::zeros(10, 6);
            for r in 0..10 {
                for c in 0..6 {
                    seed ^= seed >> 12;
                    seed ^= seed << 25;
                    seed ^= seed >> 27;
                    let value = seed.wrapping_mul(0x2545_f491_4f6c_dd1d);
                    j[(r, c)] = (value >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
                }
            }
            if rank_deficient {
                // `J.col(2) = J.col(4)`.
                let col4: DVector<f64> = j.column(4).into_owned();
                j.set_column(2, &col4);
            }

            let r_qr: DMatrix<f64> = householder_r(&j);
            let ata: DMatrix<f64> = j.transpose() * &j;

            if rank_deficient {
                // For singular `JᵀJ`, QR still yields triangular `R` with `RᵀR = JᵀJ`.
                let rtr: DMatrix<f64> = r_qr.transpose() * &r_qr;
                assert!(
                    (&rtr - &ata).norm() <= 1e-9 * ata.norm(),
                    "{case}: RᵀR != JᵀJ"
                );
                continue;
            }

            let llt = ata.clone().cholesky().expect("full rank");
            let u: DMatrix<f64> = llt.l().transpose();
            for row in 0..6 {
                // Up to the sign of the row, which is all a QR is free in.
                let sign: f64 = if r_qr[(row, row)] * u[(row, row)] < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                for col in 0..6 {
                    let got: f64 = sign * r_qr[(row, col)];
                    assert!(
                        (got - u[(row, col)]).abs() <= 1e-9 * u.norm(),
                        "{case}: R[{row},{col}] = {got}, Lᵀ = {}",
                        u[(row, col)]
                    );
                }
            }
        }
    }

    /// The upper-triangular factor of a Householder QR, driven exactly as
    /// `performQRHouseholder` drives it, through the port's own primitives.
    fn householder_r(j: &DMatrix<f64>) -> DMatrix<f64> {
        let rows: usize = j.nrows();
        let cols: usize = j.ncols();
        let mut work: DMatrix<f64> = j.clone();
        let mut scratch = vec![0.0; rows];
        for k in 0..rows.min(cols) {
            let offset = k * rows + k;
            let len = rows - k;
            let (active, beta) =
                make_householder_unchecked(&work.as_slice()[offset..offset + len], &mut scratch);
            apply_householder_unchecked(
                &mut work.as_mut_slice()[offset..],
                len,
                cols - k,
                rows,
                &scratch,
                active,
            );
            work[(k, k)] = beta;
            work.as_mut_slice()[offset + 1..offset + len].fill(0.0);
        }
        let mut r: DMatrix<f64> = DMatrix::zeros(cols, cols);
        for row in 0..cols {
            for col in row..cols {
                r[(row, col)] = work[(row, col)];
            }
        }
        r
    }

    macro_rules! pivot_boundary_properties {
        ($name:ident, $scalar:ty) => {
            proptest! {
                #[test]
                fn $name(
                    distance in 0.01f64..0.49,
                    above in any::<bool>(),
                    residual in -4.0f64..4.0,
                    lead_row in 0usize..5,
                    pivot_row in 0usize..5,
                    flip_lead in any::<bool>(),
                    flip_pivot in any::<bool>(),
                ) {
                    // Rows are placed at random and signs flipped so every accepted
                    // column needs a real two-row reflection, not a no-op. Entries
                    // stay one per column so the arithmetic is exact and the pivot
                    // sits exactly where `distance` puts it against the threshold.
                    prop_assume!(lead_row != pivot_row);
                    let threshold = <$scalar>::EPSILON.sqrt();
                    let pivot = threshold * (1.0 + if above { distance as $scalar } else { -distance as $scalar });
                    let lead_sign: $scalar = if flip_lead { -1.0 } else { 1.0 };
                    let pivot_sign: $scalar = if flip_pivot { -1.0 } else { 1.0 };
                    let mut storage = DMatrix::<$scalar>::zeros(5, 5);
                    storage[(lead_row, 0)] = 2.0 * lead_sign;
                    storage[(lead_row, 1)] = 4.0 * lead_sign; // dependent column
                    storage[(pivot_row, 3)] = pivot * pivot_sign; // column 2 is zero
                    storage[(pivot_row, 4)] = residual as $scalar * pivot_sign;
                    let mut axis = [0.0; 5];
                    let mut rank = 0;
                    for col in 0..4 {
                        let rows = 5 - rank;
                        let (active, beta) = make_householder_unchecked(&storage.as_slice()[col * 5 + rank..(col + 1) * 5], &mut axis);
                        if beta.abs() > threshold {
                            apply_householder_unchecked(&mut storage.as_mut_slice()[col * 5 + rank..], rows, 5 - col, 5, &axis, active);
                            rank += 1;
                        }
                    }
                    prop_assert_eq!(rank, 1 + usize::from(above));
                    // An accepted pivot column is reflected to row `rank`; a rejected
                    // one is left where it was, except that the lead column's
                    // reflection swaps rows 0 and `lead_row`.
                    let pivot_at: usize = if above { 1 } else if pivot_row == 0 { lead_row } else { pivot_row };
                    prop_assert!((storage.column(4).norm_squared() - (residual as $scalar).powi(2)).abs() <= 32.0 * <$scalar>::EPSILON * (1.0 + residual.abs() as $scalar).powi(2));
                    prop_assert!((storage[(pivot_at, 3)].abs() - pivot).abs() <= 8.0 * <$scalar>::EPSILON * pivot);
                    prop_assert!((storage[(pivot_at, 3)] * storage[(pivot_at, 4)] - pivot * residual as $scalar).abs() <= 32.0 * <$scalar>::EPSILON * pivot * (1.0 + residual.abs() as $scalar));
                }
            }
        };
    }
    pivot_boundary_properties!(pivot_boundary_f32, f32);
    pivot_boundary_properties!(pivot_boundary_f64, f64);
}
