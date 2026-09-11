//! In-place orthogonal transforms over column-major landmark and prior storage.
use crate::lie::LieScalar;
use nalgebra::linalg::{givens::GivensRotation, householder::reflection_axis_mut};
use nalgebra::{
    DMatrix, DVectorView, DVectorViewMut, Dim, Dyn, Matrix, Reflection, StorageMut, Unit, Vector2,
};

/// Build a unit reflection axis in caller-owned scratch, returning (active, beta).
/// Keeping the full axis lets nalgebra apply the reflection directly to views.
pub(crate) fn make_householder<S: LieScalar>(
    storage: &DMatrix<S>,
    col: usize,
    start: usize,
    len: usize,
    axis: &mut [S],
) -> (bool, S) {
    let mut axis = DVectorViewMut::from_slice(&mut axis[..len], len);
    axis.copy_from(&storage.column(col).rows(start, len));
    let (beta, active) = reflection_axis_mut(&mut axis);
    (active, beta)
}

/// Reflect a mutable matrix or vector view. The axis is unit length when active.
pub(crate) fn apply_householder_on_the_left<S: LieScalar, C: Dim, T: StorageMut<S, Dyn, C>>(
    mut view: Matrix<S, Dyn, C, T>,
    axis: &[S],
    active: bool,
) {
    if active {
        let rows = view.nrows();
        let axis = DVectorView::from_slice(&axis[..rows], rows);
        Reflection::new(Unit::new_unchecked(axis), S::zero()).reflect(&mut view);
    }
}

/// Scale before constructing the rotation to avoid squaring large coefficients.
pub(crate) fn make_givens<S: LieScalar>(p: S, q: S) -> GivensRotation<S> {
    let scale = p.abs().max(q.abs());
    if scale == S::zero() {
        GivensRotation::identity()
    } else {
        GivensRotation::cancel_y(&Vector2::new(p / scale, q / scale))
            .map_or_else(GivensRotation::identity, |(rotation, _)| rotation)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use nalgebra::DVector;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn a_sub_block_reflection_preserves_its_surroundings(
            values in prop::collection::vec(-5.0f64..5.0, 42),
            start in 0usize..3,
        ) {
            let mut a = DMatrix::from_row_slice(7, 6, &values);
            let before = a.clone();
            let mut axis = [0.0; 4];
            let (active, beta) = make_householder(&a, 2, start, 4, &mut axis);
            apply_householder_on_the_left(a.view_mut((start, 2), (4, 3)), &axis, active);
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
            apply_householder_on_the_left(residual.rows_mut(start, 4), &axis, active);
            prop_assert!((residual - a.column(2)).norm() < 1e-12);
        }

        #[test]
        fn givens_cancels_the_lower_coefficient(p in -100.0f64..100.0, q in -100.0f64..100.0) {
            let rotation = make_givens(p, q);
            let mut a = DMatrix::from_row_slice(2, 1, &[p, q]);
            rotation.rotate(&mut a.fixed_rows_mut::<2>(0));
            prop_assert!(a[(1, 0)].abs() < 1e-12);
            prop_assert!((a.norm() - p.hypot(q)).abs() < 1e-12);
        }
    }

    // Test-only driver; production callers own their scratch and validated views.
    fn reflect_column<S: LieScalar>(
        storage: &mut DMatrix<S>,
        col: usize,
        start: usize,
        len: usize,
    ) {
        let mut axis = vec![S::zero(); len];
        let (active, _) = make_householder(storage, col, start, len, &mut axis);
        let cols = storage.ncols();
        apply_householder_on_the_left(storage.view_mut((start, 0), (len, cols)), &axis, active);
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
                reflect_column(&mut augmented, col, col, 6 - col);
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

            // The port has no standalone QR — the landmark block eliminates exactly
            // three columns — so the reflections are driven here the way
            // `performQRHouseholder` drives them.
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
        for k in 0..cols {
            reflect_column(&mut work, k, k, rows - k);
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
                        let (active, beta) = make_householder(&storage, col, rank, rows, &mut axis);
                        if beta.abs() > threshold {
                            apply_householder_on_the_left(storage.view_mut((rank, col), (rows, 5 - col)), &axis, active);
                            rank += 1;
                        }
                    }
                    prop_assert_eq!(rank, 1 + usize::from(above));
                    prop_assert!((storage.column(4).norm_squared() - (residual as $scalar).powi(2)).abs() <= 32.0 * <$scalar>::EPSILON * (1.0 + residual.abs() as $scalar).powi(2));
                    prop_assert!((storage[(1, 3)].abs() - pivot).abs() <= 8.0 * <$scalar>::EPSILON * pivot);
                    prop_assert!((storage[(1, 3)] * storage[(1, 4)] - pivot * residual as $scalar).abs() <= 32.0 * <$scalar>::EPSILON * pivot * (1.0 + residual.abs() as $scalar));
                }
            }
        };
    }
    pivot_boundary_properties!(pivot_boundary_f32, f32);
    pivot_boundary_properties!(pivot_boundary_f64, f64);
}
