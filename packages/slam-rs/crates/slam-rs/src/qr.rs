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
    use super::*;
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
}
