//! Invariants of the QR sweep used by square-root marginalization.
use nalgebra::DMatrix;
use proptest::prelude::*;
use slam_rs::linearize::reflect_column;

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
            prop_assert!(reflect_column(&mut augmented, col, col, 6 - col).is_ok());
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
