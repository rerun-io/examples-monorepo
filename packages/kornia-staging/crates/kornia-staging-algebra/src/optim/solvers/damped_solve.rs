//! One scaled dense damped normal-equation solve.
use crate::Scalar;
use nalgebra::{DMatrix, DVector};

/// A damped solve refused its inputs or could not produce a finite increment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DampedSolveError {
    /// The normal matrix is empty or its dimensions disagree with the vectors.
    #[error("damped solve needs a nonempty square matrix and matching vectors")]
    ShapeMismatch,
    /// The solved increment is nonfinite or the factorization is singular.
    #[error("damped solve did not produce a finite increment")]
    NonFinite,
}

/// Attempt `(H + diag(max(lambda * diag(H), min_lambda))) x = b` in f64.
/// Full-pivot LU permits rounded indefinite normal matrices. The caller owns retries.
/// # Arguments
/// * `h` - Nonempty square normal matrix.
/// * `b` - Right-hand side, with one element per matrix row.
/// * `lambda` - Nonnegative damping multiplier.
/// * `min_lambda` - Nonnegative floor on each damping term.
/// * `increment` - Output with the same length as `b`; apply the desired step sign at the caller.
/// # Errors
/// Returns [`DampedSolveError::ShapeMismatch`] before writing the output when dimensions disagree.
/// A numerical failure returns [`DampedSolveError::NonFinite`]; its output must not be applied.
/// ```
/// use kornia_staging_algebra::optim::solvers::solve_scaled_damped;
/// use nalgebra::{DMatrix, DVector};
/// let mut step = DVector::zeros(1);
/// assert!(solve_scaled_damped(&DMatrix::identity(1,1), &DVector::repeat(1,1.0_f64), 1.0, 0.0, &mut step).is_ok());
/// assert_eq!(step[0], 0.5);
/// ```
pub fn solve_scaled_damped<S: Scalar>(
    h: &DMatrix<S>,
    b: &DVector<S>,
    lambda: S,
    min_lambda: S,
    increment: &mut DVector<S>,
) -> Result<(), DampedSolveError> {
    let size = b.len();
    if size == 0 || h.shape() != (size, size) || increment.len() != size {
        return Err(DampedSolveError::ShapeMismatch);
    }
    let mut working = h.map(|value| value.to_f64());
    for i in 0..size {
        let diagonal = h[(i, i)] * lambda;
        // A comparison rather than f32/f64::max retains a NaN on the left.
        let damped = if diagonal < min_lambda {
            min_lambda
        } else {
            diagonal
        };
        working[(i, i)] += damped.to_f64();
    }
    let scales = DVector::from_iterator(
        size,
        (0..size).map(|i| {
            let scale = working[(i, i)].abs().sqrt();
            if scale > 0.0 {
                scale
            } else {
                1.0
            }
        }),
    );
    for col in 0..size {
        for row in 0..size {
            working[(row, col)] /= scales[row] * scales[col];
        }
    }
    let factor = nalgebra::linalg::FullPivLU::new(working);
    let mut solution = b.map(|value| value.to_f64());
    solution.component_div_assign(&scales);
    if factor.solve_mut(&mut solution) {
        for i in 0..size {
            increment[i] = S::from_literal(solution[i] / scales[i]);
        }
    } else {
        increment.fill(S::from_literal(f64::NAN));
    }
    if increment.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(DampedSolveError::NonFinite)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mismatched_shapes_preserve_the_output() {
        for (rows, cols, rhs, output) in [(0, 0, 0, 0), (2, 3, 2, 2), (2, 2, 3, 3), (2, 2, 2, 3)] {
            let mut step = DVector::repeat(output, 42.0_f64);
            let before = step.clone();
            assert_eq!(
                solve_scaled_damped(
                    &DMatrix::identity(rows, cols),
                    &DVector::zeros(rhs),
                    1.0,
                    0.0,
                    &mut step
                ),
                Err(DampedSolveError::ShapeMismatch)
            );
            assert_eq!(step, before);
        }
    }

    #[test]
    fn singular_solve_returns_a_typed_failure() {
        let mut step = DVector::zeros(1);
        assert_eq!(
            solve_scaled_damped(
                &DMatrix::zeros(1, 1),
                &DVector::repeat(1, 1.0_f64),
                0.0,
                0.0,
                &mut step
            ),
            Err(DampedSolveError::NonFinite)
        );
    }

    #[test]
    fn a_rounded_indefinite_system_still_has_a_finite_solve() {
        // Rounding can leave a tiny negative eigenvalue in a normal matrix.
        // Positive damping below one ulp does not guarantee a positive factor.
        let off = 1.0f32 + f32::EPSILON;
        let h = DMatrix::from_row_slice(2, 2, &[1.0, off, off, 1.0]);
        let b = DVector::from_element(2, 1.0f32);
        let lambda = 1e-9;
        let min_lambda = 1e-9;
        let mut inc = DVector::zeros(b.len());
        let valid = solve_scaled_damped(&h, &b, lambda, min_lambda, &mut inc);
        assert!(valid.is_ok());
        assert!((h * inc - b).norm() < 1e-6);
    }

    #[test]
    fn damping_below_f32_precision_still_regularizes_a_singular_system() {
        let h = DMatrix::from_element(2, 2, 1.0f32);
        let b = DVector::from_vec(vec![1.0f32, -1.0]);
        let lambda = 1e-9;
        let min_lambda = 1e-9;
        let mut inc = DVector::zeros(b.len());
        let valid = solve_scaled_damped(&h, &b, lambda, min_lambda, &mut inc);
        assert!(valid.is_ok());
        // Check in f64: rounding the damped matrix back to f32 removes its rank.
        let mut damped = h.map(f64::from);
        damped[(0, 0)] += f64::from(lambda);
        damped[(1, 1)] += f64::from(lambda);
        assert!((damped * inc.map(f64::from) - b.map(f64::from)).norm() < 1e-6);
    }

    proptest::proptest! {
        #[test]
        fn damped_system_has_a_small_residual(
            values in proptest::collection::vec(-1.0f64..1.0, 36),
            rhs in proptest::collection::vec(-1.0f64..1.0, 6),
            exponent in -10i32..0,
        ) {
            let g = DMatrix::from_row_slice(6, 6, &values);
            let h = g.transpose() * g * 10.0f64.powi(exponent);
            let b = DVector::from_vec(rhs);
            let lambda = 1e-3;
            let min_lambda = 1e-8;
            let mut inc = DVector::zeros(b.len());
            let valid = solve_scaled_damped(&h, &b, lambda, min_lambda, &mut inc);
            proptest::prop_assert!(valid.is_ok());
            let mut damped = h.clone();
            for i in 0..6 {
                damped[(i, i)] += (h[(i, i)] * lambda).max(min_lambda);
            }
            // The caller negates inc to solve H x = -b.
            proptest::prop_assert!((damped * -inc + &b).norm() < 1e-8 * (1.0 + b.norm()));
        }
    }
}
