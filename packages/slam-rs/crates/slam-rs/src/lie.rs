//! SO(3), SE(3), group Jacobians and decoupled pose increments.
//! SO(3) operations delegate to kornia-algebra through scalar adapters (S33).
//! This module supplies reported-angle conventions, composition normalization,
//! inverse Jacobians and SE(3) operations.
//!
//! Tangents put translation first. Estimator updates add translation directly
//! and left-multiply rotation by `exp(inc[3..6])`; every state block and prior
//! uses this convention. Coupled SE(3) operations instead use a translation factor.
//! Their small-angle Taylor branches stop one term earlier than the standalone
//! SO(3) Jacobians, so each call retains its own formula.

use kornia_staging_algebra::Scalar;
/// A literal in the caller's scalar type, through `S::from_literal`.
#[inline]
pub(crate) fn c<S: Scalar>(value: f64) -> S {
    S::from_literal(value)
}

/// Return the larger value, preserving a NaN on the left.
/// The damped solver must retry on invalid input rather than suppress its NaN.
pub(crate) fn eigen_maxi<S: Scalar>(a: S, b: S) -> S {
    if a < b { b } else { a }
}

pub use kornia_staging_algebra::lie::{RigidTransform as Se3, Rotation3 as So3};

#[cfg(test)]
mod tests {
    use super::eigen_maxi;
    #[test]
    fn eigen_maxi_keeps_a_nan_on_the_left() {
        assert!(eigen_maxi(f64::NAN, 1.0).is_nan());
        assert_eq!(eigen_maxi(1.0f64, f64::NAN), 1.0);
        assert_eq!(f64::NAN.max(1.0), 1.0, "std::f64::max is the other way");
    }
}
