//! Algebra extensions prepared for kornia-algebra.
#![deny(missing_docs)]

/// Linear algebra kernels.
pub mod linalg;

/// Lie group precision and update extensions.
pub mod lie;

mod scalar;
pub use scalar::Scalar;

/// Optimization extensions.
pub mod optim;
