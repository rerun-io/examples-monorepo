//! Extension kernels for nonlinear least-squares solvers.
mod marginalization;
pub use marginalization::{marginalize, MarginalizationError, ReducedSystem};
