//! Extension kernels for nonlinear least-squares solvers.
mod marginalization;
pub use marginalization::{marginalize, MarginalizationError, ReducedSystem};

mod damped_solve;
pub use damped_solve::{solve_scaled_damped, DampedSolveError};
