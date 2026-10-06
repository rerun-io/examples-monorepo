//! Extension kernels for nonlinear least-squares solvers.
mod levenberg_marquardt;
pub use levenberg_marquardt::{
    marquardt_scaling, nielsen_damping, predicted_reduction, NielsenPolicy, ScalingFloor,
};

mod damped_solve;
pub use damped_solve::{solve_scaled_damped, DampedSolveError};

mod marginalization;
pub use marginalization::{marginalize, MarginalizationError, ReducedSystem};
