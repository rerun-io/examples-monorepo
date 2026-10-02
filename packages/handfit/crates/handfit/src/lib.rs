//! Native warm and cold UmeTrack fitting; no Python or torch in the solve.
pub mod cold;
pub mod generated;
pub mod lm;
pub mod model;
pub mod residual;
pub mod scale;
pub use lm::{fit, Config, FitError, FitResult, JacobianMode, Termination};
pub use model::{Model, Pose};
pub use nalgebra;
pub use residual::{View, Views, MAX_VIEWS};
