//! Midpoint IMU preintegration with nanosecond timestamps.
//! Navigation tangent order is translation, left rotation, velocity. Bias correction
//! pre-multiplies delta rotation. Noise inputs are discrete diagonal covariances:
//! square continuous standard deviations after multiplying by sqrt(sample_rate_hz).
mod combiner;
mod preintegration;
mod state;
pub use combiner::{CombinedImuSample, CombinerCounts, GapPolicy, ImuCombiner, ImuCombinerConfig};
pub use preintegration::{
    ImuNoise, ImuResidualJacobians, IntegratedImuMeasurement, PropagationJacobians,
};
pub use state::NavState;
