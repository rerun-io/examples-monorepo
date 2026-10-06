//! Sensor types prepared for kornia-slam/kornia-sensors.
#![deny(missing_docs)]
mod error;
mod frame;
/// Combined inertial readings and acceleration interpolation.
pub mod imu;
mod matcher;
pub use error::SensorError;
pub use frame::{CameraFrame, CaptureMeta, Frameset, SourceEvent};
pub use matcher::{FramesetMatcher, MatcherConfig, MatcherCounts};
