//! Shared CubeCL buffer layout and sampling operations.
pub mod layout;
pub mod sampling;
pub(crate) use sampling::launch_probe;
