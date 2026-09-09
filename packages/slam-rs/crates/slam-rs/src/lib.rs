//! Visual-inertial odometry core.
//!
//! The crate is deliberately free of Python, Rerun and GPU code: it consumes
//! grayscale images and IMU samples and returns plain values. Python plumbing
//! (catalog feed, evaluation, logging) lives in the `slam_rs` package and the
//! bindings in `slam-rs-py`; `slam-rs-cli` is a placeholder binary whose only
//! working subcommand is `version`.

pub mod calib;
pub mod camera;
pub mod config;
pub mod frontend;
pub mod image;
pub mod lie;
pub mod pyramid;
pub mod types;

/// Version of the core, as declared in `crates/slam-rs/Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
