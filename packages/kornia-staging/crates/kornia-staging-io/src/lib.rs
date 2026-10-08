//! Capture and video IO prepared for kornia-io.
#![deny(missing_docs)]

/// Linux multi-planar V4L2 capture with leased raw planes.
#[cfg(all(target_os = "linux", feature = "v4l"))]
pub mod v4l;

/// Annex-B parsing and optional subprocess encoding.
pub mod video;
