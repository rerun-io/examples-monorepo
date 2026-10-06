//! Image interpolation.

/// Scaled bilinear remapping with per-tap zero borders.
mod remap;
pub use remap::*;
