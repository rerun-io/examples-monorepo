//! Image interpolation.

/// Scaled bilinear remapping with per-tap zero borders.
mod remap;
pub use remap::*;

mod sample;
pub use sample::{in_bounds_u16, sample_bilinear_u16, sample_bilinear_with_gradient_u16};
