//! Feature extraction and decoding.

/// Log-quadratic heatmap peak decoding.
mod heatmap;
pub use heatmap::*;

mod cells;
pub use cells::*;
