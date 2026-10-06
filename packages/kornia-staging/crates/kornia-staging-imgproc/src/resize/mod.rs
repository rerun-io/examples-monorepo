//! Integer image resizing.

/// Integer-factor area means with half-up rounding.
mod area;
pub use area::*;

/// Four-by-four pooling with half-even and unrounded output.
mod pool;
pub use pool::*;

mod kernels;
