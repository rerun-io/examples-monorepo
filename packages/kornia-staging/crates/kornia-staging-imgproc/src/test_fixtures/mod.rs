//! Shared deterministic inputs for downstream CPU/GPU contract tests.
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod bands;
mod flow;
mod images;
pub use bands::*;
pub use flow::*;
pub use images::*;

mod pgm;
pub use pgm::{read_pgm, Pgm};
