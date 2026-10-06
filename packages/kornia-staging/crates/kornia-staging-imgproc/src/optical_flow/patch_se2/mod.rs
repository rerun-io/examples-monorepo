//! Mean-normalized inverse-compositional SE(2) patch alignment.
mod ldlt;
mod patch;
mod patterns;
mod se2;
mod simd;
pub use patch::*;
pub use patterns::*;
pub use se2::*;
#[cfg(any(test, feature = "test-oracles"))]
pub mod oracle;

#[cfg(test)]
mod tests;
