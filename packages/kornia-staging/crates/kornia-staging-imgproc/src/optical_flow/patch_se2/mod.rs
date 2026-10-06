//! Mean-normalized inverse-compositional SE(2) patch alignment.
mod ldlt;
mod patch;
mod patterns;
mod se2;
mod simd;
pub use patch::PATCH_BORDER;
pub(crate) use patch::{build_patch_group, patch_increment_rows, patch_residual_taps};
#[cfg(any(test, feature = "test-oracles"))]
pub mod oracle;
pub(crate) use patterns::MAX_PATTERN_SIZE;
pub use patterns::{PatchError, Pattern, Pattern51, Pattern52};
pub(crate) use se2::se2_exp;
pub use se2::{AffineCompact2, AffineCompact2f};

#[cfg(test)]
mod tests;
