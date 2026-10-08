//! Square-root landmark elimination and deterministic absolute-system assembly.
mod assembly;
mod dense;
mod landmark_qr;
mod prior;
pub use assembly::{eliminate_blocks, linearize_blocks};
pub use dense::{DenseBlock, DenseHbWorkspace};
pub use landmark_qr::{BackSubstitution, LandmarkQr};
pub use prior::PriorLinearization;

/// An invalid packed buffer or landmark layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SqrtBaError {
    /// A layout sum or product overflowed.
    #[error("landmark layout overflow")]
    LayoutOverflow,
    /// Storage would exceed the addressable byte capacity.
    #[error("landmark block {rows} x {cols} exceeds addressable capacity")]
    BlockTooLarge {
        /// Requested rows.
        rows: usize,
        /// Requested columns.
        cols: usize,
    },
    /// Active columns are not strictly increasing or exceed the pose width.
    #[error("active landmark pose columns are invalid")]
    ActiveColumns,
    /// Buffer lengths do not match the validated layout.
    #[error("landmark buffer does not match its layout")]
    Shape,
    /// A block cannot fit the destination system.
    #[error("system width {found} is smaller than {expected}")]
    SystemSize {
        /// Required width.
        expected: usize,
        /// Destination width.
        found: usize,
    },
}
