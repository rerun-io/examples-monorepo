//! Square-root landmark elimination, deterministic absolute-system assembly,
//! and the global square-root bundle adjustment (rootba `bal`).
mod assembly;
mod damping;
mod dense;
mod global;
mod landmark_qr;
mod lane;
mod pcg;
mod prior;
mod rel_pose;
pub use assembly::{eliminate_blocks, linearize_blocks};
pub use damping::LandmarkDamping;
pub use dense::{DenseBlock, DenseHbWorkspace};
pub use global::{
    solve_global, solve_global_with, GbaCamera, GbaCost, GbaError, GbaIteration, GbaLandmark,
    GbaObservation, GbaOptions, GbaProblem, GbaReport, GbaRigCamera, GbaStageSeconds,
    GbaTermination,
};
pub use landmark_qr::{BackSubstitution, LandmarkQr};
pub use lane::{LaneSolve, PcgLane, PcgLaneError, ReducedBlock, ReducedSystem};
pub use pcg::{
    pcg, pcg_with_dot, BlockJacobi, PcgOptions, PcgSummary, PcgTermination, GBA_CG_CHECK_EVERY,
};
pub use prior::PriorLinearization;
pub use rel_pose::{apply_left_increment, relative_pose_error, RelPoseConstraint};

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
