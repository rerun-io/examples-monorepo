//! Square-root marginalization and sliding-window updates.
//! The schedule supplies removals; this module builds ordering, linearizes,
//! splits indices, eliminates variables, freezes the newest prior state,
//! shrinks the window, and re-anchors the residual.
//!
//! Keeping a Jacobian and residual avoids forming `JᵀJ` and squaring the condition
//! number. Only square-root priors are supported (D68). Scheduling and keyframe
//! scores belong to the estimator module.

mod helper;
mod window;

pub use helper::{ReducedSystem, marginalize_helper_sqrt_to_sqrt};
pub use window::{
    MarginalizeInputs, MarginalizeOptions, MarginalizeOutput, MarginalizeSchedule, marginalize,
};

use crate::ba_base::BaError;
use crate::linearize::LinearizeError;
use crate::types::{FrameId, StateError};

/// Schedule component named in a validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleSet {
    /// `last_state_to_marg`.
    LastStateToMarg,
    /// `kfs_to_marg`.
    KfsToMarg,
    /// `poses_to_marg`.
    PosesToMarg,
    /// `states_to_marg_all`.
    StatesToMargAll,
    /// `states_to_marg_vel_bias`.
    StatesToMargVelBias,
}

impl std::fmt::Display for ScheduleSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name: &str = match self {
            Self::LastStateToMarg => "last_state_to_marg",
            Self::KfsToMarg => "kfs_to_marg",
            Self::PosesToMarg => "poses_to_marg",
            Self::StatesToMargAll => "states_to_marg_all",
            Self::StatesToMargVelBias => "states_to_marg_vel_bias",
        };
        f.write_str(name)
    }
}

/// Typed marginalization failures prevent data-dependent panics (D32).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MargError {
    /// Kept and marginalized counts do not cover the system.
    #[error("{keep} kept plus {marg} marginalized indices do not cover {total} columns")]
    IndexCountMismatch {
        /// Size of `idx_to_keep`.
        keep: usize,
        /// Size of `idx_to_marg`.
        marg: usize,
        /// Columns of the system.
        total: usize,
    },
    /// An index names a nonexistent system column.
    #[error("index {index} is out of range for a system of {total} columns")]
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// Columns of the system.
        total: usize,
    },
    /// An index is in both sets, which would double-count a column.
    #[error("index {index} is both kept and marginalized")]
    IndexInBothSets {
        /// The offending index.
        index: usize,
    },
    /// `Q2Jp.rows() == Q2r.rows()`, or a right-hand
    /// side that does not match a square system.
    #[error("the system has {rows} rows and the right-hand side {rhs}")]
    RhsLengthMismatch {
        /// Rows of the matrix.
        rows: usize,
        /// Rows of the vector.
        rhs: usize,
    },
    /// The prior ordering disagrees with the window ordering.
    #[error("the marginalization prior's ordering does not match the window at frame {frame_id}")]
    PriorOrderMismatch {
        /// The frame that disagrees.
        frame_id: FrameId,
    },
    /// `marg_data.H.cols() == marg_data.order.total_size`
    #[error("the prior is {cols} columns wide but its ordering covers {total_size}")]
    PriorWidthMismatch {
        /// Columns of `H`.
        cols: usize,
        /// Rows the ordering covers.
        total_size: usize,
    },
    /// A frame named by the schedule or the ordering is not in the window.
    #[error("frame {frame_id} is not in the window")]
    FrameNotInWindow {
        /// The frame that is missing.
        frame_id: FrameId,
    },
    /// `last_state_to_marg` was already frozen (
    /// asserts it is not).
    #[error("frame {frame_id} is already at its linearization point")]
    AlreadyLinearized {
        /// The frame that was to be frozen.
        frame_id: FrameId,
    },
    /// A full state in the ordering is in none of the three sets and is not
    /// `last_state_to_marg` ( asserts).
    #[error("state {frame_id} is neither marginalized nor the last state to marginalize")]
    UnscheduledState {
        /// The state the schedule forgot.
        frame_id: FrameId,
    },
    /// A block in the ordering is neither a pose nor a full state
    #[error("frame {frame_id} has a block of {size} rows, which is neither 6 nor 15")]
    UnexpectedBlockSize {
        /// The offending frame.
        frame_id: FrameId,
        /// Its block size.
        size: usize,
    },
    /// A schedule set names a missing frame or the wrong block kind.
    /// Validate before mutation so no state can be deleted without marginalization.
    #[error(
        "{set} names frame {frame_id}, which the ordering does not hold as a {block}-row block"
    )]
    ScheduledFrameNotInOrdering {
        /// Which set named it.
        set: ScheduleSet,
        /// The offending frame.
        frame_id: FrameId,
        /// The block size that set requires.
        block: usize,
    },
    /// Mutually exclusive schedule sets name the same frame.
    #[error("frame {frame_id} is in both {first} and {second}")]
    ScheduleSetsOverlap {
        /// The first set.
        first: ScheduleSet,
        /// The second set.
        second: ScheduleSet,
        /// The frame both name.
        frame_id: FrameId,
    },
    /// `kfs_to_marg` is not a subset of `poses_to_marg`, which
    ///  guarantees by adding every keyframe it
    /// picks to both. Without it the keyframe's landmarks are dropped
    ///  while the frame itself survives.
    #[error("frame {frame_id} is marginalized as a keyframe but not as a pose")]
    KeyframeNotInPosesToMarg {
        /// The offending frame.
        frame_id: FrameId,
    },
    /// Something the linearizer refused.
    #[error(transparent)]
    Linearize(#[from] LinearizeError),
    /// Something the bundle-adjustment base refused.
    #[error(transparent)]
    Ba(#[from] BaError),
    /// Something a state type refused.
    #[error(transparent)]
    State(#[from] StateError),
}
