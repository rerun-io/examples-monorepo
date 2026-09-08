//! Square-root marginalization: the rank-revealing QR helper and the sliding
//! window mechanics that drive it.
//!
//! Ported from `include/basalt/vi_estimator/marg_helper.h`,
//! `src/vi_estimator/marg_helper.cpp`, the second half of
//! `src/vi_estimator/sqrt_keypoint_vio.cpp`'s `marginalize()` and
//! `src/vi_estimator/sqrt_ba_base.cpp`.
//!
//! ```text
//! marginalize()                     sqrt_keypoint_vio.cpp
//!   build the absolute ordering       :726-763
//!   linearize the window + prior      :905-942
//!   split keep / marg indices         :980-1003
//!   MargHelper::...                   :1069-1083
//!   setLinTrue on the newest state    :1085-1088   (trap 7)
//!   shrink frame_states/poses/lmdb    :1090-1118
//!   new prior order                   :1120-1137
//!   b -= H * delta                    :1170-1172   (trap 8)
//! ```
//!
//! **What is here and what is stage S8's.** This module takes the four sets the
//! schedule produced — which keyframes, poses and states leave, and which state
//! becomes the prior's newest block — as [`MarginalizeSchedule`] and does
//! everything downstream of that decision. The scoring at
//! `sqrt_keypoint_vio.cpp:767-880`, the `states_to_remove` count and the
//! keyframe bookkeeping are not ported here.
//!
//! **Why the QR and not the Schur complement.** The prior is stored as a
//! Jacobian `J_m` and a residual `r_m` rather than as `H` and `b`, and
//! marginalizing is one flat, rank-revealing Householder QR over the stacked
//! `[J_marg | J_keep]` (papers-part2 §12). Squaring the system to eliminate a
//! block would square its condition number, which is the whole point of the
//! square-root formulation; the QR never forms `JᵀJ` at all. basalt keeps the
//! squared form behind `vio_sqrt_marg` as the 2019 baseline the 2021 paper
//! compared against; the port carries only the square-root form, because
//! `SqrtKeypointVio::new` refuses the flag off (D68).

mod helper;
mod window;

pub use helper::{ReducedSystem, marginalize_helper_sqrt_to_sqrt};
pub use window::{
    MarginalizeInputs, MarginalizeOptions, MarginalizeOutput, MarginalizeSchedule, marginalize,
};

use crate::ba_base::BaError;
use crate::linearize::LinearizeError;
use crate::types::{FrameId, StateError};

/// One of the five things [`MarginalizeSchedule`] decides, named so a refusal
/// can say which of them was wrong.
///
/// The C++ builds all five out of the window itself
/// (`sqrt_keypoint_vio.cpp:724-880`), so every relationship between them is an
/// invariant of that construction rather than something checked; the port
/// checks them, and this is how it reports which one broke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleSet {
    /// `last_state_to_marg` (`:724`).
    LastStateToMarg,
    /// `kfs_to_marg` (`:766`).
    KfsToMarg,
    /// `poses_to_marg` (`:729`).
    PosesToMarg,
    /// `states_to_marg_all` (`:743`).
    StatesToMargAll,
    /// `states_to_marg_vel_bias` (`:742`).
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

/// What marginalization refuses to do.
///
/// Every variant replaces a C++ assertion, an `at()` that would throw, or an
/// unchecked index; the estimator runs with the GIL released, where a panic
/// aborts the process (decision D32).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MargError {
    /// `keep_size + marg_size == abs_H.cols()` (`marg_helper.cpp:47`, `:132`,
    /// `:253`).
    #[error("{keep} kept plus {marg} marginalized indices do not cover {total} columns")]
    IndexCountMismatch {
        /// Size of `idx_to_keep`.
        keep: usize,
        /// Size of `idx_to_marg`.
        marg: usize,
        /// Columns of the system.
        total: usize,
    },
    /// An index names a column the system does not have; C++ would read out of
    /// range.
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
    /// `Q2Jp.rows() == Q2r.rows()` (`marg_helper.cpp:254`), or a right-hand
    /// side that does not match a square system.
    #[error("the system has {rows} rows and the right-hand side {rhs}")]
    RhsLengthMismatch {
        /// Rows of the matrix.
        rows: usize,
        /// Rows of the vector.
        rhs: usize,
    },
    /// The prior's ordering disagrees with the window's, which C++ asserts
    /// block by block (`sqrt_keypoint_vio.cpp:736`, `:758-759`).
    #[error("the marginalization prior's ordering does not match the window at frame {frame_id}")]
    PriorOrderMismatch {
        /// The frame that disagrees.
        frame_id: FrameId,
    },
    /// `marg_data.H.cols() == marg_data.order.total_size`
    /// (`sqrt_keypoint_vio.cpp:1145`, `sqrt_ba_base.cpp:52`).
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
    /// `last_state_to_marg` was already frozen (`sqrt_keypoint_vio.cpp:1086`
    /// asserts it is not).
    #[error("frame {frame_id} is already at its linearization point")]
    AlreadyLinearized {
        /// The frame that was to be frozen.
        frame_id: FrameId,
    },
    /// A full state in the ordering is in none of the three sets and is not
    /// `last_state_to_marg` (`sqrt_keypoint_vio.cpp:999` asserts).
    #[error("state {frame_id} is neither marginalized nor the last state to marginalize")]
    UnscheduledState {
        /// The state the schedule forgot.
        frame_id: FrameId,
    },
    /// A block in the ordering is neither a pose nor a full state
    /// (`sqrt_keypoint_vio.cpp:990`, `sqrt_ba_base.cpp:91-93`).
    #[error("frame {frame_id} has a block of {size} rows, which is neither 6 nor 15")]
    UnexpectedBlockSize {
        /// The offending frame.
        frame_id: FrameId,
        /// Its block size.
        size: usize,
    },
    /// A schedule set names a frame the marginalization ordering does not hold
    /// as the kind of block that set is about — a pose block for
    /// `poses_to_marg`, a full state for the two state sets.
    ///
    /// C++ trusts the construction: `frame_states.at(id)` throws on a frame
    /// that is not there, `frame_poses.erase(id)` silently does nothing, and a
    /// state newer than `last_state_to_marg` is not in the ordering at all —
    /// so it is deleted without ever being marginalized
    /// (`sqrt_keypoint_vio.cpp:1090-1112`). Either way the window is already
    /// half rewritten by the time it shows.
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
    /// Two schedule sets C++ fills in mutually exclusive branches
    /// (`sqrt_keypoint_vio.cpp:745-750`) name the same frame.
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
    /// `sqrt_keypoint_vio.cpp:875-876` guarantees by adding every keyframe it
    /// picks to both. Without it the keyframe's landmarks are dropped
    /// (`:1114`) while the frame itself survives.
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
