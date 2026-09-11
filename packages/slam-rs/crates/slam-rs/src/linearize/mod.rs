//! Square-root linearization into a reduced camera system.
//! Only `ABS_QR` with square-root marginalization is supported (D13, D68).
//! Each iteration linearizes, eliminates landmarks, builds `H` and `b`, solves
//! the damped system, negates the increment, and back-substitutes landmarks.
//! Damping belongs to the dense solve; landmark/pose damping and Jacobian scaling
//! are absent. Landmark reductions use fixed order independent of Rayon width.

mod abs_qr;
mod dense_hb;
mod landmark_block;
mod relative_pose;

pub(crate) use relative_pose::linearize_relative_pose;

pub use abs_qr::{ImuInput, LinearizationAbsQR, LinearizationInputs, LinearizationOptions};
pub use dense_hb::DenseHbWorkspace;
pub use landmark_block::{
    DenseHbScratch, LandmarkBlock, LandmarkBlockOptions, LandmarkBlockState, compute_error_weight,
};

use nalgebra::{Matrix4, Matrix6};

use crate::ba_base::BaError;
use crate::lie::LieScalar;
use crate::types::{CamId, FrameId, LandmarkId};

/// `RelPoseLin<Scalar>` : one (host, target) pair's
/// relative pose and the two 6x6 Jacobians of that pose against the two absolute
/// pose increments.
///
/// Hoisted per pair rather than per observation, because every landmark hosted
/// in the same image and seen in the same image shares it
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelPoseLin<S: LieScalar> {
    /// `T_t_h`, as a 4x4 so the residual can multiply the homogeneous
    /// landmark straight through.
    pub t_t_h: Matrix4<S>,
    /// `d_rel_d_h`.
    pub d_rel_d_h: Matrix6<S>,
    /// `d_rel_d_t`.
    pub d_rel_d_t: Matrix6<S>,
}

impl<S: LieScalar> Default for RelPoseLin<S> {
    /// `RelPoseLin()`'s in-class initialisers: all three zero.
    fn default() -> Self {
        Self {
            t_t_h: Matrix4::zeros(),
            d_rel_d_h: Matrix6::zeros(),
            d_rel_d_t: Matrix6::zeros(),
        }
    }
}

/// Typed linearization failures (D32).
/// Check allocation arithmetic once so repeated observation indexing is valid.
/// Non-finite Jacobians are zeroed with a warning, and singular landmark factors
/// skip back-substitution with a warning; neither is a fatal error (trap 11).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LinearizeError {
    /// A landmark's host frame is not in the absolute ordering
    /// ( asserts).
    #[error("landmark block host frame {frame_id} is not in the absolute ordering")]
    HostNotInOrdering {
        /// The host frame.
        frame_id: FrameId,
    },
    /// No relative pose was allocated for a (host, target) pair the landmark
    /// database says exists ( asserts).
    #[error("no relative pose for host ({host}, {host_cam}) -> target ({target}, {target_cam})")]
    MissingRelativePose {
        /// Host frame.
        host: FrameId,
        /// Host camera.
        host_cam: CamId,
        /// Target frame.
        target: FrameId,
        /// Target camera.
        target_cam: CamId,
    },
    /// A landmark's observation map lost an entry between allocation and
    /// linearization.
    #[error("landmark {lm_id:?} has no observation in ({target}, {target_cam})")]
    MissingObservation {
        /// The landmark.
        lm_id: LandmarkId,
        /// Target frame.
        target: FrameId,
        /// Target camera.
        target_cam: CamId,
    },
    /// A camera id is not in the calibration.
    #[error("camera {cam_id} is out of range: the rig has {camera_count}")]
    UnknownCamera {
        /// The camera asked for.
        cam_id: CamId,
        /// How many the rig has.
        camera_count: usize,
    },
    /// The block layout arithmetic of overflowed.
    #[error("landmark block layout arithmetic overflowed")]
    LayoutOverflow,
    /// Padded column count is not divisible by four.
    #[error("landmark block has {num_cols} columns, which is not a multiple of 4")]
    UnalignedBlock {
        /// The column count that failed the check.
        num_cols: usize,
    },
    /// A pose block would be written past the end of the pose columns.
    #[error("pose block at {offset} does not fit in {total_size} pose columns")]
    PoseBlockOutOfRange {
        /// The offset asked for.
        offset: usize,
        /// The ordering's total size.
        total_size: usize,
    },
    /// A method was called out of state-machine order.
    #[error("landmark block is {found:?}, expected {expected:?}")]
    WrongState {
        /// What the operation needed.
        expected: LandmarkBlockState,
        /// What the block was.
        found: LandmarkBlockState,
    },
    /// A stacked system was handed in at the wrong size.
    #[error("stacked system has {found} rows or columns, expected {expected}")]
    StackedSystemSize {
        /// What the block needed.
        expected: usize,
        /// What it got.
        found: usize,
    },
    /// `backSubstitute` was given a pose increment of the wrong length
    /// ( asserts).
    #[error("pose increment has {found} rows, expected {expected}")]
    PoseIncrementSize {
        /// The ordering's total size.
        expected: usize,
        /// What was passed.
        found: usize,
    },
    /// A landmark the database listed is not in it.
    #[error("landmark {0:?} is not in the database")]
    UnknownLandmark(LandmarkId),
    /// An IMU measurement names a frame the window does not have.
    #[error("IMU measurement spans frames {start} to {end}, which the window does not have")]
    UnknownImuFrames {
        /// Start timestamp.
        start: FrameId,
        /// End timestamp.
        end: FrameId,
    },
    /// An IMU endpoint timestamp overflows i64.
    #[error("IMU interval from {start} ns lasting {dt_ns} ns overflows the timestamp")]
    ImuIntervalOverflow {
        /// Start timestamp.
        start: FrameId,
        /// The measurement's duration.
        dt_ns: i64,
    },
    /// An IMU endpoint has a six-column pose slot instead of the required 15-column state.
    #[error("frame {frame} carries a {size}-column block, but an IMU factor needs 15")]
    ImuStateNotFullSize {
        /// The frame that disagrees.
        frame: FrameId,
        /// The size the ordering gave it.
        size: usize,
    },
    /// A landmark block's buffer would not fit in memory. `DMatrix::zeros`
    /// multiplies the two dimensions unchecked.
    #[error("a landmark block of {rows} x {cols} cannot be allocated")]
    BlockTooLarge {
        /// Rows asked for.
        rows: usize,
        /// Columns asked for.
        cols: usize,
    },
    /// Prior and window block orderings disagree.
    #[error("the marginalization prior's ordering does not match the window's at frame {frame_id}")]
    MargOrderMismatch {
        /// The frame that disagrees.
        frame_id: FrameId,
    },
    /// Something the bundle-adjustment base refused.
    #[error(transparent)]
    Ba(#[from] BaError),
}
