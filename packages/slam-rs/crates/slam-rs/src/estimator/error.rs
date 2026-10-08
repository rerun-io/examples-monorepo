//! Typed estimator failures and window roles.

#[cfg(doc)]
use super::{FrameOutcome, SqrtKeypointVio};
use crate::ba_base::BaError;
use crate::config::LinearizationType;
use crate::landmark::LandmarkError;
use crate::linearize::LinearizeError;
use crate::marg::MargError;
use crate::types::{FrameId, KeypointId, StateError};
use kornia_staging_sensors::SensorError;

/// What the eviction score wanted a keyframe as, for
/// [`EstimatorError::KeyframeNotInWindow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowRole {
    /// A pose block, `frame_poses`.
    Pose,
    /// A state block, `frame_states`.
    State,
    /// An entry in `num_points_kf`, the landmarks a keyframe hosts.
    HostedLandmarkCount,
}

impl std::fmt::Display for WindowRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name: &str = match self {
            Self::Pose => "pose",
            Self::State => "state",
            Self::HostedLandmarkCount => "hosted-landmark count",
        };
        f.write_str(name)
    }
}

/// Typed failures from the driver (D32).
/// Whether the window is unchanged depends on the call site, as described in
/// the module documentation, rather than on the error variant alone.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EstimatorError {
    /// `vio_linearization_type` is not `ABS_QR`, or `vio_sqrt_marg` is false:
    /// the other five combinations are out of scope (D13).
    #[error(
        "only ABS_QR with sqrt marginalization is ported, config says {linearization:?} sqrt_marg={sqrt_marg}"
    )]
    UnsupportedPath {
        /// What the config asked for.
        linearization: LinearizationType,
        /// Whether the config asked for the square-root prior.
        sqrt_marg: bool,
    },
    /// `vio_enforce_realtime` drops framesets, which Offline mode cannot do
    /// without letting arrival order reach a decision (D17).
    #[error("vio_enforce_realtime is not available in offline mode")]
    EnforceRealtime,
    /// `vio_max_states` or `vio_max_kfs` is not positive, so the window has no
    /// size.
    #[error("window sizes must be positive, got max_states={max_states} max_kfs={max_kfs}")]
    EmptyWindow {
        /// `vio_max_states`.
        max_states: i32,
        /// `vio_max_kfs`.
        max_kfs: i32,
    },
    /// A scalar the estimator divides by or takes the square root of is zero,
    /// negative or not finite: an observation or IMU standard deviation, the IMU
    /// update rate, or a damping bound.
    #[error("{field} must be positive and finite, got {value}")]
    NonPositiveScalar {
        /// The config or calibration field, spelled as the struct spells it.
        field: &'static str,
        /// What it holds, in the estimator's own scalar widened to `f64`.
        value: f64,
    },
    /// An initial prior weight is negative or not finite. Zero is a free gauge
    /// direction, which is a choice a caller may make; a negative weight has no
    /// square root and an infinite one has no prior.
    #[error("{field} must be non-negative and finite, got {value}")]
    NegativeScalar {
        /// The config field, spelled as the struct spells it.
        field: &'static str,
        /// What it holds.
        value: f64,
    },
    /// The damping bounds cross, so no `lambda` satisfies both (
    #[error("vio_lm_lambda_min {min} must not exceed vio_lm_lambda_max {max}")]
    DampingRangeReversed {
        /// `vio_lm_lambda_min`.
        min: f64,
        /// `vio_lm_lambda_max`.
        max: f64,
    },
    /// The rig has fewer than the two cameras requires,
    /// its intrinsics and extrinsics disagree, or a frameset carries a
    /// different number of cameras than the rig.
    #[error("expected {expected} cameras, got {actual}")]
    CameraCountMismatch {
        /// Cameras the rig is required or known to have.
        expected: usize,
        /// Cameras the rejected input carries.
        actual: usize,
    },
    /// Frame timestamps must strictly increase: asserts both the
    /// duplicate and the reordering, because a zero `dt` makes the
    /// preintegration invalid.
    #[error("frameset at {t_ns} ns does not follow the previous frameset at {previous_t_ns} ns")]
    NonMonotonicFrame {
        /// Timestamp of the last accepted frameset.
        previous_t_ns: i64,
        /// Timestamp of the rejected frameset.
        t_ns: i64,
    },
    /// The window disagrees with the marginalization prior's ordering.
    #[error("frame {frame_id} sits at {found:?} in the window but at {expected:?} in the prior")]
    PriorOrderMismatch {
        /// The disagreeing frame.
        frame_id: FrameId,
        /// `(index, size)` in the prior, or `None` when the prior has no entry.
        expected: Option<(usize, usize)>,
        /// `(index, size)` the window just built.
        found: (usize, usize),
    },
    /// An eviction candidate is missing from the window.
    #[error("keyframe {frame_id} is not in the window as a {wanted}")]
    KeyframeNotInWindow {
        /// The keyframe the score wanted.
        frame_id: FrameId,
        /// What the score wanted it as.
        wanted: WindowRole,
    },
    /// The previous state needed to predict the new state is missing.
    #[error("the previous state at {t_ns} ns is not in the window")]
    PreviousStateMissing {
        /// `last_state_t_ns`.
        t_ns: i64,
    },
    /// An IMU endpoint is in the ordering but absent from `frame_states`.
    /// Skipping it would silently change the objective and the accepted LM step.
    #[error("the imu factor over ({start_t_ns}, {end_t_ns}] ns has no state at {missing_t_ns} ns")]
    ImuFactorStateMissing {
        /// `start_timestamp_ns()`.
        start_t_ns: i64,
        /// `start_timestamp_ns() + dt_ns()`.
        end_t_ns: i64,
        /// Whichever endpoint the window is missing; the start when both are.
        missing_t_ns: i64,
    },
    /// An unconnected keypoint is missing when triangulation reads its pixel.
    /// Both maps come from the same frameset, so this violates an internal invariant.
    #[error("keypoint {kpt_id:?} is unconnected in camera {cam_id} but not in its keypoint map")]
    UnconnectedKeypointMissing {
        /// The camera whose map the id came from.
        cam_id: usize,
        /// The id `measure` filed as unconnected.
        kpt_id: KeypointId,
    },
    /// The state window is too short for the marginalization advance.
    #[error("{states} states cannot spare the {states_to_remove} the marginalization removes")]
    StateWindowTooShort {
        /// States in the window.
        states: usize,
        /// `states_to_remove`.
        states_to_remove: usize,
    },
    /// A window frame is absent from the ordering built for that window.
    #[error("frame {frame_id} is in the window but not in its ordering")]
    FrameNotInOrdering {
        /// The frame the increment could not be applied to.
        frame_id: FrameId,
    },
    /// The eviction loop found no keyframe to marginalize, which asserts
    /// is impossible ("the logic above is faulty").
    #[error("no keyframe could be selected for marginalization out of {candidates} candidates")]
    NoKeyframeToMarginalize {
        /// Keyframes the loop had to choose from.
        candidates: usize,
    },
    /// The linearization refused the problem.
    #[error("linearization: {0}")]
    Linearize(#[from] LinearizeError),
    /// The marginalization refused the schedule.
    #[error("marginalization: {0}")]
    Marginalize(#[from] MargError),
    /// The window or the error computation refused an input.
    #[error("bundle adjustment: {0}")]
    BundleAdjustment(#[from] BaError),
    /// The landmark database refused an observation.
    #[error("landmark database: {0}")]
    Landmark(#[from] LandmarkError),
    /// The preintegration refused a sample.
    #[error("imu: {0}")]
    Imu(#[from] SensorError),
    /// Frame-boundary IMU accumulation failed.
    #[error(transparent)]
    Accumulate(#[from] crate::imu::AccumulateError),
    /// A fixed-linearization state was frozen twice, or a delta was not zero
    /// when it was frozen.
    #[error("state: {0}")]
    State(#[from] StateError),
    /// The IMU queue ran dry while skipping forward to the frameset
    /// after the coverage test found a sample past it, which means
    /// [`SqrtKeypointVio::push_imu`]'s ordering invariant broke. Not
    /// [`FrameOutcome::NeedMoreImu`]: the skip has consumed samples by then, so
    /// the estimator is no longer untouched.
    #[error("the imu queue ran dry while skipping forward to the frameset at {t_ns} ns")]
    ImuQueueRanDry {
        /// The frameset being initialized on.
        t_ns: i64,
    },
    /// `linearizeProblem` reported `numerically_valid == false`, which
    /// prints as "did not expect numerical failure during linearization" and
    /// then fails the frame.
    #[error("linearization was not numerically valid at frame {t_ns} ns")]
    NumericallyInvalid {
        /// The frame being optimized.
        t_ns: i64,
    },
}
