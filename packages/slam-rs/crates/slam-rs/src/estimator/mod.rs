//! The sliding-window VIO driver: `SqrtKeypointVioEstimator<Scalar>`.
//!
//! A port of `src/vi_estimator/sqrt_keypoint_vio.cpp` and
//! `include/basalt/vi_estimator/sqrt_keypoint_vio.h` on the ABS_QR plus
//! square-root-marginalization path only (D13), composing what the stages below
//! already built: [`crate::ba_base::BundleAdjustmentBase`] for the window and
//! the reprojection error, [`crate::imu::IntegratedImuMeasurement`] for the
//! preintegration, [`crate::linearize::LinearizationAbsQR`] for the linearized
//! problem and [`crate::marg::marginalize`] for the prior.
//!
//! ## What the driver itself is
//!
//! Three functions, and they are the whole algorithm:
//!
//! * [`SqrtKeypointVio::process_frame`] is `proc_func`'s loop body
//!   (`:263-355`): initialise from one accelerometer sample if this is the first
//!   frame, preintegrate the IMU samples that fall in `(prev_t, curr_t]`, then
//!   `measure`.
//! * `SqrtKeypointVio::measure` (`:422-575`) predicts the new state, files the
//!   observations, votes on a keyframe, triangulates what the window does not
//!   yet know, and calls `optimize_and_marg`.
//! * `optimize` (`optimize.rs`, `:1201-1639`) is the Levenberg–Marquardt loop
//!   and `marginalize` (`schedule.rs`, `:707-1198`) is the schedule plus the
//!   call into the square-root helper.
//!
//! ## Threading: none (D17)
//!
//! basalt runs this on its own thread behind a bounded `vision_data_queue` and
//! a 3000-deep `imu_data_queue`, and drops framesets when
//! `vio_enforce_realtime` is set. Offline mode has no threads and no drops:
//! [`SqrtKeypointVio::push_imu`] appends to an unbounded buffer and
//! `process_frame` runs to completion in the calling thread, so no queue state
//! can reach an estimator decision. `vio_enforce_realtime` is therefore refused
//! at construction rather than silently ignored — `src/vio.cpp:307-309` forces
//! it off for offline replay anyway.
//!
//! ## What is deliberately not here
//!
//! `scheduleResetState`/`resetState` (`:120-195`): the VIT path only reaches it
//! from Monado's reset request, and `measure` returning an error is the port's
//! signal instead. `out_marg_queue` (`:952-978`), which exports `MargData` for
//! the NFR mapper that D13 puts out of scope. `takeLongTermKeyframe` is
//! implemented ([`SqrtKeypointVio::take_long_term_keyframe`]) because the
//! `take_ltkf` branch inside `measure` is on the shipped path, but nothing in
//! the VIO calls it. The visualization payloads (`:643-663`) become
//! [`SqrtKeypointVio::snapshot`], which returns values; the Rerun rung that logs
//! them is stage S9's (D03).
//!
//! ## Where an error leaves the window
//!
//! Three classes, and only the first is retryable. The class is the **call site**, not the variant:
//! [`EstimatorError::BundleAdjustment`] and [`EstimatorError::State`] are each raised from more
//! than one, and the sites fall in different classes.
//!
//! 1. **Raised before anything moves**, so the estimator is untouched and the
//!    caller may retry with a corrected input: [`EstimatorError::CameraCountMismatch`]
//!    and [`EstimatorError::NonMonotonicFrame`] from [`SqrtKeypointVio::process_frame`]'s
//!    validation, and [`EstimatorError::UnsupportedPath`], [`EstimatorError::EnforceRealtime`],
//!    [`EstimatorError::EmptyWindow`] and one [`EstimatorError::BundleAdjustment`] site from
//!    [`SqrtKeypointVio::new`]: [`BaError::Camera`], raised while
//!    [`BundleAdjustmentBase::new`] resolves the rig's projection models
//!    (`ba_base.rs:647`, `camera.rs:1058`), which returns before an estimator
//!    exists at all, so a corrected calibration may be passed to a new one.
//! 2. **Raised after the IMU queue was consumed but before the new state was
//!    filed.** [`EstimatorError::ImuQueueRanDry`] and [`EstimatorError::Imu`] come out of the
//!    preintegration loops, which have already popped samples; and
//!    [`EstimatorError::PreviousStateMissing`] comes out of `measure`'s prediction, after
//!    the same pops. The window still holds the frames it did, but the samples
//!    that interval needed are gone, so the same frameset can never be
//!    integrated again: not retryable either.
//! 3. **Raised after the new state, its observations and its preintegration
//!    were inserted** — every remaining variant, and all but one of the sites
//!    inside `measure`: the window-invariant breaks, `NumericallyInvalid` from
//!    the LM loop, and anything `Linearize`, `Marginalize`, `BundleAdjustment`,
//!    `Landmark` or `State` refuses there. The window has advanced by one
//!    frameset while `prev_frame` has not, so retrying the same frameset would
//!    file its observations twice.
//!
//!    One class-3 site sits before `measure`: `process_frame`'s initialization
//!    pushes the first ordering entry (`:281-283`) after it has filed the first
//!    state, so its [`EstimatorError::State`] would leave that same advanced window. It
//!    cannot fire — the push is the first into a fresh [`AbsOrderMap`], with
//!    nothing for `DuplicateFrame` to collide with and a fixed
//!    `POSE_VEL_BIAS_SIZE` that cannot overflow the offset — and stays a `?`
//!    because D32 leaves no room for the `unwrap` that would replace it.
//!
//! basalt has one answer to classes 2 and 3: it resets the whole estimator
//! (`proc_func`'s `return false`, `scheduleResetState` at `:120-195`), which
//! this port does not have. A caller that sees one of them must rebuild the
//! estimator. Stage S9's Realtime mode is where the reset belongs (D5 of the S8
//! simplify list).

mod frame_update;
mod optimize;
mod schedule;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Vector2, Vector3, Vector4};

use crate::ba_base::{BaError, BundleAdjustmentBase, triangulate};
use crate::calib::Calibration;
use crate::camera::CameraEnum;
use crate::config::{LinearizationType, VioConfig};
use crate::duration_ns;
use crate::imu::{
    ImuError, ImuLinData, ImuNoise, ImuSample, IntegratedImuMeasurement, Popped, gravity,
    gravity_from_first_accel,
};
use crate::landmark::{Landmark, LandmarkError, StereographicParam};
use crate::lie::{LieScalar, Se3, eigen_maxi};
use crate::linearize::LinearizeError;
use crate::marg::MargError;
use crate::types::{
    AbsOrderMap, FrameId, KeypointId, LandmarkId, MargLinData, POSE_VEL_BIAS_SIZE,
    PoseVelBiasState, PoseVelBiasStateWithLin, PoseVelState, StateError, TimeCamId,
};

pub use frame_update::{FrameUpdateDecline, FrameUpdateOutcome};
use frame_update::{FrameUpdateResult, FrameUpdateScratch};
use optimize::OptimizeScratch;
pub use optimize::{LmIteration, LmTermination};
use schedule::MarginalizationOutcome;
pub use schedule::{EvictionReason, KeyframeEviction, MarginalizationStats};

/// What the eviction score wanted a keyframe as, for
/// [`EstimatorError::KeyframeNotInWindow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowRole {
    /// A pose block, `frame_poses` (`:794`, `:849`).
    Pose,
    /// A state block, `frame_states` (`:854`).
    State,
    /// An entry in `num_points_kf`, the landmarks a keyframe hosts (`:827`).
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

/// Everything the driver can refuse.
///
/// basalt's equivalents are `BASALT_ASSERT`s, an `std::out_of_range` from a
/// `.at()`, or a `return false` that makes `proc_func` reset the whole state.
/// Under D32 none of them may panic on data, so each becomes a variant here.
///
/// **Which errors leave the window where** is a property of the call site, not
/// of the variant, and is set out under "Where an error leaves the window" in
/// the module header.
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
    /// The damping bounds cross, so no `lambda` satisfies both (`:1415`,
    /// `:1595`).
    #[error("vio_lm_lambda_min {min} must not exceed vio_lm_lambda_max {max}")]
    DampingRangeReversed {
        /// `vio_lm_lambda_min`.
        min: f64,
        /// `vio_lm_lambda_max`.
        max: f64,
    },
    /// The rig has fewer than the two cameras `optical_flow.h:210` requires,
    /// its intrinsics and extrinsics disagree, or a frameset carries a
    /// different number of cameras than the rig.
    #[error("expected {expected} cameras, got {actual}")]
    CameraCountMismatch {
        /// Cameras the rig is required or known to have.
        expected: usize,
        /// Cameras the rejected input carries.
        actual: usize,
    },
    /// Frame timestamps must strictly increase: `:309-313` asserts both the
    /// duplicate and the reordering, because a zero `dt` makes the
    /// preintegration invalid.
    #[error("frameset at {t_ns} ns does not follow the previous frameset at {previous_t_ns} ns")]
    NonMonotonicFrame {
        /// Timestamp of the last accepted frameset.
        previous_t_ns: i64,
        /// Timestamp of the rejected frameset.
        t_ns: i64,
    },
    /// The window disagrees with the marginalization prior's ordering
    /// (`:1227`, `:1237`). C++ asserts, or throws out of `.at()` when the frame
    /// is missing from the prior altogether.
    #[error("frame {frame_id} sits at {found:?} in the window but at {expected:?} in the prior")]
    PriorOrderMismatch {
        /// The disagreeing frame.
        frame_id: FrameId,
        /// `(index, size)` in the prior, or `None` when the prior has no entry.
        expected: Option<(usize, usize)>,
        /// `(index, size)` the window just built.
        found: (usize, usize),
    },
    /// A keyframe the eviction score reads is missing from the window
    /// (`:824`, `:845-856`, `:794`) — `.at()` calls that C++ would throw out
    /// of.
    #[error("keyframe {frame_id} is not in the window as a {wanted}")]
    KeyframeNotInWindow {
        /// The keyframe the score wanted.
        frame_id: FrameId,
        /// What the score wanted it as.
        wanted: WindowRole,
    },
    /// `measure` predicts the new state from `frame_states.at(last_state_t_ns)`
    /// (`:428`), which C++ throws out of when the previous state has already
    /// been marginalized.
    #[error("the previous state at {t_ns} ns is not in the window")]
    PreviousStateMissing {
        /// `last_state_t_ns`.
        t_ns: i64,
    },
    /// An IMU factor's endpoint is in the ordering but not in `frame_states`
    /// (`sc_ba_base.cpp:671-672`, two `.at()` calls C++ throws out of).
    /// Skipping the factor would drop its residual from the true cost and so
    /// change which LM step is accepted, silently.
    #[error("the imu factor over ({start_t_ns}, {end_t_ns}] ns has no state at {missing_t_ns} ns")]
    ImuFactorStateMissing {
        /// `get_start_t_ns()`.
        start_t_ns: i64,
        /// `get_start_t_ns() + get_dt_ns()`.
        end_t_ns: i64,
        /// Whichever endpoint the window is missing; the start when both are.
        missing_t_ns: i64,
    },
    /// A keypoint `measure` recorded as unconnected is missing from the camera's
    /// own keypoint map when the triangulation reads its pixel back (`:514`'s
    /// `opt_flow_meas->keypoints.at(i).at(lm_id)`, which C++ throws out of).
    /// The two come from the same frameset a few lines apart, so a miss means
    /// the frameset changed under the loop.
    #[error("keypoint {kpt_id:?} is unconnected in camera {cam_id} but not in its keypoint map")]
    UnconnectedKeypointMissing {
        /// The camera whose map the id came from.
        cam_id: usize,
        /// The id `measure` filed as unconnected.
        kpt_id: KeypointId,
    },
    /// The state window is shorter than the marginalization's own advance
    /// (`:724`): C++ advances the iterator past `end()` and dereferences it.
    #[error("{states} states cannot spare the {states_to_remove} the marginalization removes")]
    StateWindowTooShort {
        /// States in the window.
        states: usize,
        /// `states_to_remove` (`:720-724`).
        states_to_remove: usize,
    },
    /// A frame in the window is missing from the ordering `optimize` built
    /// from that same window a few lines earlier (`:1468`, `:1472`, both
    /// `.at()` calls C++ would throw out of).
    #[error("frame {frame_id} is in the window but not in its ordering")]
    FrameNotInOrdering {
        /// The frame the increment could not be applied to.
        frame_id: FrameId,
    },
    /// The eviction loop found no keyframe to marginalize, which `:872` asserts
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
    Imu(#[from] ImuError),
    /// A fixed-linearization state was frozen twice, or a delta was not zero
    /// when it was frozen.
    #[error("state: {0}")]
    State(#[from] StateError),
    /// The IMU queue ran dry while skipping forward to the frameset (`:265-271`)
    /// after the coverage test found a sample past it, which means
    /// [`SqrtKeypointVio::push_imu`]'s ordering invariant broke. Not
    /// [`FrameOutcome::NeedMoreImu`]: the skip has consumed samples by then, so
    /// the estimator is no longer untouched.
    #[error("the imu queue ran dry while skipping forward to the frameset at {t_ns} ns")]
    ImuQueueRanDry {
        /// The frameset being initialized on.
        t_ns: i64,
    },
    /// `linearizeProblem` reported `numerically_valid == false`, which `:1301`
    /// prints as "did not expect numerical failure during linearization" and
    /// then fails the frame.
    #[error("linearization was not numerically valid at frame {t_ns} ns")]
    NumericallyInvalid {
        /// The frame being optimized.
        t_ns: i64,
    },
}

/// `OpticalFlowResult` as the estimator reads it (`optical_flow.h:186-215`).
///
/// The frontend produces much more — the input images, the responses, the
/// pyramid levels, the timing block — and the backend reads exactly this: one
/// map of keypoint id to observed pixel per camera, plus the frameset
/// timestamp. Keeping the estimator's input this narrow is what lets the oracle
/// gate replay the **C++'s own** flow stream into the Rust window without a
/// frontend in the loop.
///
/// The pixels are `f32` whatever the estimator's scalar is, because
/// `AffineCompact2f` is: `:437` and `:511` cast them to `Scalar` at the point of
/// use, so an `f64` estimator sees `f32` values widened, never `f64` precision
/// the frontend never had.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowObservations {
    /// Frameset timestamp, nanoseconds on the IMU clock.
    pub t_ns: i64,
    /// `keypoints[cam]`: the ids this camera tracked, and where.
    pub cameras: Vec<BTreeMap<KeypointId, Vector2<f32>>>,
}

impl FlowObservations {
    /// An empty result for `num_cameras` cameras.
    pub fn new(t_ns: i64, num_cameras: usize) -> Self {
        Self {
            t_ns,
            cameras: vec![BTreeMap::new(); num_cameras],
        }
    }
}

/// What one `process_frame` call did.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameOutcome<S: LieScalar> {
    /// The buffered IMU samples do not yet reach past the frameset, so nothing
    /// was consumed and the window is unchanged. basalt's `pop` blocks here;
    /// Offline mode returns instead (D17).
    NeedMoreImu,
    /// The frame was measured. The statistics are per frame and are the input
    /// to the S9 Rerun rung.
    Measured(Box<FrameStats<S>>),
}

/// Nanosecond marks basalt pushes into the frame's `TimeStats`
/// (`:1627-1636`), as durations rather than cumulative timestamps.
///
/// Wall-clock measurements, so they differ run to run. Nothing reads them: they
/// are reported, never compared, which is what keeps `process_frame`
/// bit-reproducible (D17).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageTimings {
    /// IMU integration before measure plus state prediction and append inside it.
    pub predict_ns: u64,
    /// Keyframe decision and landmark initialization; zero on other frames.
    pub keyframe_ns: u64,
    /// The whole LM loop, including all four detailed optimization stages.
    pub optimize_ns: u64,
    /// `linearizeProblem` plus `performQR`, summed over the LM iterations.
    pub linearize_ns: u64,
    /// `get_dense_H_b` plus the damped LDLT solve.
    pub solver_ns: u64,
    /// `backSubstitute`.
    pub back_substitution_ns: u64,
    /// The true-cost recomputation.
    pub error_ns: u64,
    /// The whole marginalization, including its own linearization.
    pub marginalize_ns: u64,
    /// The whole `measure`.
    pub measure_ns: u64,
}

/// Everything one frame decided, in one value.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStats<S: LieScalar> {
    /// Frameset timestamp.
    pub t_ns: i64,
    /// `connected[cam]`, observations of landmarks the window already hosts.
    pub connected: Vec<usize>,
    /// `unconnected_obs[cam].size()`, keypoints the window has never seen.
    pub unconnected: Vec<usize>,
    /// Whether `take_kf` was set when this frame arrived (`:473`).
    pub took_keyframe: bool,
    /// Whether the keyframe vote of `:454-456` fired on this frame.
    pub keyframe_vote: bool,
    /// `frames_after_kf` (`:202`) after the update: the vote's rate limiter,
    /// zero on a keyframe and one more than the last frame otherwise.
    pub frames_after_kf: i32,
    /// `num_points_added` (`:552`), zero on a non-keyframe.
    pub num_points_added: usize,
    /// Keyframes after the update, oldest first.
    pub kf_ids: Vec<FrameId>,
    /// Long-term keyframes after the update.
    pub ltkfs: Vec<FrameId>,
    /// `lmdb.numLandmarks()` after `measure`.
    pub num_landmarks: usize,
    /// `lmdb.numObservations()` after `measure`.
    pub num_observations: usize,
    /// Landmarks `vio_marg_lost_landmarks` would drop this frame.
    pub num_lost_landmarks: usize,
    /// `opt_started` (`:1207`) after this frameset: false until five states
    /// have accumulated, true from the first linearization on.
    pub opt_started: bool,
    /// One entry per LM step, accepted or rejected, in order.
    pub lm: Vec<LmIteration<S>>,
    /// Why the LM loop stopped.
    pub termination: LmTermination,
    /// The marginalization, when the trigger of `:717` fired.
    pub marginalization: Option<MarginalizationStats>,
    /// What D76's frame update did with this frameset: never attempted, taken,
    /// or refused by a named precondition.
    pub frame_update: FrameUpdateOutcome,
    /// Wall-clock stage marks; see [`StageTimings`].
    pub timings: StageTimings,
}

/// The nine degrees of freedom a 15-dof state carries beyond its pose.
///
/// A pose block (`frame_poses`) has none of them, a state block
/// (`frame_states`) has all three, so they travel as one value rather than as
/// three `Option`s that are always all-`Some` or all-`None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelBias<S: LieScalar> {
    /// World-frame velocity.
    pub vel_w_i: Vector3<S>,
    /// Gyroscope bias.
    pub bias_gyro: Vector3<S>,
    /// Accelerometer bias.
    pub bias_accel: Vector3<S>,
}

/// One window state, as the S9 Rerun rung needs it.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowState<S: LieScalar> {
    /// State timestamp.
    pub t_ns: i64,
    /// `T_w_i`, the rig pose in the world frame.
    pub t_w_i: Se3<S>,
    /// The nine dof beyond the pose, `None` for a pose-only block.
    pub vel_bias: Option<VelBias<S>>,
    /// Whether the linearization point is frozen (`imu_types.h:109`).
    pub linearized: bool,
    /// Whether this frame is a keyframe.
    pub keyframe: bool,
    /// Whether this frame is a long-term keyframe.
    pub long_term_keyframe: bool,
    /// `frame_idx`, the monotonic index basalt keeps for the UI (`:207`).
    pub frame_index: Option<usize>,
}

/// One landmark as the V2 rung draws it.
///
/// The host is carried because the rung colours the point cloud by the keyframe
/// that hosts it: the position alone cannot say which frame's bearing it is a
/// distance along, and `lmdb` keys on the host rather than storing it per point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotLandmark<S: LieScalar> {
    /// The landmark's id, which is the id of the keypoint that spawned it.
    pub id: LandmarkId,
    /// Host keyframe and camera: the image the inverse distance is measured from.
    pub host: TimeCamId,
    /// Position in the world frame, metres.
    pub position_w: Vector3<S>,
}

/// The window and its landmarks, for the V2 visual-validation rung (D51).
///
/// `getAllPosesMap`, `get_current_points` and the `VioVisualizationData` fields
/// (`:643-663`) collapsed into one value the caller reads once per frame. The
/// core logs nothing itself (D03).
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSnapshot<S: LieScalar> {
    /// Frameset timestamp of the newest state.
    pub t_ns: i64,
    /// The 15-dof states, oldest first.
    pub states: Vec<WindowState<S>>,
    /// The pose-only blocks, oldest first.
    pub poses: Vec<WindowState<S>>,
    /// Landmarks the window currently holds, in `lmdb` order.
    pub landmarks: Vec<SnapshotLandmark<S>>,
    /// Frames the last marginalization removed from the window.
    pub marginalized: Vec<FrameId>,
}

/// LM damping, the four fields of `sqrt_keypoint_vio.h:241` in one place.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LmDamping<S: LieScalar> {
    /// `lambda`, reset to `vio_lm_lambda_initial` every frame (D11, `:1249`).
    lambda: S,
    /// `min_lambda`, the floor of the damped diagonal (`:1415`).
    min_lambda: S,
    /// `max_lambda`; exceeding it terminates the frame (`:1595`).
    max_lambda: S,
    /// `lambda_vee`, the Nielsen escalation factor, reset to 2 on an accept.
    lambda_vee: S,
}

/// `vee_factor` and `initial_vee`, both the compile-time constant 2.0
/// (`sqrt_keypoint_vio.h:239-240`), not config fields.
const VEE_FACTOR: f64 = 2.0;

impl<S: LieScalar> LmDamping<S> {
    /// `:1557-1562`: Nielsen's update after a step the objective accepted.
    ///
    /// `std::pow<Scalar>(x, 3)` deduces the exponent as `int`, so
    /// `__promote_2<Scalar, int>` is `double` in both instantiations and the
    /// power and the `1 −` happen in `double` before narrowing back — which is
    /// why this is `to_f64().powf(3.0)` and not `x * x * x`. Both maxima are
    /// `eigen_maxi` because `cwiseMax` is `numext::maxi`, which keeps a NaN on
    /// the left where `f32::max` would drop it.
    fn accept(&mut self, relative_decrease: S) {
        let x: S = S::from_literal(2.0) * relative_decrease - S::one();
        let gain: S = S::from_literal(1.0 - x.to_f64().powf(3.0));
        let floor: S = S::one() / S::from_literal(3.0);
        self.lambda *= eigen_maxi(floor, gain);
        self.lambda = eigen_maxi(self.min_lambda, self.lambda);
        self.lambda_vee = S::from_literal(VEE_FACTOR);
    }

    /// `:1585-1586`: the geometric escalation after a rejected step, which the
    /// damped solve's own retry on a non-finite increment (`:1424-1425`) makes
    /// with the same two lines.
    fn escalate(&mut self) {
        self.lambda = self.lambda_vee * self.lambda;
        self.lambda_vee *= S::from_literal(VEE_FACTOR);
    }

    /// `:1595`: whether the escalation has taken the frame past `max_lambda`.
    ///
    /// Both loops ask it after the rollback, where C++ asks it; nothing between
    /// the escalation and the question touches the damping.
    fn exhausted(&self) -> bool {
        self.lambda > self.max_lambda
    }
}

/// `:1565-1568`: whether an accepted step is the last one the frame takes.
///
/// Both tolerances are hard-coded in C++ too. The window solve and D76's frame
/// update ask this of their own `f_diff` and `step_norminf`, and there is one
/// predicate so the two schedules cannot come to converge on different terms.
fn lm_converged<S: LieScalar>(f_diff: S, step_norminf: S) -> bool {
    (f_diff > S::zero() && f_diff < S::from_literal(optimize::FUNCTION_TOLERANCE))
        || step_norminf < S::from_literal(optimize::STEP_TOLERANCE)
}

/// `SqrtKeypointVioEstimator<Scalar>` (`sqrt_keypoint_vio.h:50-248`).
#[derive(Debug, Clone)]
pub struct SqrtKeypointVio<S: LieScalar> {
    /// The sliding window: `frame_states`, `frame_poses`, `lmdb` and the
    /// calibration. Crate-visible because [`crate::marg::marginalize`] takes
    /// it by `&mut` and [`Self::snapshot`] reads it; outside the crate the
    /// snapshot is the window's only view.
    pub(crate) ba: BundleAdjustmentBase<S>,

    /// `prev_frame` (`:198`), the frameset the last `measure` consumed.
    prev_frame: Option<Arc<FlowObservations>>,
    /// `take_kf` (`:201`), true at construction so the first frame is a
    /// keyframe (`:61`).
    take_kf: bool,
    /// `frames_after_kf` (`:202`), the rate limiter of `:455`.
    frames_after_kf: i32,
    /// `frame_count` (`:203`), the source of `frame_idx`.
    frame_count: usize,
    /// `kf_ids` (`:204`).
    kf_ids: BTreeSet<FrameId>,
    /// `ltkfs` (`:205`), exempt from the `max_kfs` budget (`:717`).
    ltkfs: BTreeSet<FrameId>,
    /// `take_ltkf` (`:206`).
    take_ltkf: bool,
    /// `frame_idx` (`:207`).
    frame_idx: BTreeMap<FrameId, usize>,
    /// `last_state_t_ns` (`:209`).
    last_state_t_ns: i64,
    /// `imu_meas` (`:210`), one preintegration per consecutive state pair.
    imu_meas: BTreeMap<i64, IntegratedImuMeasurement<S>>,
    /// `g` (`:212`), `(0, 0, -9.81)` unless the caller overrides it.
    g: Vector3<S>,
    /// `prev_opt_flow_res` (`:216`), kept so a new keyframe can gather every
    /// observation of a keypoint across the live window (`:491-505`).
    prev_opt_flow_res: BTreeMap<FrameId, Arc<FlowObservations>>,
    /// `num_points_kf` (`:218`). Never erased, exactly as basalt never erases
    /// it — `:824` reads it for keyframes that left the window long ago.
    num_points_kf: BTreeMap<FrameId, usize>,
    /// `marg_data` (`:221`), the square-root prior.
    marg_data: MargLinData<S>,
    /// `gyro_bias_sqrt_weight` (`:226`), `1 / gyro_bias_std`.
    gyro_bias_sqrt_weight: Vector3<S>,
    /// `accel_bias_sqrt_weight` (`:226`).
    accel_bias_sqrt_weight: Vector3<S>,
    /// `max_states` (`:228`).
    max_states: usize,
    /// `max_kfs` (`:229`).
    max_kfs: usize,
    /// `T_w_i_init` (`:231`), the pose the first accelerometer sample gave.
    t_w_i_init: Se3<S>,
    /// `initialized` (`:233`).
    initialized: bool,
    /// `opt_started` (`:234`), which flips once `frame_states.size() > 4`.
    opt_started: bool,
    /// `config` (`:237`).
    config: VioConfig,
    /// The four damping fields of `:241`.
    damping: LmDamping<S>,
    /// The estimator's own preintegration noise (`:228-229` of the constructor
    /// body). The frontend runs a second, independent preintegrator (D24) and
    /// the two are deliberately not shared.
    noise: ImuNoise<S>,
    /// The gyro and accel bias the caller initialised with, re-used for every
    /// new `IntegratedImuMeasurement` before the first state exists.
    initial_bias_gyro: Vector3<S>,
    /// See [`Self::initial_bias_gyro`].
    initial_bias_accel: Vector3<S>,

    /// `imu_data_queue` (`vio_estimator.h:91`), unbounded here: Offline mode
    /// never drops a sample and never blocks (D17, D24).
    imu_queue: VecDeque<ImuSample>,
    /// C++'s loop-local `data` (`:296`), the one sample already popped and
    /// calibrated. It survives across frames, so it is state, not a local.
    pending: Option<(i64, Vector3<S>, Vector3<S>)>,
    /// The newest timestamp [`Self::push_imu`] has accepted, whether that
    /// sample is still in [`Self::imu_queue`] or has already moved into
    /// [`Self::pending`]. The queue alone cannot answer that: once its last
    /// sample has been popped it is empty, and an older sample would then be
    /// accepted behind the pending one.
    newest_imu_t_ns: Option<i64>,
    /// Frames the last marginalization removed, for [`Self::snapshot`].
    last_marginalized: Vec<FrameId>,

    /// The buffers [`Self::frame_update`]'s loop works in, kept across frames
    /// for [`Self::scratch`]'s reason (D76).
    frame_scratch: FrameUpdateScratch<S>,

    /// The buffers [`Self::optimize`]'s inner loop works in, kept across
    /// frames.
    ///
    /// Not state: every one of them is reset or overwritten before it is read,
    /// so an estimator that dropped and rebuilt them each frame would compute
    /// the same numbers. They are held because the loop runs seven times on the
    /// median MIO10 frame and each pass wanted a fresh `87x87` reduced system, a
    /// damped copy of it and the reduction's subtree partials.
    scratch: OptimizeScratch<S>,
}

/// Every live scalar the estimator's own arithmetic needs, checked before any
/// of that arithmetic runs.
///
/// The estimator divides by the two bias deviations (`:226`) and by the squared
/// observation deviation (`ba_base.cpp:181`), takes the square root of the IMU
/// rate (`calibration.hpp:186`) and of the three initial prior weights
/// (`:87-93`), and compares `lambda` against both damping bounds (`:1415`,
/// `:1595`). basalt does all of it unchecked, on numbers that reach it from a
/// device driver and a file it ships; here they reach it from a caller, so a
/// value outside its domain is bad input refused at the boundary (D32) rather
/// than a NaN or an infinity in a live prior. The `VioConfig` and `Calibration`
/// parsers stay syntax-only: what a number has to be is a property of the
/// arithmetic that reads it, and `Calibration` is also the frontend's.
///
/// The Nielsen escalation factor is not here: it is the compile-time
/// [`VEE_FACTOR`], not a config field.
fn validate_scalars<S: LieScalar>(
    calibration: &Calibration<S>,
    config: &VioConfig,
) -> Result<(), EstimatorError> {
    let positive = |field: &'static str, value: f64| -> Result<(), EstimatorError> {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(EstimatorError::NonPositiveScalar { field, value })
        }
    };

    positive("imu_update_rate", calibration.imu_update_rate.to_f64())?;
    for (field, deviations) in [
        ("gyro_noise_std", &calibration.gyro_noise_std),
        ("accel_noise_std", &calibration.accel_noise_std),
        ("gyro_bias_std", &calibration.gyro_bias_std),
        ("accel_bias_std", &calibration.accel_bias_std),
    ] {
        for deviation in deviations.iter() {
            positive(field, deviation.to_f64())?;
        }
    }
    positive("vio_obs_std_dev", config.vio_obs_std_dev)?;
    positive("vio_obs_huber_thresh", config.vio_obs_huber_thresh)?;
    positive("vio_lm_lambda_initial", config.vio_lm_lambda_initial)?;
    positive("vio_lm_lambda_min", config.vio_lm_lambda_min)?;
    positive("vio_lm_lambda_max", config.vio_lm_lambda_max)?;

    for (field, value) in [
        ("vio_init_pose_weight", config.vio_init_pose_weight),
        ("vio_init_ba_weight", config.vio_init_ba_weight),
        ("vio_init_bg_weight", config.vio_init_bg_weight),
    ] {
        if !(value.is_finite() && value >= 0.0) {
            return Err(EstimatorError::NegativeScalar { field, value });
        }
    }

    if config.vio_lm_lambda_min > config.vio_lm_lambda_max {
        return Err(EstimatorError::DampingRangeReversed {
            min: config.vio_lm_lambda_min,
            max: config.vio_lm_lambda_max,
        });
    }
    Ok(())
}

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// `SqrtKeypointVioEstimator(g, calib, config)` (`:57-117`).
    ///
    /// Sets the square-root gauge prior on the first state: `sqrt(init_pose_weight)`
    /// on indices 0 to 2 and on index 5 alone, and `sqrt(init_ba_weight)` /
    /// `sqrt(init_bg_weight)` on 9 to 11 and 12 to 14 (`:87-93`, D18). Roll and
    /// pitch (3 and 4) are left free because gravity observes them. The
    /// **square root** is what the sqrt branch stores; the Hessian branch
    /// (`:96-102`) stores the weight itself and is unreachable here (D13).
    ///
    /// # Errors
    ///
    /// [`EstimatorError`] when the config asks for a path this port does not
    /// have, when it enables realtime frame dropping, when the window sizes are
    /// not positive, when the rig has fewer than two cameras, when a camera
    /// model has no projection, or when any scalar its own arithmetic divides
    /// by, takes the square root of or compares against is outside its domain:
    /// `NonPositiveScalar`, `NegativeScalar` and `DampingRangeReversed` name
    /// the field and the value.
    pub fn new(
        g: Vector3<S>,
        calibration: Calibration<S>,
        config: VioConfig,
    ) -> Result<Self, EstimatorError> {
        if config.vio_linearization_type != LinearizationType::AbsQr || !config.vio_sqrt_marg {
            return Err(EstimatorError::UnsupportedPath {
                linearization: config.vio_linearization_type,
                sqrt_marg: config.vio_sqrt_marg,
            });
        }
        if config.vio_enforce_realtime {
            return Err(EstimatorError::EnforceRealtime);
        }
        if config.vio_max_states <= 0 || config.vio_max_kfs <= 0 {
            return Err(EstimatorError::EmptyWindow {
                max_states: config.vio_max_states,
                max_kfs: config.vio_max_kfs,
            });
        }
        // `optical_flow.h:210` and `vit_tracker.cpp:181`: a hard precondition,
        // because the epipolar filter needs a second camera.
        if calibration.t_i_c.len() < 2 {
            return Err(EstimatorError::CameraCountMismatch {
                expected: 2,
                actual: calibration.t_i_c.len(),
            });
        }
        // The rig is one list of cameras: `intrinsics` carries the projections
        // and `t_i_c` the extrinsics, and every camera id downstream indexes
        // both. basalt reads them out of one `Calibration` and never checks,
        // so a JSON with two extrinsics and one intrinsic would index out of
        // range deep inside the triangulation; refuse it here instead.
        if calibration.intrinsics.len() != calibration.t_i_c.len() {
            return Err(EstimatorError::CameraCountMismatch {
                expected: calibration.t_i_c.len(),
                actual: calibration.intrinsics.len(),
            });
        }

        validate_scalars(&calibration, &config)?;

        let noise: ImuNoise<S> = ImuNoise::from_calibration(&calibration);
        let gyro_bias_sqrt_weight: Vector3<S> = calibration.gyro_bias_std.map(|v| S::one() / v);
        let accel_bias_sqrt_weight: Vector3<S> = calibration.accel_bias_std.map(|v| S::one() / v);

        // `:71-73`.
        let obs_std_dev: S = S::from_literal(config.vio_obs_std_dev);
        let huber_thresh: S = S::from_literal(config.vio_obs_huber_thresh);
        let ba: BundleAdjustmentBase<S> =
            BundleAdjustmentBase::new(calibration, obs_std_dev, huber_thresh)?;

        let mut marg_data: MargLinData<S> = MargLinData {
            order: AbsOrderMap::new(),
            h: DMatrix::zeros(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE),
            b: DVector::zeros(POSE_VEL_BIAS_SIZE),
        };
        // `:87-93`, the square-root branch.
        let pose_weight_sqrt: S = S::from_literal(config.vio_init_pose_weight).sqrt();
        let ba_weight_sqrt: S = S::from_literal(config.vio_init_ba_weight).sqrt();
        let bg_weight_sqrt: S = S::from_literal(config.vio_init_bg_weight).sqrt();
        for i in 0..3 {
            marg_data.h[(i, i)] = pose_weight_sqrt;
            marg_data.h[(9 + i, 9 + i)] = ba_weight_sqrt;
            marg_data.h[(12 + i, 12 + i)] = bg_weight_sqrt;
        }
        marg_data.h[(5, 5)] = pose_weight_sqrt;

        Ok(Self {
            ba,
            prev_frame: None,
            // `:61`.
            take_kf: true,
            frames_after_kf: 0,
            frame_count: 0,
            kf_ids: BTreeSet::new(),
            ltkfs: BTreeSet::new(),
            take_ltkf: false,
            frame_idx: BTreeMap::new(),
            last_state_t_ns: 0,
            imu_meas: BTreeMap::new(),
            g,
            prev_opt_flow_res: BTreeMap::new(),
            num_points_kf: BTreeMap::new(),
            marg_data,
            gyro_bias_sqrt_weight,
            accel_bias_sqrt_weight,
            max_states: config.vio_max_states.unsigned_abs() as usize,
            max_kfs: config.vio_max_kfs.unsigned_abs() as usize,
            t_w_i_init: Se3::identity(),
            initialized: false,
            opt_started: false,
            damping: LmDamping {
                lambda: S::from_literal(config.vio_lm_lambda_initial),
                min_lambda: S::from_literal(config.vio_lm_lambda_min),
                max_lambda: S::from_literal(config.vio_lm_lambda_max),
                lambda_vee: S::from_literal(VEE_FACTOR),
            },
            config,
            noise,
            initial_bias_gyro: Vector3::zeros(),
            initial_bias_accel: Vector3::zeros(),
            imu_queue: VecDeque::new(),
            pending: None,
            newest_imu_t_ns: None,
            last_marginalized: Vec::new(),
            frame_scratch: FrameUpdateScratch::default(),
            scratch: OptimizeScratch::default(),
        })
    }

    /// The same estimator with basalt's own gravity, `(0, 0, -9.81)`
    /// (`imu_types.h:63`), which is what `src/vio.cpp` and the VIT path pass.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_default_gravity(
        calibration: Calibration<S>,
        config: VioConfig,
    ) -> Result<Self, EstimatorError> {
        Self::new(gravity::<S>(), calibration, config)
    }

    /// `takeLongTermKeyframe()` (`:124-127`): the next `measure` moves the
    /// newest keyframe into `ltkfs`, where the `max_kfs` budget cannot evict it.
    ///
    /// Nothing in the VIO path calls this; it is a Monado/API hook.
    pub fn take_long_term_keyframe(&mut self) {
        self.take_ltkf = true;
    }

    /// `addIMUToQueue` (`:370-373`) plus the `popFromImuDataQueue` cast
    /// (`:377-390`).
    ///
    /// Samples must arrive in order; a sample that does not follow the last one
    /// **accepted** is dropped rather than reordered, because the integration
    /// reads the stream strictly forward. "Accepted" is
    /// [`Self::newest_imu_t_ns`], not the queue's back: the newest sample may
    /// already sit in `Self::pending`, leaving the queue empty and an older
    /// sample free to slot in behind it. The drop is reachable only from a
    /// direct `SqrtKeypointVio` user, which is `tests/vio_oracle.rs`:
    /// [`Vio::push_imu`](crate::Vio::push_imu) refuses the same sample with a
    /// typed error before it gets here. The static bias calibration
    /// (`calib_bias.hpp:101-107`) is applied when the sample is popped, as
    /// `:298-299` does, not here.
    pub fn push_imu(&mut self, sample: ImuSample) {
        if let Some(newest) = self.newest_imu_t_ns
            && sample.t_ns <= newest
        {
            return;
        }
        self.newest_imu_t_ns = Some(sample.t_ns);
        self.imu_queue.push_back(sample);
    }

    /// Whether the buffered IMU reaches strictly past `t_ns`, which is what
    /// [`Self::process_frame`] needs before it may consume anything (D17).
    ///
    /// Strictly past, because a sample landing exactly on the frameset is
    /// consumed and the integration loop pops again (`:322-328`). The newest
    /// accepted sample is never popped past — the skip loop stops at the
    /// previous frameset and the integration loop at this one, both below it —
    /// so this reads `Self::newest_imu_t_ns` rather than walking the queue.
    ///
    /// [`Vio::track`](crate::Vio::track) runs this **before** the frontend, so
    /// a refused frameset leaves the whole pipeline untouched and the caller
    /// may push the missing samples and retry the same frameset.
    pub fn imu_covers_frame(&self, t_ns: i64) -> bool {
        self.newest_imu_t_ns.is_some_and(|newest| newest > t_ns)
    }

    /// The newest inertial timestamp accepted, or `None` before the first
    /// sample.
    pub fn newest_imu_t_ns(&self) -> Option<i64> {
        self.newest_imu_t_ns
    }

    /// Whether the window has a state (`initialized`, `:233`).
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// `get_t_ns()` (`:138`), the newest state's timestamp.
    pub fn last_state_t_ns(&self) -> i64 {
        self.last_state_t_ns
    }

    /// `get_state()` (`:143`), the newest state, or `None` before the window has
    /// one.
    pub fn state(&self) -> Option<&PoseVelBiasState<S>> {
        self.ba
            .frame_states
            .get(&self.last_state_t_ns)
            .map(PoseVelBiasStateWithLin::state)
    }

    /// The keyframes, oldest first (`kf_ids`, `:204`).
    pub fn kf_ids(&self) -> impl Iterator<Item = FrameId> + '_ {
        self.kf_ids.iter().copied()
    }

    /// The live marginalization prior.
    pub fn marg_data(&self) -> &MargLinData<S> {
        &self.marg_data
    }

    /// `num_points_kf` (`:218`), landmarks hosted per keyframe when it was
    /// created.
    pub fn num_points_kf(&self) -> &BTreeMap<FrameId, usize> {
        &self.num_points_kf
    }

    /// The preintegrated intervals, keyed by their start timestamp.
    pub fn imu_meas(&self) -> &BTreeMap<i64, IntegratedImuMeasurement<S>> {
        &self.imu_meas
    }

    /// The window, its landmarks in the world frame and the frames the last
    /// marginalization removed, for the V2 Rerun rung (D51).
    ///
    /// The landmark position is `host_pose * T_i_c * unproject(direction) /
    /// inv_dist`, exactly as `:628-640` builds the landmark bundle, and a
    /// landmark whose host has left the window is skipped as it is there.
    pub fn snapshot(&self) -> WindowSnapshot<S> {
        let states: Vec<WindowState<S>> = self
            .ba
            .frame_states
            .iter()
            .map(|(t_ns, state)| {
                let inner: &PoseVelBiasState<S> = state.state();
                WindowState {
                    t_ns: *t_ns,
                    t_w_i: inner.t_w_i,
                    vel_bias: Some(VelBias {
                        vel_w_i: inner.vel_w_i,
                        bias_gyro: inner.bias_gyro,
                        bias_accel: inner.bias_accel,
                    }),
                    linearized: state.is_linearized(),
                    keyframe: self.kf_ids.contains(t_ns),
                    long_term_keyframe: self.ltkfs.contains(t_ns),
                    frame_index: self.frame_idx.get(t_ns).copied(),
                }
            })
            .collect();
        let poses: Vec<WindowState<S>> = self
            .ba
            .frame_poses
            .iter()
            .map(|(t_ns, pose)| WindowState {
                t_ns: *t_ns,
                t_w_i: *pose.pose(),
                vel_bias: None,
                linearized: pose.is_linearized(),
                keyframe: self.kf_ids.contains(t_ns),
                long_term_keyframe: self.ltkfs.contains(t_ns),
                frame_index: self.frame_idx.get(t_ns).copied(),
            })
            .collect();

        let mut landmarks: Vec<SnapshotLandmark<S>> =
            Vec::with_capacity(self.ba.lmdb.num_landmarks());
        for lm in self.ba.lmdb.landmarks() {
            let Ok(host) = self.ba.get_pose_state_with_lin(lm.host_kf_id.frame_id) else {
                continue;
            };
            let Some(t_i_c) = self.ba.calib.t_i_c.get(lm.host_kf_id.cam_id) else {
                continue;
            };
            let t_w_c: Se3<S> = *host.pose() * *t_i_c;
            let bearing: Vector4<S> = StereographicParam::unproject(&lm.direction);
            let scale: S = S::one() / lm.inv_dist;
            let point_c: Vector3<S> =
                Vector3::new(bearing[0] * scale, bearing[1] * scale, bearing[2] * scale);
            landmarks.push(SnapshotLandmark {
                id: lm.id,
                host: lm.host_kf_id,
                position_w: t_w_c * point_c,
            });
        }

        WindowSnapshot {
            t_ns: self.last_state_t_ns,
            states,
            poses,
            landmarks,
            marginalized: self.last_marginalized.clone(),
        }
    }

    /// One iteration of `proc_func`'s loop (`:263-355`).
    ///
    /// Pops and calibrates the IMU samples this frameset needs, initialises the
    /// window from the first accelerometer sample if it has none
    /// (`:263-296`, D17 of papers-part2 §13: there is no other initialization
    /// stage), preintegrates `(prev_t, curr_t]` into one measurement with the
    /// **previous state's** biases as the linearization point (`:302-304`), and
    /// calls `Self::measure`.
    ///
    /// Returns [`FrameOutcome::NeedMoreImu`] and leaves everything untouched
    /// when the buffer does not yet reach past `frame.t_ns`: basalt blocks on
    /// its queue instead, and Offline mode must not let the arrival order of a
    /// sample reach a decision (D17).
    ///
    /// # Errors
    ///
    /// [`EstimatorError`] on a non-monotonic frameset, a frameset of the wrong
    /// width, or anything `measure` refuses.
    pub fn process_frame(
        &mut self,
        frame: Arc<FlowObservations>,
    ) -> Result<FrameOutcome<S>, EstimatorError> {
        let num_cams: usize = self.ba.calib.t_i_c.len();
        if frame.cameras.len() != num_cams {
            return Err(EstimatorError::CameraCountMismatch {
                expected: num_cams,
                actual: frame.cameras.len(),
            });
        }
        if let Some(prev) = &self.prev_frame
            && frame.t_ns <= prev.t_ns
        {
            // `:309-313`, both asserts.
            return Err(EstimatorError::NonMonotonicFrame {
                previous_t_ns: prev.t_ns,
                t_ns: frame.t_ns,
            });
        }

        // The one place Offline mode differs from a blocking queue: every pop
        // below must succeed, so the coverage test happens before any state
        // moves. `Vio::track` runs the same predicate before the frontend, so
        // by the time a caller gets here through the driver this is the second
        // line, not the first.
        if !self.imu_covers_frame(frame.t_ns) {
            return Ok(FrameOutcome::NeedMoreImu);
        }

        let predict_started: std::time::Instant = std::time::Instant::now();
        if self.pending.is_none() {
            self.pending = self.pop_calibrated();
        }

        let mut meas: Option<IntegratedImuMeasurement<S>> = None;

        if !self.initialized {
            // `:265-271`: skip forward to the frameset, then take that sample's
            // accelerometer reading as the whole initialization.
            //
            // No `NeedMoreImu` exit here, and that is the point: the skip has
            // already consumed samples by the time it could want one, and the
            // outcome promises the estimator was left untouched. It cannot
            // want one either — the coverage test above found a sample
            // strictly past the frameset and `push_imu` keeps the queue
            // ordered, so the skip stops on a sample at or after it.
            let accel: Vector3<S> = loop {
                match self.pending {
                    Some((t_ns, _, accel)) if t_ns >= frame.t_ns => break accel,
                    Some(_) => self.pending = self.pop_calibrated(),
                    None => return Err(EstimatorError::ImuQueueRanDry { t_ns: frame.t_ns }),
                }
            };

            // `:273-278`: zero velocity, zero translation, and the rotation that
            // takes the measured acceleration onto +Z.
            let vel_w_i_init: Vector3<S> = Vector3::zeros();
            self.t_w_i_init = Se3::new(gravity_from_first_accel(&accel), Vector3::zeros());

            self.last_state_t_ns = frame.t_ns;
            self.imu_meas.insert(
                self.last_state_t_ns,
                IntegratedImuMeasurement::new(
                    self.last_state_t_ns,
                    &self.initial_bias_gyro,
                    &self.initial_bias_accel,
                ),
            );
            self.ba.frame_states.insert(
                self.last_state_t_ns,
                PoseVelBiasStateWithLin::new(
                    PoseVelBiasState::new(
                        self.last_state_t_ns,
                        self.t_w_i_init,
                        vel_w_i_init,
                        self.initial_bias_gyro,
                        self.initial_bias_accel,
                    ),
                    true,
                ),
            );
            self.frame_idx
                .insert(self.last_state_t_ns, self.frame_count);
            self.frame_count += 1;

            // `:281-283`: the first ordering entry, one 15-dof state at index 0.
            let mut order: AbsOrderMap = AbsOrderMap::new();
            order.push(self.last_state_t_ns, POSE_VEL_BIAS_SIZE)?;
            self.marg_data.order = order;

            self.initialized = true;
        } else if let Some(prev) = self.prev_frame.clone() {
            // `:300-336`.
            let (bias_gyro, bias_accel) = match self.ba.frame_states.get(&self.last_state_t_ns) {
                Some(state) => (state.state().bias_gyro, state.state().bias_accel),
                None => (self.initial_bias_gyro, self.initial_bias_accel),
            };
            let mut pim: IntegratedImuMeasurement<S> =
                IntegratedImuMeasurement::new(prev.t_ns, &bias_gyro, &bias_accel);

            // `:315-336`, the loop `IntegratedImuMeasurement::accumulate_to`
            // owns for both preintegrators.
            let noise: ImuNoise<S> = self.noise;
            let pending: Option<Popped<S>> = self.pending.take();
            let (skip_past_ns, until_ns): (i64, i64) = (prev.t_ns, frame.t_ns);
            self.pending = pim.accumulate_to(
                pending,
                || self.pop_calibrated(),
                skip_past_ns,
                until_ns,
                &noise,
            )?;
            meas = Some(pim);
        }

        let integration_ns: u64 = duration_ns(predict_started);
        let mut stats: FrameStats<S> = self.measure(Arc::clone(&frame), meas)?;
        stats.timings.predict_ns += integration_ns;
        // `:354`, and only on success.
        self.prev_frame = Some(frame);
        Ok(FrameOutcome::Measured(Box::new(stats)))
    }

    /// `popFromImuDataQueue` (`:377-390`) followed by the static bias
    /// calibration of `:298-299`.
    ///
    /// The cast to `Scalar` happens **before** the calibration, as it does in
    /// C++: the queue holds `ImuData<double>` and `popFromImuDataQueue` casts,
    /// then `getCalibrated` runs in `Scalar`.
    fn pop_calibrated(&mut self) -> Option<(i64, Vector3<S>, Vector3<S>)> {
        let sample: ImuSample = self.imu_queue.pop_front()?;
        let gyro: Vector3<S> = sample.gyro.map(S::from_literal);
        let accel: Vector3<S> = sample.accel.map(S::from_literal);
        Some((
            sample.t_ns,
            self.ba.calib.calib_gyro_bias.calibrated(&gyro),
            self.ba.calib.calib_accel_bias.calibrated(&accel),
        ))
    }

    /// `measure(opt_flow_meas, meas)` (`:422-575`).
    ///
    /// **Contract: `frame.cameras.len() == self.ba.calib.t_i_c.len()`, which is
    /// at least two.** [`Self::process_frame`] checks the frameset width
    /// against the rig (`:309`) and [`Self::new`] refuses a rig of fewer than
    /// two cameras, so camera 0 exists and the per-camera vectors below are
    /// indexed directly rather than re-validated here.
    ///
    /// # Errors
    ///
    /// [`EstimatorError`] when an observation cannot be filed or when
    /// `optimize_and_marg` refuses.
    fn measure(
        &mut self,
        frame: Arc<FlowObservations>,
        meas: Option<IntegratedImuMeasurement<S>>,
    ) -> Result<FrameStats<S>, EstimatorError> {
        let started: std::time::Instant = std::time::Instant::now();
        let num_cams: usize = frame.cameras.len();
        debug_assert_eq!(num_cams, self.ba.calib.t_i_c.len());

        // `:427-441`: predict the new state from the previous one and the
        // preintegration, then file it under the frameset's timestamp.
        if let Some(pim) = meas {
            let previous: PoseVelBiasState<S> =
                match self.ba.frame_states.get(&self.last_state_t_ns) {
                    Some(state) => *state.state(),
                    None => {
                        return Err(EstimatorError::PreviousStateMissing {
                            t_ns: self.last_state_t_ns,
                        });
                    }
                };
            let predicted: PoseVelState<S> = pim.predict_state(&previous.pose_vel_state(), &self.g);
            let next_state: PoseVelBiasState<S> = PoseVelBiasState::new(
                frame.t_ns,
                predicted.t_w_i,
                predicted.vel_w_i,
                previous.bias_gyro,
                previous.bias_accel,
            );

            self.last_state_t_ns = frame.t_ns;
            self.ba.frame_states.insert(
                self.last_state_t_ns,
                PoseVelBiasStateWithLin::new(next_state, false),
            );
            self.frame_idx
                .insert(self.last_state_t_ns, self.frame_count);
            self.frame_count += 1;
            self.imu_meas.insert(pim.get_start_t_ns(), pim);
        }

        let predict_ns: u64 = duration_ns(started);

        // `:444`.
        self.prev_opt_flow_res
            .insert(frame.t_ns, Arc::clone(&frame));

        // `:447-451`: file every observation the window already hosts, and
        // remember the rest per camera.
        let mut connected: Vec<usize> = vec![0; num_cams];
        let mut num_points_connected: BTreeMap<FrameId, usize> = BTreeMap::new();
        let mut unconnected_obs: Vec<BTreeSet<KeypointId>> = vec![BTreeSet::new(); num_cams];
        for (cam_id, keypoints) in frame.cameras.iter().enumerate() {
            let tcid_target: TimeCamId = TimeCamId::new(frame.t_ns, cam_id);
            for (kpt_id, pos) in keypoints {
                let lm_id: LandmarkId = LandmarkId::from(*kpt_id);
                match self.ba.lmdb.get_landmark(lm_id) {
                    Some(landmark) => {
                        let host: FrameId = landmark.host_kf_id.frame_id;
                        self.ba
                            .lmdb
                            .add_observation(tcid_target, lm_id, cast_pixel::<S>(pos))?;
                        *num_points_connected.entry(host).or_insert(0) += 1;
                        connected[cam_id] += 1;
                    }
                    None => {
                        unconnected_obs[cam_id].insert(*kpt_id);
                    }
                }
            }
        }

        // `:454-456`, D21: camera 0 alone votes, rate-limited to one keyframe
        // every `vio_min_frames_after_kf + 1` frames.
        //
        // The division is `Scalar(int) / size_t`, so the denominator is
        // converted to `Scalar` too; an empty camera 0 gives `0 / 0` = NaN and
        // `NaN < thresh` is false, which is why `eigen_maxi`-style care is not
        // needed but `f32::max` semantics would be wrong. The threshold is a
        // `float` field widened to `Scalar`, which in the `f64` instantiation is
        // `0.699999988079071`, not `0.7`.
        let keyframe_started: std::time::Instant = std::time::Instant::now();
        let total0: usize = connected[0] + unconnected_obs[0].len();
        let ratio: S = S::from_literal(connected[0] as f64) / S::from_literal(total0 as f64);
        let keyframe_vote: bool = ratio
            < S::from_literal(f64::from(self.config.vio_new_kf_keypoints_thresh))
            && self.frames_after_kf > self.config.vio_min_frames_after_kf;
        if keyframe_vote {
            self.take_kf = true;
        }

        self.demote_long_term_keyframe();

        let took_keyframe: bool = self.take_kf;
        let mut num_points_added: usize = 0;
        if self.take_kf {
            // `:473-552`.
            self.take_kf = false;
            self.frames_after_kf = 0;
            self.kf_ids.insert(self.last_state_t_ns);
            num_points_added = self.triangulate_unconnected(&frame, &unconnected_obs)?;
            self.num_points_kf.insert(frame.t_ns, num_points_added);
        } else {
            self.frames_after_kf += 1;
        }

        let keyframe_ns: u64 = if took_keyframe { duration_ns(keyframe_started) } else { 0 };

        // `:555-563`: every landmark this frameset did not see, in any camera.
        let mut lost_landmarks: BTreeSet<LandmarkId> = BTreeSet::new();
        if self.config.vio_marg_lost_landmarks {
            for lm in self.ba.lmdb.landmarks() {
                let kpt: KeypointId = KeypointId::from(lm.id);
                if !frame.cameras.iter().any(|cam| cam.contains_key(&kpt)) {
                    lost_landmarks.insert(lm.id);
                }
            }
        }

        // `:566`, plus D76's gate: above zero, `port.frame_update_max_iterations`
        // moves the joint solve to the framesets that took a keyframe and gives
        // the others the newest state alone. The frame update declines a
        // frameset it cannot serve, and the joint solve owns the warmup.
        let optimize_started: std::time::Instant = std::time::Instant::now();
        let attempt_frame_update: bool =
            self.config.port_frame_update_max_iterations > 0 && !took_keyframe && self.opt_started;
        let updated: Option<FrameUpdateResult<S>> = if attempt_frame_update {
            Some(self.frame_update(frame.t_ns)?)
        } else {
            None
        };
        let frame_update: FrameUpdateOutcome = match &updated {
            None => FrameUpdateOutcome::NotAttempted,
            Some(Ok(_)) => FrameUpdateOutcome::Taken,
            Some(Err(decline)) => FrameUpdateOutcome::Declined(*decline),
        };
        let (lm, termination, mut timings) = match updated {
            Some(Ok(outcome)) => outcome,
            Some(Err(_)) | None => self.optimize(frame.t_ns)?,
        };
        timings.optimize_ns = duration_ns(optimize_started);
        timings.predict_ns = predict_ns;
        timings.keyframe_ns = keyframe_ns;
        let marg: MarginalizationOutcome =
            self.marginalize(&num_points_connected, &lost_landmarks)?;
        timings.marginalize_ns = marg.elapsed_ns;
        timings.measure_ns = duration_ns(started);

        // `:1642-1653`, read off the window the two stages above left behind.
        Ok(FrameStats {
            t_ns: frame.t_ns,
            connected,
            unconnected: unconnected_obs.iter().map(BTreeSet::len).collect(),
            took_keyframe,
            keyframe_vote,
            frames_after_kf: self.frames_after_kf,
            num_points_added,
            kf_ids: self.kf_ids.iter().copied().collect(),
            ltkfs: self.ltkfs.iter().copied().collect(),
            num_landmarks: self.ba.lmdb.num_landmarks(),
            num_observations: self.ba.lmdb.num_observations(),
            num_lost_landmarks: lost_landmarks.len(),
            opt_started: self.opt_started,
            lm,
            termination,
            marginalization: marg.marginalization,
            frame_update,
            timings,
        })
    }

    /// `:474-552`: triangulate every unconnected observation into a new
    /// landmark hosted by this frameset.
    ///
    /// For each unconnected id, gather **every** observation of it across the
    /// live `prev_opt_flow_res` (`:491-505`), then try each gathered image in
    /// `TimeCamId` order as the second view: unproject both pixels, form
    /// `T_0_1 = T_i_c[host]⁻¹ · T_i0_i1 · T_i_c[other]`, skip a baseline shorter
    /// than `vio_min_triangulation_dist` (`:530`), DLT-triangulate, and accept
    /// iff every coefficient is finite and `0 < inv_dist < 3` (`:534`). On
    /// acceptance **all** gathered observations are filed, not only the pair
    /// that triangulated (`:546-548`).
    ///
    /// The host camera is `i`, not camera 0: the database is genuinely
    /// N-camera.
    ///
    /// **Contract: every camera id below is in the rig**, so the rig is
    /// indexed directly. `cam_id` indexes `unconnected_obs`, which
    /// [`Self::measure`] builds with one entry per frameset camera, and
    /// `tcido.cam_id` names a camera of a frameset [`Self::process_frame`]
    /// already accepted; both are therefore `< t_i_c.len()`, and
    /// [`Self::new`] refuses a calibration whose `intrinsics` and `t_i_c`
    /// disagree.
    ///
    /// **Deliberate deviation from the C++ line shape:** `:519-521` re-reads
    /// the host pose and re-inverts it and the host camera's extrinsic for
    /// every candidate pair. Nothing in the loop moves the host frame's pose
    /// or the calibration, so they are resolved once each — the same values
    /// from the same inputs, in the same products.
    fn triangulate_unconnected(
        &mut self,
        frame: &FlowObservations,
        unconnected_obs: &[BTreeSet<KeypointId>],
    ) -> Result<usize, EstimatorError> {
        debug_assert_eq!(unconnected_obs.len(), self.ba.calib.t_i_c.len());
        // `:509`: the squared threshold is formed in `double` and cast, so the
        // `f32` instantiation compares against `(float)(0.05 * 0.05)`.
        let min_triang_distance2: S = S::from_literal(
            self.config.vio_min_triangulation_dist * self.config.vio_min_triangulation_dist,
        );
        let mut num_points_added: usize = 0;
        // `:519`'s `T_i0_inv`: the host is this frameset, for every landmark
        // and every pair.
        let t_i0_inv: Se3<S> = self
            .ba
            .get_pose_state_with_lin(frame.t_ns)?
            .pose()
            .inverse();

        for (cam_id, ids) in unconnected_obs.iter().enumerate() {
            let tcidl: TimeCamId = TimeCamId::new(frame.t_ns, cam_id);
            let host_keypoints: &BTreeMap<KeypointId, Vector2<f32>> = &frame.cameras[cam_id];
            let cam0: CameraEnum<S> = self.ba.cameras()[cam_id];
            let t_i_c0_inv: Se3<S> = self.ba.calib.t_i_c[cam_id].inverse();
            for kpt_id in ids {
                let lm_id: LandmarkId = LandmarkId::from(*kpt_id);
                // `:487`: another camera of this frameset may have hosted it
                // already.
                if self.ba.lmdb.landmark_exists(lm_id) {
                    continue;
                }
                // `:514`'s `.at(lm_id)`: `measure` took this id out of
                // `host_keypoints` itself, so a miss is an invariant break, not
                // a landmark to skip (D32).
                let Some(p0_pixel) = host_keypoints.get(kpt_id) else {
                    return Err(EstimatorError::UnconnectedKeypointMissing {
                        cam_id,
                        kpt_id: *kpt_id,
                    });
                };
                let p0: Vector2<S> = cast_pixel::<S>(p0_pixel);

                // `:491-505`: every image of this id in the live window,
                // ordered by `TimeCamId` because C++ collects into a
                // `std::map`.
                let mut kp_obs: BTreeMap<TimeCamId, Vector2<S>> = BTreeMap::new();
                for (other_t_ns, other) in &self.prev_opt_flow_res {
                    for (other_cam, keypoints) in other.cameras.iter().enumerate() {
                        if let Some(pixel) = keypoints.get(kpt_id) {
                            kp_obs.insert(
                                TimeCamId::new(*other_t_ns, other_cam),
                                cast_pixel::<S>(pixel),
                            );
                        }
                    }
                }

                let mut accepted: Option<Landmark<S>> = None;
                for (tcido, p1) in &kp_obs {
                    // `:512-517`: an unprojection the camera rejects skips this
                    // pair, not the landmark.
                    let mut p0_3d: Vector4<S> = Vector4::zeros();
                    let mut p1_3d: Vector4<S> = Vector4::zeros();
                    let cam1: CameraEnum<S> = self.ba.cameras()[tcido.cam_id];
                    let valid0: bool = cam0.unproject(&p0, &mut p0_3d);
                    let valid1: bool = cam1.unproject(p1, &mut p1_3d);
                    if !valid0 || !valid1 {
                        continue;
                    }

                    // `:519-522`.
                    let other_pose: Se3<S> =
                        *self.ba.get_pose_state_with_lin(tcido.frame_id)?.pose();
                    let t_i0_i1: Se3<S> = t_i0_inv * other_pose;
                    let t_0_1: Se3<S> = t_i_c0_inv * t_i0_i1 * self.ba.calib.t_i_c[tcido.cam_id];

                    // `:524`: `squaredNorm()` on a 3-vector is Eigen's
                    // three-coefficient reduction, whose order differs between
                    // the precisions (D47).
                    let t: Vector3<S> = t_0_1.translation;
                    let baseline2: S = S::eigen_redux3(t[0] * t[0], t[1] * t[1], t[2] * t[2]);
                    if baseline2 < min_triang_distance2 {
                        continue;
                    }

                    // `:526-543`. A refused DLT skips this pair, as an
                    // unprojection the camera rejects does: where basalt reads
                    // Eigen's uninitialized `V`, the port has no value at all.
                    let Some(triangulated): Option<Vector4<S>> = triangulate(
                        &Vector3::new(p0_3d[0], p0_3d[1], p0_3d[2]),
                        &Vector3::new(p1_3d[0], p1_3d[1], p1_3d[2]),
                        &t_0_1,
                    ) else {
                        continue;
                    };
                    let finite: bool = triangulated.iter().all(|v| v.is_finite());
                    // What decides this gate is the triangulated value's own
                    // reduction order (D47), not the comparison.
                    let inv_dist: S = triangulated[3];
                    // `3.0` is a `double` literal, so the `f32` instantiation
                    // promotes and compares in `double`.
                    if finite && inv_dist > S::zero() && inv_dist.to_f64() < 3.0 {
                        accepted = Some(Landmark::new(
                            lm_id,
                            tcidl,
                            StereographicParam::project(&triangulated),
                            inv_dist,
                        ));
                        break;
                    }
                }

                // `:545-549`.
                if let Some(landmark) = accepted {
                    self.ba.lmdb.add_landmark(lm_id, &landmark);
                    num_points_added += 1;
                    for (tcido, pixel) in &kp_obs {
                        self.ba.lmdb.add_observation(*tcido, lm_id, *pixel)?;
                    }
                }
            }
        }
        Ok(num_points_added)
    }

    /// `ImuLinData` as `:1264` and `:913` build it.
    fn imu_lin_data(&self) -> ImuLinData<S> {
        ImuLinData {
            g: self.g,
            gyro_bias_weight_sqrt: self.gyro_bias_sqrt_weight,
            accel_bias_weight_sqrt: self.accel_bias_sqrt_weight,
        }
    }
}

/// `fixed_kfs`: the keyframes whose pose Jacobians a linearization zeroes.
///
/// C++ builds the same set twice — `:924` for the marginalization's own
/// linearization and `:1258` for the optimization's — so with
/// `vio_fix_long_term_keyframes` on the long-term keyframes are fixed in the
/// prior the window computes as well as in the increment it solves. `None` and
/// an empty set mean the same thing to
/// [`LinearizationAbsQR`](crate::linearize::LinearizationAbsQR); `None` is what
/// the flag being off says.
fn fixed_keyframes<'a>(
    config: &VioConfig,
    ltkfs: &'a BTreeSet<FrameId>,
) -> Option<&'a BTreeSet<FrameId>> {
    if config.vio_fix_long_term_keyframes {
        Some(ltkfs)
    } else {
        None
    }
}

/// `AffineCompact2f::translation().cast<Scalar>()` (`:437`).
fn cast_pixel<S: LieScalar>(pixel: &Vector2<f32>) -> Vector2<S> {
    Vector2::new(
        S::from_literal(f64::from(pixel.x)),
        S::from_literal(f64::from(pixel.y)),
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::config::VioConfig;

    const CALIB: &str = include_str!("../../tests/fixtures/msdmi_calib.json");
    const CONFIG: &str = include_str!("../../../../configs/msdmi_config.json");

    /// The fixture rig and config, `f32` as the shipped lane runs.
    fn estimator() -> SqrtKeypointVio<f32> {
        SqrtKeypointVio::with_default_gravity(
            Calibration::<f64>::from_json_str(CALIB).unwrap().cast(),
            VioConfig::from_json_str(CONFIG).unwrap(),
        )
        .unwrap()
    }

    /// A sample the initialization can take a gravity direction from.
    fn sample(t_ns: i64) -> ImuSample {
        ImuSample {
            t_ns,
            gyro: Vector3::zeros(),
            accel: Vector3::new(0.0, 0.0, 9.81),
        }
    }

    /// The rig is one list of cameras, and
    /// [`SqrtKeypointVio::triangulate_unconnected`] indexes the projections and
    /// the extrinsics with the same id.
    /// D32 at the API boundary: every scalar the estimator's own arithmetic
    /// divides by, takes the square root of or compares against is checked
    /// before any of that arithmetic runs.
    ///
    /// basalt runs it anyway: a negative `vio_init_pose_weight` puts a NaN on
    /// the prior's diagonal (`:87-93`), a zero bias deviation an infinity in the
    /// bias weight (`:226`), a negative `imu_update_rate` a NaN in both IMU
    /// covariances (`calibration.hpp:186`), and a zero `vio_obs_std_dev` an
    /// infinity in every Huber weight (`ba_base.cpp:181`). Every shipped
    /// fixture is inside every domain, which is why no oracle lane can see
    /// this.
    ///
    /// The values are the estimator's own scalar widened back to `f64`, so the
    /// calibration probes use numbers `f32` holds exactly.
    #[test]
    fn a_scalar_outside_its_domain_is_refused_before_the_estimator_exists() {
        let refuse_config = |mutate: &dyn Fn(&mut VioConfig)| -> EstimatorError {
            let mut config: VioConfig = VioConfig::from_json_str(CONFIG).unwrap();
            mutate(&mut config);
            SqrtKeypointVio::<f32>::with_default_gravity(
                Calibration::<f64>::from_json_str(CALIB).unwrap().cast(),
                config,
            )
            .unwrap_err()
        };
        let refuse_calibration = |mutate: &dyn Fn(&mut Calibration<f64>)| -> EstimatorError {
            let mut calibration: Calibration<f64> = Calibration::from_json_str(CALIB).unwrap();
            mutate(&mut calibration);
            SqrtKeypointVio::<f32>::with_default_gravity(
                calibration.cast(),
                VioConfig::from_json_str(CONFIG).unwrap(),
            )
            .unwrap_err()
        };

        // The shipped pair, which every domain accepts.
        estimator();

        // The three initial prior weights, whose square roots the prior is.
        assert_eq!(
            refuse_config(&|config| config.vio_init_pose_weight = -1.0),
            EstimatorError::NegativeScalar {
                field: "vio_init_pose_weight",
                value: -1.0,
            }
        );
        assert_eq!(
            refuse_config(&|config| config.vio_init_ba_weight = f64::NEG_INFINITY),
            EstimatorError::NegativeScalar {
                field: "vio_init_ba_weight",
                value: f64::NEG_INFINITY,
            }
        );
        assert_eq!(
            refuse_config(&|config| config.vio_init_bg_weight = f64::INFINITY),
            EstimatorError::NegativeScalar {
                field: "vio_init_bg_weight",
                value: f64::INFINITY,
            }
        );

        // The reprojection cost's two scalars.
        assert_eq!(
            refuse_config(&|config| config.vio_obs_std_dev = 0.0),
            EstimatorError::NonPositiveScalar {
                field: "vio_obs_std_dev",
                value: 0.0,
            }
        );
        assert_eq!(
            refuse_config(&|config| config.vio_obs_huber_thresh = -1.0),
            EstimatorError::NonPositiveScalar {
                field: "vio_obs_huber_thresh",
                value: -1.0,
            }
        );

        // The damping: three positive bounds, and they must not cross.
        assert_eq!(
            refuse_config(&|config| config.vio_lm_lambda_initial = f64::INFINITY),
            EstimatorError::NonPositiveScalar {
                field: "vio_lm_lambda_initial",
                value: f64::INFINITY,
            }
        );
        assert_eq!(
            refuse_config(&|config| config.vio_lm_lambda_min = 0.0),
            EstimatorError::NonPositiveScalar {
                field: "vio_lm_lambda_min",
                value: 0.0,
            }
        );
        assert_eq!(
            refuse_config(&|config| {
                config.vio_lm_lambda_min = 1.0;
                config.vio_lm_lambda_max = 0.5;
            }),
            EstimatorError::DampingRangeReversed { min: 1.0, max: 0.5 }
        );

        // The IMU rate and the four deviations, one probe each.
        assert_eq!(
            refuse_calibration(&|calibration| calibration.imu_update_rate = -200.0),
            EstimatorError::NonPositiveScalar {
                field: "imu_update_rate",
                value: -200.0,
            }
        );
        assert_eq!(
            refuse_calibration(&|calibration| calibration.gyro_bias_std.y = 0.0),
            EstimatorError::NonPositiveScalar {
                field: "gyro_bias_std",
                value: 0.0,
            }
        );
        assert_eq!(
            refuse_calibration(&|calibration| calibration.accel_bias_std.z = -0.5),
            EstimatorError::NonPositiveScalar {
                field: "accel_bias_std",
                value: -0.5,
            }
        );
        assert_eq!(
            refuse_calibration(&|calibration| calibration.accel_noise_std.x = f64::INFINITY),
            EstimatorError::NonPositiveScalar {
                field: "accel_noise_std",
                value: f64::INFINITY,
            }
        );
        // A NaN cannot be compared with `assert_eq!`; it is refused as well.
        assert!(matches!(
            refuse_calibration(&|calibration| calibration.gyro_noise_std.x = f64::NAN),
            EstimatorError::NonPositiveScalar {
                field: "gyro_noise_std",
                ..
            }
        ));
    }

    #[test]
    fn a_rig_whose_intrinsics_and_extrinsics_disagree_is_refused() {
        let config: VioConfig = VioConfig::from_json_str(CONFIG).unwrap();
        let mut calibration: Calibration<f64> = Calibration::from_json_str(CALIB).unwrap();
        calibration.intrinsics.pop();
        let refused: EstimatorError =
            SqrtKeypointVio::new(Vector3::new(0.0, 0.0, -9.81), calibration, config).unwrap_err();
        assert!(
            matches!(
                refused,
                EstimatorError::CameraCountMismatch {
                    expected: 2,
                    actual: 1
                }
            ),
            "two extrinsics and one intrinsic is not a rig, got {refused:?}"
        );
    }

    /// `push_imu` compares with the newest sample it has **accepted**, not with
    /// the queue's back: once that sample has moved into `pending` the queue is
    /// empty, and comparing with its back let an older sample slot in behind
    /// the pending one, so a later integration met reversed timestamps
    /// (`:947`'s loop). basalt cannot hit this — its queue is the only buffer —
    /// but the port's public API could.
    #[test]
    fn an_imu_sample_behind_the_pending_one_is_dropped() {
        let mut estimator: SqrtKeypointVio<f32> = estimator();
        estimator.push_imu(sample(10));
        // One frameset before the sample: the initialization pops it into
        // `pending` and leaves the queue empty, which is the whole setup.
        estimator
            .process_frame(Arc::new(FlowObservations::new(5, 2)))
            .unwrap();
        assert!(estimator.imu_queue.is_empty());
        assert_eq!(estimator.pending.map(|(t_ns, _, _)| t_ns), Some(10));

        estimator.push_imu(sample(7));
        assert!(
            estimator.imu_queue.is_empty(),
            "a sample older than the pending one was accepted behind it"
        );
        assert_eq!(estimator.newest_imu_t_ns, Some(10));

        // A sample that does follow the pending one is still accepted.
        estimator.push_imu(sample(12));
        assert_eq!(
            estimator.imu_queue.back().map(|s| s.t_ns),
            Some(12),
            "the ordering check rejected a sample that does follow"
        );
    }

    /// `:514`'s `opt_flow_meas->keypoints.at(i).at(lm_id)`: the triangulation
    /// reads the host pixel back out of the map `measure` took the unconnected
    /// id from, and C++ throws when it is not there. The port used to skip the
    /// landmark silently (D32).
    ///
    /// Only a test can build this: `measure` fills `unconnected_obs[cam]` by
    /// iterating `frame.cameras[cam]` and hands the same `frame` on, so the
    /// two agree by construction on every ported path. The call below pairs an
    /// id with a frameset that never carried it.
    #[test]
    fn an_unconnected_keypoint_missing_from_its_own_frameset_is_refused() {
        let mut estimator: SqrtKeypointVio<f32> = estimator();
        estimator.push_imu(sample(10));
        estimator
            .process_frame(Arc::new(FlowObservations::new(5, 2)))
            .unwrap();

        let frame: FlowObservations = FlowObservations::new(5, 2);
        let unconnected: Vec<BTreeSet<KeypointId>> =
            vec![BTreeSet::from([KeypointId(7)]), BTreeSet::new()];
        assert_eq!(
            estimator
                .triangulate_unconnected(&frame, &unconnected)
                .unwrap_err(),
            EstimatorError::UnconnectedKeypointMissing {
                cam_id: 0,
                kpt_id: KeypointId(7),
            }
        );

        // The same call over an id the frameset does carry gets as far as the
        // triangulation, which one view cannot satisfy: no landmark, no error.
        let mut carried: FlowObservations = FlowObservations::new(5, 2);
        carried.cameras[0].insert(KeypointId(7), Vector2::new(480.0, 480.0));
        assert_eq!(
            estimator
                .triangulate_unconnected(&carried, &unconnected)
                .unwrap(),
            0
        );
    }
}
