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
//! * [`SqrtKeypointVio::measure`] (`:422-575`) predicts the new state, files the
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

mod optimize;
mod schedule;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Vector2, Vector3, Vector4};

use crate::ba_base::{BaError, BundleAdjustmentBase, triangulate};
use crate::calib::Calibration;
use crate::camera::CameraEnum;
use crate::config::{LinearizationType, VioConfig};
use crate::imu::{
    ImuError, ImuLinData, ImuNoise, ImuSample, IntegratedImuMeasurement, gravity,
    gravity_from_first_accel,
};
use crate::landmark::{Landmark, LandmarkError, StereographicParam};
use crate::lie::{LieScalar, Se3};
use crate::linearize::LinearizeError;
use crate::marg::{MargError, NullspaceCheck};
use crate::types::{
    AbsOrderMap, FrameId, KeypointId, LandmarkId, MargLinData, POSE_VEL_BIAS_SIZE,
    PoseVelBiasState, PoseVelBiasStateWithLin, PoseVelState, StateError, TimeCamId,
};

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
/// **Where the window is left, per error.** The validation errors — a frameset
/// of the wrong width, a non-monotonic frameset, an unsupported config — are
/// raised before anything moves, so the window is untouched and the caller may
/// retry with a corrected frameset. The errors raised inside
/// [`SqrtKeypointVio::measure`] — `NumericallyInvalid` from the LM loop, and
/// anything `Linearize`, `Marginalize` or `BundleAdjustment` refuses — come
/// **after** the new state, its observations and its preintegration were
/// inserted, so the window has advanced by one frameset while `prev_frame` has
/// not: retrying the same frameset would file its observations twice. basalt
/// resets the whole estimator instead (`proc_func`'s `return false`,
/// `scheduleResetState` at `:120-195`), which this port does not have; a caller
/// that sees one of those must rebuild the estimator. Stage S9's Realtime mode
/// is where the reset belongs (D5 of the S8 simplify list).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
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
    /// `logMargNullspace` (`:670-681`), present only with `vio_debug` or
    /// `vio_extended_logging`.
    pub nullspace: Option<NullspaceCheck>,
    /// `checkMargEigenvalues()` (`:695`), ascending, under the same gate.
    pub nullspace_eigenvalues: Option<Vec<f64>>,
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
    /// Landmark positions in the world frame, with their ids.
    pub landmarks: Vec<(LandmarkId, Vector3<S>)>,
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
    /// `nullspace_marg_data` (`:224`), the prior-free copy the diagnostics read.
    nullspace_marg_data: MargLinData<S>,
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
    /// Frames the last marginalization removed, for [`Self::snapshot`].
    last_marginalized: Vec<FrameId>,
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
    /// not positive, when the rig has fewer than two cameras, or when a camera
    /// model has no projection.
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

        let noise: ImuNoise<S> = ImuNoise::from_calibration(&calibration);
        let gyro_bias_sqrt_weight: Vector3<S> = calibration.gyro_bias_std.map(|v| S::one() / v);
        let accel_bias_sqrt_weight: Vector3<S> = calibration.accel_bias_std.map(|v| S::one() / v);

        // `:71-73`.
        let obs_std_dev: S = S::from_literal(config.vio_obs_std_dev);
        let huber_thresh: S = S::from_literal(config.vio_obs_huber_thresh);
        let ba: BundleAdjustmentBase<S> =
            BundleAdjustmentBase::new(calibration, obs_std_dev, huber_thresh)?;

        let mut marg_data: MargLinData<S> = MargLinData {
            is_sqrt: config.vio_sqrt_marg,
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

        // `:80-84`: the debug copy starts at the same shape with **no** prior.
        let nullspace_marg_data: MargLinData<S> = MargLinData {
            is_sqrt: marg_data.is_sqrt,
            order: AbsOrderMap::new(),
            h: DMatrix::zeros(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE),
            b: DVector::zeros(POSE_VEL_BIAS_SIZE),
        };

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
            nullspace_marg_data,
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
            last_marginalized: Vec::new(),
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
    /// is dropped rather than reordered, because the integration reads the
    /// stream strictly forward. The static bias calibration
    /// (`calib_bias.hpp:101-107`) is applied when the sample is popped, as
    /// `:298-299` does, not here.
    pub fn push_imu(&mut self, sample: ImuSample) {
        if let Some(last) = self.imu_queue.back()
            && sample.t_ns <= last.t_ns
        {
            return;
        }
        self.imu_queue.push_back(sample);
    }

    /// Whether the window has a state (`initialized`, `:233`).
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Whether `optimize()` has started running (`opt_started`, `:1207`).
    pub fn optimization_started(&self) -> bool {
        self.opt_started
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

    /// The long-term keyframes, oldest first (`ltkfs`, `:205`).
    pub fn ltkfs(&self) -> impl Iterator<Item = FrameId> + '_ {
        self.ltkfs.iter().copied()
    }

    /// The live marginalization prior.
    pub fn marg_data(&self) -> &MargLinData<S> {
        &self.marg_data
    }

    /// The prior-free debug copy the nullspace diagnostics read.
    pub fn nullspace_marg_data(&self) -> &MargLinData<S> {
        &self.nullspace_marg_data
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

        let mut landmarks: Vec<(LandmarkId, Vector3<S>)> = Vec::new();
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
            landmarks.push((lm.id, t_w_c * point_c));
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
    /// calls [`Self::measure`].
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
        // moves. The integration needs a sample strictly past the frameset,
        // because a sample landing exactly on `t_ns` is consumed and the loop
        // pops again (`:322-328`).
        let covered: bool = self
            .pending
            .map(|(t_ns, _, _)| t_ns > frame.t_ns)
            .unwrap_or(false)
            || self
                .imu_queue
                .back()
                .is_some_and(|last| last.t_ns > frame.t_ns);
        if !covered {
            return Ok(FrameOutcome::NeedMoreImu);
        }

        if self.pending.is_none() {
            self.pending = self.pop_calibrated();
        }

        let mut meas: Option<IntegratedImuMeasurement<S>> = None;

        if !self.initialized {
            // `:265-271`: skip forward to the frameset, then take that sample's
            // accelerometer reading as the whole initialization.
            while let Some((t_ns, _, _)) = self.pending {
                if t_ns >= frame.t_ns {
                    break;
                }
                self.pending = self.pop_calibrated();
            }
            let Some((_, _, accel)) = self.pending else {
                return Ok(FrameOutcome::NeedMoreImu);
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

            // `:315-320`: discard everything at or before the previous frameset.
            while let Some((t_ns, _, _)) = self.pending {
                if t_ns > prev.t_ns {
                    break;
                }
                self.pending = self.pop_calibrated();
            }
            // `:322-328`: integrate everything up to and including the frameset.
            while let Some((t_ns, gyro, accel)) = self.pending {
                if t_ns > frame.t_ns {
                    break;
                }
                pim.integrate_calibrated(
                    t_ns,
                    &accel,
                    &gyro,
                    &self.noise.accel_cov,
                    &self.noise.gyro_cov,
                )?;
                self.pending = self.pop_calibrated();
            }
            // `:330-336`: close the interval exactly on the frameset by
            // re-stamping the next sample. basalt restores the timestamp
            // afterwards, so the sample is still available to the next frame at
            // its own time.
            if pim.get_start_t_ns() + pim.get_dt_ns() < frame.t_ns
                && let Some((_, gyro, accel)) = self.pending
            {
                pim.integrate_calibrated(
                    frame.t_ns,
                    &accel,
                    &gyro,
                    &self.noise.accel_cov,
                    &self.noise.gyro_cov,
                )?;
            }
            meas = Some(pim);
        }

        let stats: FrameStats<S> = self.measure(Arc::clone(&frame), meas)?;
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

        // `:566`.
        let (lm, termination, mut timings) = self.optimize(frame.t_ns)?;
        let marg: MarginalizationOutcome =
            self.marginalize(&num_points_connected, &lost_landmarks)?;
        timings.marginalize_ns = marg.elapsed_ns;
        timings.measure_ns = duration_ns(started);
        let (nullspace, nullspace_eigenvalues) = marg.nullspace.unzip();

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
            nullspace,
            nullspace_eigenvalues,
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
                let Some(p0_pixel) = host_keypoints.get(kpt_id) else {
                    continue;
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

                    // `:526-543`.
                    let triangulated: Vector4<S> = triangulate(
                        &Vector3::new(p0_3d[0], p0_3d[1], p0_3d[2]),
                        &Vector3::new(p1_3d[0], p1_3d[1], p1_3d[2]),
                        &t_0_1,
                    );
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

/// `AffineCompact2f::translation().cast<Scalar>()` (`:437`).
fn cast_pixel<S: LieScalar>(pixel: &Vector2<f32>) -> Vector2<S> {
    Vector2::new(
        S::from_literal(f64::from(pixel.x)),
        S::from_literal(f64::from(pixel.y)),
    )
}

/// Elapsed nanoseconds, saturating rather than panicking on an absurd clock.
fn duration_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::config::VioConfig;

    const CALIB: &str = include_str!("../../tests/fixtures/msdmi_calib.json");
    const CONFIG: &str = include_str!("../../tests/fixtures/msdmi_config.json");

    /// The rig is one list of cameras, and
    /// [`SqrtKeypointVio::triangulate_unconnected`] indexes the projections and
    /// the extrinsics with the same id.
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
}
