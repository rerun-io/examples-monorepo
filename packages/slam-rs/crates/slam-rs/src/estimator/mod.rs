//! The offline sliding-window VIO driver.
//!
//! The driver composes bundle adjustment, IMU preintegration, square-root
//! linearization and marginalization. `process_frame` initializes from an
//! accelerometer sample, integrates `(prev_t, curr_t]`, and calls `measure`.
//! Measurement predicts the state, files observations, votes on a keyframe,
//! triangulates landmarks, then optimizes and marginalizes.
//!
//! Processing is synchronous (D17). IMU samples enter an unbounded buffer and
//! frames run to completion in the calling thread. Realtime queue dropping is
//! unsupported and `vio_enforce_realtime` is refused. Snapshots return values
//! for visualization. Long-term keyframes are supported; mapper output is not.
//!
//! ## Where an error leaves the window
//!
//! The call site determines whether an error is retryable:
//! 1. Validation before mutation leaves the estimator untouched. Correct the
//!    input and retry, or construct a new estimator after a constructor error.
//! 2. Errors after IMU consumption but before state insertion leave the old
//!    window but consume the required samples. Rebuild the estimator.
//! 3. Errors after insertion leave an advanced window; retrying would duplicate
//!    observations. Rebuild the estimator.
//!
//! `BundleAdjustment` and `State` errors occur at multiple sites, so their
//! variants alone do not establish retryability. Initialization also inserts a
//! state before its first ordering entry. That ordering insertion cannot fail
//! for a fresh map and fixed block size, but still uses a typed result (D32).

use kornia_staging_algebra::Scalar;
mod deferred;
mod error;
mod frame_update;
mod optimize;
mod report;
mod schedule;
mod snapshot;
mod triangulation;

pub use snapshot::{SnapshotLandmark, VelBias, WindowSnapshot, WindowState};

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Vector2, Vector3};

use crate::ba_base::BundleAdjustmentBase;
use crate::calib::Calibration;
use crate::config::{LinearizationType, VioConfig};
use crate::duration_ns;
use crate::frontend::parallel::WorkPool;
use crate::imu::{
    ImuLinData, ImuNoise, ImuSample, IntegratedImuMeasurement, Popped, gravity,
    gravity_from_first_accel,
};
use crate::lie::{Se3, eigen_maxi};
use crate::types::{
    AbsOrderMap, FrameId, KeypointId, LandmarkId, MargLinData, POSE_VEL_BIAS_SIZE,
    PoseVelBiasState, PoseVelBiasStateWithLin, PoseVelState, TimeCamId,
};

pub use error::{EstimatorError, WindowRole};
pub use report::{FlowObservations, FrameOutcome, FrameStats, StageTimings};

use deferred::DeferredKeyframe;
pub use deferred::DeferredKeyframeStats;
pub use frame_update::{FrameUpdateDecline, FrameUpdateOutcome};
use frame_update::{FrameUpdateResult, FrameUpdateScratch};
use optimize::OptimizeScratch;
pub use optimize::{LmIteration, LmTermination};
use schedule::MarginalizationOutcome;
pub use schedule::{EvictionReason, KeyframeEviction, MarginalizationStats};

/// LM damping, the four fields of in one place.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LmDamping<S: Scalar> {
    /// `lambda`, reset to `vio_lm_lambda_initial` every frame (D11).
    lambda: S,
    /// `min_lambda`, the floor of the damped diagonal.
    min_lambda: S,
    /// `max_lambda`; exceeding it terminates the frame.
    max_lambda: S,
    /// `lambda_vee`, the Nielsen escalation factor, reset to 2 on an accept.
    lambda_vee: S,
}

/// The compile-time Nielsen escalation constants, both 2.0.
const VEE_FACTOR: f64 = 2.0;

/// Minimum landmark support for a world start or a deferred keyframe solve.
pub(crate) const MIN_LANDMARK_SUPPORT: usize = 10;

impl<S: Scalar> LmDamping<S> {
    /// Nielsen's update after an accepted step.
    /// The cubic and subtraction run in f64 before narrowing. Both maxima retain
    /// a NaN on the left, allowing the caller to detect non-finite damping.
    fn accept(&mut self, relative_decrease: S) {
        let x: S = S::from_literal(2.0) * relative_decrease - S::one();
        let gain: S = S::from_literal(1.0 - x.to_f64().powf(3.0));
        let floor: S = S::one() / S::from_literal(3.0);
        self.lambda *= eigen_maxi(floor, gain);
        self.lambda = eigen_maxi(self.min_lambda, self.lambda);
        self.lambda_vee = S::from_literal(VEE_FACTOR);
    }

    /// Geometrically escalate damping after a rejected step or failed solve.
    fn escalate(&mut self) {
        self.lambda = self.lambda_vee * self.lambda;
        self.lambda_vee *= S::from_literal(VEE_FACTOR);
    }

    /// Check whether escalation exceeded the maximum lambda after rollback.
    fn exhausted(&self) -> bool {
        self.lambda > self.max_lambda
    }
}

/// Shared convergence predicate for the window and frame-update schedules.
/// Both use the same fixed cost and infinity-norm tolerances.
fn lm_converged<S: Scalar>(f_diff: S, step_norminf: S) -> bool {
    (f_diff > S::zero() && f_diff < S::from_literal(optimize::FUNCTION_TOLERANCE))
        || step_norminf < S::from_literal(optimize::STEP_TOLERANCE)
}

/// `SqrtKeypointVioEstimator<Scalar>`.
#[derive(Debug, Clone)]
pub struct SqrtKeypointVio<S: Scalar> {
    /// The sliding window: `frame_states`, `frame_poses`, `lmdb` and the
    /// calibration. Crate-visible because [`crate::marg::marginalize`] takes
    /// it by `&mut` and [`Self::snapshot`] reads it; outside the crate the
    /// snapshot is the window's only view.
    pub(crate) ba: BundleAdjustmentBase<S>,

    /// `prev_frame`, the frameset the last `measure` consumed.
    prev_frame: Option<Arc<FlowObservations>>,
    /// `take_kf`, true at construction so the first frame is a
    /// keyframe.
    take_kf: bool,
    /// Frames since the last keyframe, used to rate-limit keyframe selection.
    frames_after_kf: i32,
    /// `frame_count`, the source of `frame_idx`.
    frame_count: usize,
    /// `kf_ids`.
    kf_ids: BTreeSet<FrameId>,
    /// `ltkfs`, exempt from the `max_kfs` budget.
    ltkfs: BTreeSet<FrameId>,
    /// `take_ltkf`.
    take_ltkf: bool,
    /// `frame_idx`.
    frame_idx: BTreeMap<FrameId, usize>,
    /// `last_state_t_ns`.
    last_state_t_ns: i64,
    /// `imu_meas`, one preintegration per consecutive state pair.
    imu_meas: BTreeMap<i64, IntegratedImuMeasurement<S>>,
    /// `g`, `(0, 0, -9.81)` unless the caller overrides it.
    g: Vector3<S>,
    /// `prev_opt_flow_res`, kept so a new keyframe can gather every
    /// observation of a keypoint across the live window.
    prev_opt_flow_res: BTreeMap<FrameId, Arc<FlowObservations>>,
    /// Hosted-point counts retained even after a keyframe leaves the window.
    num_points_kf: BTreeMap<FrameId, usize>,
    /// `marg_data`, the square-root prior.
    marg_data: MargLinData<S>,
    /// `gyro_bias_sqrt_weight`, `1 / gyro_bias_std`.
    gyro_bias_sqrt_weight: Vector3<S>,
    /// `accel_bias_sqrt_weight`.
    accel_bias_sqrt_weight: Vector3<S>,
    /// `max_states`.
    max_states: usize,
    /// `max_kfs`.
    max_kfs: usize,
    /// `T_w_i_init`, the pose the first accelerometer sample gave.
    t_w_i_init: Se3<S>,
    /// `initialized`.
    initialized: bool,
    /// `opt_started`, which flips once `frame_states.size() > 4`.
    opt_started: bool,
    /// `config`.
    config: VioConfig,
    /// The four damping configuration fields.
    damping: LmDamping<S>,
    /// The estimator's own preintegration noise ( of the constructor
    /// body). The frontend runs a second, independent preintegrator (D24) and
    /// the two are deliberately not shared.
    noise: ImuNoise<S>,
    /// The gyro and accel bias the caller initialised with, re-used for every
    /// new `IntegratedImuMeasurement` before the first state exists.
    initial_bias_gyro: Vector3<S>,
    /// See [`Self::initial_bias_gyro`].
    initial_bias_accel: Vector3<S>,

    /// `imu_data_queue`, unbounded here: Offline mode
    /// never drops a sample and never blocks (D17, D24).
    imu_queue: VecDeque<ImuSample>,
    /// The calibrated sample already popped from the queue, retained across frames.
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

    /// A keyframe's second half, pending (D84). Every entry point that reads or
    /// moves the window finishes it first, so only the timing of that work
    /// depends on the caller, never its result.
    deferred: Option<DeferredKeyframe>,
}

/// Validate numerical domains before estimator arithmetic runs (D32).
/// Bias and observation deviations are divisors; IMU rate and initial prior
/// weights enter square roots; damping values enter comparisons. Invalid values
/// must be refused before they create NaNs or infinities in a live prior.
/// Parsers remain syntax-only because these constraints belong to the consuming
/// algorithm. The Nielsen factor is a compile-time constant, not a config field.
fn validate_scalars<S: Scalar>(
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

impl<S: Scalar> SqrtKeypointVio<S> {
    /// `SqrtKeypointVioEstimator(g, calib, config)`.
    ///
    /// Sets the square-root gauge prior on the first state: `sqrt(init_pose_weight)`
    /// on indices 0 to 2 and on index 5 alone, and `sqrt(init_ba_weight)` /
    /// `sqrt(init_bg_weight)` on 9 to 11 and 12 to 14 (D18). Roll and
    /// pitch (3 and 4) are left free because gravity observes them. The
    /// **square root** is what the sqrt branch stores; the Hessian branch
    ///  stores the weight itself and is unreachable here (D13).
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
        //  and : a hard precondition,
        // because the epipolar filter needs a second camera.
        if calibration.t_i_c.len() < 2 {
            return Err(EstimatorError::CameraCountMismatch {
                expected: 2,
                actual: calibration.t_i_c.len(),
            });
        }
        // Each camera id indexes both intrinsics and extrinsics. Reject ragged lists
        // before triangulation can read beyond either list.
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

        let obs_std_dev: S = S::from_literal(config.vio_obs_std_dev);
        let huber_thresh: S = S::from_literal(config.vio_obs_huber_thresh);
        let ba: BundleAdjustmentBase<S> =
            BundleAdjustmentBase::new(calibration, obs_std_dev, huber_thresh)?;

        let mut marg_data: MargLinData<S> = MargLinData {
            order: AbsOrderMap::new(),
            h: DMatrix::zeros(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE),
            b: DVector::zeros(POSE_VEL_BIAS_SIZE),
        };
        // the square-root branch.
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
            deferred: None,
        })
    }

    /// Construct with gravity `(0, 0, -9.81)`.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn with_default_gravity(
        calibration: Calibration<S>,
        config: VioConfig,
    ) -> Result<Self, EstimatorError> {
        Self::new(gravity::<S>(), calibration, config)
    }

    /// `takeLongTermKeyframe()` : the next `measure` moves the
    /// newest keyframe into `ltkfs`, where the `max_kfs` budget cannot evict it.
    ///
    /// Nothing in the VIO path calls this; it is a Monado/API hook.
    pub fn take_long_term_keyframe(&mut self) {
        self.take_ltkf = true;
    }

    /// Append an IMU sample only if it follows the newest accepted timestamp.
    /// That timestamp includes the pending sample even when the queue is empty.
    /// Direct callers have older samples dropped; [`Vio::push_imu`](crate::Vio::push_imu)
    /// refuses them with a typed error. Static bias calibration occurs when popping.
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
    /// consumed and the integration loop pops again. The newest
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

    /// Whether the window has a state (`initialized`).
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// `get_t_ns()`, the newest state's timestamp.
    pub fn last_state_t_ns(&self) -> i64 {
        self.last_state_t_ns
    }

    /// `get_state()`, the newest state, or `None` before the window has
    /// one.
    pub fn state(&self) -> Option<&PoseVelBiasState<S>> {
        self.ba
            .frame_states
            .get(&self.last_state_t_ns)
            .map(PoseVelBiasStateWithLin::state)
    }

    /// The keyframes, oldest first (`kf_ids`).
    pub fn kf_ids(&self) -> impl Iterator<Item = FrameId> + '_ {
        self.kf_ids.iter().copied()
    }

    /// The live marginalization prior.
    pub fn marg_data(&self) -> &MargLinData<S> {
        &self.marg_data
    }

    /// `num_points_kf`, landmarks hosted per keyframe when it was
    /// created.
    pub fn num_points_kf(&self) -> &BTreeMap<FrameId, usize> {
        &self.num_points_kf
    }

    /// The preintegrated intervals, keyed by their start timestamp.
    pub fn imu_meas(&self) -> &BTreeMap<i64, IntegratedImuMeasurement<S>> {
        &self.imu_meas
    }

    /// Process one frameset synchronously.
    /// Initialize from the first accelerometer sample only when the first
    /// keyframe hosts at least ten stereo landmarks; otherwise consume the
    /// frameset without a state. Once initialized, preintegrate
    /// `(prev_t, curr_t]` using the previous state's biases and call `measure`.
    /// Return [`FrameOutcome::NeedMoreImu`] without mutation when coverage does not
    /// extend past the frameset. Arrival order must not affect decisions (D17).
    ///
    /// # Errors
    /// [`EstimatorError`] for invalid frameset order or width, or a failure in `measure`.
    pub fn process_frame(
        &mut self,
        frame: Arc<FlowObservations>,
        pool: Option<&WorkPool>,
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
            // both asserts.
            return Err(EstimatorError::NonMonotonicFrame {
                previous_t_ns: prev.t_ns,
                t_ns: frame.t_ns,
            });
        }

        // A deferred keyframe's second half belongs before this frameset (D84).
        // `Vio::track` has normally run it beside the frontend already; a
        // caller driving the estimator directly gets it here, synchronously.
        // It reads no IMU, so the coverage refusal below stays side-effect free
        // for the IMU queue either way.
        if self.deferred.is_some() && self.imu_covers_frame(frame.t_ns) {
            self.finish_deferred_keyframe(pool)?;
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

        let initializing = !self.initialized;
        if initializing {
            // skip forward to the frameset, then take that sample's
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

            // zero velocity, zero translation, and the rotation that
            // takes the measured acceleration onto +Z.
            let vel_w_i_init: Vector3<S> = Vector3::zeros();
            let initial_pose = Se3::new(gravity_from_first_accel(&accel), Vector3::zeros());
            if self.count_initial_landmarks(&frame, initial_pose)? < MIN_LANDMARK_SUPPORT {
                self.frame_count += 1;
                self.prev_frame = Some(frame);
                return Ok(FrameOutcome::NoVisualFeatures);
            }
            self.t_w_i_init = initial_pose;

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

            // the first ordering entry, one 15-dof state at index 0.
            let mut order: AbsOrderMap = AbsOrderMap::new();
            order.push(self.last_state_t_ns, POSE_VEL_BIAS_SIZE)?;
            self.marg_data.order = order;

            self.initialized = true;
        } else if let Some(prev) = self.prev_frame.clone() {
            let (bias_gyro, bias_accel) = match self.ba.frame_states.get(&self.last_state_t_ns) {
                Some(state) => (state.state().bias_gyro, state.state().bias_accel),
                None => (self.initial_bias_gyro, self.initial_bias_accel),
            };
            let mut pim: IntegratedImuMeasurement<S> =
                IntegratedImuMeasurement::new(prev.t_ns, &bias_gyro, &bias_accel);

            // the loop `IntegratedImuMeasurement::accumulate_to`
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
        let mut stats: FrameStats<S> = self.measure(Arc::clone(&frame), meas, pool)?;
        stats.timings.predict_ns += integration_ns;
        if initializing {
            log::info!(
                "slam-rs: world started at frameset {} (t_ns={}), landmarks={}",
                self.frame_count - 1,
                frame.t_ns,
                stats.num_points_added
            );
        }
        // and only on success.
        self.prev_frame = Some(frame);
        Ok(FrameOutcome::Measured(Box::new(stats)))
    }

    /// Pop an IMU sample, cast it to the estimator scalar, then apply static bias
    /// calibration in that scalar.
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

    /// `measure(opt_flow_meas, meas)`.
    ///
    /// **Contract: `frame.cameras.len() == self.ba.calib.t_i_c.len()`, which is
    /// at least two.** [`Self::process_frame`] checks the frameset width
    /// against the rig and [`Self::new`] refuses a rig of fewer than
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
        pool: Option<&WorkPool>,
    ) -> Result<FrameStats<S>, EstimatorError> {
        let started: std::time::Instant = std::time::Instant::now();
        let num_cams: usize = frame.cameras.len();
        debug_assert_eq!(num_cams, self.ba.calib.t_i_c.len());

        // predict the new state from the previous one and the
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

        self.prev_opt_flow_res
            .insert(frame.t_ns, Arc::clone(&frame));

        // file every observation the window already hosts, and
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

        // D21: camera 0 alone votes, rate-limited to one keyframe
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

        // D84: a keyframe frameset solves its newest state the way every other
        // frameset does and returns that pose; triangulation, the joint solve
        // and the marginalization wait for `finish_deferred_keyframe`. The
        // update runs before triangulation, so it sees exactly the landmarks a
        // non-keyframe frameset would. A declined update keeps the synchronous
        // keyframe below.
        let deferred_update = if self.take_kf
            && self.config.port_keyframe_solve_deferred
            && self.config.port_frame_update_max_iterations > 0
            && self.opt_started
            && self.ba.lmdb.num_landmarks() >= MIN_LANDMARK_SUPPORT
            && connected.iter().sum::<usize>() >= MIN_LANDMARK_SUPPORT
        {
            let optimize_started = std::time::Instant::now();
            self.frame_update(frame.t_ns)?
                .ok()
                .map(|(lm, termination, mut timings)| {
                    timings.optimize_ns = duration_ns(optimize_started);
                    timings.predict_ns = predict_ns;
                    (lm, termination, timings)
                })
        } else {
            None
        };
        let keyframe_deferred = deferred_update.is_some();

        let took_keyframe: bool = self.take_kf;
        let mut num_points_added: usize = 0;
        if self.take_kf {
            self.take_kf = false;
            self.frames_after_kf = 0;
            self.kf_ids.insert(self.last_state_t_ns);
            if !keyframe_deferred {
                num_points_added = self.triangulate_unconnected(&frame, &unconnected_obs)?;
                self.num_points_kf.insert(frame.t_ns, num_points_added);
            }
        } else {
            self.frames_after_kf += 1;
        }

        let keyframe_ns: u64 = if took_keyframe && !keyframe_deferred {
            duration_ns(keyframe_started)
        } else {
            0
        };

        let lost_landmarks: BTreeSet<LandmarkId> = self.lost_landmarks(&frame);

        let num_lost_landmarks = lost_landmarks.len();
        let unconnected = unconnected_obs.iter().map(BTreeSet::len).collect();
        let mut tail: MeasureTail<S> = match deferred_update {
            Some((lm, termination, mut timings)) => {
                self.deferred = Some(DeferredKeyframe {
                    frame: Arc::clone(&frame),
                    unconnected_obs,
                    num_points_connected,
                    lost_landmarks,
                });
                timings.keyframe_ns = duration_ns(keyframe_started);
                MeasureTail {
                    lm,
                    termination,
                    timings,
                    frame_update: FrameUpdateOutcome::Taken,
                    marginalization: None,
                }
            }
            None => {
                // plus D76's gate: above zero, `port.frame_update_max_iterations`
                // moves the joint solve to the framesets that took a keyframe and gives
                // the others the newest state alone. The frame update declines a
                // frameset it cannot serve, and the joint solve owns the warmup.
                let optimize_started: std::time::Instant = std::time::Instant::now();
                let attempt_frame_update: bool = self.config.port_frame_update_max_iterations > 0
                    && !took_keyframe
                    && self.opt_started;
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
                    Some(Err(_)) | None => self.optimize(frame.t_ns, pool)?,
                };
                timings.optimize_ns = duration_ns(optimize_started);
                timings.predict_ns = predict_ns;
                timings.keyframe_ns = keyframe_ns;
                let marg: MarginalizationOutcome =
                    self.marginalize(&num_points_connected, &lost_landmarks)?;
                timings.marginalize_ns = marg.elapsed_ns;
                MeasureTail {
                    lm,
                    termination,
                    timings,
                    frame_update,
                    marginalization: marg.marginalization,
                }
            }
        };
        tail.timings.measure_ns = duration_ns(started);

        // read off the window the two stages above left behind.
        Ok(FrameStats {
            t_ns: frame.t_ns,
            visually_supported: self.ba.lmdb.num_landmarks() >= MIN_LANDMARK_SUPPORT
                && connected.iter().sum::<usize>() >= MIN_LANDMARK_SUPPORT
                && self.opt_started
                && self.state().is_some_and(|state| {
                    state.t_w_i.translation.iter().all(|v| v.is_finite())
                        && state
                            .t_w_i
                            .rotation
                            .quaternion()
                            .coords
                            .iter()
                            .all(|v| v.is_finite())
                }),
            connected,
            unconnected,
            took_keyframe,
            keyframe_vote,
            frames_after_kf: self.frames_after_kf,
            num_points_added,
            kf_ids: self.kf_ids.iter().copied().collect(),
            ltkfs: self.ltkfs.iter().copied().collect(),
            num_landmarks: self.ba.lmdb.num_landmarks(),
            num_observations: self.ba.lmdb.num_observations(),
            num_lost_landmarks,
            opt_started: self.opt_started,
            lm: tail.lm,
            termination: tail.termination,
            marginalization: tail.marginalization,
            frame_update: tail.frame_update,
            timings: tail.timings,
            keyframe_deferred,
        })
    }

    /// `ImuLinData` as and build it.
    fn imu_lin_data(&self) -> ImuLinData<S> {
        ImuLinData {
            g: self.g,
            gyro_bias_weight_sqrt: self.gyro_bias_sqrt_weight,
            accel_bias_weight_sqrt: self.accel_bias_sqrt_weight,
        }
    }
}

/// What the two ends of `measure` (a deferred keyframe, or the solve and the
/// marginalization now) hand its report.
struct MeasureTail<S: Scalar> {
    lm: Vec<LmIteration<S>>,
    termination: LmTermination,
    timings: StageTimings,
    frame_update: FrameUpdateOutcome,
    marginalization: Option<MarginalizationStats>,
}

/// Keyframes whose pose Jacobians are zeroed in both optimization and marginalization.
/// With `vio_fix_long_term_keyframes`, long-term poses stay fixed in the prior
/// and in the solved increment. `None` and an empty set have the same numerical effect.
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

/// `AffineCompact2f::translation().cast<Scalar>()`.
fn cast_pixel<S: Scalar>(pixel: &Vector2<f32>) -> Vector2<S> {
    Vector2::new(
        S::from_literal(f64::from(pixel.x)),
        S::from_literal(f64::from(pixel.y)),
    )
}

#[cfg(test)]
mod tests;
