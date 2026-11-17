//! The inertial bootstrap driver (`config.vio_bootstrap = "inertial"`).
//!
//! The stereo start needs landmarks triangulated inside the first frameset
//! and assumes the rig is at rest. A rig without stereo overlap (a
//! back-to-back 360 camera, one camera of a stereo pair) that starts moving
//! has neither. While bootstrapping, [`Vio`] runs the frontend as usual but
//! hands each frameset's observations to [`InertialBootstrap`] instead of the
//! estimator:
//!
//! 1. The frontend gets a gyro-only rotation prediction: [`Vio`]'s own
//!    frontend preintegration from an identity state at rest with zero
//!    biases, of which only the rotation is kept. The KLT has a guess before
//!    any state exists.
//! 2. The observations and the IMU (raw, and calibrated once as it arrives)
//!    are buffered. When the buffer spans the window (2 s), the observations
//!    become unit bearings ([`prepare_bearings`]) and
//!    `kornia_staging_slam::init::inertial::solve_inertial_init` runs. A
//!    rolling-shutter rig's observations arrive already corrected:
//!    `Vio::finish_track` corrects the estimator's copy before it reaches
//!    either the bootstrap or the estimator, so the buffer, the bearings and
//!    the replayed framesets are the same corrected observations.
//! 3. On a rejection of the first window it grows to 3 s. Each later
//!    rejection slides the 3 s window by 0.5 s, so the next attempt comes
//!    0.5 s of framesets later: one solve per 0.5 s, never two on
//!    consecutive framesets. Every attempt is logged and kept in the
//!    [`BootstrapReport`].
//! 4. On acceptance [`Vio`] builds a fresh estimator with
//!    [`estimator::SqrtKeypointVio::with_initial_state`] at the window's
//!    first frameset, pushes the buffered IMU, replays the buffered
//!    framesets through the normal `process_frame` and continues live.
//!    The replayed framesets publish no pose: the first pose is the
//!    frameset that completed the accepted window. The driver keeps its
//!    buffers and its report names no start until the replay has succeeded
//!    ([`InertialBootstrap::commit`]); a replay error leaves it as the
//!    attempt left it, so the next frameset solves again.
//! 5. The buffer holds at most the grown window. A window without
//!    excitation (the rig at rest) means "not yet" and costs no time; once
//!    the windows that had excitation span [`BootstrapOptions::timeout_s`]
//!    without an acceptance, `track` fails with
//!    [`BootstrapError::NeverAccepted`] on that frameset and every later one.
//!
//! Frameset times must be on the IMU clock and refer to the instant the
//! observations describe (for a rolling-shutter camera: the middle row's
//! mid-exposure, with the rows already corrected to it upstream).
use super::{Vio, VioError, VioResult, VioStatus};
use crate::calib::{BasaltCamera, Calibration};
use crate::camera::SlamCamera;
use crate::config::{VioBootstrap, VioConfig};
use crate::estimator::{self, FlowObservations, InitialState};
use crate::imu::Popped;
use crate::lie::Se3;
use kornia_staging_algebra::Scalar;
use kornia_staging_sensors::imu::CombinedImuSample;
use kornia_staging_slam::init::inertial::{
    BearingObservation, FeatureTrack, ImuSample, InertialInitError, InertialInitOptions,
    InertialInitOutcome, InertialInitProblem, InitVerdict, RejectReason, solve_inertial_init,
};
use nalgebra::{Vector3, Vector4};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// Window schedule and solver options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapOptions {
    /// First window length, seconds.
    pub window_s: f64,
    /// Length the window grows to after the first rejection, seconds.
    pub grow_s: f64,
    /// Slide of the grown window after each later rejection, seconds.
    pub slide_s: f64,
    /// Give up once the windows with excitation span this long without an
    /// acceptance, seconds.
    pub timeout_s: f64,
    /// Floor of the velocity prior's sigma, m/s.
    pub velocity_sigma_floor: f64,
    /// The solver's thresholds.
    pub solver: InertialInitOptions,
}

impl Default for BootstrapOptions {
    fn default() -> Self {
        Self {
            window_s: 2.0,
            grow_s: 3.0,
            slide_s: 0.5,
            timeout_s: 30.0,
            velocity_sigma_floor: 0.05,
            solver: InertialInitOptions::default(),
        }
    }
}

/// The name of a verdict: `Accepted` or the rejection reason's name.
pub fn verdict_name(verdict: &InitVerdict) -> &'static str {
    match verdict {
        InitVerdict::Accepted => "Accepted",
        InitVerdict::Rejected(reason) => match reason {
            RejectReason::LowExcitation { .. } => "LowExcitation",
            RejectReason::LowParallax { .. } => "LowParallax",
            RejectReason::TooFewTracks { .. } => "TooFewTracks",
            RejectReason::IllConditioned { .. } => "IllConditioned",
            RejectReason::ResidualTooHigh { .. } => "ResidualTooHigh",
            RejectReason::GravityNormInconsistent { .. } => "GravityNormInconsistent",
            RejectReason::NotConverged => "NotConverged",
        },
    }
}

/// One solve attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapAttempt {
    /// First frameset of the window, ns.
    pub first_t_ns: i64,
    /// Last frameset of the window, ns.
    pub last_t_ns: i64,
    /// Framesets in the window.
    pub frames: usize,
    /// Tracks handed to the solver.
    pub tracks: usize,
    /// The solver's verdict, with the numbers behind a rejection.
    pub verdict: InitVerdict,
    /// Excitation, m/s^2.
    pub excitation_mps2: f64,
    /// Inlier tracks.
    pub tracks_used: usize,
    /// RMS bearing error, rad.
    pub residual_rms_rad: Option<f64>,
    /// Condition number of the reduced linear system.
    pub condition: Option<f64>,
    /// Unconstrained `|g|`, m/s^2.
    pub gravity_norm_unconstrained: Option<f64>,
    /// Relative 1-sigma of the speed.
    pub scale_sigma: Option<f64>,
    /// Solved world velocity, m/s (also for late rejections).
    pub velocity_world: Option<[f64; 3]>,
    /// Solved gravity in the first IMU frame, m/s^2.
    pub gravity_body0: Option<[f64; 3]>,
    /// Solved gyro bias, rad/s.
    pub bias_gyro: Option<[f64; 3]>,
    /// Solved accelerometer bias, m/s^2.
    pub bias_accel: Option<[f64; 3]>,
}

impl BootstrapAttempt {
    /// Whether the solver accepted the window.
    pub fn accepted(&self) -> bool {
        self.verdict == InitVerdict::Accepted
    }
}

/// Every attempt and where the bootstrap started the estimator.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BootstrapReport {
    /// Attempts in order.
    pub attempts: Vec<BootstrapAttempt>,
    /// The accepted window's first frameset: the estimator's first state.
    pub start_t_ns: Option<i64>,
    /// The frameset that completed the accepted window: the first pose published.
    pub accepted_at_t_ns: Option<i64>,
    /// Framesets replayed into the fresh estimator.
    pub replayed: usize,
}

/// Why the bootstrap stopped.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BootstrapError {
    /// The windows with excitation spanned the timeout without an acceptance.
    #[error(
        "inertial bootstrap: no window accepted over {seconds:.1} s of windows with excitation after {attempts} attempts; last verdict {last_verdict}"
    )]
    NeverAccepted {
        /// Seconds of framesets covered by windows that had excitation.
        seconds: f64,
        /// Attempts made.
        attempts: usize,
        /// The last attempt's verdict name.
        last_verdict: &'static str,
    },
    /// The solver refused the window as malformed.
    #[error("inertial bootstrap: {0}")]
    Solver(#[from] InertialInitError),
    /// A camera model could not be built.
    #[error("inertial bootstrap: camera {0} has no usable model")]
    Camera(usize),
    /// The one-frame-lag schedule is not supported with the bootstrap.
    #[error("inertial bootstrap does not run with port.frontend_lag")]
    FrontendLag,
}

/// What a frameset did to the bootstrap.
#[derive(Debug)]
pub enum BootstrapStep {
    /// Still buffering or the attempt was rejected.
    Buffering,
    /// A window was accepted: start from it, then [`InertialBootstrap::commit`].
    Accepted(Box<Accepted>),
}

/// An accepted window and what the estimator needs to start from it: copies
/// of the driver's buffers, which it keeps until the start is committed.
#[derive(Debug)]
pub struct Accepted {
    /// The first state, at the first buffered frameset.
    pub initial: InitialState<f64>,
    /// Every buffered frameset, oldest first, ending with the current one.
    pub frames: Vec<Arc<FlowObservations>>,
    /// Every buffered IMU sample, raw (the estimator calibrates on pop).
    pub imu: Vec<CombinedImuSample>,
}

/// Turn a window's framesets into bearing tracks: the observation seam.
///
/// Each pixel is unprojected through its camera's model to a unit bearing in
/// that camera's frame; the frame index is the frameset's position in
/// `window`. A rolling-shutter camera's pixels are already corrected (rotated
/// by the gyro over each row's time offset from the frameset time, in
/// `Vio::finish_track`) when they get here; that is mandatory for the
/// Insta360: in the solver's Monte Carlo `sensitivity_study`
/// (`kornia_staging_slam::init::inertial` tests), an uncorrected +-8 ms
/// readout gave 6-16 % scale error. Before the bootstrap accepts there is no
/// gyro bias estimate, so that correction uses the raw gyro rate (the static
/// calibration's bias is not applied either; it is zero for the Insta360 rig).
pub fn prepare_bearings(
    window: &[Arc<FlowObservations>],
    cameras: &[SlamCamera<f64>],
) -> Vec<FeatureTrack> {
    let mut tracks: BTreeMap<u64, Vec<BearingObservation>> = BTreeMap::new();
    for (frame, observations) in window.iter().enumerate() {
        for (camera, keypoints) in observations.cameras.iter().enumerate() {
            let Some(model) = cameras.get(camera) else {
                continue;
            };
            for (id, pixel) in keypoints {
                let mut ray: Vector4<f64> = Vector4::zeros();
                if !model.unproject(&pixel.cast::<f64>(), &mut ray) {
                    continue;
                }
                tracks.entry(id.0).or_default().push(BearingObservation {
                    frame,
                    camera,
                    bearing: ray.xyz(),
                });
            }
        }
    }
    tracks
        .into_iter()
        .map(|(id, observations)| FeatureTrack { id, observations })
        .collect()
}

/// The bootstrap's buffers and schedule.
#[derive(Debug, Clone)]
pub struct InertialBootstrap {
    options: BootstrapOptions,
    cameras: Vec<SlamCamera<f64>>,
    calibration: Calibration<f64>,
    gravity_magnitude: f64,
    /// Raw samples, for the estimator a commit starts.
    imu: VecDeque<CombinedImuSample>,
    /// The same samples calibrated as they arrived, for the solver.
    imu_calibrated: VecDeque<ImuSample>,
    frames: VecDeque<Arc<FlowObservations>>,
    window_ns: i64,
    /// Frameset time covered by windows that had excitation, ns.
    excited_ns: i64,
    /// The last attempt's last frameset.
    last_attempt_t_ns: Option<i64>,
    report: BootstrapReport,
}

fn seconds_to_ns(s: f64) -> i64 {
    (s * 1e9).round() as i64
}

impl InertialBootstrap {
    /// A driver for this rig; `gravity_magnitude` must be the estimator's.
    ///
    /// # Errors
    /// [`BootstrapError::Camera`] when a camera model cannot unproject.
    pub fn new(
        calibration: Calibration<f64>,
        gravity_magnitude: f64,
        options: BootstrapOptions,
    ) -> Result<Self, BootstrapError> {
        let cameras: Vec<SlamCamera<f64>> = calibration
            .intrinsics
            .iter()
            .enumerate()
            .map(|(i, m): (usize, &BasaltCamera<f64>)| {
                SlamCamera::from_model(m).map_err(|_| BootstrapError::Camera(i))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            window_ns: seconds_to_ns(options.window_s),
            options,
            cameras,
            calibration,
            gravity_magnitude,
            imu: VecDeque::new(),
            imu_calibrated: VecDeque::new(),
            frames: VecDeque::new(),
            excited_ns: 0,
            last_attempt_t_ns: None,
            report: BootstrapReport::default(),
        })
    }

    /// The attempts so far.
    pub fn report(&self) -> &BootstrapReport {
        &self.report
    }

    /// Framesets buffered now; at most the grown window's.
    #[cfg(test)]
    pub fn buffered_frames(&self) -> usize {
        self.frames.len()
    }

    /// The newest IMU timestamp buffered.
    pub fn newest_imu_t_ns(&self) -> Option<i64> {
        self.imu.back().map(|s| s.timestamp_ns)
    }

    /// Whether the IMU reaches strictly past `t_ns` (the estimator's rule).
    pub fn imu_covers_frame(&self, t_ns: i64) -> bool {
        self.newest_imu_t_ns().is_some_and(|newest| newest > t_ns)
    }

    /// Buffer one raw sample and its calibrated value; the caller has checked
    /// its order.
    pub fn push_imu(&mut self, sample: CombinedImuSample, calibrated: Popped<f64>) {
        let (t_ns, gyro, accel) = calibrated;
        self.imu.push_back(sample);
        self.imu_calibrated
            .push_back(ImuSample { t_ns, gyro, accel });
        if self.frames.is_empty() {
            // Before the first frameset keep the timeout's span (at least one
            // second), so a caller that pushes IMU ahead of the framesets still
            // has the IMU the first frameset needs; `push_frame` trims the rest.
            let keep_after: i64 =
                sample.timestamp_ns - seconds_to_ns(self.options.timeout_s.max(1.0));
            while self.imu.len() > 1
                && self
                    .imu
                    .front()
                    .is_some_and(|s| s.timestamp_ns < keep_after)
            {
                self.pop_imu();
            }
        }
    }

    fn pop_imu(&mut self) {
        self.imu.pop_front();
        self.imu_calibrated.pop_front();
    }

    fn attempt(&mut self) -> Result<(InertialInitOutcome, usize), BootstrapError> {
        let window: Vec<Arc<FlowObservations>> = self.frames.iter().cloned().collect();
        let tracks: Vec<FeatureTrack> = prepare_bearings(&window, &self.cameras);
        let frame_times: Vec<i64> = window.iter().map(|f| f.t_ns).collect();
        let problem = InertialInitProblem {
            imu: self.imu_calibrated.make_contiguous(),
            frame_times_ns: &frame_times,
            tracks: &tracks,
            t_i_c: &self.calibration.t_i_c,
            gravity_magnitude: self.gravity_magnitude,
            gyro_bias_prior: None,
        };
        Ok((
            solve_inertial_init(&problem, &self.options.solver)?,
            tracks.len(),
        ))
    }

    fn record(&mut self, outcome: &InertialInitOutcome, tracks: usize) {
        let first_t_ns: i64 = self.frames.front().map_or(0, |f| f.t_ns);
        let last_t_ns: i64 = self.frames.back().map_or(0, |f| f.t_ns);
        let q = &outcome.quality;
        let s = outcome.solution.as_ref();
        let attempt = BootstrapAttempt {
            first_t_ns,
            last_t_ns,
            frames: self.frames.len(),
            tracks,
            verdict: outcome.verdict,
            excitation_mps2: q.excitation_mps2,
            tracks_used: q.tracks_used,
            residual_rms_rad: q.residual_rms_rad,
            condition: q.condition,
            gravity_norm_unconstrained: q.gravity_norm_unconstrained,
            scale_sigma: q.scale_sigma,
            velocity_world: s.map(|s| s.velocity_world.into()),
            gravity_body0: s.map(|s| s.gravity_body0.into()),
            bias_gyro: s.map(|s| s.bias_gyro.into()),
            bias_accel: s.map(|s| s.bias_accel.into()),
        };
        log::info!(
            "inertial bootstrap attempt {}: window {:.2} s, {} framesets, {} tracks ({} used): {:?}; excitation {:.3} m/s^2, residual {:?} rad, condition {:?}, |g| {:?}, scale sigma {:?}",
            self.report.attempts.len() + 1,
            (last_t_ns - first_t_ns) as f64 * 1e-9,
            attempt.frames,
            tracks,
            attempt.tracks_used,
            attempt.verdict,
            attempt.excitation_mps2,
            attempt.residual_rms_rad,
            attempt.condition,
            attempt.gravity_norm_unconstrained,
            attempt.scale_sigma,
        );
        // A window at rest is "not yet": only windows with excitation count
        // toward the timeout, each for the framesets since the last attempt.
        if !matches!(
            outcome.verdict,
            InitVerdict::Rejected(RejectReason::LowExcitation { .. })
        ) {
            self.excited_ns +=
                last_t_ns - self.last_attempt_t_ns.unwrap_or(first_t_ns).max(first_t_ns);
        }
        self.last_attempt_t_ns = Some(last_t_ns);
        self.report.attempts.push(attempt);
    }

    /// Slide the window start by `slide_s` and drop what precedes it.
    fn slide(&mut self) {
        let Some(first) = self.frames.front().map(|f| f.t_ns) else {
            return;
        };
        let cut: i64 = first + seconds_to_ns(self.options.slide_s);
        while self.frames.front().is_some_and(|f| f.t_ns < cut) {
            self.frames.pop_front();
        }
        self.trim_imu();
    }

    /// Keep the last sample at or before the first frameset and everything after.
    fn trim_imu(&mut self) {
        let Some(first) = self.frames.front().map(|f| f.t_ns) else {
            return;
        };
        while self.imu.len() > 1 && self.imu.get(1).is_some_and(|s| s.timestamp_ns <= first) {
            self.pop_imu();
        }
    }

    /// Buffer a frameset and, when the window is full, solve it.
    ///
    /// On acceptance the buffers stay as they are and the report names no
    /// start: the caller replays copies of them and then calls
    /// [`Self::commit`].
    ///
    /// # Errors
    /// [`BootstrapError::NeverAccepted`] once the windows with excitation
    /// span the timeout (on that frameset and every later one), and
    /// malformed windows from the solver.
    pub fn push_frame(
        &mut self,
        frame: Arc<FlowObservations>,
    ) -> Result<BootstrapStep, BootstrapError> {
        self.check_timeout()?;
        let t_ns: i64 = frame.t_ns;
        self.frames.push_back(frame);
        // A window starts where the IMU does: the solver integrates from its
        // first frameset.
        let imu_start: Option<i64> = self.imu.front().map(|s| s.timestamp_ns);
        while self
            .frames
            .front()
            .is_some_and(|f| imu_start.is_none_or(|start| f.t_ns < start))
        {
            self.frames.pop_front();
        }
        self.trim_imu();
        let first: i64 = self.frames.front().map_or(t_ns, |f| f.t_ns);
        if t_ns - first < self.window_ns {
            return Ok(BootstrapStep::Buffering);
        }
        let (outcome, tracks) = self.attempt()?;
        self.record(&outcome, tracks);
        if let Some(solution) = outcome.accepted() {
            let velocity_sigma: f64 = outcome
                .quality
                .scale_sigma
                .map_or(f64::INFINITY, |s| s * solution.velocity_world.norm())
                .max(self.options.velocity_sigma_floor);
            let initial = InitialState {
                t_ns: first,
                t_w_i: Se3::new(solution.world_from_body0, Vector3::zeros()),
                vel_w_i: solution.velocity_world,
                bias_gyro: solution.bias_gyro,
                bias_accel: solution.bias_accel,
                velocity_sigma,
            };
            return Ok(BootstrapStep::Accepted(Box::new(Accepted {
                initial,
                frames: self.frames.iter().cloned().collect(),
                imu: self.imu.iter().copied().collect(),
            })));
        }
        if self.window_ns < seconds_to_ns(self.options.grow_s) {
            self.window_ns = seconds_to_ns(self.options.grow_s);
        } else {
            self.slide();
        }
        self.check_timeout()?;
        Ok(BootstrapStep::Buffering)
    }

    /// The estimator started from the accepted window: stamp the start and
    /// hand over the report.
    pub fn commit(self) -> BootstrapReport {
        let mut report: BootstrapReport = self.report;
        report.start_t_ns = self.frames.front().map(|f| f.t_ns);
        report.accepted_at_t_ns = self.frames.back().map(|f| f.t_ns);
        report.replayed = self.frames.len();
        report
    }

    /// [`BootstrapError::NeverAccepted`] once the windows with excitation
    /// span more than the timeout.
    fn check_timeout(&self) -> Result<(), BootstrapError> {
        let seconds: f64 = self.excited_ns as f64 * 1e-9;
        if seconds > self.options.timeout_s {
            return Err(BootstrapError::NeverAccepted {
                seconds,
                attempts: self.report.attempts.len(),
                last_verdict: self
                    .report
                    .attempts
                    .last()
                    .map_or("", |a| verdict_name(&a.verdict)),
            });
        }
        Ok(())
    }
}

/// Where a [`Vio`] is in its start, and so who owns the IMU: the driver while
/// it runs, the estimator otherwise.
#[derive(Debug, Clone)]
pub(super) enum BootstrapPhase {
    /// The stereo start.
    Off,
    /// Buffering and solving.
    Running(Box<InertialBootstrap>),
    /// Started from a solved window.
    Done(Box<BootstrapReport>),
}

impl BootstrapPhase {
    /// The phase a new pipeline starts in.
    pub(super) fn for_config(
        config: &VioConfig,
        calibration: &Calibration<f64>,
    ) -> Result<Self, VioError> {
        if config.vio_bootstrap != VioBootstrap::Inertial {
            return Ok(Self::Off);
        }
        if config.port_frontend_lag {
            return Err(BootstrapError::FrontendLag.into());
        }
        // The recording's own magnitude, as the estimator uses.
        let gravity_magnitude: f64 = calibration.gravity_ms2;
        Ok(Self::Running(Box::new(InertialBootstrap::new(
            calibration.clone(),
            gravity_magnitude,
            BootstrapOptions::default(),
        )?)))
    }

    pub(super) fn is_running(&self) -> bool {
        matches!(self, Self::Running(_))
    }

    /// The newest IMU timestamp the owner holds.
    pub(super) fn newest_imu_t_ns<S: Scalar>(
        &self,
        estimator: &estimator::SqrtKeypointVio<S>,
    ) -> Option<i64> {
        match self {
            Self::Running(driver) => driver.newest_imu_t_ns(),
            Self::Off | Self::Done(_) => estimator.newest_imu_t_ns(),
        }
    }

    /// Whether the owner's IMU reaches strictly past `t_ns`.
    pub(super) fn imu_covers_frame<S: Scalar>(
        &self,
        estimator: &estimator::SqrtKeypointVio<S>,
        t_ns: i64,
    ) -> bool {
        match self {
            Self::Running(driver) => driver.imu_covers_frame(t_ns),
            Self::Off | Self::Done(_) => estimator.imu_covers_frame(t_ns),
        }
    }

    /// Hand one raw sample to the owner; the driver also takes its calibrated value.
    pub(super) fn push_imu<S: Scalar>(
        &mut self,
        estimator: &mut estimator::SqrtKeypointVio<S>,
        sample: CombinedImuSample,
        calibration: &Calibration<f32>,
    ) {
        match self {
            Self::Running(driver) => {
                driver.push_imu(sample, super::calibrated(calibration, &sample))
            }
            Self::Off | Self::Done(_) => estimator.push_imu(sample),
        }
    }

    /// Whether the frontend gets a gyro-only rotation prediction before the
    /// first state; the stereo start gives it identity.
    pub(super) fn predicts_rotation(&self) -> bool {
        match self {
            Self::Running(_) => true,
            Self::Off | Self::Done(_) => false,
        }
    }
}

impl<S: Scalar> Vio<S> {
    /// The inertial bootstrap's attempts, or `None` on the stereo start.
    pub fn bootstrap_report(&self) -> Option<&BootstrapReport> {
        match &self.bootstrap {
            BootstrapPhase::Off => None,
            BootstrapPhase::Running(driver) => Some(driver.report()),
            BootstrapPhase::Done(report) => Some(report),
        }
    }

    /// Hand a bootstrapping frameset to the driver; on acceptance start a
    /// fresh estimator and replay the window.
    pub(super) fn bootstrap_track(
        &mut self,
        observations: Arc<FlowObservations>,
    ) -> Result<VioResult, VioError> {
        let t_ns: i64 = observations.t_ns;
        let BootstrapPhase::Running(driver) = &mut self.bootstrap else {
            unreachable!("bootstrap_track runs only while bootstrapping");
        };
        let accepted: Box<Accepted> = match driver.push_frame(observations)? {
            BootstrapStep::Buffering => return Ok(self.result(VioStatus::NoVisualFeatures, t_ns)),
            BootstrapStep::Accepted(accepted) => accepted,
        };
        let calibration: Calibration<S> = driver.calibration.cast();
        let config: VioConfig = self.frontend.config().clone();
        let initial: InitialState<S> = InitialState {
            t_ns: accepted.initial.t_ns,
            t_w_i: accepted.initial.t_w_i.cast(),
            vel_w_i: accepted.initial.vel_w_i.map(S::from_literal),
            bias_gyro: accepted.initial.bias_gyro.map(S::from_literal),
            bias_accel: accepted.initial.bias_accel.map(S::from_literal),
            velocity_sigma: S::from_literal(accepted.initial.velocity_sigma),
        };
        let mut fresh =
            estimator::SqrtKeypointVio::with_initial_state(calibration, config, initial)?;
        for sample in &accepted.imu {
            fresh.push_imu(*sample);
        }

        // Replay into the local estimator. Nothing of `self` has changed so
        // far (the driver still holds the window, its report names no start):
        // an error here returns with the pipeline still bootstrapping, and the
        // next frameset solves again.
        let pool = self.frontend.cpu_pool();
        let count: usize = accepted.frames.len();
        let mut last: Option<estimator::FrameOutcome<S>> = None;
        for (index, frame) in accepted.frames.into_iter().enumerate() {
            let outcome = fresh.process_frame(frame, pool.as_ref())?;
            if index + 1 < count {
                fresh.finish_deferred_keyframe(pool.as_ref())?;
            } else {
                last = Some(outcome);
            }
        }
        self.estimator = fresh;
        let BootstrapPhase::Running(driver) =
            std::mem::replace(&mut self.bootstrap, BootstrapPhase::Off)
        else {
            unreachable!("checked above");
        };
        self.bootstrap = BootstrapPhase::Done(Box::new(driver.commit()));
        let outcome = last.unwrap_or(estimator::FrameOutcome::NoVisualFeatures);
        self.accept_outcome(t_ns, outcome)
    }
}

#[cfg(test)]
mod tests;
