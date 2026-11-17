//! The inertial bootstrap driver (`config.vio_bootstrap = "inertial"`).
//!
//! The stereo start needs landmarks triangulated inside the first frameset
//! and assumes the rig is at rest. A rig without stereo overlap (a
//! back-to-back 360 camera, one camera of a stereo pair) that starts moving
//! has neither. While bootstrapping, [`Vio`] runs the frontend as usual but
//! hands each frameset's observations to [`InertialBootstrap`] instead of the
//! estimator:
//!
//! 1. The frontend gets a gyro-only rotation prediction ([`InertialBootstrap::predict`]),
//!    so the KLT has a guess before any state exists.
//! 2. The observations and a copy of the IMU are buffered. When the buffer
//!    spans the window (2 s), the observations become unit bearings
//!    ([`prepare_bearings`]) and `kornia_staging_slam::init::inertial::solve_inertial_init`
//!    runs. A rolling-shutter rig's observations arrive already corrected:
//!    `Vio::finish_track` corrects the estimator's copy before it reaches
//!    either the bootstrap or the estimator, so the buffer, the bearings and
//!    the replayed framesets are the same corrected observations.
//! 3. On a rejection the window grows to 3 s once; then it slides by 0.5 s
//!    and starts at 2 s again. Every attempt is logged and kept in the
//!    [`BootstrapReport`].
//! 4. On acceptance [`Vio`] builds a fresh estimator with
//!    [`estimator::SqrtKeypointVio::with_initial_state`] at the window's
//!    first frameset, pushes the buffered IMU, replays the buffered
//!    framesets through the normal `process_frame` and continues live.
//!    The replayed framesets publish no pose: the first pose is the
//!    frameset that completed the accepted window.
//! 5. The buffer holds at most the grown window. If no window is accepted
//!    within [`BootstrapOptions::timeout_s`] of the first frameset,
//!    `track` fails with [`BootstrapError::NeverAccepted`].
//!
//! Frameset times must be on the IMU clock and refer to the instant the
//! observations describe (for a rolling-shutter camera: the middle row's
//! mid-exposure, with the rows already corrected to it upstream).
use super::{Vio, VioError, VioResult, VioStatus};
use crate::calib::{BasaltCamera, Calibration};
use crate::camera::SlamCamera;
use crate::config::{VioBootstrap, VioConfig};
use crate::estimator::{self, FlowObservations, InitialState};
use crate::frontend::flow::PosePrediction;
use crate::lie::{Se3, So3};
use kornia_staging_algebra::Scalar;
use kornia_staging_sensors::imu::CombinedImuSample;
use kornia_staging_slam::init::inertial::{
    BearingObservation, FeatureTrack, ImuSample, InertialInitError, InertialInitOptions,
    InertialInitOutcome, InertialInitProblem, InitVerdict, solve_inertial_init,
};
use nalgebra::{UnitQuaternion, Vector2, Vector3, Vector4};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// Window schedule and solver options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapOptions {
    /// First window length, seconds.
    pub window_s: f64,
    /// Length the window grows to after a rejection, seconds.
    pub grow_s: f64,
    /// Slide after a rejection at the grown length, seconds.
    pub slide_s: f64,
    /// Give up when no window is accepted this long after the first frameset.
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

/// One solve attempt.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct BootstrapAttempt {
    /// First frameset of the window, ns.
    pub first_t_ns: i64,
    /// Last frameset of the window, ns.
    pub last_t_ns: i64,
    /// Framesets in the window.
    pub frames: usize,
    /// Tracks handed to the solver.
    pub tracks: usize,
    /// `Accepted` or the rejection reason.
    pub verdict: String,
    /// Whether the verdict accepted.
    pub accepted: bool,
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

/// Every attempt and where the bootstrap started the estimator.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
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

impl BootstrapReport {
    /// The report as JSON, for tools that log it.
    ///
    /// # Errors
    /// A non-finite number in a quality field.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Why the bootstrap stopped.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BootstrapError {
    /// No window was accepted in time.
    #[error(
        "inertial bootstrap: no window accepted in {seconds:.1} s after {attempts} attempts; last verdict {last_verdict}"
    )]
    NeverAccepted {
        /// Seconds since the first frameset.
        seconds: f64,
        /// Attempts made.
        attempts: usize,
        /// The last attempt's verdict.
        last_verdict: String,
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
    /// A window was accepted: start from it.
    Accepted(Box<Accepted>),
}

/// An accepted window and what the estimator needs to start from it.
#[derive(Debug)]
pub struct Accepted {
    /// The first state, at the first buffered frameset.
    pub initial: InitialState<f64>,
    /// Every buffered frameset, oldest first, ending with the current one.
    pub frames: Vec<Arc<FlowObservations>>,
    /// Every buffered IMU sample, raw (the estimator calibrates on pop).
    pub imu: Vec<CombinedImuSample>,
}

/// Whether some camera pair can triangulate inside one frameset: a baseline
/// of at least `min_baseline` and a shared field of view.
///
/// The overlap test is the frontend's own stereo-matching guess: a 16 x 16
/// grid of pixels of camera `i` is unprojected, placed at 1, 3 and 10 m, and
/// projected into camera `j`; one valid in-image projection is an overlap.
pub fn rig_has_stereo(calibration: &Calibration<f64>, min_baseline: f64) -> bool {
    let cameras: Vec<Option<SlamCamera<f64>>> = calibration
        .intrinsics
        .iter()
        .map(|m| SlamCamera::from_model(m).ok())
        .collect();
    let n: usize = calibration.t_i_c.len().min(cameras.len());
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let (Some(cam_i), Some(cam_j)) = (&cameras[i], &cameras[j]) else {
                continue;
            };
            let t_ci: Se3<f64> = calibration.t_i_c[i];
            let t_cj: Se3<f64> = calibration.t_i_c[j];
            if (t_ci.translation - t_cj.translation).norm() < min_baseline {
                continue;
            }
            let j_from_i: Se3<f64> = t_cj.inverse() * t_ci;
            let [w_i, h_i] = calibration.resolution[i];
            let [w_j, h_j] = calibration.resolution[j];
            for gy in 0..16 {
                for gx in 0..16 {
                    let pixel = Vector2::new(
                        (f64::from(gx) + 0.5) / 16.0 * f64::from(w_i),
                        (f64::from(gy) + 0.5) / 16.0 * f64::from(h_i),
                    );
                    let mut ray: Vector4<f64> = Vector4::zeros();
                    if !cam_i.unproject(&pixel, &mut ray) {
                        continue;
                    }
                    for depth in [1.0, 3.0, 10.0] {
                        let p_i: Vector3<f64> = ray.xyz() * depth;
                        let p_j: Vector3<f64> =
                            j_from_i.rotation.matrix() * p_i + j_from_i.translation;
                        let mut projected: Vector2<f64> = Vector2::zeros();
                        if cam_j.project_point(&p_j.push(1.0), &mut projected, None)
                            && (0.0..f64::from(w_j)).contains(&projected.x)
                            && (0.0..f64::from(h_j)).contains(&projected.y)
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Whether this config and rig run the inertial bootstrap.
pub fn uses_inertial(config: &VioConfig, calibration: &Calibration<f64>) -> bool {
    match config.vio_bootstrap {
        VioBootstrap::Stereo => false,
        VioBootstrap::Inertial => true,
        VioBootstrap::Auto => !rig_has_stereo(calibration, config.vio_min_triangulation_dist),
    }
}

/// Turn a window's framesets into bearing tracks: the observation seam.
///
/// Each pixel is unprojected through its camera's model to a unit bearing in
/// that camera's frame; the frame index is the frameset's position in
/// `window`. A rolling-shutter camera's pixels are already corrected (rotated
/// by the gyro over each row's time offset from the frameset time, in
/// `Vio::finish_track`) when they get here; the solver's sensitivity table
/// makes that mandatory for the Insta360 (uncorrected +-8 ms gave 6-16 %
/// scale error). Before the bootstrap accepts there is no gyro bias estimate,
/// so that correction uses the raw gyro rate (the static calibration's bias is
/// not applied either; it is zero for the Insta360 rig).
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
    imu: VecDeque<CombinedImuSample>,
    newest_imu_t_ns: Option<i64>,
    frames: VecDeque<Arc<FlowObservations>>,
    window_ns: i64,
    first_t_ns: Option<i64>,
    orientation: UnitQuaternion<f64>,
    orientation_t_ns: Option<i64>,
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
            newest_imu_t_ns: None,
            frames: VecDeque::new(),
            first_t_ns: None,
            orientation: UnitQuaternion::identity(),
            orientation_t_ns: None,
            report: BootstrapReport::default(),
        })
    }

    /// The attempts so far.
    pub fn report(&self) -> &BootstrapReport {
        &self.report
    }

    /// Take the report, for the pipeline to keep after acceptance.
    pub fn into_report(self) -> BootstrapReport {
        self.report
    }

    /// Framesets buffered now; at most the grown window's.
    pub fn buffered_frames(&self) -> usize {
        self.frames.len()
    }

    /// IMU samples buffered now.
    pub fn buffered_imu(&self) -> usize {
        self.imu.len()
    }

    /// The newest IMU timestamp buffered.
    pub fn newest_imu_t_ns(&self) -> Option<i64> {
        self.newest_imu_t_ns
    }

    /// Whether the IMU reaches strictly past `t_ns` (the estimator's rule).
    pub fn imu_covers_frame(&self, t_ns: i64) -> bool {
        self.newest_imu_t_ns.is_some_and(|newest| newest > t_ns)
    }

    /// Buffer one raw sample; the caller has checked its order.
    pub fn push_imu(&mut self, sample: CombinedImuSample) {
        self.newest_imu_t_ns = Some(sample.timestamp_ns);
        self.imu.push_back(sample);
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
                self.imu.pop_front();
            }
        }
    }

    fn calibrated(&self, sample: &CombinedImuSample) -> ImuSample {
        ImuSample {
            t_ns: sample.timestamp_ns,
            gyro: self
                .calibration
                .calib_gyro_bias
                .calibrated(&Vector3::from(sample.gyro.to_array())),
            accel: self
                .calibration
                .calib_accel_bias
                .calibrated(&Vector3::from(sample.accel.to_array())),
        }
    }

    /// Gyro-only rotation prediction from the previous frameset to `t_ns`
    /// (translation zero, gyro bias zero), so the KLT has a guess before any
    /// state exists. The caller has checked IMU coverage.
    pub fn predict(&mut self, t_ns: i64) -> PosePrediction {
        let previous: UnitQuaternion<f64> = self.orientation;
        if let Some(t0) = self.orientation_t_ns {
            // Mid-point integration over (t0, t_ns] with the samples on both
            // sides of each end held constant to the edge.
            let samples: Vec<ImuSample> = self.imu.iter().map(|s| self.calibrated(s)).collect();
            let mut q: UnitQuaternion<f64> = previous;
            let mut t_a: i64 = t0;
            let gyro_at = |t: i64| -> Vector3<f64> {
                let upper: usize = samples.partition_point(|s| s.t_ns <= t);
                match (
                    upper.checked_sub(1).map(|i| &samples[i]),
                    samples.get(upper),
                ) {
                    (Some(a), Some(b)) => {
                        let w: f64 = (t - a.t_ns) as f64 / (b.t_ns - a.t_ns) as f64;
                        a.gyro * (1.0 - w) + b.gyro * w
                    }
                    (Some(a), None) => a.gyro,
                    (None, Some(b)) => b.gyro,
                    (None, None) => Vector3::zeros(),
                }
            };
            let mut knots: Vec<i64> = samples
                .iter()
                .map(|s| s.t_ns)
                .filter(|&t| t > t0 && t < t_ns)
                .collect();
            knots.push(t_ns);
            let mut gyro_a: Vector3<f64> = gyro_at(t0);
            for t_b in knots {
                let gyro_b: Vector3<f64> = gyro_at(t_b);
                let dt: f64 = (t_b - t_a) as f64 * 1e-9;
                q *= UnitQuaternion::from_scaled_axis((gyro_a + gyro_b) * (0.5 * dt));
                t_a = t_b;
                gyro_a = gyro_b;
            }
            self.orientation = q;
        }
        self.orientation_t_ns = Some(t_ns);
        let pose = |q: &UnitQuaternion<f64>| -> Se3<f32> {
            Se3::new(So3::from_unit_quaternion(*q), Vector3::zeros()).cast()
        };
        PosePrediction {
            t_w_i_previous: pose(&previous),
            t_w_i_current: pose(&self.orientation),
        }
    }

    fn attempt(&self) -> Result<(InertialInitOutcome, usize), BootstrapError> {
        let window: Vec<Arc<FlowObservations>> = self.frames.iter().cloned().collect();
        let tracks: Vec<FeatureTrack> = prepare_bearings(&window, &self.cameras);
        let frame_times: Vec<i64> = window.iter().map(|f| f.t_ns).collect();
        let imu: Vec<ImuSample> = self.imu.iter().map(|s| self.calibrated(s)).collect();
        let problem = InertialInitProblem {
            imu: &imu,
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
            verdict: match outcome.verdict {
                InitVerdict::Accepted => "Accepted".to_owned(),
                InitVerdict::Rejected(reason) => format!("{reason:?}"),
            },
            accepted: outcome.accepted().is_some(),
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
            "inertial bootstrap attempt {}: window {:.2} s, {} framesets, {} tracks ({} used): {}; excitation {:.3} m/s^2, residual {:?} rad, condition {:?}, |g| {:?}, scale sigma {:?}",
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
            self.imu.pop_front();
        }
    }

    /// Buffer a frameset and, when the window is full, solve it.
    ///
    /// # Errors
    /// [`BootstrapError::NeverAccepted`] past the timeout, and malformed
    /// windows from the solver.
    pub fn push_frame(
        &mut self,
        frame: Arc<FlowObservations>,
    ) -> Result<BootstrapStep, BootstrapError> {
        let t_ns: i64 = frame.t_ns;
        let first_ever: i64 = *self.first_t_ns.get_or_insert(t_ns);
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
            self.check_timeout(t_ns - first_ever)?;
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
            self.report.start_t_ns = Some(first);
            self.report.accepted_at_t_ns = Some(t_ns);
            self.report.replayed = self.frames.len();
            return Ok(BootstrapStep::Accepted(Box::new(Accepted {
                initial,
                frames: self.frames.drain(..).collect(),
                imu: self.imu.drain(..).collect(),
            })));
        }
        if self.window_ns < seconds_to_ns(self.options.grow_s) {
            self.window_ns = seconds_to_ns(self.options.grow_s);
        } else {
            self.window_ns = seconds_to_ns(self.options.window_s);
            self.slide();
        }
        self.check_timeout(t_ns - first_ever)?;
        Ok(BootstrapStep::Buffering)
    }

    /// [`BootstrapError::NeverAccepted`] once `elapsed_ns` since the first
    /// frameset is past the timeout; checked on every frameset.
    fn check_timeout(&self, elapsed_ns: i64) -> Result<(), BootstrapError> {
        let seconds: f64 = elapsed_ns as f64 * 1e-9;
        if seconds > self.options.timeout_s {
            return Err(BootstrapError::NeverAccepted {
                seconds,
                attempts: self.report.attempts.len(),
                last_verdict: self
                    .report
                    .attempts
                    .last()
                    .map_or_else(String::new, |a| a.verdict.clone()),
            });
        }
        Ok(())
    }
}

/// Where a [`Vio`] is in its start.
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
        if !uses_inertial(config, calibration) {
            return Ok(Self::Off);
        }
        if config.port_frontend_lag {
            return Err(BootstrapError::FrontendLag.into());
        }
        // The dataset's own magnitude (`config.vio_gravity_ms2`), as the estimator uses.
        let gravity_magnitude: f64 = config.vio_gravity_ms2;
        Ok(Self::Running(Box::new(InertialBootstrap::new(
            calibration.clone(),
            gravity_magnitude,
            BootstrapOptions::default(),
        )?)))
    }

    pub(super) fn running_mut(&mut self) -> Option<&mut InertialBootstrap> {
        match self {
            Self::Running(driver) => Some(driver),
            _ => None,
        }
    }

    pub(super) fn running(&self) -> Option<&InertialBootstrap> {
        match self {
            Self::Running(driver) => Some(driver),
            _ => None,
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
        let Some(driver) = self.bootstrap.running_mut() else {
            unreachable!("bootstrap_track runs only while bootstrapping");
        };
        let accepted: Box<Accepted> = match driver.push_frame(observations)? {
            BootstrapStep::Buffering => return Ok(self.result(VioStatus::NoVisualFeatures, t_ns)),
            BootstrapStep::Accepted(accepted) => accepted,
        };
        let BootstrapPhase::Running(driver) =
            std::mem::replace(&mut self.bootstrap, BootstrapPhase::Off)
        else {
            unreachable!("checked above");
        };
        let config: VioConfig = self.frontend.config().clone();
        let calibration: Calibration<S> = driver.calibration.cast();
        let initial: InitialState<S> = InitialState {
            t_ns: accepted.initial.t_ns,
            t_w_i: accepted.initial.t_w_i.cast(),
            vel_w_i: accepted.initial.vel_w_i.map(S::from_literal),
            bias_gyro: accepted.initial.bias_gyro.map(S::from_literal),
            bias_accel: accepted.initial.bias_accel.map(S::from_literal),
            velocity_sigma: S::from_literal(accepted.initial.velocity_sigma),
        };
        let gravity: nalgebra::Vector3<S> =
            nalgebra::Vector3::new(S::zero(), S::zero(), crate::lie::c(-config.vio_gravity_ms2));
        let mut fresh = estimator::SqrtKeypointVio::with_initial_state(
            gravity,
            calibration,
            config,
            initial,
        )?;
        for sample in &accepted.imu {
            fresh.push_imu(*sample);
        }
        self.estimator = fresh;
        self.bootstrap = BootstrapPhase::Done(Box::new(driver.into_report()));

        let pool = self.frontend.cpu_pool();
        let count: usize = accepted.frames.len();
        let mut last: Option<estimator::FrameOutcome<S>> = None;
        for (index, frame) in accepted.frames.into_iter().enumerate() {
            let outcome = self.estimator.process_frame(frame, pool.as_ref())?;
            if index + 1 < count {
                self.estimator.finish_deferred_keyframe(pool.as_ref())?;
            } else {
                last = Some(outcome);
            }
        }
        let outcome = last.unwrap_or(estimator::FrameOutcome::NoVisualFeatures);
        self.accept_outcome(t_ns, outcome)
    }
}

#[cfg(test)]
mod tests;
