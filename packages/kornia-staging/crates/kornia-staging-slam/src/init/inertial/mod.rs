//! Visual-inertial initialisation from a short moving window.
//!
//! A sliding-window VIO normally starts from stereo landmarks of the first
//! frame and zero velocity. A rig without stereo overlap (a back-to-back 360
//! camera, a monocular camera) that never stands still has neither. This
//! module recovers, from the feature tracks and IMU samples of a window of
//! one to three seconds, the body velocity at the first frame, the gravity
//! direction, the gyroscope bias and, when it is observable, the
//! accelerometer bias.
//!
//! # Frames, signs and units
//!
//! - **Body-0 frame**: the IMU (body) frame at the first frame time. Every
//!   output vector is in it unless its name says `world`.
//! - Rotations `R_k` are body-k to body-0 (`R_0 = I`); positions `p_k` are
//!   body-k origins in body-0 (`p_0 = 0`).
//! - Camera extrinsics are `T_i_c`: `R_ic` takes camera directions to the
//!   body, `t_ic` is the camera centre in the body. Several cameras may share
//!   a centre; none needs to overlap another.
//! - Bearings are unit vectors in their camera frame from any camera model,
//!   including directions more than 90 degrees from the optical axis.
//! - IMU: `gyro = w + b_g + noise` (rad/s), `accel = f + b_a + noise`
//!   (m/s^2), where `f = R_wb^T (a - g)` is the specific force: a body at rest
//!   reads `accel = -g`, about `+9.81` upwards.
//! - **Gravity** `g` is the gravitational acceleration vector (it points
//!   down), `|g| = gravity_magnitude`, so `p_k = v0 dt_k + 1/2 g dt_k^2 + dp_k`
//!   with `dp_k` the double integral of the rotated specific force.
//! - **World**: z up, origin at the first body position, gravity
//!   `(0, 0, -|g|)`. [`InertialInit::world_from_body0`] is the shortest-arc
//!   rotation that takes `-g` (the at-rest accelerometer direction) onto
//!   `+Z`: the convention of slam-rs's `gravity_from_first_accel`, applied to
//!   the solved gravity instead of one accelerometer sample. Yaw is not
//!   observable and is fixed by that shortest arc.
//! - Times are integer nanoseconds on one clock; frame times are the
//!   instants the bearings refer to (after any rolling-shutter correction).
//!
//! # Method
//!
//! 1. **Preintegration** of the IMU from the first frame to every frame at
//!    the prior gyro bias (Forster et al., T-RO 2017), with first-order bias
//!    Jacobians so the refinement moves the biases without re-integrating.
//! 2. **Gates before solving**: the excitation (RMS deviation of the
//!    body-0-frame specific force averaged per frame interval) and the track
//!    support (observations per track, rotation-compensated parallax).
//! 3. **Closed form** (Martinelli, IJCV 2014; Kaiser et al., RA-L 2017):
//!    each observation gives two equations linear in `(X, v0, g)`; each
//!    track's point is eliminated by QR, leaving a 6-unknown system.
//!    Tracks fixed in the image (the camera operator) or mistracked would
//!    bias a plain least-squares fit, so `(v0, g)` comes from MSAC over
//!    track subsets; each hypothesis already carries `|g| = G`, solved
//!    exactly as a quadratically constrained least squares (secular
//!    equation, Gander 1981), not by rescaling.
//! 4. **Refinement**: Levenberg-Marquardt over `v0`, the gravity direction
//!    (on the sphere `|g| = G`), the gyro bias, optionally the accelerometer
//!    bias, and every point (Schur complement), with a Huber loss on the
//!    chordal bearing error. Then the IMU is re-integrated at the new biases,
//!    every candidate track is re-triangulated and scored, tracks above the
//!    outlier threshold are dropped (or re-admitted), and the refinement runs
//!    again until the inlier set and the biases are stable.
//! 5. **Gates after solving**: inlier count, conditioning of the reduced
//!    linear system and the relative speed (scale) uncertainty, bearing RMS,
//!    and the unconstrained `|g|` of the closed form at the refined biases.
//!
//! # Design choices
//!
//! - The gyro bias is not first estimated from visual relative rotations
//!   (the VINS-Mono step): that needs overlapping views or an essential
//!   matrix per frame pair. It is a parameter of the bearing refinement
//!   instead, which uses every observation and needs no overlap.
//! - The excitation measure is the specific force in the body-0 frame,
//!   not `std(|a|)`: `|a|` misses accelerations orthogonal to gravity and
//!   counts rotation-induced changes that carry no scale.
//! - The constrained solve is exact (secular equation) and runs inside the
//!   consensus, and the final answer is the robust non-linear refinement.
//! - The window is not slid here; a rejected verdict tells the caller to
//!   try the next window.
//! - The accelerometer bias is estimated by default, with a Gaussian prior
//!   (0.1 m/s^2). Held at zero, a 0.05 m/s^2 bias along gravity cannot be
//!   absorbed by the `|g|` constraint and reads as a constant acceleration:
//!   in the noise trials it took the p90 scale error from 0.95 % to 2.7 % at
//!   2 s and from 0.75 % to 8.8 % at 3 s. Its horizontal part trades against
//!   a gravity tilt of `|b_a| / |g|` and is separated only by the window's
//!   rotation, so expect gravity errors of a few tenths of a degree.
//! - The bearings are whitened by [`InertialInitOptions::bearing_sigma_rad`]
//!   so the bias priors weigh against them in consistent units.
//! - The scale uncertainty ([`InitQuality::scale_sigma`]) is computed with
//!   the accelerometer bias as a free parameter under its prior even when it
//!   is held at zero, so a fixed bias cannot pass for scale information.
//!
//! # Limits measured on synthetic data (see the tests)
//!
//! Default 2 s window, 0.5 px bearing noise: the p90 scale error is about
//! 1 %. The answer is only as good as the time alignment and the rolling-
//! shutter correction: an IMU-to-camera offset of 5 ms gave a p90 scale error
//! of 5.7 % (10 ms was rejected), and uncorrected rolling shutter of +-8 ms
//! gave 13.5 % while passing every gate. A 3 cm lever-arm error costs 1.4 %
//! (3.7 % along the lens axis) and tilts gravity by about 2.5 degrees.
//!
//! # Example
//!
//! ```
//! use kornia_staging_slam::init::inertial::{
//!     solve_inertial_init, InertialInitOptions, InertialInitProblem, InitVerdict, RejectReason,
//! };
//! use kornia_staging_algebra::lie::RigidTransform;
//! use nalgebra::Vector3;
//!
//! // A rig at rest: no tracks, no excitation.
//! let imu: Vec<_> = (0..=2_000)
//!     .map(|i| kornia_staging_slam::init::inertial::ImuSample {
//!         t_ns: i * 1_000_000,
//!         gyro: Vector3::zeros(),
//!         accel: Vector3::new(0.0, 0.0, 9.81),
//!     })
//!     .collect();
//! let frames: Vec<i64> = (0..60).map(|k| k * 33_333_333).collect();
//! let cameras = [RigidTransform::<f64>::identity()];
//! let problem = InertialInitProblem {
//!     imu: &imu,
//!     frame_times_ns: &frames,
//!     tracks: &[],
//!     t_i_c: &cameras,
//!     gravity_magnitude: 9.81,
//!     gyro_bias_prior: None,
//! };
//! let outcome = solve_inertial_init(&problem, &InertialInitOptions::default())?;
//! assert!(matches!(
//!     outcome.verdict,
//!     InitVerdict::Rejected(RejectReason::LowExcitation { .. })
//! ));
//! assert!(outcome.accepted().is_none());
//! # Ok::<(), kornia_staging_slam::init::inertial::InertialInitError>(())
//! ```
mod linear;
mod preintegration;
mod refine;
#[cfg(test)]
mod tests;

use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use linear::{Camera, ConsensusParams, TrackBlock};
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3, Vector6};
use preintegration::{preintegrate, Preintegrated};

/// One IMU sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSample {
    /// Sample time, nanoseconds.
    pub t_ns: i64,
    /// Angular rate in the body frame, rad/s, bias included.
    pub gyro: Vector3<f64>,
    /// Specific force in the body frame, m/s^2, bias included.
    pub accel: Vector3<f64>,
}

/// One observation of a track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BearingObservation {
    /// Index into the window's frame times.
    pub frame: usize,
    /// Index into the rig's cameras.
    pub camera: usize,
    /// Unit bearing in the camera frame (normalised again by the solver).
    pub bearing: Vector3<f64>,
}

/// The observations of one 3-D point.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureTrack {
    /// The caller's track identifier, returned with the points.
    pub id: u64,
    /// Observations in any order; several cameras may see the same frame.
    pub observations: Vec<BearingObservation>,
}

/// The input window.
#[derive(Debug, Clone, Copy)]
pub struct InertialInitProblem<'a> {
    /// IMU samples ordered by strictly increasing time, covering the first
    /// and the last frame time.
    pub imu: &'a [ImuSample],
    /// Frame times, strictly increasing; at least three.
    pub frame_times_ns: &'a [i64],
    /// Feature tracks of the window.
    pub tracks: &'a [FeatureTrack],
    /// Camera-to-body transforms `T_i_c`, one per camera.
    pub t_i_c: &'a [RigidTransform<f64>],
    /// Magnitude of gravity, m/s^2 (slam-rs uses 9.81).
    pub gravity_magnitude: f64,
    /// A gyro bias known beforehand (rad/s); the prior mean of the refinement.
    pub gyro_bias_prior: Option<Vector3<f64>>,
}

/// Thresholds and switches. The defaults come from the synthetic noise
/// trials in this module's tests (0.5 px bearing noise at 305.6 px/rad,
/// consumer MEMS IMU).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialInitOptions {
    /// Fewest inlier tracks for an answer.
    pub min_tracks: usize,
    /// Fewest observations for a track to be used (at least 5).
    pub min_track_observations: usize,
    /// Smallest rotation-compensated angle between two rays of a track, rad.
    pub min_track_parallax_rad: f64,
    /// Smallest excitation, m/s^2 (RMS deviation of the body-0-frame
    /// specific force averaged per frame interval).
    pub min_excitation_mps2: f64,
    /// Largest condition number of the column-equilibrated reduced linear system.
    pub max_condition: f64,
    /// Largest relative 1-sigma uncertainty of the speed `|v0|` (the scale).
    pub max_scale_sigma: f64,
    /// Largest RMS bearing error of the inliers after refinement, rad.
    pub max_residual_rms_rad: f64,
    /// Largest relative deviation of the unconstrained `|g|` from the magnitude.
    pub max_gravity_norm_error: f64,
    /// Track RMS above which the consensus counts a track as an outlier, rad.
    /// Generous: it must pass the inliers before the gyro bias is refined.
    pub consensus_threshold_rad: f64,
    /// Track RMS above which the refinement drops a track, rad.
    pub outlier_threshold_rad: f64,
    /// Huber threshold on the chordal bearing error, rad.
    pub huber_rad: f64,
    /// Bearing noise (1-sigma per tangent axis), rad: weighs the bearings
    /// against the bias priors.
    pub bearing_sigma_rad: f64,
    /// Tracks per consensus hypothesis.
    pub consensus_sample_tracks: usize,
    /// Most consensus hypotheses.
    pub consensus_max_iterations: usize,
    /// Seed of the consensus sampler.
    pub seed: u64,
    /// Prior 1-sigma of the gyro bias around its prior mean, rad/s.
    pub gyro_bias_sigma: f64,
    /// Estimate the accelerometer bias (otherwise it stays zero). On by
    /// default; see the module docs for the measured cost of holding it at zero.
    pub estimate_accel_bias: bool,
    /// Prior 1-sigma of the accelerometer bias around zero, m/s^2.
    pub accel_bias_sigma: f64,
    /// Most Levenberg-Marquardt iterations per refinement.
    pub max_iterations: usize,
}

impl Default for InertialInitOptions {
    fn default() -> Self {
        Self {
            min_tracks: 20,
            min_track_observations: 5,
            min_track_parallax_rad: 0.02,
            min_excitation_mps2: 0.25,
            max_condition: 1.0e4,
            max_scale_sigma: 0.05,
            max_residual_rms_rad: 0.006,
            max_gravity_norm_error: 0.1,
            consensus_threshold_rad: 0.035,
            outlier_threshold_rad: 0.01,
            huber_rad: 0.004,
            bearing_sigma_rad: 0.002,
            consensus_sample_tracks: 6,
            consensus_max_iterations: 300,
            seed: 0x5EED_1A17,
            gyro_bias_sigma: 0.05,
            estimate_accel_bias: true,
            accel_bias_sigma: 0.1,
            max_iterations: 30,
        }
    }
}

/// Malformed input. A well-formed window that cannot be solved is not an
/// error: it gives a [`InitVerdict::Rejected`] outcome.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum InertialInitError {
    /// Fewer than three frames.
    #[error("{frames} frames; the window needs at least 3")]
    TooFewFrames {
        /// Frames given.
        frames: usize,
    },
    /// Frame times are not strictly increasing.
    #[error("frame times are not strictly increasing at frame {index}")]
    FrameTimesNotIncreasing {
        /// First offending frame.
        index: usize,
    },
    /// IMU times are not strictly increasing.
    #[error("IMU times are not strictly increasing at sample {index}")]
    ImuTimesNotIncreasing {
        /// First offending sample.
        index: usize,
    },
    /// The IMU does not reach from the first to the last frame.
    #[error("the IMU samples do not cover the window")]
    ImuDoesNotCoverWindow,
    /// A non-finite IMU sample.
    #[error("IMU sample {index} is not finite")]
    NonFiniteImu {
        /// Offending sample.
        index: usize,
    },
    /// No cameras.
    #[error("the rig has no cameras")]
    NoCameras,
    /// Gravity magnitude is not a positive finite number.
    #[error("gravity magnitude {0} is not positive and finite")]
    InvalidGravity(f64),
    /// An observation names a frame or camera that does not exist, or its
    /// bearing is not a finite non-zero vector.
    #[error("track {track_id} has an invalid observation (frame, camera or bearing)")]
    InvalidObservation {
        /// The track.
        track_id: u64,
    },
}

/// Why a window was rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RejectReason {
    /// Too little acceleration beyond gravity: scale is not observable.
    LowExcitation {
        /// Measured excitation, m/s^2.
        excitation_mps2: f64,
    },
    /// Enough long tracks, but too few with parallax (pure rotation or far scene).
    LowParallax {
        /// Tracks with parallax.
        tracks: usize,
    },
    /// Too few usable or inlier tracks.
    TooFewTracks {
        /// Tracks available at the failing step.
        tracks: usize,
        /// Tracks required.
        required: usize,
    },
    /// The reduced system or the scale is badly determined.
    IllConditioned {
        /// Condition number of the reduced linear system.
        condition: f64,
        /// Relative 1-sigma of the speed.
        scale_sigma: f64,
    },
    /// The inliers do not fit the solution to the bearing noise.
    ResidualTooHigh {
        /// RMS bearing error, rad.
        residual_rms_rad: f64,
    },
    /// The closed form, free of the norm constraint, disagrees with `|g|`.
    GravityNormInconsistent {
        /// Unconstrained `|g|`, m/s^2.
        gravity_norm: f64,
    },
    /// The solve failed numerically, or the refine and re-score passes ended without a stable inlier set.
    NotConverged,
}

/// The verdict on a window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitVerdict {
    /// Every gate passed: start the estimator from the solution.
    Accepted,
    /// Do not use the solution; try a later or longer window.
    Rejected(RejectReason),
}

/// The solved state.
#[derive(Debug, Clone, PartialEq)]
pub struct InertialInit {
    /// Body velocity at the first frame, body-0 frame, m/s.
    pub velocity_body0: Vector3<f64>,
    /// Gravity vector in body-0 (points down), `|g| = gravity_magnitude`.
    pub gravity_body0: Vector3<f64>,
    /// Gravity-aligned world-from-body rotation at the first frame (see the
    /// module docs for the convention).
    pub world_from_body0: Rotation3<f64>,
    /// `world_from_body0 * velocity_body0`.
    pub velocity_world: Vector3<f64>,
    /// Gyroscope bias, rad/s, body frame.
    pub bias_gyro: Vector3<f64>,
    /// Accelerometer bias, m/s^2, body frame (zero unless estimated).
    pub bias_accel: Vector3<f64>,
    /// Body-k-to-body-0 rotation of every frame.
    pub rotations_body0: Vec<Rotation3<f64>>,
    /// Body position of every frame in body-0, metres.
    pub positions_body0: Vec<Vector3<f64>>,
    /// Inlier points in body-0 as `(track id, point)`, for diagnostics.
    pub points_body0: Vec<(u64, Vector3<f64>)>,
    /// Tracks that passed the length and parallax tests but were rejected as outliers.
    pub outlier_track_ids: Vec<u64>,
}

/// Quality numbers; `None` where the solve stopped before computing them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitQuality {
    /// Excitation, m/s^2.
    pub excitation_mps2: f64,
    /// Tracks with enough observations.
    pub tracks_long_enough: usize,
    /// Of those, tracks with enough parallax (the candidates).
    pub tracks_with_parallax: usize,
    /// Inlier tracks used by the final refinement.
    pub tracks_used: usize,
    /// RMS bearing error of the inliers, rad.
    pub residual_rms_rad: Option<f64>,
    /// Condition number of the column-equilibrated reduced linear system.
    pub condition: Option<f64>,
    /// `|g|` of the unconstrained closed form at the refined biases.
    pub gravity_norm_unconstrained: Option<f64>,
    /// Relative 1-sigma of `|v0|` from the refinement's information matrix.
    pub scale_sigma: Option<f64>,
    /// Refinement converged.
    pub converged: bool,
}

/// The verdict, the solution when the solve got that far, and its quality.
#[derive(Debug, Clone, PartialEq)]
pub struct InertialInitOutcome {
    /// Accepted or why not.
    pub verdict: InitVerdict,
    /// The solution; present also for late rejections, for diagnostics.
    pub solution: Option<InertialInit>,
    /// Quality numbers.
    pub quality: InitQuality,
}

impl InertialInitOutcome {
    /// The solution, only when the verdict is [`InitVerdict::Accepted`].
    pub fn accepted(&self) -> Option<&InertialInit> {
        match self.verdict {
            InitVerdict::Accepted => self.solution.as_ref(),
            InitVerdict::Rejected(_) => None,
        }
    }
}

/// The shortest-arc rotation taking `-gravity` onto world `+Z`.
///
/// For gravity within rounding of world `+Z` (an upside-down rig) the axis
/// is the canonical axis least aligned with it, made orthogonal; yaw there
/// is arbitrary, as it is for every input.
pub fn world_from_gravity(gravity: &Vector3<f64>) -> Rotation3<f64> {
    let norm: f64 = gravity.norm();
    if !(norm.is_finite() && norm > 0.0) {
        return Rotation3::identity();
    }
    let up: Vector3<f64> = -gravity / norm;
    let z: Vector3<f64> = Vector3::z();
    let dot: f64 = up.dot(&z);
    if dot < -1.0 + 1e-12 {
        let (axis, _) = linear::tangent_basis(&up);
        return Rotation3::from_unit_quaternion(UnitQuaternion::new_normalize(Quaternion::new(
            0.0, axis.x, axis.y, axis.z,
        )));
    }
    let axis: Vector3<f64> = up.cross(&z);
    let s: f64 = ((1.0 + dot) * 2.0).sqrt();
    Rotation3::from_unit_quaternion(UnitQuaternion::new_normalize(Quaternion::new(
        0.5 * s,
        axis.x / s,
        axis.y / s,
        axis.z / s,
    )))
}

fn validate(problem: &InertialInitProblem<'_>) -> Result<(), InertialInitError> {
    let frames: &[i64] = problem.frame_times_ns;
    if frames.len() < 3 {
        return Err(InertialInitError::TooFewFrames {
            frames: frames.len(),
        });
    }
    if let Some(i) = (1..frames.len()).find(|&i| frames[i] <= frames[i - 1]) {
        return Err(InertialInitError::FrameTimesNotIncreasing { index: i });
    }
    let imu: &[ImuSample] = problem.imu;
    if let Some(i) = (1..imu.len()).find(|&i| imu[i].t_ns <= imu[i - 1].t_ns) {
        return Err(InertialInitError::ImuTimesNotIncreasing { index: i });
    }
    if let Some(i) = imu.iter().position(|s| {
        !(s.gyro.iter().all(|v| v.is_finite()) && s.accel.iter().all(|v| v.is_finite()))
    }) {
        return Err(InertialInitError::NonFiniteImu { index: i });
    }
    if problem.t_i_c.is_empty() {
        return Err(InertialInitError::NoCameras);
    }
    let g: f64 = problem.gravity_magnitude;
    if !(g.is_finite() && g > 0.0) {
        return Err(InertialInitError::InvalidGravity(g));
    }
    for track in problem.tracks {
        let bad = track.observations.iter().any(|o| {
            let n: f64 = o.bearing.norm();
            o.frame >= frames.len()
                || o.camera >= problem.t_i_c.len()
                || !(n.is_finite() && n > 0.0)
        });
        if bad {
            return Err(InertialInitError::InvalidObservation { track_id: track.id });
        }
    }
    Ok(())
}

/// `value <= limit`, false for NaN: a gate passes only on a number.
fn within(value: f64, limit: f64) -> bool {
    value <= limit
}

/// RMS deviation of the per-interval mean specific force in body-0.
fn excitation(preint: &Preintegrated) -> f64 {
    let n: usize = preint.dt.len();
    let forces: Vec<Vector3<f64>> = (1..n)
        .map(|k| (preint.velocity[k] - preint.velocity[k - 1]) / (preint.dt[k] - preint.dt[k - 1]))
        .collect();
    let mean: Vector3<f64> = forces.iter().sum::<Vector3<f64>>() / forces.len() as f64;
    (forces
        .iter()
        .map(|f| (f - mean).norm_squared())
        .sum::<f64>()
        / forces.len() as f64)
        .sqrt()
}

/// Largest angle between two rays of a track in body-0.
fn parallax(
    observations: &[BearingObservation],
    preint: &Preintegrated,
    cameras: &[Camera],
) -> f64 {
    let rays: Vec<Vector3<f64>> = observations
        .iter()
        .map(|o| preint.rotation[o.frame] * cameras[o.camera].r_ic * o.bearing)
        .collect();
    let mut min_cos: f64 = 1.0;
    for (i, a) in rays.iter().enumerate() {
        for b in &rays[i + 1..] {
            min_cos = min_cos.min(a.dot(b));
        }
    }
    min_cos.clamp(-1.0, 1.0).acos()
}

/// A candidate track: owned, normalised, sorted observations.
struct Candidate {
    id: u64,
    observations: Vec<BearingObservation>,
}

/// Solve one window. See the module docs for the method and conventions.
///
/// The function is pure and deterministic: the same input gives a
/// bit-identical outcome.
///
/// # Errors
/// Malformed input only; see [`InertialInitError`].
pub fn solve_inertial_init(
    problem: &InertialInitProblem<'_>,
    options: &InertialInitOptions,
) -> Result<InertialInitOutcome, InertialInitError> {
    validate(problem)?;
    let gravity_magnitude: f64 = problem.gravity_magnitude;
    let cameras: Vec<Camera> = problem
        .t_i_c
        .iter()
        .map(|t| Camera {
            r_ic: t.rotation.matrix(),
            t_ic: t.translation,
        })
        .collect();
    let bias_gyro_prior: Vector3<f64> = problem.gyro_bias_prior.unwrap_or_else(Vector3::zeros);
    let zero: Vector3<f64> = Vector3::zeros();
    let mut preint: Preintegrated =
        preintegrate(problem.imu, problem.frame_times_ns, &bias_gyro_prior, &zero)?;

    let mut quality = InitQuality {
        excitation_mps2: excitation(&preint),
        tracks_long_enough: 0,
        tracks_with_parallax: 0,
        tracks_used: 0,
        residual_rms_rad: None,
        condition: None,
        gravity_norm_unconstrained: None,
        scale_sigma: None,
        converged: false,
    };
    let reject = |reason: RejectReason, quality: InitQuality, solution: Option<InertialInit>| {
        Ok(InertialInitOutcome {
            verdict: InitVerdict::Rejected(reason),
            solution,
            quality,
        })
    };

    // Candidate tracks.
    let min_obs: usize = options.min_track_observations.max(5);
    let mut candidates: Vec<Candidate> = Vec::new();
    for track in problem.tracks {
        if track.observations.len() < min_obs {
            continue;
        }
        quality.tracks_long_enough += 1;
        let mut observations: Vec<BearingObservation> = track
            .observations
            .iter()
            .map(|o| BearingObservation {
                bearing: o.bearing.normalize(),
                ..*o
            })
            .collect();
        observations.sort_by_key(|o| (o.frame, o.camera));
        if parallax(&observations, &preint, &cameras) < options.min_track_parallax_rad {
            continue;
        }
        candidates.push(Candidate {
            id: track.id,
            observations,
        });
    }
    quality.tracks_with_parallax = candidates.len();

    if quality.excitation_mps2 < options.min_excitation_mps2 {
        let e = quality.excitation_mps2;
        return reject(
            RejectReason::LowExcitation { excitation_mps2: e },
            quality,
            None,
        );
    }
    if candidates.len() < options.min_tracks {
        let reason = if quality.tracks_long_enough >= options.min_tracks {
            RejectReason::LowParallax {
                tracks: candidates.len(),
            }
        } else {
            RejectReason::TooFewTracks {
                tracks: quality.tracks_long_enough,
                required: options.min_tracks,
            }
        };
        return reject(reason, quality, None);
    }

    // Closed form with consensus.
    let observations: Vec<&[BearingObservation]> = candidates
        .iter()
        .map(|c| c.observations.as_slice())
        .collect();
    let blocks: Option<Vec<TrackBlock>> = observations
        .iter()
        .map(|o| linear::track_block(o, &preint, &cameras))
        .collect();
    let Some(blocks) = blocks else {
        return reject(RejectReason::NotConverged, quality, None);
    };
    let params = ConsensusParams {
        gravity_magnitude,
        threshold_rad: options.consensus_threshold_rad,
        sample_tracks: options.consensus_sample_tracks,
        max_iterations: options.consensus_max_iterations,
        seed: options.seed,
    };
    let Some(consensus) = linear::consensus(&observations, &blocks, &preint, &cameras, &params)
    else {
        return reject(RejectReason::NotConverged, quality, None);
    };
    let mut inlier: Vec<bool> = consensus.inlier;
    let mut points: Vec<Vector3<f64>> = blocks
        .iter()
        .map(|b| b.triangulate(&consensus.y).unwrap_or_else(Vector3::zeros))
        .collect();
    let mut velocity: Vector3<f64> = consensus.y.fixed_rows::<3>(0).into_owned();
    let mut gravity: Vector3<f64> = consensus.y.fixed_rows::<3>(3).into_owned();
    let mut bias_gyro: Vector3<f64> = bias_gyro_prior;
    let mut bias_accel: Vector3<f64> = zero;

    // Refine, re-integrate, re-score; until the inlier set and biases settle.
    let mut refined: Option<refine::Refined> = None;
    let mut angles: Vec<f64> = Vec::new();
    // Whether a pass ended with the inlier set and the biases unchanged. Without it, the refinement and the
    // re-scored inlier set below describe different track sets, so the window is not accepted.
    let mut settled: bool = false;
    for _pass in 0..5 {
        let used: Vec<usize> = (0..candidates.len()).filter(|&i| inlier[i]).collect();
        if used.len() < options.min_tracks {
            quality.tracks_used = used.len();
            return reject(
                RejectReason::TooFewTracks {
                    tracks: used.len(),
                    required: options.min_tracks,
                },
                quality,
                None,
            );
        }
        let used_obs: Vec<&[BearingObservation]> = used.iter().map(|&i| observations[i]).collect();
        let refine_problem = refine::Problem {
            preint: &preint,
            cameras: &cameras,
            tracks: &used_obs,
            huber: options.huber_rad,
            bearing_sigma: options.bearing_sigma_rad,
            estimate_accel_bias: options.estimate_accel_bias,
            gyro_bias_prior: bias_gyro_prior,
            gyro_bias_sigma: options.gyro_bias_sigma,
            accel_bias_sigma: options.accel_bias_sigma,
        };
        let state = refine::State {
            velocity,
            gravity,
            bias_gyro,
            bias_accel,
            points: used.iter().map(|&i| points[i]).collect(),
        };
        let result = refine::refine(&refine_problem, state, options.max_iterations);
        velocity = result.state.velocity;
        gravity = result.state.gravity;
        bias_gyro = result.state.bias_gyro;
        bias_accel = result.state.bias_accel;
        for (&i, p) in used.iter().zip(&result.state.points) {
            points[i] = *p;
        }
        refined = Some(result);

        // Re-integrate at the refined biases when they moved.
        let moved: bool = (bias_gyro - preint.bias_gyro).norm() > 1e-5
            || (bias_accel - preint.bias_accel).norm() > 1e-4;
        if moved {
            preint = preintegrate(problem.imu, problem.frame_times_ns, &bias_gyro, &bias_accel)?;
        }

        // Re-score every candidate: inliers keep their refined point, the
        // others are triangulated by the closed form at the refined state.
        let poses = linear::poses(&preint, &velocity, &gravity, &bias_gyro, &bias_accel);
        let y: Vector6<f64> = Vector6::new(
            velocity.x, velocity.y, velocity.z, gravity.x, gravity.y, gravity.z,
        );
        let mut new_inlier: Vec<bool> = Vec::with_capacity(candidates.len());
        for i in 0..candidates.len() {
            if !inlier[i] {
                let block = linear::track_block(observations[i], &preint, &cameras);
                if let Some(x) = block.and_then(|b| b.triangulate(&y)) {
                    points[i] = x;
                }
            }
            linear::track_angles(observations[i], &points[i], &poses, &cameras, &mut angles);
            new_inlier.push(linear::rms(&angles) < options.outlier_threshold_rad);
        }
        let stable: bool = new_inlier == inlier && !moved;
        inlier = new_inlier;
        if stable {
            settled = true;
            break;
        }
    }
    let Some(refined) = refined else {
        return reject(RejectReason::NotConverged, quality, None);
    };

    // Quality at the final state.
    let used: Vec<usize> = (0..candidates.len()).filter(|&i| inlier[i]).collect();
    quality.tracks_used = used.len();
    quality.converged = refined.converged;
    let poses = linear::poses(&preint, &velocity, &gravity, &bias_gyro, &bias_accel);
    let mut all_angles: Vec<f64> = Vec::new();
    for &i in &used {
        linear::track_angles(observations[i], &points[i], &poses, &cameras, &mut angles);
        all_angles.extend_from_slice(&angles);
    }
    let residual_rms: f64 = linear::rms(&all_angles);
    quality.residual_rms_rad = Some(residual_rms);

    let final_blocks: Vec<TrackBlock> = used
        .iter()
        .filter_map(|&i| linear::track_block(observations[i], &preint, &cameras))
        .collect();
    let unconstrained = linear::solve_unconstrained(final_blocks.iter());
    if let Some((y, condition)) = unconstrained {
        quality.condition = Some(condition);
        quality.gravity_norm_unconstrained = Some(y.fixed_rows::<3>(3).norm());
    }
    // Scale sigma: velocity block of the inverse reduced information, with
    // the residual variance estimated from the robust cost.
    let dof: f64 = (2 * refined.observations) as f64
        - 3.0 * used.len() as f64
        - refined.information.nrows() as f64;
    let sigma2: f64 = if dof > 0.0 {
        refined.data_cost / dof
    } else {
        f64::INFINITY
    };
    let speed: f64 = velocity.norm();
    quality.scale_sigma = refined.information.clone().try_inverse().map(|cov| {
        let c_vv: Matrix3<f64> = cov.fixed_view::<3, 3>(0, 0).into_owned() * sigma2;
        if speed > 0.0 {
            let dir: Vector3<f64> = velocity / speed;
            (dir.dot(&(c_vv * dir))).max(0.0).sqrt() / speed
        } else {
            f64::INFINITY
        }
    });

    let world_from_body0: Rotation3<f64> = world_from_gravity(&gravity);
    let solution = InertialInit {
        velocity_body0: velocity,
        gravity_body0: gravity,
        velocity_world: world_from_body0.matrix() * velocity,
        world_from_body0,
        bias_gyro,
        bias_accel,
        rotations_body0: poses
            .iter()
            .map(|(r, _)| {
                Rotation3::from_unit_quaternion(UnitQuaternion::from_rotation_matrix(
                    &nalgebra::Rotation3::from_matrix_unchecked(*r),
                ))
            })
            .collect(),
        positions_body0: poses.iter().map(|(_, p)| *p).collect(),
        points_body0: used
            .iter()
            .map(|&i| (candidates[i].id, points[i]))
            .collect(),
        outlier_track_ids: (0..candidates.len())
            .filter(|&i| !inlier[i])
            .map(|i| candidates[i].id)
            .collect(),
    };

    let finite: bool = velocity
        .iter()
        .chain(gravity.iter())
        .chain(bias_gyro.iter())
        .all(|v| v.is_finite());
    let verdict: InitVerdict = if !finite || !settled {
        InitVerdict::Rejected(RejectReason::NotConverged)
    } else if used.len() < options.min_tracks {
        InitVerdict::Rejected(RejectReason::TooFewTracks {
            tracks: used.len(),
            required: options.min_tracks,
        })
    } else if quality
        .condition
        .is_none_or(|c| !within(c, options.max_condition))
        || quality
            .scale_sigma
            .is_none_or(|s| !within(s, options.max_scale_sigma))
    {
        InitVerdict::Rejected(RejectReason::IllConditioned {
            condition: quality.condition.unwrap_or(f64::INFINITY),
            scale_sigma: quality.scale_sigma.unwrap_or(f64::INFINITY),
        })
    } else if !within(residual_rms, options.max_residual_rms_rad) {
        InitVerdict::Rejected(RejectReason::ResidualTooHigh {
            residual_rms_rad: residual_rms,
        })
    } else if quality.gravity_norm_unconstrained.is_none_or(|n| {
        !within(
            (n / gravity_magnitude - 1.0).abs(),
            options.max_gravity_norm_error,
        )
    }) {
        InitVerdict::Rejected(RejectReason::GravityNormInconsistent {
            gravity_norm: quality.gravity_norm_unconstrained.unwrap_or(f64::NAN),
        })
    } else {
        InitVerdict::Accepted
    };
    Ok(InertialInitOutcome {
        verdict,
        solution: Some(solution),
        quality,
    })
}
