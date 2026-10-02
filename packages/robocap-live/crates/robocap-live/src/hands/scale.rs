//! The hand scale: handfit's joint scale solve ([`handfit::scale`], MEgATrack §3.6) and the policy around it: the live
//! calibration of `ScaleMode::Auto` (collect, solve in the background, switch) and the phi range.
#![deny(missing_docs)]

use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::time::Instant;

use handfit::Model;
pub use handfit::scale::{CalibrationBlock, CalibrationConfig, ScaleCalibration, ScaleError, Termination, calibrate_scale};

/// handtrack's `run_pipeline.PHI_RANGE`: a calibrated phi outside it is clamped.
pub const PHI_RANGE: (f64, f64) = (0.75, 1.35);

/// The outcome of the live calibration, for logs.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleOutcome {
    /// The phi in use from now on.
    pub phi: f64,
    /// Stereo observations solved over (0 for the fallback).
    pub blocks: usize,
    /// What happened ("calibrated", "clamped from x", "generic fallback: ...").
    pub note: String,
    /// Wall-clock seconds of the solve.
    pub solve_s: f64,
}

impl ScaleOutcome {
    /// Keep `phi` (no calibration) and say why.
    fn fallback(phi: f64, note: String) -> Self {
        Self { phi, blocks: 0, note, solve_s: 0.0 }
    }
}

/// The phi a calibration answer leads to (handtrack `run_pipeline.calibrate_unknown_hand`: clamp to `PHI_RANGE`, fall back to 1).
pub fn scale_choice(answer: Result<ScaleCalibration, ScaleError>, solve_s: f64) -> ScaleOutcome {
    match answer {
        Ok(calibration) if calibration.phi.is_finite() => {
            let phi = calibration.phi.clamp(PHI_RANGE.0, PHI_RANGE.1);
            let note = if phi == calibration.phi { "calibrated".to_string() } else { format!("clamped from {:.4}", calibration.phi) };
            ScaleOutcome { phi, blocks: calibration.blocks, note, solve_s }
        }
        Ok(calibration) => ScaleOutcome { solve_s, ..ScaleOutcome::fallback(1.0, format!("generic fallback: calibrated phi is {}", calibration.phi)) },
        Err(error) => ScaleOutcome::fallback(1.0, format!("generic fallback: {error}")),
    }
}

type Answer = (Result<ScaleCalibration, ScaleError>, f64);

enum State {
    /// A fixed phi, or calibration finished.
    Final,
    /// Collecting stereo evidence until `until_ns` (set by the first frameset).
    Collecting { seconds: f64, until_ns: Option<i64>, blocks: Vec<CalibrationBlock> },
    /// The solve runs on a background thread; it answers with its wall-clock seconds.
    Solving { result: Receiver<Answer> },
}

/// The live scale calibration (`ScaleMode::Auto`): the stereo evidence of the first `seconds` of tracking, solved on a
/// background thread at the lowest CPU priority; or nothing to do (a fixed scale).
pub struct LiveScale {
    state: State,
    outcome: Option<ScaleOutcome>,
    /// Wait for the solve in the step that starts it (`HandsConfig::scale_wait`).
    wait: bool,
}

impl LiveScale {
    /// A fixed scale: nothing to calibrate.
    pub fn fixed() -> Self {
        Self { state: State::Final, outcome: None, wait: false }
    }

    /// Calibrate over the first `seconds` of tracking; with `wait`, the step that starts the solve waits for it.
    pub fn auto(seconds: f64, wait: bool) -> Self {
        Self { state: State::Collecting { seconds, until_ns: None, blocks: Vec::new() }, outcome: None, wait }
    }

    /// Whether the scale is final (fixed, or calibration finished).
    pub fn is_final(&self) -> bool {
        matches!(self.state, State::Final)
    }

    /// The calibration's outcome, once it finished.
    pub fn outcome(&self) -> Option<&ScaleOutcome> {
        self.outcome.as_ref()
    }

    /// After the frameset at `t_ns` with the hands' `blocks`: collect, start the solve of `generic`, pick up its answer.
    ///
    /// # Arguments
    ///
    /// * `t_ns` - The frameset's time.
    /// * `blocks` - Its reported hands' observations.
    /// * `generic` - The generic hand model (the solve's start at phi = 1).
    /// * `phi` - The phi in use (kept by a fallback).
    ///
    /// # Returns
    ///
    /// The outcome when the calibration finished in this call; the tracker then switches to its phi.
    pub fn advance(&mut self, t_ns: i64, blocks: Vec<CalibrationBlock>, generic: &Model, phi: f64) -> Option<ScaleOutcome> {
        let state = std::mem::replace(&mut self.state, State::Final);
        let mut finished = None;
        self.state = match state {
            State::Final => State::Final,
            State::Collecting { seconds, until_ns, blocks: mut collected } => {
                // The evidence of the frames within the first `seconds` (handtrack: the first `calibration_frames` frames).
                let until = until_ns.unwrap_or_else(|| t_ns.saturating_add((seconds * 1e9).round() as i64));
                if t_ns < until {
                    collected.extend(blocks);
                    State::Collecting { seconds, until_ns: Some(until), blocks: collected }
                } else {
                    let (sender, receiver) = channel();
                    let generic = generic.clone();
                    let spawned = std::thread::Builder::new().name("hand-scale".into()).spawn(move || {
                        lower_current_thread_priority();
                        let begin = Instant::now();
                        let result = calibrate_scale(&generic, &collected, &CalibrationConfig::default());
                        let _ = sender.send((result, begin.elapsed().as_secs_f64()));
                    });
                    match spawned {
                        Ok(_) => State::Solving { result: receiver },
                        Err(error) => {
                            finished = Some(ScaleOutcome::fallback(phi, format!("generic fallback: no thread ({error})")));
                            State::Final
                        }
                    }
                }
            }
            State::Solving { result } => match result.try_recv() {
                Ok((answer, solve_s)) => {
                    finished = Some(scale_choice(answer, solve_s));
                    State::Final
                }
                Err(TryRecvError::Empty) => State::Solving { result },
                Err(TryRecvError::Disconnected) => {
                    finished = Some(ScaleOutcome::fallback(phi, "generic fallback: the solve thread died".into()));
                    State::Final
                }
            },
        };
        if finished.is_none() && self.wait {
            finished = self.finish(phi);
        }
        if let Some(outcome) = &finished {
            self.outcome = Some(outcome.clone());
        }
        finished
    }

    /// Wait for a running solve and return its outcome (`None` when none runs); tests and replays that want a deterministic
    /// switch.
    pub fn finish(&mut self, phi: f64) -> Option<ScaleOutcome> {
        if !matches!(self.state, State::Solving { .. }) {
            return None;
        }
        let State::Solving { result } = std::mem::replace(&mut self.state, State::Final) else { return None };
        let outcome = match result.recv() {
            Ok((answer, solve_s)) => scale_choice(answer, solve_s),
            Err(_) => ScaleOutcome::fallback(phi, "generic fallback: the solve thread died".into()),
        };
        self.outcome = Some(outcome.clone());
        Some(outcome)
    }
}

/// Run the calling thread at the lowest CPU priority (nice 19): the scale solve then takes only the cycles the 30 fps step and
/// the other stages leave on the cores it shares with them. Best effort: a refusal leaves the priority as it was.
fn lower_current_thread_priority() {
    #[cfg(target_os = "linux")]
    // SAFETY: gettid takes no arguments; setpriority(PRIO_PROCESS, tid) on Linux changes the nice value of that one thread,
    // our own, and touches no memory of ours.
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        let _ = libc::setpriority(libc::PRIO_PROCESS, tid, 19);
    }
}

#[cfg(test)]
mod tests {
    use handfit::model::Step;
    use handfit::nalgebra::{SMatrix, SVector, Vector2, Vector3};
    use handfit::residual::View;
    use handfit::{Model, Pose};

    use super::*;
    use crate::hands::model::{GenericHandModel, identity_pose};

    /// Two pinhole views of a hand at a known scale, observed exactly.
    fn block(generic: &Model, phi: f64, translation: Vector3<f64>, angle: f64) -> CalibrationBlock {
        let truth = Model { pivots: generic.pivots * phi, rest: generic.rest * phi, ..generic.clone() };
        let rotation = handfit::nalgebra::Rotation3::from_euler_angles(1.2, angle, 0.1).into_inner();
        let pose = Pose { rotation, translation, ..identity_pose() };
        let points = truth.landmarks(&pose, 1.0, &Step::zeros(), None);
        let views: Vec<View> = [-0.06, 0.06]
            .iter()
            .map(|&x| {
                let rotation = handfit::nalgebra::Rotation3::from_euler_angles(0.0, -x * 3.0, 0.0).into_inner();
                let translation = Vector3::new(-x, 0.0, 0.0);
                let pixels = SMatrix::<f64, 21, 2>::from_fn(|i, k| {
                    let p = rotation * points.fixed_rows::<3>(3 * i).into_owned() + translation;
                    500.0 * p[k] / p[2] + if k == 0 { 320.0 } else { 240.0 }
                });
                View {
                    rotation,
                    translation,
                    focal: Vector2::repeat(500.0),
                    principal: Vector2::new(320.0, 240.0),
                    distortion: None,
                    pixels,
                    weights: SVector::repeat(1.0),
                    distances: SVector::zeros(),
                }
            })
            .collect();
        CalibrationBlock { mirror: 1.0, views, initial: pose }
    }

    #[test]
    fn exact_stereo_observations_recover_the_scale() -> Result<(), Box<dyn std::error::Error>> {
        let generic = GenericHandModel::load()?;
        let hands: Vec<CalibrationBlock> = (0..12)
            .map(|i| block(generic.model(), 0.9, Vector3::new(0.02 * (i as f64 - 6.0), 0.01 * i as f64, 0.35 + 0.01 * i as f64), 0.1 * i as f64))
            .collect();
        let calibration = calibrate_scale(generic.model(), &hands, &CalibrationConfig::default())?;
        assert!((calibration.phi - 0.9).abs() < 2e-3, "phi {} ({} iterations, {:?})", calibration.phi, calibration.iterations, calibration.termination);
        assert_eq!(calibration.blocks, 12);
        // Exact observations drive the energy towards 0, where the relative tolerance rarely fires: no convergence claim here.
        assert_ne!(calibration.termination, Termination::NonFinite);
        Ok(())
    }

    #[test]
    fn mono_observations_are_an_error() -> Result<(), Box<dyn std::error::Error>> {
        let generic = GenericHandModel::load()?;
        let mut one = block(generic.model(), 1.0, Vector3::new(0.0, 0.0, 0.4), 0.0);
        one.views.truncate(1);
        assert_eq!(calibrate_scale(generic.model(), &[one], &CalibrationConfig::default()).err(), Some(ScaleError::NoStereo));
        Ok(())
    }
}
