//! One-frame-lag frontend/estimator scheduling (M7).

use super::{Vio, VioError, VioResult};
use crate::{duration_ns, estimator, frontend, imu, lie, types};

/// Wall times for the estimator running beside the frontend, in nanoseconds.
/// These overlap the frontend timers and must not be added to them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverlapTimings {
    /// Full estimator call, including any preceding D84 solve.
    pub estimator_ns: u64,
    /// Time waiting for the estimator after the frontend completed.
    pub wait_ns: u64,
}

impl<S: lie::LieScalar> Vio<S> {
    /// Timings for the last estimator work beside the frontend; zero at startup.
    pub fn overlap_timings(&self) -> OverlapTimings {
        self.overlap_timings
    }

    /// Drain the last queued frameset, returning its pose exactly once.
    ///
    /// Call at the end of a lagged stream. With lag disabled, or after a drain,
    /// returns `None`. Also finishes any D84 solve before exposing the window.
    /// The returned pose keeps D84's pre-solve publication rule. Tracking may
    /// resume after a drain; the next accepted frame starts a new queue.
    ///
    /// # Errors
    /// The estimator's measurement, joint-solve or marginalization errors.
    pub fn flush(&mut self) -> Result<Option<VioResult>, VioError> {
        self.last_deferred = None;
        self.overlap_timings = OverlapTimings::default();
        let result = if let Some(observations) = self.pending_observations.take() {
            let mark = std::time::Instant::now();
            let t_ns = observations.t_ns;
            self.last_deferred = self
                .estimator
                .finish_deferred_keyframe(self.frontend.cpu_pool().as_ref())?
                .map(Box::new);
            let outcome = self
                .estimator
                .process_frame(observations, self.frontend.cpu_pool().as_ref())?;
            self.overlap_timings.estimator_ns = duration_ns(mark);
            Some(self.accept_outcome(t_ns, outcome)?)
        } else {
            None
        };
        self.finish_deferred_keyframe()?;
        Ok(result)
    }

    pub(super) fn process_frame_beside_estimator(
        &mut self,
        t_ns: i64,
        prediction: &frontend::flow::PosePrediction,
    ) -> Result<Option<(i64, estimator::FrameOutcome<S>)>, VioError> {
        self.overlap_timings = OverlapTimings::default();
        let observations = self.pending_observations.as_ref().cloned();
        if observations.is_none() && !self.estimator.has_deferred_keyframe() {
            self.frontend
                .process_frame(t_ns, &self.frames, prediction, &self.masks)?;
            return Ok(None);
        }
        // The frontend owns these workers while the estimator runs beside it.
        let Self {
            frontend,
            estimator,
            frames,
            masks,
            ..
        } = self;
        let (flow, estimated, wait_ns) = std::thread::scope(|scope| -> Result<_, VioError> {
            let worker = std::thread::Builder::new()
                .name("slam-rs-estimator".to_owned())
                .spawn_scoped(scope, || {
                    let mark = std::time::Instant::now();
                    let deferred = estimator.finish_deferred_keyframe(None)?;
                    let outcome = observations
                        .map(|frame| {
                            let t_ns = frame.t_ns;
                            estimator
                                .process_frame(frame, None)
                                .map(|result| (t_ns, result))
                        })
                        .transpose()?;
                    Ok::<_, estimator::EstimatorError>((outcome, deferred, duration_ns(mark)))
                })
                .map_err(|error| VioError::EstimatorThread(error.to_string()))?;
            let flow = frontend
                .process_frame(t_ns, frames, prediction, masks)
                .map(|_| ());
            let mark = std::time::Instant::now();
            let estimated = worker.join().map_err(|_| VioError::EstimatorPanicked);
            Ok((flow, estimated, duration_ns(mark)))
        })?;
        let (outcome, deferred, estimator_ns) = estimated??;
        self.pending_observations = None;
        self.last_deferred = deferred.map(Box::new);
        self.overlap_timings = OverlapTimings {
            estimator_ns,
            wait_ns,
        };
        flow?;
        Ok(outcome)
    }

    /// Retain the extra IMU interval and predict both image poses from the
    /// same published state. Using the old state's pose directly as "previous"
    /// would incorrectly give KLT a two-frame displacement for one-frame images.
    pub(super) fn lagged_prediction(
        &mut self,
        t_ns: i64,
        latest: &types::PoseVelBiasState<f64>,
    ) -> Result<frontend::flow::PosePrediction, VioError> {
        self.drop_frontend_imu_through(latest.t_ns);
        let predict = |until_ns: i64| -> Result<_, VioError> {
            if until_ns == latest.t_ns {
                return Ok(latest.t_w_i);
            }
            let mut samples = self.frontend_imu.iter();
            let mut pim = imu::IntegratedImuMeasurement::new(
                latest.t_ns,
                &latest.bias_gyro,
                &latest.bias_accel,
            );
            pim.accumulate_to(
                None,
                || samples.next().map(|sample| self.calibrated(sample)),
                latest.t_ns,
                until_ns,
                &self.frontend_noise,
            )?;
            Ok(pim
                .predict_state(&latest.pose_vel_state(), &imu::gravity::<f64>())
                .t_w_i)
        };
        let previous_t_ns = self.last_frame_t_ns.unwrap_or(latest.t_ns);
        Ok(frontend::flow::PosePrediction {
            t_w_i_previous: predict(previous_t_ns)?.cast(),
            t_w_i_current: predict(t_ns)?.cast(),
        })
    }
}
