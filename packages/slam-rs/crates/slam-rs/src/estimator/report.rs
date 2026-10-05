//! Observation input and per-frame results reported by the estimator.

#[cfg(doc)]
use super::SqrtKeypointVio;
use super::{FrameUpdateOutcome, LmIteration, LmTermination, MarginalizationStats};
use crate::lie::LieScalar;
use crate::types::{FrameId, KeypointId};
use nalgebra::Vector2;
use std::collections::BTreeMap;

/// The estimator's flow input: a frameset timestamp and one keypoint-to-pixel map
/// per camera. This boundary permits backend tests without running the frontend.
/// Pixels are f32 even in the f64 estimator; widening cannot add precision that
/// the frontend did not produce.
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
    /// IMU coverage does not extend past this frameset. Nothing was consumed and
    /// the window is unchanged, so the caller can add samples and retry (D17).
    NeedMoreImu,
    /// The first keyframe hosted fewer than ten landmarks. The frontend may
    /// advance, but no estimator state exists; try the next frameset.
    NoVisualFeatures,
    /// The frame was measured. The statistics are per frame and are the input
    /// to the S9 Rerun rung.
    Measured(Box<FrameStats<S>>),
}

/// Per-frame stage durations in nanoseconds.
/// Wall-clock timings vary between runs and are reported but never used in
/// estimator decisions, preserving deterministic replay (D17).
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
    /// Whether `take_kf` was set when this frame arrived.
    pub took_keyframe: bool,
    /// Whether the keyframe vote of fired on this frame.
    pub keyframe_vote: bool,
    /// `frames_after_kf` after the update: the vote's rate limiter,
    /// zero on a keyframe and one more than the last frame otherwise.
    pub frames_after_kf: i32,
    /// `num_points_added`, zero on a non-keyframe.
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
    /// `opt_started` after this frameset: false until five states
    /// have accumulated, true from the first linearization on.
    pub opt_started: bool,
    /// A finite optimized pose with enough landmarks and tracked observations to publish.
    pub visually_supported: bool,
    /// One entry per LM step, accepted or rejected, in order.
    pub lm: Vec<LmIteration<S>>,
    /// Why the LM loop stopped.
    pub termination: LmTermination,
    /// The marginalization, when the trigger of fired.
    pub marginalization: Option<MarginalizationStats>,
    /// What D76's frame update did with this frameset: never attempted, taken,
    /// or refused by a named precondition.
    pub frame_update: FrameUpdateOutcome,
    /// Wall-clock stage marks; see [`StageTimings`].
    pub timings: StageTimings,
    /// Whether this keyframe frameset left its triangulation, joint solve and
    /// marginalization pending (D84): its pose is the frame update's, and
    /// [`SqrtKeypointVio::finish_deferred_keyframe`] reports the rest.
    pub keyframe_deferred: bool,
}
