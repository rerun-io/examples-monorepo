//! The deferred keyframe solve (D84).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::{
    EstimatorError, FlowObservations, LmIteration, LmTermination, MarginalizationOutcome,
    MarginalizationStats, SqrtKeypointVio, StageTimings,
};
use crate::duration_ns;
use crate::lie::LieScalar;
use crate::types::{FrameId, KeypointId, LandmarkId};

/// What a deferred keyframe's second half did (D84).
#[derive(Debug, Clone, PartialEq)]
pub struct DeferredKeyframeStats<S: LieScalar> {
    /// The keyframe frameset's timestamp.
    pub t_ns: i64,
    /// Landmarks the keyframe triangulated.
    pub num_points_added: usize,
    /// `lmdb.numLandmarks()` after the marginalization.
    pub num_landmarks: usize,
    /// The joint solve's LM steps.
    pub lm: Vec<LmIteration<S>>,
    /// Why the joint solve stopped.
    pub termination: LmTermination,
    /// The marginalization, when it fired.
    pub marginalization: Option<MarginalizationStats>,
    /// `keyframe_ns` (triangulation), `optimize_ns` (the joint solve and its
    /// detailed stages), `marginalize_ns`, and `measure_ns` for the whole half.
    pub timings: StageTimings,
}

/// What a keyframe frameset leaves for [`SqrtKeypointVio::finish_deferred_keyframe`].
#[derive(Debug, Clone)]
pub(super) struct DeferredKeyframe {
    /// The keyframe frameset's observations.
    pub(super) frame: Arc<FlowObservations>,
    /// Its keypoints the window had never seen, per camera: what triangulation reads.
    pub(super) unconnected_obs: Vec<BTreeSet<KeypointId>>,
    /// Its observations per host keyframe: the marginalization's input.
    pub(super) num_points_connected: BTreeMap<FrameId, usize>,
    /// Landmarks it did not see.
    pub(super) lost_landmarks: BTreeSet<LandmarkId>,
}

/// D84 defers a keyframe's joint solve only when the window already holds this
/// many landmarks and the frameset observes this many of them: the frame
/// update's pose then has visual support of its own (PR #270's live rule), and
/// the cold start — where the keyframe's new landmarks are the support — keeps
/// the synchronous solve.
pub(super) const DEFER_MIN_SUPPORT: usize = 10;

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// Every landmark `frame` did not see, in any camera, when
    /// `vio_marg_lost_landmarks` asks for them; empty otherwise.
    pub(super) fn lost_landmarks(&self, frame: &FlowObservations) -> BTreeSet<LandmarkId> {
        let mut lost_landmarks: BTreeSet<LandmarkId> = BTreeSet::new();
        if self.config.vio_marg_lost_landmarks {
            for lm in self.ba.lmdb.landmarks() {
                let kpt: KeypointId = KeypointId::from(lm.id);
                if !frame.cameras.iter().any(|cam| cam.contains_key(&kpt)) {
                    lost_landmarks.insert(lm.id);
                }
            }
        }
        lost_landmarks
    }

    /// Whether a keyframe's triangulation, joint solve and marginalization are
    /// pending (D84).
    pub fn has_deferred_keyframe(&self) -> bool {
        self.deferred.is_some()
    }

    /// Run a deferred keyframe's second half: triangulate its new landmarks
    /// against the updated pose, solve the window, marginalize. `None` when
    /// nothing is pending. [`Self::process_frame`] calls this itself before it
    /// touches the window; [`crate::Vio::track`] calls it on a second thread
    /// beside the next frameset's frontend.
    ///
    /// # Errors
    ///
    /// What triangulation, the joint solve or the marginalization refuse, as on
    /// a synchronous keyframe.
    pub fn finish_deferred_keyframe(
        &mut self,
    ) -> Result<Option<DeferredKeyframeStats<S>>, EstimatorError> {
        let Some(deferred) = self.deferred.take() else {
            return Ok(None);
        };
        let started: std::time::Instant = std::time::Instant::now();
        let t_ns: i64 = deferred.frame.t_ns;
        let num_points_added: usize =
            self.triangulate_unconnected(&deferred.frame, &deferred.unconnected_obs)?;
        self.num_points_kf.insert(t_ns, num_points_added);
        let keyframe_ns: u64 = duration_ns(started);
        let optimize_started: std::time::Instant = std::time::Instant::now();
        let (lm, termination, mut timings) = self.optimize(t_ns)?;
        timings.optimize_ns = duration_ns(optimize_started);
        timings.keyframe_ns = keyframe_ns;
        let marg: MarginalizationOutcome =
            self.marginalize(&deferred.num_points_connected, &deferred.lost_landmarks)?;
        timings.marginalize_ns = marg.elapsed_ns;
        timings.measure_ns = duration_ns(started);
        Ok(Some(DeferredKeyframeStats {
            t_ns,
            num_points_added,
            num_landmarks: self.ba.lmdb.num_landmarks(),
            lm,
            termination,
            marginalization: marg.marginalization,
            timings,
        }))
    }
}
