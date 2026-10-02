//! Keypoint storage and frontend boundary values.

#[cfg(doc)]
use super::FrameToFrameOpticalFlow;
use crate::frontend::se2::AffineCompact2f;
use crate::frontend::tracker::FlowTransforms;
use crate::lie::Se3;
use crate::types::KeypointId;

/// Sentinel response `-1` for a keypoint without a detector score.
pub const NO_RESPONSE: f32 = -1.0;

/// Frontend options kept separate from the serialized VIO configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontendOptions {
    /// Workers the tracking passes run on (decision D31). One is the
    /// deterministic reference lane; any value gives the same numbers.
    pub threads: usize,
    /// Maximum keypoints per camera and the capacity used for its buffers.
    /// Detection and matching stop adding in scan order when full, preserving existing
    /// tracks and keeping the frame processable. The default exceeds the shipped
    /// 50-pixel grid's capacity on a 960x960 image; the caller's upper limit is
    /// [`crate::frontend::tracker::MAX_CAPACITY`].
    pub max_keypoints: usize,
}

impl Default for FrontendOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            // The shipped grid is 50 px over a 960x960 frame with one point per
            // cell, so 361 cells; 3000 leaves room for a finer grid and for the
            // extra points the non-overlap pass adds on cameras 1 and up.
            max_keypoints: 3000,
        }
    }
}

/// Previous and predicted poses supplied by IMU preintegration.
/// Keeping them as inputs makes the frontend independent of IMU processing.
/// Two identities produce a depth-based reprojection of the same pixel until
/// the first estimated state arrives.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PosePrediction {
    /// `latest_state->T_w_i`, the pose of the previous frame.
    pub t_w_i_previous: Se3<f32>,
    /// `predicted_state->T_w_i`, the pose predicted for this frame.
    pub t_w_i_current: Se3<f32>,
}

/// Tracked keypoints in structure-of-arrays form.
/// Ids stay sorted ascending for deterministic filtering and id assignment,
/// with binary-search lookup. [`FlowTransforms`] stores six coefficient arrays
/// with keypoint index varying fastest; no per-frame hash map is needed.
#[derive(Debug, Default, PartialEq)]
pub struct Keypoints {
    /// Keypoint ids, ascending. `LandmarkId == KeypointId`.
    pub ids: Vec<KeypointId>,
    /// The 2x3 warp of each keypoint, in the same order as [`Keypoints::ids`].
    pub transforms: FlowTransforms,
    /// Detector responses, using [`NO_RESPONSE`] for tracked or stereo-matched points
    /// without a score. Detected points carry integer OpenCV-style corner scores.
    pub responses: Vec<f32>,
}

/// `Clone` by hand for the sake of `clone_from`.
///
/// `#[derive(Clone)]` writes only `clone`; the default `clone_from` is
/// `*self = source.clone()`, which drops every buffer and allocates as many
/// again — sixteen allocations and sixteen frees per stereo frame once
/// [`FrameToFrameOpticalFlow::process_frame`] takes its snapshot and again if it
/// has to restore. Copying field by field lets `Vec::clone_from` overwrite in
/// place, and `FlowTransforms` does the same one level down.
impl Clone for Keypoints {
    fn clone(&self) -> Self {
        Self {
            ids: self.ids.clone(),
            transforms: self.transforms.clone(),
            responses: self.responses.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.ids.clone_from(&source.ids);
        self.transforms.clone_from(&source.transforms);
        self.responses.clone_from(&source.responses);
    }
}

impl Keypoints {
    /// How many keypoints this camera carries.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether this camera has no keypoints.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The warp stored for `id`, or `None`.
    pub fn get(&self, id: KeypointId) -> Option<AffineCompact2f> {
        self.index_of(id).map(|index| self.transforms.get(index))
    }

    /// The warp at `index`, in id order.
    ///
    /// # Panics
    ///
    /// Whatever [`FlowTransforms::get`] panics on: an `index` past the end.
    pub fn transform(&self, index: usize) -> AffineCompact2f {
        self.transforms.get(index)
    }

    /// Where `id` sits in the sorted arrays.
    fn index_of(&self, id: KeypointId) -> Option<usize> {
        self.ids.binary_search(&id).ok()
    }

    pub(super) fn clear(&mut self) {
        self.ids.clear();
        self.transforms.clear();
        self.responses.clear();
    }

    /// `transforms->keypoints[cam][id] = kp`: insert or overwrite.
    pub(super) fn set(&mut self, id: KeypointId, transform: &AffineCompact2f, response: f32) {
        match self.ids.binary_search(&id) {
            Ok(index) => {
                self.transforms.set(index, transform);
                self.responses[index] = response;
            }
            Err(index) => {
                self.ids.insert(index, id);
                self.transforms.insert(index, transform);
                self.responses.insert(index, response);
            }
        }
    }

    /// `std::map::insert`, which **keeps** an existing entry.
    ///
    /// Returns whether the keypoint was new.
    pub(super) fn insert_if_absent(&mut self, id: KeypointId, transform: &AffineCompact2f) -> bool {
        match self.ids.binary_search(&id) {
            Ok(_) => false,
            Err(index) => {
                self.ids.insert(index, id);
                self.transforms.insert(index, transform);
                self.responses.insert(index, NO_RESPONSE);
                true
            }
        }
    }

    pub(super) fn remove(&mut self, id: KeypointId) -> Option<AffineCompact2f> {
        let index: usize = self.index_of(id)?;
        let transform: AffineCompact2f = self.transforms.get(index);
        self.ids.remove(index);
        self.responses.remove(index);
        self.transforms.remove(index);
        Some(transform)
    }
}

/// What one frameset produced: one [`Keypoints`] per camera.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FlowFrame {
    // Frameset timestamp, absent until the first successful commit.
    // Using `Option` leaves every i64 value, including negative values, valid as a timestamp.
    pub t_ns: Option<i64>,
    /// One entry per camera, in rig order.
    pub cameras: Vec<Keypoints>,
}

/// Wall time the frontend's phases took on the last frame, nanoseconds.
///
/// Reported, never compared: they are wall-clock, so they differ run to run and
/// nothing in `process_frame` reads them, which is what keeps the frame
/// bit-reproducible (D17), exactly as the estimator's own `StageTimings` are.
///
/// These do not add up to the frame: cell counts and other bookkeeping
/// remain outside the measured stages. Stereo includes cross-camera matching
/// and the epipolar filter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlowTimings {
    /// Building this frame's pyramids, every camera.
    pub pyramid_ns: u64,
    /// `detectKeypointsWithCells`, every camera.
    pub detect_ns: u64,
    /// Temporal `trackPoints` calls only.
    pub track_ns: u64,
    /// Cross-camera matching and epipolar filtering.
    pub stereo_ns: u64,
}
