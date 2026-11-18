//! What the window hands the persistent keyframe map (DESIGN §2.2).
//!
//! Two messages, both copies taken *after* the window has solved, so they carry
//! what the window ends up believing rather than what it predicted:
//!
//! - a [`KeyframeStamp`] after every keyframe's joint solve and marginalization
//!   (the synchronous keyframe, or [`SqrtKeypointVio::finish_deferred_keyframe`]
//!   for a deferred one, D84): the window keyframes' poses, every window
//!   landmark with its keyframe observations, and the new keyframe's tracked
//!   keypoints;
//! - a [`MargStamp`] when the keyframe budget evicts keyframes: their final
//!   poses and the final values of the landmarks they host, taken just before
//!   the marginalization drops them.
//!
//! RootSLAM sends a whole window `LandmarkDatabase` copy per keyframe whose
//! keyframe poses are the *predicted* poses written when the keyframe was
//! created (`sqrt_keypoint_vio.cpp:681-682`), and only the evicted ids on
//! marginalization (`:1238-1249`). Here every stamp carries solved poses, and
//! the marginalization stamp carries the final ones.
//!
//! The stamps are reads of the window: emitting them changes no number the
//! window computes. Nothing is emitted unless [`SqrtKeypointVio::emit_map_stamps`]
//! turned it on. The estimator never puts an image in a stamp; the pipeline
//! attaches the new keyframe's own images ([`KeyframeImages`]) for loop
//! closing, which drops them after that keyframe's query (the retention rule,
//! DESIGN §2.2).

use kornia_staging_algebra::Scalar;
use std::collections::{BTreeMap, BTreeSet};

use nalgebra::Vector2;

#[cfg(doc)]
use super::SqrtKeypointVio;
use crate::landmark::Landmark;
use crate::lie::Se3;
use crate::types::{FrameId, KeypointId, LandmarkId, TimeCamId};

/// Tracked keypoints per camera, each camera in id order: pixels only.
pub type Keypoints = Vec<Vec<(KeypointId, Vector2<f32>)>>;

/// One window landmark as the map stores it: host-relative, in `f64`, with
/// observations in keyframes only.
#[derive(Debug, Clone, PartialEq)]
pub struct StampLandmark {
    /// The landmark id (its source keypoint id).
    pub id: LandmarkId,
    /// The image its direction is expressed in.
    pub host: TimeCamId,
    /// Stereographic direction in the host camera frame.
    pub direction: Vector2<f64>,
    /// Inverse distance along that direction.
    pub inv_dist: f64,
    /// Where keyframes saw it. Observations in non-keyframe frames are left out:
    /// the map holds keyframes only.
    pub obs: BTreeMap<TimeCamId, Vector2<f32>>,
}

impl StampLandmark {
    /// Copy a window landmark, keeping the observations whose frame `keep`
    /// accepts. Pixels were `f32` at the frontend, so the narrowing is exact.
    pub(crate) fn from_window<S: Scalar>(
        landmark: &Landmark<S>,
        keep: impl Fn(FrameId) -> bool,
    ) -> Self {
        Self {
            id: landmark.id,
            host: landmark.host_kf_id,
            direction: landmark.direction.map(Scalar::to_f64),
            inv_dist: landmark.inv_dist.to_f64(),
            obs: landmark
                .obs
                .iter()
                .filter(|(tcid, _)| keep(tcid.frame_id))
                .map(|(tcid, pixel)| (*tcid, pixel.map(|v| v.to_f64() as f32)))
                .collect(),
        }
    }
}

/// One keyframe's images, one per camera (level 0, `u16`), shared without a
/// copy. Two values are equal when they share the same buffers.
#[derive(Clone)]
pub struct KeyframeImages(pub std::sync::Arc<Vec<kornia_image::Image<u16, 1>>>);

impl PartialEq for KeyframeImages {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for KeyframeImages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.0.iter().map(|image| [image.width(), image.height()]))
            .finish()
    }
}

/// A keyframe stamp's images on their way to the back end, which takes them
/// out; [`ImageSlots`] may empty the slot first. Equal when shared.
#[derive(Debug, Clone)]
pub struct ImageSlot(std::sync::Arc<std::sync::Mutex<Option<KeyframeImages>>>);

impl ImageSlot {
    /// A slot holding `images`.
    pub fn new(images: KeyframeImages) -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(Some(images))))
    }

    /// Take the images out, if they are still there.
    pub fn take(&self) -> Option<KeyframeImages> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

impl PartialEq for ImageSlot {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

/// The slots of the keyframe stamps sent to the back end: the newest `cap`
/// keep their images; an older one is emptied before the back end takes it
/// (Realtime, behind), whose query it would drop anyway. This bounds the
/// images queued stamps hold (retention, DESIGN §2.2).
#[derive(Debug, Default)]
pub struct ImageSlots {
    cap: usize,
    slots: std::collections::VecDeque<ImageSlot>,
}

impl ImageSlots {
    /// Keep the images of the newest `cap` stamps.
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            slots: std::collections::VecDeque::new(),
        }
    }

    /// A slot for a new stamp's images; empties the oldest past the cap.
    pub fn hold(&mut self, images: KeyframeImages) -> ImageSlot {
        let slot = ImageSlot::new(images);
        self.slots.push_back(slot.clone());
        while self.slots.len() > self.cap {
            self.slots.pop_front().and_then(|old| old.take());
        }
        slot
    }
}

/// The window after a keyframe's solve.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyframeStamp {
    /// The new keyframe's timestamp (its frame id).
    pub t_ns: i64,
    /// `T_w_i` of every keyframe in the window, the new one included.
    pub keyframe_poses: BTreeMap<FrameId, Se3<f64>>,
    /// Every window landmark, in id order.
    pub landmarks: Vec<StampLandmark>,
    /// The new keyframe's tracked keypoints per camera, in id order: what loop
    /// detection describes later. Pixels only.
    pub keypoints: Keypoints,
    /// The new keyframe's images, which the pipeline attaches for loop closing
    /// (`None` from the estimator, and when the pipeline no longer holds them).
    pub images: Option<ImageSlot>,
}

/// Keyframes leaving the window, with their final values.
#[derive(Debug, Clone, PartialEq)]
pub struct MargStamp {
    /// Final `T_w_i` of each evicted keyframe.
    pub keyframe_poses: BTreeMap<FrameId, Se3<f64>>,
    /// Final values of the landmarks the evicted keyframes host, in id order.
    pub landmarks: Vec<StampLandmark>,
}

/// One message from the window to the map, in the order the window emits them.
#[derive(Debug, Clone, PartialEq)]
pub enum MapStamp {
    /// A keyframe's solve finished.
    Keyframe(Box<KeyframeStamp>),
    /// The keyframe budget evicted keyframes.
    Marginalized(MargStamp),
}

impl<S: Scalar> super::SqrtKeypointVio<S> {
    /// Start (`true`) or stop (`false`) recording [`MapStamp`]s for
    /// [`Self::take_map_stamps`]. Off by default. Stopping drops what was not taken.
    pub fn emit_map_stamps(&mut self, on: bool) {
        self.map_stamps = on.then(Vec::new);
    }

    /// The stamps emitted since the last call, oldest first.
    pub fn take_map_stamps(&mut self) -> Vec<MapStamp> {
        self.map_stamps
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Every keyframe still in the window, long-term ones included.
    fn window_keyframes(&self) -> BTreeSet<FrameId> {
        self.kf_ids.union(&self.ltkfs).copied().collect()
    }

    /// `T_w_i` of the given window frames, in `f64`. A frame the window does not
    /// hold is a bug (a keyframe the map would never release): it fails debug
    /// builds, and release builds log it, count it in
    /// [`Self::missing_stamp_poses`] and leave it out.
    fn window_poses(&mut self, frames: &BTreeSet<FrameId>) -> BTreeMap<FrameId, Se3<f64>> {
        let mut poses: BTreeMap<FrameId, Se3<f64>> = BTreeMap::new();
        for &frame_id in frames {
            match self.ba.get_pose_state_with_lin(frame_id) {
                Ok(pose) => {
                    poses.insert(frame_id, pose.pose().cast::<f64>());
                }
                Err(error) => {
                    self.missing_stamp_poses += 1;
                    log::warn!(
                        "map stamp: {error} ({} missing so far)",
                        self.missing_stamp_poses
                    );
                }
            }
        }
        debug_assert_eq!(
            poses.len(),
            frames.len(),
            "a map stamp asked for a frame the window does not hold"
        );
        poses
    }

    /// Keyframe poses the map stamps asked for and the window did not hold.
    pub fn missing_stamp_poses(&self) -> usize {
        self.missing_stamp_poses
    }

    /// Record the window after `t_ns`'s keyframe solve, when stamps are on.
    pub(super) fn stamp_keyframe(&mut self, t_ns: i64) {
        if self.map_stamps.is_none() {
            return;
        }
        let keyframes: BTreeSet<FrameId> = self.window_keyframes();
        let keypoints: Keypoints = self
            .prev_opt_flow_res
            .get(&t_ns)
            .map(|frame| {
                frame
                    .cameras
                    .iter()
                    .map(|camera| camera.iter().map(|(id, pixel)| (*id, *pixel)).collect())
                    .collect()
            })
            .unwrap_or_default();
        let stamp = KeyframeStamp {
            t_ns,
            keyframe_poses: self.window_poses(&keyframes),
            landmarks: self
                .ba
                .lmdb
                .landmarks()
                .iter()
                .map(|landmark| {
                    StampLandmark::from_window(landmark, |frame| keyframes.contains(&frame))
                })
                .collect(),
            keypoints,
            images: None,
        };
        if let Some(stamps) = self.map_stamps.as_mut() {
            stamps.push(MapStamp::Keyframe(Box::new(stamp)));
        }
    }

    /// Record the final values of keyframes about to be evicted, when stamps are
    /// on. Called before the marginalization removes them from the window.
    pub(super) fn stamp_marginalized(&mut self, kfs_to_marg: &BTreeSet<FrameId>) {
        if self.map_stamps.is_none() || kfs_to_marg.is_empty() {
            return;
        }
        let keyframes: BTreeSet<FrameId> = self.window_keyframes();
        let stamp = MargStamp {
            keyframe_poses: self.window_poses(kfs_to_marg),
            landmarks: self
                .ba
                .lmdb
                .landmarks()
                .iter()
                .filter(|landmark| kfs_to_marg.contains(&landmark.host_kf_id.frame_id))
                .map(|landmark| {
                    StampLandmark::from_window(landmark, |frame| {
                        keyframes.contains(&frame) || kfs_to_marg.contains(&frame)
                    })
                })
                .collect(),
        };
        if let Some(stamps) = self.map_stamps.as_mut() {
            stamps.push(MapStamp::Marginalized(stamp));
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn a_window_landmark_keeps_only_its_keyframe_observations_in_f64() {
        let host = TimeCamId::new(10, 0);
        let mut landmark: Landmark<f32> =
            Landmark::new(LandmarkId(7), host, Vector2::new(0.25, -0.5), 0.5);
        landmark.obs.insert(host, Vector2::new(1.5, 2.5));
        landmark
            .obs
            .insert(TimeCamId::new(20, 1), Vector2::new(3.0, 4.0));
        landmark
            .obs
            .insert(TimeCamId::new(30, 0), Vector2::new(5.0, 6.0));

        let stamped = StampLandmark::from_window(&landmark, |frame| frame != 20);

        assert_eq!(stamped.id, LandmarkId(7));
        assert_eq!(stamped.host, host);
        assert_eq!(stamped.direction, Vector2::new(0.25, -0.5));
        assert_eq!(stamped.inv_dist, 0.5);
        assert_eq!(
            stamped.obs.keys().copied().collect::<Vec<_>>(),
            vec![host, TimeCamId::new(30, 0)]
        );
        assert_eq!(stamped.obs[&host], Vector2::new(1.5, 2.5));
    }
}
