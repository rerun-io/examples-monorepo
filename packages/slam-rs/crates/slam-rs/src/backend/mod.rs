//! The back end: a persistent keyframe map beside the window (DESIGN §2, PR a1).
//!
//! [`MapBackend::step`] takes the window's [`MapStamp`]s in order and keeps the
//! [`KeyframeMap`], its [`CovisibilityGraph`] and the culling rules
//! ([`cull`]) up to date. It is a synchronous state machine with no thread:
//! [`crate::Vio`] steps it inline after each frameset (Offline); a later PR
//! wraps the same `step` in a thread for Realtime.
//!
//! Host only, behind the `backend` cargo feature; wasm, iOS and embedded builds
//! leave it out. The window never reads the map, so the raw VIO poses are the
//! same with the back end on or off.

pub mod covisibility;
pub mod cull;
pub mod map;

use std::collections::{BTreeMap, BTreeSet};

use crate::config::VioConfig;
use crate::estimator::{KeyframeStamp, MapStamp, MargStamp};
use crate::types::FrameId;

pub use covisibility::{CovisibilityGraph, PoseGraphEdge, PoseGraphEdgeKind};
pub use cull::CovisibilityThreshold;
pub use map::{KeyframeMap, MapKeyframe, MapLandmark, MapStats};

/// The map's knobs, read from [`VioConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct MapConfig {
    /// `config.map_enable_culling`.
    pub enable_culling: bool,
    /// `config.map_max_redundancy_score`.
    pub max_redundancy_score: f32,
    /// `config.map_redundancy_exclude_hosted_lms`.
    pub redundancy_exclude_hosted: bool,
    /// `config.map_max_keyframes`.
    pub max_keyframes: usize,
    /// `config.map_batch_cull_size`.
    pub batch_cull_size: usize,
    /// `config.loop_closing_pgo_min_covisibility_weight`, or
    /// `backend.high_covisibility_landmark_ratio` when that is unset.
    pub high_covisibility: CovisibilityThreshold,
}

impl From<&VioConfig> for MapConfig {
    fn from(config: &VioConfig) -> Self {
        Self {
            enable_culling: config.map_enable_culling,
            max_redundancy_score: config.map_max_redundancy_score as f32,
            redundancy_exclude_hosted: config.map_redundancy_exclude_hosted_lms,
            max_keyframes: config.map_max_keyframes,
            batch_cull_size: config.map_batch_cull_size,
            high_covisibility: match config.loop_closing_pgo_min_covisibility_weight {
                Some(weight) => CovisibilityThreshold::Fixed(weight),
                None => CovisibilityThreshold::PerMedianLandmarks(
                    config.backend_high_covisibility_landmark_ratio,
                ),
            },
        }
    }
}

/// Why a keyframe left the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullReason {
    /// Redundant as it left the window.
    Redundant,
    /// Lowest-scored while the map was over its keyframe budget.
    OverBudget,
}

/// What one [`MapBackend::step`] did, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapEvent {
    /// A keyframe joined the map.
    KeyframeAdded(FrameId),
    /// A keyframe left the window; the map keeps it.
    KeyframeMarginalized(FrameId),
    /// A keyframe was culled.
    KeyframeCulled {
        /// The keyframe.
        id: FrameId,
        /// Which rule culled it.
        reason: CullReason,
    },
}

/// The map, its covisibility graph and what protects keyframes from culling.
#[derive(Debug, Clone)]
pub struct MapBackend {
    config: MapConfig,
    /// Image size per camera, for the coverage score.
    resolutions: Vec<[u32; 2]>,
    map: KeyframeMap,
    graph: CovisibilityGraph,
    /// Keyframes still in the window.
    active: BTreeSet<FrameId>,
    /// Keyframes loop closing holds (an island from request to apply).
    pinned: BTreeSet<FrameId>,
    /// Keyframes the map culled; a stamp never brings one back.
    culled: BTreeSet<FrameId>,
}

impl MapBackend {
    /// An empty map for a rig with these image sizes.
    pub fn new(config: MapConfig, resolutions: Vec<[u32; 2]>) -> Self {
        Self {
            config,
            resolutions,
            map: KeyframeMap::new(),
            graph: CovisibilityGraph::new(),
            active: BTreeSet::new(),
            pinned: BTreeSet::new(),
            culled: BTreeSet::new(),
        }
    }

    /// The map.
    pub fn map(&self) -> &KeyframeMap {
        &self.map
    }

    /// The covisibility graph.
    pub fn graph(&self) -> &CovisibilityGraph {
        &self.graph
    }

    /// The `stats_map.json` counts.
    pub fn stats(&self) -> MapStats {
        self.map.stats()
    }

    /// Keyframes still in the window.
    pub fn active_keyframes(&self) -> &BTreeSet<FrameId> {
        &self.active
    }

    /// The shared-landmark count at which an edge is "high", for the map as it is.
    pub fn high_covisibility_weight(&self) -> u32 {
        self.config.high_covisibility.weight(&self.map)
    }

    /// Whether culling must leave `id` alone: the root, a window keyframe, or a
    /// pinned one.
    pub fn is_protected(&self, id: FrameId) -> bool {
        self.graph.root() == Some(id) || self.active.contains(&id) || self.pinned.contains(&id)
    }

    /// Protect keyframes from culling until [`Self::unpin`] (B10).
    pub fn pin(&mut self, ids: impl IntoIterator<Item = FrameId>) {
        self.pinned.extend(ids);
    }

    /// Release keyframes [`Self::pin`] protected.
    pub fn unpin(&mut self, ids: impl IntoIterator<Item = FrameId>) {
        for id in ids {
            self.pinned.remove(&id);
        }
    }

    /// Take one stamp from the window.
    pub fn step(&mut self, stamp: MapStamp) -> Vec<MapEvent> {
        match stamp {
            MapStamp::Keyframe(stamp) => self.merge_keyframe(&stamp),
            MapStamp::Marginalized(stamp) => self.merge_marginalized(&stamp),
        }
    }

    /// `write_map_stamp`: window poses and landmarks into the map, new keyframes
    /// into the graph, then a batch cull if the map is over budget.
    fn merge_keyframe(&mut self, stamp: &KeyframeStamp) -> Vec<MapEvent> {
        let mut events: Vec<MapEvent> = Vec::new();
        let mut added: Vec<FrameId> = Vec::new();
        for (&id, pose) in &stamp.keyframe_poses {
            if self.culled.contains(&id) {
                continue;
            }
            let keypoints = (id == stamp.t_ns).then(|| stamp.keypoints.clone());
            if !self.map.contains_keyframe(id) {
                self.active.insert(id);
                added.push(id);
                events.push(MapEvent::KeyframeAdded(id));
            }
            self.map.upsert_keyframe(id, *pose, keypoints);
        }
        self.map.merge_landmarks(&stamp.landmarks);
        for id in added {
            let shared: BTreeMap<FrameId, u32> = self.shared_landmarks(id);
            self.graph.insert_keyframe(id, &shared);
        }
        if self.config.enable_culling && self.map.keyframes().len() > self.config.max_keyframes {
            let high_weight: u32 = self.high_covisibility_weight();
            let batch: Vec<FrameId> = cull::batch_to_cull(
                &self.map,
                &self.graph,
                |id| self.is_protected(id),
                high_weight,
                &self.resolutions,
                self.config.batch_cull_size,
            );
            for id in batch {
                if self.cull_keyframe(id) {
                    events.push(MapEvent::KeyframeCulled {
                        id,
                        reason: CullReason::OverBudget,
                    });
                }
            }
        }
        events
    }

    /// `write_map_marg`, with final values: the evicted keyframes' poses and
    /// landmarks, then the redundancy cull of each.
    fn merge_marginalized(&mut self, stamp: &MargStamp) -> Vec<MapEvent> {
        let mut events: Vec<MapEvent> = Vec::new();
        for (&id, pose) in &stamp.keyframe_poses {
            if self.map.contains_keyframe(id) {
                self.map.upsert_keyframe(id, *pose, None);
            }
        }
        self.map.merge_landmarks(&stamp.landmarks);
        for &id in stamp.keyframe_poses.keys() {
            if !self.active.remove(&id) {
                continue;
            }
            events.push(MapEvent::KeyframeMarginalized(id));
            if self.config.enable_culling
                && self
                    .map
                    .redundancy_score(id, self.config.redundancy_exclude_hosted)
                    >= self.config.max_redundancy_score
                && self.cull_keyframe(id)
            {
                events.push(MapEvent::KeyframeCulled {
                    id,
                    reason: CullReason::Redundant,
                });
            }
        }
        events
    }

    /// Shared landmarks between keyframe `id` and every graph node (and the root).
    fn shared_landmarks(&self, id: FrameId) -> BTreeMap<FrameId, u32> {
        let mut shared: BTreeMap<FrameId, u32> = BTreeMap::new();
        for landmark_id in self.map.landmarks_of_keyframe(id) {
            let Some(landmark) = self.map.landmark(landmark_id) else {
                continue;
            };
            let others: BTreeSet<FrameId> = landmark
                .obs
                .keys()
                .chain([&landmark.host])
                .map(|image| image.frame_id)
                .filter(|&other| other != id && self.graph.contains(other))
                .collect();
            for other in others {
                *shared.entry(other).or_insert(0) += 1;
            }
        }
        shared
    }

    /// `cullKeyframe`: remove `id` from the map and the graph, and lower the
    /// edges between the keyframes that shared a landmark the removal took.
    /// `false`, changing nothing, for a protected or unknown keyframe.
    fn cull_keyframe(&mut self, id: FrameId) -> bool {
        if self.is_protected(id) {
            return false;
        }
        let Some(removed) = self.map.remove_keyframe(id) else {
            return false;
        };
        self.graph.remove_keyframe(id);
        for (_, landmark) in &removed {
            let observers: Vec<FrameId> = landmark
                .obs
                .keys()
                .map(|image| image.frame_id)
                .filter(|&other| other != id)
                .collect::<BTreeSet<FrameId>>()
                .into_iter()
                .collect();
            for (k, &a) in observers.iter().enumerate() {
                for &b in &observers[k + 1..] {
                    self.graph.decrement_edge(a, b, 1);
                }
            }
        }
        self.culled.insert(id);
        true
    }
}

#[cfg(test)]
mod tests;
