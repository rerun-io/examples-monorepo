//! The back end: a persistent keyframe map beside the window (DESIGN §2, PR a1).
//!
//! [`MapBackend::step`] takes the window's [`MapStamp`]s in order and keeps the
//! [`KeyframeMap`], its [`CovisibilityGraph`] and the culling rules
//! ([`cull`]) up to date. It is a synchronous state machine with no thread.
//! [`BackendRunner`] runs it inline (Offline) or on its own thread (Realtime,
//! PR a2); [`MapSnapshot`] copies it out.
//!
//! Host only, behind the `backend` cargo feature; wasm, iOS and embedded builds
//! leave it out. The window never reads the map, so the raw VIO poses are the
//! same with the back end on or off.

pub mod covisibility;
pub mod cull;
pub mod loop_closing;
pub mod map;
pub mod pgo;
pub mod runner;
pub mod snapshot;

use std::collections::{BTreeMap, BTreeSet};

use crate::config::VioConfig;
use crate::estimator::{KeyframeStamp, MapStamp, MargStamp};
use crate::types::FrameId;

pub use covisibility::{CovisibilityGraph, PoseGraphEdge, PoseGraphEdgeKind};
pub use loop_closing::{LoopClosing, LoopCounts, LoopDetection, LoopEvent, LoopOutcome};
pub use map::{KeyframeMap, MapKeyframe, MapLandmark, MapStats};
pub use pgo::{MapCorrection, PgoRun, PgoState};
pub use runner::{BackendCounters, BackendRunner, BackendView, BackendWork};
pub use snapshot::{MapSnapshot, SnapshotLoop};

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
    /// `config.loop_closing_pgo_min_covisibility_weight` (shared landmarks).
    pub high_covisibility: u32,
}

impl From<&VioConfig> for MapConfig {
    fn from(config: &VioConfig) -> Self {
        Self {
            enable_culling: config.map_enable_culling,
            max_redundancy_score: config.map_max_redundancy_score as f32,
            redundancy_exclude_hosted: config.map_redundancy_exclude_hosted_lms,
            max_keyframes: config.map_max_keyframes,
            batch_cull_size: config.map_batch_cull_size,
            high_covisibility: config.loop_closing_pgo_min_covisibility_weight,
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
    /// A loop passed verification at a new keyframe (PR b2).
    LoopVerified {
        /// The new keyframe.
        current: FrameId,
        /// The keyframe it closes a loop with.
        candidate: FrameId,
        /// Fused, optimized then fused, left unapplied, or stale.
        outcome: LoopOutcome,
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
    /// How many of `culled` the over-budget batches took.
    culled_over_budget: usize,
    /// `T_i_c` per camera, which places the points of a [`BackendWork::view`].
    t_i_c: Vec<crate::lie::Se3<f64>>,
    /// Loop closing (PR b2); `None` when it is off.
    loops: Option<LoopClosing>,
}

/// Wall time of back-end calls, as [`BackendRunner`] measured them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepTiming {
    /// Steps timed.
    pub steps: u64,
    /// Their total, in nanoseconds.
    pub total_ns: u64,
    /// The slowest one, in nanoseconds.
    pub max_ns: u64,
}

impl StepTiming {
    /// Add one step that took `ns`.
    pub fn record(&mut self, ns: u64) {
        self.steps += 1;
        self.total_ns += ns;
        self.max_ns = self.max_ns.max(ns);
    }
}

/// How many keyframes each culling rule has taken so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CullCounts {
    /// Redundant as they left the window.
    pub redundant: usize,
    /// Lowest-scored in an over-budget batch.
    pub over_budget: usize,
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
            culled_over_budget: 0,
            t_i_c: Vec::new(),
            loops: None,
        }
    }

    /// The rig's `T_i_c` per camera, so a view can place the map's points.
    #[must_use]
    pub fn with_extrinsics(mut self, t_i_c: Vec<crate::lie::Se3<f64>>) -> Self {
        self.t_i_c = t_i_c;
        self
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

    /// How many keyframes each culling rule has taken.
    pub fn cull_counts(&self) -> CullCounts {
        CullCounts {
            redundant: self.culled.len() - self.culled_over_budget,
            over_budget: self.culled_over_budget,
        }
    }

    /// Keyframes still in the window.
    pub fn active_keyframes(&self) -> &BTreeSet<FrameId> {
        &self.active
    }

    /// The shared-landmark count at which an edge is "high" (at least 1).
    pub fn high_covisibility_weight(&self) -> u32 {
        self.config.high_covisibility.max(1)
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
        if let (Some(loops), Some(images), true) = (
            self.loops.as_mut(),
            &stamp.images,
            added.contains(&stamp.t_ns),
        ) {
            loops.hold_images(stamp.t_ns, images.clone());
        }
        self.map.merge_landmarks(&stamp.landmarks);
        for id in added {
            let shared: BTreeMap<FrameId, u32> = self.shared_landmarks(id);
            self.graph.insert_keyframe(id, &shared);
        }
        if self.config.enable_culling && self.map.keyframes().len() > self.config.max_keyframes {
            for id in self.cull_over_budget() {
                events.push(MapEvent::KeyframeCulled {
                    id,
                    reason: CullReason::OverBudget,
                });
            }
        }
        events
    }

    /// `cullKeyframes`: cull up to `batch_cull_size` unprotected keyframes, the
    /// lowest score first (equal scores by id). As in RootSLAM
    /// (`handleGraphScoreChange`), every cull re-ranks the keyframes it touched
    /// before the next pick: graph neighbours, loop partners and the observers of
    /// the landmarks it took.
    fn cull_over_budget(&mut self) -> Vec<FrameId> {
        let high_weight: u32 = self.high_covisibility_weight();
        let score = |backend: &Self, id: FrameId| {
            cull::keyframe_score(
                &backend.map,
                &backend.graph,
                id,
                high_weight,
                &backend.resolutions,
            )
        };
        let mut scores: BTreeMap<FrameId, f32> = self
            .map
            .keyframes()
            .keys()
            .copied()
            .filter(|&id| !self.is_protected(id))
            .map(|id| (id, score(self, id)))
            .collect();
        let mut culled: Vec<FrameId> = Vec::new();
        while culled.len() < self.config.batch_cull_size {
            let Some(id) = scores
                .iter()
                .min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(b.0)))
                .map(|(&id, _)| id)
            else {
                break;
            };
            scores.remove(&id);
            let Some(touched) = self.cull_keyframe(id) else {
                continue;
            };
            culled.push(id);
            self.culled_over_budget += 1;
            for other in touched {
                if let Some(entry) = scores.get_mut(&other) {
                    *entry = score(self, other);
                }
            }
        }
        culled
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
                && self.cull_keyframe(id).is_some()
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
    /// Returns the keyframes whose score inputs changed: `id`'s graph neighbours
    /// and loop partners, and the observers of the landmarks it took. `None`,
    /// changing nothing, for a protected or unknown keyframe.
    fn cull_keyframe(&mut self, id: FrameId) -> Option<BTreeSet<FrameId>> {
        if self.is_protected(id) {
            return None;
        }
        let removed = self.map.remove_keyframe(id)?;
        let mut touched: BTreeSet<FrameId> = self
            .graph
            .neighbours(id)
            .map(|(other, _)| other)
            .chain(self.graph.loop_closures(id))
            .collect();
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
            touched.extend(observers);
            touched.insert(landmark.host.frame_id);
        }
        self.culled.insert(id);
        if let Some(loops) = self.loops.as_mut() {
            loops.forget(id);
        }
        Some(touched)
    }
}

impl BackendWork for MapBackend {
    type View = MapSnapshot;

    /// [`MapBackend::snapshot`] with the rig's extrinsics.
    fn view(&self) -> MapSnapshot {
        self.snapshot(&self.t_i_c)
    }

    fn step(&mut self, stamp: MapStamp) -> Vec<MapEvent> {
        MapBackend::step(self, stamp)
    }

    /// Loop detection and, below the drift gate, fusion (PR b2).
    fn query(&mut self, keyframe: FrameId) -> Vec<MapEvent> {
        self.close_loops(keyframe)
    }

    /// Free the images a dropped query would have read.
    fn drop_query(&mut self, keyframe: FrameId) {
        if let Some(loops) = self.loops.as_mut() {
            loops.drop_images(keyframe);
        }
    }
}

#[cfg(test)]
mod tests;
