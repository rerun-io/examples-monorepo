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

pub mod anchor;
pub mod covisibility;
pub mod cull;
pub mod gba;
pub mod loop_closing;
pub mod map;
pub mod pgo;
pub mod runner;
pub mod snapshot;

use std::collections::{BTreeMap, BTreeSet};

use crate::calib::{BasaltCamera, CalibError, Calibration};
use crate::config::VioConfig;
use crate::estimator::{KeyframeStamp, MapStamp, MargStamp};
use crate::lie::Se3;
use crate::types::{FrameId, LandmarkId};

pub use anchor::{AnchorError, reanchor, reanchor_arrays};
pub use covisibility::{CovisibilityGraph, PoseGraphEdge, PoseGraphEdgeKind};
pub use gba::{GbaDriverError, GbaRig, GbaRun, MapResult};
pub use loop_closing::{LoopClosing, LoopCounts, LoopDetection, LoopEvent, LoopOutcome};
pub use map::{KeyframeMap, MapKeyframe, MapLandmark, MapStats};
pub use pgo::{PgoRun, PgoState};
pub use runner::{BackendCounters, BackendRunner, BackendView, BackendWork};
pub use snapshot::{MapDelta, MapSnapshot, SnapshotLoop};

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

/// The rig the back end works in, one entry per camera, built once from the
/// calibration with its lists checked to the same length. Culling scores
/// coverage with its image sizes, the snapshot, loop closing and the global
/// BA place landmarks with its extrinsics, and loop closing and the global BA
/// derive their camera models from its models.
#[derive(Debug, Clone)]
pub struct BackendRig {
    /// Projection model per camera.
    pub(crate) intrinsics: Vec<BasaltCamera<f64>>,
    /// `T_i_c` per camera, its rotation renormalized (`Se3::cast`): the map has
    /// always used the normalized extrinsics; the GBA's cameras are cast to f32 anyway.
    pub(crate) t_i_c: Vec<Se3<f64>>,
    /// `[width, height]` per camera, in pixels.
    pub(crate) resolutions: Vec<[u32; 2]>,
}

impl BackendRig {
    /// The rig of `calibration`.
    ///
    /// # Errors
    ///
    /// [`CalibError::RaggedCameraLists`] when the camera lists differ in length.
    pub fn new(calibration: &Calibration<f64>) -> Result<Self, CalibError> {
        let (extrinsics, intrinsics, resolutions) = (
            calibration.t_i_c.len(),
            calibration.intrinsics.len(),
            calibration.resolution.len(),
        );
        if intrinsics != extrinsics || resolutions != extrinsics {
            return Err(CalibError::RaggedCameraLists {
                extrinsics,
                intrinsics,
                resolutions,
            });
        }
        Ok(Self {
            intrinsics: calibration.intrinsics.clone(),
            t_i_c: calibration
                .t_i_c
                .iter()
                .map(|t_i_c| t_i_c.cast::<f64>())
                .collect(),
            resolutions: calibration.resolution.clone(),
        })
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
        /// Fused, optimized then fused, or left unapplied.
        outcome: LoopOutcome,
        /// Wall time of the apply, when it ran a pose-graph optimization
        /// (graph, solve, map write and fusion; PR c2). It differs run to run.
        pgo_ns: Option<u64>,
    },
}

/// The map, its covisibility graph and what protects keyframes from culling.
#[derive(Debug, Clone)]
pub struct MapBackend {
    config: MapConfig,
    rig: BackendRig,
    map: KeyframeMap,
    graph: CovisibilityGraph,
    /// Keyframes still in the window.
    active: BTreeSet<FrameId>,
    /// Keyframes the map culled; a stamp never brings one back.
    culled: BTreeSet<FrameId>,
    /// How many of `culled` the over-budget batches took.
    culled_over_budget: usize,
    /// Loop closing (PR b2); `None` when it is off.
    loops: Option<LoopClosing>,
    /// The rig's camera models for the global BA ([`Self::finish`]); `None`
    /// when the rig has a model the GBA cannot project.
    gba_rig: Option<gba::GbaRig>,
    /// Where the global BA's PCG runs (`select_backend`'s GBA lane).
    gba_lane: crate::Backend,
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
    /// An empty map for `rig` with the map knobs of `config`, loop closing as
    /// `config` says and the global BA's PCG on `gba_lane`. A camera model loop
    /// closing or the global BA cannot use leaves that part off, with one
    /// warning: the map still runs.
    pub fn new(config: &VioConfig, rig: BackendRig, gba_lane: crate::Backend) -> Self {
        let loops: Option<LoopClosing> =
            LoopClosing::new(&config.backend_loop_closing, &rig.intrinsics).unwrap_or_else(
                |error| {
                    log::warn!("loop closing off: {error}");
                    None
                },
            );
        let gba_rig: Option<gba::GbaRig> = gba::GbaRig::of(&rig);
        Self {
            config: MapConfig::from(config),
            rig,
            map: KeyframeMap::new(),
            graph: CovisibilityGraph::new(),
            active: BTreeSet::new(),
            culled: BTreeSet::new(),
            culled_over_budget: 0,
            loops,
            gba_rig,
            gba_lane,
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

    /// Whether culling must leave `id` alone: the root or a window keyframe.
    pub fn is_protected(&self, id: FrameId) -> bool {
        self.graph.root() == Some(id) || self.active.contains(&id)
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
        if let (Some(loops), Some(slot), true) = (
            self.loops.as_mut(),
            &stamp.images,
            added.contains(&stamp.t_ns),
        ) {
            loops.hold_images(stamp.t_ns, slot);
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
                &backend.rig.resolutions,
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
            // Each observing keyframe once, and the host if it observes none.
            let host: FrameId = landmark.host.frame_id;
            let host_alone: Option<FrameId> =
                (!landmark.frames().any(|f| f == host)).then_some(host);
            for other in landmark.frames().chain(host_alone) {
                if other != id && self.graph.contains(other) {
                    *shared.entry(other).or_insert(0) += 1;
                }
            }
        }
        shared
    }

    /// `cullKeyframe`: remove `id` from the map and the graph; each landmark
    /// the removal took lowers the edge between every two of its other
    /// observers by one. Returns the keyframes whose score inputs changed:
    /// `id`'s graph neighbours and loop partners, and the observers and hosts
    /// of the landmarks it took. `None`, changing nothing, for a protected or
    /// unknown keyframe.
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
        let mut change = EdgeChange::default();
        for (_, landmark) in &removed {
            let observers: BTreeSet<FrameId> = landmark.frames().filter(|&f| f != id).collect();
            change.pairs(&observers, -1);
            touched.extend(observers);
            touched.insert(landmark.host.frame_id);
        }
        change.apply(&mut self.graph);
        self.culled.insert(id);
        if let Some(loops) = self.loops.as_mut() {
            loops.forget(id);
        }
        Some(touched)
    }

    /// Carry out loop fusion's landmark edits (`merge_loop_landmarks`) and
    /// raise the covisibility edges they create: an added observation links
    /// `current` to every observer of the landmark (once per landmark); a
    /// fusion links every observer of one landmark to every observer of the
    /// other. Returns (landmarks fused, observations added).
    ///
    /// Deliberate difference from RootSLAM: `merge_loop_landmarks`
    /// (`map_database.cpp:259-336`) does every added observation before any
    /// fusion; this does the plan's steps in match order. Both orders were
    /// measured (2026-11-17): RootSLAM's made MGO13's final trajectory worse
    /// (5.51 cm against 4.76) and left EMH01 and TC3 the same, so the
    /// interleaved order stays. Do not "fix" it without a new measurement.
    fn fuse_landmarks(
        &mut self,
        current: FrameId,
        plan: Vec<loop_closing::LoopFusion>,
    ) -> (usize, usize) {
        let observers = |map: &KeyframeMap, id: LandmarkId| -> BTreeSet<FrameId> {
            map.landmark(id)
                .map(|landmark| landmark.frames().collect())
                .unwrap_or_default()
        };
        let mut change = EdgeChange::default();
        let (mut landmarks_fused, mut observations_added) = (0, 0);
        let mut linked: BTreeSet<LandmarkId> = BTreeSet::new();
        for step in plan {
            match step {
                loop_closing::LoopFusion::AddObs {
                    landmark,
                    image,
                    pixel,
                } => {
                    if linked.insert(landmark) {
                        change.across(&[current].into(), &observers(&self.map, landmark), 1);
                    }
                    observations_added +=
                        usize::from(self.map.add_observation(landmark, image, pixel));
                }
                loop_closing::LoopFusion::Fuse { from, into } => {
                    let (old, new) = (observers(&self.map, from), observers(&self.map, into));
                    if self.map.merge_landmark(from, into) {
                        change.across(&old, &new, 1);
                        landmarks_fused += 1;
                    }
                }
            }
        }
        change.apply(&mut self.graph);
        (landmarks_fused, observations_added)
    }
}

/// Covisibility edge changes, summed per keyframe pair and applied once.
#[derive(Default)]
struct EdgeChange(BTreeMap<(FrameId, FrameId), i64>);

impl EdgeChange {
    /// Add `by` to the edge between every two of `keyframes`.
    fn pairs(&mut self, keyframes: &BTreeSet<FrameId>, by: i64) {
        for &a in keyframes {
            for &b in keyframes.range(a + 1..) {
                *self.0.entry((a, b)).or_insert(0) += by;
            }
        }
    }

    /// Add `by` to the edge of `(a, b)` for every `a` in `left` and `b` in
    /// `right`, `a != b`.
    fn across(&mut self, left: &BTreeSet<FrameId>, right: &BTreeSet<FrameId>, by: i64) {
        for &a in left {
            for &b in right.iter().filter(|&&b| b != a) {
                *self.0.entry((a.min(b), a.max(b))).or_insert(0) += by;
            }
        }
    }

    fn apply(self, graph: &mut CovisibilityGraph) {
        for ((a, b), by) in self.0 {
            let size: u32 = u32::try_from(by.unsigned_abs()).unwrap_or(u32::MAX);
            if by > 0 {
                graph.increment_edge(a, b, size);
            } else if by < 0 {
                graph.decrement_edge(a, b, size);
            }
        }
    }
}

impl BackendWork for MapBackend {
    type View = MapSnapshot;

    /// [`MapBackend::snapshot`] with the rig's extrinsics.
    fn view(&self) -> MapSnapshot {
        self.snapshot()
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
