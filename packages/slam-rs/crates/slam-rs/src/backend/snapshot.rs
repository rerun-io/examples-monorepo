//! A copy of the map for readers outside the back end (PR a2): the Python
//! bindings and the Rerun layer. Everything is copied, in id order, so a
//! snapshot stays valid while the back end goes on (D51).

use super::pgo::PgoRun;
use super::{
    CullCounts, KeyframeMap, LoopCounts, LoopOutcome, MapBackend, MapStats, PoseGraphEdge,
};
use crate::lie::Se3;
use crate::types::{FrameId, LandmarkId};

/// The map at one moment.
#[derive(Debug, Clone, PartialEq)]
pub struct MapSnapshot {
    /// Map keyframes, oldest first.
    pub keyframe_t_ns: Vec<FrameId>,
    /// `T_w_i` of each keyframe as the map holds it: the window's solved (or,
    /// once evicted, final) pose, as the pose-graph optimization corrected it.
    pub keyframe_poses: Vec<Se3<f64>>,
    /// The window's pose of each keyframe, before any correction.
    pub keyframe_raw_poses: Vec<Se3<f64>>,
    /// Whether each keyframe is still in the window.
    pub keyframe_active: Vec<bool>,
    /// Map landmarks the snapshot could place, in id order.
    pub point_ids: Vec<LandmarkId>,
    /// The keyframe hosting each point.
    pub point_hosts: Vec<FrameId>,
    /// Each point in the world frame, metres.
    pub points_world: Vec<[f32; 3]>,
    /// Covisibility edges `(smaller id, larger id, shared landmarks)`, each once.
    pub covisibility_edges: Vec<(FrameId, FrameId, u32)>,
    /// RootSLAM's `stats_map.json` counts (all landmarks, placed or not).
    pub stats: MapStats,
    /// Keyframes each culling rule took.
    pub culls: CullCounts,
    /// The shared-landmark weight at which an edge is "high".
    pub high_covisibility_weight: u32,
    /// The edges a pose-graph optimization would take now (the spanning tree,
    /// the loop closures, the high-covisibility edges; PR f1 draws them).
    pub pose_graph_edges: Vec<PoseGraphEdge>,
    /// Median number of landmarks a map keyframe sees.
    pub median_keyframe_landmarks: usize,
    /// The map's payload in bytes (a lower bound on its heap).
    pub retained_bytes: usize,
    /// What loop closing has done (all zero when it is off).
    pub loop_counts: LoopCounts,
    /// Verified loops whose two keyframes are still in the map, oldest first.
    pub loops: Vec<SnapshotLoop>,
    /// Applied PGOs so far, for re-anchoring (PR f1): a reader re-anchors
    /// when it changes.
    pub pgo_epoch: u64,
    /// Applied PGOs, oldest first.
    pub pgo_runs: Vec<PgoRun>,
}

/// The keyframe half of the map, which stays small: what a [`MapDelta`]
/// carries whole and a [`MapSnapshot`] holds in its own fields.
#[derive(Debug, Clone, PartialEq)]
pub struct MapKeyframes {
    /// Map keyframes, oldest first.
    pub t_ns: Vec<FrameId>,
    /// `T_w_i` of each keyframe as the map holds it ([`MapSnapshot::keyframe_poses`]).
    pub poses: Vec<Se3<f64>>,
    /// The window's pose of each keyframe, before any correction.
    pub raw_poses: Vec<Se3<f64>>,
    /// Whether each keyframe is still in the window.
    pub active: Vec<bool>,
    /// Verified loops whose two keyframes are still in the map, oldest first.
    pub loops: Vec<SnapshotLoop>,
}

/// What a reader needs to redraw the map since its last read (D51's copy, minus
/// what did not change): the keyframe half, and the points of only the
/// keyframes a write moved after `since`. Reading it costs those keyframes'
/// hosted points, not the whole map's.
#[derive(Debug, Clone, PartialEq)]
pub struct MapDelta {
    /// The map's revision when the delta was taken: the reader's next `since`.
    pub revision: u64,
    /// Every keyframe's pose and the loops.
    pub keyframes: MapKeyframes,
    /// Keyframes in the map whose hosted points moved after `since`, in id
    /// order. A keyframe the reader holds that is missing from
    /// [`MapKeyframes::t_ns`] was removed.
    pub changed_hosts: Vec<FrameId>,
    /// How many of the points below each changed host has, aligned with
    /// `changed_hosts` (0: it hosts none the map can place).
    pub point_counts: Vec<usize>,
    /// The changed hosts' points, host by host in `changed_hosts` order, then
    /// by camera and id.
    pub point_ids: Vec<LandmarkId>,
    /// Each point in the world frame, metres.
    pub points_world: Vec<[f32; 3]>,
}

/// One verified loop in a [`MapSnapshot`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotLoop {
    /// The keyframe that found the loop.
    pub current: FrameId,
    /// The keyframe it closes with.
    pub candidate: FrameId,
    /// Its landmarks were fused, at once or after a pose-graph optimization
    /// (`false`: left unapplied, the optimization off or refused).
    pub fused: bool,
    /// Translation of `T_map^-1 T_pnp` when it was verified, metres
    /// ([`LoopEvent::drift_m`](super::loop_closing::LoopEvent::drift_m)).
    pub drift_m: f64,
    /// Its rotation angle, degrees.
    pub drift_deg: f64,
}

/// A world point as `f32`, or `None` when it overflows (kept out of the
/// points, as RootSLAM's export keeps only finite positions).
fn narrow(point_w: nalgebra::Vector3<f64>) -> Option<[f32; 3]> {
    let point: [f32; 3] = [point_w.x as f32, point_w.y as f32, point_w.z as f32];
    point.iter().all(|value| value.is_finite()).then_some(point)
}

impl MapBackend {
    /// Copy the map. A point is [`super::KeyframeMap::landmark_world`]; a
    /// landmark it cannot place, or whose position overflows `f32`, is left out
    /// of the points (it still counts in `stats`), as RootSLAM's map export
    /// keeps only finite positions.
    pub fn snapshot(&self) -> MapSnapshot {
        let mut point_ids: Vec<LandmarkId> = Vec::with_capacity(self.map.landmarks().len());
        let mut point_hosts: Vec<FrameId> = Vec::with_capacity(self.map.landmarks().len());
        let mut points_world: Vec<[f32; 3]> = Vec::with_capacity(self.map.landmarks().len());
        for (&id, landmark) in self.map.landmarks() {
            let Some(point) = self.placed_point(id) else {
                continue;
            };
            point_ids.push(id);
            point_hosts.push(landmark.host.frame_id);
            points_world.push(point);
        }
        let graph = self.graph();
        let pgo = self.loops.as_ref().map(|loops| loops.pgo());
        let MapKeyframes {
            t_ns,
            poses,
            raw_poses,
            active,
            loops,
        } = self.map_keyframes();
        MapSnapshot {
            keyframe_t_ns: t_ns,
            keyframe_poses: poses,
            keyframe_raw_poses: raw_poses,
            keyframe_active: active,
            point_ids,
            point_hosts,
            points_world,
            covisibility_edges: graph.edges().collect(),
            stats: self.stats(),
            culls: self.cull_counts(),
            high_covisibility_weight: self.high_covisibility_weight(),
            pose_graph_edges: graph.pose_graph_edges(self.high_covisibility_weight()),
            median_keyframe_landmarks: self.map.median_keyframe_landmarks(),
            retained_bytes: self.map.retained_bytes(),
            loop_counts: self
                .loops
                .as_ref()
                .map(|loops| loops.counts())
                .unwrap_or_default(),
            loops,
            pgo_epoch: pgo.map_or(0, |pgo| pgo.epoch()),
            pgo_runs: pgo.map(|pgo| pgo.runs.clone()).unwrap_or_default(),
        }
    }

    /// What changed since revision `since` ([`MapDelta`]): the keyframe half,
    /// and the points of the keyframes a write moved after it. `since = 0`
    /// gives every point, as a first read.
    pub fn delta(&self, since: u64) -> MapDelta {
        let changed_hosts: Vec<FrameId> = self.map.hosts_changed_since(since);
        let mut point_counts: Vec<usize> = Vec::with_capacity(changed_hosts.len());
        let mut point_ids: Vec<LandmarkId> = Vec::new();
        let mut points_world: Vec<[f32; 3]> = Vec::new();
        for &host in &changed_hosts {
            let before: usize = point_ids.len();
            if let Some(keyframe) = self.map.keyframe(host) {
                for (id, landmark) in self.map.hosted_landmarks(host) {
                    if let Some(point) =
                        KeyframeMap::place(landmark, keyframe, &self.rig.t_i_c).and_then(narrow)
                    {
                        point_ids.push(id);
                        points_world.push(point);
                    }
                }
            }
            point_counts.push(point_ids.len() - before);
        }
        MapDelta {
            revision: self.map.revision(),
            keyframes: self.map_keyframes(),
            changed_hosts,
            point_counts,
            point_ids,
            points_world,
        }
    }

    /// Every keyframe's pose, raw pose and window state, and the loops.
    pub fn map_keyframes(&self) -> MapKeyframes {
        let keyframes = self.map.keyframes();
        MapKeyframes {
            t_ns: keyframes.keys().copied().collect(),
            poses: keyframes.values().map(|keyframe| keyframe.pose).collect(),
            raw_poses: self.raw_poses(),
            active: keyframes
                .keys()
                .map(|id| self.active.contains(id))
                .collect(),
            loops: self.snapshot_loops(),
        }
    }

    /// A landmark in the world frame as `f32`, or `None` when the map cannot
    /// place it or it overflows `f32` (kept out of the points, as RootSLAM's
    /// export keeps only finite positions).
    fn placed_point(&self, id: LandmarkId) -> Option<[f32; 3]> {
        narrow(self.map.landmark_world(id, &self.rig.t_i_c)?)
    }

    /// The window's pose of each keyframe before any PGO correction.
    fn raw_poses(&self) -> Vec<Se3<f64>> {
        let pgo = self.loops.as_ref().map(|loops| loops.pgo());
        self.map
            .keyframes()
            .iter()
            .map(|(id, keyframe)| {
                pgo.and_then(|pgo| pgo.raw.get(id).copied())
                    .unwrap_or(keyframe.pose)
            })
            .collect()
    }

    /// Verified loops whose two keyframes are still in the map, oldest first.
    fn snapshot_loops(&self) -> Vec<SnapshotLoop> {
        let keyframes = self.map.keyframes();
        self.loops
            .iter()
            .flat_map(|loops| loops.events())
            .filter(|event| {
                keyframes.contains_key(&event.detection.current)
                    && keyframes.contains_key(&event.detection.candidate)
            })
            .map(|event| SnapshotLoop {
                current: event.detection.current,
                candidate: event.detection.candidate,
                fused: matches!(
                    event.outcome,
                    LoopOutcome::Fused { .. } | LoopOutcome::Optimized { .. }
                ),
                drift_m: event.drift_m,
                drift_deg: event.drift_deg,
            })
            .collect()
    }
}
