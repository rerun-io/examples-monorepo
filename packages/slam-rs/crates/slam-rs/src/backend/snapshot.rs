//! A copy of the map for readers outside the back end (PR a2): the Python
//! bindings and the Rerun layer. Everything is copied, in id order, so a
//! snapshot stays valid while the back end goes on (D51).

use nalgebra::{Vector3, Vector4};

use super::pgo::{MapCorrection, PgoRun, SolveTimes};
use super::{CullCounts, LoopCounts, LoopOutcome, MapBackend, MapStats};
use crate::landmark::StereographicParam;
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
    /// Median number of landmarks a map keyframe sees.
    pub median_keyframe_landmarks: usize,
    /// The map's payload in bytes (a lower bound on its heap).
    pub retained_bytes: usize,
    /// What loop closing has done (all zero when it is off).
    pub loop_counts: LoopCounts,
    /// Verified loops whose two keyframes are still in the map, oldest first.
    pub loops: Vec<SnapshotLoop>,
    /// The PGO's corrections and epoch, for re-anchoring (PR f1).
    pub correction: MapCorrection,
    /// Applied PGOs, oldest first.
    pub pgo_runs: Vec<PgoRun>,
    /// PGOs the solver refused.
    pub pgo_failures: u64,
    /// Wall time of each applied PGO's solve.
    pub pgo_solve_ns: SolveTimes,
}

/// One verified loop in a [`MapSnapshot`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotLoop {
    /// The keyframe that found the loop.
    pub current: FrameId,
    /// The keyframe it closes with.
    pub candidate: FrameId,
    /// Final PnP inliers.
    pub inliers: usize,
    /// Fused at once (`false`: at or above the drift gate, the pose-graph
    /// optimization's).
    pub fused: bool,
}

impl MapBackend {
    /// Copy the map. A point is `T_w_i(host) * t_i_c[cam] * unproject(direction)
    /// / inv_dist`, as the window builds it. A landmark whose host keyframe or
    /// camera is unknown, or with no finite position (`inv_dist <= 0`, at
    /// infinity), is left out of the points (it still counts in `stats`), as
    /// RootSLAM's map export keeps only finite positions.
    pub fn snapshot(&self, t_i_c: &[Se3<f64>]) -> MapSnapshot {
        let keyframes = self.map.keyframes();
        let mut point_ids: Vec<LandmarkId> = Vec::with_capacity(self.map.landmarks().len());
        let mut point_hosts: Vec<FrameId> = Vec::with_capacity(self.map.landmarks().len());
        let mut points_world: Vec<[f32; 3]> = Vec::with_capacity(self.map.landmarks().len());
        for (&id, landmark) in self.map.landmarks() {
            let (Some(host), Some(extrinsic)) = (
                keyframes.get(&landmark.host.frame_id),
                t_i_c.get(landmark.host.cam_id),
            ) else {
                continue;
            };
            if landmark.inv_dist.is_nan() || landmark.inv_dist <= 0.0 {
                continue;
            }
            let bearing: Vector4<f64> = StereographicParam::unproject(&landmark.direction);
            let point_c: Vector3<f64> =
                Vector3::new(bearing[0], bearing[1], bearing[2]) / landmark.inv_dist;
            let point_w: Vector3<f64> = (host.pose * *extrinsic) * point_c;
            let point: [f32; 3] = [point_w.x as f32, point_w.y as f32, point_w.z as f32];
            if !point.iter().all(|value| value.is_finite()) {
                continue;
            }
            point_ids.push(id);
            point_hosts.push(landmark.host.frame_id);
            points_world.push(point);
        }
        let graph = self.graph();
        let pgo = self.loops.as_ref().map(|loops| loops.pgo());
        MapSnapshot {
            keyframe_t_ns: keyframes.keys().copied().collect(),
            keyframe_poses: keyframes.values().map(|keyframe| keyframe.pose).collect(),
            keyframe_raw_poses: keyframes
                .iter()
                .map(|(id, keyframe)| {
                    pgo.and_then(|pgo| pgo.raw.get(id).copied())
                        .unwrap_or(keyframe.pose)
                })
                .collect(),
            keyframe_active: keyframes
                .keys()
                .map(|id| self.active.contains(id))
                .collect(),
            point_ids,
            point_hosts,
            points_world,
            covisibility_edges: graph
                .node_ids()
                .flat_map(|a| {
                    graph
                        .neighbours(a)
                        .filter(move |&(b, _)| b > a)
                        .map(move |(b, weight)| (a, b, weight))
                })
                .collect(),
            stats: self.stats(),
            culls: self.cull_counts(),
            high_covisibility_weight: self.high_covisibility_weight(),
            median_keyframe_landmarks: self.map.median_keyframe_landmarks(),
            retained_bytes: self.map.retained_bytes(),
            loop_counts: self
                .loops
                .as_ref()
                .map(|loops| loops.counts())
                .unwrap_or_default(),
            loops: self
                .loops
                .iter()
                .flat_map(|loops| loops.events())
                .filter(|event| {
                    event.outcome != LoopOutcome::Stale
                        && keyframes.contains_key(&event.current)
                        && keyframes.contains_key(&event.candidate)
                })
                .map(|event| SnapshotLoop {
                    current: event.current,
                    candidate: event.candidate,
                    inliers: if event.stages.reprojection_matches > 0 {
                        event.stages.reprojection_inliers
                    } else {
                        event.stages.inliers
                    },
                    fused: matches!(event.outcome, LoopOutcome::Fused { .. }),
                })
                .collect(),
            correction: self.map_correction(),
            pgo_runs: pgo.map(|pgo| pgo.runs.clone()).unwrap_or_default(),
            pgo_failures: pgo.map_or(0, |pgo| pgo.failures),
            pgo_solve_ns: pgo.map(|pgo| pgo.solve_ns.clone()).unwrap_or_default(),
        }
    }
}
