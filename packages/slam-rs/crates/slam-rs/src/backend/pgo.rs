//! The pose-graph optimization a large-drift loop runs (RootSLAM §III-B;
//! `loop_closing.cpp:725-902`, `map_database.cpp:164-243`; DESIGN §3 row 12,
//! §4 B11/B12/B29, PR c2). The solver is `kornia_staging_slam::pgo` (c1).
//!
//! When a verified loop would move the current keyframe by at least the drift
//! gate, [`MapBackend::apply_loop`] runs one PGO before it fuses:
//!
//! 1. **Graph** ([`PoseGraph::build`]): every map keyframe; the covisibility
//!    graph's pose-graph edges (spanning tree, past loops, edges at or above
//!    `pgo_min_covisibility_weight`), each measured from the map poses as they
//!    are now; and one edge from each island keyframe to the current one,
//!    measured with the PnP pose. Each edge weighs `pgo_edge_sigmas` (B12;
//!    RootSLAM's I6 under `rootslam-parity`). Of several edges between one pair
//!    the first is kept. The loop's edges come first, so an island keyframe
//!    that already shares an edge with the current one keeps the loop's
//!    measurement (B29, also under parity: RootSLAM lists them last and drops
//!    them).
//! 2. **Gauge** (DESIGN §11 Q3): the window keyframes and the newest keyframe
//!    are fixed and stay bit-identical. The window never reads the map, so the
//!    raw VIO poses do not change; the old map moves onto the current drift.
//! 3. **Apply** ([`apply`]): each moved keyframe takes its optimized pose.
//!    The PGO runs inside the back-end step (Offline and Realtime), so no
//!    keyframe changes between the copy and the apply and B11 cannot occur.
//!    Landmarks are host-relative and move with their host keyframe.
//!
//! Every applied PGO raises the epoch of the [`MapCorrection`] that the
//! snapshot publishes for re-anchoring (PR f1).

use std::collections::{BTreeMap, BTreeSet};

use kornia_staging_slam::pgo::{
    EdgeSigmas, PgoConfig, PgoError, PgoReport, PoseGraphEdge as SolverEdge, optimize_pose_graph,
};

use super::MapBackend;
use super::covisibility::CovisibilityGraph;
use super::loop_closing::LoopDetection;
use super::map::KeyframeMap;
use crate::config::LoopClosingConfig;
use crate::lie::Se3;
use crate::types::FrameId;

/// The pose graph of one loop, measured from a copy of the map poses.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraph {
    /// Map keyframes, in id order; index `i` is solver pose `i`.
    pub ids: Vec<FrameId>,
    /// `T_w_i` of each keyframe when the graph was built (`T_copy`).
    pub copy: Vec<Se3<f64>>,
    /// The gauge: window keyframes and the newest one.
    pub fixed: Vec<bool>,
    /// Edges in priority order (the first of a pair is kept).
    pub edges: Vec<SolverEdge>,
}

impl PoseGraph {
    /// The graph RootSLAM's `buildPoseGraph` builds for a loop between
    /// `current` (at `corrected`, from PnP) and `island`.
    pub fn build(
        map: &KeyframeMap,
        graph: &CovisibilityGraph,
        high_covisibility: u32,
        active: &BTreeSet<FrameId>,
        loop_closure: (FrameId, &[FrameId], &Se3<f64>),
        config: &LoopClosingConfig,
    ) -> Self {
        let (current, island, corrected) = loop_closure;
        let ids: Vec<FrameId> = map.keyframes().keys().copied().collect();
        let copy: Vec<Se3<f64>> = map.keyframes().values().map(|kf| kf.pose).collect();
        let index: BTreeMap<FrameId, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let newest: Option<FrameId> = ids.last().copied();
        let fixed: Vec<bool> = ids
            .iter()
            .map(|id| active.contains(id) || Some(*id) == newest)
            .collect();
        let [sigma_t, sigma_r] = config.pgo_edge_sigmas;
        let sigmas = EdgeSigmas {
            translation_m: sigma_t,
            rotation_rad: sigma_r,
        };
        let edge = |a: usize, b: usize, b_pose: &Se3<f64>| {
            SolverEdge::new(a, b, copy[a].inverse() * *b_pose, &sigmas)
        };
        let map_edges: Vec<SolverEdge> = graph
            .pose_graph_edges(high_covisibility)
            .into_iter()
            .filter_map(|e| Some((*index.get(&e.from)?, *index.get(&e.to)?)))
            .map(|(a, b)| edge(a, b, &copy[b]))
            .collect();
        let loop_edges: Vec<SolverEdge> = match index.get(&current) {
            Some(&c) => island
                .iter()
                .filter_map(|id| index.get(id))
                .filter(|&&i| i != c)
                .map(|&i| edge(i, c, corrected))
                .collect(),
            None => Vec::new(),
        };
        Self {
            ids,
            copy,
            fixed,
            edges: loop_edges.into_iter().chain(map_edges).collect(),
        }
    }

    /// Solve with RootSLAM's limits (LM, at most 20 iterations); the optimized
    /// poses, index for index with [`Self::ids`].
    ///
    /// # Errors
    ///
    /// The solver's refusal of a malformed graph.
    pub fn solve(&self) -> Result<(Vec<Se3<f64>>, PgoReport), PgoError> {
        let mut poses: Vec<Se3<f64>> = self.copy.clone();
        let report =
            optimize_pose_graph(&mut poses, &self.fixed, &self.edges, &PgoConfig::default())?;
        Ok((poses, report))
    }
}

/// Write a PGO result into the map: each keyframe the PGO moved takes its
/// optimized pose. Returns the moved keyframes.
pub fn apply(map: &mut KeyframeMap, graph: &PoseGraph, optimized: &[Se3<f64>]) -> Vec<FrameId> {
    let mut moved: Vec<FrameId> = Vec::new();
    for ((&id, copy), pgo) in graph.ids.iter().zip(&graph.copy).zip(optimized) {
        if pgo != copy {
            map.upsert_keyframe(id, *pgo, None);
            moved.push(id);
        }
    }
    moved
}

/// One applied PGO.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PgoRun {
    /// The loop's current keyframe.
    pub current: FrameId,
    /// The keyframe it closed with.
    pub candidate: FrameId,
    /// Keyframes the result moved.
    pub moved: usize,
    /// Parallel edges the solver dropped: a map edge between an island
    /// keyframe and the current one (B29).
    pub dropped_edges: usize,
    /// LM steps.
    pub iterations: usize,
}

/// What the PGO has done: the runs, the corrections, the epoch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PgoState {
    /// Applied PGOs, oldest first.
    pub runs: Vec<PgoRun>,
    /// Loops at or above the drift gate whose PGO the solver refused.
    pub failures: u64,
    /// The pose each moved keyframe had before its first correction (the
    /// window's final pose); culled keyframes leave.
    pub raw: BTreeMap<FrameId, Se3<f64>>,
}

impl PgoState {
    /// Applied PGOs so far; the [`MapCorrection`] epoch.
    pub fn epoch(&self) -> u64 {
        self.runs.len() as u64
    }
}

/// The corrections every applied PGO left on the map, for re-anchoring (PR f1).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MapCorrection {
    /// Applied PGOs so far; a reader re-anchors when it changes.
    pub epoch: u64,
    /// Corrected `T_w_i` of each map keyframe a PGO moved. Every other map
    /// keyframe keeps its raw pose.
    pub poses: BTreeMap<FrameId, Se3<f64>>,
    /// The pose each of those keyframes had before its first correction (the
    /// window's), so a reader can move a frame by `T_corr · T_raw⁻¹` (PR f1).
    pub raw: BTreeMap<FrameId, Se3<f64>>,
}

impl MapBackend {
    /// Build, solve and apply the PGO of a verified loop at or above the drift
    /// gate. `false`, moving nothing, when loop closing or its PGO is off or the
    /// solver refuses the graph.
    pub(super) fn optimize_loop(&mut self, detection: &LoopDetection) -> bool {
        let Some(loops) = self.loops.as_ref().filter(|loops| loops.config().pgo) else {
            return false;
        };
        let graph = PoseGraph::build(
            &self.map,
            &self.graph,
            self.high_covisibility_weight(),
            &self.active,
            (
                detection.current,
                &detection.island,
                &detection.corrected_pose,
            ),
            loops.config(),
        );
        let solved = graph.solve();
        let Some(loops) = self.loops.as_mut() else {
            return false;
        };
        let (poses, report) = match solved {
            Ok(solved) => solved,
            Err(error) => {
                log::warn!(
                    "pose-graph optimization of loop {} -> {} refused: {error}",
                    detection.current,
                    detection.candidate
                );
                loops.pgo.failures += 1;
                return false;
            }
        };
        for ((&id, copy), pose) in graph.ids.iter().zip(&graph.copy).zip(&poses) {
            if pose != copy {
                loops.pgo.raw.entry(id).or_insert(*copy);
            }
        }
        let moved: Vec<FrameId> = apply(&mut self.map, &graph, &poses);
        loops.pgo.runs.push(PgoRun {
            current: detection.current,
            candidate: detection.candidate,
            moved: moved.len(),
            dropped_edges: report.dropped_edges,
            iterations: report.iterations,
        });
        self.corrections.publish(self.map_correction());
        true
    }

    /// The corrections the PGO left on the map, with its epoch.
    pub fn map_correction(&self) -> MapCorrection {
        let Some(loops) = self.loops.as_ref() else {
            return MapCorrection::default();
        };
        let moved = loops
            .pgo
            .raw
            .iter()
            .filter_map(|(&id, &raw)| Some((id, raw, self.map.keyframe(id)?.pose)));
        MapCorrection {
            epoch: loops.pgo.epoch(),
            poses: moved.clone().map(|(id, _, pose)| (id, pose)).collect(),
            raw: moved.map(|(id, raw, _)| (id, raw)).collect(),
        }
    }
}

#[cfg(test)]
mod tests;
