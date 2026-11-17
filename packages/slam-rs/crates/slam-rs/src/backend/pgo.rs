//! The pose-graph optimization a large-drift loop runs (RootSLAM §III-B;
//! `loop_closing.cpp:725-902`, `map_database.cpp:164-243`; DESIGN §3 row 12,
//! §4 B11/B12, PR c2). The solver is `kornia_staging_slam::pgo` (c1).
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
//!    the first is kept: the loop's edges come first, so an island keyframe that
//!    already shares an edge with the current one keeps the loop's measurement
//!    (RootSLAM lists them last and drops them; parity keeps that order).
//! 2. **Gauge** (DESIGN §11 Q3): the window keyframes and the newest keyframe
//!    are fixed and stay bit-identical. The window never reads the map, so the
//!    raw VIO poses do not change; the old map moves onto the current drift.
//! 3. **Apply** ([`apply_delta`], B11): each moved keyframe takes the PGO's
//!    change, `T_new = T_pgo · T_copy⁻¹ · T_now`, so a pose the window wrote
//!    after the copy keeps that write. Landmarks are host-relative and move
//!    with their host keyframe.
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
    /// How many of `edges` are the current loop's.
    pub loop_edges: usize,
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
        let n_loop: usize = loop_edges.len();
        let edges: Vec<SolverEdge> = if config.pgo_loop_edges_first {
            loop_edges.into_iter().chain(map_edges).collect()
        } else {
            map_edges.into_iter().chain(loop_edges).collect()
        };
        Self {
            ids,
            copy,
            fixed,
            edges,
            loop_edges: n_loop,
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

/// Write a PGO result into the map as a change (B11): keyframe `i` takes
/// `T_pgo · T_copy⁻¹ · T_now`. RootSLAM writes `T_pgo` itself, which undoes any
/// window update made since the copy. A pose the PGO left bit-identical, or a
/// keyframe culled since the copy, is not touched; one added since the copy is
/// not in the graph and keeps its pose. Returns the moved keyframes.
pub fn apply_delta(
    map: &mut KeyframeMap,
    graph: &PoseGraph,
    optimized: &[Se3<f64>],
) -> Vec<FrameId> {
    let mut moved: Vec<FrameId> = Vec::new();
    for ((&id, copy), pgo) in graph.ids.iter().zip(&graph.copy).zip(optimized) {
        if pgo == copy {
            continue;
        }
        let Some(now) = map.keyframe(id).map(|kf| kf.pose) else {
            continue;
        };
        let new = if now == *copy {
            *pgo
        } else {
            *pgo * copy.inverse() * now
        };
        map.upsert_keyframe(id, new, None);
        moved.push(id);
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
    /// Keyframes in the graph.
    pub keyframes: usize,
    /// Of those, fixed (the gauge).
    pub fixed: usize,
    /// Edges given to the solver.
    pub edges: usize,
    /// Of those, the current loop's.
    pub loop_edges: usize,
    /// Parallel edges the solver dropped.
    pub dropped_edges: usize,
    /// LM steps.
    pub iterations: usize,
    /// Cost before and after.
    pub cost: (f64, f64),
    /// Keyframes the result moved.
    pub moved: usize,
}

/// Wall time of each PGO solve, in nanoseconds. It changes run to run, so it
/// never takes part in equality or `Debug` (`Vio`'s fingerprint, and the
/// Offline-against-Realtime snapshot comparison).
#[derive(Clone, Default)]
pub struct SolveTimes(pub Vec<u64>);

impl PartialEq for SolveTimes {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for SolveTimes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SolveTimes(..)")
    }
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
    /// Wall time of each solve.
    pub solve_ns: SolveTimes,
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
        let started = std::time::Instant::now();
        let solved = graph.solve();
        let solve_ns: u64 = crate::duration_ns(started);
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
        let moved: Vec<FrameId> = apply_delta(&mut self.map, &graph, &poses);
        loops.pgo.runs.push(PgoRun {
            current: detection.current,
            candidate: detection.candidate,
            keyframes: graph.ids.len(),
            fixed: graph.fixed.iter().filter(|&&fixed| fixed).count(),
            edges: graph.edges.len(),
            loop_edges: graph.loop_edges,
            dropped_edges: report.dropped_edges,
            iterations: report.iterations,
            cost: (report.initial_cost, report.final_cost),
            moved: moved.len(),
        });
        loops.pgo.solve_ns.0.push(solve_ns);
        true
    }

    /// The corrections the PGO left on the map, with its epoch.
    pub fn map_correction(&self) -> MapCorrection {
        let Some(loops) = self.loops.as_ref() else {
            return MapCorrection::default();
        };
        MapCorrection {
            epoch: loops.pgo.epoch(),
            poses: loops
                .pgo
                .raw
                .keys()
                .filter_map(|&id| Some((id, self.map.keyframe(id)?.pose)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests;
