//! `Vio.map_snapshot`'s result: the persistent keyframe map and the back
//! end's counters (PR a2). It holds the core's copy (D51) and flattens each
//! field when a getter reads it.

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use slam_rs::backend::{BackendCounters, MapDelta as CoreMapDelta, MapSnapshot as CoreMapSnapshot, PoseGraphEdgeKind};
use slam_rs::types::LandmarkId;
use slam_rs::config::BackendMode;

use crate::flat_poses;

/// The persistent keyframe map and the back end's counters, copied out (D51).
///
/// Arrays are flat copies in id order; scalars are counts. The map's poses are
/// the window's solved (or, once evicted, final) poses as the pose-graph
/// optimization corrected them; the raw poses are the window's.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
#[derive(Debug)]
pub struct MapSnapshot {
    inner: Arc<CoreMapSnapshot>,
    counters: BackendCounters,
    mode: BackendMode,
}

impl MapSnapshot {
    pub fn new(inner: Arc<CoreMapSnapshot>, counters: BackendCounters, mode: BackendMode) -> Self {
        Self {
            inner,
            counters,
            mode,
        }
    }
}

/// What loop closing has done so far (all zero when it is off).
#[pyclass(module = "slam_rs._core", frozen, get_all, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct LoopCounts {
    /// Keyframes described and added to the database.
    indexed: u64,
    /// Queries that found no images for their keyframe.
    without_images: u64,
    /// Keyframe stamps whose images were dropped before the back end took
    /// them (Realtime, more stamps queued than the query cap).
    images_stripped: u64,
    /// Queries that retrieved at least one candidate.
    with_candidates: u64,
    /// Candidates validated.
    candidates: u64,
    /// GP3P RANSAC calls that refused their input (a bug upstream of it).
    pnp_errors: u64,
    /// Loops that passed verification.
    verified: u64,
    /// Loops fused at once.
    fused: u64,
    /// Loops at or above the drift gate: the pose-graph optimization's triggers.
    needs_pgo: u64,
    /// Of those, closed by an applied pose-graph optimization.
    optimized: u64,
}

#[pymethods]
impl MapSnapshot {
    /// Where the back end runs: ``"offline"`` or ``"realtime"``.
    #[getter]
    fn mode(&self) -> &'static str {
        crate::backend_mode_name(self.mode)
    }

    /// Map keyframe timestamps, oldest first: `int64[k]`.
    #[getter]
    fn keyframe_t_ns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.keyframe_t_ns.to_pyarray(py)
    }

    /// Keyframe poses `T_w_i` as the map holds them (PGO-corrected),
    /// `[tx, ty, tz, qx, qy, qz, qw]`: `float64[k, 7]`.
    #[getter]
    fn keyframe_poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        flat_poses(&self.inner.keyframe_poses)
            .to_pyarray(py)
            .reshape((self.inner.keyframe_poses.len(), 7))
    }

    /// The window's keyframe poses, before any correction: `float64[k, 7]`.
    #[getter]
    fn keyframe_raw_poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        flat_poses(&self.inner.keyframe_raw_poses)
            .to_pyarray(py)
            .reshape((self.inner.keyframe_raw_poses.len(), 7))
    }

    /// Applied pose-graph optimizations so far (the correction epoch).
    #[getter]
    fn pgo_epoch(&self) -> u64 {
        self.inner.pgo_epoch
    }

    /// Each applied PGO, oldest first: `[current, candidate, moved,
    /// dropped_edges, iterations]` (the loop's keyframe timestamps, keyframes
    /// moved, parallel edges dropped, LM steps): `int64[r, 5]`.
    #[getter]
    fn pgo_runs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let runs: Vec<i64> = self
            .inner
            .pgo_runs
            .iter()
            .flat_map(|run| {
                let counts = [run.moved, run.dropped_edges, run.iterations]
                    .map(|n| i64::try_from(n).unwrap_or(i64::MAX));
                [run.current, run.candidate].into_iter().chain(counts)
            })
            .collect();
        runs.to_pyarray(py).reshape((self.inner.pgo_runs.len(), 5))
    }

    /// Wall time of the loops that ran a PGO (graph, solve, map write and
    /// fusion), total and slowest, microseconds.
    #[getter]
    fn pgo_us(&self) -> (u64, u64) {
        (
            self.counters.pgo.total_ns / 1_000,
            self.counters.pgo.max_ns / 1_000,
        )
    }

    /// Whether each keyframe is still in the window: `bool[k]`.
    #[getter]
    fn keyframe_active<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.inner.keyframe_active.to_pyarray(py)
    }

    /// Map point (landmark) ids: `int64[p]`. `ValueError` when an id has
    /// outgrown `int64`, the refusal `VioSnapshot` makes for the same ids.
    #[getter]
    fn point_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(landmark_ids(&self.inner.point_ids)?.to_pyarray(py))
    }

    /// Timestamp of the keyframe hosting each point: `int64[p]`.
    #[getter]
    fn point_hosts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.point_hosts.to_pyarray(py)
    }

    /// Map points in the world frame, metres: `float32[p, 3]`.
    #[getter]
    fn points_world<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.inner
            .points_world
            .as_flattened()
            .to_pyarray(py)
            .reshape((self.inner.points_world.len(), 3))
    }

    /// Covisibility edges as keyframe timestamp pairs, smaller first: `int64[e, 2]`.
    #[getter]
    fn covisibility_edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let edges: Vec<i64> = self
            .inner
            .covisibility_edges
            .iter()
            .flat_map(|&(a, b, _)| [a, b])
            .collect();
        edges
            .to_pyarray(py)
            .reshape((self.inner.covisibility_edges.len(), 2))
    }

    /// Shared landmarks per covisibility edge: `uint32[e]`.
    #[getter]
    fn covisibility_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let weights: Vec<u32> = self
            .inner
            .covisibility_edges
            .iter()
            .map(|&(_, _, weight)| weight)
            .collect();
        weights.to_pyarray(py)
    }

    /// The edges a pose-graph optimization would take now, `[from, to]` keyframe timestamps: `int64[g, 2]`.
    #[getter]
    fn pose_graph_edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let edges: Vec<i64> = self
            .inner
            .pose_graph_edges
            .iter()
            .flat_map(|edge| [edge.from, edge.to])
            .collect();
        edges
            .to_pyarray(py)
            .reshape((self.inner.pose_graph_edges.len(), 2))
    }

    /// Kind of each pose-graph edge: 0 spanning tree, 1 loop closure, 2 high covisibility: `uint8[g]`.
    #[getter]
    fn pose_graph_edge_kinds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let kinds: Vec<u8> = self
            .inner
            .pose_graph_edges
            .iter()
            .map(|edge| match edge.kind {
                PoseGraphEdgeKind::SpanningTree => 0,
                PoseGraphEdgeKind::LoopClosure => 1,
                PoseGraphEdgeKind::HighCovisibility => 2,
            })
            .collect();
        kinds.to_pyarray(py)
    }

    /// Map keyframes (RootSLAM's ``num_kfs``).
    #[getter]
    fn keyframes(&self) -> usize {
        self.inner.stats.keyframes
    }

    /// Map landmarks (``num_lms``), placed in ``points_world`` or not.
    #[getter]
    fn landmarks(&self) -> usize {
        self.inner.stats.landmarks
    }

    /// (landmark, image) observations (``num_obs``).
    #[getter]
    fn observations(&self) -> usize {
        self.inner.stats.observations
    }

    /// Keyframes culled as redundant when they left the window.
    #[getter]
    fn culled_redundant(&self) -> usize {
        self.inner.culls.redundant
    }

    /// Keyframes culled while the map was over its keyframe budget.
    #[getter]
    fn culled_over_budget(&self) -> usize {
        self.inner.culls.over_budget
    }

    /// Shared landmarks at which a covisibility edge is "high".
    #[getter]
    fn high_covisibility_weight(&self) -> u32 {
        self.inner.high_covisibility_weight
    }

    /// Median number of landmarks a map keyframe sees.
    #[getter]
    fn median_keyframe_landmarks(&self) -> usize {
        self.inner.median_keyframe_landmarks
    }

    /// The map's payload in bytes (a lower bound on its heap).
    #[getter]
    fn retained_bytes(&self) -> usize {
        self.inner.retained_bytes
    }

    /// Stamps the back end has taken.
    #[getter]
    fn stamps(&self) -> u64 {
        self.counters.stamps
    }

    /// Loop queries run.
    #[getter]
    fn queries_run(&self) -> u64 {
        self.counters.queries_run
    }

    /// Loop queries dropped because the back end was behind (Realtime only).
    #[getter]
    fn queries_dropped(&self) -> u64 {
        self.counters.queries_dropped
    }

    /// Most stamps ever waiting for the back-end thread (0 Offline).
    #[getter]
    fn max_pending_stamps(&self) -> u64 {
        self.counters.max_pending_stamps
    }

    /// Steps timed.
    #[getter]
    fn step_count(&self) -> u64 {
        self.counters.step.steps
    }

    /// Wall time of all steps, microseconds.
    #[getter]
    fn step_total_us(&self) -> u64 {
        self.counters.step.total_ns / 1_000
    }

    /// Wall time of the slowest step, microseconds.
    #[getter]
    fn step_max_us(&self) -> u64 {
        self.counters.step.max_ns / 1_000
    }

    /// Wall time of all loop queries, microseconds.
    #[getter]
    fn query_total_us(&self) -> u64 {
        self.counters.query.total_ns / 1_000
    }

    /// Wall time of the slowest loop query, microseconds.
    #[getter]
    fn query_max_us(&self) -> u64 {
        self.counters.query.max_ns / 1_000
    }

    /// Views the Realtime back-end thread posted for readers (0 Offline).
    #[getter]
    fn publish_count(&self) -> u64 {
        self.counters.publish.steps
    }

    /// Wall time of the slowest posted view, microseconds.
    #[getter]
    fn publish_max_us(&self) -> u64 {
        self.counters.publish.max_ns / 1_000
    }

    /// Wall time of all posted views, microseconds.
    #[getter]
    fn publish_total_us(&self) -> u64 {
        self.counters.publish.total_ns / 1_000
    }

    /// Verified loops whose keyframes are still in the map, oldest first, as
    /// `[current, candidate]` keyframe timestamps: `int64[l, 2]`.
    #[getter]
    fn loop_edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let edges: Vec<i64> = self
            .inner
            .loops
            .iter()
            .flat_map(|edge| [edge.current, edge.candidate])
            .collect();
        edges.to_pyarray(py).reshape((self.inner.loops.len(), 2))
    }

    /// Whether each loop's landmarks were fused, at once or after a pose-graph
    /// optimization (else it was left unapplied): `bool[l]`.
    #[getter]
    fn loop_fused<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let fused: Vec<bool> = self.inner.loops.iter().map(|edge| edge.fused).collect();
        fused.to_pyarray(py)
    }

    /// Drift of each loop when it was verified (translation and rotation of
    /// `T_map^-1 T_pnp`), in the order of `loop_edges`: `float64[l, 2]`
    /// (metres, degrees).
    #[getter]
    fn loop_drift<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let drift: Vec<f64> = self
            .inner
            .loops
            .iter()
            .flat_map(|edge| [edge.drift_m, edge.drift_deg])
            .collect();
        drift.to_pyarray(py).reshape((self.inner.loops.len(), 2))
    }

    /// What loop closing has done so far.
    #[getter]
    fn loop_counts(&self) -> LoopCounts {
        let counts = self.inner.loop_counts;
        LoopCounts {
            indexed: counts.indexed,
            without_images: counts.without_images,
            images_stripped: counts.images_stripped,
            with_candidates: counts.with_candidates,
            candidates: counts.candidates,
            pnp_errors: counts.pnp_errors,
            verified: counts.verified,
            fused: counts.fused,
            needs_pgo: counts.needs_pgo,
            optimized: counts.optimized,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MapSnapshot(mode={}, keyframes={}, points={}, edges={}, queries_dropped={})",
            self.mode(),
            self.inner.keyframe_t_ns.len(),
            self.inner.point_ids.len(),
            self.inner.covisibility_edges.len(),
            self.counters.queries_dropped
        )
    }
}

/// Landmark ids as `int64`; `ValueError` when one has outgrown it, the refusal
/// `VioSnapshot` makes for the same ids.
fn landmark_ids(ids: &[LandmarkId]) -> PyResult<Vec<i64>> {
    ids.iter()
        .map(|id| {
            i64::try_from(id.0).map_err(|_| PyValueError::new_err(format!("landmark id {} does not fit in an int64", id.0)))
        })
        .collect()
}

/// `Vio.map_delta`'s result: what changed in the map since a revision, so a
/// reader redraws only that (every keyframe pose and the loops, which stay
/// small, and the points of the keyframes a write touched). Arrays as in
/// :class:`MapSnapshot`.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
#[derive(Debug)]
pub struct MapDelta {
    inner: CoreMapDelta,
}

impl MapDelta {
    pub fn new(inner: CoreMapDelta) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl MapDelta {
    /// The map's revision when the delta was taken: the next read's `since`.
    #[getter]
    fn revision(&self) -> u64 {
        self.inner.revision
    }

    /// Map keyframe timestamps, oldest first: `int64[k]`.
    #[getter]
    fn keyframe_t_ns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.keyframe_t_ns.to_pyarray(py)
    }

    /// Keyframe poses `T_w_i` as the map holds them: `float64[k, 7]`.
    #[getter]
    fn keyframe_poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        flat_poses(&self.inner.keyframe_poses)
            .to_pyarray(py)
            .reshape((self.inner.keyframe_poses.len(), 7))
    }

    /// The window's keyframe poses, before any correction: `float64[k, 7]`.
    #[getter]
    fn keyframe_raw_poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        flat_poses(&self.inner.keyframe_raw_poses)
            .to_pyarray(py)
            .reshape((self.inner.keyframe_raw_poses.len(), 7))
    }

    /// Whether each keyframe is still in the window: `bool[k]`.
    #[getter]
    fn keyframe_active<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.inner.keyframe_active.to_pyarray(py)
    }

    /// Keyframes whose hosted points changed since the read's revision, removed
    /// ones included: `int64[c]`.
    #[getter]
    fn changed_hosts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.changed_hosts.to_pyarray(py)
    }

    /// The changed hosts' point ids, by host then id: `int64[p]`.
    #[getter]
    fn point_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(landmark_ids(&self.inner.point_ids)?.to_pyarray(py))
    }

    /// Timestamp of the keyframe hosting each point: `int64[p]`.
    #[getter]
    fn point_hosts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.point_hosts.to_pyarray(py)
    }

    /// The changed hosts' points in the world frame, metres: `float32[p, 3]`.
    #[getter]
    fn points_world<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.inner
            .points_world
            .as_flattened()
            .to_pyarray(py)
            .reshape((self.inner.points_world.len(), 3))
    }

    /// Verified loops whose keyframes are still in the map, oldest first, as
    /// `[current, candidate]` keyframe timestamps: `int64[l, 2]`.
    #[getter]
    fn loop_edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let edges: Vec<i64> = self.inner.loops.iter().flat_map(|edge| [edge.current, edge.candidate]).collect();
        edges.to_pyarray(py).reshape((self.inner.loops.len(), 2))
    }

    /// Whether each loop's landmarks were fused: `bool[l]`.
    #[getter]
    fn loop_fused<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let fused: Vec<bool> = self.inner.loops.iter().map(|edge| edge.fused).collect();
        fused.to_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "MapDelta(revision={}, keyframes={}, changed_hosts={}, points={})",
            self.inner.revision,
            self.inner.keyframe_t_ns.len(),
            self.inner.changed_hosts.len(),
            self.inner.point_ids.len()
        )
    }
}
