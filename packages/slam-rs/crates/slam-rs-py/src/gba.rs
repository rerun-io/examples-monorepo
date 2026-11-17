//! `Vio.finish_map`'s result: the map's keyframes before and after the global
//! BA, and the GBA's counts and per-iteration scalars (PR d2). Flat copies (D51).

use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use slam_rs::backend::{GbaRun as CoreGbaRun, MapResult as CoreMapResult};

use crate::flat_poses;

/// The map's keyframes at the end of a sequence (`Vio.finish_map`).
///
/// Poses are `T_w_i` (world from IMU), `[tx, ty, tz, qx, qy, qz, qw]`, on the
/// clock `track` was given. `keyframe_poses_vio` is what the window left in the
/// map; `keyframe_poses` is the same after the global BA (equal without it).
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
#[derive(Debug)]
pub struct MapResult {
    keyframe_t_ns: Vec<i64>,
    keyframe_poses_vio: Vec<f64>,
    keyframe_poses: Vec<f64>,
    gba: Option<Py<GbaRun>>,
}

impl MapResult {
    /// Flatten one result.
    pub fn build(py: Python<'_>, result: &CoreMapResult) -> PyResult<Self> {
        Ok(Self {
            keyframe_t_ns: result.keyframe_t_ns.clone(),
            keyframe_poses_vio: flat_poses(&result.keyframe_poses_vio),
            keyframe_poses: flat_poses(&result.keyframe_poses),
            gba: result
                .gba
                .as_ref()
                .map(|run| Py::new(py, GbaRun::build(run)))
                .transpose()?,
        })
    }
}

#[pymethods]
impl MapResult {
    /// Map keyframe timestamps, oldest first: `int64[k]`.
    #[getter]
    fn keyframe_t_ns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.keyframe_t_ns.to_pyarray(py)
    }

    /// `T_w_i` as the window left them: `float64[k, 7]`.
    #[getter]
    fn keyframe_poses_vio<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.keyframe_poses_vio
            .to_pyarray(py)
            .reshape((self.keyframe_t_ns.len(), 7))
    }

    /// `T_w_i` after the global BA: `float64[k, 7]`.
    #[getter]
    fn keyframe_poses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.keyframe_poses
            .to_pyarray(py)
            .reshape((self.keyframe_t_ns.len(), 7))
    }

    /// The global BA, or `None` when `finish_map` ran without it.
    #[getter]
    fn gba(&self, py: Python<'_>) -> Option<Py<GbaRun>> {
        self.gba.as_ref().map(|run| run.clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "MapResult(keyframes={}, gba={})",
            self.keyframe_t_ns.len(),
            self.gba.is_some()
        )
    }
}

/// One global BA over the map: what the problem held, the LM attempts and the
/// wall of each stage.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
#[derive(Debug)]
pub struct GbaRun {
    run: CoreGbaRun,
}

impl GbaRun {
    fn build(run: &CoreGbaRun) -> Self {
        Self { run: run.clone() }
    }

    fn per_iteration<T: numpy::Element + Copy>(
        &self,
        value: impl Fn(&slam_rs::backend::gba::GbaIteration) -> T,
    ) -> Vec<T> {
        self.run.report.iterations.iter().map(value).collect()
    }
}

#[pymethods]
impl GbaRun {
    /// Keyframe variables.
    #[getter]
    fn keyframes(&self) -> usize {
        self.run.counts.keyframes
    }

    /// Keyframes under `backend.gba_min_obs_per_kf`: pose-graph rows only (B22).
    #[getter]
    fn keyframes_without_reprojection(&self) -> usize {
        self.run.counts.keyframes_without_reprojection
    }

    /// Landmarks in the problem.
    #[getter]
    fn landmarks(&self) -> usize {
        self.run.counts.landmarks
    }

    /// Reprojection observations in the problem.
    #[getter]
    fn observations(&self) -> usize {
        self.run.counts.observations
    }

    /// Pose-graph rows.
    #[getter]
    fn pose_graph_rows(&self) -> usize {
        self.run.counts.pose_graph_rows
    }

    /// Landmarks left out for want of a finite position.
    #[getter]
    fn nonfinite_landmarks(&self) -> usize {
        self.run.counts.nonfinite_landmarks
    }

    /// `ERROR_VALID` before the first step.
    #[getter]
    fn initial_cost(&self) -> f64 {
        self.run.report.initial_cost.error
    }

    /// `ERROR_VALID` of the returned state.
    #[getter]
    fn final_cost(&self) -> f64 {
        self.run.report.final_cost.error
    }

    /// Why the LM loop stopped: ``"converged"``, ``"max_iterations"`` or ``"lambda_overflow"``.
    #[getter]
    fn termination(&self) -> &'static str {
        match self.run.report.termination {
            slam_rs::backend::gba::GbaTermination::Converged => "converged",
            slam_rs::backend::gba::GbaTermination::MaxIterations => "max_iterations",
            slam_rs::backend::gba::GbaTermination::LambdaOverflow => "lambda_overflow",
        }
    }

    /// LM damping of each attempt: `float64[n]`.
    #[getter]
    fn iteration_lambda<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.per_iteration(|i| i.lambda).to_pyarray(py)
    }

    /// Cost after each attempt's step, NaN where the step was not finite: `float64[n]`.
    #[getter]
    fn iteration_cost<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.per_iteration(|i| i.cost_after.map_or(f64::NAN, |c| c.error))
            .to_pyarray(py)
    }

    /// Whether each attempt's step was kept: `bool[n]`.
    #[getter]
    fn iteration_accepted<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.per_iteration(|i| i.accepted).to_pyarray(py)
    }

    /// PCG iterations of each attempt: `int64[n]`.
    #[getter]
    fn iteration_cg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.per_iteration(|i| i.pcg.iterations as i64)
            .to_pyarray(py)
    }

    /// Wall seconds of each attempt's stages, columns stage 1 (linearize, QR),
    /// stage 2 (damping, preconditioner), PCG, back-substitution, evaluation:
    /// `float64[n, 5]`.
    #[getter]
    fn iteration_stage_s<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let flat: Vec<f64> = self
            .run
            .report
            .iterations
            .iter()
            .flat_map(|i| {
                [
                    i.seconds.stage1,
                    i.seconds.stage2,
                    i.seconds.pcg,
                    i.seconds.back_substitution,
                    i.seconds.evaluation,
                ]
            })
            .collect();
        flat.to_pyarray(py)
            .reshape((self.run.report.iterations.len(), 5))
    }

    /// Wall seconds turning the map into the problem.
    #[getter]
    fn build_s(&self) -> f64 {
        self.run.build_s
    }

    /// Wall seconds of the solve (centring, LM, uncentring, gauge).
    #[getter]
    fn solve_s(&self) -> f64 {
        self.run.solve_s
    }

    /// Wall seconds writing the result into the map.
    #[getter]
    fn apply_s(&self) -> f64 {
        self.run.apply_s
    }

    /// How far the free gauge had tilted the solution's world z, degrees,
    /// before the driver brought it back onto the input poses.
    #[getter]
    fn gauge_tilt_deg(&self) -> f64 {
        self.run.gauge.rotation.matrix()[(2, 2)]
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }

    /// Worker threads.
    #[getter]
    fn threads(&self) -> usize {
        self.run.threads
    }

    /// Whether it solved in f64.
    #[getter]
    fn use_double(&self) -> bool {
        self.run.use_double
    }

    fn __repr__(&self) -> String {
        format!(
            "GbaRun(keyframes={}, landmarks={}, attempts={}, cost {:.4e} -> {:.4e}, {} s)",
            self.run.counts.keyframes,
            self.run.counts.landmarks,
            self.run.report.iterations.len(),
            self.run.report.initial_cost.error,
            self.run.report.final_cost.error,
            self.run.solve_s
        )
    }
}
