//! The inertial bootstrap's report (`config.vio_bootstrap = "inertial"`).
use pyo3::prelude::*;

/// One solve attempt of the inertial bootstrap.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct BootstrapAttempt {
    inner: slam_rs::bootstrap::BootstrapAttempt,
}

#[pymethods]
impl BootstrapAttempt {
    /// First frameset of the window, ns.
    #[getter]
    fn first_t_ns(&self) -> i64 {
        self.inner.first_t_ns
    }

    /// Last frameset of the window, ns: when the attempt ran.
    #[getter]
    fn last_t_ns(&self) -> i64 {
        self.inner.last_t_ns
    }

    /// Framesets in the window.
    #[getter]
    fn frames(&self) -> usize {
        self.inner.frames
    }

    /// Tracks handed to the solver.
    #[getter]
    fn tracks(&self) -> usize {
        self.inner.tracks
    }

    /// Inlier tracks of the solve.
    #[getter]
    fn tracks_used(&self) -> usize {
        self.inner.tracks_used
    }

    /// The verdict's name: `Accepted` or the rejection reason, e.g. `LowExcitation`.
    #[getter]
    fn verdict(&self) -> &'static str {
        slam_rs::bootstrap::verdict_name(&self.inner.verdict)
    }

    /// Whether the solver accepted the window.
    #[getter]
    fn accepted(&self) -> bool {
        self.inner.accepted()
    }

    /// Excitation, m/s^2.
    #[getter]
    fn excitation_mps2(&self) -> f64 {
        self.inner.excitation_mps2
    }

    /// RMS bearing error of the solve, rad; `None` when the attempt reports none.
    #[getter]
    fn residual_rms_rad(&self) -> Option<f64> {
        self.inner.residual_rms_rad
    }

    /// Condition number of the reduced linear system; `None` when the attempt reports none.
    #[getter]
    fn condition(&self) -> Option<f64> {
        self.inner.condition
    }

    /// Unconstrained `|g|`, m/s^2; `None` when the attempt reports none.
    #[getter]
    fn gravity_norm_unconstrained(&self) -> Option<f64> {
        self.inner.gravity_norm_unconstrained
    }

    /// Relative 1-sigma of the speed; `None` when the attempt reports none.
    #[getter]
    fn scale_sigma(&self) -> Option<f64> {
        self.inner.scale_sigma
    }

    fn __repr__(&self) -> String {
        format!(
            "BootstrapAttempt(verdict={:?}, frames={}, first_t_ns={}, last_t_ns={})",
            self.verdict(),
            self.inner.frames,
            self.inner.first_t_ns,
            self.inner.last_t_ns
        )
    }
}

/// Every attempt of the inertial bootstrap and where it started the estimator.
#[pyclass(module = "slam_rs._core", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct BootstrapReport {
    inner: slam_rs::bootstrap::BootstrapReport,
}

impl From<&slam_rs::bootstrap::BootstrapReport> for BootstrapReport {
    fn from(report: &slam_rs::bootstrap::BootstrapReport) -> Self {
        Self {
            inner: report.clone(),
        }
    }
}

#[pymethods]
impl BootstrapReport {
    /// The attempts in order.
    #[getter]
    fn attempts(&self) -> Vec<BootstrapAttempt> {
        self.inner
            .attempts
            .iter()
            .map(|attempt| BootstrapAttempt {
                inner: attempt.clone(),
            })
            .collect()
    }

    /// The accepted window's first frameset, ns: the estimator's first state.
    /// `None` before a window is accepted.
    #[getter]
    fn start_t_ns(&self) -> Option<i64> {
        self.inner.start_t_ns
    }

    /// The frameset that completed the accepted window, ns: the first pose
    /// published. `None` before a window is accepted.
    #[getter]
    fn accepted_at_t_ns(&self) -> Option<i64> {
        self.inner.accepted_at_t_ns
    }

    /// Framesets replayed into the fresh estimator.
    #[getter]
    fn replayed(&self) -> usize {
        self.inner.replayed
    }

    fn __repr__(&self) -> String {
        format!(
            "BootstrapReport(attempts={}, start_t_ns={:?}, accepted_at_t_ns={:?}, replayed={})",
            self.inner.attempts.len(),
            self.inner.start_t_ns,
            self.inner.accepted_at_t_ns,
            self.inner.replayed
        )
    }
}
