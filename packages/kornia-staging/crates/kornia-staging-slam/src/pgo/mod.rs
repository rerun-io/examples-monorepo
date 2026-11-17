//! SE(3) pose-graph optimization over keyframe poses, in f64.
//!
//! The solver behind RootSLAM's loop closure (Ceres `SPARSE_NORMAL_CHOLESKY`, LM, 20 iterations): relative-pose
//! residuals with analytic Jacobians ([`relative_pose_residual`]), a block-sparse Cholesky over 6x6 blocks with a
//! minimum-degree ordering ([`SymbolicCholesky`]), and the staged Levenberg–Marquardt driver. The gauge is the
//! caller's set of fixed poses (RootSLAM fixes the sliding window and the newest keyframe). Each edge carries
//! its own square-root information; [`EdgeSigmas`] builds the diagonal ones.
//!
//! Single-threaded and allocation-bounded by the graph: a 1500-keyframe map solves in a few milliseconds.

mod residual;
mod sparse;

pub use residual::{relative_pose_residual, RelPoseLinearization};
pub use sparse::{BlockSymmetric, NotPositiveDefinite, SymbolicCholesky};

use kornia_staging_algebra::lie::RigidTransform;
use kornia_staging_algebra::optim::solvers::{
    levenberg_marquardt, DampingRule, LmConfig, LmProblem, LmTermination, ScalingFloor,
};
use nalgebra::{Matrix6, Vector6};

/// Standard deviations of a relative-pose measurement, the same on every axis of each part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeSigmas {
    /// Translation sigma, metres.
    pub translation_m: f64,
    /// Rotation sigma, radians.
    pub rotation_rad: f64,
}

impl EdgeSigmas {
    /// Unit information (1 m ≙ 1 rad): RootSLAM's PGO (`c.information = I6`).
    pub const UNIT: Self = Self {
        translation_m: 1.0,
        rotation_rad: 1.0,
    };

    /// The diagonal square-root information `diag(1/σ_t x3, 1/σ_r x3)`.
    pub fn sqrt_information(&self) -> Matrix6<f64> {
        let t = 1.0 / self.translation_m;
        let r = 1.0 / self.rotation_rad;
        Matrix6::from_diagonal(&Vector6::new(t, t, t, r, r, r))
    }
}

impl Default for EdgeSigmas {
    /// 0.05 m / 0.01 rad: rootba's pose-graph rows in RootSLAM's global BA, used for the PGO too so both
    /// weigh the pose graph alike.
    fn default() -> Self {
        Self {
            translation_m: 0.05,
            rotation_rad: 0.01,
        }
    }
}

/// One relative-pose measurement between two keyframes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseGraphEdge {
    /// Index of keyframe `a`.
    pub from: usize,
    /// Index of keyframe `b`.
    pub to: usize,
    /// Measured `a_from_b = T_wa^-1 T_wb`.
    pub from_t_to: RigidTransform<f64>,
    /// Square-root information `S`; the edge costs `|S r|^2`.
    pub sqrt_information: Matrix6<f64>,
}

impl PoseGraphEdge {
    /// An edge with diagonal information from `sigmas`.
    pub fn new(
        from: usize,
        to: usize,
        from_t_to: RigidTransform<f64>,
        sigmas: &EdgeSigmas,
    ) -> Self {
        Self {
            from,
            to,
            from_t_to,
            sqrt_information: sigmas.sqrt_information(),
        }
    }

    fn cost(&self, poses: &[RigidTransform<f64>]) -> f64 {
        let r =
            relative_pose_residual(&self.from_t_to, &poses[self.from], &poses[self.to]).residual;
        (self.sqrt_information * r).norm_squared()
    }
}

/// Levenberg–Marquardt limits for [`optimize_pose_graph`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PgoConfig {
    /// Attempted steps, accepted or not (RootSLAM: 20).
    pub max_iterations: usize,
    /// Stop when an accepted step lowers the cost by less than this fraction (Ceres' function tolerance).
    pub relative_tolerance: f64,
    /// Initial damping on the Marquardt-scaled diagonal (Ceres: 1 / initial trust radius 1e4).
    pub initial_damping: f64,
    /// Stop when the damping exceeds this.
    pub max_damping: f64,
}

impl Default for PgoConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            relative_tolerance: 1e-6,
            initial_damping: 1e-4,
            max_damping: 1e16,
        }
    }
}

impl PgoConfig {
    /// The staged LM driver's limits for this configuration.
    pub fn lm_config(&self) -> LmConfig<f64> {
        LmConfig {
            max_iterations: self.max_iterations,
            relative_tolerance: self.relative_tolerance,
            absolute_tolerance: 0.0,
            initial_damping: self.initial_damping,
            max_damping: self.max_damping,
        }
    }
}

/// Outcome of [`optimize_pose_graph`]; costs are sums of squared whitened residuals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PgoReport {
    /// Attempted LM steps.
    pub iterations: usize,
    /// Cost before the solve.
    pub initial_cost: f64,
    /// Cost of the returned poses.
    pub final_cost: f64,
    /// Why the LM loop stopped.
    pub termination: LmTermination,
}

/// An invalid pose graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PgoError {
    /// `fixed` does not have one flag per pose.
    #[error("{fixed} fixed flags for {poses} poses")]
    FixedLength {
        /// Number of poses.
        poses: usize,
        /// Number of fixed flags.
        fixed: usize,
    },
    /// An edge names a pose that does not exist.
    #[error("edge {edge} names pose {pose}, out of range")]
    PoseIndex {
        /// The edge.
        edge: usize,
        /// The missing pose.
        pose: usize,
    },
    /// An edge joins a pose to itself.
    #[error("edge {edge} joins a pose to itself")]
    SelfEdge {
        /// The edge.
        edge: usize,
    },
    /// An edge's measurement or information is not finite.
    #[error("edge {edge} is not finite")]
    NonFinite {
        /// The edge.
        edge: usize,
    },
    /// A pose is not finite.
    #[error("pose {pose} is not finite")]
    NonFinitePose {
        /// The pose.
        pose: usize,
    },
    /// A connected part of the graph has no fixed pose, so its gauge is free.
    #[error("pose {pose} is in a component without a fixed pose")]
    Unanchored {
        /// The lowest-numbered pose of that component.
        pose: usize,
    },
}

/// Optimize keyframe poses against relative-pose edges, in place.
///
/// Poses flagged in `fixed` are the gauge and stay bit-identical; so do poses that no edge touches. Every
/// connected component of the graph needs at least one fixed pose. On return the poses are the best LM
/// iterate; the report says why the loop stopped.
///
/// # Arguments
/// * `world_from_poses` - Keyframe poses `T_w_i`, updated in place.
/// * `fixed` - One flag per pose: `true` holds that pose constant.
/// * `edges` - Relative-pose measurements; parallel edges add up.
/// * `config` - LM limits.
///
/// # Errors
/// [`PgoError`] for a malformed graph or one with a free gauge; the poses are untouched then.
pub fn optimize_pose_graph(
    world_from_poses: &mut [RigidTransform<f64>],
    fixed: &[bool],
    edges: &[PoseGraphEdge],
    config: &PgoConfig,
) -> Result<PgoReport, PgoError> {
    let n = world_from_poses.len();
    if fixed.len() != n {
        return Err(PgoError::FixedLength {
            poses: n,
            fixed: fixed.len(),
        });
    }
    for (index, edge) in edges.iter().enumerate() {
        for pose in [edge.from, edge.to] {
            if pose >= n {
                return Err(PgoError::PoseIndex { edge: index, pose });
            }
        }
        if edge.from == edge.to {
            return Err(PgoError::SelfEdge { edge: index });
        }
        let finite = edge.sqrt_information.iter().all(|v| v.is_finite())
            && edge.from_t_to.translation.iter().all(|v| v.is_finite())
            && edge
                .from_t_to
                .rotation
                .quaternion_xyzw()
                .iter()
                .all(|v| v.is_finite());
        if !finite {
            return Err(PgoError::NonFinite { edge: index });
        }
    }
    for (pose, t) in world_from_poses.iter().enumerate() {
        let finite = t.translation.iter().all(|v| v.is_finite())
            && t.rotation.quaternion_xyzw().iter().all(|v| v.is_finite());
        if !finite {
            return Err(PgoError::NonFinitePose { pose });
        }
    }
    check_anchored(n, fixed, edges)?;

    let mut touched = vec![false; n];
    for edge in edges {
        touched[edge.from] = true;
        touched[edge.to] = true;
    }
    let mut variables = Vec::new();
    let mut variable_of = vec![None; n];
    for pose in 0..n {
        if touched[pose] && !fixed[pose] {
            variable_of[pose] = Some(variables.len());
            variables.push(pose);
        }
    }
    let initial_cost: f64 = edges.iter().map(|e| e.cost(world_from_poses)).sum();
    if variables.is_empty() {
        return Ok(PgoReport {
            iterations: 0,
            initial_cost,
            final_cost: initial_cost,
            termination: LmTermination::Tolerance,
        });
    }

    let symbolic = SymbolicCholesky::new(
        variables.len(),
        edges
            .iter()
            .filter_map(|e| Some((variable_of[e.from]?, variable_of[e.to]?))),
    );
    let layout = edges
        .iter()
        .map(|e| {
            let from = variable_of[e.from];
            let to = variable_of[e.to];
            let slot = from.zip(to).and_then(|(a, b)| symbolic.slot(a, b));
            EdgeLayout { from, to, slot }
        })
        .collect();
    let mut problem = PgoProblem {
        edges,
        layout,
        poses: world_from_poses.to_vec(),
        trial: world_from_poses.to_vec(),
        variables,
        h: symbolic.zeros(),
        symbolic,
        g: Vec::new(),
        scale: Vec::new(),
        linearized: false,
    };
    let lm = levenberg_marquardt(
        &mut problem,
        &config.lm_config(),
        DampingRule::Nielsen {
            policy: Default::default(),
        },
    );
    for &pose in &problem.variables {
        world_from_poses[pose] = problem.poses[pose];
    }
    Ok(PgoReport {
        iterations: lm.iterations,
        initial_cost,
        final_cost: lm.final_energy,
        termination: lm.termination,
    })
}

/// Every component of the edge graph holds a fixed pose.
fn check_anchored(n: usize, fixed: &[bool], edges: &[PoseGraphEdge]) -> Result<(), PgoError> {
    fn root(parent: &mut [usize], mut v: usize) -> usize {
        while parent[v] != v {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        v
    }
    let mut parent: Vec<usize> = (0..n).collect();
    let mut touched = vec![false; n];
    for edge in edges {
        touched[edge.from] = true;
        touched[edge.to] = true;
        let (a, b) = (root(&mut parent, edge.from), root(&mut parent, edge.to));
        parent[a.max(b)] = a.min(b);
    }
    let mut anchored = vec![false; n];
    for (pose, _) in fixed.iter().enumerate().filter(|(_, &f)| f) {
        let r = root(&mut parent, pose);
        anchored[r] = true;
    }
    for pose in 0..n {
        if touched[pose] && !anchored[root(&mut parent, pose)] {
            return Err(PgoError::Unanchored { pose });
        }
    }
    Ok(())
}

/// Where an edge's Jacobian blocks land in the normal equations.
struct EdgeLayout {
    from: Option<usize>,
    to: Option<usize>,
    /// Lower slot of `H_from,to` when both ends are free.
    slot: Option<(usize, bool)>,
}

/// The LM state: retained and trial poses, and the normal equations of the retained poses.
struct PgoProblem<'a> {
    edges: &'a [PoseGraphEdge],
    layout: Vec<EdgeLayout>,
    poses: Vec<RigidTransform<f64>>,
    trial: Vec<RigidTransform<f64>>,
    /// Pose index of each variable.
    variables: Vec<usize>,
    symbolic: SymbolicCholesky,
    h: BlockSymmetric,
    g: Vec<f64>,
    /// Marquardt diagonal scaling of `h`.
    scale: Vec<f64>,
    /// `h`, `g` and `scale` belong to `poses`.
    linearized: bool,
}

impl PgoProblem<'_> {
    fn cost(&self, poses: &[RigidTransform<f64>]) -> f64 {
        self.edges.iter().map(|e| e.cost(poses)).sum()
    }
}

impl LmProblem<f64> for PgoProblem<'_> {
    type Normal = ();
    type Step = Vec<f64>;

    fn energy(&self) -> f64 {
        self.cost(&self.poses)
    }

    fn linearize(&mut self) {
        if self.linearized {
            return;
        }
        let dim = 6 * self.variables.len();
        self.h.clear();
        self.g.clear();
        self.g.resize(dim, 0.0);
        for (edge, layout) in self.edges.iter().zip(&self.layout) {
            let lin = relative_pose_residual(
                &edge.from_t_to,
                &self.poses[edge.from],
                &self.poses[edge.to],
            );
            let s = &edge.sqrt_information;
            let r = s * lin.residual;
            let j_from = s * lin.d_from;
            let j_to = s * lin.d_to;
            for (variable, j) in [(layout.from, &j_from), (layout.to, &j_to)] {
                if let Some(v) = variable {
                    self.symbolic.add_diagonal(&mut self.h, v, &j.tr_mul(j));
                    let g = j.tr_mul(&r);
                    for k in 0..6 {
                        self.g[6 * v + k] += g[k];
                    }
                }
            }
            if let Some((slot, transposed)) = layout.slot {
                self.symbolic
                    .add_to_slot(&mut self.h, slot, transposed, &j_from.tr_mul(&j_to));
            }
        }
        let mut diagonal = nalgebra::DVector::from_iterator(
            dim,
            (0..self.variables.len()).flat_map(|v| {
                let block = self.symbolic.diagonal(&self.h, v);
                (0..6).map(move |k| block[(k, k)])
            }),
        );
        kornia_staging_algebra::optim::solvers::marquardt_scaling(
            &mut diagonal,
            ScalingFloor::default(),
        );
        self.scale = diagonal.as_slice().to_vec();
        self.linearized = true;
    }

    fn solve_damped(&self, _: &(), lambda: f64) -> Option<Vec<f64>> {
        let mut factor = self.h.clone();
        factor.add_scaled_identity(&self.symbolic, lambda, &self.scale);
        self.symbolic.factor(&mut factor).ok()?;
        let mut step: Vec<f64> = self.g.iter().map(|g| -g).collect();
        self.symbolic.solve(&factor, &mut step);
        Some(step)
    }

    fn predicted_reduction(&self, _: &(), step: &Vec<f64>) -> f64 {
        let mut h_step = vec![0.0; step.len()];
        self.symbolic.multiply(&self.h, step, &mut h_step);
        let g_step: f64 = self.g.iter().zip(step).map(|(g, s)| g * s).sum();
        let step_h_step: f64 = step.iter().zip(&h_step).map(|(s, h)| s * h).sum();
        -(2.0 * g_step + step_h_step)
    }

    fn try_step(&mut self, step: Vec<f64>) -> f64 {
        self.trial.copy_from_slice(&self.poses);
        for (v, &pose) in self.variables.iter().enumerate() {
            self.trial[pose].apply_inc(&Vector6::from_column_slice(&step[6 * v..6 * v + 6]));
        }
        self.cost(&self.trial)
    }

    fn accept(&mut self) {
        std::mem::swap(&mut self.poses, &mut self.trial);
        self.linearized = false;
    }
}

#[cfg(test)]
mod tests;
