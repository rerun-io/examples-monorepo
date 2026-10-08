use crate::{
    model::{orthonormalize, retract, LandmarkJacobian, Model, Pose, Step},
    residual::{
        evaluate, evaluate_into, evaluate_rigid, evaluate_rigid_into, linearize, normal_equations,
        rigid_landmarks, JacobianRows, Residual, View, Views,
    },
};
use kornia_staging_algebra::optim::solvers::{
    levenberg_marquardt, marquardt_scaling, DampingRule, LmConfig, LmProblem, LmTermination,
};
use nalgebra::{SMatrix, SVector, SymmetricEigen};

/// Damping above this ends the loop with an undamped stationarity test.
pub(crate) const MAX_DAMPING: f64 = 1e10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JacobianMode {
    Analytic,
    CentralDifference,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub phi: f64,
    pub dist_weight: f64,
    pub temporal_weight: f64,
    pub translation_unit: f64,
    pub margin: f64,
    pub max_iterations: usize,
    pub relative_tolerance: f64,
    pub absolute_tolerance: f64,
    pub initial_damping: f64,
    pub init_iterations: usize,
    pub init_relative_tolerance: f64,
    pub rotation_hypotheses: usize,
    pub full_fit_hypotheses: usize,
    pub finger_starts: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            phi: 1.0,
            dist_weight: 0.04,
            temporal_weight: 20.0,
            translation_unit: 0.1,
            margin: 5.0_f64.to_radians(),
            max_iterations: 10,
            relative_tolerance: 1e-3,
            absolute_tolerance: 1e-6,
            initial_damping: 1e-3,
            init_iterations: 40,
            init_relative_tolerance: 1e-6,
            rotation_hypotheses: 24,
            full_fit_hypotheses: 2,
            finger_starts: 2,
        }
    }
}
/// Why an LM solve stopped (handtrack's names: [`Termination::as_str`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// An accepted step lowered the energy by less than the tolerance.
    Tolerance,
    /// The damping ran away at a stationary point.
    Stationary,
    /// The iteration budget ran out.
    Iterations,
    /// The damping ran away at a point that is not stationary.
    Damping,
    /// The energy or the pose left the finite numbers.
    NonFinite,
    /// A cold fit without three observed points in any view: no solve ran.
    NoEvidence,
}

impl From<LmTermination> for Termination {
    fn from(reason: LmTermination) -> Self {
        match reason {
            LmTermination::Tolerance => Self::Tolerance,
            LmTermination::Stationary => Self::Stationary,
            LmTermination::Iterations => Self::Iterations,
            LmTermination::Damping => Self::Damping,
            LmTermination::NonFinite | LmTermination::SolveFailed => Self::NonFinite,
        }
    }
}

impl Termination {
    /// Whether the final termination denotes convergence.
    pub fn converged(self) -> bool {
        matches!(self, Self::Tolerance | Self::Stationary)
    }

    /// handtrack's name: `tolerance`, `stationary`, `iterations`, `damping`, `non_finite` or `no_evidence`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tolerance => "tolerance",
            Self::Stationary => "stationary",
            Self::Iterations => "iterations",
            Self::Damping => "damping",
            Self::NonFinite => "non_finite",
            Self::NoEvidence => "no_evidence",
        }
    }
}

/// Why [`Config::from_numbers`] refuses a configuration.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ConfigError {
    /// Neither 9 (warm fit) nor 14 (warm and cold fit) numbers.
    #[error("config must have 9 warm-fit or 14 warm/cold fit numbers")]
    Length,
    /// A non-finite number, or a warm-fit number out of range.
    #[error("invalid fit config")]
    Warm,
    /// A cold-fit number out of range.
    #[error("invalid cold fit config")]
    Cold,
}

impl Config {
    /// The configuration from `handfit.HandFitter`'s numbers: phi, dist_weight, temporal_weight, translation_unit, margin,
    /// max_iterations, relative_tolerance, absolute_tolerance, initial_damping, then optionally init_iterations,
    /// init_relative_tolerance, rotation_hypotheses, full_fit_hypotheses, finger_starts ([`Config::default`]'s without them).
    ///
    /// # Errors
    ///
    /// [`ConfigError::Length`] for another count, [`ConfigError::Warm`] for a non-finite number or a warm-fit number out of
    /// range (counts must be whole), [`ConfigError::Cold`] for a cold-fit number out of range.
    pub fn from_numbers(numbers: &[f64]) -> Result<Self, ConfigError> {
        let Some((warm, cold)) = numbers.split_first_chunk::<9>() else {
            return Err(ConfigError::Length);
        };
        let cold: Option<&[f64; 5]> = if cold.is_empty() {
            None
        } else {
            Some(cold.try_into().map_err(|_| ConfigError::Length)?)
        };
        let &[phi, dist_weight, temporal_weight, translation_unit, margin, max_iterations, relative_tolerance, absolute_tolerance, initial_damping] =
            warm;
        if numbers.iter().any(|x| !x.is_finite())
            || phi <= 0.0
            || dist_weight < 0.0
            || temporal_weight < 0.0
            || translation_unit <= 0.0
            || margin < 0.0
            || max_iterations < 0.0
            || max_iterations.fract() != 0.0
            || relative_tolerance < 0.0
            || absolute_tolerance < 0.0
            || initial_damping <= 0.0
        {
            return Err(ConfigError::Warm);
        }
        let mut config = Self {
            phi,
            dist_weight,
            temporal_weight,
            translation_unit,
            margin,
            max_iterations: max_iterations as usize,
            relative_tolerance,
            absolute_tolerance,
            initial_damping,
            ..Self::default()
        };
        if let Some(
            &[init_iterations, init_relative_tolerance, rotation_hypotheses, full_fit_hypotheses, finger_starts],
        ) = cold
        {
            if init_iterations < 0.0
                || init_iterations.fract() != 0.0
                || init_relative_tolerance < 0.0
                || !(0.0..=24.0).contains(&rotation_hypotheses)
                || rotation_hypotheses.fract() != 0.0
                || full_fit_hypotheses < 1.0
                || full_fit_hypotheses.fract() != 0.0
                || !(1.0..=2.0).contains(&finger_starts)
                || finger_starts.fract() != 0.0
            {
                return Err(ConfigError::Cold);
            }
            config.init_iterations = init_iterations as usize;
            config.init_relative_tolerance = init_relative_tolerance;
            config.rotation_hypotheses = rotation_hypotheses as usize;
            config.full_fit_hypotheses = full_fit_hypotheses as usize;
            config.finger_starts = finger_starts as usize;
        }
        Ok(config)
    }
}

/// Why a fit refuses its input.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq)]
pub enum FitError {
    /// Invalid view count or calibration.
    #[error(transparent)]
    Views(#[from] crate::residual::ViewValidationError),
    /// A cold fit whose config asks for no full fit: `full_fit_hypotheses` or `finger_starts` is 0.
    #[error(
        "the cold fit config asks for no full fit (full_fit_hypotheses or finger_starts is 0)"
    )]
    NoFullFit,
}

#[derive(Debug)]
pub struct FitResult {
    pub pose: Pose,
    pub energies: [f64; 4],
    pub iterations: usize,
    pub converged: bool,
    pub termination: Termination,
    /// Energy before final polar correction, used to rank cold hypotheses.
    pub solve_energy: f64,
    pub cold: Option<crate::cold::Diagnostics>,
}
/// `J^T J` and `J^T r` over the solve's free coordinates: all 26 (wrist rotation, translation, 20 finger angles), or the first
/// 6 in a rigid solve, whose fingers are held.
type Gauss<const N: usize> = (SMatrix<f64, N, N>, SVector<f64, N>);

struct PoseNormal<const N: usize> {
    h: SMatrix<f64, N, N>,
    g: SVector<f64, N>,
    mask: SVector<f64, N>,
    scaling: SVector<f64, N>,
}

struct PoseStep<const N: usize> {
    tangent: SVector<f64, N>,
    candidate: Pose,
}

fn equations<const N: usize>(
    pose: &Pose,
    normal: &Gauss<N>,
    limits: &SMatrix<f64, 20, 2>,
) -> PoseNormal<N> {
    let (mut h, mut g) = *normal;
    let mut mask = SVector::<f64, N>::repeat(1.0);
    for k in 6..N {
        let i = k - 6;
        if (pose.angles[i] <= limits[(i, 0)] + 1e-6 && g[k] > 0.0)
            || (pose.angles[i] >= limits[(i, 1)] - 1e-6 && g[k] < 0.0)
        {
            mask[k] = 0.0;
            g[k] = 0.0;
            h.row_mut(k).fill(0.0);
            h.column_mut(k).fill(0.0);
        }
    }
    let mut diagonal = h.diagonal();
    marquardt_scaling(&mut diagonal, Default::default());
    for i in 0..N {
        if mask[i] == 0.0 {
            diagonal[i] = 1.0;
        }
    }
    PoseNormal { h, g, mask, scaling: diagonal }
}

/// The 26-coordinate tangent step of an `N`-coordinate one (held fingers do not move).
fn widen<const N: usize>(step: &SVector<f64, N>) -> Step {
    Step::from_fn(|i, _| if i < N { step[i] } else { 0.0 })
}

/// Quadratic energy decrease for the applied step, the joint angles' effective (clamped) step included: coordinates 6..26 are
/// the joint angles, a 27th (the scale solve's phi) keeps its step.
pub(crate) fn effective_predicted_reduction<const N: usize>(
    pose: &Pose,
    candidate: &Pose,
    step: &SVector<f64, N>,
    h: &SMatrix<f64, N, N>,
    g: &SVector<f64, N>,
) -> f64 {
    let mut effective = *step;
    for k in 6..N.min(26) {
        effective[k] = candidate.angles[k - 6] - pose.angles[k - 6];
    }
    kornia_staging_algebra::optim::solvers::predicted_reduction(&effective, h, g)
}

/// The warm fit of one hand: an LM solve from `prior`, with the temporal prior pulling towards it (handtrack `fit_hand`).
///
/// # Arguments
///
/// * `model` - The hand model at the hand's scale.
/// * `config` - The residual weights and the solver settings.
/// * `prior` - The previous frame's pose: the start (joint angles clamped to the widened limits) and the temporal prior.
/// * `mirror` - +1 left hand, −1 right hand.
/// * `views` - The hand's views, at most [`crate::residual::MAX_VIEWS`]; with none the solve runs on the temporal prior alone.
/// * `mode` - The analytic Jacobian, or handtrack's central differences.
///
/// # Returns
///
/// The fitted pose, its energies and why the solve stopped.
///
/// # Errors
///
/// [`FitError::Views`] for more than [`crate::residual::MAX_VIEWS`] views.
pub fn fit(
    model: &Model,
    config: &Config,
    prior: &Pose,
    mirror: f64,
    views: &[View],
    mode: JacobianMode,
) -> Result<FitResult, FitError> {
    let checked =
        Views::new(views)?;
    Ok(solve(model, config, prior, mirror, checked, mode, false))
}

/// One LM solve; rigid fits hold the fingers and solve for the six wrist coordinates only.
///
/// # Arguments
///
/// * `model`, `config`, `prior`, `mirror`, `mode` - As [`fit`].
/// * `views` - The hand's views.
/// * `rigid` - Hold the finger angles (the palm-only stage of a cold fit); the result then keeps the pose of its last
///   residual, before the polar correction.
///
/// # Returns
///
/// The solved pose, its energies and why the solve stopped.
pub fn solve(
    model: &Model,
    config: &Config,
    prior: &Pose,
    mirror: f64,
    views: Views<'_>,
    mode: JacobianMode,
    rigid: bool,
) -> FitResult {
    let limits = SMatrix::<f64, 20, 2>::from_fn(|i, k| {
        model.limits[(i, k)]
            + if k == 0 {
                -config.margin
            } else {
                config.margin
            }
    });
    let mut start = prior.clone();
    for i in 0..20 {
        start.angles[i] = start.angles[i].clamp(limits[(i, 0)], limits[(i, 1)]);
    }
    let zero = Step::zeros();
    let full_residual =
        |pose: &Pose| evaluate(model, config, pose, prior, mirror, views, &zero, None);
    let outcome = match (mode, rigid) {
        (JacobianMode::Analytic, true) => {
            // Rigid steps never move the (already clamped) finger angles: the hand-frame landmarks are fixed for the solve.
            let local = rigid_landmarks(model, &start, mirror);
            let mut rows = Box::new(JacobianRows::<6>::zeros());
            iterate::<6>(
                config,
                start,
                &limits,
                &|pose| evaluate_rigid(config, pose, prior, &local, views, None),
                &mut |pose| {
                    let r = evaluate_rigid_into(config, pose, prior, &local, views, &mut rows);
                    normal_equations(config, views, &r, &rows)
                },
            )
        }
        (JacobianMode::Analytic, false) => {
            // Zeroed once: every linearisation of this solve writes the same rows (see `evaluate_into`).
            let mut landmarks = Box::new(LandmarkJacobian::zeros());
            let mut rows = Box::new(JacobianRows::<26>::zeros());
            iterate::<26>(config, start, &limits, &full_residual, &mut |pose| {
                let r = evaluate_into(
                    model,
                    config,
                    pose,
                    prior,
                    mirror,
                    views,
                    &mut landmarks,
                    &mut rows,
                );
                normal_equations(config, views, &r, &rows)
            })
        }
        (JacobianMode::CentralDifference, true) => {
            iterate::<6>(config, start, &limits, &full_residual, &mut |pose| {
                let (r, j) = linearize(model, config, pose, prior, mirror, views, mode);
                normal_equations(config, views, &r, &j.fixed_rows::<6>(0).into_owned())
            })
        }
        (JacobianMode::CentralDifference, false) => {
            iterate::<26>(config, start, &limits, &full_residual, &mut |pose| {
                let (r, j) = linearize(model, config, pose, prior, mirror, views, mode);
                normal_equations(config, views, &r, &j)
            })
        }
    };
    let Outcome {
        mut pose,
        residual,
        energy,
        iterations,
        mut termination,
    } = outcome;
    // Rigid hypotheses feed the full stage before the reference's final polar correction.
    if !rigid && pose.rotation.iter().all(|x| x.is_finite()) {
        pose.rotation = orthonormalize(&pose.rotation);
    }
    // A rigid solve ends at the pose of its last residual; a full solve evaluates the corrected pose.
    let final_r = if rigid {
        residual
    } else {
        full_residual(&pose)
    };
    let e2d = final_r.fixed_rows::<84>(0).norm_squared();
    let dist = final_r.fixed_rows::<42>(84).norm_squared();
    let temporal = final_r.fixed_rows::<32>(126).norm_squared();
    let energies = [
        e2d,
        if config.dist_weight > 0.0 {
            dist / config.dist_weight
        } else {
            0.0
        },
        if config.temporal_weight > 0.0 {
            temporal / config.temporal_weight
        } else {
            0.0
        },
        e2d + dist + temporal,
    ];
    if !energies[3].is_finite() {
        termination = Termination::NonFinite;
    }
    FitResult {
        pose,
        energies,
        iterations,
        converged: termination.converged(),
        termination,
        solve_energy: energy,
        cold: None,
    }
}

/// The dense linear algebra of the LM loop, at the two sizes it runs at (nalgebra's decompositions need concrete sizes).
trait Dense<const N: usize> {
    /// Solves the damped system: Cholesky (it is symmetric positive definite), LU with pivoting if rounding says otherwise.
    fn solve_damped(self, rhs: &SVector<f64, N>) -> Option<SVector<f64, N>>;
    /// Eigenvalues and eigenvectors of a symmetric matrix.
    fn eigenpairs(self) -> (SVector<f64, N>, SMatrix<f64, N, N>);
}

macro_rules! dense {
    ($n:literal) => {
        impl Dense<$n> for SMatrix<f64, $n, $n> {
            fn solve_damped(self, rhs: &SVector<f64, $n>) -> Option<SVector<f64, $n>> {
                cholesky_solve(&self, rhs).or_else(|| {
                    self.lu()
                        .solve(rhs)
                        .filter(|step| step.iter().all(|v| v.is_finite()))
                })
            }
            fn eigenpairs(self) -> (SVector<f64, $n>, SMatrix<f64, $n, $n>) {
                let eigen = SymmetricEigen::new(self);
                (eigen.eigenvalues, eigen.eigenvectors)
            }
        }
    };
}
dense!(6);
dense!(26);

/// `a x = b` for a symmetric positive definite `a` by Cholesky, or `None` when a pivot is not positive.
///
/// The factor is kept by rows so every entry is one dot product over contiguous memory, with four running sums: on the in-order
/// cores of the RoboCap cap this is about 1.7x faster than nalgebra's LU at 26x26.
fn cholesky_solve<const N: usize>(
    a: &SMatrix<f64, N, N>,
    b: &SVector<f64, N>,
) -> Option<SVector<f64, N>> {
    let mut lower = [[0.0_f64; N]; N];
    for j in 0..N {
        let pivot = a[(j, j)] - dot(&lower[j][..j], &lower[j][..j]);
        if pivot.is_nan() || pivot <= 0.0 {
            return None;
        }
        lower[j][j] = pivot.sqrt();
        let row_j = lower[j];
        let inverse = 1.0 / row_j[j];
        for i in j + 1..N {
            lower[i][j] = (a[(i, j)] - dot(&lower[i][..j], &row_j[..j])) * inverse;
        }
    }
    let mut y = [0.0_f64; N];
    for i in 0..N {
        y[i] = (b[i] - dot(&lower[i][..i], &y[..i])) / lower[i][i];
    }
    let mut x = SVector::<f64, N>::zeros();
    for i in (0..N).rev() {
        let mut sum = y[i];
        for k in i + 1..N {
            sum -= lower[k][i] * x[k];
        }
        x[i] = sum / lower[i][i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Dot product of equal-length slices with four running sums.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let split = a.len() / 4 * 4;
    let mut sums = [0.0_f64; 4];
    for (x, y) in a[..split].chunks_exact(4).zip(b[..split].chunks_exact(4)) {
        sums[0] += x[0] * y[0];
        sums[1] += x[1] * y[1];
        sums[2] += x[2] * y[2];
        sums[3] += x[3] * y[3];
    }
    let tail: f64 = a[split..].iter().zip(&b[split..]).map(|(x, y)| x * y).sum();
    (sums[0] + sums[1]) + (sums[2] + sums[3]) + tail
}

/// Where the LM loop stopped: the pose, its residual and energy, and why.
struct Outcome {
    pose: Pose,
    residual: Residual,
    energy: f64,
    iterations: usize,
    termination: Termination,
}

/// Adapt the pose problem to the shared LM driver, retaining its static matrix sizes.
fn iterate<const N: usize>(
    config: &Config,
    pose: Pose,
    limits: &SMatrix<f64, 20, 2>,
    residual: &dyn Fn(&Pose) -> Residual,
    normal_equations: &mut dyn FnMut(&Pose) -> Gauss<N>,
) -> Outcome
where
    SMatrix<f64, N, N>: Dense<N>,
{
    let current = residual(&pose);
    let energy = current.norm_squared();
    let mut problem = PoseProblem {
        relative_tolerance: config.relative_tolerance,
        pose,
        limits,
        residual,
        normal_equations,
        current,
        energy,
        normal: None,
        candidate: None,
    };
    let solver = LmConfig {
        max_iterations: config.max_iterations,
        relative_tolerance: config.relative_tolerance,
        absolute_tolerance: config.absolute_tolerance,
        initial_damping: config.initial_damping,
        max_damping: MAX_DAMPING,
    };
    let report = levenberg_marquardt(
        &mut problem,
        &solver,
        DampingRule::Nielsen {
            policy: Default::default(),
        },
    );
    let termination = Termination::from(report.termination);
    Outcome {
        pose: problem.pose,
        residual: problem.current,
        energy: report.final_energy,
        iterations: report.iterations,
        termination,
    }
}

struct PoseProblem<'a, const N: usize> {
    relative_tolerance: f64,
    pose: Pose,
    limits: &'a SMatrix<f64, 20, 2>,
    residual: &'a dyn Fn(&Pose) -> Residual,
    normal_equations: &'a mut dyn FnMut(&Pose) -> Gauss<N>,
    current: Residual,
    energy: f64,
    normal: Option<Gauss<N>>,
    candidate: Option<(Pose, Residual, f64)>,
}

impl<const N: usize> LmProblem<f64> for PoseProblem<'_, N>
where
    SMatrix<f64, N, N>: Dense<N>,
{
    type Normal = PoseNormal<N>;
    type Step = PoseStep<N>;

    fn energy(&self) -> f64 {
        self.energy
    }

    #[inline]
    fn linearize(&mut self) -> Self::Normal {
        let linear = *self
            .normal
            .get_or_insert_with(|| (self.normal_equations)(&self.pose));
        equations(&self.pose, &linear, self.limits)
    }

    fn solve_damped(&self, normal: &Self::Normal, damping: f64) -> Option<Self::Step> {
        let PoseNormal { h, g, mask, scaling: d } = normal;
        let mut system = *h;
        for i in 0..N {
            system[(i, i)] += damping * d[i] + (1.0 - mask[i]);
            if !system[(i, i)].is_finite() {
                return None;
            }
        }
        let step = system.solve_damped(&(-g))?;
        let candidate = retract(&self.pose, &widen(&step), self.limits);
        Some(PoseStep { tangent: step, candidate })
    }

    fn predicted_reduction(&self, normal: &Self::Normal, step: &Self::Step) -> f64 {
        effective_predicted_reduction(&self.pose, &step.candidate, &step.tangent, &normal.h, &normal.g)
    }

    fn try_step(&mut self, step: Self::Step) -> f64 {
        let pose = step.candidate;
        let residual = (self.residual)(&pose);
        let energy = residual.norm_squared();
        self.candidate = Some((pose, residual, energy));
        energy
    }

    fn accept(&mut self) {
        if let Some((pose, residual, energy)) = self.candidate.take() {
            self.pose = pose;
            self.energy = energy;
            self.current = residual;
            self.normal = None;
        }
    }

    #[cold]
    fn is_stationary(&mut self) -> bool {
        let PoseNormal { h, g, scaling: d, .. } = self.linearize();
        let units = d.map(|x| 1.0 / x.sqrt());
        let normalized = SMatrix::<f64, N, N>::from_fn(|i, k| h[(i, k)] * units[i] * units[k]);
        let (eigenvalues, eigenvectors) = normalized.eigenpairs();
        let threshold = N as f64 * f64::EPSILON * eigenvalues.abs().max();
        let inverse = eigenvalues.map(|x| if x.abs() > threshold { 1.0 / x } else { 0.0 });
        let scaled_g = g.component_mul(&units);
        let gn = -(eigenvectors * (inverse.component_mul(&(eigenvectors.transpose() * scaled_g))))
            .component_mul(&units);
        let candidate = retract(&self.pose, &widen(&gn), self.limits);
        let predicted = effective_predicted_reduction(&self.pose, &candidate, &gn, &h, &g);
        stationary(
            predicted,
            scaled_g.map(|x| x * x).max(),
            self.relative_tolerance,
            self.energy,
        )
    }
}

/// Handtrack's stationarity test at the damping limit.
pub(crate) fn stationary(
    gn_prediction: f64,
    max_scaled_gradient_sq: f64,
    relative_tolerance: f64,
    energy: f64,
) -> bool {
    gn_prediction.abs() <= relative_tolerance * energy
        || max_scaled_gradient_sq <= f32::EPSILON as f64 * energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infinite_normal_diagonal_stops_before_applying_a_step() {
        let pose = Pose {
            rotation: nalgebra::Matrix3::identity(),
            translation: nalgebra::Vector3::zeros(),
            angles: SVector::zeros(),
        };
        let outcome = iterate::<6>(
            &Config::default(),
            pose,
            &SMatrix::zeros(),
            &|_| Residual::repeat(1.0),
            &mut |_| {
                (
                    SMatrix::from_diagonal(&SVector::repeat(f64::INFINITY)),
                    SVector::repeat(1.0),
                )
            },
        );
        assert_eq!(outcome.termination, Termination::NonFinite);
        assert_eq!(outcome.iterations, 0);
        assert!(outcome.pose.translation.iter().all(|v| *v == 0.0));
    }
}
