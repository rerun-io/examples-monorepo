//! Shared LM control flow and extension kernels. `LmConfig` initial/max damping and
//! iteration budget map to upstream `lambda_init`, `lambda_max`, and `max_iterations`;
//! the damping rule factor maps to `lambda_factor`.

use crate::Scalar;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector};

// Upstream Levenberg acceptance floor; keep parity with kornia-algebra.
const LEVENBERG_MIN_DAMPING: f64 = 1e-10;

/// Floors used to scale a normal-equation diagonal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalingFloor<S> {
    /// Floor relative to the largest diagonal entry.
    pub relative: S,
    /// Lower bound on that largest entry before relative scaling.
    pub absolute: S,
}

impl<S: Scalar> Default for ScalingFloor<S> {
    fn default() -> Self {
        Self {
            relative: S::from_literal(1e-9),
            absolute: S::from_literal(1e-12),
        }
    }
}

/// Numerical policy for an accepted-step Nielsen damping update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NielsenPolicy<S> {
    /// Smallest multiplier applied to damping.
    pub min_ratio: S,
    /// Positive floor on the predicted energy decrease.
    pub predicted_floor: S,
    /// Multiplier on the actual-to-predicted reduction ratio.
    pub growth: S,
}

impl<S: Scalar> Default for NielsenPolicy<S> {
    fn default() -> Self {
        Self {
            min_ratio: S::from_literal(1.0 / 3.0),
            predicted_floor: S::from_literal(1e-30),
            growth: S::from_literal(2.0),
        }
    }
}

/// Floor a normal-equation diagonal for Marquardt damping, before applying a variable mask.
/// # Arguments
/// * `diagonal` - Diagonal entries replaced in place by their floored values.
/// * `policy` - Relative and absolute scaling floors.
/// ```
/// use kornia_staging_algebra::optim::solvers::marquardt_scaling;
/// let mut d = nalgebra::DVector::from_vec(vec![2.0_f64, 0.0]);
/// marquardt_scaling(&mut d, Default::default());
/// assert_eq!(d[1], 2e-9);
/// ```
#[inline]
pub fn marquardt_scaling<S: Scalar, D: Dim>(diagonal: &mut OVector<S, D>, policy: ScalingFloor<S>)
where
    DefaultAllocator: Allocator<D>,
{
    let floor = policy.relative * diagonal.max().max(policy.absolute);
    diagonal
        .iter_mut()
        .for_each(|value| *value = value.max(floor));
}

/// Quadratic energy decrease for an already projected effective step.
/// # Arguments
/// * `step` - Applied tangent step.
/// * `h` - Normal-equation Hessian.
/// * `g` - Normal-equation gradient.
#[inline]
pub fn predicted_reduction<S: Scalar, D: Dim>(
    step: &OVector<S, D>,
    h: &OMatrix<S, D, D>,
    g: &OVector<S, D>,
) -> S
where
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    -(S::from_literal(2.0) * g.dot(step) + step.dot(&(h * step)))
}

/// Nielsen damping update after an accepted step.
/// # Arguments
/// * `damping` - Current nonnegative damping.
/// * `reduction` - Actual energy decrease.
/// * `predicted` - Predicted energy decrease.
/// * `policy` - Reduction floor and damping multiplier policy.
#[inline]
pub fn nielsen_damping<S: Scalar>(
    damping: S,
    reduction: S,
    predicted: S,
    policy: NielsenPolicy<S>,
) -> S {
    let ratio = reduction / predicted.max(policy.predicted_floor);
    damping * (S::one() - (policy.growth * ratio - S::one()).powi(3)).max(policy.min_ratio)
}

/// Problem-owned linear algebra and trial state for the LM control loop.
///
/// The normal equations and step retain their concrete types. Implementations may cache
/// linearizations until acceptance. A trial may be stored separately or applied in place;
/// `reject` must restore the retained state when a trial was applied in place.
pub trait LmProblem<S: Scalar> {
    /// Normal equations, including any active-set information.
    type Normal;
    /// A damped step, optionally carrying its projected candidate.
    type Step;
    /// Energy of the retained state.
    fn energy(&self) -> S;
    /// Form the normal equations at the retained state.
    fn linearize(&mut self) -> Self::Normal;
    /// Solve once at `lambda`; `None` means the damped system cannot be solved.
    fn solve_damped(&self, normal: &Self::Normal, lambda: S) -> Option<Self::Step>;
    /// Predicted energy decrease of the effective, projected step.
    fn predicted_reduction(&self, normal: &Self::Normal, step: &Self::Step) -> S;
    /// Evaluate a candidate and retain it for `accept` or `reject`.
    /// Called once per step, after prediction when required by the damping rule.
    fn try_step(&mut self, step: Self::Step) -> S;
    /// Retain the candidate and invalidate any stale linearization.
    fn accept(&mut self);
    /// Undo a trial that was applied in place. Separate candidate storage needs no action.
    fn reject(&mut self) {}
    /// Test stationarity at the retained state when damping exceeds its maximum.
    ///
    /// Problems that need an undamped solve keep it here, with their own tolerance
    /// and projection rules. The default reports a damping limit without convergence.
    fn is_stationary(&mut self) -> bool {
        false
    }
}

/// Damping adaptation after acceptance or rejection.
#[derive(Clone, Copy, Debug)]
pub enum DampingRule<S> {
    /// Divide on acceptance (floor 1e-10); multiply on rejection.
    Levenberg {
        /// Finite factor greater than one.
        factor: S,
    },
    /// Nielsen update on acceptance; multiply by an increasing rejection factor.
    Nielsen {
        /// Accepted-step policy; `growth` also sets the initial rejection factor
        /// and its escalation, reset on acceptance. Growth must be finite and > 1.
        policy: NielsenPolicy<S>,
    },
}

/// Limits for the matrix-independent LM driver.
///
/// The caller owns parameter validation, as with upstream `LevenbergMarquardt`.
#[derive(Clone, Copy, Debug)]
pub struct LmConfig<S> {
    /// Maximum number of attempted candidates.
    pub max_iterations: usize,
    /// Relative accepted-energy decrease tolerance.
    pub relative_tolerance: S,
    /// Absolute accepted-energy decrease tolerance.
    pub absolute_tolerance: S,
    /// Damping at the first attempt.
    pub initial_damping: S,
    /// Exit threshold, checked after adaptation.
    pub max_damping: S,
}

/// Why the driver stopped; names and presentation belong to the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmTermination {
    /// An accepted decrease met the configured tolerance.
    Tolerance,
    /// The problem was stationary at the damping limit.
    Stationary,
    /// The iteration budget ended.
    Iterations,
    /// Damping exceeded its maximum away from a stationary point.
    Damping,
    /// The initial or candidate energy was non-finite.
    NonFinite,
    /// A damped solve failed before a candidate was evaluated.
    SolveFailed,
}

/// Final state of the driver; the retained parameters stay in the problem.
#[derive(Clone, Copy, Debug)]
pub struct LmReport<S> {
    /// Number of evaluated candidates, including rejected and non-finite candidates.
    /// A failed damped solve is not an evaluated candidate.
    pub iterations: usize,
    /// Energy of the retained parameters.
    pub final_energy: S,
    /// Final damping, useful for diagnostics.
    pub damping: S,
    /// Exit reason.
    pub termination: LmTermination,
}

/// Minimize a problem without taking ownership of its matrix representation.
///
/// # Arguments
/// * `problem` - Concrete normal equations, projected steps and trial state.
/// * `config` - Tolerances and iteration/damping limits.
/// * `rule` - Damping adaptation; its factor must be finite and greater than one.
///
/// The driver allocates nothing. Problem methods may allocate as their representation
/// requires. Failed solves and non-finite candidate energies stop without accepting a trial.
/// This follows handtrack: a non-finite trial ends the solve. Upstream Kornia instead
/// rejects that trial, increases damping and retries. Levenberg damping/iterate parity
/// on finite problems does not imply identical stopping policies.
///
/// ```
/// use kornia_staging_algebra::optim::solvers::{
///     levenberg_marquardt, DampingRule, LmConfig, LmProblem, LmTermination,
/// };
/// struct Square { x: f64, trial: f64 }
/// impl LmProblem<f64> for Square {
///     type Normal = f64;
///     type Step = f64;
///     fn energy(&self) -> f64 { self.x * self.x }
///     fn linearize(&mut self) -> f64 { self.x }
///     fn solve_damped(&self, g: &f64, lambda: f64) -> Option<f64> { Some(-g / (1.0 + lambda)) }
///     fn predicted_reduction(&self, g: &f64, step: &f64) -> f64 { -(2.0 * g * step + step * step) }
///     fn try_step(&mut self, step: f64) -> f64 { self.trial = self.x + step; self.trial * self.trial }
///     fn accept(&mut self) { self.x = self.trial; }
/// }
/// let mut problem = Square { x: 1.0, trial: 1.0 };
/// let limits = LmConfig {
///     max_iterations: 20, relative_tolerance: 1e-6, absolute_tolerance: 1e-12,
///     initial_damping: 1e-3, max_damping: 1e10,
/// };
/// let result = levenberg_marquardt(&mut problem, &limits, DampingRule::Nielsen { policy: Default::default() });
/// assert_eq!(result.termination, LmTermination::Tolerance);
/// assert!(problem.x.abs() < 1e-9);
/// ```
pub fn levenberg_marquardt<S: Scalar, P: LmProblem<S>>(
    problem: &mut P,
    config: &LmConfig<S>,
    rule: DampingRule<S>,
) -> LmReport<S> {
    let mut report = LmReport {
        iterations: 0,
        final_energy: problem.energy(),
        damping: config.initial_damping,
        termination: LmTermination::Iterations,
    };
    let factor = match rule {
        DampingRule::Levenberg { factor } => factor,
        DampingRule::Nielsen { policy } => policy.growth,
    };
    debug_assert!(factor.is_finite() && factor > S::one());
    if !report.final_energy.is_finite() {
        report.termination = LmTermination::NonFinite;
        return report;
    }
    let mut growth = factor;
    for _ in 0..config.max_iterations {
        let normal = problem.linearize();
        let Some(step) = problem.solve_damped(&normal, report.damping) else {
            report.termination = LmTermination::SolveFailed;
            break;
        };
        let predicted = match rule {
            DampingRule::Nielsen { .. } => problem.predicted_reduction(&normal, &step),
            DampingRule::Levenberg { .. } => S::zero(),
        };
        let candidate_energy = problem.try_step(step);
        let reduction = report.final_energy - candidate_energy;
        report.iterations += 1;
        if !candidate_energy.is_finite() {
            problem.reject();
            report.termination = LmTermination::NonFinite;
            break;
        }
        let small = reduction
            <= config.relative_tolerance * report.final_energy + config.absolute_tolerance;
        if reduction > S::zero() {
            problem.accept();
            report.final_energy = candidate_energy;
            report.damping = match rule {
                DampingRule::Levenberg { factor } => {
                    (report.damping / factor).max(S::from_literal(LEVENBERG_MIN_DAMPING))
                }
                DampingRule::Nielsen { policy } => {
                    nielsen_damping(report.damping, reduction, predicted, policy)
                }
            };
            growth = factor;
            if small {
                report.termination = LmTermination::Tolerance;
                break;
            }
        } else {
            problem.reject();
            report.damping *= growth;
            if let DampingRule::Nielsen { .. } = rule {
                growth *= factor;
            }
        }
        if report.damping > config.max_damping {
            report.termination = if problem.is_stationary() {
                LmTermination::Stationary
            } else {
                LmTermination::Damping
            };
            break;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector, SMatrix, SVector};

    #[test]
    fn default_policy_preserves_numeric_floors() {
        let scaling = ScalingFloor::<f64>::default();
        assert_eq!(scaling.relative.to_bits(), 1e-9_f64.to_bits());
        assert_eq!(scaling.absolute.to_bits(), 1e-12_f64.to_bits());
        let damping = NielsenPolicy::<f64>::default();
        assert_eq!(damping.predicted_floor.to_bits(), 1e-30_f64.to_bits());
        assert_eq!(damping.min_ratio.to_bits(), (1.0_f64 / 3.0).to_bits());
        assert_eq!(damping.growth.to_bits(), 2.0_f64.to_bits());
    }

    #[test]
    fn diagonal_floor_and_quadratic_prediction() {
        let mut diagonal = SVector::<f64, 3>::new(4.0, 0.0, 2.0);
        marquardt_scaling(&mut diagonal, Default::default());
        assert_eq!(diagonal.as_slice(), &[4.0, 4e-9, 2.0]);
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 4.0]));
        let g = DVector::from_vec(vec![-2.0, -4.0]);
        let step = DVector::from_vec(vec![1.0, 1.0]);
        assert_eq!(predicted_reduction(&step, &h, &g), 6.0);
        assert_eq!(nielsen_damping(3.0_f64, 2.0, 2.0, Default::default()), 1.0);
    }

    struct Quadratic {
        x: f64,
        candidate: f64,
    }

    impl LmProblem<f64> for Quadratic {
        type Normal = (SMatrix<f64, 1, 1>, SVector<f64, 1>);
        type Step = SVector<f64, 1>;
        fn energy(&self) -> f64 {
            (self.x - 2.0).powi(2)
        }
        fn linearize(&mut self) -> Self::Normal {
            (SMatrix::identity(), SVector::repeat(self.x - 2.0))
        }
        fn solve_damped(&self, normal: &Self::Normal, lambda: f64) -> Option<Self::Step> {
            Some(-normal.1 / (normal.0[(0, 0)] + lambda))
        }
        fn predicted_reduction(&self, normal: &Self::Normal, step: &Self::Step) -> f64 {
            -(2.0 * normal.1.dot(step) + step.dot(&(normal.0 * step)))
        }
        fn try_step(&mut self, step: Self::Step) -> f64 {
            self.candidate = self.x + step[0];
            (self.candidate - 2.0).powi(2)
        }
        fn accept(&mut self) {
            self.x = self.candidate;
        }
        fn is_stationary(&mut self) -> bool {
            self.x == 2.0
        }
    }

    #[test]
    fn fixed_size_problem_converges_without_erasing_its_matrix_type() {
        let mut problem = Quadratic {
            x: 0.0,
            candidate: 0.0,
        };
        let config = LmConfig {
            max_iterations: 20,
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-12,
            initial_damping: 1e-3,
            max_damping: 1e10,
        };
        let report = levenberg_marquardt(
            &mut problem,
            &config,
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        assert_eq!(report.termination, LmTermination::Tolerance);
        assert!((problem.x - 2.0).abs() < 1e-9);
        assert_eq!(report.final_energy, problem.energy());
    }

    #[test]
    fn nielsen_uses_the_callers_acceptance_policy() {
        // One damped step takes x=0 to x=1: actual and predicted decrease are 3.
        // These policies independently exercise the ratio floor, prediction floor,
        // and growth coefficient at the public driver boundary.
        for (min_ratio, predicted_floor, growth, expected_damping) in [
            (0.5, 6.0, 3.0, 0.875),
            (0.9, 6.0, 3.0, 0.9),
            (0.5, 12.0, 3.0, 1.015625),
            (0.5, 6.0, 4.0, 0.5),
        ] {
            let mut problem = Quadratic {
                x: 0.0,
                candidate: 0.0,
            };
            let config = LmConfig {
                max_iterations: 1,
                relative_tolerance: 0.0,
                absolute_tolerance: 0.0,
                initial_damping: 1.0,
                max_damping: 10.0,
            };
            let report = levenberg_marquardt(
                &mut problem,
                &config,
                DampingRule::Nielsen {
                    policy: NielsenPolicy {
                        min_ratio,
                        predicted_floor,
                        growth,
                    },
                },
            );
            assert_eq!(problem.x, 1.0);
            assert_eq!(report.damping, expected_damping);
        }
    }

    // r(x) = x² - 1, starting near its flat point: undamped steps overshoot.
    struct SquaredFactor;
    impl kornia_algebra::optim::Factor for SquaredFactor {
        fn linearize(
            &self,
            params: &[&[f32]],
            jacobian: bool,
        ) -> Result<kornia_algebra::optim::LinearizationResult, kornia_algebra::optim::FactorError>
        {
            let x = params[0][0];
            Ok(kornia_algebra::optim::LinearizationResult::new(
                vec![x * x - 1.0],
                jacobian.then(|| vec![2.0 * x]),
                1,
            ))
        }
        fn residual_dim(&self) -> usize {
            1
        }
        fn num_variables(&self) -> usize {
            1
        }
        fn variable_local_dim(&self, _: usize) -> usize {
            1
        }
    }

    struct DynamicProblem {
        x: f32,
        candidate: f32,
        trace: Vec<(f32, f32)>,
        lambdas: std::cell::RefCell<Vec<f32>>,
    }
    impl DynamicProblem {
        fn new(x: f32) -> Self {
            Self {
                x,
                candidate: x,
                trace: vec![],
                lambdas: Default::default(),
            }
        }
        fn cost(x: f32) -> f32 {
            (x * x - 1.0).powi(2)
        }
    }
    impl LmProblem<f32> for DynamicProblem {
        type Normal = (nalgebra::DMatrix<f32>, nalgebra::DVector<f32>);
        type Step = nalgebra::DVector<f32>;
        fn energy(&self) -> f32 {
            Self::cost(self.x)
        }
        fn linearize(&mut self) -> Self::Normal {
            let j = 2.0 * self.x;
            (
                nalgebra::DMatrix::from_element(1, 1, j * j),
                nalgebra::DVector::from_element(1, j * (self.x * self.x - 1.0)),
            )
        }
        fn solve_damped(&self, normal: &Self::Normal, lambda: f32) -> Option<Self::Step> {
            self.lambdas.borrow_mut().push(lambda);
            let mut h = normal.0.clone();
            h[(0, 0)] += lambda;
            h.lu().solve(&(-&normal.1))
        }
        fn predicted_reduction(&self, normal: &Self::Normal, step: &Self::Step) -> f32 {
            -(2.0 * normal.1.dot(step) + step.dot(&(&normal.0 * step)))
        }
        fn try_step(&mut self, step: Self::Step) -> f32 {
            self.candidate = self.x + step[0];
            Self::cost(self.candidate)
        }
        fn accept(&mut self) {
            self.x = self.candidate;
            self.trace.push((self.x, self.energy()));
        }
        fn reject(&mut self) {
            self.trace.push((self.x, self.energy()));
        }
    }

    #[test]
    fn levenberg_rule_reproduces_upstream_damping_sequence() {
        use kornia_algebra::optim::{LevenbergMarquardt, Problem, Variable};
        let mut upstream = Problem::new();
        upstream
            .add_variable(Variable::euclidean("x", 1), vec![0.1])
            .unwrap();
        upstream
            .add_factor(Box::new(SquaredFactor), vec!["x".into()])
            .unwrap();
        // A finite iterate prefix includes rejection, recovery, and accepted steps. Tolerance
        // stopping differs upstream: it keeps tiny cost increases, unlike handfit's driver.
        let optimizer = LevenbergMarquardt {
            max_iterations: 6,
            cost_tolerance: 0.0,
            gradient_tolerance: 0.0,
            ..Default::default()
        };
        let mut reference = vec![];
        let mut damping = vec![];
        let upstream_report = optimizer
            .optimize_with_callback(&mut upstream, |p, state| {
                if state.iteration > 0 {
                    reference.push((p.get_variables()["x"].values[0], state.cost));
                }
                damping.push(state.lambda);
                true
            })
            .unwrap();
        let mut problem = DynamicProblem::new(0.1);
        let limits = LmConfig {
            max_iterations: 6,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            initial_damping: 1e-3,
            max_damping: 1e10,
        };
        let result = levenberg_marquardt(
            &mut problem,
            &limits,
            DampingRule::Levenberg { factor: 10.0 },
        );
        assert_eq!(result.termination, LmTermination::Iterations);
        assert_eq!(
            upstream_report.termination_reason,
            kornia_algebra::optim::TerminationReason::MaxIterations
        );
        assert_eq!(result.iterations, upstream_report.iterations);
        assert_eq!(problem.trace, reference); // Exact f32 values, not an ULP tolerance.
        assert_eq!(*problem.lambdas.borrow(), damping[..6]);
        assert_eq!(
            result.final_energy.to_bits(),
            upstream_report.final_cost.to_bits()
        );
        assert_eq!(result.damping.to_bits(), damping[6].to_bits());
    }

    #[test]
    fn nielsen_rejection_streak_grows_then_resets() {
        let mut problem = DynamicProblem::new(0.1);
        let limits = LmConfig {
            max_iterations: 6,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            initial_damping: 1e-3,
            max_damping: 1e10,
        };
        let result = levenberg_marquardt(
            &mut problem,
            &limits,
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        assert_eq!(result.termination, LmTermination::Iterations);
        assert_eq!(
            &problem.lambdas.borrow()[..5],
            &[0.001, 0.002, 0.008, 0.064, 1.024]
        );
        assert!(problem.trace[..4].iter().all(|&(x, _)| x == 0.1));
        assert!(problem.trace[4].0 > 0.1);
        assert!(problem.lambdas.borrow()[5] < 1.024);
    }

    #[test]
    fn acceptance_resets_growth_before_the_next_rejection_streak() {
        let mut problem = DynamicProblem::new(0.1);
        let limits = LmConfig {
            max_iterations: 30,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            initial_damping: 1e-3,
            max_damping: 1e10,
        };
        levenberg_marquardt(
            &mut problem,
            &limits,
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        let exact = problem
            .trace
            .iter()
            .position(|&(_, energy)| energy == 0.0)
            .unwrap();
        let lambdas = problem.lambdas.borrow();
        // At the exact solution the next step has zero reduction and is rejected.
        // Its first rejection must double lambda, even after the earlier four rejections.
        assert_eq!(lambdas[exact + 2], 2.0 * lambdas[exact + 1]);
    }

    #[test]
    fn damping_cap_preserves_the_last_accepted_state() {
        let mut problem = DynamicProblem::new(0.1);
        let limits = LmConfig {
            max_iterations: 20,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            initial_damping: 1e-3,
            max_damping: 0.007,
        };
        let report = levenberg_marquardt(
            &mut problem,
            &limits,
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        assert_eq!(report.termination, LmTermination::Damping);
        assert_eq!(report.iterations, 2);
        assert_eq!(report.damping, 0.008);
        assert_eq!(problem.x, 0.1);
        assert_eq!(report.final_energy, DynamicProblem::cost(0.1));
    }

    #[test]
    fn zero_budget_and_nonfinite_initial_energy_do_not_evaluate_candidates() {
        let limits = LmConfig {
            max_iterations: 0,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            initial_damping: 1e-3,
            max_damping: 1e10,
        };
        let mut problem = Quadratic {
            x: 0.0,
            candidate: 42.0,
        };
        let report = levenberg_marquardt(
            &mut problem,
            &limits,
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        assert_eq!(report.termination, LmTermination::Iterations);
        assert_eq!(report.iterations, 0);
        assert_eq!(problem.candidate, 42.0);
        problem.x = f64::INFINITY;
        let report = levenberg_marquardt(
            &mut problem,
            &limits,
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        assert_eq!(report.termination, LmTermination::NonFinite);
        assert_eq!(report.iterations, 0);
    }

    #[test]
    fn nonfinite_trial_is_rejected_without_changing_the_retained_state() {
        let mut problem = DynamicProblem::new(1e-20);
        let limits = LmConfig {
            max_iterations: 10,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            initial_damping: 1e-38,
            max_damping: 1e10,
        };
        let report = levenberg_marquardt(
            &mut problem,
            &limits,
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        assert_eq!(report.termination, LmTermination::NonFinite);
        assert_eq!(report.iterations, 1);
        assert_eq!(problem.x, 1e-20);
        assert_eq!(report.final_energy, 1.0);
    }

    #[test]
    fn levenberg_damping_has_the_upstream_floor() {
        let mut problem = Quadratic {
            x: 0.0,
            candidate: 0.0,
        };
        let limits = LmConfig {
            max_iterations: 1,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            initial_damping: 1e-11,
            max_damping: 1e10,
        };
        let report = levenberg_marquardt(
            &mut problem,
            &limits,
            DampingRule::Levenberg { factor: 10.0 },
        );
        assert_eq!(report.damping, 1e-10);
    }

    #[test]
    fn stationary_at_the_cap_is_distinct_from_damping_exhaustion() {
        let mut problem = Quadratic {
            x: 2.0,
            candidate: 2.0,
        };
        let limits = LmConfig {
            max_iterations: 10,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            initial_damping: 1.0,
            max_damping: 1.0,
        };
        let report = levenberg_marquardt(
            &mut problem,
            &limits,
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        assert_eq!(report.termination, LmTermination::Stationary);
        assert_eq!(report.iterations, 1);
        assert_eq!(report.final_energy, 0.0);
    }

    struct Unsolvable;
    impl LmProblem<f64> for Unsolvable {
        type Normal = ();
        type Step = ();
        fn energy(&self) -> f64 {
            1.0
        }
        fn linearize(&mut self) {}
        fn solve_damped(&self, _: &(), _: f64) -> Option<()> {
            None
        }
        fn predicted_reduction(&self, _: &(), _: &()) -> f64 {
            unreachable!()
        }
        fn try_step(&mut self, _: ()) -> f64 {
            unreachable!()
        }
        fn accept(&mut self) {
            unreachable!()
        }
    }

    #[test]
    fn failed_solve_preserves_energy_and_candidate_count() {
        let limits = LmConfig {
            max_iterations: 10,
            relative_tolerance: 0.0,
            absolute_tolerance: 0.0,
            initial_damping: 1.0,
            max_damping: 1e10,
        };
        let report = levenberg_marquardt(
            &mut Unsolvable,
            &limits,
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        assert_eq!(report.termination, LmTermination::SolveFailed);
        assert_eq!(report.iterations, 0);
        assert_eq!(report.final_energy, 1.0);
    }
}
