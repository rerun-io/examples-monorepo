//! Seeded RANSAC around [`gp3p`](super::gp3p).
//!
//! [`draw_hypotheses`] is upstream `kornia_3d::ransac::run_with_rng`'s plain
//! serial path (no SPRT, no LO-RANSAC) with an inlier-count consensus on the
//! `1 - cos` bearing residual, so it keeps upstream's adaptive iteration cap
//! (`m >= log(1 - p) / log(1 - w^k)`, tightened as better models land). It is a
//! copy, checked draw for draw against upstream under the `kornia-3d-interop`
//! feature (`tests/gp3p_ransac.rs`), because kornia-3d depends on faer, which
//! slam-rs removed for an aarch64 LLVM crash; the `ransac` feature needs only
//! `rand`. Every random draw comes from one seeded [`ChaCha8Rng`]: the same
//! input and seed give the same pose and inliers, on any thread count.
//!
//! RootSLAM (`loop_closing.cpp:583-605`) runs OpenGV's `Ransac` with a sample of
//! four (three for GP3P, the fourth picks among its solutions), confidence 0.99
//! and `max_iterations_` 20-35, then refines on the inliers and re-selects them.
//! OpenGV's count is adaptive too, but the cap binds (bug B15). At 30 % inliers
//! a sample of four is all-inlier with probability 0.8 %, so 36 draws contain
//! one only 25 % of the time; that is a lower bound, because a sample whose
//! three GP3P points are inliers often succeeds when the fourth is an outlier
//! (the outlier then picks among solutions that include the truth). Measured
//! on 100 synthetic scenes at w = 0.3 (`tests/gp3p_ransac.rs`), the parity
//! schedule finds the pose in about half of them, the adaptive one in all.
//! [`IterationSchedule::Adaptive`] samples three, scores every solution, and caps
//! at 500 with `p = 0.999`; [`IterationSchedule::RootSlamParity`] keeps OpenGV's
//! sampling, confidence and cap for the rootslam-parity profile.
//!
//! A caller that rejects loops below an inlier ratio (RootSLAM's
//! `pnp_inliers_ratio`, DESIGN B1) passes it as
//! [`Gp3pRansacConfig::min_inlier_ratio`]: the draws then stop once an
//! acceptable model would have been found with the schedule's confidence, so a
//! rejected candidate costs about 52 draws at 0.5 (17 at 0.7) instead of the cap.

use super::{refine, solve, validate, GeneralizedCorrespondence, Gp3pError, Gp3pPose, Solutions};
use kornia_staging_algebra::lie::RigidTransform;
use rand::rngs::ChaCha8Rng;
use rand::SeedableRng;

/// RootSLAM's `loop_closing_pnp_ransac_threshold`: `1 - cos(angle)`, about
/// 0.57° (10 mrad).
pub const ROOTSLAM_THRESHOLD: f64 = 5e-5;
/// RootSLAM's `loop_closing_pnp_ransac_iterations` on MSD and EuRoC (20 on TUM-VI).
pub const ROOTSLAM_ITERATIONS: u32 = 35;

/// How many hypotheses RANSAC draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IterationSchedule {
    /// Minimal samples of three, every solution scored; stop once a model has
    /// been found with probability `confidence`, never after `max_iterations`.
    Adaptive {
        /// Probability of drawing at least one all-inlier sample.
        confidence: f64,
        /// Hard cap on hypotheses.
        max_iterations: u32,
    },
    /// OpenGV as RootSLAM calls it: samples of four, the fourth correspondence
    /// picks among the GP3P solutions, confidence 0.99, and at most
    /// `iterations + 1` draws (OpenGV breaks on `iterations_ > max_iterations_`).
    /// For the rootslam-parity profile only.
    ///
    /// Two known differences from OpenGV's draw count: here every draw counts,
    /// also one whose triple has no solution, where OpenGV skips a draw whose
    /// `computeModelCoefficients` fails (up to `10 x max_iterations_` skips)
    /// without counting it; and the solver drops roots with a negative depth,
    /// which OpenGV keeps and scores. Draws without a solution are frequent
    /// (40-57 % of random triples at w = 0.3), so OpenGV evaluates somewhat
    /// more hypotheses for the same cap.
    RootSlamParity {
        /// `loop_closing_pnp_ransac_iterations`.
        iterations: u32,
    },
}

impl Default for IterationSchedule {
    /// `p = 0.999`, cap 500 (DESIGN B15).
    fn default() -> Self {
        Self::Adaptive {
            confidence: 0.999,
            max_iterations: 500,
        }
    }
}

impl IterationSchedule {
    /// RootSLAM's MSD setting.
    pub fn rootslam_parity() -> Self {
        Self::RootSlamParity {
            iterations: ROOTSLAM_ITERATIONS,
        }
    }
}

/// Configuration of [`gp3p_ransac`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gp3pRansacConfig {
    /// Inlier threshold on `1 - cos(angle)` between bearing and point direction.
    pub threshold: f64,
    /// How many hypotheses to draw.
    pub schedule: IterationSchedule,
    /// RNG seed: same seed and input, same result.
    pub seed: u64,
    /// Refine on the inliers and re-select them, as RootSLAM does.
    pub refine: bool,
    /// The smallest inlier ratio the caller accepts (`None`: no bound). The
    /// draws stop after `ceil(log(1 - p) / log(1 - w_min^k))` (sample size `k`
    /// and confidence `p` of the schedule) when that is below the schedule's
    /// cap: by then a model with `w_min` inliers would have been drawn with
    /// probability `p`, and a model below `w_min` is rejected by the caller.
    /// Must lie in `(0, 1]`. The rootslam-parity profile passes `None`.
    pub min_inlier_ratio: Option<f64>,
}

impl Default for Gp3pRansacConfig {
    fn default() -> Self {
        Self {
            threshold: ROOTSLAM_THRESHOLD,
            schedule: IterationSchedule::default(),
            seed: 0,
            refine: true,
            min_inlier_ratio: None,
        }
    }
}

/// The result of [`gp3p_ransac`].
#[derive(Debug, Clone, PartialEq)]
pub struct Gp3pRansacResult {
    /// `T_w_b` of the body.
    pub pose: RigidTransform<f64>,
    /// Inlier mask over the input correspondences, under `pose`.
    pub inliers: Vec<bool>,
    /// Number of `true` entries in `inliers`.
    pub num_inliers: usize,
    /// Hypotheses drawn.
    pub iterations: u32,
}

/// The GP3P hypothesis generator. `SAMPLE == 3` returns every GP3P solution;
/// `SAMPLE == 4` keeps the one that fits the fourth correspondence best
/// (OpenGV `AbsolutePoseSacProblem`). Under the `kornia-3d-interop` feature it
/// also implements upstream's `kornia_3d::ransac::Estimator`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gp3pEstimator<const SAMPLE: usize>;

fn fit_triple(samples: &[GeneralizedCorrespondence]) -> Solutions {
    let mut solutions = Solutions::default();
    let triple = [samples[0], samples[1], samples[2]];
    if !super::collinear(&triple[0].point, &triple[1].point, &triple[2].point) {
        solve(&triple, &mut solutions);
    }
    solutions
}

impl Gp3pEstimator<3> {
    /// Every GP3P solution of the first three samples, appended to `out`.
    pub fn fit(&self, samples: &[GeneralizedCorrespondence], out: &mut Vec<Gp3pPose>) {
        out.extend_from_slice(fit_triple(samples).as_slice());
    }
}

impl Gp3pEstimator<4> {
    /// The GP3P solution of the first three samples that fits the fourth best,
    /// appended to `out` (none when the triple has no solution).
    pub fn fit(&self, samples: &[GeneralizedCorrespondence], out: &mut Vec<Gp3pPose>) {
        let solutions = fit_triple(samples);
        let best = solutions.as_slice().iter().min_by(|a, b| {
            samples[3]
                .angular_residual(a)
                .total_cmp(&samples[3].angular_residual(b))
        });
        out.extend(best.copied());
    }
}

#[cfg(feature = "kornia-3d-interop")]
mod interop {
    use super::{GeneralizedCorrespondence, Gp3pEstimator, Gp3pPose};
    use kornia_3d::ransac::Estimator;

    impl Estimator for Gp3pEstimator<3> {
        type Model = Gp3pPose;
        type Sample = GeneralizedCorrespondence;
        const SAMPLE_SIZE: usize = 3;

        fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
            Gp3pEstimator::<3>::fit(self, samples, out);
        }

        fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
            sample.angular_residual(model)
        }

        /// No refit inside the driver: [`super::gp3p_ransac`] refines after it.
        fn refit(&self, _inliers: &[Self::Sample], _out: &mut Vec<Self::Model>) {}
    }

    impl Estimator for Gp3pEstimator<4> {
        type Model = Gp3pPose;
        type Sample = GeneralizedCorrespondence;
        const SAMPLE_SIZE: usize = 4;

        fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
            Gp3pEstimator::<4>::fit(self, samples, out);
        }

        fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
            sample.angular_residual(model)
        }

        /// See the sample-of-three estimator.
        fn refit(&self, _inliers: &[Self::Sample], _out: &mut Vec<Self::Model>) {}
    }
}

/// The best hypothesis of [`draw_hypotheses`].
#[derive(Debug, Clone, PartialEq)]
pub struct BestHypothesis {
    /// The model with the most inliers (the first one drawn on a tie).
    pub pose: Gp3pPose,
    /// Its inlier mask over the input.
    pub inliers: Vec<bool>,
    /// Hypotheses drawn.
    pub iterations: u32,
}

/// Serial RANSAC: draw `sample` distinct correspondences with a [`ChaCha8Rng`]
/// seeded from `seed` (`rand::seq::index::sample`), fit GP3P
/// ([`Gp3pEstimator`]), count the inliers (`1 - cos < threshold`) of every
/// solution, keep the strictly better one, and tighten `max_iters` to
/// `ceil(log(1 - p) / log(1 - w^sample))` after each better model. Upstream
/// `kornia_3d::ransac::run_with_rng` with SPRT and LO off, draw for draw.
/// `None` when fewer than `sample` correspondences or no draw gave a model.
///
/// # Panics
///
/// If `sample` is not 3 or 4.
pub fn draw_hypotheses(
    correspondences: &[GeneralizedCorrespondence],
    sample: usize,
    threshold: f64,
    confidence: f64,
    max_iters: u32,
    seed: u64,
) -> Option<BestHypothesis> {
    assert!(sample == 3 || sample == 4, "GP3P samples 3 or 4 rays");
    let n = correspondences.len();
    if n < sample || max_iters == 0 {
        return None;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut drawn: Vec<GeneralizedCorrespondence> = Vec::with_capacity(sample);
    let mut models: Vec<Gp3pPose> = Vec::with_capacity(super::MAX_SOLUTIONS);
    let mut inliers: Vec<bool> = Vec::with_capacity(n);
    let mut best: Option<(Gp3pPose, Vec<bool>)> = None;
    let mut best_count: Option<usize> = None;
    let mut cap = max_iters;
    let mut iterations: u32 = 0;
    while iterations < cap {
        drawn.clear();
        drawn.extend(
            rand::seq::index::sample(&mut rng, n, sample)
                .iter()
                .map(|index| correspondences[index]),
        );
        models.clear();
        if sample == 3 {
            Gp3pEstimator::<3>.fit(&drawn, &mut models);
        } else {
            Gp3pEstimator::<4>.fit(&drawn, &mut models);
        }
        for model in &models {
            inliers.clear();
            inliers.extend(
                correspondences
                    .iter()
                    .map(|c| c.angular_residual(model) < threshold),
            );
            let count = inliers.iter().filter(|&&is_in| is_in).count();
            if best_count.is_none_or(|best| count > best) {
                best_count = Some(count);
                best = Some((*model, inliers.clone()));
                if count > 0 {
                    cap = adaptive_cap(count as f64 / n as f64, sample, confidence, cap);
                }
            }
        }
        iterations += 1;
    }
    best.map(|(pose, inliers)| BestHypothesis {
        pose,
        inliers,
        iterations,
    })
}

/// Upstream's `adaptive_max_iters`: the draws that contain an all-inlier sample
/// with probability `confidence` at inlier ratio `w`, never above `current`.
fn adaptive_cap(w: f64, sample: usize, confidence: f64, current: u32) -> u32 {
    let all_inlier = w.powi(sample as i32);
    if all_inlier >= 1.0 {
        return 1;
    }
    let denominator = (1.0 - all_inlier).ln();
    if denominator >= 0.0 || !denominator.is_finite() {
        return current;
    }
    let raw = (1.0 - confidence.clamp(0.0, 1.0 - 1e-12)).ln() / denominator;
    if !raw.is_finite() || raw <= 0.0 {
        return current;
    }
    // Saturating float-to-int cast.
    (raw.ceil() as u32).min(current).max(1)
}

/// Robust generalized absolute pose: seeded RANSAC over GP3P, then (optionally)
/// refinement on the inliers and inlier re-selection under the refined pose.
///
/// Returns `Ok(None)` when no hypothesis was found (every sample degenerate).
///
/// # Errors
/// [`Gp3pError::TooFewCorrespondences`] below four correspondences (three for
/// the solver, one to tell its solutions apart); [`Gp3pError::NonFinite`] /
/// [`Gp3pError::NonUnitBearing`] for malformed input.
pub fn gp3p_ransac(
    correspondences: &[GeneralizedCorrespondence],
    config: &Gp3pRansacConfig,
) -> Result<Option<Gp3pRansacResult>, Gp3pError> {
    if correspondences.len() < 4 {
        return Err(Gp3pError::TooFewCorrespondences {
            got: correspondences.len(),
            needed: 4,
        });
    }
    validate(correspondences)?;
    if config
        .min_inlier_ratio
        .is_some_and(|w| !(w > 0.0 && w <= 1.0))
    {
        return Err(Gp3pError::InvalidMinInlierRatio);
    }
    let (confidence, cap, sample) = match config.schedule {
        IterationSchedule::Adaptive {
            confidence,
            max_iterations,
        } => (confidence, max_iterations, 3),
        IterationSchedule::RootSlamParity { iterations } => (0.99, iterations.saturating_add(1), 4),
    };
    let max_iters = config
        .min_inlier_ratio
        .map_or(cap, |w| cap.min(draws_for(w, sample as i32, confidence)));
    let Some(best) = draw_hypotheses(
        correspondences,
        sample,
        config.threshold,
        confidence,
        max_iters,
        config.seed,
    ) else {
        return Ok(None);
    };
    let mut pose = best.pose;
    let mut inliers = best.inliers;
    if config.refine {
        let selected: Vec<GeneralizedCorrespondence> = correspondences
            .iter()
            .zip(&inliers)
            .filter(|(_, &is_in)| is_in)
            .map(|(c, _)| *c)
            .collect();
        pose = refine(&selected, &pose);
        inliers = correspondences
            .iter()
            .map(|c| c.angular_residual(&pose) < config.threshold)
            .collect();
    }
    Ok(Some(Gp3pRansacResult {
        pose: pose.to_rigid_transform(),
        num_inliers: inliers.iter().filter(|&&b| b).count(),
        inliers,
        iterations: best.iterations,
    }))
}

/// Draws that contain an all-inlier sample of `sample` points with probability
/// `confidence` at inlier ratio `w`: `ceil(log(1 - p) / log(1 - w^k))`, at least
/// one (upstream's adaptive formula).
fn draws_for(w: f64, sample: i32, confidence: f64) -> u32 {
    let all_inlier = w.powi(sample);
    if all_inlier >= 1.0 {
        return 1;
    }
    let draws = (1.0 - confidence).ln() / (1.0 - all_inlier).ln();
    if draws.is_finite() {
        // Saturating float-to-int cast.
        draws.ceil().max(1.0) as u32
    } else {
        u32::MAX
    }
}
