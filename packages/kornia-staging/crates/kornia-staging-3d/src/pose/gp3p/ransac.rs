//! Seeded RANSAC around [`gp3p`](super::gp3p), on upstream `kornia_3d::ransac`.
//!
//! The estimator implements upstream's [`Estimator`] trait and runs in its
//! serial driver ([`run_with_rng`]) with [`ThresholdConsensus`] on the
//! `1 - cos` bearing residual, so it inherits upstream's adaptive iteration cap
//! (`m >= log(1 - p) / log(1 - w^k)`, tightened as better models land). The
//! driver is serial and every random draw comes from one seeded
//! [`ChaCha8Rng`]: the same input and seed give the same pose and inliers, on
//! any thread count.
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
//! at 500 with `p = 0.999`; [`IterationSchedule::OpenGv`] keeps OpenGV's
//! sampling, confidence and cap, for a caller that needs OpenGV's draws.
//!
//! A caller that rejects loops below an inlier ratio (RootSLAM's
//! `pnp_inliers_ratio`) passes it as
//! [`Gp3pRansacConfig::min_inlier_ratio`]: the draws then stop once an
//! acceptable model would have been found with the schedule's confidence, so a
//! rejected candidate costs about 52 draws at 0.5 (17 at 0.7) instead of the cap.

use super::{refine, solve, validate, GeneralizedCorrespondence, Gp3pError, Gp3pPose, Solutions};
use kornia_3d::ransac::{
    run_with_rng, Estimator, RansacConfig, ThresholdConsensus, UniformSampler,
};
use kornia_staging_algebra::lie::RigidTransform;
use rand::rngs::ChaCha8Rng;
use rand::SeedableRng;

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
    ///
    /// Two known differences from OpenGV's draw count: here every draw counts,
    /// also one whose triple has no solution, where OpenGV skips a draw whose
    /// `computeModelCoefficients` fails (up to `10 x max_iterations_` skips)
    /// without counting it; and the solver drops roots with a negative depth,
    /// which OpenGV keeps and scores. Draws without a solution are frequent
    /// (40-57 % of random triples at w = 0.3), so OpenGV evaluates somewhat
    /// more hypotheses for the same cap.
    OpenGv {
        /// OpenGV's `max_iterations_`.
        iterations: u32,
    },
}

impl Default for IterationSchedule {
    /// `p = 0.999`, cap 500 (OpenGV's 20-35 binds: see the module doc).
    fn default() -> Self {
        Self::Adaptive {
            confidence: 0.999,
            max_iterations: 500,
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
    /// Must lie in `(0, 1]`.
    pub min_inlier_ratio: Option<f64>,
}

impl Default for Gp3pRansacConfig {
    fn default() -> Self {
        Self {
            // `1 - cos(angle)`, about 0.57° (10 mrad): RootSLAM's loop-closing value.
            threshold: 5e-5,
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

/// [`Estimator`] for generalized absolute pose. `SAMPLE == 3` scores every GP3P
/// solution; `SAMPLE == 4` keeps the one that fits the fourth correspondence
/// best (OpenGV `AbsolutePoseSacProblem`).
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

impl Estimator for Gp3pEstimator<3> {
    type Model = Gp3pPose;
    type Sample = GeneralizedCorrespondence;
    const SAMPLE_SIZE: usize = 3;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        out.extend_from_slice(fit_triple(samples).as_slice());
    }

    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        sample.angular_residual(model)
    }

    /// No refit inside the driver: [`gp3p_ransac`] refines after it, on the
    /// inliers, and keeps the refined pose unconditionally (as RootSLAM does).
    fn refit(&self, _inliers: &[Self::Sample], _out: &mut Vec<Self::Model>) {}
}

impl Estimator for Gp3pEstimator<4> {
    type Model = Gp3pPose;
    type Sample = GeneralizedCorrespondence;
    const SAMPLE_SIZE: usize = 4;

    fn fit(&self, samples: &[Self::Sample], out: &mut Vec<Self::Model>) {
        let solutions = fit_triple(samples);
        let best = solutions.as_slice().iter().min_by(|a, b| {
            samples[3]
                .angular_residual(a)
                .total_cmp(&samples[3].angular_residual(b))
        });
        out.extend(best.copied());
    }

    fn residual(&self, model: &Self::Model, sample: &Self::Sample) -> f64 {
        sample.angular_residual(model)
    }

    /// See [`Gp3pEstimator::<3>::refit`].
    fn refit(&self, _inliers: &[Self::Sample], _out: &mut Vec<Self::Model>) {}
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
    let consensus = ThresholdConsensus {
        threshold: config.threshold,
    };
    let mut sampler = UniformSampler::new(ChaCha8Rng::seed_from_u64(config.seed));
    // The driver's own RNG only feeds SPRT, which is off; seed it anyway.
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
    let (confidence, cap, sample) = match config.schedule {
        IterationSchedule::Adaptive {
            confidence,
            max_iterations,
        } => (confidence, max_iterations, 3),
        IterationSchedule::OpenGv { iterations } => (0.99, iterations.saturating_add(1), 4),
    };
    let max_iters = config
        .min_inlier_ratio
        .map_or(cap, |w| cap.min(draws_for(w, sample, confidence)));
    let ransac = RansacConfig {
        max_iters,
        confidence,
        inlier_threshold: config.threshold,
        ..RansacConfig::default()
    };
    let result = match config.schedule {
        IterationSchedule::Adaptive { .. } => run_with_rng(
            &Gp3pEstimator::<3>,
            &consensus,
            &mut sampler,
            correspondences,
            &ransac,
            &mut rng,
        ),
        IterationSchedule::OpenGv { .. } => run_with_rng(
            &Gp3pEstimator::<4>,
            &consensus,
            &mut sampler,
            correspondences,
            &ransac,
            &mut rng,
        ),
    };
    let Some(mut pose) = result.model else {
        return Ok(None);
    };
    let mut inliers = result.inliers;
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
        iterations: result.num_iters,
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
