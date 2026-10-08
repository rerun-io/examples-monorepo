//! Hand-scale calibration (MEgATrack §3.6), the port of handtrack's `fit/scale.py::calibrate_scale`: one scale phi, shared by both
//! hands, from observations seen in stereo.
//!
//! min over θ_1..θ_n and phi of Σ_t E_2D(θ_t, phi), where the model is the generic model with its rest geometry scaled by phi. The
//! Jacobian is block-diagonal in the θ_t plus one dense column for phi, so each LM step eliminates the θ_t blocks (Schur complement
//! on phi): one 26x26 solve per block and a scalar equation for phi. The joint solve lets the optimiser trade the hand's distance
//! against its size. Solver rules (damping, active set at the joint limits, stopping) are handtrack's.
//!
//! The pose columns of the Jacobian come from handfit's analytic residual Jacobian; the phi column is analytic too: the skinned
//! landmarks are linear in the rest geometry's scale, so d p_world / d phi = (p_world − t) / phi. handtrack takes central differences
//! in float32 instead; the two agree to the solver's tolerance (see the golden test).
#![deny(missing_docs)]

use nalgebra::{SMatrix, SVector, SymmetricEigen};

pub use crate::lm::Termination;
use crate::lm::{nielsen_damping, predicted_reduction, DAMPING_GROWTH, MAX_DAMPING};
use crate::model::{orthonormalize, retract, Step};
use crate::residual::{evaluate, project_camera, Residual, View, Views};
use crate::{Config, Model, Pose};

const LANDMARKS: usize = 21;
const POSE: usize = 26;
const ROWS_2D: usize = 84;

/// Settings of the joint LM (handtrack `CalibrationConfig`).
#[derive(Clone, Debug)]
pub struct CalibrationConfig {
    /// LM iterations.
    pub iterations: usize,
    /// Stop when an accepted step lowers Σ E_2D by less than this fraction of it.
    pub relative_tolerance: f64,
    /// λ at the first iteration, relative to each block's diag(JᵀJ).
    pub initial_damping: f64,
    /// A view counts towards stereo when at least this many of its keypoints have weight > 0.
    pub min_view_keypoints: usize,
    /// The joint limits are widened by this much (handtrack `FitConfig.joint_limit_margin_rad`).
    pub joint_limit_margin_rad: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            iterations: 30,
            relative_tolerance: 1e-5,
            initial_damping: 1e-3,
            min_view_keypoints: 10,
            joint_limit_margin_rad: 5.0_f64.to_radians(),
        }
    }
}

/// One observation of a hand: its views (as the fit saw them) and the tracker's pose at phi = 1.
#[derive(Clone)]
pub struct CalibrationBlock {
    /// +1 left hand, −1 right hand.
    pub mirror: f64,
    /// At most [`crate::residual::MAX_VIEWS`] views; only an observation with two of them can enter the solve.
    pub views: Vec<View>,
    /// The tracker's fitted pose (the solve's start).
    pub initial: Pose,
}

/// The calibrated scale and how the solve went (handtrack `ScaleCalibration`).
#[derive(Clone, Debug)]
pub struct ScaleCalibration {
    /// The hand's scale relative to the generic hand (NaN when the energy became non-finite).
    pub phi: f64,
    /// θ_t of every stereo observation used.
    pub poses: Vec<Pose>,
    /// Indices of those observations in the input.
    pub used: Vec<usize>,
    /// Σ_t E_2D at the solution, pixels².
    pub e_2d: f64,
    /// Stereo observations in the solve.
    pub blocks: usize,
    /// LM iterations run.
    pub iterations: usize,
    /// Tolerance reached or stationary.
    pub converged: bool,
    /// Why the solve stopped.
    pub termination: Termination,
}

/// Why no scale came out.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ScaleError {
    /// No observation had two views with enough keypoints.
    #[error("no observation is seen in stereo")]
    NoStereo,
    /// Invalid views in one observation.
    #[error("observation {observation}: {source}")]
    Views {
        /// Its index in the input.
        observation: usize,
        /// View admission error.
        source: crate::residual::ViewValidationError,
    },
    /// The scale left the finite numbers.
    #[error("the scale calibration diverged")]
    Diverged,
}

type Jacobian = SMatrix<f64, ROWS_2D, 27>;
type Hessian = SMatrix<f64, 27, 27>;
type PoseHessian = SMatrix<f64, POSE, POSE>;

/// The 2D residual of one block (its `mirror` and `views`) at `pose` and `scale`, with its Jacobian over the 26 pose
/// parameters and the scale.
fn linearize(
    generic: &Model,
    scale: f64,
    pose: &Pose,
    (mirror, views): (f64, Views<'_>),
) -> (SVector<f64, ROWS_2D>, Jacobian) {
    let model = Model {
        pivots: generic.pivots * scale,
        rest: generic.rest * scale,
        ..generic.clone()
    };
    let config = Config {
        phi: 1.0,
        dist_weight: 0.0,
        temporal_weight: 0.0,
        translation_unit: 1.0,
        ..Config::default()
    };
    let mut full = crate::residual::Jacobian::zeros();
    let residual: Residual = evaluate(
        &model,
        &config,
        pose,
        pose,
        mirror,
        views,
        &Step::zeros(),
        Some(&mut full),
    );
    let mut jacobian = Jacobian::zeros();
    jacobian
        .fixed_view_mut::<ROWS_2D, POSE>(0, 0)
        .tr_copy_from(&full.fixed_view::<POSE, ROWS_2D>(0, 0));
    let points = model.landmarks(pose, mirror, &Step::zeros(), None);
    for (v, view) in views.iter().enumerate() {
        for i in 0..LANDMARKS {
            if view.weights[i] <= 0.0 {
                continue;
            }
            let world = points.fixed_rows::<3>(3 * i).into_owned();
            let point = view.rotation * world + view.translation;
            let d_point = view.rotation * ((world - pose.translation) / scale);
            let mut projection = SMatrix::<f64, 2, 3>::zeros();
            project_camera(views.camera(v), &point, Some(&mut projection));
            let column = projection * d_point * view.weights[i].sqrt();
            jacobian[(42 * v + 2 * i, POSE)] = column[0];
            jacobian[(42 * v + 2 * i + 1, POSE)] = column[1];
        }
    }
    (residual.fixed_rows::<ROWS_2D>(0).into_owned(), jacobian)
}

/// JᵀJ and Jᵀr with the rows and columns of held joint angles zeroed, and the free mask (handtrack `normal_equations`).
fn normal_equations(
    pose: &Pose,
    r: &SVector<f64, ROWS_2D>,
    j: &Jacobian,
    limits: &SMatrix<f64, 20, 2>,
) -> (Hessian, SVector<f64, 27>, SVector<f64, 27>) {
    let mut h: Hessian = j.transpose() * j;
    let mut g: SVector<f64, 27> = j.transpose() * r;
    let mut mask = SVector::<f64, 27>::repeat(1.0);
    for joint in 0..20 {
        let k = 6 + joint;
        let angle = pose.angles[joint];
        if (angle <= limits[(joint, 0)] + 1e-6 && g[k] > 0.0)
            || (angle >= limits[(joint, 1)] - 1e-6 && g[k] < 0.0)
        {
            mask[k] = 0.0;
        }
    }
    for a in 0..27 {
        g[a] *= mask[a];
        for b in 0..27 {
            h[(a, b)] *= mask[a] * mask[b];
        }
    }
    (h, g, mask)
}

/// D of (JᵀJ + λD)δ = −Jᵀr: diag(JᵀJ) floored, 1 for held parameters (handtrack `marquardt_scaling`).
fn marquardt_scaling(
    diagonal: &SVector<f64, POSE>,
    mask: &SVector<f64, POSE>,
) -> SVector<f64, POSE> {
    let floor = 1e-9 * diagonal.max().max(1e-12);
    SVector::from_fn(|i, _| diagonal[i].max(floor) * mask[i] + (1.0 - mask[i]))
}

/// The pseudo-inverse of the column-normalised pose block (handtrack `undamped_inverse`).
fn undamped_inverse(h: &PoseHessian, mask: &SVector<f64, POSE>) -> PoseHessian {
    let units = marquardt_scaling(&h.diagonal(), mask).map(|x| 1.0 / x.sqrt());
    let normalized = PoseHessian::from_fn(|i, k| h[(i, k)] * units[i] * units[k]);
    let eigen = SymmetricEigen::new(normalized);
    let threshold = POSE as f64 * f64::EPSILON * eigen.eigenvalues.abs().max();
    let inverse = eigen
        .eigenvalues
        .map(|x| if x.abs() > threshold { 1.0 / x } else { 0.0 });
    let pinv =
        eigen.eigenvectors * PoseHessian::from_diagonal(&inverse) * eigen.eigenvectors.transpose();
    PoseHessian::from_fn(|i, k| pinv[(i, k)] * units[i] * units[k])
}

struct Linearized {
    residuals: Vec<SVector<f64, ROWS_2D>>,
    jacobians: Vec<Jacobian>,
    energy: f64,
}

fn linearize_all(
    generic: &Model,
    scale: f64,
    poses: &[Pose],
    blocks: &[(f64, Views<'_>)],
) -> Linearized {
    let (residuals, jacobians): (Vec<_>, Vec<_>) = poses
        .iter()
        .zip(blocks)
        .map(|(pose, &block)| linearize(generic, scale, pose, block))
        .unzip();
    let energy = residuals.iter().map(|r| r.norm_squared()).sum();
    Linearized {
        residuals,
        jacobians,
        energy,
    }
}

/// One Schur-complement step for every block: the pose steps and the shared scale step, from damped (or pseudo-inverted) pose blocks.
struct Normal {
    h: Vec<Hessian>,
    g: Vec<SVector<f64, 27>>,
    mask: Vec<SVector<f64, 27>>,
}

fn normal_all(poses: &[Pose], lin: &Linearized, limits: &SMatrix<f64, 20, 2>) -> Normal {
    let mut normal = Normal {
        h: Vec::with_capacity(poses.len()),
        g: Vec::with_capacity(poses.len()),
        mask: Vec::with_capacity(poses.len()),
    };
    for ((pose, r), j) in poses.iter().zip(&lin.residuals).zip(&lin.jacobians) {
        let (h, g, mask) = normal_equations(pose, r, j, limits);
        normal.h.push(h);
        normal.g.push(g);
        normal.mask.push(mask);
    }
    normal
}

/// The pose-block solves of a Schur step, `(H_pp⁻¹ c, H_pp⁻¹ g)` per block, and the sums of the scalar scale equation.
struct Schur {
    solved: Vec<(SVector<f64, POSE>, SVector<f64, POSE>)>,
    /// Σ H_φφ.
    curvature: f64,
    /// Σ g_φ.
    gradient: f64,
    /// Σ cᵀ H_pp⁻¹ c.
    coupling: f64,
    /// Σ cᵀ H_pp⁻¹ g.
    coupled_gradient: f64,
}

type BlockSolve = (SVector<f64, POSE>, SVector<f64, POSE>);

/// Solve every pose block with `solve(pose block, free mask, coupling c, gradient g)` and sum the scale equation's terms;
/// `None` when a block cannot be solved.
fn schur(
    normal: &Normal,
    mut solve: impl FnMut(
        &PoseHessian,
        &SVector<f64, POSE>,
        &SVector<f64, POSE>,
        &SVector<f64, POSE>,
    ) -> Option<BlockSolve>,
) -> Option<Schur> {
    let mut out = Schur {
        solved: Vec::with_capacity(normal.h.len()),
        curvature: 0.0,
        gradient: 0.0,
        coupling: 0.0,
        coupled_gradient: 0.0,
    };
    for b in 0..normal.h.len() {
        let (pose_block, coupling, gradient, pose_mask) =
            pose_parts(&normal.h[b], &normal.g[b], &normal.mask[b]);
        let (s0, s1) = solve(&pose_block, &pose_mask, &coupling, &gradient)?;
        out.curvature += normal.h[b][(POSE, POSE)];
        out.gradient += normal.g[b][POSE];
        out.coupling += coupling.dot(&s0);
        out.coupled_gradient += coupling.dot(&s1);
        out.solved.push((s0, s1));
    }
    Some(out)
}

/// Each block's 27-parameter step for a scale step: the pose part `−(H_pp⁻¹ g + H_pp⁻¹ c · Δφ)` and `Δφ`.
fn block_steps(solved: &[BlockSolve], scale_step: f64) -> Vec<SVector<f64, 27>> {
    solved
        .iter()
        .map(|(s0, s1)| {
            let pose_step = -(s1 + s0 * scale_step);
            SVector::from_fn(|i, _| if i < POSE { pose_step[i] } else { scale_step })
        })
        .collect()
}

fn pose_parts(
    h: &Hessian,
    g: &SVector<f64, 27>,
    mask: &SVector<f64, 27>,
) -> (
    PoseHessian,
    SVector<f64, POSE>,
    SVector<f64, POSE>,
    SVector<f64, POSE>,
) {
    (
        h.fixed_view::<POSE, POSE>(0, 0).into_owned(),
        h.fixed_view::<POSE, 1>(0, POSE).into_owned(),
        g.fixed_rows::<POSE>(0).into_owned(),
        mask.fixed_rows::<POSE>(0).into_owned(),
    )
}

/// Solve for one phi shared by every stereo observation in `hands` and a θ_t per observation (handtrack `calibrate_scale`).
///
/// # Arguments
///
/// * `generic` - The model to scale (the generic hand at phi = 1).
/// * `hands` - Observations of either hand in any frames, each with at most [`crate::residual::MAX_VIEWS`] views; only those with two views of
///   `min_view_keypoints` observed keypoints enter the solve.
/// * `config` - Solver settings.
///
/// # Returns
///
/// The calibration; `phi` is NaN when the energy became non-finite.
///
/// # Errors
///
/// [`ScaleError::Views`] when an observation has more than [`crate::residual::MAX_VIEWS`] views, [`ScaleError::NoStereo`] when no
/// observation is seen in stereo, [`ScaleError::Diverged`] when phi is not finite.
pub fn calibrate_scale(
    generic: &Model,
    hands: &[CalibrationBlock],
    config: &CalibrationConfig,
) -> Result<ScaleCalibration, ScaleError> {
    let views: Vec<Views<'_>> = hands
        .iter()
        .enumerate()
        .map(|(observation, hand)| {
            Views::new(&hand.views).map_err(|source| ScaleError::Views { observation, source })
        })
        .collect::<Result<_, _>>()?;
    let used: Vec<usize> = (0..hands.len())
        .filter(|&i| {
            hands[i]
                .views
                .iter()
                .filter(|view| {
                    view.weights.iter().filter(|w| **w > 0.0).count() >= config.min_view_keypoints
                })
                .count()
                >= 2
        })
        .collect();
    if used.is_empty() {
        return Err(ScaleError::NoStereo);
    }
    let blocks: Vec<(f64, Views<'_>)> = used.iter().map(|&i| (hands[i].mirror, views[i])).collect();
    let limits = SMatrix::<f64, 20, 2>::from_fn(|i, k| {
        generic.limits[(i, k)]
            + if k == 0 {
                -config.joint_limit_margin_rad
            } else {
                config.joint_limit_margin_rad
            }
    });
    let mut poses: Vec<Pose> = used
        .iter()
        .map(|&i| {
            let mut pose = hands[i].initial.clone();
            for joint in 0..20 {
                pose.angles[joint] =
                    pose.angles[joint].clamp(limits[(joint, 0)], limits[(joint, 1)]);
            }
            pose
        })
        .collect();
    let mut scale = 1.0;
    let mut lin = linearize_all(generic, scale, &poses, &blocks);
    let mut damping = config.initial_damping;
    let mut growth = DAMPING_GROWTH;
    let mut converged = false;
    let mut termination = if lin.energy.is_finite() {
        Termination::Iterations
    } else {
        Termination::NonFinite
    };
    let mut iterations = 0;
    for _ in 0..config.iterations {
        if termination == Termination::NonFinite {
            break;
        }
        iterations += 1;
        let normal = normal_all(&poses, &lin, &limits);
        // Damped pose blocks: solve [coupling, gradient] per block, then the scalar Schur equation for the scale step.
        let damped = schur(&normal, |pose_block, pose_mask, coupling, gradient| {
            let scaling = marquardt_scaling(&pose_block.diagonal(), pose_mask);
            let mut damped = *pose_block;
            for i in 0..POSE {
                damped[(i, i)] += damping * scaling[i] + (1.0 - pose_mask[i]);
            }
            let lu = damped.lu();
            Some((lu.solve(coupling)?, lu.solve(gradient)?))
        });
        let Some(damped) = damped else {
            termination = Termination::NonFinite;
            break;
        };
        let schur_complement = damped.curvature * (1.0 + damping) - damped.coupling;
        let scale_step = (-damped.gradient + damped.coupled_gradient) / schur_complement;
        let steps = block_steps(&damped.solved, scale_step);
        let candidates: Vec<Pose> = poses
            .iter()
            .zip(&steps)
            .map(|(pose, step)| retract(pose, &step.fixed_rows::<POSE>(0).into_owned(), &limits))
            .collect();
        let candidate_scale = scale + scale_step as f32 as f64;
        let predicted: f64 = (0..poses.len())
            .map(|b| {
                predicted_reduction(
                    &poses[b],
                    &candidates[b],
                    &steps[b],
                    &normal.h[b],
                    &normal.g[b],
                )
            })
            .sum();
        let new_lin = linearize_all(generic, candidate_scale, &candidates, &blocks);
        if !new_lin.energy.is_finite() {
            termination = Termination::NonFinite;
            break;
        }
        let reduction = lin.energy - new_lin.energy;
        if reduction > 0.0 {
            converged = reduction <= config.relative_tolerance * lin.energy;
            poses = candidates;
            scale = candidate_scale;
            lin = new_lin;
            damping = nielsen_damping(damping, reduction, predicted);
            growth = DAMPING_GROWTH;
        } else {
            damping *= growth;
            growth *= DAMPING_GROWTH;
        }
        if converged {
            termination = Termination::Tolerance;
            break;
        }
        if damping > MAX_DAMPING {
            // Recompute at the retained solution: the last step may have been accepted. Undamped (pseudo-inverted) blocks.
            let normal = normal_all(&poses, &lin, &limits);
            let mut scaled_gradient_sq: f64 = 0.0;
            let undamped = schur(&normal, |pose_block, pose_mask, coupling, gradient| {
                let scaling = marquardt_scaling(&pose_block.diagonal(), pose_mask);
                for i in 0..POSE {
                    scaled_gradient_sq =
                        scaled_gradient_sq.max(gradient[i] * gradient[i] / scaling[i]);
                }
                let inverse = undamped_inverse(pose_block, pose_mask);
                Some((inverse * coupling, inverse * gradient))
            });
            // The pseudo-inverse always solves.
            let Some(undamped) = undamped else {
                termination = Termination::NonFinite;
                break;
            };
            let schur_complement = undamped.curvature - undamped.coupling;
            let mut gn_prediction = f64::INFINITY;
            if schur_complement > f64::EPSILON * undamped.curvature {
                let scale_step =
                    (-undamped.gradient + undamped.coupled_gradient) / schur_complement;
                gn_prediction = 0.0;
                for (b, step) in block_steps(&undamped.solved, scale_step).iter().enumerate() {
                    let candidate =
                        retract(&poses[b], &step.fixed_rows::<POSE>(0).into_owned(), &limits);
                    gn_prediction += predicted_reduction(
                        &poses[b],
                        &candidate,
                        step,
                        &normal.h[b],
                        &normal.g[b],
                    );
                }
            }
            scaled_gradient_sq = scaled_gradient_sq
                .max(undamped.gradient * undamped.gradient / undamped.curvature.max(1e-30));
            converged = gn_prediction.abs() <= config.relative_tolerance * lin.energy
                || scaled_gradient_sq <= f32::EPSILON as f64 * lin.energy;
            termination = if converged {
                Termination::Stationary
            } else {
                Termination::Damping
            };
            break;
        }
    }
    if !scale.is_finite() {
        return Err(ScaleError::Diverged);
    }
    if termination != Termination::NonFinite {
        for pose in &mut poses {
            pose.rotation = orthonormalize(&pose.rotation);
        }
    }
    let e_2d: f64 = poses
        .iter()
        .zip(&blocks)
        .map(|(pose, &block)| linearize(generic, scale, pose, block).0.norm_squared())
        .sum();
    if !e_2d.is_finite() {
        termination = Termination::NonFinite;
    }
    Ok(ScaleCalibration {
        phi: if termination == Termination::NonFinite {
            f64::NAN
        } else {
            scale
        },
        poses,
        used,
        e_2d,
        blocks: blocks.len(),
        iterations,
        converged: converged && termination != Termination::NonFinite,
        termination,
    })
}
