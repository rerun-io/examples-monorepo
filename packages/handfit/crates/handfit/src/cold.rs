//! Acquisition: ray alignment, cube wrists, palm-only LM, then full finger fits.
use crate::{
    lm::solve, model::Step, residual::Views, Config, FitError, FitResult, JacobianMode, Model,
    Pose, Termination, View,
};
use nalgebra::{Matrix3, SVector, Vector3};
use std::time::{Duration, Instant};

pub const PALM: [usize; 6] = [5, 8, 11, 14, 17, 20];

/// How a cold fit went: every palm-only and full solve, the palms chosen for the full fits, the winner, and the wall time of
/// each stage.
#[derive(Debug)]
pub struct Diagnostics {
    pub rigid_iterations: Vec<usize>,
    pub rigid_terminations: Vec<Termination>,
    pub chosen: Vec<usize>,
    pub full_iterations: Vec<usize>,
    pub full_terminations: Vec<Termination>,
    pub winner: usize,
    /// The wrist hypotheses.
    pub hypotheses_time: Duration,
    /// The palm-only solves.
    pub rigid_time: Duration,
    /// The full solves.
    pub full_time: Duration,
}

/// The cold fit's finger start: the identity wrist at the origin, every joint at the middle of its limits.
///
/// # Arguments
///
/// * `model` - The hand model (its joint limits).
///
/// # Returns
///
/// The neutral pose.
pub fn neutral(model: &Model) -> Pose {
    Pose {
        rotation: Matrix3::identity(),
        translation: Vector3::zeros(),
        angles: SVector::from_fn(|i, _| {
            if i < 20 {
                (model.limits[(i, 0)] + model.limits[(i, 1)]) / 2.0
            } else {
                0.0
            }
        }),
    }
}

/// The cold fit's wrist hypotheses: one ray-aligned wrist per view (Kabsch on the palm keypoints, or every keypoint when fewer
/// than three palm points are observed), then the first `config.rotation_hypotheses` axis-aligned rotations about the best
/// view's anchor. Aligned slots 0 and 1 always precede the cube rotations, even for a single view.
///
/// # Arguments
///
/// * `model` - The hand model.
/// * `config` - `phi` (scales the relative distances) and `rotation_hypotheses`.
/// * `mirror` - +1 left hand, −1 right hand.
/// * `views` - The hand's views.
///
/// # Returns
///
/// The hypotheses and, per hypothesis, whether it is usable: an aligned wrist needs three observed keypoints in its view, the
/// cube rotations need one usable view.
pub fn wrist_hypotheses(
    model: &Model,
    config: &Config,
    mirror: f64,
    views: Views<'_>,
) -> (Vec<Pose>, Vec<bool>) {
    let neutral = neutral(model);
    let local = model.landmarks(&neutral, mirror, &Step::zeros(), None);
    let mut poses = vec![neutral.clone(); 2];
    let mut usable = vec![false; 2];
    let mut best_count: isize = -2;
    let mut anchor = Vector3::zeros();
    let mut model_anchor = Vector3::zeros();
    for (v, view) in views.iter().enumerate() {
        let palm_count = PALM.iter().filter(|&&i| view.weights[i] > 0.0).count();
        let used: Vec<usize> = (0..21)
            .filter(|i| view.weights[*i] > 0.0 && (palm_count < 3 || PALM.contains(i)))
            .collect();
        let total = used.len().max(1) as f64;
        usable[v] = used.len() >= 3;
        let rays: Vec<Vector3<f64>> = used
            .iter()
            .map(|&i| {
                Vector3::from(
                    views
                        .camera(v)
                        .unproject([view.pixels[(i, 0)], view.pixels[(i, 1)]])
                        .unwrap_or([0.0; 3]),
                )
            })
            .collect();
        let offsets: Vec<Vector3<f64>> = used
            .iter()
            .zip(&rays)
            .map(|(&i, ray)| ray * (config.phi * view.distances[i] / 1000.0))
            .collect();
        let source: Vec<Vector3<f64>> = used
            .iter()
            .map(|&i| local.fixed_rows::<3>(i * 3).into_owned())
            .collect();
        let mean =
            |points: &[Vector3<f64>]| points.iter().fold(Vector3::zeros(), |a, p| a + p) / total;
        let ray_mean = mean(&rays);
        let offset_mean = mean(&offsets);
        let source_mean = mean(&source);
        let mut qa = 0.0_f64;
        let mut qb = 0.0_f64;
        let mut qc = 0.0_f64;
        for i in 0..used.len() {
            let a = rays[i] - ray_mean;
            let c = offsets[i] - offset_mean;
            qa += a.norm_squared();
            qb += a.dot(&c);
            qc += c.norm_squared() - (source[i] - source_mean).norm_squared();
        }
        qa = qa.max(1e-12);
        let distance = ((-qb + (qb * qb - qa * qc).max(0.0).sqrt()) / qa).clamp(0.1, 2.0);
        let targets: Vec<Vector3<f64>> = rays
            .iter()
            .zip(&offsets)
            .map(|(r, o)| distance * r + o)
            .collect();
        let target_mean = mean(&targets);
        let mut covariance = Matrix3::zeros();
        for i in 0..used.len() {
            covariance += (source[i] - source_mean) * (targets[i] - target_mean).transpose();
        }
        let svd = covariance.svd(true, true);
        // Asked for, `u` and `v_t` are always computed; without them the view has no aligned wrist.
        let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
            usable[v] = false;
            continue;
        };
        let vmat = v_t.transpose();
        let mut correction = Matrix3::identity();
        correction[(2, 2)] = (vmat * u.transpose()).determinant().signum();
        let cam_rotation = vmat * correction * u.transpose();
        let world_rotation = view.rotation.transpose();
        let world_mean = world_rotation * (target_mean - view.translation);
        poses[v].rotation = world_rotation * cam_rotation;
        poses[v].translation = world_mean - poses[v].rotation * source_mean;
        let score = if usable[v] { used.len() as isize } else { -1 };
        if score > best_count {
            best_count = score;
            anchor = world_mean;
            model_anchor = source_mean;
        }
    }
    let any_valid = usable.iter().any(|x| *x);
    // itertools.permutations(range(3)) x product((1, -1), repeat=3), determinant +1.
    let mut count = 0;
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for bits in 0..8 {
            let mut rotation = Matrix3::zeros();
            for row in 0..3 {
                rotation[(row, permutation[row])] = if bits & (1 << (2 - row)) == 0 {
                    1.0
                } else {
                    -1.0
                };
            }
            if rotation.determinant() > 0.0 && count < config.rotation_hypotheses {
                poses.push(Pose {
                    rotation,
                    translation: anchor - rotation * model_anchor,
                    ..neutral.clone()
                });
                usable.push(any_valid);
                count += 1;
            }
        }
    }
    (poses, usable)
}

/// `f` over `items` on up to `threads` scoped threads (thread `t` takes items `t, t + threads, ...`), results in input order.
fn parallel_map<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    parallel_map_with_builder(items, threads, f, |_| std::thread::Builder::new())
}

// Inject the OS thread builder so tests can exercise rejected spawns without exhausting resources.
fn parallel_map_with_builder<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    f: impl Fn(&T) -> R + Sync,
    builder: impl Fn(usize) -> std::thread::Builder,
) -> Vec<R> {
    let threads = threads.clamp(1, items.len().max(1));
    if threads == 1 {
        return items.iter().map(&f).collect();
    }
    let f = &f;
    let mut slots: Vec<Option<R>> = std::iter::repeat_with(|| None).take(items.len()).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for t in 0..threads {
            let work = move || {
                (t..items.len())
                    .step_by(threads)
                    .map(|i| (i, f(&items[i])))
                    .collect::<Vec<(usize, R)>>()
            };
            match builder(t).spawn_scoped(scope, work) {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    for (i, r) in work() {
                        slots[i] = Some(r);
                    }
                }
            }
        }
        for handle in handles {
            match handle.join() {
                Ok(results) => {
                    for (i, r) in results {
                        slots[i] = Some(r);
                    }
                }
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
    });
    // Every index was filled by exactly one thread.
    slots.into_iter().flatten().collect()
}

/// The acquisition fit from no prior: wrist hypotheses, palm-only LM on each, then full fits of the best palms from two
/// finger starts; the lowest-energy full fit wins. [`initial_pose_parallel`] on one thread.
///
/// # Arguments
///
/// * `model` - The hand model at the hand's scale.
/// * `config` - The residual weights and the cold-fit settings (`init_iterations`, `rotation_hypotheses`, ...).
/// * `mirror` - +1 left hand, −1 right hand.
/// * `views` - The hand's views, at most [`crate::residual::MAX_VIEWS`].
/// * `mode` - The analytic Jacobian, or handtrack's central differences.
///
/// # Returns
///
/// The winning full fit with the [`Diagnostics`] of every stage, or the neutral pose with [`Termination::NoEvidence`] when no
/// view has three observed keypoints.
///
/// # Errors
///
/// [`FitError::Views`] for more than two views, [`FitError::NoFullFit`] when `config` asks for no full fit.
pub fn initial_pose(
    model: &Model,
    config: &Config,
    mirror: f64,
    views: &[View],
    mode: JacobianMode,
) -> Result<FitResult, FitError> {
    initial_pose_parallel(model, config, mirror, views, mode, 1)
}

/// [`initial_pose`] with its independent LM solves (the palm-only hypotheses, then the full fits) on up to `threads` scoped
/// threads. The result does not depend on `threads`: every solve runs the same code on the same inputs, and the ranking and
/// the winner use the same order. The threads inherit the caller's CPU affinity: give the caller a core set, not one core.
///
/// # Arguments
///
/// * `model`, `config`, `mirror`, `views`, `mode` - As [`initial_pose`].
/// * `threads` - The most threads the solves run on (0 and 1 = the caller's thread alone).
///
/// # Returns
///
/// As [`initial_pose`].
///
/// # Errors
///
/// As [`initial_pose`].
pub fn initial_pose_parallel(
    model: &Model,
    config: &Config,
    mirror: f64,
    views: &[View],
    mode: JacobianMode,
    threads: usize,
) -> Result<FitResult, FitError> {
    let checked =
        Views::new(views)?;
    let started = Instant::now();
    let (hypotheses, usable) = wrist_hypotheses(model, config, mirror, checked);
    let hypotheses_time = started.elapsed();
    if !usable.iter().any(|x| *x) {
        return Ok(FitResult {
            pose: neutral(model),
            energies: [f64::NAN; 4],
            solve_energy: f64::NAN,
            iterations: 0,
            converged: false,
            termination: Termination::NoEvidence,
            cold: None,
        });
    }
    let stage = Config {
        max_iterations: config.init_iterations,
        relative_tolerance: config.init_relative_tolerance,
        temporal_weight: 0.0,
        ..config.clone()
    };
    let mut rigid_views = views.to_vec();
    if views
        .iter()
        .any(|v| PALM.iter().filter(|&&i| v.weights[i] > 0.0).count() >= 3)
    {
        for view in &mut rigid_views {
            for i in 0..21 {
                if !PALM.contains(&i) {
                    view.weights[i] = 0.0;
                }
            }
        }
    }
    // As many views as `views`.
    let rigid_views = Views::new(&rigid_views)?;
    let started = Instant::now();
    let rigid: Vec<FitResult> = parallel_map(&hypotheses, threads, |p| {
        solve(model, &stage, p, mirror, rigid_views, mode, true)
    });
    let rigid_time = started.elapsed();
    let energies: Vec<f64> = rigid
        .iter()
        .zip(&usable)
        .map(|(r, usable)| {
            if *usable && r.solve_energy.is_finite() {
                r.solve_energy
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..rigid.len()).collect();
    order.sort_by(|&a, &b| energies[a].total_cmp(&energies[b]).then(a.cmp(&b)));
    let best = &rigid[order[0]].pose.rotation;
    let mut ranked = vec![f64::INFINITY; rigid.len()];
    for (i, r) in rigid.iter().enumerate() {
        let cosine = ((best.transpose() * r.pose.rotation).trace() - 1.0) / 2.0;
        if cosine.clamp(-1.0, 1.0).acos() > 30.0_f64.to_radians() {
            ranked[i] = energies[i];
        }
    }
    ranked[order[0]] = f64::NEG_INFINITY;
    let mut chosen: Vec<usize> = (0..rigid.len()).collect();
    chosen.sort_by(|&a, &b| ranked[a].total_cmp(&ranked[b]).then(a.cmp(&b)));
    chosen.truncate(config.full_fit_hypotheses.min(rigid.len()));
    for (k, index) in chosen.iter_mut().enumerate() {
        if ranked[*index] == f64::INFINITY {
            *index = order[k];
        }
    }
    let started = Instant::now();
    let mut starts = Vec::new();
    for &index in &chosen {
        for fingers in 0..config.finger_starts.min(2) {
            let mut start = rigid[index].pose.clone();
            if fingers == 1 {
                for i in 0..20 {
                    start.angles[i] = 0.0_f64.clamp(model.limits[(i, 0)], model.limits[(i, 1)]);
                }
            }
            starts.push(start);
        }
    }
    let mut full: Vec<FitResult> = parallel_map(&starts, threads, |start| {
        solve(model, &stage, start, mirror, checked, mode, false)
    });
    let full_time = started.elapsed();
    // No full fit only when `full_fit_hypotheses` or `finger_starts` is 0 (there are always two aligned hypotheses).
    let Some(winner) = (0..full.len()).min_by(|&a, &b| {
        let energy = |i: usize| {
            if full[i].solve_energy.is_finite() {
                full[i].solve_energy
            } else {
                f64::INFINITY
            }
        };
        energy(a).total_cmp(&energy(b)).then(a.cmp(&b))
    }) else {
        return Err(FitError::NoFullFit);
    };
    let diagnostics = Diagnostics {
        rigid_iterations: rigid.iter().map(|r| r.iterations).collect(),
        rigid_terminations: rigid.iter().map(|r| r.termination).collect(),
        chosen,
        full_iterations: full.iter().map(|r| r.iterations).collect(),
        full_terminations: full.iter().map(|r| r.termination).collect(),
        winner,
        hypotheses_time,
        rigid_time,
        full_time,
    };
    let mut result = full.swap_remove(winner);
    result.cold = Some(diagnostics);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_thread_spawns_keep_every_result_in_input_order() {
        // An impossible stack size makes the OS reject creation without exhausting host resources.
        assert!(std::thread::Builder::new()
            .stack_size(usize::MAX)
            .spawn(|| ())
            .is_err());
        for failed_threads in [vec![0, 1, 2], vec![1]] {
            let calls: Vec<std::sync::atomic::AtomicUsize> = (0..7)
                .map(|_| std::sync::atomic::AtomicUsize::new(0))
                .collect();
            let caller = std::thread::current().id();
            let result = parallel_map_with_builder(
                &[0, 1, 2, 3, 4, 5, 6],
                3,
                |&i| {
                    calls[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if failed_threads.contains(&(i % 3)) {
                        assert_eq!(std::thread::current().id(), caller);
                    }
                    i * i
                },
                |t| {
                    let builder = std::thread::Builder::new();
                    if failed_threads.contains(&t) {
                        builder.stack_size(usize::MAX)
                    } else {
                        builder
                    }
                },
            );
            assert_eq!(result, [0, 1, 4, 9, 16, 25, 36]);
            assert!(calls
                .iter()
                .all(|count| count.load(std::sync::atomic::Ordering::Relaxed) == 1));
        }
        assert_eq!(parallel_map(&[3, 1, 2], 1, |&i| i * i), [9, 1, 4]);
        assert_eq!(parallel_map(&[3, 1, 2], 8, |&i| i * i), [9, 1, 4]);
        assert!(parallel_map::<usize, usize>(&[], 0, |&i| i).is_empty());
    }
}
