#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::sparse::{BlockSymmetric, SymbolicCholesky};
use super::*;
use kornia_staging_algebra::lie::Rotation3;
use kornia_staging_algebra::optim::solvers::{
    levenberg_marquardt, marquardt_scaling, predicted_reduction, DampingRule, LmProblem,
    LmTermination,
};
use nalgebra::{DMatrix, DVector, Matrix6, Vector3, Vector6};

/// A small deterministic generator, so every graph is reproducible without a fixture file.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
    }

    /// Uniform in `[-scale, scale]`.
    fn symmetric(&mut self, scale: f64) -> f64 {
        (2.0 * self.next_f64() - 1.0) * scale
    }

    fn vector3(&mut self, scale: f64) -> Vector3<f64> {
        Vector3::new(
            self.symmetric(scale),
            self.symmetric(scale),
            self.symmetric(scale),
        )
    }

    fn tangent(&mut self, translation: f64, rotation: f64) -> Vector6<f64> {
        let t = self.vector3(translation);
        let w = self.vector3(rotation);
        Vector6::new(t.x, t.y, t.z, w.x, w.y, w.z)
    }

    fn pose(&mut self, translation: f64, rotation: f64) -> RigidTransform<f64> {
        RigidTransform::exp_decoupled(&self.tangent(translation, rotation))
    }

    fn index(&mut self, n: usize) -> usize {
        ((self.next_f64() * n as f64) as usize).min(n - 1)
    }
}

fn perturbed(pose: &RigidTransform<f64>, inc: &Vector6<f64>) -> RigidTransform<f64> {
    let mut out = *pose;
    out.apply_inc(inc);
    out
}

fn max_pose_difference(a: &[RigidTransform<f64>], b: &[RigidTransform<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x.inverse() * *y;
            d.translation.norm().max(d.rotation.log().norm())
        })
        .fold(0.0, f64::max)
}

// ───── residual ─────────────────────────────────────────────────────────────

#[test]
fn residual_vanishes_at_the_measurement() {
    let mut rng = Lcg(1);
    for _ in 0..32 {
        let a = rng.pose(5.0, 3.0);
        let b = rng.pose(5.0, 3.0);
        let measured = a.inverse() * b;
        let lin = relative_pose_residual(&measured, &a, &b);
        assert!(lin.residual.norm() < 1e-12, "{}", lin.residual);
    }
}

#[test]
fn residual_jacobians_match_central_differences() {
    let mut rng = Lcg(2);
    let eps = 1e-6;
    for _ in 0..64 {
        let a = rng.pose(10.0, 3.0);
        let b = rng.pose(10.0, 3.0);
        // Measurement errors up to ~0.5 rad keep the rotation log far from the pi branch cut.
        let measured = a.inverse() * b * rng.pose(0.5, 0.3);
        let lin = relative_pose_residual(&measured, &a, &b);
        for k in 0..6 {
            let mut inc = Vector6::zeros();
            inc[k] = eps;
            let num_a = (relative_pose_residual(&measured, &perturbed(&a, &inc), &b).residual
                - relative_pose_residual(&measured, &perturbed(&a, &-inc), &b).residual)
                / (2.0 * eps);
            let num_b = (relative_pose_residual(&measured, &a, &perturbed(&b, &inc)).residual
                - relative_pose_residual(&measured, &a, &perturbed(&b, &-inc)).residual)
                / (2.0 * eps);
            let err_a = (num_a - lin.d_from.column(k)).amax();
            let err_b = (num_b - lin.d_to.column(k)).amax();
            assert!(err_a < 1e-6, "d_from column {k}: {err_a}");
            assert!(err_b < 1e-6, "d_to column {k}: {err_b}");
        }
    }
}

#[test]
fn residual_cost_matches_the_ceres_pose_graph_term() {
    // RootSLAM's Ceres `PoseGraph3dErrorTerm`: [R_a^T (p_b - p_a) - p_ab; 2 vec(q_ab q_est^-1)].
    // Ours is the decoupled SE(3) log in the measurement frame: the same squared norm to second order.
    let mut rng = Lcg(3);
    for _ in 0..32 {
        let a = rng.pose(10.0, 3.0);
        let b = rng.pose(10.0, 3.0);
        let measured = a.inverse() * b * rng.pose(0.01, 0.01);
        let ours = relative_pose_residual(&measured, &a, &b).residual;
        let est = a.inverse() * b;
        let t = est.translation - measured.translation;
        let dq = (*measured.rotation.quaternion()) * est.rotation.quaternion().inverse();
        let ceres_rot = 2.0 * dq.imag() * dq.w.signum();
        let ceres = Vector6::new(t.x, t.y, t.z, ceres_rot.x, ceres_rot.y, ceres_rot.z);
        assert!((ours.norm_squared() - ceres.norm_squared()).abs() <= 1e-4 * ceres.norm_squared());
    }
}

// ───── sigmas ───────────────────────────────────────────────────────────────

#[test]
fn edge_sigmas_whiten_translation_and_rotation_rows() {
    let unit = EdgeSigmas::UNIT.sqrt_information();
    assert_eq!(unit, Matrix6::identity());
    let default = EdgeSigmas::default();
    assert_eq!(default.translation_m, 0.05);
    assert_eq!(default.rotation_rad, 0.01);
    let s = default.sqrt_information();
    let expected = Matrix6::from_diagonal(&Vector6::new(20.0, 20.0, 20.0, 100.0, 100.0, 100.0));
    assert!((s - expected).amax() < 1e-12);
}

// ───── block-sparse Cholesky ────────────────────────────────────────────────

fn random_graph(rng: &mut Lcg, n: usize, extra_edges: usize) -> Vec<(usize, usize)> {
    let mut edges: Vec<(usize, usize)> = (1..n).map(|i| (i - 1, i)).collect();
    for _ in 0..extra_edges {
        let a = rng.index(n);
        let b = rng.index(n);
        if a != b {
            edges.push((a, b));
        }
    }
    edges
}

fn random_spd_blocks(
    rng: &mut Lcg,
    symbolic: &SymbolicCholesky,
    n: usize,
    edges: &[(usize, usize)],
) -> (BlockSymmetric, DMatrix<f64>) {
    let mut h = symbolic.zeros();
    let mut dense = DMatrix::<f64>::zeros(6 * n, 6 * n);
    for &(a, b) in edges {
        let ja = Matrix6::from_fn(|_, _| rng.symmetric(1.0));
        let jb = Matrix6::from_fn(|_, _| rng.symmetric(1.0));
        symbolic.add_diagonal(&mut h, a, &(ja.transpose() * ja));
        symbolic.add_diagonal(&mut h, b, &(jb.transpose() * jb));
        symbolic.add_off_diagonal(&mut h, a, b, &(ja.transpose() * jb));
        let mut add = |r: usize, c: usize, m: &Matrix6<f64>| {
            let mut view = dense.view_mut((6 * r, 6 * c), (6, 6));
            view += m;
        };
        add(a, a, &(ja.transpose() * ja));
        add(b, b, &(jb.transpose() * jb));
        add(a, b, &(ja.transpose() * jb));
        add(b, a, &(jb.transpose() * ja));
    }
    // Anchor every node so the system is strictly positive definite.
    for i in 0..n {
        symbolic.add_diagonal(&mut h, i, &Matrix6::identity());
        let mut view = dense.view_mut((6 * i, 6 * i), (6, 6));
        view += Matrix6::<f64>::identity();
    }
    (h, dense)
}

#[test]
fn ordering_is_a_permutation_and_the_pattern_covers_the_graph() {
    let mut rng = Lcg(4);
    for n in [1, 2, 7, 40] {
        let edges = random_graph(&mut rng, n, n);
        let symbolic = SymbolicCholesky::new(n, edges.iter().copied());
        let mut seen = symbolic.ordering().to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>());
        for &(a, b) in &edges {
            assert!(
                symbolic.has_block(a, b),
                "edge ({a}, {b}) has no factor block"
            );
        }
    }
}

#[test]
fn minimum_degree_eliminates_a_star_hub_last() {
    // Eliminating the hub first fills the whole matrix; minimum degree takes the leaves first.
    let n = 30;
    let edges: Vec<(usize, usize)> = (1..n).map(|i| (0, i)).collect();
    let symbolic = SymbolicCholesky::new(n, edges.iter().copied());
    // The hub goes once it has a single neighbour left: one of the last two positions.
    let hub = symbolic.ordering().iter().position(|&v| v == 0).unwrap();
    assert!(hub >= n - 2, "hub eliminated at {hub}");
    assert_eq!(symbolic.off_diagonal_blocks(), n - 1);
}

#[test]
fn a_long_loop_factors_with_linear_fill() {
    // A ring of n nodes: minimum degree keeps the fill O(n), not O(n^2).
    let n = 1500;
    let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    let symbolic = SymbolicCholesky::new(n, edges.iter().copied());
    assert!(
        symbolic.off_diagonal_blocks() <= 2 * n,
        "{}",
        symbolic.off_diagonal_blocks()
    );
}

#[test]
fn sparse_solve_matches_dense_cholesky() {
    let mut rng = Lcg(5);
    for (n, extra) in [(1, 0), (2, 1), (12, 10), (50, 80)] {
        let edges = random_graph(&mut rng, n, extra);
        let symbolic = SymbolicCholesky::new(n, edges.iter().copied());
        let (h, dense) = random_spd_blocks(&mut rng, &symbolic, n, &edges);
        let rhs = DVector::from_fn(6 * n, |_, _| rng.symmetric(1.0));

        let mut y = DVector::zeros(6 * n);
        symbolic.multiply(&h, rhs.as_slice(), y.as_mut_slice());
        assert!((&y - &dense * &rhs).amax() < 1e-10, "block multiply");

        let mut factor = h.clone();
        symbolic.factor(&mut factor).unwrap();
        let mut x = rhs.clone();
        symbolic.solve(&factor, x.as_mut_slice());
        let expected = dense.clone().cholesky().unwrap().solve(&rhs);
        let scale = expected.amax().max(1.0);
        assert!((&x - &expected).amax() / scale < 1e-10, "n = {n}");
    }
}

#[test]
fn factor_rejects_an_indefinite_matrix() {
    let symbolic = SymbolicCholesky::new(2, [(0, 1)]);
    let mut h = symbolic.zeros();
    symbolic.add_diagonal(&mut h, 0, &Matrix6::identity());
    symbolic.add_diagonal(&mut h, 1, &Matrix6::identity());
    symbolic.add_off_diagonal(&mut h, 0, 1, &(Matrix6::identity() * 2.0));
    assert!(symbolic.factor(&mut h).is_err());
}

// ───── pose graph problems ──────────────────────────────────────────────────

struct Graph {
    truth: Vec<RigidTransform<f64>>,
    initial: Vec<RigidTransform<f64>>,
    edges: Vec<PoseGraphEdge>,
}

/// A trajectory with odometry edges, a few skip edges and loop edges, measured with `noise`;
/// the initial guess integrates the noisy odometry, so it drifts like a VIO does.
fn drifting_graph(rng: &mut Lcg, n: usize, noise: f64, sigmas: &EdgeSigmas) -> Graph {
    let mut truth = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let radius = 3.0 + 0.002 * n as f64;
        let position = Vector3::new(
            radius * angle.cos(),
            radius * angle.sin(),
            0.3 * (3.0 * angle).sin(),
        );
        let rotation = Rotation3::exp(&Vector3::new(0.1 * angle.sin(), 0.05, angle + 0.3));
        truth.push(RigidTransform::new(rotation, position));
    }
    let mut pairs: Vec<(usize, usize)> = (1..n).map(|i| (i - 1, i)).collect();
    pairs.extend((2..n).step_by(3).map(|i| (i - 2, i)));
    for k in 0..3.min(n / 4) {
        pairs.push((k, n - 1 - k));
    }
    let edges: Vec<PoseGraphEdge> = pairs
        .into_iter()
        .map(|(a, b)| {
            let measured = truth[a].inverse() * truth[b] * rng.pose(noise, noise);
            PoseGraphEdge::new(a, b, measured, sigmas)
        })
        .collect();
    let mut initial = vec![truth[0]];
    for i in 1..n {
        let odometry = truth[i - 1].inverse() * truth[i] * rng.pose(0.01, 0.003);
        initial.push(initial[i - 1] * odometry);
    }
    Graph {
        truth,
        initial,
        edges,
    }
}

/// The RootSLAM gauge: the newest pose and the window before it are fixed.
fn window_fixed(n: usize, window: usize) -> Vec<bool> {
    (0..n).map(|i| i + window >= n).collect()
}

#[test]
fn pgo_recovers_a_consistent_graph() {
    let mut rng = Lcg(6);
    let n = 60;
    let mut graph = drifting_graph(&mut rng, n, 0.0, &EdgeSigmas::default());
    // Fix the true window so the optimum is the ground truth itself.
    let fixed = window_fixed(n, 7);
    for ((initial, truth), _) in graph
        .initial
        .iter_mut()
        .zip(&graph.truth)
        .zip(&fixed)
        .filter(|(_, &f)| f)
    {
        *initial = *truth;
    }
    let mut poses = graph.initial.clone();
    let report =
        optimize_pose_graph(&mut poses, &fixed, &graph.edges, &PgoConfig::default()).unwrap();
    assert!(
        report.final_cost < 1e-16 * report.initial_cost.max(1.0),
        "{report:?}"
    );
    assert!(max_pose_difference(&poses, &graph.truth) < 1e-8);
}

#[test]
fn fixed_poses_are_bit_unchanged_and_loop_drift_moves_the_old_map() {
    let mut rng = Lcg(7);
    let n = 80;
    let graph = drifting_graph(&mut rng, n, 0.002, &EdgeSigmas::default());
    let fixed = window_fixed(n, 8);
    let mut poses = graph.initial.clone();
    let report =
        optimize_pose_graph(&mut poses, &fixed, &graph.edges, &PgoConfig::default()).unwrap();
    assert!(report.final_cost < 0.1 * report.initial_cost, "{report:?}");
    for i in 0..n {
        if fixed[i] {
            assert_eq!(poses[i], graph.initial[i], "fixed pose {i} moved");
        }
    }
    // Pose 0 starts where the truth is and the window carries the drift: the old map follows the window.
    assert!(max_pose_difference(&poses[..1], &graph.initial[..1]) > 1e-3);
}

#[test]
fn edges_between_fixed_poses_and_isolated_poses_are_inert() {
    let mut rng = Lcg(8);
    let n = 20;
    let mut graph = drifting_graph(&mut rng, n, 0.002, &EdgeSigmas::UNIT);
    // A pose that no edge touches stays put and does not make the system singular.
    graph.initial.push(rng.pose(1.0, 1.0));
    let mut fixed = window_fixed(n, 3);
    fixed.push(false);
    let mut poses = graph.initial.clone();
    optimize_pose_graph(&mut poses, &fixed, &graph.edges, &PgoConfig::default()).unwrap();
    assert_eq!(poses[n], graph.initial[n]);
}

#[test]
fn pgo_needs_a_gauge_in_every_component() {
    let mut rng = Lcg(9);
    let graph = drifting_graph(&mut rng, 10, 0.0, &EdgeSigmas::UNIT);
    let mut poses = graph.initial.clone();
    let none = vec![false; 10];
    assert_eq!(
        optimize_pose_graph(&mut poses, &none, &graph.edges, &PgoConfig::default()).unwrap_err(),
        PgoError::Unanchored { pose: 0 }
    );
    // Two components, the gauge only in the first.
    let mut poses: Vec<RigidTransform<f64>> = (0..4).map(|_| rng.pose(1.0, 0.5)).collect();
    let edges = [
        PoseGraphEdge::new(0, 1, poses[0].inverse() * poses[1], &EdgeSigmas::UNIT),
        PoseGraphEdge::new(2, 3, poses[2].inverse() * poses[3], &EdgeSigmas::UNIT),
    ];
    let fixed = [true, false, false, false];
    assert_eq!(
        optimize_pose_graph(&mut poses, &fixed, &edges, &PgoConfig::default()).unwrap_err(),
        PgoError::Unanchored { pose: 2 }
    );
}

#[test]
fn invalid_inputs_are_rejected() {
    let mut poses = vec![RigidTransform::<f64>::identity(); 3];
    let fixed = [true, false, false];
    let edge = |a, b| PoseGraphEdge::new(a, b, RigidTransform::identity(), &EdgeSigmas::UNIT);
    let config = PgoConfig::default();
    assert_eq!(
        optimize_pose_graph(&mut poses, &fixed, &[edge(0, 3)], &config).unwrap_err(),
        PgoError::PoseIndex { edge: 0, pose: 3 }
    );
    assert_eq!(
        optimize_pose_graph(&mut poses, &fixed, &[edge(0, 1), edge(1, 1)], &config).unwrap_err(),
        PgoError::SelfEdge { edge: 1 }
    );
    assert_eq!(
        optimize_pose_graph(&mut poses, &fixed[..2], &[edge(0, 1)], &config).unwrap_err(),
        PgoError::FixedLength { poses: 3, fixed: 2 }
    );
    let mut bad = edge(0, 1);
    bad.sqrt_information[(0, 0)] = f64::NAN;
    assert_eq!(
        optimize_pose_graph(&mut poses, &fixed, &[bad], &config).unwrap_err(),
        PgoError::NonFinite { edge: 0 }
    );
}

#[test]
fn a_stiffer_edge_is_satisfied_more_closely() {
    // Two conflicting measurements of the same relative pose: the weighted mean favours the stiffer one.
    let measured_a = RigidTransform::new(Rotation3::identity(), Vector3::new(1.0, 0.0, 0.0));
    let measured_b = RigidTransform::new(Rotation3::identity(), Vector3::new(1.2, 0.0, 0.0));
    let stiff = EdgeSigmas {
        translation_m: 0.01,
        rotation_rad: 0.01,
    };
    let loose = EdgeSigmas {
        translation_m: 0.1,
        rotation_rad: 0.01,
    };
    let edges = [
        PoseGraphEdge::new(0, 1, measured_a, &stiff),
        PoseGraphEdge::new(0, 1, measured_b, &loose),
    ];
    let mut poses = vec![RigidTransform::<f64>::identity(); 2];
    optimize_pose_graph(&mut poses, &[true, false], &edges, &PgoConfig::default()).unwrap();
    // Weighted mean: (1.0 * 100^2 + 1.2 * 10^2) / (100^2 + 10^2).
    let expected = (1.0 * 1e4 + 1.2 * 1e2) / (1e4 + 1e2);
    assert!(
        (poses[1].translation.x - expected).abs() < 1e-9,
        "{}",
        poses[1].translation.x
    );
}

// ───── dense f64 reference ──────────────────────────────────────────────────

/// The same problem with a dense normal matrix and nalgebra's dense Cholesky, driven by the same LM
/// schedule, so the block-sparse solver must reproduce it step for step.
struct DensePgo<'a> {
    poses: Vec<RigidTransform<f64>>,
    trial: Vec<RigidTransform<f64>>,
    variable: Vec<Option<usize>>,
    edges: &'a [PoseGraphEdge],
    h: DMatrix<f64>,
    g: DVector<f64>,
}

impl DensePgo<'_> {
    fn cost(&self, poses: &[RigidTransform<f64>]) -> f64 {
        self.edges
            .iter()
            .map(|e| {
                let r = relative_pose_residual(&e.from_t_to, &poses[e.from], &poses[e.to]).residual;
                (e.sqrt_information * r).norm_squared()
            })
            .sum()
    }
}

impl LmProblem<f64> for DensePgo<'_> {
    type Normal = ();
    type Step = DVector<f64>;

    fn energy(&self) -> f64 {
        self.cost(&self.poses)
    }

    fn linearize(&mut self) {
        self.h.fill(0.0);
        self.g.fill(0.0);
        for e in self.edges {
            let lin = relative_pose_residual(&e.from_t_to, &self.poses[e.from], &self.poses[e.to]);
            let r = e.sqrt_information * lin.residual;
            let blocks = [
                (self.variable[e.from], e.sqrt_information * lin.d_from),
                (self.variable[e.to], e.sqrt_information * lin.d_to),
            ];
            for (vi, ji) in &blocks {
                let Some(vi) = vi else { continue };
                let mut g = self.g.rows_mut(6 * vi, 6);
                g += ji.transpose() * r;
                for (vj, jj) in &blocks {
                    let Some(vj) = vj else { continue };
                    let mut h = self.h.view_mut((6 * vi, 6 * vj), (6, 6));
                    h += ji.transpose() * jj;
                }
            }
        }
    }

    fn solve_damped(&self, _: &(), lambda: f64) -> Option<DVector<f64>> {
        let mut diagonal = self.h.diagonal();
        marquardt_scaling(&mut diagonal, Default::default());
        let mut damped = self.h.clone();
        for i in 0..damped.nrows() {
            damped[(i, i)] += lambda * diagonal[i];
        }
        Some(damped.cholesky()?.solve(&-&self.g))
    }

    fn predicted_reduction(&self, _: &(), step: &DVector<f64>) -> f64 {
        predicted_reduction(step, &self.h, &self.g)
    }

    fn try_step(&mut self, step: DVector<f64>) -> f64 {
        self.trial = self.poses.clone();
        for (pose, variable) in self.trial.iter_mut().zip(&self.variable) {
            if let Some(v) = variable {
                pose.apply_inc(&step.fixed_rows::<6>(6 * v).into_owned());
            }
        }
        self.cost(&self.trial)
    }

    fn accept(&mut self) {
        std::mem::swap(&mut self.poses, &mut self.trial);
    }
}

#[test]
fn pgo_matches_a_dense_f64_reference() {
    let mut rng = Lcg(10);
    for (n, noise) in [(8, 0.01), (25, 0.003), (40, 0.02)] {
        let graph = drifting_graph(&mut rng, n, noise, &EdgeSigmas::default());
        let fixed = window_fixed(n, 3);
        let config = PgoConfig::default();

        let mut poses = graph.initial.clone();
        let report = optimize_pose_graph(&mut poses, &fixed, &graph.edges, &config).unwrap();

        let mut next = 0;
        let variable: Vec<Option<usize>> = fixed
            .iter()
            .map(|&f| {
                (!f).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        let mut dense = DensePgo {
            poses: graph.initial.clone(),
            trial: Vec::new(),
            variable,
            edges: &graph.edges,
            h: DMatrix::zeros(6 * next, 6 * next),
            g: DVector::zeros(6 * next),
        };
        let dense_report = levenberg_marquardt(
            &mut dense,
            &config.lm_config(),
            DampingRule::Nielsen {
                policy: Default::default(),
            },
        );
        assert_eq!(report.iterations, dense_report.iterations, "n = {n}");
        assert!(
            (report.final_cost - dense_report.final_energy).abs()
                <= 1e-9 * dense_report.final_energy.max(1e-12),
            "n = {n}: {} vs {}",
            report.final_cost,
            dense_report.final_energy
        );
        assert!(max_pose_difference(&poses, &dense.poses) < 1e-9, "n = {n}");
    }
}

// ───── upstream kornia-3d oracle ────────────────────────────────────────────

/// kornia-3d 0.2 `pose_graph_optimize`: f32, numeric Jacobians, dense LU, world->camera poses and
/// `T_ab = T_b T_a^-1`. A test-only oracle; it is too slow and too dense for the back end.
fn upstream_optimize(
    poses: &[RigidTransform<f64>],
    fixed: &[bool],
    edges: &[PoseGraphEdge],
) -> Vec<RigidTransform<f64>> {
    use kornia_3d::pgo::{pose_graph_optimize, PgoEdge, PgoParams};
    use kornia_3d::pose::Pose3d;
    use kornia_algebra::{Mat3F64, QuatF32, Vec3AF32, Vec3F64, SE3F32};

    let to_upstream = |pose: &RigidTransform<f64>| {
        let camera_from_world = pose.inverse();
        let m = camera_from_world.rotation.matrix();
        let t = camera_from_world.translation;
        Pose3d {
            rotation: Mat3F64::from_cols(
                Vec3F64::new(m[(0, 0)], m[(1, 0)], m[(2, 0)]),
                Vec3F64::new(m[(0, 1)], m[(1, 1)], m[(2, 1)]),
                Vec3F64::new(m[(0, 2)], m[(1, 2)], m[(2, 2)]),
            ),
            translation: Vec3F64::new(t.x, t.y, t.z),
        }
    };
    let upstream_edges: Vec<PgoEdge> = edges
        .iter()
        .map(|e| {
            // Our `from_t_to` is a_from_b; upstream wants b_from_a for world->camera poses.
            let b_from_a = e.from_t_to.inverse();
            let [x, y, z, w] = b_from_a.rotation.quaternion_xyzw();
            let t = b_from_a.translation;
            PgoEdge {
                pose_a: e.from,
                pose_b: e.to,
                t_ab_meas: SE3F32::from_qxyz(
                    QuatF32::from_xyzw(x as f32, y as f32, z as f32, w as f32),
                    Vec3AF32::new(t.x as f32, t.y as f32, t.z as f32),
                ),
                weight: 1.0,
            }
        })
        .collect();
    let fixed_indices: Vec<usize> = (0..poses.len()).filter(|&i| fixed[i]).collect();
    let params = PgoParams {
        max_iterations: 50,
        ..Default::default()
    };
    let result = pose_graph_optimize(
        &poses.iter().map(to_upstream).collect::<Vec<_>>(),
        &upstream_edges,
        &fixed_indices,
        &params,
    )
    .unwrap();
    result
        .poses
        .iter()
        .map(|p| {
            let r = p.rotation;
            let m = nalgebra::Matrix3::new(
                r.col(0).x,
                r.col(1).x,
                r.col(2).x,
                r.col(0).y,
                r.col(1).y,
                r.col(2).y,
                r.col(0).z,
                r.col(1).z,
                r.col(2).z,
            );
            let q = nalgebra::UnitQuaternion::from_matrix(&m);
            let camera_from_world = RigidTransform::new(
                Rotation3::from_unit_quaternion(q),
                Vector3::new(p.translation.x, p.translation.y, p.translation.z),
            );
            camera_from_world.inverse()
        })
        .collect()
}

#[test]
fn pgo_agrees_with_upstream_kornia_3d() {
    let mut rng = Lcg(11);
    for (n, noise) in [(6, 0.0), (12, 0.0), (12, 0.005)] {
        // Unit information: upstream has one scalar weight per edge.
        let mut graph = drifting_graph(&mut rng, n, noise, &EdgeSigmas::UNIT);
        let fixed = window_fixed(n, 2);
        for ((initial, truth), _) in graph
            .initial
            .iter_mut()
            .zip(&graph.truth)
            .zip(&fixed)
            .filter(|(_, &f)| f)
        {
            *initial = *truth;
        }
        let mut ours = graph.initial.clone();
        optimize_pose_graph(&mut ours, &fixed, &graph.edges, &PgoConfig::default()).unwrap();
        let upstream = upstream_optimize(&graph.initial, &fixed, &graph.edges);
        // Upstream is f32 with a coupled log residual in the other keyframe's frame: on a consistent graph
        // both reach the truth; on a noisy one the two optima differ at the noise level, and ours is the
        // better point of our cost.
        let diff = max_pose_difference(&ours, &upstream);
        if noise == 0.0 {
            assert!(diff < 1e-4, "n = {n}: {diff}");
            assert!(max_pose_difference(&ours, &graph.truth) < 1e-8);
        } else {
            assert!(diff < 4.0 * noise, "n = {n}, noise = {noise}: {diff}");
            let cost = |poses: &[RigidTransform<f64>]| {
                graph.edges.iter().map(|e| e.cost(poses)).sum::<f64>()
            };
            assert!(cost(&ours) <= cost(&upstream) * (1.0 + 1e-9));
        }
    }
}

// ───── RootSLAM-sized loop ──────────────────────────────────────────────────

/// The shape of a RootSLAM map at its 1500-keyframe cap: a spanning-tree chain, covisibility edges to
/// the next few keyframes, past loop edges, and the current loop from the newest keyframe to an island
/// near the start (`benches/pgo.rs` builds the same shape).
fn rootslam_loop_graph(n: usize) -> (Vec<RigidTransform<f64>>, Vec<bool>, Vec<PoseGraphEdge>) {
    let mut rng = Lcg(12);
    let sigmas = EdgeSigmas::default();
    let mut graph = drifting_graph(&mut rng, n, 0.001, &sigmas);
    for i in (0..n).step_by(5) {
        for k in [3, 4] {
            if i + k < n {
                let measured =
                    graph.truth[i].inverse() * graph.truth[i + k] * rng.pose(0.001, 0.001);
                graph
                    .edges
                    .push(PoseGraphEdge::new(i, i + k, measured, &sigmas));
            }
        }
    }
    for k in 0..5 {
        let measured = graph.truth[k].inverse() * graph.truth[n - 1] * rng.pose(0.001, 0.001);
        graph
            .edges
            .push(PoseGraphEdge::new(k, n - 1, measured, &sigmas));
    }
    (graph.initial, window_fixed(n, 8), graph.edges)
}

#[test]
fn a_1500_keyframe_loop_closes() {
    let (initial, fixed, edges) = rootslam_loop_graph(1500);
    let mut poses = initial.clone();
    let report = optimize_pose_graph(&mut poses, &fixed, &edges, &PgoConfig::default()).unwrap();
    assert!(report.iterations <= 20);
    // 0.001 noise on ~2,700 edges whitened by 0.05 m / 0.01 rad leaves a cost floor of ~15.
    assert!(report.final_cost < 1e-2 * report.initial_cost, "{report:?}");
    assert_eq!(report.termination, LmTermination::Tolerance);
}
