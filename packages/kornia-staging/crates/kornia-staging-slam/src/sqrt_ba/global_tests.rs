use super::*;
use kornia_staging_3d::camera::Pinhole;
use nalgebra::{DMatrix, DVector, Matrix3, UnitQuaternion};

type Rig = Vec<GbaRigCamera<f64, CameraModelKind<f64>>>;

fn wobble(i: usize, k: usize) -> f64 {
    (((i * 7919 + k * 104729) % 9973) as f64 * 0.6180339887).sin()
}

fn rotation(m: Matrix3<f64>) -> Rotation3<f64> {
    Rotation3::from_unit_quaternion(UnitQuaternion::from_matrix(&m))
}

/// `cameras` cameras on a body, `n_kf` keyframes on a circle looking at a
/// cloud of `n_lm` points, consecutive pose-graph rows. Returns (truth, noisy, rig).
fn synthetic(n_kf: usize, n_lm: usize, cameras: usize) -> (GbaProblem<f64>, GbaProblem<f64>, Rig) {
    let rig: Rig = (0..cameras)
        .map(|c| {
            let yaw = 0.15 * c as f64 - 0.2;
            GbaRigCamera {
                model: CameraModelKind::Pinhole(
                    Pinhole::new([300.0, 300.0, 320.0, 240.0]).unwrap(),
                ),
                imu_from_camera: RigidTransform::new(
                    Rotation3::exp(&Vector3::new(0.02 * c as f64, yaw, 0.01)),
                    Vector3::new(0.08 * c as f64 - 0.1, 0.01 * c as f64, 0.02),
                ),
            }
        })
        .collect();
    let world_from_imu: Vec<RigidTransform<f64>> = (0..n_kf)
        .map(|k| {
            let th = 0.25 * k as f64;
            let (s, c) = th.sin_cos();
            let m = Matrix3::from_columns(&[
                Vector3::new(-s, c, 0.0),
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(-c, -s, 0.0),
            ]);
            RigidTransform::new(
                rotation(m),
                Vector3::new(5.0 * c, 5.0 * s, 0.3 * (k as f64).sin()),
            )
        })
        .collect();
    let imu_from_world: Vec<_> = world_from_imu.iter().map(|p| p.inverse()).collect();
    let mut landmarks = Vec::new();
    for l in 0..n_lm {
        let p = Vector3::new(1.5 * wobble(l, 1), 1.5 * wobble(l, 2), 1.0 * wobble(l, 3));
        let mut observations = Vec::new();
        for (k, pose) in imu_from_world.iter().enumerate() {
            for (c, cam) in rig.iter().enumerate() {
                if (k + c + l) % 3 == 0 {
                    continue;
                }
                let p_c = cam.imu_from_camera.inverse() * (*pose * p);
                if let Ok(px) = cam.model.project([p_c.x, p_c.y, p_c.z]) {
                    if (0.0..640.0).contains(&px[0]) && (0.0..480.0).contains(&px[1]) {
                        observations.push(GbaObservation {
                            keyframe: k,
                            camera: c,
                            pixel: [
                                px[0] + 0.3 * wobble(l, 10 * k + c),
                                px[1] + 0.3 * wobble(l, 10 * k + c + 5),
                            ],
                        });
                    }
                }
            }
        }
        if observations.len() >= 2 {
            landmarks.push(GbaLandmark {
                position: p,
                observations,
            });
        }
    }
    let rel_poses = (1..n_kf)
        .map(|k| {
            RelPoseConstraint::with_sigmas(
                k - 1,
                k,
                imu_from_world[k - 1] * imu_from_world[k].inverse(),
                0.05,
                0.01,
            )
        })
        .collect();
    let truth = GbaProblem {
        imu_from_world,
        landmarks,
        rel_poses,
    };
    let mut noisy = truth.clone();
    for (k, pose) in noisy.imu_from_world.iter_mut().enumerate().skip(1) {
        let xi = Vector6::from_fn(|i, _| wobble(k, 50 + i) * if i < 3 { 0.03 } else { 0.01 });
        *pose = apply_left_increment(pose, &xi);
    }
    for (l, landmark) in noisy.landmarks.iter_mut().enumerate() {
        landmark.position += Vector3::from_fn(|i, _| 0.05 * wobble(l, 70 + i));
    }
    (truth, noisy, rig)
}

fn cameras(rig: &Rig) -> Vec<Camera<'_, f64, CameraModelKind<f64>>> {
    rig.iter()
        .map(|c| {
            let cam_from_imu = c.imu_from_camera.inverse();
            Camera {
                model: &c.model,
                adjoint: cam_from_imu.adjoint(),
                cam_from_imu,
            }
        })
        .collect()
}

#[test]
fn four_camera_rig_jacobians_match_central_differences() {
    let (_, problem, rig) = synthetic(6, 40, 4);
    let cams = cameras(&rig);
    let mut checked = 0;
    for landmark in problem.landmarks.iter().take(10) {
        for obs in &landmark.observations {
            let pose = problem.imu_from_world[obs.keyframe];
            let cam = &cams[obs.camera];
            let (_, jp, jl) = reprojection(obs, &landmark.position, &pose, cam, true).unwrap();
            let (jp, jl) = (jp.unwrap(), jl.unwrap());
            let h = 1e-6;
            for k in 0..6 {
                let mut d = Vector6::zeros();
                d[k] = h;
                let plus = reprojection(
                    obs,
                    &landmark.position,
                    &apply_left_increment(&pose, &d),
                    cam,
                    false,
                )
                .unwrap()
                .0;
                let minus = reprojection(
                    obs,
                    &landmark.position,
                    &apply_left_increment(&pose, &(-d)),
                    cam,
                    false,
                )
                .unwrap()
                .0;
                let numeric = (plus - minus) / (2.0 * h);
                assert!(
                    (numeric - jp.column(k)).amax() < 1e-4 * (1.0 + numeric.amax()),
                    "pose col {k}"
                );
            }
            for k in 0..3 {
                let mut d = Vector3::zeros();
                d[k] = h;
                let plus = reprojection(obs, &(landmark.position + d), &pose, cam, false)
                    .unwrap()
                    .0;
                let minus = reprojection(obs, &(landmark.position - d), &pose, cam, false)
                    .unwrap()
                    .0;
                let numeric = (plus - minus) / (2.0 * h);
                assert!(
                    (numeric - jl.column(k)).amax() < 1e-4 * (1.0 + numeric.amax()),
                    "point col {k}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 20);
}

/// The first LM step against the dense f64 solve of the full damped, scaled
/// system `(Js'Js + lambda I) d = -Js' r` over poses and landmarks.
#[test]
fn one_lm_step_matches_a_dense_damped_solve() {
    let (_, problem, rig) = synthetic(6, 30, 2);
    let cams = cameras(&rig);
    let n_kf = problem.imu_from_world.len();
    let n_lm = problem.landmarks.len();
    let cols = 6 * n_kf + 3 * n_lm;
    let mut rows_j: Vec<Vec<f64>> = Vec::new();
    let mut rows_r: Vec<f64> = Vec::new();
    for (l, landmark) in problem.landmarks.iter().enumerate() {
        for obs in &landmark.observations {
            let (res, jp, jl) = reprojection(
                obs,
                &landmark.position,
                &problem.imu_from_world[obs.keyframe],
                &cams[obs.camera],
                true,
            )
            .unwrap();
            let (jp, jl) = (jp.unwrap(), jl.unwrap());
            let e = res.norm();
            let sw = if e < 1.0 { 1.0 } else { (1.0 / e).sqrt() };
            for r in 0..2 {
                let mut row = vec![0.0; cols];
                for c in 0..6 {
                    row[6 * obs.keyframe + c] = sw * jp[(r, c)];
                }
                for c in 0..3 {
                    row[6 * n_kf + 3 * l + c] = sw * jl[(r, c)];
                }
                rows_j.push(row);
                rows_r.push(sw * res[r]);
            }
        }
    }
    for c in &problem.rel_poses {
        let (mut ja, mut jb) = (Matrix6::zeros(), Matrix6::zeros());
        let r = c.sqrt_info
            * relative_pose_error(
                &c.a_from_b,
                &problem.imu_from_world[c.frame_a],
                &problem.imu_from_world[c.frame_b],
                Some(&mut ja),
                Some(&mut jb),
            );
        let (ja, jb) = (c.sqrt_info * ja, c.sqrt_info * jb);
        for i in 0..6 {
            let mut row = vec![0.0; cols];
            for j in 0..6 {
                row[6 * c.frame_a + j] = ja[(i, j)];
                row[6 * c.frame_b + j] = jb[(i, j)];
            }
            rows_j.push(row);
            rows_r.push(r[i]);
        }
    }
    let j = DMatrix::from_fn(rows_j.len(), cols, |r, c| rows_j[r][c]);
    let r = DVector::from_vec(rows_r);
    let scale = DVector::from_fn(cols, |c, _| 1.0 / (f64::EPSILON + j.column(c).norm()));
    let js = &j * DMatrix::from_diagonal(&scale);
    let lambda = 1e-4;
    let h = js.transpose() * &js + DMatrix::identity(cols, cols) * lambda;
    let step_s = h.cholesky().unwrap().solve(&(-(js.transpose() * &r)));
    let step = step_s.component_mul(&scale);
    let jd = &j * &step;
    let model_decrease = -jd.dot(&(&r + 0.5 * &jd));

    let mut solved = problem.clone();
    let options = GbaOptions {
        max_iterations: 1,
        pcg: PcgOptions {
            max_iterations: 2000,
            eta: 1e-15,
        },
        ..Default::default()
    };
    let report = solve_global(&mut solved, &rig, &options, None).unwrap();
    let it = report.iterations[0];
    assert!(it.accepted, "{it:?}");
    assert!(
        (it.model_decrease - model_decrease).abs() < 1e-7 * model_decrease,
        "{} vs {model_decrease}",
        it.model_decrease
    );
    for k in 0..n_kf {
        let expected = apply_left_increment(
            &problem.imu_from_world[k],
            &Vector6::from_fn(|i, _| step[6 * k + i]),
        );
        let got = solved.imu_from_world[k];
        let dt = (expected.translation - got.translation).amax();
        let dr = (expected.rotation.inverse() * got.rotation).log().amax();
        let st = Vector6::from_fn(|i, _| step[6 * k + i]).amax();
        assert!(
            dt < 1e-6 * st && dr < 1e-6 * st,
            "kf {k}: dt {dt} dr {dr} step {st}"
        );
    }
    for l in 0..n_lm {
        let expected =
            problem.landmarks[l].position + Vector3::from_fn(|i, _| step[6 * n_kf + 3 * l + i]);
        assert!(
            (expected - solved.landmarks[l].position).amax() < 1e-7,
            "lm {l}"
        );
    }
}

/// The synthetic rig in f32 (the product lane).
fn rig32(rig: &Rig) -> Vec<GbaRigCamera<f32, CameraModelKind<f32>>> {
    rig.iter()
        .map(|c| GbaRigCamera {
            model: match c.model {
                CameraModelKind::Pinhole(_) => {
                    CameraModelKind::Pinhole(Pinhole::new([300.0f32, 300.0, 320.0, 240.0]).unwrap())
                }
                _ => unreachable!(),
            },
            imu_from_camera: c.imu_from_camera.cast(),
        })
        .collect()
}

/// The same scene with the world origin moved by `-offset` (every world point
/// gains `offset`).
fn shifted(problem: &GbaProblem<f64>, offset: Vector3<f64>) -> GbaProblem<f64> {
    let mut out = problem.clone();
    for pose in &mut out.imu_from_world {
        *pose = *pose * RigidTransform::new(Rotation3::identity(), -offset);
    }
    for l in &mut out.landmarks {
        l.position += offset;
    }
    out
}

fn pose_rms(a: &GbaProblem<f64>, b: &GbaProblem<f64>) -> f64 {
    let sum: f64 = a
        .imu_from_world
        .iter()
        .zip(&b.imu_from_world)
        .map(|(x, y)| (x.inverse().translation - y.inverse().translation).norm_squared())
        .sum();
    (sum / a.imu_from_world.len() as f64).sqrt()
}

#[test]
fn f32_reaches_the_f64_result_within_the_band() {
    let (truth, problem, rig) = synthetic(12, 120, 2);
    let mut solved64 = problem.clone();
    let r64 = solve_global(&mut solved64, &rig, &GbaOptions::default(), None).unwrap();
    let (mut centred32, centroid) = problem.centred::<f32>();
    let r32 = solve_global(&mut centred32, &rig32(&rig), &GbaOptions::default(), None).unwrap();
    let mut solved32 = problem.clone();
    solved32.uncentre_from(&centred32, &centroid);
    assert!(
        r64.final_cost.error < 0.5 * r64.initial_cost.error,
        "{r64:?}"
    );
    // DESIGN §7.4: final cost f32 vs f64 within 1e-3 relative, KF poses within 1 mm.
    let rel = (r32.final_cost.error - r64.final_cost.error).abs() / r64.final_cost.error;
    assert!(
        rel < 1e-3,
        "f32 {} vs f64 {}",
        r32.final_cost.error,
        r64.final_cost.error
    );
    assert!(
        pose_rms(&solved32, &solved64) < 1e-3,
        "{}",
        pose_rms(&solved32, &solved64)
    );
    // From the noisy start it reaches the minimum it reaches from the truth
    // (the gauge is free, as in rootba, so poses are compared by cost only).
    let from_truth = solve_global(&mut truth.clone(), &rig, &GbaOptions::default(), None).unwrap();
    assert!(
        (r64.final_cost.error - from_truth.final_cost.error).abs()
            < 1e-6 * from_truth.final_cost.error
    );
}

fn assert_thread_count_independent<S: Scalar>(
    problem: &GbaProblem<S>,
    rig: &[GbaRigCamera<S, CameraModelKind<S>>],
) {
    let mut first: Option<GbaProblem<S>> = None;
    for threads in [0, 1, 3, 8] {
        let pool = (threads > 0).then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        });
        let mut solved = problem.clone();
        solve_global(&mut solved, rig, &GbaOptions::default(), pool.as_ref()).unwrap();
        match &first {
            None => first = Some(solved),
            Some(f) => assert_eq!(f, &solved, "threads {threads}"),
        }
    }
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let (_, problem, rig) = synthetic(10, 80, 2);
    assert_thread_count_independent(&problem, &rig);
    // The product lane: f32 after centring.
    assert_thread_count_independent(&problem.centred::<f32>().0, &rig32(&rig));
}

/// DESIGN B24: 10 km from the origin an f32 world point resolves to about 1 mm,
/// which costs ~0.1 px at these depths. Centring in f64 before the cast keeps
/// the f32 solve within the §7.4 band of f64; the uncentred cast leaves it.
#[test]
fn f32_far_from_the_origin_needs_centring() {
    let (_, near, rig) = synthetic(12, 120, 2);
    let problem = shifted(&near, Vector3::new(1e4, -2e3, 5.0));
    let mut solved64 = problem.clone();
    let r64 = solve_global(&mut solved64, &rig, &GbaOptions::default(), None).unwrap();
    let rig32 = rig32(&rig);
    let band = |r32: &GbaReport, solved32: &GbaProblem<f64>| {
        (
            (r32.final_cost.error - r64.final_cost.error).abs() / r64.final_cost.error,
            pose_rms(solved32, &solved64),
        )
    };

    let (mut centred32, centroid) = problem.centred::<f32>();
    let r32 = solve_global(&mut centred32, &rig32, &GbaOptions::default(), None).unwrap();
    let mut solved32 = problem.clone();
    solved32.uncentre_from(&centred32, &centroid);
    let (rel, rms) = band(&r32, &solved32);
    assert!(
        rel < 1e-3 && rms < 1e-3,
        "centred f32: cost rel {rel}, KF RMS {rms} m"
    );

    let mut uncentred32 = problem.cast::<f32>();
    let r32 = solve_global(&mut uncentred32, &rig32, &GbaOptions::default(), None).unwrap();
    let (rel, rms) = band(&r32, &uncentred32.cast());
    assert!(
        rel >= 1e-3 || rms >= 1e-3,
        "uncentred f32 stayed in the band (cost rel {rel}, KF RMS {rms} m): the test no longer shows the fix"
    );
}

/// Landmarks moved by up to 3 m, a tiny `lambda` and PCG run to tolerance: the
/// first Gauss-Newton-like attempts overshoot and are rejected. A
/// rejected attempt leaves the state bit-identical, multiplies `lambda` by
/// `vee` (x2, then x4) and damps the same linearization again: the attempt that
/// is finally accepted equals a fresh solve started at its `lambda`, bit for bit.
#[test]
fn a_rejected_step_redamps_the_same_linearization() {
    let (_, mut start, rig) = synthetic(8, 60, 2);
    for (k, pose) in start.imu_from_world.iter_mut().enumerate().skip(1) {
        let xi = Vector6::from_fn(|i, _| wobble(k, 90 + i) * if i < 3 { 0.05 } else { 0.02 });
        *pose = apply_left_increment(pose, &xi);
    }
    for (l, landmark) in start.landmarks.iter_mut().enumerate() {
        landmark.position += Vector3::from_fn(|i, _| 3.0 * wobble(l, 120 + i));
    }
    let options = |max_iterations: usize, initial_lambda: f64| GbaOptions {
        max_iterations,
        initial_lambda,
        lambda_min: 1e-14,
        pcg: PcgOptions {
            max_iterations: 500,
            eta: 1e-10,
        },
        ..Default::default()
    };
    let lambda0 = 1e-12;

    let mut once = start.clone();
    let first = solve_global(&mut once, &rig, &options(1, lambda0), None).unwrap();
    let it = first.iterations[0];
    let cost_after = it.cost_after.expect("a finite step whose cost rose");
    assert!(!it.accepted, "{it:?}");
    assert!(cost_after.error > first.initial_cost.error, "{it:?}");
    assert_eq!(once, start, "a rejected attempt changed the state");
    assert_eq!(first.final_cost, first.initial_cost);

    let mut solved = start.clone();
    let report = solve_global(&mut solved, &rig, &options(20, lambda0), None).unwrap();
    let its = &report.iterations;
    let k = its
        .iter()
        .position(|i| i.accepted)
        .expect("an accepted step");
    assert!(
        k >= 2,
        "need two rejections before the first accepted step: {its:?}"
    );
    assert_eq!(its[0].lambda, lambda0);
    assert_eq!(its[1].lambda, 2.0 * lambda0);
    assert_eq!(its[2].lambda, 4.0 * its[1].lambda);
    // Up to and including step k, one linearization was damped k + 1 times.
    let mut upto = start.clone();
    solve_global(&mut upto, &rig, &options(k + 1, lambda0), None).unwrap();
    let mut fresh = start.clone();
    let fresh_report = solve_global(&mut fresh, &rig, &options(1, its[k].lambda), None).unwrap();
    assert_eq!(fresh_report.iterations[0].pcg, its[k].pcg);
    assert_eq!(
        fresh_report.iterations[0].model_decrease.to_bits(),
        its[k].model_decrease.to_bits()
    );
    assert_eq!(fresh_report.iterations[0].cost_after, its[k].cost_after);
    assert_eq!(
        fresh, upto,
        "the re-damped step differs from a fresh solve at its lambda"
    );
    assert!(report.final_cost.error < report.initial_cost.error);
}

#[test]
fn error_valid_counts_only_points_in_front_of_the_camera() {
    let (_, mut problem, rig) = synthetic(6, 30, 2);
    let before = solve_global(
        &mut problem.clone(),
        &rig,
        &GbaOptions {
            max_iterations: 0,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    // Move one landmark through the centre: rows with z <= 0 leave the cost.
    problem.landmarks[0].position *= -40.0;
    let cams = cameras(&rig);
    let behind = problem.landmarks[0]
        .observations
        .iter()
        .filter(|o| {
            (cams[o.camera].cam_from_imu
                * (problem.imu_from_world[o.keyframe] * problem.landmarks[0].position))
                .z
                <= 0.0
        })
        .count();
    assert!(behind >= 1);
    let report = solve_global(
        &mut problem,
        &rig,
        &GbaOptions {
            max_iterations: 3,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert!(report.initial_cost.invalid >= behind);
    assert_eq!(
        report.initial_cost.valid + report.initial_cost.invalid,
        before.initial_cost.valid
    );
    assert!(report.final_cost.error.is_finite());
    assert_eq!(report.nonfinite_landmark_steps, 0);
}

#[test]
fn sparse_keyframes_keep_their_pose_graph_rows() {
    let (_, mut problem, rig) = synthetic(6, 30, 2);
    // Keyframe 5 keeps only one observation: under min_obs_per_kf it loses the
    // reprojection rows (B22), but keeps both pose-graph rows and its pose.
    let mut kept = false;
    for landmark in &mut problem.landmarks {
        landmark.observations.retain(|o| {
            let keep = o.keyframe != 5 || !kept;
            kept |= o.keyframe == 5;
            keep
        });
    }
    let single = problem
        .landmarks
        .iter()
        .flat_map(|l| &l.observations)
        .filter(|o| o.keyframe == 5)
        .count();
    assert!(single >= 1);
    problem.landmarks.push(GbaLandmark {
        position: Vector3::zeros(),
        observations: vec![GbaObservation {
            keyframe: 0,
            camera: 0,
            pixel: [1.0, 1.0],
        }],
    });
    let (obs, lms) = problem.drop_sparse_observations(10);
    assert_eq!(obs, single);
    assert!(lms >= 1);
    assert!(problem.landmarks.iter().all(|l| l.observations.len() >= 2));
    assert!(problem
        .landmarks
        .iter()
        .flat_map(|l| &l.observations)
        .all(|o| o.keyframe != 5));
    assert_eq!(problem.imu_from_world.len(), 6);
    assert_eq!(
        problem.rel_poses.iter().filter(|c| c.frame_b == 5).count(),
        1
    );
    let start = problem.imu_from_world[5];
    let report = solve_global(&mut problem, &rig, &GbaOptions::default(), None).unwrap();
    assert!(report.final_cost.error < report.initial_cost.error);
    assert_ne!(
        problem.imu_from_world[5], start,
        "the pose-graph row still moves keyframe 5"
    );
}

#[test]
fn fixed_keyframes_stay_bit_identical() {
    let (_, mut problem, rig) = synthetic(8, 60, 2);
    let start = problem.imu_from_world.clone();
    let options = GbaOptions {
        fixed_keyframes: vec![0, 3],
        ..Default::default()
    };
    let report = solve_global(&mut problem, &rig, &options, None).unwrap();
    assert!(report.iterations.iter().any(|i| i.accepted));
    assert_eq!(problem.imu_from_world[0], start[0]);
    assert_eq!(problem.imu_from_world[3], start[3]);
    assert_ne!(problem.imu_from_world[1], start[1]);
    assert!(matches!(
        solve_global(
            &mut problem,
            &rig,
            &GbaOptions {
                fixed_keyframes: vec![99],
                ..Default::default()
            },
            None
        ),
        Err(GbaError::Keyframe(99))
    ));
}

#[test]
fn centring_round_trips_in_f64() {
    let (_, mut problem, _) = synthetic(5, 20, 1);
    for pose in &mut problem.imu_from_world {
        *pose = *pose * RigidTransform::new(Rotation3::identity(), Vector3::new(-1e4, 2e3, 5.0));
    }
    for l in &mut problem.landmarks {
        l.position += Vector3::new(1e4, -2e3, -5.0);
    }
    let (centred, c) = problem.centred::<f64>();
    assert!(
        centred
            .landmarks
            .iter()
            .map(|l| l.position)
            .sum::<Vector3<f64>>()
            .amax()
            < 1e-6
    );
    let mut back = problem.clone();
    back.uncentre_from(&centred, &c);
    for (a, b) in back.imu_from_world.iter().zip(&problem.imu_from_world) {
        assert!((a.translation - b.translation).amax() < 1e-8);
    }
    for (a, b) in back.landmarks.iter().zip(&problem.landmarks) {
        assert!((a.position - b.position).amax() < 1e-9);
    }
}

#[test]
fn invalid_inputs_are_rejected() {
    let (_, problem, rig) = synthetic(4, 10, 1);
    let mut p = problem.clone();
    p.landmarks[0].observations.truncate(1);
    assert_eq!(
        solve_global(&mut p, &rig, &GbaOptions::default(), None).unwrap_err(),
        GbaError::Observations(0)
    );
    let mut p = problem.clone();
    p.landmarks[0].observations[0].camera = 7;
    assert!(matches!(
        solve_global(&mut p, &rig, &GbaOptions::default(), None),
        Err(GbaError::Index { .. })
    ));
    let mut p = problem;
    p.rel_poses[0].frame_b = 40;
    assert_eq!(
        solve_global(&mut p, &rig, &GbaOptions::default(), None).unwrap_err(),
        GbaError::Keyframe(40)
    );
}
