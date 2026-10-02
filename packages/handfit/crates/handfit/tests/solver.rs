use handfit::model::{LandmarkJacobian, Step};
use handfit::residual::{
    evaluate, evaluate_rigid, linearize, normal_equations, rigid_landmarks, Jacobian, RigidJacobian,
};
use handfit::{fit, Config, JacobianMode, Model, Pose, Termination, View, Views};
use nalgebra::{Matrix3, Rotation3, SMatrix, SVector, Vector2, Vector3};

fn problem() -> (Model, Pose, View) {
    let model = Model {
        axes: SMatrix::from_fn(|i, k| if k == i % 3 { 1.0 } else { 0.0 }),
        pivots: SMatrix::from_fn(|i, k| ((i * 3 + k) as f64).cos() * 20.0),
        rest: SMatrix::from_fn(|i, k| ((i * 3 + k) as f64).sin() * 50.0),
        weights: SMatrix::from_fn(|_, k| if k == 0 { 1.0 } else { 0.0 }),
        limits: SMatrix::from_fn(|_, k| if k == 0 { -1.0 } else { 1.0 }),
    };
    let pose = Pose {
        rotation: Matrix3::identity(),
        translation: Vector3::new(0.02, -0.01, 0.6),
        angles: SVector::repeat(0.1),
    };
    let p = model.landmarks(&pose, 1.0, &Step::zeros(), None);
    let view = View {
        rotation: Matrix3::identity(),
        translation: Vector3::zeros(),
        focal: Vector2::repeat(500.0),
        principal: Vector2::new(320.0, 240.0),
        distortion: None,
        pixels: SMatrix::from_fn(|i, k| {
            500.0 * p[3 * i + k] / p[3 * i + 2] + if k == 0 { 320.0 } else { 240.0 }
        }),
        distances: SVector::from_fn(|i, _| p.fixed_rows::<3>(3 * i).norm() * 1000.0),
        weights: SVector::repeat(1.0),
    };
    (model, pose, view)
}

#[test]
fn synthetic_hand_fit_lowers_energy_and_recovers_landmarks() {
    let (model, target, view) = problem();
    let config = Config {
        max_iterations: 50,
        temporal_weight: 0.0,
        relative_tolerance: 1e-9,
        ..Config::default()
    };
    let mut prior = target.clone();
    prior.translation += Vector3::new(0.005, -0.003, 0.02);
    for i in 0..20 {
        prior.angles[i] += 0.01;
    }
    for mode in [JacobianMode::Analytic, JacobianMode::CentralDifference] {
        let result = fit(
            &model,
            &config,
            &prior,
            1.0,
            std::slice::from_ref(&view),
            mode,
        )
        .unwrap();
        let a = model.landmarks(&result.pose, 1.0, &Step::zeros(), None);
        let b = model.landmarks(&target, 1.0, &Step::zeros(), None);
        assert!((a - b).norm() < 1e-5, "{:?}", result);
        assert!(result.energies[3] < 1e-6);
        assert!(result.converged);
        assert!((result.pose.rotation.determinant() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn stationary_iterations_and_limit_clamp() {
    let (model, target, view) = problem();
    let config = Config {
        max_iterations: 50,
        temporal_weight: 0.0,
        ..Config::default()
    };
    let mut exact_model = model.clone();
    exact_model.rest.fill(0.0);
    exact_model.pivots.fill(0.0);
    let exact_pose = Pose {
        rotation: Matrix3::identity(),
        translation: Vector3::new(0.0, 0.0, 1.0),
        angles: SVector::zeros(),
    };
    let mut exact_view = view.clone();
    exact_view.pixels = SMatrix::from_fn(|_, k| exact_view.principal[k]);
    exact_view.distances.fill(1000.0);
    let result = fit(
        &exact_model,
        &config,
        &exact_pose,
        1.0,
        &[exact_view],
        JacobianMode::Analytic,
    )
    .unwrap();
    assert!(result.converged, "{result:?}");
    assert_eq!(result.termination, Termination::Stationary);
    let mut outside = target;
    outside.angles[0] = 8.0;
    outside.angles[1] = -8.0;
    let config = Config {
        max_iterations: 0,
        ..config
    };
    let result = fit(
        &model,
        &config,
        &outside,
        1.0,
        &[view],
        JacobianMode::Analytic,
    )
    .unwrap();
    assert_eq!(result.termination, Termination::Iterations);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.pose.angles[0], 1.0 + config.margin);
    assert_eq!(result.pose.angles[1], -1.0 - config.margin);
}

#[test]
fn residual_chain_matches_numeric_with_palm_reference_and_temporal_prior() {
    let (model, target, mut view) = problem();
    view.weights[5] = 0.0;
    view.pixels.row_mut(5).fill(f64::NAN);
    view.distances[5] = f64::NAN;
    let mut pose = target.clone();
    pose.translation.x += 0.01;
    pose.angles[7] += 0.02;
    let config = Config::default();
    let views = [view];
    let views = Views::new(&views).unwrap();
    let (r, j) = linearize(
        &model,
        &config,
        &pose,
        &target,
        -1.0,
        views,
        JacobianMode::Analytic,
    );
    assert!(r.iter().all(|v| v.is_finite()));
    for k in 0..26 {
        let mut delta = Step::zeros();
        delta[k] = 1e-6;
        let a = evaluate(&model, &config, &pose, &target, -1.0, views, &delta, None);
        delta[k] = -1e-6;
        let b = evaluate(&model, &config, &pose, &target, -1.0, views, &delta, None);
        let numeric = (a - b) / 2e-6;
        let error = (numeric - j.row(k).transpose()).norm() / numeric.norm().max(1e-10);
        assert!(error < 1e-6, "column {k}: {error}");
    }
    // An observed, exact synthetic hand also has a finite generated Jacobian at zero angles.
    let mut flat = target;
    flat.angles.fill(0.0);
    let mut jac = LandmarkJacobian::zeros();
    model.landmarks(&flat, 1.0, &Step::zeros(), Some(&mut jac));
    assert!(jac.iter().all(|v| v.is_finite()));
}

#[test]
fn warm_without_observations_runs_prior_only() {
    let (model, prior, mut view) = problem();
    view.weights.fill(0.0);
    for temporal_weight in [0.0, 20.0] {
        let config = Config {
            temporal_weight,
            ..Config::default()
        };
        let result = fit(
            &model,
            &config,
            &prior,
            1.0,
            std::slice::from_ref(&view),
            JacobianMode::Analytic,
        )
        .unwrap();
        assert_eq!(result.energies, [0.0; 4]);
        assert_eq!(result.pose.translation, prior.translation);
        assert_eq!(result.termination, Termination::Stationary);
        assert!(result.converged);
        assert!(result.iterations > 0);
    }
}

#[test]
fn cold_synthetic_one_and_two_views_both_lenses() {
    use handfit::cold::initial_pose;
    use handfit::residual::project;
    let (model, mut target, view) = problem();
    target.angles.fill(0.12);
    for mirror in [1.0, -1.0] {
        let points = model.landmarks(&target, mirror, &Step::zeros(), None);
        for fisheye in [false, true] {
            let mut views = vec![view.clone(), view.clone()];
            views[1].translation.x = -0.15;
            for v in &mut views {
                if fisheye {
                    v.distortion = Some(SVector::from_row_slice(&[
                        0.01, -0.002, 0.0, 0.0, 0.0, 0.0, 0.001, -0.002,
                    ]));
                }
                for i in 0..21 {
                    let p = v.rotation * points.fixed_rows::<3>(i * 3) + v.translation;
                    let pixel = project(v, &p);
                    v.pixels.row_mut(i).copy_from(&pixel.transpose());
                    v.distances[i] = p.norm() * 1000.0 - target.translation.norm() * 1000.0;
                }
            }
            for count in [1, 2] {
                let mut displaced = target.clone();
                displaced.translation += Vector3::new(0.02, -0.01, 0.05);
                let mut palm_views = views[..count].to_vec();
                for view in &mut palm_views {
                    for i in 0..21 {
                        if !handfit::cold::PALM.contains(&i) {
                            view.weights[i] = 0.0;
                        }
                    }
                }
                let rigid_config = Config {
                    temporal_weight: 0.0,
                    max_iterations: 40,
                    relative_tolerance: 1e-6,
                    ..Config::default()
                };
                let rigid = handfit::lm::solve(
                    &model,
                    &rigid_config,
                    &displaced,
                    mirror,
                    Views::new(&palm_views).unwrap(),
                    JacobianMode::Analytic,
                    true,
                );
                assert_eq!(rigid.pose.angles, target.angles);
                assert!((rigid.pose.translation - target.translation).norm() < 1e-5);
                assert!(rigid.energies[3] < 1e-5);
                let result = initial_pose(
                    &model,
                    &Config::default(),
                    mirror,
                    &views[..count],
                    JacobianMode::Analytic,
                )
                .unwrap();
                let actual = model.landmarks(&result.pose, mirror, &Step::zeros(), None);
                assert!(
                    (actual - points).norm() < 1e-4,
                    "views={count}, fisheye={fisheye}: {result:?}"
                );
                assert!(result.energies[3] < 1e-5);
                assert!(result.converged);
            }
        }
    }
}

#[test]
fn identical_cold_wrists_use_rank_fallback_and_first_full_tie() {
    let (model, _, view) = problem();
    let config = Config {
        init_iterations: 0,
        rotation_hypotheses: 0,
        ..Config::default()
    };
    let result = handfit::cold::initial_pose(
        &model,
        &config,
        1.0,
        &[view.clone(), view],
        JacobianMode::Analytic,
    )
    .unwrap();
    let diagnostics = result.cold.unwrap();
    // Both aligned wrists coincide, so neither is more than 30 degrees from the best.
    assert_eq!(diagnostics.chosen, vec![0, 1]);
    assert_eq!(diagnostics.full_iterations, vec![0; 4]);
    // Neutral and open are identical for these symmetric limits: all four energies tie.
    assert_eq!(diagnostics.winner, 0);
}

#[test]
fn a_zero_weight_bone_slot_may_name_any_bone() {
    use handfit::generated;
    use handfit::model::Model;
    let mut indices: Vec<i64> = generated::BONE_INDICES.iter().flatten().copied().collect();
    let topology: Vec<i64> = [
        generated::JOINT_PARENT,
        generated::JOINT_FRAME_INDEX,
        generated::JOINT_FIRST_CHILD,
        generated::JOINT_NEXT_SIBLING,
    ]
    .concat();
    let mut weights: Vec<f64> = vec![1.0 / 3.0; 63];
    assert!(Model::validate_topology(&indices, &weights, &topology).is_ok());
    // landmark 20, slot 2: another bone at weight 0 is accepted (UmeTrack synthetic profiles), at a non-zero weight it is not
    indices[20 * 3 + 2] += 3;
    assert!(Model::validate_topology(&indices, &weights, &topology).is_err());
    weights[20 * 3 + 2] = 0.0;
    assert!(Model::validate_topology(&indices, &weights, &topology).is_ok());
}

#[test]
fn a_checked_model_refuses_what_the_kernels_cannot_fit() {
    use handfit::generated;
    use handfit::model::ModelError;
    let (model, _, _) = problem();
    let mut indices: Vec<i64> = generated::BONE_INDICES.iter().flatten().copied().collect();
    let topology: Vec<i64> = [
        generated::JOINT_PARENT,
        generated::JOINT_FRAME_INDEX,
        generated::JOINT_FIRST_CHILD,
        generated::JOINT_NEXT_SIBLING,
    ]
    .concat();
    let checked = |weights: SMatrix<f64, 21, 3>, limits: SMatrix<f64, 20, 2>, indices: &[i64]| {
        Model::new(
            model.axes,
            model.pivots,
            model.rest,
            weights,
            limits,
            indices,
            &topology,
        )
        .map(|_| ())
    };
    assert_eq!(checked(model.weights, model.limits, &indices), Ok(()));
    // Landmark 0's second slot names another bone: accepted at weight 0, refused at a non-zero weight.
    indices[1] += 1;
    assert_eq!(checked(model.weights, model.limits, &indices), Ok(()));
    let mut weights = model.weights;
    weights[(0, 1)] = 0.5;
    assert_eq!(
        checked(weights, model.limits, &indices),
        Err(ModelError::Topology)
    );
    // All three of its slots name bone 4: two non-zero weights on it are a blend torch does not compute.
    indices[1] -= 1;
    assert_eq!(
        checked(weights, model.limits, &indices),
        Err(ModelError::DuplicateBone)
    );
    let mut limits = model.limits;
    limits[(3, 0)] = 2.0;
    assert_eq!(
        checked(model.weights, limits, &indices),
        Err(ModelError::NonFinite)
    );
}

#[test]
fn config_numbers_are_checked_and_unpacked_in_binding_order() {
    use handfit::lm::ConfigError;
    let warm = [1.1, 0.04, 20.0, 0.1, 0.087, 10.0, 1e-3, 1e-6, 1e-4];
    let config = Config::from_numbers(&warm).unwrap();
    let defaults = Config::default();
    assert_eq!(
        (
            config.phi,
            config.margin,
            config.max_iterations,
            config.initial_damping
        ),
        (1.1, 0.087, 10, 1e-4)
    );
    assert_eq!(
        (
            config.init_iterations,
            config.rotation_hypotheses,
            config.finger_starts
        ),
        (
            defaults.init_iterations,
            defaults.rotation_hypotheses,
            defaults.finger_starts
        )
    );
    let cold = [&warm[..], &[30.0, 1e-5, 12.0, 1.0, 1.0]].concat();
    let config = Config::from_numbers(&cold).unwrap();
    assert_eq!(
        (
            config.init_iterations,
            config.init_relative_tolerance,
            config.rotation_hypotheses,
            config.full_fit_hypotheses,
            config.finger_starts
        ),
        (30, 1e-5, 12, 1, 1)
    );
    assert_eq!(
        Config::from_numbers(&cold[..10]).unwrap_err(),
        ConfigError::Length
    );
    for (index, value, error) in [
        (5, 2.5, ConfigError::Warm),
        (12, f64::NAN, ConfigError::Warm),
        (13, 3.0, ConfigError::Cold),
        (11, 25.0, ConfigError::Cold),
    ] {
        let mut numbers = cold.clone();
        numbers[index] = value;
        assert_eq!(
            Config::from_numbers(&numbers).unwrap_err(),
            error,
            "number {index}"
        );
    }
}

/// Two views (pinhole and fisheye), an unobserved palm reference in the second, a temporal prior and a rotated wrist.
fn two_view_problem() -> (Model, Pose, Pose, Vec<View>) {
    let (model, target, view) = problem();
    let mut pose = target.clone();
    pose.rotation = Rotation3::from_euler_angles(0.3, -0.2, 0.5).into_inner();
    pose.translation += Vector3::new(0.01, 0.02, -0.03);
    pose.angles[3] += 0.05;
    let mut second = view.clone();
    second.translation.x = -0.15;
    second.distortion = Some(SVector::from_row_slice(&[
        0.01, -0.002, 0.0005, 0.0, 0.0, 0.0, 0.001, -0.002,
    ]));
    second.weights[5] = 0.0;
    second.weights[9] = 0.0;
    (model, pose, target, vec![view, second])
}

#[test]
fn rigid_residual_and_jacobian_match_the_full_chain() {
    let (model, pose, prior, views) = two_view_problem();
    let views = Views::new(&views).unwrap();
    for (mirror, temporal_weight) in [(1.0, 20.0), (-1.0, 0.0)] {
        let config = Config {
            temporal_weight,
            ..Config::default()
        };
        let mut full = Jacobian::zeros();
        let r = evaluate(
            &model,
            &config,
            &pose,
            &prior,
            mirror,
            views,
            &Step::zeros(),
            Some(&mut full),
        );
        let local = rigid_landmarks(&model, &pose, mirror);
        let mut rigid = RigidJacobian::zeros();
        let r6 = evaluate_rigid(&config, &pose, &prior, &local, views, Some(&mut rigid));
        assert!(
            (r - r6).norm() <= 1e-12 * r.norm(),
            "residual {}",
            (r - r6).norm()
        );
        let reference = full.fixed_rows::<6>(0).into_owned();
        assert!(
            (reference - rigid).norm() <= 1e-12 * reference.norm(),
            "jacobian {}",
            (reference - rigid).norm()
        );
        assert_eq!(
            evaluate_rigid(&config, &pose, &prior, &local, views, None),
            r6
        );
    }
}

#[test]
fn normal_equations_skip_only_rows_that_are_zero() {
    let (model, pose, prior, views) = two_view_problem();
    for (count, temporal_weight) in [(2, 20.0), (1, 0.0), (1, 20.0)] {
        let config = Config {
            temporal_weight,
            ..Config::default()
        };
        let views = Views::new(&views[..count]).unwrap();
        let mut j = Jacobian::zeros();
        let r = evaluate(
            &model,
            &config,
            &pose,
            &prior,
            1.0,
            views,
            &Step::zeros(),
            Some(&mut j),
        );
        let (h, g) = normal_equations(&config, views, &r, &j);
        let (want_h, want_g) = (j * j.transpose(), j * r);
        assert!(
            (h - want_h).norm() <= 1e-12 * want_h.norm(),
            "views={count}"
        );
        assert!(
            (g - want_g).norm() <= 1e-12 * want_g.norm(),
            "views={count}"
        );
        assert_eq!(h, h.transpose());
    }
}

#[test]
fn a_hand_takes_up_to_two_views_and_more_are_refused_at_every_entry_point() {
    use handfit::cold::{initial_pose, initial_pose_parallel};
    use handfit::residual::project;
    use handfit::scale::{calibrate_scale, CalibrationBlock, CalibrationConfig, ScaleError};
    use handfit::FitError;
    let (model, target, view) = problem();
    // Two consistent views of `target`, the second 15 cm to the side; a third repeats the first.
    let points = model.landmarks(&target, 1.0, &Step::zeros(), None);
    let mut views = vec![view.clone(), view];
    views[1].translation.x = -0.15;
    for v in &mut views {
        for i in 0..21 {
            let p = v.rotation * points.fixed_rows::<3>(i * 3) + v.translation;
            let pixel = project(v, &p);
            v.pixels.row_mut(i).copy_from(&pixel.transpose());
            v.distances[i] = p.norm() * 1000.0;
        }
    }
    views.push(views[0].clone());
    for count in 0..=3 {
        assert_eq!(
            Views::new(&views[..count]).is_some(),
            count <= 2,
            "views={count}"
        );
    }
    let mode = JacobianMode::Analytic;
    let config = Config::default();
    let mut prior = target.clone();
    prior.translation += Vector3::new(0.004, -0.002, 0.01);

    // Warm: no views runs on the temporal prior alone, exactly as one view without observed keypoints does.
    let mut unobserved = views[0].clone();
    unobserved.weights.fill(0.0);
    let alone = fit(&model, &config, &prior, 1.0, &[], mode).unwrap();
    let blind = fit(&model, &config, &prior, 1.0, &[unobserved], mode).unwrap();
    assert_eq!(
        (
            alone.pose.translation,
            alone.pose.angles,
            alone.energies,
            alone.termination
        ),
        (
            blind.pose.translation,
            blind.pose.angles,
            blind.energies,
            blind.termination
        )
    );
    assert_eq!(alone.pose.rotation, blind.pose.rotation);
    for count in 1..=2 {
        let result = fit(&model, &config, &prior, 1.0, &views[..count], mode).unwrap();
        assert!(
            result.iterations > 0 && result.energies.iter().all(|e| e.is_finite()),
            "views={count}: {result:?}"
        );
    }
    assert_eq!(
        fit(&model, &config, &prior, 1.0, &views, mode).unwrap_err(),
        FitError::TooManyViews { views: 3 }
    );

    // Cold: no views is no evidence; three are refused before any solve, on any thread count.
    let none = initial_pose(&model, &config, 1.0, &[], mode).unwrap();
    assert_eq!(none.termination, Termination::NoEvidence);
    for count in 1..=2 {
        let result = initial_pose(&model, &config, 1.0, &views[..count], mode).unwrap();
        assert!(result.cold.is_some(), "views={count}");
    }
    for threads in [1, 4] {
        assert_eq!(
            initial_pose_parallel(&model, &config, 1.0, &views, mode, threads).unwrap_err(),
            FitError::TooManyViews { views: 3 }
        );
    }
    // A config that asks for no full fit has no winner to return.
    for config in [
        Config {
            full_fit_hypotheses: 0,
            ..Config::default()
        },
        Config {
            finger_starts: 0,
            ..Config::default()
        },
    ] {
        assert_eq!(
            initial_pose(&model, &config, 1.0, &views[..2], mode).unwrap_err(),
            FitError::NoFullFit
        );
    }

    // Scale: fewer than two views is not stereo; any observation with three refuses the whole solve.
    let block = |views: &[View]| CalibrationBlock {
        mirror: 1.0,
        views: views.to_vec(),
        initial: prior.clone(),
    };
    let calibration = CalibrationConfig::default();
    for count in 0..=1 {
        assert_eq!(
            calibrate_scale(&model, &[block(&views[..count])], &calibration).unwrap_err(),
            ScaleError::NoStereo,
            "views={count}"
        );
    }
    let stereo = calibrate_scale(&model, &[block(&views[..2])], &calibration).unwrap();
    assert_eq!((stereo.blocks, stereo.used.as_slice()), (1, &[0][..]));
    assert_eq!(
        calibrate_scale(&model, &[block(&views[..2]), block(&views)], &calibration).unwrap_err(),
        ScaleError::TooManyViews {
            observation: 1,
            views: 3
        }
    );
}
