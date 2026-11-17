//! GP3P minimal solver and refinement on synthetic rigs.
#![allow(clippy::unwrap_used)]
mod gp3p_common;

use gp3p_common::{observe, pose_error, random_pose, rig, Rng};
use kornia_staging_3d::pose::gp3p::{gp3p, refine, GeneralizedCorrespondence, Gp3pError, Gp3pPose};
use kornia_staging_algebra::lie::RigidTransform;
use nalgebra::Vector3;

/// Every trial: 2-4 cameras, three correspondences on distinct cameras when
/// possible, exact bearings; one solution must be the truth.
fn exact_recovery(offset: f64, trials: u64) {
    exact_recovery_from(offset, 0, trials)
}

fn exact_recovery_from(offset: f64, first: u64, trials: u64) {
    let mut worst_rotation = 0.0f64;
    let mut worst_translation = 0.0f64;
    for seed in first..trials {
        let mut rng = Rng::new(seed);
        let cameras = 2 + (seed % 3) as usize;
        let rig = rig(&mut rng, cameras, offset);
        let truth = random_pose(&mut rng);
        let c: Vec<GeneralizedCorrespondence> = (0..3)
            .map(|i| observe(&mut rng, &rig, i % cameras, &truth, 0.0))
            .collect();
        let poses = gp3p(&c).unwrap();
        assert!(poses.len() <= 8);
        let (r, t) = poses
            .iter()
            .map(|p| pose_error(p, &truth))
            .min_by(|a, b| (a.0 + a.1).total_cmp(&(b.0 + b.1)))
            .unwrap_or_else(|| panic!("seed {seed}: no solution"));
        assert!(
            r < 1e-7 && t < 1e-6,
            "seed {seed}: rotation {r}, translation {t}"
        );
        worst_rotation = worst_rotation.max(r);
        worst_translation = worst_translation.max(t);
        // Every returned solution explains all three rays exactly.
        for p in poses.iter() {
            for ci in c.iter() {
                assert!(ci.angular_residual(p).abs() < 1e-9, "seed {seed}");
            }
        }
    }
    eprintln!("offset {offset}: worst rotation {worst_rotation:e} rad, translation {worst_translation:e} m");
}

#[test]
fn exact_recovery_non_central_rigs() {
    exact_recovery(0.15, 2000);
}

#[test]
fn exact_recovery_central_rigs() {
    exact_recovery(0.0, 2000);
}

#[test]
fn exact_recovery_wide_baseline_rig() {
    // Stereo-like baselines of a metre, comparable to the point depths.
    exact_recovery(1.0, 500);
}

#[test]
fn noisy_minimal_solution_stays_close() {
    let mut max_rotation = 0.0f64;
    let mut lost = Vec::new();
    for seed in 0..300 {
        let mut rng = Rng::new(10_000 + seed);
        let rig = rig(&mut rng, 4, 0.1);
        let truth = random_pose(&mut rng);
        let c: Vec<GeneralizedCorrespondence> = (0..3)
            .map(|i| observe(&mut rng, &rig, i, &truth, 1e-4))
            .collect();
        // Noise can push a true root through a double root into a complex
        // pair; such a triple has no solution near the truth. Count them.
        let nearest = gp3p(&c)
            .unwrap()
            .iter()
            .map(|p| pose_error(p, &truth).0)
            .min_by(f64::total_cmp);
        match nearest {
            Some(r) if r < 0.1 => max_rotation = max_rotation.max(r),
            _ => lost.push(seed),
        }
    }
    eprintln!("worst rotation {max_rotation} rad, no solution near the truth: {lost:?}");
    // Minimal solutions amplify noise, but the solution near the truth stays
    // within a few degrees at 0.1 mrad of bearing noise, and it is seldom lost.
    assert!(
        lost.len() <= 3,
        "{} of 300 lost the truth: {lost:?}",
        lost.len()
    );
}

#[test]
fn refine_recovers_truth_from_a_perturbed_start() {
    for seed in 0..50 {
        let mut rng = Rng::new(20_000 + seed);
        let rig = rig(&mut rng, 3, 0.12);
        let truth = random_pose(&mut rng);
        let c: Vec<GeneralizedCorrespondence> = (0..40)
            .map(|i| observe(&mut rng, &rig, i % 3, &truth, 0.0))
            .collect();
        let start = Gp3pPose::from_rigid_transform(
            &(truth
                * RigidTransform::exp_decoupled(&nalgebra::Vector6::new(
                    0.05, -0.03, 0.04, 0.02, -0.01, 0.03,
                ))),
        );
        let refined = refine(&c, &start);
        let (r, t) = pose_error(&refined, &truth);
        assert!(
            r < 1e-9 && t < 1e-8,
            "seed {seed}: rotation {r}, translation {t}"
        );
    }
}

#[test]
fn refine_reduces_noisy_error_and_never_raises_cost() {
    let mut better = 0;
    for seed in 0..50 {
        let mut rng = Rng::new(30_000 + seed);
        let rig = rig(&mut rng, 2, 0.1);
        let truth = random_pose(&mut rng);
        let c: Vec<GeneralizedCorrespondence> = (0..100)
            .map(|i| observe(&mut rng, &rig, i % 2, &truth, 2e-3))
            .collect();
        let Ok(poses) = gp3p(&c[..3]) else { continue };
        let Some(start) = poses
            .iter()
            .min_by(|a, b| pose_error(a, &truth).0.total_cmp(&pose_error(b, &truth).0))
        else {
            continue;
        };
        // The refinement objective: squared sine of the bearing error.
        let cost = |p: &Gp3pPose| -> f64 {
            c.iter()
                .map(|ci| {
                    let cos = 1.0 - ci.angular_residual(p);
                    1.0 - cos * cos
                })
                .sum()
        };
        let refined = refine(&c, start);
        assert!(cost(&refined) <= cost(start) * (1.0 + 1e-12), "seed {seed}");
        if pose_error(&refined, &truth).0 < pose_error(start, &truth).0 {
            better += 1;
        }
        // 100 bearings at 2 mrad pin the rotation well below a degree.
        assert!(pose_error(&refined, &truth).0 < 2e-3, "seed {seed}");
    }
    assert!(better >= 45, "refinement improved only {better} of 50");
}

#[test]
fn refine_with_too_few_correspondences_is_identity() {
    let start = Gp3pPose {
        rotation: nalgebra::Matrix3::identity(),
        translation: Vector3::new(1.0, 2.0, 3.0),
    };
    assert_eq!(refine(&[], &start), start);
}

fn triple() -> Vec<GeneralizedCorrespondence> {
    let mut rng = Rng::new(7);
    let rig = rig(&mut rng, 3, 0.1);
    let truth = random_pose(&mut rng);
    (0..3)
        .map(|i| observe(&mut rng, &rig, i, &truth, 0.0))
        .collect()
}

#[test]
fn fewer_than_three_is_refused() {
    let c = triple();
    for n in 0..3 {
        assert_eq!(
            gp3p(&c[..n]),
            Err(Gp3pError::TooFewCorrespondences { got: n, needed: 3 })
        );
    }
    let mut four = c.clone();
    four.push(c[0]);
    assert_eq!(
        gp3p(&four),
        Err(Gp3pError::TooManyCorrespondences {
            got: 4,
            expected: 3
        })
    );
}

#[test]
fn non_finite_and_non_unit_are_refused() {
    let base = triple();
    for index in 0..3 {
        for field in 0..3 {
            let mut c = base.clone();
            match field {
                0 => c[index].origin.x = f64::NAN,
                1 => c[index].bearing.y = f64::INFINITY,
                _ => c[index].point.z = f64::NEG_INFINITY,
            }
            assert_eq!(gp3p(&c), Err(Gp3pError::NonFinite { index }));
        }
        let mut c = base.clone();
        c[index].bearing *= 2.0;
        assert_eq!(gp3p(&c), Err(Gp3pError::NonUnitBearing { index }));
    }
    // A zero bearing normalizes to NaN and is refused as non-finite.
    let zero = GeneralizedCorrespondence::from_camera(
        &RigidTransform::identity(),
        &Vector3::zeros(),
        &Vector3::new(1.0, 2.0, 3.0),
    );
    let mut c = base.clone();
    c[1] = zero;
    assert_eq!(gp3p(&c), Err(Gp3pError::NonFinite { index: 1 }));
}

#[test]
fn collinear_and_coincident_points_are_refused() {
    let mut c = triple();
    let (a, b) = (c[0].point, c[1].point);
    c[2].point = a + (b - a) * 2.5;
    assert_eq!(gp3p(&c), Err(Gp3pError::CollinearPoints));
    c[2].point = a + (b - a) * 0.5 + Vector3::new(1e-13, 0.0, 0.0);
    assert_eq!(gp3p(&c), Err(Gp3pError::CollinearPoints));
    c[1].point = a;
    c[2].point = a;
    assert_eq!(gp3p(&c), Err(Gp3pError::CollinearPoints));
}

#[test]
fn to_and_from_rigid_transform_round_trip() {
    let mut rng = Rng::new(99);
    for _ in 0..20 {
        let t = random_pose(&mut rng);
        let back = Gp3pPose::from_rigid_transform(&t).to_rigid_transform();
        assert!((back.rotation.matrix() - t.rotation.matrix()).amax() < 1e-14);
        assert_eq!(back.translation, t.translation);
    }
}

/// A correspondence behind its camera (`f · u <= 0`) is not a fit: the
/// tangent-plane residual `B u` vanishes at the antipode `u = -f`, so the
/// refinement must not pull the pose toward putting points there.
#[test]
fn refine_does_not_fit_points_behind_the_camera() {
    for seed in 0..20 {
        let mut rng = Rng::new(60_000 + seed);
        let rig = rig(&mut rng, 2, 0.1);
        let truth = random_pose(&mut rng);
        let near = truth
            * RigidTransform::exp_decoupled(&nalgebra::Vector6::new(
                0.02, -0.01, 0.015, 0.01, -0.008, 0.012,
            ));
        // 10 exact inliers of `truth`, and 30 bearings that are antipodal under `near`.
        let mut c: Vec<GeneralizedCorrespondence> = (0..10)
            .map(|i| observe(&mut rng, &rig, i % 2, &truth, 0.0))
            .collect();
        c.extend((0..30).map(|i| {
            let mut o = observe(&mut rng, &rig, i % 2, &near, 0.0);
            o.bearing = -o.bearing;
            o
        }));
        let start = Gp3pPose::from_rigid_transform(&near);
        for o in &c[10..] {
            assert!(o.angular_residual(&start) > 1.999, "seed {seed}");
        }
        let refined = refine(&c, &start);
        let (r, t) = pose_error(&refined, &truth);
        // The 30 rows behind cost 30 in total, so the relative tolerance stops
        // a little earlier than on inliers alone.
        assert!(
            r < 1e-7 && t < 1e-6,
            "seed {seed}: rotation {r}, translation {t}"
        );
    }
}
