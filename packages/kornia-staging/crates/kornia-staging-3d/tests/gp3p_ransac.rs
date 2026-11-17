//! Seeded, adaptive GP3P RANSAC on synthetic multi-camera rigs.
#![cfg(feature = "ransac")]
#![allow(clippy::unwrap_used)]
mod gp3p_common;

use gp3p_common::{observe, pose_error, random_pose, rig, scene, Rng};
use kornia_staging_3d::pose::gp3p::ransac::{
    gp3p_ransac, Gp3pRansacConfig, Gp3pRansacResult, IterationSchedule, ROOTSLAM_ITERATIONS,
};
use kornia_staging_3d::pose::gp3p::{GeneralizedCorrespondence, Gp3pError, Gp3pPose};
use kornia_staging_algebra::lie::RigidTransform;
use nalgebra::Vector3;

/// RootSLAM's threshold is 5e-5 (0.57°); synthetic noise of 1 mrad sits well inside.
const NOISE: f64 = 1e-3;

fn run(c: &[GeneralizedCorrespondence], config: &Gp3pRansacConfig) -> Gp3pRansacResult {
    gp3p_ransac(c, config).unwrap().unwrap()
}

fn error(result: &Gp3pRansacResult, truth: &RigidTransform<f64>) -> (f64, f64) {
    pose_error(&Gp3pPose::from_rigid_transform(&result.pose), truth)
}

#[test]
fn recovers_pose_with_outliers_on_two_to_four_cameras() {
    for cameras in 2..=4 {
        for (offset, label) in [(0.0, "central"), (0.12, "non-central")] {
            for seed in 0..10 {
                let mut rng = Rng::new(1000 * cameras as u64 + seed);
                let rig = rig(&mut rng, cameras, offset);
                let truth = random_pose(&mut rng);
                let (c, truth_mask) = scene(&mut rng, &rig, &truth, 80, 0.5, 0.0);
                let result = run(&c, &Gp3pRansacConfig::default());
                let (r, t) = error(&result, &truth);
                assert!(
                    r < 1e-9 && t < 1e-8,
                    "{label} {cameras} cams seed {seed}: {r} {t}"
                );
                assert_eq!(
                    result.inliers, truth_mask,
                    "{label} {cameras} cams seed {seed}"
                );
                assert_eq!(result.num_inliers, 40);
            }
        }
    }
}

#[test]
fn noisy_recovery_within_tolerance() {
    let mut worst = (0.0f64, 0.0f64);
    for seed in 0..40 {
        let mut rng = Rng::new(5000 + seed);
        let rig = rig(&mut rng, 4, 0.1);
        let truth = random_pose(&mut rng);
        let (c, truth_mask) = scene(&mut rng, &rig, &truth, 150, 0.6, NOISE);
        let config = Gp3pRansacConfig {
            // 1 - cos(2 mrad) ≈ 2e-6: about 2σ for 1 mrad bearing noise.
            threshold: 1e-5,
            ..Gp3pRansacConfig::default()
        };
        let result = run(&c, &config);
        let (r, t) = error(&result, &truth);
        worst = (worst.0.max(r), worst.1.max(t));
        // 90 inliers at 1 mrad, 1-8 m deep: well under 0.1° and a few cm.
        assert!(
            r < 2e-3 && t < 0.05,
            "seed {seed}: rotation {r} rad, translation {t} m"
        );
        let missed = truth_mask
            .iter()
            .zip(&result.inliers)
            .filter(|(&t, &e)| t && !e)
            .count();
        let false_in = truth_mask
            .iter()
            .zip(&result.inliers)
            .filter(|(&t, &e)| !t && e)
            .count();
        assert!(
            missed <= 9 && false_in <= 2,
            "seed {seed}: missed {missed}, false {false_in}"
        );
    }
    eprintln!("worst rotation {} rad, translation {} m", worst.0, worst.1);
}

#[test]
fn refinement_improves_on_the_ransac_hypothesis() {
    let mut improved = 0;
    for seed in 0..30 {
        let mut rng = Rng::new(7000 + seed);
        let rig = rig(&mut rng, 3, 0.1);
        let truth = random_pose(&mut rng);
        let (c, _) = scene(&mut rng, &rig, &truth, 120, 0.7, NOISE);
        let base = Gp3pRansacConfig {
            threshold: 1e-5,
            seed,
            ..Gp3pRansacConfig::default()
        };
        let raw = run(
            &c,
            &Gp3pRansacConfig {
                refine: false,
                ..base
            },
        );
        let refined = run(&c, &base);
        if error(&refined, &truth).0 < error(&raw, &truth).0 {
            improved += 1;
        }
    }
    assert!(improved >= 27, "refinement improved {improved} of 30");
}

fn fixture(
    seed: u64,
    n: usize,
    ratio: f64,
) -> (Vec<GeneralizedCorrespondence>, RigidTransform<f64>) {
    let mut rng = Rng::new(seed);
    let rig = rig(&mut rng, 4, 0.1);
    let truth = random_pose(&mut rng);
    let (c, _) = scene(&mut rng, &rig, &truth, n, ratio, NOISE);
    (c, truth)
}

#[test]
fn same_seed_same_result_on_any_thread() {
    let (c, _) = fixture(42, 120, 0.35);
    for schedule in [
        IterationSchedule::default(),
        IterationSchedule::rootslam_parity(),
    ] {
        let config = Gp3pRansacConfig {
            schedule,
            threshold: 1e-5,
            seed: 7,
            ..Gp3pRansacConfig::default()
        };
        let first = run(&c, &config);
        assert_eq!(run(&c, &config), first);
        // No global state: four threads at once give the same bits.
        let results: Vec<Gp3pRansacResult> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..4).map(|_| s.spawn(|| run(&c, &config))).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for r in results {
            assert_eq!(r, first);
        }
    }
}

#[test]
fn different_seeds_draw_different_hypotheses() {
    let (c, _) = fixture(43, 120, 0.3);
    // Without refinement the pose is the winning minimal sample's: its bits
    // depend on which inliers the seed drew.
    let poses: Vec<[u64; 3]> = (0..8)
        .map(|seed| {
            let config = Gp3pRansacConfig {
                threshold: 1e-5,
                seed,
                refine: false,
                ..Gp3pRansacConfig::default()
            };
            run(&c, &config).pose.translation.map(f64::to_bits).into()
        })
        .collect();
    let distinct: std::collections::BTreeSet<[u64; 3]> = poses.iter().copied().collect();
    assert!(distinct.len() > 1, "{poses:?}");
}

#[test]
fn adaptive_count_reaches_the_cap_only_at_low_inlier_ratios() {
    let cap = 500;
    let schedule = IterationSchedule::Adaptive {
        confidence: 0.999,
        max_iterations: cap,
    };
    let config = |seed| Gp3pRansacConfig {
        schedule,
        threshold: 1e-5,
        seed,
        ..Gp3pRansacConfig::default()
    };
    for seed in 0..10 {
        // w = 0.9: log(0.001) / log(1 - 0.9^3) ≈ 6 draws.
        let (c, _) = fixture(100 + seed, 100, 0.9);
        let high = run(&c, &config(seed)).iterations;
        assert!(high <= 20, "w 0.9 seed {seed}: {high} iterations");
        // w = 0.5: ≈ 52 draws.
        let (c, _) = fixture(200 + seed, 100, 0.5);
        let mid = run(&c, &config(seed)).iterations;
        assert!(mid < cap / 2, "w 0.5 seed {seed}: {mid} iterations");
        // w = 0.12: ≈ 4000 draws needed, so the cap binds.
        let (c, _) = fixture(300 + seed, 200, 0.12);
        let low = gp3p_ransac(&c, &config(seed)).unwrap().unwrap().iterations;
        assert_eq!(low, cap, "w 0.12 seed {seed}");
    }
}

#[test]
fn parity_schedule_never_exceeds_opengv_draws() {
    for (seed, ratio) in [(0, 0.9), (1, 0.5), (2, 0.2)] {
        let (c, _) = fixture(400 + seed, 100, ratio);
        let config = Gp3pRansacConfig {
            schedule: IterationSchedule::rootslam_parity(),
            threshold: 1e-5,
            ..Gp3pRansacConfig::default()
        };
        if let Some(result) = gp3p_ransac(&c, &config).unwrap() {
            assert!(
                result.iterations <= ROOTSLAM_ITERATIONS + 1,
                "{}",
                result.iterations
            );
        }
    }
}

/// Bug B15: at a 30 % inlier ratio RootSLAM's capped draws miss the pose often;
/// the adaptive schedule does not. Same scenes, same seeds.
#[test]
fn b15_adaptive_finds_low_inlier_loops_that_the_fixed_cap_misses() {
    let trials = 100;
    let mut found = [0usize; 2];
    for seed in 0..trials {
        let (c, truth) = fixture(600 + seed, 100, 0.3);
        for (slot, schedule) in [
            IterationSchedule::default(),
            IterationSchedule::rootslam_parity(),
        ]
        .into_iter()
        .enumerate()
        {
            let config = Gp3pRansacConfig {
                schedule,
                threshold: 1e-5,
                seed,
                ..Gp3pRansacConfig::default()
            };
            let ok = gp3p_ransac(&c, &config)
                .unwrap()
                .is_some_and(|r| error(&r, &truth).0 < 1e-2);
            found[slot] += ok as usize;
        }
    }
    eprintln!(
        "found at w = 0.3: adaptive {} / {trials}, parity {} / {trials}",
        found[0], found[1]
    );
    assert!(found[0] >= 98, "adaptive found {}", found[0]);
    assert!(found[1] <= 80, "parity found {}", found[1]);
}

#[test]
fn fewer_than_four_is_refused() {
    let (c, _) = fixture(1, 10, 1.0);
    for n in 0..4 {
        assert_eq!(
            gp3p_ransac(&c[..n], &Gp3pRansacConfig::default()),
            Err(Gp3pError::TooFewCorrespondences { got: n, needed: 4 })
        );
    }
}

#[test]
fn non_finite_input_is_refused_with_its_index() {
    let (mut c, _) = fixture(2, 30, 1.0);
    c[17].point.x = f64::NAN;
    assert_eq!(
        gp3p_ransac(&c, &Gp3pRansacConfig::default()),
        Err(Gp3pError::NonFinite { index: 17 })
    );
    let (mut c, _) = fixture(2, 30, 1.0);
    c[5].bearing *= 0.5;
    assert_eq!(
        gp3p_ransac(&c, &Gp3pRansacConfig::default()),
        Err(Gp3pError::NonUnitBearing { index: 5 })
    );
}

#[test]
fn all_collinear_world_points_give_no_pose() {
    let mut rng = Rng::new(3);
    let rig = rig(&mut rng, 3, 0.1);
    let c: Vec<GeneralizedCorrespondence> = (0..30)
        .map(|i| {
            let point = Vector3::new(0.2 * i as f64, 0.0, 5.0);
            let cam = i % 3;
            let origin = rig[cam].translation;
            GeneralizedCorrespondence {
                origin,
                bearing: (point - origin).normalize(),
                point,
            }
        })
        .collect();
    assert_eq!(gp3p_ransac(&c, &Gp3pRansacConfig::default()), Ok(None));
}

#[test]
fn default_is_rootslam_threshold_with_adaptive_schedule() {
    let config = Gp3pRansacConfig::default();
    assert_eq!(config.threshold, 5e-5);
    assert!(config.refine);
    assert_eq!(
        config.schedule,
        IterationSchedule::Adaptive {
            confidence: 0.999,
            max_iterations: 500
        }
    );
    assert_eq!(
        IterationSchedule::rootslam_parity(),
        IterationSchedule::RootSlamParity { iterations: 35 }
    );
    // Exact correspondences at the default threshold: every one is an inlier.
    let mut rng = Rng::new(4);
    let rig = rig(&mut rng, 2, 0.1);
    let truth = random_pose(&mut rng);
    let c: Vec<GeneralizedCorrespondence> = (0..20)
        .map(|i| observe(&mut rng, &rig, i % 2, &truth, 0.0))
        .collect();
    assert_eq!(run(&c, &config).num_inliers, 20);
}
