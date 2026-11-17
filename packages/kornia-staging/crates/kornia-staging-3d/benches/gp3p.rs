//! GP3P minimal solve and RANSAC at RootSLAM's island sizes.
//!
//! Island matches fed to PnP in the RootSLAM runs on the NAS (23k accepted
//! verifications): median 40, p95 77, max 263; inlier ratio median 0.5, p10 0.33.
//! Rejected candidates (w 0.1 and 0, false loops) run to the cap unless the
//! caller's acceptance ratio bounds the draws (`min_inlier_ratio`: 0.5 MSD,
//! 0.7 EuRoC). Every case uses the shipped threshold (5e-5).
use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_3d::pose::gp3p::ransac::{gp3p_ransac, Gp3pRansacConfig, IterationSchedule};
use kornia_staging_3d::pose::gp3p::{gp3p, GeneralizedCorrespondence};
use std::hint::black_box;

#[path = "../tests/gp3p_common/mod.rs"]
mod gp3p_common;
use gp3p_common::{observe, random_pose, rig, scene, Rng};

fn bench(c: &mut Criterion) {
    let mut rng = Rng::new(1);
    let rig = rig(&mut rng, 4, 0.1);
    let truth = random_pose(&mut rng);
    let triple: Vec<GeneralizedCorrespondence> = (0..3)
        .map(|i| observe(&mut rng, &rig, i, &truth, 0.0))
        .collect();
    c.bench_function("gp3p_minimal", |b| b.iter(|| gp3p(black_box(&triple))));

    let mut group = c.benchmark_group("gp3p_ransac");
    let accepted = [(40, 0.5), (77, 0.5), (263, 0.5), (40, 0.33), (77, 0.33)];
    let rejected = [(77, 0.1), (263, 0.1), (77, 0.0), (263, 0.0)];
    for &(n, ratio) in accepted.iter().chain(&rejected) {
        let (correspondences, _) = scene(&mut rng, &rig, &truth, n, ratio, 1e-3);
        for (label, schedule, min_inlier_ratio) in [
            ("adaptive", IterationSchedule::default(), None),
            ("adaptive_wmin0.5", IterationSchedule::default(), Some(0.5)),
            ("adaptive_wmin0.7", IterationSchedule::default(), Some(0.7)),
            ("parity35", IterationSchedule::rootslam_parity(), None),
        ] {
            let config = Gp3pRansacConfig {
                schedule,
                min_inlier_ratio,
                ..Gp3pRansacConfig::default()
            };
            group.bench_function(format!("{label}_n{n}_w{ratio}"), |b| {
                b.iter(|| gp3p_ransac(black_box(&correspondences), &config))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
