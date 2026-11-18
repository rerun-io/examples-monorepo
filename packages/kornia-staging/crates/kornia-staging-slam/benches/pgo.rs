//! PGO on a RootSLAM-sized map: 1500 keyframes (the map cap), a chain, covisibility edges to the next few
//! keyframes and a loop from the newest keyframe back to the start; the window (8) and the newest are fixed.
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use kornia_staging_slam::pgo::{optimize_pose_graph, EdgeSigmas, PgoConfig, PoseGraphEdge};
use nalgebra::{Vector3, Vector6};
use std::hint::black_box;

/// A deterministic small perturbation, distinct per `seed`.
fn wiggle(seed: usize, scale: f64) -> RigidTransform<f64> {
    let s = seed as f64;
    let v = |k: f64| scale * (s * 12.9898 + k * 78.233).sin();
    RigidTransform::exp_decoupled(&Vector6::new(
        v(1.0),
        v(2.0),
        v(3.0),
        v(4.0),
        v(5.0),
        v(6.0),
    ))
}

fn loop_graph(n: usize) -> (Vec<RigidTransform<f64>>, Vec<bool>, Vec<PoseGraphEdge>) {
    let truth: Vec<RigidTransform<f64>> = (0..n)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let radius = 6.0;
            RigidTransform::new(
                Rotation3::exp(&Vector3::new(0.1 * angle.sin(), 0.05, angle + 0.3)),
                Vector3::new(
                    radius * angle.cos(),
                    radius * angle.sin(),
                    0.3 * (3.0 * angle).sin(),
                ),
            )
        })
        .collect();
    let sigmas = EdgeSigmas::default();
    let mut pairs: Vec<(usize, usize)> = (1..n).map(|i| (i - 1, i)).collect();
    for i in (0..n).step_by(5) {
        pairs.extend(
            [2, 3, 4]
                .into_iter()
                .filter(|k| i + k < n)
                .map(|k| (i, i + k)),
        );
    }
    pairs.extend((0..5).map(|k| (k, n - 1)));
    let edges = pairs
        .iter()
        .enumerate()
        .map(|(e, &(a, b))| {
            PoseGraphEdge::new(
                a,
                b,
                truth[a].inverse() * truth[b] * wiggle(e, 0.001),
                &sigmas,
            )
        })
        .collect();
    // Integrated drifting odometry, as a VIO hands it over.
    let mut initial = vec![truth[0]];
    for i in 1..n {
        let odometry = truth[i - 1].inverse() * truth[i] * wiggle(i + 7919, 0.003);
        initial.push(initial[i - 1] * odometry);
    }
    let fixed = (0..n).map(|i| i + 8 >= n).collect();
    (initial, fixed, edges)
}

fn bench(c: &mut Criterion) {
    let (initial, fixed, edges) = loop_graph(1500);
    let config = PgoConfig::default();
    c.bench_function("pgo_1500_keyframe_loop", |b| {
        b.iter_batched(
            || initial.clone(),
            |mut poses| {
                let report =
                    optimize_pose_graph(&mut poses, black_box(&fixed), black_box(&edges), &config)
                        .unwrap();
                black_box((poses, report))
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
