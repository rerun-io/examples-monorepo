use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_3d::camera::{CameraModelKind, Pinhole};
use kornia_staging_algebra::lie::RigidTransform;
use kornia_staging_slam::factors::{compute_rel_pose, linearize_point, LinearizePointOut};
use nalgebra::{Matrix2x3, Matrix2x6, Matrix4, Matrix6, Vector2, Vector3};
use std::hint::black_box;
fn bench(c: &mut Criterion) {
    let camera =
        CameraModelKind::Pinhole(Pinhole::<f64>::new([400.0, 400.0, 320.0, 240.0]).unwrap());
    let transform = Matrix4::from([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.1, 0.0, 0.0, 1.0],
    ]);
    c.bench_function("hosted_reprojection_f64", |b| {
        b.iter(|| {
            let mut r = Vector2::zeros();
            let mut jp = Matrix2x6::zeros();
            let mut jl = Matrix2x3::zeros();
            linearize_point(
                black_box(&Vector2::new(340.0, 250.0)),
                black_box(&Vector2::new(0.1, 0.02)),
                black_box(0.5),
                black_box(&transform),
                black_box(&camera),
                &mut r,
                &mut LinearizePointOut {
                    d_res_d_xi: Some(&mut jp),
                    d_res_d_p: Some(&mut jl),
                    proj: None,
                },
            )
            .unwrap();
            black_box((r, jp, jl))
        })
    });
    c.bench_function("relative_pose_f64", |b| {
        b.iter(|| {
            let p = RigidTransform::default();
            let q = RigidTransform {
                translation: Vector3::new(0.1, 0.0, 0.0),
                ..p
            };
            let mut jh = Matrix6::zeros();
            let mut jt = jh;
            black_box(compute_rel_pose(
                black_box(&q),
                &p,
                &p,
                &q,
                Some(&mut jh),
                Some(&mut jt),
            ));
            black_box((jh, jt))
        })
    });
}
criterion_group!(benches, bench);
criterion_main!(benches);
