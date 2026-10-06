use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_3d::camera::{CameraModel, KannalaBrandt4};
use kornia_staging_algebra::Scalar;
use std::hint::black_box;
fn lane<S: Scalar>(c: &mut Criterion, label: &str) {
    let param = [400.0, 410.0, 320.0, 240.0, 0.01, -0.001, 0.0001, 0.0].map(S::from_literal);
    let camera = KannalaBrandt4::new(param).expect("valid camera calibration");
    let point = [0.7, -0.4, 1.2].map(S::from_literal);
    let pixel = camera.project_unchecked(point);
    let mut group = c.benchmark_group(label);
    let homogeneous = nalgebra::Vector4::new(point[0], point[1], point[2], S::one());

    group.bench_function("kb4_homogeneous_point_jacobian", |b| {
        b.iter(|| {
            let p = black_box(homogeneous);
            let mut jacobian = [[S::zero(); 3]; 2];
            let pixel = black_box(&camera).project_with_jacobians(
                [p[0], p[1], p[2]],
                Some(&mut jacobian),
                None,
            );
            let matrix =
                nalgebra::Matrix2x4::from_fn(|r, c| if c < 3 { jacobian[r][c] } else { S::zero() });
            black_box((pixel, matrix))
        })
    });
    group.bench_function("kb4_project", |b| {
        b.iter(|| black_box(&camera).project(black_box(point)))
    });
    group.bench_function("kb4_project_unchecked", |b| {
        b.iter(|| black_box(&camera).project_unchecked(black_box(point)))
    });

    group.bench_function("kb4_unproject", |b| {
        b.iter(|| black_box(&camera).unproject(black_box(pixel)))
    });
    group.finish();
}
fn cameras(c: &mut Criterion) {
    lane::<f64>(c, "f64");
    lane::<f32>(c, "f32");
}
criterion_group!(benches, cameras);
criterion_main!(benches);
