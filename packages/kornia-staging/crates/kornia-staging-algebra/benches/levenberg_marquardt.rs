use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_algebra::optim::solvers::{marquardt_scaling, predicted_reduction};
use nalgebra::{SMatrix, SVector};
use std::hint::black_box;
fn bench(c: &mut Criterion) {
    let h = SMatrix::<f64, 26, 26>::identity();
    let g = SVector::<f64, 26>::repeat(1.0);
    c.bench_function("prediction_26", |b| {
        b.iter(|| predicted_reduction(black_box(&g), black_box(&h), black_box(&g)))
    });
    c.bench_function("marquardt_26", |b| {
        b.iter(|| {
            let mut d = black_box(g);
            marquardt_scaling(&mut d, Default::default());
            black_box(d)
        })
    });
}
criterion_group!(benches, bench);
criterion_main!(benches);
