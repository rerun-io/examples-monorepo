use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_algebra::optim::solvers::solve_scaled_damped;
use nalgebra::{DMatrix, DVector};

fn damped(c: &mut Criterion) {
    let h: Vec<f64> = (0..60 * 60)
        .map(|i| {
            let row = i % 60;
            let col = i / 60;
            0.01 / (row + col + 1) as f64 + if row == col { 4.0 } else { 0.0 }
        })
        .collect();
    let h = DMatrix::from_vec(60, 60, h);
    let b = DVector::repeat(60, 0.3);
    let mut step = DVector::zeros(60);
    c.bench_function("scaled_damped_solve_60", |bench| {
        bench.iter(|| {
            let report = solve_scaled_damped(&h, &b, 1e-4, 1e-12, &mut step);
            let _ = std::hint::black_box((&step, report));
        });
    });
}
criterion_group!(benches, damped);
criterion_main!(benches);
