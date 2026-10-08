use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use kornia_staging_algebra::optim::solvers::marginalize;
use nalgebra::{DMatrix, DVector};

fn marginalization(c: &mut Criterion) {
    let mut seed = 0x713a_85cdu32;
    let input: Vec<f64> = (0..48 * 24)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed as f64 / u32::MAX as f64 - 0.5) * 0.02;
            noise + if i % 48 == i / 48 { 4.0 } else { 0.0 }
        })
        .collect();
    let input = DMatrix::from_vec(48, 24, input);
    let residual = DVector::repeat(48, 0.3);
    let keep = (6..24).collect();
    let marg = (0..6).collect();
    c.bench_function("marginalize_48x24", |b| {
        b.iter_batched(
            || (input.clone(), residual.clone()),
            |(matrix, residual)| marginalize(matrix, residual, &keep, &marg).unwrap(),
            BatchSize::SmallInput,
        );
    });
}
criterion_group!(benches, marginalization);
criterion_main!(benches);
