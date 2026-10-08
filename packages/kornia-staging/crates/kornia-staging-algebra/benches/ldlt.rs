use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_algebra::linalg::ldlt::ldlt_in_place;
use std::hint::black_box;
fn ldlt(c: &mut Criterion) {
    c.bench_function("ldlt_3x3_f32", |b| {
        b.iter(|| {
            let mut columns = black_box([[4.0f32, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]);
            black_box(ldlt_in_place(&mut columns));
        })
    });
}
criterion_group!(benches, ldlt);
criterion_main!(benches);
