use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use kornia_staging_algebra::linalg::qr::{apply_householder_unchecked, make_householder_unchecked};

fn qr(c: &mut Criterion) {
    let input: Vec<f64> = (0..48 * 24).map(|i| (i as f64 + 0.3).sin()).collect();
    c.bench_function("householder_reflection_48x24", |b| {
        b.iter_batched_ref(
            || (input.clone(), vec![0.0; 48]),
            |(matrix, scratch)| {
                let (active, _) = make_householder_unchecked(&matrix[..48], scratch);
                apply_householder_unchecked(matrix, 48, 24, 48, scratch, active);
                std::hint::black_box(matrix);
            },
            BatchSize::SmallInput,
        );
    });
}
criterion_group!(benches, qr);
criterion_main!(benches);
