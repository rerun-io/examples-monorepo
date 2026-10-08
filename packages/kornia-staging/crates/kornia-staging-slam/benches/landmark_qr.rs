use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use kornia_staging_slam::sqrt_ba::LandmarkQr;
use std::hint::black_box;
fn bench(c: &mut Criterion) {
    let qr = LandmarkQr::<f64>::new(8, 30, (0..12).collect()).unwrap();
    let mut storage = vec![0.0; qr.rows() * qr.columns()];
    for column in (0..12).chain(qr.landmark_column()..qr.columns()) {
        for row in 0..16 {
            storage[column * qr.rows() + row] = ((row * 7 + column * 13 + 1) as f64).sin();
        }
    }
    for k in 0..3 {
        storage[(qr.landmark_column() + k) * qr.rows() + k] += 5.0;
    }
    c.bench_function("landmark_householder_f64", |b| {
        b.iter_batched(
            || (qr.clone(), storage.clone()),
            |(mut qr, mut storage)| {
                qr.eliminate_householder_unchecked(black_box(&mut storage));
                black_box((qr, storage))
            },
            BatchSize::SmallInput,
        )
    });
    let mut eliminated = qr.clone();
    eliminated.eliminate_householder_unchecked(&mut storage);
    c.bench_function("landmark_back_substitute_f64", |b| {
        b.iter(|| {
            black_box(eliminated.back_substitute_unchecked(
                black_box(&storage),
                black_box(&[0.001; 30]),
                true,
            ))
        })
    });
}
criterion_group!(benches, bench);
criterion_main!(benches);
