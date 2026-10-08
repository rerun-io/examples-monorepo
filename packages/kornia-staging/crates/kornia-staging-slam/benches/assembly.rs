use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_slam::sqrt_ba::{DenseBlock, DenseHbWorkspace, LandmarkQr, PriorLinearization};
use nalgebra::{DMatrix, DVector};
use std::hint::black_box;
fn bench(c: &mut Criterion) {
    let qr = LandmarkQr::<f64>::new(8, 30, (0..12).collect()).unwrap();
    let mut data = vec![0.0; qr.rows() * qr.columns()];
    for col in (0..12).chain(std::iter::once(qr.residual_column())) {
        for r in 3..qr.rows() {
            data[r + col * qr.rows()] = ((r * 13 + col * 7) as f64).sin();
        }
    }
    let block = DenseBlock::new(&qr, &data);
    let mut workspace = DenseHbWorkspace::default();
    let mut h = DMatrix::zeros(30, 30);
    let mut b = DVector::zeros(30);
    c.bench_function("dense_reduction_100_landmarks_f64", |bench| {
        bench.iter(|| {
            workspace
                .reduce_into(
                    std::iter::repeat_n(black_box(block), 100),
                    black_box(&mut h),
                    black_box(&mut b),
                )
                .unwrap();
        })
    });
    let jac = DMatrix::repeat(30, 30, 0.1);
    let res = DVector::repeat(30, 0.2);
    let delta = DVector::repeat(30, 0.01);
    let prior = PriorLinearization::new(&jac, &res, &delta).unwrap();
    let mut h = DMatrix::zeros(30, 30);
    let mut b = DVector::zeros(30);
    c.bench_function("prior_dense_30_f64", |bench| {
        bench.iter(|| black_box(prior).add_dense(black_box(&mut h), black_box(&mut b)))
    });
}
criterion_group!(benches, bench);
criterion_main!(benches);
