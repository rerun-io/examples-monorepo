use criterion::{criterion_group, criterion_main, Criterion};
use kornia_staging_sensors::imu::{CombinedImuSample, ImuNoise, IntegratedImuMeasurement};
use nalgebra::Vector3;
use std::hint::black_box;
fn bench(c: &mut Criterion) {
    let measurement = IntegratedImuMeasurement::<f64>::new(0, &Vector3::zeros(), &Vector3::zeros());
    let sample = CombinedImuSample {
        timestamp_ns: 5_000_000,
        gyro: [0.1, 0.2, 0.3].into(),
        accel: [0.0, 0.0, 9.81].into(),
    };
    let noise = ImuNoise::new(Vector3::repeat(0.01), Vector3::repeat(0.001)).unwrap();
    c.bench_function("imu_integrate_f64", |b| {
        b.iter_batched(
            || measurement,
            |mut m| {
                m.integrate(black_box(&sample), black_box(&noise)).unwrap();
                black_box(m)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}
criterion_group!(benches, bench);
criterion_main!(benches);
