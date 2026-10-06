use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::pyramid::PyramidPlanU16;
use std::hint::black_box;
fn bench(c: &mut Criterion) {
    let source = Image::new(
        ImageSize {
            width: 960,
            height: 720,
        },
        (0u32..960 * 720)
            .map(|v| v.wrapping_mul(7919) as u16)
            .collect(),
    )
    .unwrap();
    let mut plan = PyramidPlanU16::new(source.size(), 1).unwrap();
    c.bench_function("pyramid_plan_u16_960x720_one_reduction", |b| {
        b.iter(|| black_box(&mut plan).run(black_box(&source)).unwrap())
    });
}
criterion_group!(benches, bench);
criterion_main!(benches);
