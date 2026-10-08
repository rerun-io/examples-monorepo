use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::resize::resize_area_u8;
use std::hint::black_box;

fn area(c: &mut Criterion) {
    let src = Image::<u8, 1>::from_size_val(
        ImageSize {
            width: 1920,
            height: 1080,
        },
        127,
    )
    .unwrap();
    let mut dst = Image::<u8, 1>::from_size_val(
        ImageSize {
            width: 640,
            height: 360,
        },
        0,
    )
    .unwrap();
    c.bench_function("area3_mono_1080p", |b| {
        b.iter(|| resize_area_u8(black_box(&src), black_box(&mut dst)).unwrap())
    });
}
criterion_group!(benches, area);
criterion_main!(benches);
