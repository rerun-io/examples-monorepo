use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::pyramid::{pyrdown_u16_unchecked, PyrDownU16Scratch};
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
    let mut destination = Image::from_size_val(
        ImageSize {
            width: 480,
            height: 360,
        },
        0u16,
    )
    .unwrap();
    let mut scratch = PyrDownU16Scratch::new(960).unwrap();
    c.bench_function("pyrdown_u16_960x720", |b| {
        b.iter(|| {
            pyrdown_u16_unchecked(
                black_box(&source),
                black_box(&mut destination),
                black_box(&mut scratch),
            )
        })
    });
}
criterion_group!(benches, bench);
criterion_main!(benches);
