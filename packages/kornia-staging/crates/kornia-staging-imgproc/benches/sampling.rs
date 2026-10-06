use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{ops::cast_and_scale, Image, ImageSize};
use kornia_staging_imgproc::{
    color::widen_u8_shift8_strided,
    interpolation::{sample_bilinear_u16, sample_bilinear_with_gradient_u16},
};
use std::hint::black_box;

fn sampling(c: &mut Criterion) {
    let size = ImageSize {
        width: 960,
        height: 960,
    };
    let source = Image::new(size, (0..960 * 960).map(|i| (i % 251) as u8).collect()).unwrap();
    let mut image = Image::from_size_val(size, 0u16).unwrap();
    c.bench_function("cast_and_scale_960", |b| {
        b.iter(|| cast_and_scale(black_box(&source), black_box(&mut image), 256u16).unwrap())
    });
    let padded: Vec<u8> = (0..968 * 960).map(|i| (i % 251) as u8).collect();
    c.bench_function("widen_shift8_padded_960", |b| {
        b.iter(|| widen_u8_shift8_strided(black_box(&padded), 968, black_box(&mut image)).unwrap())
    });
    c.bench_function("bilinear_u16", |b| {
        b.iter(|| sample_bilinear_u16(black_box(&image), black_box(231.4), black_box(123.34345)))
    });
    c.bench_function("bilinear_u16_gradient", |b| {
        b.iter(|| {
            sample_bilinear_with_gradient_u16(
                black_box(&image),
                black_box(231.4),
                black_box(123.34345),
            )
        })
    });
}
criterion_group!(benches, sampling);
criterion_main!(benches);
