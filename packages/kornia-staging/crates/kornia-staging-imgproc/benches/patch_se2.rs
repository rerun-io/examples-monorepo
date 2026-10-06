use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::optical_flow::patch_se2::{oracle::OpticalFlowPatch, Pattern51};
use std::hint::black_box;
fn patches(c: &mut Criterion) {
    let pixels = (0..64 * 64)
        .map(|i| (i as u32).wrapping_mul(7919) as u16)
        .collect();
    let image = Image::<u16, 1>::new(
        ImageSize {
            width: 64,
            height: 64,
        },
        pixels,
    )
    .unwrap();
    let positions = [[24.25, 24.5], [32.25, 24.5], [24.25, 32.5], [32.25, 32.5]];
    c.bench_function("patch_se2_scalar", |b| {
        b.iter(|| {
            black_box(
                OpticalFlowPatch::<Pattern51>::new(black_box(&image), black_box(positions[0]))
                    .unwrap(),
            )
        })
    });
}
criterion_group!(benches, patches);
criterion_main!(benches);
