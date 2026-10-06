use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kornia_image::{Image, ImageSize};
use kornia_staging_gpu::{pyramid::GpuPyramidBuilder, runtime::gpu_client};

fn pyramid(c: &mut Criterion) {
    let size = ImageSize {
        width: 640,
        height: 480,
    };
    let image = Image::new(
        size,
        (0..size.width * size.height)
            .map(|i| (i * 1237) as u16)
            .collect(),
    )
    .unwrap();
    let mut builder = GpuPyramidBuilder::new(gpu_client().unwrap());
    let mut output = builder.allocate(size.width, size.height, 3).unwrap();
    let mut last = Image::from_size_val(
        ImageSize {
            width: 80,
            height: 60,
        },
        0u16,
    )
    .unwrap();
    builder.build(0, &image, &mut output).unwrap();
    output.read_level_into(3, &mut last).unwrap();
    let mut group = c.benchmark_group("gpu_pyramid_u16");
    group.throughput(Throughput::Elements((size.width * size.height) as u64));
    group.bench_function("upload_build_and_wait_640x480", |b| {
        b.iter(|| {
            builder.build(0, &image, &mut output).unwrap();
            output.read_level_into(3, &mut last).unwrap();
            std::hint::black_box(&last);
        })
    });
    group.finish();
}
criterion_group!(benches, pyramid);
criterion_main!(benches);
