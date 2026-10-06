use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kornia_staging_gpu::{features::GpuCornerScan, runtime::gpu_client};
use kornia_staging_imgproc::features::{CellGrid, CellSelect, CornerScan};
use kornia_staging_imgproc::test_fixtures as images;

fn features(c: &mut Criterion) {
    let image = images::cornered_image(640, 480);
    let mut scanner = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let policy = CellSelect {
        grid: CellGrid::new(640, 480, 50).unwrap(),
        threshold: 5,
        safe_radius: 0.0,
    };
    let mut corners = Vec::new();
    let mut group = c.benchmark_group("gpu_fast");
    group.throughput(Throughput::Elements((640 * 480) as u64));
    group.bench_function("scan_and_wait", |b| {
        b.iter(|| scanner.scan(0, &image).unwrap())
    });
    group.bench_function("select_cells_and_wait", |b| {
        b.iter(|| {
            scanner
                .select_cells(0, &image, &policy, None, &mut corners)
                .unwrap();
            std::hint::black_box(&corners);
        })
    });
    group.finish();
}
criterion_group!(benches, features);
criterion_main!(benches);
