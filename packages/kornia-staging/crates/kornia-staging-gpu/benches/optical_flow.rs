use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kornia_staging_gpu::{
    optical_flow::FusedKltPlan, pyramid::GpuPyramidBuilder, runtime::gpu_client,
};
use kornia_staging_imgproc::optical_flow::{patch_se2::Pattern51, patch_tracker::FlowResult};
use kornia_staging_imgproc::test_fixtures as flow;
use kornia_staging_imgproc::test_fixtures as images;
fn optical_flow(c: &mut Criterion) {
    let client = gpu_client().unwrap();
    let mut builder = GpuPyramidBuilder::new(client.clone());
    let mut prev = builder.allocate(512, 512, flow::LEVELS).unwrap();
    let mut next = builder.allocate(512, 512, flow::LEVELS).unwrap();
    builder
        .build(0, &images::textured_image(512, 512, 0.0, 0.0), &mut prev)
        .unwrap();
    builder
        .build(0, &images::textured_image(512, 512, 2.75, -1.5), &mut next)
        .unwrap();
    let points = flow::grid_positions(512);
    let guesses = flow::guesses_at(&points);
    let mut plan = FusedKltPlan::<Pattern51, _>::new(
        client,
        flow::MAX_KEYPOINTS,
        flow::LEVELS + 1,
        flow::MAX_ITERATIONS,
        flow::MAX_RECOVERED_DIST2,
    )
    .unwrap();
    let mut out = FlowResult::with_capacity(flow::MAX_KEYPOINTS);
    let mut group = c.benchmark_group("gpu_patch_se2");
    group.throughput(Throughput::Elements(points.len() as u64));
    group.bench_function("fused_forward_backward_and_wait", |b| {
        b.iter(|| {
            plan.track(&prev, &next, &points, &guesses, None, &mut out)
                .unwrap();
            std::hint::black_box(&out);
        })
    });
    group.finish();
}
criterion_group!(benches, optical_flow);
criterion_main!(benches);
