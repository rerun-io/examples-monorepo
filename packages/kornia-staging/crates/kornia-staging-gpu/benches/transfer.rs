use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kornia_staging_gpu::{
    runtime::{gpu_client, GpuError},
    transfer::{execute_exclusive, read_buffers, upload},
};
fn transfer(c: &mut Criterion) {
    let client = gpu_client().unwrap();
    let bytes = vec![17u8; 4096];
    let mut group = c.benchmark_group("gpu_transfer");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("exclusive_upload_and_read", |b| {
        b.iter(|| {
            let output = execute_exclusive(&client, "transfer benchmark", || {
                let handle = upload(&client, &bytes)?;
                read_buffers(&client, vec![handle], "benchmark")
            })
            .unwrap_or_else(|error: GpuError| panic!("{error}"));
            std::hint::black_box(output);
        })
    });
    group.finish();
}
criterion_group!(benches, transfer);
criterion_main!(benches);
