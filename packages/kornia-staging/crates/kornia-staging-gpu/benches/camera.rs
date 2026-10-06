// The benchmark launches the same inline inverse as a larger stereo kernel.
#![allow(unsafe_code)]
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use cubecl::prelude::*;
use kornia_staging_gpu::{
    camera::{brown8, Brown8, BROWN8_PARAMETERS},
    runtime::gpu_client,
    transfer,
};

#[cube(launch_unchecked)]
fn inverse(params: &[f32], pixels: &[f32], output: &mut [f32], count: usize) {
    let i = ABSOLUTE_POS;
    if i >= count {
        terminate!();
    }
    let mut bearing = Array::<f32>::new(3usize);
    let status = brown8::unproject(
        params,
        0usize,
        pixels[i * 2usize],
        pixels[i * 2usize + 1usize],
        &mut bearing,
    );
    output[i * 4usize] = bearing[0usize];
    output[i * 4usize + 1usize] = bearing[1usize];
    output[i * 4usize + 2usize] = bearing[2usize];
    output[i * 4usize + 3usize] = f32::cast_from(status);
}
fn camera(c: &mut Criterion) {
    let client = gpu_client().unwrap();
    let camera = Brown8::new(
        [
            400., 400., 320., 240., 0.1, 0.01, 0.001, -0.001, 0., 0., 0., 0.,
        ],
        None,
    )
    .unwrap();
    let pixels: Vec<_> = (0..1024)
        .flat_map(|i| [(i % 32) as f32 * 20., (i / 32) as f32 * 15.])
        .collect();
    let params = transfer::upload(&client, f32::as_bytes(&camera.device_parameters())).unwrap();
    let input = transfer::upload(&client, f32::as_bytes(&pixels)).unwrap();
    let output = client.empty(1024 * 4 * size_of::<f32>());
    let mut group = c.benchmark_group("gpu_brown8");
    group.throughput(Throughput::Elements(1024));
    group.bench_function("inline_inverse_1024", |b| {
        b.iter(|| {
            // SAFETY: All bindings hold complete records for the fixed point count.
            unsafe {
                inverse::launch_unchecked::<kornia_staging_gpu::GpuRuntime>(
                    &client,
                    CubeCount::Static(16, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(params.clone(), BROWN8_PARAMETERS),
                    BufferArg::from_raw_parts(input.clone(), 2048),
                    BufferArg::from_raw_parts(output.clone(), 4096),
                    1024,
                );
            }
            std::hint::black_box(
                transfer::read_buffers(&client, vec![output.clone()], "camera benchmark").unwrap(),
            );
        })
    });
    group.finish();
}
criterion_group!(benches, camera);
criterion_main!(benches);
