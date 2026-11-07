//! A sphere of radius 1.7 seen from 200 directions around its center (golden-angle spiral), 192x256 depth.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tsdf::{Camera, DepthFrame, TsdfVolume};

const FRAME_COUNT: usize = 200;
const HEIGHT: usize = 192;
const WIDTH: usize = 256;
const FOCAL: f32 = 230.0;

fn spherical_depth() -> Vec<f32> {
    let principal_x = WIDTH as f32 / 2.0 - 0.5;
    let principal_y = HEIGHT as f32 / 2.0 - 0.5;
    (0..HEIGHT * WIDTH)
        .map(|index| {
            let x = ((index % WIDTH) as f32 - principal_x) / FOCAL;
            let y = ((index / WIDTH) as f32 - principal_y) / FOCAL;
            1.7 / (x * x + y * y + 1.0).sqrt()
        })
        .collect()
}

fn cameras() -> Vec<Camera> {
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    let principal = [WIDTH as f32 / 2.0 - 0.5, HEIGHT as f32 / 2.0 - 0.5];
    (0..FRAME_COUNT)
        .map(|frame| {
            let z = 1.0 - 2.0 * (frame as f32 + 0.5) / FRAME_COUNT as f32;
            let (sin_yaw, cos_yaw) = (frame as f32 * golden_angle).sin_cos();
            let (sin_pitch, cos_pitch) = z.asin().sin_cos();
            #[rustfmt::skip]
            let camera_from_world = [
                cos_yaw, 0.0, -sin_yaw, 0.0,
                sin_pitch * sin_yaw, cos_pitch, sin_pitch * cos_yaw, 0.0,
                cos_pitch * sin_yaw, -sin_pitch, cos_pitch * cos_yaw, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ];
            Camera::pinhole(
                [FOCAL, FOCAL, principal[0], principal[1]],
                camera_from_world,
            )
        })
        .collect()
}

fn fused(depth: &[f32], cameras: &[Camera]) -> TsdfVolume {
    let mut volume = TsdfVolume::new(0.04, 0.12, false).expect("valid volume parameters");
    for camera in cameras {
        volume
            .integrate(&DepthFrame::new(depth, [HEIGHT, WIDTH]), camera)
            .expect("valid frame");
    }
    volume
}

fn tsdf(c: &mut Criterion) {
    let depth = spherical_depth();
    let cameras = cameras();
    let mut group = c.benchmark_group("sphere_200_frames_4cm");
    group.sample_size(10);

    // 200 integrations into a fresh volume; divide by 200 for the per-frame cost.
    group.bench_function("integrate_all_frames", |b| {
        b.iter(|| fused(black_box(&depth), black_box(&cameras)))
    });
    let volume = fused(&depth, &cameras);
    group.bench_function("extract_mesh", |b| {
        b.iter(|| volume.extract_mesh().expect("mesh"))
    });
    group.bench_function("dirty_block_meshes_all_blocks", |b| {
        b.iter(|| volume.dirty_block_meshes().expect("meshes"))
    });
    group.finish();
}

criterion_group!(benches, tsdf);
criterion_main!(benches);
