use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use tsdf_rs::TsdfVolume;

const FRAME_COUNT: usize = 200;
const HEIGHT: usize = 192;
const WIDTH: usize = 256;
const SAMPLE_COUNT: usize = 5;

#[derive(Clone, Copy)]
struct BenchResult {
    integrate_mean_ms: f64,
    integrate_tail_mean_ms: f64,
    extract_mesh_ms: f64,
    vertex_count: usize,
    triangle_count: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let depth = spherical_depth();
    let intrinsics = [
        230.0,
        0.0,
        WIDTH as f32 / 2.0 - 0.5,
        0.0,
        230.0,
        HEIGHT as f32 / 2.0 - 0.5,
        0.0,
        0.0,
        1.0,
    ];
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for sample_index in 0..SAMPLE_COUNT {
        let result = run_scene(&depth, intrinsics)?;
        println!(
            "sample {}: integrate mean {:.3} ms, tail {:.3} ms; extract_mesh {:.3} ms; {} vertices, {} triangles",
            sample_index + 1,
            result.integrate_mean_ms,
            result.integrate_tail_mean_ms,
            result.extract_mesh_ms,
            result.vertex_count,
            result.triangle_count,
        );
        samples.push(result);
    }

    let median_integrate = median(
        samples
            .iter()
            .map(|sample| sample.integrate_mean_ms)
            .collect(),
    );
    let median_integrate_tail = median(
        samples
            .iter()
            .map(|sample| sample.integrate_tail_mean_ms)
            .collect(),
    );
    let median_extract = median(
        samples
            .iter()
            .map(|sample| sample.extract_mesh_ms)
            .collect(),
    );
    let scene = samples[0];
    println!(
        "median: integrate mean {:.3} ms, tail {:.3} ms; extract_mesh {:.3} ms; {} vertices, {} triangles",
        median_integrate,
        median_integrate_tail,
        median_extract,
        scene.vertex_count,
        scene.triangle_count,
    );
    Ok(())
}

fn run_scene(depth: &[f32], intrinsics: [f32; 9]) -> Result<BenchResult, Box<dyn Error>> {
    let mut volume = TsdfVolume::new(0.04, 0.12, false)?;
    let mut integrate_times = Vec::with_capacity(FRAME_COUNT);
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());

    for frame_index in 0..FRAME_COUNT {
        let z = 1.0 - 2.0 * (frame_index as f32 + 0.5) / FRAME_COUNT as f32;
        let pitch = z.asin();
        let yaw = frame_index as f32 * golden_angle;
        let transform = camera_from_world(yaw, pitch);
        let started = Instant::now();
        volume.integrate(
            black_box(depth),
            [HEIGHT, WIDTH],
            intrinsics,
            transform,
            None,
        )?;
        integrate_times.push(started.elapsed());
    }

    let extract_started = Instant::now();
    let mesh = volume.extract_mesh()?;
    let extract_time = extract_started.elapsed();
    let result = BenchResult {
        integrate_mean_ms: mean_milliseconds(&integrate_times),
        integrate_tail_mean_ms: mean_milliseconds(&integrate_times[FRAME_COUNT - 50..]),
        extract_mesh_ms: extract_time.as_secs_f64() * 1_000.0,
        vertex_count: mesh.vertices.len(),
        triangle_count: mesh.triangles.len(),
    };
    black_box(mesh);
    Ok(result)
}

fn spherical_depth() -> Vec<f32> {
    let principal_x = WIDTH as f32 / 2.0 - 0.5;
    let principal_y = HEIGHT as f32 / 2.0 - 0.5;
    (0..HEIGHT * WIDTH)
        .map(|index| {
            let x = ((index % WIDTH) as f32 - principal_x) / 230.0;
            let y = ((index / WIDTH) as f32 - principal_y) / 230.0;
            1.7 / (x * x + y * y + 1.0).sqrt()
        })
        .collect()
}

fn camera_from_world(yaw: f32, pitch: f32) -> [f32; 16] {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    [
        cos_yaw,
        0.0,
        -sin_yaw,
        0.0,
        sin_pitch * sin_yaw,
        cos_pitch,
        sin_pitch * cos_yaw,
        0.0,
        cos_pitch * sin_yaw,
        -sin_pitch,
        cos_pitch * cos_yaw,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

fn mean_milliseconds(durations: &[Duration]) -> f64 {
    durations.iter().map(Duration::as_secs_f64).sum::<f64>() * 1_000.0 / durations.len() as f64
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
