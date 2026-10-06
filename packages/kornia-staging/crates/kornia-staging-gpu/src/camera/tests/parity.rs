//! Brown8 contracts share the CPU/OpenCV fixture and typed rejection rules.
#![cfg(feature = "wgpu")]
use crate::{camera::Brown8, runtime::gpu_client};
use kornia_staging_3d::camera::{BrownConrady, CameraModel};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    intrinsics: [f32; 4],
    distortion: [f32; 8],
    points: Vec<[f32; 3]>,
    pixels: Vec<[f32; 2]>,
}
fn cameras(params: [f32; 12], radius: Option<f32>) -> (BrownConrady<f32>, Brown8) {
    let mut full = [0.0; 18];
    full[..12].copy_from_slice(&params);
    (
        BrownConrady::new(full, radius).unwrap(),
        Brown8::new(params, radius).unwrap(),
    )
}
fn assert_inverse(cpu: &BrownConrady<f32>, gpu: &Brown8, pixels: &[[f32; 2]]) {
    let mut actual = Vec::new();
    gpu.unproject(&gpu_client().unwrap(), pixels, &mut actual)
        .unwrap();
    let p = cpu.params();
    assert_eq!(actual.len(), pixels.len());
    for (pixel, value) in pixels.iter().zip(actual) {
        match (cpu.unproject(*pixel), value) {
            (Ok(a), Ok(b)) => {
                let gap = ((a[0] / a[2] - b[0] / b[2]) * p[0])
                    .abs()
                    .max(((a[1] / a[2] - b[1] / b[2]) * p[1]).abs());
                assert!(gap <= 1e-3, "pixel={pixel:?}, bearing gap={gap} px");
                let norm = b.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 2e-6);
                let projected = cpu.project(b).unwrap();
                let error = (projected[0] - pixel[0])
                    .abs()
                    .max((projected[1] - pixel[1]).abs());
                assert!(error <= 1e-3, "pixel={pixel:?}, roundtrip={error} px");
            }
            (Err(a), Err(b)) => assert_eq!(a, b, "pixel={pixel:?}"),
            pair => panic!("CPU/GPU validity differs at {pixel:?}: {pair:?}"),
        }
    }
}
#[test]
fn brown8_matches_shared_opencv_fixture_and_cpu_inverse() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../../fixtures/cameras/brown8_opencv.json"
    ))
    .unwrap();
    let mut params = [0.0; 12];
    params[..4].copy_from_slice(&fixture.intrinsics);
    params[4..].copy_from_slice(&fixture.distortion);
    let (cpu, gpu) = cameras(params, None);
    let mut actual = Vec::new();
    gpu.project(&gpu_client().unwrap(), &fixture.points, &mut actual)
        .unwrap();
    assert_eq!(actual.len(), fixture.points.len());
    for ((point, expected), got) in fixture.points.iter().zip(&fixture.pixels).zip(actual) {
        let got = got.unwrap();
        let cpu = cpu.project(*point).unwrap();
        for axis in 0..2 {
            assert!((got[axis] - expected[axis]).abs() <= 1e-3);
            assert!((got[axis] - cpu[axis]).abs() <= 1e-3);
        }
    }
    assert_inverse(&cpu, &gpu, &fixture.pixels);
}

#[test]
fn damped_inverse_converges_beyond_the_old_five_step_budget() {
    let (cpu, gpu) = cameras(
        [
            100., 110., 0., 0., 0.4, 0.04, 0.01, -0.02, 0.002, 0., 0., 0.,
        ],
        None,
    );
    let pixels: Vec<_> = [[2.5, 1.5, 1.], [-2.2, 0.9, 1.], [0., 0., 1.]]
        .map(|point| cpu.project(point).unwrap())
        .into();
    assert_inverse(&cpu, &gpu, &pixels);
}

#[test]
fn brown8_preserves_nonfinite_depth_radius_and_inverse_rejections() {
    let (cpu, gpu) = cameras([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], Some(1.));
    let points = [
        [0., 0., -1.],
        [0., 0., 0.001],
        [0., 0., 1.],
        [1., 0., 1.],
        [1.1, 0., 1.],
        [f32::NAN, 0., 1.],
        [0., f32::INFINITY, 1.],
    ];
    let mut output = Vec::new();
    gpu.project(&gpu_client().unwrap(), &points, &mut output)
        .unwrap();
    for (point, got) in points.iter().zip(output) {
        assert_eq!(cpu.project(*point), got);
    }
    assert_inverse(
        &cpu,
        &gpu,
        &[
            [0., 0.],
            [1., 0.],
            [1.1, 0.],
            [f32::NAN, 0.],
            [0., f32::INFINITY],
        ],
    );
    // Radial derivative singularity, nonconvergent pixel, and denominator pole.
    for (params, pixels) in [
        (
            [1., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0.],
            vec![[1., 0.], [0.8, 0.]],
        ),
        (
            [1., 1., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0.],
            vec![[1., 0.]],
        ),
    ] {
        let (cpu, gpu) = cameras(params, None);
        assert_inverse(&cpu, &gpu, &pixels);
    }
}

#[test]
fn finite_extreme_brown8_inputs_preserve_cpu_overflow_rejection() {
    let (cpu, gpu) = cameras([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], None);
    let mut actual = Vec::new();
    let points = [[1e10, 0., 1.], [0., -1e10, 1.]];
    gpu.project(&gpu_client().unwrap(), &points, &mut actual)
        .unwrap();
    for (point, value) in points.into_iter().zip(actual) {
        assert_eq!(cpu.project(point), value);
    }
    assert_inverse(&cpu, &gpu, &[[1e10, 0.], [0., -1e10]]);
}
