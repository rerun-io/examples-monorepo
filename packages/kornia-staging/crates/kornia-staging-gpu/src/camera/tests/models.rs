//! Device cameras against the CPU f32 models: pixels, bearings, Jacobians and typed rejections.
//!
//! Inputs are real calibrations: RoboCap KB4 (cam 0, copied into `ROBOCAP_KB4`),
//! TUM-VI double sphere and EUCM (Basalt `tumvi_512_{ds,eucm}_calib.json` cam 0),
//! an equirect hemisphere tile 1088x960 at 960/pi px/rad with `lon_max` 102 deg, and a
//! 1920x960 panorama. The grids cover each model's whole valid domain and beyond.
#![cfg(feature = "wgpu")]
use crate::{
    camera::{brown8, dispatch, DeviceCamera},
    runtime::gpu_client,
    transfer, GpuRuntime,
};
use cubecl::prelude::*;
use kornia_staging_3d::camera::{
    BrownConrady, CameraModel, CameraModelKind, DoubleSphere, Equirectangular, Eucm,
    KannalaBrandt4, ProjectionReject, UnprojectError,
};

/// Values per output record: pixel or bearing, the 2x3 Jacobian, then the status.
const STRIDE: usize = 9;
const PROJECT: u32 = 0;
const UNPROJECT: u32 = 1;
const JACOBIAN: u32 = 2;
/// Pixel parity bound.
const PIXEL_TOLERANCE: f32 = 1e-3;
/// Jacobian bound, relative to the largest entry of the CPU row.
const JACOBIAN_TOLERANCE: f32 = 1e-4;

#[cube(launch_unchecked)]
fn model_batch(
    params: &[f32],
    input: &[f32],
    out: &mut [f32],
    count: usize,
    #[comptime] model: u32,
    #[comptime] operation: u32,
) {
    let index = ABSOLUTE_POS;
    if index >= count {
        terminate!();
    }
    let mut result = Array::<f32>::new(8usize);
    for slot in 0..8usize {
        result[slot] = 0.0f32;
    }
    let status = if comptime!(operation == UNPROJECT) {
        dispatch::unproject(
            params,
            0usize,
            input[index * 2usize],
            input[index * 2usize + 1usize],
            &mut result,
            model,
        )
    } else if comptime!(operation == JACOBIAN) {
        let base = index * 3usize;
        dispatch::project_with_point_jacobian(
            params,
            0usize,
            input[base],
            input[base + 1usize],
            input[base + 2usize],
            &mut result,
            model,
        )
    } else {
        let base = index * 3usize;
        dispatch::project(
            params,
            0usize,
            input[base],
            input[base + 1usize],
            input[base + 2usize],
            &mut result,
            model,
        )
    };
    for slot in 0..8usize {
        out[index * STRIDE + slot] = result[slot];
    }
    out[index * STRIDE + 8usize] = f32::cast_from(status);
}

fn run(camera: &DeviceCamera, operation: u32, input: &[f32]) -> Vec<[f32; STRIDE]> {
    let stride = if operation == UNPROJECT { 2 } else { 3 };
    let count = input.len() / stride;
    let client = gpu_client().unwrap();
    let parameters = camera.device_parameters();
    let params = transfer::upload(&client, f32::as_bytes(parameters)).unwrap();
    let source = transfer::upload(&client, f32::as_bytes(input)).unwrap();
    let output = client.empty(count * STRIDE * size_of::<f32>());
    // SAFETY: one record per lane; the kernel bounds the rounded-up launch.
    unsafe {
        model_batch::launch_unchecked::<GpuRuntime>(
            &client,
            CubeCount::Static(count.div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            BufferArg::from_raw_parts(params, parameters.len()),
            BufferArg::from_raw_parts(source, input.len()),
            BufferArg::from_raw_parts(output.clone(), count * STRIDE),
            count,
            camera.model_id(),
            operation,
        );
    }
    let bytes = transfer::read_buffers(&client, vec![output], "model output").unwrap();
    f32::from_bytes(&bytes[0]).as_chunks::<STRIDE>().0.to_vec()
}

fn projection_status(status: f32) -> Result<(), ProjectionReject> {
    match status as u32 {
        brown8::VALID => Ok(()),
        brown8::NON_FINITE => Err(ProjectionReject::NonFinite),
        brown8::BELOW_DEPTH => Err(ProjectionReject::BelowMinDepth),
        brown8::OUTSIDE_DOMAIN => Err(ProjectionReject::OutsideDomain),
        other => panic!("unexpected projection status {other}"),
    }
}

fn unprojection_status(status: f32) -> Result<(), UnprojectError> {
    match status as u32 {
        brown8::VALID => Ok(()),
        brown8::NON_FINITE => Err(UnprojectError::NonFinite),
        brown8::OUTSIDE_DOMAIN => Err(UnprojectError::OutsideDomain),
        brown8::SINGULAR => Err(UnprojectError::Singular),
        brown8::NO_CONVERGENCE => Err(UnprojectError::NoConvergence),
        other => panic!("unexpected unprojection status {other}"),
    }
}

/// Points on spheres of two radii: polar angle 0..180 deg from +z, every azimuth, plus edge cases.
fn sphere_points() -> Vec<[f32; 3]> {
    let mut points = Vec::new();
    for polar in 0..=110 {
        // Odd step so no sample sits on a model's domain boundary to within f32 rounding.
        let theta = (polar as f64 * 1.6363).min(180.0).to_radians();
        for azimuth in 0..36 {
            let phi = (azimuth as f64 * 10.0 + 3.7).to_radians();
            for distance in [0.7, 4.0] {
                points.push([
                    (distance * theta.sin() * phi.cos()) as f32,
                    (distance * theta.sin() * phi.sin()) as f32,
                    (distance * theta.cos()) as f32,
                ]);
            }
        }
    }
    points.extend([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [f32::NAN, 0.0, 1.0],
        [0.0, f32::INFINITY, 1.0],
        [0.0, 0.0, f32::NEG_INFINITY],
        [1e30, 0.0, 1.0],
    ]);
    points
}

fn image_pixels(width: f32, height: f32) -> Vec<[f32; 2]> {
    let mut pixels = Vec::new();
    let mut v = -40.0;
    while v <= height + 40.0 {
        let mut u = -40.0;
        while u <= width + 40.0 {
            pixels.push([u, v]);
            u += 7.31;
        }
        v += 6.97;
    }
    pixels.extend([[f32::NAN, 0.0], [0.0, f32::INFINITY], [1e30, 0.0]]);
    pixels
}

/// Device projection and Jacobian agree with the CPU model; returns the worst errors.
fn assert_projection(cpu: &CameraModelKind<f32>, points: &[[f32; 3]]) -> (f32, f32, usize) {
    let device = DeviceCamera::from_model(cpu).unwrap();
    let pixels = run(&device, PROJECT, points.as_flattened());
    let jacobians = run(&device, JACOBIAN, points.as_flattened());
    let (mut pixel_worst, mut jacobian_worst, mut valid) = (0f32, 0f32, 0);
    for ((point, got), jac) in points.iter().zip(&pixels).zip(&jacobians) {
        let mut jp = [[0.0f32; 3]; 2];
        let expected = match cpu {
            CameraModelKind::Kb4(m) => m.project_with_jacobians(*point, Some(&mut jp), None),
            CameraModelKind::DoubleSphere(m) => {
                m.project_with_jacobians(*point, Some(&mut jp), None)
            }
            CameraModelKind::Eucm(m) => m.project_with_jacobians(*point, Some(&mut jp), None),
            CameraModelKind::Equirectangular(m) => {
                m.project_with_jacobians(*point, Some(&mut jp), None)
            }
            CameraModelKind::BrownConrady(m) => {
                m.project_with_jacobians(*point, Some(&mut jp), None)
            }
            other => panic!("no device camera for {other:?}"),
        };
        let status = projection_status(got[8]);
        assert_eq!(
            status,
            projection_status(jac[8]),
            "point={point:?}: Jacobian status"
        );
        match (expected, status) {
            (Ok(pixel), Ok(())) => {
                valid += 1;
                for axis in 0..2 {
                    let error = (pixel[axis] - got[axis]).abs();
                    pixel_worst = pixel_worst.max(error);
                    assert!(
                        error <= PIXEL_TOLERANCE,
                        "point={point:?}: pixel {pixel:?} vs {got:?}"
                    );
                    // The Jacobian variant may contract differently (FMA): same bound, not same bits.
                    assert!(
                        (pixel[axis] - jac[axis]).abs() <= PIXEL_TOLERANCE,
                        "point={point:?}: Jacobian variant pixel"
                    );
                    let scale = jp[axis]
                        .iter()
                        .fold(f32::MIN_POSITIVE, |m, v| m.max(v.abs()));
                    for column in 0..3 {
                        let error = (jp[axis][column] - jac[2 + 3 * axis + column]).abs() / scale;
                        jacobian_worst = jacobian_worst.max(error);
                        assert!(
                            error <= JACOBIAN_TOLERANCE,
                            "point={point:?}: J[{axis}][{column}] {} vs {}",
                            jp[axis][column],
                            jac[2 + 3 * axis + column]
                        );
                    }
                }
            }
            (Err(a), Err(b)) => assert_eq!(a, b, "point={point:?}"),
            pair => panic!("CPU/device validity differs at {point:?}: {pair:?}"),
        }
    }
    (pixel_worst, jacobian_worst, valid)
}

/// Device bearings agree with the CPU inverse to `PIXEL_TOLERANCE` times the focal length.
///
/// Where `1 - b r^2` of the DS/EUCM closed form is below 1e-2 (the rim of the inverse's
/// disc, outside the TUM-VI image) f32 cancellation dominates, and beyond 98 % of KB4's
/// `r_max` the radial slope tends to zero: on these rims the bearing bound is 5e-3 px and
/// the round trip through the CPU projection must still land within 1e-3 px.
fn assert_inverse(cpu: &CameraModelKind<f32>, pixels: &[[f32; 2]]) -> (f32, usize) {
    let device = DeviceCamera::from_model(cpu).unwrap();
    let focal = device.device_parameters()[0].max(device.device_parameters()[1]);
    let bearings = run(&device, UNPROJECT, pixels.as_flattened());
    let (mut worst, mut valid) = (0f32, 0);
    for (pixel, got) in pixels.iter().zip(&bearings) {
        match (cpu.unproject(*pixel), unprojection_status(got[8])) {
            (Ok(a), Ok(())) => {
                valid += 1;
                let norm = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                let b = [got[0], got[1], got[2]];
                assert!(
                    (norm(b) - 1.0).abs() < 2e-6,
                    "pixel={pixel:?}: norm {}",
                    norm(b)
                );
                let cross = [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ];
                let gap = norm(cross) / (norm(a) * norm(b)) * focal;
                let p = device.device_parameters();
                let r2 = ((pixel[0] - p[2]) / p[0]).powi(2) + ((pixel[1] - p[3]) / p[1]).powi(2);
                let rim = match cpu {
                    CameraModelKind::DoubleSphere(_) => 1.0 - (2.0 * p[5] - 1.0) * r2 < 1e-2,
                    CameraModelKind::Eucm(_) => 1.0 - (2.0 * p[4] - 1.0) * p[5] * r2 < 1e-2,
                    CameraModelKind::Kb4(_) => r2.sqrt() > 0.98 * p[9],
                    _ => false,
                };
                if rim {
                    let back = cpu.project(b).unwrap();
                    let error = (back[0] - pixel[0]).abs().max((back[1] - pixel[1]).abs());
                    assert!(
                        error <= PIXEL_TOLERANCE,
                        "pixel={pixel:?}: rim round trip {error} px"
                    );
                }
                if !rim {
                    worst = worst.max(gap);
                }
                assert!(
                    gap <= if rim { 5e-3 } else { PIXEL_TOLERANCE },
                    "pixel={pixel:?}: bearing gap {gap} px"
                );
            }
            (Err(a), Err(b)) => assert_eq!(a, b, "pixel={pixel:?}"),
            pair => panic!("CPU/device inverse validity differs at {pixel:?}: {pair:?}"),
        }
    }
    (worst, valid)
}

/// RoboCap cam 0 KB4 `fx fy cx cy k1 k2 k3 k4`, copied from slam-rs `configs/robocap_calib_downscale3.json`.
#[allow(clippy::excessive_precision)]
const ROBOCAP_KB4: [f64; 8] = [
    212.14536598347505,
    211.57082242862452,
    318.4011350411036,
    174.8127089872141,
    0.06166616664989166,
    -0.02109092590818281,
    0.037163389786906705,
    -0.013518619346447233,
];

fn robocap_kb4() -> KannalaBrandt4<f32> {
    KannalaBrandt4::new(ROBOCAP_KB4.map(|v| v as f32)).unwrap()
}

#[allow(clippy::excessive_precision)]
const TUMVI_DS: [f32; 6] = [
    158.28600034966976,
    158.2743455478755,
    254.96116578191652,
    256.8894394501779,
    -0.17213086034353242,
    0.5931177593944744,
];
#[allow(clippy::excessive_precision)]
const TUMVI_EUCM: [f32; 6] = [
    191.14799836282188,
    191.13150963902817,
    254.9585771534443,
    256.88154645599445,
    0.6291060881178562,
    1.0418067381860867,
];

#[test]
fn kb4_matches_cpu_beyond_ninety_degrees() {
    // RoboCap's branch ends at 91.25 deg; TartanAir's equidistant fisheye (1000x1000 over
    // 195 deg, dataforge `tartanair.py`) is valid to 180 deg.
    let tartanair = 1000.0 / 195f32.to_radians();
    for (camera, width, height) in [
        (robocap_kb4(), 640.0, 360.0),
        (
            KannalaBrandt4::new([tartanair, tartanair, 499.5, 499.5, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            1000.0,
            1000.0,
        ),
    ] {
        let cpu = CameraModelKind::Kb4(camera);
        let mut points = sphere_points();
        for degrees in [90.5f32, 91.0, 91.2, 91.3, 92.0] {
            let (sin, cos) = degrees.to_radians().sin_cos();
            points.extend([
                [sin, 0.0, cos],
                [0.0, -2.0 * sin, 2.0 * cos],
                [0.6 * sin, 0.8 * sin, cos],
            ]);
        }
        let (pixel, jacobian, valid) = assert_projection(&cpu, &points);
        let rear = points
            .iter()
            .filter(|p| p[2] < 0.0 && camera.project(**p).is_ok())
            .count();
        let outside = points
            .iter()
            .filter(|p| camera.project(**p) == Err(ProjectionReject::OutsideDomain))
            .count();
        assert!(rear >= 6, "rear rays inside the domain: {rear}");
        let (bearing, inverse_valid) = assert_inverse(&cpu, &image_pixels(width, height));
        // Beyond the branch-end radius the inverse refuses, as on the CPU.
        let [fx, _, cx, cy, ..] = camera.params();
        assert_inverse(&cpu, &[[camera.max_radius() * fx + cx + 5.0, cy], [cx, cy]]);
        println!(
            "KB4 theta_max={:.2} deg: pixel {pixel:e} px, Jacobian {jacobian:e} rel ({valid} points, {rear} rear, {outside} beyond), bearing {bearing:e} px ({inverse_valid} pixels)",
            camera.max_angle().to_degrees()
        );
    }
}

#[test]
fn double_sphere_and_eucm_match_cpu_over_their_domains() {
    for cpu in [
        CameraModelKind::DoubleSphere(DoubleSphere::new(TUMVI_DS).unwrap()),
        CameraModelKind::Eucm(Eucm::new(TUMVI_EUCM).unwrap()),
    ] {
        let points = sphere_points();
        let (pixel, jacobian, valid) = assert_projection(&cpu, &points);
        let rejected = points
            .iter()
            .filter(|p| cpu.project(**p) == Err(ProjectionReject::OutsideDomain))
            .count();
        assert!(
            rejected > 100,
            "the grid must cross the domain bound: {rejected}"
        );
        let (bearing, inverse_valid) = assert_inverse(&cpu, &image_pixels(512.0, 512.0));
        // Pixels far outside the image disc reach the closed-form inverse's own bound.
        let (_, far_valid) = assert_inverse(
            &cpu,
            &image_pixels(512.0, 512.0)
                .iter()
                .map(|p| [p[0] * 3.0 - 512.0, p[1] * 3.0 - 512.0])
                .collect::<Vec<_>>(),
        );
        println!("{cpu:?}: pixel {pixel:e} px, Jacobian {jacobian:e} rel ({valid} points), bearing {bearing:e} px ({inverse_valid}+{far_valid} pixels)");
    }
}

#[test]
fn equirect_tile_and_panorama_match_cpu_to_the_window_and_poles() {
    for (cpu, width, height) in [
        (
            Equirectangular::hemisphere_tile(
                1088,
                960,
                960.0 / std::f32::consts::PI,
                102f32.to_radians(),
            )
            .unwrap(),
            1088.0,
            960.0,
        ),
        (Equirectangular::panorama(1920, 960).unwrap(), 1920.0, 960.0),
    ] {
        let cpu = CameraModelKind::Equirectangular(cpu);
        let mut points = sphere_points();
        // Near the poles: inside and outside the f32 pole epsilon (3.2e-3 of the norm).
        for rho in [1e-2f32, 5e-3, 2e-3, 1e-3] {
            points.extend([[rho, 1.0, 0.0], [0.0, -1.0, rho], [-rho, 1.0, rho]]);
        }
        let (pixel, jacobian, valid) = assert_projection(&cpu, &points);
        let (bearing, inverse_valid) = assert_inverse(&cpu, &image_pixels(width, height));
        println!("{cpu:?}: pixel {pixel:e} px, Jacobian {jacobian:e} rel ({valid} points), bearing {bearing:e} px ({inverse_valid} pixels)");
    }
}

#[test]
fn brown8_point_jacobian_matches_cpu() {
    let mut params = [0.0f32; 18];
    params[..12].copy_from_slice(&[
        380.0, 381.0, 320.0, 240.0, -0.28, 0.07, 1e-3, -2e-4, -0.01, 0.0, 0.0, 0.0,
    ]);
    let cpu = CameraModelKind::BrownConrady(BrownConrady::new(params, Some(1.2)).unwrap());
    let (pixel, jacobian, valid) = assert_projection(&cpu, &sphere_points());
    assert!(valid > 500);
    println!("Brown8: pixel {pixel:e} px, Jacobian {jacobian:e} rel ({valid} points)");
}

#[test]
fn device_camera_packs_cpu_constants_and_refuses_general_models() {
    let kb4 = robocap_kb4();
    let device = DeviceCamera::from_model(&CameraModelKind::Kb4(kb4)).unwrap();
    assert_eq!(device.model_id(), crate::camera::MODEL_KB4);
    assert_eq!(device.device_parameters()[8], kb4.max_angle());
    assert_eq!(device.device_parameters()[9], kb4.max_radius());
    let ds = DoubleSphere::new(TUMVI_DS).unwrap();
    let device = DeviceCamera::from_model(&CameraModelKind::DoubleSphere(ds)).unwrap();
    assert_eq!(device.device_parameters()[6], ds.domain_bound());
    let pinhole = kornia_staging_3d::camera::Pinhole::new([400.0, 400.0, 320.0, 240.0]).unwrap();
    assert!(DeviceCamera::from_model(&CameraModelKind::Pinhole(pinhole)).is_none());
}
