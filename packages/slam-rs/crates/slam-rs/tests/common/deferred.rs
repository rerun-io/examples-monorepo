//! The moving-ceiling fixture of the deferred-keyframe integration test.

use super::api::calib::{BasaltCamera, Calibration, PinholeParams};
use super::api::config::VioConfig;
use super::api::frontend::flow::FrontendOptions;
use super::api::lie::{Se3, So3};
use super::api::{ImageView, Vio, VioResult, VioStatus};
use nalgebra::Vector3;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;
const FOCAL: f64 = 300.0;
const CEILING_M: f64 = 3.0;
/// Texture lattice pitch on the ceiling, metres (6 px at 3 m).
const CELL_M: f64 = 0.06;
pub const FRAME_NS: i64 = 33_333_333;
const IMU_NS: i64 = 5_000_000;
pub const FRAMES: i64 = 180;
/// Sideways excursion and period: `x(t) = A (1 - cos(2 pi t / T))`, from rest.
const AMPLITUDE_M: [f64; 2] = [0.6, 0.25];
const PERIOD_S: [f64; 2] = [4.0, 3.0];

pub fn calibration() -> Calibration<f64> {
    let mut calibration: Calibration<f64> = Calibration::from_json_str(include_str!(
        "../../../../configs/robocap_calib_downscale3.json"
    ))
    .unwrap();
    for camera in 0..4 {
        calibration.t_i_c[camera] =
            Se3::new(So3::identity(), [camera as f64 * 0.08, 0.0, 0.0].into());
        calibration.intrinsics[camera] = BasaltCamera::Pinhole(PinholeParams {
            fx: FOCAL,
            fy: FOCAL,
            cx: WIDTH as f64 / 2.0,
            cy: HEIGHT as f64 / 2.0,
        });
    }
    calibration
}

pub fn config(deferred: bool) -> VioConfig {
    let mut config: VioConfig =
        VioConfig::from_json_str(include_str!("../../../../configs/msdmo_config.json")).unwrap();
    // The fast profile, which the deferral builds on.
    config.port_redetect_survivor_ratio = 0.85;
    config.port_frame_update_max_iterations = 5;
    config.optical_flow_image_safe_radius = 0.0;
    config.port_keyframe_solve_deferred = deferred;
    config
}

/// The rig's true position (world = initial rig frame: gravity along -z, no rotation).
pub fn position(t_s: f64) -> Vector3<f64> {
    let axis = |k: usize| {
        let w = 2.0 * std::f64::consts::PI / PERIOD_S[k];
        AMPLITUDE_M[k] * (1.0 - (w * t_s).cos())
    };
    Vector3::new(axis(0), axis(1), 0.0)
}

/// The rig's true acceleration, world frame.
pub fn acceleration(t_s: f64) -> Vector3<f64> {
    let axis = |k: usize| {
        let w = 2.0 * std::f64::consts::PI / PERIOD_S[k];
        AMPLITUDE_M[k] * w * w * (w * t_s).cos()
    };
    Vector3::new(axis(0), axis(1), 0.0)
}

/// Bilinear value noise on a `CELL_M` lattice: smooth enough to track, cornered enough to detect.
fn texture(x: f64, y: f64) -> u8 {
    let (gx, gy) = (x / CELL_M, y / CELL_M);
    let (ix, iy) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - ix, gy - iy);
    let value = |i: f64, j: f64| {
        let hash = ((i as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (j as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_mul(0x1656_67B1_9E37_79F9);
        30.0 + (hash >> 56) as f64 * (195.0 / 255.0)
    };
    let top = value(ix, iy) * (1.0 - fx) + value(ix + 1.0, iy) * fx;
    let bottom = value(ix, iy + 1.0) * (1.0 - fx) + value(ix + 1.0, iy + 1.0) * fx;
    (top * (1.0 - fy) + bottom * fy).round() as u8
}

pub fn render(t_s: f64, camera: usize) -> Vec<u8> {
    let centre = position(t_s) + Vector3::new(camera as f64 * 0.08, 0.0, 0.0);
    let depth = CEILING_M - centre.z;
    let mut pixels = vec![0u8; WIDTH * HEIGHT];
    for (v, row) in pixels.as_chunks_mut::<WIDTH>().0.iter_mut().enumerate() {
        for (u, pixel) in row.iter_mut().enumerate() {
            let x = centre.x + depth * (u as f64 - WIDTH as f64 / 2.0) / FOCAL;
            let y = centre.y + depth * (v as f64 - HEIGHT as f64 / 2.0) / FOCAL;
            *pixel = texture(x, y);
        }
    }
    pixels
}

pub struct Run {
    pub results: Vec<VioResult>,
    pub deferred_keyframes: usize,
    pub reported_solves: usize,
    pub solves: Vec<super::api::estimator::DeferredKeyframeStats<f32>>,
    pub errors_m: Vec<f64>,
}

pub fn pipeline(deferred: bool) -> Vio<f32> {
    let options = FrontendOptions {
        threads: 2,
        ..FrontendOptions::default()
    };
    Vio::new(config(deferred), calibration(), options).unwrap()
}

pub fn run(frames: &[[Vec<u8>; 4]], vio: &mut Vio<f32>) -> Run {
    let t0: i64 = 1_000_000_000;
    let mut imu_t: i64 = t0 - 100_000_000;
    let mut out = Run {
        results: Vec::new(),
        deferred_keyframes: 0,
        reported_solves: 0,
        solves: Vec::new(),
        errors_m: Vec::new(),
    };
    let mut pending = false;
    for (index, images) in frames.iter().enumerate() {
        let t_ns: i64 = t0 + index as i64 * FRAME_NS;
        while imu_t <= t_ns + IMU_NS {
            let t_s = ((imu_t - t0) as f64 / 1e9).max(0.0);
            let accel = acceleration(t_s) + Vector3::new(0.0, 0.0, 9.81);
            vio.push_imu(imu_t, [0.0; 3], [accel.x, accel.y, accel.z])
                .unwrap();
            imu_t += IMU_NS;
        }
        let views: Vec<ImageView<'_>> = images
            .iter()
            .map(|data| ImageView {
                width: WIDTH,
                height: HEIGHT,
                stride: WIDTH,
                data,
            })
            .collect();
        let result: VioResult = vio.track(t_ns, &views).unwrap();
        assert_eq!(result.status, VioStatus::Tracking);
        // A deferred solve is reported by the frameset after its keyframe, and only then.
        assert_eq!(
            vio.last_deferred_keyframe().is_some(),
            pending,
            "frameset {index}"
        );
        if let Some(solve) = vio.last_deferred_keyframe() {
            out.reported_solves += 1;
            let mut comparable = solve.clone();
            comparable.timings = Default::default();
            out.solves.push(comparable);
            assert!(
                solve.num_points_added > 0,
                "frameset {index}: the deferred keyframe triangulated nothing"
            );
        }
        let stats = vio.last_stats().unwrap();
        pending = stats.keyframe_deferred;
        if stats.keyframe_deferred {
            assert!(stats.took_keyframe);
            out.deferred_keyframes += 1;
        }
        if stats.opt_started {
            let truth = position((t_ns - t0) as f64 / 1e9);
            let estimate = Vector3::new(
                result.pose.unwrap().world_from_rig[0],
                result.pose.unwrap().world_from_rig[1],
                result.pose.unwrap().world_from_rig[2],
            );
            out.errors_m.push((estimate - truth).norm());
        }
        out.results.push(result);
    }
    out
}
