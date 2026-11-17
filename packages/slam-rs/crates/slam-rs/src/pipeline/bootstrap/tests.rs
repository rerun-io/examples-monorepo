//! The driver on a synthetic walk seen by the MSD Index rig (KB4), with an
//! exact IMU; and the estimator started from its answer.
#![allow(clippy::unwrap_used)]

use super::*;
use crate::config::VioConfig;
use crate::estimator::{FrameOutcome, SqrtKeypointVio};
use crate::types::KeypointId;
use nalgebra::{Matrix3, Vector2};

const MSDMI_CONFIG: &str = include_str!("../../../../../configs/msdmi_config.json");
const MSDMI_CALIB: &str = include_str!("../../../tests/fixtures/msdmi_calib.json");
const GRAVITY: f64 = 9.81;
const FRAME_NS: i64 = 33_333_333;

fn calibration() -> Calibration<f64> {
    Calibration::from_json_str(MSDMI_CALIB).unwrap()
}

/// A handheld walk that starts after `static_s` seconds at rest, through a
/// quintic onset of one second.
struct Walk {
    static_s: f64,
}

fn rot(axis: Vector3<f64>, angle: f64) -> Matrix3<f64> {
    So3::exp(&(axis * angle)).matrix()
}

impl Walk {
    fn onset(&self, t: f64) -> f64 {
        let x: f64 = (t - self.static_s).clamp(0.0, 1.0);
        x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
    }
    fn position(&self, t: f64) -> Vector3<f64> {
        let e: f64 = self.onset(t);
        let tau: f64 = t - self.static_s;
        let tau_pi: f64 = std::f64::consts::TAU * tau;
        Vector3::new(
            e * (1.2 * tau + 0.02 * (2.0 * tau_pi).sin()),
            e * (0.03 * tau_pi.sin() + 0.075 * tau * tau.max(0.0)),
            1.5 + e * 0.02 * (2.0 * tau_pi + 1.0).sin(),
        )
    }
    /// World from IMU: the IMU z axis (the MSD cameras' axis) forward along x.
    fn rotation(&self, t: f64) -> Matrix3<f64> {
        let e: f64 = self.onset(t);
        let tau_pi: f64 = std::f64::consts::TAU * (t - self.static_s);
        let nominal = Matrix3::from_columns(&[
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(1.0, 0.0, 0.0),
        ]);
        rot(Vector3::z(), e * 0.25 * (0.6 * tau_pi + 0.3).sin())
            * rot(Vector3::y(), e * 0.08 * (2.0 * tau_pi + 0.7).sin())
            * rot(Vector3::x(), e * 0.10 * (tau_pi + 1.9).sin())
            * nominal
    }
    fn imu(&self, t_ns: i64) -> CombinedImuSample {
        let t: f64 = t_ns as f64 * 1e-9;
        let h: f64 = 1e-3;
        let accel_w: Vector3<f64> =
            (self.position(t + h) - self.position(t) * 2.0 + self.position(t - h)) / (h * h);
        let r: Matrix3<f64> = self.rotation(t);
        let specific: Vector3<f64> = r.transpose() * (accel_w + Vector3::new(0.0, 0.0, GRAVITY));
        let hr: f64 = 1e-5;
        let delta = nalgebra::Rotation3::from_matrix_unchecked(
            self.rotation(t - hr).transpose() * self.rotation(t + hr),
        );
        let gyro: Vector3<f64> = delta.scaled_axis() / (2.0 * hr);
        CombinedImuSample {
            timestamp_ns: t_ns,
            gyro: [gyro.x, gyro.y, gyro.z].into(),
            accel: [specific.x, specific.y, specific.z].into(),
        }
    }
}

/// Points on the walls, floor and ceiling of a room around the walk.
fn room() -> Vec<Vector3<f64>> {
    let mut state: u64 = 0x5EED;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..900)
        .map(|i| {
            let (a, b) = (uniform(), uniform());
            match i % 4 {
                0 => Vector3::new(9.0, -4.0 + 8.0 * a, 3.0 * b),
                1 => Vector3::new(-2.0 + 11.0 * a, 4.0, 3.0 * b),
                2 => Vector3::new(-2.0 + 11.0 * a, -4.0, 3.0 * b),
                _ => Vector3::new(-2.0 + 11.0 * a, -4.0 + 8.0 * b, 0.0),
            }
        })
        .collect()
}

/// Camera 0's pixels of every visible point; camera 1 sees nothing (the
/// mono proxy).
fn frameset(
    walk: &Walk,
    calibration: &Calibration<f64>,
    points: &[Vector3<f64>],
    t_ns: i64,
) -> Arc<FlowObservations> {
    let t: f64 = t_ns as f64 * 1e-9;
    let camera = SlamCamera::from_model(&calibration.intrinsics[0]).unwrap();
    let t_i_c: Se3<f64> = calibration.t_i_c[0];
    let r: Matrix3<f64> = walk.rotation(t);
    let p: Vector3<f64> = walk.position(t);
    let mut observations = FlowObservations::new(t_ns, calibration.t_i_c.len());
    for (id, point) in points.iter().enumerate() {
        let x: Vector3<f64> =
            t_i_c.rotation.matrix().transpose() * (r.transpose() * (point - p) - t_i_c.translation);
        if x.z < 0.3 || x.norm() > 12.0 {
            continue;
        }
        let mut pixel: Vector2<f64> = Vector2::zeros();
        if camera.project_point(&x.push(1.0), &mut pixel, None)
            && (20.0..940.0).contains(&pixel.x)
            && (20.0..940.0).contains(&pixel.y)
        {
            observations.cameras[0].insert(KeypointId(id as u64), pixel.cast::<f32>());
        }
    }
    Arc::new(observations)
}

struct Run {
    walk: Walk,
    calibration: Calibration<f64>,
    points: Vec<Vector3<f64>>,
    driver: InertialBootstrap,
    /// Next IMU sample index (1 kHz), pushed as the feed would: just past each frameset.
    next_imu: i64,
}

impl Run {
    fn new(static_s: f64, options: BootstrapOptions) -> Self {
        let walk = Walk { static_s };
        let calibration: Calibration<f64> = calibration();
        let driver = InertialBootstrap::new(calibration.clone(), GRAVITY, options).unwrap();
        Self {
            walk,
            calibration,
            points: room(),
            driver,
            next_imu: -50,
        }
    }
    /// Push the IMU up to 2 ms past frameset `k`, as the feed does.
    fn imu_through(&mut self, k: i64) {
        while self.next_imu * 1_000_000 <= k * FRAME_NS + 2_000_000 {
            self.driver
                .push_imu(self.walk.imu(self.next_imu * 1_000_000));
            self.next_imu += 1;
        }
    }
    fn frame(&self, k: i64) -> Arc<FlowObservations> {
        frameset(&self.walk, &self.calibration, &self.points, k * FRAME_NS)
    }
    /// Feed framesets until one is accepted or `max_frames` pass.
    fn until_accepted(&mut self, max_frames: i64) -> Result<(i64, Box<Accepted>), BootstrapError> {
        for k in 0..max_frames {
            self.imu_through(k);
            self.driver.predict(k * FRAME_NS);
            if let BootstrapStep::Accepted(accepted) = self.driver.push_frame(self.frame(k))? {
                return Ok((k, accepted));
            }
            assert!(
                self.driver.buffered_frames() <= 92,
                "the buffer outgrew the 3 s window"
            );
        }
        panic!(
            "no window accepted in {max_frames} framesets: {:#?}",
            self.driver.report()
        );
    }
}

fn truth_gravity_body0(walk: &Walk) -> Vector3<f64> {
    walk.rotation(0.0).transpose() * Vector3::new(0.0, 0.0, -GRAVITY)
}

#[test]
fn accepts_a_walking_window_and_hands_over_the_whole_buffer() {
    let mut run = Run::new(-1.0, BootstrapOptions::default());
    let (k, accepted) = run.until_accepted(120).unwrap();
    // 60 framesets of 33.333333 ms fall 20 ns short of 2 s.
    assert_eq!(k, 61, "a 2 s window at 30 Hz completes at frameset 61");
    assert_eq!(accepted.frames.len(), 62);
    assert_eq!(accepted.initial.t_ns, 0);
    assert!(!accepted.imu.is_empty() && accepted.imu[0].timestamp_ns <= 0);
    let report = run.driver.report();
    assert_eq!(report.attempts.len(), 1);
    assert_eq!(report.start_t_ns, Some(0));

    let r_w_b0: Matrix3<f64> = accepted.initial.t_w_i.rotation.matrix();
    let gravity: Vector3<f64> = r_w_b0.transpose() * Vector3::new(0.0, 0.0, -GRAVITY);
    let angle: f64 = gravity.angle(&truth_gravity_body0(&run.walk));
    assert!(
        angle < 0.1f64.to_radians(),
        "gravity off by {} deg",
        angle.to_degrees()
    );
    let h: f64 = 1e-4;
    let v_world_truth: Vector3<f64> = (run.walk.position(h) - run.walk.position(-h)) / (2.0 * h);
    let v_body_truth: Vector3<f64> = run.walk.rotation(0.0).transpose() * v_world_truth;
    let v_body: Vector3<f64> = r_w_b0.transpose() * accepted.initial.vel_w_i;
    assert!(
        (v_body - v_body_truth).norm() < 0.01,
        "{v_body:?} vs {v_body_truth:?}"
    );
    assert!(accepted.initial.velocity_sigma >= 0.05);
}

#[test]
fn gyro_prediction_follows_the_rotation() {
    let mut run = Run::new(-1.0, BootstrapOptions::default());
    run.imu_through(0);
    run.driver.predict(0);
    for k in 1..20 {
        run.imu_through(k);
        let prediction = run.driver.predict(k * FRAME_NS);
        let predicted: Matrix3<f64> = (prediction.t_w_i_previous.rotation.inverse()
            * prediction.t_w_i_current.rotation)
            .matrix()
            .cast();
        let t0: f64 = (k - 1) as f64 * FRAME_NS as f64 * 1e-9;
        let t1: f64 = k as f64 * FRAME_NS as f64 * 1e-9;
        let truth: Matrix3<f64> = run.walk.rotation(t0).transpose() * run.walk.rotation(t1);
        // Frobenius norm of R^T R' - I is sqrt(2) times the angle; f32 poses.
        let error: f64 = (truth.transpose() * predicted - Matrix3::identity()).norm();
        assert!(error < 2e-5, "frameset {k}: {error}");
        assert_eq!(prediction.t_w_i_current.translation, Vector3::zeros());
    }
}

#[test]
fn slides_past_a_still_start_then_accepts() {
    let mut run = Run::new(3.0, BootstrapOptions::default());
    let (_, accepted) = run.until_accepted(200).unwrap();
    let report = run.driver.report();
    assert!(
        report.attempts[0].verdict.starts_with("LowExcitation"),
        "{:?}",
        report.attempts[0]
    );
    assert!(report.attempts.len() >= 3);
    assert!(
        report.attempts.iter().any(|a| a.first_t_ns > 0),
        "the window never slid"
    );
    assert!(report.attempts.last().unwrap().accepted);
    assert!(accepted.initial.t_ns >= 500_000_000);
    let first = accepted.initial.t_ns;
    assert_eq!(accepted.frames[0].t_ns, first);
}

#[test]
fn never_accepting_is_a_typed_error_with_a_bounded_buffer() {
    let options = BootstrapOptions {
        timeout_s: 4.0,
        ..BootstrapOptions::default()
    };
    let mut run = Run::new(1e6, options);
    let error = run.until_accepted(400).unwrap_err();
    match error {
        BootstrapError::NeverAccepted {
            seconds,
            attempts,
            last_verdict,
        } => {
            assert!((4.0..4.6).contains(&seconds), "{seconds}");
            assert!(attempts >= 3);
            assert!(last_verdict.starts_with("LowExcitation"), "{last_verdict}");
        }
        other => panic!("{other:?}"),
    }
}

/// Push the IMU through `imu_ahead_s` first, then framesets `0..frames`;
/// return the first step that is not `Buffering`, or `None`.
fn imu_first(
    imu_ahead_s: f64,
    options: BootstrapOptions,
    frames: i64,
) -> Option<Result<BootstrapStep, BootstrapError>> {
    let mut run = Run::new(-1.0, options);
    while run.next_imu * 1_000_000 <= seconds_to_ns(imu_ahead_s) {
        run.driver.push_imu(run.walk.imu(run.next_imu * 1_000_000));
        run.next_imu += 1;
    }
    for k in 0..frames {
        run.driver.predict(k * FRAME_NS);
        match run.driver.push_frame(run.frame(k)) {
            Ok(BootstrapStep::Buffering) => {}
            other => return Some(other),
        }
    }
    None
}

/// A caller may push IMU seconds ahead of the framesets: the framesets are
/// still buffered and solved, or the timeout fires; the driver never buffers
/// forever.
#[test]
fn imu_pushed_far_ahead_still_ends_the_bootstrap() {
    // 5 s of IMU, then the 150 framesets it covers: the 2 s window is solved.
    let step = imu_first(5.0, BootstrapOptions::default(), 150);
    assert!(
        matches!(step, Some(Ok(BootstrapStep::Accepted(_)))),
        "{step:?}"
    );
    // 40 s of IMU ahead of a 4 s timeout: the first framesets fall before the
    // IMU kept, and the timeout still fires on a buffering frameset.
    let options = BootstrapOptions {
        timeout_s: 4.0,
        ..BootstrapOptions::default()
    };
    match imu_first(40.0, options, 300) {
        Some(Err(BootstrapError::NeverAccepted {
            seconds, attempts, ..
        })) => {
            assert!((4.0..4.1).contains(&seconds), "{seconds}");
            assert_eq!(attempts, 0);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn estimator_starts_from_the_solved_window_and_tracks_monocular() {
    let mut run = Run::new(-1.0, BootstrapOptions::default());
    let (k_accepted, accepted) = run.until_accepted(120).unwrap();
    let config: VioConfig = VioConfig::from_json_str(MSDMI_CONFIG).unwrap();
    let initial = accepted.initial;
    let mut estimator: SqrtKeypointVio<f64> = SqrtKeypointVio::with_initial_state(
        crate::imu::gravity(),
        run.calibration.clone(),
        config,
        initial,
    )
    .unwrap();
    for sample in &accepted.imu {
        estimator.push_imu(*sample);
    }
    for i in run.next_imu..=12_000 {
        estimator.push_imu(run.walk.imu(i * 1_000_000));
    }
    for frame in accepted.frames {
        let outcome = estimator.process_frame(frame, None).unwrap();
        assert!(matches!(outcome, FrameOutcome::Measured(_)));
    }
    let last: i64 = 180;
    for k in k_accepted + 1..=last {
        let outcome = estimator.process_frame(run.frame(k), None).unwrap();
        assert!(matches!(outcome, FrameOutcome::Measured(_)), "frameset {k}");
    }
    let state = estimator.state().unwrap();
    let landmarks: usize = estimator.snapshot().landmarks.len();
    println!("landmarks {landmarks}");
    // Compare in the first IMU frame: yaw is not observable.
    let r_w_b0: Matrix3<f64> = initial.t_w_i.rotation.matrix();
    let estimate: Vector3<f64> = r_w_b0.transpose() * state.t_w_i.translation;
    let t: f64 = last as f64 * FRAME_NS as f64 * 1e-9;
    let truth: Vector3<f64> =
        run.walk.rotation(0.0).transpose() * (run.walk.position(t) - run.walk.position(0.0));
    let error: f64 = (estimate - truth).norm();
    println!("error {error} over {}", truth.norm());
    assert!(landmarks > 20, "temporal triangulation made no map");
    assert!(
        error < 0.03 * truth.norm(),
        "position error {error} m over {} m",
        truth.norm()
    );
}

#[test]
fn stereo_stays_the_default_and_is_not_written_out() {
    let config: VioConfig = VioConfig::from_json_str(MSDMI_CONFIG).unwrap();
    assert_eq!(config.vio_bootstrap, VioBootstrap::Stereo);
    assert!(!config.to_json_string().unwrap().contains("vio_bootstrap"));
    let inertial =
        VioConfig::with_overlay(MSDMI_CONFIG, r#"{"config.vio_bootstrap": "inertial"}"#).unwrap();
    assert_eq!(inertial.vio_bootstrap, VioBootstrap::Inertial);
    assert!(
        inertial
            .to_json_string()
            .unwrap()
            .contains(r#""config.vio_bootstrap": "inertial""#)
    );

    let vio: Vio<f32> = Vio::new(
        config,
        calibration(),
        crate::frontend::flow::FrontendOptions::default(),
    )
    .unwrap();
    assert!(vio.bootstrap_report().is_none());
    assert!(matches!(vio.bootstrap, BootstrapPhase::Off));
}

#[test]
fn frontend_lag_is_refused_with_the_bootstrap() {
    let config = VioConfig::with_overlay(
        MSDMI_CONFIG,
        r#"{"config.vio_bootstrap": "inertial", "port.frontend_lag": true}"#,
    )
    .unwrap();
    let error = Vio::<f32>::new(
        config,
        calibration(),
        crate::frontend::flow::FrontendOptions::default(),
    )
    .err();
    assert_eq!(
        error,
        Some(VioError::Bootstrap(BootstrapError::FrontendLag))
    );
}
