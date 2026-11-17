//! The driver on a synthetic walk seen by the MSD Index rig (KB4), with an
//! exact IMU; and the estimator started from its answer.
#![allow(clippy::unwrap_used)]

use super::*;
use crate::config::VioConfig;
use crate::estimator::{FrameOutcome, SqrtKeypointVio};
use crate::lie::So3;
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
    calib_f32: Calibration<f32>,
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
            calib_f32: calibration.cast(),
            calibration,
            points: room(),
            driver,
            next_imu: -50,
        }
    }
    /// Push the IMU up to 2 ms past frameset `k`, as the feed does.
    fn imu_through(&mut self, k: i64) {
        while self.next_imu * 1_000_000 <= k * FRAME_NS + 2_000_000 {
            self.push_next_imu();
        }
    }
    fn push_next_imu(&mut self) {
        let sample: CombinedImuSample = self.walk.imu(self.next_imu * 1_000_000);
        self.driver
            .push_imu(sample, super::super::calibrated(&self.calib_f32, &sample));
        self.next_imu += 1;
    }
    fn frame(&self, k: i64) -> Arc<FlowObservations> {
        frameset(&self.walk, &self.calibration, &self.points, k * FRAME_NS)
    }
    /// Feed framesets until one is accepted or `max_frames` pass.
    fn until_accepted(&mut self, max_frames: i64) -> Result<(i64, Box<Accepted>), BootstrapError> {
        for k in 0..max_frames {
            self.imu_through(k);
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
    // Nothing is stamped before the caller commits the start.
    assert_eq!(report.start_t_ns, None);
    assert_eq!(
        run.driver.buffered_frames(),
        62,
        "the window stays buffered"
    );

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
    let report = run.driver.commit();
    assert_eq!(
        (report.start_t_ns, report.accepted_at_t_ns, report.replayed),
        (Some(0), Some(61 * FRAME_NS), 62)
    );
}

fn inertial_vio() -> Vio<f64> {
    let config =
        VioConfig::with_overlay(MSDMI_CONFIG, r#"{"config.vio_bootstrap": "inertial"}"#).unwrap();
    Vio::new(
        config,
        calibration(),
        crate::frontend::flow::FrontendOptions::default(),
    )
    .unwrap()
}

/// Push the walk's IMU at 1 kHz through `through_ns` into `vio`, from sample `next`.
fn push_walk_imu(vio: &mut Vio<f64>, walk: &Walk, next: &mut i64, through_ns: i64) {
    while *next * 1_000_000 <= through_ns {
        let sample = walk.imu(*next * 1_000_000);
        vio.push_imu(
            sample.timestamp_ns,
            sample.gyro.to_array(),
            sample.accel.to_array(),
        )
        .unwrap();
        *next += 1;
    }
}

/// Before any state the frontend gets the gyro's rotation between framesets,
/// from the pipeline's own frontend preintegration.
#[test]
fn gyro_prediction_follows_the_rotation() {
    let walk = Walk { static_s: -1.0 };
    let mut vio = inertial_vio();
    let mut next: i64 = -50;
    push_walk_imu(&mut vio, &walk, &mut next, 2_000_000);
    vio.drop_frontend_imu_through(0);
    for k in 1..20 {
        push_walk_imu(&mut vio, &walk, &mut next, k * FRAME_NS + 2_000_000);
        let prediction = vio
            .gyro_prediction((k - 1) * FRAME_NS, k * FRAME_NS)
            .unwrap();
        vio.last_frame_t_ns = Some(k * FRAME_NS);
        let predicted: Matrix3<f64> = (prediction.t_w_i_previous.rotation.inverse()
            * prediction.t_w_i_current.rotation)
            .matrix()
            .cast();
        let t0: f64 = (k - 1) as f64 * FRAME_NS as f64 * 1e-9;
        let t1: f64 = k as f64 * FRAME_NS as f64 * 1e-9;
        let truth: Matrix3<f64> = walk.rotation(t0).transpose() * walk.rotation(t1);
        // Frobenius norm of R^T R' - I is sqrt(2) times the angle. The
        // pipeline's preintegration holds each 1 kHz sample over its interval,
        // so it lags the truth by about half a sample: up to 2.2e-4 rad here,
        // 0.07 px at the MSD cameras' 300 px/rad.
        let error: f64 = (truth.transpose() * predicted - Matrix3::identity()).norm();
        assert!(error < 5e-4, "frameset {k}: {error}");
        assert_eq!(prediction.t_w_i_current.translation, Vector3::zeros());
    }
}

/// After a still start the schedule is: a 2 s window, the 3 s window, then
/// the 3 s window slid by 0.5 s per rejection, so attempts are 0.5 s of
/// framesets apart and never on consecutive framesets.
#[test]
fn slides_past_a_still_start_then_accepts() {
    let mut run = Run::new(3.0, BootstrapOptions::default());
    let (_, accepted) = run.until_accepted(200).unwrap();
    let report = run.driver.report();
    assert_eq!(verdict_name(&report.attempts[0].verdict), "LowExcitation");
    assert!(report.attempts.len() >= 3);
    assert!(report.attempts.last().unwrap().accepted());
    let span = |a: &BootstrapAttempt| (a.last_t_ns - a.first_t_ns) as f64 * 1e-9;
    let frame_s: f64 = FRAME_NS as f64 * 1e-9;
    assert!(
        (span(&report.attempts[0]) - 2.0).abs() <= frame_s,
        "{:?}",
        report.attempts[0]
    );
    for pair in report.attempts.windows(2) {
        assert!((span(&pair[1]) - 3.0).abs() <= frame_s, "{:?}", pair[1]);
        let gap: f64 = (pair[1].last_t_ns - pair[0].last_t_ns) as f64 * 1e-9;
        assert!(
            gap >= 0.5 - frame_s,
            "attempts {gap} s apart: {:?}",
            pair[1]
        );
    }
    let slid = &report.attempts[2];
    assert!(
        (slid.first_t_ns as f64 * 1e-9 - 0.5).abs() <= frame_s,
        "the third window starts 0.5 s in: {slid:?}"
    );
    let first = accepted.initial.t_ns;
    assert_eq!(accepted.frames[0].t_ns, first);
}

/// A rig at rest for longer than the timeout is "not yet": the bootstrap
/// waits, and accepts once the rig moves.
#[test]
fn a_still_start_longer_than_the_timeout_still_accepts() {
    let mut run = Run::new(35.0, BootstrapOptions::default());
    let (k, accepted) = run.until_accepted(1_300).unwrap();
    assert!(
        k * FRAME_NS > seconds_to_ns(35.0),
        "accepted at frameset {k}"
    );
    let report = run.driver.report();
    assert!(report.attempts.len() > 60);
    assert!(
        report
            .attempts
            .iter()
            .filter(|a| a.last_t_ns < seconds_to_ns(35.0))
            .all(|a| verdict_name(&a.verdict) == "LowExcitation")
    );
    assert!(accepted.initial.t_ns > seconds_to_ns(30.0));
}

/// Windows with excitation that never solve (no tracks at all) end the
/// bootstrap once they span the timeout, on that frameset and every later
/// one; the buffer stays bounded.
#[test]
fn never_accepting_is_a_typed_error_with_a_bounded_buffer() {
    let options = BootstrapOptions {
        timeout_s: 4.0,
        ..BootstrapOptions::default()
    };
    let mut run = Run::new(-1.0, options);
    run.points.clear();
    let error = run.until_accepted(400).unwrap_err();
    match error {
        BootstrapError::NeverAccepted {
            seconds,
            attempts,
            last_verdict,
        } => {
            assert!((4.0..4.6).contains(&seconds), "{seconds}");
            assert!(attempts >= 3);
            assert_eq!(last_verdict, "TooFewTracks");
        }
        other => panic!("{other:?}"),
    }
    let attempts: usize = run.driver.report().attempts.len();
    let k: i64 = 400;
    run.imu_through(k);
    assert!(matches!(
        run.driver.push_frame(run.frame(k)),
        Err(BootstrapError::NeverAccepted { .. })
    ));
    assert_eq!(run.driver.report().attempts.len(), attempts);
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
        run.push_next_imu();
    }
    for k in 0..frames {
        match run.driver.push_frame(run.frame(k)) {
            Ok(BootstrapStep::Buffering) => {}
            other => return Some(other),
        }
    }
    None
}

/// A caller may push IMU seconds ahead of the framesets: the framesets the
/// IMU kept still covers are buffered and solved, the others are dropped; the
/// driver never buffers forever.
#[test]
fn imu_pushed_far_ahead_still_ends_the_bootstrap() {
    // 5 s of IMU, then the 150 framesets it covers: the 2 s window is solved.
    let step = imu_first(5.0, BootstrapOptions::default(), 150);
    assert!(
        matches!(step, Some(Ok(BootstrapStep::Accepted(_)))),
        "{step:?}"
    );
    // 40 s of IMU ahead of a 4 s timeout: the IMU kept starts at 36 s, so
    // the first 300 framesets (10 s) fall before it and are dropped.
    let options = BootstrapOptions {
        timeout_s: 4.0,
        ..BootstrapOptions::default()
    };
    assert!(imu_first(40.0, options, 300).is_none());
}

#[test]
fn estimator_starts_from_the_solved_window_and_tracks_monocular() {
    let mut run = Run::new(-1.0, BootstrapOptions::default());
    let (k_accepted, accepted) = run.until_accepted(120).unwrap();
    let config: VioConfig = VioConfig::from_json_str(MSDMI_CONFIG).unwrap();
    let initial = accepted.initial;
    let mut estimator: SqrtKeypointVio<f64> = SqrtKeypointVio::with_initial_state(
        crate::imu::gravity(9.81),
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
fn stereo_stays_the_default_and_is_written_out() {
    let config: VioConfig = VioConfig::from_json_str(MSDMI_CONFIG).unwrap();
    assert_eq!(config.vio_bootstrap, VioBootstrap::Stereo);
    assert!(
        config
            .to_json_string()
            .unwrap()
            .contains(r#""config.vio_bootstrap": "stereo""#)
    );
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

/// A replay that fails leaves the pipeline bootstrapping with the accepted
/// window still buffered and no start in the report; the estimator is the
/// one it had.
#[test]
fn a_failed_replay_changes_nothing() {
    let walk = Walk { static_s: -1.0 };
    let calibration: Calibration<f64> = calibration();
    let points: Vec<Vector3<f64>> = room();
    let mut vio = inertial_vio();
    let mut next: i64 = -50;
    let mut result = None;
    for k in 0..70 {
        push_walk_imu(&mut vio, &walk, &mut next, k * FRAME_NS + 2_000_000);
        let mut frame: FlowObservations =
            (*frameset(&walk, &calibration, &points, k * FRAME_NS)).clone();
        if k == 5 {
            // The solver ignores a camera the rig does not have; the estimator refuses it.
            frame.cameras.push(Default::default());
        }
        match vio.bootstrap_track(Arc::new(frame)) {
            Ok(r) => assert_eq!(r.status, VioStatus::NoVisualFeatures, "frameset {k}"),
            Err(error) => {
                result = Some((k, error));
                break;
            }
        }
    }
    let Some((k, error)) = result else {
        panic!("the replay of the accepted window did not fail");
    };
    assert_eq!(k, 61);
    assert!(matches!(error, VioError::Estimator(_)), "{error:?}");
    let BootstrapPhase::Running(driver) = &vio.bootstrap else {
        panic!("the pipeline left the bootstrap");
    };
    assert_eq!(driver.buffered_frames(), 62);
    let report = driver.report();
    assert_eq!(report.attempts.len(), 1);
    assert!(report.attempts[0].accepted());
    assert_eq!(
        (report.start_t_ns, report.accepted_at_t_ns, report.replayed),
        (None, None, 0)
    );
    assert!(vio.estimator.state().is_none());
}
