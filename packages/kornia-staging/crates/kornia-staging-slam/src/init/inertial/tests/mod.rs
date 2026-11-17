//! Synthetic tests of the inertial initialisation.
mod synth;

use super::*;
use synth::{generate, Motion, Rig, Scenario, Synthetic};

fn solve(data: &Synthetic, options: &InertialInitOptions) -> InertialInitOutcome {
    let problem = InertialInitProblem {
        imu: &data.imu,
        frame_times_ns: &data.frame_times_ns,
        tracks: &data.tracks,
        t_i_c: &data.t_i_c,
        gravity_magnitude: synth::GRAVITY,
        gyro_bias_prior: None,
    };
    solve_inertial_init(&problem, options).expect("well-formed input")
}

/// Velocity error (m/s), gravity direction error (deg), scale error (fraction)
/// of the window trajectory: `argmin_s |s p_hat - p|` minus one.
#[derive(Debug, Clone, Copy)]
struct Errors {
    velocity: f64,
    gravity_deg: f64,
    scale: f64,
    bias_gyro: f64,
    bias_accel: f64,
}

fn errors(solution: &InertialInit, data: &Synthetic) -> Errors {
    let truth = &data.truth;
    let (num, den) = solution
        .positions_body0
        .iter()
        .zip(&truth.positions_body0)
        .fold((0.0, 0.0), |(n, d), (e, t)| {
            (n + e.dot(t), d + t.norm_squared())
        });
    // scale of the estimate relative to the truth: p_hat = s p
    let scale: f64 = num / den - 1.0;
    let cos: f64 = solution
        .gravity_body0
        .normalize()
        .dot(&truth.gravity_body0.normalize());
    Errors {
        velocity: (solution.velocity_body0 - truth.velocity_body0).norm(),
        gravity_deg: cos.clamp(-1.0, 1.0).acos().to_degrees(),
        scale,
        bias_gyro: (solution.bias_gyro - truth.bias_gyro).norm(),
        bias_accel: (solution.bias_accel - truth.bias_accel).norm(),
    }
}

fn exact_recovery(rig: Rig) {
    let data = generate(&Scenario::clean(rig, 2.0));
    let outcome = solve(&data, &InertialInitOptions::default());
    let solution = outcome
        .accepted()
        .unwrap_or_else(|| panic!("{rig:?}: {:?} {:?}", outcome.verdict, outcome.quality));
    let e = errors(solution, &data);
    println!("{rig:?}: {e:?} {:?}", outcome.quality);
    assert!(e.velocity < 2e-3, "{rig:?} velocity {e:?}");
    assert!(e.gravity_deg < 0.02, "{rig:?} gravity {e:?}");
    assert!(e.scale.abs() < 1e-3, "{rig:?} scale {e:?}");
    assert!(e.bias_gyro < 1e-4, "{rig:?} gyro bias {e:?}");
    assert!(outcome.quality.residual_rms_rad.unwrap() < 1e-4);
    assert!(solution.outlier_track_ids.is_empty());
}

#[test]
fn exact_recovery_two_centre() {
    exact_recovery(Rig::TwoCentre);
}

#[test]
fn exact_recovery_central() {
    exact_recovery(Rig::Central);
}

#[test]
fn exact_recovery_mono_with_blind_camera() {
    exact_recovery(Rig::Mono);
}

/// The world rotation follows slam-rs's convention: it takes the at-rest
/// accelerometer direction `-g` onto +Z, and world gravity is `(0, 0, -|g|)`.
#[test]
fn world_rotation_takes_up_to_plus_z() {
    let data = generate(&Scenario::clean(Rig::TwoCentre, 2.0));
    let outcome = solve(&data, &InertialInitOptions::default());
    let s = outcome.accepted().expect("accepted");
    let r: Matrix3<f64> = s.world_from_body0.matrix();
    let g_world: Vector3<f64> = r * s.gravity_body0;
    assert!((g_world - Vector3::new(0.0, 0.0, -synth::GRAVITY)).norm() < 1e-9);
    assert!((s.velocity_world - r * s.velocity_body0).norm() < 1e-12);
    for g in [
        Vector3::new(0.3, -9.7, 1.0),
        Vector3::new(0.0, 0.0, 9.81),
        Vector3::new(0.0, 0.0, -9.81),
    ] {
        let up = world_from_gravity(&g).matrix() * (-g.normalize());
        assert!((up - Vector3::z()).norm() < 1e-12, "{g:?}");
    }
}

/// Noise-free accelerometer bias. Its component along gravity is separable
/// from scale through the excitation; the horizontal part trades against a
/// gravity tilt of `|b_a| / g` and is separated only by the window's rotation,
/// so it is recovered to a few mm/s^2, not exactly.
#[test]
fn accel_bias_is_recovered_when_estimated() {
    let mut scenario = Scenario::clean(Rig::TwoCentre, 2.0);
    scenario.accel_bias = Vector3::new(0.05, -0.03, 0.04);
    let data = generate(&scenario);
    let outcome = solve(&data, &InertialInitOptions::default());
    let s = outcome.accepted().expect("accepted");
    let e = errors(s, &data);
    println!("{e:?}");
    assert!(e.bias_accel < 0.01, "{e:?}");
    assert!(e.gravity_deg < 0.06, "{e:?}");
    assert!(e.scale.abs() < 2e-3, "{e:?}");
}

#[test]
fn deterministic() {
    let data = generate(&Scenario::noisy(Rig::TwoCentre, 2.0, 3));
    let a = solve(&data, &InertialInitOptions::default());
    let b = solve(&data, &InertialInitOptions::default());
    assert_eq!(a, b);
    let bits = |o: &InertialInitOutcome| -> Vec<u64> {
        let s = o.solution.as_ref().expect("solution");
        s.velocity_body0
            .iter()
            .chain(s.gravity_body0.iter())
            .chain(s.bias_gyro.iter())
            .map(|v| v.to_bits())
            .collect()
    };
    assert_eq!(bits(&a), bits(&b));
}

/// An accepted answer must lie within its own uncertainty.
fn assert_not_confidently_wrong(label: &str, outcome: &InertialInitOutcome, data: &Synthetic) {
    println!("{label}: {:?} {:?}", outcome.verdict, outcome.quality);
    if let Some(solution) = outcome.accepted() {
        let e = errors(solution, data);
        let sigma: f64 = outcome.quality.scale_sigma.expect("accepted has a sigma");
        println!("{label}: accepted with {e:?}");
        assert!(
            e.scale.abs() <= 3.0 * sigma + 0.005,
            "{label}: {e:?} sigma {sigma}"
        );
    }
}

#[test]
fn rejects_constant_velocity() {
    let gate_off = InertialInitOptions {
        min_excitation_mps2: 0.0,
        ..InertialInitOptions::default()
    };
    for rig in [Rig::TwoCentre, Rig::Central, Rig::Mono] {
        let mut scenario = Scenario::noisy(rig, 2.0, 11);
        scenario.motion = Motion::ConstantVelocity;
        let data = generate(&scenario);
        let outcome = solve(&data, &InertialInitOptions::default());
        assert!(
            matches!(
                outcome.verdict,
                InitVerdict::Rejected(RejectReason::LowExcitation { .. })
            ),
            "{rig:?}: {:?}",
            outcome.verdict
        );
        // Without the excitation gate: a single-centre rig has no scale at
        // all and must be refused. The two-centre rig keeps the
        // 33 mm baseline of the tracks seen by both lenses; it may answer,
        // but only within its stated uncertainty.
        let outcome = solve(&data, &gate_off);
        assert_not_confidently_wrong(
            &format!("constant velocity {rig:?}, gate off"),
            &outcome,
            &data,
        );
        if rig != Rig::TwoCentre {
            assert!(
                outcome.accepted().is_none(),
                "{rig:?}: {:?}",
                outcome.verdict
            );
        }
    }
}

#[test]
fn rejects_pure_rotation() {
    let mut scenario = Scenario::noisy(Rig::TwoCentre, 2.0, 12);
    scenario.motion = Motion::PureRotation;
    let data = generate(&scenario);
    let outcome = solve(&data, &InertialInitOptions::default());
    assert!(outcome.accepted().is_none(), "{:?}", outcome.verdict);
    let options = InertialInitOptions {
        min_excitation_mps2: 0.0,
        ..InertialInitOptions::default()
    };
    let outcome = solve(&data, &options);
    println!(
        "pure rotation, excitation gate off: {:?} {:?}",
        outcome.verdict, outcome.quality
    );
    assert!(outcome.accepted().is_none(), "{:?}", outcome.verdict);
    assert!(outcome.solution.is_none());
}

#[test]
fn rejects_too_few_tracks() {
    let mut scenario = Scenario::noisy(Rig::TwoCentre, 2.0, 13);
    scenario.inlier_tracks = 12;
    scenario.outlier_fraction = 0.0;
    let data = generate(&scenario);
    let outcome = solve(&data, &InertialInitOptions::default());
    assert!(
        matches!(
            outcome.verdict,
            InitVerdict::Rejected(RejectReason::TooFewTracks { .. })
        ),
        "{:?}",
        outcome.verdict
    );
    assert!(outcome.solution.is_none());
}

#[test]
fn rejects_malformed_input() {
    let data = generate(&Scenario::clean(Rig::TwoCentre, 1.0));
    let mut problem = InertialInitProblem {
        imu: &data.imu,
        frame_times_ns: &data.frame_times_ns,
        tracks: &data.tracks,
        t_i_c: &data.t_i_c,
        gravity_magnitude: synth::GRAVITY,
        gyro_bias_prior: None,
    };
    let short = &data.imu[200..];
    problem.imu = short;
    assert_eq!(
        solve_inertial_init(&problem, &InertialInitOptions::default()),
        Err(InertialInitError::ImuDoesNotCoverWindow)
    );
    problem.imu = &data.imu;
    problem.t_i_c = &data.t_i_c[..1];
    assert!(matches!(
        solve_inertial_init(&problem, &InertialInitOptions::default()),
        Err(InertialInitError::InvalidObservation { .. })
    ));
}

// ---------------------------------------------------------------------------
// Noise trials and sensitivity. Printed with `--nocapture`.

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let i: usize = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

/// Median, p90 and max.
type Summary = (f64, f64, f64);

struct Stats {
    accepted: usize,
    trials: usize,
    velocity: Vec<f64>,
    gravity_deg: Vec<f64>,
    scale: Vec<f64>,
    bias_gyro: Vec<f64>,
    scale_sigma: Vec<f64>,
    residual: Vec<f64>,
    rejected: std::collections::BTreeMap<String, usize>,
}

impl Stats {
    fn new() -> Self {
        Self {
            accepted: 0,
            trials: 0,
            velocity: vec![],
            gravity_deg: vec![],
            scale: vec![],
            bias_gyro: vec![],
            scale_sigma: vec![],
            residual: vec![],
            rejected: std::collections::BTreeMap::new(),
        }
    }
    fn push(&mut self, result: Result<(Errors, f64, f64), String>) {
        self.trials += 1;
        match result {
            Ok((e, sigma, residual)) => {
                self.residual.push(residual * 1e3);
                self.accepted += 1;
                self.velocity.push(e.velocity);
                self.gravity_deg.push(e.gravity_deg);
                self.scale.push(e.scale.abs() * 100.0);
                self.bias_gyro.push(e.bias_gyro.to_degrees());
                self.scale_sigma.push(sigma * 100.0);
            }
            Err(reason) => *self.rejected.entry(reason).or_default() += 1,
        }
    }
    fn summary(v: &mut [f64]) -> (f64, f64, f64) {
        if v.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        v.sort_by(f64::total_cmp);
        (percentile(v, 0.5), percentile(v, 0.9), v[v.len() - 1])
    }
    fn row(&mut self, label: &str) -> [(f64, f64, f64); 4] {
        let rows = [
            Self::summary(&mut self.velocity),
            Self::summary(&mut self.gravity_deg),
            Self::summary(&mut self.scale),
            Self::summary(&mut self.bias_gyro),
        ];
        let f = |(a, b, c): (f64, f64, f64)| format!("{a:.4} / {b:.4} / {c:.4}");
        let sigma = Self::summary(&mut self.scale_sigma).0;
        let residual = Self::summary(&mut self.residual).0;
        let rejected: Vec<String> = self
            .rejected
            .iter()
            .map(|(k, v)| format!("{k} x{v}"))
            .collect();
        println!(
            "| {label} | {}/{} | {} | {} | {} | {sigma:.2} | {residual:.2} | {} | {} |",
            self.accepted,
            self.trials,
            f(rows[0]),
            f(rows[1]),
            f(rows[2]),
            f(rows[3]),
            rejected.join(", ")
        );
        rows
    }
}

fn trial(scenario: &Scenario, options: &InertialInitOptions) -> Result<(Errors, f64, f64), String> {
    let data = generate(scenario);
    let outcome = solve(&data, options);
    match (outcome.accepted(), outcome.verdict) {
        (Some(s), _) => Ok((
            errors(s, &data),
            outcome.quality.scale_sigma.unwrap_or(f64::NAN),
            outcome.quality.residual_rms_rad.unwrap_or(f64::NAN),
        )),
        (None, InitVerdict::Rejected(reason)) => Err(format!("{reason:?}")
            .split([' ', '{'])
            .next()
            .unwrap_or("")
            .to_string()),
        (None, InitVerdict::Accepted) => Err("no solution".to_string()),
    }
}

const HEADER: &str = "| case | accepted | velocity err m/s (median / p90 / max) | gravity err deg | abs scale err % | scale sigma % (median) | residual RMS mrad (median) | gyro bias err deg/s | rejections |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |";

/// A few noisy windows in the default lane; the full trials are below.
#[test]
fn noise_trials_smoke() {
    for seed in 0..3 {
        let data = generate(&Scenario::noisy(Rig::TwoCentre, 2.0, 1000 + seed));
        let outcome = solve(&data, &InertialInitOptions::default());
        let solution = outcome
            .accepted()
            .unwrap_or_else(|| panic!("seed {seed}: {:?}", outcome.verdict));
        let e = errors(solution, &data);
        assert!(
            e.scale.abs() < 0.03 && e.velocity < 0.05 && e.gravity_deg < 1.0,
            "seed {seed}: {e:?}"
        );
        assert!(
            solution.outlier_track_ids.len() >= 30,
            "seed {seed}: outliers {}",
            solution.outlier_track_ids.len()
        );
    }
}

/// 50 seeded trials per window length with realistic noise, biases and 20 %
/// outlier tracks fixed in the image; the two-centre 360 rig. About 10 s in
/// release and too slow unoptimised, so it runs on request:
/// `cargo test --release -p kornia-staging-slam --lib init::inertial -- --ignored --nocapture`.
#[test]
#[ignore = "Monte Carlo; run with --release -- --ignored --nocapture"]
fn noise_trials() {
    let seeds: u64 = 50;
    println!("\nnoise trials, two-centre rig, {seeds} seeds each");
    println!("{HEADER}");
    let mut results: Vec<(f64, bool, usize, [Summary; 4])> = Vec::new();
    for (estimate_accel_bias, tag) in [(false, "ba fixed 0"), (true, "ba estimated")] {
        let options = InertialInitOptions {
            estimate_accel_bias,
            ..InertialInitOptions::default()
        };
        for window in [1.0, 2.0, 3.0] {
            let mut stats = Stats::new();
            for seed in 0..seeds {
                stats.push(trial(
                    &Scenario::noisy(Rig::TwoCentre, window, 1000 + seed),
                    &options,
                ));
            }
            let rows = stats.row(&format!("{window:.0} s, {tag}"));
            results.push((window, estimate_accel_bias, stats.accepted, rows));
        }
    }
    // The other rigs at the default window, default options.
    for rig in [Rig::Central, Rig::Mono] {
        let mut stats = Stats::new();
        for seed in 0..seeds {
            stats.push(trial(
                &Scenario::noisy(rig, 2.0, 1000 + seed),
                &InertialInitOptions::default(),
            ));
        }
        let rows = stats.row(&format!("2 s, {rig:?} rig, ba estimated"));
        assert!(
            stats.accepted as f64 >= 0.9 * seeds as f64,
            "{rig:?}: accepted {}",
            stats.accepted
        );
        assert!(rows[2].1 < 3.0 && rows[2].2 < 5.0, "{rig:?}: {rows:?}");
    }
    for (window, estimated, accepted, rows) in results {
        let label = format!("{window} s, ba estimated {estimated}");
        let [velocity, gravity, scale, bias_gyro] = rows;
        if !estimated {
            // The comparison lane: the accelerometer bias held at zero.
            assert!(scale.2 < 20.0, "{label}: {rows:?}");
            continue;
        }
        // The default: bounds with margin over the measured table.
        assert!(
            accepted as f64 >= 0.9 * seeds as f64,
            "{label}: accepted {accepted}"
        );
        assert!(velocity.2 < 0.05, "{label}: {rows:?}");
        assert!(gravity.2 < 1.0, "{label}: {rows:?}");
        assert!(bias_gyro.2 < 0.15, "{label}: {rows:?}");
        let (p90, max) = if window < 1.5 { (2.5, 4.0) } else { (2.0, 3.0) };
        assert!(scale.1 < p90 && scale.2 < max, "{label}: {rows:?}");
    }
}

/// Scale error against calibration errors, 2 s window, 20 seeds each. Not
/// asserted tightly: the table sets the calibration budget. Run as
/// [`noise_trials`].
#[test]
#[ignore = "Monte Carlo; run with --release -- --ignored --nocapture"]
fn sensitivity_study() {
    let seeds: u64 = 20;
    let options = InertialInitOptions::default();
    println!("\nsensitivity, two-centre rig, 2 s window, {seeds} seeds each");
    println!("{HEADER}");
    let cases: [(&str, f64, Vector3<f64>, f64); 6] = [
        ("baseline", 0.0, Vector3::zeros(), 0.0),
        ("time offset 5 ms", 0.005, Vector3::zeros(), 0.0),
        ("time offset 10 ms", 0.010, Vector3::zeros(), 0.0),
        ("lever arm 3 cm", 0.0, Vector3::new(0.03, 0.0, 0.0), 0.0),
        (
            "lever arm 3 cm on axis",
            0.0,
            Vector3::new(0.0, 0.0, 0.03),
            0.0,
        ),
        ("rolling shutter +-8 ms", 0.0, Vector3::zeros(), 0.008),
    ];
    for (label, offset, lever, rs) in cases {
        let mut stats = Stats::new();
        for seed in 0..seeds {
            let mut scenario = Scenario::noisy(Rig::TwoCentre, 2.0, 2000 + seed);
            scenario.time_offset_s = offset;
            scenario.lever_arm_error = lever;
            scenario.rolling_shutter_s = rs;
            stats.push(trial(&scenario, &options));
        }
        let rows = stats.row(label);
        assert!(rows[2].2.is_nan() || rows[2].2 < 50.0, "{label}: {rows:?}");
    }
}

/// The refinement's gradient matches central differences of its cost, for
/// every reduced parameter (velocity, gravity tangent, gyro bias, accel bias).
#[test]
fn refinement_gradient_matches_finite_differences() {
    let data = generate(&Scenario::clean(Rig::TwoCentre, 2.0));
    let cameras: Vec<linear::Camera> = data
        .t_i_c
        .iter()
        .map(|t| linear::Camera {
            r_ic: t.rotation.matrix(),
            t_ic: t.translation,
        })
        .collect();
    let zero: Vector3<f64> = Vector3::zeros();
    let preint = preintegration::preintegrate(&data.imu, &data.frame_times_ns, &zero, &zero)
        .expect("covered");
    let tracks: Vec<&[BearingObservation]> = data
        .tracks
        .iter()
        .take(30)
        .map(|t| t.observations.as_slice())
        .collect();
    let problem = refine::Problem {
        preint: &preint,
        cameras: &cameras,
        tracks: &tracks,
        huber: 1.0, // quadratic everywhere: cost = sum |r|^2
        bearing_sigma: 1.0,
        estimate_accel_bias: true,
        gyro_bias_prior: zero,
        gyro_bias_sigma: 1e9,
        accel_bias_sigma: 1e9,
    };
    // Off the solution, at the gyro linearisation point (where the first-order
    // rotation model is exact in its derivative).
    let state = refine::State {
        velocity: data.truth.velocity_body0 + Vector3::new(0.01, 0.02, -0.01),
        gravity: data.truth.gravity_body0,
        bias_gyro: zero,
        bias_accel: Vector3::new(0.01, 0.0, -0.01),
        points: vec![Vector3::new(1.0, 2.0, 3.0); tracks.len()],
    };
    let basis = refine::gravity_basis(&state.gravity);
    let normal = refine::linearize(&problem, &state, &basis, 11);
    let h: f64 = 1e-7;
    let no_point_step = vec![zero; tracks.len()];
    for i in 0..11 {
        let mut dy: nalgebra::DVector<f64> = nalgebra::DVector::zeros(11);
        dy[i] = h;
        let plus = refine::apply(&state, &dy, &no_point_step, &basis, true);
        dy[i] = -h;
        let minus = refine::apply(&state, &dy, &no_point_step, &basis, true);
        // g_y is J^T r, half the gradient of sum |r|^2.
        let numeric: f64 =
            (refine::data_cost(&problem, &plus) - refine::data_cost(&problem, &minus)) / (4.0 * h);
        let analytic: f64 = normal.g_y[i];
        assert!(
            (analytic - numeric).abs() <= 1e-5 * analytic.abs().max(1.0),
            "parameter {i}: analytic {analytic:e} numeric {numeric:e}"
        );
    }
}
