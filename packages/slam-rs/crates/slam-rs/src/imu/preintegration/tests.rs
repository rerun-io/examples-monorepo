#![allow(clippy::unwrap_used)]

use super::*;
use crate::lie::Se3;
use crate::types::PoseVelBiasState;
use approx::assert_abs_diff_eq;
use nalgebra::{SMatrix, SVector};
use proptest::prelude::*;

// Test support uses a smooth analytic trajectory with closed-form pose, velocity,
// acceleration and body angular velocity, plus a seeded xorshift generator.
// This keeps tests deterministic without a spline dependency.

/// xorshift64*, so every ported test runs the same numbers each time.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x: u64 = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform value in `[-1, 1]`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// Uniform on `[low, high]`.
    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (self.uniform() + 1.0) * 0.5 * (high - low)
    }

    /// Three independent uniform values in `[-1, 1]`.
    fn vector3(&mut self) -> Vector3<f64> {
        Vector3::new(self.uniform(), self.uniform(), self.uniform())
    }

    /// `Sophus::Vector6d::Random()`.
    fn vector6(&mut self) -> SVector<f64, 6> {
        SVector::<f64, 6>::from_fn(|_, _| self.uniform())
    }
}

/// Smooth analytic trajectory: sinusoidal position gives exact velocity and
/// acceleration. For `R = exp(phi(t))`, body angular velocity is `J_r(phi) phi_dot`.
/// Low frequencies keep midpoint integration error bounded over the 20-second test.
struct Trajectory {
    pos_amp: Vector3<f64>,
    pos_freq: Vector3<f64>,
    pos_phase: Vector3<f64>,
    rot_amp: Vector3<f64>,
    rot_freq: Vector3<f64>,
    rot_phase: Vector3<f64>,
}

impl Trajectory {
    fn new(rng: &mut Rng) -> Self {
        Self {
            pos_amp: Vector3::from_fn(|_, _| rng.range(0.8, 1.5)),
            pos_freq: Vector3::from_fn(|_, _| rng.range(0.15, 0.35)),
            pos_phase: Vector3::from_fn(|_, _| rng.range(0.0, std::f64::consts::TAU)),
            rot_amp: Vector3::from_fn(|_, _| rng.range(0.15, 0.30)),
            rot_freq: Vector3::from_fn(|_, _| rng.range(0.15, 0.30)),
            rot_phase: Vector3::from_fn(|_, _| rng.range(0.0, std::f64::consts::TAU)),
        }
    }

    fn seconds(t_ns: i64) -> f64 {
        t_ns as f64 * 1e-9
    }

    fn rotation_vector(&self, t: f64) -> Vector3<f64> {
        Vector3::from_fn(|i, _| self.rot_amp[i] * (self.rot_freq[i] * t + self.rot_phase[i]).sin())
    }

    fn rotation_vector_dot(&self, t: f64) -> Vector3<f64> {
        Vector3::from_fn(|i, _| {
            self.rot_amp[i] * self.rot_freq[i] * (self.rot_freq[i] * t + self.rot_phase[i]).cos()
        })
    }

    fn pose(&self, t_ns: i64) -> Se3<f64> {
        let t: f64 = Self::seconds(t_ns);
        Se3::new(
            So3::exp(&self.rotation_vector(t)),
            Vector3::from_fn(|i, _| {
                self.pos_amp[i] * (self.pos_freq[i] * t + self.pos_phase[i]).sin()
            }),
        )
    }

    fn trans_vel_world(&self, t_ns: i64) -> Vector3<f64> {
        let t: f64 = Self::seconds(t_ns);
        Vector3::from_fn(|i, _| {
            self.pos_amp[i] * self.pos_freq[i] * (self.pos_freq[i] * t + self.pos_phase[i]).cos()
        })
    }

    fn trans_accel_world(&self, t_ns: i64) -> Vector3<f64> {
        let t: f64 = Self::seconds(t_ns);
        Vector3::from_fn(|i, _| {
            -self.pos_amp[i]
                * self.pos_freq[i]
                * self.pos_freq[i]
                * (self.pos_freq[i] * t + self.pos_phase[i]).sin()
        })
    }

    /// `omega_body = J_r(phi) phi_dot` for `R(t) = exp(phi(t))`.
    fn rot_vel_body(&self, t_ns: i64) -> Vector3<f64> {
        let t: f64 = Self::seconds(t_ns);
        right_jacobian_so3(&self.rotation_vector(t)) * self.rotation_vector_dot(t)
    }

    /// Body-frame specific force and angular velocity sampled at the interval midpoint.
    fn sample(&self, t_ns: i64, dt_ns: i64) -> ImuSample {
        let pose: Se3<f64> = self.pose(t_ns);
        ImuSample {
            t_ns: t_ns + dt_ns / 2,
            gyro: self.rot_vel_body(t_ns),
            accel: pose.rotation.inverse() * (self.trans_accel_world(t_ns) - gravity::<f64>()),
        }
    }
}

/// Central differences at zero with epsilon `1e-8` and norm tolerance `1e-3`.
/// Zero comparison uses the absolute norm; relative comparison scales by the
/// smaller input norm.
fn test_jacobian<const R: usize, const C: usize>(
    name: &str,
    ja: &SMatrix<f64, R, C>,
    func: impl Fn(&SVector<f64, C>) -> SVector<f64, R>,
    eps: f64,
    max_norm: f64,
) {
    let mut jn: SMatrix<f64, R, C> = SMatrix::zeros();
    for i in 0..C {
        let mut inc: SVector<f64, C> = SVector::zeros();
        inc[i] = eps;
        let fpe: SVector<f64, R> = func(&inc);
        let fme: SVector<f64, R> = func(&(-inc));
        jn.set_column(i, &((fpe - fme) / (2.0 * eps)));
    }

    assert!(
        ja.iter().all(|v: &f64| v.is_finite()),
        "{name}: Ja not finite\n{ja}"
    );
    assert!(
        jn.iter().all(|v: &f64| v.is_finite()),
        "{name}: Jn not finite\n{jn}"
    );

    let difference: f64 = (jn - ja).norm();
    if jn.norm() <= max_norm && ja.norm() <= max_norm {
        assert!(
            difference <= max_norm,
            "{name}: Ja not equal to Jn (diff norm {difference})\nJa\n{ja}\nJn\n{jn}"
        );
    } else {
        let bound: f64 = max_norm * jn.norm().min(ja.norm());
        assert!(
            difference <= bound,
            "{name}: Ja not equal to Jn (diff norm {difference} > {bound})\nJa\n{ja}\nJn\n{jn}"
        );
    }
}

/// `TestConstants<double>::epsilon`.
const DEFAULT_EPS: f64 = 1e-8;
/// `TestConstants<double>::max_norm`.
const DEFAULT_MAX_NORM: f64 = 1e-3;
const ACCEL_STD_DEV: f64 = 0.23;
const GYRO_STD_DEV: f64 = 0.0027;

fn noise_from_std_dev() -> ImuNoise<f64> {
    ImuNoise {
        accel_cov: Vector3::repeat(ACCEL_STD_DEV * ACCEL_STD_DEV),
        gyro_cov: Vector3::repeat(GYRO_STD_DEV * GYRO_STD_DEV),
    }
}

/// Relative matrix comparison scaled by the smaller input norm.
fn is_approx(a: &Vector3<f64>, b: &Vector3<f64>, precision: f64) -> bool {
    (a - b).norm() <= precision * a.norm().min(b.norm())
}

// Preintegration invariants include the first covariance step, symmetric positive
// semidefiniteness and the whitening identity. These directly check covariance
// behavior without a large Monte Carlo run.

/// `ImuPreintegrationTestCase.PredictTestGT`.
///
/// 2 000 samples over 20 s, then `predictState` from the true state at 0 has
/// to land on the true state at the end: velocity and translation within a
/// relative `1e-4`, orientation within `1e-6` rad.
#[test]
fn predict_state_reaches_the_ground_truth_state() {
    let mut rng: Rng = Rng::new(0x5eed_0001);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let mut meas: IntegratedImuMeasurement<f64> =
        IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());

    let state0: PoseVelState<f64> =
        PoseVelState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));

    let dt_ns: i64 = 10_000_000;
    let ones: Vector3<f64> = Vector3::repeat(1.0);
    let mut t_ns: i64 = dt_ns / 2;
    while t_ns < 20_000_000_000 {
        meas.integrate(&trajectory.sample(t_ns, dt_ns), &ones, &ones)
            .unwrap();
        t_ns += dt_ns;
    }

    let end_t_ns: i64 = meas.get_dt_ns();
    let state1: PoseVelState<f64> = meas.predict_state(&state0, &gravity::<f64>());
    let gt_pose: Se3<f64> = trajectory.pose(end_t_ns);
    let gt_vel: Vector3<f64> = trajectory.trans_vel_world(end_t_ns);

    assert!(
        is_approx(&gt_vel, &state1.vel_w_i, 1e-4),
        "vel_gt {gt_vel} vel {}",
        state1.vel_w_i
    );
    let angular_distance: f64 = gt_pose
        .rotation
        .quaternion()
        .angle_to(state1.t_w_i.rotation.quaternion());
    assert!(
        angular_distance <= 1e-6,
        "angular distance {angular_distance}"
    );
    assert!(
        is_approx(&gt_pose.translation, &state1.t_w_i.translation, 1e-4),
        "p_gt {} p {}",
        gt_pose.translation,
        state1.t_w_i.translation
    );
}

/// Compare transition and noise-input Jacobians with central differences at each step.
#[test]
fn propagate_state_jacobians_match_finite_differences() {
    let mut rng: Rng = Rng::new(0x5eed_0002);
    let trajectory: Trajectory = Trajectory::new(&mut rng);

    let dt_ns: i64 = 10_000_000;
    let mut t_ns: i64 = dt_ns / 2;
    while t_ns < 2_000_000_000 {
        let sample: ImuSample = trajectory.sample(t_ns, dt_ns);
        let curr_t_ns: i64 = t_ns - dt_ns / 2;
        let curr_state: PoseVelState<f64> = PoseVelState::new(
            curr_t_ns,
            trajectory.pose(curr_t_ns),
            trajectory.trans_vel_world(curr_t_ns),
        );
        let accel: Vector3<f64> = sample.accel;
        let gyro: Vector3<f64> = sample.gyro;
        let (next_state, j) = IntegratedImuMeasurement::<f64>::propagate_state(
            &curr_state,
            sample.t_ns,
            &accel,
            &gyro,
        )
        .unwrap();

        test_jacobian(
            "F_TEST",
            &j.d_next_d_curr,
            |x: &Vector9<f64>| {
                let mut perturbed: PoseVelState<f64> = curr_state;
                perturbed.apply_inc(x);
                let (next, _) = IntegratedImuMeasurement::<f64>::propagate_state(
                    &perturbed,
                    sample.t_ns,
                    &accel,
                    &gyro,
                )
                .unwrap();
                next_state.diff(&next)
            },
            DEFAULT_EPS,
            DEFAULT_MAX_NORM,
        );

        test_jacobian(
            "A_TEST",
            &j.d_next_d_accel,
            |x: &Vector3<f64>| {
                let (next, _) = IntegratedImuMeasurement::<f64>::propagate_state(
                    &curr_state,
                    sample.t_ns,
                    &(accel + x),
                    &gyro,
                )
                .unwrap();
                next_state.diff(&next)
            },
            DEFAULT_EPS,
            DEFAULT_MAX_NORM,
        );

        test_jacobian(
            "G_TEST",
            &j.d_next_d_gyro,
            |x: &Vector3<f64>| {
                let (next, _) = IntegratedImuMeasurement::<f64>::propagate_state(
                    &curr_state,
                    sample.t_ns,
                    &accel,
                    &(gyro + x),
                )
                .unwrap();
                next_state.diff(&next)
            },
            1e-8,
            DEFAULT_MAX_NORM,
        );

        t_ns += dt_ns;
    }
}

/// `ImuPreintegrationTestCase.ResidualTest`.
///
/// The residual at the true end state is zero to `1e-6` per coefficient, and
/// the four Jacobians match central differences at a perturbed end state.
#[test]
fn residual_vanishes_at_the_truth_and_its_jacobians_match() {
    let mut rng: Rng = Rng::new(0x5eed_0003);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;

    let mut meas: IntegratedImuMeasurement<f64> = IntegratedImuMeasurement::new(0, &bg, &ba);
    let state0: PoseVelState<f64> =
        PoseVelState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));

    let dt_ns: i64 = 10_000_000;
    let ones: Vector3<f64> = Vector3::repeat(1.0);
    let mut t_ns: i64 = dt_ns / 2;
    while t_ns < 100_000_000 {
        let mut sample: ImuSample = trajectory.sample(t_ns, dt_ns);
        sample.accel += ba;
        sample.gyro += bg;
        meas.integrate(&sample, &ones, &ones).unwrap();
        t_ns += dt_ns;
    }

    let end_t_ns: i64 = meas.get_dt_ns();
    let g: Vector3<f64> = gravity::<f64>();
    let state1_gt: PoseVelState<f64> = PoseVelState::new(
        end_t_ns,
        trajectory.pose(end_t_ns),
        trajectory.trans_vel_world(end_t_ns),
    );
    let res_gt: Vector9<f64> = meas.residual(&state0, &g, &state1_gt, &bg, &ba);
    assert!(res_gt.amax() <= 1e-6, "res_gt {}", res_gt.transpose());

    let state1: PoseVelState<f64> = PoseVelState::new(
        end_t_ns,
        trajectory.pose(end_t_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
        trajectory.trans_vel_world(end_t_ns) + rng.vector3() / 10.0,
    );
    let (_, jacobians) = meas.residual_with_jacobians(&state0, &g, &state1, &bg, &ba);

    test_jacobian(
        "d_res_d_state0",
        &jacobians.d_res_d_state0,
        |x: &Vector9<f64>| {
            let mut perturbed: PoseVelState<f64> = state0;
            perturbed.apply_inc(x);
            meas.residual(&perturbed, &g, &state1, &bg, &ba)
        },
        DEFAULT_EPS,
        DEFAULT_MAX_NORM,
    );
    test_jacobian(
        "d_res_d_state1",
        &jacobians.d_res_d_state1,
        |x: &Vector9<f64>| {
            let mut perturbed: PoseVelState<f64> = state1;
            perturbed.apply_inc(x);
            meas.residual(&state0, &g, &perturbed, &bg, &ba)
        },
        DEFAULT_EPS,
        DEFAULT_MAX_NORM,
    );
    test_jacobian(
        "d_res_d_bg",
        &jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 0).into_owned(),
        |x: &Vector3<f64>| meas.residual(&state0, &g, &state1, &(bg + x), &ba),
        DEFAULT_EPS,
        DEFAULT_MAX_NORM,
    );
    test_jacobian(
        "d_res_d_ba",
        &jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 3).into_owned(),
        |x: &Vector3<f64>| meas.residual(&state0, &g, &state1, &bg, &(ba + x)),
        DEFAULT_EPS,
        DEFAULT_MAX_NORM,
    );
}

/// The samples `BiasTest` and `ResidualBiasTest` share
/// 100 samples over 1 s with both
/// biases added in.
fn biased_samples(trajectory: &Trajectory, bg: &Vector3<f64>, ba: &Vector3<f64>) -> Vec<ImuSample> {
    let dt_ns: i64 = 10_000_000;
    let mut samples: Vec<ImuSample> = Vec::new();
    let mut t_ns: i64 = dt_ns / 2;
    while t_ns < 1_000_000_000 {
        let mut sample: ImuSample = trajectory.sample(t_ns, dt_ns);
        sample.accel += ba;
        sample.gyro += bg;
        samples.push(sample);
        t_ns += dt_ns;
    }
    samples
}

fn integrate_all(
    start_t_ns: i64,
    bg: &Vector3<f64>,
    ba: &Vector3<f64>,
    samples: &[ImuSample],
) -> IntegratedImuMeasurement<f64> {
    let mut meas: IntegratedImuMeasurement<f64> = IntegratedImuMeasurement::new(start_t_ns, bg, ba);
    let ones: Vector3<f64> = Vector3::repeat(1.0);
    for sample in samples {
        meas.integrate(sample, &ones, &ones).unwrap();
    }
    meas
}

/// `ImuPreintegrationTestCase.BiasTest`.
///
/// The two bias Jacobians against re-integrating the same samples about a
/// perturbed linearization point, compared through `PoseVelState::diff`.
#[test]
fn bias_jacobians_match_reintegration_at_a_perturbed_bias() {
    let mut rng: Rng = Rng::new(0x5eed_0004);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);

    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
    let delta_state: PoseVelState<f64> = *meas.get_delta_state();

    test_jacobian(
        "d_state_d_bg",
        meas.get_d_state_d_bg(),
        |x: &Vector3<f64>| {
            let perturbed: IntegratedImuMeasurement<f64> =
                integrate_all(0, &(bg + x), &ba, &samples);
            delta_state.diff(perturbed.get_delta_state())
        },
        DEFAULT_EPS,
        DEFAULT_MAX_NORM,
    );
    test_jacobian(
        "d_state_d_ba",
        meas.get_d_state_d_ba(),
        |x: &Vector3<f64>| {
            let perturbed: IntegratedImuMeasurement<f64> =
                integrate_all(0, &bg, &(ba + x), &samples);
            delta_state.diff(perturbed.get_delta_state())
        },
        DEFAULT_EPS,
        DEFAULT_MAX_NORM,
    );
}

/// Bias correction must agree with reintegration about the shifted bias within
/// relative `1e-4`. Compare both bias Jacobians through reintegration, with the
/// gyro Jacobian's norm tolerance `1e-2`.
#[test]
fn residual_bias_correction_agrees_with_reintegration() {
    let mut rng: Rng = Rng::new(0x5eed_0005);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);

    let g: Vector3<f64> = gravity::<f64>();
    let state0: PoseVelState<f64> =
        PoseVelState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));
    let end_t_ns: i64 = meas.get_dt_ns();
    let state1: PoseVelState<f64> = PoseVelState::new(
        end_t_ns,
        trajectory.pose(end_t_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
        trajectory.trans_vel_world(end_t_ns) + rng.vector3() / 10.0,
    );

    let bg_test: Vector3<f64> = bg + rng.vector3() / 1000.0;
    let ba_test: Vector3<f64> = ba + rng.vector3() / 100.0;
    let (res, jacobians) = meas.residual_with_jacobians(&state0, &g, &state1, &bg_test, &ba_test);

    let reintegrated: IntegratedImuMeasurement<f64> =
        integrate_all(0, &bg_test, &ba_test, &samples);
    let res1: Vector9<f64> = reintegrated.residual(&state0, &g, &state1, &bg_test, &ba_test);
    assert!(
        (res - res1).norm() <= 1e-4 * res.norm().min(res1.norm()),
        "res {}\nres1 {}",
        res.transpose(),
        res1.transpose()
    );

    test_jacobian(
        "d_res_d_ba",
        &jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 3).into_owned(),
        |x: &Vector3<f64>| {
            let perturbed: IntegratedImuMeasurement<f64> =
                integrate_all(0, &bg_test, &(ba_test + x), &samples);
            perturbed.residual(&state0, &g, &state1, &bg_test, &(ba_test + x))
        },
        DEFAULT_EPS,
        DEFAULT_MAX_NORM,
    );
    test_jacobian(
        "d_res_d_bg",
        &jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 0).into_owned(),
        |x: &Vector3<f64>| {
            let perturbed: IntegratedImuMeasurement<f64> =
                integrate_all(0, &(bg_test + x), &ba_test, &samples);
            perturbed.residual(&state0, &g, &state1, &(bg_test + x), &ba_test)
        },
        1e-8,
        1e-2,
    );
}

// ── Port-specific tests ─────────────────────────────────────────────

/// Accelerometer bias cannot move delta rotation: its input Jacobian has zero
/// rotation rows and the transition's rotation rows are `[0 I 0]`.
#[test]
fn the_accel_bias_jacobian_never_touches_rotation() {
    let mut rng: Rng = Rng::new(0x5eed_0008);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
    assert_eq!(
        meas.get_d_state_d_ba()
            .fixed_view::<3, 3>(3, 0)
            .into_owned(),
        Matrix3::zeros()
    );
}

/// The empty measurement is exactly zero everywhere, and its square-root
/// inverse covariance is zero rather than infinite — the pseudo-inverse
/// branch.
#[test]
fn an_empty_measurement_has_a_zero_pseudo_inverse() {
    let meas: IntegratedImuMeasurement<f64> = IntegratedImuMeasurement::default();
    assert_eq!(meas.get_dt_ns(), 0);
    assert_eq!(*meas.get_cov(), Matrix9::zeros());
    assert_eq!(meas.get_cov_inv_sqrt(), Matrix9::zeros());
    assert_eq!(meas.get_cov_inv(), Matrix9::zeros());
}

/// `F`, `A` and `G` for a rig whose gyroscope reads exactly zero, written
/// out here from Paper 1 Eq. (13) instead of being taken from
/// [`IntegratedImuMeasurement::propagate_state`].
///
/// With no rotation the delta rotation stays the identity, so `accel_world`
/// is the measurement itself and `rightJacobianSO3(0)` is the identity: the
/// three Jacobians are the *same* at every step, which turns the covariance
/// recurrence into a closed-form sum. Being written twice is the point —
/// a check that rebuilds the expected covariance out of the implementation's
/// own `F` cannot see a wrong `F`.
fn constant_jacobians(
    dt: f64,
    accel: &Vector3<f64>,
) -> (Matrix9<f64>, Matrix9x3<f64>, Matrix9x3<f64>) {
    let hat: Matrix3<f64> = So3::hat(&(-accel * dt));

    let mut f: Matrix9<f64> = Matrix9::identity();
    f.fixed_view_mut::<3, 3>(0, 6)
        .copy_from(&(Matrix3::identity() * dt));
    f.fixed_view_mut::<3, 3>(6, 3).copy_from(&hat);
    f.fixed_view_mut::<3, 3>(0, 3).copy_from(&(hat * dt * 0.5));

    let mut a: Matrix9x3<f64> = Matrix9x3::zeros();
    a.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(Matrix3::identity() * 0.5 * dt * dt));
    a.fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&(Matrix3::identity() * dt));

    let mut g: Matrix9x3<f64> = Matrix9x3::zeros();
    g.fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(Matrix3::identity() * dt));
    let d_vel_d_gyro: Matrix3<f64> = hat * 0.5 * dt;
    g.fixed_view_mut::<3, 3>(6, 0).copy_from(&d_vel_d_gyro);
    g.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(d_vel_d_gyro * 0.5 * dt));

    (f, a, g)
}

/// The relative Frobenius distance between two matrices.
fn relative_distance<const R: usize, const C: usize>(
    got: &SMatrix<f64, R, C>,
    want: &SMatrix<f64, R, C>,
) -> f64 {
    let scale: f64 = want.norm().max(f64::MIN_POSITIVE);
    (got - want).norm() / scale
}

proptest! {
    /// The covariance and both bias Jacobians against the closed forms of
    /// their recurrences, evaluated with an independently written `F`, `A`
    /// and `G`.
    ///
    /// `cov_n = Σ_{k<n} F^k Q (F^k)ᵀ` with `Q = A Σa Aᵀ + G Σg Gᵀ`
    /// `d_state_d_ba_n = -Σ_{k<n} F^k A` and
    /// `d_state_d_bg_n = -Σ_{k<n} F^k G`. One step, thirty
    /// steps and sub-millisecond intervals all fall out of the ranges.
    #[test]
    fn the_covariance_recurrence_matches_its_closed_form(
        steps in 1usize..30,
        dt_ns in 50_000i64..20_000_000,
        ax in -12.0f64..12.0,
        ay in -12.0f64..12.0,
        az in -12.0f64..12.0,
    ) {
        let noise: ImuNoise<f64> = noise_from_std_dev();
        let accel: Vector3<f64> = Vector3::new(ax, ay, az);
        let dt: f64 = dt_ns as f64 * 1e-9;

        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
        for step in 1..=steps as i64 {
            meas.integrate(
                &ImuSample { t_ns: step * dt_ns, gyro: Vector3::zeros(), accel },
                &noise.accel_cov,
                &noise.gyro_cov,
            )?;
        }

        let (f, a, g) = constant_jacobians(dt, &accel);
        let q: Matrix9<f64> = a * Matrix3::from_diagonal(&noise.accel_cov) * a.transpose()
            + g * Matrix3::from_diagonal(&noise.gyro_cov) * g.transpose();

        let mut power: Matrix9<f64> = Matrix9::identity();
        let mut cov: Matrix9<f64> = Matrix9::zeros();
        let mut d_ba: Matrix9x3<f64> = Matrix9x3::zeros();
        let mut d_bg: Matrix9x3<f64> = Matrix9x3::zeros();
        for _ in 0..steps {
            cov += power * q * power.transpose();
            d_ba -= power * a;
            d_bg -= power * g;
            power *= f;
        }

        prop_assert!(
            relative_distance(meas.get_cov(), &cov) <= 1e-9,
            "cov off by {} relative",
            relative_distance(meas.get_cov(), &cov)
        );
        prop_assert!(
            relative_distance(meas.get_d_state_d_ba(), &d_ba) <= 1e-12,
            "d_state_d_ba off by {} relative",
            relative_distance(meas.get_d_state_d_ba(), &d_ba)
        );
        prop_assert!(
            relative_distance(meas.get_d_state_d_bg(), &d_bg) <= 1e-12,
            "d_state_d_bg off by {} relative",
            relative_distance(meas.get_d_state_d_bg(), &d_bg)
        );
    }
}

/// The same closed form in `f32`, at one fixed configuration.
///
/// The recurrence and the explicit sum are different orders of the same
/// arithmetic, so the tolerance is the `f32` accumulation of 20 steps, not
/// the agreement of two exact quantities.
#[test]
fn the_covariance_recurrence_matches_its_closed_form_in_float() {
    let accel: Vector3<f64> = Vector3::new(0.35, -1.25, 9.75);
    let dt_ns: i64 = 2_500_000;
    let steps: i64 = 20;
    let noise: ImuNoise<f32> = ImuNoise {
        accel_cov: Vector3::repeat((ACCEL_STD_DEV * ACCEL_STD_DEV) as f32),
        gyro_cov: Vector3::repeat((GYRO_STD_DEV * GYRO_STD_DEV) as f32),
    };

    let mut meas: IntegratedImuMeasurement<f32> =
        IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
    for step in 1..=steps {
        meas.integrate(
            &ImuSample {
                t_ns: step * dt_ns,
                gyro: Vector3::zeros(),
                accel,
            },
            &noise.accel_cov,
            &noise.gyro_cov,
        )
        .unwrap();
    }

    let dt: f64 = dt_ns as f64 * 1e-9;
    let (f, a, g) = constant_jacobians(dt, &accel);
    let accel_cov: Vector3<f64> = Vector3::repeat(f64::from(noise.accel_cov.x));
    let gyro_cov: Vector3<f64> = Vector3::repeat(f64::from(noise.gyro_cov.x));
    let q: Matrix9<f64> = a * Matrix3::from_diagonal(&accel_cov) * a.transpose()
        + g * Matrix3::from_diagonal(&gyro_cov) * g.transpose();

    let mut power: Matrix9<f64> = Matrix9::identity();
    let mut cov: Matrix9<f64> = Matrix9::zeros();
    for _ in 0..steps {
        cov += power * q * power.transpose();
        power *= f;
    }

    let got: Matrix9<f64> = meas.get_cov().map(f64::from);
    assert!(
        relative_distance(&got, &cov) <= 1e-5,
        "cov off by {} relative",
        relative_distance(&got, &cov)
    );
}

/// A duplicate or reordered sample is rejected and nothing is integrated.
#[test]
fn duplicate_and_reordered_samples_are_rejected() {
    let ones: Vector3<f64> = Vector3::repeat(1.0);
    let mut meas: IntegratedImuMeasurement<f64> =
        IntegratedImuMeasurement::new(1_000, &Vector3::zeros(), &Vector3::zeros());
    let sample = |t_ns: i64| ImuSample {
        t_ns,
        gyro: Vector3::new(0.01, 0.0, 0.0),
        accel: Vector3::new(0.0, 0.0, 9.81),
    };

    meas.integrate(&sample(2_000), &ones, &ones).unwrap();
    assert_eq!(meas.get_dt_ns(), 1_000);
    assert_eq!(
        meas.integrate(&sample(2_000), &ones, &ones),
        Err(ImuError::NonMonotonicSample {
            previous_t_ns: 1_000,
            t_ns: 1_000
        })
    );
    assert_eq!(
        meas.integrate(&sample(1_500), &ones, &ones),
        Err(ImuError::NonMonotonicSample {
            previous_t_ns: 1_000,
            t_ns: 500
        })
    );
    // The measurement is unchanged after both refusals.
    assert_eq!(meas.get_dt_ns(), 1_000);

    // A sample at exactly the start time would be a zero-length step.
    let mut fresh: IntegratedImuMeasurement<f64> =
        IntegratedImuMeasurement::new(1_000, &Vector3::zeros(), &Vector3::zeros());
    assert_eq!(
        fresh.integrate(&sample(1_000), &ones, &ones),
        Err(ImuError::NonMonotonicSample {
            previous_t_ns: 0,
            t_ns: 0
        })
    );
}

/// The five ways the shared accumulation loop refuses a malformed interval,
/// and the closing step that must still happen when a sample does follow.
///
/// `integrate_between` carried these cases and RC9 deleted it with its
/// tests, leaving the two live producers driving a loop that accepted an
/// empty interval, an interval starting somewhere else, and an interval it
/// could not close — the last as `Ok` with a measurement shorter than the
/// frame gap. Both producers precheck a sample strictly after the frame
/// ( through `imu_covers_frame`, and
/// `Vio::track`'s own coverage test), so none of this can fire on the
/// shipped path; a public method promising to close the interval exactly
/// must say so anyway (D32).
#[test]
fn accumulate_to_rejects_bad_intervals() {
    let noise: ImuNoise<f64> = noise_from_std_dev();
    let sample =
        |t_ns: i64| -> Popped<f64> { (t_ns, Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81)) };
    let meas = || IntegratedImuMeasurement::<f64>::new(0, &Vector3::zeros(), &Vector3::zeros());
    // The queue both producers pop from, as a closure over a list.
    let feed = |samples: Vec<Popped<f64>>| {
        let mut samples = samples.into_iter();
        move || samples.next()
    };

    // An empty interval: asserts it, because a zero time delta
    // "leads to invalid IMU integration".
    assert_eq!(
        meas().accumulate_to(None, feed(vec![sample(1)]), 0, 0, &noise),
        Err(ImuError::NonMonotonicFrames { t0_ns: 0, t1_ns: 0 })
    );
    // An interval that does not start where the measurement was built:
    // every sample would be timed against the wrong origin.
    assert_eq!(
        meas().accumulate_to(None, feed(vec![sample(1)]), 5, 10, &noise),
        Err(ImuError::StartTimeMismatch {
            start_t_ns: 0,
            t0_ns: 5
        })
    );
    // A duplicate and a reordered sample, both from the per-sample step.
    assert_eq!(
        meas().accumulate_to(None, feed(vec![sample(2), sample(2)]), 0, 10, &noise),
        Err(ImuError::NonMonotonicSample {
            previous_t_ns: 2,
            t_ns: 2
        })
    );
    assert_eq!(
        meas().accumulate_to(None, feed(vec![sample(3), sample(1)]), 0, 10, &noise),
        Err(ImuError::NonMonotonicSample {
            previous_t_ns: 3,
            t_ns: 1
        })
    );
    // Nothing after the frame to close the interval with.
    assert_eq!(
        meas().accumulate_to(None, feed(vec![sample(2), sample(4)]), 0, 10, &noise),
        Err(ImuError::MissingSampleAfterFrame { t1_ns: 10 })
    );

    // The sample that does follow closes the interval exactly on the frame
    // and comes back out at its own time.
    let mut closed: IntegratedImuMeasurement<f64> = meas();
    let pending: Option<Popped<f64>> = closed
        .accumulate_to(
            None,
            feed(vec![sample(2), sample(4), sample(12)]),
            0,
            10,
            &noise,
        )
        .unwrap();
    assert_eq!(pending.map(|(t_ns, _, _)| t_ns), Some(12));
    assert_eq!(closed.get_dt_ns(), 10);
}

/// `Quaternion::FromTwoVectors(accel, UnitZ)`
/// rotates the measured specific force onto the world `+Z` axis, so gravity
/// lands along `-Z`.
#[test]
fn gravity_init_aligns_the_accelerometer_with_world_up() {
    let mut rng: Rng = Rng::new(0x5eed_0009);
    for _ in 0..64 {
        let accel: Vector3<f64> = rng.vector3() * 9.81;
        if accel.norm() < 1e-3 {
            continue;
        }
        let rotation: So3<f64> = gravity_from_first_accel(&accel);
        let up: Vector3<f64> = rotation * (accel / accel.norm());
        assert_abs_diff_eq!(up, Vector3::new(0.0, 0.0, 1.0), epsilon = 1e-12);
        // The rig-frame gravity is minus the measured specific force.
        let g_body: Vector3<f64> = rotation.inverse() * gravity::<f64>();
        assert_abs_diff_eq!(g_body.normalize(), -accel.normalize(), epsilon = 1e-12);
    }
}

/// Both degenerate directions: a sample already along `+Z` gives the
/// identity, and the anti-parallel sample still lands on `+Z`.
#[test]
fn gravity_init_handles_the_degenerate_directions() {
    let up: So3<f64> = gravity_from_first_accel(&Vector3::new(0.0, 0.0, 9.81));
    assert_abs_diff_eq!(up.log(), Vector3::zeros(), epsilon = 1e-12);

    let down: So3<f64> = gravity_from_first_accel(&Vector3::new(0.0, 0.0, -9.81));
    assert_abs_diff_eq!(
        down * Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 0.0, 1.0),
        epsilon = 1e-12
    );

    // Nothing to align: the identity, not a NaN.
    assert_eq!(
        gravity_from_first_accel(&Vector3::<f64>::zeros()),
        So3::identity()
    );
    assert_eq!(
        gravity_from_first_accel(&Vector3::new(f64::NAN, 0.0, 0.0)),
        So3::identity()
    );
}

/// `gravity_from_first_accel` recovers the orientation of a rig at rest for
/// any roll and pitch, up to the yaw it cannot see.
#[test]
fn gravity_init_recovers_roll_and_pitch() {
    let mut rng: Rng = Rng::new(0x5eed_000b);
    let g: Vector3<f64> = gravity::<f64>();
    for _ in 0..64 {
        let truth: So3<f64> = So3::exp(&(rng.vector3() * 1.2));
        // What a rig at rest measures: minus gravity, in the rig frame.
        let accel: Vector3<f64> = truth.inverse() * (-g);
        let estimate: So3<f64> = gravity_from_first_accel(&accel);
        // Both map the measured direction onto world up, so the two differ
        // by a rotation about the world z axis: yaw only.
        let residual: Vector3<f64> = (estimate * truth.inverse()).log();
        assert_abs_diff_eq!(residual.x, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(residual.y, 0.0, epsilon = 1e-9);
    }
}

/// The same 100-sample integration in `f32` and `f64` agrees to `1e-4`
/// (decision D05: the measurement is generic and both are instantiated).
#[test]
fn f32_agrees_with_f64_over_a_hundred_samples() {
    let mut rng: Rng = Rng::new(0x5eed_000a);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
    assert_eq!(samples.len(), 100);

    let wide: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);

    let mut narrow: IntegratedImuMeasurement<f32> = IntegratedImuMeasurement::new(
        0,
        &Vector3::new(bg.x as f32, bg.y as f32, bg.z as f32),
        &Vector3::new(ba.x as f32, ba.y as f32, ba.z as f32),
    );
    let ones: Vector3<f32> = Vector3::repeat(1.0);
    for sample in &samples {
        narrow.integrate(sample, &ones, &ones).unwrap();
    }

    assert_eq!(narrow.get_dt_ns(), wide.get_dt_ns());
    let wide_delta: &PoseVelState<f64> = wide.get_delta_state();
    let narrow_delta: &PoseVelState<f32> = narrow.get_delta_state();
    for axis in 0..3 {
        assert!(
            (f64::from(narrow_delta.t_w_i.translation[axis]) - wide_delta.t_w_i.translation[axis])
                .abs()
                <= 1e-4
        );
        assert!((f64::from(narrow_delta.vel_w_i[axis]) - wide_delta.vel_w_i[axis]).abs() <= 1e-4);
    }
    let narrow_log: Vector3<f32> = narrow_delta.t_w_i.rotation.log();
    let wide_log: Vector3<f64> = wide_delta.t_w_i.rotation.log();
    for axis in 0..3 {
        assert!((f64::from(narrow_log[axis]) - wide_log[axis]).abs() <= 1e-4);
    }
}

/// The 9x6 bias Jacobian carries the gyro block in columns 0-2 and the accel
/// block in 3-5, which is what the estimator's `start_idx + 9` and
/// `start_idx + 12` column offsets mean.
#[test]
fn the_bias_jacobian_columns_are_gyro_then_accel() {
    let mut rng: Rng = Rng::new(0x5eed_000c);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);

    let g: Vector3<f64> = gravity::<f64>();
    let state0: PoseVelState<f64> =
        PoseVelState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));
    let end_t_ns: i64 = meas.get_dt_ns();
    let state1: PoseVelState<f64> = PoseVelState::new(
        end_t_ns,
        trajectory.pose(end_t_ns),
        trajectory.trans_vel_world(end_t_ns),
    );
    let (res, jacobians) = meas.residual_with_jacobians(&state0, &g, &state1, &bg, &ba);

    // The accel half is exactly `-d_state_d_ba`.
    assert_eq!(
        jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 3).into_owned(),
        -meas.get_d_state_d_ba()
    );
    // The gyro half is `-d_state_d_bg` with its rotation rows replaced by
    // `+ leftJacobianInv(res_rot) * d_state_d_bg(3,0)`.
    let gyro_half: Matrix9x3<f64> = jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 0).into_owned();
    assert_eq!(
        gyro_half.fixed_view::<3, 3>(0, 0).into_owned(),
        -meas.get_d_state_d_bg().fixed_view::<3, 3>(0, 0)
    );
    assert_eq!(
        gyro_half.fixed_view::<3, 3>(6, 0).into_owned(),
        -meas.get_d_state_d_bg().fixed_view::<3, 3>(6, 0)
    );
    let res_rot: Vector3<f64> = res.fixed_rows::<3>(3).into_owned();
    assert_abs_diff_eq!(
        gyro_half.fixed_view::<3, 3>(3, 0).into_owned(),
        left_jacobian_inv_so3(&res_rot) * meas.get_d_state_d_bg().fixed_view::<3, 3>(3, 0),
        epsilon = 1e-15
    );
}

// ── Properties ──────────────────────────────────────────────────────

fn zero_motion_measurement(
    steps: usize,
    dt_ns: i64,
    accel: Vector3<f64>,
) -> IntegratedImuMeasurement<f64> {
    let noise: ImuNoise<f64> = noise_from_std_dev();
    let mut meas: IntegratedImuMeasurement<f64> =
        IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
    for step in 1..=steps as i64 {
        meas.integrate(
            &ImuSample {
                t_ns: step * dt_ns,
                gyro: Vector3::zeros(),
                accel,
            },
            &noise.accel_cov,
            &noise.gyro_cov,
        )
        .unwrap();
    }
    meas
}

proptest! {
    /// Gravity-only accelerometer, zero gyro, zero bias: the delta state has
    /// no rotation, its velocity is `-g T` and its position `-½ g T²`, so
    /// `predict_state` leaves a rig that started at rest exactly where it
    /// was, and the residual against that prediction vanishes.
    #[test]
    fn a_rig_at_rest_stays_at_rest(steps in 1usize..40, dt_ns in 1_000_000i64..20_000_000) {
        let g: Vector3<f64> = gravity::<f64>();
        let meas: IntegratedImuMeasurement<f64> = zero_motion_measurement(steps, dt_ns, -g);
        let total: f64 = meas.get_dt_ns() as f64 * 1e-9;

        let delta: &PoseVelState<f64> = meas.get_delta_state();
        prop_assert!(delta.t_w_i.rotation.log().norm() <= 1e-15);
        prop_assert!((delta.vel_w_i - (-g * total)).norm() <= 1e-9);
        prop_assert!((delta.t_w_i.translation - (-g * 0.5 * total * total)).norm() <= 1e-9);

        let state0: PoseVelState<f64> = PoseVelState::default();
        let state1: PoseVelState<f64> = meas.predict_state(&state0, &g);
        prop_assert!(state1.t_w_i.translation.norm() <= 1e-9);
        prop_assert!(state1.vel_w_i.norm() <= 1e-9);
        prop_assert!(state1.t_w_i.rotation.log().norm() <= 1e-15);
        prop_assert_eq!(state1.t_ns, meas.get_dt_ns());

        let res = meas.residual(&state0, &g, &state1, &Vector3::zeros(), &Vector3::zeros());
        prop_assert!(res.amax() <= 1e-9);
    }

    /// A free-falling rig — zero specific force — accumulates nothing, and
    /// the prediction is pure gravity: `v = g T`, `p = ½ g T²`.
    #[test]
    fn free_fall_is_pure_gravity(steps in 1usize..40, dt_ns in 1_000_000i64..20_000_000) {
        let g: Vector3<f64> = gravity::<f64>();
        let meas: IntegratedImuMeasurement<f64> =
            zero_motion_measurement(steps, dt_ns, Vector3::zeros());
        let total: f64 = meas.get_dt_ns() as f64 * 1e-9;

        let delta: &PoseVelState<f64> = meas.get_delta_state();
        prop_assert_eq!(delta.vel_w_i, Vector3::zeros());
        prop_assert_eq!(delta.t_w_i.translation, Vector3::zeros());

        let state1: PoseVelState<f64> = meas.predict_state(&PoseVelState::default(), &g);
        prop_assert!((state1.vel_w_i - g * total).norm() <= 1e-12);
        prop_assert!((state1.t_w_i.translation - g * 0.5 * total * total).norm() <= 1e-12);
    }

    #[test]
    fn random_spd_covariance_is_inverted(values in prop::collection::vec(-1.0f64..1.0, 81)) {
        let g = Matrix9::from_row_slice(&values);
        let a = g.transpose() * g + Matrix9::identity();
        let mut meas = IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
        meas.cov = a;
        let inverse = meas.get_cov_inv();
        let reference = a.try_inverse().unwrap();
        prop_assert!((inverse - reference).norm() < 1e-10 * (1.0 + reference.norm()));
    }

    #[test]
    fn rank_deficient_gram_covariance_has_a_generalized_inverse(
        weights in prop::array::uniform4(0.25f64..4.0), shift in 0usize..9,
    ) {
        // Four independent rows and scaled duplicate columns. This tests
        // oblique null directions without rounding GᵀG into full rank.
        let mut g = Matrix9::zeros();
        for i in 0..4 {
            g[(i, i)] = weights[i];
            g[(i, i + 4)] = 2.0 * weights[i];
        }
        g.swap_columns(0, shift);
        let a = g.transpose() * g;
        let mut meas = IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
        meas.cov = a;
        let inverse = meas.get_cov_inv();
        prop_assert!(inverse.iter().all(|v| v.is_finite()));
        prop_assert!((inverse - inverse.transpose()).norm() < 1e-12 * (1.0 + inverse.norm()));
        prop_assert!((a * inverse * a - a).norm() < 1e-10 * (1.0 + a.norm()));
    }

    /// The covariance stays symmetric and positive semi-definite, and the
    /// square-root inverse really is one: `(MᵀM) cov = I`.
    #[test]
    fn the_covariance_is_symmetric_psd_and_its_factor_inverts_it(
        steps in 2usize..25,
        dt_ns in 1_000_000i64..10_000_000,
        gyro_scale in 0.0f64..1.0,
    ) {
        let noise: ImuNoise<f64> = noise_from_std_dev();
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
        for step in 1..=steps as i64 {
            meas.integrate(
                &ImuSample {
                    t_ns: step * dt_ns,
                    gyro: Vector3::new(0.3, -0.2, 0.5) * gyro_scale,
                    accel: Vector3::new(0.4, -0.3, 9.7),
                },
                &noise.accel_cov,
                &noise.gyro_cov,
            )?;
        }

        let cov: Matrix9<f64> = *meas.get_cov();
        let asymmetry: f64 = (cov - cov.transpose()).amax();
        prop_assert!(asymmetry <= 1e-12 * cov.amax().max(1.0), "asymmetry {}", asymmetry);

        let smallest: f64 = cov.symmetric_eigenvalues().min();
        prop_assert!(smallest >= -1e-9 * cov.amax(), "smallest eigenvalue {}", smallest);

        let m: Matrix9<f64> = meas.get_cov_inv_sqrt();
        let deviation: f64 = (m.transpose() * m * cov - Matrix9::identity()).amax();
        prop_assert!(deviation <= 1e-6, "MᵀM cov - I = {}", deviation);
    }

    /// A duplicate or backwards timestamp is always refused, whatever the
    /// start time, and the measurement does not move.
    #[test]
    fn out_of_order_samples_are_always_refused(
        start_t_ns in -1_000_000_000i64..1_000_000_000,
        step_ns in 1i64..10_000_000,
        back_ns in 0i64..10_000_000,
    ) {
        let ones: Vector3<f64> = Vector3::repeat(1.0);
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(start_t_ns, &Vector3::zeros(), &Vector3::zeros());
        let sample = |t_ns: i64| ImuSample {
            t_ns,
            gyro: Vector3::new(0.02, 0.0, -0.01),
            accel: Vector3::new(0.0, 0.0, 9.81),
        };
        meas.integrate(&sample(start_t_ns + step_ns), &ones, &ones)?;
        let before: IntegratedImuMeasurement<f64> = meas;

        let result = meas.integrate(&sample(start_t_ns + step_ns - back_ns), &ones, &ones);
        prop_assert_eq!(
            result,
            Err(ImuError::NonMonotonicSample {
                previous_t_ns: step_ns,
                t_ns: step_ns - back_ns
            })
        );
        prop_assert_eq!(meas, before);
    }
}

mod factor;
