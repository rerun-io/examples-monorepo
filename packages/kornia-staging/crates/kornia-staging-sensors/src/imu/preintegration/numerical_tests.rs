//! Propagation, residual, bias, and covariance regression tests.
use super::test_support::{
    biased_samples, integrate_all, Rng, Trajectory, ACCEL_STD_DEV, GRAVITY, GYRO_STD_DEV,
};
use super::{CombinedImuSample, IntegratedImuMeasurement, Matrix9, Matrix9x3, NavState, Vector9};
use approx::assert_abs_diff_eq;
use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use proptest::prelude::*;

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

fn noise_covariances() -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::repeat(ACCEL_STD_DEV * ACCEL_STD_DEV),
        Vector3::repeat(GYRO_STD_DEV * GYRO_STD_DEV),
    )
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

    let state0: NavState<f64> = NavState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));

    let dt_ns: i64 = 10_000_000;
    let ones: Vector3<f64> = Vector3::repeat(1.0);
    let mut timestamp_ns: i64 = dt_ns / 2;
    while timestamp_ns < 20_000_000_000 {
        meas.integrate(
            &trajectory.sample(timestamp_ns, dt_ns),
            &crate::imu::ImuNoise::new(ones, ones).unwrap(),
        )
        .unwrap();
        timestamp_ns += dt_ns;
    }

    let end_t_ns: i64 = meas.dt_ns();
    let state1: NavState<f64> = meas.predict_state(&state0, &GRAVITY);
    let gt_pose: RigidTransform<f64> = trajectory.pose(end_t_ns);
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
    let mut timestamp_ns: i64 = dt_ns / 2;
    while timestamp_ns < 2_000_000_000 {
        let sample: CombinedImuSample = trajectory.sample(timestamp_ns, dt_ns);
        let curr_t_ns: i64 = timestamp_ns - dt_ns / 2;
        let curr_state: NavState<f64> = NavState::new(
            curr_t_ns,
            trajectory.pose(curr_t_ns),
            trajectory.trans_vel_world(curr_t_ns),
        );
        let accel: Vector3<f64> = Vector3::from(sample.accel.to_array());
        let gyro: Vector3<f64> = Vector3::from(sample.gyro.to_array());
        let (next_state, j) = IntegratedImuMeasurement::<f64>::propagate_state(
            &curr_state,
            sample.timestamp_ns,
            &accel,
            &gyro,
        )
        .unwrap();

        test_jacobian(
            "F_TEST",
            &j.d_next_d_curr,
            |x: &Vector9<f64>| {
                let mut perturbed: NavState<f64> = curr_state;
                perturbed.apply_inc(x);
                let (next, _) = IntegratedImuMeasurement::<f64>::propagate_state(
                    &perturbed,
                    sample.timestamp_ns,
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
                    sample.timestamp_ns,
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
                    sample.timestamp_ns,
                    &accel,
                    &(gyro + x),
                )
                .unwrap();
                next_state.diff(&next)
            },
            1e-8,
            DEFAULT_MAX_NORM,
        );

        timestamp_ns += dt_ns;
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
    let state0: NavState<f64> = NavState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));

    let dt_ns: i64 = 10_000_000;
    let ones: Vector3<f64> = Vector3::repeat(1.0);
    let mut timestamp_ns: i64 = dt_ns / 2;
    while timestamp_ns < 100_000_000 {
        let mut sample: CombinedImuSample = trajectory.sample(timestamp_ns, dt_ns);
        let shift = ba;
        sample.accel.x += shift.x;
        sample.accel.y += shift.y;
        sample.accel.z += shift.z;
        let shift = bg;
        sample.gyro.x += shift.x;
        sample.gyro.y += shift.y;
        sample.gyro.z += shift.z;
        meas.integrate(&sample, &crate::imu::ImuNoise::new(ones, ones).unwrap())
            .unwrap();
        timestamp_ns += dt_ns;
    }

    let end_t_ns: i64 = meas.dt_ns();
    let g: Vector3<f64> = GRAVITY;
    let state1_gt: NavState<f64> = NavState::new(
        end_t_ns,
        trajectory.pose(end_t_ns),
        trajectory.trans_vel_world(end_t_ns),
    );
    let res_gt: Vector9<f64> = meas.residual(&state0, &g, &state1_gt, &bg, &ba);
    assert!(res_gt.amax() <= 1e-6, "res_gt {}", res_gt.transpose());

    let state1: NavState<f64> = NavState::new(
        end_t_ns,
        trajectory.pose(end_t_ns) * RigidTransform::exp_decoupled(&(rng.vector6() / 10.0)),
        trajectory.trans_vel_world(end_t_ns) + rng.vector3() / 10.0,
    );
    let (_, jacobians) = meas.residual_with_jacobians(&state0, &g, &state1, &bg, &ba);

    test_jacobian(
        "d_res_d_state0",
        &jacobians.d_res_d_state0,
        |x: &Vector9<f64>| {
            let mut perturbed: NavState<f64> = state0;
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
            let mut perturbed: NavState<f64> = state1;
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

/// `ImuPreintegrationTestCase.BiasTest`.
///
/// The two bias Jacobians against re-integrating the same samples about a
/// perturbed linearization point, compared through `NavState::diff`.
#[test]
fn bias_jacobians_match_reintegration_at_a_perturbed_bias() {
    let mut rng: Rng = Rng::new(0x5eed_0004);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<CombinedImuSample> = biased_samples(&trajectory, &bg, &ba);

    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
    let delta_state: NavState<f64> = *meas.delta_state();

    test_jacobian(
        "d_state_d_bg",
        meas.d_state_d_bias_gyro(),
        |x: &Vector3<f64>| {
            let perturbed: IntegratedImuMeasurement<f64> =
                integrate_all(0, &(bg + x), &ba, &samples);
            delta_state.diff(perturbed.delta_state())
        },
        DEFAULT_EPS,
        DEFAULT_MAX_NORM,
    );
    test_jacobian(
        "d_state_d_ba",
        meas.d_state_d_bias_accel(),
        |x: &Vector3<f64>| {
            let perturbed: IntegratedImuMeasurement<f64> =
                integrate_all(0, &bg, &(ba + x), &samples);
            delta_state.diff(perturbed.delta_state())
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
    let samples: Vec<CombinedImuSample> = biased_samples(&trajectory, &bg, &ba);
    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);

    let g: Vector3<f64> = GRAVITY;
    let state0: NavState<f64> = NavState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));
    let end_t_ns: i64 = meas.dt_ns();
    let state1: NavState<f64> = NavState::new(
        end_t_ns,
        trajectory.pose(end_t_ns) * RigidTransform::exp_decoupled(&(rng.vector6() / 10.0)),
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
    let samples: Vec<CombinedImuSample> = biased_samples(&trajectory, &bg, &ba);
    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
    assert_eq!(
        meas.d_state_d_bias_accel()
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
    assert_eq!(meas.dt_ns(), 0);
    assert_eq!(*meas.cov(), Matrix9::zeros());
    assert_eq!(meas.cov_inv_sqrt(), Matrix9::zeros());
    assert_eq!(meas.cov_inv(), Matrix9::zeros());
}

/// `F`, `A` and `G` for a rig whose gyroscope reads exactly zero, written
/// out analytically instead of being taken from
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
    let hat: Matrix3<f64> = Rotation3::hat(&(-accel * dt));

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
        let (accel_cov, gyro_cov) = noise_covariances();
        let accel: Vector3<f64> = Vector3::new(ax, ay, az);
        let dt: f64 = dt_ns as f64 * 1e-9;

        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
        for step in 1..=steps as i64 {
            meas.integrate(&CombinedImuSample { timestamp_ns: step * dt_ns, gyro: kornia_algebra::Vec3F64::ZERO, accel: kornia_algebra::Vec3F64::from_array(accel.into()) }, &crate::imu::ImuNoise::new(accel_cov, gyro_cov).unwrap())?;
        }

        let (f, a, g) = constant_jacobians(dt, &accel);
        let q: Matrix9<f64> = a * Matrix3::from_diagonal(&accel_cov) * a.transpose()
            + g * Matrix3::from_diagonal(&gyro_cov) * g.transpose();

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
            relative_distance(meas.cov(), &cov) <= 1e-9,
            "cov off by {} relative",
            relative_distance(meas.cov(), &cov)
        );
        prop_assert!(
            relative_distance(meas.d_state_d_bias_accel(), &d_ba) <= 1e-12,
            "d_state_d_ba off by {} relative",
            relative_distance(meas.d_state_d_bias_accel(), &d_ba)
        );
        prop_assert!(
            relative_distance(meas.d_state_d_bias_gyro(), &d_bg) <= 1e-12,
            "d_state_d_bg off by {} relative",
            relative_distance(meas.d_state_d_bias_gyro(), &d_bg)
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
    let accel_cov: Vector3<f32> = Vector3::repeat((ACCEL_STD_DEV * ACCEL_STD_DEV) as f32);
    let gyro_cov: Vector3<f32> = Vector3::repeat((GYRO_STD_DEV * GYRO_STD_DEV) as f32);

    let mut meas: IntegratedImuMeasurement<f32> =
        IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
    for step in 1..=steps {
        meas.integrate(
            &CombinedImuSample {
                timestamp_ns: step * dt_ns,
                gyro: kornia_algebra::Vec3F64::ZERO,
                accel: kornia_algebra::Vec3F64::from_array(accel.into()),
            },
            &crate::imu::ImuNoise::new(accel_cov, gyro_cov).unwrap(),
        )
        .unwrap();
    }

    let dt: f64 = dt_ns as f64 * 1e-9;
    let (f, a, g) = constant_jacobians(dt, &accel);
    let accel_cov: Vector3<f64> = Vector3::repeat(f64::from(accel_cov.x));
    let gyro_cov: Vector3<f64> = Vector3::repeat(f64::from(gyro_cov.x));
    let q: Matrix9<f64> = a * Matrix3::from_diagonal(&accel_cov) * a.transpose()
        + g * Matrix3::from_diagonal(&gyro_cov) * g.transpose();

    let mut power: Matrix9<f64> = Matrix9::identity();
    let mut cov: Matrix9<f64> = Matrix9::zeros();
    for _ in 0..steps {
        cov += power * q * power.transpose();
        power *= f;
    }

    let got: Matrix9<f64> = meas.cov().map(f64::from);
    assert!(
        relative_distance(&got, &cov) <= 1e-5,
        "cov off by {} relative",
        relative_distance(&got, &cov)
    );
}

/// The same 100-sample integration in `f32` and `f64` agrees to `1e-4`
/// (the measurement is generic and both precisions are instantiated).
#[test]
fn f32_agrees_with_f64_over_a_hundred_samples() {
    let mut rng: Rng = Rng::new(0x5eed_000a);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<CombinedImuSample> = biased_samples(&trajectory, &bg, &ba);
    assert_eq!(samples.len(), 100);

    let wide: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);

    let mut narrow: IntegratedImuMeasurement<f32> = IntegratedImuMeasurement::new(
        0,
        &Vector3::new(bg.x as f32, bg.y as f32, bg.z as f32),
        &Vector3::new(ba.x as f32, ba.y as f32, ba.z as f32),
    );
    let ones: Vector3<f32> = Vector3::repeat(1.0);
    for sample in &samples {
        narrow
            .integrate(sample, &crate::imu::ImuNoise::new(ones, ones).unwrap())
            .unwrap();
    }

    assert_eq!(narrow.dt_ns(), wide.dt_ns());
    let wide_delta: &NavState<f64> = wide.delta_state();
    let narrow_delta: &NavState<f32> = narrow.delta_state();
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
    let samples: Vec<CombinedImuSample> = biased_samples(&trajectory, &bg, &ba);
    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);

    let g: Vector3<f64> = GRAVITY;
    let state0: NavState<f64> = NavState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0));
    let end_t_ns: i64 = meas.dt_ns();
    let state1: NavState<f64> = NavState::new(
        end_t_ns,
        trajectory.pose(end_t_ns),
        trajectory.trans_vel_world(end_t_ns),
    );
    let (res, jacobians) = meas.residual_with_jacobians(&state0, &g, &state1, &bg, &ba);

    // The accel half is exactly `-d_state_d_ba`.
    assert_eq!(
        jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 3).into_owned(),
        -meas.d_state_d_bias_accel()
    );
    // The gyro half is `-d_state_d_bg` with its rotation rows replaced by
    // `+ leftJacobianInv(res_rot) * d_state_d_bg(3,0)`.
    let gyro_half: Matrix9x3<f64> = jacobians.d_res_d_bias.fixed_view::<9, 3>(0, 0).into_owned();
    assert_eq!(
        gyro_half.fixed_view::<3, 3>(0, 0).into_owned(),
        -meas.d_state_d_bias_gyro().fixed_view::<3, 3>(0, 0)
    );
    assert_eq!(
        gyro_half.fixed_view::<3, 3>(6, 0).into_owned(),
        -meas.d_state_d_bias_gyro().fixed_view::<3, 3>(6, 0)
    );
    let res_rot: Vector3<f64> = res.fixed_rows::<3>(3).into_owned();
    assert_abs_diff_eq!(
        gyro_half.fixed_view::<3, 3>(3, 0).into_owned(),
        kornia_staging_algebra::lie::left_jacobian_inv_so3(&res_rot)
            * meas.d_state_d_bias_gyro().fixed_view::<3, 3>(3, 0),
        epsilon = 1e-15
    );
}

// ── Properties ──────────────────────────────────────────────────────

fn zero_motion_measurement(
    steps: usize,
    dt_ns: i64,
    accel: Vector3<f64>,
) -> IntegratedImuMeasurement<f64> {
    let (accel_cov, gyro_cov) = noise_covariances();
    let mut meas: IntegratedImuMeasurement<f64> =
        IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
    for step in 1..=steps as i64 {
        meas.integrate(
            &CombinedImuSample {
                timestamp_ns: step * dt_ns,
                gyro: kornia_algebra::Vec3F64::ZERO,
                accel: kornia_algebra::Vec3F64::from_array(accel.into()),
            },
            &crate::imu::ImuNoise::new(accel_cov, gyro_cov).unwrap(),
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
        let g: Vector3<f64> = GRAVITY;
        let meas: IntegratedImuMeasurement<f64> = zero_motion_measurement(steps, dt_ns, -g);
        let total: f64 = meas.dt_ns() as f64 * 1e-9;

        let delta: &NavState<f64> = meas.delta_state();
        prop_assert!(delta.t_w_i.rotation.log().norm() <= 1e-15);
        prop_assert!((delta.vel_w_i - (-g * total)).norm() <= 1e-9);
        prop_assert!((delta.t_w_i.translation - (-g * 0.5 * total * total)).norm() <= 1e-9);

        let state0: NavState<f64> = NavState::default();
        let state1: NavState<f64> = meas.predict_state(&state0, &g);
        prop_assert!(state1.t_w_i.translation.norm() <= 1e-9);
        prop_assert!(state1.vel_w_i.norm() <= 1e-9);
        prop_assert!(state1.t_w_i.rotation.log().norm() <= 1e-15);
        prop_assert_eq!(state1.timestamp_ns, meas.dt_ns());

        let res = meas.residual(&state0, &g, &state1, &Vector3::zeros(), &Vector3::zeros());
        prop_assert!(res.amax() <= 1e-9);
    }

    /// A free-falling rig — zero specific force — accumulates nothing, and
    /// the prediction is pure gravity: `v = g T`, `p = ½ g T²`.
    #[test]
    fn free_fall_is_pure_gravity(steps in 1usize..40, dt_ns in 1_000_000i64..20_000_000) {
        let g: Vector3<f64> = GRAVITY;
        let meas: IntegratedImuMeasurement<f64> =
            zero_motion_measurement(steps, dt_ns, Vector3::zeros());
        let total: f64 = meas.dt_ns() as f64 * 1e-9;

        let delta: &NavState<f64> = meas.delta_state();
        prop_assert_eq!(delta.vel_w_i, Vector3::zeros());
        prop_assert_eq!(delta.t_w_i.translation, Vector3::zeros());

        let state1: NavState<f64> = meas.predict_state(&NavState::default(), &g);
        prop_assert!((state1.vel_w_i - g * total).norm() <= 1e-12);
        prop_assert!((state1.t_w_i.translation - g * 0.5 * total * total).norm() <= 1e-12);
    }

    /// The covariance stays symmetric and positive semi-definite, and the
    /// square-root inverse really is one: `(MᵀM) cov = I`.
    #[test]
    fn the_covariance_is_symmetric_psd_and_its_factor_inverts_it(
        steps in 2usize..25,
        dt_ns in 1_000_000i64..10_000_000,
        gyro_scale in 0.0f64..1.0,
    ) {
        let (accel_cov, gyro_cov) = noise_covariances();
        let mut meas: IntegratedImuMeasurement<f64> =
            IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
        for step in 1..=steps as i64 {
            meas.integrate(&CombinedImuSample {
                    timestamp_ns: step * dt_ns,
                    gyro: kornia_algebra::Vec3F64::new(0.3 * gyro_scale, -0.2 * gyro_scale, 0.5 * gyro_scale),
                    accel: kornia_algebra::Vec3F64::new(0.4, -0.3, 9.7),
                }, &crate::imu::ImuNoise::new(accel_cov, gyro_cov).unwrap())?;
        }

        let cov: Matrix9<f64> = *meas.cov();
        let asymmetry: f64 = (cov - cov.transpose()).amax();
        prop_assert!(asymmetry <= 1e-12 * cov.amax().max(1.0), "asymmetry {}", asymmetry);

        let smallest: f64 = cov.symmetric_eigenvalues().min();
        prop_assert!(smallest >= -1e-9 * cov.amax(), "smallest eigenvalue {}", smallest);

        let m: Matrix9<f64> = meas.cov_inv_sqrt();
        let deviation: f64 = (m.transpose() * m * cov - Matrix9::identity()).amax();
        prop_assert!(deviation <= 1e-6, "MᵀM cov - I = {}", deviation);
    }

}
