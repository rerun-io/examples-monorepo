//! Factor checks share the parent module's analytic trajectory fixtures.
use super::*;
use crate::imu::{ImuBlock, ImuLinData};
use crate::types::{AbsOrderMap, POSE_SIZE, POSE_VEL_BIAS_SIZE, PoseVelBiasStateWithLin, Vector15};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

// IMU factor invariants.

/// `ScBundleAdjustmentBase::checkNullspace`,
/// restricted to the pose-velocity-bias blocks the two IMU tests use — the
/// pose-only branch belongs to the visual factors.
///
/// Returns `xHx + xb` for six global directions plus one random direction: a
/// shift of every position along x, y and z, and a rotation of every pose
/// *and velocity* about the position centroid in roll, pitch and yaw. For a
/// VIO problem only yaw and the three translations are unobservable, so only
/// entries 0, 1, 2 and 5 are expected to vanish.
fn check_nullspace(
    h: &DMatrix<f64>,
    b: &DVector<f64>,
    order: &AbsOrderMap,
    frame_states: &HashMap<i64, PoseVelBiasStateWithLin<f64>>,
    rng: &mut Rng,
) -> [f64; 7] {
    let size: usize = order.total_size();
    let mut increments: [DVector<f64>; 6] = std::array::from_fn(|_| DVector::zeros(size));

    // the centroid the rotations turn about.
    let mut mean_trans: Vector3<f64> = Vector3::zeros();
    for (frame_id, _, _) in order.iter() {
        mean_trans += frame_states[&frame_id].state_lin().t_w_i.translation;
    }
    mean_trans /= order.items() as f64;

    let eps: f64 = 0.01; // `:541`
    for (frame_id, offset, _) in order.iter() {
        let state: &PoseVelBiasState<f64> = frame_states[&frame_id].state_lin();
        for axis in 0..3 {
            increments[axis][offset + axis] = eps; // `:545-547`
            increments[3 + axis][offset + 3 + axis] = eps; // `:548-550`
        }

        //  and : the rotation increments also move the
        // translations (about the centroid) and the velocities.
        let j: Matrix3<f64> = -So3::hat(&(state.t_w_i.translation - mean_trans)) * eps;
        let j_vel: Matrix3<f64> = -So3::hat(&state.vel_w_i) * eps;
        for axis in 0..3 {
            for row in 0..3 {
                increments[3 + axis][offset + row] = j[(row, axis)];
                increments[3 + axis][offset + POSE_SIZE + row] = j_vel[(row, axis)];
            }
        }
    }

    let mut result: [f64; 7] = [0.0; 7];
    for (index, increment) in increments.iter().enumerate() {
        let unit: DVector<f64> = increment / increment.norm(); // `:589-594`
        result[index] = unit.dot(&(h * &unit)) + unit.dot(b);
    }
    let random: DVector<f64> = DVector::from_fn(size, |_, _| rng.uniform());
    let random: DVector<f64> = &random / random.norm(); // `:601-603`
    result[6] = random.dot(&(h * &random)) + random.dot(b);
    result
}

/// `ScBundleAdjustmentBase::computeImuError`,
/// for the measurements the two IMU tests hold, summed into one number.
///
/// Note `gyro_bias_weight / dt` here against `gyro_bias_weight_sqrt / sqrt(dt)`
/// in the block : the caller passes the squared weight.
fn compute_imu_error(
    order: &AbsOrderMap,
    states: &HashMap<i64, PoseVelBiasStateWithLin<f64>>,
    measurements: &[IntegratedImuMeasurement<f64>],
    gyro_bias_weight: &Vector3<f64>,
    accel_bias_weight: &Vector3<f64>,
    g: &Vector3<f64>,
) -> f64 {
    let mut total: f64 = 0.0;
    for meas in measurements {
        if meas.get_dt_ns() == 0 {
            continue;
        }
        let start_t: i64 = meas.get_start_t_ns();
        let end_t: i64 = start_t + meas.get_dt_ns();
        if !order.contains(start_t) || !order.contains(end_t) {
            continue;
        }
        let start: &PoseVelBiasState<f64> = states[&start_t].state();
        let end: &PoseVelBiasState<f64> = states[&end_t].state();
        let res: Vector9<f64> = meas.residual(
            &start.pose_vel_state(),
            g,
            &end.pose_vel_state(),
            &start.bias_gyro,
            &start.bias_accel,
        );
        total += 0.5 * res.dot(&(meas.get_cov_inv() * res));

        let dt: f64 = meas.get_dt_ns() as f64 * 1e-9;
        let res_bg: Vector3<f64> = start.bias_gyro - end.bias_gyro;
        total += 0.5 * res_bg.dot(&(gyro_bias_weight / dt).component_mul(&res_bg));
        let res_ba: Vector3<f64> = start.bias_accel - end.bias_accel;
        total += 0.5 * res_ba.dot(&(accel_bias_weight / dt).component_mul(&res_ba));
    }
    total
}

/// Noisy samples over `[from_ns, to_ns)`, as both nullspace tests build them
fn noisy_samples(
    trajectory: &Trajectory,
    bg: &Vector3<f64>,
    ba: &Vector3<f64>,
    from_ns: i64,
    to_ns: i64,
    rng: &mut Rng,
) -> Vec<ImuSample> {
    let dt_ns: i64 = 10_000_000;
    let mut samples: Vec<ImuSample> = Vec::new();
    let mut t_ns: i64 = from_ns;
    while t_ns < to_ns {
        let mut sample: ImuSample = trajectory.sample(t_ns, dt_ns);
        sample.accel += ba + rng.vector3() * ACCEL_STD_DEV;
        sample.gyro += bg + rng.vector3() * GYRO_STD_DEV;
        samples.push(sample);
        t_ns += dt_ns;
    }
    samples
}

fn integrate_noisy(
    start_t_ns: i64,
    bg: &Vector3<f64>,
    ba: &Vector3<f64>,
    samples: &[ImuSample],
) -> IntegratedImuMeasurement<f64> {
    let noise: ImuNoise<f64> = noise_from_std_dev();
    let mut meas: IntegratedImuMeasurement<f64> = IntegratedImuMeasurement::new(start_t_ns, bg, ba);
    for sample in samples {
        meas.integrate(sample, &noise.accel_cov, &noise.gyro_cov)
            .unwrap();
    }
    meas
}

/// both weights are `1e3`.
fn lin_data() -> ImuLinData<f64> {
    ImuLinData {
        g: gravity::<f64>(),
        gyro_bias_weight_sqrt: Vector3::repeat(1e3),
        accel_bias_weight_sqrt: Vector3::repeat(1e3),
    }
}

/// `VioTestSuite.ImuNullspace2Test`.
///
/// One IMU factor between two full states: the block's `H` and `b` have to
/// reproduce the error change to `2e-2` for ten small random increments, and
/// the three global translations and the yaw rotation have to lie in the
/// nullspace of `H` and `b` to `1e-8` and `1e-6`.
#[test]
fn imu_nullspace_2() {
    let mut rng: Rng = Rng::new(0x5eed_0006);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;

    let samples: Vec<ImuSample> =
        noisy_samples(&trajectory, &bg, &ba, 5_000_000, 100_000_000, &mut rng);
    let meas: IntegratedImuMeasurement<f64> = integrate_noisy(0, &bg, &ba, &samples);

    let state0: PoseVelBiasState<f64> =
        PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
    let end_t_ns: i64 = meas.get_dt_ns();
    let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
        end_t_ns,
        trajectory.pose(end_t_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
        trajectory.trans_vel_world(end_t_ns) + rng.vector3() / 10.0,
        bg,
        ba,
    );

    let mut frame_states: HashMap<i64, PoseVelBiasStateWithLin<f64>> = HashMap::new();
    frame_states.insert(0, PoseVelBiasStateWithLin::new(state0, false));
    frame_states.insert(end_t_ns, PoseVelBiasStateWithLin::new(state1, false));

    let mut order: AbsOrderMap = AbsOrderMap::new();
    order.push(0, POSE_VEL_BIAS_SIZE).unwrap();
    order.push(end_t_ns, POSE_VEL_BIAS_SIZE).unwrap();
    let size: usize = order.total_size();

    let ild: ImuLinData<f64> = lin_data();
    let block: ImuBlock<f64> =
        ImuBlock::linearize(&meas, &ild, &frame_states[&0], &frame_states[&end_t_ns]);
    let mut h: DMatrix<f64> = DMatrix::zeros(size, size);
    let mut b: DVector<f64> = DVector::zeros(size);
    block.add_dense_h_b(0, POSE_VEL_BIAS_SIZE, &mut h, &mut b);
    let e0: f64 = block.error;

    // the quadratic model has to predict the error change.
    let gyro_weight: Vector3<f64> = ild.gyro_bias_weight_sqrt.map(|v: f64| v * v);
    let accel_weight: Vector3<f64> = ild.accel_bias_weight_sqrt.map(|v: f64| v * v);
    for _ in 0..10 {
        let raw: DVector<f64> = DVector::from_fn(size, |_, _| rng.uniform());
        let inc: DVector<f64> = &raw / raw.norm() / 10_000.0;

        let mut moved: HashMap<i64, PoseVelBiasStateWithLin<f64>> = frame_states.clone();
        let mut inc0: Vector15<f64> = Vector15::zeros();
        let mut inc1: Vector15<f64> = Vector15::zeros();
        for i in 0..POSE_VEL_BIAS_SIZE {
            inc0[i] = inc[i];
            inc1[i] = inc[POSE_VEL_BIAS_SIZE + i];
        }
        moved.get_mut(&0).unwrap().apply_inc(&inc0);
        moved.get_mut(&end_t_ns).unwrap().apply_inc(&inc1);

        let e1: f64 = compute_imu_error(
            &order,
            &moved,
            std::slice::from_ref(&meas),
            &gyro_weight,
            &accel_weight,
            &ild.g,
        ) - e0;
        let e2: f64 = 0.5 * inc.dot(&(&h * &inc)) + inc.dot(&b);
        assert!((e1 - e2).abs() <= 2e-2, "e1 {e1} e2 {e2}");
    }

    let null_res: [f64; 7] = check_nullspace(&h, &b, &order, &frame_states, &mut rng);
    assert!(null_res[0].abs() <= 1e-8, "x {}", null_res[0]);
    assert!(null_res[1].abs() <= 1e-8, "y {}", null_res[1]);
    assert!(null_res[2].abs() <= 1e-8, "z {}", null_res[2]);
    assert!(null_res[5].abs() <= 1e-6, "yaw {}", null_res[5]);
    // Gravity makes roll and pitch observable; require real information in those
    // and random directions so an all-zero Hessian cannot pass vacuously.
    assert!(null_res[3].abs() > 1.0, "roll {}", null_res[3]);
    assert!(null_res[4].abs() > 1.0, "pitch {}", null_res[4]);
    assert!(null_res[6].abs() > 1.0, "random {}", null_res[6]);
}

/// `VioTestSuite.ImuNullspace3Test`.
///
/// Two consecutive IMU factors over three states; the same four directions
/// have to stay in the nullspace once both blocks are accumulated.
#[test]
fn imu_nullspace_3() {
    let mut rng: Rng = Rng::new(0x5eed_0007);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;

    let samples1: Vec<ImuSample> =
        noisy_samples(&trajectory, &bg, &ba, 5_000_000, 1_000_000_000, &mut rng);
    let meas1: IntegratedImuMeasurement<f64> = integrate_noisy(0, &bg, &ba, &samples1);
    let t1_ns: i64 = meas1.get_dt_ns();

    let samples2: Vec<ImuSample> = noisy_samples(
        &trajectory,
        &bg,
        &ba,
        t1_ns + 5_000_000,
        2_000_000_000,
        &mut rng,
    );
    let meas2: IntegratedImuMeasurement<f64> = integrate_noisy(t1_ns, &bg, &ba, &samples2);
    let t2_ns: i64 = t1_ns + meas2.get_dt_ns();

    let state0: PoseVelBiasState<f64> =
        PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
    let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
        t1_ns,
        trajectory.pose(t1_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
        trajectory.trans_vel_world(t1_ns) + rng.vector3() / 10.0,
        bg,
        ba,
    );
    let state2: PoseVelBiasState<f64> = PoseVelBiasState::new(
        t2_ns,
        trajectory.pose(t2_ns) * Se3::exp_decoupled(&(rng.vector6() / 10.0)),
        trajectory.trans_vel_world(t2_ns) + rng.vector3() / 10.0,
        bg,
        ba,
    );

    let mut frame_states: HashMap<i64, PoseVelBiasStateWithLin<f64>> = HashMap::new();
    frame_states.insert(0, PoseVelBiasStateWithLin::new(state0, false));
    frame_states.insert(t1_ns, PoseVelBiasStateWithLin::new(state1, false));
    frame_states.insert(t2_ns, PoseVelBiasStateWithLin::new(state2, false));

    let mut order: AbsOrderMap = AbsOrderMap::new();
    order.push(0, POSE_VEL_BIAS_SIZE).unwrap();
    order.push(t1_ns, POSE_VEL_BIAS_SIZE).unwrap();
    order.push(t2_ns, POSE_VEL_BIAS_SIZE).unwrap();
    let size: usize = order.total_size();

    let ild: ImuLinData<f64> = lin_data();
    let mut h: DMatrix<f64> = DMatrix::zeros(size, size);
    let mut b: DVector<f64> = DVector::zeros(size);
    ImuBlock::linearize(&meas1, &ild, &frame_states[&0], &frame_states[&t1_ns]).add_dense_h_b(
        0,
        POSE_VEL_BIAS_SIZE,
        &mut h,
        &mut b,
    );
    ImuBlock::linearize(&meas2, &ild, &frame_states[&t1_ns], &frame_states[&t2_ns]).add_dense_h_b(
        POSE_VEL_BIAS_SIZE,
        2 * POSE_VEL_BIAS_SIZE,
        &mut h,
        &mut b,
    );

    let null_res: [f64; 7] = check_nullspace(&h, &b, &order, &frame_states, &mut rng);
    assert!(null_res[0].abs() <= 1e-8, "x {}", null_res[0]);
    assert!(null_res[1].abs() <= 1e-8, "y {}", null_res[1]);
    assert!(null_res[2].abs() <= 1e-8, "z {}", null_res[2]);
    assert!(null_res[5].abs() <= 1e-6, "yaw {}", null_res[5]);
    // Gravity makes roll and pitch observable; require real information in those
    // and random directions so an all-zero Hessian cannot pass vacuously.
    assert!(null_res[3].abs() > 1.0, "roll {}", null_res[3]);
    assert!(null_res[4].abs() > 1.0, "pitch {}", null_res[4]);
    assert!(null_res[6].abs() > 1.0, "random {}", null_res[6]);
}

/// An out-of-range block offset is ignored, and the *check itself* does not
/// overflow: `usize::MAX + 15` panics in debug and wraps to a small,
/// accepted number in release (decision D32).
#[test]
fn the_block_ignores_offsets_that_do_not_fit() {
    let mut rng: Rng = Rng::new(0x5eed_000e);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
    let end_t_ns: i64 = meas.get_dt_ns();

    let state0: PoseVelBiasState<f64> =
        PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
    let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
        end_t_ns,
        trajectory.pose(end_t_ns),
        trajectory.trans_vel_world(end_t_ns),
        bg,
        ba,
    );
    let block: ImuBlock<f64> = ImuBlock::linearize(
        &meas,
        &lin_data(),
        &PoseVelBiasStateWithLin::new(state0, false),
        &PoseVelBiasStateWithLin::new(state1, false),
    );

    let size: usize = 2 * POSE_VEL_BIAS_SIZE;
    for (start_idx, end_idx) in [
        (usize::MAX, 0),
        (0, usize::MAX),
        (usize::MAX, usize::MAX),
        (size, 0),
        (0, size),
    ] {
        let mut h: DMatrix<f64> = DMatrix::zeros(size, size);
        let mut b: DVector<f64> = DVector::zeros(size);
        block.add_dense_h_b(start_idx, end_idx, &mut h, &mut b);
        assert_eq!(h.norm(), 0.0, "({start_idx}, {end_idx}) wrote to H");
        assert_eq!(b.norm(), 0.0, "({start_idx}, {end_idx}) wrote to b");
    }

    // The offsets that do fit still work.
    let mut h: DMatrix<f64> = DMatrix::zeros(size, size);
    let mut b: DVector<f64> = DVector::zeros(size);
    block.add_dense_h_b(0, POSE_VEL_BIAS_SIZE, &mut h, &mut b);
    assert!(h.norm() > 0.0);
}

/// The square-root and the squared form of the same factor agree:
/// `Q2Jpᵀ Q2Jp = H` and `Q2Jpᵀ Q2r = b`.
///
/// `addJp_diag2` must likewise be the squared column norms of
/// the same scattered Jacobian, and `backSubstitute` the model
/// cost change of the same `J` and `r`.
#[test]
fn the_imu_block_exports_agree_with_each_other() {
    let mut rng: Rng = Rng::new(0x5eed_00e1);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
    let end_t_ns: i64 = meas.get_dt_ns();
    let state0: PoseVelBiasState<f64> =
        PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
    let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
        end_t_ns,
        trajectory.pose(end_t_ns),
        trajectory.trans_vel_world(end_t_ns),
        bg + rng.vector3() / 1000.0,
        ba + rng.vector3() / 1000.0,
    );
    let block: ImuBlock<f64> = ImuBlock::linearize(
        &meas,
        &lin_data(),
        &PoseVelBiasStateWithLin::new(state0, false),
        &PoseVelBiasStateWithLin::new(state1, false),
    );

    // Two states, side by side, plus a third that nothing touches — so the
    // scatter really is checked against the offsets it was given.
    let size: usize = POSE_VEL_BIAS_SIZE;
    let total: usize = 3 * size;
    let (start_idx, end_idx) = (size, 2 * size);

    let mut q2jp: DMatrix<f64> = DMatrix::zeros(size, total);
    let mut q2r: DVector<f64> = DVector::zeros(size);
    block.add_dense_q2jp_q2r(start_idx, end_idx, 0, &mut q2jp, &mut q2r);

    let mut h: DMatrix<f64> = DMatrix::zeros(total, total);
    let mut b: DVector<f64> = DVector::zeros(total);
    block.add_dense_h_b(start_idx, end_idx, &mut h, &mut b);

    let h_sqrt: DMatrix<f64> = q2jp.transpose() * &q2jp;
    let b_sqrt: DVector<f64> = q2jp.transpose() * &q2r;
    assert_abs_diff_eq!(h_sqrt, h, epsilon = 1e-9 * h.norm());
    assert_abs_diff_eq!(b_sqrt, b, epsilon = 1e-9 * b.norm().max(1.0));

    // The first state's block is untouched.
    assert_eq!(q2jp.columns(0, size).norm(), 0.0);

    // `backSubstitute`: the model cost change of the scattered system.
    let inc: DVector<f64> =
        DVector::from_iterator(total, (0..total).map(|_| rng.uniform() / 100.0));
    let mut l_diff: f64 = 0.0;
    block.back_substitute(start_idx, end_idx, &inc, &mut l_diff);
    let jinc: DVector<f64> = &q2jp * &inc;
    let jinc: DVector<f64> = DVector::from_column_slice(jinc.as_slice());
    let want: f64 = -(jinc.transpose() * (0.5 * &jinc + &q2r))[(0, 0)];
    assert_abs_diff_eq!(l_diff, want, epsilon = 1e-12 * want.abs().max(1.0));

    // Offsets that do not fit are ignored, not a panic (decision D32).
    let mut l_diff: f64 = 0.0;
    block.back_substitute(0, total, &inc, &mut l_diff);
    assert_eq!(l_diff, 0.0);
}

/// A frozen linearization point changes only the residual *value*, not the
/// Jacobians (trap 7 of the architecture dossier).
#[test]
fn the_block_re_evaluates_the_residual_at_a_linearized_state() {
    let mut rng: Rng = Rng::new(0x5eed_000d);
    let trajectory: Trajectory = Trajectory::new(&mut rng);
    let bg: Vector3<f64> = rng.vector3() / 100.0;
    let ba: Vector3<f64> = rng.vector3() / 10.0;
    let samples: Vec<ImuSample> = biased_samples(&trajectory, &bg, &ba);
    let meas: IntegratedImuMeasurement<f64> = integrate_all(0, &bg, &ba, &samples);
    let end_t_ns: i64 = meas.get_dt_ns();

    let state0: PoseVelBiasState<f64> =
        PoseVelBiasState::new(0, trajectory.pose(0), trajectory.trans_vel_world(0), bg, ba);
    let state1: PoseVelBiasState<f64> = PoseVelBiasState::new(
        end_t_ns,
        trajectory.pose(end_t_ns),
        trajectory.trans_vel_world(end_t_ns),
        bg,
        ba,
    );
    let ild: ImuLinData<f64> = lin_data();

    let plain0: PoseVelBiasStateWithLin<f64> = PoseVelBiasStateWithLin::new(state0, false);
    let plain1: PoseVelBiasStateWithLin<f64> = PoseVelBiasStateWithLin::new(state1, false);
    let reference: ImuBlock<f64> = ImuBlock::linearize(&meas, &ild, &plain0, &plain1);

    let mut frozen0: PoseVelBiasStateWithLin<f64> = PoseVelBiasStateWithLin::new(state0, false);
    frozen0.set_linearized().unwrap();
    let mut inc: Vector15<f64> = Vector15::zeros();
    inc.fixed_rows_mut::<3>(0).copy_from(&Vector3::repeat(0.01));
    frozen0.apply_inc(&inc);

    let frozen: ImuBlock<f64> = ImuBlock::linearize(&meas, &ild, &frozen0, &plain1);
    // Same linearization point, so the Jacobian is untouched...
    assert_eq!(frozen.jp, reference.jp);
    // but the residual moved with the state.
    assert!((frozen.r - reference.r).norm() > 1e-6);
}
