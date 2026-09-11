//! Sliding-window marginalization on a synthetic window.
//! Two pose keyframes, three full states and forty landmarks exercise removals,
//! demotion and observations outside the ordering. Check the resulting maps and
//! prior against dense Schur elimination, residual re-anchoring and the quadratic
//! at accumulated delta. Invalid schedules and unfrozen blocks must be refused
//! before mutation; a valid frozen demotion must succeed.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, DVector, Vector2, Vector3, Vector4, Vector6};
use slam_rs::ba_base::{BaError, BundleAdjustmentBase};
use slam_rs::calib::Calibration;
use slam_rs::imu::IntegratedImuMeasurement;
use slam_rs::landmark::{Landmark, StereographicParam};
use slam_rs::lie::{Se3, So3};
use slam_rs::marg::{
    MargError, MarginalizeInputs, MarginalizeOptions, MarginalizeOutput, MarginalizeSchedule,
    ScheduleSet, marginalize, marginalize_helper_sqrt_to_sqrt,
};
use slam_rs::types::{
    AbsOrderMap, FrameId, LandmarkId, MargLinData, POSE_SIZE, POSE_VEL_BIAS_SIZE, PoseStateWithLin,
    PoseVelBiasState, PoseVelBiasStateWithLin, TimeCamId,
};

mod common;
use common::{Rng, dense_schur, test_calibration};

const KF0: FrameId = 0;
const KF1: FrameId = 100;
const STATE0: FrameId = 200;
const STATE1: FrameId = 300;
const STATE2: FrameId = 400;

/// A window ready to marginalize.
#[derive(Clone)]
struct Window {
    estimator: BundleAdjustmentBase<f64>,
    marg: MargLinData<f64>,
    imu_meas: BTreeMap<i64, IntegratedImuMeasurement<f64>>,
}

/// The pose of a frame in the synthetic trajectory: a metre of forward motion
/// with a small rotation on top.
fn pose_at(index: usize, rng: &mut Rng) -> Se3<f64> {
    let mut t_w_i: Se3<f64> = Se3::identity();
    t_w_i.rotation = So3::exp(&(rng.vector3() / 20.0));
    t_w_i.translation = Vector3::new(index as f64 * 0.2, index as f64 * 0.02, 0.0);
    t_w_i
}

/// Build the window.
///
/// `prior_covers_state0` chooses between the two shapes the estimator really
/// sees: a steady-state prior over both keyframes **and** the state that
/// entered it last ( produces exactly that),
/// and an empty prior over the keyframes alone, which is what a window looks
/// like before the first state has been marginalized.
fn build_window(seed: u64, prior_covers_state0: bool) -> Window {
    let mut rng: Rng = Rng::new(seed);

    let calib: Calibration<f64> = test_calibration(&mut rng);
    let mut estimator: BundleAdjustmentBase<f64> =
        BundleAdjustmentBase::new(calib, 2.0, 0.5).unwrap();

    // Two keyframes as pose blocks, three frames as full states.
    let frames: [FrameId; 5] = [KF0, KF1, STATE0, STATE1, STATE2];
    let mut poses: Vec<Se3<f64>> = Vec::new();
    for (index, frame_id) in frames.iter().enumerate() {
        let t_w_i: Se3<f64> = pose_at(index, &mut rng);
        poses.push(t_w_i);
        if *frame_id == KF0 || *frame_id == KF1 {
            // Both keyframes are in the prior, so both are frozen at their
            // linearization point (`computeDelta` asserts it).
            estimator
                .frame_poses
                .insert(*frame_id, PoseStateWithLin::new(*frame_id, t_w_i, true));
        } else {
            let state: PoseVelBiasState<f64> = PoseVelBiasState::new(
                *frame_id,
                t_w_i,
                Vector3::new(0.2, 0.0, 0.0),
                rng.vector3() / 1000.0,
                rng.vector3() / 1000.0,
            );
            let linearized: bool = prior_covers_state0 && *frame_id == STATE0;
            estimator
                .frame_states
                .insert(*frame_id, PoseVelBiasStateWithLin::new(state, linearized));
        }
    }

    // Forty landmarks: thirty hosted by the keyframe that leaves, ten by the
    // one that stays.
    let add_landmarks = |estimator: &mut BundleAdjustmentBase<f64>,
                         rng: &mut Rng,
                         host_index: usize,
                         first_id: u64,
                         count: u64| {
        let host_frame: FrameId = frames[host_index];
        for k in 0..count {
            let id: LandmarkId = LandmarkId(first_id + k);
            let mut point: Vector3<f64> = rng.vector3();
            point[2] += 5.0;
            let p3d: Vector3<f64> = poses[host_index] * point;

            let t_c_w: Se3<f64> = (poses[host_index] * estimator.calib.t_i_c[0]).inverse();
            let p_host: Vector3<f64> = t_c_w * p3d;
            let landmark: Landmark<f64> = Landmark::new(
                id,
                TimeCamId::new(host_frame, 0),
                StereographicParam::project(&Vector4::new(p_host[0], p_host[1], p_host[2], 1.0)),
                1.0 / p_host.norm(),
            );
            estimator.lmdb.add_landmark(id, &landmark);

            // Every image of the four frames inside the ordering, and for one
            // landmark in five also camera 0 of the frame outside it.
            for (f, frame_id) in frames.iter().enumerate() {
                let cams: &[usize] = if *frame_id == STATE2 {
                    if k % 5 == 0 { &[0] } else { &[] }
                } else {
                    &[0, 1]
                };
                for &c in cams {
                    let t_c_w: Se3<f64> = (poses[f] * estimator.calib.t_i_c[c]).inverse();
                    let p_cam: Vector3<f64> = t_c_w * p3d;
                    let mut pixel: Vector2<f64> = Vector2::zeros();
                    let mut jacobian = nalgebra::Matrix2x4::zeros();
                    let ok: bool = estimator.cameras()[c].project_with_jacobian(
                        &Vector4::new(p_cam[0], p_cam[1], p_cam[2], 1.0),
                        &mut pixel,
                        &mut jacobian,
                    );
                    assert!(ok, "the synthetic window must project");
                    pixel[0] += rng.symmetric() / 100.0;
                    pixel[1] += rng.symmetric() / 100.0;
                    estimator
                        .lmdb
                        .add_observation(TimeCamId::new(*frame_id, c), id, pixel)
                        .unwrap();
                }
            }
        }
    };
    add_landmarks(&mut estimator, &mut rng, 0, 0, 30);
    add_landmarks(&mut estimator, &mut rng, 1, 1000, 10);

    // The prior. Its ordering is a prefix of the window's, at the same offsets
    let mut order: AbsOrderMap = AbsOrderMap::new();
    order.push(KF0, POSE_SIZE).unwrap();
    order.push(KF1, POSE_SIZE).unwrap();
    if prior_covers_state0 {
        order.push(STATE0, POSE_VEL_BIAS_SIZE).unwrap();
    }
    let marg_size: usize = order.total_size();
    let (h, b): (DMatrix<f64>, DVector<f64>) = if prior_covers_state0 {
        // A real square-root prior: a well-conditioned `J_m` and a residual.
        let rows: usize = marg_size;
        let mut h: DMatrix<f64> = DMatrix::from_fn(rows, marg_size, |_, _| rng.symmetric());
        for i in 0..marg_size {
            h[(i, i)] += 30.0;
        }
        let b: DVector<f64> = DVector::from_fn(rows, |_, _| rng.symmetric());
        (h, b)
    } else {
        // The empty prior: no rows, but the ordering is still real, which is
        // what the window before the first marginalization looks like.
        (DMatrix::zeros(0, marg_size), DVector::zeros(0))
    };

    // Two preintegrated intervals, one per gap between the full states. They
    // carry no samples: `imu_lin_data` is `None` in these tests, so nothing
    // reads them, and what is under test is that the consumed one is dropped.
    let mut imu_meas: BTreeMap<i64, IntegratedImuMeasurement<f64>> = BTreeMap::new();
    for start in [STATE0, STATE1] {
        imu_meas.insert(
            start,
            IntegratedImuMeasurement::new(start, &Vector3::zeros(), &Vector3::zeros()),
        );
    }

    Window {
        estimator,
        marg: MargLinData { order, h, b },
        imu_meas,
    }
}

/// The schedule the tests marginalize with: the oldest keyframe and the oldest
/// full state leave, and `STATE1` becomes the prior's newest block.
fn schedule() -> MarginalizeSchedule {
    MarginalizeSchedule {
        last_state_to_marg: STATE1,
        kfs_to_marg: [KF0].into_iter().collect(),
        poses_to_marg: [KF0].into_iter().collect(),
        states_to_marg_all: [STATE0].into_iter().collect(),
        states_to_marg_vel_bias: BTreeSet::new(),
    }
}

/// The schedule that demotes: `STATE1` keeps its pose block and loses its
/// velocity and biases, so `STATE2` becomes the prior's newest
/// block.
fn demote_state1() -> MarginalizeSchedule {
    MarginalizeSchedule {
        last_state_to_marg: STATE2,
        states_to_marg_vel_bias: [STATE1].into_iter().collect(),
        ..schedule()
    }
}

/// One marginalization of `window`, with everything the tests ever vary.
fn try_run(
    window: &mut Window,
    sched: &MarginalizeSchedule,
    options: MarginalizeOptions,
) -> Result<MarginalizeOutput<f64>, MargError> {
    let inputs: MarginalizeInputs<'_, f64> = MarginalizeInputs {
        schedule: sched,
        imu_lin_data: None,
        lost_landmarks: None,
        fixed_frames: None,
        options,
    };
    marginalize(
        &mut window.estimator,
        &mut window.marg,
        &mut window.imu_meas,
        &inputs,
    )
}

fn run(window: &mut Window, options: MarginalizeOptions) -> MarginalizeOutput<f64> {
    try_run(window, &schedule(), options).expect("the synthetic window marginalizes")
}

/// The whole window, every coefficient of it: `f64`'s `Debug` is the shortest
/// representation that round-trips, so equal strings are equal states, and this
/// covers the landmark database and the IMU intervals as well as the frame maps
/// and the prior.
fn window_debug(window: &Window) -> String {
    format!("{:?}", (&window.estimator, &window.marg, &window.imu_meas))
}

// ─── the tests ─────────────────────────────────────────────────────────────

/// what the window looks like afterwards.
#[test]
fn marginalizing_a_keyframe_shrinks_the_window() {
    let mut window: Window = build_window(0xB00C, true);
    let landmarks_before: usize = window.estimator.lmdb.num_landmarks();
    assert_eq!(landmarks_before, 40);

    let out: MarginalizeOutput<f64> = run(&mut window, MarginalizeOptions::default());

    // The ordering the marginalization ran over stops at `last_state_to_marg`
    // two poses and two states, not three.
    assert_eq!(out.aom.items(), 4);
    assert_eq!(out.aom.total_size(), 2 * POSE_SIZE + 2 * POSE_VEL_BIAS_SIZE);
    assert!(!out.aom.contains(STATE2), "the newest state is outside it");

    // the split covers every column exactly once.
    assert_eq!(
        out.idx_to_keep.len() + out.idx_to_marg.len(),
        out.aom.total_size()
    );
    assert!(out.idx_to_keep.is_disjoint(&out.idx_to_marg));
    // `KF0` (6) and `STATE0` (15) go; `KF1` (6) and `STATE1` (15) stay.
    assert_eq!(out.idx_to_marg.len(), POSE_SIZE + POSE_VEL_BIAS_SIZE);
    assert_eq!(out.idx_to_keep.len(), POSE_SIZE + POSE_VEL_BIAS_SIZE);
    assert!(out.numerically_valid);

    assert_eq!(
        window
            .estimator
            .frame_poses
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![KF1]
    );
    assert_eq!(
        window
            .estimator
            .frame_states
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![STATE1, STATE2]
    );

    // trap 7: the state entering the prior is frozen, and its
    // delta is still zero, so it contributes nothing to the re-anchoring.
    let state1 = window.estimator.frame_states.get(&STATE1).unwrap();
    assert!(state1.is_linearized());
    assert_eq!(*state1.delta(), nalgebra::SVector::<f64, 15>::zeros());

    // every landmark hosted by the marginalized keyframe is gone, and
    // the ten hosted by the surviving one are not.
    assert_eq!(window.estimator.lmdb.num_landmarks(), 10);
    for id in 0..30u64 {
        assert!(!window.estimator.lmdb.landmark_exists(LandmarkId(id)));
    }
    for id in 1000..1010u64 {
        assert!(window.estimator.lmdb.landmark_exists(LandmarkId(id)));
    }
    // and no observation is left in a frame that left the window.
    for targets in window.estimator.lmdb.observations().values() {
        for tcid in targets.keys() {
            assert_ne!(tcid.frame_id, KF0);
            assert_ne!(tcid.frame_id, STATE0);
        }
    }

    // the interval that started at the marginalized state is consumed.
    assert_eq!(
        window.imu_meas.keys().copied().collect::<Vec<_>>(),
        vec![STATE1]
    );

    // the new prior's ordering is the window that survives.
    assert_eq!(window.marg.order.items(), 2);
    assert_eq!(window.marg.order.get(KF1), Some((0, POSE_SIZE)));
    assert_eq!(
        window.marg.order.get(STATE1),
        Some((POSE_SIZE, POSE_VEL_BIAS_SIZE))
    );
    assert_eq!(
        window.marg.order.total_size(),
        POSE_SIZE + POSE_VEL_BIAS_SIZE
    );
    assert_eq!(window.marg.h.ncols(), window.marg.order.total_size());
    assert_eq!(window.marg.b.nrows(), window.marg.h.nrows());
}

/// The prior really is the marginal of the window it came from.
///
/// The flat QR of `marginalizeHelperSqrtToSqrt` never forms `JᵀJ`; this test
/// forms it, takes the dense Schur complement over the same index split, and
/// checks that the square-root prior squares to it. That is the same argument
/// `VoMargSqrtLinearizationTest` makes about the linearization
/// one level up.
///
/// It also checks the statement that matters to the estimator: the increment
/// that minimizes the reduced quadratic is the increment the full system would
/// have given the kept variables.
#[test]
fn the_prior_is_the_schur_complement_of_the_window() {
    let window: Window = build_window(0xB00D, true);

    // The same system the marginalization is about to consume, taken through
    // the public linearizer on an untouched copy of the window.
    let (q2jp, q2r) = linearized_system(&window);

    let mut marginalized: Window = window.clone();
    let out: MarginalizeOutput<f64> = run(&mut marginalized, MarginalizeOptions::default());

    let keep: Vec<usize> = out.idx_to_keep.iter().copied().collect();
    let marg: Vec<usize> = out.idx_to_marg.iter().copied().collect();
    let h: DMatrix<f64> = q2jp.transpose() * &q2jp;
    let b: DVector<f64> = q2jp.transpose() * &q2r;
    let (schur_h, schur_b) = dense_schur(&h, &b, &keep, &marg);

    // Undo the re-anchoring of to compare like with like.
    let delta: DVector<f64> = marginalized
        .estimator
        .compute_delta(&marginalized.marg.order)
        .unwrap();
    let b_pre: DVector<f64> = &marginalized.marg.b + &marginalized.marg.h * &delta;

    let got_h: DMatrix<f64> = marginalized.marg.h.transpose() * &marginalized.marg.h;
    let got_b: DVector<f64> = marginalized.marg.h.transpose() * &b_pre;

    let scale: f64 = schur_h.iter().fold(1.0f64, |a, v| a.max(v.abs()));
    for i in 0..keep.len() {
        for j in 0..keep.len() {
            assert!(
                (got_h[(i, j)] - schur_h[(i, j)]).abs() < 1e-8 * scale,
                "H[{i},{j}]: {} vs {}",
                got_h[(i, j)],
                schur_h[(i, j)]
            );
        }
        assert!(
            (got_b[i] - schur_b[i]).abs() < 1e-8 * scale,
            "b[{i}]: {} vs {}",
            got_b[i],
            schur_b[i]
        );
    }

    // The optimum over the kept variables is unchanged.
    //
    // Both reduced Hessians are singular, and for the same reason: a
    // visual-only window says nothing about the newest state's velocity or
    // biases, so nine directions carry no information. The solve is therefore
    // damped, with the same `lambda` on both sides — which is what the
    // estimator itself does (`H.diagonal() * lambda`,
    // ), so this compares the increment the
    // optimizer would really take.
    let lambda: f64 = 1e-6 * scale;
    let damped = |h: &DMatrix<f64>| -> DMatrix<f64> {
        let mut out: DMatrix<f64> = h.clone();
        for i in 0..h.nrows() {
            out[(i, i)] += lambda;
        }
        out
    };
    let want: DVector<f64> = damped(&schur_h).try_inverse().unwrap() * &schur_b;
    let got: DVector<f64> = damped(&got_h).try_inverse().unwrap() * &got_b;
    let want_scale: f64 = want.norm().max(1e-12);
    for i in 0..keep.len() {
        assert!(
            (got[i] - want[i]).abs() < 1e-7 * want_scale,
            "increment[{i}]: {} vs {}",
            got[i],
            want[i]
        );
    }
}

/// Trap 8: `marg_data.b -= marg_data.H * delta`.
///
/// The helper hands back a prior linearized at `x = 0`; the estimator stores
/// priors in the delta-independent form, so the residual has to lose
/// `J_m · delta` on the way in. This runs the helper by hand on the same
/// system and checks the difference is exactly that product.
#[test]
fn the_prior_is_re_anchored_on_the_delta() {
    let mut window: Window = build_window(0xB00E, true);
    // Give the surviving keyframe a non-zero delta, so the re-anchoring has
    // something to subtract. `KF1` is frozen, so `applyInc` accumulates
    let inc: Vector6<f64> = Vector6::from_iterator((0..6).map(|k| 0.001 * (k as f64 + 1.0)));
    window
        .estimator
        .frame_poses
        .get_mut(&KF1)
        .unwrap()
        .apply_inc(&inc);

    let (q2jp, q2r) = linearized_system(&window);

    let mut marginalized: Window = window.clone();
    let out: MarginalizeOutput<f64> = run(&mut marginalized, MarginalizeOptions::default());

    // The helper's own output, un-anchored.
    let raw =
        marginalize_helper_sqrt_to_sqrt(q2jp, q2r, &out.idx_to_keep, &out.idx_to_marg).unwrap();

    let delta: DVector<f64> = marginalized
        .estimator
        .compute_delta(&marginalized.marg.order)
        .unwrap();
    // `KF1`'s six rows carry the increment; `STATE1` was frozen with a zero
    // delta a moment ago and carries nothing.
    assert!(delta.rows(0, POSE_SIZE).norm() > 1e-6);
    assert_eq!(delta.rows(POSE_SIZE, POSE_VEL_BIAS_SIZE).norm(), 0.0);

    assert_eq!(raw.h.nrows(), marginalized.marg.h.nrows());
    let expected: DVector<f64> = &raw.b - &raw.h * &delta;
    for i in 0..expected.nrows() {
        assert!(
            (marginalized.marg.b[i] - expected[i]).abs() < 1e-12,
            "b[{i}]: {} vs {}",
            marginalized.marg.b[i],
            expected[i]
        );
    }
    // The Jacobian itself is untouched by the re-anchoring.
    for i in 0..raw.h.nrows() {
        for j in 0..raw.h.ncols() {
            assert_eq!(marginalized.marg.h[(i, j)], raw.h[(i, j)]);
        }
    }
}

/// `computeMargPriorError` after an increment is the quadratic model evaluated
/// at the accumulated delta.
///
/// The prior is `P(x) = 0.5‖J(delta + x) + r‖²` and `computeMargPriorError`
/// returns it with the constant `0.5 rᵀr` dropped, i.e.
/// `(J delta)ᵀ(0.5 J delta + r)`. The point of the test is that `applyInc` on a
/// **frozen** frame accumulates into `delta` rather than moving the
/// linearization point, so the prior sees the increment at all — trap 7 the
/// other way round.
#[test]
fn the_prior_error_is_the_quadratic_at_the_delta() {
    let mut window: Window = build_window(0xB00F, true);
    run(&mut window, MarginalizeOptions::default());

    let quadratic = |delta: &DVector<f64>| -> f64 {
        let hd: DVector<f64> = &window.marg.h * delta;
        hd.iter()
            .zip(window.marg.b.iter())
            .map(|(d, b)| d * (0.5 * d + b))
            .sum()
    };

    // Right after marginalizing, only the surviving keyframe can carry a delta,
    // and here it is zero, so the prior costs nothing.
    let delta0: DVector<f64> = window.estimator.compute_delta(&window.marg.order).unwrap();
    assert_eq!(delta0.norm(), 0.0);
    let e0: f64 = window
        .estimator
        .compute_marg_prior_error(&window.marg)
        .unwrap();
    assert!(e0.abs() < 1e-12, "no drift, no cost: {e0}");

    // Now move both blocks of the prior.
    let pose_inc: Vector6<f64> = Vector6::from_iterator((0..6).map(|k| 0.002 * (k as f64 - 2.0)));
    let state_inc: nalgebra::SVector<f64, 15> =
        nalgebra::SVector::from_iterator((0..15).map(|k| 0.001 * (k as f64 - 7.0)));
    window
        .estimator
        .frame_poses
        .get_mut(&KF1)
        .unwrap()
        .apply_inc(&pose_inc);
    window
        .estimator
        .frame_states
        .get_mut(&STATE1)
        .unwrap()
        .apply_inc(&state_inc);

    let delta1: DVector<f64> = window.estimator.compute_delta(&window.marg.order).unwrap();
    // `applyInc` accumulated exactly what was asked for, because both blocks
    // are frozen.
    for k in 0..POSE_SIZE {
        assert!((delta1[k] - pose_inc[k]).abs() < 1e-15);
    }
    for k in 0..POSE_VEL_BIAS_SIZE {
        assert!((delta1[POSE_SIZE + k] - state_inc[k]).abs() < 1e-15);
    }

    let e1: f64 = window
        .estimator
        .compute_marg_prior_error(&window.marg)
        .unwrap();
    let want: f64 = quadratic(&delta1);
    assert!(
        (e1 - want).abs() < 1e-9 * want.abs().max(1.0),
        "prior error {e1} vs quadratic {want}"
    );
    assert!(e1 > 0.0, "moving away from the linearization point costs");
}

/// Ordering mismatches return typed errors.
///
/// The schedule's own relationships to the window are
/// `an_invalid_schedule_is_refused_before_anything_changes`; this is the other
/// direction, the *prior* disagreeing with the window it is supposed to cover.
#[test]
fn a_window_that_disagrees_with_the_prior_is_refused() {
    let mut window: Window = build_window(0xB011, true);
    let mut order: AbsOrderMap = AbsOrderMap::new();
    order.push(KF0, POSE_SIZE).unwrap();
    order.push(STATE0, POSE_VEL_BIAS_SIZE).unwrap();
    window.marg.order = order;
    window.marg.h = DMatrix::zeros(0, POSE_SIZE + POSE_VEL_BIAS_SIZE);
    window.marg.b = DVector::zeros(0);
    assert_eq!(
        try_run(&mut window, &schedule(), MarginalizeOptions::default()),
        Err(MargError::PriorOrderMismatch { frame_id: KF1 })
    );
}

/// Break schedule relationships one at a time. All eight invalid schedules must
/// return typed errors and leave the window unchanged. Frozen-state validation
/// is checked separately with an otherwise valid schedule.
#[test]
fn an_invalid_schedule_is_refused_before_anything_changes() {
    let unknown: FrameId = 999;
    let cases: [(&str, MarginalizeSchedule, MargError); 8] = [
        (
            "a last_state_to_marg the window does not have",
            MarginalizeSchedule {
                last_state_to_marg: unknown,
                ..schedule()
            },
            MargError::FrameNotInWindow { frame_id: unknown },
        ),
        (
            "a state in neither set that is not last_state_to_marg (`:999`)",
            MarginalizeSchedule {
                states_to_marg_all: BTreeSet::new(),
                ..schedule()
            },
            MargError::UnscheduledState { frame_id: STATE0 },
        ),
        (
            "a frame the window does not have in states_to_marg_vel_bias",
            MarginalizeSchedule {
                states_to_marg_vel_bias: [unknown].into_iter().collect(),
                ..schedule()
            },
            MargError::ScheduledFrameNotInOrdering {
                set: ScheduleSet::StatesToMargVelBias,
                frame_id: unknown,
                block: POSE_VEL_BIAS_SIZE,
            },
        ),
        (
            "a state in both state sets",
            MarginalizeSchedule {
                states_to_marg_vel_bias: [STATE0].into_iter().collect(),
                ..schedule()
            },
            MargError::ScheduleSetsOverlap {
                first: ScheduleSet::StatesToMargAll,
                second: ScheduleSet::StatesToMargVelBias,
                frame_id: STATE0,
            },
        ),
        (
            "a state past last_state_to_marg, so outside the ordering",
            MarginalizeSchedule {
                states_to_marg_all: [STATE0, STATE2].into_iter().collect(),
                ..schedule()
            },
            MargError::ScheduledFrameNotInOrdering {
                set: ScheduleSet::StatesToMargAll,
                frame_id: STATE2,
                block: POSE_VEL_BIAS_SIZE,
            },
        ),
        (
            "a pose block named as a full state",
            MarginalizeSchedule {
                states_to_marg_vel_bias: [KF1].into_iter().collect(),
                ..schedule()
            },
            MargError::ScheduledFrameNotInOrdering {
                set: ScheduleSet::StatesToMargVelBias,
                frame_id: KF1,
                block: POSE_VEL_BIAS_SIZE,
            },
        ),
        (
            "a keyframe marginalized without its pose block",
            MarginalizeSchedule {
                kfs_to_marg: [KF0, KF1].into_iter().collect(),
                ..schedule()
            },
            MargError::KeyframeNotInPosesToMarg { frame_id: KF1 },
        ),
        (
            "last_state_to_marg marginalized outright",
            MarginalizeSchedule {
                states_to_marg_all: [STATE0, STATE1].into_iter().collect(),
                ..schedule()
            },
            MargError::ScheduleSetsOverlap {
                first: ScheduleSet::LastStateToMarg,
                second: ScheduleSet::StatesToMargAll,
                frame_id: STATE1,
            },
        ),
    ];

    let pristine: Window = build_window(0xB015, true);
    let before: String = window_debug(&pristine);

    for (name, sched, want) in cases {
        let mut window: Window = pristine.clone();
        let got = try_run(&mut window, &sched, MarginalizeOptions::default());
        assert_eq!(got.err(), Some(want), "{name}");
        assert_eq!(before, window_debug(&window), "{name}: the window changed");
    }
}

/// Check frozen-state preconditions before mutation for both surviving poses and
/// states being demoted. A valid schedule does not make an unfrozen block valid.
#[test]
fn a_window_that_is_not_frozen_is_refused_before_anything_changes() {
    let refused = |window: &mut Window, sched: &MarginalizeSchedule, frame_id: FrameId| {
        let before: String = window_debug(window);
        let got = try_run(window, sched, MarginalizeOptions::default());
        assert_eq!(
            got.err(),
            Some(MargError::Ba(BaError::NotLinearized { frame_id }))
        );
        assert_eq!(before, window_debug(window), "the window changed");
    };

    // The demoted state: `STATE1` is a free variable in the fixture, and
    // demotion would carry that flag into the new prior.
    let mut window: Window = build_window(0xB017, true);
    refused(&mut window, &demote_state1(), STATE1);

    // The surviving pose: `KF1` stays whole under the normal schedule. Thawed,
    // with the same linearization point and no increment, it is the other half
    // of the same walk.
    let mut window: Window = build_window(0xB017, true);
    let thawed: PoseStateWithLin<f64> =
        PoseStateWithLin::new(KF1, *window.estimator.frame_poses[&KF1].pose_lin(), false);
    window.estimator.frame_poses.insert(KF1, thawed);
    refused(&mut window, &schedule(), KF1);
}

/// The control: freezing that one state, and nothing else, makes the same
/// schedule marginalize and demote the selected state.
#[test]
fn a_frozen_demoted_state_marginalizes() {
    let mut window: Window = build_window(0xB017, true);
    window
        .estimator
        .frame_states
        .get_mut(&STATE1)
        .unwrap()
        .set_linearized()
        .unwrap();

    try_run(&mut window, &demote_state1(), MarginalizeOptions::default())
        .expect("freezing STATE1 makes the demotion schedule marginalize");

    // `STATE1` kept its pose and lost its velocity and biases, so it is a
    // 6-row block of the new prior, behind `KF1` and ahead of `STATE2`.
    assert!(window.estimator.frame_poses.contains_key(&STATE1));
    assert!(!window.estimator.frame_states.contains_key(&STATE1));
    assert_eq!(
        window.marg.order.get(STATE1),
        Some((POSE_SIZE, POSE_SIZE)),
        "the demoted block"
    );
}

// ─── helpers ───────────────────────────────────────────────────────────────

/// The stacked square-root system `marginalize` is about to consume, taken
/// through the public linearizer with the same inputs
fn linearized_system(window: &Window) -> (DMatrix<f64>, DVector<f64>) {
    use slam_rs::linearize::{LinearizationAbsQR, LinearizationInputs, LinearizationOptions};

    let mut aom: AbsOrderMap = AbsOrderMap::new();
    for frame_id in window.estimator.frame_poses.keys() {
        aom.push(*frame_id, POSE_SIZE).unwrap();
    }
    for frame_id in window.estimator.frame_states.keys() {
        if *frame_id > STATE1 {
            break;
        }
        aom.push(*frame_id, POSE_VEL_BIAS_SIZE).unwrap();
    }

    let used: BTreeSet<FrameId> = [KF0].into_iter().collect();
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        marg: Some(&window.marg),
        imu: None,
        used_frames: Some(&used),
        lost_landmarks: None,
        fixed_frames: None,
    };
    let mut lqr: LinearizationAbsQR<f64> = LinearizationAbsQR::new(
        &window.estimator,
        &aom,
        LinearizationOptions::default(),
        &inputs,
    )
    .unwrap();
    lqr.linearize_problem(&window.estimator, &inputs).unwrap();
    lqr.perform_qr().unwrap();
    lqr.get_dense_q2jp_q2r(&window.estimator, &inputs).unwrap()
}
