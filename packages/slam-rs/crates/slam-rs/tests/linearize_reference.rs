//! basalt's own linearization tests, ported.
//!
//! `test/src/test_linearization.cpp` has four tests and
//! `test/src/test_qr.cpp` three. Every one of them checks `ABS_QR` against a
//! *second implementation* — `LinearizationAbsSC`, `LinearizationRelSC`, or
//! Eigen's `HouseholderQR` and `LLT`. Only `ABS_QR` is in this port
//! (decision D13), so the second implementation has to come from somewhere else:
//!
//! * **`VoNoMargLinearizationTest`, `VoMargLinearizationTest`** compare the
//!   error, `H` and `b` of the three linearizations at 1e-8
//!   (`test_linearization.cpp:147-161`, `:211-225`). Here the reference is
//!   [`dense_schur_reference`] below: the whole problem written out as one dense
//!   `[J_p | J_l]` per landmark and eliminated with an explicit Schur complement
//!   `H = J_pᵀJ_p − J_pᵀJ_l (J_lᵀJ_l)⁻¹ J_lᵀJ_p`. That is what
//!   `LinearizationAbsSC` computes; it is 40 lines here and it shares no code
//!   with the thing under test, which is the whole point of the C++ test.
//! * **`VoMargBacksubstituteTest`** additionally solves `inc = −H⁻¹b` and
//!   compares the three `l_diff` values (`:290-297`). The reference `l_diff` is
//!   the model cost change of the same dense system, with the landmark
//!   increments recovered by Schur back-substitution.
//! * **`VoMargSqrtLinearizationTest`** is the square-root identity
//!   `Q₂ᵀJ_p` ᵀ`Q₂ᵀJ_p = H` and `Q₂ᵀJ_p` ᵀ`Q₂ᵀr = b` at 1e-3 / 1e-5
//!   (`:379-388`). That one needs no second implementation at all and is ported
//!   as it stands, with basalt's own tolerances.
//! * **`QRvsLLT`, `QRvsLLTRankDef`** (`test_qr.cpp:9-35`) print `R` from a
//!   Householder QR next to `Lᵀ` from the Cholesky of `JᵀJ` and assert nothing.
//!   Ported as the assertion they demonstrate: the two agree row by row up to a
//!   sign, and the port's own [`slam_rs`] Householder is what builds `R`.
//! * **`RankDefLeastSquares`** (`:38-119`) is a marginalization test: it drives
//!   `MargHelper::marginalizeHelperSq*`, which is stage S7. **Not ported here**;
//!   it belongs with the code it tests.
//!
//! The problem itself is basalt's (`test_linearization.cpp:9-76`): six frames,
//! two cameras sharing `KannalaBrandtCamera4::getTestProjections()[0]`, ten
//! landmarks hosted per frame, every landmark observed in **every** image, a
//! pixel of noise on each observation, `huber_thresh = 0.5`,
//! `obs_std_dev = 2.0`. Eigen's `Random()` is replaced by a seeded xorshift, for
//! the same reason the IMU tests replace it: `Random()` never reseeds, so the
//! C++ test is deterministic too, and a Rust test that flakes is worse than one
//! that is merely differently arbitrary.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use nalgebra::{DMatrix, DVector, Vector2, Vector3, Vector4, Vector6};
use slam_rs::ba_base::BaError;
use slam_rs::ba_base::{BundleAdjustmentBase, LinearizePointOut, linearize_point};
use slam_rs::calib::Calibration;
use slam_rs::imu::{ImuLinData, ImuSample, IntegratedImuMeasurement};
use slam_rs::landmark::{Landmark, StereographicParam};
use slam_rs::lie::{Se3, So3};
use slam_rs::linearize::{
    ImuInput, LandmarkBlockOptions, LinearizationAbsQR, LinearizationInputs, LinearizationOptions,
    LinearizeError,
};
use slam_rs::types::{
    AbsOrderMap, LandmarkId, MargLinData, POSE_SIZE, PoseStateWithLin, TimeCamId,
};

mod common;
use common::{Rng, test_calibration};

/// The window `get_vo_estimator` builds (`test_linearization.cpp:9-76`), plus
/// the ordering it fills in.
struct Problem {
    estimator: BundleAdjustmentBase<f64>,
    aom: AbsOrderMap,
    marg: Option<MargLinData<f64>>,
}

/// `get_vo_estimator(num_frames, estimator, aom)` (`:9-76`).
fn vo_problem(num_frames: usize, seed: u64) -> Problem {
    let mut rng: Rng = Rng::new(seed);

    // `:15-22`: the two camera-to-IMU transforms and both intrinsics.
    let calib: Calibration<f64> = test_calibration(&mut rng);

    // `:25-26`: the 3-D points, five metres in front.
    let points: Vec<Vector3<f64>> = (0..num_frames * 10)
        .map(|_| {
            let mut p: Vector3<f64> = rng.vector3();
            p[2] += 5.0;
            p
        })
        .collect();

    // `:31-40`: the poses.
    let mut estimator: BundleAdjustmentBase<f64> =
        BundleAdjustmentBase::new(calib, 2.0, 0.5).unwrap();
    let mut aom: AbsOrderMap = AbsOrderMap::new();
    let mut poses: Vec<Se3<f64>> = Vec::new();
    for i in 0..num_frames {
        let mut t_w_i: Se3<f64> = Se3::identity();
        t_w_i.rotation = So3::exp(&(rng.vector3() / 100.0));
        t_w_i.translation[0] = i as f64 * 0.1;
        poses.push(t_w_i);
        aom.push(i as i64, POSE_SIZE).unwrap();
        estimator
            .frame_poses
            .insert(i as i64, PoseStateWithLin::new(i as i64, t_w_i, false));
    }

    // `:42-74`: ten landmarks hosted per frame, each seen in every image.
    for i in 0..num_frames {
        for j in 0..10 {
            let kp_idx: usize = 10 * i + j;
            let p3d: Vector3<f64> = points[kp_idx];

            // `:47-52`: the landmark in its host camera's frame.
            let t_c_w: Se3<f64> = (poses[i] * estimator.calib.t_i_c[0]).inverse();
            let p3d_cam: Vector3<f64> = t_c_w * p3d;
            let id: LandmarkId = LandmarkId(kp_idx as u64);
            let landmark: Landmark<f64> = Landmark::new(
                id,
                TimeCamId::new(i as i64, 0),
                StereographicParam::project(&Vector4::new(p3d_cam[0], p3d_cam[1], p3d_cam[2], 1.0)),
                1.0 / p3d_cam.norm(),
            );
            estimator.lmdb.add_landmark(id, &landmark);

            // `:56-73`: an observation in every image of every frame.
            for (f, pose) in poses.iter().enumerate() {
                for c in 0..2 {
                    let t_c_w: Se3<f64> = (*pose * estimator.calib.t_i_c[c]).inverse();
                    let p_cam: Vector3<f64> = t_c_w * p3d;
                    let mut pixel: Vector2<f64> = Vector2::zeros();
                    let mut jacobian = nalgebra::Matrix2x4::zeros();
                    let ok: bool = estimator.cameras()[c].project_with_jacobian(
                        &Vector4::new(p_cam[0], p_cam[1], p_cam[2], 1.0),
                        &mut pixel,
                        &mut jacobian,
                    );
                    assert!(ok, "the synthetic problem must project");
                    // `:65`: `Random() / 100` pixels of noise.
                    pixel[0] += rng.symmetric() / 100.0;
                    pixel[1] += rng.symmetric() / 100.0;
                    estimator
                        .lmdb
                        .add_observation(TimeCamId::new(f as i64, c), id, pixel)
                        .unwrap();
                }
            }
        }
    }

    Problem {
        estimator,
        aom,
        marg: None,
    }
}

/// `get_vo_estimator_with_marg` (`test_linearization.cpp:78-100`): the same
/// window, plus a prior over the first two frames and both of them frozen at
/// their linearization point with a non-zero delta.
fn vo_problem_with_marg(num_frames: usize, seed: u64) -> Problem {
    let mut problem: Problem = vo_problem(num_frames, seed);
    let mut rng: Rng = Rng::new(seed ^ 0x5eed);

    let marg_size: usize = 2 * POSE_SIZE;
    // `:85-89`: `H = 1e6 I`, `b = 10 * Random()`.
    let mut h: DMatrix<f64> = DMatrix::identity(marg_size, marg_size);
    h *= 1e6;
    let b: DVector<f64> =
        DVector::from_iterator(marg_size, (0..marg_size).map(|_| rng.symmetric() * 10.0));

    // `:91-93`.
    let mut order: AbsOrderMap = AbsOrderMap::new();
    order.push(0, POSE_SIZE).unwrap();
    order.push(1, POSE_SIZE).unwrap();

    // `:95-99`: freeze, then drift.
    for frame_id in [0i64, 1] {
        let pose = problem.estimator.frame_poses.get_mut(&frame_id).unwrap();
        pose.set_linearized().unwrap();
        pose.apply_inc(&(rng.vector6() / 100.0));
    }

    problem.marg = Some(MargLinData { order, h, b });
    problem
}

/// The second implementation: one dense `[J_p | J_l]` per landmark, eliminated
/// with an explicit Schur complement.
///
/// This is what `LinearizationAbsSC` computes and it shares no line of code with
/// [`LinearizationAbsQR`]. Returns `(error, H, b, l_diff_of)`, where the last is
/// a closure that gives the model cost change for a pose increment.
#[allow(clippy::type_complexity)]
fn dense_schur_reference(
    problem: &Problem,
) -> (
    f64,
    DMatrix<f64>,
    DVector<f64>,
    Box<dyn Fn(&DVector<f64>) -> f64>,
) {
    let estimator: &BundleAdjustmentBase<f64> = &problem.estimator;
    let total: usize = problem.aom.total_size();
    let mut h: DMatrix<f64> = DMatrix::zeros(total, total);
    let mut b: DVector<f64> = DVector::zeros(total);
    let mut error: f64 = 0.0;
    let huber: f64 = estimator.huber_thresh;
    let sigma: f64 = estimator.obs_std_dev;

    // Everything a landmark contributes, kept so the cost-change closure can
    // replay it without rebuilding the Jacobians.
    let mut per_landmark: Vec<(DMatrix<f64>, DMatrix<f64>, DVector<f64>)> = Vec::new();

    for lm in estimator.lmdb.landmarks() {
        let rows: usize = 2 * lm.obs.len();
        let mut jp: DMatrix<f64> = DMatrix::zeros(rows, total);
        let mut jl: DMatrix<f64> = DMatrix::zeros(rows, 3);
        let mut r: DVector<f64> = DVector::zeros(rows);

        for (i, (tcid_t, kpt_obs)) in lm.obs.iter().enumerate() {
            let host: TimeCamId = lm.host_kf_id;
            let state_h = estimator.get_pose_state_with_lin(host.frame_id).unwrap();
            let state_t = estimator.get_pose_state_with_lin(tcid_t.frame_id).unwrap();
            let t_i_c_h: Se3<f64> = estimator.calib.t_i_c[host.cam_id];
            let t_i_c_t: Se3<f64> = estimator.calib.t_i_c[tcid_t.cam_id];

            let (t_t_h, d_rel_d_h, d_rel_d_t) = if host == *tcid_t {
                (
                    nalgebra::Matrix4::identity(),
                    nalgebra::Matrix6::zeros(),
                    nalgebra::Matrix6::zeros(),
                )
            } else {
                let mut d_h = nalgebra::Matrix6::zeros();
                let mut d_t = nalgebra::Matrix6::zeros();
                let lin: Se3<f64> = slam_rs::ba_base::compute_rel_pose(
                    state_h.pose_lin(),
                    &t_i_c_h,
                    state_t.pose_lin(),
                    &t_i_c_t,
                    Some(&mut d_h),
                    Some(&mut d_t),
                );
                let value: Se3<f64> = if state_h.is_linearized() || state_t.is_linearized() {
                    slam_rs::ba_base::compute_rel_pose(
                        state_h.pose(),
                        &t_i_c_h,
                        state_t.pose(),
                        &t_i_c_t,
                        None,
                        None,
                    )
                } else {
                    lin
                };
                (value.matrix(), d_h, d_t)
            };

            let mut res: Vector2<f64> = Vector2::zeros();
            let mut d_res_d_xi = nalgebra::Matrix2x6::zeros();
            let mut d_res_d_p = nalgebra::Matrix2x3::zeros();
            let valid: bool = linearize_point(
                kpt_obs,
                lm,
                &t_t_h,
                &estimator.cameras()[tcid_t.cam_id],
                &mut res,
                &mut LinearizePointOut {
                    d_res_d_xi: Some(&mut d_res_d_xi),
                    d_res_d_p: Some(&mut d_res_d_p),
                    proj: None,
                },
            );
            if !valid {
                continue;
            }

            // `landmark_block_abs_dynamic.hpp:456-470` and `:168-179`.
            let res_squared: f64 = res.norm_squared();
            let weight: f64 = if res_squared <= huber * huber {
                1.0
            } else {
                huber / res_squared.sqrt()
            };
            error += 0.5 * (2.0 - weight) * weight * res_squared / (sigma * sigma);
            let sqrt_weight: f64 = weight.sqrt() / sigma;

            let (h_idx, _) = problem.aom.get(host.frame_id).unwrap();
            let (t_idx, _) = problem.aom.get(tcid_t.frame_id).unwrap();
            let host_block = (sqrt_weight * d_res_d_xi) * d_rel_d_h;
            let target_block = (sqrt_weight * d_res_d_xi) * d_rel_d_t;
            for row in 0..2 {
                for col in 0..POSE_SIZE {
                    jp[(2 * i + row, h_idx + col)] += host_block[(row, col)];
                    jp[(2 * i + row, t_idx + col)] += target_block[(row, col)];
                }
                for col in 0..3 {
                    jl[(2 * i + row, col)] = sqrt_weight * d_res_d_p[(row, col)];
                }
                r[2 * i + row] = sqrt_weight * res[row];
            }
        }

        // The Schur complement.
        let h_ll: nalgebra::Matrix3<f64> = (jl.transpose() * &jl).fixed_view::<3, 3>(0, 0).into();
        let h_ll_inv: nalgebra::Matrix3<f64> = h_ll.try_inverse().expect("J_l is full rank here");
        let h_pl: DMatrix<f64> = jp.transpose() * &jl;
        let b_l: Vector3<f64> = (jl.transpose() * &r).fixed_view::<3, 1>(0, 0).into();
        h += jp.transpose() * &jp - &h_pl * h_ll_inv * h_pl.transpose();
        b += jp.transpose() * &r - &h_pl * (h_ll_inv * b_l);
        per_landmark.push((jp, jl, r));
    }

    if let Some(marg) = &problem.marg {
        error += estimator
            .linearize_marg_prior(marg, &problem.aom, &mut h, &mut b)
            .unwrap();
    }

    // The model cost change of the same dense system: recover each landmark's
    // increment by Schur back-substitution, then
    // `l_diff = -(J inc)ᵀ (r + 0.5 (J inc))`
    // (`landmark_block_abs_dynamic.hpp:276-320`).
    let marg: Option<MargLinData<f64>> = problem.marg.clone();
    let delta: Option<DVector<f64>> = marg
        .as_ref()
        .map(|m| estimator.compute_delta(&m.order).unwrap());
    let l_diff_of = move |pose_inc: &DVector<f64>| -> f64 {
        let mut l_diff: f64 = 0.0;
        for (jp, jl, r) in &per_landmark {
            let h_ll: nalgebra::Matrix3<f64> =
                (jl.transpose() * jl).fixed_view::<3, 3>(0, 0).into();
            let h_ll_inv: nalgebra::Matrix3<f64> = h_ll.try_inverse().unwrap();
            let rhs: Vector3<f64> = (jl.transpose() * (r + jp * pose_inc))
                .fixed_view::<3, 1>(0, 0)
                .into();
            let inc_l: Vector3<f64> = -(h_ll_inv * rhs);
            let jinc: DVector<f64> = jp * pose_inc + jl * inc_l;
            l_diff -= (jinc.transpose() * (0.5 * &jinc + r))[(0, 0)];
        }
        if let (Some(m), Some(delta)) = (marg.as_ref(), delta.as_ref()) {
            let size: usize = m.h.ncols();
            let inc: DVector<f64> = pose_inc.rows(0, size).into_owned();
            let b_jdelta: DVector<f64> = &m.h * delta + &m.b;
            let j_inc: DVector<f64> = &m.h * inc;
            l_diff -= (j_inc.transpose() * (b_jdelta + 0.5 * &j_inc))[(0, 0)];
        }
        l_diff
    };

    (error, h, b, Box::new(l_diff_of))
}

/// Run the port's linearizer over a problem, through `performQR`.
fn linearize(problem: &Problem) -> (f64, LinearizationAbsQR<f64>) {
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        marg: problem.marg.as_ref(),
        ..Default::default()
    };
    let mut lqr: LinearizationAbsQR<f64> = LinearizationAbsQR::new(
        &problem.estimator,
        &problem.aom,
        LinearizationOptions::default(),
        &inputs,
    )
    .unwrap();
    let (error, valid) = lqr.linearize_problem(&problem.estimator, &inputs).unwrap();
    assert!(valid);
    lqr.perform_qr().unwrap();
    (error, lqr)
}

/// `VoNoMargLinearizationTest` (`test_linearization.cpp:103-162`): the error,
/// `H` and `b` agree with a second implementation to 1e-8.
#[test]
fn vo_no_marg_linearization() {
    let problem: Problem = vo_problem(6, 0x1234_5678);
    let (error_qr, lqr) = linearize(&problem);
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs::default();
    let (h_qr, b_qr) = lqr.get_dense_h_b(&problem.estimator, &inputs).unwrap();

    let (error_ref, h_ref, b_ref, _) = dense_schur_reference(&problem);

    // basalt's own tolerances (`:155-157`).
    assert!(
        (error_qr - error_ref).abs() <= 1e-8,
        "{error_qr} vs {error_ref}"
    );
    assert!(
        (&h_qr - &h_ref).norm() <= 1e-8,
        "{}",
        (&h_qr - &h_ref).norm()
    );
    assert!(
        (&b_qr - &b_ref).norm() <= 1e-8,
        "{}",
        (&b_qr - &b_ref).norm()
    );
}

/// `VoMargLinearizationTest` (`:165-227`): the same with a marginalization
/// prior and two frozen linearization points.
#[test]
fn vo_marg_linearization() {
    let problem: Problem = vo_problem_with_marg(6, 0x1234_5678);
    let (error_qr, lqr) = linearize(&problem);
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        marg: problem.marg.as_ref(),
        ..Default::default()
    };
    let (h_qr, b_qr) = lqr.get_dense_h_b(&problem.estimator, &inputs).unwrap();

    let (error_ref, h_ref, b_ref, _) = dense_schur_reference(&problem);

    assert!(
        (error_qr - error_ref).abs() <= 1e-8,
        "{error_qr} vs {error_ref}"
    );
    // The prior's `H` is `1e6 I`, so the absolute norm of `H` is around 1e12 and
    // basalt's own absolute 1e-8 would be a 1e-20 relative demand. Scaled.
    let scale: f64 = h_ref.norm().max(1.0);
    assert!(
        (&h_qr - &h_ref).norm() <= 1e-8 * scale,
        "{}",
        (&h_qr - &h_ref).norm() / scale
    );
    assert!(
        (&b_qr - &b_ref).norm() <= 1e-8 * b_ref.norm().max(1.0),
        "{}",
        (&b_qr - &b_ref).norm()
    );
}

/// `VoMargBacksubstituteTest` (`:230-308`): solve, back-substitute, and compare
/// the model cost change against the second implementation.
#[test]
fn vo_marg_backsubstitute() {
    let mut problem: Problem = vo_problem_with_marg(6, 0x1234_5678);
    let (_, mut lqr) = linearize(&problem);
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        marg: problem.marg.as_ref(),
        ..Default::default()
    };
    let (h, b) = lqr.get_dense_h_b(&problem.estimator, &inputs).unwrap();

    let (_, _, _, l_diff_of) = dense_schur_reference(&problem);

    // `:290`: `inc = -H.ldlt().solve(b)`.
    let inc: DVector<f64> = -h.clone().lu().solve(&b).expect("the system is solvable");
    let l_diff_ref: f64 = l_diff_of(&inc);

    let error_before: f64 = problem.estimator.compute_error(None, 0.0).unwrap().0;

    // The prior borrow has to end before the estimator is mutated.
    let marg: Option<MargLinData<f64>> = problem.marg.clone();
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        marg: marg.as_ref(),
        ..Default::default()
    };
    let l_diff_qr: f64 = lqr
        .back_substitute(&mut problem.estimator, &inputs, &inc)
        .unwrap();

    assert!(
        (l_diff_qr - l_diff_ref).abs() <= 1e-8 * l_diff_ref.abs().max(1.0),
        "l_diff {l_diff_qr} vs {l_diff_ref}"
    );

    // The increment really does help: apply it to the poses too and the
    // reprojection error drops. `:299-306` compares `computeError` across the
    // three estimators; with one estimator, the meaningful statement is that the
    // step the model predicted a decrease for delivers one.
    assert!(l_diff_qr > 0.0, "the solved step must predict a decrease");
    for (frame_id, offset, _) in problem.aom.iter().collect::<Vec<_>>() {
        let pose = problem.estimator.frame_poses.get_mut(&frame_id).unwrap();
        pose.apply_inc(&Vector6::from_iterator(
            (0..POSE_SIZE).map(|k| inc[offset + k]),
        ));
    }
    let error_after: f64 = problem.estimator.compute_error(None, 0.0).unwrap().0;
    assert!(
        error_after < error_before,
        "error {error_before} -> {error_after}"
    );
}

/// `VoMargSqrtLinearizationTest` (`:311-394`): the square-root identity
/// `Q₂ᵀJ_pᵀ Q₂ᵀJ_p = H` and `Q₂ᵀJ_pᵀ Q₂ᵀr = b`, at basalt's own 1e-3 and 1e-5.
#[test]
fn vo_marg_sqrt_linearization() {
    let problem: Problem = vo_problem_with_marg(6, 0x1234_5678);
    let (_, lqr) = linearize(&problem);
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        marg: problem.marg.as_ref(),
        ..Default::default()
    };
    let (h, b) = lqr.get_dense_h_b(&problem.estimator, &inputs).unwrap();
    let (q2jp, q2r) = lqr.get_dense_q2jp_q2r(&problem.estimator, &inputs).unwrap();

    // `:380-383`.
    let h_diff: f64 = (q2jp.transpose() * &q2jp - &h).norm();
    let b_diff: f64 = (q2jp.transpose() * &q2r - &b).norm();
    assert!(h_diff <= 1e-3, "H differs by {h_diff}");
    assert!(b_diff <= 1e-5, "b differs by {b_diff}");
}

/// `QRvsLLT` and `QRvsLLTRankDef` (`test_qr.cpp:9-35`), as the assertion they
/// demonstrate: for a `10 x 6` `J`, the `R` of a Householder QR and the `Lᵀ` of
/// the Cholesky of `JᵀJ` agree row by row up to a sign.
///
/// The QR is built from this port's own [`slam_rs`] Householder primitives, so
/// the test is on the port, not on nalgebra.
#[test]
fn householder_qr_matches_the_cholesky_of_the_normal_equations() {
    for (case, rank_deficient) in [("full rank", false), ("rank deficient", true)] {
        let mut rng: Rng = Rng::new(0xfeed_1234);
        let mut j: DMatrix<f64> = DMatrix::zeros(10, 6);
        for r in 0..10 {
            for c in 0..6 {
                j[(r, c)] = rng.symmetric();
            }
        }
        if rank_deficient {
            // `test_qr.cpp:26`: `J.col(2) = J.col(4)`.
            let col4: DVector<f64> = j.column(4).into_owned();
            j.set_column(2, &col4);
        }

        // The port has no standalone QR — the landmark block eliminates exactly
        // three columns — so the reflections are driven here the way
        // `performQRHouseholder` drives them (`landmark_block_abs_dynamic.hpp:445-453`).
        let r_qr: DMatrix<f64> = householder_r(&j);
        let ata: DMatrix<f64> = j.transpose() * &j;

        if rank_deficient {
            // `JᵀJ` is singular, so there is no Cholesky; what survives is that
            // the QR still produces an upper-triangular `R` with `RᵀR = JᵀJ`,
            // which is the point the C++ test prints.
            let rtr: DMatrix<f64> = r_qr.transpose() * &r_qr;
            assert!(
                (&rtr - &ata).norm() <= 1e-9 * ata.norm(),
                "{case}: RᵀR != JᵀJ"
            );
            continue;
        }

        let llt = ata.clone().cholesky().expect("full rank");
        let u: DMatrix<f64> = llt.l().transpose();
        for row in 0..6 {
            // Up to the sign of the row, which is all a QR is free in.
            let sign: f64 = if r_qr[(row, row)] * u[(row, row)] < 0.0 {
                -1.0
            } else {
                1.0
            };
            for col in 0..6 {
                let got: f64 = sign * r_qr[(row, col)];
                assert!(
                    (got - u[(row, col)]).abs() <= 1e-9 * u.norm(),
                    "{case}: R[{row},{col}] = {got}, Lᵀ = {}",
                    u[(row, col)]
                );
            }
        }
    }
}

/// The upper-triangular factor of a Householder QR, driven exactly as
/// `performQRHouseholder` drives it, through the port's own primitives.
fn householder_r(j: &DMatrix<f64>) -> DMatrix<f64> {
    let rows: usize = j.nrows();
    let cols: usize = j.ncols();
    let mut work: DMatrix<f64> = j.clone();
    for k in 0..cols {
        slam_rs::linearize::reflect_column(&mut work, k, k, rows - k).unwrap();
    }
    let mut r: DMatrix<f64> = DMatrix::zeros(cols, cols);
    for row in 0..cols {
        for col in row..cols {
            r[(row, col)] = work[(row, col)];
        }
    }
    r
}

// ── properties ─────────────────────────────────────────────────────────────

/// `H = Q₂ᵀJ_pᵀ Q₂ᵀJ_p` is symmetric and positive semi-definite by
/// construction. A sign slip anywhere in the QR shows up here.
#[test]
fn the_reduced_hessian_is_symmetric_and_positive_semidefinite() {
    for seed in [1u64, 2, 3, 4] {
        let problem: Problem = vo_problem(3, seed.wrapping_mul(0x9E37_79B9));
        let (_, lqr) = linearize(&problem);
        let inputs: LinearizationInputs<'_, f64> = LinearizationInputs::default();
        let (h, _) = lqr.get_dense_h_b(&problem.estimator, &inputs).unwrap();

        let asymmetry: f64 = (&h - h.transpose()).norm();
        assert!(asymmetry <= 1e-9 * h.norm(), "asymmetry {asymmetry}");

        let eigenvalues = h.symmetric_eigenvalues();
        let smallest: f64 = eigenvalues.iter().fold(f64::INFINITY, |m, v| m.min(*v));
        assert!(
            smallest >= -1e-7 * h.norm(),
            "smallest eigenvalue {smallest} on a matrix of norm {}",
            h.norm()
        );
    }
}

/// The `Q₂` rows are the null space of the landmark columns: after the QR,
/// `Q₂ᵀJ_l = 0` (ICCV 2021 §3.3, CVPR 2021 Eq. (15)). This is what makes the
/// reduced camera system exact rather than an approximation.
#[test]
fn the_q2_rows_are_orthogonal_to_the_landmark_columns() {
    let problem: Problem = vo_problem(3, 0xabcd_ef01);
    let (_, lqr) = linearize(&problem);
    for block in lqr.landmark_blocks() {
        let (num_rows, _, _, lm_idx, _) = block.layout();
        let storage: &DMatrix<f64> = block.storage();
        // Rows 3 .. num_rows - 3 are `Q₂`; the last three are the (zero)
        // damping rows.
        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for row in 0..num_rows {
            for col in 0..3 {
                let v: f64 = storage[(row, lm_idx + col)].abs();
                if row < 3 {
                    scale = scale.max(v);
                } else {
                    worst = worst.max(v);
                }
            }
        }
        assert!(
            worst <= 1e-9 * scale.max(1.0),
            "Q2^T J_l has a coefficient of {worst} against a scale of {scale}"
        );
    }
}

/// `backSubstitute` returns the *linearized* cost decrease, and it is not the
/// reduced system's decrease alone.
///
/// For one block with the orthogonal `Q = [Q₁, Q₂]` and the optimal landmark
/// increment `R inc_l = −(Q₁ᵀr + Q₁ᵀJ_p inc_p)`, the first three rows of
/// `Qᵀ J inc` are `−Q₁ᵀr` — so the *updated residual* `Qᵀ J inc + Qᵀ r` is zero
/// there, which is what "the landmarks move to their own optimum" means — and
///
/// ```text
/// l_diff = 0.5 ‖Q₁ᵀr‖²  −  inc_pᵀ b  −  0.5 inc_pᵀ H inc_p
/// ```
///
/// with `H` and `b` the reduced camera system. The first term does not depend on
/// the pose increment at all: it is what the landmarks gain by moving to their
/// own optimum, and it is why basalt's `l_diff` is **nonnegative at `inc = 0`**
/// — zero exactly when the eliminated residual `Q₁ᵀr` already is.
/// The identity is `landmark_block_abs_dynamic.hpp:276-320` written out, and
/// getting the constant term wrong would make every Levenberg-Marquardt gain
/// ratio wrong in the same direction — which is exactly the kind of bug that
/// still converges, only worse.
#[test]
fn back_substitution_reduces_the_linearized_cost() {
    let mut problem: Problem = vo_problem(3, 0x0f0f_0f0f);
    let (_, mut lqr) = linearize(&problem);
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs::default();
    let (h, b) = lqr.get_dense_h_b(&problem.estimator, &inputs).unwrap();

    // The landmarks' own gain, `0.5 ‖Q₁ᵀr‖²` summed over the blocks.
    let landmark_gain: f64 = lqr
        .landmark_blocks()
        .iter()
        .map(|block| {
            let (_, _, _, _, res_idx) = block.layout();
            (0..3)
                .map(|row| 0.5 * block.storage()[(row, res_idx)].powi(2))
                .sum::<f64>()
        })
        .sum();

    // The Gauss-Newton step, and basalt's sign convention: the solve gives
    // `H inc = b`, and the increment applied is its negation
    // (`sqrt_keypoint_vio.cpp:1450`).
    let inc: DVector<f64> = -h.clone().lu().solve(&b).expect("solvable");
    let quadratic: f64 =
        -(inc.transpose() * &b)[(0, 0)] - 0.5 * (inc.transpose() * &h * &inc)[(0, 0)];
    assert!(
        quadratic > 0.0,
        "the Gauss-Newton step must reduce the model"
    );

    let l_diff: f64 = lqr
        .back_substitute(&mut problem.estimator, &inputs, &inc)
        .unwrap();

    let want: f64 = quadratic + landmark_gain;
    assert!(
        (l_diff - want).abs() <= 1e-9 * want.abs(),
        "l_diff {l_diff} vs {want} (quadratic {quadratic} + landmark gain {landmark_gain})"
    );

    // And a zero increment leaves exactly the landmarks' own gain. On its own
    // window: the call above already moved this one's landmarks.
    let mut untouched: Problem = vo_problem(3, 0x0f0f_0f0f);
    let (_, mut fresh) = linearize(&untouched);
    let gain_zero: f64 = fresh
        .landmark_blocks()
        .iter()
        .map(|block| {
            let (_, _, _, _, res_idx) = block.layout();
            (0..3)
                .map(|row| 0.5 * block.storage()[(row, res_idx)].powi(2))
                .sum::<f64>()
        })
        .sum();
    let zero: DVector<f64> = DVector::zeros(untouched.aom.total_size());
    let l_diff_zero: f64 = fresh
        .back_substitute(&mut untouched.estimator, &inputs, &zero)
        .unwrap();
    assert!(
        (l_diff_zero - gain_zero).abs() <= 1e-9 * gain_zero.abs(),
        "l_diff(0) {l_diff_zero} vs {gain_zero}"
    );
}

/// First-estimate Jacobians (`linearization_abs_qr.cpp:229-232`, trap 7): once a
/// frame is frozen, moving its current state changes the relative pose's
/// **value** and leaves `d_rel_d_h` and `d_rel_d_t` exactly as they were.
///
/// Note what this does **not** say. `d_res_d_xi` comes out of `linearizePoint`
/// against the re-evaluated `T_t_h` (`landmark_block_abs_dynamic.hpp:148`), so
/// the block's own Jacobian columns do move; what is frozen is the pose
/// composition, which is the only place basalt applies the first-estimate rule.
/// A test that demanded frozen block columns would be testing a property basalt
/// does not have.
#[test]
fn freezing_a_state_freezes_its_pose_jacobians_but_not_its_residuals() {
    let base: Problem = vo_problem_with_marg(4, 0x2222_3333);
    let (_, lqr_before) = linearize(&base);

    // The same window with a larger drift on the two frozen frames.
    let mut moved: Problem = vo_problem_with_marg(4, 0x2222_3333);
    for frame_id in [0i64, 1] {
        let pose = moved.estimator.frame_poses.get_mut(&frame_id).unwrap();
        assert!(pose.is_linearized());
        pose.apply_inc(&Vector6::from_iterator(
            (0..6).map(|k| 0.01 * (k as f64 + 1.0)),
        ));
    }
    let (_, lqr_after) = linearize(&moved);

    let mut any_value_moved: bool = false;
    for (before, after) in lqr_before
        .relative_poses()
        .iter()
        .zip(lqr_after.relative_poses().iter())
    {
        assert_eq!(
            before.d_rel_d_h, after.d_rel_d_h,
            "d_rel_d_h moved with the current state"
        );
        assert_eq!(
            before.d_rel_d_t, after.d_rel_d_t,
            "d_rel_d_t moved with the current state"
        );
        if (before.t_t_h - after.t_t_h).norm() > 1e-6 {
            any_value_moved = true;
        }
    }
    assert!(
        any_value_moved,
        "the drift did not move a single relative pose, so the test proved nothing"
    );

    // And the residual column of at least one block moved with it.
    let mut any_residual_moved: bool = false;
    for (before, after) in lqr_before
        .landmark_blocks()
        .iter()
        .zip(lqr_after.landmark_blocks().iter())
    {
        let (num_rows, _, _, _, res_idx) = before.layout();
        for row in 0..num_rows {
            if (before.storage()[(row, res_idx)] - after.storage()[(row, res_idx)]).abs() > 1e-6 {
                any_residual_moved = true;
            }
        }
    }
    assert!(any_residual_moved, "no residual moved");
}

/// The four fixed-order reductions are order-stable: linearizing the same
/// problem twice returns bit-identical errors and increments (decision D31).
#[test]
fn the_reductions_are_reproducible() {
    let problem: Problem = vo_problem_with_marg(4, 0x7777_8888);
    let (error_a, lqr_a) = linearize(&problem);
    let (error_b, lqr_b) = linearize(&problem);
    assert_eq!(error_a.to_bits(), error_b.to_bits());
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        marg: problem.marg.as_ref(),
        ..Default::default()
    };
    let (h_a, b_a) = lqr_a.get_dense_h_b(&problem.estimator, &inputs).unwrap();
    let (h_b, b_b) = lqr_b.get_dense_h_b(&problem.estimator, &inputs).unwrap();
    for (a, b) in h_a.iter().zip(h_b.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in b_a.iter().zip(b_b.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// The square-root export writes the prior into the *first* columns of the
/// stacked system (`linearization_abs_qr.cpp:587-589`), so it is only correct
/// when the prior's ordering is the window's prefix. C++ checks that in the
/// Hessian path (`ba_base.cpp:383-388`) and not in the square-root one, where a
/// disagreeing ordering silently attaches one frame's columns to another.
#[test]
fn a_prior_ordering_that_disagrees_is_refused_by_both_exports() {
    let mut problem: Problem = vo_problem_with_marg(4, 0x3333_4444);

    // The window has frame 0 at offset 0 and frame 1 at offset 6; give the
    // prior the same two frames the other way round.
    let mut reversed: AbsOrderMap = AbsOrderMap::new();
    reversed.push(1, POSE_SIZE).unwrap();
    reversed.push(0, POSE_SIZE).unwrap();
    if let Some(marg) = problem.marg.as_mut() {
        marg.order = reversed;
    }

    let (_, lqr) = linearize(&problem);
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        marg: problem.marg.as_ref(),
        ..Default::default()
    };

    // The Hessian path already refused it...
    assert!(matches!(
        lqr.get_dense_h_b(&problem.estimator, &inputs),
        Err(LinearizeError::Ba(BaError::MargOrderMismatch { .. }))
    ));
    // ...and now so does the square-root path.
    assert!(matches!(
        lqr.get_dense_q2jp_q2r(&problem.estimator, &inputs),
        Err(LinearizeError::Ba(BaError::MargOrderMismatch { .. }))
    ));
}

/// An IMU factor is fifteen columns wide at each end (`imu_block.hpp:21`), and
/// C++ discards the ordering's block sizes (`linearization_abs_qr.cpp:163-167`)
/// and computes `start_t + dt` unchecked (`imu_block.hpp:30`). With pose-sized
/// slots the scatter writes over the neighbouring state; the port refuses at
/// construction, where it costs nothing.
#[test]
fn an_imu_factor_over_pose_sized_slots_is_refused() {
    // A one-nanosecond measurement, so its endpoints are the two frames
    // `vo_problem` numbers 0 and 1.
    let noise: Vector3<f64> = Vector3::repeat(1e-4);
    let mut meas: IntegratedImuMeasurement<f64> =
        IntegratedImuMeasurement::new(0, &Vector3::zeros(), &Vector3::zeros());
    meas.integrate(
        &ImuSample {
            t_ns: 1,
            gyro: Vector3::new(0.01, -0.02, 0.03),
            accel: Vector3::new(0.0, 0.0, 9.81),
        },
        &noise,
        &noise,
    )
    .unwrap();
    assert_eq!(meas.get_dt_ns(), 1);

    let problem: Problem = vo_problem(2, 0x4444_5555);
    assert_eq!(problem.aom.get(0), Some((0, POSE_SIZE)));
    assert_eq!(problem.aom.get(1), Some((POSE_SIZE, POSE_SIZE)));

    let imu: ImuInput<'_, f64> = ImuInput {
        lin_data: ImuLinData {
            g: Vector3::new(0.0, 0.0, -9.81),
            gyro_bias_weight_sqrt: Vector3::repeat(100.0),
            accel_bias_weight_sqrt: Vector3::repeat(100.0),
        },
        measurements: vec![(0, &meas)],
    };
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        imu: Some(&imu),
        ..Default::default()
    };
    let err = LinearizationAbsQR::new(
        &problem.estimator,
        &problem.aom,
        LinearizationOptions::default(),
        &inputs,
    )
    .unwrap_err();
    assert_eq!(
        err,
        LinearizeError::ImuStateNotFullSize {
            frame: 0,
            size: POSE_SIZE
        },
        "a six-column slot must not carry a fifteen-column factor"
    );

    // And an end timestamp that does not fit in an `i64`.
    let overflowing: ImuInput<'_, f64> = ImuInput {
        lin_data: imu.lin_data,
        measurements: vec![(i64::MAX, &meas)],
    };
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        imu: Some(&overflowing),
        ..Default::default()
    };
    assert_eq!(
        LinearizationAbsQR::new(
            &problem.estimator,
            &problem.aom,
            LinearizationOptions::default(),
            &inputs,
        )
        .unwrap_err(),
        LinearizeError::ImuIntervalOverflow {
            start: i64::MAX,
            dt_ns: 1
        }
    );

    // A frame the window does not have at all is still the older error.
    let unknown: ImuInput<'_, f64> = ImuInput {
        lin_data: imu.lin_data,
        measurements: vec![(7, &meas)],
    };
    let inputs: LinearizationInputs<'_, f64> = LinearizationInputs {
        imu: Some(&unknown),
        ..Default::default()
    };
    assert!(matches!(
        LinearizationAbsQR::new(
            &problem.estimator,
            &problem.aom,
            LinearizationOptions::default(),
            &inputs,
        ),
        Err(LinearizeError::UnknownImuFrames { start: 7, end: 8 })
    ));
}

/// The block layout arithmetic of `landmark_block_abs_dynamic.hpp:83-96` on
/// every ordering size, not just the ones the fixture happens to use.
#[test]
fn the_block_layout_follows_the_padding_rule() {
    for frames in 1..=5usize {
        let problem: Problem = vo_problem(frames, 0x1111 + frames as u64);
        let inputs: LinearizationInputs<'_, f64> = LinearizationInputs::default();
        let lqr: LinearizationAbsQR<f64> = LinearizationAbsQR::new(
            &problem.estimator,
            &problem.aom,
            LinearizationOptions {
                lb_options: LandmarkBlockOptions::default(),
            },
            &inputs,
        )
        .unwrap();
        let total: usize = problem.aom.total_size();
        for block in lqr.landmark_blocks() {
            let (num_rows, num_cols, padding_idx, lm_idx, res_idx) = block.layout();
            assert_eq!(padding_idx, total);
            assert_eq!(lm_idx, total + (4 - total % 4) % 4);
            assert_eq!(res_idx, lm_idx + 3);
            assert_eq!(num_cols, res_idx + 1);
            assert_eq!(num_cols % 4, 0);
            assert_eq!(num_rows % 2, 1);
            assert_eq!(block.num_q2rows(), num_rows - 3);
        }
    }
}
