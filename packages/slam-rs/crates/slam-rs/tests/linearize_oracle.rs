//! The landmark block and the absolute-pose QR driver against basalt's own
//! C++, matrix for matrix.
//!
//! `fixtures/linearize/linearize_oracle.json` is the output of
//! `tools/linearize_oracle.cpp` on the fork's `slam-rs-reference` branch
//! (target `basalt_linearize_oracle`), built and run out of tree against the
//! basalt sources, since the monorepo never compiles C++ (decision D15).
//!
//! Four problems, each emitted in `f64` and in `f32`:
//!
//! | name | frames | landmarks | what it adds |
//! |---|---|---|---|
//! | `two_frames` | 2 | 3 | the minimum: `padding_size == 0`, 2 to 4 observations per landmark |
//! | `three_frames_huber` | 3 | 4 | `padding_size == 2`, and every third observation is 3 px off so the Huber branch fires |
//! | `marg_fej` | 3 | 4 | a square-root marginalization prior over the first two frames, both frozen at their linearization point with a non-zero delta |
//! | `marg_fej_wide` | 3 | 6 | the same with more landmarks and a different prior |
//!
//! Every case carries the whole problem as plain numbers — the two
//! camera-to-IMU transforms, the intrinsics, the poses with their deltas, the
//! ordering, and every landmark with its observations — so the Rust side
//! rebuilds the identical problem without reading a calibration file, and then:
//!
//! * the **block layout** (`num_rows`, `num_cols`, `padding_idx`, `lm_idx`,
//!   `res_idx`), the arithmetic of `landmark_block_abs_dynamic.hpp:83-96`;
//! * `storage` after `linearizeLandmark` and after `performQR`, so the residual,
//!   the Huber weighting, the two 2x6 pose blocks and all three Householder
//!   reflections are compared coefficient by coefficient;
//! * the per-block `Q2Jp`/`Q2r` and `JᵀJ`/`Jᵀr`;
//! * `backSubstitute`: the model cost change and the landmark parameters it
//!   leaves behind, including the `max(0, inv_dist + inc[2])` projection;
//! * and the whole problem through `LinearizationAbsQR`: the error
//!   (marginalization prior included), the dense `H` and `b`, the stacked
//!   `Q2Jp`/`Q2r` with the prior's re-anchored residual `H delta + b`
//!   (`linearization_abs_qr.cpp:592`, trap 8), and the total `l_diff`.
//!
//! The fixture also carries `storage` after `setLandmarkDamping(lambda)` and
//! after `setLandmarkDamping(0)` undoes it; nothing reads those entries since
//! D68 removed the damping stack the shipped VIO never called. It is left as
//! the C++ emitted it.
//!
//! **Tolerances, and what they measure.** The port reproduces Eigen's formulas
//! (`makeHouseholder`, the triangular solve) coefficient for
//! coefficient, but not Eigen's *product kernels*: `essential.adjoint() * bottom`
//! inside `applyHouseholderOnTheLeft` and `J.transpose() * J` inside
//! `add_dense_H_b` are `gemv`/`gemm` calls whose blocking associates a sum
//! differently from a plain loop. The constants below are the measured
//! agreement, not an aspiration; see the report for the exact figures.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::excessive_precision)]

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector, Quaternion, UnitQuaternion, Vector2, Vector3, Vector6};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use slam_rs::ba_base::BundleAdjustmentBase;
use slam_rs::calib::CameraModel;
use slam_rs::calib::{Calibration, Kb4Params};
use slam_rs::landmark::Landmark;
use slam_rs::lie::{LieScalar, Se3, So3};
use slam_rs::linearize::{
    DenseHbScratch, LandmarkBlock, LinearizationAbsQR, LinearizationInputs, LinearizationOptions,
};
use slam_rs::types::{AbsOrderMap, LandmarkId, MargLinData, PoseStateWithLin, TimeCamId};

use common::Compare;

mod common;

const ORACLE: &str = include_str!("fixtures/linearize/linearize_oracle.json");

/// Agreement with the C++ number in `f64`, relative to the array's own scale.
///
/// The measured worst case over all four problems is **6.3e-16**, three ulps:
/// every formula is Eigen's, and what is left is the association inside Eigen's
/// product kernels, which the port does not reproduce. The constant is two
/// orders looser so a compiler or nalgebra bump does not turn a one-ulp drift
/// into a failure.
const TOLERANCE_F64: f64 = 1e-14;

/// And in `f32`, where the measured worst case is **1.4e-6** — about eleven
/// ulps, and dominated by the sub-diagonal entries the QR drives to zero.
const TOLERANCE_F32: f64 = 5e-6;

#[derive(Debug, Deserialize)]
struct Oracle {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    scalar: String,
    huber_thresh: f64,
    obs_std_dev: f64,
    t_i_c: Vec<Pose>,
    kb4_params: Vec<f64>,
    frames: Vec<Frame>,
    aom: Vec<OrderEntry>,
    landmarks: Vec<OracleLandmark>,
    marg: Option<Marg>,
    pose_inc: Vec<f64>,
    blocks: Vec<Block>,
    problem: Problem,
}

#[derive(Debug, Deserialize)]
struct Pose {
    q: [f64; 4],
    t: [f64; 3],
}

#[derive(Debug, Deserialize)]
struct Frame {
    t_ns: i64,
    q: [f64; 4],
    t: [f64; 3],
    linearized: bool,
    delta: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct OrderEntry {
    t_ns: i64,
    idx: usize,
    size: usize,
}

#[derive(Debug, Deserialize)]
struct OracleLandmark {
    id: u64,
    host: [i64; 2],
    direction: [f64; 2],
    inv_dist: f64,
    obs: Vec<OracleObs>,
}

#[derive(Debug, Deserialize)]
struct OracleObs {
    frame: i64,
    cam: usize,
    pos: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct Marg {
    rows: usize,
    cols: usize,
    order: Vec<OrderEntry>,
    #[serde(rename = "H")]
    h: Vec<f64>,
    b: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Block {
    lm_id: u64,
    num_rows: usize,
    num_cols: usize,
    padding_idx: usize,
    lm_idx: usize,
    res_idx: usize,
    error: f64,
    storage_pre: Vec<f64>,
    storage_post: Vec<f64>,
    q2jp: Vec<f64>,
    q2r: Vec<f64>,
    #[serde(rename = "block_H")]
    block_h: Vec<f64>,
    block_b: Vec<f64>,
    l_diff: f64,
    direction_after: [f64; 2],
    inv_dist_after: f64,
}

#[derive(Debug, Deserialize)]
struct Problem {
    error: f64,
    numerically_valid: bool,
    h_rows: usize,
    #[serde(rename = "H")]
    h: Vec<f64>,
    b: Vec<f64>,
    q2jp_rows: usize,
    q2jp_cols: usize,
    q2jp: Vec<f64>,
    q2r: Vec<f64>,
    l_diff: f64,
    landmarks_after: Vec<LandmarkAfter>,
}

#[derive(Debug, Deserialize)]
struct LandmarkAfter {
    id: u64,
    direction: [f64; 2],
    inv_dist: f64,
}

fn se3_from<S: LieScalar>(q: &[f64; 4], t: &[f64; 3]) -> Se3<S> {
    let quat: Quaternion<S> = Quaternion::new(
        S::from_literal(q[3]),
        S::from_literal(q[0]),
        S::from_literal(q[1]),
        S::from_literal(q[2]),
    );
    Se3::new(
        So3::from_unit_quaternion(UnitQuaternion::new_unchecked(quat)),
        Vector3::new(
            S::from_literal(t[0]),
            S::from_literal(t[1]),
            S::from_literal(t[2]),
        ),
    )
}

/// The window the fixture describes.
struct Rebuilt<S: LieScalar> {
    estimator: BundleAdjustmentBase<S>,
    aom: AbsOrderMap,
    marg: Option<MargLinData<S>>,
    pose_inc: DVector<S>,
}

fn rebuild<S: LieScalar + Serialize + DeserializeOwned>(case: &Case) -> Rebuilt<S> {
    // The msd-index calibration only for the fields the linearizer never reads
    // (resolution, IMU noise); the rig and the intrinsics come from the fixture,
    // which carries `KannalaBrandtCamera4::getTestProjections()[0]` cast to this
    // scalar.
    let mut calib: Calibration<S> =
        Calibration::from_json_str(common::calibration_text("msdmi")).unwrap();
    calib.t_i_c = case
        .t_i_c
        .iter()
        .map(|p| se3_from::<S>(&p.q, &p.t))
        .collect();
    let p: &[f64] = &case.kb4_params;
    calib.intrinsics = vec![
        CameraModel::Kb4(Kb4Params {
            fx: S::from_literal(p[0]),
            fy: S::from_literal(p[1]),
            cx: S::from_literal(p[2]),
            cy: S::from_literal(p[3]),
            k1: S::from_literal(p[4]),
            k2: S::from_literal(p[5]),
            k3: S::from_literal(p[6]),
            k4: S::from_literal(p[7]),
        });
        2
    ];

    let mut estimator: BundleAdjustmentBase<S> = BundleAdjustmentBase::new(
        calib,
        S::from_literal(case.obs_std_dev),
        S::from_literal(case.huber_thresh),
    )
    .unwrap();

    for frame in &case.frames {
        let mut pose: PoseStateWithLin<S> =
            PoseStateWithLin::new(frame.t_ns, se3_from::<S>(&frame.q, &frame.t), false);
        if frame.linearized {
            pose.set_linearized().unwrap();
            let delta: Vector6<S> =
                Vector6::from_iterator(frame.delta.iter().map(|v| S::from_literal(*v)));
            pose.apply_inc(&delta);
        }
        estimator.frame_poses.insert(frame.t_ns, pose);
    }

    let mut aom: AbsOrderMap = AbsOrderMap::new();
    for entry in &case.aom {
        let offset: usize = aom.push(entry.t_ns, entry.size).unwrap();
        assert_eq!(offset, entry.idx, "the ordering disagrees with the fixture");
    }

    for lm in &case.landmarks {
        let id: LandmarkId = LandmarkId(lm.id);
        let host: TimeCamId = TimeCamId::new(lm.host[0], lm.host[1] as usize);
        let landmark: Landmark<S> = Landmark::new(
            id,
            host,
            Vector2::new(
                S::from_literal(lm.direction[0]),
                S::from_literal(lm.direction[1]),
            ),
            S::from_literal(lm.inv_dist),
        );
        estimator.lmdb.add_landmark(id, &landmark);
        for obs in &lm.obs {
            estimator
                .lmdb
                .add_observation(
                    TimeCamId::new(obs.frame, obs.cam),
                    id,
                    Vector2::new(S::from_literal(obs.pos[0]), S::from_literal(obs.pos[1])),
                )
                .unwrap();
        }
    }

    let marg: Option<MargLinData<S>> = case.marg.as_ref().map(|m| {
        let mut order: AbsOrderMap = AbsOrderMap::new();
        for entry in &m.order {
            let offset: usize = order.push(entry.t_ns, entry.size).unwrap();
            assert_eq!(offset, entry.idx);
        }
        // The fixture is row major.
        let mut h: DMatrix<S> = DMatrix::zeros(m.rows, m.cols);
        for r in 0..m.rows {
            for c in 0..m.cols {
                h[(r, c)] = S::from_literal(m.h[r * m.cols + c]);
            }
        }
        MargLinData {
            order,
            h,
            b: DVector::from_iterator(m.rows, m.b.iter().map(|v| S::from_literal(*v))),
        }
    });

    let pose_inc: DVector<S> = DVector::from_iterator(
        case.pose_inc.len(),
        case.pose_inc.iter().map(|v| S::from_literal(*v)),
    );

    Rebuilt {
        estimator,
        aom,
        marg,
        pose_inc,
    }
}

fn check_case<S: LieScalar + Serialize + DeserializeOwned>(case: &Case, tolerance: f64) -> f64 {
    let mut cmp: Compare = Compare::new(tolerance);
    let mut rebuilt: Rebuilt<S> = rebuild::<S>(case);
    let inputs: LinearizationInputs<'_, S> = LinearizationInputs {
        marg: rebuilt.marg.as_ref(),
        ..Default::default()
    };
    let options: LinearizationOptions<S> = LinearizationOptions::default();
    let mut lqr: LinearizationAbsQR<S> =
        LinearizationAbsQR::new(&rebuilt.estimator, &rebuilt.aom, options, &inputs).unwrap();
    // The driver copies the Huber threshold and the pixel sigma off the
    // estimator (`linearization_abs_qr.cpp:69-73` asserts they agree); the
    // per-block calls below need the same options.
    let lb_options = lqr.options().lb_options;

    let label = |what: &str| format!("{} {} {what}", case.name, case.scalar);

    // -- linearizeProblem ------------------------------------------------
    let (error, numerically_valid) = lqr.linearize_problem(&rebuilt.estimator, &inputs).unwrap();
    assert_eq!(numerically_valid, case.problem.numerically_valid);
    cmp.close(
        error.to_f64(),
        case.problem.error,
        &label("linearize_problem error"),
    );

    // The layout arithmetic, the per-landmark error and the block before the QR.
    assert_eq!(lqr.landmark_blocks().len(), case.blocks.len());
    for (block, want) in lqr.landmark_blocks().iter().zip(case.blocks.iter()) {
        let name = label(&format!("lm {}", want.lm_id));
        assert_eq!(block.landmark_id(), LandmarkId(want.lm_id), "{name}: id");
        assert_eq!(
            block.layout(),
            (
                want.num_rows,
                want.num_cols,
                want.padding_idx,
                want.lm_idx,
                want.res_idx
            ),
            "{name}: layout"
        );
        cmp.close_matrix(
            block.storage(),
            &want.storage_pre,
            want.num_rows,
            want.num_cols,
            &format!("{name}: storage_pre"),
        );

        // `linearizeLandmark`'s return value, which the driver only ever sums.
        let lm: &Landmark<S> = rebuilt
            .estimator
            .lmdb
            .get_landmark(LandmarkId(want.lm_id))
            .unwrap();
        let mut solo: LandmarkBlock<S> = block.clone();
        let block_error: S = solo
            .linearize_landmark(
                lm,
                lqr.relative_poses(),
                rebuilt.estimator.cameras(),
                &lb_options,
            )
            .unwrap();
        cmp.close(
            block_error.to_f64(),
            want.error,
            &format!("{name}: linearize_landmark error"),
        );
    }

    // -- performQR -------------------------------------------------------
    lqr.perform_qr().unwrap();
    for (block, want) in lqr.landmark_blocks().iter().zip(case.blocks.iter()) {
        let name = label(&format!("lm {}", want.lm_id));
        cmp.close_matrix(
            block.storage(),
            &want.storage_post,
            want.num_rows,
            want.num_cols,
            &format!("{name}: storage_post"),
        );

        // The per-block exports.
        let rows: usize = block.num_q2rows();
        let mut q2jp: DMatrix<S> = DMatrix::zeros(rows, want.padding_idx);
        let mut q2r: DVector<S> = DVector::zeros(rows);
        block.get_dense_q2jp_q2r(&mut q2jp, &mut q2r, 0).unwrap();
        cmp.close_matrix(
            &q2jp,
            &want.q2jp,
            rows,
            want.padding_idx,
            &format!("{name}: q2jp"),
        );
        cmp.close_slice(q2r.as_slice(), &want.q2r, &format!("{name}: q2r"));

        let mut h: DMatrix<S> = DMatrix::zeros(want.padding_idx, want.padding_idx);
        let mut b: DVector<S> = DVector::zeros(want.padding_idx);
        block
            .add_dense_h_b(&mut h, &mut b, &mut DenseHbScratch::default())
            .unwrap();
        cmp.close_matrix(
            &h,
            &want.block_h,
            want.padding_idx,
            want.padding_idx,
            &format!("{name}: block H"),
        );
        cmp.close_slice(b.as_slice(), &want.block_b, &format!("{name}: block b"));

        // The per-block back substitution, on copies of both the block and the
        // landmark, so the driver-level pass below still starts from the
        // pristine state the C++ used.
        let mut solo: LandmarkBlock<S> = block.clone();
        let mut lm: Landmark<S> = rebuilt
            .estimator
            .lmdb
            .get_landmark(LandmarkId(want.lm_id))
            .unwrap()
            .clone();
        let mut l_diff: S = S::zero();
        solo.back_substitute(&mut lm, &rebuilt.pose_inc, &mut l_diff)
            .unwrap();
        cmp.close(
            l_diff.to_f64(),
            want.l_diff,
            &format!("{name}: block l_diff"),
        );
        cmp.close_slice(
            &[lm.direction[0], lm.direction[1], lm.inv_dist],
            &[
                want.direction_after[0],
                want.direction_after[1],
                want.inv_dist_after,
            ],
            &format!("{name}: landmark after"),
        );
    }

    // -- get_dense_H_b ---------------------------------------------------
    let (h, b) = lqr.get_dense_h_b(&rebuilt.estimator, &inputs).unwrap();
    cmp.close_matrix(
        &h,
        &case.problem.h,
        case.problem.h_rows,
        case.problem.h_rows,
        &label("H"),
    );
    cmp.close_slice(b.as_slice(), &case.problem.b, &label("b"));

    // -- get_dense_Q2Jp_Q2r ----------------------------------------------
    let (q2jp, q2r) = lqr.get_dense_q2jp_q2r(&rebuilt.estimator, &inputs).unwrap();
    cmp.close_matrix(
        &q2jp,
        &case.problem.q2jp,
        case.problem.q2jp_rows,
        case.problem.q2jp_cols,
        &label("Q2Jp"),
    );
    cmp.close_slice(q2r.as_slice(), &case.problem.q2r, &label("Q2r"));

    // -- backSubstitute --------------------------------------------------
    let pose_inc: DVector<S> = rebuilt.pose_inc.clone();
    let l_diff: S = lqr
        .back_substitute(&mut rebuilt.estimator, &inputs, &pose_inc)
        .unwrap();
    cmp.close(l_diff.to_f64(), case.problem.l_diff, &label("l_diff"));
    for want in &case.problem.landmarks_after {
        let lm: &Landmark<S> = rebuilt
            .estimator
            .lmdb
            .get_landmark(LandmarkId(want.id))
            .unwrap();
        cmp.close_slice(
            &[lm.direction[0], lm.direction[1], lm.inv_dist],
            &[want.direction[0], want.direction[1], want.inv_dist],
            &label(&format!("lm {} after", want.id)),
        );
    }

    println!(
        "{} {}: worst {:.3e} at {}",
        case.name, case.scalar, cmp.worst, cmp.worst_what
    );
    cmp.worst
}

fn oracle() -> Oracle {
    serde_json::from_str(ORACLE).expect("the linearization oracle fixture parses")
}

#[test]
fn the_linearization_reproduces_the_cpp_in_double() {
    let oracle: Oracle = oracle();
    let mut seen: usize = 0;
    let mut worst: f64 = 0.0;
    for case in &oracle.cases {
        if case.scalar != "f64" {
            continue;
        }
        worst = worst.max(check_case::<f64>(case, TOLERANCE_F64));
        seen += 1;
    }
    assert_eq!(seen, 4, "four double-precision problems in the fixture");
    println!("worst f64 deviation: {worst:.3e}");
}

#[test]
fn the_linearization_reproduces_the_cpp_in_float() {
    let oracle: Oracle = oracle();
    let mut seen: usize = 0;
    let mut worst: f64 = 0.0;
    for case in &oracle.cases {
        if case.scalar != "f32" {
            continue;
        }
        worst = worst.max(check_case::<f32>(case, TOLERANCE_F32));
        seen += 1;
    }
    assert_eq!(seen, 4, "four single-precision problems in the fixture");
    println!("worst f32 deviation: {worst:.3e}");
}

/// The fixture really does exercise what the module docs claim: the padding
/// branch, the Huber branch, the prior and the first-estimate Jacobians.
#[test]
fn the_fixture_covers_the_branches_it_claims_to() {
    let oracle: Oracle = oracle();
    let f64_cases: Vec<&Case> = oracle.cases.iter().filter(|c| c.scalar == "f64").collect();

    // `padding_size == 0` in one problem and `2` in the others
    // (`landmark_block_abs_dynamic.hpp:87-88`).
    let paddings: BTreeSet<usize> = f64_cases
        .iter()
        .flat_map(|c| c.blocks.iter().map(|b| b.lm_idx - b.padding_idx))
        .collect();
    assert!(paddings.contains(&0), "no unpadded block in the fixture");
    assert!(paddings.contains(&2), "no padded block in the fixture");
    for case in &f64_cases {
        for block in &case.blocks {
            assert_eq!(block.num_cols % 4, 0, "the `% 4` assertion of `:96`");
            assert_eq!(
                block.num_rows % 2,
                1,
                "2 per observation plus 3 damping rows"
            );
        }
    }

    // Two problems carry a prior with both of their first frames frozen.
    let with_marg: usize = f64_cases.iter().filter(|c| c.marg.is_some()).count();
    assert_eq!(with_marg, 2);
    for case in f64_cases.iter().filter(|c| c.marg.is_some()) {
        let frozen: usize = case.frames.iter().filter(|f| f.linearized).count();
        assert_eq!(
            frozen, 2,
            "{}: the FEJ branch needs frozen frames",
            case.name
        );
        assert!(
            case.frames
                .iter()
                .any(|f| f.linearized && f.delta.iter().any(|d| *d != 0.0)),
            "{}: a zero delta would not exercise the re-evaluation",
            case.name
        );
    }

    // The Huber branch: `three_frames_huber` has observations well past the
    // 0.5 px threshold, so its error is an order of magnitude above the others'.
    let huber: &Case = f64_cases
        .iter()
        .find(|c| c.name == "three_frames_huber")
        .unwrap();
    let plain: &Case = f64_cases.iter().find(|c| c.name == "two_frames").unwrap();
    assert!(huber.problem.error > 10.0 * plain.problem.error);
}
