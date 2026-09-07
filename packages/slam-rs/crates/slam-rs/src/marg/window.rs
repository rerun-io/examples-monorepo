//! The sliding-window half of `SqrtKeypointVioEstimator::marginalize`
//! (`src/vi_estimator/sqrt_keypoint_vio.cpp:707-1198`): everything that happens
//! **after** the schedule has decided what leaves.
//!
//! The schedule itself — the keyframe scoring at `:767-880`, the
//! `states_to_remove` count at `:720-724`, and the `kf_ids` bookkeeping — is
//! stage S8's and is not here. This module takes those four sets as
//! [`MarginalizeSchedule`] and does the rest: build the absolute ordering,
//! linearize the window with the current prior, split the ordering into kept
//! and marginalized indices, run [`crate::marg::MargHelper`], shrink the window
//! and the landmark database, and re-anchor the new prior.
//!
//! Two traps of the architecture dossier live here and are called out at the
//! lines that implement them:
//!
//! * **Trap 7** — the state that stays behind as the prior's newest block is
//!   frozen at its linearization point, `setLinTrue()` (`:1086-1088`). Without
//!   it the prior's `delta` means nothing and the drift is silent.
//! * **Trap 8** — the prior comes out of the helper linearized at `x = 0`, and
//!   is put back into its delta-independent form by
//!   `marg_data.b -= marg_data.H * delta` (`:1170-1172`). Its mirror on the way
//!   in is `linearization_abs_qr.cpp:592`.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use crate::ba_base::BundleAdjustmentBase;
use crate::imu::{ImuLinData, IntegratedImuMeasurement};
use crate::lie::{LieScalar, So3};
use crate::linearize::{ImuInput, LinearizationAbsQR, LinearizationInputs, LinearizationOptions};
use crate::marg::helper::{
    ReducedSystem, marginalize_helper_sq_to_sq, marginalize_helper_sqrt_to_sqrt,
};
use crate::marg::{MargError, ScheduleSet};
use crate::types::{
    AbsOrderMap, FrameId, LandmarkId, MargLinData, POSE_SIZE, POSE_VEL_BIAS_SIZE, PoseStateWithLin,
};

/// What the schedule decided, `sqrt_keypoint_vio.cpp:724-880`.
///
/// The four sets are disjoint by construction in C++ and the port checks it:
/// `poses_to_marg` names pose blocks that leave, `kfs_to_marg` the subset of
/// those that were keyframes hosting landmarks, `states_to_marg_all` full
/// states that leave outright, and `states_to_marg_vel_bias` full states that
/// keep their pose and lose their velocity and biases.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MarginalizeSchedule {
    /// `last_state_to_marg` (`:724`): the newest state the ordering reaches and
    /// the one that becomes the prior's 15-dof block.
    pub last_state_to_marg: FrameId,
    /// `kfs_to_marg` (`:766`).
    pub kfs_to_marg: BTreeSet<FrameId>,
    /// `poses_to_marg` (`:729`), a superset of `kfs_to_marg`.
    pub poses_to_marg: BTreeSet<FrameId>,
    /// `states_to_marg_all` (`:743`).
    pub states_to_marg_all: BTreeSet<FrameId>,
    /// `states_to_marg_vel_bias` (`:742`).
    pub states_to_marg_vel_bias: BTreeSet<FrameId>,
}

/// The configuration bits `marginalize()` reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MarginalizeOptions {
    /// `config.vio_marg_lost_landmarks` (`:1116`).
    pub marg_lost_landmarks: bool,
    /// `config.vio_debug || config.vio_extended_logging` (`:1012`, `:1174`):
    /// run the whole linearization a second time against `nullspace_marg_data`.
    pub keep_nullspace_marg_data: bool,
}

/// Everything `marginalize()` needs besides the window itself.
#[derive(Debug)]
pub struct MarginalizeInputs<'a, S: LieScalar> {
    /// What the schedule decided.
    pub schedule: &'a MarginalizeSchedule,
    /// Gravity and the two bias random-walk weights; `None` for a visual-only
    /// window (`ImuLinData` at `:913`).
    pub imu_lin_data: Option<ImuLinData<S>>,
    /// `lost_landmaks` (`:709`), the landmarks the frontend stopped tracking.
    pub lost_landmarks: Option<&'a BTreeSet<LandmarkId>>,
    /// `fixed_kfs` (`:924`): `ltkfs` when `config.vio_fix_long_term_keyframes`
    /// is on, empty otherwise.
    pub fixed_frames: Option<&'a BTreeSet<FrameId>>,
    /// Flags.
    pub options: MarginalizeOptions,
}

/// What one marginalization produced besides the updated prior.
#[derive(Debug, Clone, PartialEq)]
pub struct MarginalizeOutput<S: LieScalar> {
    /// `aom` (`:726-763`): the ordering the marginalization linearized over.
    pub aom: AbsOrderMap,
    /// `idx_to_keep` (`:980-1003`).
    pub idx_to_keep: BTreeSet<usize>,
    /// `idx_to_marg` (`:980-1003`).
    pub idx_to_marg: BTreeSet<usize>,
    /// What `linearizeProblem` reported for the window that is leaving.
    pub error: S,
    /// `numerically_valid`: false when a landmark block held a non-finite
    /// Jacobian, which basalt zeroes with a warning rather than failing.
    pub numerically_valid: bool,
}

/// Build the absolute ordering the marginalization runs over
/// (`sqrt_keypoint_vio.cpp:726-763`).
///
/// Every pose block first, in timestamp order, then the full states up to and
/// including `last_state_to_marg` — states newer than that are simply not in
/// the system. Each block is checked against the prior's ordering as C++
/// asserts (`:736`, `:758-759`): the prior must be a prefix of the window at
/// exactly the same offsets, or the two systems do not describe the same
/// variables.
fn build_absolute_ordering<S: LieScalar>(
    estimator: &BundleAdjustmentBase<S>,
    marg_data: &MargLinData<S>,
    last_state_to_marg: FrameId,
) -> Result<AbsOrderMap, MargError> {
    let mut aom: AbsOrderMap = AbsOrderMap::new();
    for frame_id in estimator.frame_poses.keys() {
        let offset: usize = aom.push(*frame_id, POSE_SIZE)?;
        // `:736`: unconditional, because C++ uses `at()`.
        if marg_data.order.get(*frame_id) != Some((offset, POSE_SIZE)) {
            return Err(MargError::PriorOrderMismatch {
                frame_id: *frame_id,
            });
        }
    }
    for frame_id in estimator.frame_states.keys() {
        // `:745`.
        if *frame_id > last_state_to_marg {
            break;
        }
        let offset: usize = aom.push(*frame_id, POSE_VEL_BIAS_SIZE)?;
        // `:758-759`: the comparison is against `aom.items` **before** the
        // increment at `:762`, so a state past the end of the prior is not
        // checked — it is the one entering it.
        if aom.items() <= marg_data.order.items()
            && marg_data.order.get(*frame_id) != Some((offset, POSE_VEL_BIAS_SIZE))
        {
            return Err(MargError::PriorOrderMismatch {
                frame_id: *frame_id,
            });
        }
    }
    Ok(aom)
}

/// Check the whole schedule against the linearized ordering, before anything
/// mutates.
///
/// C++ builds the five sets out of the window itself
/// (`sqrt_keypoint_vio.cpp:724-880`) and then trusts them: `:1090-1112` erases
/// what they name with `frame_states.at()` (which throws on a frame that is not
/// there) and `frame_poses.erase()` (which silently does nothing), in that
/// order, so a schedule that disagrees with the window takes effect *before* it
/// is noticed — and a state newer than `last_state_to_marg` is not in the
/// ordering at all, so it is deleted without having been marginalized. Every
/// relationship below is an invariant of C++'s construction, and checking them
/// here is what keeps a refused marginalization from leaving a half-rewritten
/// window (decision D32).
///
/// Membership in `aom` is the same question as membership in the live maps:
/// [`build_absolute_ordering`] just walked `frame_poses` and the prefix of
/// `frame_states` up to `last_state_to_marg`, and nothing has changed since.
/// The block size is what tells a pose from a full state, which is also what
/// makes the four sets pairwise disjoint: `poses_to_marg` must be 6-row blocks
/// and both state sets 15-row blocks.
fn validate_schedule(aom: &AbsOrderMap, schedule: &MarginalizeSchedule) -> Result<(), MargError> {
    // `:729` and `:876`: every pose that leaves is a pose block of the window.
    for frame_id in &schedule.poses_to_marg {
        if !matches!(aom.get(*frame_id), Some((_, POSE_SIZE))) {
            return Err(MargError::ScheduledFrameNotInOrdering {
                set: ScheduleSet::PosesToMarg,
                frame_id: *frame_id,
                block: POSE_SIZE,
            });
        }
    }

    // `:875-876`.
    for frame_id in &schedule.kfs_to_marg {
        if !schedule.poses_to_marg.contains(frame_id) {
            return Err(MargError::KeyframeNotInPosesToMarg {
                frame_id: *frame_id,
            });
        }
    }

    // `:743` and `:742`: both state sets are full states of the window, and
    // `:745`'s `if (kv.first != last_state_to_marg)` keeps the newest state out
    // of both — it is the one that becomes the prior's own block.
    for (set, frames) in [
        (ScheduleSet::StatesToMargAll, &schedule.states_to_marg_all),
        (
            ScheduleSet::StatesToMargVelBias,
            &schedule.states_to_marg_vel_bias,
        ),
    ] {
        for frame_id in frames {
            if !matches!(aom.get(*frame_id), Some((_, POSE_VEL_BIAS_SIZE))) {
                return Err(MargError::ScheduledFrameNotInOrdering {
                    set,
                    frame_id: *frame_id,
                    block: POSE_VEL_BIAS_SIZE,
                });
            }
            if *frame_id == schedule.last_state_to_marg {
                return Err(MargError::ScheduleSetsOverlap {
                    first: ScheduleSet::LastStateToMarg,
                    second: set,
                    frame_id: *frame_id,
                });
            }
        }
    }

    // `:747-750`: the two state sets are an if/else over the same frame.
    if let Some(frame_id) = schedule
        .states_to_marg_all
        .intersection(&schedule.states_to_marg_vel_bias)
        .next()
    {
        return Err(MargError::ScheduleSetsOverlap {
            first: ScheduleSet::StatesToMargAll,
            second: ScheduleSet::StatesToMargVelBias,
            frame_id: *frame_id,
        });
    }

    Ok(())
}

/// The ordering the new prior gets, `marg_order_new` (`:1120-1133`).
///
/// C++ builds it by walking the **already shrunk** `frame_poses`; the port
/// builds it from the sets instead, so the width check at `:1145` can run
/// before anything mutates and a refusal leaves `marg_data` alone. The two
/// give the same ordering: C++'s `frame_poses` at that point is
/// `(frame_poses ∪ states_to_marg_vel_bias) \ poses_to_marg`, a `std::map` in
/// ascending key order, which is what a [`BTreeSet`] of the same ids iterates.
fn new_prior_ordering<S: LieScalar>(
    estimator: &BundleAdjustmentBase<S>,
    schedule: &MarginalizeSchedule,
) -> Result<AbsOrderMap, MargError> {
    let surviving: BTreeSet<FrameId> = estimator
        .frame_poses
        .keys()
        .copied()
        .chain(schedule.states_to_marg_vel_bias.iter().copied())
        .filter(|id| !schedule.poses_to_marg.contains(id))
        .collect();
    let mut order: AbsOrderMap = AbsOrderMap::new();
    for frame_id in &surviving {
        order.push(*frame_id, POSE_SIZE)?;
    }
    order.push(schedule.last_state_to_marg, POSE_VEL_BIAS_SIZE)?;
    Ok(order)
}

/// Split the ordering into the indices that stay and the indices that go
/// (`sqrt_keypoint_vio.cpp:980-1003`).
///
/// A pose block goes whole or stays whole. A full state either goes whole
/// (`states_to_marg_all`), keeps its six pose rows and loses the other nine
/// (`states_to_marg_vel_bias`), or stays whole — and the last case is asserted
/// to be `last_state_to_marg` alone (`:999`).
fn split_indices(
    aom: &AbsOrderMap,
    schedule: &MarginalizeSchedule,
) -> Result<(BTreeSet<usize>, BTreeSet<usize>), MargError> {
    let mut idx_to_keep: BTreeSet<usize> = BTreeSet::new();
    let mut idx_to_marg: BTreeSet<usize> = BTreeSet::new();
    for (frame_id, start_idx, size) in aom.iter() {
        match size {
            POSE_SIZE => {
                let target: &mut BTreeSet<usize> = if schedule.poses_to_marg.contains(&frame_id) {
                    &mut idx_to_marg
                } else {
                    &mut idx_to_keep
                };
                target.extend(start_idx..start_idx + POSE_SIZE);
            }
            POSE_VEL_BIAS_SIZE => {
                if schedule.states_to_marg_all.contains(&frame_id) {
                    idx_to_marg.extend(start_idx..start_idx + POSE_VEL_BIAS_SIZE);
                } else if schedule.states_to_marg_vel_bias.contains(&frame_id) {
                    idx_to_keep.extend(start_idx..start_idx + POSE_SIZE);
                    idx_to_marg.extend(start_idx + POSE_SIZE..start_idx + POSE_VEL_BIAS_SIZE);
                } else {
                    // `:999`.
                    if frame_id != schedule.last_state_to_marg {
                        return Err(MargError::UnscheduledState { frame_id });
                    }
                    idx_to_keep.extend(start_idx..start_idx + POSE_VEL_BIAS_SIZE);
                }
            }
            // C++ asserts `POSE_SIZE` or `POSE_VEL_BIAS_SIZE` (`:990`).
            size => return Err(MargError::UnexpectedBlockSize { frame_id, size }),
        }
    }
    Ok((idx_to_keep, idx_to_marg))
}

/// Linearize the window over `aom` with `prior` included, and return whichever
/// dense form the prior's own representation calls for
/// (`sqrt_keypoint_vio.cpp:905-942`).
///
/// Only `ABS_QR` is ported (decision D13), so `isLinearizationSqrt` is
/// constantly true and the branch at `:935-939` is decided by
/// `marg_data.is_sqrt` alone: a square-root prior takes `get_dense_Q2Jp_Q2r`,
/// a squared one takes `get_dense_H_b`.
fn linearize_for_marginalization<S: LieScalar>(
    estimator: &BundleAdjustmentBase<S>,
    aom: &AbsOrderMap,
    prior: &MargLinData<S>,
    imu_input: Option<&ImuInput<'_, S>>,
    inputs: &MarginalizeInputs<'_, S>,
) -> Result<LinearizedWindow<S>, MargError> {
    let lin_inputs: LinearizationInputs<'_, S> = LinearizationInputs {
        marg: Some(prior),
        imu: imu_input,
        // `:925`: only landmarks hosted by a marginalized keyframe, or lost.
        used_frames: Some(&inputs.schedule.kfs_to_marg),
        lost_landmarks: inputs.lost_landmarks,
        fixed_frames: inputs.fixed_frames,
    };
    // The defaults are safe: `LinearizationAbsQR::new` overwrites
    // `huber_parameter` and `obs_std_dev` from the estimator
    // (`linearization_abs_qr.cpp:69-73`, where C++ asserts they agree).
    let mut lqr: LinearizationAbsQR<S> =
        LinearizationAbsQR::new(estimator, aom, LinearizationOptions::default(), &lin_inputs)?;
    // `:928`.
    let (error, numerically_valid) = lqr.linearize_problem(estimator, &lin_inputs)?;
    // `:932`.
    lqr.perform_qr()?;
    // `:935-939`.
    let (h, b) = if prior.is_sqrt {
        lqr.get_dense_q2jp_q2r(estimator, &lin_inputs)?
    } else {
        lqr.get_dense_h_b(estimator, &lin_inputs)?
    };
    Ok(LinearizedWindow {
        h,
        b,
        error,
        numerically_valid,
    })
}

/// What one pass of `:905-942` produced.
struct LinearizedWindow<S: LieScalar> {
    /// `Q2Jp_or_H`.
    h: DMatrix<S>,
    /// `Q2r_or_b`.
    b: DVector<S>,
    /// What `linearizeProblem` reported.
    error: S,
    /// Whether every landmark block held finite Jacobians.
    numerically_valid: bool,
}

/// Run the helper the prior's representation calls for
/// (`sqrt_keypoint_vio.cpp:1071-1080`).
///
/// `marginalizeHelperSqToSqrt` is the third branch there and is unreachable in
/// this port: it needs a squared linearization with a square-root prior, and
/// only `ABS_QR` is ported. It is still exercised by the oracle and by
/// `test_qr.cpp`'s `RankDefLeastSquares`.
fn run_helper<S: LieScalar>(
    is_sqrt: bool,
    h: DMatrix<S>,
    b: DVector<S>,
    idx_to_keep: &BTreeSet<usize>,
    idx_to_marg: &BTreeSet<usize>,
) -> Result<ReducedSystem<S>, MargError> {
    if is_sqrt {
        marginalize_helper_sqrt_to_sqrt(h, b, idx_to_keep, idx_to_marg)
    } else {
        marginalize_helper_sq_to_sq(h, b, idx_to_keep, idx_to_marg)
    }
}

/// `SqrtKeypointVioEstimator::marginalize` from `:896` to `:1178`.
///
/// Mutates the window in place: `marg_data` becomes the new prior,
/// `estimator.frame_states` / `frame_poses` / `lmdb` shrink, and the consumed
/// IMU intervals leave `imu_meas`. Returns the ordering and the index split so
/// stage S8 can log them.
///
/// The order of operations is basalt's, and two of the steps only work because
/// of where they sit:
///
/// * `setLinTrue` on `last_state_to_marg` (`:1086-1088`) happens **before**
///   `computeDelta`, so that state contributes a zero delta to the re-anchoring
///   rather than the increment it accumulated as a free variable;
/// * `marg_order_new` describes the window that *survives* — C++ builds it at
///   `:1120-1133`, after the `states_to_marg_vel_bias` frames have been demoted
///   into `frame_poses` and the `poses_to_marg` frames removed. The port
///   computes the same set from the schedule instead
///   ([`new_prior_ordering`]) so that `:1145`'s width check can run before the
///   first mutation.
pub fn marginalize<S: LieScalar>(
    estimator: &mut BundleAdjustmentBase<S>,
    marg_data: &mut MargLinData<S>,
    mut nullspace_marg_data: Option<&mut MargLinData<S>>,
    imu_meas: &mut BTreeMap<i64, IntegratedImuMeasurement<S>>,
    inputs: &MarginalizeInputs<'_, S>,
) -> Result<MarginalizeOutput<S>, MargError> {
    let schedule: &MarginalizeSchedule = inputs.schedule;
    let last_state_to_marg: FrameId = schedule.last_state_to_marg;

    // `:726-763`.
    let aom: AbsOrderMap = build_absolute_ordering(estimator, marg_data, last_state_to_marg)?;

    // `:1086`: C++ asserts this before it freezes the state; the port checks it
    // up front, so a refused marginalization leaves the window untouched.
    match estimator.frame_states.get(&last_state_to_marg) {
        None => {
            return Err(MargError::FrameNotInWindow {
                frame_id: last_state_to_marg,
            });
        }
        Some(state) if state.is_linearized() => {
            return Err(MargError::AlreadyLinearized {
                frame_id: last_state_to_marg,
            });
        }
        Some(_) => {}
    }

    // Everything the schedule claims about the window, checked here rather
    // than discovered halfway through `:1090-1112`.
    validate_schedule(&aom, schedule)?;

    // `:915-922`: the intervals whose two ends are both in the ordering.
    let imu_input: Option<ImuInput<'_, S>> = inputs.imu_lin_data.map(|lin_data| ImuInput {
        lin_data,
        measurements: imu_meas
            .iter()
            .filter(|(_, meas)| {
                let start_t: i64 = meas.get_start_t_ns();
                let end_t: i64 = start_t + meas.get_dt_ns();
                aom.contains(start_t) && aom.contains(end_t)
            })
            .map(|(start_t, meas)| (*start_t, meas))
            .collect(),
    });

    // `:905-942`.
    let live: LinearizedWindow<S> =
        linearize_for_marginalization(estimator, &aom, marg_data, imu_input.as_ref(), inputs)?;

    // `:980-1003`.
    let (idx_to_keep, idx_to_marg) = split_indices(&aom, schedule)?;

    // `:1120-1133`, hoisted: the new prior's ordering is a function of the
    // window and the schedule, so it and the `:1145` width check both run
    // before the first mutation.
    let marg_order_new: AbsOrderMap = new_prior_ordering(estimator, schedule)?;
    if idx_to_keep.len() != marg_order_new.total_size() {
        return Err(MargError::PriorWidthMismatch {
            cols: idx_to_keep.len(),
            total_size: marg_order_new.total_size(),
        });
    }

    // `:1012-1064`: the debug copy. A second full linearization against the
    // *previous* nullspace prior, so the two can be compared without the
    // fixed-linearization bookkeeping the live prior carries.
    let nullspace_reduced: Option<ReducedSystem<S>> = match nullspace_marg_data.as_deref_mut() {
        Some(nullspace) if inputs.options.keep_nullspace_marg_data => {
            // `:1021`: `nullspace_marg_data.order = marg_data.order`, the
            // order *before* `:1137` replaces it, so the second linearization
            // runs against the same variables the live one does. C++ writes it
            // into `nullspace_marg_data` itself; the port carries it in this
            // local copy and assigns the field at the end of `marginalize`,
            // where `:1186`'s `logMargNullspace()` assigns the new one.
            let mut prior: MargLinData<S> = MargLinData {
                is_sqrt: nullspace.is_sqrt,
                order: marg_data.order.clone(),
                h: nullspace.h.clone(),
                b: nullspace.b.clone(),
            };
            if prior.h.ncols() != prior.order.total_size() {
                // The very first marginalization has an empty nullspace prior;
                // basalt linearizes with an empty `MargLinData` too.
                prior.h = DMatrix::zeros(0, prior.order.total_size());
                prior.b = DVector::zeros(0);
            }
            let debug: LinearizedWindow<S> =
                linearize_for_marginalization(estimator, &aom, &prior, imu_input.as_ref(), inputs)?;
            Some(run_helper(
                marg_data.is_sqrt,
                debug.h,
                debug.b,
                &idx_to_keep,
                &idx_to_marg,
            )?)
        }
        _ => None,
    };

    // `:1069-1083`.
    let reduced: ReducedSystem<S> = run_helper(
        marg_data.is_sqrt,
        live.h,
        live.b,
        &idx_to_keep,
        &idx_to_marg,
    )?;

    // The linearization is done with the window; everything from here mutates.
    drop(imu_input);

    // `:1085-1088`, trap 7. `validate_schedule` and the `:1086` check above
    // both prove the lookup, so the `else` is unreachable; it is an error
    // rather than a skip because silently not freezing the state is trap 7
    // happening.
    let Some(state) = estimator.frame_states.get_mut(&last_state_to_marg) else {
        return Err(MargError::FrameNotInWindow {
            frame_id: last_state_to_marg,
        });
    };
    state.set_linearized()?;

    // `:1090-1096`.
    for id in &schedule.states_to_marg_all {
        estimator.frame_states.remove(id);
        imu_meas.remove(id);
    }

    // `:1098-1105`: a keyframe that keeps its pose is demoted to a pose block,
    // carrying the first six entries of its delta and its `linearized` flag
    // (`imu_types.h:206-215`).
    for id in &schedule.states_to_marg_vel_bias {
        // Proven by `validate_schedule`, like the lookup above: every frame in
        // this set is a 15-row block of the ordering, and the ordering was
        // built from `frame_states`.
        let Some(state) = estimator.frame_states.get(id) else {
            return Err(MargError::FrameNotInWindow { frame_id: *id });
        };
        let pose: PoseStateWithLin<S> = PoseStateWithLin::from_pose_vel_bias(state);
        estimator.frame_poses.insert(*id, pose);
        estimator.frame_states.remove(id);
        imu_meas.remove(id);
    }

    // `:1107-1112`.
    for id in &schedule.poses_to_marg {
        estimator.frame_poses.remove(id);
    }

    // `:1114`.
    estimator.lmdb.remove_keyframes(
        &schedule.kfs_to_marg,
        &schedule.poses_to_marg,
        &schedule.states_to_marg_all,
    );

    // `:1116-1118`.
    if inputs.options.marg_lost_landmarks
        && let Some(lost) = inputs.lost_landmarks
    {
        for lm_id in lost {
            estimator.lmdb.remove_landmark(*lm_id);
        }
    }

    // `:1135-1137`. The width the helper produced is `idx_to_keep.len()`, and
    // `:1145`'s check on it already ran above, against the same ordering.
    marg_data.h = reduced.h;
    marg_data.b = reduced.b;
    marg_data.order = marg_order_new;

    // `:1147-1172`, trap 8. The prior comes out of the helper as
    // `P(x) = 0.5‖J x + res‖²`; putting it back into the delta-independent form
    // `P(x) = 0.5‖J (delta + x) + (res − J delta)‖²` is one subtraction.
    let delta: DVector<S> = estimator.compute_delta(&marg_data.order)?;
    subtract_h_delta(&mut marg_data.b, &marg_data.h, &delta);

    // `:1174-1178`: the same re-anchoring on the debug copy, with the same
    // delta, and then the order.
    if let (Some(nullspace), Some(reduced_ns)) = (nullspace_marg_data, nullspace_reduced) {
        nullspace.is_sqrt = marg_data.is_sqrt;
        nullspace.h = reduced_ns.h;
        nullspace.b = reduced_ns.b;
        subtract_h_delta(&mut nullspace.b, &nullspace.h, &delta);
        // `:1186` calls `logMargNullspace()`, whose **first** statement is
        // `nullspace_marg_data.order = marg_data.order` (`:672`) — the *new*
        // order, since `:1137` has already replaced it — before
        // `checkMargNullspace()` and `checkMargEigenvalues()` read the pair.
        // So the debug prior's order does not lag: it is the same ordering the
        // live prior just got. (C++ also assigns the *old* order at `:1021`,
        // for the second linearization; the port carries that one in the local
        // `prior` above, which is the same value at the same point.)
        nullspace.order = marg_data.order.clone();
    }

    Ok(MarginalizeOutput {
        aom,
        idx_to_keep,
        idx_to_marg,
        error: live.error,
        numerically_valid: live.numerically_valid,
    })
}

/// `b -= H * delta` (`sqrt_keypoint_vio.cpp:1172`), written out so the
/// summation order is fixed rather than nalgebra's.
fn subtract_h_delta<S: LieScalar>(b: &mut DVector<S>, h: &DMatrix<S>, delta: &DVector<S>) {
    let cols: usize = h.ncols().min(delta.nrows());
    for i in 0..h.nrows().min(b.nrows()) {
        let mut acc: S = S::zero();
        for j in 0..cols {
            acc += h[(i, j)] * delta[j];
        }
        b[i] -= acc;
    }
}

// ─── the two diagnostics ───────────────────────────────────────────────────

/// What [`check_marg_nullspace`] measured, one entry per probe direction.
///
/// C++ prints these and returns `xHx + xb` (`sqrt_ba_base.cpp:197-207`); the
/// port returns all three, because the two halves say different things — `xHx`
/// is spurious information on an unobservable direction, `xb` is a spurious
/// gradient — and a caller that only wants basalt's number can read
/// [`Self::total`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NullspaceCheck {
    /// `xHx`: `incᵀ H inc` for x, y, z, roll, pitch, yaw and the random probe,
    /// in that order (`:180-186`).
    pub xhx: [f64; 7],
    /// `xb`: `incᵀ b` for the same seven (`:189-195`).
    pub xb: [f64; 7],
}

impl NullspaceCheck {
    /// `xHx + xb`, what `checkNullspace` returns (`:207`).
    pub fn total(&self) -> [f64; 7] {
        let mut out: [f64; 7] = [0.0; 7];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.xhx[i] + self.xb[i];
        }
        out
    }
}

/// `SqrtBundleAdjustmentBase::checkNullspace`
/// (`src/vi_estimator/sqrt_ba_base.cpp:42-208`).
///
/// Builds the six increments that *should* lie in the prior's null space —
/// three global translations and three global rotations about the translation
/// centroid — normalizes each, and reports how much information the prior has
/// accumulated along them. For a visual-inertial problem only yaw is truly
/// unobservable: gravity fixes roll and pitch, so those two are expected to
/// carry information. A seventh, random, direction is the control.
///
/// **The rotational increments rotate translations and velocities too**
/// (`:125-142`), because poses are cam-to-world with a left increment, so a
/// world-frame rotation moves both. The centre of rotation is the mean
/// translation rather than the origin, "for better numerics" (`:117-121`).
///
/// `inc_random` replaces C++'s `inc_random.setRandom()` (`:158`): it is a
/// parameter so the diagnostic is reproducible, and it is normalized here as
/// C++ normalizes it. Everything runs in `f64` whatever the estimator's scalar
/// is, as `:165-176` does.
pub fn check_marg_nullspace<S: LieScalar>(
    mld: &MargLinData<S>,
    estimator: &BundleAdjustmentBase<S>,
    inc_random: &DVector<f64>,
) -> Result<NullspaceCheck, MargError> {
    // `:52`.
    let marg_size: usize = mld.order.total_size();
    if mld.h.ncols() != marg_size {
        return Err(MargError::PriorWidthMismatch {
            cols: mld.h.ncols(),
            total_size: marg_size,
        });
    }
    if inc_random.nrows() != marg_size {
        return Err(MargError::ProbeLengthMismatch {
            expected: marg_size,
            actual: inc_random.nrows(),
        });
    }
    // The two shapes `:165-176` and `:180-195` then rely on and C++ does not
    // assert (decision D32). A square-root prior needs `b` as tall as `H`, or
    // `Hᵀb` is not formed; a squared one needs `H` square as well as
    // `marg_size` wide, or the quadratic `xᵀHx` does not close.
    if mld.is_sqrt {
        if mld.b.nrows() != mld.h.nrows() {
            return Err(MargError::RhsLengthMismatch {
                rows: mld.h.nrows(),
                rhs: mld.b.nrows(),
            });
        }
    } else {
        if mld.h.nrows() != marg_size {
            return Err(MargError::NotSquare {
                rows: mld.h.nrows(),
                cols: marg_size,
            });
        }
        if mld.b.nrows() != marg_size {
            return Err(MargError::RhsLengthMismatch {
                rows: marg_size,
                rhs: mld.b.nrows(),
            });
        }
    }

    // `:82-96`: the mean translation over the prior's blocks.
    let mut mean_trans: Vector3<f64> = Vector3::zeros();
    let mut num_trans: usize = 0;
    for (frame_id, _, size) in mld.order.iter() {
        mean_trans += translation_of(estimator, frame_id, size)?;
        num_trans += 1;
    }
    if num_trans == 0 {
        return Err(MargError::EmptyPriorOrder);
    }
    mean_trans /= num_trans as f64;

    // `:98`.
    let eps: f64 = 0.01;

    let mut inc: [DVector<f64>; 6] = std::array::from_fn(|_| DVector::zeros(marg_size));

    // `:101-144`.
    for (frame_id, offset, size) in mld.order.iter() {
        for (axis, vector) in inc.iter_mut().enumerate() {
            vector[offset + axis] = eps;
        }

        let trans: Vector3<f64> = translation_of(estimator, frame_id, size)? - mean_trans;

        // `:125-126`: `J = -SO3::hat(trans) * eps`, one column per rotation.
        let j: Matrix3<f64> = -So3::<f64>::hat(&trans) * eps;
        for axis in 0..3 {
            for row in 0..3 {
                inc[3 + axis][offset + row] = j[(row, axis)];
            }
        }

        if size == POSE_VEL_BIAS_SIZE {
            // `:129-141`.
            let Some(state) = estimator.frame_states.get(&frame_id) else {
                return Err(MargError::FrameNotInWindow { frame_id });
            };
            let vel: Vector3<f64> = state.state_lin().vel_w_i.map(|v| v.to_f64());
            let j_vel: Matrix3<f64> = -So3::<f64>::hat(&vel) * eps;
            for axis in 0..3 {
                for row in 0..3 {
                    inc[3 + axis][offset + POSE_SIZE + row] = j_vel[(row, axis)];
                }
            }
        }
    }

    // `:146-151`.
    for vector in &mut inc {
        let norm: f64 = vector.norm();
        if norm > 0.0 {
            *vector /= norm;
        }
    }
    let mut random: DVector<f64> = inc_random.clone();
    let random_norm: f64 = random.norm();
    if random_norm > 0.0 {
        random /= random_norm;
    }

    // `:165-176`: the squared form, always in double.
    let h_d: DMatrix<f64> = mld.h.map(|v| v.to_f64());
    let b_d: DVector<f64> = mld.b.map(|v| v.to_f64());
    let (h, b): (DMatrix<f64>, DVector<f64>) = if mld.is_sqrt {
        (h_d.transpose() * &h_d, h_d.transpose() * &b_d)
    } else {
        (h_d, b_d)
    };

    let mut xhx: [f64; 7] = [0.0; 7];
    let mut xb: [f64; 7] = [0.0; 7];
    for (i, vector) in inc.iter().chain(std::iter::once(&random)).enumerate() {
        let hv: DVector<f64> = &h * vector;
        xhx[i] = vector.dot(&hv);
        xb[i] = vector.dot(&b);
    }

    Ok(NullspaceCheck { xhx, xb })
}

/// The linearization-point translation of one block, `getPoseLin().translation()`
/// or `getStateLin().T_w_i.translation()` (`sqrt_ba_base.cpp:85-93`).
fn translation_of<S: LieScalar>(
    estimator: &BundleAdjustmentBase<S>,
    frame_id: FrameId,
    size: usize,
) -> Result<Vector3<f64>, MargError> {
    match size {
        POSE_SIZE => {
            let Some(pose) = estimator.frame_poses.get(&frame_id) else {
                return Err(MargError::FrameNotInWindow { frame_id });
            };
            Ok(pose.pose_lin().translation.map(|v| v.to_f64()))
        }
        POSE_VEL_BIAS_SIZE => {
            let Some(state) = estimator.frame_states.get(&frame_id) else {
                return Err(MargError::FrameNotInWindow { frame_id });
            };
            Ok(state.state_lin().t_w_i.translation.map(|v| v.to_f64()))
        }
        // `:91-93`: C++ prints and aborts.
        size => Err(MargError::UnexpectedBlockSize { frame_id, size }),
    }
}

/// `SqrtBundleAdjustmentBase::checkEigenvalues`
/// (`src/vi_estimator/sqrt_ba_base.cpp:210-233`).
///
/// The eigenvalues of `JᵀJ` in ascending order, computed in `f64` whatever the
/// estimator's scalar is and on the *squared* matrix deliberately, "to easily
/// notice if we have negative EVs (numerically)" (`:212-214`).
///
/// **One substitution.** C++ uses `Eigen::SelfAdjointEigenSolver`, whose
/// tridiagonalisation-plus-implicit-QL is not ported; this is nalgebra's
/// symmetric eigendecomposition with the values sorted ascending as Eigen sorts
/// them. Unlike the LDLT (decision D41) and the complete orthogonal
/// decomposition, nothing downstream branches on the result: `checkEigenvalues`
/// is called once, with `verbose = false`, and its output goes into a statistics
/// log (`sqrt_keypoint_vio.cpp:690`).
///
/// A squared prior that is not square is refused rather than handed to the
/// eigensolver, which asserts on it. The square-root branch cannot be
/// non-square: `HᵀH` is square whatever `H` is.
pub fn check_eigenvalues<S: LieScalar>(mld: &MargLinData<S>) -> Result<DVector<f64>, MargError> {
    let h_d: DMatrix<f64> = mld.h.map(|v| v.to_f64());
    let h: DMatrix<f64> = if mld.is_sqrt {
        h_d.transpose() * &h_d
    } else {
        if h_d.nrows() != h_d.ncols() {
            return Err(MargError::NotSquare {
                rows: h_d.nrows(),
                cols: h_d.ncols(),
            });
        }
        h_d
    };
    let mut values: DVector<f64> = h.symmetric_eigenvalues();
    values.as_mut_slice().sort_by(f64::total_cmp);
    Ok(values)
}
