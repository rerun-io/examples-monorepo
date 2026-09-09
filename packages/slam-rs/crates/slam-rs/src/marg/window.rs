//! The sliding-window half of `SqrtKeypointVioEstimator::marginalize`
//! (`src/vi_estimator/sqrt_keypoint_vio.cpp:707-1198`): everything that happens
//! **after** the schedule has decided what leaves.
//!
//! The schedule itself — the keyframe scoring at `:767-880`, the
//! `states_to_remove` count at `:720-724`, and the `kf_ids` bookkeeping — is
//! stage S8's and is not here. This module takes those four sets as
//! [`MarginalizeSchedule`] and does the rest: build the absolute ordering,
//! linearize the window with the current prior, split the ordering into kept
//! and marginalized indices, run [`marginalize_helper_sqrt_to_sqrt`], shrink
//! the window and the landmark database, and re-anchor the new prior.
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

use nalgebra::{DMatrix, DVector};

use crate::ba_base::{BaError, BundleAdjustmentBase};
use crate::imu::{ImuLinData, IntegratedImuMeasurement};
use crate::lie::LieScalar;
use crate::linearize::{ImuInput, LinearizationAbsQR, LinearizationInputs, LinearizationOptions};
use crate::marg::helper::{ReducedSystem, marginalize_helper_sqrt_to_sqrt};
use crate::marg::{MargError, ScheduleSet};
use crate::types::{
    AbsOrderMap, FrameId, LandmarkId, MargLinData, POSE_SIZE, POSE_VEL_BIAS_SIZE, PoseStateWithLin,
};

/// What the schedule decided, `sqrt_keypoint_vio.cpp:724-880`.
///
/// `poses_to_marg` names pose blocks that leave, `kfs_to_marg` the subset of
/// those that were keyframes hosting landmarks, `states_to_marg_all` full
/// states that leave outright, and `states_to_marg_vel_bias` full states that
/// keep their pose and lose their velocity and biases. So `poses_to_marg` is
/// disjoint from both state sets, the two state sets are disjoint from each
/// other, and `kfs_to_marg` is a subset of `poses_to_marg` rather than
/// disjoint from it. C++ gets all of that from how it builds them; the port
/// checks it (`validate_schedule`).
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
/// C++ builds the four sets and `last_state_to_marg` out of the window itself
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
/// The block size is what tells a pose from a full state, and it is what keeps
/// `poses_to_marg` disjoint from the two state sets: it must hold 6-row blocks
/// and they must hold 15-row blocks. The two state sets are checked against
/// each other below; `kfs_to_marg` is checked to be a *subset* of
/// `poses_to_marg`, which is what C++ builds it as.
///
/// The window's own linearization precondition is a different question, over a
/// different ordering: `check_prior_blocks_linearized`.
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

/// `computeDelta`'s own precondition on the ordering it is handed
/// (`ba_base.cpp:294`, `:297`): every block of it frozen at its linearization
/// point.
///
/// At `sqrt_keypoint_vio.cpp:1171` that ordering is the new prior's, so its
/// 6-row blocks are the poses that survive plus the states that are about to be
/// demoted ([`new_prior_ordering`]) — and demotion copies the state's
/// `linearized` flag over (`imu_types.h:206-215`), which is exactly what
/// [`BundleAdjustmentBase::get_pose_state_with_lin`] answers with for a block
/// that is still a full state here. So both kinds are decidable before
/// `:1090-1112` has moved anything, and the error is the one `compute_delta`
/// would raise at `:1171`, only with the window still intact. The lookup cannot
/// miss — [`new_prior_ordering`] builds the ordering out of those same two maps
/// — so `?` carries its typed error rather than a branch that cannot be taken.
/// The one 15-row block is `last_state_to_marg` (`ba_base.cpp:297`), which the
/// caller checked is *not* linearized and is about to freeze (`:1086-1088`).
fn check_prior_blocks_linearized<S: LieScalar>(
    estimator: &BundleAdjustmentBase<S>,
    marg_order_new: &AbsOrderMap,
) -> Result<(), MargError> {
    for (frame_id, _, size) in marg_order_new.iter() {
        if size == POSE_SIZE && !estimator.get_pose_state_with_lin(frame_id)?.is_linearized() {
            return Err(BaError::NotLinearized { frame_id }.into());
        }
    }
    Ok(())
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

/// Linearize the window over `aom` with `prior` included and export the
/// stacked square-root system (`sqrt_keypoint_vio.cpp:905-942`).
///
/// The branch at `:935-939` chooses `get_dense_Q2Jp_Q2r` for a square-root
/// prior and `get_dense_H_b` for a squared one. Only `ABS_QR` is ported
/// (decision D13) and only a square-root prior can exist (D68), so the port
/// takes the first unconditionally.
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
    let (h, b) = lqr.get_dense_q2jp_q2r(estimator, &lin_inputs)?;
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
///   (`new_prior_ordering`) so that `:1145`'s width check can run before the
///   first mutation.
pub fn marginalize<S: LieScalar>(
    estimator: &mut BundleAdjustmentBase<S>,
    marg_data: &mut MargLinData<S>,
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

    // `:1120-1133`, hoisted: the new prior's ordering is a function of the
    // window and the schedule, so both the checks it carries — the
    // linearization precondition `compute_delta` reaches only at `:1171`, and
    // the `:1145` width check below — run before the first mutation.
    let marg_order_new: AbsOrderMap = new_prior_ordering(estimator, schedule)?;
    check_prior_blocks_linearized(estimator, &marg_order_new)?;

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

    // `:1145`.
    if idx_to_keep.len() != marg_order_new.total_size() {
        return Err(MargError::PriorWidthMismatch {
            cols: idx_to_keep.len(),
            total_size: marg_order_new.total_size(),
        });
    }

    // `:1069-1083`.
    let reduced: ReducedSystem<S> =
        marginalize_helper_sqrt_to_sqrt(live.h, live.b, &idx_to_keep, &idx_to_marg)?;

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
    // This ordering's blocks were proven frozen before the first mutation
    // (`check_prior_blocks_linearized`, and `:1086` above for the last state).
    let delta: DVector<S> = estimator.compute_delta(&marg_data.order)?;
    subtract_h_delta(&mut marg_data.b, &marg_data.h, &delta);

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
///
/// **Contract: `b.nrows() == h.nrows()` and `delta.nrows() == h.ncols()`.**
/// Both hold where this is called: `h` and `b` are the two halves of one
/// [`ReducedSystem`], and `delta` is `compute_delta` over the ordering the
/// width check above matched `h` against. Clamping the extents instead would
/// turn a shape that does not close into a *partial* re-anchoring, which is a
/// prior that is quietly wrong rather than one that is refused.
fn subtract_h_delta<S: LieScalar>(b: &mut DVector<S>, h: &DMatrix<S>, delta: &DVector<S>) {
    debug_assert_eq!(b.nrows(), h.nrows());
    debug_assert_eq!(delta.nrows(), h.ncols());
    for i in 0..h.nrows() {
        let mut acc: S = S::zero();
        for j in 0..h.ncols() {
            acc += h[(i, j)] * delta[j];
        }
        b[i] -= acc;
    }
}
