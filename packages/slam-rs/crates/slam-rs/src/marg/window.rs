//! Apply a marginalization schedule to the sliding window.
//! Build ordering, linearize with the current prior, split indices, eliminate,
//! shrink state and landmark maps, then re-anchor the new prior.
//!
//! Freeze the newest prior state before computing its delta (trap 7).
//! Convert the helper's residual back to delta-independent form by subtracting
//! `H * delta` (trap 8); omitting either step causes silent drift.

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

/// Scheduled removals: poses, their hosted keyframes, full states, and state
/// velocity/bias blocks. Pose removals and the two state sets are disjoint;
/// keyframe removals are a subset of pose removals. Validate these relationships
/// before changing the window.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MarginalizeSchedule {
    /// `last_state_to_marg` : the newest state the ordering reaches and
    /// the one that becomes the prior's 15-dof block.
    pub last_state_to_marg: FrameId,
    /// `kfs_to_marg`.
    pub kfs_to_marg: BTreeSet<FrameId>,
    /// `poses_to_marg`, a superset of `kfs_to_marg`.
    pub poses_to_marg: BTreeSet<FrameId>,
    /// `states_to_marg_all`.
    pub states_to_marg_all: BTreeSet<FrameId>,
    /// `states_to_marg_vel_bias`.
    pub states_to_marg_vel_bias: BTreeSet<FrameId>,
}

/// The configuration bits `marginalize()` reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MarginalizeOptions {
    /// `config.vio_marg_lost_landmarks`.
    pub marg_lost_landmarks: bool,
}

/// Everything `marginalize()` needs besides the window itself.
#[derive(Debug)]
pub struct MarginalizeInputs<'a, S: LieScalar> {
    /// What the schedule decided.
    pub schedule: &'a MarginalizeSchedule,
    /// Gravity and the two bias random-walk weights; `None` for a visual-only
    /// window (`ImuLinData` at ).
    pub imu_lin_data: Option<ImuLinData<S>>,
    /// `lost_landmaks`, the landmarks the frontend stopped tracking.
    pub lost_landmarks: Option<&'a BTreeSet<LandmarkId>>,
    /// `fixed_kfs` : `ltkfs` when `config.vio_fix_long_term_keyframes`
    /// is on, empty otherwise.
    pub fixed_frames: Option<&'a BTreeSet<FrameId>>,
    /// Flags.
    pub options: MarginalizeOptions,
}

/// What one marginalization produced besides the updated prior.
#[derive(Debug, Clone, PartialEq)]
pub struct MarginalizeOutput<S: LieScalar> {
    /// `aom` : the ordering the marginalization linearized over.
    pub aom: AbsOrderMap,
    /// `idx_to_keep`.
    pub idx_to_keep: BTreeSet<usize>,
    /// `idx_to_marg`.
    pub idx_to_marg: BTreeSet<usize>,
    /// What `linearizeProblem` reported for the window that is leaving.
    pub error: S,
    /// False if a landmark Jacobian was non-finite and zeroed with a warning.
    pub numerically_valid: bool,
}

/// Order all poses by timestamp, then states through `last_state_to_marg`.
/// The prior must be an exact prefix at the same offsets; otherwise the two
/// systems assign different variables to the same columns.
fn build_absolute_ordering<S: LieScalar>(
    estimator: &BundleAdjustmentBase<S>,
    marg_data: &MargLinData<S>,
    last_state_to_marg: FrameId,
) -> Result<AbsOrderMap, MargError> {
    let mut aom: AbsOrderMap = AbsOrderMap::new();
    for frame_id in estimator.frame_poses.keys() {
        let offset: usize = aom.push(*frame_id, POSE_SIZE)?;
        // Every prior pose must have a matching ordering entry.
        if marg_data.order.get(*frame_id) != Some((offset, POSE_SIZE)) {
            return Err(MargError::PriorOrderMismatch {
                frame_id: *frame_id,
            });
        }
    }
    for frame_id in estimator.frame_states.keys() {
        if *frame_id > last_state_to_marg {
            break;
        }
        let offset: usize = aom.push(*frame_id, POSE_VEL_BIAS_SIZE)?;
        // the comparison is against `aom.items` **before** the
        // increment, so a state past the end of the prior is not
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

/// Validate the entire schedule before mutation (D32).
/// Ordering membership establishes frame membership; block sizes distinguish
/// poses from full states. Check disjoint state sets and that removed keyframes
/// are a subset of removed poses. This prevents partial updates on refusal.
/// Frozen-linearization preconditions are checked separately over the new prior.
fn validate_schedule(aom: &AbsOrderMap, schedule: &MarginalizeSchedule) -> Result<(), MargError> {
    //  and : every pose that leaves is a pose block of the window.
    for frame_id in &schedule.poses_to_marg {
        if !matches!(aom.get(*frame_id), Some((_, POSE_SIZE))) {
            return Err(MargError::ScheduledFrameNotInOrdering {
                set: ScheduleSet::PosesToMarg,
                frame_id: *frame_id,
                block: POSE_SIZE,
            });
        }
    }

    for frame_id in &schedule.kfs_to_marg {
        if !schedule.poses_to_marg.contains(frame_id) {
            return Err(MargError::KeyframeNotInPosesToMarg {
                frame_id: *frame_id,
            });
        }
    }

    //  and : both state sets are full states of the window, and
    // 's `if (kv.first != last_state_to_marg)` keeps the newest state out
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

    // the two state sets are an if/else over the same frame.
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

/// Construct the surviving prior ordering before mutation from
/// `(frame_poses ∪ states_to_marg_vel_bias) \ poses_to_marg` in sorted order.
/// This lets width validation fail without changing the prior or window.
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

/// Check that all surviving pose blocks are frozen before computing deltas.
/// Demoted states carry their first six delta entries and frozen flag, so their
/// precondition can be checked before demotion. The one full-state block is
/// `last_state_to_marg`, which is checked unfrozen and will be frozen next.
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
///
/// A pose block goes whole or stays whole. A full state either goes whole
/// (`states_to_marg_all`), keeps its six pose rows and loses the other nine
/// (`states_to_marg_vel_bias`), or stays whole — and the last case is asserted
/// to be `last_state_to_marg` alone.
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
                    if frame_id != schedule.last_state_to_marg {
                        return Err(MargError::UnscheduledState { frame_id });
                    }
                    idx_to_keep.extend(start_idx..start_idx + POSE_VEL_BIAS_SIZE);
                }
            }
            // Only pose and full-state block sizes are valid.
            size => return Err(MargError::UnexpectedBlockSize { frame_id, size }),
        }
    }
    Ok((idx_to_keep, idx_to_marg))
}

/// Linearize the window over `aom` with `prior` included and export the
/// stacked square-root system.
///
/// Only square-root priors exist, so this always exports `Q2Jp` and `Q2r` (D68).
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
        // only landmarks hosted by a marginalized keyframe, or lost.
        used_frames: Some(&inputs.schedule.kfs_to_marg),
        lost_landmarks: inputs.lost_landmarks,
        fixed_frames: inputs.fixed_frames,
    };
    // The linearizer copies Huber threshold and observation deviation from the estimator.
    let mut lqr: LinearizationAbsQR<S> =
        LinearizationAbsQR::new(estimator, aom, LinearizationOptions::default(), &lin_inputs)?;
    let (error, numerically_valid) = lqr.linearize_problem(estimator, &lin_inputs)?;
    lqr.perform_qr()?;
    let (h, b) = lqr.get_dense_q2jp_q2r(estimator, &lin_inputs)?;
    Ok(LinearizedWindow {
        h,
        b,
        error,
        numerically_valid,
    })
}

/// What one pass of produced.
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

/// Marginalize in place and return ordering and index split for diagnostics.
/// The prior is replaced, frame and landmark maps shrink, and consumed IMU factors leave.
/// Freeze `last_state_to_marg` before computing delta so its new prior delta is zero.
/// Construct the surviving ordering from the schedule before mutation, allowing
/// width and frozen-state checks to fail without a partial window update.
pub fn marginalize<S: LieScalar>(
    estimator: &mut BundleAdjustmentBase<S>,
    marg_data: &mut MargLinData<S>,
    imu_meas: &mut BTreeMap<i64, IntegratedImuMeasurement<S>>,
    inputs: &MarginalizeInputs<'_, S>,
) -> Result<MarginalizeOutput<S>, MargError> {
    let schedule: &MarginalizeSchedule = inputs.schedule;
    let last_state_to_marg: FrameId = schedule.last_state_to_marg;

    let aom: AbsOrderMap = build_absolute_ordering(estimator, marg_data, last_state_to_marg)?;

    // Check the freeze precondition before mutation so refusal leaves the window intact.
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
    // than discovered halfway through.
    validate_schedule(&aom, schedule)?;

    // Compute the new ordering before mutation to check its width and frozen blocks.
    let marg_order_new: AbsOrderMap = new_prior_ordering(estimator, schedule)?;
    check_prior_blocks_linearized(estimator, &marg_order_new)?;

    // the intervals whose two ends are both in the ordering.
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

    let live: LinearizedWindow<S> =
        linearize_for_marginalization(estimator, &aom, marg_data, imu_input.as_ref(), inputs)?;

    let (idx_to_keep, idx_to_marg) = split_indices(&aom, schedule)?;

    if idx_to_keep.len() != marg_order_new.total_size() {
        return Err(MargError::PriorWidthMismatch {
            cols: idx_to_keep.len(),
            total_size: marg_order_new.total_size(),
        });
    }

    let reduced: ReducedSystem<S> =
        marginalize_helper_sqrt_to_sqrt(live.h, live.b, &idx_to_keep, &idx_to_marg)?;

    // The linearization is done with the window; everything from here mutates.
    drop(imu_input);

    // trap 7. `validate_schedule` and the check above
    // both prove the lookup, so the `else` is unreachable; it is an error
    // rather than a skip because silently not freezing the state is trap 7
    // happening.
    let Some(state) = estimator.frame_states.get_mut(&last_state_to_marg) else {
        return Err(MargError::FrameNotInWindow {
            frame_id: last_state_to_marg,
        });
    };
    state.set_linearized()?;

    for id in &schedule.states_to_marg_all {
        estimator.frame_states.remove(id);
        imu_meas.remove(id);
    }

    // a keyframe that keeps its pose is demoted to a pose block,
    // carrying the first six entries of its delta and its `linearized` flag
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

    for id in &schedule.poses_to_marg {
        estimator.frame_poses.remove(id);
    }

    estimator.lmdb.remove_keyframes(
        &schedule.kfs_to_marg,
        &schedule.poses_to_marg,
        &schedule.states_to_marg_all,
    );

    if inputs.options.marg_lost_landmarks
        && let Some(lost) = inputs.lost_landmarks
    {
        for lm_id in lost {
            estimator.lmdb.remove_landmark(*lm_id);
        }
    }

    // The helper output width equals the kept-index count, already checked against ordering.
    marg_data.h = reduced.h;
    marg_data.b = reduced.b;
    marg_data.order = marg_order_new;

    // trap 8. The prior comes out of the helper as
    // `P(x) = 0.5‖J x + res‖²`; putting it back into the delta-independent form
    // `P(x) = 0.5‖J (delta + x) + (res − J delta)‖²` is one subtraction.
    // This ordering's blocks were proven frozen before the first mutation
    // (`check_prior_blocks_linearized`, and above for the last state).
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

/// `b -= H * delta`, written out so the
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
