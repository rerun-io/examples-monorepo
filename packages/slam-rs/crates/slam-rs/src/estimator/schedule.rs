//! `SqrtKeypointVioEstimator::marginalize()` (`sqrt_keypoint_vio.cpp:707-1198`).
//!
//! Only the **schedule** lives here: which states lose their velocity and bias,
//! which lose everything, which pose blocks go, and which keyframe the budget
//! evicts. The algebra — the second linearization over the evicted keyframes'
//! landmarks, the index split, the rank-revealing flat QR and the re-anchoring —
//! is [`crate::marg::marginalize`], from stage S7.
//!
//! Two things make the schedule subtle:
//!
//! * `states_to_remove` is `frame_states.size() − max_states + 1`, so
//!   `last_state_to_marg` is the **second** oldest state when the window is
//!   full, not the oldest (`:720-724`). The oldest state leaves; the second
//!   oldest is kept whole and has its linearization point frozen.
//! * The keyframe budget is only enforced when the departing state was itself a
//!   keyframe: the loop condition is
//!   `kf_ids.size() > max_kfs && !states_to_marg_vel_bias.empty()` (`:767`), and
//!   nothing in the body shrinks `states_to_marg_vel_bias`. With the shipped
//!   `vio_min_frames_after_kf = 5` keyframes are at least six frames apart, so
//!   `states_to_marg_vel_bias` holds at most one entry and the newest keyframe
//!   is always inside the two the eviction score skips.
//!
//! `KF_MARG_DEFAULT`'s second pass is a DSO-derived distance score whose own
//! comment admits it "seems to mostly marginalize the oldest keyframe"
//! (`:832-836`, D22).

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::Vector3;

use super::{EstimatorError, FrameStats, SqrtKeypointVio, duration_ns};
use crate::config::KeyframeMargCriteria;
use crate::lie::{LieScalar, Se3};
use crate::marg::{
    MarginalizeInputs, MarginalizeOptions, MarginalizeSchedule, NullspaceCheck, check_eigenvalues,
    check_marg_nullspace, marginalize as marginalize_window,
};
use crate::types::{FrameId, LandmarkId, MargLinData};

/// Which pass of the criterion chose a keyframe for eviction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionReason {
    /// `KF_MARG_DEFAULT` first pass: the oldest keyframe whose tracked
    /// fraction fell below `vio_kf_marg_feature_ratio`, or which the current
    /// frame does not observe at all (`:822-829`).
    FeatureRatio,
    /// `KF_MARG_DEFAULT` second pass: the minimum of the DSO distance score
    /// (`:845-867`).
    DistanceScore,
    /// `KF_MARG_FORWARD_VECTOR`: the least distinctive viewing direction
    /// (`:788-812`).
    ForwardVector,
}

/// One keyframe the budget evicted, and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyframeEviction {
    /// The evicted keyframe.
    pub frame_id: FrameId,
    /// Which pass selected it.
    pub reason: EvictionReason,
}

/// What one marginalization did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarginalizationStats {
    /// `states_to_remove` (`:720`).
    pub states_to_remove: usize,
    /// The state that is kept whole and frozen (`:724`, `:1086-1087`).
    pub last_state_to_marg: FrameId,
    /// Pose blocks that leave: the non-keyframes, plus the evicted keyframes
    /// (`:733`, `:875`).
    pub poses_to_marg: Vec<FrameId>,
    /// States that leave entirely (`:754`).
    pub states_to_marg_all: Vec<FrameId>,
    /// Keyframe states demoted to pose blocks (`:752`).
    pub states_to_marg_vel_bias: Vec<FrameId>,
    /// Keyframes the budget evicted, oldest first (`:874`).
    pub kfs_to_marg: Vec<FrameId>,
    /// Which criterion pass chose each eviction, in the order they happened.
    pub evictions: Vec<KeyframeEviction>,
    /// `idx_to_keep.size()` (`:1005`).
    pub kept_indices: usize,
    /// `idx_to_marg.size()`.
    pub marg_indices: usize,
    /// Whether the marginalization's own linearization was numerically valid.
    /// basalt discards this; the port reports it, because a marginalization run
    /// on an invalid linearization is worth knowing about.
    pub numerically_valid: bool,
    /// `asize`, the width of the ordering the split was taken over (`:898`).
    /// `kept_indices + marg_indices` is exactly this, by construction.
    pub ordering_size: usize,
    /// `marg_order_new`, as `(frame, index, size)` (`:1120-1133`).
    pub prior_order: Vec<(FrameId, usize, usize)>,
}

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// `marginalize(num_points_connected, lost_landmaks)` (`:707-1198`).
    ///
    /// # Errors
    ///
    /// [`EstimatorError`] where C++ asserts or reads out of range: a keyframe
    /// the eviction score needs that is not in the window, a window that
    /// disagrees with the prior's ordering, or no selectable keyframe.
    pub(super) fn marginalize(
        &mut self,
        num_points_connected: &BTreeMap<FrameId, usize>,
        lost_landmarks: &BTreeSet<LandmarkId>,
        stats: &mut FrameStats<S>,
    ) -> Result<(), EstimatorError> {
        // `:710-713`.
        if !self.opt_started {
            return Ok(());
        }
        // `:717`.
        if !(self.ba.frame_poses.len() > self.ltkfs.len() + self.max_kfs
            || self.ba.frame_states.len() >= self.max_states)
        {
            return Ok(());
        }
        let mark: std::time::Instant = std::time::Instant::now();

        // `:720-724`. C++ computes this in `size_t` and stores it in an `int`,
        // so a window shorter than `max_states - 1` wraps to a negative count
        // and the advance does not happen; saturating arithmetic is that
        // behaviour written down.
        let states_to_remove: usize =
            (self.ba.frame_states.len() + 1).saturating_sub(self.max_states);
        let Some(last_state_to_marg) = self.ba.frame_states.keys().nth(states_to_remove).copied()
        else {
            // C++ advances the iterator past `end()` and dereferences it.
            return Err(EstimatorError::KeyframeNotInWindow {
                frame_id: self.last_state_t_ns,
                wanted: "state",
            });
        };

        // `:731-740`: every pose block that is neither a keyframe nor a
        // long-term keyframe. The ordering assertion of `:737` happens inside
        // `marg::marginalize`, which rebuilds the same `AbsOrderMap`.
        let mut poses_to_marg: BTreeSet<FrameId> = BTreeSet::new();
        for frame_id in self.ba.frame_poses.keys().copied() {
            if !self.kf_ids.contains(&frame_id) && !self.ltkfs.contains(&frame_id) {
                poses_to_marg.insert(frame_id);
            }
        }

        // `:744-763`: the states older than `last_state_to_marg` split by
        // whether they are keyframes.
        let mut states_to_marg_vel_bias: BTreeSet<FrameId> = BTreeSet::new();
        let mut states_to_marg_all: BTreeSet<FrameId> = BTreeSet::new();
        for frame_id in self.ba.frame_states.keys().copied() {
            if frame_id > last_state_to_marg {
                break;
            }
            if frame_id != last_state_to_marg {
                if self.kf_ids.contains(&frame_id) || self.ltkfs.contains(&frame_id) {
                    states_to_marg_vel_bias.insert(frame_id);
                } else {
                    states_to_marg_all.insert(frame_id);
                }
            }
        }

        // `:767-880`.
        let mut kfs_to_marg: BTreeSet<FrameId> = BTreeSet::new();
        let mut evictions: Vec<KeyframeEviction> = Vec::new();
        while self.kf_ids.len() > self.max_kfs && !states_to_marg_vel_bias.is_empty() {
            let eviction: KeyframeEviction = match self.config.vio_kf_marg_criteria {
                KeyframeMargCriteria::ForwardVector => self.evict_by_forward_vector()?,
                KeyframeMargCriteria::Default => self.evict_by_default(num_points_connected)?,
            };
            kfs_to_marg.insert(eviction.frame_id);
            poses_to_marg.insert(eviction.frame_id);
            self.kf_ids.remove(&eviction.frame_id);
            evictions.push(eviction);
        }

        let schedule: MarginalizeSchedule = MarginalizeSchedule {
            last_state_to_marg,
            kfs_to_marg,
            poses_to_marg,
            states_to_marg_all,
            states_to_marg_vel_bias,
        };
        // `:1015-1065`: the debug copy is only built under the same gate that
        // logs it.
        let keep_nullspace: bool = self.config.vio_debug || self.config.vio_extended_logging;
        let inputs: MarginalizeInputs<'_, S> = MarginalizeInputs {
            schedule: &schedule,
            imu_lin_data: Some(self.imu_lin_data()),
            lost_landmarks: Some(lost_landmarks),
            fixed_frames: None,
            options: MarginalizeOptions {
                marg_lost_landmarks: self.config.vio_marg_lost_landmarks,
                keep_nullspace_marg_data: keep_nullspace,
            },
        };
        let nullspace_slot: Option<&mut MargLinData<S>> = if keep_nullspace {
            Some(&mut self.nullspace_marg_data)
        } else {
            None
        };
        let output = marginalize_window(
            &mut self.ba,
            &mut self.marg_data,
            nullspace_slot,
            &mut self.imu_meas,
            &inputs,
        )?;

        // `:1091-1094` and `:1108-1111`: the two estimator-side maps
        // `marg::marginalize` does not own. A demoted keyframe keeps both — it
        // is still in the window as a pose.
        self.last_marginalized.clear();
        for frame_id in schedule
            .states_to_marg_all
            .iter()
            .chain(schedule.poses_to_marg.iter())
            .copied()
        {
            self.frame_idx.remove(&frame_id);
            self.prev_opt_flow_res.remove(&frame_id);
            self.last_marginalized.push(frame_id);
        }
        self.last_marginalized.sort_unstable();
        self.last_marginalized.dedup();

        stats.marginalization = Some(MarginalizationStats {
            states_to_remove,
            last_state_to_marg,
            poses_to_marg: schedule.poses_to_marg.iter().copied().collect(),
            states_to_marg_all: schedule.states_to_marg_all.iter().copied().collect(),
            states_to_marg_vel_bias: schedule.states_to_marg_vel_bias.iter().copied().collect(),
            kfs_to_marg: schedule.kfs_to_marg.iter().copied().collect(),
            evictions,
            kept_indices: output.idx_to_keep.len(),
            marg_indices: output.idx_to_marg.len(),
            numerically_valid: output.numerically_valid,
            ordering_size: output.aom.total_size(),
            prior_order: self.marg_data.order.iter().collect(),
        });

        // `:1180-1184`, `:670-681`.
        if keep_nullspace {
            let (check, eigenvalues) = self.log_marg_nullspace()?;
            stats.nullspace = Some(check);
            stats.nullspace_eigenvalues = Some(eigenvalues);
        }
        stats.timings.marginalize_ns = duration_ns(mark);
        Ok(())
    }

    /// `logMargNullspace()` (`:670-681`).
    ///
    /// The order is copied from the live prior **here**, which is what makes the
    /// debug copy's `H` and its ordering agree again after the marginalization
    /// replaced both. The control direction is a parameter of
    /// [`check_marg_nullspace`] rather than `inc_random.setRandom()`
    /// (`sqrt_ba_base.cpp:158`), which is `rand()`-seeded and not reproducible;
    /// a fixed direction keeps the diagnostic deterministic (D17).
    fn log_marg_nullspace(&mut self) -> Result<(NullspaceCheck, Vec<f64>), EstimatorError> {
        self.nullspace_marg_data.order = self.marg_data.order.clone();
        let width: usize = self.nullspace_marg_data.order.total_size();
        let mut direction: nalgebra::DVector<f64> = nalgebra::DVector::zeros(width);
        for (i, slot) in direction.iter_mut().enumerate() {
            // A fixed, non-degenerate direction: the diagnostic normalizes it
            // per probe, so only its non-alignment with the gauge directions
            // matters.
            *slot = 1.0 / f64::from(u32::try_from(i).unwrap_or(u32::MAX) + 1);
        }
        let check: NullspaceCheck =
            check_marg_nullspace(&self.nullspace_marg_data, &self.ba, &direction)?;
        let eigenvalues: nalgebra::DVector<f64> = check_eigenvalues(&self.nullspace_marg_data);
        Ok((check, eigenvalues.iter().copied().collect()))
    }

    /// `KF_MARG_DEFAULT` (`:814-868`).
    ///
    /// First pass: walk from the oldest keyframe, skipping the newest two, and
    /// take the first whose tracked fraction is below
    /// `vio_kf_marg_feature_ratio` — or which the current frame does not
    /// observe at all, which is what `num_points_connected.count(*it) == 0`
    /// means. The ratio is computed in **`float`** whatever the estimator's
    /// scalar is (`static_cast<float>` at `:826`) and then promoted to `double`
    /// for the comparison against the config field, so the `f64` instantiation
    /// compares a single-precision quotient.
    ///
    /// Second pass: the DSO score `sqrt(‖p_i − p_last‖) · Σ_j 1/(‖p_i − p_j‖ +
    /// 1e-5)`, minimized. The norms are Eigen's three-coefficient reduction,
    /// whose order differs between the precisions (D47).
    fn evict_by_default(
        &self,
        num_points_connected: &BTreeMap<FrameId, usize>,
    ) -> Result<KeyframeEviction, EstimatorError> {
        let candidates: Vec<FrameId> = self.eviction_candidates();

        // `:820-830`.
        for frame_id in &candidates {
            let hosted: usize = match self.num_points_kf.get(frame_id) {
                Some(count) => *count,
                // `:827` reads `num_points_kf.at(*it)`; basalt never erases the
                // map, so a keyframe always has an entry.
                None => {
                    return Err(EstimatorError::KeyframeNotInWindow {
                        frame_id: *frame_id,
                        wanted: "hosted-landmark count",
                    });
                }
            };
            let connected: Option<usize> = num_points_connected.get(frame_id).copied();
            let below: bool = match connected {
                None => true,
                Some(connected) => {
                    let ratio: f32 = connected as f32 / hosted as f32;
                    f64::from(ratio) < self.config.vio_kf_marg_feature_ratio
                }
            };
            if below {
                return Ok(KeyframeEviction {
                    frame_id: *frame_id,
                    reason: EvictionReason::FeatureRatio,
                });
            }
        }

        // `:838-867`.
        let Some(last_kf) = self.kf_ids.iter().next_back().copied() else {
            return Err(EstimatorError::NoKeyframeToMarginalize { candidates: 0 });
        };
        // `:854`: `frame_states.at(last_kf)`. The newest keyframe is the newest
        // frame whenever `take_kf` fired on it, and keyframes are at least six
        // frames apart, so it is a state in every shipped configuration.
        let Some(last_state) = self.ba.frame_states.get(&last_kf) else {
            return Err(EstimatorError::KeyframeNotInWindow {
                frame_id: last_kf,
                wanted: "state",
            });
        };
        let last_translation: Vector3<S> = last_state.state().t_w_i.translation;

        let mut min_score: S = S::max_value().unwrap_or_else(S::one);
        let mut min_score_id: Option<FrameId> = None;
        for frame_id in &candidates {
            let here: Vector3<S> = self.keyframe_pose(*frame_id)?.translation;
            // `:848-853`: the sum runs over the same candidate set, so a
            // keyframe's distance to itself contributes `1 / 1e-5`.
            let mut denom: S = S::zero();
            for other in &candidates {
                let there: Vector3<S> = self.keyframe_pose(*other)?.translation;
                let d: Vector3<S> = here - there;
                denom += S::one() / (eigen_norm3(&d) + S::from_literal(1e-5));
            }
            let d: Vector3<S> = here - last_translation;
            let score: S = eigen_norm3(&d).sqrt() * denom;
            if score < min_score {
                min_score = score;
                min_score_id = Some(*frame_id);
            }
        }
        // `:872`: "if no frame was selected, the logic above is faulty".
        min_score_id
            .map(|frame_id| KeyframeEviction {
                frame_id,
                reason: EvictionReason::DistanceScore,
            })
            .ok_or(EstimatorError::NoKeyframeToMarginalize {
                candidates: candidates.len(),
            })
    }

    /// `KF_MARG_FORWARD_VECTOR` (`:770-813`), the fork's own criterion.
    ///
    /// Score each candidate keyframe by the sum of angles between its camera-0
    /// forward vector and every other keyframe's — long-term keyframes
    /// included, which is the difference from the default criterion's candidate
    /// set — and evict the minimum, i.e. the least distinctive viewing
    /// direction. `acos` is called unqualified inside `namespace basalt`, so the
    /// `float` instantiation resolves to `::acosf` (the same overload trap as
    /// `atan2` in D42); computing in `S` reproduces that.
    fn evict_by_forward_vector(&self) -> Result<KeyframeEviction, EstimatorError> {
        let candidates: Vec<FrameId> = self.eviction_candidates();
        // `:781-786`: the scored-against set is `ltkfs ∪ kf_ids`, again without
        // its own newest two.
        let mut all_kfs: BTreeSet<FrameId> = self.ltkfs.clone();
        all_kfs.extend(self.kf_ids.iter().copied());
        let against: Vec<FrameId> = all_kfs
            .iter()
            .copied()
            .take(all_kfs.len().saturating_sub(2))
            .collect();

        let mut min_score: S = S::max_value().unwrap_or_else(S::one);
        let mut min_score_id: Option<FrameId> = None;
        for frame_id in &candidates {
            let fwd1: (S, S) = self.forward_vector_2d(*frame_id)?;
            let mut score: S = S::zero();
            for other in &against {
                let fwd2: (S, S) = self.forward_vector_2d(*other)?;
                // `fwd1.dot(fwd2)` on a 2-vector is Eigen's coefficient-based
                // product: `a0 * b0 + a1 * b1`.
                let dot: S = fwd1.0 * fwd2.0 + fwd1.1 * fwd2.1;
                // `std::clamp(v, lo, hi)` = `v < lo ? lo : (hi < v ? hi : v)`,
                // which returns a NaN unchanged.
                let clamped: S = if dot < -S::one() {
                    -S::one()
                } else if S::one() < dot {
                    S::one()
                } else {
                    dot
                };
                score += clamped.acos();
            }
            if score < min_score {
                min_score = score;
                min_score_id = Some(*frame_id);
            }
        }
        min_score_id
            .map(|frame_id| KeyframeEviction {
                frame_id,
                reason: EvictionReason::ForwardVector,
            })
            .ok_or(EstimatorError::NoKeyframeToMarginalize {
                candidates: candidates.len(),
            })
    }

    /// `kf_ids` without its newest two (`std::prev(kf_ids.end(), 2)` at `:819`
    /// and `:840`), empty when there are two or fewer — which is the
    /// `kf_ids.size() > 2` guard C++ writes to keep `std::prev` valid.
    fn eviction_candidates(&self) -> Vec<FrameId> {
        if self.kf_ids.len() <= 2 {
            return Vec::new();
        }
        self.kf_ids
            .iter()
            .copied()
            .take(self.kf_ids.len() - 2)
            .collect()
    }

    /// `frame_poses.at(ts).getPose()` (`:794`, `:849`), which C++ throws out of
    /// when the keyframe is not a pose block.
    fn keyframe_pose(&self, frame_id: FrameId) -> Result<Se3<S>, EstimatorError> {
        self.ba
            .frame_poses
            .get(&frame_id)
            .map(|pose| *pose.pose())
            .ok_or(EstimatorError::KeyframeNotInWindow {
                frame_id,
                wanted: "pose",
            })
    }

    /// `get_forward_vector2d` (`:792-798`): the camera-0 optical axis in the
    /// world frame, projected onto the horizontal plane.
    fn forward_vector_2d(&self, frame_id: FrameId) -> Result<(S, S), EstimatorError> {
        let t_w_i: Se3<S> = self.keyframe_pose(frame_id)?;
        let Some(t_i_c0) = self.ba.calib.t_i_c.first() else {
            return Err(EstimatorError::CameraCountMismatch {
                expected: 1,
                actual: 0,
            });
        };
        let t_w_c0: Se3<S> = t_w_i * *t_i_c0;
        let fwd: Vector3<S> = t_w_c0.rotation * Vector3::new(S::zero(), S::zero(), S::one());
        Ok((fwd[0], fwd[1]))
    }
}

/// `Vector3::norm()`, which is `sqrt(squaredNorm())` and therefore Eigen's
/// three-coefficient reduction (D47).
fn eigen_norm3<S: LieScalar>(v: &Vector3<S>) -> S {
    crate::landmark::eigen_norm3(v[0], v[1], v[2])
}
