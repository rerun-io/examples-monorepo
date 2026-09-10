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
//! * The keyframe budget is **lazy**. The eviction loop runs only while
//!   `kf_ids.size() > max_kfs && !states_to_marg_vel_bias.empty()` (`:767`),
//!   and nothing in the body shrinks `states_to_marg_vel_bias` — the keyframes
//!   leaving the *state* window this step. A frame that has just been voted a
//!   keyframe is still a state, so it cannot be evicted, and `kf_ids` sits one
//!   over `max_kfs` until it is demoted to a pose block. On the smoke segment
//!   that is framesets 49-50 and 56-57, eight keyframes against
//!   `vio_max_kfs = 7`, and the C++ does exactly the same. The overshoot is at
//!   most one because `vio_min_frames_after_kf = 5` puts keyframes six
//!   framesets apart while a state leaves after `vio_max_states = 3` — which is
//!   also why `states_to_marg_vel_bias` holds at most one entry and the newest
//!   keyframe is always inside the two the eviction score skips.
//!
//! `KF_MARG_DEFAULT`'s second pass is a DSO-derived distance score whose own
//! comment admits it "seems to mostly marginalize the oldest keyframe"
//! (`:832-836`, D22) — which is not what it does; see
//! [`SqrtKeypointVio::evict_by_default`].

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::Vector3;

use super::{EstimatorError, SqrtKeypointVio, WindowRole, duration_ns, fixed_keyframes};
use crate::config::KeyframeMargCriteria;
use crate::lie::{LieScalar, Se3};
use crate::marg::{
    MarginalizeInputs, MarginalizeOptions, MarginalizeSchedule, marginalize as marginalize_window,
};
use crate::types::{FrameId, LandmarkId};

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
    ///
    /// It witnesses that the split covers the ordering: `kept_indices +
    /// marg_indices == ordering_size` is what "every variable is either kept or
    /// marginalized" means, and `tests/vio_oracle.rs` asserts that sum.
    pub ordering_size: usize,
    /// `marg_order_new`, as `(frame, index, size)` (`:1120-1133`).
    pub prior_order: Vec<(FrameId, usize, usize)>,
}

/// What one call to [`SqrtKeypointVio::marginalize`] did, on its way into
/// [`FrameStats`].
///
/// Everything is empty when the trigger of `:717` did not fire, which is the
/// common case: the window marginalizes on roughly one frameset in two.
#[derive(Debug, Clone, Default)]
pub(super) struct MarginalizationOutcome {
    /// The marginalization itself.
    pub marginalization: Option<MarginalizationStats>,
    /// `StageTimings::marginalize_ns`.
    pub elapsed_ns: u64,
}

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// `:465-472`: move the newest keyframe out of the `max_kfs` budget, if
    /// [`SqrtKeypointVio::take_long_term_keyframe`] asked for it.
    ///
    /// The long-term keyframes are counted on the other side of `:717`'s pose
    /// budget and are never eviction candidates, so this is the one way a
    /// keyframe leaves the budget without leaving the window. The request is
    /// consumed whether or not there was a keyframe to demote, as in C++.
    pub(super) fn demote_long_term_keyframe(&mut self) {
        if !self.take_ltkf {
            return;
        }
        if let Some(last_kf) = self.kf_ids.iter().next_back().copied() {
            self.ltkfs.insert(last_kf);
            self.kf_ids.remove(&last_kf);
        }
        self.take_ltkf = false;
    }

    /// `marginalize(num_points_connected, lost_landmaks)` (`:707-1198`),
    /// returning what it did and how long it took. Every component is
    /// `None`/zero when the trigger of `:717` did not fire.
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
    ) -> Result<MarginalizationOutcome, EstimatorError> {
        // `:710-713`.
        if !self.opt_started {
            return Ok(MarginalizationOutcome::default());
        }
        // `:717`.
        if !(self.ba.frame_poses.len() > self.ltkfs.len() + self.max_kfs
            || self.ba.frame_states.len() >= self.max_states)
        {
            return Ok(MarginalizationOutcome::default());
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
            return Err(EstimatorError::StateWindowTooShort {
                states: self.ba.frame_states.len(),
                states_to_remove,
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
        let inputs: MarginalizeInputs<'_, S> = MarginalizeInputs {
            schedule: &schedule,
            imu_lin_data: Some(self.imu_lin_data()),
            lost_landmarks: Some(lost_landmarks),
            // `:924`: the same set `optimize` builds at `:1258`.
            fixed_frames: fixed_keyframes(&self.config, &self.ltkfs),
            options: MarginalizeOptions {
                marg_lost_landmarks: self.config.vio_marg_lost_landmarks,
            },
        };
        let output = marginalize_window(
            &mut self.ba,
            &mut self.marg_data,
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

        let marginalization: MarginalizationStats = MarginalizationStats {
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
        };

        Ok(MarginalizationOutcome {
            marginalization: Some(marginalization),
            elapsed_ns: duration_ns(mark),
        })
    }

    /// `KF_MARG_DEFAULT` (`:814-868`).
    ///
    /// First pass: walk from the oldest keyframe, skipping the newest two, and
    /// take the first whose tracked fraction is below
    /// `vio_kf_marg_feature_ratio` — or which the current frame does not
    /// observe at all, which is what `num_points_connected.count(*it) == 0`
    /// means. The ratio is computed in **`float`** whatever the estimator's
    /// scalar is (`static_cast<float>` at `:826`) and then promoted to `double`
    /// for the comparison, so the `f64` instantiation compares a
    /// single-precision quotient. That cast cannot change the outcome for any
    /// count the window can hold: distinct quotients of counts below a million
    /// are at least `1e-6` apart while one `f32` ulp near `0.1` is `7.5e-9`,
    /// and the one quotient that lands exactly on the threshold —
    /// `connected · 10 == hosted` — rounds *up* in `f32` and so fails the
    /// strict `<` in both precisions. It is ported because it is what the C++
    /// computes, not because a decision turns on it.
    ///
    /// Second pass: the DSO score `sqrt(‖p_i − p_last‖) · Σ_j 1/(‖p_i − p_j‖ +
    /// 1e-5)`, minimized, with the sum running over the candidate set
    /// *including `i` itself* (`:848`). That self term is `1/1e-5 = 1e5` and
    /// swamps the metre-scale distances to the other keyframes, so what
    /// actually discriminates is `sqrt(‖p_i − p_last‖)` and the candidate
    /// **nearest the newest keyframe** is evicted — not the oldest that the
    /// comment at `:832-836` expects. The norms are Eigen's three-coefficient
    /// reduction, whose order differs between the precisions (D47).
    fn evict_by_default(
        &self,
        num_points_connected: &BTreeMap<FrameId, usize>,
    ) -> Result<KeyframeEviction, EstimatorError> {
        let candidates: Vec<FrameId> = self.eviction_candidates();

        // `:820-830`. The `||` at `:823-824` short-circuits on the missing
        // connection *before* it reads `num_points_kf.at(*it)`, so a keyframe
        // absent from both maps is evicted rather than refused.
        for frame_id in &candidates {
            let Some(connected) = num_points_connected.get(frame_id).copied() else {
                return Ok(KeyframeEviction {
                    frame_id: *frame_id,
                    reason: EvictionReason::FeatureRatio,
                });
            };
            // `:827` reads `num_points_kf.at(*it)`; basalt never erases the
            // map, so a keyframe the frame does see always has an entry.
            let Some(hosted) = self.num_points_kf.get(frame_id).copied() else {
                return Err(EstimatorError::KeyframeNotInWindow {
                    frame_id: *frame_id,
                    wanted: WindowRole::HostedLandmarkCount,
                });
            };
            let ratio: f32 = connected as f32 / hosted as f32;
            if f64::from(ratio) < self.config.vio_kf_marg_feature_ratio {
                return Ok(KeyframeEviction {
                    frame_id: *frame_id,
                    reason: EvictionReason::FeatureRatio,
                });
            }
        }

        // `:838-867`. `:842`: `std::numeric_limits<Scalar>::max()`.
        let mut min_score: S = S::largest();
        let mut min_score_id: Option<FrameId> = None;
        // `:841`: `*kf_ids.crbegin()`. Every candidate comes out of `kf_ids`,
        // so an empty keyframe set has no candidate either and the one
        // `NoKeyframeToMarginalize` below reports it.
        if let Some(last_kf) = self.kf_ids.iter().next_back().copied() {
            // `:854`: `frame_states.at(last_kf)`. The newest keyframe is the
            // newest frame whenever `take_kf` fired on it, and keyframes are at
            // least six frames apart, so it is a state in every shipped
            // configuration.
            let Some(last_state) = self.ba.frame_states.get(&last_kf) else {
                return Err(EstimatorError::KeyframeNotInWindow {
                    frame_id: last_kf,
                    wanted: WindowRole::State,
                });
            };
            let last_translation: Vector3<S> = last_state.state().t_w_i.translation;
            for frame_id in &candidates {
                let here: Vector3<S> = self.keyframe_pose(*frame_id)?.translation;
                // `:848-853`: the sum runs over the same candidate set, so a
                // keyframe's distance to itself contributes `1 / 1e-5`.
                let mut denom: S = S::zero();
                for other in &candidates {
                    let there: Vector3<S> = self.keyframe_pose(*other)?.translation;
                    let d: Vector3<S> = here - there;
                    denom += S::one() / (d.norm() + S::from_literal(1e-5));
                }
                let d: Vector3<S> = here - last_translation;
                let score: S = d.norm().sqrt() * denom;
                if score < min_score {
                    min_score = score;
                    min_score_id = Some(*frame_id);
                }
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

        // `:788`: `std::numeric_limits<Scalar>::max()`.
        let mut min_score: S = S::largest();
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
                wanted: WindowRole::Pose,
            })
    }

    /// `get_forward_vector2d` (`:792-798`): the camera-0 optical axis in the
    /// world frame, projected onto the horizontal plane.
    ///
    /// Camera 0 exists: [`SqrtKeypointVio::new`] refuses a rig of fewer than
    /// two cameras.
    fn forward_vector_2d(&self, frame_id: FrameId) -> Result<(S, S), EstimatorError> {
        let t_w_i: Se3<S> = self.keyframe_pose(frame_id)?;
        let t_w_c0: Se3<S> = t_w_i * self.ba.calib.t_i_c[0];
        let fwd: Vector3<S> = t_w_c0.rotation * Vector3::new(S::zero(), S::zero(), S::one());
        Ok((fwd[0], fwd[1]))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use nalgebra::Vector3;

    use super::*;
    use crate::calib::Calibration;
    use crate::config::VioConfig;
    use crate::lie::So3;
    use crate::types::{PoseStateWithLin, PoseVelBiasState, PoseVelBiasStateWithLin};

    const CALIB: &str = include_str!("../../tests/fixtures/msdmi_calib.json");
    const CONFIG: &str = include_str!("../../../../configs/msdmi_config.json");

    /// Six keyframes all facing the same way: the default criterion never
    /// reads the azimuth.
    const FLAT: [f64; 6] = [0.0; 6];

    /// Six keyframes one metre apart along `x` at the given azimuths, the
    /// newest also a state because the distance score reads
    /// `frame_states.at(last_kf)` (`:854`), each hosting ten landmarks.
    ///
    /// The azimuth is a rotation about the **world z**, so it turns the
    /// camera-0 forward vector's `(x, y)` part without changing its length —
    /// which makes `fwd_i · fwd_j` a monotone function of the azimuth
    /// difference alone and the forward-vector score predictable. A rotation
    /// about `y` would not: it changes the length of the `head<2>()` the
    /// criterion takes, and the scores stop being comparable.
    fn a_window_of_keyframes(
        criteria: KeyframeMargCriteria,
        azimuths: [f64; 6],
    ) -> SqrtKeypointVio<f64> {
        let mut config: VioConfig = VioConfig::from_json_str(CONFIG).unwrap();
        config.vio_kf_marg_criteria = criteria;
        let calibration: Calibration<f64> = Calibration::from_json_str(CALIB).unwrap();
        let mut vio: SqrtKeypointVio<f64> =
            SqrtKeypointVio::new(Vector3::new(0.0, 0.0, -9.81), calibration, config).unwrap();
        for (index, azimuth) in azimuths.into_iter().enumerate() {
            let t_ns: FrameId = i64::try_from(index).unwrap() * 1_000_000;
            #[expect(
                clippy::cast_precision_loss,
                reason = "six small integers, exactly representable"
            )]
            let pose: Se3<f64> = Se3::new(
                So3::exp(&Vector3::new(0.0, 0.0, azimuth)),
                Vector3::new(index as f64, 0.0, 0.0),
            );
            vio.kf_ids.insert(t_ns);
            vio.num_points_kf.insert(t_ns, 10);
            vio.ba
                .frame_poses
                .insert(t_ns, PoseStateWithLin::new(t_ns, pose, true));
            if index == azimuths.len() - 1 {
                vio.ba.frame_states.insert(
                    t_ns,
                    PoseVelBiasStateWithLin::new(
                        PoseVelBiasState::new(
                            t_ns,
                            pose,
                            Vector3::zeros(),
                            Vector3::zeros(),
                            Vector3::zeros(),
                        ),
                        false,
                    ),
                );
            }
        }
        vio
    }

    /// Every keyframe well tracked, so nothing to marginalize on the ratio.
    fn all_connected(vio: &SqrtKeypointVio<f64>) -> BTreeMap<FrameId, usize> {
        vio.kf_ids.iter().map(|t_ns| (*t_ns, 10)).collect()
    }

    /// `fixed_kfs` reaches **both** linearizations. C++ builds it at `:924`
    /// for the marginalization and at `:1258` for the optimization, so with
    /// `vio_fix_long_term_keyframes` on a long-term keyframe's pose Jacobians
    /// are zeroed in the prior it computes as well as in the increment it
    /// solves. No shipped config sets the flag and nothing in the VIO calls
    /// `take_long_term_keyframe`, so this is the only coverage.
    #[test]
    fn a_long_term_keyframe_is_fixed_in_both_linearizations() {
        let mut vio: SqrtKeypointVio<f64> =
            a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        vio.take_long_term_keyframe();
        vio.demote_long_term_keyframe();
        assert!(!vio.ltkfs.is_empty());

        assert_eq!(
            fixed_keyframes(&vio.config, &vio.ltkfs),
            None,
            "the shipped config leaves every keyframe free"
        );

        vio.config.vio_fix_long_term_keyframes = true;
        assert_eq!(
            fixed_keyframes(&vio.config, &vio.ltkfs),
            Some(&vio.ltkfs),
            "the flag fixes the long-term keyframes"
        );
    }

    /// `takeLongTermKeyframe()` (`:124-127`) and the demotion it asks for
    /// (`:465-472`): the newest keyframe leaves `kf_ids` for `ltkfs`, which
    /// moves it to the other side of `:717`'s pose budget and out of the
    /// eviction candidates. Nothing in the VIO path calls it — it is the
    /// Monado/API hook — so this is its only coverage.
    #[test]
    fn a_long_term_keyframe_leaves_the_keyframe_budget() {
        let mut vio: SqrtKeypointVio<f64> =
            a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        let newest: FrameId = 5_000_000;
        let budget_before: usize = vio.ltkfs.len() + vio.max_kfs;
        assert!(vio.kf_ids.contains(&newest));

        vio.take_long_term_keyframe();
        vio.demote_long_term_keyframe();

        assert!(
            !vio.kf_ids.contains(&newest),
            "the demoted keyframe is still in the budget"
        );
        assert!(vio.ltkfs.contains(&newest), "and not in ltkfs");
        assert!(!vio.take_ltkf, "the request was not consumed");
        assert_eq!(
            vio.ltkfs.len() + vio.max_kfs,
            budget_before + 1,
            "`:717`'s pose budget did not grow by the demoted keyframe"
        );
        assert!(
            !vio.eviction_candidates().contains(&newest),
            "a long-term keyframe cannot be evicted"
        );

        // A second demotion needs a second request.
        vio.demote_long_term_keyframe();
        assert_eq!(vio.ltkfs.len(), 1);
    }

    /// `std::prev(kf_ids.end(), 2)` (`:819`, `:840`) and the `kf_ids.size() > 2`
    /// guard that keeps it valid.
    #[test]
    fn the_newest_two_keyframes_are_never_candidates() {
        let vio: SqrtKeypointVio<f64> = a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        assert_eq!(
            vio.eviction_candidates(),
            vec![0, 1_000_000, 2_000_000, 3_000_000]
        );

        let mut two: SqrtKeypointVio<f64> =
            a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        two.kf_ids.retain(|t_ns| *t_ns >= 4_000_000);
        assert!(two.eviction_candidates().is_empty());
    }

    /// `:822-829` first pass: the oldest keyframe below
    /// `vio_kf_marg_feature_ratio`, and nothing newer even if it is worse.
    #[test]
    fn the_ratio_pass_takes_the_oldest_poorly_tracked_keyframe() {
        let vio: SqrtKeypointVio<f64> = a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        let mut connected: BTreeMap<FrameId, usize> = all_connected(&vio);
        connected.insert(1_000_000, 0);
        connected.insert(2_000_000, 0);
        assert_eq!(
            vio.evict_by_default(&connected).unwrap(),
            KeyframeEviction {
                frame_id: 1_000_000,
                reason: EvictionReason::FeatureRatio,
            }
        );
    }

    /// `num_points_connected.count(*it) == 0` (`:823`): a keyframe the current
    /// frame does not observe at all is taken by the **first** pass, whatever
    /// its hosted count — the missing entry and the low ratio are one `||`.
    #[test]
    fn a_keyframe_the_frame_does_not_see_goes_first() {
        let vio: SqrtKeypointVio<f64> = a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        let mut connected: BTreeMap<FrameId, usize> = all_connected(&vio);
        connected.remove(&2_000_000);
        assert_eq!(
            vio.evict_by_default(&connected).unwrap(),
            KeyframeEviction {
                frame_id: 2_000_000,
                reason: EvictionReason::FeatureRatio,
            }
        );
    }

    /// `:845-867` second pass: the DSO score is
    /// `sqrt(‖p_i − p_last‖) · Σ_j 1/(‖p_i − p_j‖ + 1e-5)`, and the sum runs
    /// over the candidate set **including `i` itself** (`:848`), so every
    /// candidate carries a `1/1e-5 = 1e5` self term that swamps the metre-scale
    /// distances to the others. What is left to discriminate is
    /// `sqrt(‖p_i − p_last‖)`, so the candidate **nearest the newest keyframe**
    /// is evicted — on six keyframes a metre apart, the newest candidate, not
    /// the oldest that basalt's comment expects (`:832-836`, D22). Dropping the
    /// self term, which reads like a bug, would invert the answer.
    #[test]
    fn the_distance_pass_takes_the_candidate_nearest_the_newest_keyframe() {
        let vio: SqrtKeypointVio<f64> = a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        assert_eq!(
            vio.evict_by_default(&all_connected(&vio)).unwrap(),
            KeyframeEviction {
                frame_id: 3_000_000,
                reason: EvictionReason::DistanceScore,
            }
        );
    }

    /// `:788-812`: the score is the sum of angles to every other keyframe, so
    /// the minimum is the least distinctive viewing direction. With three
    /// candidates sharing an azimuth and a fourth a radian away, the eviction
    /// has to come out of the cluster — which of the three it is depends on
    /// `acos`'s curvature and is not a property worth pinning.
    #[test]
    fn the_forward_vector_pass_takes_a_direction_from_the_cluster() {
        let vio: SqrtKeypointVio<f64> = a_window_of_keyframes(
            KeyframeMargCriteria::ForwardVector,
            [0.0, 0.02, 0.04, 1.0, 2.0, 3.0],
        );
        let evicted: KeyframeEviction = vio.evict_by_forward_vector().unwrap();
        assert_eq!(evicted.reason, EvictionReason::ForwardVector);
        assert!(
            [0, 1_000_000, 2_000_000].contains(&evicted.frame_id),
            "evicted {} instead of one of the three clustered azimuths",
            evicted.frame_id
        );
    }

    /// `frame_poses.at(ts)` (`:794`, `:849`) and `frame_states.at(last_kf)`
    /// (`:854`) both throw when the window disagrees with `kf_ids`; the port
    /// returns the typed error instead (D32).
    #[test]
    fn a_keyframe_missing_from_the_window_is_a_typed_error() {
        let mut vio: SqrtKeypointVio<f64> =
            a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        vio.ba.frame_states.clear();
        assert_eq!(
            vio.evict_by_default(&all_connected(&vio)),
            Err(EstimatorError::KeyframeNotInWindow {
                frame_id: 5_000_000,
                wanted: WindowRole::State,
            })
        );

        let mut hostless: SqrtKeypointVio<f64> =
            a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        hostless.num_points_kf.remove(&1_000_000);
        assert_eq!(
            hostless.evict_by_default(&all_connected(&hostless)),
            Err(EstimatorError::KeyframeNotInWindow {
                frame_id: 1_000_000,
                wanted: WindowRole::HostedLandmarkCount,
            })
        );

        // Missing from **both** maps: `:823` sees `count(*it) == 0` first, so
        // the keyframe is evicted and the hosted count is never read.
        let mut unseen: SqrtKeypointVio<f64> =
            a_window_of_keyframes(KeyframeMargCriteria::Default, FLAT);
        unseen.num_points_kf.remove(&1_000_000);
        let mut connected: BTreeMap<FrameId, usize> = all_connected(&unseen);
        connected.remove(&1_000_000);
        assert_eq!(
            unseen.evict_by_default(&connected),
            Ok(KeyframeEviction {
                frame_id: 1_000_000,
                reason: EvictionReason::FeatureRatio,
            })
        );

        let mut poseless: SqrtKeypointVio<f64> =
            a_window_of_keyframes(KeyframeMargCriteria::ForwardVector, FLAT);
        poseless.ba.frame_poses.remove(&2_000_000);
        assert_eq!(
            poseless.evict_by_forward_vector(),
            Err(EstimatorError::KeyframeNotInWindow {
                frame_id: 2_000_000,
                wanted: WindowRole::Pose,
            })
        );
    }
}
