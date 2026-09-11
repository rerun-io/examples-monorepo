//! Marginalization scheduling: select full states, velocity/bias blocks, poses
//! and keyframes to remove. [`crate::marg::marginalize`] owns the algebra.
//!
//! `states_to_remove = frame_states.size() - max_states + 1`: at a full window,
//! the oldest state leaves and the second oldest is kept and frozen.
//! The keyframe budget is lazy: eviction requires both an exceeded keyframe
//! budget and a keyframe leaving the state window. A newly selected keyframe
//! can therefore keep the count one above the configured budget until demotion.
//! The shipped frame spacing and state-window sizes bound this overshoot.
//!
//! The default rule first checks tracked-feature ratios, then minimizes a
//! DSO-derived distance score. Its self-distance term makes proximity to the
//! newest keyframe dominate the second pass (D22).

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
    /// frame does not observe at all.
    FeatureRatio,
    /// `KF_MARG_DEFAULT` second pass: the minimum of the DSO distance score
    DistanceScore,
    /// `KF_MARG_FORWARD_VECTOR`: the least distinctive viewing direction
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
    /// `states_to_remove`.
    pub states_to_remove: usize,
    /// The state that is kept whole and frozen.
    pub last_state_to_marg: FrameId,
    /// Pose blocks that leave: the non-keyframes, plus the evicted keyframes
    pub poses_to_marg: Vec<FrameId>,
    /// States that leave entirely.
    pub states_to_marg_all: Vec<FrameId>,
    /// Keyframe states demoted to pose blocks.
    pub states_to_marg_vel_bias: Vec<FrameId>,
    /// Keyframes the budget evicted, oldest first.
    pub kfs_to_marg: Vec<FrameId>,
    /// Which criterion pass chose each eviction, in the order they happened.
    pub evictions: Vec<KeyframeEviction>,
    /// `idx_to_keep.size()`.
    pub kept_indices: usize,
    /// `idx_to_marg.size()`.
    pub marg_indices: usize,
    /// Whether the marginalization linearization was numerically valid.
    pub numerically_valid: bool,
    /// Ordering width; kept and marginalized index counts must sum to this value.
    pub ordering_size: usize,
    /// `marg_order_new`, as `(frame, index, size)`.
    pub prior_order: Vec<(FrameId, usize, usize)>,
}

/// What one call to [`SqrtKeypointVio::marginalize`] did, on its way into
/// [`FrameStats`].
///
/// Everything is empty when the trigger of did not fire, which is the
/// common case: the window marginalizes on roughly one frameset in two.
#[derive(Debug, Clone, Default)]
pub(super) struct MarginalizationOutcome {
    /// The marginalization itself.
    pub marginalization: Option<MarginalizationStats>,
    /// `StageTimings::marginalize_ns`.
    pub elapsed_ns: u64,
}

impl<S: LieScalar> SqrtKeypointVio<S> {
    /// Move the newest keyframe outside the ordinary budget when requested.
    /// Long-term keyframes are never eviction candidates. Consume the request even
    /// when no keyframe can be moved.
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

    /// Run marginalization and return its decisions and duration.
    /// Components stay empty or zero if the trigger does not fire.
    ///
    /// # Errors
    /// Returns [`EstimatorError`] for missing candidate frames, prior-order mismatch
    /// or absence of a selectable keyframe.
    pub(super) fn marginalize(
        &mut self,
        num_points_connected: &BTreeMap<FrameId, usize>,
        lost_landmarks: &BTreeSet<LandmarkId>,
    ) -> Result<MarginalizationOutcome, EstimatorError> {
        if !self.opt_started {
            return Ok(MarginalizationOutcome::default());
        }
        if !(self.ba.frame_poses.len() > self.ltkfs.len() + self.max_kfs
            || self.ba.frame_states.len() >= self.max_states)
        {
            return Ok(MarginalizationOutcome::default());
        }
        let mark: std::time::Instant = std::time::Instant::now();

        // Saturating arithmetic leaves a short window with no states to advance past.
        let states_to_remove: usize =
            (self.ba.frame_states.len() + 1).saturating_sub(self.max_states);
        let Some(last_state_to_marg) = self.ba.frame_states.keys().nth(states_to_remove).copied()
        else {
            // Refuse an advance beyond the state window.
            return Err(EstimatorError::StateWindowTooShort {
                states: self.ba.frame_states.len(),
                states_to_remove,
            });
        };

        // every pose block that is neither a keyframe nor a
        // long-term keyframe. The ordering assertion of happens inside
        // `marg::marginalize`, which rebuilds the same `AbsOrderMap`.
        let mut poses_to_marg: BTreeSet<FrameId> = BTreeSet::new();
        for frame_id in self.ba.frame_poses.keys().copied() {
            if !self.kf_ids.contains(&frame_id) && !self.ltkfs.contains(&frame_id) {
                poses_to_marg.insert(frame_id);
            }
        }

        // the states older than `last_state_to_marg` split by
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
            // Fix the same long-term keyframes as optimization.
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

        //  and : the two estimator-side maps
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

    /// Default keyframe eviction.
    ///
    /// First choose the oldest eligible keyframe whose tracked-feature ratio is
    /// below the threshold, or which the current frame does not observe. Skip the
    /// newest two. The quotient is computed in f32 and widened for comparison.
    ///
    /// Otherwise minimize `sqrt(‖p_i − p_last‖) · Σ_j 1/(‖p_i − p_j‖ + 1e-5)`.
    /// The sum includes the candidate itself, contributing `1e5`. This dominates
    /// metre-scale distances, so proximity to the newest keyframe drives eviction.
    fn evict_by_default(
        &self,
        num_points_connected: &BTreeMap<FrameId, usize>,
    ) -> Result<KeyframeEviction, EstimatorError> {
        let candidates: Vec<FrameId> = self.eviction_candidates();

        // The `||` at short-circuits on the missing
        // connection *before* it reads `num_points_kf.at(*it)`, so a keyframe
        // absent from both maps is evicted rather than refused.
        for frame_id in &candidates {
            let Some(connected) = num_points_connected.get(frame_id).copied() else {
                return Ok(KeyframeEviction {
                    frame_id: *frame_id,
                    reason: EvictionReason::FeatureRatio,
                });
            };
            // Hosted counts are retained, so an observed keyframe must have an entry.
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

        // : `std::numeric_limits<Scalar>::max()`.
        let mut min_score: S = S::largest();
        let mut min_score_id: Option<FrameId> = None;
        // `*kf_ids.crbegin()`. Every candidate comes out of `kf_ids`,
        // so an empty keyframe set has no candidate either and the one
        // `NoKeyframeToMarginalize` below reports it.
        if let Some(last_kf) = self.kf_ids.iter().next_back().copied() {
            // `frame_states.at(last_kf)`. The newest keyframe is the
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
                // the sum runs over the same candidate set, so a
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
        // "if no frame was selected, the logic above is faulty".
        min_score_id
            .map(|frame_id| KeyframeEviction {
                frame_id,
                reason: EvictionReason::DistanceScore,
            })
            .ok_or(EstimatorError::NoKeyframeToMarginalize {
                candidates: candidates.len(),
            })
    }

    /// Forward-vector eviction: sum camera-0 viewing-direction angles against other
    /// keyframes, including long-term keyframes, and evict the minimum.
    /// The least distinctive direction leaves. Angles use the estimator scalar.
    fn evict_by_forward_vector(&self) -> Result<KeyframeEviction, EstimatorError> {
        let candidates: Vec<FrameId> = self.eviction_candidates();
        // the scored-against set is `ltkfs ∪ kf_ids`, again without
        // its own newest two.
        let mut all_kfs: BTreeSet<FrameId> = self.ltkfs.clone();
        all_kfs.extend(self.kf_ids.iter().copied());
        let against: Vec<FrameId> = all_kfs
            .iter()
            .copied()
            .take(all_kfs.len().saturating_sub(2))
            .collect();

        // `std::numeric_limits<Scalar>::max()`.
        let mut min_score: S = S::largest();
        let mut min_score_id: Option<FrameId> = None;
        for frame_id in &candidates {
            let fwd1: (S, S) = self.forward_vector_2d(*frame_id)?;
            let mut score: S = S::zero();
            for other in &against {
                let fwd2: (S, S) = self.forward_vector_2d(*other)?;
                // Two-vector dot product: `a0 * b0 + a1 * b1`.
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

    /// All keyframes except the newest two; empty when there are at most two.
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

    /// Look up a keyframe pose block, returning an error if absent.
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

    /// `get_forward_vector2d` : the camera-0 optical axis in the
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
    /// `frame_states.at(last_kf)`, each hosting ten landmarks.
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

    /// Long-term keyframes are fixed in both the optimization and marginalization
    /// linearizations, so their pose Jacobians are zero in both.
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

    /// `takeLongTermKeyframe()` and the demotion it asks for
    /// the newest keyframe leaves `kf_ids` for `ltkfs`, which
    /// moves it to the other side of 's pose budget and out of the
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

    /// `std::prev(kf_ids.end(), 2)` and the `kf_ids.size() > 2`
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

    ///  first pass: the oldest keyframe below
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

    /// `num_points_connected.count(*it) == 0` : a keyframe the current
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

    /// The distance score includes a `1e5` self term. On equally spaced keyframes,
    /// this makes the candidate nearest the newest keyframe leave (D22).
    /// Removing the self term would change the selection.
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

    /// the score is the sum of angles to every other keyframe, so
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

    /// `frame_poses.at(ts)` and `frame_states.at(last_kf)`
    ///  both throw when the window disagrees with `kf_ids`; the port
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

        // Missing from **both** maps: sees `count(*it) == 0` first, so
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
