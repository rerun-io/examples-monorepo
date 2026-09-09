//! Landmarks: the stereographic parameterisation, the landmark record, and the
//! database the linearizer walks.
//!
//! Ported from `thirdparty/basalt-headers/include/basalt/camera/stereographic_param.hpp`
//! and `include/basalt/vi_estimator/landmark_database.{h,cpp}`.
//!
//! A basalt landmark is **three** parameters — a 2-D stereographic `direction`
//! and a scalar `inv_dist` — anchored to a host frame *and camera*
//! (`landmark_database.h:54-92`). The 3-D point is
//! `unproject(direction)` with the fourth homogeneous slot overwritten by
//! `inv_dist` (`ba_utils.h:91-92`), so a landmark at infinity is `inv_dist = 0`
//! and needs no special case.
//!
//! Two deliberate departures from the C++ containers, both about determinism
//! and about the shape the GPU phase wants:
//!
//! * **Every map is a `BTreeMap`, never a `HashMap`** (decision D31). C++ uses
//!   `unordered_map` for `kpts` and for the outer `observations` level, and the
//!   error sum in [`crate::ba_base::BundleAdjustmentBase::compute_error`] walks
//!   host frames in that container's order. Floating-point addition is not
//!   associative, so the port cannot reproduce libstdc++'s bucket order and does
//!   not try: it sums in sorted [`TimeCamId`] order, which is reproducible on
//!   every machine and every run. The difference is at the level of the last
//!   bits of one sum.
//! * **The landmarks live in a `Vec`, with a `BTreeMap` from id to index**
//!   (the GPU seam rule: dense arrays plus an index map, never a per-element
//!   hash map). The `Vec` is kept sorted by [`LandmarkId`], so iterating it is
//!   the same order as iterating the index, and a later CubeCL kernel can take
//!   `landmarks()` as one contiguous buffer. Mutation keeps the two in step:
//!   [`LandmarkDatabase::add_landmark`] inserts in place and
//!   [`LandmarkDatabase::remove_landmark`] removes in place, both rebuilding the
//!   index entries the shift invalidated.

use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

use nalgebra::{Matrix2x4, Matrix4x2, Vector2, Vector4};

use crate::eigen::norm3;
use crate::lie::{LieScalar, c};
use crate::types::{FrameId, LandmarkId, TimeCamId};

/// Stereographic projection: the minimal 2-parameter chart on the unit sphere
/// basalt parameterises landmark directions with
/// (`stereographic_param.hpp:53-152`).
///
/// The chart is smooth and bijective everywhere except at the projection point,
/// which is why basalt can carry a direction as two unconstrained numbers and
/// increment them additively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StereographicParam<S: LieScalar> {
    _scalar: PhantomData<S>,
}

impl<S: LieScalar> StereographicParam<S> {
    /// `project` (`stereographic_param.hpp:79-106`): `[x, y] / (z + |p|)`.
    ///
    /// The fourth component of `p3d` is ignored, exactly as the C++ ignores it:
    /// only `head<3>()` reaches the norm.
    #[inline]
    pub fn project(p3d: &Vector4<S>) -> Vector2<S> {
        // `p3d.template head<3>().norm()` (`:80`) — the name `sqrt` is basalt's.
        let sqrt: S = norm3(p3d[0], p3d[1], p3d[2]);
        let norm: S = p3d[2] + sqrt;
        let norm_inv: S = S::one() / norm;
        Vector2::new(p3d[0] * norm_inv, p3d[1] * norm_inv)
    }

    /// [`Self::project`] with the 2x4 Jacobian `d_r_d_p`
    /// (`stereographic_param.hpp:86-103`).
    #[inline]
    pub fn project_with_jacobian(p3d: &Vector4<S>, d_r_d_p: &mut Matrix2x4<S>) -> Vector2<S> {
        let sqrt: S = norm3(p3d[0], p3d[1], p3d[2]);
        let norm: S = p3d[2] + sqrt;
        let norm_inv: S = S::one() / norm;
        let res: Vector2<S> = Vector2::new(p3d[0] * norm_inv, p3d[1] * norm_inv);

        let norm_inv2: S = norm_inv * norm_inv;
        let tmp: S = -norm_inv2 / sqrt;

        d_r_d_p.fill(S::zero());
        d_r_d_p[(0, 0)] = norm_inv + p3d[0] * p3d[0] * tmp;
        d_r_d_p[(1, 0)] = p3d[0] * p3d[1] * tmp;

        d_r_d_p[(1, 1)] = norm_inv + p3d[1] * p3d[1] * tmp;
        d_r_d_p[(0, 1)] = p3d[0] * p3d[1] * tmp;

        d_r_d_p[(0, 2)] = p3d[0] * norm * tmp;
        d_r_d_p[(1, 2)] = p3d[1] * norm * tmp;

        d_r_d_p[(0, 3)] = S::zero();
        d_r_d_p[(1, 3)] = S::zero();

        res
    }

    /// `unproject` (`stereographic_param.hpp:124-151`):
    /// `eta * [u, v, 1] - [0, 0, 1]` with `eta = 2 / (1 + u^2 + v^2)`.
    ///
    /// The fourth component comes back **zero**, not one: the caller overwrites
    /// it with the landmark's inverse distance (`ba_utils.h:92`), which is what
    /// makes the 4-vector a homogeneous point.
    #[inline]
    pub fn unproject(proj: &Vector2<S>) -> Vector4<S> {
        let x2: S = proj[0] * proj[0];
        let y2: S = proj[1] * proj[1];
        let r2: S = x2 + y2;

        let norm_inv: S = c::<S>(2.0) / (S::one() + r2);

        Vector4::new(
            proj[0] * norm_inv,
            proj[1] * norm_inv,
            norm_inv - S::one(),
            S::zero(),
        )
    }

    /// [`Self::unproject`] with the 4x2 Jacobian `d_r_d_p`
    /// (`stereographic_param.hpp:133-148`).
    ///
    /// Row 3 is zero: the homogeneous slot does not depend on the direction, so
    /// the inverse-distance column of the residual Jacobian is separate
    /// (`ba_utils.h:132`).
    #[inline]
    pub fn unproject_with_jacobian(proj: &Vector2<S>, d_r_d_p: &mut Matrix4x2<S>) -> Vector4<S> {
        let x2: S = proj[0] * proj[0];
        let y2: S = proj[1] * proj[1];
        let r2: S = x2 + y2;

        let norm_inv: S = c::<S>(2.0) / (S::one() + r2);
        let res: Vector4<S> = Vector4::new(
            proj[0] * norm_inv,
            proj[1] * norm_inv,
            norm_inv - S::one(),
            S::zero(),
        );

        let norm_inv2: S = norm_inv * norm_inv;
        let xy: S = proj[0] * proj[1];

        d_r_d_p[(0, 0)] = norm_inv - x2 * norm_inv2;
        d_r_d_p[(0, 1)] = -xy * norm_inv2;

        d_r_d_p[(1, 0)] = -xy * norm_inv2;
        d_r_d_p[(1, 1)] = norm_inv - y2 * norm_inv2;

        d_r_d_p[(2, 0)] = -proj[0] * norm_inv2;
        d_r_d_p[(2, 1)] = -proj[1] * norm_inv2;

        d_r_d_p[(3, 0)] = S::zero();
        d_r_d_p[(3, 1)] = S::zero();

        res
    }
}

/// One landmark: three optimised parameters, a host image, and its observations
/// (`landmark_database.h:54-92`).
#[derive(Debug, Clone, PartialEq)]
pub struct Landmark<S: LieScalar> {
    /// Stereographic direction in the host camera frame, `direction`.
    pub direction: Vector2<S>,
    /// Inverse distance along that direction, `inv_dist`. Never negative: the
    /// increment is projected at zero (`landmark_block_abs_dynamic.hpp:326`).
    pub inv_dist: S,
    /// The image the direction is expressed in, `host_kf_id`.
    pub host_kf_id: TimeCamId,
    /// Where the landmark was seen, keyed by image, `obs`.
    pub obs: BTreeMap<TimeCamId, Vector2<S>>,
    /// The database key, `id` (`landmark_database.h:70`).
    pub id: LandmarkId,
    backup_direction: Vector2<S>,
    backup_inv_dist: S,
}

impl<S: LieScalar> Landmark<S> {
    /// A landmark with no observations yet.
    ///
    /// The four fields here are exactly the four `addLandmark` copies
    /// (`landmark_database.cpp:41-47`).
    pub fn new(id: LandmarkId, host_kf_id: TimeCamId, direction: Vector2<S>, inv_dist: S) -> Self {
        Self {
            direction,
            inv_dist,
            host_kf_id,
            obs: BTreeMap::new(),
            id,
            backup_direction: direction,
            backup_inv_dist: inv_dist,
        }
    }

    /// Save the three optimised parameters, `backup` (`landmark_database.h:72-75`).
    ///
    /// Only the parameters: the observations and the host are not touched by a
    /// Levenberg-Marquardt step, so C++ does not save them either.
    pub fn backup(&mut self) {
        self.backup_direction = self.direction;
        self.backup_inv_dist = self.inv_dist;
    }

    /// Undo the last increment, `restore` (`landmark_database.h:77-80`).
    pub fn restore(&mut self) {
        self.direction = self.backup_direction;
        self.inv_dist = self.backup_inv_dist;
    }
}

/// What the database refuses to do.
///
/// basalt asserts in both cases (`landmark_database.cpp:123`, `:216`) and
/// `getLandmark` calls `std::unordered_map::at`, which throws. The port returns
/// a typed error instead: the estimator runs inside a released GIL where a
/// panic aborts the process (decision D32, trap 15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LandmarkError {
    /// An observation, a removal or a query named a landmark the database has
    /// never been given.
    #[error("landmark {0:?} is not in the database")]
    UnknownLandmark(LandmarkId),
}

/// basalt's `min_num_obs` (`landmark_database.h:159`): a landmark with fewer
/// observations than this carries no information and is deleted.
const MIN_NUM_OBS: usize = 2;

/// The landmark store and the host -> target -> landmark adjacency the
/// linearizer iterates (`landmark_database.h:94-160`).
///
/// The adjacency is not a cache: `observations` is what
/// [`crate::ba_base::BundleAdjustmentBase::compute_error`] and the linearizer
/// walk, and every mutation keeps it exactly consistent with the landmarks'
/// own `obs` maps — an empty target set drops the target, an empty target map
/// drops the host (`landmark_database.cpp:178-205`).
#[derive(Debug, Clone)]
pub struct LandmarkDatabase<S: LieScalar> {
    /// Sorted by `LandmarkId`; `index` maps an id to a position here.
    kpts: Vec<Landmark<S>>,
    index: BTreeMap<LandmarkId, usize>,
    observations: BTreeMap<TimeCamId, BTreeMap<TimeCamId, BTreeSet<LandmarkId>>>,
}

impl<S: LieScalar> Default for LandmarkDatabase<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: LieScalar> LandmarkDatabase<S> {
    /// An empty database.
    pub fn new() -> Self {
        Self {
            kpts: Vec::new(),
            index: BTreeMap::new(),
            observations: BTreeMap::new(),
        }
    }

    /// Forget everything, `clear` (`landmark_database.h:106-109`).
    pub fn clear(&mut self) {
        self.kpts.clear();
        self.index.clear();
        self.observations.clear();
    }

    /// Add or overwrite a landmark's parameters, `addLandmark`
    /// (`landmark_database.cpp:41-47`).
    ///
    /// C++ writes through `kpts[lm_id]`, which default-constructs a landmark
    /// when the id is new and then copies **four** fields: `direction`,
    /// `inv_dist`, `host_kf_id` and `id`. It does *not* copy `obs`, so
    /// re-adding an existing id keeps that landmark's observations and its
    /// adjacency entries. The port reproduces that.
    ///
    /// **Deviation.** Re-adding an existing id with a *different* host leaves
    /// C++'s adjacency filed under the old host while the landmark claims the
    /// new one, and `removeLandmarkHelper` then looks the landmark up under a
    /// host that does not list it (`landmark_database.cpp:180-187`). The
    /// estimator never does this — a landmark id comes from a keypoint and is
    /// added once (`sqrt_keypoint_vio.cpp:535-539`) — so rather than reproduce a
    /// latent inconsistency the port moves the existing observations to the new
    /// host and keeps the invariant.
    pub fn add_landmark(&mut self, lm_id: LandmarkId, pos: &Landmark<S>) {
        match self.index.get(&lm_id) {
            Some(&at) => {
                if let Some(kpt) = self.kpts.get_mut(at) {
                    // The adjacency is keyed by the host, so a changed host has
                    // to carry the existing observations with it.
                    let host_changed: bool = kpt.host_kf_id != pos.host_kf_id;
                    let old_host: TimeCamId = kpt.host_kf_id;
                    kpt.direction = pos.direction;
                    kpt.inv_dist = pos.inv_dist;
                    kpt.host_kf_id = pos.host_kf_id;
                    kpt.id = lm_id;
                    if host_changed {
                        let targets: Vec<TimeCamId> = kpt.obs.keys().copied().collect();
                        let new_host: TimeCamId = pos.host_kf_id;
                        for target in targets {
                            self.detach_observation(old_host, target, lm_id);
                            self.observations
                                .entry(new_host)
                                .or_default()
                                .entry(target)
                                .or_default()
                                .insert(lm_id);
                        }
                    }
                }
            }
            None => {
                let at: usize = self.kpts.partition_point(|lm| lm.id < lm_id);
                self.kpts.insert(
                    at,
                    Landmark::new(lm_id, pos.host_kf_id, pos.direction, pos.inv_dist),
                );
                self.reindex_from(at);
            }
        }
    }

    /// Record that `lm_id` was seen at `pos` in `tcid_target`, `addObservation`
    /// (`landmark_database.cpp:121-128`).
    ///
    /// C++ asserts the landmark exists; the port returns
    /// [`LandmarkError::UnknownLandmark`].
    pub fn add_observation(
        &mut self,
        tcid_target: TimeCamId,
        lm_id: LandmarkId,
        pos: Vector2<S>,
    ) -> Result<(), LandmarkError> {
        let at: usize = *self
            .index
            .get(&lm_id)
            .ok_or(LandmarkError::UnknownLandmark(lm_id))?;
        let kpt: &mut Landmark<S> = self
            .kpts
            .get_mut(at)
            .ok_or(LandmarkError::UnknownLandmark(lm_id))?;
        kpt.obs.insert(tcid_target, pos);
        let host: TimeCamId = kpt.host_kf_id;
        self.observations
            .entry(host)
            .or_default()
            .entry(tcid_target)
            .or_default()
            .insert(lm_id);
        Ok(())
    }

    /// A landmark by id, `getLandmark` (`landmark_database.cpp:131-138`).
    ///
    /// C++ uses `at`, which throws; the port returns `None`.
    pub fn get_landmark(&self, lm_id: LandmarkId) -> Option<&Landmark<S>> {
        self.index.get(&lm_id).and_then(|&at| self.kpts.get(at))
    }

    /// A landmark by id, mutable (`landmark_database.cpp:131-133`).
    ///
    /// Only the three optimised parameters may be written through this handle;
    /// changing `obs` or `host_kf_id` here would desynchronise the adjacency,
    /// which is why the mutating API above exists.
    pub fn get_landmark_mut(&mut self, lm_id: LandmarkId) -> Option<&mut Landmark<S>> {
        match self.index.get(&lm_id) {
            Some(&at) => self.kpts.get_mut(at),
            None => None,
        }
    }

    /// Whether an id is in the database, `landmarkExists`
    /// (`landmark_database.cpp:151-154`).
    pub fn landmark_exists(&self, lm_id: LandmarkId) -> bool {
        self.index.contains_key(&lm_id)
    }

    /// Every landmark, in id order — one dense slice, for the GPU seam.
    ///
    /// `getLandmarks` (`landmark_database.cpp:146-149`) hands back the whole
    /// `unordered_map`; this is the same content in a reproducible order.
    pub fn landmarks(&self) -> &[Landmark<S>] {
        &self.kpts
    }

    /// The host -> target -> landmark-set adjacency, `getObservations`
    /// (`landmark_database.cpp:140-144`).
    pub fn observations(&self) -> &BTreeMap<TimeCamId, BTreeMap<TimeCamId, BTreeSet<LandmarkId>>> {
        &self.observations
    }

    /// The images that host at least one landmark, `getHostKfs`
    /// (`landmark_database.cpp:89-97`), in sorted order.
    pub fn host_kfs(&self) -> Vec<TimeCamId> {
        self.observations.keys().copied().collect()
    }

    /// The targets one host is observed in, and the landmarks in each.
    ///
    /// The per-(host, target) iteration the linearizer needs: one relative pose
    /// `T_t_h` is hoisted per pair (`ba_base.cpp:145-160`), and every landmark
    /// in the set reuses it.
    pub fn targets_for_host(
        &self,
        tcid: TimeCamId,
    ) -> Option<&BTreeMap<TimeCamId, BTreeSet<LandmarkId>>> {
        self.observations.get(&tcid)
    }

    /// Number of landmarks, `numLandmarks` (`landmark_database.cpp:156-159`).
    pub fn num_landmarks(&self) -> usize {
        self.kpts.len()
    }

    /// Number of (landmark, target) pairs in the adjacency, `numObservations`
    /// (`landmark_database.cpp:161-170`).
    pub fn num_observations(&self) -> usize {
        self.observations
            .values()
            .flat_map(|targets| targets.values())
            .map(BTreeSet::len)
            .sum()
    }

    /// Delete one landmark and its adjacency, `removeLandmark`
    /// (`landmark_database.cpp:207-211`). A missing id is a no-op, as in C++.
    pub fn remove_landmark(&mut self, lm_id: LandmarkId) {
        if let Some(&at) = self.index.get(&lm_id) {
            self.remove_landmark_at(at);
        }
    }

    /// The marginalization sweep, `removeKeyframes`
    /// (`landmark_database.cpp:65-87`).
    ///
    /// Three rules, in this order and no other:
    ///
    /// 1. a landmark **hosted** by a frame in `kfs_to_marg` is deleted outright
    ///    (`:70-71`) — its direction is expressed in a frame that is about to
    ///    stop existing, so nothing can be salvaged;
    /// 2. otherwise, every observation made in a frame in any of the three sets
    ///    is dropped (`:73-78`);
    /// 3. then the `min_num_obs = 2` sweep deletes what is left with fewer than
    ///    two observations (`:80-84`).
    ///
    /// Note rule 1 short-circuits rule 2: a landmark hosted by a marginalized
    /// keyframe never reaches the observation loop.
    pub fn remove_keyframes(
        &mut self,
        kfs_to_marg: &BTreeSet<FrameId>,
        poses_to_marg: &BTreeSet<FrameId>,
        states_to_marg_all: &BTreeSet<FrameId>,
    ) {
        self.retain_observations(
            |target| {
                let fid: FrameId = target.frame_id;
                !(poses_to_marg.contains(&fid)
                    || states_to_marg_all.contains(&fid)
                    || kfs_to_marg.contains(&fid))
            },
            |host| kfs_to_marg.contains(&host.frame_id),
        );
    }

    /// Save every landmark's parameters, `backup` (`landmark_database.h:141-143`).
    pub fn backup(&mut self) {
        for kpt in &mut self.kpts {
            kpt.backup();
        }
    }

    /// Undo every landmark's last increment, `restore`
    /// (`landmark_database.h:145-147`).
    pub fn restore(&mut self) {
        for kpt in &mut self.kpts {
            kpt.restore();
        }
    }

    // ─── internals ────────────────────────────────────────────────────────

    /// The shared body of `removeFrame` and `removeKeyframes`: delete whole
    /// landmarks whose host `drop_host` selects, drop the observations
    /// `keep_obs` rejects, then run the `min_num_obs` sweep
    /// (`landmark_database.cpp:49-87`).
    ///
    /// One pass over the dense `Vec` with `retain`, and one index rebuild, in
    /// place of C++'s iterator-erasing loop; both leave the same database.
    fn retain_observations(
        &mut self,
        keep_obs: impl Fn(TimeCamId) -> bool,
        drop_host: impl Fn(TimeCamId) -> bool,
    ) {
        let observations: &mut BTreeMap<TimeCamId, BTreeMap<TimeCamId, BTreeSet<LandmarkId>>> =
            &mut self.observations;
        self.kpts.retain_mut(|kpt| {
            let host: TimeCamId = kpt.host_kf_id;
            if drop_host(host) {
                detach_landmark(observations, host, &kpt.obs, kpt.id);
                return false;
            }
            let dropped: Vec<TimeCamId> = kpt
                .obs
                .keys()
                .copied()
                .filter(|target| !keep_obs(*target))
                .collect();
            for target in dropped {
                kpt.obs.remove(&target);
                detach_one(observations, host, target, kpt.id);
            }
            if kpt.obs.len() < MIN_NUM_OBS {
                detach_landmark(observations, host, &kpt.obs, kpt.id);
                return false;
            }
            true
        });
        self.reindex_from(0);
    }

    /// Erase the landmark at a known position and every adjacency entry it owns,
    /// `removeLandmarkHelper` (`landmark_database.cpp:177-192`).
    fn remove_landmark_at(&mut self, at: usize) {
        let Some(kpt) = self.kpts.get(at) else {
            return;
        };
        let host: TimeCamId = kpt.host_kf_id;
        let lm_id: LandmarkId = kpt.id;
        let targets: Vec<TimeCamId> = kpt.obs.keys().copied().collect();
        for target in targets {
            self.detach_observation(host, target, lm_id);
        }
        self.kpts.remove(at);
        self.reindex_from(at);
    }

    /// One adjacency entry, `removeLandmarkObservationHelper`
    /// (`landmark_database.cpp:194-205`) minus the landmark-side erase.
    ///
    /// C++ dereferences the result of `observations.find` unchecked; a landmark
    /// added but never observed makes that a null dereference. The port skips
    /// the missing entry.
    fn detach_observation(&mut self, host: TimeCamId, target: TimeCamId, lm_id: LandmarkId) {
        detach_one(&mut self.observations, host, target, lm_id);
    }

    /// Rebuild `index` for every position from `at` on, which is exactly the
    /// range an insert or a remove at `at` shifted.
    ///
    /// Entries at or past `at` are dropped first: a removal shrinks the `Vec`,
    /// and re-adding alone would leave the erased id pointing past the end.
    fn reindex_from(&mut self, at: usize) {
        self.index.retain(|_, &mut pos| pos < at);
        for (pos, lm) in self.kpts.iter().enumerate().skip(at) {
            self.index.insert(lm.id, pos);
        }
    }
}

/// Drop `lm_id` from `observations[host][target]`, then the empty containers
/// above it (`landmark_database.cpp:198-202`).
fn detach_one(
    observations: &mut BTreeMap<TimeCamId, BTreeMap<TimeCamId, BTreeSet<LandmarkId>>>,
    host: TimeCamId,
    target: TimeCamId,
    lm_id: LandmarkId,
) {
    let Some(targets) = observations.get_mut(&host) else {
        return;
    };
    if let Some(ids) = targets.get_mut(&target) {
        ids.remove(&lm_id);
        if ids.is_empty() {
            targets.remove(&target);
        }
    }
    if targets.is_empty() {
        observations.remove(&host);
    }
}

/// The same for every target of one landmark, `removeLandmarkHelper`
/// (`landmark_database.cpp:180-189`).
fn detach_landmark<S: LieScalar>(
    observations: &mut BTreeMap<TimeCamId, BTreeMap<TimeCamId, BTreeSet<LandmarkId>>>,
    host: TimeCamId,
    obs: &BTreeMap<TimeCamId, Vector2<S>>,
    lm_id: LandmarkId,
) {
    for &target in obs.keys() {
        detach_one(observations, host, target, lm_id);
    }
    if let Some(targets) = observations.get(&host)
        && targets.is_empty()
    {
        observations.remove(&host);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    const CASES: u32 = 128;

    fn config() -> ProptestConfig {
        ProptestConfig::with_cases(CASES)
    }

    fn tcid(frame_id: i64, cam_id: usize) -> TimeCamId {
        TimeCamId::new(frame_id, cam_id)
    }

    fn landmark(id: u64, host: TimeCamId) -> Landmark<f64> {
        Landmark::new(LandmarkId(id), host, Vector2::new(0.1, -0.2), 0.5)
    }

    /// Every invariant `removeKeyframes` and the sweep are supposed to leave.
    fn check_invariants(db: &LandmarkDatabase<f64>) {
        for lm in db.landmarks() {
            assert!(
                lm.obs.len() >= MIN_NUM_OBS,
                "{:?} survived the sweep",
                lm.id
            );
            for target in lm.obs.keys() {
                assert!(
                    db.observations()
                        .get(&lm.host_kf_id)
                        .and_then(|t| t.get(target))
                        .is_some_and(|ids| ids.contains(&lm.id)),
                    "adjacency missing {:?} at {target}",
                    lm.id
                );
            }
        }
        for (host, targets) in db.observations() {
            assert!(!targets.is_empty(), "empty host {host}");
            for (target, ids) in targets {
                assert!(!ids.is_empty(), "empty target {target}");
                for id in ids {
                    let lm: &Landmark<f64> = db.get_landmark(*id).unwrap();
                    assert_eq!(lm.host_kf_id, *host);
                    assert!(lm.obs.contains_key(target));
                }
            }
        }
        // The dense array and the index agree, and the array is sorted.
        for (pos, lm) in db.landmarks().iter().enumerate() {
            assert_eq!(db.index.get(&lm.id), Some(&pos));
        }
        assert_eq!(db.index.len(), db.landmarks().len());
        assert!(db.landmarks().windows(2).all(|w| w[0].id < w[1].id));
    }

    /// A small database: three landmarks hosted in frame 10 cam 0, each seen in
    /// frames 10, 20 and 30.
    fn a_database() -> LandmarkDatabase<f64> {
        let mut db: LandmarkDatabase<f64> = LandmarkDatabase::new();
        let host: TimeCamId = tcid(10, 0);
        for id in 0..3u64 {
            db.add_landmark(LandmarkId(id), &landmark(id, host));
            for frame in [10i64, 20, 30] {
                db.add_observation(
                    tcid(frame, 0),
                    LandmarkId(id),
                    Vector2::new(f64::from(frame as i32), f64::from(id as i32)),
                )
                .unwrap();
            }
        }
        db
    }

    #[test]
    fn unproject_inverts_project() {
        for &(u, v) in &[(0.0, 0.0), (0.3, -0.7), (-1.2, 0.4), (2.0, 2.0)] {
            let proj: Vector2<f64> = Vector2::new(u, v);
            let bearing: Vector4<f64> = StereographicParam::unproject(&proj);
            assert_abs_diff_eq!(bearing.fixed_rows::<3>(0).norm(), 1.0, epsilon = 1e-15);
            assert_eq!(bearing[3], 0.0);
            let back: Vector2<f64> = StereographicParam::project(&bearing);
            assert_abs_diff_eq!(back, proj, epsilon = 1e-14);
        }
    }

    #[test]
    fn adding_a_landmark_twice_keeps_its_observations() {
        // `addLandmark` copies four fields and leaves `obs` alone
        // (`landmark_database.cpp:41-47`).
        let mut db: LandmarkDatabase<f64> = a_database();
        let host: TimeCamId = tcid(10, 0);
        let mut again: Landmark<f64> = landmark(0, host);
        again.direction = Vector2::new(9.0, 9.0);
        again.inv_dist = 1.5;
        db.add_landmark(LandmarkId(0), &again);
        let lm: &Landmark<f64> = db.get_landmark(LandmarkId(0)).unwrap();
        assert_eq!(lm.obs.len(), 3);
        assert_eq!(lm.direction, Vector2::new(9.0, 9.0));
        assert_eq!(lm.inv_dist, 1.5);
        check_invariants(&db);
    }

    #[test]
    fn an_observation_on_an_unknown_landmark_is_an_error() {
        let mut db: LandmarkDatabase<f64> = LandmarkDatabase::new();
        assert_eq!(
            db.add_observation(tcid(1, 0), LandmarkId(7), Vector2::zeros()),
            Err(LandmarkError::UnknownLandmark(LandmarkId(7)))
        );
    }

    #[test]
    fn removing_a_landmark_empties_its_adjacency() {
        let mut db: LandmarkDatabase<f64> = a_database();
        assert_eq!(db.num_landmarks(), 3);
        assert_eq!(db.num_observations(), 9);
        db.remove_landmark(LandmarkId(1));
        assert_eq!(db.num_landmarks(), 2);
        assert_eq!(db.num_observations(), 6);
        assert!(!db.landmark_exists(LandmarkId(1)));
        check_invariants(&db);
        // Removing all three drops the host entry entirely.
        db.remove_landmark(LandmarkId(0));
        db.remove_landmark(LandmarkId(2));
        assert!(db.observations().is_empty());
        assert!(db.host_kfs().is_empty());
    }

    #[test]
    fn remove_keyframes_deletes_hosted_landmarks_first() {
        let mut db: LandmarkDatabase<f64> = a_database();
        // A second host that survives.
        let other: TimeCamId = tcid(20, 1);
        db.add_landmark(LandmarkId(9), &landmark(9, other));
        for frame in [20i64, 30] {
            db.add_observation(tcid(frame, 1), LandmarkId(9), Vector2::zeros())
                .unwrap();
        }
        let kfs: BTreeSet<i64> = [10i64].into_iter().collect();
        db.remove_keyframes(&kfs, &BTreeSet::new(), &BTreeSet::new());
        // Everything hosted by frame 10 is gone; the frame-20 host survives with
        // both of its observations, neither of which is in frame 10.
        assert_eq!(db.num_landmarks(), 1);
        assert!(db.landmark_exists(LandmarkId(9)));
        check_invariants(&db);
    }

    #[test]
    fn remove_keyframes_drops_observations_then_sweeps() {
        let mut db: LandmarkDatabase<f64> = a_database();
        let poses: BTreeSet<i64> = [20i64].into_iter().collect();
        db.remove_keyframes(&BTreeSet::new(), &poses, &BTreeSet::new());
        // Each landmark loses its frame-20 observation and keeps two.
        assert_eq!(db.num_landmarks(), 3);
        assert_eq!(db.num_observations(), 6);
        check_invariants(&db);
        // One more marginalized frame takes every landmark below two.
        let states: BTreeSet<i64> = [30i64].into_iter().collect();
        db.remove_keyframes(&BTreeSet::new(), &BTreeSet::new(), &states);
        assert_eq!(db.num_landmarks(), 0);
        assert!(db.observations().is_empty());
    }

    #[test]
    fn backup_and_restore_move_only_the_three_parameters() {
        let mut db: LandmarkDatabase<f64> = a_database();
        db.backup();
        let lm: &mut Landmark<f64> = db.get_landmark_mut(LandmarkId(0)).unwrap();
        lm.direction += Vector2::new(0.5, 0.5);
        lm.inv_dist += 0.25;
        assert_eq!(db.get_landmark(LandmarkId(0)).unwrap().inv_dist, 0.75);
        db.restore();
        let lm: &Landmark<f64> = db.get_landmark(LandmarkId(0)).unwrap();
        assert_eq!(lm.direction, Vector2::new(0.1, -0.2));
        assert_eq!(lm.inv_dist, 0.5);
        assert_eq!(lm.obs.len(), 3);
    }

    proptest! {
        #![proptest_config(config())]

        /// `unproject` is the inverse of `project` on the unit sphere, in both
        /// precisions, and the bearing always has unit norm.
        #[test]
        fn stereographic_round_trips(u in -8.0f64..8.0, v in -8.0f64..8.0) {
            let proj: Vector2<f64> = Vector2::new(u, v);
            let bearing: Vector4<f64> = StereographicParam::unproject(&proj);
            prop_assert!((bearing.fixed_rows::<3>(0).norm() - 1.0).abs() < 1e-14);
            let back: Vector2<f64> = StereographicParam::project(&bearing);
            prop_assert!((back - proj).norm() < 1e-12);

            let proj32: Vector2<f32> = Vector2::new(u as f32, v as f32);
            let bearing32: Vector4<f32> = StereographicParam::unproject(&proj32);
            let back32: Vector2<f32> = StereographicParam::project(&bearing32);
            prop_assert!((back32 - proj32).norm() < 1e-4);
        }

        /// `project` also inverts `unproject` when it starts from a 3-D point
        /// that is not unit norm: the chart only sees the direction.
        #[test]
        fn project_ignores_the_length(
            x in -3.0f64..3.0,
            y in -3.0f64..3.0,
            z in 0.05f64..4.0,
            scale in 0.1f64..10.0,
        ) {
            let p: Vector4<f64> = Vector4::new(x, y, z, 1.0);
            let scaled: Vector4<f64> = Vector4::new(x * scale, y * scale, z * scale, 1.0);
            let a: Vector2<f64> = StereographicParam::project(&p);
            let b: Vector2<f64> = StereographicParam::project(&scaled);
            prop_assert!((a - b).norm() < 1e-12);
        }

        /// A landmark hosted by a removed keyframe never survives, no
        /// observation from a removed frame survives, and everything left has at
        /// least two observations and a consistent adjacency.
        #[test]
        fn remove_keyframes_leaves_a_consistent_database(
            hosts in prop::collection::vec(0i64..4, 1..6),
            frames in prop::collection::vec(prop::collection::vec(0i64..6, 0..5), 1..6),
            kfs in prop::collection::vec(0i64..4, 0..3),
            poses in prop::collection::vec(0i64..6, 0..3),
            states in prop::collection::vec(0i64..6, 0..3),
        ) {
            let mut db: LandmarkDatabase<f64> = LandmarkDatabase::new();
            for (i, host_frame) in hosts.iter().enumerate() {
                let id: LandmarkId = LandmarkId(i as u64);
                let host: TimeCamId = tcid(*host_frame, i % 2);
                db.add_landmark(id, &landmark(i as u64, host));
                let targets: &Vec<i64> = &frames[i % frames.len()];
                for (j, frame) in targets.iter().enumerate() {
                    db.add_observation(tcid(*frame, j % 2), id, Vector2::zeros()).unwrap();
                }
            }
            let kfs: BTreeSet<i64> = kfs.into_iter().collect();
            let poses: BTreeSet<i64> = poses.into_iter().collect();
            let states: BTreeSet<i64> = states.into_iter().collect();
            db.remove_keyframes(&kfs, &poses, &states);

            for lm in db.landmarks() {
                prop_assert!(!kfs.contains(&lm.host_kf_id.frame_id));
                prop_assert!(lm.obs.len() >= MIN_NUM_OBS);
                for target in lm.obs.keys() {
                    prop_assert!(!kfs.contains(&target.frame_id));
                    prop_assert!(!poses.contains(&target.frame_id));
                    prop_assert!(!states.contains(&target.frame_id));
                }
            }
            check_invariants(&db);
            prop_assert_eq!(
                db.num_observations(),
                db.landmarks().iter().map(|lm| lm.obs.len()).sum::<usize>()
            );
        }
    }
}
