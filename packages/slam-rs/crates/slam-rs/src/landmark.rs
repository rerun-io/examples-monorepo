//! Landmark parameterization, records and database.
//! A landmark has a two-dimensional stereographic direction and inverse distance,
//! anchored to a host frame and camera. The homogeneous fourth component is
//! inverse distance; zero represents infinity.
//!
//! Maps use sorted keys for deterministic accumulation (D31). Landmarks occupy
//! an id-sorted vector with an id-to-index map, supporting contiguous access.
//! Insertion and removal rebuild index entries affected by the shift.

use kornia_staging_algebra::Scalar;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

use nalgebra::{Matrix2x4, Matrix4x2, Vector2, Vector4};

use crate::lie::{c};
use crate::types::{FrameId, LandmarkId, TimeCamId};

/// Two-parameter stereographic chart on the unit sphere.
/// It is smooth and bijective except at the projection point, permitting additive
/// direction increments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StereographicParam<S: Scalar> {
    _scalar: PhantomData<S>,
}

impl<S: Scalar> StereographicParam<S> {
    /// Project through staging; invalid directions retain the estimator's NaN sentinel.
    pub fn project(p3d: &Vector4<S>) -> Vector2<S> {
        Vector2::from(
            kornia_staging_3d::pose::stereographic_project((*p3d).into())
                .unwrap_or([c::<S>(f64::NAN); 2]),
        )
    }
    /// Project with a 2x4 Jacobian; invalid directions produce NaN sentinels.
    pub fn project_with_jacobian(p3d: &Vector4<S>, jacobian: &mut Matrix2x4<S>) -> Vector2<S> {
        let (value, j) =
            kornia_staging_3d::pose::stereographic_project_with_jacobian((*p3d).into())
                .unwrap_or(([c::<S>(f64::NAN); 2], [[c::<S>(f64::NAN); 2]; 4]));
        jacobian.data.0 = j;
        Vector2::from(value)
    }
    /// Unit spatial bearing with a zero inverse-distance component.
    pub fn unproject(point: &Vector2<S>) -> Vector4<S> {
        Vector4::from(kornia_staging_3d::pose::stereographic_unproject(
            (*point).into(),
        ))
    }
    /// Unit bearing and its 4x2 Jacobian; the fourth row is zero.
    pub fn unproject_with_jacobian(point: &Vector2<S>, jacobian: &mut Matrix4x2<S>) -> Vector4<S> {
        let (value, j) =
            kornia_staging_3d::pose::stereographic_unproject_with_jacobian((*point).into());
        jacobian.data.0 = j;
        Vector4::from(value)
    }
}

/// One landmark: three optimised parameters, a host image, and its observations
#[derive(Debug, Clone, PartialEq)]
pub struct Landmark<S: Scalar> {
    /// Stereographic direction in the host camera frame, `direction`.
    pub direction: Vector2<S>,
    /// Inverse distance along that direction, `inv_dist`. Never negative: the
    /// increment is projected at zero.
    pub inv_dist: S,
    /// The image the direction is expressed in, `host_kf_id`.
    pub host_kf_id: TimeCamId,
    /// Where the landmark was seen, keyed by image, `obs`.
    pub obs: BTreeMap<TimeCamId, Vector2<S>>,
    /// The database key, `id`.
    pub id: LandmarkId,
    backup_direction: Vector2<S>,
    backup_inv_dist: S,
}

impl<S: Scalar> Landmark<S> {
    /// A landmark with no observations yet.
    ///
    /// The four fields here are exactly the four `addLandmark` copies
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

    /// Save the three optimized parameters. LM steps do not change observations or hosts.
    pub fn backup(&mut self) {
        self.backup_direction = self.direction;
        self.backup_inv_dist = self.inv_dist;
    }

    /// Undo the last increment, `restore`.
    pub fn restore(&mut self) {
        self.direction = self.backup_direction;
        self.inv_dist = self.backup_inv_dist;
    }
}

/// Typed database failures prevent data errors from panicking with the GIL released (D32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LandmarkError {
    /// An observation, a removal or a query named a landmark the database has
    /// never been given.
    #[error("landmark {0:?} is not in the database")]
    UnknownLandmark(LandmarkId),
}

/// Minimum observations needed for a landmark to retain information.
const MIN_NUM_OBS: usize = 2;

/// The landmark store and the host -> target -> landmark adjacency the
/// linearizer iterates.
///
/// The adjacency is not a cache: `observations` is what
/// [`crate::ba_base::BundleAdjustmentBase::compute_error`] and the linearizer
/// walk, and every mutation keeps it exactly consistent with the landmarks'
/// own `obs` maps — an empty target set drops the target, an empty target map
/// drops the host.
#[derive(Debug, Clone)]
pub struct LandmarkDatabase<S: Scalar> {
    /// Sorted by `LandmarkId`; `index` maps an id to a position here.
    kpts: Vec<Landmark<S>>,
    index: BTreeMap<LandmarkId, usize>,
    observations: BTreeMap<TimeCamId, BTreeMap<TimeCamId, BTreeSet<LandmarkId>>>,
}

impl<S: Scalar> Default for LandmarkDatabase<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> LandmarkDatabase<S> {
    /// An empty database.
    pub fn new() -> Self {
        Self {
            kpts: Vec::new(),
            index: BTreeMap::new(),
            observations: BTreeMap::new(),
        }
    }

    /// Forget everything, `clear`.
    pub fn clear(&mut self) {
        self.kpts.clear();
        self.index.clear();
        self.observations.clear();
    }

    /// Add or overwrite parameters while preserving existing observations.
    /// If the host changes, move adjacency entries to the new host so both database
    /// views stay consistent.
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

    /// Record an observation, or return [`LandmarkError::UnknownLandmark`] for an absent id.
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

    /// Look up a landmark by id; return `None` if absent.
    pub fn get_landmark(&self, lm_id: LandmarkId) -> Option<&Landmark<S>> {
        self.index.get(&lm_id).and_then(|&at| self.kpts.get(at))
    }

    /// A landmark by id, mutable.
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
    pub fn landmark_exists(&self, lm_id: LandmarkId) -> bool {
        self.index.contains_key(&lm_id)
    }

    /// Every landmark, in id order — one dense slice, for the GPU seam.
    ///
    /// `getLandmarks` hands back the whole
    /// `unordered_map`; this is the same content in a reproducible order.
    pub fn landmarks(&self) -> &[Landmark<S>] {
        &self.kpts
    }

    /// The host -> target -> landmark-set adjacency, `getObservations`
    pub fn observations(&self) -> &BTreeMap<TimeCamId, BTreeMap<TimeCamId, BTreeSet<LandmarkId>>> {
        &self.observations
    }

    /// The images that host at least one landmark, `getHostKfs`
    /// in sorted order.
    pub fn host_kfs(&self) -> Vec<TimeCamId> {
        self.observations.keys().copied().collect()
    }

    /// The targets one host is observed in, and the landmarks in each.
    ///
    /// The per-(host, target) iteration the linearizer needs: one relative pose
    /// `T_t_h` is hoisted per pair, and every landmark
    /// in the set reuses it.
    pub fn targets_for_host(
        &self,
        tcid: TimeCamId,
    ) -> Option<&BTreeMap<TimeCamId, BTreeSet<LandmarkId>>> {
        self.observations.get(&tcid)
    }

    /// Number of landmarks, `numLandmarks`.
    pub fn num_landmarks(&self) -> usize {
        self.kpts.len()
    }

    /// Number of (landmark, target) pairs in the adjacency, `numObservations`
    pub fn num_observations(&self) -> usize {
        self.observations
            .values()
            .flat_map(|targets| targets.values())
            .map(BTreeSet::len)
            .sum()
    }

    /// Delete a landmark and its adjacency. A missing id is a no-op.
    pub fn remove_landmark(&mut self, lm_id: LandmarkId) {
        if let Some(&at) = self.index.get(&lm_id) {
            self.remove_landmark_at(at);
        }
    }

    /// The marginalization sweep, `removeKeyframes`
    ///
    /// Three rules, in this order and no other:
    ///
    /// 1. a landmark **hosted** by a frame in `kfs_to_marg` is deleted outright
    ///    — its direction is expressed in a frame that is about to
    ///    stop existing, so nothing can be salvaged;
    /// 2. otherwise, every observation made in a frame in any of the three sets
    ///    is dropped;
    /// 3. then the `min_num_obs = 2` sweep deletes what is left with fewer than
    ///    two observations.
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

    /// Save every landmark's parameters, `backup`.
    pub fn backup(&mut self) {
        for kpt in &mut self.kpts {
            kpt.backup();
        }
    }

    /// Undo every landmark's last increment, `restore`
    pub fn restore(&mut self) {
        for kpt in &mut self.kpts {
            kpt.restore();
        }
    }

    // ─── internals ────────────────────────────────────────────────────────

    /// Remove selected hosts and observations, then delete under-observed landmarks.
    /// Use one retain pass over the dense vector and one index rebuild.
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
    /// `removeLandmarkHelper`.
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

    /// Remove one adjacency entry, skipping it if absent.
    /// A landmark added without observations has no entry to remove.
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
/// above it.
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
fn detach_landmark<S: Scalar>(
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
    fn rejected_chart_projection_writes_the_nan_sentinel() {
        let south_pole = Vector4::new(0.0f64, 0.0, -1.0, 0.0);
        assert!(
            StereographicParam::project(&south_pole)
                .iter()
                .all(|v| v.is_nan())
        );
        let mut jacobian = nalgebra::Matrix2x4::zeros();
        let pixel = StereographicParam::project_with_jacobian(&south_pole, &mut jacobian);
        assert!(pixel.iter().all(|v| v.is_nan()));
        assert!(jacobian.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn adding_a_landmark_twice_keeps_its_observations() {
        // `addLandmark` copies four fields and leaves `obs` alone
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
