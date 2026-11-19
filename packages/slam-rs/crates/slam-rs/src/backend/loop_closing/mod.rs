//! Loop closing (RootSLAM §III-B; `loop_closing.cpp:120-694`,
//! `map_database.cpp:164-336`; DESIGN §3 rows 7-11, PR b2).
//!
//! At each new keyframe:
//!
//! 1. **Index.** Describe the keyframe's tracked keypoints (rotated BRIEF on its
//!    own images, the only images loop closing reads, dropped after its query:
//!    DESIGN §2.2), and add one bag of words per camera to the HashBoW database.
//! 2. **Retrieve.** Query with every camera's bag (B5; camera 0 under
//!    `rootslam-parity`), score each keyframe by its best (query camera, entry
//!    camera) pair before keeping the best N (B4), sort by score then id and
//!    accept the first that passes in that order (B3). Candidates are at least
//!    `frame_time_margin_s` older.
//! 3. **Validate** each candidate until one passes: reject it if it shares a
//!    tracked keypoint with the current keyframe (covisible, not a loop);
//!    match keyframe to keyframe (mutual, Hamming < 70, strict ratio 1.2: B6,
//!    B17; ids without a descriptor are skipped: B7); read the island (the
//!    candidate and its strongest covisibility neighbours) and match against
//!    its landmarks; GP3P RANSAC on the rig (#614) with its minimum inlier
//!    count; then the reprojection pass: unmatched landmarks of the candidate
//!    are projected with the PnP pose and searched for in a 16 px window of
//!    the current images, on a local copy of the matches (B8), and PnP runs
//!    again. The final PnP alone has the ratio gate `inliers >= 0.5 x` the
//!    matches it was fed (B1, DESIGN §4; B2), also RANSAC's acceptance bound.
//!
//! Then, in the same query (no stamp or cull runs in between, so the island
//! needs no protection: B10), apply measures the drift the loop would
//! correct; a small one fuses at once (landmark fusion, [`LoopFusion`], B9, and
//! a loop edge in the covisibility graph); a large one first runs the
//! pose-graph optimization ([`super::pgo`], PR c2), then fuses the same way.

mod window;

use std::collections::{BTreeMap, BTreeSet};

use kornia_bow::BoW;
use kornia_staging_3d::pose::gp3p::GeneralizedCorrespondence;
use kornia_staging_3d::pose::gp3p::ransac::{Gp3pRansacConfig, IterationSchedule, gp3p_ransac};
use kornia_staging_bow::hash_bow::{FrameCamId, HashBow};
use kornia_staging_imgproc::features::{
    BriefConfig, BriefSmoothing, Descriptor256, MatchConfig, describe_keypoints_u16, match_mutual,
};
use nalgebra::{Vector2, Vector3, Vector4};
use rayon::prelude::*;

use super::pgo::PgoState;
use super::{MapBackend, MapEvent};
use crate::calib::BasaltCamera;
use crate::camera::SlamCamera;
use crate::config::{BriefTests, LoopClosingConfig, LoopQueryCameras};
use crate::estimator::{ImageSlot, KeyframeImages};
use crate::lie::Se3;
use crate::types::{CamId, FrameId, KeypointId, LandmarkId, TimeCamId};

/// One camera's descriptors in ID order, contiguous for Hamming matching.
#[derive(Debug, Clone, Default)]
struct DescriptorSet {
    ids: Vec<u64>,
    values: Vec<Descriptor256>,
}

impl DescriptorSet {
    fn from_ordered(descriptors: impl IntoIterator<Item = (u64, Descriptor256)>) -> Self {
        let (ids, values): (Vec<_>, Vec<_>) = descriptors.into_iter().unzip();
        debug_assert!(
            ids.is_sorted_by(|a, b| a < b),
            "ids must be unique and ascending"
        );
        Self { ids, values }
    }

    fn get(&self, id: u64) -> Option<&Descriptor256> {
        self.ids.binary_search(&id).ok().map(|i| &self.values[i])
    }
}

type KeyframeDescriptors = Vec<DescriptorSet>;
type CameraMatch = (CamId, KeypointId, u64);

struct LoopQuery<'a> {
    current: FrameId,
    current_ids: BTreeSet<u64>,
    keypoints: &'a crate::estimator::Keypoints,
    images: &'a KeyframeImages,
    // The map does not change during verification. Candidates can share
    // island members, so compute each member's matches once per query.
    island_matches: BTreeMap<FrameId, Vec<CameraMatch>>,
}

impl<'a> LoopQuery<'a> {
    fn new(
        current: FrameId,
        ours: &[DescriptorSet],
        keypoints: &'a crate::estimator::Keypoints,
        images: &'a KeyframeImages,
    ) -> Self {
        Self {
            current,
            current_ids: ours
                .iter()
                .flat_map(|cam| cam.ids.iter().copied())
                .collect(),
            keypoints,
            images,
            island_matches: BTreeMap::new(),
        }
    }
}

/// One correspondence between the current keyframe and a map landmark.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopMatch {
    /// Camera of the current keyframe.
    pub cam: CamId,
    /// The current keyframe's tracked keypoint, or `None` for a corner the
    /// reprojection pass found.
    pub keypoint: Option<KeypointId>,
    /// Where the current keyframe sees it.
    pub pixel: Vector2<f32>,
    /// The island landmark.
    pub landmark: LandmarkId,
}

/// What fusion does with one verified match (B9: RootSLAM decides by
/// comparing the two ids, so a redetected match fuses only by id coincidence).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoopFusion {
    /// The current keypoint is a map landmark: fold the island landmark into
    /// it (it keeps its own host-relative coordinates).
    Fuse {
        /// The island landmark, removed.
        from: LandmarkId,
        /// The current keyframe's landmark, kept.
        into: LandmarkId,
    },
    /// Otherwise: the current keyframe observes the island landmark.
    AddObs {
        /// The island landmark.
        landmark: LandmarkId,
        /// The current keyframe's image.
        image: TimeCamId,
        /// Where it sees it.
        pixel: Vector2<f32>,
    },
}

/// A loop that passed verification.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopDetection {
    /// The new keyframe.
    pub current: FrameId,
    /// The keyframe it closes a loop with.
    pub candidate: FrameId,
    /// The candidate and its covisibility neighbours.
    pub island: Vec<FrameId>,
    /// `T_w_i` of the current keyframe from PnP against the island.
    pub corrected_pose: Se3<f64>,
    /// The final inlier matches, which the fusion reads.
    pub matches: Vec<LoopMatch>,
    /// How many matches survived each stage.
    pub stages: LoopStages,
}

/// Match counts of one verification, RootSLAM's `[LC]` debug line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoopStages {
    /// Keyframe-to-keyframe matches.
    pub initial_matches: usize,
    /// Matches against the island.
    pub island_matches: usize,
    /// PnP inliers among them.
    pub inliers: usize,
    /// Matches after the reprojection pass (0 when it is off).
    pub reprojection_matches: usize,
    /// PnP inliers among those.
    pub reprojection_inliers: usize,
}

/// What applying a verified loop did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopOutcome {
    /// Small drift: landmarks fused and a loop edge added.
    Fused {
        /// Island landmarks folded into current ones.
        landmarks_fused: usize,
        /// Observations of island landmarks added to the current keyframe.
        observations_added: usize,
    },
    /// Drift at or above the gate, but the pose-graph optimization is off or
    /// refused the graph: nothing changed.
    NeedsPgo,
    /// Drift at or above the gate: the pose-graph optimization moved the map,
    /// then landmarks were fused and a loop edge added (c2).
    Optimized {
        /// Island landmarks folded into current ones.
        landmarks_fused: usize,
        /// Observations of island landmarks added to the current keyframe.
        observations_added: usize,
    },
}

/// One verified loop and what became of it.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopEvent {
    /// The loop as verified; its matches are not kept once applied (empty).
    pub detection: LoopDetection,
    /// Translation of `T_map^-1 T_pnp`, metres.
    pub drift_m: f64,
    /// Rotation angle of `T_map^-1 T_pnp`, degrees.
    pub drift_deg: f64,
    /// What applying it did.
    pub outcome: LoopOutcome,
}

/// What loop closing has done so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoopCounts {
    /// Keyframes described and added to the database.
    pub indexed: u64,
    /// Queries that found no images for their keyframe (dropped, culled, or a
    /// lagged pipeline that no longer held them).
    pub without_images: u64,
    /// Keyframe stamps whose images the pipeline dropped before the back end
    /// took them (Realtime, more stamps queued than the query cap).
    pub images_stripped: u64,
    /// Queries that retrieved at least one candidate.
    pub with_candidates: u64,
    /// Candidates validated.
    pub candidates: u64,
    /// GP3P RANSAC calls that refused their input (non-finite or non-unit
    /// rays, a bad ratio): a bug upstream of it, logged and counted, never a
    /// silent "no loop".
    pub pnp_errors: u64,
    /// Loops that passed verification.
    pub verified: u64,
    /// Loops fused at once.
    pub fused: u64,
    /// Loops at or above the drift gate: the pose-graph optimization's triggers.
    pub needs_pgo: u64,
    /// Of those, closed by an applied pose-graph optimization.
    pub optimized: u64,
}

/// Loop-closing state beside the map: the HashBoW database, the descriptors of
/// every indexed keyframe, the images of keyframes whose query has not run,
/// and the verified loops.
#[derive(Debug, Clone)]
pub struct LoopClosing {
    config: LoopClosingConfig,
    cameras: Vec<SlamCamera<f64>>,
    bow: HashBow,
    descriptors: BTreeMap<FrameId, KeyframeDescriptors>,
    images: BTreeMap<FrameId, KeyframeImages>,
    events: Vec<LoopEvent>,
    counts: LoopCounts,
    /// What the pose-graph optimization has done (PR c2).
    pub(super) pgo: PgoState,
}

impl LoopClosing {
    /// Loop closing for a rig with these cameras; `None` when the config turns
    /// it off.
    ///
    /// # Errors
    ///
    /// A camera model the projection code does not support, or a word size
    /// outside HashBoW's range.
    pub fn new(
        config: &LoopClosingConfig,
        intrinsics: &[BasaltCamera<f64>],
    ) -> Result<Option<Self>, String> {
        if !config.enabled {
            return Ok(None);
        }
        let cameras = intrinsics
            .iter()
            .map(SlamCamera::from_model)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        Ok(Some(Self {
            config: config.clone(),
            cameras,
            bow: HashBow::new(config.bow_num_bits).map_err(|error| error.to_string())?,
            descriptors: BTreeMap::new(),
            images: BTreeMap::new(),
            events: Vec::new(),
            counts: LoopCounts::default(),
            pgo: PgoState::default(),
        }))
    }

    /// Every verified loop, oldest first.
    pub fn events(&self) -> &[LoopEvent] {
        &self.events
    }

    /// The counters; the verified-loop ones are counted from the events.
    pub fn counts(&self) -> LoopCounts {
        let mut counts: LoopCounts = self.counts;
        for event in &self.events {
            counts.verified += 1;
            match event.outcome {
                LoopOutcome::Fused { .. } => counts.fused += 1,
                LoopOutcome::NeedsPgo => counts.needs_pgo += 1,
                LoopOutcome::Optimized { .. } => {
                    counts.needs_pgo += 1;
                    counts.optimized += 1;
                }
            }
        }
        counts
    }

    /// What the pose-graph optimization has done.
    pub fn pgo(&self) -> &PgoState {
        &self.pgo
    }

    /// The section this loop closing runs with.
    pub fn config(&self) -> &LoopClosingConfig {
        &self.config
    }

    /// Keyframes whose images wait for their query.
    pub fn held_images(&self) -> usize {
        self.images.len()
    }

    /// Hold a new keyframe's images until its query; count a slot the
    /// pipeline emptied first.
    pub(super) fn hold_images(&mut self, keyframe: FrameId, slot: &ImageSlot) {
        match slot.take() {
            Some(images) => _ = self.images.insert(keyframe, images),
            None => self.counts.images_stripped += 1,
        }
    }

    /// A culled keyframe leaves the database; a dropped query frees its images.
    pub(super) fn forget(&mut self, keyframe: FrameId) {
        self.bow.remove_frame(keyframe);
        self.descriptors.remove(&keyframe);
        self.images.remove(&keyframe);
        self.pgo.raw.remove(&keyframe);
    }

    /// Release the images of a query that will not run.
    pub(super) fn drop_images(&mut self, keyframe: FrameId) {
        self.images.remove(&keyframe);
    }

    fn brief(&self) -> BriefConfig {
        match self.config.brief_tests {
            BriefTests::Box => BriefConfig::default(),
            BriefTests::Raw => BriefConfig {
                smoothing: BriefSmoothing::None,
            },
        }
    }

    /// `validateCandidate`'s covisibility check and keyframe matching: the
    /// initial match count, if the candidate passes both.
    fn initial_matches(&self, query: &LoopQuery<'_>, candidate: FrameId) -> Option<usize> {
        let ours = self.descriptors.get(&query.current)?;
        let theirs = self.descriptors.get(&candidate)?;
        if share_a_track(&query.current_ids, theirs) {
            return None;
        }
        let initial = consolidate(match_cameras(ours, theirs, self.matching())).len();
        (initial >= self.config.min_initial_matches).then_some(initial)
    }

    fn matching(&self) -> MatchConfig {
        MatchConfig {
            max_distance: self.config.max_hamming_distance,
            second_best_ratio: self.config.second_best_test_ratio,
        }
    }

    /// Describe the keyframe's keypoints and add its bags (`indexKeyframe`).
    /// Returns the bag of each camera that has one.
    fn index(
        &mut self,
        keyframe: FrameId,
        keypoints: &crate::estimator::Keypoints,
        images: &KeyframeImages,
    ) -> Vec<(CamId, BoW)> {
        let brief: BriefConfig = self.brief();
        let described: KeyframeDescriptors = keypoints
            .par_iter()
            .enumerate()
            .map(|(cam, camera_keypoints)| {
                let Some(image) = images.0.get(cam) else {
                    return DescriptorSet::default();
                };
                let pixels: Vec<[f32; 2]> =
                    camera_keypoints.iter().map(|(_, p)| [p.x, p.y]).collect();
                // Keyframe keypoints are in id order, so the described subset is too.
                DescriptorSet::from_ordered(
                    camera_keypoints
                        .iter()
                        .zip(describe_keypoints_u16(image, &pixels, brief))
                        .filter_map(|((id, _), described)| Some((id.0, described?.descriptor))),
                )
            })
            .collect();

        // Bags enter the database in camera order, so insert them serially.
        let mut bags: Vec<(CamId, BoW)> = Vec::new();
        for (cam, descriptors) in described.iter().take(images.0.len()).enumerate() {
            let words: Vec<[u64; 4]> = descriptors.values.iter().map(|d| d.0).collect();
            let bag: BoW = self.bow.bag(&words);
            self.bow.insert(
                FrameCamId {
                    frame: keyframe,
                    cam: cam as u32,
                },
                &bag,
            );
            bags.push((cam, bag));
        }
        self.descriptors.insert(keyframe, described);
        self.counts.indexed += 1;
        bags
    }
}

/// Candidate keyframes for a query, best first (`retrieveCandidates` with B3,
/// B4, B5 fixed): every query bag is scored against every entry older than
/// `keyframe - frame_time_margin_s`; a keyframe keeps its best score over all
/// (query camera, entry camera) pairs; keyframes are sorted by score, then id;
/// the best `num_frames_to_match` at or above `frames_to_match_threshold` are
/// returned.
pub(crate) fn retrieve(
    bow: &HashBow,
    bags: &[(CamId, BoW)],
    keyframe: FrameId,
    config: &LoopClosingConfig,
) -> Vec<FrameId> {
    let before: i64 = keyframe.saturating_sub((config.frame_time_margin_s * 1e9) as i64);
    let mut best: BTreeMap<FrameId, f32> = BTreeMap::new();
    for (cam, bag) in bags {
        if config.query_cameras == LoopQueryCameras::Cam0 && *cam != 0 {
            continue;
        }
        for hit in bow.query_frames(bag, 0, Some(before)) {
            let score = best.entry(hit.frame).or_insert(hit.score);
            *score = score.max(hit.score);
        }
    }
    let mut ranked: Vec<(FrameId, f32)> = best
        .into_iter()
        .filter(|&(frame, _)| frame != keyframe)
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked
        .into_iter()
        .take(config.num_frames_to_match)
        .filter(|&(_, score)| f64::from(score) >= config.frames_to_match_threshold)
        .map(|(frame, _)| frame)
        .collect()
}

/// Whether any camera shares a described track with the current keyframe
/// (`checkCovisibility`): then they are covisible, not a loop.
fn share_a_track(current_ids: &BTreeSet<u64>, candidate: &[DescriptorSet]) -> bool {
    candidate
        .iter()
        .flat_map(|cam| &cam.ids)
        .any(|id| current_ids.contains(id))
}

/// Mutual matches for every camera pair, in camera and descriptor id order,
/// as `(current camera, current keypoint, candidate id)` (`matchKeyframes`).
/// Only described ids take part (B7).
fn match_cameras(
    current: &[DescriptorSet],
    candidates: &[DescriptorSet],
    config: MatchConfig,
) -> Vec<CameraMatch> {
    let mut matches = Vec::new();
    for (cam, described) in current.iter().enumerate() {
        for candidate in candidates {
            for (i, j) in match_mutual(&described.values, &candidate.values, config) {
                matches.push((cam, KeypointId(described.ids[i]), candidate.ids[j]));
            }
        }
    }
    matches
}

/// Keep the first match of each (camera, current keypoint) (`consolidateMatches`).
fn consolidate(matches: Vec<CameraMatch>) -> Vec<CameraMatch> {
    let mut seen: BTreeSet<(CamId, KeypointId)> = BTreeSet::new();
    matches
        .into_iter()
        .filter(|&(cam, id, _)| seen.insert((cam, id)))
        .collect()
}

/// The PnP gate: at least `min_inliers` inliers and at least `ratio` of the
/// `fed` matches that PnP saw (B1: RootSLAM's integer cast makes the ratio 0;
/// B2: its second gate divides by the matches before the reprojection pass).
pub(crate) fn passes_pnp_gate(inliers: usize, fed: usize, min_inliers: usize, ratio: f64) -> bool {
    inliers >= min_inliers && inliers as f64 >= ratio * fed as f64
}

/// The translation (metres) and angle (degrees) of `T_map^-1 T_pnp`. In the
/// body frame by construction; RootSLAM's world-frame difference
/// `|t_pnp - t_map|` has the same norm (`R^T` keeps lengths), so B18 changes
/// no number and needs no knob.
pub(crate) fn drift(map: &Se3<f64>, pnp: &Se3<f64>) -> (f64, f64) {
    let relative: Se3<f64> = map.inverse() * *pnp;
    (
        relative.translation.norm(),
        relative.rotation.log().norm().to_degrees(),
    )
}

/// The fusions of a verified loop's matches, in match order (`merge_loop_landmarks`
/// with B9 fixed): a match whose current keypoint is a map landmark other than
/// the island one fuses; every other match adds an observation. Matches whose
/// island landmark left the map are skipped, as is a second fusion of either
/// landmark.
pub(crate) fn fusion_plan(
    map: &super::KeyframeMap,
    current: FrameId,
    matches: &[LoopMatch],
) -> Vec<LoopFusion> {
    let mut used: BTreeSet<LandmarkId> = BTreeSet::new();
    let mut plan: Vec<LoopFusion> = Vec::new();
    for m in matches {
        if map.landmark(m.landmark).is_none() {
            continue;
        }
        let into: Option<LandmarkId> = m
            .keypoint
            .map(LandmarkId::from)
            .filter(|&id| id != m.landmark && map.landmark(id).is_some());
        match into {
            Some(into) => {
                if !used.contains(&m.landmark) && !used.contains(&into) {
                    used.extend([m.landmark, into]);
                    plan.push(LoopFusion::Fuse {
                        from: m.landmark,
                        into,
                    });
                }
            }
            None => plan.push(LoopFusion::AddObs {
                landmark: m.landmark,
                image: TimeCamId::new(current, m.cam),
                pixel: m.pixel,
            }),
        }
    }
    plan
}

/// A RANSAC seed from the two keyframes and the pass: the same loop draws the
/// same samples in every run and in any order of queries.
fn seed(current: FrameId, candidate: FrameId, pass: u64) -> u64 {
    let mut z: u64 = (current as u64)
        ^ (candidate as u64).rotate_left(29)
        ^ pass.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl MapBackend {
    /// The loop-closing state; `None` when it is off.
    pub fn loop_closing(&self) -> Option<&LoopClosing> {
        self.loops.as_ref()
    }

    /// The query of one new keyframe: detect, then apply at once. The
    /// keyframe's images are dropped here, with
    /// those of any older keyframe whose query never ran.
    pub(super) fn close_loops(&mut self, keyframe: FrameId) -> Vec<MapEvent> {
        let Some(detection) = self.detect_loop(keyframe) else {
            return Vec::new();
        };
        let started = std::time::Instant::now();
        let Some(&LoopEvent {
            ref detection,
            outcome,
            ..
        }) = self.apply_loop(detection)
        else {
            return Vec::new();
        };
        let pgo_ns =
            matches!(outcome, LoopOutcome::Optimized { .. }).then(|| crate::duration_ns(started));
        vec![MapEvent::LoopVerified {
            current: detection.current,
            candidate: detection.candidate,
            outcome,
            pgo_ns,
        }]
    }

    /// Index `keyframe` and look for a loop at it.
    fn detect_loop(&mut self, keyframe: FrameId) -> Option<LoopDetection> {
        let loops = self.loops.as_mut()?;
        let images: Option<KeyframeImages> = loops.images.remove(&keyframe);
        loops.images.retain(|&id, _| id > keyframe);
        let Some(images) = images else {
            loops.counts.without_images += 1;
            return None;
        };
        let keypoints = self.map.keyframe(keyframe)?.keypoints.clone();
        let bags = loops.index(keyframe, &keypoints, &images);
        let candidates: Vec<FrameId> = retrieve(&loops.bow, &bags, keyframe, &loops.config);
        if candidates.is_empty() {
            return None;
        }
        loops.counts.with_candidates += 1;

        let mut query = LoopQuery::new(
            keyframe,
            loops.descriptors.get(&keyframe)?,
            &keypoints,
            &images,
        );
        // The first matching stage reads descriptors only. Run it across
        // candidates, then keep the original score order for geometry and PnP.
        let initial_matches: Vec<_> = candidates
            .par_iter()
            .map(|&candidate| loops.initial_matches(&query, candidate))
            .collect();
        for (candidate, initial) in candidates.into_iter().zip(initial_matches) {
            self.loops.as_mut()?.counts.candidates += 1;
            if let Some(initial) = initial
                && let Some(detection) = self.validate(&mut query, candidate, initial)
            {
                return Some(detection);
            }
        }
        None
    }

    /// `validateCandidate` after the covisibility check and keyframe matching,
    /// which `detect_loop` runs across candidates: island matching, PnP, the
    /// reprojection pass and PnP again.
    fn validate(
        &mut self,
        query: &mut LoopQuery<'_>,
        candidate: FrameId,
        initial: usize,
    ) -> Option<LoopDetection> {
        let current = query.current;
        let loops = self.loops.as_ref()?;
        let config: LoopClosingConfig = loops.config.clone();
        let matching: MatchConfig = loops.matching();
        let ours = loops.descriptors.get(&current)?;

        // The island and its landmarks, each with the descriptor its keyframe
        // gave the keypoint of the same id (B7: none, no match).
        let island: Vec<FrameId> = std::iter::once(candidate)
            .chain(self.graph.top_covisible(candidate, config.island_size))
            .collect();
        let mut island_matches: Vec<CameraMatch> = Vec::new();
        for &kf in &island {
            let matches = query.island_matches.entry(kf).or_insert_with(|| {
                let Some(described) = loops.descriptors.get(&kf) else {
                    return Vec::new();
                };
                let sets: Vec<DescriptorSet> = described
                    .iter()
                    .enumerate()
                    .map(|(cam, by_id)| {
                        let ids = self.map.landmarks_in_image(TimeCamId::new(kf, cam));
                        DescriptorSet::from_ordered(
                            ids.filter_map(|lm| Some((lm.0, *by_id.get(lm.0)?))),
                        )
                    })
                    .collect();
                match_cameras(ours, &sets, matching)
            });
            island_matches.extend_from_slice(matches);
        }
        let island_matches = consolidate(island_matches);
        let mut stages = LoopStages {
            initial_matches: initial,
            island_matches: island_matches.len(),
            ..LoopStages::default()
        };
        if island_matches.len() < config.min_island_matches {
            return None;
        }
        let pixel_of = |cam: CamId, id: KeypointId| -> Option<Vector2<f32>> {
            let list = query.keypoints.get(cam)?;
            let index = list.binary_search_by_key(&id, |(k, _)| *k).ok()?;
            Some(list[index].1)
        };
        let matches: Vec<LoopMatch> = island_matches
            .iter()
            .filter_map(|&(cam, id, lm)| {
                Some(LoopMatch {
                    cam,
                    keypoint: Some(id),
                    pixel: pixel_of(cam, id)?,
                    landmark: LandmarkId(lm),
                })
            })
            .collect();

        // B1 (DESIGN §4): the ratio gate reads the final PnP only; the first
        // keeps its minimum inlier count.
        let ratio: f64 = config.pnp_inliers_ratio;
        let first_ratio: f64 = if config.reproject_landmarks {
            0.0
        } else {
            ratio
        };
        let (mut pose, mut matches) = self.solve_pnp(
            &matches,
            seed(current, candidate, 0),
            config.pnp_min_inliers,
            first_ratio,
        )?;
        stages.inliers = matches.len();
        if config.reproject_landmarks {
            matches.extend(self.reproject(candidate, &matches, &pose, query.images));
            stages.reprojection_matches = matches.len();
            (pose, matches) = self.solve_pnp(
                &matches,
                seed(current, candidate, 1),
                config.reprojected_pnp_min_inliers,
                ratio,
            )?;
            stages.reprojection_inliers = matches.len();
        }
        Some(LoopDetection {
            current,
            candidate,
            island,
            corrected_pose: pose,
            matches,
            stages,
        })
    }

    /// GP3P RANSAC of the rig body pose from the matches; the pose and the
    /// inlier matches when they pass [`passes_pnp_gate`] over the matches fed
    /// to RANSAC (those with a landmark and a bearing, B2), else `None`.
    fn solve_pnp(
        &mut self,
        matches: &[LoopMatch],
        seed: u64,
        min_inliers: usize,
        ratio: f64,
    ) -> Option<(Se3<f64>, Vec<LoopMatch>)> {
        let loops = self.loops.as_ref()?;
        let config: &LoopClosingConfig = &loops.config;
        let mut used: Vec<LoopMatch> = Vec::with_capacity(matches.len());
        let mut correspondences: Vec<GeneralizedCorrespondence> = Vec::with_capacity(matches.len());
        for m in matches {
            let (Some(point), Some(camera), Some(t_i_c)) = (
                self.map.landmark_world(m.landmark, &self.rig.t_i_c),
                loops.cameras.get(m.cam),
                self.rig.t_i_c.get(m.cam),
            ) else {
                continue;
            };
            let mut bearing: Vector4<f64> = Vector4::zeros();
            if !camera.unproject(&m.pixel.cast::<f64>(), &mut bearing) {
                continue;
            }
            correspondences.push(GeneralizedCorrespondence::from_camera(
                t_i_c,
                &bearing.xyz(),
                &point,
            ));
            used.push(*m);
        }
        if correspondences.len() < 4 {
            return None;
        }
        let ransac = Gp3pRansacConfig {
            threshold: config.pnp_ransac_threshold,
            schedule: config
                .pnp_ransac_fixed_iterations
                .map_or_else(IterationSchedule::default, |iterations| {
                    IterationSchedule::OpenGv { iterations }
                }),
            seed,
            refine: true,
            min_inlier_ratio: (ratio > 0.0).then_some(ratio.min(1.0)),
        };
        let result = match gp3p_ransac(&correspondences, &ransac) {
            Ok(result) => result?,
            Err(error) => {
                log::warn!(
                    "loop closing: GP3P RANSAC refused {} correspondences: {error}",
                    correspondences.len()
                );
                self.loops.as_mut()?.counts.pnp_errors += 1;
                return None;
            }
        };
        let fed: usize = used.len();
        let inliers: Vec<LoopMatch> = used
            .into_iter()
            .zip(&result.inliers)
            .filter(|(_, is_in)| **is_in)
            .map(|(m, _)| m)
            .collect();
        passes_pnp_gate(inliers.len(), fed, min_inliers, ratio).then_some((result.pose, inliers))
    }

    /// The reprojection pass (`reprojectLandmarks` + `searchInWindow`): every
    /// landmark the candidate observes in camera `c` and no match holds yet is
    /// projected into the current camera `c` with `pose`; inside the image, a
    /// corner matching the candidate's descriptor of it near that pixel becomes
    /// a new match. Returns the new matches only; the caller's matches and the
    /// keyframe's keypoints are left as they are (B8).
    fn reproject(
        &self,
        candidate: FrameId,
        matches: &[LoopMatch],
        pose: &Se3<f64>,
        images: &KeyframeImages,
    ) -> Vec<LoopMatch> {
        let Some(loops) = self.loops.as_ref() else {
            return Vec::new();
        };
        let Some(described) = loops.descriptors.get(&candidate) else {
            return Vec::new();
        };
        let search = window::WindowSearch {
            side: loops.config.fast_grid_size,
            fast_threshold: loops.config.fast_threshold,
            brief: loops.brief(),
            matching: MatchConfig {
                max_distance: loops.config.redetect_max_hamming_distance,
                second_best_ratio: loops.config.redetect_second_best_test_ratio,
            },
        };
        let mut found: Vec<LoopMatch> = Vec::new();
        for (cam, (camera, image)) in loops.cameras.iter().zip(images.0.iter()).enumerate() {
            let (Some(t_i_c), Some(by_id)) = (self.rig.t_i_c.get(cam), described.get(cam)) else {
                continue;
            };
            let matched: BTreeSet<LandmarkId> = matches
                .iter()
                .filter(|m| m.cam == cam)
                .map(|m| m.landmark)
                .collect();
            let t_c_w: Se3<f64> = (*pose * *t_i_c).inverse();
            let (width, height) = (image.width() as f64, image.height() as f64);
            for landmark in self.map.landmarks_in_image(TimeCamId::new(candidate, cam)) {
                if matched.contains(&landmark) {
                    continue;
                }
                let (Some(point), Some(target)) = (
                    self.map.landmark_world(landmark, &self.rig.t_i_c),
                    by_id.get(landmark.0),
                ) else {
                    continue;
                };
                let point_c: Vector3<f64> = t_c_w * point;
                let mut uv: Vector2<f64> = Vector2::zeros();
                if point_c.z <= 0.0 || !camera.project_point(&point_c.push(1.0), &mut uv, None) {
                    continue;
                }
                if !(0.0..width).contains(&uv.x) || !(0.0..height).contains(&uv.y) {
                    continue;
                }
                if let Some(pixel) = window::search_window(image, uv, target, &search) {
                    found.push(LoopMatch {
                        cam,
                        keypoint: None,
                        pixel,
                        landmark,
                    });
                }
            }
        }
        found
    }

    /// Apply a verified loop (`handle_loop_detection`): measure the drift it
    /// would correct; below the gate, fuse its landmarks and add the loop edge;
    /// at or above it, run the pose-graph optimization first
    /// (`handle_loop_closure`), then do the same. Records the event, without
    /// its matches, and returns it.
    fn apply_loop(&mut self, mut detection: LoopDetection) -> Option<&LoopEvent> {
        let config: &LoopClosingConfig = &self.loops.as_ref()?.config;
        let (min_m, min_deg) = (config.min_drift_reduction_m, config.min_rot_reduction_deg);
        let map_pose: Se3<f64> = self.map.keyframe(detection.current)?.pose;
        let (drift_m, drift_deg) = drift(&map_pose, &detection.corrected_pose);
        let outcome: LoopOutcome = if drift_m >= min_m || drift_deg >= min_deg {
            if self.optimize_loop(&detection) {
                self.graph
                    .add_loop_closure(detection.candidate, detection.current);
                match self.fuse(detection.current, &detection.matches) {
                    LoopOutcome::Fused {
                        landmarks_fused,
                        observations_added,
                    } => LoopOutcome::Optimized {
                        landmarks_fused,
                        observations_added,
                    },
                    other => other,
                }
            } else {
                LoopOutcome::NeedsPgo
            }
        } else {
            self.graph
                .add_loop_closure(detection.candidate, detection.current);
            self.fuse(detection.current, &detection.matches)
        };
        detection.matches = Vec::new();
        let loops = self.loops.as_mut()?;
        loops.events.push(LoopEvent {
            detection,
            drift_m,
            drift_deg,
            outcome,
        });
        loops.events.last()
    }

    /// Carry out [`fusion_plan`] ([`MapBackend::fuse_landmarks`]).
    fn fuse(&mut self, current: FrameId, matches: &[LoopMatch]) -> LoopOutcome {
        let plan: Vec<LoopFusion> = fusion_plan(&self.map, current, matches);
        let (landmarks_fused, observations_added) = self.fuse_landmarks(current, plan);
        LoopOutcome::Fused {
            landmarks_fused,
            observations_added,
        }
    }
}

#[cfg(test)]
mod tests;
