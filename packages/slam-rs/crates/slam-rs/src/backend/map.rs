//! The persistent keyframe map (RootSLAM §III-A, `map_database.cpp`,
//! `landmark_database.cpp`).
//!
//! Keyframe poses (`T_w_i`, `f64`), host-relative landmarks with their keyframe
//! observations, and each keyframe's tracked keypoints. Landmarks stay in the
//! host-relative form the window uses, so moving a keyframe moves the landmarks
//! it hosts. Every container is ordered by id (D31).
//!
//! Retention (DESIGN §2.2): a keyframe keeps keypoints (and, from b1,
//! descriptors), never an image.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::Vector2;

use crate::estimator::{Keypoints, StampLandmark};
use crate::lie::Se3;
use crate::types::{CamId, FrameId, KeypointId, LandmarkId, TimeCamId};

/// Fewest observations a map landmark keeps (`min_num_obs`).
const MIN_OBSERVATIONS: usize = 2;

/// A landmark observed by more than this many keyframes counts as redundant
/// in the keyframe that also sees it (`getKeyframeRedundancyScore`).
const REDUNDANT_ABOVE_KEYFRAMES: usize = 3;

/// One map keyframe.
#[derive(Debug, Clone, PartialEq)]
pub struct MapKeyframe {
    /// `T_w_i`, the newest value the window or the back end gave it.
    pub pose: Se3<f64>,
    /// Tracked keypoints per camera, in id order. Empty for a keyframe the map
    /// first saw in a stamp of another keyframe.
    pub keypoints: Keypoints,
}

/// One map landmark.
#[derive(Debug, Clone, PartialEq)]
pub struct MapLandmark {
    /// The image the direction is expressed in.
    pub host: TimeCamId,
    /// Stereographic direction in the host camera frame.
    pub direction: Vector2<f64>,
    /// Inverse distance along that direction.
    pub inv_dist: f64,
    /// Where map keyframes saw it; at least two entries.
    pub obs: BTreeMap<TimeCamId, Vector2<f32>>,
}

/// The counts RootSLAM writes to `stats_map.json`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MapStats {
    /// `num_kfs`.
    pub keyframes: usize,
    /// `num_lms`.
    pub landmarks: usize,
    /// `num_obs`: (landmark, image) pairs.
    pub observations: usize,
}

/// Keyframes, landmarks and the image -> landmark index.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeyframeMap {
    keyframes: BTreeMap<FrameId, MapKeyframe>,
    landmarks: BTreeMap<LandmarkId, MapLandmark>,
    /// Landmarks each image observes or hosts (`keyframe_obs`).
    by_image: BTreeMap<TimeCamId, BTreeSet<LandmarkId>>,
}

impl KeyframeMap {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every keyframe, in id order.
    pub fn keyframes(&self) -> &BTreeMap<FrameId, MapKeyframe> {
        &self.keyframes
    }

    /// One keyframe.
    pub fn keyframe(&self, id: FrameId) -> Option<&MapKeyframe> {
        self.keyframes.get(&id)
    }

    /// Whether `id` is a map keyframe.
    pub fn contains_keyframe(&self, id: FrameId) -> bool {
        self.keyframes.contains_key(&id)
    }

    /// Every landmark, in id order.
    pub fn landmarks(&self) -> &BTreeMap<LandmarkId, MapLandmark> {
        &self.landmarks
    }

    /// One landmark.
    pub fn landmark(&self, id: LandmarkId) -> Option<&MapLandmark> {
        self.landmarks.get(&id)
    }

    /// The `stats_map.json` counts.
    pub fn stats(&self) -> MapStats {
        MapStats {
            keyframes: self.keyframes.len(),
            landmarks: self.landmarks.len(),
            observations: self.landmarks.values().map(|lm| lm.obs.len()).sum(),
        }
    }

    /// The median number of landmarks a map keyframe sees (the upper median);
    /// 0 for an empty map.
    pub fn median_keyframe_landmarks(&self) -> usize {
        let mut seen: Vec<usize> = self
            .keyframes
            .keys()
            .map(|&id| self.landmarks_of_keyframe(id).len())
            .collect();
        seen.sort_unstable();
        seen.get(seen.len() / 2).copied().unwrap_or(0)
    }

    /// Landmarks one image observes or hosts.
    pub fn landmarks_in_image(&self, image: TimeCamId) -> impl Iterator<Item = LandmarkId> + '_ {
        self.by_image.get(&image).into_iter().flatten().copied()
    }

    /// Landmarks keyframe `id` observes or hosts in any camera, each once.
    pub fn landmarks_of_keyframe(&self, id: FrameId) -> BTreeSet<LandmarkId> {
        self.by_image
            .range(TimeCamId::new(id, 0)..=TimeCamId::new(id, CamId::MAX))
            .flat_map(|(_, landmarks)| landmarks.iter().copied())
            .collect()
    }

    /// Add keyframe `id`, or update the pose of the one already there. Keypoints
    /// replace the stored ones only when given.
    pub fn upsert_keyframe(&mut self, id: FrameId, pose: Se3<f64>, keypoints: Option<Keypoints>) {
        match self.keyframes.get_mut(&id) {
            Some(keyframe) => {
                keyframe.pose = pose;
                if let Some(keypoints) = keypoints {
                    keyframe.keypoints = keypoints;
                }
            }
            None => {
                self.keyframes.insert(
                    id,
                    MapKeyframe {
                        pose,
                        keypoints: keypoints.unwrap_or_default(),
                    },
                );
            }
        }
    }

    /// Write window landmarks over the map's (`mergeLMDB` with `override`): new
    /// values for direction, inverse distance and host, observations added or
    /// updated, none removed. A landmark hosted outside the map is skipped, and
    /// so is an observation in an image outside it (a culled keyframe; RootSLAM's
    /// `removeReferencesToCulledKeyframes`). A landmark left with fewer than two
    /// observations is removed.
    pub fn merge_landmarks(&mut self, incoming: &[StampLandmark]) {
        for stamped in incoming {
            if !self.keyframes.contains_key(&stamped.host.frame_id) {
                continue;
            }
            let obs = stamped
                .obs
                .iter()
                .filter(|(image, _)| self.keyframes.contains_key(&image.frame_id))
                .map(|(image, pixel)| (*image, *pixel));
            let landmark: &mut MapLandmark =
                self.landmarks
                    .entry(stamped.id)
                    .or_insert_with(|| MapLandmark {
                        host: stamped.host,
                        direction: stamped.direction,
                        inv_dist: stamped.inv_dist,
                        obs: BTreeMap::new(),
                    });
            let old_host: TimeCamId = landmark.host;
            landmark.host = stamped.host;
            landmark.direction = stamped.direction;
            landmark.inv_dist = stamped.inv_dist;
            landmark.obs.extend(obs);
            let images: Vec<TimeCamId> = landmark.obs.keys().copied().collect();
            if old_host != stamped.host && !images.contains(&old_host) {
                detach(&mut self.by_image, old_host, stamped.id);
            }
            for image in images.into_iter().chain([stamped.host]) {
                self.by_image.entry(image).or_default().insert(stamped.id);
            }
            if self
                .landmarks
                .get(&stamped.id)
                .is_some_and(|lm| lm.obs.len() < MIN_OBSERVATIONS)
            {
                self.remove_landmark(stamped.id);
            }
        }
    }

    /// Move a landmark to a new direction and inverse distance in its host
    /// camera (the global BA's write-back); host and observations stay. A
    /// missing id is a no-op.
    pub fn set_landmark_position(
        &mut self,
        id: LandmarkId,
        direction: Vector2<f64>,
        inv_dist: f64,
    ) {
        if let Some(landmark) = self.landmarks.get_mut(&id) {
            landmark.direction = direction;
            landmark.inv_dist = inv_dist;
        }
    }

    /// Delete a landmark and its index entries; a missing id is a no-op.
    pub fn remove_landmark(&mut self, id: LandmarkId) -> Option<MapLandmark> {
        let landmark: MapLandmark = self.landmarks.remove(&id)?;
        for image in landmark.obs.keys().chain([&landmark.host]) {
            detach(&mut self.by_image, *image, id);
        }
        Some(landmark)
    }

    /// Delete keyframe `id` (`removeKeyframe`): the landmarks it hosts go, its
    /// observations go, and a landmark left with fewer than two observations
    /// goes. Returns the removed landmarks as they were before the removal,
    /// for the covisibility bookkeeping; `None` for an unknown keyframe.
    pub fn remove_keyframe(&mut self, id: FrameId) -> Option<Vec<(LandmarkId, MapLandmark)>> {
        self.keyframes.remove(&id)?;
        let mut removed: Vec<(LandmarkId, MapLandmark)> = Vec::new();
        for landmark_id in self.landmarks_of_keyframe(id) {
            let Some(landmark) = self.landmarks.get_mut(&landmark_id) else {
                continue;
            };
            if landmark.host.frame_id == id {
                if let Some(landmark) = self.remove_landmark(landmark_id) {
                    removed.push((landmark_id, landmark));
                }
                continue;
            }
            let before: MapLandmark = landmark.clone();
            landmark.obs.retain(|image, _| image.frame_id != id);
            if landmark.obs.len() < MIN_OBSERVATIONS {
                self.remove_landmark(landmark_id);
                removed.push((landmark_id, before));
            }
        }
        self.by_image.retain(|image, _| image.frame_id != id);
        Some(removed)
    }

    /// The fraction of the landmarks keyframe `id` sees that more than three
    /// keyframes see (`getKeyframeRedundancyScore`); 0 when it sees none.
    pub fn redundancy_score(&self, id: FrameId, exclude_hosted: bool) -> f32 {
        let seen: BTreeSet<LandmarkId> = self.landmarks_of_keyframe(id);
        let redundant: usize = seen
            .iter()
            .filter_map(|landmark_id| self.landmarks.get(landmark_id))
            .filter(|landmark| !(exclude_hosted && landmark.host.frame_id == id))
            .filter(|landmark| {
                let keyframes: BTreeSet<FrameId> =
                    landmark.obs.keys().map(|image| image.frame_id).collect();
                keyframes.len() > REDUNDANT_ABOVE_KEYFRAMES
            })
            .count();
        if seen.is_empty() {
            0.0
        } else {
            redundant as f32 / seen.len() as f32
        }
    }

    /// How much of each image the observations of keyframe `id` span: per camera
    /// with observations, bounding-box area over image area, averaged
    /// (`computeImgCoverageScore`, which divides cumulative sums instead).
    pub fn image_coverage(&self, id: FrameId, resolutions: &[[u32; 2]]) -> f32 {
        let mut ratios: Vec<f32> = Vec::new();
        for (cam_id, [width, height]) in resolutions.iter().enumerate() {
            let image = TimeCamId::new(id, cam_id);
            let pixels: Vec<Vector2<f32>> = self
                .landmarks_in_image(image)
                .filter_map(|landmark_id| self.landmarks.get(&landmark_id)?.obs.get(&image))
                .copied()
                .collect();
            if pixels.is_empty() || *width == 0 || *height == 0 {
                continue;
            }
            let (min, max) = pixels.iter().fold(
                (Vector2::repeat(f32::MAX), Vector2::repeat(f32::MIN)),
                |(min, max), pixel| (min.inf(pixel), max.sup(pixel)),
            );
            let area: f32 = (max.x - min.x) * (max.y - min.y);
            ratios.push(area / (*width as f32 * *height as f32));
        }
        if ratios.is_empty() {
            0.0
        } else {
            ratios.iter().sum::<f32>() / ratios.len() as f32
        }
    }

    /// `n / (n + 10)` for the `n` landmarks keyframe `id` sees
    /// (`computeTotalObservedLmsScore`).
    pub fn observed_landmarks_score(&self, id: FrameId) -> f32 {
        let seen: f32 = self.landmarks_of_keyframe(id).len() as f32;
        seen / (seen + 10.0)
    }

    /// Bytes the map's own buffers hold, counted by payload: a lower bound on
    /// what the allocator holds for it (tree nodes carry slack).
    /// `tests/backend_map.rs` checks it against a counting allocator.
    pub fn retained_bytes(&self) -> usize {
        use std::mem::size_of;
        let keyframes: usize = self
            .keyframes
            .values()
            .map(|keyframe| {
                size_of::<(FrameId, MapKeyframe)>()
                    + keyframe
                        .keypoints
                        .iter()
                        .map(|camera| {
                            size_of::<Vec<(KeypointId, Vector2<f32>)>>()
                                + camera.capacity() * size_of::<(KeypointId, Vector2<f32>)>()
                        })
                        .sum::<usize>()
            })
            .sum();
        let landmarks: usize = self
            .landmarks
            .values()
            .map(|landmark| {
                size_of::<(LandmarkId, MapLandmark)>()
                    + landmark.obs.len() * size_of::<(TimeCamId, Vector2<f32>)>()
            })
            .sum();
        let index: usize = self
            .by_image
            .values()
            .map(|set| size_of::<TimeCamId>() + set.len() * size_of::<LandmarkId>())
            .sum();
        keyframes + landmarks + index
    }
}

/// Drop `id` from `by_image[image]`, then the entry if it empties.
fn detach(
    by_image: &mut BTreeMap<TimeCamId, BTreeSet<LandmarkId>>,
    image: TimeCamId,
    id: LandmarkId,
) {
    if let Some(set) = by_image.get_mut(&image) {
        set.remove(&id);
        if set.is_empty() {
            by_image.remove(&image);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn landmark(id: u64, host: (FrameId, CamId), obs: &[(FrameId, CamId)]) -> StampLandmark {
        StampLandmark {
            id: LandmarkId(id),
            host: TimeCamId::new(host.0, host.1),
            direction: Vector2::new(0.1 * id as f64, 0.0),
            inv_dist: 0.5,
            obs: obs
                .iter()
                .enumerate()
                .map(|(k, &(frame, cam))| {
                    (
                        TimeCamId::new(frame, cam),
                        Vector2::new(10.0 * k as f32, 5.0 + k as f32),
                    )
                })
                .collect(),
        }
    }

    fn with_keyframes(ids: &[FrameId]) -> KeyframeMap {
        let mut map = KeyframeMap::new();
        for &id in ids {
            map.upsert_keyframe(id, Se3::identity(), None);
        }
        map
    }

    #[test]
    fn merging_adds_overrides_and_never_drops_an_observation() {
        let mut map = with_keyframes(&[1, 2, 3]);
        map.merge_landmarks(&[landmark(7, (1, 0), &[(1, 0), (2, 1)])]);
        assert_eq!(
            map.stats(),
            MapStats {
                keyframes: 3,
                landmarks: 1,
                observations: 2
            }
        );

        let mut update = landmark(7, (1, 0), &[(1, 0), (3, 0)]);
        update.inv_dist = 0.25;
        map.merge_landmarks(&[update]);
        let merged = map.landmark(LandmarkId(7)).unwrap();
        assert_eq!(merged.inv_dist, 0.25);
        assert_eq!(
            merged.obs.keys().copied().collect::<Vec<_>>(),
            vec![
                TimeCamId::new(1, 0),
                TimeCamId::new(2, 1),
                TimeCamId::new(3, 0)
            ]
        );
        assert_eq!(
            map.landmarks_of_keyframe(2),
            BTreeSet::from([LandmarkId(7)])
        );
        assert_eq!(map.stats().observations, 3);
    }

    #[test]
    fn merging_skips_what_lies_outside_the_map() {
        let mut map = with_keyframes(&[1, 2]);
        map.merge_landmarks(&[
            // host culled: skipped.
            landmark(1, (9, 0), &[(9, 0), (1, 0), (2, 0)]),
            // one observation in a culled keyframe: dropped, two remain.
            landmark(2, (1, 0), &[(1, 0), (2, 0), (9, 1)]),
            // left with one observation: removed.
            landmark(3, (1, 0), &[(1, 0), (9, 0)]),
        ]);
        assert_eq!(
            map.landmarks().keys().copied().collect::<Vec<_>>(),
            vec![LandmarkId(2)]
        );
        assert_eq!(map.landmark(LandmarkId(2)).unwrap().obs.len(), 2);
        assert!(
            map.landmarks_in_image(TimeCamId::new(9, 0))
                .next()
                .is_none()
        );
    }

    #[test]
    fn a_new_host_moves_the_index_entry() {
        let mut map = with_keyframes(&[1, 2, 3]);
        map.merge_landmarks(&[landmark(4, (1, 1), &[(2, 0), (3, 0)])]);
        assert_eq!(
            map.landmarks_of_keyframe(1),
            BTreeSet::from([LandmarkId(4)])
        );
        map.merge_landmarks(&[landmark(4, (2, 0), &[(2, 0), (3, 0)])]);
        assert!(map.landmarks_of_keyframe(1).is_empty());
        assert_eq!(
            map.landmark(LandmarkId(4)).unwrap().host,
            TimeCamId::new(2, 0)
        );
    }

    #[test]
    fn removing_a_keyframe_takes_its_landmarks_and_starved_ones() {
        let mut map = with_keyframes(&[1, 2, 3, 4]);
        map.merge_landmarks(&[
            landmark(1, (2, 0), &[(2, 0), (3, 0), (4, 0)]),
            landmark(2, (1, 0), &[(1, 0), (2, 1)]),
            landmark(3, (1, 0), &[(1, 0), (2, 0), (3, 0)]),
            landmark(4, (3, 0), &[(3, 0), (4, 1)]),
        ]);
        let removed = map.remove_keyframe(2).unwrap();
        let ids: Vec<u64> = removed.iter().map(|(id, _)| id.0).collect();
        assert_eq!(ids, vec![1, 2], "hosted, then starved below two");
        assert_eq!(removed[1].1.obs.len(), 2, "returned as it was");
        assert!(!map.contains_keyframe(2));
        assert_eq!(
            map.landmarks().keys().map(|id| id.0).collect::<Vec<_>>(),
            vec![3, 4]
        );
        assert_eq!(map.landmark(LandmarkId(3)).unwrap().obs.len(), 2);
        assert!(map.landmarks_of_keyframe(2).is_empty());
        assert!(map.remove_keyframe(2).is_none());
    }

    #[test]
    fn redundancy_counts_landmarks_seen_by_more_than_three_keyframes() {
        let mut map = with_keyframes(&[1, 2, 3, 4, 5]);
        map.merge_landmarks(&[
            landmark(1, (1, 0), &[(1, 0), (2, 0), (3, 0), (4, 0)]),
            landmark(2, (2, 0), &[(1, 1), (2, 0), (3, 0), (4, 0), (5, 0)]),
            landmark(3, (2, 0), &[(1, 0), (2, 0)]),
            landmark(4, (1, 0), &[(1, 0), (1, 1), (2, 0), (3, 0)]),
        ]);
        // keyframe 1 sees 1, 2, 3, 4; 1 and 2 are seen by four or more keyframes.
        assert_eq!(map.redundancy_score(1, false), 0.5);
        // excluding its hosted 1 and 4: 2 of 4 seen are counted, 1 of them redundant.
        assert_eq!(map.redundancy_score(1, true), 0.25);
        assert_eq!(map.redundancy_score(9, false), 0.0);
    }

    #[test]
    fn coverage_averages_each_cameras_box_over_its_image() {
        let mut map = with_keyframes(&[1, 2]);
        let mut a = landmark(1, (2, 0), &[(1, 0), (2, 0)]);
        a.obs.insert(TimeCamId::new(1, 0), Vector2::new(0.0, 0.0));
        let mut b = landmark(2, (2, 0), &[(1, 0), (2, 0)]);
        b.obs.insert(TimeCamId::new(1, 0), Vector2::new(50.0, 20.0));
        let mut c = landmark(3, (2, 0), &[(1, 1), (2, 0)]);
        c.obs.insert(TimeCamId::new(1, 1), Vector2::new(5.0, 5.0));
        map.merge_landmarks(&[a, b, c]);
        // camera 0: 50 x 20 of 100 x 100 = 0.1; camera 1: one point = 0.
        let coverage = map.image_coverage(1, &[[100, 100], [100, 100], [100, 100]]);
        assert!((coverage - 0.05).abs() < 1e-6, "{coverage}");
        assert_eq!(map.image_coverage(9, &[[100, 100]]), 0.0);
        assert!((map.observed_landmarks_score(1) - 3.0 / 13.0).abs() < 1e-6);
    }
}
