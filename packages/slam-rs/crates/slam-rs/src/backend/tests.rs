//! The map against hand-built stamp sequences, and the culling protection as a
//! property (B10, B19).

#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{Vector2, Vector3};
use proptest::prelude::*;

use super::*;
use crate::estimator::StampLandmark;
use crate::lie::{Se3, So3};
use crate::types::{CamId, KeypointId, LandmarkId, TimeCamId};

fn config(max_keyframes: usize, batch: usize) -> MapConfig {
    MapConfig {
        enable_culling: true,
        max_redundancy_score: 0.9,
        redundancy_exclude_hosted: false,
        max_keyframes,
        batch_cull_size: batch,
        high_covisibility: 2,
    }
}

fn pose(x: f64) -> Se3<f64> {
    Se3::new(So3::identity(), Vector3::new(x, 0.0, 0.0))
}

fn landmark(id: u64, host: FrameId, obs: &[(FrameId, CamId)]) -> StampLandmark {
    StampLandmark {
        id: LandmarkId(id),
        host: TimeCamId::new(host, 0),
        direction: Vector2::new(0.0, 0.0),
        inv_dist: 0.5,
        obs: obs
            .iter()
            .map(|&(frame, cam)| (TimeCamId::new(frame, cam), Vector2::new(1.0, 1.0)))
            .collect(),
    }
}

fn keyframe(t_ns: FrameId, window: &[FrameId], landmarks: Vec<StampLandmark>) -> MapStamp {
    MapStamp::Keyframe(Box::new(KeyframeStamp {
        t_ns,
        keyframe_poses: window.iter().map(|&id| (id, pose(id as f64))).collect(),
        landmarks,
        keypoints: vec![vec![(KeypointId(t_ns as u64), Vector2::new(2.0, 3.0))]; 2],
        images: None,
    }))
}

fn marg(ids: &[FrameId], x: f64, landmarks: Vec<StampLandmark>) -> MapStamp {
    MapStamp::Marginalized(MargStamp {
        keyframe_poses: ids.iter().map(|&id| (id, pose(x))).collect(),
        landmarks,
    })
}

/// A back end on two 640 x 480 cameras with these extrinsics and no camera
/// models (loop closing and the global BA off), with these map knobs.
fn backend_with(config: MapConfig, t_i_c: Vec<Se3<f64>>) -> MapBackend {
    let rig = BackendRig {
        intrinsics: Vec::new(),
        t_i_c,
        resolutions: vec![[640, 480]; 2],
    };
    let mut backend = MapBackend::new(&VioConfig::default(), rig, crate::Backend::Cpu);
    backend.config = config;
    backend.loops = None;
    backend.gba_rig = None;
    backend
}

fn backend(max_keyframes: usize, batch: usize) -> MapBackend {
    backend_with(config(max_keyframes, batch), Vec::new())
}

#[test]
fn keyframe_stamps_build_the_map_and_the_graph() {
    let mut backend = backend(100, 10);
    assert_eq!(
        backend.step(keyframe(1, &[1], vec![])),
        vec![MapEvent::KeyframeAdded(1)]
    );
    backend.step(keyframe(
        2,
        &[1, 2],
        vec![
            landmark(10, 1, &[(1, 0), (2, 0)]),
            landmark(11, 1, &[(1, 0), (1, 1), (2, 1)]),
            landmark(12, 2, &[(2, 0)]),
        ],
    ));
    backend.step(keyframe(
        3,
        &[1, 2, 3],
        vec![
            landmark(10, 1, &[(1, 0), (2, 0), (3, 0)]),
            landmark(13, 2, &[(2, 0), (3, 1)]),
        ],
    ));
    // 12 has one observation: dropped. 10 gains keyframe 3.
    assert_eq!(
        backend.stats(),
        MapStats {
            keyframes: 3,
            landmarks: 3,
            observations: 3 + 3 + 2
        }
    );
    assert_eq!(backend.active_keyframes(), &BTreeSet::from([1, 2, 3]));
    let graph = backend.graph();
    assert_eq!(graph.root(), Some(1));
    // 2 shares 10 and 11 with 1 (11 counts once over two cameras).
    assert_eq!(graph.weight(1, 2), Some(2));
    // 3 shares 10 with 1, and 10 and 13 with 2.
    assert_eq!(graph.weight(1, 3), Some(1));
    assert_eq!(graph.weight(2, 3), Some(2));
    assert_eq!(graph.parent(3), Some(2));
    assert_eq!(backend.map().keyframe(3).unwrap().keypoints[1].len(), 1);
    assert_eq!(backend.map().keyframe(2).unwrap().pose, pose(2.0));
}

#[test]
fn a_marginalization_stamp_writes_final_values_and_releases_the_keyframe() {
    let mut backend = backend(100, 10);
    backend.step(keyframe(1, &[1], vec![]));
    backend.step(keyframe(
        2,
        &[1, 2],
        vec![landmark(10, 1, &[(1, 0), (2, 0)])],
    ));
    let mut last = landmark(10, 1, &[(1, 0), (2, 0)]);
    last.inv_dist = 0.125;
    let events = backend.step(marg(&[1], 7.0, vec![last]));
    assert_eq!(events, vec![MapEvent::KeyframeMarginalized(1)]);
    assert_eq!(backend.map().keyframe(1).unwrap().pose, pose(7.0));
    assert_eq!(
        backend.map().landmark(LandmarkId(10)).unwrap().inv_dist,
        0.125
    );
    assert_eq!(backend.active_keyframes(), &BTreeSet::from([2]));
    // An eviction the map never saw changes nothing.
    assert!(backend.step(marg(&[99], 0.0, vec![])).is_empty());
}

/// Five keyframes all see landmarks 10 and 11; keyframe 2 sees nothing else,
/// so it is fully redundant when it leaves the window.
fn redundant_map() -> MapBackend {
    let mut backend = backend(100, 10);
    let seen_by_all = |id| landmark(id, 1, &[(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
    backend.step(keyframe(1, &[1], vec![]));
    for t in 2..=5 {
        let window: Vec<FrameId> = (1..=t).collect();
        backend.step(keyframe(t, &window, vec![seen_by_all(10), seen_by_all(11)]));
    }
    backend.step(keyframe(
        6,
        &[1, 2, 3, 4, 5, 6],
        vec![
            landmark(20, 3, &[(3, 0), (6, 0)]),
            landmark(21, 1, &[(1, 0), (3, 1), (6, 1)]),
        ],
    ));
    backend
}

#[test]
fn a_redundant_keyframe_is_culled_as_it_leaves_and_never_comes_back() {
    let mut backend = redundant_map();
    assert_eq!(backend.graph().weight(2, 3), Some(2));
    let events = backend.step(marg(&[2], 2.0, vec![]));
    assert_eq!(
        events,
        vec![
            MapEvent::KeyframeMarginalized(2),
            MapEvent::KeyframeCulled {
                id: 2,
                reason: CullReason::Redundant
            }
        ]
    );
    assert!(!backend.map().contains_keyframe(2));
    assert!(!backend.graph().contains(2));
    assert_eq!(backend.graph().parent(3), Some(1), "3's parent was 2");
    // A late stamp naming the culled keyframe does not bring it back.
    backend.step(keyframe(
        7,
        &[2, 7],
        vec![landmark(30, 2, &[(2, 0), (7, 0)])],
    ));
    assert!(!backend.map().contains_keyframe(2));
    assert!(backend.map().landmark(LandmarkId(30)).is_none());
}

#[test]
fn the_root_is_never_culled() {
    let mut backend = redundant_map();
    // Keyframe 1 is the root: redundant or not, it stays.
    let events = backend.step(marg(&[1], 1.0, vec![]));
    assert_eq!(events, vec![MapEvent::KeyframeMarginalized(1)]);
    assert!(backend.map().contains_keyframe(1));
}

#[test]
fn an_over_budget_map_culls_its_lowest_scored_inactive_keyframes() {
    let mut backend = backend(4, 2);
    backend.step(keyframe(1, &[1], vec![]));
    for t in 2..=4 {
        backend.step(keyframe(
            t,
            &[t - 1, t],
            vec![landmark(t as u64, t - 1, &[(t - 1, 0), (t, 0)])],
        ));
        backend.step(marg(&[t - 1], (t - 1) as f64, vec![]));
    }
    assert_eq!(backend.map().keyframes().len(), 4);
    let events = backend.step(keyframe(
        5,
        &[4, 5],
        vec![landmark(5, 4, &[(4, 0), (5, 0)])],
    ));
    let culled: Vec<FrameId> = events
        .iter()
        .filter_map(|event| match event {
            MapEvent::KeyframeCulled {
                id,
                reason: CullReason::OverBudget,
            } => Some(*id),
            _ => None,
        })
        .collect();
    // 1 is the root, 4 and 5 are in the window: only 2 and 3 can go, 3 first
    // (the middle of the time span scores lowest).
    assert_eq!(culled, vec![3, 2]);
    assert_eq!(
        backend
            .map()
            .keyframes()
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 4, 5]
    );
}

/// RootSLAM re-ranks after each cull (`handleGraphScoreChange`), and so does
/// the batch: a cull that takes a neighbour's landmarks lowers its score before
/// the next pick. Ranked once, the batch would take 3 then 4.
#[test]
fn the_batch_re_ranks_after_each_cull() {
    let mut backend = backend_with(
        MapConfig {
            high_covisibility: 100,
            ..config(5, 2)
        },
        Vec::new(),
    );
    backend.step(keyframe(1, &[1], vec![]));
    // 2 sees only ten landmarks hosted by 3; 4 sees fifteen of its own with 5.
    // No edge is high and every pixel is the same, so graph = 1 and coverage =
    // 0 everywhere: score = 0.4 + 0.1 time + 0.2 n / (n + 10).
    let landmarks: Vec<StampLandmark> = (0..10)
        .map(|k| landmark(100 + k, 3, &[(2, 0), (3, 0)]))
        .chain((0..15).map(|k| landmark(200 + k, 4, &[(4, 0), (5, 0)])))
        .collect();
    backend.step(keyframe(5, &[2, 3, 4, 5], landmarks));
    backend.step(marg(&[2, 3, 4], 0.0, vec![]));
    let score = |backend: &MapBackend, id| {
        cull::keyframe_score(backend.map(), backend.graph(), id, 100, &[[640, 480]; 2])
    };
    let events = backend.step(keyframe(6, &[5, 6], vec![]));
    let culled: Vec<FrameId> = events
        .iter()
        .filter_map(|event| match event {
            MapEvent::KeyframeCulled {
                id,
                reason: CullReason::OverBudget,
            } => Some(*id),
            _ => None,
        })
        .collect();
    // Before: 3 = 0.4 + 0.02 + 0.1 = 0.52, 4 = 0.4 + 0.02 + 0.12 = 0.54,
    // 2 = 0.4 + 0.06 + 0.1 = 0.56. Culling 3 takes its ten landmarks, so
    // 2 drops to 0.46 and goes before 4.
    assert_eq!(culled, vec![3, 2]);
    assert!((score(&backend, 4) - 0.54).abs() < 1e-6);
}

// ── property: culling never touches a protected keyframe ────────────────

/// One step of a random session.
#[derive(Debug, Clone)]
enum Step {
    /// A new keyframe; the window keeps the newest `window` keyframes. Each
    /// landmark is `(host offset back, observing offsets back)`.
    Keyframe {
        window: usize,
        landmarks: Vec<(usize, Vec<(usize, CamId)>)>,
    },
    /// Evict the oldest window keyframe.
    Marginalize,
}

fn step() -> impl Strategy<Value = Step> {
    let landmark = (
        0usize..4,
        prop::collection::vec((0usize..6, 0usize..2), 1..5),
    );
    prop_oneof![
        4 => (2usize..5, prop::collection::vec(landmark, 0..8))
            .prop_map(|(window, landmarks)| Step::Keyframe { window, landmarks }),
        3 => Just(Step::Marginalize),
    ]
}

/// What must survive a step that marginalizes `releasing` (if any): the root,
/// and every window keyframe but the one leaving.
fn protected(backend: &MapBackend, releasing: Option<FrameId>) -> BTreeSet<FrameId> {
    let mut protected: BTreeSet<FrameId> = backend.active_keyframes().clone();
    if let Some(releasing) = releasing {
        protected.remove(&releasing);
    }
    protected.extend(backend.graph().root());
    protected
}

/// Step `backend` and fail if a culled keyframe was in `protected`. Returns
/// the over-budget culls, in order.
fn checked_step(
    backend: &mut MapBackend,
    stamp: MapStamp,
    protected: &BTreeSet<FrameId>,
) -> Result<Vec<FrameId>, TestCaseError> {
    let mut over_budget: Vec<FrameId> = Vec::new();
    for event in backend.step(stamp) {
        if let MapEvent::KeyframeCulled { id, reason } = event {
            prop_assert!(!protected.contains(&id), "culled protected keyframe {}", id);
            if reason == CullReason::OverBudget {
                over_budget.push(id);
            }
        }
    }
    Ok(over_budget)
}

/// The over-budget batch the slow way: merge `stamp` without culling, then
/// score every unprotected keyframe afresh before each pick.
fn rescored_batch(backend: &MapBackend, stamp: MapStamp) -> Vec<FrameId> {
    let mut reference: MapBackend = backend.clone();
    reference.config.enable_culling = false;
    reference.step(stamp);
    if reference.map.keyframes().len() <= reference.config.max_keyframes {
        return Vec::new();
    }
    let high_weight: u32 = reference.high_covisibility_weight();
    let mut culled: Vec<FrameId> = Vec::new();
    while culled.len() < reference.config.batch_cull_size {
        let lowest: Option<(f32, FrameId)> = reference
            .map
            .keyframes()
            .keys()
            .copied()
            .filter(|&id| !reference.is_protected(id))
            .map(|id| {
                let score = cull::keyframe_score(
                    &reference.map,
                    &reference.graph,
                    id,
                    high_weight,
                    &reference.rig.resolutions,
                );
                (score, id)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let Some((_, id)) = lowest else {
            break;
        };
        reference.cull_keyframe(id);
        culled.push(id);
    }
    culled
}

proptest! {
    #[test]
    fn culling_never_removes_a_protected_keyframe_and_the_map_stays_consistent(
        steps in prop::collection::vec(step(), 1..60),
        max_keyframes in 3usize..8,
    ) {
        let mut backend = backend_with(
            MapConfig { max_redundancy_score: 0.5, ..config(max_keyframes, 2) },
            Vec::new(),
        );
        let mut made: Vec<FrameId> = Vec::new();
        let mut window: Vec<FrameId> = Vec::new();
        let mut next_landmark: u64 = 0;
        for step in steps {
            match step {
                Step::Keyframe { window: size, landmarks } => {
                    let t = made.len() as FrameId + 1;
                    made.push(t);
                    window.push(t);
                    let landmarks = landmarks
                        .into_iter()
                        .map(|(host_back, obs)| {
                            next_landmark += 1;
                            let back = |offset: usize| window[window.len() - 1 - offset.min(window.len() - 1)];
                            let host = back(host_back);
                            let mut obs: Vec<(FrameId, CamId)> = obs.into_iter().map(|(o, cam)| (back(o), cam)).collect();
                            obs.push((host, 0));
                            landmark(next_landmark, host, &obs)
                        })
                        .collect();
                    let mut guard = protected(&backend, None);
                    guard.extend(window.iter().copied());
                    let stamp = keyframe(t, &window, landmarks);
                    let expected = rescored_batch(&backend, stamp.clone());
                    // The incremental re-rank picks what a full re-score would.
                    prop_assert_eq!(checked_step(&mut backend, stamp, &guard)?, expected);
                    while window.len() > size {
                        let oldest = window.remove(0);
                        let guard = protected(&backend, Some(oldest));
                        checked_step(&mut backend, marg(&[oldest], 0.0, vec![]), &guard)?;
                    }
                }
                Step::Marginalize => {
                    if window.len() > 1 {
                        let oldest = window.remove(0);
                        let guard = protected(&backend, Some(oldest));
                        checked_step(&mut backend, marg(&[oldest], 0.0, vec![]), &guard)?;
                    }
                }
            }
            let map = backend.map();
            prop_assert_eq!(
                backend.graph().node_ids().collect::<Vec<_>>(),
                map.keyframes().keys().copied().collect::<Vec<_>>()
            );
            prop_assert_eq!(
                backend.active_keyframes().iter().copied().collect::<Vec<_>>(),
                window.iter().copied().filter(|id| map.contains_keyframe(*id)).collect::<Vec<_>>()
            );
            for id in backend.active_keyframes() {
                prop_assert!(map.contains_keyframe(*id));
            }
            let mut by_keyframe: BTreeMap<FrameId, BTreeSet<LandmarkId>> = BTreeMap::new();
            for (&landmark_id, landmark) in map.landmarks() {
                prop_assert!(landmark.obs.len() >= 2);
                prop_assert!(map.contains_keyframe(landmark.host.frame_id));
                for image in landmark.obs.keys().chain([&landmark.host]) {
                    prop_assert!(map.contains_keyframe(image.frame_id));
                    by_keyframe.entry(image.frame_id).or_default().insert(landmark_id);
                }
            }
            for &id in map.keyframes().keys() {
                prop_assert_eq!(
                    map.landmarks_of_keyframe(id),
                    by_keyframe.remove(&id).unwrap_or_default()
                );
            }
        }
    }
}

/// The snapshot (PR a2) copies the map in id order: keyframes with their
/// window flag, world points from host pose x extrinsic x stereographic
/// direction / inverse distance, and each covisibility edge once.
#[test]
fn a_map_snapshot_copies_keyframes_world_points_and_edges() {
    // Camera 0 sits 0.1 m along y of the IMU, camera 1 0.2 m along z.
    let t_i_c: Vec<Se3<f64>> = vec![
        Se3::new(So3::identity(), Vector3::new(0.0, 0.1, 0.0)),
        Se3::new(So3::identity(), Vector3::new(0.0, 0.0, 0.2)),
    ];
    let mut backend = backend_with(config(100, 10), t_i_c);
    backend.step(keyframe(1, &[1], vec![]));
    backend.step(keyframe(
        2,
        &[1, 2],
        vec![
            landmark(10, 1, &[(1, 0), (2, 0)]),
            landmark(11, 2, &[(2, 1), (1, 1)]),
        ],
    ));
    backend.step(marg(&[1], 1.5, vec![]));

    let snapshot: MapSnapshot = backend.snapshot();

    assert_eq!(snapshot.keyframe_t_ns, vec![1, 2]);
    assert_eq!(snapshot.keyframe_poses, vec![pose(1.5), pose(2.0)]);
    assert_eq!(snapshot.keyframe_active, vec![false, true]);
    assert_eq!(snapshot.point_ids, vec![LandmarkId(10), LandmarkId(11)]);
    assert_eq!(snapshot.point_hosts, vec![1, 2]);
    // Direction (0, 0) is the optical axis; inverse distance 0.5 is 2 m.
    // Landmark 10: host 1 (x = 1.5 after the marg stamp), camera 0.
    // Landmark 11: host 2 (x = 2), camera 0 too (`landmark` hosts in camera 0).
    assert_eq!(
        snapshot.points_world,
        vec![[1.5, 0.1, 2.0], [2.0, 0.1, 2.0]]
    );
    assert_eq!(snapshot.covisibility_edges, vec![(1, 2, 2)]);
    assert_eq!(
        snapshot.stats,
        MapStats {
            keyframes: 2,
            landmarks: 2,
            observations: 4
        }
    );
    assert_eq!(snapshot.high_covisibility_weight, 2);
    assert_eq!(snapshot.culls, CullCounts::default());
}

/// A delta from revision 0 is the whole map's points; from a later revision it
/// has every keyframe pose but only the points of the hosts a write touched,
/// each where the snapshot puts it.
#[test]
fn a_map_delta_carries_every_pose_and_only_the_changed_hosts_points() {
    let mut backend = backend_with(config(100, 10), two_cameras());
    backend.step(keyframe(1, &[1], vec![]));
    backend.step(keyframe(
        2,
        &[1, 2],
        vec![
            landmark(10, 1, &[(1, 0), (2, 0)]),
            landmark(11, 2, &[(2, 1), (1, 1)]),
        ],
    ));
    let first: MapDelta = backend.delta(0);
    assert_eq!(first.changed_hosts, vec![1, 2]);
    assert_eq!(first.point_counts, vec![1, 1]);
    assert_eq!(
        hosted(
            &first.changed_hosts,
            &first.point_counts,
            &first.point_ids,
            &first.points_world
        ),
        by_host(&backend.snapshot())
    );

    let read: u64 = first.revision;
    let quiet: MapDelta = backend.delta(read);
    assert!(
        quiet.changed_hosts.is_empty() && quiet.point_ids.is_empty(),
        "nothing changed since the read"
    );
    assert_eq!(quiet.keyframes, backend.map_keyframes());

    backend.step(marg(&[1], 1.5, vec![]));
    let moved: MapDelta = backend.delta(read);
    let now: MapSnapshot = backend.snapshot();
    assert_eq!(
        moved.changed_hosts,
        vec![1],
        "the marginalization moved keyframe 1 only"
    );
    assert_eq!(moved.keyframes.poses, now.keyframe_poses);
    assert_eq!(moved.keyframes.active, now.keyframe_active);
    assert_eq!(
        moved.point_ids,
        vec![LandmarkId(10)],
        "keyframe 1 hosts landmark 10"
    );
}

fn two_cameras() -> Vec<Se3<f64>> {
    vec![
        Se3::new(So3::identity(), Vector3::new(0.0, 0.1, 0.0)),
        Se3::new(So3::identity(), Vector3::new(0.0, 0.0, 0.2)),
    ]
}

type HostedPoints = BTreeMap<FrameId, BTreeMap<LandmarkId, [f32; 3]>>;

/// Delta points by host (hosts with none left out).
fn hosted(
    hosts: &[FrameId],
    counts: &[usize],
    ids: &[LandmarkId],
    points: &[[f32; 3]],
) -> HostedPoints {
    let mut out = HostedPoints::new();
    let mut start: usize = 0;
    for (&host, &count) in hosts.iter().zip(counts) {
        if count > 0 {
            out.insert(
                host,
                ids[start..start + count]
                    .iter()
                    .copied()
                    .zip(points[start..start + count].iter().copied())
                    .collect(),
            );
        }
        start += count;
    }
    out
}

/// A snapshot's points by host.
fn by_host(snapshot: &MapSnapshot) -> HostedPoints {
    let mut out = HostedPoints::new();
    for ((id, host), point) in snapshot
        .point_ids
        .iter()
        .zip(&snapshot.point_hosts)
        .zip(&snapshot.points_world)
    {
        out.entry(*host).or_default().insert(*id, *point);
    }
    out
}

/// One write to the map, as the window, culling, loop fusion, the PGO and the
/// global BA's write-back make them. Indices pick among what exists.
#[derive(Debug, Clone)]
enum MapWrite {
    /// A keyframe stamp: a new keyframe and landmarks hosted in the window.
    Keyframe(Vec<(u8, u8, u8)>),
    /// A marginalization stamp moving one keyframe.
    Marginalize(u8, i8),
    /// A pose write (the PGO): `true` writes the pose it already has.
    Pose(u8, i8, bool),
    /// The global BA's write-back of one landmark.
    Position(u8, u8),
    /// An observation-only merge: the landmark again, same geometry, a new image.
    Observe(u8, u8),
    /// A landmark merged with new geometry from the window (a rehost when the host differs).
    Rewrite(u8, u8, u8),
    RemoveLandmark(u8),
    /// A cull.
    RemoveKeyframe(u8),
    /// Loop fusion.
    Fold(u8, u8),
}

fn map_write() -> impl Strategy<Value = MapWrite> {
    prop_oneof![
        3 => prop::collection::vec((any::<u8>(), 0u8..2, 1u8..8), 0..4).prop_map(MapWrite::Keyframe),
        1 => (any::<u8>(), any::<i8>()).prop_map(|(k, x)| MapWrite::Marginalize(k, x)),
        1 => (any::<u8>(), any::<i8>(), any::<bool>()).prop_map(|(k, x, same)| MapWrite::Pose(k, x, same)),
        1 => (any::<u8>(), 1u8..8).prop_map(|(l, d)| MapWrite::Position(l, d)),
        1 => (any::<u8>(), any::<u8>()).prop_map(|(l, k)| MapWrite::Observe(l, k)),
        1 => (any::<u8>(), any::<u8>(), 1u8..8).prop_map(|(l, k, d)| MapWrite::Rewrite(l, k, d)),
        1 => any::<u8>().prop_map(MapWrite::RemoveLandmark),
        1 => any::<u8>().prop_map(MapWrite::RemoveKeyframe),
        1 => (any::<u8>(), any::<u8>()).prop_map(|(a, b)| MapWrite::Fold(a, b)),
    ]
}

fn pick<T: Copy>(items: &[T], index: u8) -> Option<T> {
    (!items.is_empty()).then(|| items[usize::from(index) % items.len()])
}

proptest! {
    /// The delta contract: a reader that folds each delta into what it holds
    /// (dropping keyframes that left the map) holds exactly the snapshot's
    /// points, and `delta(0)`'s, after every kind of write; every public write
    /// takes at most one revision, and a read with nothing changed is empty.
    #[test]
    fn folded_deltas_hold_the_map(writes in prop::collection::vec(map_write(), 1..40)) {
        let mut backend = backend_with(config(100, 10), two_cameras());
        let mut next_keyframe: FrameId = 1;
        let mut next_landmark: u64 = 1;
        let mut held = HostedPoints::new();
        let mut since: u64 = 0;
        for write in writes {
            let write: MapWrite = write;
            let keyframes: Vec<FrameId> = backend.map.keyframes().keys().copied().collect();
            let landmarks: Vec<LandmarkId> = backend.map.landmarks().keys().copied().collect();
            let before: u64 = backend.map.revision();
            match write.clone() {
                MapWrite::Keyframe(hosted_landmarks) => {
                    let id: FrameId = next_keyframe;
                    next_keyframe += 1;
                    let window: Vec<FrameId> = keyframes.iter().copied().chain([id]).collect();
                    let stamped: Vec<StampLandmark> = hosted_landmarks
                        .iter()
                        .map(|&(host, cam, depth)| {
                            let host: FrameId = pick(&window, host).unwrap_or(id);
                            let other: FrameId = if host == id { window[0] } else { id };
                            let mut stamped = landmark(next_landmark, host, &[(host, usize::from(cam)), (other, 1 - usize::from(cam))]);
                            stamped.host = TimeCamId::new(host, usize::from(cam));
                            stamped.inv_dist = f64::from(depth) / 8.0;
                            stamped.direction = Vector2::new(0.05 * next_landmark as f64, 0.01);
                            next_landmark += 1;
                            stamped
                        })
                        .collect();
                    // A stamp is several map writes (one pose per window keyframe, then the landmarks).
                    backend.step(keyframe(id, &window, stamped));
                }
                MapWrite::Marginalize(k, x) => {
                    let Some(k) = pick(&keyframes, k) else { continue };
                    backend.step(marg(&[k], f64::from(x) / 10.0, vec![]));
                }
                MapWrite::Pose(k, x, same) => {
                    let Some(k) = pick(&keyframes, k) else { continue };
                    let pose_now: Se3<f64> = backend.map.keyframe(k).unwrap().pose;
                    backend.map.upsert_keyframe(k, if same { pose_now } else { pose(f64::from(x) / 10.0) }, None);
                    if same {
                        prop_assert_eq!(backend.map.revision(), before, "an unchanged pose takes no revision");
                    }
                }
                MapWrite::Position(l, depth) => {
                    let Some(l) = pick(&landmarks, l) else { continue };
                    backend.map.set_landmark_position(l, Vector2::new(0.03, -0.02), f64::from(depth) / 8.0);
                }
                MapWrite::Observe(l, k) => {
                    let (Some(l), Some(k)) = (pick(&landmarks, l), pick(&keyframes, k)) else { continue };
                    let current: MapLandmark = backend.map.landmark(l).unwrap().clone();
                    let again = StampLandmark {
                        id: l,
                        host: current.host,
                        direction: current.direction,
                        inv_dist: current.inv_dist,
                        obs: [(TimeCamId::new(k, 1), Vector2::new(4.0, 4.0))].into_iter().collect(),
                    };
                    backend.map.merge_landmarks(&[again]);
                    prop_assert_eq!(backend.map.revision(), before, "an observation-only merge takes no revision");
                }
                MapWrite::Rewrite(l, k, depth) => {
                    let (Some(l), Some(k)) = (pick(&landmarks, l), pick(&keyframes, k)) else { continue };
                    let current: MapLandmark = backend.map.landmark(l).unwrap().clone();
                    let rewritten = StampLandmark {
                        id: l,
                        host: TimeCamId::new(k, 0),
                        direction: Vector2::new(-0.04, 0.02),
                        inv_dist: f64::from(depth) / 8.0,
                        obs: current.obs.clone(),
                    };
                    backend.map.merge_landmarks(&[rewritten]);
                }
                MapWrite::RemoveLandmark(l) => {
                    let Some(l) = pick(&landmarks, l) else { continue };
                    backend.map.remove_landmark(l);
                }
                MapWrite::RemoveKeyframe(k) => {
                    let Some(k) = pick(&keyframes, k) else { continue };
                    backend.map.remove_keyframe(k);
                }
                MapWrite::Fold(a, b) => {
                    let (Some(a), Some(b)) = (pick(&landmarks, a), pick(&landmarks, b)) else { continue };
                    backend.map.merge_landmark(a, b);
                }
            }
            if !matches!(write, MapWrite::Keyframe(_) | MapWrite::Marginalize(..)) {
                prop_assert!(backend.map.revision() <= before + 1, "one revision per public write");
            }
            let delta: MapDelta = backend.delta(since);
            held.retain(|host, _| delta.keyframes.t_ns.contains(host));
            for (host, points) in hosted(&delta.changed_hosts, &delta.point_counts, &delta.point_ids, &delta.points_world) {
                held.insert(host, points);
            }
            for (host, count) in delta.changed_hosts.iter().zip(&delta.point_counts) {
                if *count == 0 {
                    held.remove(host);
                }
            }
            since = delta.revision;
            let whole: MapDelta = backend.delta(0);
            let snapshot: MapSnapshot = backend.snapshot();
            prop_assert_eq!(&held, &by_host(&snapshot));
            prop_assert_eq!(&hosted(&whole.changed_hosts, &whole.point_counts, &whole.point_ids, &whole.points_world), &by_host(&snapshot));
            prop_assert_eq!(&delta.keyframes, &backend.map_keyframes());
            prop_assert!(backend.delta(since).changed_hosts.is_empty(), "a read with nothing changed is empty");
        }
    }
}

/// A landmark whose host camera has no extrinsic is left out, not invented.
#[test]
fn a_map_snapshot_skips_points_it_cannot_place() {
    let mut backend = backend(100, 10);
    backend.step(keyframe(1, &[1], vec![]));
    backend.step(keyframe(
        2,
        &[1, 2],
        vec![landmark(10, 1, &[(1, 0), (2, 0)])],
    ));
    let snapshot: MapSnapshot = backend.snapshot();
    assert!(snapshot.point_ids.is_empty());
    assert!(snapshot.points_world.is_empty());
    assert_eq!(snapshot.stats.landmarks, 1);
}

/// A landmark at infinity (`inv_dist` 0, which the window allows) has no
/// finite position: left out of the points, still counted (RootSLAM's map
/// export keeps finite positions only).
#[test]
fn a_map_snapshot_skips_points_at_infinity() {
    let mut backend = backend_with(
        config(100, 10),
        vec![Se3::new(So3::identity(), Vector3::zeros()); 2],
    );
    let mut far: StampLandmark = landmark(11, 1, &[(1, 0), (2, 0)]);
    far.inv_dist = 0.0;
    backend.step(keyframe(1, &[1], vec![]));
    backend.step(keyframe(
        2,
        &[1, 2],
        vec![landmark(10, 1, &[(1, 0), (2, 0)]), far],
    ));
    let snapshot: MapSnapshot = backend.view();
    assert_eq!(snapshot.point_ids, vec![LandmarkId(10)]);
    assert!(
        snapshot
            .points_world
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    assert_eq!(snapshot.stats.landmarks, 2);
}

#[test]
fn reanchoring_refuses_a_frame_without_a_rotation() {
    // A zero quaternion is no rotation: refused, as the Python binding refuses it,
    // never read as the identity.
    let snapshot: MapSnapshot = backend(100, 10).snapshot();
    let mut frames = vec![(10, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (20, [0.0; 7])];
    assert_eq!(
        snapshot.reanchor(&frames),
        Err(AnchorError::NoRotation { t_ns: 20 })
    );
    frames[1].1[6] = 1.0;
    assert_eq!(snapshot.reanchor(&frames).unwrap()[1], frames[1].1);
}

#[test]
fn a_rig_with_ragged_camera_lists_is_refused() {
    let mut calibration = crate::calib::Calibration::<f64>::default();
    calibration.resolution.push([640, 480]);
    assert!(matches!(
        BackendRig::new(&calibration),
        Err(crate::calib::CalibError::RaggedCameraLists {
            extrinsics: 0,
            intrinsics: 0,
            resolutions: 1
        })
    ));
}
