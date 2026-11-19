//! Loop closing against hand-built databases and a synthetic revisit: a rig
//! that sees a textured scene, leaves, and comes back 40 s later with new
//! track ids and a drifted map pose.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use kornia_image::{Image, ImageSize};
use nalgebra::{Vector2, Vector3, Vector4};

use super::*;
use crate::backend::{BackendWork, MapConfig};
use crate::calib::{BasaltCamera, Calibration, PinholeParams};
use crate::config::VioConfig;
use crate::estimator::{KeyframeStamp, MapStamp, MargStamp, StampLandmark};
use crate::landmark::StereographicParam;
use crate::lie::So3;

const SECOND: i64 = 1_000_000_000;
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
/// Points per camera; the revisit tracks the first `SEEN` of them.
const POINTS: usize = 60;
const SEEN: usize = 42;
/// First keyframe, its covisible neighbour, the keyframe before the revisit,
/// and the revisit 39 s after the first.
const A: FrameId = SECOND;
const A2: FrameId = SECOND + SECOND / 2;
const D: FrameId = 39 * SECOND + SECOND / 2;
const C: FrameId = 40 * SECOND;

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn calibration(cameras: usize) -> Calibration<f64> {
    Calibration {
        t_i_c: (0..cameras)
            .map(|c| {
                Se3::new(
                    So3::exp(&Vector3::new(0.0, 1.2 * c as f64, 0.0)),
                    Vector3::new(0.06 * c as f64, 0.0, 0.0),
                )
            })
            .collect(),
        intrinsics: vec![
            BasaltCamera::Pinhole(PinholeParams {
                fx: 300.0,
                fy: 300.0,
                cx: 320.0,
                cy: 240.0,
            });
            cameras
        ],
        resolution: vec![[WIDTH as u32, HEIGHT as u32]; cameras],
        ..Calibration::default()
    }
}

/// One camera's view: bright dots at integer pixels on a noise background,
/// and the depth of each dot.
struct View {
    image: Image<u16, 1>,
    pixels: Vec<Vector2<f32>>,
    depths: Vec<f64>,
}

fn view(rng: &mut Rng) -> View {
    let mut pixels: Vec<Vector2<f32>> = Vec::new();
    while pixels.len() < POINTS {
        let p = Vector2::new(40.0 + rng.below(560) as f32, 40.0 + rng.below(400) as f32);
        if pixels.iter().all(|q| (p - q).norm() >= 24.0) {
            pixels.push(p);
        }
    }
    let mut data: Vec<u16> = (0..WIDTH * HEIGHT)
        .map(|_| (rng.below(100) as u16) << 8)
        .collect();
    for p in &pixels {
        data[p.y as usize * WIDTH + p.x as usize] = 255 << 8;
    }
    View {
        image: Image::new(
            ImageSize {
                width: WIDTH,
                height: HEIGHT,
            },
            data,
        )
        .unwrap(),
        depths: (0..POINTS)
            .map(|_| 3.0 + rng.below(300) as f64 / 100.0)
            .collect(),
        pixels,
    }
}

/// The revisit: the views and a back end holding A and A2 (out of the
/// window), with the revisit C stepped `drift` metres off (and, with
/// `fuse_landmarks`, D before it, hosting landmarks C also sees).
struct Scene {
    views: Vec<View>,
    backend: MapBackend,
    /// Every stamp the scene stepped, in order, to replay through a runner.
    stamps: Vec<MapStamp>,
}

/// What a test changes in the revisit.
#[derive(Clone)]
struct Setup {
    cameras: usize,
    drift: f64,
    fuse_landmarks: bool,
    loops: LoopClosingConfig,
    max_keyframes: usize,
    /// C tracks the first `seen` points of each camera.
    seen: usize,
    /// Island landmarks `k < wrong` sit 30 px off their dot: outliers of the
    /// first PnP.
    wrong: usize,
    /// Island landmarks `k >= seen` sit 4 px off: the reprojection pass finds
    /// their dots and PnP rejects them (at [`tight`]'s threshold).
    shifted_unseen: bool,
    /// A decoy keyframe E (and its neighbour E2) before A: the same images,
    /// its own track ids, its landmarks `k >= seen` 4 px off. Its score ties
    /// A's, so it is validated first, and it fails after its reprojection pass.
    decoy: bool,
}

impl Setup {
    fn new(cameras: usize, drift: f64, loops: LoopClosingConfig) -> Self {
        Self {
            cameras,
            drift,
            fuse_landmarks: false,
            loops,
            max_keyframes: 100,
            seen: SEEN,
            wrong: 0,
            shifted_unseen: false,
            decoy: false,
        }
    }
}

/// A PnP threshold under which a landmark 4 px off its dot is an outlier
/// anywhere in the image (about 0.6 px).
fn tight(loops: LoopClosingConfig) -> LoopClosingConfig {
    LoopClosingConfig {
        pnp_ransac_threshold: 2e-6,
        ..loops
    }
}

/// The decoy keyframe and its neighbour ([`Setup::decoy`]).
const E: FrameId = SECOND / 2;
const E2: FrameId = 3 * SECOND / 4;

fn keypoint_id(cam: usize, k: usize, offset: u64) -> KeypointId {
    KeypointId(offset + 1 + 1000 * cam as u64 + k as u64)
}

fn landmark(
    views: &[View],
    calib: &Calibration<f64>,
    cam: usize,
    k: usize,
    id: KeypointId,
    host: FrameId,
    obs: &[FrameId],
) -> StampLandmark {
    placed(views, calib, (cam, k), id, host, obs, 0.0)
}

/// [`landmark`] placed `off` pixels from its dot, in a direction drawn from
/// `(cam, k)`: observed at the dot, it is a PnP outlier.
fn placed(
    views: &[View],
    calib: &Calibration<f64>,
    (cam, k): (usize, usize),
    id: KeypointId,
    host: FrameId,
    obs: &[FrameId],
    off: f32,
) -> StampLandmark {
    let camera = SlamCamera::from_model(&calib.intrinsics[cam]).unwrap();
    let mut bearing = Vector4::zeros();
    let angle = Rng::new(1000 * cam as u64 + k as u64).below(360) as f32;
    let shift = off * Vector2::new(angle.to_radians().cos(), angle.to_radians().sin());
    assert!(camera.unproject(&(views[cam].pixels[k] + shift).cast::<f64>(), &mut bearing));
    StampLandmark {
        id: LandmarkId::from(id),
        host: TimeCamId::new(host, cam),
        direction: StereographicParam::project(&bearing),
        inv_dist: 1.0 / views[cam].depths[k],
        obs: obs
            .iter()
            .map(|&f| (TimeCamId::new(f, cam), views[cam].pixels[k]))
            .collect(),
    }
}

fn keyframe_stamp(
    t_ns: FrameId,
    window: &[(FrameId, Se3<f64>)],
    landmarks: Vec<StampLandmark>,
    keypoints: crate::estimator::Keypoints,
    images: Option<&KeyframeImages>,
) -> MapStamp {
    MapStamp::Keyframe(Box::new(KeyframeStamp {
        t_ns,
        keyframe_poses: window.iter().copied().collect(),
        landmarks,
        keypoints,
        images: images.cloned().map(ImageSlot::new),
    }))
}

fn map_config(max_keyframes: usize) -> MapConfig {
    MapConfig {
        enable_culling: true,
        max_redundancy_score: 2.0,
        redundancy_exclude_hosted: false,
        max_keyframes,
        batch_cull_size: 2,
        high_covisibility: 10,
    }
}

/// The back end on `calib` with these map knobs and `config`'s loop closing.
fn backend_on(calib: &Calibration<f64>, config: &VioConfig, map: MapConfig) -> MapBackend {
    let rig = crate::backend::BackendRig::new(calib).unwrap();
    let mut backend = MapBackend::new(config, rig, crate::Backend::Cpu);
    backend.config = map;
    backend
}

fn scene(
    cameras: usize,
    drift: f64,
    fuse_landmarks: bool,
    loops: LoopClosingConfig,
    max_keyframes: usize,
) -> Scene {
    scene_with(Setup {
        fuse_landmarks,
        max_keyframes,
        ..Setup::new(cameras, drift, loops)
    })
}

/// A copy of `stamp` with its own image slot: a clone shares the slot, and the
/// first step that reads it takes the images out.
fn copy_stamp(stamp: &mut MapStamp) -> MapStamp {
    let mut copy = stamp.clone();
    if let (MapStamp::Keyframe(original), MapStamp::Keyframe(copied)) = (&mut *stamp, &mut copy)
        && let Some(images) = original.images.as_ref().and_then(ImageSlot::take)
    {
        original.images = Some(ImageSlot::new(images.clone()));
        copied.images = Some(ImageSlot::new(images));
    }
    copy
}

fn scene_with(setup: Setup) -> Scene {
    let Setup {
        cameras,
        drift,
        fuse_landmarks,
        loops,
        max_keyframes,
        seen,
        ..
    } = setup.clone();
    let calib = calibration(cameras);
    let mut rng = Rng::new(7 + cameras as u64);
    let views: Vec<View> = (0..cameras).map(|_| view(&mut rng)).collect();
    let images = KeyframeImages(Arc::new(views.iter().map(|v| v.image.clone()).collect()));
    let config = VioConfig {
        backend_loop_closing: loops,
        ..VioConfig::default()
    };
    let mut backend = backend_on(&calib, &config, map_config(max_keyframes));
    let mut stamps: Vec<MapStamp> = Vec::new();
    let mut record = |backend: &mut MapBackend, mut stamp: MapStamp| {
        stamps.push(copy_stamp(&mut stamp));
        backend.step(stamp);
    };
    let tracked = |offset: u64, count: usize| -> crate::estimator::Keypoints {
        (0..cameras)
            .map(|cam| {
                (0..count)
                    .map(|k| (keypoint_id(cam, k, offset), views[cam].pixels[k]))
                    .collect()
            })
            .collect()
    };
    let identity = Se3::identity();
    // A keyframe and its neighbour, out of the window, hosting one landmark per
    // point (`shifted` for those C does not track: 4 px off).
    let mut pair = |(first, second): (FrameId, FrameId), offset: u64, shifted: bool| {
        record(
            &mut backend,
            keyframe_stamp(
                first,
                &[(first, identity)],
                vec![],
                tracked(offset, POINTS),
                Some(&images),
            ),
        );
        assert!(backend.query(first).is_empty());
        let island: Vec<StampLandmark> = (0..cameras)
            .flat_map(|cam| (0..POINTS).map(move |k| (cam, k)))
            .map(|(cam, k)| {
                let off: f32 = match k {
                    k if k < setup.wrong => 30.0,
                    k if k >= seen && shifted => 4.0,
                    _ => 0.0,
                };
                let id = keypoint_id(cam, k, offset);
                placed(&views, &calib, (cam, k), id, first, &[first, second], off)
            })
            .collect();
        record(
            &mut backend,
            keyframe_stamp(
                second,
                &[(first, identity), (second, identity)],
                island,
                tracked(offset, POINTS),
                Some(&images),
            ),
        );
        assert!(
            backend.query(second).is_empty(),
            "too recent to close a loop with {first}"
        );
        record(
            &mut backend,
            MapStamp::Marginalized(MargStamp {
                keyframe_poses: [(first, identity), (second, identity)]
                    .into_iter()
                    .collect(),
                landmarks: vec![],
            }),
        );
    };
    if setup.decoy {
        pair((E, E2), 50_000, true);
    }
    pair((A, A2), 0, setup.shifted_unseen);
    let drifted = Se3::new(So3::identity(), Vector3::new(drift, 0.0, 0.0));
    let mut window = vec![(C, drifted)];
    let mut current = Vec::new();
    if fuse_landmarks {
        window.insert(0, (D, drifted));
        record(
            &mut backend,
            keyframe_stamp(D, &[(D, drifted)], vec![], tracked(100_000, seen), None),
        );
        current = (0..cameras)
            .flat_map(|cam| (0..10).map(move |k| (cam, k)))
            .map(|(cam, k)| {
                landmark(
                    &views,
                    &calib,
                    cam,
                    k,
                    keypoint_id(cam, k, 100_000),
                    D,
                    &[D, C],
                )
            })
            .collect();
    }
    record(
        &mut backend,
        keyframe_stamp(C, &window, current, tracked(100_000, seen), Some(&images)),
    );
    Scene {
        views,
        backend,
        stamps,
    }
}

fn hand_bow(entries: &[(FrameId, u32, &[[u64; 4]])]) -> HashBow {
    let mut bow = HashBow::new(16).unwrap();
    for &(frame, cam, words) in entries {
        let bag = bow.bag(words);
        bow.insert(FrameCamId { frame, cam }, &bag);
    }
    bow
}

/// A descriptor whose 16-bit HashBoW word is `word`.
fn with_word(word: u16) -> [u64; 4] {
    // The first 16 word bits of `hash_bow.rs`.
    const WORD_BITS: [u8; 16] = [
        170, 215, 41, 38, 96, 172, 52, 1, 182, 89, 234, 98, 217, 73, 195, 113,
    ];
    let mut d = [0u64; 4];
    for (i, &bit) in WORD_BITS.iter().enumerate() {
        if word >> i & 1 == 1 {
            d[usize::from(bit) / 64] |= 1 << (bit % 64);
        }
    }
    d
}

#[test]
fn retrieval_sorts_by_score_then_id_and_drops_recent_and_weak_keyframes() {
    // B3: RootSLAM returns at most N unsorted hits and validates the first.
    let (w1, w2, w3) = (with_word(1), with_word(2), with_word(3));
    let bow = hand_bow(&[
        (9 * SECOND, 0, &[w1, w2]),
        (2 * SECOND, 0, &[w1, w2]),
        (5 * SECOND, 0, &[w1, w1, w1, w3]),
        (7 * SECOND, 0, &[w3]),
        (35 * SECOND, 0, &[w1, w1]),
    ]);
    let query = bow.bag(&[w1, w1]);
    let config = LoopClosingConfig::default();
    let found = retrieve(&bow, &[(0, query.clone())], 40 * SECOND, &config);
    // 5 s scores 0.75; 2 s and 9 s tie at 0.5 (id order); 7 s scores 0 (below
    // 0.02); 35 s is less than 20 s old.
    assert_eq!(found, vec![5 * SECOND, 2 * SECOND, 9 * SECOND]);
    let top = LoopClosingConfig {
        num_frames_to_match: 2,
        ..config
    };
    assert_eq!(
        retrieve(&bow, &[(0, query)], 40 * SECOND, &top),
        vec![5 * SECOND, 2 * SECOND]
    );
}

#[test]
fn retrieval_ranks_keyframes_by_their_best_camera_before_the_top_n() {
    // B4: a four-camera keyframe must not take every slot of the top N.
    let (w1, w2) = (with_word(1), with_word(2));
    let bow = hand_bow(&[
        (SECOND, 0, &[w1]),
        (SECOND, 1, &[w1]),
        (SECOND, 2, &[w1]),
        (SECOND, 3, &[w1]),
        (2 * SECOND, 0, &[w1, w2]),
        (3 * SECOND, 1, &[w1, w2, w2]),
    ]);
    let config = LoopClosingConfig {
        num_frames_to_match: 2,
        ..LoopClosingConfig::default()
    };
    let found = retrieve(&bow, &[(0, bow.bag(&[w1]))], 40 * SECOND, &config);
    assert_eq!(found, vec![SECOND, 2 * SECOND]);
}

#[test]
fn every_camera_queries_by_default_and_camera_0_only_under_parity() {
    // B5: the revisit is seen by camera 1 only.
    let (w1, w2) = (with_word(1), with_word(2));
    let bow = hand_bow(&[(SECOND, 1, &[w2])]);
    let bags = [(0, bow.bag(&[w1])), (1, bow.bag(&[w2]))];
    let all = LoopClosingConfig::default();
    assert_eq!(retrieve(&bow, &bags, 40 * SECOND, &all), vec![SECOND]);
    let cam0 = LoopClosingConfig {
        query_cameras: LoopQueryCameras::Cam0,
        ..all
    };
    assert!(retrieve(&bow, &bags, 40 * SECOND, &cam0).is_empty());
}

#[test]
fn matching_skips_undescribed_ids_keeps_one_match_per_keypoint_and_spots_shared_tracks() {
    let d = |seed: u64| {
        let mut rng = Rng::new(seed);
        Descriptor256([rng.next(), rng.next(), rng.next(), rng.next()])
    };
    let current: KeyframeDescriptors = vec![DescriptorSet::from_ordered([(1, d(1)), (2, d(2))])];
    // B7: an id without a descriptor is absent from the set, never an all-zero one.
    let island = [
        DescriptorSet::from_ordered([(5, d(1)), (6, d(2))]),
        DescriptorSet::from_ordered([(7, d(1))]),
    ];
    let matches = match_cameras(&current, &island, MatchConfig::default());
    assert_eq!(
        matches,
        vec![
            (0, KeypointId(1), 5),
            (0, KeypointId(2), 6),
            (0, KeypointId(1), 7)
        ]
    );
    assert_eq!(
        consolidate(matches),
        vec![(0, KeypointId(1), 5), (0, KeypointId(2), 6)]
    );
    let current_ids = BTreeSet::from([1, 2]);
    let other = [
        DescriptorSet::default(),
        DescriptorSet::from_ordered([(2, d(9))]),
    ];
    assert!(
        share_a_track(&current_ids, &other),
        "a shared track id means covisible"
    );
    assert!(!share_a_track(
        &current_ids,
        &[DescriptorSet::from_ordered([(3, d(1))])]
    ));
}

#[test]
fn a_candidate_that_shares_a_track_is_never_validated() {
    let mut s = scene(1, 0.05, false, LoopClosingConfig::default(), 100);
    // Give C one of A's track ids: covisible, so no loop.
    let mut keypoints = s.backend.map().keyframe(C).unwrap().keypoints.clone();
    keypoints[0][0].0 = keypoint_id(0, 0, 0);
    keypoints[0].sort_by_key(|(id, _)| *id);
    s.backend.map.upsert_keyframe(
        C,
        s.backend.map().keyframe(C).unwrap().pose,
        Some(keypoints),
    );
    assert!(s.backend.detect_loop(C).is_none());
    let counts = s.backend.loop_closing().unwrap().counts();
    assert!(counts.candidates >= 1 && counts.verified == 0, "{counts:?}");
}

#[test]
fn the_pnp_gate_takes_the_ratio_over_the_matches_fed_to_that_pnp() {
    // B1: RootSLAM's MGO13 loop "35 -> 14 inliers" passes its gate only
    // because 0.5 is cast to 0.
    assert!(passes_pnp_gate(14, 35, 10, 0.0));
    assert!(!passes_pnp_gate(14, 35, 10, 0.5));
    // B2: after the reprojection pass, 25 inliers of 60 fed matches fail at
    // 0.5 even though RootSLAM's denominator (the 40 island matches) passes them.
    assert!(!passes_pnp_gate(25, 60, 20, 0.5));
    assert!(passes_pnp_gate(25, 40, 20, 0.5));
    assert!(!passes_pnp_gate(9, 10, 10, 0.0), "the minimum still holds");
}

#[test]
fn drift_is_the_relative_transform_and_its_length_is_frame_free() {
    // B18: |t(T_map^-1 T_pnp)| = |R_map^T (t_pnp - t_map)| = |t_pnp - t_map|.
    let mut rng = Rng::new(3);
    let mut random = || (rng.below(2000) as f64 / 1000.0) - 1.0;
    for _ in 0..50 {
        let map = Se3::new(
            So3::exp(&Vector3::new(random(), random(), random())),
            Vector3::new(30.0 * random(), 30.0 * random(), random()),
        );
        let small = Se3::new(
            So3::exp(&(0.02 * Vector3::new(random(), random(), random()))),
            0.05 * Vector3::new(random(), random(), random()),
        );
        let pnp = map * small;
        let (metres, degrees) = drift(&map, &pnp);
        assert!((metres - small.translation.norm()).abs() < 1e-9);
        assert!((metres - (pnp.translation - map.translation).norm()).abs() < 1e-9);
        assert!((degrees - small.rotation.log().norm().to_degrees()).abs() < 1e-9);
    }
}

#[test]
fn fusion_folds_into_a_current_landmark_and_adds_an_observation_otherwise() {
    let calib = calibration(1);
    let mut rng = Rng::new(11);
    let views = vec![view(&mut rng)];
    let mut map = super::super::KeyframeMap::new();
    for id in [A, A2, D, C] {
        map.upsert_keyframe(id, Se3::identity(), None);
    }
    map.merge_landmarks(&[
        landmark(&views, &calib, 0, 0, KeypointId(1), A, &[A, A2]),
        landmark(&views, &calib, 0, 1, KeypointId(2), A, &[A, A2]),
        landmark(&views, &calib, 0, 2, KeypointId(3), A, &[A, A2]),
        landmark(&views, &calib, 0, 3, KeypointId(50), D, &[D, C]),
        landmark(&views, &calib, 0, 4, KeypointId(51), D, &[D, C]),
    ]);
    let m = |keypoint: Option<u64>, landmark: u64| LoopMatch {
        cam: 0,
        keypoint: keypoint.map(KeypointId),
        pixel: Vector2::new(1.0, 2.0),
        landmark: LandmarkId(landmark),
    };
    let plan = fusion_plan(
        &map,
        C,
        &[
            m(Some(50), 1), // a current landmark: fuse 1 into 50
            m(Some(60), 2), // a keypoint without a landmark: observe 2
            m(None, 3),     // redetected: observe 3 (B9: never fused by id)
            m(Some(50), 3), // 50 already took a fusion: skipped
            m(Some(70), 9), // the island landmark left the map: skipped
            m(Some(51), 3), // 3 is still free: fused into 51
        ],
    );
    let image = TimeCamId::new(C, 0);
    let pixel = Vector2::new(1.0, 2.0);
    assert_eq!(
        plan,
        vec![
            LoopFusion::Fuse {
                from: LandmarkId(1),
                into: LandmarkId(50)
            },
            LoopFusion::AddObs {
                landmark: LandmarkId(2),
                image,
                pixel
            },
            LoopFusion::AddObs {
                landmark: LandmarkId(3),
                image,
                pixel
            },
            LoopFusion::Fuse {
                from: LandmarkId(3),
                into: LandmarkId(51)
            },
        ]
    );
}

#[test]
fn a_revisit_is_verified_fused_and_linked_to_its_island() {
    for cameras in [1, 2] {
        let mut s = scene(cameras, 0.05, true, LoopClosingConfig::default(), 100);
        let before = s.backend.stats();
        let events = s.backend.query(C);
        assert_eq!(events.len(), 1, "{cameras} cameras");
        let loops = s.backend.loop_closing().unwrap();
        let event = loops.events()[0].clone();
        let detection = &event.detection;
        assert_eq!(
            (detection.current, detection.candidate),
            (C, A),
            "equal scores: the older id first"
        );
        assert_eq!(detection.island, vec![A, A2]);
        assert_eq!(
            detection.stages,
            LoopStages {
                initial_matches: SEEN * cameras,
                island_matches: SEEN * cameras,
                inliers: SEEN * cameras,
                // Every unmatched landmark redetected.
                reprojection_matches: POINTS * cameras,
                reprojection_inliers: POINTS * cameras,
            }
        );
        assert!(
            (event.drift_m - 0.05).abs() < 1e-6 && event.drift_deg < 1e-4,
            "{event:?}"
        );
        let fused = 10 * cameras;
        let observed = (POINTS - 10) * cameras;
        assert_eq!(
            event.outcome,
            LoopOutcome::Fused {
                landmarks_fused: fused,
                observations_added: observed
            }
        );
        let after = s.backend.stats();
        assert_eq!(after.landmarks, before.landmarks - fused);
        // A fusion moves the island landmark's two observations; an added one is new.
        assert_eq!(after.observations, before.observations + observed);
        assert_eq!(
            s.backend.graph().loop_closures(C).collect::<Vec<_>>(),
            vec![A]
        );
        assert!(s.backend.graph().weight(A, C).unwrap() >= observed as u32);
        // The fused landmark keeps the current, host-relative coordinates.
        let kept = s
            .backend
            .map()
            .landmark(LandmarkId::from(keypoint_id(0, 0, 100_000)))
            .unwrap();
        assert_eq!(kept.host, TimeCamId::new(D, 0));
        assert!(kept.obs.contains_key(&TimeCamId::new(A, 0)));
        assert!(
            s.backend
                .map()
                .landmark(LandmarkId::from(keypoint_id(0, 0, 0)))
                .is_none()
        );
        assert_eq!(loops.held_images(), 0, "images dropped after the query");
        assert_eq!(loops.counts().fused, 1);
        let snapshot = s.backend.snapshot();
        assert_eq!(snapshot.loop_counts.verified, 1);
        assert_eq!(snapshot.loops.len(), 1);
        assert!(snapshot.loops[0].fused);
        // The snapshot carries each loop's drift, so a run can report its largest (k1).
        assert_eq!(
            (snapshot.loops[0].drift_m, snapshot.loops[0].drift_deg),
            (event.drift_m, event.drift_deg)
        );
    }
}

#[test]
fn the_reprojection_pass_reads_the_image_and_leaves_the_keyframe_as_it_was() {
    // B8: RootSLAM writes redetected corners into the shared keypoints.
    let mut s = scene(2, 0.05, false, LoopClosingConfig::default(), 100);
    let keypoints = s.backend.map().keyframe(C).unwrap().keypoints.clone();
    let detection = s.backend.detect_loop(C).unwrap();
    let redetected: Vec<&LoopMatch> = detection
        .matches
        .iter()
        .filter(|m| m.keypoint.is_none())
        .collect();
    assert_eq!(redetected.len(), 2 * (POINTS - SEEN));
    for m in redetected {
        let k = (m.landmark.0 - 1 - 1000 * m.cam as u64) as usize;
        assert_eq!(m.pixel, s.views[m.cam].pixels[k], "found at the dot");
    }
    assert_eq!(s.backend.map().keyframe(C).unwrap().keypoints, keypoints);
    let off = LoopClosingConfig {
        reproject_landmarks: false,
        ..LoopClosingConfig::default()
    };
    let mut s = scene(2, 0.05, false, off, 100);
    let detection = s.backend.detect_loop(C).unwrap();
    assert!(detection.matches.iter().all(|m| m.keypoint.is_some()));
    assert_eq!(detection.stages.reprojection_matches, 0);
}

#[test]
fn with_the_pose_graph_off_a_large_drift_changes_no_map_entry() {
    let off = LoopClosingConfig {
        pgo: false,
        ..LoopClosingConfig::default()
    };
    let mut s = scene(1, 0.3, true, off, 100);
    let before = (s.backend.map().clone(), s.backend.graph().clone());
    s.backend.query(C);
    let event = &s.backend.loop_closing().unwrap().events()[0];
    assert_eq!(event.outcome, LoopOutcome::NeedsPgo);
    assert!((event.drift_m - 0.3).abs() < 1e-6);
    assert_eq!((s.backend.map().clone(), s.backend.graph().clone()), before);
    assert_eq!(s.backend.loop_closing().unwrap().counts().needs_pgo, 1);
}

#[test]
fn a_graph_the_solver_refuses_releases_the_loop_unapplied_and_records_it_once() {
    // Zero sigmas give an infinite information matrix, which the solver refuses.
    let refused = LoopClosingConfig {
        pgo_edge_sigmas: [0.0, 0.0],
        ..LoopClosingConfig::default()
    };
    let mut s = scene(1, 0.3, true, refused, 100);
    let before = (s.backend.map().clone(), s.backend.graph().clone());
    s.backend.query(C);
    let loops = s.backend.loop_closing().unwrap();
    assert_eq!(loops.events().len(), 1);
    assert_eq!(loops.events()[0].outcome, LoopOutcome::NeedsPgo);
    assert_eq!(loops.pgo().failures, 1);
    assert!(loops.pgo().runs.is_empty());
    assert_eq!((loops.counts().needs_pgo, loops.counts().optimized), (1, 0));
    assert_eq!((s.backend.map().clone(), s.backend.graph().clone()), before);
    assert_eq!(s.backend.snapshot().pgo_epoch, 0);
}

#[test]
fn realtime_and_offline_give_the_same_final_trajectory_through_a_pose_graph_optimization() {
    // f1: the large-drift scene's stamps through the runner in both modes; the
    // runner queries each new keyframe, and C's query runs the PGO.
    let loops = LoopClosingConfig::default();
    let mut scene = scene(1, 0.3, true, loops.clone(), 100);
    let calib = calibration(1);
    let config = VioConfig {
        backend_loop_closing: loops,
        ..VioConfig::default()
    };
    let mut run = |mode: crate::config::BackendMode| -> crate::backend::MapSnapshot {
        let backend = backend_on(&calib, &config, map_config(100));
        let mut runner = crate::backend::BackendRunner::new(backend, mode, 16).unwrap();
        for stamp in &mut scene.stamps {
            runner.submit(copy_stamp(stamp));
        }
        runner.flush();
        assert_eq!(runner.counters().queries_dropped, 0);
        crate::backend::MapSnapshot::clone(&runner.view().view)
    };
    let offline = run(crate::config::BackendMode::Offline);
    let realtime = run(crate::config::BackendMode::Realtime);
    assert_eq!(offline.pgo_epoch, 1, "one pose-graph optimization ran");
    assert_eq!(realtime, offline);
    // Each keyframe's raw pose, and a frame halfway to the next keyframe that
    // keeps the raw pose before it.
    let pose = crate::pose_to_xyzw;
    let mut frames: Vec<(i64, [f64; 7])> = Vec::new();
    for (row, (&t_ns, raw)) in offline
        .keyframe_t_ns
        .iter()
        .zip(&offline.keyframe_raw_poses)
        .enumerate()
    {
        frames.push((t_ns, pose(raw)));
        if let Some(&next) = offline.keyframe_t_ns.get(row + 1) {
            frames.push(((t_ns + next) / 2, pose(raw)));
        }
    }
    let final_offline = offline.reanchor(&frames).unwrap();
    assert_eq!(realtime.reanchor(&frames).unwrap(), final_offline);
    assert!(
        final_offline
            .iter()
            .zip(&frames)
            .any(|(corrected, (_, raw))| (corrected[0] - raw[0]).abs() > 0.1),
        "the PGO moved the island's frames"
    );
}

#[test]
fn the_global_ba_starts_from_the_map_after_loops_and_pgo() {
    // k0: `Vio::finish_map` is `runner.update(|b| b.finish(..))` with no flush
    // before it. The update must take every queued stamp, the revisit's query and
    // its PGO included, before the GBA reads the map; a GBA that ran first would
    // start from the raw poses.
    let loops = LoopClosingConfig::default();
    let mut scene = scene(1, 0.3, true, loops.clone(), 100);
    let calib = calibration(1);
    let config = VioConfig {
        backend_loop_closing: loops,
        ..VioConfig::default()
    };
    let mut runner_in = |mode: crate::config::BackendMode| {
        let backend = backend_on(&calib, &config, map_config(100));
        let mut runner = crate::backend::BackendRunner::new(backend, mode, 16).unwrap();
        for stamp in &mut scene.stamps {
            runner.submit(copy_stamp(stamp));
        }
        runner
    };
    // The map after loops and PGO, flushed on its own.
    let mut flushed = runner_in(crate::config::BackendMode::Offline);
    flushed.flush();
    let after_pgo = crate::backend::MapSnapshot::clone(&flushed.view().view);
    assert_eq!(after_pgo.pgo_epoch, 1, "one pose-graph optimization ran");
    assert_ne!(
        after_pgo.keyframe_poses, after_pgo.keyframe_raw_poses,
        "the PGO moved no keyframe"
    );
    for mode in [
        crate::config::BackendMode::Offline,
        crate::config::BackendMode::Realtime,
    ] {
        let mut runner = runner_in(mode);
        let result = runner.update(|backend| backend.finish(true, &config));
        assert!(result.gba.is_some(), "{mode:?}: the GBA ran");
        assert_eq!(result.keyframe_t_ns, after_pgo.keyframe_t_ns, "{mode:?}");
        assert_eq!(
            result.keyframe_poses_before_gba, after_pgo.keyframe_poses,
            "{mode:?}: the GBA did not start from the map after loops and PGO"
        );
        assert_eq!(
            runner.view().view.pgo_epoch,
            1,
            "{mode:?}: the PGO ran before the GBA"
        );
    }
}

#[test]
fn a_large_drift_runs_the_pose_graph_on_the_old_map_then_fuses() {
    let mut s = scene(1, 0.3, true, LoopClosingConfig::default(), 100);
    let t_i_c = calibration(1).t_i_c;
    let before = s.backend.map().clone();
    let kept = LandmarkId::from(keypoint_id(0, SEEN, 0));
    let kept_before = before.landmark_world(kept, &t_i_c).unwrap();
    let events = s.backend.query(C);
    assert!(
        matches!(
            events[..],
            [MapEvent::LoopVerified {
                pgo_ns: Some(_),
                ..
            }]
        ),
        "timed for the runner's counters: {events:?}"
    );
    let loops = s.backend.loop_closing().unwrap();
    let event = &loops.events()[0];
    assert_eq!(
        event.outcome,
        LoopOutcome::Optimized {
            landmarks_fused: 10,
            observations_added: POINTS - 10
        }
    );
    assert!((event.drift_m - 0.3).abs() < 1e-6);
    assert_eq!((loops.counts().needs_pgo, loops.counts().optimized), (1, 1));
    let map = s.backend.map();
    for id in [C, D] {
        assert_eq!(
            map.keyframe(id).unwrap().pose,
            before.keyframe(id).unwrap().pose,
            "the window is the gauge"
        );
    }
    for id in [A, A2] {
        let x = map.keyframe(id).unwrap().pose.translation.x;
        assert!(
            x > 0.15 && x < 0.35,
            "the island follows the current drift: {x}"
        );
    }
    let run = loops.pgo().runs[0];
    assert_eq!((run.current, run.candidate, run.moved), (C, A, 2));
    let host = map.keyframe(A).unwrap().pose * before.keyframe(A).unwrap().pose.inverse();
    let kept_after = map.landmark_world(kept, &t_i_c).unwrap();
    assert!(
        (kept_after - host * kept_before).norm() < 1e-9,
        "a landmark moves with its host"
    );
    assert_eq!(
        s.backend.graph().loop_closures(C).collect::<Vec<_>>(),
        vec![A]
    );
    let snapshot = s.backend.snapshot();
    assert_eq!(snapshot.pgo_epoch, 1);
    let corrected: Vec<FrameId> = snapshot
        .keyframe_t_ns
        .iter()
        .zip(
            snapshot
                .keyframe_poses
                .iter()
                .zip(&snapshot.keyframe_raw_poses),
        )
        .filter(|(_, (pose, raw))| pose != raw)
        .map(|(&id, _)| id)
        .collect();
    assert_eq!(corrected, vec![A, A2]);
    let row_a = snapshot
        .keyframe_t_ns
        .iter()
        .position(|&id| id == A)
        .unwrap();
    assert_eq!(snapshot.keyframe_raw_poses[row_a], Se3::identity());
    assert_eq!(
        snapshot.keyframe_poses[row_a],
        map.keyframe(A).unwrap().pose
    );
    assert_eq!(snapshot.pgo_runs, loops.pgo().runs);
    assert!(snapshot.loops[0].fused, "fused after the PGO");
}

#[test]
fn a_small_drift_runs_no_pose_graph() {
    let mut s = scene(1, 0.05, true, LoopClosingConfig::default(), 100);
    let events = s.backend.query(C);
    assert!(matches!(
        events[..],
        [MapEvent::LoopVerified { pgo_ns: None, .. }]
    ));
    let loops = s.backend.loop_closing().unwrap();
    assert!(matches!(
        loops.events()[0].outcome,
        LoopOutcome::Fused { .. }
    ));
    assert!(loops.pgo().runs.is_empty());
    let snapshot = s.backend.snapshot();
    assert_eq!(snapshot.pgo_epoch, 0);
    assert_eq!(snapshot.keyframe_raw_poses, snapshot.keyframe_poses);
}

#[test]
fn two_runs_of_a_large_drift_loop_give_the_same_pose_graph_and_map() {
    let run = || {
        let mut s = scene(2, 0.3, true, LoopClosingConfig::default(), 100);
        s.backend.query(C);
        let loops = s.backend.loop_closing().unwrap();
        (
            loops.events().to_vec(),
            loops.pgo().clone(),
            s.backend.map().clone(),
        )
    };
    let (first, second) = (run(), run());
    assert_eq!(first.1.runs.len(), 1);
    assert_eq!(first, second);
}

#[test]
fn the_parity_knobs_close_the_same_revisit() {
    let parity = LoopClosingConfig {
        query_cameras: LoopQueryCameras::Cam0,
        brief_tests: BriefTests::Raw,
        pnp_ransac_fixed_iterations: Some(35),
        pnp_inliers_ratio: 0.0,
        ..LoopClosingConfig::default()
    };
    let mut s = scene(2, 0.05, false, parity, 100);
    assert_eq!(s.backend.query(C).len(), 1);
    assert_eq!(
        s.backend.loop_closing().unwrap().events()[0]
            .detection
            .stages
            .reprojection_inliers,
        2 * POINTS
    );
}

#[test]
fn two_runs_give_the_same_loop_events_and_map() {
    let run = || {
        let mut s = scene(2, 0.05, true, LoopClosingConfig::default(), 100);
        s.backend.query(C);
        (
            s.backend.loop_closing().unwrap().events().to_vec(),
            s.backend.map().clone(),
        )
    };
    assert_eq!(run(), run());
}

#[test]
fn a_query_without_images_or_a_dropped_one_holds_nothing() {
    let mut s = scene(1, 0.05, false, LoopClosingConfig::default(), 100);
    assert_eq!(s.backend.loop_closing().unwrap().held_images(), 1);
    s.backend.drop_query(C);
    assert_eq!(s.backend.loop_closing().unwrap().held_images(), 0);
    assert!(s.backend.query(C).is_empty());
    assert_eq!(s.backend.loop_closing().unwrap().counts().without_images, 1);
}

#[test]
fn the_ratio_gate_reads_the_final_pnp_only() {
    // B1, lead decision (DESIGN §4): 20 of the 42 island matches are inliers
    // (0.48, as RootSLAM's MGO13 loops: median 0.42), and after the
    // reprojection pass the final PnP keeps all its matches.
    let setup = Setup {
        wrong: 22,
        ..Setup::new(1, 0.05, LoopClosingConfig::default())
    };
    let detection = scene_with(setup.clone()).backend.detect_loop(C).unwrap();
    assert_eq!(
        (detection.stages.island_matches, detection.stages.inliers),
        (SEEN, SEEN - 22)
    );
    assert_eq!(detection.stages.reprojection_inliers, POINTS - 22);
    // Without the reprojection pass the first PnP is the final one.
    let no_pass = LoopClosingConfig {
        reproject_landmarks: false,
        ..LoopClosingConfig::default()
    };
    let mut without = scene_with(Setup {
        loops: no_pass.clone(),
        ..setup.clone()
    });
    assert!(without.backend.detect_loop(C).is_none());
    let parity = LoopClosingConfig {
        pnp_inliers_ratio: 0.0,
        ..no_pass
    };
    assert!(
        scene_with(Setup {
            loops: parity,
            ..setup
        })
        .backend
        .detect_loop(C)
        .is_some()
    );
}

#[test]
fn the_final_ratio_divides_by_the_matches_fed_to_that_pnp() {
    // B2: all 20 island matches are inliers; the reprojection pass adds 40
    // dots whose landmarks sit 4 px off, so the final PnP keeps 20 of 60.
    // RootSLAM divides by the 20 island matches and would pass it.
    let setup = Setup {
        seen: 20,
        shifted_unseen: true,
        ..Setup::new(1, 0.05, tight(LoopClosingConfig::default()))
    };
    assert!(scene_with(setup.clone()).backend.detect_loop(C).is_none());
    let parity = tight(LoopClosingConfig {
        pnp_inliers_ratio: 0.0,
        ..LoopClosingConfig::default()
    });
    let detection = scene_with(Setup {
        loops: parity,
        ..setup
    })
    .backend
    .detect_loop(C)
    .unwrap();
    assert_eq!(
        detection.stages,
        LoopStages {
            initial_matches: 20,
            island_matches: 20,
            inliers: 20,
            reprojection_matches: POINTS,
            reprojection_inliers: 20,
        }
    );
}

#[test]
fn an_island_landmark_its_keyframe_never_described_matches_nothing() {
    // B7: A2 observes a landmark of D (an added observation, no descriptor in
    // A2), and C gets a keypoint whose descriptor is all zero. RootSLAM
    // default-inserts a zero descriptor for the landmark, and the two match.
    let mut s = scene(1, 0.05, true, LoopClosingConfig::default(), 100);
    let keypoints = s.backend.map().keyframe(C).unwrap().keypoints.clone();
    let loops = s.backend.loops.as_mut().unwrap();
    let images = loops.images.remove(&C).unwrap();
    loops.index(C, &keypoints, &images);
    let validate = |backend: &mut MapBackend| {
        let loops = backend.loops.as_ref().unwrap();
        let mut query = LoopQuery::new(C, &loops.descriptors[&C], &keypoints, &images);
        let initial = loops.initial_matches(&query, A).unwrap();
        backend.validate(&mut query, A, initial).unwrap()
    };
    let before = validate(&mut s.backend);
    let loops = s.backend.loops.as_mut().unwrap();
    let described = &mut loops.descriptors.get_mut(&C).unwrap()[0];
    let index = described.ids.binary_search(&9_999_999).unwrap_err();
    described.ids.insert(index, 9_999_999);
    described.values.insert(index, Descriptor256([0; 4]));
    let hosted = LandmarkId::from(keypoint_id(0, 0, 100_000));
    assert!(
        s.backend
            .map
            .add_observation(hosted, TimeCamId::new(A2, 0), Vector2::new(5.0, 5.0))
    );
    let after = validate(&mut s.backend);
    assert_eq!(after.stages, before.stages);
}

#[test]
fn a_candidate_that_failed_after_its_reprojection_pass_leaves_the_next_one_as_it_is() {
    // B8: RootSLAM writes a candidate's redetected corners into the current
    // keyframe, and the next candidate of the same query sees them.
    let setup = Setup {
        seen: 20,
        ..Setup::new(1, 0.05, tight(LoopClosingConfig::default()))
    };
    // Its island may differ (the spanning tree links A to E2); what it found must not.
    let found = |d: LoopDetection| (d.candidate, d.corrected_pose, d.matches, d.stages);
    let alone = scene_with(setup.clone()).backend.detect_loop(C).unwrap();
    let mut s = scene_with(Setup {
        decoy: true,
        ..setup.clone()
    });
    assert_eq!(s.backend.detect_loop(C).map(found), Some(found(alone)));
    assert_eq!(
        s.backend.loop_closing().unwrap().counts().candidates,
        3,
        "E and E2 failed first"
    );
    // The decoy reaches its reprojection pass: without the ratio gate it closes.
    let parity = tight(LoopClosingConfig {
        pnp_inliers_ratio: 0.0,
        ..LoopClosingConfig::default()
    });
    let decoy = scene_with(Setup {
        decoy: true,
        loops: parity,
        ..setup
    })
    .backend
    .detect_loop(C)
    .unwrap();
    assert_eq!(
        (decoy.candidate, decoy.stages.reprojection_matches),
        (E, POINTS)
    );
}

#[test]
fn queued_stamps_past_the_query_cap_lose_their_images_before_the_back_end_takes_them() {
    // Realtime, behind: only the newest `cap` stamps keep their images.
    let image = || {
        KeyframeImages(Arc::new(vec![
            Image::from_size_val([4, 4].into(), 0u16).unwrap(),
        ]))
    };
    let mut slots = crate::estimator::ImageSlots::new(2);
    let queued: Vec<ImageSlot> = (0..4).map(|_| slots.hold(image())).collect();
    let taken: Vec<bool> = queued.iter().map(|slot| slot.take().is_some()).collect();
    assert_eq!(taken, [false, false, true, true]);
    // The back end counts a stamp whose slot came empty, and its query runs without.
    let mut s = scene(1, 0.05, false, LoopClosingConfig::default(), 100);
    let loops = s.backend.loops.as_mut().unwrap();
    loops.hold_images(D, &queued[0]);
    assert_eq!(loops.counts().images_stripped, 1);
    assert_eq!(loops.held_images(), 1, "C's images only");
}
