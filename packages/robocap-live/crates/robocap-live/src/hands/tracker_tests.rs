//! The tracker's state machine with fake perception: the port of handtrack's `tests/test_tracker.py` for the ported code paths.
//!
//! The scene is Python's: four 636x480 pinhole cameras at the rig origin, yawed -0.9, -0.3, 0.3 and 0.9 rad, a still headset,
//! and the generic hand model with both hands held still in front. The fake detector reports the ground-truth circle on
//! scripted frames; the fake KeyNet returns the exact projected ground truth with a scripted presence. The fit is the real one.
//! The same expectations as the Python tests hold (camera counts, views, frames), which checks the geometry against Python too.
//! `tests/tracker_perception.rs` drives the tracker through the real estimator with heatmaps rendered from the ground truth.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use handfit::model::Step;
use handfit::nalgebra::{Rotation3, SMatrix, SVector};
use kornia_image::{Image, ImageSize};
use nalgebra::{Translation3, UnitQuaternion};

use super::*;
use crate::frame::RigCamera;
use crate::hands::model::identity_pose;
use crate::nets::{DetNetRaw, KeyNetRaw, NetFrame, NetsError};
use kornia_staging_sensors::CaptureMeta;

const YAWS: [f64; 4] = [-0.9, -0.3, 0.3, 0.9];
const WRISTS: [[f64; 3]; 2] = [[-0.3, 0.0, 0.3], [0.3, 0.0, 0.3]];
/// Keypoints inside each camera, per hand, in this scene (Python's `VISIBLE`).
const VISIBLE: [[usize; 4]; 2] = [[21, 21, 14, 0], [0, 15, 21, 21]];

fn rotation(axis: usize, angle: f64) -> Matrix3<f64> {
    let (c, s) = (angle.cos(), angle.sin());
    let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut m = Matrix3::identity();
    m[(i, i)] = c;
    m[(i, j)] = -s;
    m[(j, i)] = s;
    m[(j, j)] = c;
    m
}

/// Ground truth of the still scene in every camera.
struct Scene {
    rig: Rig,
    cameras: Vec<RigCameraModel>,
    landmarks: [[[f64; 3]; NUM_LANDMARKS]; 2],
    /// [camera][side]
    points_cam: Vec<[[Vector3<f64>; NUM_LANDMARKS]; 2]>,
    net_xy: Vec<[[[f64; 2]; NUM_LANDMARKS]; 2]>,
    visible: Vec<[usize; 2]>,
    circles: Vec<[[f64; 3]; 2]>,
}

fn scene() -> Result<Scene, HandsError> {
    let cameras: Vec<RigCamera> = YAWS
        .iter()
        .enumerate()
        .map(|(c, &yaw)| {
            let r = rotation(1, yaw).transpose();
            let mut cam_from_rig = [[0.0; 4]; 4];
            for (i, row) in cam_from_rig.iter_mut().enumerate() {
                for (k, value) in row.iter_mut().enumerate() {
                    *value = if i < 3 && k < 3 {
                        r[(i, k)]
                    } else if i == k {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
            RigCamera {
                name: format!("cam_0{c}"),
                width: 636,
                height: 480,
                cam_from_rig,
                focal: [240.0, 240.0],
                principal: [317.5, 239.5],
                fisheye62: None,
            }
        })
        .collect();
    let rig = Rig {
        cameras,
        source: "handtrack tests/test_tracker.py scene".into(),
        device: "test".into(),
    };
    let geometry: Vec<RigCameraModel> = rig
        .cameras
        .iter()
        .map(RigCameraModel::from_rig_camera)
        .collect::<Result<_, _>>()
        .map_err(|error| HandsError::Invalid(error.to_string()))?;
    let model = GenericHandModel::load()?;
    let poses: [Pose; 2] = std::array::from_fn(|side| Pose {
        rotation: rotation(0, 1.2),
        translation: Vector3::from_column_slice(&WRISTS[side]),
        angles: SVector::zeros(),
    });
    let landmarks: [[[f64; 3]; NUM_LANDMARKS]; 2] =
        std::array::from_fn(|side| landmarks_world(model.model(), &pose_f32(&poses[side]), side));
    let world = Matrix4::identity();
    let mut points_cam = Vec::new();
    let mut net_xy = Vec::new();
    let mut visible = Vec::new();
    let mut circles = Vec::new();
    for camera in &geometry {
        let points: [[Vector3<f64>; NUM_LANDMARKS]; 2] = std::array::from_fn(|side| {
            std::array::from_fn(|i| camera.fit.cam_from_world_point(&world, &landmarks[side][i]))
        });
        let pixels: [[Vector2<f64>; NUM_LANDMARKS]; 2] = std::array::from_fn(|side| {
            std::array::from_fn(|i| camera.fit.project(&points[side][i]))
        });
        let net: [[[f64; 2]; NUM_LANDMARKS]; 2] = std::array::from_fn(|side| {
            std::array::from_fn(|i| {
                camera
                    .net
                    .to_net(&Vector2::new(pixels[side][i][0], pixels[side][i][1]))
                    .into()
            })
        });
        visible.push(std::array::from_fn(|side| {
            (0..NUM_LANDMARKS)
                .filter(|&i| points[side][i][2] > 0.0 && camera.inside_image(&pixels[side][i]))
                .count()
        }));
        circles.push(std::array::from_fn(|side| {
            let front: Vec<[f64; 2]> = (0..NUM_LANDMARKS)
                .filter(|&i| points[side][i][2] > 0.0)
                .map(|i| net[side][i])
                .collect();
            min_enclosing_circle(&front).map_or([f64::NAN; 3], |c| c.to_array())
        }));
        points_cam.push(points);
        net_xy.push(net);
    }
    Ok(Scene {
        rig,
        cameras: geometry,
        landmarks,
        points_cam,
        net_xy,
        visible,
        circles,
    })
}

#[derive(Default)]
struct Log {
    frame: usize,
    detnet: Vec<(usize, usize)>,
    keynet: Vec<(usize, Vec<ViewRequest>)>,
}

type PresenceFn = Box<dyn Fn(usize, usize, usize, usize) -> f64 + Send>;
type KeypointsFn =
    Box<dyn Fn(usize, usize, &mut [[f64; 2]; NUM_LANDMARKS], &mut [f64; NUM_LANDMARKS]) + Send>;

/// Python's `FakeDetector` and `FakeKeyNet` in one.
struct Fake {
    scene: Arc<Scene>,
    /// (frame, camera, side) that DetNet reports.
    detections: HashSet<(usize, usize, usize)>,
    /// Presence per (frame, camera, side, KeyNet call of the frame); 0.9 by default.
    presence: PresenceFn,
    /// Edits the keypoints (net frame) and confidences of (camera, side).
    keypoints: Option<KeypointsFn>,
    log: Arc<Mutex<Log>>,
}

impl Perception for Fake {
    fn detect(
        &mut self,
        _nets: &mut dyn HandNets,
        cameras: &[usize],
        _small: &[&Luma],
    ) -> Result<Vec<Detections>, HandsError> {
        let mut log = self
            .log
            .lock()
            .map_err(|_| HandsError::Invalid("poisoned".into()))?;
        let frame = log.frame;
        Ok(cameras
            .iter()
            .map(|&camera| {
                log.detnet.push((frame, camera));
                Detections {
                    circle_net: self.scene.circles[camera].map(|c| c.map(|x| x as f32)),
                    probability: std::array::from_fn(|side| {
                        if self.detections.contains(&(frame, camera, side)) {
                            0.9
                        } else {
                            0.1
                        }
                    }),
                }
            })
            .collect())
    }

    fn estimate(
        &mut self,
        _nets: &mut dyn HandNets,
        _full: &[Option<&CameraFrame>; NUM_CAMERAS],
        _turned_180: &[bool; NUM_CAMERAS],
        _w: &Isometry3<f64>,
        views: &[ViewRequest],
    ) -> Result<(Vec<KeypointEstimate>, f64), HandsError> {
        let mut log = self
            .log
            .lock()
            .map_err(|_| HandsError::Invalid("poisoned".into()))?;
        let frame = log.frame;
        let call = log.keynet.iter().filter(|(f, _)| *f == frame).count();
        log.keynet.push((frame, views.to_vec()));
        let estimates = views
            .iter()
            .map(|view| {
                let camera = &self.scene.cameras[view.camera];
                let mut points_net = self.scene.net_xy[view.camera][view.side];
                let mut confidence = [1.0; NUM_LANDMARKS];
                if let Some(edit) = &self.keypoints {
                    edit(view.camera, view.side, &mut points_net, &mut confidence);
                }
                let distances: [f64; NUM_LANDMARKS] = std::array::from_fn(|i| {
                    self.scene.points_cam[view.camera][view.side][i].norm()
                });
                let mean = distances.iter().sum::<f64>() / NUM_LANDMARKS as f64;
                let points_px: [[f64; 2]; NUM_LANDMARKS] =
                    points_net.map(|p| camera.net.from_net(&Vector2::from(p)).into());
                KeypointEstimate {
                    points_net: points_net.map(|p| p.map(|x| x as f32)),
                    points_px: points_px.map(|p| p.map(|x| x as f32)),
                    d_rel_mm: distances.map(|d| ((d - mean) * 1000.0) as f32),
                    presence: (self.presence)(frame, view.camera, view.side, call) as f32,
                    confidence: confidence.map(|c| c as f32),
                    pinch: None,
                    usable: true,
                    crop: None,
                }
            })
            .collect();
        Ok((estimates, 0.0))
    }

    fn set_phi(&mut self, _phi: f64) {}
}

struct NoNets;

impl HandNets for NoNets {
    fn detnet(&mut self, _frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        Err(NetsError::Run {
            net: "detnet",
            message: "fake perception only".into(),
        })
    }
    fn keynet(
        &mut self,
        _crops: &[&[f32]],
        _keypoints: &[[f32; 63]],
    ) -> Result<Vec<KeyNetRaw>, NetsError> {
        Err(NetsError::Run {
            net: "keynet",
            message: "fake perception only".into(),
        })
    }
    fn describe(&self) -> String {
        "none".into()
    }
}

struct Setup {
    scene: Arc<Scene>,
    tracker: Tracker,
    log: Arc<Mutex<Log>>,
}

fn setup(
    detections: &[(usize, usize, usize)],
    presence: PresenceFn,
    keypoints: Option<KeypointsFn>,
    config: TrackerConfig,
) -> Result<Setup, HandsError> {
    setup_with(detections, presence, keypoints, config, test_hands())
}

/// The tests' `HandsConfig`: four cameras, a fixed scale, handtrack's views and round-robin DetNet.
fn test_hands() -> HandsConfig {
    HandsConfig {
        scale: ScaleMode::Fixed(1.0),
        cameras: vec![0, 1, 2, 3],
        max_views: 2,
        detnet_cameras: None,
        // One group per camera: round robin, handtrack's default.
        detnet_groups: 4,
        scale_wait: false,
        acquire_threads: 4,
    }
}

fn setup_with(
    detections: &[(usize, usize, usize)],
    presence: PresenceFn,
    keypoints: Option<KeypointsFn>,
    config: TrackerConfig,
    hands: HandsConfig,
) -> Result<Setup, HandsError> {
    let scene = Arc::new(scene()?);
    let log = Arc::new(Mutex::new(Log::default()));
    let fake = Fake {
        scene: scene.clone(),
        detections: detections.iter().copied().collect(),
        presence,
        keypoints,
        log: log.clone(),
    };
    let tracker = Tracker::new(&scene.rig, &hands, config, Box::new(fake))?;
    Ok(Setup {
        scene,
        tracker,
        log,
    })
}

fn always(value: f64) -> PresenceFn {
    Box::new(move |_, _, _, _| value)
}

fn identity() -> Isometry3<f64> {
    Isometry3::identity()
}

impl Setup {
    fn step(
        &mut self,
        frame: usize,
        world: &Isometry3<f64>,
    ) -> Result<HandFrameResult, HandsError> {
        let image: Luma = Arc::new(
            Image::new(
                ImageSize {
                    width: 1,
                    height: 1,
                },
                vec![0u8],
            )
            .map_err(|e| HandsError::Invalid(e.to_string()))?,
        );
        let images: [Option<&Luma>; NUM_CAMERAS] =
            std::array::from_fn(|c| (c < 4).then_some(&image));
        let camera_frame = CameraFrame {
            meta: CaptureMeta::default(),
            full: image.clone(),
        };
        let full: [Option<&CameraFrame>; NUM_CAMERAS] =
            std::array::from_fn(|c| (c < 4).then_some(&camera_frame));
        self.log
            .lock()
            .map_err(|_| HandsError::Invalid("poisoned".into()))?
            .frame = frame;
        let inputs = HandInputs {
            turned_180: [false; NUM_CAMERAS],
            index: frame as u64,
            t_ns: frame as i64 * 33_333_333,
            full,
            small: images,
        };
        self.tracker.track(&inputs, world, &mut NoNets)
    }

    fn run(&mut self, frames: usize) -> Result<Vec<HandFrameResult>, HandsError> {
        (0..frames)
            .map(|frame| self.step(frame, &identity()))
            .collect()
    }

    /// The cameras KeyNet ran on for a hand in a frame, sorted (Python `_views`).
    fn views(&self, frame: usize, side: usize) -> Vec<usize> {
        self.log.lock().map_or_else(
            |_| Vec::new(),
            |log| {
                let mut cameras: Vec<usize> = log
                    .keynet
                    .iter()
                    .filter(|(f, _)| *f == frame)
                    .flat_map(|(_, views)| {
                        views.iter().filter(|v| v.side == side).map(|v| v.camera)
                    })
                    .collect();
                cameras.sort_unstable();
                cameras
            },
        )
    }

    fn keynet_calls(&self) -> Vec<(usize, Vec<ViewRequest>)> {
        self.log
            .lock()
            .map_or_else(|_| Vec::new(), |log| log.keynet.clone())
    }

    fn detnet_calls(&self) -> Vec<(usize, usize)> {
        self.log
            .lock()
            .map_or_else(|_| Vec::new(), |log| log.detnet.clone())
    }
}

fn reported(result: &HandFrameResult) -> [bool; 2] {
    [result.hands[0].reported, result.hands[1].reported]
}

fn max_landmark_error(result: &HandFrameResult, scene: &Scene, side: usize) -> f64 {
    result.hands[side]
        .landmarks_world
        .map_or(f64::INFINITY, |landmarks| {
            landmarks
                .iter()
                .zip(&scene.landmarks[side])
                .flat_map(|(p, q)| (0..3).map(move |k| (p[k] - q[k]).abs()))
                .fold(0.0, f64::max)
        })
}

#[test]
fn scene_counts_match_python() -> Result<(), HandsError> {
    let scene = scene()?;
    for (side, expected) in VISIBLE.iter().enumerate() {
        let counts: Vec<usize> = (0..4).map(|c| scene.visible[c][side]).collect();
        assert_eq!(&counts, expected);
    }
    Ok(())
}

#[test]
fn detnet_runs_round_robin_while_a_hand_is_untracked() -> Result<(), HandsError> {
    let mut setup = setup(&[], always(0.9), None, TrackerConfig::handtrack_default())?;
    let results = setup.run(6)?;
    assert_eq!(
        setup.detnet_calls(),
        vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (5, 1)]
    );
    assert_eq!(
        results.iter().map(|r| r.detnet_camera).collect::<Vec<_>>(),
        [0, 1, 2, 3, 0, 1].map(Some).to_vec()
    );
    assert!(setup.keynet_calls().is_empty());
    assert!(
        results
            .iter()
            .all(|r| !r.hands[0].tracked && !r.hands[1].tracked)
    );
    Ok(())
}

#[test]
fn acquisition_then_stereo_tracking_then_no_detnet() -> Result<(), HandsError> {
    // The left hand is found in cam_00 on frame 0; the right hand in cam_01 on frame 1 (its round-robin camera).
    let mut setup = setup(
        &[(0, 0, LEFT), (1, 1, RIGHT)],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
    )?;
    let results = setup.run(4)?;
    assert_eq!(reported(&results[0]), [true, false]);
    assert!(results[0].hands[LEFT].detnet_circle.is_some());
    assert_eq!(setup.views(0, LEFT), vec![0]);
    let calls = setup.keynet_calls();
    assert!(
        calls[0]
            .1
            .iter()
            .all(|v| v.planning_pose_landmarks_world.is_none()),
        "an acquisition runs KeyNet with no planning pose"
    );
    assert!(max_landmark_error(&results[0], &setup.scene, LEFT) < 2e-3);

    // Frame 1: the left hand's views come from its pose; KeyNet runs on its two best views.
    assert_eq!(reported(&results[1]), [true, true]);
    assert_eq!(setup.views(1, LEFT), vec![0, 1]);
    assert_eq!(setup.views(1, RIGHT), vec![1]);
    let frame_one: Vec<&ViewRequest> = calls
        .iter()
        .filter(|(f, _)| *f == 1)
        .flat_map(|(_, v)| v.iter())
        .collect();
    assert!(
        frame_one
            .iter()
            .filter(|v| v.side == LEFT)
            .all(|v| v.planning_pose_landmarks_world.is_some()),
        "a tracked hand plans from its pose"
    );
    assert_eq!(results[1].detnet_camera, Some(1));

    // Frame 2 on: both hands tracked, no DetNet; the right hand in its two best views.
    assert_eq!(results[2].detnet_camera, None);
    assert_eq!(results[3].detnet_camera, None);
    assert_eq!(setup.detnet_calls(), vec![(0, 0), (1, 1)]);
    assert_eq!(setup.views(2, RIGHT), vec![2, 3]);
    assert_eq!(setup.views(3, LEFT), vec![0, 1]);
    for side in [LEFT, RIGHT] {
        assert!(
            max_landmark_error(&results[3], &setup.scene, side) < 1e-3,
            "side {side}"
        );
    }
    Ok(())
}

#[test]
fn low_presence_drops_the_view_then_the_track() -> Result<(), HandsError> {
    let presence: PresenceFn = Box::new(|frame, camera, side, _| match (side, frame, camera) {
        (LEFT, 2, 1) => 0.3, // one view below 0.5: it stays out of the fit
        (LEFT, 3, _) => 0.2, // every view below 0.5: the track ends
        _ => 0.9,
    });
    let mut setup = setup(
        &[(0, 0, LEFT)],
        presence,
        None,
        TrackerConfig::handtrack_default(),
    )?;
    let results = setup.run(5)?;
    assert_eq!(results[2].hands[LEFT].fitted_views().count(), 1);
    assert!(results[2].hands[LEFT].reported);
    assert_eq!(setup.views(3, LEFT), vec![0, 1]);
    assert!(!results[3].hands[LEFT].tracked && results[3].hands[LEFT].landmarks_world.is_none());
    // Dropped: no tracked views on the next frame, and DetNet looks for it again (cam_00 was frame 0; frames 1-3 kept cycling).
    assert!(
        setup
            .keynet_calls()
            .iter()
            .filter(|(f, _)| *f == 4)
            .all(|(_, views)| views
                .iter()
                .all(|v| v.side != LEFT || v.planning_pose_landmarks_world.is_none()))
    );
    assert_eq!(
        results.iter().map(|r| r.detnet_camera).collect::<Vec<_>>(),
        [0, 1, 2, 3, 0].map(Some).to_vec()
    );
    Ok(())
}

#[test]
fn end_on_view_rejection_drops_a_track_left_with_one_view() -> Result<(), HandsError> {
    let presence: PresenceFn = Box::new(|frame, camera, side, _| {
        if side == LEFT && frame == 2 && camera == 1 {
            0.3
        } else {
            0.9
        }
    });
    let config = TrackerConfig {
        end_on_view_rejection: true,
        ..TrackerConfig::handtrack_default()
    };
    let mut setup = setup(&[(0, 0, LEFT)], presence, None, config)?;
    let results = setup.run(4)?;
    assert!(results[1].hands[LEFT].reported && !results[2].hands[LEFT].tracked); // the one-view frame ends the track
    assert!(setup.views(3, LEFT).is_empty()); // DetNet looks for it again (cam_03 does not report it)
    Ok(())
}

#[test]
fn rejection_patience_two_keeps_a_track_through_one_rejected_frame() -> Result<(), HandsError> {
    let presence: PresenceFn = Box::new(|frame, camera, side, _| {
        if side == LEFT && frame == 2 && camera == 1 {
            0.3
        } else {
            0.9
        }
    });
    let config = TrackerConfig {
        end_on_view_rejection: true,
        rejection_patience: 2,
        ..TrackerConfig::handtrack_default()
    };
    let mut setup = setup(&[(0, 0, LEFT)], presence, None, config)?;
    let results = setup.run(4)?;
    assert!(results.iter().all(|r| r.hands[LEFT].reported));
    assert_eq!(results[2].hands[LEFT].fitted_views().count(), 1);
    Ok(())
}

#[test]
fn a_detection_that_keynet_rejects_is_not_tracked() -> Result<(), HandsError> {
    let mut setup = setup(
        &[(0, 0, LEFT)],
        always(0.1),
        None,
        TrackerConfig::handtrack_default(),
    )?;
    let results = setup.run(2)?;
    assert!(results[0].hands[LEFT].detnet_circle.is_some());
    assert!(!results[0].hands[LEFT].tracked && !results[0].hands[RIGHT].tracked);
    assert_eq!(results[1].detnet_camera, Some(1));
    Ok(())
}

#[test]
fn a_lost_headset_pose_resets_the_tracks() -> Result<(), HandsError> {
    let mut setup = setup(
        &[(0, 0, LEFT)],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
    )?;
    assert!(setup.step(0, &identity())?.hands[LEFT].tracked);
    let lost = Isometry3::from_parts(
        Translation3::new(f64::NAN, 0.0, 0.0),
        UnitQuaternion::identity(),
    );
    let result = setup.step(1, &lost)?;
    assert!(
        !result.hands[LEFT].tracked
            && !result.hands[RIGHT].tracked
            && result.detnet_camera.is_none()
    );
    assert_eq!(setup.step(2, &identity())?.detnet_camera, Some(1));
    Ok(())
}

#[test]
fn a_hand_outside_every_image_is_dropped_without_keynet() -> Result<(), HandsError> {
    let mut setup = setup(
        &[(0, 0, LEFT)],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
    )?;
    assert!(setup.step(0, &identity())?.hands[LEFT].tracked);
    // The headset looks the other way: the hand is behind every camera.
    let turned = Isometry3::from_parts(
        Translation3::identity(),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation(
            1,
            std::f64::consts::PI,
        ))),
    );
    let away = setup.step(1, &turned)?;
    assert!(!away.hands[LEFT].tracked);
    assert!(setup.views(1, LEFT).is_empty());
    assert_eq!(
        away.detnet_camera,
        Some(1),
        "the right hand was still untracked, so DetNet ran; the left hand is looked for from frame 2"
    );
    Ok(())
}

#[test]
fn a_keypoint_with_an_empty_heatmap_is_left_out_of_the_fit() -> Result<(), HandsError> {
    let dead_wrist: KeypointsFn = Box::new(|_, _, points, confidence| {
        points[5] = [0.0, 0.0]; // an empty wrist heatmap decodes to the crop corner
        confidence[5] = 0.0;
    });
    let mut setup = setup(
        &[(0, 0, LEFT)],
        always(0.9),
        Some(dead_wrist),
        TrackerConfig::handtrack_default(),
    )?;
    let results = setup.run(3)?;
    assert!(results[2].hands[LEFT].reported);
    assert!(max_landmark_error(&results[2], &setup.scene, LEFT) < 3e-3);
    // The weights themselves: the dead keypoint weighs 0, the others 1.
    let estimate = fake_estimate(
        &setup.scene,
        0,
        LEFT,
        std::array::from_fn(|i| if i == 5 { 0.0 } else { 1.0 }),
    );
    let view = ViewRequest {
        camera: 0,
        side: LEFT,
        planning_pose_landmarks_world: None,
        circle_net: None,
        source: CropSource::DetNet,
    };
    let seen = setup.tracker.seen(&view, &estimate, &Matrix4::identity());
    assert_eq!(seen.weights[5], 0.0);
    assert_eq!(seen.weights.iter().sum::<f64>(), 20.0);
    Ok(())
}

/// KeyNet's answer for the scene's true keypoints of (camera, side), at presence 0.9, with these heatmap confidences.
fn fake_estimate(
    scene: &Scene,
    camera: usize,
    side: usize,
    confidence: [f32; NUM_LANDMARKS],
) -> KeypointEstimate {
    let points_net = scene.net_xy[camera][side];
    let points_px: [[f64; 2]; NUM_LANDMARKS] =
        points_net.map(|p| scene.cameras[camera].net.from_net(&Vector2::from(p)).into());
    KeypointEstimate {
        points_net: points_net.map(|p| p.map(|x| x as f32)),
        points_px: points_px.map(|p| p.map(|x| x as f32)),
        d_rel_mm: [0.0; NUM_LANDMARKS],
        presence: 0.9,
        confidence,
        pinch: None,
        usable: true,
        crop: None,
    }
}

#[test]
fn mask_out_of_image_zeroes_keypoints_the_planning_pose_puts_outside() -> Result<(), HandsError> {
    let config = TrackerConfig {
        mask_out_of_image: true,
        ..TrackerConfig::handtrack_default()
    };
    let setup = setup(&[], always(0.9), None, config)?;
    // The left hand in cam_02 has 14 of its 21 keypoints inside [0, W) x [0, H); the mask keeps pixel centres in
    // [-0.5, W - 0.5), which also holds the keypoint at x = -0.017 (handtrack's `_clear` gives 15 too).
    let pose = Pose {
        rotation: rotation(0, 1.2),
        translation: Vector3::from_column_slice(&WRISTS[LEFT]),
        angles: SVector::zeros(),
    };
    let estimate = fake_estimate(&setup.scene, 2, LEFT, [1.0; NUM_LANDMARKS]);
    let landmarks = landmarks_world(&setup.tracker.model, &pose, LEFT);
    let view = ViewRequest {
        camera: 2,
        side: LEFT,
        planning_pose_landmarks_world: Some(landmarks),
        circle_net: None,
        source: CropSource::Pose,
    };
    let seen = setup.tracker.seen(&view, &estimate, &Matrix4::identity());
    assert_eq!(seen.weights.iter().sum::<f64>(), 15.0);
    Ok(())
}

#[test]
fn a_fit_out_of_reach_ends_the_track() -> Result<(), HandsError> {
    let config = TrackerConfig {
        max_reach_m: 0.2,
        ..TrackerConfig::handtrack_default()
    }; // the wrists are 0.42 m from the headset
    let mut setup = setup(&[(0, 0, LEFT)], always(0.9), None, config)?;
    let results = setup.run(2)?;
    assert!(results[0].hands[LEFT].detnet_circle.is_some());
    assert!(!results[0].hands[LEFT].tracked);
    assert_eq!(results[1].detnet_camera, Some(1));
    Ok(())
}

#[test]
fn unconverged_acquisitions_are_not_tracked() -> Result<(), HandsError> {
    let configs = [
        TrackerConfig {
            min_keypoint_confidence: 2.0,
            ..TrackerConfig::handtrack_default()
        },
        TrackerConfig {
            fit: Config {
                init_iterations: 0,
                ..Config::default()
            },
            ..TrackerConfig::handtrack_default()
        },
    ];
    for config in configs {
        let mut setup = setup(&[(0, 0, LEFT), (1, 1, LEFT)], always(0.9), None, config)?;
        let results = setup.run(2)?;
        assert!(
            results
                .iter()
                .all(|r| !r.hands[LEFT].tracked && !r.hands[RIGHT].tracked)
        );
        assert!(setup.keynet_calls().iter().all(|(_, views)| {
            views
                .iter()
                .all(|v| v.planning_pose_landmarks_world.is_none())
        }));
    }
    Ok(())
}

#[test]
fn confirm_frames_delays_reporting_but_not_tracking() -> Result<(), HandsError> {
    for confirm in [0u32, 1, 3] {
        let config = TrackerConfig {
            confirm_frames: confirm,
            ..TrackerConfig::handtrack_default()
        };
        let mut setup = setup(&[(0, 0, LEFT)], always(0.9), None, config)?;
        let results = setup.run(5)?;
        assert!(results.iter().all(|r| r.hands[LEFT].tracked));
        let first = results.iter().position(|r| r.hands[LEFT].reported);
        assert_eq!(first, Some(confirm as usize), "confirm_frames {confirm}");
    }
    Ok(())
}

#[test]
fn the_robust_preset_tracks_the_fake_scene() -> Result<(), HandsError> {
    let mut setup = setup(&[(0, 0, LEFT)], always(0.9), None, TrackerConfig::robust())?;
    let results = setup.run(6)?;
    // confirm_frames 2: reported from the 3rd frame; the damped guess still converges onto the exact keypoints.
    assert_eq!(
        results
            .iter()
            .map(|r| r.hands[LEFT].reported)
            .collect::<Vec<_>>(),
        vec![false, false, true, true, true, true]
    );
    assert!(max_landmark_error(&results[5], &setup.scene, LEFT) < 2e-3);
    Ok(())
}

#[test]
fn the_robust_preset_recrops_an_acquisition_once() -> Result<(), HandsError> {
    let mut setup = setup(&[(0, 0, LEFT)], always(0.9), None, TrackerConfig::robust())?;
    setup.run(1)?;
    let calls = setup.keynet_calls();
    assert_eq!(calls.len(), 2, "the re-crop pass, then the fit's pass");
    let first = calls[0].1[0].circle_net;
    let second = calls[1].1[0]
        .circle_net
        .ok_or(HandsError::Invalid("no re-crop circle".into()))?;
    let expected = min_enclosing_circle(&setup.scene.net_xy[0][LEFT])
        .map(|c| c.to_array())
        .ok_or(HandsError::Invalid("no circle".into()))?;
    assert_eq!(
        first,
        Some(setup.scene.circles[0][LEFT].map(|x| x as f32)),
        "the first pass uses DetNet's circle"
    );
    // KeyNet's keypoints are float32 (as the estimator returns them): the enclosing circle matches to float32 precision.
    assert!(
        (0..3).all(|k| (f64::from(second[k]) - expected[k]).abs() < 1e-3),
        "the second pass encloses KeyNet's keypoints"
    );
    Ok(())
}

/// Deterministic standard normal samples (splitmix64 + Box-Muller).
fn normals(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            (-2.0 * uniform().max(1e-300).ln()).sqrt()
                * (2.0 * std::f64::consts::PI * uniform()).cos()
        })
        .collect()
}

#[test]
fn an_acquisition_whose_fit_leaves_a_large_residual_for_its_keypoints_size_is_rejected()
-> Result<(), HandsError> {
    for (noise, max_relative_rms, tracked) in
        [(0.0, 0.02, true), (0.25, 1e9, true), (0.25, 0.02, false)]
    {
        let jitter = normals(0, 4 * NUM_LANDMARKS * 2);
        // KeyNet's left-hand keypoints jittered by this share of their spread: a fit converges but cannot explain them.
        let edit: KeypointsFn = Box::new(move |camera, side, points, _| {
            if side != LEFT {
                return;
            }
            let mean =
                [0, 1].map(|k| points.iter().map(|p| p[k]).sum::<f64>() / NUM_LANDMARKS as f64);
            let spread = (points
                .iter()
                .map(|p| (p[0] - mean[0]).powi(2) + (p[1] - mean[1]).powi(2))
                .sum::<f64>()
                / NUM_LANDMARKS as f64)
                .sqrt();
            for (i, p) in points.iter_mut().enumerate() {
                p[0] += noise * spread * jitter[(camera * NUM_LANDMARKS + i) * 2];
                p[1] += noise * spread * jitter[(camera * NUM_LANDMARKS + i) * 2 + 1];
            }
        });
        let config = TrackerConfig {
            acquire_max_relative_rms: max_relative_rms,
            ..TrackerConfig::handtrack_default()
        };
        let mut setup = setup(&[(0, 0, LEFT)], always(0.9), Some(edit), config)?;
        let results = setup.run(1)?;
        assert_eq!(
            results[0].hands[LEFT].tracked, tracked,
            "noise {noise}, max relative rms {max_relative_rms}"
        );
    }
    Ok(())
}

#[test]
fn the_tracker_rejects_unsupported_view_counts() {
    for max_views in [0, 3] {
        let hands = HandsConfig {
            max_views,
            ..test_hands()
        };
        assert!(
            setup_with(
                &[],
                always(0.9),
                None,
                TrackerConfig::handtrack_default(),
                hands
            )
            .is_err(),
            "max_views {max_views}"
        );
    }
}

#[test]
fn extrapolation_is_damped_and_clamped() {
    let before = Pose {
        rotation: Matrix3::identity(),
        translation: Vector3::new(0.0, 0.0, 0.3),
        angles: SVector::repeat(0.1),
    };
    let previous = Pose {
        rotation: rotation(2, 0.2),
        translation: Vector3::new(0.4, 0.0, 0.3),
        angles: SVector::repeat(0.3),
    };
    let guess = extrapolate(&previous, &before, 0.5, Some(0.15));
    assert!(
        (guess.translation - Vector3::new(0.55, 0.0, 0.3)).norm() < 1e-12,
        "a 0.2 m step is clamped to 0.15 m"
    );
    assert!(
        (guess.angles[0] - 0.4).abs() < 1e-12,
        "joint angles move by half their velocity"
    );
    let angle =
        Rotation3::from_matrix_unchecked(guess.rotation * previous.rotation.transpose()).angle();
    assert!(
        (angle - 0.1).abs() < 1e-3,
        "half the rotation step: {angle}"
    );
    let constant = extrapolate(&previous, &before, 1.0, None);
    assert!((constant.translation - Vector3::new(0.8, 0.0, 0.3)).norm() < 1e-12);
}

#[test]
fn detnet_cycles_through_its_own_camera_subset() -> Result<(), HandsError> {
    let hands = HandsConfig {
        detnet_cameras: Some(vec![3, 1]),
        ..test_hands()
    };
    let mut setup = setup_with(
        &[],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
        hands,
    )?;
    setup.run(4)?;
    assert_eq!(setup.detnet_calls(), vec![(0, 1), (1, 3), (2, 1), (3, 3)]);
    Ok(())
}

#[test]
fn detnet_on_all_cameras_acquires_in_stereo() -> Result<(), HandsError> {
    // The left hand is reported by cam_00 and cam_01 on frame 0: one acquisition with two views.
    let hands = HandsConfig {
        detnet_groups: 1,
        ..test_hands()
    };
    let mut setup = setup_with(
        &[(0, 0, LEFT), (0, 1, LEFT), (0, 2, LEFT)],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
        hands,
    )?;
    let results = setup.run(2)?;
    assert_eq!(
        setup.detnet_calls(),
        vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3)
        ]
    );
    assert_eq!(
        setup.views(0, LEFT),
        vec![0, 1],
        "at most max_views of the reporting cameras, ties to the lower index"
    );
    assert_eq!(results[0].hands[LEFT].fitted_views().count(), 2);
    assert!(
        results[0].hands[LEFT].reported
            && max_landmark_error(&results[0], &setup.scene, LEFT) < 1e-3
    );
    assert_eq!(results[0].detnet_camera, Some(0));
    Ok(())
}

#[test]
fn a_detnet_camera_outside_the_rig_is_an_error() {
    let hands = HandsConfig {
        detnet_cameras: Some(vec![7]),
        ..test_hands()
    };
    assert!(
        setup_with(
            &[],
            always(0.9),
            None,
            TrackerConfig::handtrack_default(),
            hands
        )
        .is_err()
    );
}

#[test]
fn detnet_groups_alternate_interleaved_camera_halves() -> Result<(), HandsError> {
    let hands = HandsConfig {
        detnet_groups: 2,
        ..test_hands()
    };
    let mut two = setup_with(
        &[],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
        hands,
    )?;
    two.run(3)?;
    assert_eq!(
        two.detnet_calls(),
        vec![(0, 0), (0, 2), (1, 1), (1, 3), (2, 0), (2, 2)]
    );
    // As many groups as cameras is the round robin.
    let hands = HandsConfig {
        detnet_groups: 4,
        ..test_hands()
    };
    let mut four = setup_with(
        &[],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
        hands,
    )?;
    four.run(5)?;
    assert_eq!(
        four.detnet_calls(),
        vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 0)]
    );
    Ok(())
}

/// handfit's threaded cold fit (the tracker's acquisitions) is the sequential one bit for bit.
#[test]
fn the_parallel_acquisition_fit_is_handfits_bit_for_bit() -> Result<(), Box<dyn std::error::Error>>
{
    let generic = GenericHandModel::load()?;
    let model = generic.model();
    for (k, mirror) in [(0, 1.0), (1, -1.0), (2, 1.0)] {
        let pose = Pose {
            rotation: Rotation3::from_euler_angles(1.0 + 0.3 * k as f64, 0.2, -0.1 * k as f64)
                .into_inner(),
            translation: Vector3::new(0.05 * k as f64, 0.1, 0.4),
            ..identity_pose()
        };
        let points = model.landmarks(&pose, mirror, &Step::zeros(), None);
        let views: Vec<View> = [-0.05, 0.05]
            .iter()
            .take(1 + k % 2)
            .map(|&x| {
                let rotation = Rotation3::from_euler_angles(0.0, x, 0.0).into_inner();
                let translation = Vector3::new(x, 0.0, 0.0);
                let cam: Vec<Vector3<f64>> = (0..21)
                    .map(|i| rotation * points.fixed_rows::<3>(3 * i).into_owned() + translation)
                    .collect();
                let mean = cam.iter().map(|p| p.norm()).sum::<f64>() / 21.0;
                View {
                    rotation,
                    translation,
                    camera: handfit::residual::camera_model(&Vector2::repeat(500.0), &Vector2::new(320.0, 240.0), None).unwrap(),
                    pixels: SMatrix::from_fn(|i, c| {
                        500.0 * cam[i][c] / cam[i][2]
                            + if c == 0 { 320.0 } else { 240.0 }
                            + 0.7 * ((i * 7 + c) as f64).sin()
                    }),
                    weights: SVector::repeat(1.0),
                    distances: SVector::from_fn(|i, _| (cam[i].norm() - mean) * 1000.0),
                }
            })
            .collect();
        let config = Config::default();
        let sequential =
            handfit::cold::initial_pose(model, &config, mirror, &views, JacobianMode::Analytic)?;
        for threads in [2, 4, 7] {
            let parallel = handfit::cold::initial_pose_parallel(
                model,
                &config,
                mirror,
                &views,
                JacobianMode::Analytic,
                threads,
            )?;
            assert_eq!(parallel.pose.rotation, sequential.pose.rotation);
            assert_eq!(parallel.pose.translation, sequential.pose.translation);
            assert_eq!(parallel.pose.angles, sequential.pose.angles);
            assert_eq!(
                parallel.energies.map(f64::to_bits),
                sequential.energies.map(f64::to_bits)
            );
            assert_eq!(
                (
                    parallel.converged,
                    parallel.termination,
                    parallel.iterations
                ),
                (
                    sequential.converged,
                    sequential.termination,
                    sequential.iterations
                )
            );
        }
    }
    Ok(())
}

/// (camera, outcome) of a hand's KeyNet views in a frame, sorted by camera.
fn outcomes(result: &HandFrameResult, side: usize) -> Vec<(usize, ViewOutcome)> {
    let mut views: Vec<(usize, ViewOutcome)> = result.hands[side]
        .keynet_views
        .iter()
        .map(|v| (v.camera, v.outcome))
        .collect();
    views.sort_by_key(|&(camera, _)| camera);
    views
}

#[test]
fn every_detnet_answer_for_an_untracked_hand_is_reported_with_its_verdict() -> Result<(), HandsError>
{
    // DetNet on all four cameras: the left hand in cam_00 and cam_01 (0.9), nowhere else (0.1).
    let hands = HandsConfig {
        detnet_groups: 1,
        ..test_hands()
    };
    let mut setup = setup_with(
        &[(0, 0, LEFT), (0, 1, LEFT)],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
        hands,
    )?;
    let results = setup.run(2)?;
    let verdicts = |frame: usize, side: usize| -> Vec<(usize, f32, bool)> {
        results[frame].hands[side]
            .detnet_hits
            .iter()
            .map(|h| (h.camera, h.probability, h.accepted))
            .collect()
    };
    assert_eq!(
        verdicts(0, LEFT),
        vec![
            (0, 0.9, true),
            (1, 0.9, true),
            (2, 0.1, false),
            (3, 0.1, false)
        ]
    );
    assert_eq!(
        verdicts(0, RIGHT),
        vec![
            (0, 0.1, false),
            (1, 0.1, false),
            (2, 0.1, false),
            (3, 0.1, false)
        ]
    );
    assert_eq!(
        results[0].hands[LEFT].detnet_hits[0].circle,
        setup.scene.circles[0][LEFT].map(|x| x as f32),
        "DetNet's own circle, net frame"
    );
    // Frame 1: the left hand is tracked, so DetNet only looks for the right one.
    assert!(results[1].hands[LEFT].detnet_hits.is_empty());
    assert_eq!(verdicts(1, RIGHT).len(), 4);
    Ok(())
}

#[test]
fn keynet_views_carry_keypoints_confidences_presence_and_the_fitted_verdict()
-> Result<(), HandsError> {
    let edit: KeypointsFn = Box::new(|_, _, _, confidence| confidence[3] = 0.25);
    let mut setup = setup(
        &[(0, 0, LEFT)],
        always(0.9),
        Some(edit),
        TrackerConfig::handtrack_default(),
    )?;
    let results = setup.run(1)?;
    let views = &results[0].hands[LEFT].keynet_views;
    assert_eq!(views.len(), 1);
    let view = &views[0];
    assert_eq!(
        (view.camera, view.outcome, view.presence),
        (0, ViewOutcome::Fitted, 0.9)
    );
    assert_eq!((view.confidence[3], view.confidence[4]), (0.25, 1.0));
    let camera = &setup.scene.cameras[0];
    let expected = camera
        .net
        .from_net(&Vector2::from(setup.scene.net_xy[0][LEFT][0]));
    assert!(
        (f64::from(view.keypoints_px[0][0]) - expected[0]).abs() < 1e-3
            && (f64::from(view.keypoints_px[0][1]) - expected[1]).abs() < 1e-3
    );
    assert!(results[0].hands[RIGHT].keynet_views.is_empty());
    assert_eq!(
        results[0].world_from_rig,
        Some(identity()),
        "the result names the headset pose it used"
    );
    Ok(())
}

#[test]
fn a_view_below_the_presence_threshold_is_reported_as_low_presence() -> Result<(), HandsError> {
    let presence: PresenceFn = Box::new(|frame, camera, side, _| match (side, frame, camera) {
        (LEFT, 2, 1) => 0.3,
        (LEFT, 3, _) => 0.2,
        _ => 0.9,
    });
    let mut setup = setup(
        &[(0, 0, LEFT)],
        presence,
        None,
        TrackerConfig::handtrack_default(),
    )?;
    let results = setup.run(4)?;
    assert_eq!(
        outcomes(&results[2], LEFT),
        vec![(0, ViewOutcome::Fitted), (1, ViewOutcome::LowPresence)]
    );
    assert_eq!(
        outcomes(&results[3], LEFT),
        vec![(0, ViewOutcome::LowPresence), (1, ViewOutcome::LowPresence)]
    );
    assert!(!results[3].hands[LEFT].tracked);
    Ok(())
}

#[test]
fn a_good_view_dropped_with_its_rejected_pair_is_reported_as_lost_pair() -> Result<(), HandsError> {
    let presence: PresenceFn = Box::new(|frame, camera, side, _| {
        if side == LEFT && frame == 2 && camera == 1 {
            0.3
        } else {
            0.9
        }
    });
    let config = TrackerConfig {
        end_on_view_rejection: true,
        ..TrackerConfig::handtrack_default()
    };
    let mut setup = setup(&[(0, 0, LEFT)], presence, None, config)?;
    let results = setup.run(3)?;
    assert_eq!(
        outcomes(&results[2], LEFT),
        vec![(0, ViewOutcome::LostPair), (1, ViewOutcome::LowPresence)]
    );
    Ok(())
}

#[test]
fn an_acquisition_rejected_by_the_residual_gate_reports_its_rms_and_limit() -> Result<(), HandsError>
{
    let jitter = normals(0, 4 * NUM_LANDMARKS * 2);
    let edit: KeypointsFn = Box::new(move |camera, side, points, _| {
        if side != LEFT {
            return;
        }
        let mean = [0, 1].map(|k| points.iter().map(|p| p[k]).sum::<f64>() / NUM_LANDMARKS as f64);
        let spread = (points
            .iter()
            .map(|p| (p[0] - mean[0]).powi(2) + (p[1] - mean[1]).powi(2))
            .sum::<f64>()
            / NUM_LANDMARKS as f64)
            .sqrt();
        for (i, p) in points.iter_mut().enumerate() {
            p[0] += 0.25 * spread * jitter[(camera * NUM_LANDMARKS + i) * 2];
            p[1] += 0.25 * spread * jitter[(camera * NUM_LANDMARKS + i) * 2 + 1];
        }
    });
    let config = TrackerConfig {
        acquire_max_relative_rms: 0.02,
        ..TrackerConfig::handtrack_default()
    };
    let mut setup = setup(&[(0, 0, LEFT)], always(0.9), Some(edit), config)?;
    let results = setup.run(1)?;
    let views = outcomes(&results[0], LEFT);
    assert_eq!(views.len(), 1);
    let ViewOutcome::FitResidual { rms_px, limit_px } = views[0].1 else {
        panic!("expected a residual rejection, got {:?}", views[0].1)
    };
    assert!(
        limit_px > 0.0 && rms_px > limit_px,
        "rms {rms_px} px, limit {limit_px} px"
    );
    assert!(!results[0].hands[LEFT].tracked);
    Ok(())
}

#[test]
fn a_fit_out_of_reach_is_reported_as_failed() -> Result<(), HandsError> {
    let config = TrackerConfig {
        max_reach_m: 0.2,
        ..TrackerConfig::handtrack_default()
    };
    let mut setup = setup(&[(0, 0, LEFT)], always(0.9), None, config)?;
    let results = setup.run(1)?;
    assert_eq!(
        outcomes(&results[0], LEFT),
        vec![(0, ViewOutcome::FitFailed)]
    );
    Ok(())
}

#[test]
fn keynet_views_name_where_their_crop_came_from_and_carry_relative_depths() -> Result<(), HandsError>
{
    let mut plain = setup(
        &[(0, 0, LEFT)],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
    )?;
    let results = plain.run(2)?;
    let first = &results[0].hands[LEFT].keynet_views[0];
    assert_eq!(
        first.crop_source,
        CropSource::DetNet,
        "an acquisition crops around DetNet's circle"
    );
    let distances: [f64; NUM_LANDMARKS] =
        std::array::from_fn(|i| plain.scene.points_cam[0][LEFT][i].norm());
    let mean = distances.iter().sum::<f64>() / NUM_LANDMARKS as f64;
    assert!(
        (0..NUM_LANDMARKS)
            .all(|i| (f64::from(first.d_rel_mm[i]) - (distances[i] - mean) * 1000.0).abs() < 1e-3)
    );
    assert!(
        results[1].hands[LEFT]
            .keynet_views
            .iter()
            .all(|v| v.crop_source == CropSource::Pose),
        "a tracked hand crops around its predicted pose"
    );
    let mut robust = setup(&[(0, 0, LEFT)], always(0.9), None, TrackerConfig::robust())?;
    let results = robust.run(1)?;
    assert_eq!(
        results[0].hands[LEFT].keynet_views[0].crop_source,
        CropSource::Recrop,
        "the robust preset re-crops around KeyNet's first answer"
    );
    Ok(())
}

#[test]
fn a_tracked_hand_reports_the_predicted_pose_its_views_were_planned_from() -> Result<(), HandsError>
{
    let mut setup = setup(
        &[(0, 0, LEFT)],
        always(0.9),
        None,
        TrackerConfig::handtrack_default(),
    )?;
    let results = setup.run(2)?;
    assert!(
        results[0].hands[LEFT].predicted_landmarks_world.is_none(),
        "an acquisition has no prediction"
    );
    let calls = setup.keynet_calls();
    let planned = calls
        .iter()
        .filter(|(f, _)| *f == 1)
        .flat_map(|(_, views)| views.iter())
        .find(|v| v.side == LEFT)
        .and_then(|v| v.planning_pose_landmarks_world);
    assert!(planned.is_some());
    assert_eq!(results[1].hands[LEFT].predicted_landmarks_world, planned);
    Ok(())
}
