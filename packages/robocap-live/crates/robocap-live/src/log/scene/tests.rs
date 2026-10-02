use super::*;
use crate::hands::perspective::CropCamera;
use crate::hands::{CropSource, DetNetHit, HandFrameResult, HandOutput, KeyNetView, ViewOutcome};
use nalgebra::Vector3;

fn pose(x: f64) -> Isometry3<f64> {
    Isometry3::translation(x, 0.0, 1.0)
}

/// The producer and one sink, as the worker drives them (with the last snapshot, which the record borrows).
#[derive(Default)]
struct Sink {
    state: RecordState,
    delivered: DeliveredState,
    snapshot: Option<SceneSnapshot>,
}

impl Sink {
    fn new(rig: &Rig, overlays: HandOverlays) -> Self {
        Self { state: RecordState::new(rig, overlays), ..Self::default() }
    }

    fn record(&mut self, t: i64, pose: Option<Isometry3<f64>>, status: &str, hands: Option<&HandFrameResult>, signals: &Signals) -> FrameRecord<'_> {
        let snapshot = self.snapshot.insert(self.state.prepare(t, pose, status, hands, signals));
        self.delivered.record(snapshot)
    }
}

#[test]
fn pixels_map_to_the_small_image_by_pixel_centres() {
    // The centre of the 3x3 block (0..3, 0..3) is full pixel 1, small pixel 0.
    assert_eq!(small_from_full([1.0, 1.0], [1.0 / 3.0; 2]), [0.0, 0.0]);
    assert_eq!(small_from_full([301.0, 601.0], [1.0 / 3.0; 2]), [100.0, 200.0]);
    assert_eq!(small_box_from_detnet([320.0, 240.0, 50.0]), ([320.0, 180.0], [50.0, 50.0]));
    assert_eq!(small_box_from_detnet([1.0, 61.0, 1.5]), ([1.0, 1.0], [1.5, 1.5]), "small circles are pixels too");
    // A full-resolution pixel through the tracker's letterbox lands where the small image has it.
    let net = BarLetterbox::robocap().to_net(&nalgebra::Vector2::new(301.0, 601.0));
    assert_eq!(small_box_from_detnet([net.x as f32, net.y as f32, 1.0]).0, small_from_full([301.0, 601.0], [1.0 / 3.0; 2]));
}

#[test]
fn camera_poses_invert_cam_from_rig() {
    let cam_from_rig = [[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]];
    let centre = rig_from_cam(&cam_from_rig).translation.vector;
    let m = nalgebra::Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    // The camera centre in the rig frame maps to the camera origin.
    assert!((m * centre + Vector3::new(0.1, 0.2, 0.3)).norm() < 1e-12);
}

#[test]
fn the_trajectory_gets_an_edge_every_few_framesets_and_restarts_after_a_lost_pose() {
    let mut state = Sink::default();
    let signals = Signals::default();
    let edges: Vec<Option<[[f32; 3]; 2]>> = (0..7).map(|i| state.record(i, Some(pose(i as f64)), "ok", None, &signals).scene.edge).collect();
    assert_eq!(edges, vec![None, None, None, Some([[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]]), None, None, Some([[3.0, 0.0, 1.0], [6.0, 0.0, 1.0]])]);
    let lost = state.record(7, None, "lost", None, &signals);
    assert!(lost.pose_lost && lost.scene.edge.is_none() && lost.status == Some("lost"));
    let again: Vec<bool> = (8..12).map(|i| state.record(i, Some(pose(i as f64)), "ok", None, &signals).scene.edge.is_some()).collect();
    assert_eq!(again, vec![false, false, false, true]);
}

/// The logger tests' rig (six 1920x1080 pinholes, f 600, camera i 0.05*i m along x) with camera 0 an equidistant fisheye
/// (KB4 with zero coefficients: r = f * theta) and camera 5 looking up the rig's y axis (no landmark lies in front of it).
fn lens_rig() -> Rig {
    let mut rig = crate::log::tests::test_rig();
    rig.cameras[0].fisheye62 = Some([0.0; 8]);
    rig.cameras[5].cam_from_rig = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    rig
}

/// A reported hand whose landmark 1 sits 45 degrees right of camera 0's axis, landmark 2 behind every forward camera and
/// the rest 1 m ahead; the step's headset pose is the identity.
fn reported_hand() -> HandFrameResult {
    let mut landmarks = [[0.0, 0.0, 1.0]; 21];
    landmarks[1] = [1.0, 0.0, 1.0];
    landmarks[2] = [0.0, 0.0, -1.0];
    let left = HandOutput { tracked: true, reported: true, landmarks_world: Some(landmarks), ..HandOutput::default() };
    HandFrameResult { hands: [left, HandOutput::default()], world_from_rig: Some(Isometry3::identity()), ..HandFrameResult::default() }
}

fn items<'a>(record: &FrameRecord<'a>, layer: Layer) -> Vec<&'a PaneItem> {
    record.scene.panes.iter().filter(|item| item.layer == layer).collect()
}

/// The vertex and triangle counts a sink writes for each hand's mesh.
fn mesh_sizes(record: &FrameRecord<'_>) -> [(Option<usize>, Option<usize>); 2] {
    record.hands.map(|hand| match hand {
        Hand3d::Draw { mesh, triangles, .. } => (mesh.map(<[_]>::len), triangles.map(<[_]>::len)),
        _ => (None, None),
    })
}

fn keynet_view(camera: usize, presence: f32, outcome: ViewOutcome, crop_source: CropSource) -> KeyNetView {
    KeyNetView {
        camera,
        keypoints_px: [[301.0, 601.0]; 21],
        confidence: std::array::from_fn(|i| i as f32 / 20.0),
        d_rel_mm: [7.0; 21],
        presence,
        pinch: None,
        outcome,
        // Straight down the camera's axis; 96 crop pixels per unit of normalised coordinate: the crop spans +-0.5.
        crop: Some(CropCamera { rotation: nalgebra::Matrix3::identity(), focal: 96.0, mirror: false }),
        crop_source,
    }
}

fn near(a: [f32; 2], b: [f32; 2]) -> bool {
    (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3
}

#[test]
fn the_fit_is_projected_through_each_lens_into_every_camera_that_sees_it() {
    let mut state = Sink::new(&lens_rig(), HandOverlays::Fit);
    let record = state.record(0, None, "", Some(&reported_hand()), &Signals::default());
    let fit = items(&record, Layer::Fit);
    assert_eq!(fit.iter().map(|item| (item.camera, item.side)).collect::<Vec<_>>(), (0..5).map(|c| (c, 0)).collect::<Vec<_>>(), "camera 5 looks away");
    let PaneDraw::Skeleton { points, keypoint_ids } = &fit[0].draw else { panic!("a skeleton, got {:?}", fit[0].draw) };
    assert!(!keypoint_ids.contains(&2) && keypoint_ids.len() == 20, "the landmark behind the camera is left out");
    let one = keypoint_ids.iter().position(|&id| id == 1).map(|k| points[k]);
    // The fisheye puts 45 degrees at f * pi / 4 from the centre; a pinhole would put it at f * tan(45 deg) = 600 px.
    let fisheye = small_from_full([959.5 + 600.0 * std::f32::consts::FRAC_PI_4, 539.5], [1.0 / 3.0; 2]);
    assert!(one.is_some_and(|p| near(p, fisheye)), "{one:?} vs {fisheye:?}");
    let PaneDraw::Skeleton { points, keypoint_ids } = &fit[1].draw else { panic!("a skeleton") };
    let ahead = keypoint_ids.iter().position(|&id| id == 0).map(|k| points[k]);
    assert!(ahead.is_some_and(|p| near(p, small_from_full([959.5 + 600.0 * 0.05, 539.5], [1.0 / 3.0; 2]))), "camera 1 sits 5 cm along x");
    assert!(items(&record, Layer::KeyNet).is_empty() && items(&record, Layer::Crop).is_empty(), "the fit level shows the fit only");
}

#[test]
fn keynet_confidences_take_simplecvs_red_yellow_green_colours() {
    let cases = [(0.0, [255, 0, 0]), (0.25, [255, 127, 0]), (0.5, [255, 255, 0]), (0.75, [127, 255, 0]), (1.0, [0, 255, 0])];
    for (confidence, rgb) in cases {
        assert_eq!(confidence_rgb(confidence), rgb, "confidence {confidence}");
    }
    assert_eq!((confidence_rgb(f32::NAN), confidence_rgb(-1.0), confidence_rgb(2.0)), ([255, 0, 0], [255, 0, 0], [0, 255, 0]));
}

#[test]
fn debug_overlays_show_every_keynet_view_with_its_crop_outline_and_verdict() {
    let mut state = Sink::new(&lens_rig(), HandOverlays::Debug);
    let mut result = reported_hand();
    result.hands[0].keynet_views =
        vec![keynet_view(1, 0.98, ViewOutcome::Fitted, CropSource::Pose), keynet_view(2, 0.01, ViewOutcome::LowPresence, CropSource::DetNet)];
    let record = state.record(0, None, "", Some(&result), &Signals::default());
    let keynet = items(&record, Layer::KeyNet);
    assert_eq!(keynet.iter().map(|item| item.camera).collect::<Vec<_>>(), vec![1, 2]);
    let PaneDraw::KeyNet { points, confidence, d_rel_mm, .. } = &keynet[0].draw else { panic!("KeyNet dots") };
    assert_eq!((points[0], confidence[20], d_rel_mm.is_none()), ([100.0, 200.0], 1.0, true), "relative depths are verbose only");
    let crops = items(&record, Layer::Crop);
    let PaneDraw::Outline { points, color, label } = &crops[0].draw else { panic!("an outline") };
    // The crop's corners are the rays (-0.5, -0.5, 1) and (0.5, 0.5, 1) through camera 1's pinhole.
    assert!(points.iter().any(|&p| near(p, [219.5, 79.5])) && points.iter().any(|&p| near(p, [419.5, 279.5])), "{points:?}");
    assert!(points.first() == points.last(), "closed");
    assert_eq!((*color, label.as_str()), (FITTED_COLOR, "L keynet 0.98 ok (pose)"));
    let PaneDraw::Outline { color, label, .. } = &crops[1].draw else { panic!("an outline") };
    assert_eq!((*color, label.as_str()), (REJECTED_COLOR, "L keynet 0.01 low presence (detnet)"));
}

#[test]
fn each_camera_scales_to_the_small_image_from_its_own_calibration_size() {
    let mut rig = lens_rig();
    (rig.cameras[2].width, rig.cameras[2].height) = (3840, 2160);
    let mut state = Sink::new(&rig, HandOverlays::Debug);
    let mut result = reported_hand();
    result.hands[0].keynet_views =
        vec![keynet_view(1, 0.98, ViewOutcome::Fitted, CropSource::Pose), keynet_view(2, 0.98, ViewOutcome::Fitted, CropSource::Pose)];
    let record = state.record(0, None, "", Some(&result), &Signals::default());
    let first: Vec<[f32; 2]> = items(&record, Layer::KeyNet)
        .iter()
        .filter_map(|item| match &item.draw {
            PaneDraw::KeyNet { points, .. } => Some(points[0]),
            _ => None,
        })
        .collect();
    assert_eq!(first, vec![[100.0, 200.0], small_from_full([301.0, 601.0], [1.0 / 6.0; 2])], "1920x1080 is /3, 3840x2160 /6");
}

#[test]
fn rejection_reasons_read_in_the_crop_label() {
    let mut state = Sink::new(&lens_rig(), HandOverlays::Debug);
    let mut result = reported_hand();
    result.hands[0].reported = false;
    result.hands[0].keynet_views = vec![
        keynet_view(0, 0.9, ViewOutcome::Fitted, CropSource::Recrop),
        keynet_view(1, 0.9, ViewOutcome::LostPair, CropSource::Pose),
        keynet_view(2, 0.9, ViewOutcome::FitResidual { rms_px: 12.34, limit_px: 8.06 }, CropSource::DetNet),
        keynet_view(3, 0.9, ViewOutcome::FitFailed, CropSource::DetNet),
    ];
    let record = state.record(0, None, "", Some(&result), &Signals::default());
    let labels: Vec<String> = items(&record, Layer::Crop).iter().filter_map(|item| match &item.draw {
        PaneDraw::Outline { label, .. } => Some(label.clone()),
        _ => None,
    }).collect();
    assert_eq!(labels, vec![
        "L keynet 0.90 ok, tentative (recrop)",
        "L keynet 0.90 lost its pair (pose)",
        "L keynet 0.90 fit rms 12.3 > 8.1 px (detnet)",
        "L keynet 0.90 fit failed (detnet)",
    ]);
}

#[test]
fn detnet_answers_show_by_verdict_at_debug_and_only_the_strongest_at_fit() {
    let hits = vec![
        DetNetHit { camera: 0, circle: [320.0, 240.0, 30.0], probability: 0.93, accepted: true },
        DetNetHit { camera: 3, circle: [100.0, 160.0, 20.0], probability: 0.6, accepted: false },
        DetNetHit { camera: 4, circle: [100.0, 160.0, 20.0], probability: 0.2, accepted: false },
    ];
    let acquiring = HandOutput { detnet_circle: Some([320.0, 240.0, 30.0]), detnet_camera: Some(0), detnet_hits: hits, ..HandOutput::default() };
    let result = HandFrameResult { hands: [acquiring, HandOutput::default()], ..HandFrameResult::default() };
    let mut debug = Sink::new(&lens_rig(), HandOverlays::Debug);
    let record = debug.record(0, None, "", Some(&result), &Signals::default());
    let boxes: Vec<(usize, PaneDraw)> = items(&record, Layer::DetNet).iter().map(|item| (item.camera, item.draw.clone())).collect();
    assert_eq!(boxes, vec![
        (0, PaneDraw::Box { center: [320.0, 180.0], half: [30.0, 30.0], color: DETNET_COLOR, label: "DetNet left 0.93".into() }),
        (3, PaneDraw::Box { center: [100.0, 100.0], half: [20.0, 20.0], color: DETNET_REJECTED_COLOR, label: "DetNet left 0.60".into() }),
    ]);
    let mut fit = Sink::new(&lens_rig(), HandOverlays::Fit);
    let record = fit.record(0, None, "", Some(&result), &Signals::default());
    let strongest: Vec<(usize, PaneDraw)> = items(&record, Layer::DetNet).iter().map(|item| (item.camera, item.draw.clone())).collect();
    assert_eq!(strongest, boxes[..1], "the strongest accepted answer, drawn as at debug");
    assert_eq!(record.scene.panes.len(), 1);
}

#[test]
fn verbose_overlays_add_relative_depths_and_the_predicted_pose() {
    let mut state = Sink::new(&lens_rig(), HandOverlays::Verbose);
    let mut result = reported_hand();
    result.hands[0].keynet_views = vec![keynet_view(1, 0.98, ViewOutcome::Fitted, CropSource::Pose)];
    result.hands[0].predicted_landmarks_world = result.hands[0].landmarks_world;
    let record = state.record(0, None, "", Some(&result), &Signals::default());
    let PaneDraw::KeyNet { d_rel_mm, .. } = &items(&record, Layer::KeyNet)[0].draw else { panic!("KeyNet dots") };
    assert_eq!(d_rel_mm.as_deref(), Some(&[7.0; 21]));
    assert_eq!(items(&record, Layer::Predicted).len(), 5, "the predicted pose in every camera that sees it");
}

#[test]
fn a_reported_hand_with_a_pose_gets_its_skinned_mesh_and_the_triangles_each_time_it_appears() {
    let mut state = Sink::new(&lens_rig(), HandOverlays::Fit);
    let signals = Signals::default();
    let mut result = reported_hand();
    result.scale = 0.97;
    result.hands[0].pose = Some(handfit::Pose {
        rotation: nalgebra::Matrix3::identity(),
        translation: nalgebra::Vector3::new(0.0, 0.0, 0.5),
        angles: handfit::nalgebra::SVector::zeros(),
    });
    let vertices = Some(crate::hands::mesh::MESH_VERTICES);
    let first = state.record(0, None, "", Some(&result), &signals);
    assert_eq!(mesh_sizes(&first), [(vertices, Some(1544)), (None, None)], "the triangles go with the first frameset of a hand");
    let second = state.record(1, None, "", Some(&result), &signals);
    assert_eq!(mesh_sizes(&second), [(vertices, None), (None, None)]);
    let gone = state.record(2, None, "", Some(&HandFrameResult::default()), &signals);
    assert_eq!(gone.hands[0], Hand3d::Clear);
    let back = state.record(3, None, "", Some(&result), &signals);
    assert_eq!(mesh_sizes(&back), [(vertices, Some(1544)), (None, None)], "a Clear drops the triangles, so they go again");
}

#[test]
fn panes_are_drawn_while_present_and_cleared_once_when_they_go() {
    let mut state = Sink::new(&lens_rig(), HandOverlays::Fit);
    let signals = Signals::default();
    let record = state.record(0, None, "", Some(&reported_hand()), &signals);
    assert!(matches!(record.hands[0], Hand3d::Draw { .. }) && record.hands[1] == Hand3d::Keep);
    assert_eq!(items(&record, Layer::Fit).len(), 5);
    let gone = HandFrameResult::default();
    let record = state.record(1, None, "", Some(&gone), &signals);
    assert_eq!(record.hands[0], Hand3d::Clear);
    assert_eq!(record.pane_clears, (0..5).map(|camera| (camera, 0, Layer::Fit)).collect::<Vec<_>>());
    let record = state.record(2, None, "", Some(&gone), &signals);
    assert!(record.pane_clears.is_empty() && record.scene.panes.is_empty() && record.hands[0] == Hand3d::Keep);
}

#[test]
fn each_timing_series_reads_the_stage_of_its_name() -> Result<(), serde_json::Error> {
    let t =
        FrameTimings { slam_ms: 2.0, hands_ms: 3.0, detnet_ms: 4.0, crops_ms: 5.0, keynet_ms: 6.0, fit_ms: 7.0, pipeline_ms: 8.0, ..FrameTimings::downsampled(1.0) };
    // The record JSONL names the stages by their field names, as the series do.
    let named = serde_json::to_value(t)?;
    for (name, value) in TIMING_SERIES.iter().zip(timing_row(&t, 9.0)) {
        let expected = if *name == "log_ms" { Some(9.0) } else { named[name].as_f64() };
        assert_eq!(Some(value), expected, "{name}");
    }
    Ok(())
}

#[test]
fn timings_are_thinned_in_the_fixed_series_order() {
    let mut state = Sink::default();
    let mut timings = [f64::NAN; TIMING_SERIES.len()];
    timings[2] = 5.0;
    let mut signals = Signals { timings, fps: Some(30.0), counters: [0.0, 1.0, 2.0] };
    let first = state.record(0, None, "", None, &signals);
    assert_eq!((first.scene.fps, first.scene.counters), (Some(30.0), Some([0.0, 1.0, 2.0])));
    assert_eq!(first.scene.timings.map(|t| t[2]), Some(5.0), "slam_ms is the third series");
    assert!(state.record(1, None, "", None, &signals).scene.timings.is_none());
    signals.timings[3] = 12.0;
    let third = state.record(2, None, "", None, &signals).scene.timings.unwrap_or([0.0; TIMING_SERIES.len()]);
    assert!(third[0].is_nan() && third[2] == 5.0 && third[3] == 12.0);
    assert_eq!((TIMING_SERIES[2], TIMING_SERIES[3]), ("slam_ms", "hands_ms"));
}
