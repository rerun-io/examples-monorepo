//! Logger tests: a saved recording holds what the viewer needs, and an unreachable viewer never blocks the caller.

use std::collections::BTreeMap;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use kornia_image::Image;
use nalgebra::Isometry3;

use super::preview::{KeyGate, PreviewItem};
use super::video::VideoSample;
use super::*;
use crate::frame::{CAMERA_NAMES, RigCamera};
use crate::hands::perspective::CropCamera;
use crate::hands::{CropSource, DetNetHit, HandOutput, KeyNetView, ViewOutcome};
use crate::sched::FrameTimings;

pub(crate) fn test_rig() -> Rig {
    let cameras = CAMERA_NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| RigCamera {
            name: name.to_string(),
            width: 1920,
            height: 1080,
            cam_from_rig: [
                [1.0, 0.0, 0.0, 0.05 * i as f64],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            focal: [600.0, 600.0],
            principal: [959.5, 539.5],
            fisheye62: None,
        })
        .collect();
    Rig {
        cameras,
        source: "test".into(),
        device: "cap_a".into(),
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("robocap-live-log-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn image(frame: u64) -> Luma {
    let (w, h) = (SMALL_SIZE.width, SMALL_SIZE.height);
    let data: Vec<u8> = (0..w * h)
        .map(|i| ((i % w + i / w) as u64 + frame * 5) as u8)
        .collect();
    Arc::new(
        Image::new(SMALL_SIZE, data).unwrap_or_else(|_| {
            Image::from_size_val(SMALL_SIZE, 0).unwrap_or_else(|_| unreachable!())
        }),
    )
}

fn hands(frame: u64) -> HandFrameResult {
    let x = 0.01 * frame as f64;
    let left = HandOutput {
        tracked: true,
        reported: true,
        pose: Some(handfit::Pose {
            rotation: nalgebra::Matrix3::identity(),
            translation: nalgebra::Vector3::new(x, 0.3, 1.0),
            angles: handfit::nalgebra::SVector::zeros(),
        }),
        // 1 m ahead of every camera, so the fit shows on all six panes.
        landmarks_world: Some(std::array::from_fn(|i| [x + 0.01 * i as f64, 0.3, 1.0])),
        keynet_views: vec![KeyNetView {
            camera: 2,
            keypoints_px: std::array::from_fn(|i| [900.0 + 10.0 * i as f32, 500.0]),
            confidence: std::array::from_fn(|i| i as f32 / 20.0),
            d_rel_mm: [3.0; 21],
            presence: 0.9,
            pinch: Some(0.1),
            outcome: ViewOutcome::Fitted,
            crop: Some(CropCamera {
                rotation: nalgebra::Matrix3::identity(),
                focal: 96.0,
                mirror: false,
            }),
            crop_source: CropSource::Pose,
        }],
        detnet_circle: None,
        detnet_camera: None,
        ..HandOutput::default()
    };
    HandFrameResult {
        hands: [left, HandOutput::default()],
        detnet_camera: None,
        scale: 0.97,
        scale_final: true,
        world_from_rig: Some(Isometry3::identity()),
        ..HandFrameResult::default()
    }
}

/// An untracked hand DetNet found on `camera` only, as the tracker reports it (that answer is the hand's strongest).
fn detected(hand: &mut HandOutput, camera: usize) {
    let circle = [100.0, 160.0, 20.0];
    hand.detnet_hits = vec![DetNetHit {
        camera,
        circle,
        probability: 0.9,
        accepted: true,
    }];
    (hand.detnet_circle, hand.detnet_camera) = (Some(circle), Some(camera));
}

fn host_encoder() -> Option<EncoderConfig> {
    let has = |program: &str, args: &[&str]| {
        Command::new(program)
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    };
    if has("gst-inspect-1.0", &["openh264enc"]) {
        video::openh264(SMALL_SIZE, 30, 1_000_000, 10).ok()
    } else if has("ffmpeg", &["-hide_banner", "-h", "encoder=libx264"]) {
        video::x264(SMALL_SIZE, 30, 1_000_000, 10).ok()
    } else {
        None
    }
}

/// Temporal rows per entity path in a saved recording's data store (static rows are not counted).
fn rows_per_entity(path: &Path) -> BTreeMap<String, usize> {
    let mut rows = BTreeMap::new();
    let Ok(file) = std::fs::File::open(path) else {
        return rows;
    };
    let Ok(decoder) = re_log_encoding::DecoderApp::decode_eager(BufReader::new(file)) else {
        return rows;
    };
    for message in decoder.flatten() {
        if let rerun::log::LogMsg::ArrowMsg(store, arrow) = &message
            && store.kind() == rerun::external::re_log_types::StoreKind::Recording
            && let Ok(chunk) = rerun::log::Chunk::from_arrow_msg(arrow)
            && !chunk.is_static()
        {
            *rows.entry(chunk.entity_path().to_string()).or_insert(0) += chunk.num_rows();
        }
    }
    rows
}

/// The timelines of a saved recording's temporal data.
fn timeline_names(path: &Path) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    let Ok(file) = std::fs::File::open(path) else {
        return names;
    };
    let Ok(decoder) = re_log_encoding::DecoderApp::decode_eager(BufReader::new(file)) else {
        return names;
    };
    for message in decoder.flatten() {
        if let rerun::log::LogMsg::ArrowMsg(_, arrow) = &message
            && let Ok(chunk) = rerun::log::Chunk::from_arrow_msg(arrow)
        {
            names.extend(chunk.timelines().keys().map(|name| name.to_string()));
        }
    }
    names
}

/// Component names per entity path in a saved recording (static and temporal).
fn components_per_entity(path: &Path) -> BTreeMap<String, std::collections::BTreeSet<String>> {
    let mut components: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    let Ok(file) = std::fs::File::open(path) else {
        return components;
    };
    let Ok(decoder) = re_log_encoding::DecoderApp::decode_eager(BufReader::new(file)) else {
        return components;
    };
    for message in decoder.flatten() {
        if let rerun::log::LogMsg::ArrowMsg(_, arrow) = &message
            && let Ok(chunk) = rerun::log::Chunk::from_arrow_msg(arrow)
        {
            let names = components
                .entry(chunk.entity_path().to_string())
                .or_default();
            names.extend(
                chunk
                    .component_descriptors()
                    .map(|d| d.component.to_string()),
            );
        }
    }
    components
}

/// Temporal rows of `entity` in a saved recording that carry `component` (a row may log some of its entity's components only).
fn rows_with_component(path: &Path, entity: &str, component: &str) -> usize {
    use rerun::external::arrow::array::Array;
    let mut rows = 0;
    let Ok(file) = std::fs::File::open(path) else {
        return rows;
    };
    let Ok(decoder) = re_log_encoding::DecoderApp::decode_eager(BufReader::new(file)) else {
        return rows;
    };
    for message in decoder.flatten() {
        if let rerun::log::LogMsg::ArrowMsg(_, arrow) = &message
            && let Ok(chunk) = rerun::log::Chunk::from_arrow_msg(arrow)
            && !chunk.is_static()
            && chunk.entity_path().to_string() == entity
        {
            let columns = chunk
                .components()
                .0
                .values()
                .filter(|column| column.descriptor.component == component);
            rows += columns
                .map(|column| column.list_array.len() - column.list_array.null_count())
                .sum::<usize>();
        }
    }
    rows
}

/// Log `frames` framesets of [`hands`] into a saved recording with these overlays; the recording's path.
fn save_hands(
    name: &str,
    frames: u64,
    hand_overlays: scene::HandOverlays,
) -> Result<PathBuf, LogError> {
    let save = temp_dir(name).join("out.rrd");
    let options = LoggerConfig {
        save: Some(save.clone()),
        video: VideoMode::Off,
        hand_overlays,
        recording_id: Some(name.into()),
        input_queue: 64,
        ..LoggerConfig::default()
    };
    let mut logger = Logger::new(&test_rig(), options)?;
    for frame in 0..frames {
        let result = hands(frame);
        let timings = FrameTimings::downsampled(1.0);
        logger.log_frameset(&FrameLog {
            t_ns: frame as i64 * 33_333_333,
            small: [None; NUM_CAMERAS],
            world_from_rig: None,
            slam_status: "tracking",
            hands: Some(&result),
            timings: &timings,
        })?;
    }
    logger.finish()?;
    Ok(save)
}

#[test]
fn a_hands_layer_saves_only_the_hands_and_no_recording_properties_and_drops_nothing()
-> Result<(), LogError> {
    let save = temp_dir("hands-layer").join("out.rrd");
    // A one-frameset queue: a lossless logger waits for its worker instead of dropping.
    let options = LoggerConfig {
        save: Some(save.clone()),
        video: VideoMode::Off,
        hand_overlays: scene::HandOverlays::Debug,
        recording_id: Some("segment".into()),
        time_origin_ns: Some(0),
        content: scene::Content::HandsLayer,
        lossless: true,
        input_queue: 1,
        ..LoggerConfig::default()
    };
    let mut logger = Logger::new(&test_rig(), options)?;
    for frame in 0..40 {
        let result = hands(frame);
        let timings = FrameTimings::downsampled(1.0);
        let pose = Isometry3::identity();
        logger.log_frameset(&FrameLog {
            t_ns: 1_000_000_000 + frame as i64 * 33_333_333,
            small: [None; NUM_CAMERAS],
            world_from_rig: Some(&pose),
            slam_status: "reference",
            hands: Some(&result),
            timings: &timings,
        })?;
    }
    let (stats, _) = logger.finish()?;
    assert_eq!((stats.framesets_in, stats.framesets_dropped), (40, 0));
    let components = components_per_entity(&save);
    let foreign: Vec<&String> = components
        .keys()
        .filter(|entity| {
            !(entity.starts_with("/world/hands/")
                || entity.contains("/pinhole/hands/")
                || *entity == "/world")
        })
        .collect();
    assert!(
        foreign.is_empty(),
        "a layer holds only the hands: {foreign:?}"
    );
    assert_eq!(
        components
            .get("/world")
            .map(|names| names.iter().map(String::as_str).collect::<Vec<_>>()),
        Some(vec!["AnnotationContext:context"])
    );
    let rows = rows_per_entity(&save);
    assert_eq!(
        rows.get("/world/rig_00/cam_02/pinhole/hands/left/keynet"),
        Some(&40),
        "{rows:?}"
    );
    assert!(
        rows.get("/world/hands/left/keypoints")
            .is_some_and(|&n| n == 40),
        "{rows:?}"
    );
    Ok(())
}

#[test]
fn debug_overlays_save_keynet_dots_with_simplecvs_confidence_components_and_crop_outlines()
-> Result<(), LogError> {
    let save = save_hands("debug-overlays", 5, scene::HandOverlays::Debug)?;
    let rows = rows_per_entity(&save);
    let keynet = "/world/rig_00/cam_02/pinhole/hands/left/keynet";
    assert_eq!(
        (
            rows.get(keynet),
            rows.get("/world/rig_00/cam_02/pinhole/hands/left/crop")
        ),
        (Some(&5), Some(&5)),
        "{rows:?}"
    );
    let components = components_per_entity(&save);
    let names = components.get(keynet).cloned().unwrap_or_default();
    for name in [
        "simplecv.KeypointConfidence2D:confidences",
        "simplecv.KeypointConfidence2D:average_confidence",
        "Points2D:colors",
    ] {
        assert!(names.contains(name), "{name} missing from {names:?}");
    }
    assert!(
        !names.contains("robocap.KeyNet2D:relative_depth_mm"),
        "relative depths are verbose only"
    );
    let mesh = components
        .get("/world/hands/left/mesh")
        .cloned()
        .unwrap_or_default();
    for name in [
        "Mesh3D:vertex_positions",
        "Mesh3D:triangle_indices",
        "Mesh3D:albedo_factor",
    ] {
        assert!(mesh.contains(name), "{name} missing from {mesh:?}");
    }
    assert_eq!(
        rows.get("/world/hands/left/mesh"),
        Some(&5),
        "the skinned mesh every frameset"
    );
    assert_eq!(
        rows_with_component(&save, "/world/hands/left/mesh", "Mesh3D:triangle_indices"),
        1,
        "the triangles with the first only"
    );
    let verbose = components_per_entity(&save_hands(
        "verbose-overlays",
        2,
        scene::HandOverlays::Verbose,
    )?);
    assert!(
        verbose
            .get(keynet)
            .is_some_and(|names| names.contains("robocap.KeyNet2D:relative_depth_mm"))
    );
    let fit = rows_per_entity(&save_hands("fit-overlays", 2, scene::HandOverlays::Fit)?);
    assert!(
        !fit.contains_key(keynet)
            && fit.get("/world/rig_00/cam_05/pinhole/hands/left/fit") == Some(&2),
        "{fit:?}"
    );
    Ok(())
}

#[test]
fn a_saved_recording_holds_scene_video_pose_trajectory_hands_and_timings() -> Result<(), LogError> {
    let (video, encoder) = match host_encoder() {
        Some(encoder) => (VideoMode::H264, encoder),
        None => (VideoMode::Raw, video::mpp(SMALL_SIZE, 30, 1_000_000, 30)?),
    };
    let dir = temp_dir("save");
    let save = dir.join("out.rrd");
    let options = LoggerConfig {
        save: Some(save.clone()),
        video,
        encoder,
        input_queue: 64,
        recording_id: Some("test".into()),
        ..LoggerConfig::default()
    };
    let mut logger = Logger::new(&test_rig(), options)?;
    let frames = 20u64;
    for frame in 0..frames {
        let images: Vec<Luma> = (0..NUM_CAMERAS as u64).map(|c| image(frame + c)).collect();
        let pose = Isometry3::translation(0.1 * frame as f64, 0.0, 1.5);
        let result = hands(frame);
        let timings = FrameTimings {
            slam_ms: 5.0 + frame as f64,
            hands_ms: 12.0,
            ..FrameTimings::downsampled(1.0)
        };
        logger.log_frameset(&FrameLog {
            t_ns: 1_000_000_000 + frame as i64 * 33_333_333,
            small: std::array::from_fn(|c| images.get(c)),
            world_from_rig: Some(&pose),
            slam_status: "tracking",
            hands: Some(&result),
            timings: &timings,
        })?;
    }
    let (stats, encoders) = logger.finish()?;
    assert_eq!(
        (stats.framesets_in, stats.framesets_dropped, stats.errors),
        (frames, 0, 0)
    );
    let rows = rows_per_entity(&save);
    let video_entity = if video == VideoMode::H264 {
        "/world/rig_00/cam_00/pinhole/video"
    } else {
        "/world/rig_00/cam_00/pinhole/image"
    };
    if video == VideoMode::H264 {
        assert_eq!(encoders.len(), NUM_CAMERAS);
        assert_eq!(stats.video_samples, frames * NUM_CAMERAS as u64);
    }
    let n = frames as usize;
    assert_eq!(rows.get(video_entity).copied(), Some(n), "{rows:?}");
    assert_eq!(rows.get("/world/rig_00").copied(), Some(n));
    // An edge every third frameset; timings every second one; fps from the second sample on; counters every 15th.
    assert_eq!(
        rows.get("/world/runs/slam_rs/trajectory").copied(),
        Some((n - 1) / scene::TRAJECTORY_EVERY as usize)
    );
    assert_eq!(rows.get("/world/hands/left/keypoints").copied(), Some(n));
    assert_eq!(
        rows.get("/world/rig_00/cam_02/pinhole/hands/left/fit")
            .copied(),
        Some(n),
        "the fit on every pane"
    );
    assert_eq!(
        rows.get("/timings").copied(),
        Some(n.div_ceil(scene::TIMINGS_EVERY as usize))
    );
    assert_eq!(
        rows.get("/fps").copied(),
        Some(n.div_ceil(scene::TIMINGS_EVERY as usize) - 1)
    );
    assert_eq!(
        rows.get("/log").copied(),
        Some(n.div_ceil(scene::COUNTERS_EVERY as usize))
    );
    assert!(
        !rows.contains_key("/world/hands/right/keypoints")
            && !rows.contains_key("/world/rig_00/cam_02/pinhole/hands/left/detnet")
    );
    // Few chunks per frameset: what the wire pays for (each chunk carries its own schema): video and the fit per camera.
    let temporal_rows: usize = rows.values().sum();
    assert!(
        temporal_rows <= n * (2 * NUM_CAMERAS + 5),
        "{temporal_rows} rows for {n} framesets: {rows:?}"
    );
    assert_eq!(
        timeline_names(&save),
        [scene::TIMELINE.to_string()].into(),
        "no log_time"
    );
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

#[test]
fn an_unreachable_viewer_never_blocks_the_caller_and_drops_are_counted() -> Result<(), LogError> {
    // Port 9 (discard) is closed on the test hosts: the gRPC client keeps retrying, the SDK buffers fill, the queue drops.
    let options = LoggerConfig {
        viewer: Some("rerun+http://127.0.0.1:9/proxy".into()),
        video: VideoMode::Raw,
        preview_queue: 16,
        preview_max_bytes_in_flight: 1024 * 1024,
        input_queue: 64,
        ..LoggerConfig::default()
    };
    let mut logger = Logger::new(&test_rig(), options)?;
    let mut slowest = Duration::ZERO;
    for frame in 0..90u64 {
        let images: Vec<Luma> = (0..NUM_CAMERAS as u64).map(|c| image(frame + c)).collect();
        let start = Instant::now();
        logger.log_frameset(&FrameLog {
            t_ns: frame as i64 * 33_333_333,
            small: std::array::from_fn(|c| images.get(c)),
            world_from_rig: None,
            slam_status: "off",
            hands: None,
            timings: &FrameTimings::downsampled(1.0),
        })?;
        slowest = slowest.max(start.elapsed());
        std::thread::sleep(Duration::from_millis(5));
    }
    let started = Instant::now();
    let (stats, _) = logger.finish()?;
    assert!(
        slowest < Duration::from_millis(20),
        "log_frameset took {slowest:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "finish took {:?}",
        started.elapsed()
    );
    assert_eq!(stats.framesets_in, 90);
    assert!(stats.preview_dropped > 0, "{stats:?}");
    Ok(())
}

#[test]
fn a_gap_in_a_cameras_preview_samples_withholds_them_until_its_next_keyframe() {
    // Two encoder readers share a 3-item preview queue; they number every sample and always offer it.
    let counters = Arc::new(LogCounters::default());
    let (tx, rx) = sync_channel(3);
    let queue = PreviewQueue {
        tx,
        counters: counters.clone(),
    };
    let mut readers: Vec<VideoSink> = (0..2)
        .map(|_| VideoSink {
            save: Arc::new(Mutex::new(None)),
            preview: Some(queue.clone()),
            counters: counters.clone(),
            seq: 0,
        })
        .collect();
    let mut send = |camera: usize, keyframe: bool| {
        readers[camera].send(VideoSample {
            camera,
            t_ns: 0,
            data: vec![0u8; 16].into(),
            keyframe,
        });
    };
    // The sender takes what is queued; the samples that reach the viewer, as (camera, seq).
    let deliver = |gate: &mut KeyGate| -> Vec<(usize, u64)> {
        rx.try_iter()
            .filter_map(|item| match item {
                PreviewItem::Video { seq, sample } => gate
                    .admit(sample.camera, seq, sample.keyframe)
                    .then_some((sample.camera, seq)),
                _ => None,
            })
            .collect()
    };
    let mut gate = KeyGate::new();
    // A fresh stream starts at each camera's keyframe.
    send(1, false);
    send(0, true);
    send(1, true);
    assert_eq!(deliver(&mut gate), vec![(0, 0), (1, 1)]);
    // The link stalls: camera 0's sample 3 finds the queue full and is dropped.
    for (camera, keyframe) in [(0, false), (1, false), (0, false), (0, false)] {
        send(camera, keyframe);
    }
    assert_eq!(counters.preview_dropped.load(Ordering::Relaxed), 1);
    assert_eq!(deliver(&mut gate), vec![(0, 1), (1, 2), (0, 2)]);
    // Sample 4 follows the gap: camera 0 waits for its keyframe (5); camera 1 goes on.
    for (camera, keyframe) in [(0, false), (1, false), (0, true)] {
        send(camera, keyframe);
    }
    assert_eq!(deliver(&mut gate), vec![(1, 3), (0, 5)]);
    send(0, false);
    assert_eq!(deliver(&mut gate), vec![(0, 6)]);
    // A reconnect: every camera waits for its next keyframe.
    gate.reconnected();
    for (camera, keyframe) in [(0, false), (1, true), (0, true)] {
        send(camera, keyframe);
    }
    assert_eq!(deliver(&mut gate), vec![(1, 4), (0, 8)]);
}

#[test]
fn video_modes_parse() {
    assert_eq!("h264".parse::<VideoMode>().ok(), Some(VideoMode::H264));
    assert_eq!("raw".parse::<VideoMode>().ok(), Some(VideoMode::Raw));
    assert_eq!("off".parse::<VideoMode>().ok(), Some(VideoMode::Off));
    assert!("av1".parse::<VideoMode>().is_err());
}

#[test]
fn saving_is_refused_below_the_free_space_floor_and_the_video_set_is_respected()
-> Result<(), LogError> {
    let dir = temp_dir("floor");
    let options = LoggerConfig {
        save: Some(dir.join("never.rrd")),
        save_min_free_bytes: u64::MAX,
        video: VideoMode::Raw,
        video_cameras: vec![1, 4],
        input_queue: 16,
        ..LoggerConfig::default()
    };
    let mut logger = Logger::new(&test_rig(), options)?;
    let images: Vec<Luma> = (0..NUM_CAMERAS as u64).map(image).collect();
    logger.log_frameset(&FrameLog {
        t_ns: 0,
        small: std::array::from_fn(|c| images.get(c)),
        world_from_rig: None,
        slam_status: "",
        hands: None,
        timings: &FrameTimings::downsampled(1.0),
    })?;
    let (stats, _) = logger.finish()?;
    assert!(stats.save_stopped_low_disk);
    assert!(!dir.join("never.rrd").exists());
    assert!(free_bytes(&dir).is_some_and(|free| free > 0));
    let bad = LoggerConfig {
        video_cameras: vec![6],
        ..LoggerConfig::default()
    };
    assert!(matches!(
        Logger::new(&test_rig(), bad),
        Err(LogError::Invalid(_))
    ));
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

#[test]
fn preview_clears_a_hand_pose_and_old_camera_after_the_transition_was_dropped() {
    let counters = Arc::new(LogCounters::default());
    let (tx, rx) = sync_channel(1);
    let queue = PreviewQueue {
        tx,
        counters: counters.clone(),
    };
    let mut state = RecordState::new(&test_rig(), scene::HandOverlays::Fit);
    let mut delivered = DeliveredState::default();
    let mut seen = hands(0);
    detected(&mut seen.hands[0], 2);
    let pose = Isometry3::identity();
    let first = state.prepare(
        0,
        Some(pose),
        "tracking",
        Some(&seen),
        &scene::Signals::default(),
    );
    assert!(queue.offer(PreviewItem::Frame(Arc::new(first))));
    let mut absent = HandFrameResult::default();
    detected(&mut absent.hands[0], 4);
    let dropped = state.prepare(1, None, "lost", Some(&absent), &scene::Signals::default());
    assert!(!queue.offer(PreviewItem::Frame(Arc::new(dropped))));
    let PreviewItem::Frame(first) = rx.recv().unwrap() else {
        panic!("expected scene")
    };
    let first = delivered.record(&first);
    assert!(matches!(first.hands[0], scene::Hand3d::Draw { .. }));
    let next = state.prepare(2, None, "lost", Some(&absent), &scene::Signals::default());
    assert!(queue.offer(PreviewItem::Frame(Arc::new(next))));
    let PreviewItem::Frame(next) = rx.recv().unwrap() else {
        panic!("expected scene")
    };
    let next = delivered.record(&next);
    assert_eq!(next.hands[0], scene::Hand3d::Clear);
    assert!(next.pose_lost);
    let mut expected: Vec<(usize, usize, scene::Layer)> = (0..NUM_CAMERAS)
        .map(|camera| (camera, 0, scene::Layer::Fit))
        .collect();
    expected.push((2, 0, scene::Layer::DetNet));
    expected.sort_unstable();
    assert_eq!(next.pane_clears, expected);
    assert_eq!(next.status, Some("lost"));
}

#[test]
fn a_reconnected_preview_restores_present_entities_and_clears_every_absent_entity() {
    let mut producer = RecordState::new(&test_rig(), scene::HandOverlays::Fit);
    let signals = scene::Signals::default();
    let _ = producer.prepare(0, None, "lost", None, &signals);
    // Metadata is not scheduled for this frame, but a fresh connection must still get it.
    let mut current = hands(1);
    detected(&mut current.hands[0], 4);
    let snapshot = producer.prepare(1, None, "lost", Some(&current), &signals);
    let mut delivered = DeliveredState::reconnected();
    let record = delivered.record(&snapshot);
    assert!(matches!(record.hands[0], scene::Hand3d::Draw { .. }));
    assert_eq!(record.hands[1], scene::Hand3d::Clear);
    assert!(record.pose_lost);
    // Present: the fit on all six panes and the DetNet box on camera 4; every other (camera, side, layer) is cleared.
    let present: Vec<(usize, usize, scene::Layer)> = record
        .scene
        .panes
        .iter()
        .map(|item| (item.camera, item.side, item.layer))
        .collect();
    assert_eq!(present.len(), NUM_CAMERAS + 1);
    assert!(
        present.contains(&(4, 0, scene::Layer::DetNet))
            && present.contains(&(2, 0, scene::Layer::Fit))
    );
    assert_eq!(
        record.pane_clears.len(),
        NUM_CAMERAS * 2 * scene::Layer::ALL.len() - present.len()
    );
    assert!(
        present
            .iter()
            .all(|item| !record.pane_clears.contains(item))
    );
    assert_eq!(record.status, Some("lost"));
    assert!(
        record.scene.timings.is_none(),
        "keep the producer's thinning"
    );
    let next = delivered.record(&snapshot);
    assert!(!next.pose_lost);
    assert_eq!(next.hands[1], scene::Hand3d::Keep);
    assert!(next.pane_clears.is_empty() && next.status.is_none());
}

#[test]
fn six_encoders_close_together_before_any_is_drained() -> Result<(), LogError> {
    let options = LoggerConfig {
        video: VideoMode::H264,
        encoder: EncoderConfig {
            program: "sh".into(),
            args: vec!["-c".into(), "cat >/dev/null; sleep 0.25".into()],
            size: SMALL_SIZE,
        },
        ..LoggerConfig::default()
    };
    let logger = Logger::new(&test_rig(), options)?;
    let start = Instant::now();
    let (stats, encoders) = logger.finish()?;
    eprintln!("six encoder shutdown: {:?}", start.elapsed());
    assert_eq!(encoders.len(), 6);
    assert_eq!(stats.errors, 0);
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "six serial drains took {:?}",
        start.elapsed()
    );
    Ok(())
}

fn logger_stopped_for_low_disk(save: RecordingStream, path: PathBuf) -> Result<Logger, LogError> {
    let counters = Arc::new(LogCounters::default());
    let shutdown = Arc::new(Shutdown::default());
    let mut worker = Worker {
        shutdown: shutdown.clone(),
        finalizer: None,
        video: VideoMode::Off,
        video_cameras: Vec::new(),
        encoders: Vec::new(),
        save: Arc::new(Mutex::new(Some(save))),
        save_path: Some(path),
        save_min_free_bytes: 0,
        last_disk_check: Instant::now(),
        preview: None,
        counters: counters.clone(),
        state: RecordState::default(),
        delivered: DeliveredState::default(),
        content: scene::Content::Full,
        notice: None,
        fps_window: Default::default(),
        last_worker_ms: 0.0,
    };
    // Saving has already started. Simulate the next filesystem check crossing its configured floor.
    worker.frameset(FrameItem {
        t_ns: 123,
        small: std::array::from_fn(|_| None),
        world_from_rig: Some(Isometry3::identity()),
        slam_status: "tracking".into(),
        hands: None,
        timings: FrameTimings::downsampled(1.0),
        received: Instant::now(),
    })?;
    worker.save_min_free_bytes = u64::MAX;
    worker.last_disk_check = Instant::now() - Duration::from_secs(3);
    assert!(worker.check_disk()?.is_some());
    assert!(worker.finalizer.is_some());
    let (input, rx) = sync_channel(1);
    Ok(Logger {
        input: Some(input),
        worker: Some(std::thread::spawn(move || worker.run(rx))),
        preview: None,
        shutdown,
        counters,
        started: Instant::now(),
        time_origin_ns: None,
        recording_id: "low-disk".into(),
        lossless: false,
    })
}

#[test]
fn low_disk_finalization_after_recording_starts_is_complete_when_finish_returns()
-> Result<(), LogError> {
    let dir = temp_dir("low-disk-finish");
    let path = dir.join("saved.rrd");
    let save = rerun::RecordingStreamBuilder::new("low-disk").save(&path)?;
    let logger = logger_stopped_for_low_disk(save, path.clone())?;
    let (stats, _) = logger.finish()?;
    assert!(stats.save_stopped_low_disk);
    assert_eq!(rows_per_entity(&path).get("/world/rig_00"), Some(&1));
    std::fs::remove_dir_all(dir).unwrap();
    Ok(())
}

#[test]
fn logger_finish_reports_a_low_disk_finalizer_failure() -> Result<(), LogError> {
    struct FailingFile;
    impl rerun::sink::LogSink for FailingFile {
        fn send(&self, _: rerun::log::LogMsg) {}
        fn flush_blocking(&self, _: Duration) -> Result<(), rerun::sink::SinkFlushError> {
            Err(rerun::sink::SinkFlushError::failed(
                "injected file flush failure",
            ))
        }
    }
    let save = rerun::RecordingStreamBuilder::new("low-disk-failure")
        .set_sinks(vec![Box::new(FailingFile) as Box<dyn rerun::sink::LogSink>])?;
    let logger = logger_stopped_for_low_disk(save, PathBuf::from("/tmp"))?;
    let error = logger
        .finish()
        .expect_err("the retained finalizer must report its flush failure");
    assert!(
        error.to_string().contains("flushing the save file"),
        "{error}"
    );
    Ok(())
}

/// The pixel bytes of each serialised component of `archetype` that holds a blob (an image buffer or a video sample).
fn serialized_blob_pointers(archetype: &dyn rerun::AsComponents) -> Vec<*const u8> {
    archetype
        .as_serialized_batches()
        .iter()
        .filter_map(|batch| {
            rerun::datatypes::Blob::serialized_blob_as_slice(batch)
                .filter(|bytes| bytes.len() > 16)
                .map(<[u8]>::as_ptr)
        })
        .collect()
}

#[test]
fn logged_images_and_video_samples_reach_rerun_without_a_copy()
-> Result<(), Box<dyn std::error::Error>> {
    // A raw image: the blob borrows the image's own pixels and keeps the image alive.
    let luma: Luma = Arc::new(Image::new(
        SMALL_SIZE,
        (0..SMALL_SIZE.width * SMALL_SIZE.height)
            .map(|i| (i % 251) as u8)
            .collect(),
    )?);
    let pixels: *const u8 = luma.as_slice().as_ptr();
    let blob = worker::luma_blob(&luma);
    assert_eq!(
        (blob.as_ptr(), blob.len(), Arc::strong_count(&luma)),
        (pixels, luma.as_slice().len(), 2)
    );
    let image = rerun::Image::from_l8(blob, [SMALL_SIZE.width as u32, SMALL_SIZE.height as u32]);
    assert_eq!(
        serialized_blob_pointers(&image),
        vec![pixels],
        "the serialised image buffer is the image's memory"
    );
    drop(image);
    assert_eq!(
        Arc::strong_count(&luma),
        1,
        "the buffer releases the image when Rerun drops it"
    );
    // An access unit: the save stream's and the preview's samples share the splitter's allocation.
    let unit = VideoSample {
        camera: 0,
        t_ns: 0,
        data: vec![7u8; 4096].into(),
        keyframe: true,
    };
    let shared: *const u8 = unit.data.as_ptr();
    for _sink in ["save", "preview"] {
        let video = rerun::VideoStream::update_fields()
            .with_sample(unit.data.clone())
            .with_is_keyframe(unit.keyframe);
        assert_eq!(serialized_blob_pointers(&video), vec![shared]);
    }
    Ok(())
}
