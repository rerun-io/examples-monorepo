//! What the logger writes: the static scene from the rig, and one [`FrameRecord`] per frameset (SLAM pose and trajectory,
//! hands in 3D and on the camera panes, timings), on DataForge's RoboCap entity layout so the display asset's blueprint fits.
//!
//! Every chunk costs about a kilobyte of Arrow schema on the wire, whatever it holds, so the live stream keeps the number of
//! chunks per frameset small: what never changes (classes, keypoint ids, colours, radii, series names) is logged once as static
//! data, the per-frameset rows carry only positions, and the slow signals (trajectory edges, timings, counters) are thinned.
//!
//! Entity paths (rig 0, cameras `cam_00..cam_05`, sides `left`/`right`):
//! - `/world`: `ViewCoordinates` right-handed Z up (slam-rs's gravity-aligned world) and the `AnnotationContext` that draws the
//!   UmeTrack hand skeleton (class 0 = left, cyan; class 1 = right, orange; handtrack's colours).
//! - `/world/rig_00`: `Transform3D`, the SLAM `world_from_rig`, on `video_time`; `/world/rig_00/axes`: the rig's axes.
//! - `/world/rig_00/cam_NN`: static `Transform3D` (`rig_from_cam`); `.../pinhole`: static `Pinhole` at the 640x360 video size.
//! - `.../pinhole/video`: `VideoStream` (H.264) or `.../pinhole/image`: `Image` (raw luma).
//! - `/world/runs/slam_rs/{trajectory,trail}`: `LineStrips3D`, one edge every [`TRAJECTORY_EVERY`] framesets (the blueprint
//!   accumulates them).
//! - `/world/hands/{side}/keypoints`: `Points3D`, the 21 fitted landmarks in world metres, skeleton from the annotation context.
//! - `/world/hands/{side}/mesh`: `Mesh3D`, UmeTrack's generic hand mesh skinned on the fitted pose (as show3d and handtrack draw
//!   hands; their semi-transparent albedo). The triangle list goes with the first frameset of each appearance, not as static
//!   data, so a Clear takes it too.
//! - `.../pinhole/hands/{side}/{layer}`: the hand pipeline on each camera pane in 640x360 pixels, as much as [`HandOverlays`]
//!   asks for ([`Layer`]): `fit` (the fitted hand projected through the camera's own lens model: a skeleton in the hand's
//!   colour), `keynet` (KeyNet's keypoints as dots coloured by confidence, with simplecv's `Points2DWithConfidence`
//!   components), `crop` (the outline of the perspective crop KeyNet saw, traced through the lens, coloured and labelled by what
//!   the tracker did with the view), `predicted` (the pose a tracked hand's crops were planned from) and `detnet` (`Boxes2D`).
//!   Rerun's `Pinhole` has no lens distortion, so nothing here relies on the viewer projecting 3D points into the fisheye panes.
//! - `/timings`: `Scalars`, one series per stage ([`TIMING_SERIES`], named by a static `SeriesLines`); `/fps`; `/log`: the logger's
//!   counters (framesets dropped, preview items dropped, preview queue); `/slam/status`: `TextLog` when the status changes;
//!   `/log/status`: the logger's own warnings (saving stopped below the free-space floor).

use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::Isometry3;
use rerun::RecordingStream;

use super::{LogCounters, LogError};
use crate::frame::{NUM_CAMERAS, Rig, SMALL_SIZE};
use crate::hands::letterbox::BarLetterbox;
use crate::sched::FrameTimings;
use crate::frame::isometry_from_matrix;

mod overlays;
mod record;
#[cfg(test)]
mod tests;

pub use overlays::{HandDraw, RecordState, SceneSnapshot, Signals, SkinnedMesh};
pub use record::{FrameRecord, Hand3d, PaneDraw, PaneItem, write_record};
pub(in crate::log) use record::DeliveredState;

/// The single timeline (DataForge `schema.TIMELINE`): duration since the session start. Both streams turn Rerun's own
/// `log_time` off.
pub const TIMELINE: &str = "video_time";
/// Hand slot names (handtrack `SIDES`).
pub const SIDES: [&str; 2] = ["left", "right"];
/// Predicted hand colours (handtrack `rerun_layers.PRED_COLORS`): left cyan, right orange.
pub const HAND_COLORS: [[u8; 3]; 2] = [[0, 200, 255], [255, 150, 0]];
/// DetNet box colour (handtrack `SOURCE_COLORS[DETNET]`): an answer above the DetNet threshold.
pub const DETNET_COLOR: [u8; 3] = [255, 60, 255];
/// A DetNet answer below the threshold.
pub const DETNET_REJECTED_COLOR: [u8; 3] = [150, 150, 150];
/// DetNet answers below this probability are not drawn.
pub const DETNET_SHOWN: f32 = 0.5;
/// A KeyNet crop whose view the fit used.
pub const FITTED_COLOR: [u8; 3] = [0, 220, 90];
/// A KeyNet crop whose view the tracker rejected.
pub const REJECTED_COLOR: [u8; 3] = [255, 64, 64];
/// The predicted pose's skeleton.
pub const PREDICTED_COLOR: [u8; 3] = [170, 170, 170];
/// Annotation class of the predicted pose (0 and 1 are the hands).
pub const PREDICTED_CLASS: u16 = 2;
/// Points per side of a crop outline.
pub const CROP_OUTLINE_STEPS: usize = 8;
/// About half a crop label's size, in small-image pixels: Rerun centres a label on its anchor, so the anchor keeps this far
/// from the pane's sides. The text is sized in screen pixels, so this holds at about one zoom, and long labels are wider.
const CROP_LABEL_HALF_SIZE: [f32; 2] = [60.0, 8.0];
/// Hand mesh albedo, RGBA (dataforge `hands.HAND_ALBEDO`, the show3d look).
pub const HAND_ALBEDO: [[u8; 4]; 2] = [[90, 160, 240, 110], [240, 170, 130, 110]];
/// SLAM trajectory colour (PR #270's live trajectory).
pub const TRAJECTORY_COLOR: [u8; 3] = [255, 180, 40];
/// Frustum length in metres (DataForge `robocap.IMAGE_PLANE_DISTANCE`).
pub const IMAGE_PLANE_DISTANCE: f32 = 0.025;
/// Length of the rig's axes, metres.
pub const RIG_AXIS_LENGTH: f32 = 0.1;
/// simplecv's `UME_HAND_CONNECTIONS`: wrist (5) to each fingertip (0..4) through its joints.
pub const HAND_CONNECTIONS: [(u16, u16); 19] = [
    (5, 6), (6, 7), (7, 0), (5, 8), (8, 9), (9, 10), (10, 1), (5, 11), (11, 12), (12, 13), (13, 2), (5, 14), (14, 15), (15, 16), (16, 3),
    (5, 17), (17, 18), (18, 19), (19, 4),
];
/// A trajectory edge every this many framesets (10 Hz at 30 fps).
pub const TRAJECTORY_EVERY: u64 = 3;
/// Timings and fps every this many framesets (15 Hz).
pub const TIMINGS_EVERY: u64 = 2;
/// Logger counters every this many framesets (2 Hz).
pub const COUNTERS_EVERY: u64 = 15;

/// `/world/rig_00`.
pub const RIG_PATH: &str = "/world/rig_00";
/// The SLAM run source under `/world/runs` (DataForge RoboCap's extra run source).
pub const RUN_PATH: &str = "/world/runs/slam_rs";
/// Stage timings, one series per stage.
pub const TIMINGS_PATH: &str = "/timings";
/// The `/timings` series in order; a stage that did not run for a frameset is NaN there (no point).
pub const TIMING_SERIES: [&str; 9] = ["downsample_ms", "pipeline_ms", "slam_ms", "hands_ms", "detnet_ms", "crops_ms", "keynet_ms", "fit_ms", "log_ms"];
/// Frame rate of the framesets reaching the logger.
pub const FPS_PATH: &str = "/fps";
/// The logger's own counters.
pub const COUNTERS_PATH: &str = "/log";
/// SLAM's status text, logged when it changes.
pub const SLAM_STATUS_PATH: &str = "/slam/status";
/// The logger's own warnings, under [`COUNTERS_PATH`].
pub const LOG_STATUS_PATH: &str = "/log/status";
/// Names of the [`COUNTERS_PATH`] series.
pub const COUNTER_NAMES: [&str; 3] = ["framesets_dropped", "preview_dropped", "preview_queue"];

/// One `/timings` row in [`TIMING_SERIES`] order: the frameset's stage timings and the logger's own time per frameset.
pub fn timing_row(t: &FrameTimings, log_ms: f64) -> [f64; TIMING_SERIES.len()] {
    [t.downsample_ms, t.pipeline_ms, t.slam_ms, t.hands_ms, t.detnet_ms, t.crops_ms, t.keynet_ms, t.fit_ms, log_ms]
}

/// One [`COUNTERS_PATH`] row in [`COUNTER_NAMES`] order.
pub(super) fn counter_row(counters: &LogCounters) -> [f64; COUNTER_NAMES.len()] {
    let get = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64;
    [get(&counters.framesets_dropped), get(&counters.preview_dropped), get(&counters.preview_queued)]
}

/// `/world/rig_00/cam_NN`.
pub fn cam_path(camera: usize) -> String {
    format!("{RIG_PATH}/cam_{camera:02}")
}

/// `/world/rig_00/cam_NN/pinhole`.
pub fn pinhole_path(camera: usize) -> String {
    format!("{}/pinhole", cam_path(camera))
}

/// `/world/rig_00/cam_NN/pinhole/video`.
pub fn video_path(camera: usize) -> String {
    format!("{}/video", pinhole_path(camera))
}

/// `/world/rig_00/cam_NN/pinhole/image`.
pub fn image_path(camera: usize) -> String {
    format!("{}/image", pinhole_path(camera))
}

/// `/world/hands/{side}/keypoints`.
pub fn hand3d_path(side: usize) -> String {
    format!("/world/hands/{}/keypoints", SIDES[side])
}

/// `/world/hands/{side}/mesh`.
pub fn mesh_path(side: usize) -> String {
    format!("/world/hands/{}/mesh", SIDES[side])
}

/// How much of the hand pipeline the camera panes show.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum HandOverlays {
    /// The fitted hands projected into every camera that sees them, and each untracked hand's strongest DetNet box.
    #[default]
    Fit,
    /// Also every KeyNet view (dots coloured by keypoint confidence, the crop outline coloured and labelled by what the tracker
    /// did with it) and every DetNet answer from [`DETNET_SHOWN`] up (magenta above the threshold, grey below).
    Debug,
    /// Also KeyNet's relative depths (a component on the dots) and the predicted pose tracked hands were cropped from.
    Verbose,
}

impl FromStr for HandOverlays {
    type Err = LogError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "fit" => Ok(Self::Fit),
            "debug" => Ok(Self::Debug),
            "verbose" => Ok(Self::Verbose),
            other => Err(LogError::Invalid(format!("hand overlays {other:?}: expected fit, debug or verbose"))),
        }
    }
}

/// One kind of hand overlay on a camera pane (the last entity path component).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Layer {
    /// The fitted hand, projected.
    Fit,
    /// The predicted pose a tracked hand's crops were planned from, projected.
    Predicted,
    /// KeyNet's keypoints.
    KeyNet,
    /// The KeyNet crop's outline.
    Crop,
    /// DetNet's box.
    DetNet,
}

impl Layer {
    /// Every layer.
    pub const ALL: [Layer; 5] = [Layer::Fit, Layer::Predicted, Layer::KeyNet, Layer::Crop, Layer::DetNet];

    /// The entity name.
    pub fn name(self) -> &'static str {
        match self {
            Layer::Fit => "fit",
            Layer::Predicted => "predicted",
            Layer::KeyNet => "keynet",
            Layer::Crop => "crop",
            Layer::DetNet => "detnet",
        }
    }
}

/// `.../pinhole/hands/{side}/{layer}`.
pub fn pane_path(camera: usize, side: usize, layer: Layer) -> String {
    format!("{}/hands/{}/{}", pinhole_path(camera), SIDES[side], layer.name())
}

/// simplecv's `confidence_scores_to_rgb`: red at 0, yellow at 0.5, green at 1 (clipped; NaN is 0; channels truncated as numpy's
/// `astype(np.uint8)` does).
pub fn confidence_rgb(confidence: f32) -> [u8; 3] {
    let c = if confidence.is_nan() { 0.0 } else { confidence.clamp(0.0, 1.0) };
    if c <= 0.5 { [255, (c * 2.0 * 255.0) as u8, 0] } else { [((1.0 - (c - 0.5) * 2.0) * 255.0) as u8, 255, 0] }
}

/// Map a full-resolution pixel to the 640x360 small image (pixel centres: `(u + 0.5) * s - 0.5`, the area /3 of SPEC).
pub fn small_from_full(uv: [f32; 2], scale: [f32; 2]) -> [f32; 2] {
    [(uv[0] + 0.5) * scale[0] - 0.5, (uv[1] + 0.5) * scale[1] - 0.5]
}

/// A DetNet circle `(cx, cy, r)` in net-frame pixels ([`crate::hands::HandOutput::detnet_circle`]) to a centre and half size
/// in the small image, which DetNet's letterbox ([`BarLetterbox::robocap`], the tracker's) places at its padding.
pub fn small_box_from_detnet(circle: [f32; 3]) -> ([f32; 2], [f32; 2]) {
    let letterbox = BarLetterbox::robocap();
    let [cx, cy, r] = circle;
    ([cx - letterbox.pad_x as f32, cy - letterbox.pad_y as f32], [r, r])
}

/// The camera's pose in the rig (identity for a non-finite calibration, which the hands refuse anyway).
fn rig_from_cam(cam_from_rig: &[[f64; 4]; 4]) -> Isometry3<f64> {
    let flat: [f64; 16] = std::array::from_fn(|i| cam_from_rig[i / 4][i % 4]);
    isometry_from_matrix(&flat).map_or_else(Isometry3::identity, |cam_from_rig| cam_from_rig.inverse())
}

/// A pose as Rerun translation + quaternion (xyzw).
pub fn transform_from_isometry(pose: &Isometry3<f64>) -> rerun::Transform3D {
    let t = pose.translation.vector;
    let q = pose.rotation.quaternion();
    rerun::Transform3D::from_translation_rotation([t.x as f32, t.y as f32, t.z as f32], rerun::Quaternion::from_xyzw([q.i as f32, q.j as f32, q.k as f32, q.w as f32]))
}

/// The per-camera scale from the rig's calibration size to the 640x360 small image.
pub fn small_scale(rig: &Rig, camera: usize) -> [f32; 2] {
    rig.cameras.get(camera).map_or([1.0 / 3.0; 2], |c| [SMALL_SIZE.width as f32 / c.width as f32, SMALL_SIZE.height as f32 / c.height as f32])
}

fn rgb(c: [u8; 3]) -> rerun::Color {
    rerun::Color::from_rgb(c[0], c[1], c[2])
}

/// Log the static scene: world axes and the hand classes, the six cameras (pose in the rig, pinhole at the 640x360 video size),
/// the video codec when `h264` is set, and every per-entity constant the frameset rows then leave out.
///
/// # Errors
///
/// The SDK's error if a component fails to serialise.
pub fn log_static_scene(rec: &RecordingStream, rig: &Rig, h264: bool) -> Result<(), rerun::RecordingStreamError> {
    rec.log_static(TIMINGS_PATH, &rerun::SeriesLines::new().with_names(TIMING_SERIES))?;
    rec.log_static("/world", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())?;
    let connections = rerun::encodings::KeypointPair::vec_from(HAND_CONNECTIONS);
    let class = |id: u16, label: &str, color: [u8; 3]| rerun::encodings::ClassDescription {
        info: rerun::encodings::AnnotationInfo { id, label: Some(label.into()), color: Some(rerun::encodings::Rgba32::from_rgb(color[0], color[1], color[2])) },
        keypoint_annotations: Vec::new(),
        keypoint_connections: connections.clone(),
    };
    let classes = [class(0, SIDES[0], HAND_COLORS[0]), class(1, SIDES[1], HAND_COLORS[1]), class(PREDICTED_CLASS, "predicted", PREDICTED_COLOR)];
    rec.log_static("/world", &rerun::AnnotationContext::new(classes))?;
    // The cap's own axes, so its pose reads at room scale (the cap mesh rides in the display asset when it is readable).
    rec.log_static(format!("{RIG_PATH}/axes"), &rerun::TransformAxes3D::new(RIG_AXIS_LENGTH))?;
    for (camera, cam) in rig.cameras.iter().enumerate().take(NUM_CAMERAS) {
        rec.log_static(cam_path(camera), &transform_from_isometry(&rig_from_cam(&cam.cam_from_rig)))?;
        let scale = small_scale(rig, camera);
        let principal = small_from_full([cam.principal[0] as f32, cam.principal[1] as f32], scale);
        let (fx, fy) = (cam.focal[0] as f32 * scale[0], cam.focal[1] as f32 * scale[1]);
        // Column-major image_from_camera.
        let pinhole = rerun::Pinhole::new([[fx, 0.0, 0.0], [0.0, fy, 0.0], [principal[0], principal[1], 1.0]])
            .with_resolution([SMALL_SIZE.width as f32, SMALL_SIZE.height as f32])
            .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
            .with_image_plane_distance(IMAGE_PLANE_DISTANCE);
        rec.log_static(pinhole_path(camera), &pinhole)?;
        if h264 {
            rec.log_static(video_path(camera), &rerun::VideoStream::new(rerun::components::VideoCodec::H264))?;
        }
        // Only what never changes is static: a static component wins over every logged row of it (keypoint ids, colours and
        // labels vary per frameset here).
        for side in 0..2 {
            let fit = rerun::Points2D::update_fields().with_class_ids([side as u16]).with_radii([rerun::Radius::new_ui_points(1.5)]).with_show_labels(false);
            rec.log_static(pane_path(camera, side, Layer::Fit), &fit)?;
            let predicted =
                rerun::Points2D::update_fields().with_class_ids([PREDICTED_CLASS]).with_radii([rerun::Radius::new_ui_points(1.0)]).with_show_labels(false);
            rec.log_static(pane_path(camera, side, Layer::Predicted), &predicted)?;
            let keynet = rerun::Points2D::update_fields().with_radii([rerun::Radius::new_ui_points(3.5)]).with_show_labels(false);
            rec.log_static(pane_path(camera, side, Layer::KeyNet), &keynet)?;
            rec.log_static(pane_path(camera, side, Layer::Crop), &rerun::LineStrips2D::update_fields().with_radii([rerun::Radius::new_ui_points(1.0)]))?;
        }
    }
    for (side, [r, g, b, a]) in HAND_ALBEDO.into_iter().enumerate() {
        let points = rerun::Points3D::update_fields().with_class_ids([side as u16]).with_keypoint_ids(0..21u16).with_radii([0.005]).with_show_labels(false);
        rec.log_static(hand3d_path(side), &points)?;
        rec.log_static(mesh_path(side), &rerun::Mesh3D::update_fields().with_albedo_factor(rerun::Rgba32::from_unmultiplied_rgba(r, g, b, a)))?;
    }
    for path in [format!("{RUN_PATH}/trajectory"), format!("{RUN_PATH}/trail")] {
        rec.log_static(path, &rerun::LineStrips3D::update_fields().with_colors([rgb(TRAJECTORY_COLOR)]).with_radii([0.004]))?;
    }
    rec.log_static(FPS_PATH, &rerun::SeriesLines::new().with_names(["fps"]))?;
    rec.log_static(COUNTERS_PATH, &rerun::SeriesLines::new().with_names(COUNTER_NAMES))?;
    Ok(())
}
