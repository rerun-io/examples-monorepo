//! The producer side of the scene: what each frameset draws, as a [`SceneSnapshot`] (trajectory thinning, timings, the hands
//! in 3D and the overlays on the camera panes, traced through each camera's lens model).

use std::sync::Arc;

use nalgebra::{Isometry3, Vector2, Vector3};

use super::{
    COUNTER_NAMES, COUNTERS_EVERY, CROP_OUTLINE_STEPS, DETNET_COLOR, DETNET_REJECTED_COLOR, DETNET_SHOWN, FITTED_COLOR, HandOverlays, Layer, PaneDraw,
    PaneItem, REJECTED_COLOR, SIDES, TIMING_SERIES, TIMINGS_EVERY, TRAJECTORY_EVERY, small_box_from_detnet, small_from_full, small_scale,
};
use crate::frame::{NUM_CAMERAS, Rig};
use crate::hands::camera::{RigCameraModel, in_front, rig_models};
use crate::hands::mesh::HandMesh;
use crate::hands::model::{GenericHandModel, mirror};
use crate::hands::perspective::{CROP_SIZE, CropCamera};
use crate::hands::{CropSource, HandFrameResult, HandOutput, KeyNetView, ViewOutcome};

/// Landmarks (world, metres) through one camera's lens into the small image: the points in front of the camera and their
/// keypoint ids, when at least one lands inside the image.
fn project_landmarks(model: &RigCameraModel, world_from_rig: &Isometry3<f64>, landmarks: &[[f64; 3]; 21], scale: [f32; 2]) -> Option<(Vec<[f32; 2]>, Vec<u16>)> {
    let mut points = Vec::with_capacity(21);
    let mut ids = Vec::with_capacity(21);
    let mut inside = false;
    for (id, point) in landmarks.iter().enumerate() {
        let p_cam = model.cam_from_world_point(world_from_rig, &Vector3::new(point[0], point[1], point[2]));
        let Some(px) = in_front(&p_cam).then(|| model.project(&p_cam)).flatten().filter(|px| px.iter().all(|v| v.is_finite())) else { continue };
        inside |= model.inside_image(&px);
        points.push(small_from_full([px.x as f32, px.y as f32], scale));
        ids.push(id as u16);
    }
    inside.then_some((points, ids))
}

/// The border of a KeyNet crop traced through the camera's lens into the small image, closed; empty for an unusable crop.
fn crop_outline(model: &RigCameraModel, crop: &CropCamera, scale: [f32; 2]) -> Vec<[f32; 2]> {
    if !crop.usable() {
        return Vec::new();
    }
    let (lo, hi) = (-0.5, CROP_SIZE as f64 - 0.5);
    let corners = [[lo, lo], [hi, lo], [hi, hi], [lo, hi], [lo, lo]];
    let mut outline = Vec::with_capacity(4 * CROP_OUTLINE_STEPS + 1);
    for edge in corners.windows(2) {
        for step in 0..CROP_OUTLINE_STEPS {
            let t = step as f64 / CROP_OUTLINE_STEPS as f64;
            let uv = Vector2::new(edge[0][0] + t * (edge[1][0] - edge[0][0]), edge[0][1] + t * (edge[1][1] - edge[0][1]));
            let ray = crop.from_crop(&uv);
            if let Some(px) = in_front(&ray).then(|| model.project(&ray)).flatten().filter(|px| px.iter().all(|v| v.is_finite())) {
                outline.push(small_from_full([px.x as f32, px.y as f32], scale));
            }
        }
    }
    if let Some(&first) = outline.first() {
        outline.push(first);
    }
    outline
}

/// What a crop label says about a KeyNet view: `L keynet 0.98 ok (pose)`.
fn crop_label(side: usize, view: &KeyNetView, reported: bool) -> String {
    let verdict = match view.outcome {
        ViewOutcome::Fitted if reported => "ok".to_string(),
        ViewOutcome::Fitted => "ok, tentative".to_string(),
        ViewOutcome::LowPresence => "low presence".to_string(),
        ViewOutcome::LostPair => "lost its pair".to_string(),
        ViewOutcome::FitResidual { rms_px, limit_px } => format!("fit rms {rms_px:.1} > {limit_px:.1} px"),
        ViewOutcome::FitFailed => "fit failed".to_string(),
    };
    let source = match view.crop_source {
        CropSource::DetNet => "detnet",
        CropSource::Recrop => "recrop",
        CropSource::Pose => "pose",
    };
    format!("{} keynet {:.2} {verdict} ({source})", SIDES[side][..1].to_ascii_uppercase(), view.presence)
}

/// Producer state: trajectory thinning, timing series and the hand overlays (with each camera's lens model and its scale to
/// the small image).
#[derive(Default)]
pub struct RecordState {
    framesets: u64,
    edge_start: Option<[f32; 3]>,
    since_edge: u64,
    models: Vec<RigCameraModel>,
    scales: [[f32; 2]; NUM_CAMERAS],
    overlays: HandOverlays,
    mesh: Option<(HandMesh, GenericHandModel)>,
}

impl std::fmt::Debug for RecordState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordState").field("framesets", &self.framesets).field("cameras", &self.models.len()).field("overlays", &self.overlays).finish()
    }
}

/// One frameset's scene as the producer prepared it: everything present, with sampling and timestamps fixed before queueing.
/// Each sink's [`super::DeliveredState`] turns it into the [`super::FrameRecord`] it writes.
#[derive(Debug, Default)]
pub struct SceneSnapshot {
    /// `video_time`, nanoseconds.
    pub t_ns: i64,
    /// `world_from_rig` when SLAM has a pose.
    pub pose: Option<Isometry3<f64>>,
    /// New trajectory edge (last edge's end to the current rig position).
    pub edge: Option<[[f32; 3]; 2]>,
    /// The reported hands in 3D, by side.
    pub hands: [Option<HandDraw>; 2],
    /// Hand overlays on the camera panes.
    pub panes: Vec<PaneItem>,
    /// Stage timings in [`TIMING_SERIES`] order, every [`TIMINGS_EVERY`] framesets.
    pub timings: Option<[f64; TIMING_SERIES.len()]>,
    /// Frames per second, with the timings.
    pub fps: Option<f64>,
    /// The logger's counters in [`COUNTER_NAMES`] order, every [`COUNTERS_EVERY`] framesets.
    pub counters: Option<[f64; COUNTER_NAMES.len()]>,
    /// SLAM's status text.
    pub status: String,
    /// A message from the logger itself (e.g. saving stopped for low disk).
    pub notice: Option<String>,
}

/// One reported hand in 3D.
#[derive(Debug)]
pub struct HandDraw {
    /// Its 21 landmarks, world metres.
    pub points: Box<[[f32; 3]; 21]>,
    /// Its skinned mesh, when the hand has a pose and the mesh loaded.
    pub mesh: Option<SkinnedMesh>,
}

/// UmeTrack's generic hand mesh skinned on a fitted pose.
#[derive(Debug)]
pub struct SkinnedMesh {
    /// The vertices, world metres.
    pub vertices: Vec<[f32; 3]>,
    /// The triangle list (shared by every frameset).
    pub triangles: Arc<[[u32; 3]]>,
}

/// The slow per-frameset signals, as the worker measured them.
#[derive(Clone, Debug, Default)]
pub struct Signals {
    /// Stage timings in [`TIMING_SERIES`] order.
    pub timings: [f64; TIMING_SERIES.len()],
    /// Frames per second, when known.
    pub fps: Option<f64>,
    /// The logger's counters, in [`COUNTER_NAMES`] order.
    pub counters: [f64; COUNTER_NAMES.len()],
}

impl RecordState {
    /// For a rig: the overlays draw through each camera's own lens model and scale from its calibration size to the small
    /// image. Without the lens models nothing is projected into the panes (fit, predicted pose, crop outlines), and without
    /// the hand mesh no mesh is drawn; either is said once on stderr.
    pub fn new(rig: &Rig, overlays: HandOverlays) -> Self {
        let mesh = HandMesh::load().and_then(|mesh| Ok((mesh, GenericHandModel::load()?)));
        let mesh = mesh.map_err(|error| eprintln!("robocap-live log: no hand mesh: {error}")).ok();
        let models = rig_models(rig).unwrap_or_else(|error| {
            eprintln!("robocap-live log: no projections into the panes: {error}");
            Vec::new()
        });
        let scales = std::array::from_fn(|camera| small_scale(rig, camera));
        Self { models, scales, overlays, mesh, ..Self::default() }
    }

    /// Build one frameset's snapshot (without the worker's notice); each sink's [`super::DeliveredState`] turns it into the record
    /// it writes.
    ///
    /// # Arguments
    ///
    /// * `t_ns` - The frameset's `video_time`.
    /// * `pose` - SLAM's `world_from_rig`, if it has one.
    /// * `status` - SLAM's status text (logged when it changes).
    /// * `hands` - The tracker's result, if hands ran.
    /// * `signals` - Timings, fps and counters (thinned here).
    pub(in crate::log) fn prepare(
        &mut self,
        t_ns: i64,
        pose: Option<Isometry3<f64>>,
        status: &str,
        hands: Option<&HandFrameResult>,
        signals: &Signals,
    ) -> SceneSnapshot {
        let n = self.framesets;
        self.framesets += 1;
        let mut snapshot = SceneSnapshot { t_ns, pose, status: status.to_string(), ..SceneSnapshot::default() };
        match pose {
            Some(pose) => {
                let t = pose.translation.vector;
                let position = [t.x as f32, t.y as f32, t.z as f32];
                if position.iter().all(|v| v.is_finite()) {
                    match self.edge_start {
                        None => {
                            self.edge_start = Some(position);
                            self.since_edge = 0;
                        }
                        Some(start) => {
                            self.since_edge += 1;
                            if self.since_edge >= TRAJECTORY_EVERY {
                                snapshot.edge = Some([start, position]);
                                self.edge_start = Some(position);
                                self.since_edge = 0;
                            }
                        }
                    }
                }
            }
            None => {
                self.edge_start = None;
            }
        }
        self.hands(&mut snapshot, hands, pose);
        if n % TIMINGS_EVERY == 0 {
            snapshot.timings = Some(signals.timings);
            snapshot.fps = signals.fps;
        }
        if n % COUNTERS_EVERY == 0 {
            snapshot.counters = Some(signals.counters);
        }
        snapshot
    }

    fn hands(&mut self, snapshot: &mut SceneSnapshot, hands: Option<&HandFrameResult>, pose: Option<Isometry3<f64>>) {
        // The landmarks are in the world of the pose the hands step used.
        let world_from_rig = hands.and_then(|h| h.world_from_rig).or(pose).unwrap_or_else(Isometry3::identity);
        for (side, name) in SIDES.iter().enumerate() {
            let Some(hand) = hands.map(|h| &h.hands[side]) else { continue };
            if let Some(world) = hand.landmarks_world.as_ref().filter(|_| hand.reported) {
                let mesh = match (&hand.pose, &self.mesh) {
                    (Some(pose), Some((mesh, generic))) => {
                        let phi = hands.map_or(1.0, |h| h.scale);
                        Some(SkinnedMesh { vertices: mesh.skin(generic, phi, pose, mirror(side)), triangles: mesh.triangles().clone() })
                    }
                    _ => None,
                };
                snapshot.hands[side] = Some(HandDraw { points: Box::new(world.map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])), mesh });
                self.project(snapshot, side, Layer::Fit, &world_from_rig, world);
            }
            if self.overlays >= HandOverlays::Verbose
                && let Some(predicted) = &hand.predicted_landmarks_world
            {
                self.project(snapshot, side, Layer::Predicted, &world_from_rig, predicted);
            }
            let debug = self.overlays >= HandOverlays::Debug;
            if debug {
                self.keynet(snapshot, side, hand);
            }
            // Debug: every DetNet answer from DETNET_SHOWN up. Fit: the hand's strongest accepted answer, on its DetNet camera.
            let shown = hand.detnet_hits.iter().filter(|hit| if debug { hit.probability >= DETNET_SHOWN } else { Some(hit.camera) == hand.detnet_camera });
            for hit in shown.filter(|hit| hit.camera < NUM_CAMERAS && hit.circle.iter().all(|v| v.is_finite())) {
                let (center, half) = small_box_from_detnet(hit.circle);
                let color = if hit.accepted { DETNET_COLOR } else { DETNET_REJECTED_COLOR };
                let label = format!("DetNet {name} {:.2}", hit.probability);
                snapshot.panes.push(PaneItem { camera: hit.camera, side, layer: Layer::DetNet, draw: PaneDraw::Box { center, half, color, label } });
            }
        }
    }

    /// Landmarks projected into every camera that sees them.
    fn project(&self, snapshot: &mut SceneSnapshot, side: usize, layer: Layer, world_from_rig: &Isometry3<f64>, landmarks: &[[f64; 3]; 21]) {
        for (camera, model) in self.models.iter().enumerate().take(NUM_CAMERAS) {
            if let Some((points, keypoint_ids)) = project_landmarks(model, world_from_rig, landmarks, self.scales[camera]) {
                snapshot.panes.push(PaneItem { camera, side, layer, draw: PaneDraw::Skeleton { points, keypoint_ids } });
            }
        }
    }

    /// Each KeyNet view: its keypoints, and its crop's outline with what the tracker did with it.
    fn keynet(&self, snapshot: &mut SceneSnapshot, side: usize, hand: &HandOutput) {
        for view in hand.keynet_views.iter().filter(|v| v.camera < NUM_CAMERAS) {
            let points = Box::new(view.keypoints_px.map(|uv| small_from_full(uv, self.scales[view.camera])));
            let d_rel_mm = (self.overlays >= HandOverlays::Verbose).then(|| Box::new(view.d_rel_mm));
            snapshot.panes.push(PaneItem { camera: view.camera, side, layer: Layer::KeyNet, draw: PaneDraw::KeyNet { points, confidence: Box::new(view.confidence), d_rel_mm } });
            let outline = match (self.models.get(view.camera), &view.crop) {
                (Some(model), Some(crop)) => crop_outline(model, crop, self.scales[view.camera]),
                _ => Vec::new(),
            };
            if !outline.is_empty() {
                let color = if view.outcome == ViewOutcome::Fitted { FITTED_COLOR } else { REJECTED_COLOR };
                let label = crop_label(side, view, hand.reported);
                snapshot.panes.push(PaneItem { camera: view.camera, side, layer: Layer::Crop, draw: PaneDraw::Outline { points: outline, color, label } });
            }
        }
    }
}
