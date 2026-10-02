//! The sink side of the scene: the record one frameset logs, each sink's delivery state (clears, mesh topology, status), and
//! how a record is written into a stream.

use std::collections::HashSet;
use std::sync::Arc;

use rerun::RecordingStream;

use super::{
    COUNTERS_PATH, CROP_LABEL_HALF_SIZE, Content, FPS_PATH, LOG_STATUS_PATH, Layer, RIG_PATH, RUN_PATH, SLAM_STATUS_PATH, SceneSnapshot, TIMELINE,
    TIMINGS_PATH, confidence_rgb, hand3d_path, mesh_path, pane_path, rgb, transform_from_isometry,
};
use crate::frame::{NUM_CAMERAS, SMALL_SIZE};

/// What a sink does with one hand in 3D this frameset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hand3d<'a> {
    /// Draw the hand.
    Draw {
        /// Its 21 world landmarks.
        points: &'a [[f32; 3]; 21],
        /// Its skinned mesh's vertices, when it has one.
        mesh: Option<&'a [[f32; 3]]>,
        /// The mesh's triangle list, with the hand's first mesh on this sink since its last Clear or since the stream began.
        triangles: Option<&'a [[u32; 3]]>,
    },
    /// The hand was drawn on this sink and is gone now.
    Clear,
    /// Nothing to draw or clear.
    Keep,
}

/// What one pane overlay draws, in small-image pixels.
#[derive(Clone, Debug, PartialEq)]
pub enum PaneDraw {
    /// Projected landmarks with their keypoint ids (the skeleton comes from the annotation context).
    Skeleton {
        /// Positions.
        points: Vec<[f32; 2]>,
        /// Keypoint id of each position.
        keypoint_ids: Vec<u16>,
    },
    /// KeyNet's keypoints with their confidences.
    KeyNet {
        /// The 21 keypoints.
        points: Box<[[f32; 2]; 21]>,
        /// Per keypoint: the heatmap's peak value.
        confidence: Box<[f32; 21]>,
        /// Per keypoint: relative depth, millimetres ([`HandOverlays::Verbose`]).
        d_rel_mm: Option<Box<[f32; 21]>>,
    },
    /// A closed outline.
    Outline {
        /// Positions, the first repeated at the end.
        points: Vec<[f32; 2]>,
        /// Colour.
        color: [u8; 3],
        /// Label.
        label: String,
    },
    /// A box.
    Box {
        /// Centre.
        center: [f32; 2],
        /// Half size.
        half: [f32; 2],
        /// Colour.
        color: [u8; 3],
        /// Label.
        label: String,
    },
}

/// One overlay on one camera pane for one hand.
#[derive(Clone, Debug, PartialEq)]
pub struct PaneItem {
    /// Camera index.
    pub camera: usize,
    /// Hand slot.
    pub side: usize,
    /// Which overlay.
    pub layer: Layer,
    /// What it draws.
    pub draw: PaneDraw,
}

/// Everything one delivered frameset logs besides video: the producer's snapshot and this sink's scene transitions.
#[derive(Debug)]
pub struct FrameRecord<'a> {
    /// What the producer prepared.
    pub scene: &'a SceneSnapshot,
    /// The SLAM pose was lost since this sink's last frameset (clear the rig transform).
    pub pose_lost: bool,
    /// The two hands in 3D, by side.
    pub hands: [Hand3d<'a>; 2],
    /// Pane overlays this sink drew before and gone now, as `(camera, side, layer)`.
    pub pane_clears: Vec<(usize, usize, Layer)>,
    /// SLAM's status text, when it changed on this sink.
    pub status: Option<&'a str>,
    /// The logger's own message, when it changed on this sink; logged as a warning on [`LOG_STATUS_PATH`].
    pub notice: Option<&'a str>,
}

/// What one sink has delivered, independent of the producer and other sinks.
#[derive(Debug, Default)]
pub(in crate::log) struct DeliveredState {
    had_pose: bool,
    drawn3d: [bool; 2],
    /// The mesh triangles went out since the hand's last Clear.
    topology_sent: [bool; 2],
    drawn_panes: HashSet<(usize, usize, Layer)>,
    last_status: Option<String>,
    last_notice: Option<String>,
}

impl DeliveredState {
    pub(in crate::log) fn reconnected() -> Self {
        let all = (0..NUM_CAMERAS).flat_map(|camera| (0..2).flat_map(move |side| Layer::ALL.map(|layer| (camera, side, layer)))).collect();
        Self { had_pose: true, drawn3d: [true; 2], drawn_panes: all, ..Self::default() }
    }

    /// The record this sink writes for `scene`, and what it has delivered after it.
    pub(in crate::log) fn record<'a>(&mut self, scene: &'a SceneSnapshot) -> FrameRecord<'a> {
        let pose_lost = self.had_pose && scene.pose.is_none();
        self.had_pose = scene.pose.is_some();
        let hands = std::array::from_fn(|side| {
            let was_drawn = std::mem::replace(&mut self.drawn3d[side], scene.hands[side].is_some());
            match &scene.hands[side] {
                Some(hand) => {
                    let mesh = hand.mesh.as_ref();
                    let triangles = mesh.filter(|_| !self.topology_sent[side]).map(|mesh| &mesh.triangles[..]);
                    self.topology_sent[side] |= mesh.is_some();
                    Hand3d::Draw { points: &hand.points, mesh: mesh.map(|mesh| &mesh.vertices[..]), triangles }
                }
                None if was_drawn => {
                    self.topology_sent[side] = false;
                    Hand3d::Clear
                }
                None => Hand3d::Keep,
            }
        });
        let drawn: HashSet<(usize, usize, Layer)> = scene.panes.iter().map(|item| (item.camera, item.side, item.layer)).collect();
        let mut pane_clears: Vec<(usize, usize, Layer)> = self.drawn_panes.difference(&drawn).copied().collect();
        pane_clears.sort_unstable();
        self.drawn_panes = drawn;
        let status = (self.last_status.as_deref() != Some(scene.status.as_str())).then(|| {
            self.last_status = Some(scene.status.clone());
            scene.status.as_str()
        });
        let notice = if scene.notice == self.last_notice {
            None
        } else {
            self.last_notice.clone_from(&scene.notice);
            scene.notice.as_deref()
        };
        FrameRecord { scene, pose_lost, hands, pane_clears, status, notice }
    }
}

/// simplecv's `Points2DWithConfidence` components (`simplecv.KeypointConfidence2D:confidences` and `:average_confidence`), so a
/// recording reads the same in its tools.
fn confidence_components(confidence: &[f32; 21]) -> [rerun::SerializedComponentBatch; 2] {
    use rerun::external::arrow::array::Float32Array;
    let finite: Vec<f32> = confidence.iter().copied().filter(|c| !c.is_nan()).collect();
    let mean = if finite.is_empty() { f32::NAN } else { finite.iter().sum::<f32>() / finite.len() as f32 };
    let descriptor = |component: &'static str, component_type: &'static str| {
        rerun::ComponentDescriptor::partial(component).with_archetype("simplecv.KeypointConfidence2D".into()).with_component_type(component_type.into())
    };
    [
        rerun::SerializedComponentBatch::new(
            Arc::new(Float32Array::from(confidence.to_vec())),
            descriptor("simplecv.KeypointConfidence2D:confidences", "simplecv.components.KeypointConfidence"),
        ),
        rerun::SerializedComponentBatch::new(
            Arc::new(Float32Array::from(vec![mean])),
            descriptor("simplecv.KeypointConfidence2D:average_confidence", "simplecv.components.KeypointConfidenceMean"),
        ),
    ]
}

/// KeyNet's relative depths as a component on its dots (`robocap.KeyNet2D:relative_depth_mm`).
fn relative_depth_component(d_rel_mm: &[f32; 21]) -> rerun::SerializedComponentBatch {
    rerun::SerializedComponentBatch::new(
        Arc::new(rerun::external::arrow::array::Float32Array::from(d_rel_mm.to_vec())),
        rerun::ComponentDescriptor::partial("robocap.KeyNet2D:relative_depth_mm")
            .with_archetype("robocap.KeyNet2D".into())
            .with_component_type("robocap.components.RelativeDepthMm".into()),
    )
}

/// Write one record into a stream, on `video_time` (this thread's time is set here).
///
/// # Errors
///
/// The SDK's error if a component fails to serialise.
pub fn write_record(rec: &RecordingStream, record: &FrameRecord<'_>, content: Content) -> Result<(), rerun::RecordingStreamError> {
    let scene = record.scene;
    let full = content == Content::Full;
    rec.set_time(TIMELINE, rerun::TimeCell::from_duration_nanos(scene.t_ns));
    if full {
        if let Some(pose) = &scene.pose {
            rec.log(RIG_PATH, &transform_from_isometry(pose))?;
        } else if record.pose_lost {
            rec.log(RIG_PATH, &rerun::Transform3D::clear_fields())?;
        }
        if let Some(edge) = scene.edge {
            let line = rerun::LineStrips3D::update_fields().with_strips([edge.to_vec()]);
            rec.log(format!("{RUN_PATH}/trajectory"), &line)?;
            rec.log(format!("{RUN_PATH}/trail"), &line)?;
        }
    }
    for (side, hand) in record.hands.iter().enumerate() {
        match hand {
            Hand3d::Draw { points, mesh, triangles } => {
                rec.log(hand3d_path(side), &rerun::Points3D::update_fields().with_positions(points.iter().copied()))?;
                if let Some(vertices) = mesh {
                    let mut mesh = rerun::Mesh3D::update_fields().with_vertex_positions(vertices.iter().copied());
                    if let Some(triangles) = triangles {
                        mesh = mesh.with_triangle_indices(triangles.iter().map(|&[a, b, c]| rerun::datatypes::UVec3D::new(a, b, c)));
                    }
                    rec.log(mesh_path(side), &mesh)?;
                }
            }
            Hand3d::Clear => {
                rec.log(hand3d_path(side), &rerun::Clear::flat())?;
                rec.log(mesh_path(side), &rerun::Clear::flat())?;
            }
            Hand3d::Keep => {}
        }
    }
    // The overlays are made in 640x360 pane pixels; a layer logs them under the base layer's full-resolution pinholes.
    let k = content.pane_scale();
    let px = |p: [f32; 2]| [p[0] * k, p[1] * k];
    for item in &scene.panes {
        let path = pane_path(item.camera, item.side, item.layer);
        match &item.draw {
            PaneDraw::Skeleton { points, keypoint_ids } => {
                rec.log(path, &rerun::Points2D::update_fields().with_positions(points.iter().copied().map(px)).with_keypoint_ids(keypoint_ids.iter().copied()))?;
            }
            PaneDraw::KeyNet { points, confidence, d_rel_mm } => {
                let dots = rerun::Points2D::update_fields()
                    .with_positions(points.iter().copied().map(px))
                    .with_colors(confidence.iter().map(|&c| rgb(confidence_rgb(c))));
                let [confidences, average] = confidence_components(confidence);
                match d_rel_mm {
                    Some(depths) => {
                        let depths = relative_depth_component(depths);
                        rec.log(path, &[&dots as &dyn rerun::AsComponents, &confidences, &average, &depths])?;
                    }
                    None => rec.log(path, &[&dots as &dyn rerun::AsComponents, &confidences, &average])?,
                }
            }
            PaneDraw::Outline { points, color, label } => {
                // Rerun puts a strip's label at its centre, on the hand: the label rides a zero-length second strip at the
                // outline's top-left corner instead, kept inside the image (a crop at the edge reaches past it).
                let corner = points.iter().copied().min_by(|a, b| (a[0] + a[1]).total_cmp(&(b[0] + b[1]))).unwrap_or_default();
                let [half_w, half_h] = CROP_LABEL_HALF_SIZE;
                let corner = px([corner[0].clamp(half_w, SMALL_SIZE.width as f32 - half_w), corner[1].clamp(half_h, SMALL_SIZE.height as f32 - half_h)]);
                let strip = rerun::LineStrips2D::update_fields()
                    .with_strips([points.iter().copied().map(px).collect(), vec![corner, corner]])
                    .with_colors([rgb(*color)])
                    .with_labels(["", label.as_str()]);
                rec.log(path, &strip)?;
            }
            PaneDraw::Box { center, half, color, label } => {
                let boxes = rerun::Boxes2D::update_fields().with_centers([px(*center)]).with_half_sizes([px(*half)]).with_colors([rgb(*color)]).with_labels([label.as_str()]);
                rec.log(path, &boxes)?;
            }
        }
    }
    for &(camera, side, layer) in &record.pane_clears {
        rec.log(pane_path(camera, side, layer), &rerun::Clear::flat())?;
    }
    if !full {
        return Ok(());
    }
    if let Some(values) = &scene.timings {
        rec.log(TIMINGS_PATH, &rerun::Scalars::new(values.iter().copied()))?;
    }
    if let Some(fps) = scene.fps {
        rec.log(FPS_PATH, &rerun::Scalars::single(fps))?;
    }
    if let Some(counters) = scene.counters {
        rec.log(COUNTERS_PATH, &rerun::Scalars::new(counters))?;
    }
    if let Some(status) = record.status {
        rec.log(SLAM_STATUS_PATH, &rerun::TextLog::new(status))?;
    }
    if let Some(notice) = record.notice {
        rec.log(LOG_STATUS_PATH, &rerun::TextLog::new(notice).with_level(rerun::TextLogLevel::WARN))?;
    }
    Ok(())
}
