//! The tracker's shared fixtures, pulled in with `#[path]` by tests/tracker_golden.rs, tests/tracker_perception.rs and
//! examples/tracker_bench.rs (so the bench stays one self-contained binary for the cap; the data is embedded):
//! - the golden scene of `tools/golden_tracker.py` (`tests/data/tracker/`): its record, the recorded KeyNet and DetNet answers,
//!   and a [`Perception`] that replays them by (frame, camera, hand);
//! - the still two-hand scene, with fake networks whose outputs are rendered from the ground truth just before each call
//!   (handtrack `labels/heatmaps.py::render_heatmaps` / `render_distance`, ported below).
#![allow(dead_code, reason = "each test binary and the bench use a subset")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use handfit::Pose;
use kornia_image::Image;
use kornia_staging_imgproc::contours::min_enclosing_circle;
use kornia_staging_sensors::CameraFrame;
use nalgebra::{
    Isometry3, Matrix3, Matrix4, Rotation3, SVector, Translation3, UnitQuaternion, Vector3,
};
use robocap_live::frame::Luma;
use robocap_live::frame::isometry_from_matrix;
use robocap_live::frame::{NUM_CAMERAS, Rig};
use robocap_live::hands::camera::{RigCameraModel, fit_view, in_front, rig_models};
use robocap_live::hands::detect::{Detections, detect};
use robocap_live::hands::estimator::{KeypointEstimate, PerspectiveKeyNet, ViewRequest};
use robocap_live::hands::heatmaps::{DISTANCE_RANGE_MM, crop_to_heatmap, relative_distances};
use robocap_live::hands::letterbox::BarLetterbox;
use robocap_live::hands::model::{GenericHandModel, landmarks_world};
use robocap_live::hands::scale::CalibrationBlock;
use robocap_live::hands::tracker::Perception;
use robocap_live::hands::{HandsError, LEFT, RIGHT};
use robocap_live::nets::golden::f32_values;
use robocap_live::nets::{
    CROP_LEN, DISTANCE_BINS, DISTANCE_LEN, DetNetRaw, HEATMAP_LEN, HEATMAP_SIDE, HandNets,
    KeyNetRaw, NUM_LANDMARKS, NetFrame, NetsError,
};
use serde::Deserialize;

pub type Error = Box<dyn std::error::Error>;

const GOLDEN_JSON: &str = include_str!("../data/tracker/golden.json");
const ESTIMATES: &[u8] = include_bytes!("../data/tracker/estimates.bin");
const DETECTIONS: &[u8] = include_bytes!("../data/tracker/detections.bin");
/// One recorded KeyNet answer: points_net (42), points_px (42), d_rel_mm (21), confidence (21), presence.
const ESTIMATE_VALUES: usize = 127;
/// One recorded DetNet answer: two circles (cx, cy, r) and two probabilities.
const DETECTION_VALUES: usize = 8;

/// The still scene's hands are this much smaller than the generic hand.
pub const TRUE_PHI: f64 = 0.92;
const HEATMAP_SIGMA: f32 = 1.0;

#[derive(Deserialize)]
pub struct RequestRecord {
    pub camera: usize,
    pub side: usize,
    pub acquisition: bool,
    pub circle: [f64; 3],
}

#[derive(Deserialize)]
pub struct PoseRecord {
    pub rotation: [f64; 9],
    pub translation: [f64; 3],
    pub joint_angles: [f64; 22],
}

#[derive(Deserialize)]
pub struct FrameRecord {
    pub frame: usize,
    pub detnet_camera: i64,
    pub keynet_calls: Vec<Vec<RequestRecord>>,
    pub tracked: [bool; 2],
    pub reported: [bool; 2],
    pub poses: [Option<PoseRecord>; 2],
    pub landmarks: [Option<Vec<[f64; 3]>>; 2],
    pub view_cameras: [Vec<usize>; 2],
}

#[derive(Deserialize)]
pub struct RunRecord {
    pub phi: f64,
    pub frames: Vec<FrameRecord>,
}

#[derive(Deserialize)]
pub struct BlockView {
    pub camera: usize,
    pub weights: [f64; NUM_LANDMARKS],
}

#[derive(Deserialize)]
pub struct BlockRecord {
    pub frame: usize,
    pub side: usize,
    pub initial: PoseRecord,
    pub views: Vec<BlockView>,
}

#[derive(Deserialize)]
pub struct CalibrationRecord {
    pub phi_raw: f64,
    pub phi: f64,
    pub blocks: usize,
    pub iterations: usize,
    pub e_2d: f64,
    pub converged: bool,
    pub termination: String,
    pub hands: Vec<BlockRecord>,
}

#[derive(Deserialize)]
pub struct GoldenRecord {
    pub format: String,
    pub frames: usize,
    pub calibration_frames: usize,
    pub true_phi: f64,
    pub rig: Rig,
    pub world_from_rig: Vec<Option<[f64; 16]>>,
    pub calibration_run: RunRecord,
    pub calibration: CalibrationRecord,
    pub tracking_run: RunRecord,
    pub all_cameras_run: RunRecord,
    pub events: Vec<String>,
}

/// The golden scene and the estimator outputs Python's tracker was driven by.
pub struct Golden {
    pub record: GoldenRecord,
    estimates: Vec<f32>,
    detections: Vec<f32>,
}

/// The golden scene (its record checked against the tables' sizes).
pub fn golden() -> Result<Golden, Error> {
    let record: GoldenRecord = serde_json::from_str(GOLDEN_JSON)?;
    let estimates = f32_values(ESTIMATES).ok_or("estimates.bin: not whole f32 values")?;
    let detections = f32_values(DETECTIONS).ok_or("detections.bin: not whole f32 values")?;
    if record.format != "robocap-live-tracker-golden/1"
        || estimates.len() != record.frames * NUM_CAMERAS * 2 * ESTIMATE_VALUES
        || detections.len() != record.frames * NUM_CAMERAS * DETECTION_VALUES
    {
        let (format, frames) = (&record.format, record.frames);
        return Err(format!(
            "golden {format}: {} estimate and {} detection values for {frames} frames",
            estimates.len(),
            detections.len()
        )
        .into());
    }
    Ok(Golden {
        record,
        estimates,
        detections,
    })
}

impl Golden {
    /// The recorded KeyNet answer for a hand in a camera on a frame.
    pub fn estimate(&self, frame: usize, camera: usize, side: usize) -> KeypointEstimate {
        let start = ((frame * NUM_CAMERAS + camera) * 2 + side) * ESTIMATE_VALUES;
        let row = &self.estimates[start..start + ESTIMATE_VALUES];
        KeypointEstimate {
            points_net: std::array::from_fn(|i| [row[2 * i], row[2 * i + 1]]),
            points_px: std::array::from_fn(|i| [row[42 + 2 * i], row[42 + 2 * i + 1]]),
            d_rel_mm: std::array::from_fn(|i| row[84 + i]),
            confidence: std::array::from_fn(|i| row[105 + i]),
            presence: row[126],
            pinch: None,
            usable: true,
            crop: None,
        }
    }

    /// The recorded DetNet answer for a camera on a frame.
    pub fn detection(&self, frame: usize, camera: usize) -> Detections {
        let start = (frame * NUM_CAMERAS + camera) * DETECTION_VALUES;
        let row = &self.detections[start..start + DETECTION_VALUES];
        Detections {
            circle_net: [[row[0], row[1], row[2]], [row[3], row[4], row[5]]],
            probability: [row[6], row[7]],
        }
    }

    /// The headset pose of a frame as the runtime hands it over (an isometry), NaN when lost.
    pub fn isometry(&self, frame: usize) -> Isometry3<f64> {
        self.record.world_from_rig[frame]
            .as_ref()
            .and_then(isometry_from_matrix)
            .unwrap_or_else(|| {
                Isometry3::from_parts(
                    Translation3::new(f64::NAN, f64::NAN, f64::NAN),
                    UnitQuaternion::identity(),
                )
            })
    }

    /// The headset pose of a frame as the 4x4 Python recorded.
    pub fn matrix(&self, frame: usize) -> Option<Matrix4<f64>> {
        self.record.world_from_rig[frame].map(|m| Matrix4::from_fn(|i, k| m[4 * i + k]))
    }

    /// Python's calibration observations (`calibrate_scale`'s input) as the live calibration's blocks.
    pub fn calibration_blocks(&self) -> Result<Vec<CalibrationBlock>, Error> {
        let cameras: Vec<RigCameraModel> = rig_models(&self.record.rig)?;
        let mut blocks = Vec::new();
        for hand in &self.record.calibration.hands {
            let world = self
                .matrix(hand.frame)
                .ok_or("an observation on a frame without a headset pose")?;
            let views = hand
                .views
                .iter()
                .map(|view| {
                    let estimate = self.estimate(hand.frame, view.camera, hand.side);
                    fit_view(
                        &cameras[view.camera],
                        &world,
                        &estimate.points_px.map(|p| p.map(f64::from)),
                        &view.weights,
                        &estimate.d_rel_mm.map(f64::from),
                    )
                })
                .collect();
            let initial = Pose {
                rotation: Matrix3::from_fn(|i, k| hand.initial.rotation[3 * i + k]),
                translation: Vector3::from_column_slice(&hand.initial.translation),
                angles: SVector::<f64, 22>::from_column_slice(&hand.initial.joint_angles),
            };
            blocks.push(CalibrationBlock {
                mirror: if hand.side == 0 { 1.0 } else { -1.0 },
                views,
                initial,
            });
        }
        Ok(blocks)
    }
}

/// One KeyNet request as the Rust tracker made it.
#[derive(Clone, Debug)]
pub struct Request {
    pub camera: usize,
    pub side: usize,
    pub acquisition: bool,
    pub circle: [f64; 3],
}

/// The frame being run (set by the caller before each step) and every call the tracker made.
#[derive(Default)]
pub struct CallLog {
    pub frame: usize,
    pub detnet: Vec<(usize, usize)>,
    pub keynet: Vec<(usize, Vec<Request>)>,
}

/// The recorded fake estimator: answers depend only on (frame, camera, hand).
pub struct TablePerception {
    pub golden: Arc<Golden>,
    pub log: Arc<Mutex<CallLog>>,
}

impl Perception for TablePerception {
    fn detect(
        &mut self,
        _nets: &mut dyn HandNets,
        cameras: &[usize],
        _small: &[&Luma],
    ) -> Result<Vec<Detections>, HandsError> {
        let mut log = self
            .log
            .lock()
            .map_err(|_| HandsError::Invalid("log poisoned".into()))?;
        let frame = log.frame;
        log.detnet
            .extend(cameras.iter().map(|&camera| (frame, camera)));
        Ok(cameras
            .iter()
            .map(|&camera| self.golden.detection(frame, camera))
            .collect())
    }

    fn estimate(
        &mut self,
        _nets: &mut dyn HandNets,
        _full: &[Option<&CameraFrame>; NUM_CAMERAS],
        _turned_180: &[bool; NUM_CAMERAS],
        _world_from_rig: &Isometry3<f64>,
        views: &[ViewRequest],
    ) -> Result<(Vec<KeypointEstimate>, f64), HandsError> {
        let mut log = self
            .log
            .lock()
            .map_err(|_| HandsError::Invalid("log poisoned".into()))?;
        let frame = log.frame;
        log.keynet.push((
            frame,
            views
                .iter()
                .map(|v| Request {
                    camera: v.camera,
                    side: v.side,
                    acquisition: v.planning_pose_landmarks_world.is_none(),
                    circle: v.circle_net.map_or([f64::NAN; 3], |c| c.map(f64::from)),
                })
                .collect(),
        ));
        Ok((
            views
                .iter()
                .map(|v| self.golden.estimate(frame, v.camera, v.side))
                .collect(),
            0.0,
        ))
    }

    fn set_phi(&mut self, _phi: f64) {}
}

/// A backend that must never be called (the table perception answers instead).
pub struct NoNets;

impl HandNets for NoNets {
    fn detnet(&mut self, _frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        Err(NetsError::Run {
            net: "detnet",
            message: "not with the table perception".into(),
        })
    }
    fn keynet(
        &mut self,
        _crops: &[&[f32]],
        _keypoints: &[[f32; 3 * NUM_LANDMARKS]],
    ) -> Result<Vec<KeyNetRaw>, NetsError> {
        Err(NetsError::Run {
            net: "keynet",
            message: "not with the table perception".into(),
        })
    }
    fn describe(&self) -> String {
        "none".into()
    }
}

/// The still two-hand scene: both hands in front, or the right hand hidden behind the headset (so it stays untracked and DetNet
/// runs on every frameset). The hands are `TRUE_PHI` of the generic hand.
pub fn scene_landmarks(right_hidden: bool) -> Result<[[[f64; 3]; NUM_LANDMARKS]; 2], Error> {
    let generic = GenericHandModel::load()?;
    let truth = generic.scaled(TRUE_PHI);
    let rotation = |axis: usize, angle: f64| {
        Rotation3::from_axis_angle(
            &[Vector3::x_axis(), Vector3::y_axis(), Vector3::z_axis()][axis],
            angle,
        )
        .into_inner()
    };
    Ok(std::array::from_fn(|side| {
        let sign = if side == LEFT { -1.0 } else { 1.0 };
        let limits = generic.model().limits;
        let translation = if side == RIGHT && right_hidden {
            Vector3::new(0.0, 0.0, -1.0)
        } else {
            Vector3::new(sign * 0.11, 0.24, 0.30)
        };
        let pose = Pose {
            rotation: rotation(1, -0.35 * sign) * rotation(0, 1.2),
            translation,
            angles: SVector::from_fn(|j, _| {
                if j < 20 {
                    limits[(j, 0)] + 0.3 * (limits[(j, 1)] - limits[(j, 0)])
                } else {
                    0.0
                }
            }),
        };
        landmarks_world(&truth, &pose, side)
    }))
}

/// handtrack `render_heatmaps`: unit Gaussians (sigma one heatmap pixel) at crop points mapped to heatmap pixel centres,
/// 21 x 18 x 18 row-major.
fn render_heatmaps(points_crop: &[[f64; 2]; NUM_LANDMARKS]) -> Vec<f32> {
    let side = HEATMAP_SIDE;
    let mut out = vec![0.0f32; HEATMAP_LEN];
    for (landmark, point) in points_crop.iter().enumerate() {
        let (cx, cy) = (
            crop_to_heatmap(point[0] as f32),
            crop_to_heatmap(point[1] as f32),
        );
        for y in 0..side {
            let gy = (-((y as f32 - cy).powi(2)) / (2.0 * HEATMAP_SIGMA * HEATMAP_SIGMA)).exp();
            for x in 0..side {
                let gx = (-((x as f32 - cx).powi(2)) / (2.0 * HEATMAP_SIGMA * HEATMAP_SIGMA)).exp();
                out[(landmark * side + y) * side + x] = gy * gx;
            }
        }
    }
    out
}

/// handtrack `render_distance`: Gaussians at 18 bin centres spanning [-130, 130] mm, distances clamped first.
fn render_distance(d_rel_mm: &[f64; NUM_LANDMARKS]) -> Vec<f32> {
    let mut out = vec![0.0f32; DISTANCE_LEN];
    for (landmark, d) in d_rel_mm.iter().enumerate() {
        let centre = ((*d as f32).clamp(-DISTANCE_RANGE_MM, DISTANCE_RANGE_MM) + DISTANCE_RANGE_MM)
            * ((DISTANCE_BINS as f32 - 1.0) / (2.0 * DISTANCE_RANGE_MM));
        for bin in 0..DISTANCE_BINS {
            out[landmark * DISTANCE_BINS + bin] = (-0.5 * (bin as f32 - centre).powi(2)).exp();
        }
    }
    out
}

/// The rendered network outputs, consumed in order, and what the networks were asked.
#[derive(Default)]
pub struct Queue {
    pub detnet: VecDeque<DetNetRaw>,
    pub keynet: VecDeque<KeyNetRaw>,
    pub detnet_calls: usize,
    pub keynet_crops: usize,
}

/// Networks that return the outputs queued for them, after checking their inputs' shapes.
pub struct QueuedNets {
    pub queue: Arc<Mutex<Queue>>,
}

impl HandNets for QueuedNets {
    fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        let mut queue = self.queue.lock().map_err(|_| NetsError::Run {
            net: "detnet",
            message: "poisoned".into(),
        })?;
        if frames.iter().any(|f| f.rows().is_err()) {
            return Err(NetsError::Input {
                net: "detnet",
                message: "not a 640x480 net frame".into(),
            });
        }
        queue.detnet_calls += frames.len();
        (0..frames.len())
            .map(|_| {
                queue.detnet.pop_front().ok_or(NetsError::Run {
                    net: "detnet",
                    message: "nothing queued".into(),
                })
            })
            .collect()
    }

    fn keynet(
        &mut self,
        crops: &[&[f32]],
        keypoints: &[[f32; 3 * NUM_LANDMARKS]],
    ) -> Result<Vec<KeyNetRaw>, NetsError> {
        let mut queue = self.queue.lock().map_err(|_| NetsError::Run {
            net: "keynet",
            message: "poisoned".into(),
        })?;
        if crops.len() != keypoints.len() || crops.iter().any(|c| c.len() != CROP_LEN) {
            return Err(NetsError::Input {
                net: "keynet",
                message: "bad crops".into(),
            });
        }
        queue.keynet_crops += crops.len();
        (0..crops.len())
            .map(|_| {
                queue.keynet.pop_front().ok_or(NetsError::Run {
                    net: "keynet",
                    message: "nothing queued".into(),
                })
            })
            .collect()
    }

    fn describe(&self) -> String {
        "queued ground-truth renders".into()
    }
}

/// The ground truth of a still scene: the rig's lenses, the net-frame letterbox and both hands' landmarks (world = rig).
pub struct Truth {
    pub models: Vec<RigCameraModel>,
    pub letterbox: BarLetterbox,
    pub landmarks: [[[f64; 3]; NUM_LANDMARKS]; 2],
}

impl Truth {
    pub fn new(rig: &Rig, landmarks: [[[f64; 3]; NUM_LANDMARKS]; 2]) -> Result<Self, Error> {
        Ok(Self {
            models: rig_models(rig)?,
            letterbox: BarLetterbox::robocap(),
            landmarks,
        })
    }

    fn points_cam(
        &self,
        camera: usize,
        side: usize,
        world_from_rig: &Isometry3<f64>,
    ) -> [Vector3<f64>; NUM_LANDMARKS] {
        std::array::from_fn(|i| {
            let p = self.landmarks[side][i];
            self.models[camera]
                .cam_from_world_point(world_from_rig, &Vector3::new(p[0], p[1], p[2]))
        })
    }
}

/// The real DetNet decode + PerspectiveKeyNet with the ground truth rendered into the networks' outputs just before each call.
pub struct RenderedPerception {
    pub truth: Arc<Truth>,
    pub estimator: PerspectiveKeyNet,
    pub queue: Arc<Mutex<Queue>>,
}

impl RenderedPerception {
    /// Queue DetNet's raw output for `camera`: each hand that shows at least 12 keypoints, as the circle around them.
    fn render_detnet(&self, camera: usize) -> Result<(), HandsError> {
        let world = Isometry3::identity();
        let mut raw = DetNetRaw {
            center: [[0.0; 2]; 2],
            radius: [0.0; 2],
            presence_logit: [-4.0; 2],
        };
        for side in [LEFT, RIGHT] {
            let points = self.truth.points_cam(camera, side, &world);
            let mut net = Vec::new();
            let mut visible = 0;
            for p in points.iter().filter(|p| in_front(p)) {
                if let Some(px) = self.truth.models[camera].project(p) {
                    visible += usize::from(self.truth.models[camera].inside_image(&px));
                    let uv = self.truth.letterbox.to_net(&px);
                    net.push([uv.x, uv.y]);
                }
            }
            if let (true, Some(circle)) = (visible >= 12, min_enclosing_circle(&net)) {
                raw.center[side] = [(circle.cx / 640.0) as f32, (circle.cy / 480.0) as f32];
                raw.radius[side] = (circle.radius * 1.2 / 640.0) as f32;
                raw.presence_logit[side] = 4.0;
            }
        }
        self.queue
            .lock()
            .map_err(|_| HandsError::Invalid("poisoned".into()))?
            .detnet
            .push_back(raw);
        Ok(())
    }
}

impl Perception for RenderedPerception {
    fn detect(
        &mut self,
        nets: &mut dyn HandNets,
        cameras: &[usize],
        small: &[&Luma],
    ) -> Result<Vec<Detections>, HandsError> {
        for &camera in cameras {
            self.render_detnet(camera)?;
        }
        let images: Vec<&Image<u8, 1>> = small.iter().map(|image| image.as_ref()).collect();
        detect(nets, &BarLetterbox::robocap(), &images)
    }

    /// Renders each view's KeyNet output (heatmaps of the truth in its planned crop, present when 15 keypoints fall inside), then
    /// runs the real estimator; the crop time is the estimator's, as the runtime's perception reports it.
    fn estimate(
        &mut self,
        nets: &mut dyn HandNets,
        full: &[Option<&CameraFrame>; NUM_CAMERAS],
        turned_180: &[bool; NUM_CAMERAS],
        world: &Isometry3<f64>,
        requests: &[ViewRequest],
    ) -> Result<(Vec<KeypointEstimate>, f64), HandsError> {
        for request in requests {
            let plan = self.estimator.plan(world, request)?;
            let points = self.truth.points_cam(request.camera, request.side, world);
            let uv: [[f64; 2]; NUM_LANDMARKS] = std::array::from_fn(|i| {
                let (p, _) = plan.crop.to_crop(&points[i]);
                [p.x, p.y]
            });
            let inside = uv
                .iter()
                .filter(|p| p.iter().all(|x| (0.0..96.0).contains(x)))
                .count();
            let raw = KeyNetRaw {
                heatmaps: render_heatmaps(&uv),
                distance: render_distance(&relative_distances(&points, TRUE_PHI)),
                presence_logit: if inside >= 15 { 4.0 } else { -4.0 },
                pinch_logit: Some(-3.0),
            };
            self.queue
                .lock()
                .map_err(|_| HandsError::Invalid("poisoned".into()))?
                .keynet
                .push_back(raw);
        }
        self.estimator
            .estimate(nets, full, turned_180, world, requests)
    }

    fn set_phi(&mut self, phi: f64) {
        self.estimator.set_phi(phi);
    }
}
