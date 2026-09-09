//! `basalt::FrameToFrameOpticalFlow`, ported from
//! `frame_to_frame_optical_flow.h:103-749`.
//!
//! One call to [`FrameToFrameOpticalFlow::process_frame`] is basalt's
//! `processFrame` (`:203-292`): build a pyramid per camera, track every keypoint
//! from the previous frame, refresh the occupancy grid, detect and match new
//! points, and drop the ones that fail the epipolar test.
//!
//! The driver is generic over the pyramid builder and the tracker
//! ([`crate::pyramid::PyramidBuilder`], [`PatchTracker`]), both defaulting to the
//! CPU backends, and no signature here names a concrete pyramid: a CubeCL
//! backend arrives through [`FrameToFrameOpticalFlow::with_backends`]
//! (`cubecl-portability.md` §12.1).
//!
//! ## What the port leaves out, and why
//!
//! * **The queues and the processing thread.** `processingLoop` (`:122-155`) pops
//!   images off a TBB queue, drains four feedback queues and preintegrates the
//!   IMU itself. Here the caller drives the frontend one frameset at a time and
//!   hands in the two things those queues carried: the predicted pose pair and
//!   the average scene depth. Nothing in this module knows the IMU exists
//!   (decision D17: offline lockstep first).
//! * **Recall.** `recallPoints` (`:526-575`) is off in every shipped config, and
//!   it leaks a patch stack per landmark by design (`:590`, trap 18).
//! * **The GUI bookkeeping.** `tracking_guesses`, `matching_guesses` and
//!   `recall_guesses` (`optical_flow.h:110-112`) are only filled when `show_gui`
//!   is set and are never read by the estimator.
//! * **`OpticalFlowResult::pyramid_levels`.** Only
//!   `MultiscaleFrameToFrameOpticalFlow` writes it (`:57-59`), and that variant
//!   is out of scope, so [`Keypoints`] does not carry the field: it was always
//!   empty, and a `Vec` cloned twice per frameset for a variant that does not
//!   exist is storage nothing fills or reads.
//!
//! ## Five places the port does not match the C++ exactly
//!
//! * **The essential matrix is per camera** (deviation X03). `optical_flow.h:210`
//!   computes the cam0-cam1 matrix once and stores it under every index, which is
//!   wrong for cameras 2 and up. [`FrontendOptions::epipolar_per_camera`] defaults
//!   to the fix; set it to `false` for a C++-parity run. For a two-camera rig the
//!   two agree exactly.
//! * **Image bounds come from each camera's own resolution.** basalt reads
//!   `calib.resolution.at(0)` for the whole rig (`:108-109`, trap 16); msd-g2's
//!   cameras are stored rotated and do not share one (decision D30). The
//!   detection grid is per camera as well, which is what the C++ does
//!   (`keypoints.cpp:140-144` derives it from the image it is handed); the
//!   *occupancy* matrix keeps camera 0's shape, which is also what the C++ does
//!   (`:119`), so the two can disagree on a mixed-resolution rig and a cell whose
//!   index falls outside the matrix is skipped where the C++ reads out of range.
//! * **`getNumCams() >= 2` is not required** (trap 17). With one camera the
//!   matching and filtering passes are skipped instead of indexing `T_i_c[1]`.
//!   Asking for the C++ essential-matrix bug on a one-camera rig is refused.
//! * **The second mask test moves one step later.** `trackPoints` tests
//!   `masks2.inBounds` between the forward and the backward track (`:352`); here
//!   the whole forward pass runs, then the whole backward pass, so the test is
//!   applied to the same tracked position afterwards. The same points are
//!   dropped; the only cost is a backward track that would have been skipped.
//! * **There is a keypoint budget.** See [`FrontendOptions::max_keypoints`].

use std::marker::PhantomData;

use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

use crate::calib::Calibration;
use crate::camera::{CameraError, RigCamera};
use crate::config::{MatchingGuessType, VioConfig};
use crate::duration_ns;
use crate::frontend::detect::{
    CellGrid, CornerScan, CpuCornerScan, DetectError, DetectorConfig, DetectorScratch,
    KeypointsData, LOWEST_THRESHOLD_RUNG, MAX_CELLS, Masks, Occupancy, Rect,
    detect_keypoints_with_cells,
};
use crate::frontend::parallel::{MAX_THREADS, WorkPool};
use crate::frontend::patterns::Pattern;
use crate::frontend::se2::AffineCompact2f;
use crate::frontend::tracker::{
    CpuPatchTracker, FlowResult, FlowTransforms, MAX_CAPACITY, MAX_LEVELS, PatchTracker, PointsSoA,
    SourcePatches, TrackerError,
};
use crate::image::ImageU16;
use crate::lie::{Se3, So3};
use crate::pyramid::{CpuPyramidBuilder, Pyramid, PyramidBuilder, PyramidError};
use crate::types::KeypointId;

/// What [`Keypoints::responses`] holds where basalt's `keypoint_responses` map
/// has no entry: the same `-1` its `addKeypoint` default argument stores (`:726`).
pub const NO_RESPONSE: f32 = -1.0;

/// The port's own frontend knobs, kept out of [`VioConfig`] so that basalt's
/// JSON still round-trips byte for byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontendOptions {
    /// Compute `E[i]` from `T_c0_ci` per camera instead of reusing the cam0-cam1
    /// matrix for every camera (deviation X03). `false` reproduces the C++ bug.
    pub epipolar_per_camera: bool,
    /// Workers the tracking passes run on (decision D31). One is the
    /// deterministic reference lane; any value gives the same numbers.
    pub threads: usize,
    /// Keypoints one camera may carry, and the capacity every buffer is sized for.
    ///
    /// basalt has no such cap: its maps grow. The port needs one because the
    /// tracker's storage is preallocated (§12.2 forbids growing a buffer on the
    /// per-frame path), so detection and stereo matching **stop adding** once a
    /// camera reaches it, in the detector's own scan order. Keypoints that
    /// already exist are never dropped to make room, and the frame stays
    /// processable. The default is about eight times what the shipped 50-pixel
    /// grid can produce on a 960x960 frame, so nothing in the reference
    /// configuration comes near it, and
    /// [`crate::frontend::tracker::MAX_CAPACITY`] is the ceiling a caller may
    /// ask for.
    pub max_keypoints: usize,
}

impl Default for FrontendOptions {
    fn default() -> Self {
        Self {
            epipolar_per_camera: true,
            threads: 1,
            // The shipped grid is 50 px over a 960x960 frame with one point per
            // cell, so 361 cells; 3000 leaves room for a finer grid and for the
            // extra points the non-overlap pass adds on cameras 1 and up.
            max_keypoints: 3000,
        }
    }
}

/// The pose pair basalt's own IMU preintegration produces (`:261-262`).
///
/// `processingLoop` predicts the current frame's pose from the last state it got
/// back from the estimator (`:147-150`) and `processFrame` turns the pair into a
/// per-camera `T_c1_c2` that seeds the tracker. Handing it in keeps this module
/// free of the IMU: with both poses at the identity — which is what basalt itself
/// runs with until the first state arrives (`:140-147`) — the guess is the same
/// pixel reprojected through `depth_guess`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PosePrediction {
    /// `latest_state->T_w_i`, the pose of the previous frame.
    pub t_w_i_previous: Se3<f32>,
    /// `predicted_state->T_w_i`, the pose predicted for this frame.
    pub t_w_i_current: Se3<f32>,
}

/// One camera's tracked keypoints, in structure-of-arrays form.
///
/// basalt's `Keypoints` is an `Eigen::aligned_map<KeypointId, AffineCompact2f>`
/// (`optical_flow.h:68`), i.e. an ordered map. The order is load-bearing — it is
/// the order `filterPointsForCam` walks and the order new ids are handed out in —
/// so this keeps the ids **sorted ascending** with the parallel arrays beside
/// them, and looks up by binary search. The warps live in a [`FlowTransforms`],
/// six flat coefficient arrays with the keypoint index fast-varying, so no
/// per-keypoint record is an array of structs (§12.2). No hash map appears
/// anywhere on the per-frame path.
#[derive(Debug, Default, PartialEq)]
pub struct Keypoints {
    /// Keypoint ids, ascending. `LandmarkId == KeypointId` (`optical_flow.h:71`).
    pub ids: Vec<KeypointId>,
    /// The 2x3 warp of each keypoint, in the same order as [`Keypoints::ids`].
    pub transforms: FlowTransforms,
    /// The detector response of each keypoint, `-1` where there is none.
    ///
    /// basalt keeps the responses in a second map that only `addKeypoint` writes
    /// (`:730`), so a keypoint carried forward by `trackPoints` or inserted by
    /// `addKeypoints` from the stereo match (`:734-741`) has **no entry at all**.
    /// [`NO_RESPONSE`] is what `addKeypoint`'s own default argument stores for
    /// "no response" (`:726`), so one array with that sentinel says the same
    /// thing as the C++'s two maps. The value itself is OpenCV's integer
    /// `cornerScore`, which is what the C++ records.
    pub responses: Vec<f32>,
}

/// `Clone` by hand for the sake of `clone_from`.
///
/// `#[derive(Clone)]` writes only `clone`; the default `clone_from` is
/// `*self = source.clone()`, which drops every buffer and allocates as many
/// again — sixteen allocations and sixteen frees per stereo frame once
/// [`FrameToFrameOpticalFlow::process_frame`] takes its snapshot and again if it
/// has to restore. Copying field by field lets `Vec::clone_from` overwrite in
/// place, and `FlowTransforms` does the same one level down.
impl Clone for Keypoints {
    fn clone(&self) -> Self {
        Self {
            ids: self.ids.clone(),
            transforms: self.transforms.clone(),
            responses: self.responses.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.ids.clone_from(&source.ids);
        self.transforms.clone_from(&source.transforms);
        self.responses.clone_from(&source.responses);
    }
}

impl Keypoints {
    /// How many keypoints this camera carries.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether this camera has no keypoints.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The warp stored for `id`, or `None`.
    pub fn get(&self, id: KeypointId) -> Option<AffineCompact2f> {
        self.index_of(id).map(|index| self.transforms.get(index))
    }

    /// The warp at `index`, in id order.
    ///
    /// # Panics
    ///
    /// Whatever [`FlowTransforms::get`] panics on: an `index` past the end.
    pub fn transform(&self, index: usize) -> AffineCompact2f {
        self.transforms.get(index)
    }

    /// Where `id` sits in the sorted arrays.
    fn index_of(&self, id: KeypointId) -> Option<usize> {
        self.ids.binary_search(&id).ok()
    }

    fn clear(&mut self) {
        self.ids.clear();
        self.transforms.clear();
        self.responses.clear();
    }

    /// `transforms->keypoints[cam][id] = kp`: insert or overwrite.
    fn set(&mut self, id: KeypointId, transform: &AffineCompact2f, response: f32) {
        match self.ids.binary_search(&id) {
            Ok(index) => {
                self.transforms.set(index, transform);
                self.responses[index] = response;
            }
            Err(index) => {
                self.ids.insert(index, id);
                self.transforms.insert(index, transform);
                self.responses.insert(index, response);
            }
        }
    }

    /// `std::map::insert`, which **keeps** an existing entry (`:740`).
    ///
    /// Returns whether the keypoint was new.
    fn insert_if_absent(&mut self, id: KeypointId, transform: &AffineCompact2f) -> bool {
        match self.ids.binary_search(&id) {
            Ok(_) => false,
            Err(index) => {
                self.ids.insert(index, id);
                self.transforms.insert(index, transform);
                self.responses.insert(index, NO_RESPONSE);
                true
            }
        }
    }

    fn remove(&mut self, id: KeypointId) -> Option<AffineCompact2f> {
        let index: usize = self.index_of(id)?;
        let transform: AffineCompact2f = self.transforms.get(index);
        self.ids.remove(index);
        self.responses.remove(index);
        self.transforms.remove(index);
        Some(transform)
    }
}

/// What one frameset produced: one [`Keypoints`] per camera.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FlowFrame {
    // `Vec::clone_from` copies element by element through `Keypoints::clone_from`
    // above, so deriving here keeps the buffer-preserving property.
    /// Frameset timestamp in nanoseconds, `None` before the first frameset commits.
    ///
    /// basalt writes `t_ns = -1` until then (`optical_flow.h:172`), which makes
    /// `-1` — and every other negative timestamp — ambiguous. The port carries
    /// the absence in the type instead, so any `i64` is a timestamp like any
    /// other.
    pub t_ns: Option<i64>,
    /// One entry per camera, in rig order.
    pub cameras: Vec<Keypoints>,
}

/// What the frontend can refuse.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FrontendError {
    /// The calibration carries no cameras.
    #[error("the calibration carries no cameras")]
    NoCameras,
    /// The calibration has fewer extrinsics than cameras.
    #[error("the calibration has {intrinsics} cameras but {extrinsics} extrinsics")]
    RaggedExtrinsics {
        /// Camera models the calibration carries.
        intrinsics: usize,
        /// `T_i_c` entries it carries.
        extrinsics: usize,
    },
    /// A frameset arrived at or before the last accepted one.
    #[error("frameset timestamps must increase: got {t_ns} after {previous_t_ns}")]
    NonMonotonicFrameset {
        /// Timestamp of the last accepted frameset.
        previous_t_ns: i64,
        /// Timestamp of the frameset handed in.
        t_ns: i64,
    },
    /// The frameset does not hold one image per camera.
    #[error("expected {expected} images, got {actual}")]
    CameraCountMismatch {
        /// Cameras in the rig.
        expected: usize,
        /// Images in the frameset.
        actual: usize,
    },
    /// `optical_flow_pattern` does not name the pattern this instance runs.
    #[error("config asks for pattern {config}, this frontend runs pattern {built}")]
    PatternMismatch {
        /// `optical_flow_pattern` from the config file.
        config: i32,
        /// `Pattern::CODE` of the type parameter.
        built: i32,
    },
    /// `optical_flow_type` names an implementation that is not ported.
    #[error("optical flow type {0:?} is not ported; only frame_to_frame is")]
    UnsupportedFlowType(String),
    /// A camera's frame is smaller than one detection cell.
    #[error("camera {camera}: a {width}x{height} frame cannot carry a {cell}-pixel detection grid")]
    FrameTooSmall {
        /// Which camera.
        camera: usize,
        /// Frame width.
        width: usize,
        /// Frame height.
        height: usize,
        /// `optical_flow_detection_grid_size`.
        cell: usize,
    },
    /// A camera's detection grid has more cells than one occupancy buffer holds.
    #[error(
        "camera {camera}: the calibrated resolution over the detection grid size is a \
         {rows}x{columns} occupancy grid; the ceiling is {ceiling} cells"
    )]
    TooManyCells {
        /// Which camera.
        camera: usize,
        /// Rows the grid asks for.
        rows: usize,
        /// Columns the grid asks for.
        columns: usize,
        /// [`MAX_CELLS`].
        ceiling: usize,
    },
    /// A config field that indexes or counts is negative.
    #[error("{field} must not be negative, got {value}")]
    NegativeConfig {
        /// The config key.
        field: &'static str,
        /// What it holds.
        value: i32,
    },
    /// The keypoint budget is larger than the tracker can carry.
    #[error("max_keypoints is {max_keypoints}, the tracker's capacity is {capacity}")]
    BudgetExceedsCapacity {
        /// What the options ask for.
        max_keypoints: usize,
        /// What the tracker was built for.
        capacity: usize,
    },
    /// The tracker was built for a different pyramid depth than the config asks.
    #[error("config asks for {config} pyramid levels, the tracker runs {tracker}")]
    LevelMismatch {
        /// `optical_flow_levels + 1`.
        config: usize,
        /// What the tracker was built for.
        tracker: usize,
    },
    /// The epipolar filter needs a second camera when the C++ bug is reproduced.
    #[error("epipolar_per_camera = false needs at least two cameras, the rig has {cameras}")]
    NeedsTwoCameras {
        /// Cameras in the rig.
        cameras: usize,
    },
    /// `optical_flow_detection_min_threshold` cannot stop the halving ladder.
    #[error(
        "optical_flow_detection_min_threshold is {min_threshold}, which must be at least {rung}: \
         the detector halves the FAST threshold until it drops below it, and integer division \
         never gets a threshold of zero past zero"
    )]
    ThresholdLadderNeverEnds {
        /// `optical_flow_detection_min_threshold` from the config file.
        min_threshold: i32,
        /// The lowest rung the ladder can stop at ([`LOWEST_THRESHOLD_RUNG`]).
        rung: i32,
    },
    /// The threshold ladder starts below where it stops, so it never runs.
    #[error(
        "optical_flow_detection_max_threshold is {max_threshold} and \
         optical_flow_detection_min_threshold is {min_threshold}: the ladder starts below where it \
         stops, so the detector can never add a keypoint"
    )]
    EmptyThresholdLadder {
        /// `optical_flow_detection_min_threshold` from the config file.
        min_threshold: i32,
        /// `optical_flow_detection_max_threshold` from the config file.
        max_threshold: i32,
    },
    /// A frameset image is not the size the calibration gives that camera.
    #[error(
        "camera {camera}: the calibration is for {expected_width}x{expected_height} frames, \
         got {actual_width}x{actual_height}"
    )]
    FrameSizeMismatch {
        /// Which camera.
        camera: usize,
        /// Width the calibration gives the camera.
        expected_width: usize,
        /// Height the calibration gives the camera.
        expected_height: usize,
        /// Width of the image handed in.
        actual_width: usize,
        /// Height of the image handed in.
        actual_height: usize,
    },
    /// A camera model the projection layer does not implement.
    #[error("camera: {0}")]
    Camera(#[from] CameraError),
    /// The pyramid refused the geometry.
    #[error("pyramid: {0}")]
    Pyramid(#[from] PyramidError),
    /// The tracker refused the inputs.
    #[error("tracker: {0}")]
    Tracker(#[from] TrackerError),
    /// The detector refused the inputs.
    #[error("detector: {0}")]
    Detect(#[from] DetectError),
    /// The thread pool could not be built.
    #[error("could not build a pool of {threads} threads")]
    ThreadPool {
        /// Threads asked for.
        threads: usize,
    },
    /// No workers were asked for, which is not a pool anything can run on.
    #[error("threads must be at least 1")]
    NoThreads,
    /// More workers were asked for than [`MAX_THREADS`].
    #[error("threads is {threads}, the ceiling is {ceiling}")]
    TooManyThreads {
        /// Workers asked for.
        threads: usize,
        /// [`MAX_THREADS`].
        ceiling: usize,
    },
    /// A larger keypoint budget was asked for than [`MAX_CAPACITY`].
    #[error("max_keypoints is {max_keypoints}, the ceiling is {ceiling}")]
    TooManyKeypoints {
        /// Keypoints asked for.
        max_keypoints: usize,
        /// [`MAX_CAPACITY`].
        ceiling: usize,
    },
    /// `optical_flow_levels` asks for a deeper pyramid than the buffers allow.
    #[error("optical_flow_levels is {levels}, so {num_levels} levels; the ceiling is {ceiling}")]
    TooManyLevels {
        /// `optical_flow_levels` from the config file.
        levels: i32,
        /// `optical_flow_levels + 1`, which is what every buffer is sized with.
        num_levels: usize,
        /// [`MAX_LEVELS`].
        ceiling: usize,
    },
}

/// `basalt::FrameToFrameOpticalFlow<Scalar, Pattern>` with `Scalar = f32`.
///
/// Generic over the pyramid builder and the tracker, both defaulting to the CPU
/// backends: a CubeCL implementation of [`PyramidBuilder`] and [`PatchTracker`]
/// drops in through [`FrameToFrameOpticalFlow::with_backends`] and no public
/// signature here names a concrete pyramid (§12.1).
#[derive(Debug)]
pub struct FrameToFrameOpticalFlow<
    P: Pattern,
    B: PyramidBuilder = CpuPyramidBuilder,
    T: PatchTracker<Pattern = P, Pyramid = B::Pyramid> = CpuPatchTracker<P>,
> {
    config: VioConfig,
    options: FrontendOptions,
    calib: Calibration<f32>,
    cameras: Vec<RigCamera<f32>>,
    /// `E`, one 4x4 essential matrix per camera (`optical_flow.h:207-213`).
    essential: Vec<Matrix4<f32>>,
    /// One detection grid per camera, from that camera's own image size, as the
    /// C++ derives it inside `detectKeypointsWithCells` (`keypoints.cpp:140-144`).
    detection_grids: Vec<CellGrid>,
    /// The occupancy grid, shaped from camera 0 as `cells` is (`:119`).
    occupancy_grid: CellGrid,
    /// `cells`, `(h/c + 1) x (w/c + 1)` counts per camera (`:119`).
    cells: Vec<Vec<i32>>,
    /// `last_keypoint_id` (`optical_flow.h:174`), the global landmark id space.
    last_keypoint_id: u64,
    /// `last_keypoint_id` as it stood before the last **committed** frameset,
    /// which is what makes "how many of these keypoints are new" answerable
    /// after the fact rather than only inside the call that produced them.
    last_keypoint_id_before_frame: u64,
    /// `t_ns` (`optical_flow.h:172`), `None` until the first frame commits.
    ///
    /// basalt's `-1` sentinel is not ported: `processFrame` reads `t_ns < 0` as
    /// "no previous frame", which would make a frameset at a negative timestamp
    /// reset tracking instead of continuing it.
    t_ns: Option<i64>,
    /// `frame_counter` (`optical_flow.h:173`).
    frame_counter: u64,
    /// `depth_guess`, seeded from the config and refreshed by the estimator.
    depth_guess: f32,
    /// Masks for this frame, grown by `cam0OverlapCellsMasksForCam`.
    masks: Vec<Masks>,

    pyramid_builder: B,
    /// The last **committed** frame's pyramids, one per camera.
    ///
    /// While a frame is in flight this is the *previous* frame, and `staging`
    /// holds the one being processed; the two swap only once every step has
    /// succeeded. See [`FrameToFrameOpticalFlow::process_frame`].
    pyramid: Vec<B::Pyramid>,
    /// The frame in flight.
    staging: Vec<B::Pyramid>,
    tracker: T,
    patches: T::Patches,
    /// The source keypoint ids of the call in flight, in map order (`:299`, `:306`).
    ids: Vec<KeypointId>,
    /// The source warps, in the same order (`:300`, `:307`).
    source: FlowTransforms,
    /// Which entries of `ids` survived the `masks1` test and were offered to the
    /// tracker; the tracker's index space is this vector's.
    offered: Vec<usize>,
    /// The source positions the forward patches are built at.
    positions: PointsSoA,
    /// `transform_2` once the depth guess has been applied (`:342`).
    guesses: FlowTransforms,
    /// The tracker's dense output.
    result: FlowResult,
    /// The ids that survived, in ascending source order, with their warps.
    tracked_ids: Vec<KeypointId>,
    /// The warps of [`FrameToFrameOpticalFlow::tracked_ids`].
    tracked: FlowTransforms,
    detector: DetectorScratch,
    detected: KeypointsData,
    /// The keypoints `addPointsForCamera(0)` produced, to be matched onward.
    new_cam0: Keypoints,
    /// Ids the epipolar filter removes, ascending (`std::set`, `:669`).
    to_remove: Vec<KeypointId>,
    frame: FlowFrame,
    /// The keypoint state as it was before the frame in flight; see
    /// [`FrameToFrameOpticalFlow::process_frame`].
    snapshot: FrameState,
    /// What the last frame's phases cost; see [`FlowTimings`].
    timings: FlowTimings,
    pattern: PhantomData<P>,
}

/// Wall time the frontend's phases took on the last frame, nanoseconds.
///
/// Reported, never compared: they are wall-clock, so they differ run to run and
/// nothing in `process_frame` reads them, which is what keeps the frame
/// bit-reproducible (D17), exactly as the estimator's own `StageTimings` are.
///
/// These do not add up to the frame: cell counts and other bookkeeping
/// remain outside the measured stages. Stereo includes cross-camera matching
/// and the epipolar filter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlowTimings {
    /// Building this frame's pyramids, every camera.
    pub pyramid_ns: u64,
    /// `detectKeypointsWithCells`, every camera.
    pub detect_ns: u64,
    /// Temporal `trackPoints` calls only.
    pub track_ns: u64,
    /// Cross-camera matching and epipolar filtering.
    pub stereo_ns: u64,
}

/// The state one `processFrame` mutates, kept so a failed frame can be undone.
///
/// Held in the frontend rather than allocated per frame, and every type in it
/// implements `clone_from` by hand so the copy overwrites the existing buffers
/// instead of replacing them: once the buffers have reached their high-water
/// mark, taking the snapshot and restoring from it are **allocation-free**.
/// `tests/frame_allocations.rs` counts that with a global allocator rather than
/// asserting it.
#[derive(Debug, Clone, Default)]
struct FrameState {
    cameras: Vec<Keypoints>,
    cells: Vec<Vec<i32>>,
    last_keypoint_id: u64,
}

impl<P: Pattern> FrameToFrameOpticalFlow<P, CpuPyramidBuilder, CpuPatchTracker<P>> {
    /// `FrameToFrameOpticalFlow(conf, cal)` (`frame_to_frame_optical_flow.h:103-120`)
    /// on the CPU backends.
    ///
    /// The calibration arrives in `f64` — that is what the JSON gives — and is
    /// cast to `f32` here, as `OpticalFlowTyped`'s constructor does
    /// (`optical_flow.h:204`).
    ///
    /// # Errors
    ///
    /// [`FrontendError`] when the config names another flow type or pattern, when
    /// the rig is empty, ragged, too small for the grid or asks for more
    /// occupancy cells than [`MAX_CELLS`], when the detector's threshold ladder
    /// would never end, when a camera model has no projection, when the thread
    /// pool cannot be built, or when `threads` or `max_keypoints` is outside
    /// what [`FrontendOptions`] allows.
    pub fn new(
        config: VioConfig,
        calibration: &Calibration<f64>,
        options: FrontendOptions,
    ) -> Result<Self, FrontendError> {
        Self::validate_config(&config)?;
        Self::validate_options(&options)?;
        let num_levels: usize = config.optical_flow_levels as usize + 1;
        let pool: WorkPool =
            WorkPool::new(options.threads).map_err(|_| FrontendError::ThreadPool {
                threads: options.threads,
            })?;
        let tracker: CpuPatchTracker<P> = CpuPatchTracker::new(
            options.max_keypoints,
            num_levels,
            config.optical_flow_max_iterations as usize,
            config.optical_flow_max_recovered_dist2,
            pool,
        )?;
        Self::with_backends(
            config,
            calibration,
            options,
            CpuPyramidBuilder::new(),
            tracker,
            Box::new(CpuCornerScan::default()),
        )
    }
}

impl<P: Pattern, B: PyramidBuilder, T: PatchTracker<Pattern = P, Pyramid = B::Pyramid>>
    FrameToFrameOpticalFlow<P, B, T>
{
    /// The config checks that do not depend on the backends.
    fn validate_config(config: &VioConfig) -> Result<(), FrontendError> {
        if config.optical_flow_type != "frame_to_frame" {
            return Err(FrontendError::UnsupportedFlowType(
                config.optical_flow_type.clone(),
            ));
        }
        if config.optical_flow_pattern != P::CODE {
            return Err(FrontendError::PatternMismatch {
                config: config.optical_flow_pattern,
                built: P::CODE,
            });
        }
        for (field, value) in [
            ("optical_flow_levels", config.optical_flow_levels),
            (
                "optical_flow_max_iterations",
                config.optical_flow_max_iterations,
            ),
            (
                "optical_flow_detection_grid_size",
                config.optical_flow_detection_grid_size,
            ),
            (
                "optical_flow_detection_num_points_cell",
                config.optical_flow_detection_num_points_cell,
            ),
        ] {
            if value < 0 {
                return Err(FrontendError::NegativeConfig { field, value });
            }
        }
        // The detector's threshold ladder halves by integer division, so a
        // `min_threshold` at or below zero never ends it — in basalt as much as
        // here (`keypoints.cpp:162`, `:187`). The detector floors its own last
        // rung as well ([`LOWEST_THRESHOLD_RUNG`]), but a config that asks for a
        // ladder the C++ would hang on is refused rather than quietly run at a
        // threshold nobody asked for.
        let min_threshold: i32 = config.optical_flow_detection_min_threshold;
        if min_threshold < LOWEST_THRESHOLD_RUNG {
            return Err(FrontendError::ThresholdLadderNeverEnds {
                min_threshold,
                rung: LOWEST_THRESHOLD_RUNG,
            });
        }
        let max_threshold: i32 = config.optical_flow_detection_max_threshold;
        if max_threshold < min_threshold {
            return Err(FrontendError::EmptyThresholdLadder {
                min_threshold,
                max_threshold,
            });
        }
        // Every per-patch buffer is sized with `optical_flow_levels + 1`, so a
        // config asking for a pyramid nothing could hold is refused here rather
        // than at the allocation, which aborts instead of returning.
        let num_levels: usize = config.optical_flow_levels as usize + 1;
        if num_levels > MAX_LEVELS {
            return Err(FrontendError::TooManyLevels {
                levels: config.optical_flow_levels,
                num_levels,
                ceiling: MAX_LEVELS,
            });
        }
        Ok(())
    }

    /// The port's own knobs, which arrive from the caller rather than a basalt file.
    ///
    /// Every value a caller may type is bounded here, whatever backends the
    /// frontend is then built on: [`FrameToFrameOpticalFlow::with_backends`]
    /// takes a tracker that is already built, so [`PatchSoA::new`]'s own
    /// ceilings — the second line under these — never run on that seam.
    ///
    /// [`PatchSoA::new`]: crate::frontend::tracker::PatchSoA::new
    fn validate_options(options: &FrontendOptions) -> Result<(), FrontendError> {
        // rayon spawns exactly what it is asked for, so an unbounded `threads`
        // exhausts the machine's threads instead of returning an error; zero is
        // a pool no work can run on, which `WorkPool::new` reads as one.
        if options.threads == 0 {
            return Err(FrontendError::NoThreads);
        }
        if options.threads > MAX_THREADS {
            return Err(FrontendError::TooManyThreads {
                threads: options.threads,
                ceiling: MAX_THREADS,
            });
        }
        // The budget sizes every per-patch buffer, and a `Vec` too long to
        // allocate panics rather than returning (decision D32).
        if options.max_keypoints > MAX_CAPACITY {
            return Err(FrontendError::TooManyKeypoints {
                max_keypoints: options.max_keypoints,
                ceiling: MAX_CAPACITY,
            });
        }
        Ok(())
    }

    /// Build a frontend on caller-supplied stages.
    ///
    /// This is the seam a GPU backend enters through: `builder` and `tracker` are
    /// any pair whose pyramid types agree, the patch storage comes from the
    /// tracker itself, and `scanner` is the detector's corner stage — the three
    /// are independent, so a backend may replace any subset of them.
    ///
    /// `scanner` is a trait object where the other two are type parameters, and
    /// that asymmetry is deliberate rather than a leftover: a third parameter
    /// would have to be spelled at every `FrameToFrameOpticalFlow<..>` in the
    /// crate and in both [`crate::FrontendLane`] arms, and the scanner is
    /// entered once per camera per frameset — the ~1.8 k `band` calls inside a
    /// frame go through [`DetectorScratch`], not through this seam. The price
    /// is that [`CornerScan`] must be `Send + Sync` and `DetectorScratch` hand-
    /// writes `Default`.
    ///
    /// # Errors
    ///
    /// As [`FrameToFrameOpticalFlow::new`], plus
    /// [`FrontendError::BudgetExceedsCapacity`] and
    /// [`FrontendError::LevelMismatch`] when the backends were built for a
    /// different shape than the config asks for.
    pub fn with_backends(
        config: VioConfig,
        calibration: &Calibration<f64>,
        options: FrontendOptions,
        builder: B,
        tracker: T,
        scanner: Box<dyn CornerScan>,
    ) -> Result<Self, FrontendError> {
        Self::validate_config(&config)?;
        Self::validate_options(&options)?;

        let num_levels: usize = config.optical_flow_levels as usize + 1;
        if tracker.num_levels() != num_levels {
            return Err(FrontendError::LevelMismatch {
                config: num_levels,
                tracker: tracker.num_levels(),
            });
        }
        if options.max_keypoints > tracker.capacity() {
            return Err(FrontendError::BudgetExceedsCapacity {
                max_keypoints: options.max_keypoints,
                capacity: tracker.capacity(),
            });
        }

        let calib: Calibration<f32> = calibration.cast();
        // `calib.T_i_c[i]` is indexed for every camera (`:264-265`, `:651`); a
        // calibration with fewer poses than models would index past the end.
        if calib.t_i_c.len() != calib.intrinsics.len() {
            return Err(FrontendError::RaggedExtrinsics {
                intrinsics: calib.intrinsics.len(),
                extrinsics: calib.t_i_c.len(),
            });
        }
        let cameras: Vec<RigCamera<f32>> = RigCamera::from_calibration(&calib)?;
        let num_cams: usize = cameras.len();
        if num_cams == 0 {
            return Err(FrontendError::NoCameras);
        }
        if !options.epipolar_per_camera && num_cams < 2 {
            return Err(FrontendError::NeedsTwoCameras { cameras: num_cams });
        }

        // `E[i]` (`optical_flow.h:207-213`), from `T_c0_ci` per camera unless the
        // C++ behaviour is asked for (deviation X03).
        let essential: Vec<Matrix4<f32>> = (0..num_cams)
            .map(|index| {
                if options.epipolar_per_camera && index == 0 {
                    // `T_c0_c0` has no baseline, so `t.normalized()` would be
                    // NaN; nothing reads index 0 either way (`:704`).
                    return Matrix4::zeros();
                }
                let other: usize = if options.epipolar_per_camera {
                    index
                } else {
                    1
                };
                let t_i_j: Se3<f32> = calib.t_i_c[0].inverse() * calib.t_i_c[other];
                cast_matrix4(&compute_essential(&t_i_j.cast::<f64>()))
            })
            .collect();

        // `detectKeypointsWithCells` derives its grid from the image it is given
        // (`keypoints.cpp:140-144`), so every camera gets its own; `cells` is
        // shaped from camera 0 alone (`:119`), so that shape is separate.
        let cell: usize = config.optical_flow_detection_grid_size as usize;
        let mut detection_grids: Vec<CellGrid> = Vec::with_capacity(num_cams);
        for (camera, rig_camera) in cameras.iter().enumerate() {
            let width: usize = rig_camera.width() as usize;
            let height: usize = rig_camera.height() as usize;
            let grid: CellGrid =
                CellGrid::new(width, height, cell).ok_or(FrontendError::FrameTooSmall {
                    camera,
                    width,
                    height,
                    cell,
                })?;
            // Two reasons the ceiling is per camera: camera 0's grid sizes
            // `cells` below, one `i32` per cell per camera, and a `Vec` too long
            // to exist panics rather than returning (decision D32); and every
            // camera is detected on its own grid, which bounds that camera's
            // detection scan on every frame (`detect.rs:494`). Saturating like
            // the sibling ceilings: a product that leaves `usize` is past it.
            if grid.rows.saturating_mul(grid.columns) > MAX_CELLS {
                return Err(FrontendError::TooManyCells {
                    camera,
                    rows: grid.rows,
                    columns: grid.columns,
                    ceiling: MAX_CELLS,
                });
            }
            detection_grids.push(grid);
        }
        let occupancy_grid: CellGrid = detection_grids[0];

        let patches: T::Patches = tracker.make_patches()?;
        Ok(Self {
            depth_guess: config.optical_flow_matching_default_depth,
            patches,
            tracker,
            cells: vec![vec![0; occupancy_grid.rows * occupancy_grid.columns]; num_cams],
            masks: vec![Masks::default(); num_cams],
            pyramid: Vec::new(),
            staging: Vec::new(),
            snapshot: FrameState::default(),
            timings: FlowTimings::default(),
            pyramid_builder: builder,
            ids: Vec::new(),
            source: FlowTransforms::default(),
            offered: Vec::new(),
            positions: PointsSoA::default(),
            guesses: FlowTransforms::default(),
            result: FlowResult::default(),
            tracked_ids: Vec::new(),
            tracked: FlowTransforms::default(),
            detector: DetectorScratch::with_scanner(scanner),
            detected: KeypointsData::default(),
            new_cam0: Keypoints::default(),
            to_remove: Vec::new(),
            frame: FlowFrame {
                t_ns: None,
                cameras: vec![Keypoints::default(); num_cams],
            },
            last_keypoint_id: 0,
            last_keypoint_id_before_frame: 0,
            t_ns: None,
            frame_counter: 0,
            config,
            options,
            calib,
            cameras,
            essential,
            detection_grids,
            occupancy_grid,
            pattern: PhantomData,
        })
    }

    /// Cameras on the rig.
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// The next keypoint id that will be handed out (`optical_flow.h:174`).
    pub fn last_keypoint_id(&self) -> u64 {
        self.last_keypoint_id
    }

    /// The id space's watermark before the last committed frameset: every
    /// keypoint at or above it was handed out on that frameset.
    ///
    /// Zero before the first frameset commits, and unchanged by a frameset the
    /// frontend refuses — that one is rolled back whole, so the last committed
    /// frame and its watermark still describe each other.
    pub fn last_keypoint_id_before_frame(&self) -> u64 {
        self.last_keypoint_id_before_frame
    }

    /// Framesets processed so far (`optical_flow.h:173`).
    pub fn frame_counter(&self) -> u64 {
        self.frame_counter
    }

    /// The keypoints of the most recent frameset.
    pub fn frame(&self) -> &FlowFrame {
        &self.frame
    }

    /// What the most recent frame's phases cost, wall clock.
    pub fn timings(&self) -> FlowTimings {
        self.timings
    }

    /// One camera's occupancy counts, row-major over
    /// [`FrameToFrameOpticalFlow::occupancy_grid`].
    ///
    /// # Panics
    ///
    /// If `camera` is past the end of the rig.
    pub fn cell_counts(&self, camera: usize) -> &[i32] {
        &self.cells[camera]
    }

    /// The grid `cells` is shaped and indexed by, from camera 0 (`:119`).
    pub fn occupancy_grid(&self) -> CellGrid {
        self.occupancy_grid
    }

    /// The grid one camera is *detected* on, from its own image size
    /// (`keypoints.cpp:140-144`).
    ///
    /// # Panics
    ///
    /// If `camera` is past the end of the rig.
    pub fn detection_grid(&self, camera: usize) -> CellGrid {
        self.detection_grids[camera]
    }

    /// One camera's essential matrix against camera 0 (`optical_flow.h:221`).
    ///
    /// Index 0 is the zero matrix: camera 0 has no epipolar geometry against
    /// itself, and `filterPoints` starts at camera 1 (`:704`), so nothing reads
    /// it. The C++ leaves the cam0-cam1 matrix there instead, equally unread.
    ///
    /// # Panics
    ///
    /// If `camera` is past the end of the rig.
    pub fn essential(&self, camera: usize) -> Matrix4<f32> {
        self.essential[camera]
    }

    /// The configuration this frontend runs.
    pub fn config(&self) -> &VioConfig {
        &self.config
    }

    /// Timestamp of the most recent frameset, `None` before the first
    /// (`optical_flow.h:172`).
    pub fn t_ns(&self) -> Option<i64> {
        self.t_ns
    }

    /// `depth_guess`: the estimator's average scene depth (`optical_flow.h:164`).
    pub fn depth_guess(&self) -> f32 {
        self.depth_guess
    }

    /// `depth_guess`: the estimator's average scene depth (`optical_flow.h:164`).
    pub fn set_depth_guess(&mut self, depth: f32) {
        self.depth_guess = depth;
    }

    /// Everything a frameset must be before [`Self::process_frame`] touches
    /// anything: the clock, the camera count, and each image's size.
    ///
    /// `sizes` is one `(width, height)` per camera, in rig order. It is a
    /// precondition rather than the first lines of `process_frame` because a
    /// caller that owns more than the frontend has to be able to ask it first:
    /// [`slam_rs::Vio::track`](crate::Vio::track) runs the frontend's own IMU
    /// preintegration before this call, and a preintegration spent on a frameset
    /// that is then refused cannot be spent again (D17). `process_frame` still
    /// runs it, so the frontend's own entry point keeps the guarantee alone.
    ///
    /// # Errors
    ///
    /// [`FrontendError::NonMonotonicFrameset`] when the frameset does not follow
    /// the last accepted one, [`FrontendError::CameraCountMismatch`] on the
    /// wrong width, and [`FrontendError::FrameSizeMismatch`] when an image is
    /// not the size the calibration gives that camera.
    pub fn check_frameset(
        &self,
        t_ns: i64,
        sizes: impl ExactSizeIterator<Item = (usize, usize)>,
    ) -> Result<(), FrontendError> {
        // Tracking is frame to frame, so a frameset that does not follow the last
        // one has no previous frame of its own; basalt never sees one because its
        // `processingLoop` reads a monotonic queue.
        if let Some(previous_t_ns) = self.t_ns
            && t_ns <= previous_t_ns
        {
            return Err(FrontendError::NonMonotonicFrameset {
                previous_t_ns,
                t_ns,
            });
        }
        let num_cams: usize = self.cameras.len();
        if sizes.len() != num_cams {
            return Err(FrontendError::CameraCountMismatch {
                expected: num_cams,
                actual: sizes.len(),
            });
        }
        // The geometry is the calibration's from here on: the camera model
        // projects with the calibrated intrinsics, the detection grid is derived
        // from the calibrated size, and the occupancy matrix is allocated from
        // it. A frame of another size is not a smaller view of the same scene —
        // its pixels mean different bearings — so it is refused rather than
        // tracked against geometry it does not belong to. basalt never checks:
        // its `img_data` comes from the device the calibration describes.
        for (camera, (actual_width, actual_height)) in sizes.enumerate() {
            let expected_width: usize = self.cameras[camera].width() as usize;
            let expected_height: usize = self.cameras[camera].height() as usize;
            if actual_width != expected_width || actual_height != expected_height {
                return Err(FrontendError::FrameSizeMismatch {
                    camera,
                    expected_width,
                    expected_height,
                    actual_width,
                    actual_height,
                });
            }
        }
        Ok(())
    }

    /// `processFrame` (`frame_to_frame_optical_flow.h:203-292`).
    ///
    /// `images` holds one frame per camera, already widened to 16 bits.
    /// `prediction` is the pose pair the estimator's feedback would have
    /// produced; two identities mean "no state has arrived yet", which is what
    /// basalt runs with until the backend answers (`:140-147`). `masks` may be
    /// shorter than the rig or empty, in which case the missing cameras suppress
    /// nothing.
    ///
    /// **A rejected frame is as if it never happened.** The C++ has no such
    /// problem because it never rejects a frame; the port can, at three points —
    /// the frameset width, a pyramid geometry, and any error a pluggable backend
    /// returns from the middle of tracking — and every one of them must leave the
    /// frontend exactly as the last successful frame did. So the whole call runs
    /// against a *staging* pyramid set with `pyramid` still holding the previous
    /// frame, over a snapshot of the keypoint state; only after tracking, the
    /// cell counts, the add/match passes and the epipolar filter have all
    /// succeeded do the two pyramid sets swap and the clock advance. On any error
    /// the keypoints, the occupancy counts and the id counter are restored and
    /// the timestamp never moved.
    ///
    /// # Errors
    ///
    /// [`FrontendError`] when the frameset does not move the clock forward, when
    /// it is the wrong width, when an image is not the size the calibration gives
    /// that camera, when a pyramid refuses the geometry, or when the tracker or
    /// detector refuses an input.
    pub fn process_frame(
        &mut self,
        t_ns: i64,
        images: &[ImageU16],
        prediction: &PosePrediction,
        masks: &[Masks],
    ) -> Result<&FlowFrame, FrontendError> {
        self.check_frameset(
            t_ns,
            images.iter().map(|image| (image.width(), image.height())),
        )?;

        // The frame in flight, built where nothing else can see it.
        self.timings = FlowTimings::default();
        let mark: std::time::Instant = std::time::Instant::now();
        self.build_staging(images)?;
        self.timings.pyramid_ns = duration_ns(mark);

        // What the passes below mutate, kept so they can be undone.
        let mut snapshot: FrameState = std::mem::take(&mut self.snapshot);
        snapshot.cameras.clone_from(&self.frame.cameras);
        snapshot.cells.clone_from(&self.cells);
        snapshot.last_keypoint_id = self.last_keypoint_id;

        let outcome: Result<(), FrontendError> = self.run_passes(images, prediction, masks);

        match &outcome {
            Ok(()) => {
                // Commit: the frame in flight becomes the committed one, and the
                // set it displaces is next frame's staging buffer.
                std::mem::swap(&mut self.pyramid, &mut self.staging);
                self.t_ns = Some(t_ns);
                self.frame.t_ns = Some(t_ns);
                self.frame_counter += 1;
                // The snapshot holds the id space as it stood before this frame,
                // which is exactly what a reader of the committed frame needs.
                self.last_keypoint_id_before_frame = snapshot.last_keypoint_id;
            }
            Err(_) => {
                self.frame.cameras.clone_from(&snapshot.cameras);
                self.cells.clone_from(&snapshot.cells);
                self.last_keypoint_id = snapshot.last_keypoint_id;
            }
        }
        self.snapshot = snapshot;
        outcome?;
        Ok(&self.frame)
    }

    /// Steps 4 to 7 of `processFrame`, against `staging` as the current frame.
    ///
    /// Split out so [`FrameToFrameOpticalFlow::process_frame`] can undo them: no
    /// step here touches the pyramid sets, the timestamp or the frame counter.
    fn run_passes(
        &mut self,
        images: &[ImageU16],
        prediction: &PosePrediction,
        masks: &[Masks],
    ) -> Result<(), FrontendError> {
        for (index, mask) in self.masks.iter_mut().enumerate() {
            mask.masks.clear();
            if let Some(source) = masks.get(index) {
                mask.extend(source);
            }
        }

        let num_cams: usize = self.cameras.len();
        if self.t_ns.is_none() {
            for keypoints in &mut self.frame.cameras {
                keypoints.clear();
            }
            for counts in &mut self.cells {
                counts.fill(0);
            }
        } else {
            // `for (i) trackPoints(old_pyramid[i], pyramid[i], ..., T_c1_c2, i, i)`
            // (`:261-271`).
            let t_i1: Se3<f32> = prediction.t_w_i_previous;
            let t_i2: Se3<f32> = prediction.t_w_i_current;
            for camera in 0..num_cams {
                let t_c1: Se3<f32> = t_i1 * self.calib.t_i_c[camera];
                let t_c2: Se3<f32> = t_i2 * self.calib.t_i_c[camera];
                let t_c1_c2: Se3<f32> = t_c1.inverse() * t_c2;
                self.track_camera(camera, &t_c1_c2)?;
            }
            // `for (i) updateCellCounts(i)` (`:277`).
            for camera in 0..num_cams {
                self.update_cell_counts(camera);
            }
        }

        self.add_points(images)?;
        let mark: std::time::Instant = std::time::Instant::now();
        self.filter_points();
        self.timings.stereo_ns += duration_ns(mark);
        Ok(())
    }

    /// Build this frame's pyramids into the staging set, touching nothing else.
    ///
    /// `pyramid->at(i).setFromImage(img, config.optical_flow_levels)` (`:245-249`),
    /// with the allocation reused whenever the geometry is unchanged.
    fn build_staging(&mut self, images: &[ImageU16]) -> Result<(), FrontendError> {
        let levels: usize = self.config.optical_flow_levels as usize;
        self.staging.truncate(images.len());
        for (index, image) in images.iter().enumerate() {
            let fits: bool = self.staging.get(index).is_some_and(|pyramid| {
                pyramid.num_levels() == levels + 1
                    && pyramid
                        .level_size(0)
                        .is_some_and(|(w, h, _)| w == image.width() && h == image.height())
            });
            if !fits {
                let fresh: B::Pyramid =
                    self.pyramid_builder
                        .allocate(image.width(), image.height(), levels)?;
                match self.staging.get_mut(index) {
                    Some(slot) => *slot = fresh,
                    None => self.staging.push(fresh),
                }
            }
            if !B::PREPARE_IMAGES {
                self.pyramid_builder
                    .build(index, image, &mut self.staging[index])?;
            }
        }
        if B::PREPARE_IMAGES {
            self.pyramid_builder.prepare_images(images)?;
            for (index, image) in images.iter().enumerate() {
                self.pyramid_builder
                    .build(index, image, &mut self.staging[index])?;
            }
        }
        Ok(())
    }

    /// One camera's frame-to-frame track: `trackPoints(..., cam, cam)` (`:267-270`).
    fn track_camera(&mut self, camera: usize, t_c1_c2: &Se3<f32>) -> Result<(), FrontendError> {
        // Source and destination are the same slot (`:299-308`, `:371`), so the
        // ids and warps are copied out first — and the slot is only cleared once
        // the track has succeeded, so a refused frame does not lose the camera's
        // keypoints.
        self.ids.clear();
        self.source.clear();
        self.ids.extend_from_slice(&self.frame.cameras[camera].ids);
        for index in 0..self.frame.cameras[camera].len() {
            self.source
                .push(&self.frame.cameras[camera].transforms.get(index));
        }

        self.run_track_points(camera, camera, t_c1_c2, true)?;

        self.frame.cameras[camera].clear();
        for (slot, id) in self.tracked_ids.iter().enumerate() {
            // `keypoint_map_2.insert(result.begin(), result.end())` (`:372`); the
            // cell counts are rebuilt afterwards by `updateCellCounts`.
            self.frame.cameras[camera].set(*id, &self.tracked.get(slot), NO_RESPONSE);
        }
        Ok(())
    }

    /// The body of `trackPoints` (`:294-375`) minus the map bookkeeping.
    ///
    /// Reads `ids` and `source`, writes `tracked_ids` and `tracked`. The caller
    /// decides what to do with the survivors, because `trackPoints` serves two
    /// purposes: carrying a camera's own keypoints forward in time, and matching
    /// camera 0's new keypoints into camera *i*.
    fn run_track_points(
        &mut self,
        cam1: usize,
        cam2: usize,
        t_c1_c2: &Se3<f32>,
        tracking: bool,
    ) -> Result<(), FrontendError> {
        // `use_depth = tracking || (matching && guess_type != SAME_PIXEL)` (`:316`).
        let use_depth: bool = tracking
            || self.config.optical_flow_matching_guess_type != MatchingGuessType::SamePixel;
        let depth: f32 = self.depth_guess;

        self.offered.clear();
        self.positions.clear();
        self.guesses.clear();
        self.tracked_ids.clear();
        self.tracked.clear();

        for index in 0..self.source.len() {
            let transform_1: AffineCompact2f = self.source.get(index);
            let t1: Vector2<f32> = transform_1.translation;
            // `if (masks1.inBounds(t1.x(), t1.y())) continue;` (`:329`).
            if self.masks[cam1].in_bounds(t1.x, t1.y) {
                continue;
            }
            // `off = t2 - t2_guess` with `t2 == t1` (`:333-340`), then `t2 -= off`
            // (`:342`), so the guess is simply `t2_guess`.
            let translation: Vector2<f32> = if use_depth {
                project_between_cams(&self.cameras, &t1, depth, t_c1_c2, cam1, cam2).1
            } else {
                t1
            };
            self.offered.push(index);
            self.positions.push(t1);
            self.guesses.push(&AffineCompact2f {
                linear: transform_1.linear,
                translation,
            });
        }

        // The forward source patches come from the previous frame when tracking
        // and from this frame's camera 0 when matching (`:267`, `:652`). This
        // frame is `staging` until the call commits.
        let source_pyramid: &B::Pyramid = if tracking {
            &self.pyramid[cam1]
        } else {
            &self.staging[cam1]
        };
        let mark: std::time::Instant = std::time::Instant::now();
        self.patches
            .prepare(source_pyramid, &self.positions, None)?;
        self.tracker.track_prepared(
            source_pyramid,
            &self.staging[cam2],
            &self.patches,
            &self.guesses,
            &mut self.result,
        )?;
        if tracking {
            self.timings.track_ns += duration_ns(mark);
        }

        for slot in self.result.tracked() {
            let slot: usize = *slot as usize;
            let transform: AffineCompact2f = self.result.transform(slot);
            // `if (masks2.inBounds(t2.x(), t2.y())) continue;` (`:352`).
            if self.masks[cam2].in_bounds(transform.translation.x, transform.translation.y) {
                continue;
            }
            self.tracked_ids.push(self.ids[self.offered[slot]]);
            self.tracked.push(&transform);
        }
        Ok(())
    }

    /// `updateCellCounts` (`:707-716`): rebuild one camera's occupancy from scratch.
    fn update_cell_counts(&mut self, camera: usize) {
        self.cells[camera].fill(0);
        for index in 0..self.frame.cameras[camera].len() {
            let position: Vector2<f32> = self.frame.cameras[camera].transforms.translation(index);
            // `if (p[0] < x_start || ... || p[1] >= y_stop + c) continue;` (`:711`).
            if !self.occupancy_grid.contains(position.x, position.y) {
                continue;
            }
            let (row, column) = self.occupancy_grid.cell_of(position.x, position.y);
            self.cells[camera][row * self.occupancy_grid.columns + column] += 1;
        }
    }

    /// The `cells(y, x)++` half of `addKeypoint`/`addKeypoints` (`:729`, `:738`).
    fn bump_cell(&mut self, camera: usize, transform: &AffineCompact2f) {
        let (row, column) = self
            .occupancy_grid
            .cell_of(transform.translation.x, transform.translation.y);
        self.cells[camera][row * self.occupancy_grid.columns + column] += 1;
    }

    /// `removeKeypoint` (`:743-749`): drop a keypoint and decrement its cell.
    fn remove_keypoint(&mut self, camera: usize, id: KeypointId) {
        let Some(transform) = self.frame.cameras[camera].remove(id) else {
            return;
        };
        let (row, column) = self
            .occupancy_grid
            .cell_of(transform.translation.x, transform.translation.y);
        self.cells[camera][row * self.occupancy_grid.columns + column] -= 1;
    }

    /// `addPointsForCamera` (`:577-610`): detect in the empty cells, register the
    /// corners under fresh ids, and record camera 0's as the ones to match onward.
    ///
    /// Detection is capped at the camera's remaining budget
    /// ([`FrontendOptions::max_keypoints`]), so the frame it produces is always
    /// one the tracker can carry next time.
    fn add_points_for_camera(
        &mut self,
        camera: usize,
        images: &[ImageU16],
    ) -> Result<(), FrontendError> {
        let config: DetectorConfig = DetectorConfig {
            num_points_cell: self.config.optical_flow_detection_num_points_cell as usize,
            min_threshold: self.config.optical_flow_detection_min_threshold,
            max_threshold: self.config.optical_flow_detection_max_threshold,
            safe_radius: self.config.optical_flow_image_safe_radius,
        };
        let budget: usize = self
            .options
            .max_keypoints
            .saturating_sub(self.frame.cameras[camera].len());

        // `detectKeypointsWithCells(pyramid->at(cam_id).lvl(0), ...)` (`:579-582`),
        // on this camera's own grid and the rig's shared occupancy matrix.
        //
        // Level 0 of the staging pyramid is this frame's input image copied in
        // unchanged (`image_pyr.h:73`), and the caller still holds it, so the
        // detector reads the image itself: same pixels, and no copy out of the
        // pyramid — whose seam lends nothing (deviation X04).
        self.detected.corners.clear();
        self.detected.responses.clear();
        let mark: std::time::Instant = std::time::Instant::now();
        if budget > 0 {
            detect_keypoints_with_cells(
                &images[camera],
                camera,
                &self.detection_grids[camera],
                &Occupancy {
                    counts: &self.cells[camera],
                    rows: self.occupancy_grid.rows,
                    columns: self.occupancy_grid.columns,
                },
                &config,
                &self.masks[camera],
                budget,
                &mut self.detector,
                &mut self.detected,
            )?;
        }
        self.timings.detect_ns += duration_ns(mark);

        self.new_cam0.clear();
        for index in 0..self.detected.corners.len() {
            let corner: [f32; 2] = self.detected.corners[index];
            let response: f32 = self.detected.responses[index];
            let transform: AffineCompact2f =
                AffineCompact2f::at(Vector2::new(corner[0], corner[1]));
            let id: KeypointId = KeypointId(self.last_keypoint_id);
            // `addKeypoint` (`:726-732`): bump the cell, then register.
            self.bump_cell(camera, &transform);
            self.frame.cameras[camera].set(id, &transform, response);
            if camera == 0 {
                self.new_cam0.set(id, &transform, response);
            }
            // `last_keypoint_id++` (`:606`), the global landmark id space.
            self.last_keypoint_id += 1;
        }
        Ok(())
    }

    /// `addKeypoints` (`:734-741`): bump a cell for **every** offered keypoint,
    /// then insert the ones whose id is not already present.
    ///
    /// The double count when a keypoint is both tracked in camera *i* and matched
    /// into it from camera 0 is basalt's, not a slip: `std::map::insert` keeps the
    /// existing entry while the loop above it has already incremented the cell.
    /// The budget is the port's own: once the camera is full, the remaining
    /// matches are dropped and their cells are not bumped either.
    fn add_keypoints(&mut self, camera: usize) {
        let count: usize = self.tracked_ids.len();
        for slot in 0..count {
            if self.frame.cameras[camera].len() >= self.options.max_keypoints {
                break;
            }
            let transform: AffineCompact2f = self.tracked.get(slot);
            self.bump_cell(camera, &transform);
            self.frame.cameras[camera].insert_if_absent(self.tracked_ids[slot], &transform);
        }
    }

    /// `cam0OverlapCellsMasksForCam` (`:612-635`): mask the cells of camera
    /// `camera` that project into camera 0 at `depth_guess`.
    ///
    /// The grid walked here is the frontend's own `x_start`/`x_stop`, which the
    /// C++ takes from camera 0 (`:110-113`) whatever camera is being masked.
    /// The rectangles are appended to camera `camera`'s own mask list, which
    /// `run_passes` clears once per frameset: `optical_flow_detection_nonoverlap`
    /// is `true` in every shipped config, so returning a fresh `Masks` here
    /// allocated and freed up to `rows x columns` = 361 `Rect` per camera per
    /// frameset on the msd rigs — the one per-frame allocation left in a module
    /// whose doc promises none. The destructure is what lets one field be
    /// written while the others are read.
    fn append_cam0_overlap_masks(&mut self, camera: usize) {
        let Self {
            masks,
            cameras,
            calib,
            occupancy_grid,
            depth_guess,
            ..
        } = self;
        let grid: CellGrid = *occupancy_grid;
        let cell: usize = grid.cell;
        let half: usize = cell / 2;
        let x_first: usize = grid.x_start + half;
        let y_first: usize = grid.y_start + half;
        let x_last: usize = grid.x_stop + half;
        let y_last: usize = grid.y_stop + half;

        let width: f32 = cameras[0].resolution[0] as f32;
        let height: f32 = cameras[0].resolution[1] as f32;
        let t_ci_c0: Se3<f32> = calib.t_i_c[camera].inverse() * calib.t_i_c[0];

        let out: &mut Masks = &mut masks[camera];
        let mut y: usize = y_first;
        while y <= y_last {
            let mut x: usize = x_first;
            while x <= x_last {
                let ci_uv: Vector2<f32> = Vector2::new(x as f32, y as f32);
                let (projected, c0_uv) =
                    project_between_cams(cameras, &ci_uv, *depth_guess, &t_ci_c0, camera, 0);
                let in_bounds: bool =
                    c0_uv.x >= 0.0 && c0_uv.x < width && c0_uv.y >= 0.0 && c0_uv.y < height;
                if projected && in_bounds {
                    out.masks.push(Rect {
                        x: (x - half) as f32,
                        y: (y - half) as f32,
                        w: cell as f32,
                        h: cell as f32,
                    });
                }
                x += cell;
            }
            y += cell;
        }
    }

    /// `addPoints` (`:637-666`): detect on camera 0, match onward, then detect
    /// again on the cameras that do not overlap camera 0.
    fn add_points(&mut self, images: &[ImageU16]) -> Result<(), FrontendError> {
        self.add_points_for_camera(0, images)?;

        // `for (i = 1; i < getNumCams(); i++) trackPoints(pyr0, pyri, kpts0, ...)`
        // (`:643-654`). With one camera there is nothing to match into (trap 17).
        let mark: std::time::Instant = std::time::Instant::now();
        for camera in 1..self.cameras.len() {
            self.ids.clear();
            self.source.clear();
            self.ids.extend_from_slice(&self.new_cam0.ids);
            for index in 0..self.new_cam0.len() {
                self.source.push(&self.new_cam0.transforms.get(index));
            }
            let t_c0_ci: Se3<f32> = self.calib.t_i_c[0].inverse() * self.calib.t_i_c[camera];
            self.run_track_points(0, camera, &t_c0_ci, false)?;
            self.add_keypoints(camera);
        }

        self.timings.stereo_ns += duration_ns(mark);

        // `if (!config.optical_flow_detection_nonoverlap) continue;` (`:657-664`).
        if self.config.optical_flow_detection_nonoverlap {
            for camera in 1..self.cameras.len() {
                self.append_cam0_overlap_masks(camera);
                self.add_points_for_camera(camera, images)?;
            }
        }
        Ok(())
    }

    /// `filterPointsForCam` (`:668-701`): drop the keypoints camera `camera`
    /// shares with camera 0 whose epipolar error is too large.
    fn filter_points_for_cam(&mut self, camera: usize) {
        self.to_remove.clear();
        let essential: Matrix4<f32> = self.essential[camera];
        let threshold: f32 = self.config.optical_flow_epipolar_error;

        for index in 0..self.frame.cameras[camera].ids.len() {
            let id: KeypointId = self.frame.cameras[camera].ids[index];
            let Some(in_cam0) = self.frame.cameras[0].get(id) else {
                continue;
            };
            let proj1: Vector2<f32> = self.frame.cameras[camera].transforms.translation(index);

            let mut p3d0: Vector4<f32> = Vector4::zeros();
            let mut p3d1: Vector4<f32> = Vector4::zeros();
            let ok0: bool = self.cameras[0]
                .model
                .unproject(&in_cam0.translation, &mut p3d0);
            let ok1: bool = self.cameras[camera].model.unproject(&proj1, &mut p3d1);

            if ok0 && ok1 {
                // `std::abs(p3d0.transpose() * E[cam] * p3d1)` (`:692`), in the
                // estimator's scalar and only then widened for the comparison.
                let error: f32 = (p3d0.transpose() * essential * p3d1)[(0, 0)].abs();
                if error > threshold {
                    self.to_remove.push(id);
                }
            } else {
                self.to_remove.push(id);
            }
        }

        // `for (int id : kp_to_remove) removeKeypoint(cam_id, id);` (`:700`), a
        // `std::set`, so ascending; `self.to_remove` is built in id order
        // already. Indexed rather than drained because `remove_keypoint` takes
        // `&mut self`, and the list does not change while it runs.
        for slot in 0..self.to_remove.len() {
            let id: KeypointId = self.to_remove[slot];
            self.remove_keypoint(camera, id);
        }
        self.to_remove.clear();
    }

    /// `filterPoints` (`:703-705`): every camera but camera 0.
    fn filter_points(&mut self) {
        for camera in 1..self.cameras.len() {
            self.filter_points_for_cam(camera);
        }
    }
}

/// `computeEssential` (`utils/keypoints.h:101-107`).
///
/// `E.topLeftCorner<3,3>() = hat(t.normalized()) * R`, computed in `double`
/// because that is what `optical_flow.h:209-212` casts to before calling it, and
/// cast back to the estimator's scalar afterwards.
fn compute_essential(t_0_1: &Se3<f64>) -> Matrix4<f64> {
    let translation: Vector3<f64> = t_0_1.translation;
    let rotation = t_0_1.rotation.matrix();
    let mut essential: Matrix4<f64> = Matrix4::zeros();
    // `Eigen::normalized()` is `v / v.norm()`, which is NaN for a zero baseline;
    // the C++ does the same, and a rig with two coincident cameras has no
    // epipolar geometry to test against anyway.
    let normalized: Vector3<f64> = translation / translation.norm();
    essential
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(So3::hat(&normalized) * rotation));
    essential
}

/// `Ed.cast<Scalar>()` (`optical_flow.h:212`).
fn cast_matrix4(matrix: &Matrix4<f64>) -> Matrix4<f32> {
    Matrix4::from_iterator(matrix.iter().map(|value| *value as f32))
}

/// `Calibration::projectBetweenCams` (`calibration/calibration.hpp:79-95`).
///
/// Returns the validity flag *and* the pixel. `trackPoints` discards the flag and
/// uses whatever `project` wrote (`frame_to_frame_optical_flow.h:338`), while
/// `cam0OverlapCellsMasksForCam` reads it (`:625-627`); both behaviours are the
/// C++'s, so the pixel is always produced and the caller decides.
///
/// # Panics
///
/// If `i` or `j` is past the end of `cameras`.
pub fn project_between_cams(
    cameras: &[RigCamera<f32>],
    ci_uv: &Vector2<f32>,
    ci_depth: f32,
    t_ci_cj: &Se3<f32>,
    i: usize,
    j: usize,
) -> (bool, Vector2<f32>) {
    let mut ci_xyzw: Vector4<f32> = Vector4::zeros();
    let mut valid: bool = cameras[i].model.unproject(ci_uv, &mut ci_xyzw);
    ci_xyzw *= ci_depth;
    ci_xyzw.w = 1.0;

    // `T_ci_cj.inverse() * ci_xyzw`: Sophus rotates and translates the first
    // three components and carries the fourth through unchanged (`se3.hpp`).
    let inverse: Se3<f32> = t_ci_cj.inverse();
    let point: Vector3<f32> = inverse * Vector3::new(ci_xyzw.x, ci_xyzw.y, ci_xyzw.z);
    let cj_xyzw: Vector4<f32> = Vector4::new(point.x, point.y, point.z, ci_xyzw.w);

    let mut cj_uv: Vector2<f32> = Vector2::zeros();
    valid &= cameras[j].model.project(&cj_xyzw, &mut cj_uv);
    (valid, cj_uv)
}
