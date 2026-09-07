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
//! * **`pyramid_levels`.** `FrameToFrameOpticalFlow` never writes it; only
//!   `MultiscaleFrameToFrameOpticalFlow` does (`:57-59`). The field is on
//!   [`Keypoints`] so the shape of the output does not change when that variant
//!   arrives, and it stays empty here.
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
use crate::frontend::detect::{
    CellGrid, DetectError, DetectorConfig, DetectorScratch, KeypointsData, Masks, Occupancy, Rect,
    detect_keypoints_with_cells,
};
use crate::frontend::parallel::WorkPool;
use crate::frontend::patterns::Pattern;
use crate::frontend::se2::AffineCompact2f;
use crate::frontend::tracker::{
    CpuPatchTracker, FlowResult, FlowTransforms, PatchTracker, PointsSoA, SourcePatches,
    TrackerError,
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
    /// configuration comes near it.
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
    /// `OpticalFlowResult::pyramid_levels`, always empty for this variant.
    pub pyramid_levels: Vec<u32>,
}

/// `Clone` by hand for the sake of `clone_from`.
///
/// `#[derive(Clone)]` writes only `clone`; the default `clone_from` is
/// `*self = source.clone()`, which drops all four buffers and allocates four
/// more — sixteen allocations and sixteen frees per stereo frame once
/// [`FrameToFrameOpticalFlow::process_frame`] takes its snapshot and again if it
/// has to restore. Copying field by field lets `Vec::clone_from` overwrite in
/// place, and `FlowTransforms` does the same one level down.
impl Clone for Keypoints {
    fn clone(&self) -> Self {
        Self {
            ids: self.ids.clone(),
            transforms: self.transforms.clone(),
            responses: self.responses.clone(),
            pyramid_levels: self.pyramid_levels.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.ids.clone_from(&source.ids);
        self.transforms.clone_from(&source.transforms);
        self.responses.clone_from(&source.responses);
        self.pyramid_levels.clone_from(&source.pyramid_levels);
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

    /// The response stored for `id`, or `None`.
    pub fn response(&self, id: KeypointId) -> Option<f32> {
        self.index_of(id).map(|index| self.responses[index])
    }

    /// The warp at `index`, in id order.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
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
        self.pyramid_levels.clear();
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
    /// Frameset timestamp, nanoseconds.
    pub t_ns: i64,
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
    /// `t_ns`, negative until the first frame (`optical_flow.h:172`).
    t_ns: i64,
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
    pattern: PhantomData<P>,
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
    /// [`FrontendError`] when the config names another flow type or pattern,
    /// when the rig is empty, ragged or too small for the grid, when a camera
    /// model has no projection, or when the thread pool cannot be built.
    pub fn new(
        config: VioConfig,
        calibration: &Calibration<f64>,
        options: FrontendOptions,
    ) -> Result<Self, FrontendError> {
        Self::validate_config(&config)?;
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
        );
        Self::with_backends(
            config,
            calibration,
            options,
            CpuPyramidBuilder::new(),
            tracker,
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
        Ok(())
    }

    /// Build a frontend on caller-supplied stages.
    ///
    /// This is the seam a GPU backend enters through: `builder` and `tracker` are
    /// any pair whose pyramid types agree, and the patch storage comes from the
    /// tracker itself.
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
    ) -> Result<Self, FrontendError> {
        Self::validate_config(&config)?;

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
            detection_grids.push(CellGrid::new(width, height, cell).ok_or(
                FrontendError::FrameTooSmall {
                    camera,
                    width,
                    height,
                    cell,
                },
            )?);
        }
        let occupancy_grid: CellGrid = detection_grids[0];

        Ok(Self {
            depth_guess: config.optical_flow_matching_default_depth,
            patches: tracker.make_patches(),
            tracker,
            cells: vec![vec![0; occupancy_grid.rows * occupancy_grid.columns]; num_cams],
            masks: vec![Masks::default(); num_cams],
            pyramid: Vec::new(),
            staging: Vec::new(),
            snapshot: FrameState::default(),
            pyramid_builder: builder,
            ids: Vec::new(),
            source: FlowTransforms::default(),
            offered: Vec::new(),
            positions: PointsSoA::default(),
            guesses: FlowTransforms::default(),
            result: FlowResult::default(),
            tracked_ids: Vec::new(),
            tracked: FlowTransforms::default(),
            detector: DetectorScratch::default(),
            detected: KeypointsData::default(),
            new_cam0: Keypoints::default(),
            to_remove: Vec::new(),
            frame: FlowFrame {
                t_ns: -1,
                cameras: vec![Keypoints::default(); num_cams],
            },
            last_keypoint_id: 0,
            t_ns: -1,
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

    /// Framesets processed so far (`optical_flow.h:173`).
    pub fn frame_counter(&self) -> u64 {
        self.frame_counter
    }

    /// The keypoints of the most recent frameset.
    pub fn frame(&self) -> &FlowFrame {
        &self.frame
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

    /// The port-specific options this frontend was built with.
    pub fn options(&self) -> FrontendOptions {
        self.options
    }

    /// The configuration this frontend runs.
    pub fn config(&self) -> &VioConfig {
        &self.config
    }

    /// Timestamp of the most recent frameset, negative before the first
    /// (`optical_flow.h:172`).
    pub fn t_ns(&self) -> i64 {
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
    /// [`FrontendError`] when the frameset is the wrong width, a pyramid refuses
    /// the geometry, or the tracker or detector refuses an input.
    pub fn process_frame(
        &mut self,
        t_ns: i64,
        images: &[ImageU16],
        prediction: &PosePrediction,
        masks: &[Masks],
    ) -> Result<&FlowFrame, FrontendError> {
        let num_cams: usize = self.cameras.len();
        if images.len() != num_cams {
            return Err(FrontendError::CameraCountMismatch {
                expected: num_cams,
                actual: images.len(),
            });
        }

        // The frame in flight, built where nothing else can see it.
        self.build_staging(images)?;

        // What the passes below mutate, kept so they can be undone.
        let mut snapshot: FrameState = std::mem::take(&mut self.snapshot);
        snapshot.cameras.clone_from(&self.frame.cameras);
        snapshot.cells.clone_from(&self.cells);
        snapshot.last_keypoint_id = self.last_keypoint_id;

        let outcome: Result<(), FrontendError> = self.run_passes(prediction, masks);

        match &outcome {
            Ok(()) => {
                // Commit: the frame in flight becomes the committed one, and the
                // set it displaces is next frame's staging buffer.
                std::mem::swap(&mut self.pyramid, &mut self.staging);
                self.t_ns = t_ns;
                self.frame.t_ns = t_ns;
                self.frame_counter += 1;
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
        if self.t_ns < 0 {
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

        self.add_points()?;
        self.filter_points();
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
            self.pyramid_builder
                .build(image, &mut self.staging[index])?;
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
        self.patches.build(source_pyramid, &self.positions, None)?;
        self.tracker.track(
            source_pyramid,
            &self.staging[cam2],
            &self.patches,
            &self.guesses,
            &mut self.result,
        )?;

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

    /// `addKeypoint` (`:726-732`): register a keypoint and bump its cell.
    fn add_keypoint(
        &mut self,
        camera: usize,
        id: KeypointId,
        transform: &AffineCompact2f,
        response: f32,
    ) {
        self.bump_cell(camera, transform);
        self.frame.cameras[camera].set(id, transform, response);
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
    fn add_points_for_camera(&mut self, camera: usize) -> Result<(), FrontendError> {
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
        self.detected.corners.clear();
        self.detected.responses.clear();
        if budget > 0 {
            let mut level0: ImageU16 = std::mem::take(&mut self.detector.level0);
            let copied: Result<(), PyramidError> =
                self.staging[camera].copy_level_into(0, &mut level0);
            let outcome: Result<(), FrontendError> =
                copied
                    .map_err(FrontendError::from)
                    .and_then(|()| -> Result<(), FrontendError> {
                        detect_keypoints_with_cells(
                            &level0,
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
                        Ok(())
                    });
            self.detector.level0 = level0;
            outcome?;
        }

        self.new_cam0.clear();
        for index in 0..self.detected.corners.len() {
            let corner: [f32; 2] = self.detected.corners[index];
            let response: f32 = self.detected.responses[index];
            let transform: AffineCompact2f =
                AffineCompact2f::at(Vector2::new(corner[0], corner[1]));
            let id: KeypointId = KeypointId(self.last_keypoint_id);
            self.add_keypoint(camera, id, &transform, response);
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
    fn cam0_overlap_masks(&self, camera: usize) -> Masks {
        let grid: CellGrid = self.occupancy_grid;
        let cell: usize = grid.cell;
        let half: usize = cell / 2;
        let x_first: usize = grid.x_start + half;
        let y_first: usize = grid.y_start + half;
        let x_last: usize = grid.x_stop + half;
        let y_last: usize = grid.y_stop + half;

        let width: f32 = self.cameras[0].resolution[0] as f32;
        let height: f32 = self.cameras[0].resolution[1] as f32;
        let t_ci_c0: Se3<f32> = self.calib.t_i_c[camera].inverse() * self.calib.t_i_c[0];

        let mut masks: Masks = Masks::default();
        let mut y: usize = y_first;
        while y <= y_last {
            let mut x: usize = x_first;
            while x <= x_last {
                let ci_uv: Vector2<f32> = Vector2::new(x as f32, y as f32);
                let (projected, c0_uv) = project_between_cams(
                    &self.cameras,
                    &ci_uv,
                    self.depth_guess,
                    &t_ci_c0,
                    camera,
                    0,
                );
                let in_bounds: bool =
                    c0_uv.x >= 0.0 && c0_uv.x < width && c0_uv.y >= 0.0 && c0_uv.y < height;
                if projected && in_bounds {
                    masks.masks.push(Rect {
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
        masks
    }

    /// `addPoints` (`:637-666`): detect on camera 0, match onward, then detect
    /// again on the cameras that do not overlap camera 0.
    fn add_points(&mut self) -> Result<(), FrontendError> {
        self.add_points_for_camera(0)?;

        // `for (i = 1; i < getNumCams(); i++) trackPoints(pyr0, pyri, kpts0, ...)`
        // (`:643-654`). With one camera there is nothing to match into (trap 17).
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

        // `if (!config.optical_flow_detection_nonoverlap) continue;` (`:657-664`).
        if self.config.optical_flow_detection_nonoverlap {
            for camera in 1..self.cameras.len() {
                let overlap: Masks = self.cam0_overlap_masks(camera);
                self.masks[camera].extend(&overlap);
                self.add_points_for_camera(camera)?;
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
        // `std::set`, so ascending; `self.to_remove` is built in id order already.
        let mut to_remove: Vec<KeypointId> = std::mem::take(&mut self.to_remove);
        for id in to_remove.drain(..) {
            self.remove_keypoint(camera, id);
        }
        self.to_remove = to_remove;
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::calib::{CalibAccelBias, CalibGyroBias, CameraModel, PinholeParams};
    use crate::frontend::parallel::WorkPool;
    use crate::frontend::patterns::{Pattern51, Pattern52};
    use crate::frontend::tracker::{PatchSoA, TrackerError};
    use crate::pyramid::{CpuPyramidBuilder, PyramidU16};
    use std::collections::BTreeMap;

    const WIDTH: usize = 200;
    const HEIGHT: usize = 200;

    /// A rig of `count` identical pinhole cameras on a straight baseline.
    ///
    /// The extrinsics are pure translations along `x`, so `computeEssential`
    /// gives `hat(x_hat) * I` for every pair and two cameras that see the same
    /// bearing have exactly zero epipolar error.
    fn rig(count: usize) -> Calibration<f64> {
        let intrinsics: CameraModel<f64> = CameraModel::Pinhole(PinholeParams {
            fx: 180.0,
            fy: 180.0,
            cx: WIDTH as f64 / 2.0,
            cy: HEIGHT as f64 / 2.0,
        });
        Calibration {
            t_i_c: (0..count)
                .map(|index| Se3::new(So3::identity(), Vector3::new(0.05 * index as f64, 0.0, 0.0)))
                .collect(),
            intrinsics: vec![intrinsics; count],
            resolution: vec![[WIDTH as u32, HEIGHT as u32]; count],
            vignette: Vec::new(),
            cam_time_offset_ns: 0,
            calib_accel_bias: CalibAccelBias::default(),
            calib_gyro_bias: CalibGyroBias::default(),
            imu_update_rate: 200.0,
            gyro_noise_std: Vector3::repeat(1e-4),
            accel_noise_std: Vector3::repeat(1e-3),
            gyro_bias_std: Vector3::repeat(1e-5),
            accel_bias_std: Vector3::repeat(1e-4),
            unknown: BTreeMap::new(),
        }
    }

    /// basalt's shipped configuration, with the matching guess set to the same
    /// pixel so that the synthetic rig's cameras — which see identical frames —
    /// really do match.
    fn config() -> VioConfig {
        VioConfig {
            optical_flow_matching_guess_type: MatchingGuessType::SamePixel,
            ..VioConfig::default()
        }
    }

    /// Bright 5x5 squares on a regular lattice: four strong FAST corners each,
    /// and enough texture in between for the KLT to follow them.
    fn dotted_image(shift: i32) -> ImageU16 {
        let mut image: ImageU16 = ImageU16::zeros(WIDTH, HEIGHT).unwrap();
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let fx: f64 = f64::from(x as i32 - shift);
                let fy: f64 = f64::from(y as i32);
                let base: f64 = 18_000.0 + 5_000.0 * (fx * 0.07).sin() * (fy * 0.05).cos();
                image.set(x, y, base as u16);
            }
        }
        let mut cy: usize = 14;
        while cy + 5 < HEIGHT {
            let mut cx: usize = 14;
            while cx + 5 < WIDTH {
                for dy in 0..5 {
                    for dx in 0..5 {
                        let x: i32 = (cx + dx) as i32 + shift;
                        if x >= 0 && (x as usize) < WIDTH {
                            image.set(x as usize, cy + dy, 0xF000);
                        }
                    }
                }
                cx += 17;
            }
            cy += 17;
        }
        image
    }

    fn frontend(cameras: usize, options: FrontendOptions) -> FrameToFrameOpticalFlow<Pattern51> {
        FrameToFrameOpticalFlow::new(config(), &rig(cameras), options).unwrap()
    }

    #[test]
    fn the_first_frame_detects_on_camera_zero_and_matches_into_camera_one() {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        let frame: &FlowFrame = flow
            .process_frame(1_000, &images, &PosePrediction::default(), &[])
            .unwrap();

        assert_eq!(frame.t_ns, 1_000);
        assert_eq!(frame.cameras.len(), 2);
        assert!(!frame.cameras[0].is_empty(), "camera 0 detected nothing");
        assert!(
            !frame.cameras[1].is_empty(),
            "nothing matched into camera 1 out of {} on camera 0",
            frame.cameras[0].len()
        );
        // Every keypoint of camera 1 that came from the match shares camera 0's id.
        let shared: usize = frame.cameras[1]
            .ids
            .iter()
            .filter(|id| frame.cameras[0].get(**id).is_some())
            .count();
        assert!(shared > 0, "camera 1 shares no id with camera 0");
    }

    /// `last_keypoint_id` is the global landmark id space (`optical_flow.h:71`,
    /// `:174`): ids are handed out in detection order and never reused.
    #[test]
    fn keypoint_ids_are_one_monotonic_space() {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        flow.process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap();
        let after_first: u64 = flow.last_keypoint_id();
        assert!(after_first > 0);
        for camera in &flow.frame().cameras {
            assert!(camera.ids.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(camera.ids.iter().all(|id| id.0 < after_first));
        }

        let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
        flow.process_frame(1, &moved, &PosePrediction::default(), &[])
            .unwrap();
        assert!(flow.last_keypoint_id() >= after_first);
        assert_eq!(flow.frame_counter(), 2);
    }

    /// `updateCellCounts` / `addKeypoint` / `removeKeypoint` (`:707-749`) keep
    /// `cells` equal to the number of keypoints in each grid cell — except where
    /// `addKeypoints` deliberately double-counts, which cannot happen on camera 0.
    #[test]
    fn camera_zero_cell_counts_match_its_keypoints() {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        flow.process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap();

        let grid: CellGrid = flow.occupancy_grid();
        let mut expected: Vec<i32> = vec![0; grid.rows * grid.columns];
        for index in 0..flow.frame().cameras[0].len() {
            let translation = flow.frame().cameras[0].transforms.translation(index);
            let (row, column) = grid.cell_of(translation.x, translation.y);
            expected[row * grid.columns + column] += 1;
        }
        assert_eq!(flow.cell_counts(0), &expected[..]);
        // One point per cell is the shipped budget, so no cell may exceed it.
        assert!(flow.cell_counts(0).iter().all(|count| *count <= 1));
    }

    #[test]
    fn tracking_carries_keypoints_across_a_shifted_frame() {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        let first: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        flow.process_frame(0, &first, &PosePrediction::default(), &[])
            .unwrap();
        let before: Vec<KeypointId> = flow.frame().cameras[0].ids.clone();

        let second: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
        flow.process_frame(1, &second, &PosePrediction::default(), &[])
            .unwrap();
        let survived: usize = flow.frame().cameras[0]
            .ids
            .iter()
            .filter(|id| before.contains(id))
            .count();
        assert!(
            survived * 2 >= before.len(),
            "only {survived} of {} keypoints survived a one-pixel shift",
            before.len()
        );
        // A surviving keypoint moved by about one pixel, and no further.
        for id in flow.frame().cameras[0].ids.clone() {
            if !before.contains(&id) {
                continue;
            }
            let moved: AffineCompact2f = flow.frame().cameras[0].get(id).unwrap();
            assert!(moved.translation.x.is_finite());
        }
    }

    #[test]
    fn one_thread_and_four_threads_produce_the_same_frame() {
        let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];

        let mut single: FrameToFrameOpticalFlow<Pattern51> = frontend(
            2,
            FrontendOptions {
                threads: 1,
                ..FrontendOptions::default()
            },
        );
        let mut wide: FrameToFrameOpticalFlow<Pattern51> = frontend(
            2,
            FrontendOptions {
                threads: 4,
                ..FrontendOptions::default()
            },
        );
        for (t_ns, frameset) in [(0i64, &images), (1, &moved)] {
            single
                .process_frame(t_ns, frameset, &PosePrediction::default(), &[])
                .unwrap();
            wide.process_frame(t_ns, frameset, &PosePrediction::default(), &[])
                .unwrap();
            assert_eq!(single.frame(), wide.frame());
        }
        assert_eq!(single.last_keypoint_id(), wide.last_keypoint_id());
    }

    #[test]
    fn two_runs_of_the_same_input_produce_the_same_frame() {
        let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
        let mut frames: Vec<FlowFrame> = Vec::new();
        for _ in 0..2 {
            let mut flow: FrameToFrameOpticalFlow<Pattern51> =
                frontend(2, FrontendOptions::default());
            flow.process_frame(0, &images, &PosePrediction::default(), &[])
                .unwrap();
            flow.process_frame(1, &moved, &PosePrediction::default(), &[])
                .unwrap();
            frames.push(flow.frame().clone());
        }
        assert_eq!(frames[0], frames[1]);
    }

    /// Trap 17: `getNumCams() >= 2` is a hard precondition in the C++
    /// (`optical_flow.h:210`). The port allows one camera and skips the passes
    /// that need a second.
    #[test]
    fn a_single_camera_rig_detects_and_skips_matching() {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(1, FrontendOptions::default());
        let images: [ImageU16; 1] = [dotted_image(0)];
        let frame: &FlowFrame = flow
            .process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(frame.cameras.len(), 1);
        assert!(!frame.cameras[0].is_empty());
    }

    /// Deviation X03: `E[i]` from `T_c0_ci` per camera, or the C++'s cam0-cam1
    /// matrix everywhere. On a three-camera baseline rig the two differ for
    /// camera 2 and agree for camera 1.
    #[test]
    fn the_essential_matrix_is_per_camera_unless_the_cpp_bug_is_asked_for() {
        let fixed: FrameToFrameOpticalFlow<Pattern51> = frontend(
            3,
            FrontendOptions {
                epipolar_per_camera: true,
                ..FrontendOptions::default()
            },
        );
        let faithful: FrameToFrameOpticalFlow<Pattern51> = frontend(
            3,
            FrontendOptions {
                epipolar_per_camera: false,
                ..FrontendOptions::default()
            },
        );
        assert_eq!(fixed.essential(1), faithful.essential(1));
        assert_eq!(faithful.essential(2), faithful.essential(1));
        // Camera 2's baseline is twice camera 1's, but `computeEssential`
        // normalizes the translation, so a pure-translation rig gives the same
        // matrix either way; the rotated rig below is what separates them.
        assert_eq!(fixed.essential(2), fixed.essential(1));

        // With camera 2 rotated, the two disagree.
        let mut rotated: Calibration<f64> = rig(3);
        rotated.t_i_c[2] = Se3::new(
            So3::exp(&Vector3::new(0.0, 0.4, 0.0)),
            Vector3::new(0.10, 0.0, 0.0),
        );
        let fixed: FrameToFrameOpticalFlow<Pattern51> = FrameToFrameOpticalFlow::new(
            config(),
            &rotated,
            FrontendOptions {
                epipolar_per_camera: true,
                ..FrontendOptions::default()
            },
        )
        .unwrap();
        let faithful: FrameToFrameOpticalFlow<Pattern51> = FrameToFrameOpticalFlow::new(
            config(),
            &rotated,
            FrontendOptions {
                epipolar_per_camera: false,
                ..FrontendOptions::default()
            },
        )
        .unwrap();
        assert_ne!(fixed.essential(2), faithful.essential(2));
        assert_eq!(faithful.essential(2), faithful.essential(1));
    }

    /// For a two-camera rig the flag makes no difference at all, which is what
    /// keeps a C++-parity run on msd-index unaffected either way.
    #[test]
    fn on_a_stereo_rig_the_epipolar_flag_changes_nothing() {
        let fixed: FrameToFrameOpticalFlow<Pattern51> = frontend(
            2,
            FrontendOptions {
                epipolar_per_camera: true,
                ..FrontendOptions::default()
            },
        );
        let faithful: FrameToFrameOpticalFlow<Pattern51> = frontend(
            2,
            FrontendOptions {
                epipolar_per_camera: false,
                ..FrontendOptions::default()
            },
        );
        // Index 0 differs by construction and is never read; camera 1 is the
        // one `filterPoints` uses, and there the two agree exactly.
        assert_eq!(fixed.essential(1), faithful.essential(1));
        assert_eq!(fixed.essential(0), Matrix4::zeros());
    }

    #[test]
    fn a_config_that_names_another_pattern_is_refused() {
        let error = FrameToFrameOpticalFlow::<Pattern52>::new(
            config(),
            &rig(2),
            FrontendOptions::default(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            FrontendError::PatternMismatch {
                config: 51,
                built: 52
            }
        );
    }

    #[test]
    fn a_config_that_names_another_flow_type_is_refused() {
        let mut config: VioConfig = config();
        config.optical_flow_type = "patch".to_owned();
        let error =
            FrameToFrameOpticalFlow::<Pattern51>::new(config, &rig(2), FrontendOptions::default())
                .unwrap_err();
        assert_eq!(
            error,
            FrontendError::UnsupportedFlowType("patch".to_owned())
        );
    }

    #[test]
    fn a_frameset_of_the_wrong_width_is_refused() {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        let images: [ImageU16; 1] = [dotted_image(0)];
        let error = flow
            .process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap_err();
        assert_eq!(
            error,
            FrontendError::CameraCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn the_cpp_essential_bug_needs_a_second_camera() {
        let error = FrameToFrameOpticalFlow::<Pattern51>::new(
            config(),
            &rig(1),
            FrontendOptions {
                epipolar_per_camera: false,
                ..FrontendOptions::default()
            },
        )
        .unwrap_err();
        assert_eq!(error, FrontendError::NeedsTwoCameras { cameras: 1 });
    }

    /// `keypoints.cpp:179`: a mask covering the frame leaves nothing to detect.
    #[test]
    fn a_mask_over_the_whole_frame_suppresses_detection() {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        let masks: Vec<Masks> = vec![
            Masks {
                masks: vec![Rect {
                    x: 0.0,
                    y: 0.0,
                    w: WIDTH as f32,
                    h: HEIGHT as f32,
                }],
            };
            2
        ];
        let frame: &FlowFrame = flow
            .process_frame(0, &images, &PosePrediction::default(), &masks)
            .unwrap();
        assert!(frame.cameras[0].is_empty());
        assert!(frame.cameras[1].is_empty());
    }
    /// A frame the frontend refuses must leave it usable.
    ///
    /// The review's sequence: a 1x1 first image is rejected by the pyramid, and
    /// the *next*, valid frame used to take the tracking path against an empty
    /// previous pyramid and panic. Nothing commits until every pyramid is built.
    #[test]
    fn a_rejected_frame_leaves_the_frontend_usable() {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        let tiny: [ImageU16; 2] = [
            ImageU16::zeros(1, 1).unwrap(),
            ImageU16::zeros(1, 1).unwrap(),
        ];
        let error = flow
            .process_frame(1, &tiny, &PosePrediction::default(), &[])
            .unwrap_err();
        assert!(
            matches!(error, FrontendError::Pyramid(_)),
            "expected a pyramid error, got {error}"
        );
        // Nothing moved: the clock is still before the first frame.
        assert_eq!(flow.t_ns(), -1);
        assert_eq!(flow.frame_counter(), 0);
        assert_eq!(flow.last_keypoint_id(), 0);

        // The next valid frame is treated as the first, and the one after it
        // tracks against a pyramid that exists.
        let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        let frame: &FlowFrame = flow
            .process_frame(2, &images, &PosePrediction::default(), &[])
            .unwrap();
        assert!(!frame.cameras[0].is_empty());
        let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
        flow.process_frame(3, &moved, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(flow.frame_counter(), 2);
    }

    /// A frameset of the wrong width is refused before anything commits too.
    #[test]
    fn a_frameset_of_the_wrong_width_commits_nothing() {
        let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        let images: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        flow.process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap();
        let before: FlowFrame = flow.frame().clone();

        let short: [ImageU16; 1] = [dotted_image(1)];
        assert!(
            flow.process_frame(1, &short, &PosePrediction::default(), &[])
                .is_err()
        );
        assert_eq!(flow.t_ns(), 0);
        assert_eq!(flow.frame_counter(), 1);
        assert_eq!(flow.frame(), &before);

        // And the frontend still works afterwards.
        let moved: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
        flow.process_frame(2, &moved, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(flow.frame_counter(), 2);
    }

    /// The keypoint budget is enforced where the keypoints are created, so a
    /// frame is never produced that the tracker cannot then carry.
    #[test]
    fn the_keypoint_budget_is_never_exceeded() {
        for max_keypoints in [1usize, 3, 7] {
            let mut flow: FrameToFrameOpticalFlow<Pattern51> = frontend(
                2,
                FrontendOptions {
                    max_keypoints,
                    ..FrontendOptions::default()
                },
            );
            for step in 0..3 {
                let images: [ImageU16; 2] = [dotted_image(step), dotted_image(step)];
                let frame: &FlowFrame = flow
                    .process_frame(step.into(), &images, &PosePrediction::default(), &[])
                    .unwrap();
                for camera in &frame.cameras {
                    assert!(
                        camera.len() <= max_keypoints,
                        "{} keypoints against a budget of {max_keypoints}",
                        camera.len()
                    );
                }
            }
            assert!(!flow.frame().cameras[0].is_empty());
        }
    }

    /// The budget may not exceed what the tracker was built for.
    #[test]
    fn a_budget_larger_than_the_tracker_is_refused() {
        let config: VioConfig = config();
        let tracker: CpuPatchTracker<Pattern51> = CpuPatchTracker::new(
            4,
            config.optical_flow_levels as usize + 1,
            config.optical_flow_max_iterations as usize,
            config.optical_flow_max_recovered_dist2,
            WorkPool::new(1).unwrap(),
        );
        let error = FrameToFrameOpticalFlow::<Pattern51, _, _>::with_backends(
            config,
            &rig(2),
            FrontendOptions {
                max_keypoints: 8,
                ..FrontendOptions::default()
            },
            CpuPyramidBuilder::new(),
            tracker,
        )
        .unwrap_err();
        assert_eq!(
            error,
            FrontendError::BudgetExceedsCapacity {
                max_keypoints: 8,
                capacity: 4
            }
        );
    }

    /// A calibration with fewer poses than camera models used to index past the
    /// end of `T_i_c`; it is a typed error now (decision D32).
    #[test]
    fn a_calibration_missing_an_extrinsic_is_refused() {
        let mut ragged: Calibration<f64> = rig(2);
        ragged.t_i_c.pop();
        let error = FrameToFrameOpticalFlow::<Pattern51>::new(
            config(),
            &ragged,
            FrontendOptions::default(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            FrontendError::RaggedExtrinsics {
                intrinsics: 2,
                extrinsics: 1
            }
        );
    }

    /// A camera too small for one detection cell names itself in the error.
    #[test]
    fn a_camera_smaller_than_a_cell_is_refused() {
        let mut small: Calibration<f64> = rig(2);
        small.resolution[1] = [30, 30];
        let error =
            FrameToFrameOpticalFlow::<Pattern51>::new(config(), &small, FrontendOptions::default())
                .unwrap_err();
        assert_eq!(
            error,
            FrontendError::FrameTooSmall {
                camera: 1,
                width: 30,
                height: 30,
                cell: 50
            }
        );
    }

    /// Every camera is detected on **its own** grid (`keypoints.cpp:140-144`),
    /// even though the occupancy matrix keeps camera 0's shape (`:119`).
    ///
    /// The review's probe: with 50-pixel cells a 200x200 camera starts at 0 and a
    /// 240x240 camera at 20, and those two grids disagree about a corner near
    /// (210, 80) — camera 0's grid stops at 150 + 50 = 200, so the port would
    /// have missed it entirely if it had imposed camera 0's geometry.
    #[test]
    fn each_camera_is_detected_on_its_own_grid() {
        let mut mixed: Calibration<f64> = rig(2);
        mixed.resolution[1] = [240, 240];
        let flow: FrameToFrameOpticalFlow<Pattern51> =
            FrameToFrameOpticalFlow::new(config(), &mixed, FrontendOptions::default()).unwrap();

        assert_eq!(flow.detection_grid(0).x_start, 0);
        assert_eq!(flow.detection_grid(1).x_start, 20);
        // The occupancy matrix follows camera 0, which is what `cells` is shaped
        // from; both grids happen to need the same number of columns here.
        assert_eq!(flow.occupancy_grid(), flow.detection_grid(0));
        assert_eq!(flow.detection_grid(1).rows, flow.occupancy_grid().rows);

        assert!(flow.detection_grid(1).contains(210.0, 80.0));
        assert!(!flow.detection_grid(0).contains(210.0, 80.0));
    }

    /// A mixed-resolution rig still runs end to end, and camera 1's keypoints
    /// come from its own image.
    #[test]
    fn a_mixed_resolution_rig_runs() {
        let mut mixed: Calibration<f64> = rig(2);
        mixed.resolution[1] = [240, 240];
        let mut flow: FrameToFrameOpticalFlow<Pattern51> =
            FrameToFrameOpticalFlow::new(config(), &mixed, FrontendOptions::default()).unwrap();

        let wide: ImageU16 = {
            let mut image: ImageU16 = ImageU16::zeros(240, 240).unwrap();
            let source: ImageU16 = dotted_image(0);
            for y in 0..240 {
                for x in 0..240 {
                    let value: u16 = source.get(x % WIDTH, y % HEIGHT).unwrap_or(0);
                    image.set(x, y, value);
                }
            }
            image
        };
        let images: [ImageU16; 2] = [dotted_image(0), wide];
        let frame: &FlowFrame = flow
            .process_frame(0, &images, &PosePrediction::default(), &[])
            .unwrap();
        assert!(!frame.cameras[0].is_empty());
        for index in 0..frame.cameras[1].len() {
            let position = frame.cameras[1].transforms.translation(index);
            assert!(position.x >= 0.0 && position.x < 240.0);
            assert!(position.y >= 0.0 && position.y < 240.0);
        }
    }
    /// A tracker that forwards to the CPU one and refuses the *n*-th call.
    ///
    /// The point is a failure from inside a pluggable backend, in the middle of
    /// `processFrame`, after the frame's pyramids are built and after some of the
    /// cameras have already been tracked. Nothing else can produce that.
    #[derive(Debug)]
    struct FailingTracker {
        inner: CpuPatchTracker<Pattern51>,
        calls: std::cell::Cell<usize>,
        fail_on: usize,
    }

    impl PatchTracker for FailingTracker {
        type Pattern = Pattern51;
        type Pyramid = PyramidU16;
        type Patches = PatchSoA<Pattern51>;

        fn capacity(&self) -> usize {
            self.inner.capacity()
        }

        fn num_levels(&self) -> usize {
            self.inner.num_levels()
        }

        fn make_patches(&self) -> PatchSoA<Pattern51> {
            self.inner.make_patches()
        }

        fn track(
            &mut self,
            prev: &PyramidU16,
            next: &PyramidU16,
            patches: &PatchSoA<Pattern51>,
            transforms_in: &FlowTransforms,
            out: &mut FlowResult,
        ) -> Result<(), TrackerError> {
            self.calls.set(self.calls.get() + 1);
            if self.calls.get() == self.fail_on {
                return Err(TrackerError::CapacityExceeded {
                    offered: usize::MAX,
                    capacity: 0,
                });
            }
            self.inner.track(prev, next, patches, transforms_in, out)
        }
    }

    fn failing_frontend(
        fail_on: usize,
    ) -> FrameToFrameOpticalFlow<Pattern51, CpuPyramidBuilder, FailingTracker> {
        let config: VioConfig = config();
        let options: FrontendOptions = FrontendOptions::default();
        let inner: CpuPatchTracker<Pattern51> = CpuPatchTracker::new(
            options.max_keypoints,
            config.optical_flow_levels as usize + 1,
            config.optical_flow_max_iterations as usize,
            config.optical_flow_max_recovered_dist2,
            WorkPool::new(options.threads).unwrap(),
        );
        FrameToFrameOpticalFlow::with_backends(
            config,
            &rig(2),
            options,
            CpuPyramidBuilder::new(),
            FailingTracker {
                inner,
                calls: std::cell::Cell::new(0),
                fail_on,
            },
        )
        .unwrap()
    }

    /// A backend error part-way through frame 2 must leave the frontend exactly
    /// as frame 1 left it, so frame 3 comes out of `{1, 2 fails, 3}` identical to
    /// frame 3 of a clean `{1, 3}`.
    ///
    /// The rotation used to commit before tracking, which paired frame 1's
    /// keypoints with frame 2's pyramid on the next call.
    #[test]
    fn a_backend_error_leaves_the_frame_as_the_last_good_one() {
        let first: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        let second: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
        let third: [ImageU16; 2] = [dotted_image(2), dotted_image(2)];

        // The clean run skips the frame the other run fails on.
        let mut clean: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        clean
            .process_frame(0, &first, &PosePrediction::default(), &[])
            .unwrap();
        clean
            .process_frame(2, &third, &PosePrediction::default(), &[])
            .unwrap();

        // Frame 1 makes one call (the stereo match), so call 2 is camera 0 of
        // frame 2: the failure lands after the first frame committed and before
        // the second one could.
        let mut faulty = failing_frontend(2);
        faulty
            .process_frame(0, &first, &PosePrediction::default(), &[])
            .unwrap();
        let after_first: FlowFrame = faulty.frame().clone();
        let ids_after_first: u64 = faulty.last_keypoint_id();
        let cells_after_first: Vec<i32> = faulty.cell_counts(0).to_vec();

        let error = faulty
            .process_frame(1, &second, &PosePrediction::default(), &[])
            .unwrap_err();
        assert!(matches!(error, FrontendError::Tracker(_)), "got {error}");

        // Nothing moved.
        assert_eq!(faulty.frame(), &after_first);
        assert_eq!(faulty.last_keypoint_id(), ids_after_first);
        assert_eq!(faulty.cell_counts(0), &cells_after_first[..]);
        assert_eq!(faulty.t_ns(), 0);
        assert_eq!(faulty.frame_counter(), 1);

        // And the next good frame is the one the clean run produced.
        faulty
            .process_frame(2, &third, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(faulty.frame(), clean.frame());
        assert_eq!(faulty.last_keypoint_id(), clean.last_keypoint_id());
        assert_eq!(faulty.cell_counts(0), clean.cell_counts(0));
        assert_eq!(faulty.cell_counts(1), clean.cell_counts(1));
    }

    /// The same, with the failure on camera 1 — after camera 0 has already been
    /// tracked and its keypoint map rewritten inside the same call.
    #[test]
    fn a_backend_error_after_the_first_camera_is_undone_too() {
        let first: [ImageU16; 2] = [dotted_image(0), dotted_image(0)];
        let second: [ImageU16; 2] = [dotted_image(1), dotted_image(1)];
        let third: [ImageU16; 2] = [dotted_image(2), dotted_image(2)];

        let mut clean: FrameToFrameOpticalFlow<Pattern51> = frontend(2, FrontendOptions::default());
        clean
            .process_frame(0, &first, &PosePrediction::default(), &[])
            .unwrap();
        clean
            .process_frame(2, &third, &PosePrediction::default(), &[])
            .unwrap();

        // Frame 1 makes one matching call, so frame 2's camera-1 track is call 3.
        let mut faulty = failing_frontend(3);
        faulty
            .process_frame(0, &first, &PosePrediction::default(), &[])
            .unwrap();
        assert!(
            faulty
                .process_frame(1, &second, &PosePrediction::default(), &[])
                .is_err()
        );
        faulty
            .process_frame(2, &third, &PosePrediction::default(), &[])
            .unwrap();
        assert_eq!(faulty.frame(), clean.frame());
        assert_eq!(faulty.last_keypoint_id(), clean.last_keypoint_id());
    }
}
