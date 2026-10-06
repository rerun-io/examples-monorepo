//! Frame-to-frame optical flow.
//! Each frameset builds camera pyramids, tracks existing points, updates occupancy,
//! detects and matches new points, then applies the epipolar filter.
//!
//! The driver is generic over pyramid and tracker backends. Callers supply pose
//! prediction and depth feedback; this module has no IMU queue or processing
//! thread (D17). Recall and multiscale-flow variants are unsupported.
//!
//! Essential matrices and image bounds are per camera. Detection uses each
//! camera's geometry while occupancy retains camera 0's shape; out-of-range
//! cells are skipped. Single-camera rigs skip stereo matching and filtering.
//! Forward and backward passes are batched, with masks applied afterwards to
//! the same positions. A keypoint budget bounds preallocated storage.

mod data;
mod detection;
mod error;
mod tracking;

use super::stages::{CpuStages, FrameExecutor, FrameStages, StereoContext};

pub use data::{FlowFrame, FlowTimings, FrontendOptions, Keypoints, NO_RESPONSE, PosePrediction};
pub use error::FrontendError;
pub use tracking::project_between_cams;
use tracking::{cast_matrix4, compute_essential};

use std::marker::PhantomData;

use nalgebra::Matrix4;

use crate::calib::Calibration;
use crate::camera::RigCamera;
use crate::config::VioConfig;
use crate::duration_ns;
use crate::frontend::parallel::{MAX_THREADS, WorkPool};
use crate::frontend::patterns::Pattern;
use crate::frontend::tracker::{
    CpuPatchTracker, FlowTransforms, MAX_CAPACITY, MAX_LEVELS, PatchTracker, TrackInput,
};
use crate::lie::Se3;
use crate::pyramid::CpuPyramidBuilder;
use crate::types::KeypointId;
use kornia_image::Image;
use kornia_staging_imgproc::features::{
    CellGrid, CellSelect, CpuCornerScan, DetectorConfig, DetectorScratch, KeypointsData,
    LOWEST_THRESHOLD_RUNG, Masks, cell_select,
};

/// Frame-to-frame optical flow in f32, generic over pyramid and tracker backends.
/// [`FrameToFrameOpticalFlow::with_stages`] selects implementations without
/// exposing a concrete pyramid in the driver's public signatures.
#[derive(Debug)]
pub struct FrameToFrameOpticalFlow<
    P: Pattern,
    F: FrameStages<Tracker: PatchTracker<Pattern = P>> = CpuStages<CpuPatchTracker<P>>,
> {
    config: VioConfig,
    options: FrontendOptions,
    calib: Calibration<f32>,
    cameras: Vec<RigCamera<f32>>,
    /// `E`, one 4x4 essential matrix per camera.
    essential: Vec<Matrix4<f32>>,
    /// One detection grid per camera, derived from that camera's image size.
    detection_grids: Vec<CellGrid>,
    /// The occupancy grid, shaped from camera 0 as `cells` is.
    occupancy_grid: CellGrid,
    /// `cells`, `(h/c + 1) x (w/c + 1)` counts per camera.
    cells: Vec<Vec<i32>>,
    /// `last_keypoint_id`, the global landmark id space.
    last_keypoint_id: u64,
    /// Camera 0's keypoint count as the last **detecting** frameset left it,
    /// which is what `port.redetect_survivor_ratio` measures survivors against
    /// (D75). Zero until one frameset has detected.
    last_detect_count: usize,
    /// `last_keypoint_id` as it stood before the last **committed** frameset,
    /// which is what makes "how many of these keypoints are new" answerable
    /// after the fact rather than only inside the call that produced them.
    last_keypoint_id_before_frame: u64,
    /// Timestamp of the last committed frame; `None` means no previous frame.
    /// Negative timestamps remain valid and do not reset tracking.
    t_ns: Option<i64>,
    /// `frame_counter`.
    frame_counter: u64,
    /// `depth_guess`, seeded from the config and refreshed by the estimator.
    depth_guess: f32,
    /// Masks for this frame, grown by `cam0OverlapCellsMasksForCam`.
    masks: Vec<Masks>,

    stages: F,
    /// The source keypoint ids of each lane of the batch in flight, in map order
    ///
    /// Per lane rather than per call because a batch's passes are all launched
    /// before any of them is read, and mapping a tracked slot back to its
    /// keypoint needs this after the download.
    passes: Vec<TrackInput>,
    /// The ids that survived, in ascending source order, with their warps.
    tracked_ids: Vec<KeypointId>,
    /// The warps of [`FrameToFrameOpticalFlow::tracked_ids`].
    tracked: FlowTransforms,
    /// The device cell selection each camera can take, or `None` where its shape
    /// cannot ([`cell_select`]); rebuilt at the top of every frameset, before
    /// anything about it has been decided.
    cell_selects: Vec<Option<CellSelect>>,
    detected: Vec<KeypointsData>,
    /// Independent scanners for side-camera work, only when the selected
    /// backend supports it. Otherwise all cameras use `detector` serially.
    side_detectors: Option<Vec<DetectorScratch<F::Scanner>>>,
    /// Hosts all of `process_frame`, including pyramid builds and detection.
    /// Backends must not depend on worker identity. A one-worker pool keeps
    /// work on the caller. See
    /// [`FrameToFrameOpticalFlow::with_stages`].
    host_pool: WorkPool,
    /// The keypoints `addPointsForCamera(0)` produced, to be matched onward.
    new_cam0: Keypoints,
    /// Ids the epipolar filter removes, ascending (`std::set`).
    to_remove: Vec<KeypointId>,
    frame: FlowFrame,
    /// The keypoint state as it was before the frame in flight; see
    /// [`FrameToFrameOpticalFlow::process_frame`].
    snapshot: FrameState,
    /// What the last frame's phases cost; see [`FlowTimings`].
    timings: FlowTimings,
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
    last_detect_count: usize,
}

#[cfg(feature = "gpu-wgpu")]
impl<P: Pattern> FrameToFrameOpticalFlow<P, crate::gpu::GpuStages<P, crate::gpu::GpuRuntime>> {
    pub(crate) fn discard_lookahead(&mut self) {
        self.stages.discard_lookahead();
    }

    pub(crate) fn prepare_packed_inputs(&mut self, views: &[crate::ImageView<'_>]) {
        self.stages.prepare_packed_inputs(views);
    }

    pub(crate) fn queue_lookahead(
        &mut self,
        t_ns: i64,
        images: &mut Vec<Image<u16, 1>>,
        views: &[crate::ImageView<'_>],
    ) -> Result<(), crate::VioError> {
        self.update_cell_selects(images);
        self.stages
            .queue_lookahead(t_ns, images, views, &self.cell_selects)
    }
}

impl<P: Pattern> FrameToFrameOpticalFlow<P> {
    /// Workers shared by the CPU stages and the synchronous estimator.
    pub(crate) fn pool(&self) -> &WorkPool {
        &self.host_pool
    }

    /// `FrameToFrameOpticalFlow(conf, cal)`
    /// on the CPU backends.
    ///
    /// The calibration arrives in `f64` — that is what the JSON gives — and is
    /// cast to `f32` here, as `OpticalFlowTyped`'s constructor does
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
            pool.clone(),
        )?;
        Self::with_stages(
            config,
            calibration,
            options,
            CpuStages::new(
                CpuPyramidBuilder::new(),
                tracker,
                DetectorScratch::with_scanner(Box::new(CpuCornerScan::default())),
            )?,
            pool,
        )
    }
}

impl<P: Pattern, F: FrameStages<Tracker: PatchTracker<Pattern = P>>> FrameToFrameOpticalFlow<P, F> {
    /// The config checks that do not depend on the backends.
    fn validate_config(config: &VioConfig) -> Result<(), FrontendError> {
        if config
            .port_klt_exit_step_px
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(crate::frontend::tracker::TrackerError::InvalidExitStep.into());
        }
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
        // Refuse a non-positive minimum: integer halving would remain at zero forever.
        // The public detector also floors its last rung as a second line of protection.
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

    /// Validate caller-supplied frontend options independently of the selected backend.
    /// A prebuilt tracker passed to `with_stages` does not run `PatchSoA::new`, so
    /// the frontend must enforce its own bounds at this seam.
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
    /// Construct the flow with its frame-stage owner and host worker pool.
    pub fn with_stages(
        config: VioConfig,
        calibration: &Calibration<f64>,
        options: FrontendOptions,
        mut stages: F,
        host_pool: WorkPool,
    ) -> Result<Self, FrontendError> {
        Self::validate_config(&config)?;
        Self::validate_options(&options)?;

        let tracker = stages.tracker_mut();
        tracker.set_klt_exit_step_px(config.port_klt_exit_step_px);

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
        // `calib.T_i_c[i]` is indexed for every camera; a
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
        // Each essential matrix uses that camera's pose relative to camera 0.
        let essential: Vec<Matrix4<f32>> = (0..num_cams)
            .map(|index| {
                if index == 0 {
                    // `T_c0_c0` has no baseline, so `t.normalized()` would be
                    // NaN; nothing reads index 0 either way.
                    return Matrix4::zeros();
                }
                let t_i_j: Se3<f32> = calib.t_i_c[0].inverse() * calib.t_i_c[index];
                cast_matrix4(&compute_essential(&t_i_j.cast::<f64>()))
            })
            .collect();

        // `detectKeypointsWithCells` derives its grid from the image it is given
        // so every camera gets its own; `cells` is
        // shaped from camera 0 alone, so that shape is separate.
        let cell: usize = config.optical_flow_detection_grid_size as usize;
        let mut detection_grids: Vec<CellGrid> = Vec::with_capacity(num_cams);
        for (camera, rig_camera) in cameras.iter().enumerate() {
            let width: usize = rig_camera.width() as usize;
            let height: usize = rig_camera.height() as usize;
            let grid = CellGrid::new(width, height, cell).map_err(|error| match error {
                kornia_staging_imgproc::features::CellGridError::TooManyCells {
                    rows,
                    columns,
                    ceiling,
                } => FrontendError::TooManyCells {
                    camera,
                    rows,
                    columns,
                    ceiling,
                },
                _ => FrontendError::FrameTooSmall {
                    camera,
                    width,
                    height,
                    cell,
                },
            })?;
            detection_grids.push(grid);
        }
        let occupancy_grid: CellGrid = detection_grids[0];

        let side_detectors = stages.side_detectors(num_cams);
        Ok(Self {
            depth_guess: config.optical_flow_matching_default_depth,
            stages,
            cells: vec![vec![0; occupancy_grid.rows * occupancy_grid.columns]; num_cams],
            masks: vec![Masks::default(); num_cams],
            snapshot: FrameState::default(),
            timings: FlowTimings::default(),
            passes: (0..num_cams).map(|_| TrackInput::default()).collect(),
            tracked_ids: Vec::new(),
            tracked: FlowTransforms::default(),
            cell_selects: vec![None; num_cams],
            detected: (0..num_cams).map(|_| KeypointsData::default()).collect(),
            side_detectors,
            host_pool,
            new_cam0: Keypoints::default(),
            to_remove: Vec::new(),
            frame: FlowFrame {
                t_ns: None,
                cameras: vec![Keypoints::default(); num_cams],
            },
            last_keypoint_id: 0,
            last_keypoint_id_before_frame: 0,
            last_detect_count: 0,
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

    /// The next keypoint id that will be handed out.
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

    /// Framesets processed so far.
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

    /// The grid `cells` is shaped and indexed by, from camera 0.
    pub fn occupancy_grid(&self) -> CellGrid {
        self.occupancy_grid
    }

    /// The grid one camera is *detected* on, from its own image size
    ///
    /// # Panics
    ///
    /// If `camera` is past the end of the rig.
    pub fn detection_grid(&self, camera: usize) -> CellGrid {
        self.detection_grids[camera]
    }

    /// This camera's essential matrix relative to camera 0.
    /// Entry zero is the zero matrix and is not read by epipolar filtering.
    ///
    /// # Panics
    /// If `camera` is outside the rig.
    pub fn essential(&self, camera: usize) -> Matrix4<f32> {
        self.essential[camera]
    }

    /// The configuration this frontend runs.
    pub fn config(&self) -> &VioConfig {
        &self.config
    }

    /// Timestamp of the most recent frameset, `None` before the first
    pub fn t_ns(&self) -> Option<i64> {
        self.t_ns
    }

    /// `depth_guess`: the estimator's average scene depth.
    pub fn depth_guess(&self) -> f32 {
        self.depth_guess
    }

    /// `depth_guess`: the estimator's average scene depth.
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
        // Tracking requires a frameset strictly after the last committed timestamp.
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
        // Image size must match calibration: a different pixel geometry changes bearings,
        // so reject it before tracking with incompatible intrinsics and cell grids.
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

    /// Process one frameset of widened 16-bit images.
    /// `prediction` supplies previous and predicted poses; identities mean no state yet.
    /// Missing mask lists suppress nothing.
    ///
    /// A rejected frame leaves the frontend at its last successful state. Build into
    /// staging pyramids and snapshot keypoints; swap pyramids and advance the clock
    /// only after every pass succeeds. On failure restore keypoints, counts and ids.
    ///
    /// # Errors
    /// Returns [`FrontendError`] for invalid timestamp, camera count, image geometry,
    /// pyramid geometry, or tracker/detector input.
    pub fn process_frame(
        &mut self,
        t_ns: i64,
        images: &[Image<u16, 1>],
        prediction: &PosePrediction,
        masks: &[Masks],
    ) -> Result<&FlowFrame, FrontendError>
    where
        F: Send,
    {
        if let Some(executor) = self.stages.executor() {
            executor.run(|| self.process_frame_inner(t_ns, images, prediction, masks))?;
            return Ok(&self.frame);
        }
        let pool = self.host_pool.clone();
        match pool.install(|| self.process_frame_inner(t_ns, images, prediction, masks)) {
            Some(result) => result?,
            None => self.process_frame_inner(t_ns, images, prediction, masks)?,
        }
        Ok(&self.frame)
    }

    fn process_frame_inner(
        &mut self,
        t_ns: i64,
        images: &[Image<u16, 1>],
        prediction: &PosePrediction,
        masks: &[Masks],
    ) -> Result<(), FrontendError> {
        self.check_frameset(
            t_ns,
            images.iter().map(|image| (image.width(), image.height())),
        )?;

        self.timings = FlowTimings::default();
        self.stages.prepare(
            t_ns,
            images,
            self.config.optical_flow_levels as usize,
            &self.host_pool,
            &mut self.timings,
        )?;

        // What the passes below mutate, kept so they can be undone.
        let mut snapshot: FrameState = std::mem::take(&mut self.snapshot);
        snapshot.cameras.clone_from(&self.frame.cameras);
        snapshot.cells.clone_from(&self.cells);
        snapshot.last_keypoint_id = self.last_keypoint_id;
        snapshot.last_detect_count = self.last_detect_count;

        let outcome: Result<(), FrontendError> = self.run_passes(images, prediction, masks);
        let outcome = outcome.and_then(|()| self.stages.finish());

        match &outcome {
            Ok(()) => {
                // Commit: the frame in flight becomes the committed one, and the
                // set it displaces is next frame's staging buffer.
                self.t_ns = Some(t_ns);
                self.frame.t_ns = Some(t_ns);
                self.frame_counter += 1;
                // The snapshot holds the id space as it stood before this frame,
                // which is exactly what a reader of the committed frame needs.
                self.last_keypoint_id_before_frame = snapshot.last_keypoint_id;
            }
            Err(_) => {
                // A frameset refused between the launches and the download
                // leaves a batch in flight; the next one starts empty.
                self.stages.discard();
                self.frame.cameras.clone_from(&snapshot.cameras);
                self.cells.clone_from(&snapshot.cells);
                self.last_keypoint_id = snapshot.last_keypoint_id;
                self.last_detect_count = snapshot.last_detect_count;
            }
        }
        self.snapshot = snapshot;
        outcome?;
        Ok(())
    }

    fn update_cell_selects(&mut self, images: &[Image<u16, 1>]) {
        let config: DetectorConfig = self.detector_config();
        let Self {
            cell_selects,
            detection_grids,
            ..
        } = self;
        for ((slot, image), grid) in cell_selects
            .iter_mut()
            .zip(images)
            .zip(detection_grids.iter())
        {
            *slot = cell_select(image, grid, &config);
        }
    }

    /// Steps 4 to 7 of `processFrame`, against `staging` as the current frame.
    ///
    /// Split out so [`FrameToFrameOpticalFlow::process_frame`] can undo them: no
    /// step here touches the pyramid sets, the timestamp or the frame counter.
    fn run_passes(
        &mut self,
        images: &[Image<u16, 1>],
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

        // Camera 0's cell winners, launched here and downloaded by the temporal
        // tracks below (D78). The selection kernels read the frame alone — the
        // occupancy counts and the masks stay on the host and are applied to
        // the downloaded keys — so there is nothing about this frameset they
        // need to know, including whether it detects at all. That last part is
        // the speculation: on a frameset that skips `add_points` these kernels
        // are wasted GPU time, and what they buy on one that detects is a whole
        // synchronising read, 0.12 ms of host time on this lane. Camera 0 alone
        // because it is the only camera the match needs before it launches; the
        // others go behind the matches, where another download is waiting.
        self.update_cell_selects(images);
        self.stages.prepare_detection(
            images,
            &self.cell_selects,
            StereoContext {
                cameras: &self.cameras,
                calib: &self.calib,
                config: &self.config,
                depth: self.depth_guess,
                last_detect_count: self.last_detect_count,
                eligible: self.t_ns.is_some()
                    && self.masks.iter().all(|masks| masks.masks.is_empty()),
            },
            &mut self.timings,
        )?;

        if self.t_ns.is_none() {
            for keypoints in &mut self.frame.cameras {
                keypoints.clear();
            }
            for counts in &mut self.cells {
                counts.fill(0);
            }
        } else {
            // `for (i) trackPoints(old_pyramid[i], pyramid[i], ..., T_c1_c2, i, i)`
            // as one batch: every camera's pass is launched, then
            // one download answers all of them. A camera's pass reads only its
            // own two pyramids and writes only its own lane, so batching moves
            // no arithmetic — it removes the per-camera wait, which on the GPU
            // lane is the frameset's dominant host cost (D77).
            let mark: std::time::Instant = std::time::Instant::now();
            self.prepare_tracks(Some(prediction));
            self.stages.temporal(&mut self.passes, &mut self.timings)?;
            self.timings.track_ns += duration_ns(mark);
            for camera in 0..num_cams {
                self.finish_camera(camera);
            }
            // `for (i) updateCellCounts(i)`.
            for camera in 0..num_cams {
                self.update_cell_counts(camera);
            }
        }

        // Camera 0's keys, out of the download the tracks just made. On the
        // first frameset of a run there was no track to carry them and this
        // reads them itself, which is the read the detector used to make on
        // every frameset.
        let mark: std::time::Instant = std::time::Instant::now();
        self.stages.detector().take_cells().map_err(Into::into)?;
        self.timings.detect_ns += duration_ns(mark);

        if self.should_detect() {
            self.add_points(images)?;
            self.last_detect_count = self.frame.cameras[0].len();
        }
        let mark: std::time::Instant = std::time::Instant::now();
        self.filter_points();
        self.timings.stereo_ns += duration_ns(mark);
        Ok(())
    }
}
