//! Construction and dispatch for the selected frontend backend.

#[cfg(doc)]
use crate::Vio;
#[cfg(feature = "gpu-wgpu")]
use crate::gpu;
use crate::{Backend, ImageView, VioError, calib, config, frontend, image};

/// The frontend of a [`Vio`], on whichever backend it was built for.
///
/// A two-arm enum rather than a type parameter on [`Vio`]: the estimator is
/// already generic over its scalar, the dispatch happens once per frameset, and
/// every method below returns a type neither backend owns — so the whole cost of
/// the choice is this forwarding.
// 2.8 kB against 3.2 kB, and a pipeline holds exactly one, so boxing a variant
// would buy an indirection on the per-frame path and nothing else.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum FrontendLane {
    /// The CPU pyramid builder and patch tracker.
    Cpu(
        frontend::flow::FrameToFrameOpticalFlow<
            kornia_staging_imgproc::optical_flow::patch_se2::Pattern51,
        >,
    ),
    /// The CubeCL pyramid builder and patch tracker.
    #[cfg(feature = "gpu-wgpu")]
    Gpu(
        frontend::flow::FrameToFrameOpticalFlow<
            kornia_staging_imgproc::optical_flow::patch_se2::Pattern51,
            gpu::GpuStages<
                kornia_staging_imgproc::optical_flow::patch_se2::Pattern51,
                gpu::GpuRuntime,
            >,
        >,
    ),
}

/// Run the same expression against whichever backend the lane holds.
macro_rules! on_lane {
    ($lane:expr, |$flow:ident| $body:expr) => {
        match $lane {
            FrontendLane::Cpu($flow) => $body,
            #[cfg(feature = "gpu-wgpu")]
            FrontendLane::Gpu($flow) => $body,
        }
    };
}

impl FrontendLane {
    /// Retain byte inputs for the GPU upload, outside the host image type.
    pub(super) fn prepare_packed_inputs(&mut self, _views: &[ImageView<'_>]) {
        #[cfg(feature = "gpu-wgpu")]
        if let Self::Gpu(flow) = self {
            flow.prepare_packed_inputs(_views);
        }
    }

    /// Queue image-only work on lanes that support lookahead.
    pub(super) fn queue_lookahead(
        &mut self,
        _t_ns: i64,
        _images: &mut Vec<kornia_image::Image<u16, 1>>,
        _views: &[ImageView<'_>],
    ) -> Result<(), VioError> {
        #[cfg(feature = "gpu-wgpu")]
        if let Self::Gpu(flow) = self {
            flow.queue_lookahead(_t_ns, _images, _views)?;
        }
        Ok(())
    }

    /// Cancel a previous hint before an unhinted frame.
    pub(super) fn discard_lookahead(&mut self) {
        #[cfg(feature = "gpu-wgpu")]
        if let Self::Gpu(flow) = self {
            flow.discard_lookahead();
        }
    }

    /// Densify and widen into each lane's reusable host image.
    pub(super) fn fill_frame(
        &self,
        frame: &mut kornia_image::Image<u16, 1>,
        view: &ImageView<'_>,
    ) -> Result<(), image::ImageError> {
        image::fill_from_u8_strided(frame, view.data, view.width, view.height, view.stride)
    }

    /// Share the CPU frontend's workers with the synchronous estimator.
    pub(super) fn cpu_pool(&self) -> Option<frontend::parallel::WorkPool> {
        match self {
            Self::Cpu(flow) => Some(flow.pool().clone()),
            #[cfg(feature = "gpu-wgpu")]
            Self::Gpu(_) => None,
        }
    }

    // ── the seven [`Vio`] drives ──────────────────────────────────────────

    /// Which backend this lane runs.
    pub fn backend(&self) -> Backend {
        match self {
            Self::Cpu(_) => Backend::Cpu,
            #[cfg(feature = "gpu-wgpu")]
            Self::Gpu(_) => Backend::Gpu,
        }
    }

    /// `FrameToFrameOpticalFlow::check_frameset`.
    ///
    /// # Errors
    ///
    /// What the frontend refuses: a frameset of the wrong shape or a timestamp
    /// that does not follow the last one.
    pub fn check_frameset(
        &self,
        t_ns: i64,
        sizes: impl ExactSizeIterator<Item = (usize, usize)>,
    ) -> Result<(), frontend::flow::FrontendError> {
        on_lane!(self, |flow| flow.check_frameset(t_ns, sizes))
    }

    /// `FrameToFrameOpticalFlow::process_frame`.
    ///
    /// # Errors
    ///
    /// What the frontend refuses, plus a device failure on the GPU lane.
    pub fn process_frame(
        &mut self,
        t_ns: i64,
        images: &[kornia_image::Image<u16, 1>],
        prediction: &frontend::flow::PosePrediction,
        masks: &[kornia_staging_imgproc::features::Masks],
    ) -> Result<&frontend::flow::FlowFrame, frontend::flow::FrontendError> {
        on_lane!(self, |flow| flow
            .process_frame(t_ns, images, prediction, masks))
    }

    /// What the last frame's phases cost.
    pub fn timings(&self) -> frontend::flow::FlowTimings {
        on_lane!(self, |flow| flow.timings())
    }

    /// The last committed frame's tracked keypoints.
    pub fn frame(&self) -> &frontend::flow::FlowFrame {
        on_lane!(self, |flow| flow.frame())
    }

    /// The config the frontend was built from.
    pub fn config(&self) -> &config::VioConfig {
        on_lane!(self, |flow| flow.config())
    }

    /// Publish a new average scene depth.
    pub fn set_depth_guess(&mut self, depth: f32) {
        on_lane!(self, |flow| flow.set_depth_guess(depth));
    }

    // ── the rest, which only `slam-rs-py`'s standalone `OpticalFlow` reads ─

    /// Cameras in the rig.
    pub fn camera_count(&self) -> usize {
        on_lane!(self, |flow| flow.camera_count())
    }

    /// Framesets committed so far.
    pub fn frame_counter(&self) -> u64 {
        on_lane!(self, |flow| flow.frame_counter())
    }

    /// The high-water mark of the keypoint id space.
    pub fn last_keypoint_id(&self) -> u64 {
        on_lane!(self, |flow| flow.last_keypoint_id())
    }

    /// The same mark as it stood before the last committed frameset.
    pub fn last_keypoint_id_before_frame(&self) -> u64 {
        on_lane!(self, |flow| flow.last_keypoint_id_before_frame())
    }

    /// One camera's occupancy counts.
    pub fn cell_counts(&self, camera: usize) -> &[i32] {
        on_lane!(self, |flow| flow.cell_counts(camera))
    }

    /// The occupancy grid's geometry.
    pub fn occupancy_grid(&self) -> kornia_staging_imgproc::features::CellGrid {
        on_lane!(self, |flow| flow.occupancy_grid())
    }

    /// The last committed frameset's timestamp, `None` before the first.
    pub fn t_ns(&self) -> Option<i64> {
        on_lane!(self, |flow| flow.t_ns())
    }
}

/// Build the frontend lane a [`Backend`] names.
///
/// The CPU arm is `FrameToFrameOpticalFlow::new`. The GPU arm makes the two
/// CubeCL stage backends on one shared client and hands them to
/// `with_stages`, which is the whole of what selecting a backend costs.
pub(super) fn build_frontend(
    config: &config::VioConfig,
    calibration: &calib::Calibration<f64>,
    options: frontend::flow::FrontendOptions,
    backend: Backend,
) -> Result<FrontendLane, VioError> {
    // The two counts a backend-specific arm casts to `usize` to size its
    // buffers, checked here rather than after the cast: the GPU arm's
    // `optical_flow_levels as usize + 1` panics on `-1` in a debug build and
    // wraps to zero in a release one, either way before
    // `FrameToFrameOpticalFlow::with_stages` can run the frontend's own
    // refusal. Both arms return that refusal now, on the same field and value,
    // and no device is constructed for a config no backend can run.
    for (field, value) in [
        ("optical_flow_levels", config.optical_flow_levels),
        (
            "optical_flow_max_iterations",
            config.optical_flow_max_iterations,
        ),
    ] {
        if value < 0 {
            return Err(frontend::flow::FrontendError::NegativeConfig { field, value }.into());
        }
    }

    match backend {
        Backend::Cpu => Ok(FrontendLane::Cpu(
            frontend::flow::FrameToFrameOpticalFlow::new(config.clone(), calibration, options)?,
        )),
        #[cfg(feature = "gpu-wgpu")]
        Backend::Gpu => {
            let num_levels: usize = config.optical_flow_levels as usize + 1;
            let stages =
                gpu::gpu_stages::<kornia_staging_imgproc::optical_flow::patch_se2::Pattern51>(
                    options.max_keypoints,
                    num_levels,
                    config.optical_flow_max_iterations as usize,
                    config.optical_flow_max_recovered_dist2,
                    calibration.intrinsics.len(),
                )
                .map_err(frontend::flow::FrontendError::from)?;
            // One worker: the side cameras' detection stays on the caller.
            let host_pool = frontend::parallel::WorkPool::new(1)
                .map_err(|_| frontend::flow::FrontendError::ThreadPool { threads: 1 })?;
            Ok(FrontendLane::Gpu(
                frontend::flow::FrameToFrameOpticalFlow::with_stages(
                    config.clone(),
                    calibration,
                    options,
                    stages,
                    host_pool,
                )?,
            ))
        }
        #[cfg(not(feature = "gpu-wgpu"))]
        Backend::Gpu => Err(VioError::GpuUnavailable),
    }
}
