//! Visual-inertial odometry core.
//!
//! The crate is deliberately free of Python, Rerun and GPU code: it consumes
//! grayscale images and IMU samples and returns plain values. Python plumbing
//! (catalog feed, evaluation, logging) lives in the `slam_rs` package and the
//! bindings in `slam-rs-py`; `slam-rs-cli` is a placeholder binary whose only
//! working subcommand is `version`.

pub mod calib;
pub mod camera;
pub mod config;
pub mod frontend;
pub mod image;
pub mod lie;
pub mod pyramid;
pub mod types;

/// Version of the core, as declared in `crates/slam-rs/Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Elapsed nanoseconds, saturating rather than panicking on an absurd clock.
///
/// The one place a stage mark is taken: the estimator's six
/// ([`estimator::StageTimings`]) and the frontend's three
/// ([`frontend::flow::FlowTimings`]) are the same measurement of different work.
pub(crate) fn duration_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Which frontend backend a [`Vio`] runs.
///
/// The stage traits make the choice a construction-time one (decision D21): the
/// CPU implementations stay in the crate permanently and a GPU build only adds a
/// second pair. `Cpu` is the default everywhere — the fleet's installs, the
/// gates and the accuracy references all run it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// The ported CPU frontend.
    #[default]
    Cpu,
    /// The CubeCL frontend on this host's GPU.
    ///
    /// Available only in a build with the `gpu` feature; [`Vio::with_backend`]
    /// returns [`VioError::GpuUnavailable`] otherwise, so the Python surface
    /// carries the same signature either way.
    Gpu,
}

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
    Cpu(frontend::flow::FrameToFrameOpticalFlow<frontend::patterns::Pattern51>),
}

/// Run the same expression against whichever backend the lane holds.
macro_rules! on_lane {
    ($lane:expr, |$flow:ident| $body:expr) => {
        match $lane {
            FrontendLane::Cpu($flow) => $body,
        }
    };
}

impl FrontendLane {
    // ── the seven [`Vio`] drives ──────────────────────────────────────────

    /// Which backend this lane runs.
    pub fn backend(&self) -> Backend {
        match self {
            Self::Cpu(_) => Backend::Cpu,
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
        images: &[image::ImageU16],
        prediction: &frontend::flow::PosePrediction,
        masks: &[frontend::detect::Masks],
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
    pub fn occupancy_grid(&self) -> frontend::detect::CellGrid {
        on_lane!(self, |flow| flow.occupancy_grid())
    }

    /// The last committed frameset's timestamp, `None` before the first.
    pub fn t_ns(&self) -> Option<i64> {
        on_lane!(self, |flow| flow.t_ns())
    }
}
