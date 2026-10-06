//! Frame preparation and tracking phases owned by the frontend.

use crate::calib::Calibration;
use crate::camera::RigCamera;
use crate::config::VioConfig;
use crate::duration_ns;
use crate::frontend::detect::FrameCornerScan;
use crate::frontend::input::FrameImages;
use crate::frontend::parallel::WorkPool;
use crate::pyramid::{CpuPyramidBuilder, PyramidBuilder, ensure_pyramid_sizes};
use kornia_staging_imgproc::features::{CellSelect, DetectorScratch};
use kornia_staging_imgproc::pyramid::PyramidPlanU16;
use kornia_staging_slam::tracking::optical_flow::{PatchTracker, TrackInput, TrackPhase};

use super::flow::{FlowTimings, FrontendError};

/// Per-frame inputs to device stereo selection. Durable rig state stays in the flow.
pub struct StereoContext<'a> {
    pub cameras: &'a [RigCamera<f32>],
    pub calib: &'a Calibration<f32>,
    pub config: &'a VioConfig,
    pub depth: f32,
    pub last_detect_count: usize,
    pub eligible: bool,
}

/// The frame-level seam. Patch trackers expose only patch operations.
pub trait FrameStages {
    type Tracker: PatchTracker<Error: Into<FrontendError>>;
    type Scanner: FrameCornerScan<Error: Into<FrontendError>> + ?Sized;

    fn tracker(&self) -> &Self::Tracker;
    fn tracker_mut(&mut self) -> &mut Self::Tracker;
    fn detector(&mut self) -> &mut DetectorScratch<Self::Scanner>;

    fn side_detectors(&self, cameras: usize) -> Option<Vec<DetectorScratch<Self::Scanner>>>;

    fn executor(&self) -> Option<impl FrameExecutor + use<Self>> {
        None::<std::convert::Infallible>
    }

    fn prepare(
        &mut self,
        t_ns: i64,
        images: FrameImages<'_>,
        levels: usize,
        pool: &WorkPool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError>;

    fn prepare_detection(
        &mut self,
        images: FrameImages<'_>,
        selects: &[Option<CellSelect>],
        context: StereoContext<'_>,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError>;

    fn temporal(
        &mut self,
        inputs: &[TrackInput],
        slots: &mut [usize],
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError>;

    fn stereo(
        &mut self,
        phase: TrackPhase<'_>,
        slots: &mut [usize],
        images: FrameImages<'_>,
        selects: &[Option<CellSelect>],
        nonoverlap: bool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError>;

    fn finish(&mut self) -> Result<(), FrontendError>;
    fn discard(&mut self);
}

/// CPU frame storage. The previous pyramid changes only when a frame commits.
#[derive(Debug)]
pub struct CpuStages<T: PatchTracker<Pyramid = PyramidPlanU16>, S: FrameCornerScan = kornia_staging_imgproc::features::CpuCornerScan> {
    builder: CpuPyramidBuilder,
    previous: Vec<PyramidPlanU16>,
    current: Vec<PyramidPlanU16>,
    detector: DetectorScratch<S>,
    tracker: T,
    patches: T::Patches,
}

impl<T: PatchTracker<Pyramid = PyramidPlanU16>, S: FrameCornerScan> CpuStages<T, S>
where
    FrontendError: From<T::Error>,
{
    pub fn new(
        builder: CpuPyramidBuilder,
        tracker: T,
        detector: DetectorScratch<S>,
    ) -> Result<Self, FrontendError> {
        let patches = tracker.make_patches()?;
        Ok(Self {
            builder,
            previous: Vec::new(),
            current: Vec::new(),
            detector,
            tracker,
            patches,
        })
    }
}

impl<T: PatchTracker<Pyramid = PyramidPlanU16>, S: FrameCornerScan> FrameStages for CpuStages<T, S>
where
    FrontendError: From<T::Error>,
{
    type Tracker = T;
    type Scanner = S;

    fn tracker(&self) -> &T {
        &self.tracker
    }
    fn tracker_mut(&mut self) -> &mut T {
        &mut self.tracker
    }
    fn detector(&mut self) -> &mut DetectorScratch<Self::Scanner> {
        &mut self.detector
    }
    fn side_detectors(&self, cameras: usize) -> Option<Vec<DetectorScratch<Self::Scanner>>> {
        (1..cameras)
            .map(|_| {
                self.detector
                    .scanner()
                    .fork_frame()
                    .map(DetectorScratch::with_scanner)
            })
            .collect()
    }

    fn prepare(
        &mut self,
        _t_ns: i64,
        images: FrameImages<'_>,
        levels: usize,
        pool: &WorkPool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        let mark = std::time::Instant::now();
        let FrameImages::Dense(images) = images else {
            unreachable!("CPU stage receives dense frames")
        };
        ensure_pyramid_sizes(
            &self.builder,
            &mut self.current,
            images.iter().map(kornia_image::Image::size),
            levels,
        )?;
        self.builder.build_frames(images, &mut self.current, pool)?;
        timings.pyramid_ns = duration_ns(mark);
        Ok(())
    }

    fn prepare_detection(
        &mut self,
        _images: FrameImages<'_>,
        _selects: &[Option<CellSelect>],
        _context: StereoContext<'_>,
        _timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        Ok(())
    }

    fn temporal(
        &mut self,
        inputs: &[TrackInput],
        slots: &mut [usize],
        _timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        self.tracker.submit_batch(
            &self.previous,
            &self.current,
            TrackPhase::Temporal(inputs),
            &mut self.patches,
            slots,
        )?;
        self.tracker.collect()?;
        Ok(())
    }

    fn stereo(
        &mut self,
        phase: TrackPhase<'_>,
        slots: &mut [usize],
        _images: FrameImages<'_>,
        _selects: &[Option<CellSelect>],
        _nonoverlap: bool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        let mark = std::time::Instant::now();
        if !slots.is_empty() {
            self.tracker.submit_batch(
                &self.current,
                &self.current,
                phase,
                &mut self.patches,
                slots,
            )?;
        }
        timings.stereo_ns += duration_ns(mark);
        let mark = std::time::Instant::now();
        if !slots.is_empty() {
            self.tracker.collect()?;
        }
        timings.stereo_ns += duration_ns(mark);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), FrontendError> {
        std::mem::swap(&mut self.previous, &mut self.current);
        Ok(())
    }

    fn discard(&mut self) {
        self.tracker.discard();
    }
}

/// Runs a whole frontend frame on the backend's submission thread.
pub trait FrameExecutor {
    fn run(
        self,
        body: impl FnOnce() -> Result<(), super::flow::FrontendError> + Send,
    ) -> Result<(), super::flow::FrontendError>;
}

impl FrameExecutor for std::convert::Infallible {
    fn run(
        self,
        _body: impl FnOnce() -> Result<(), super::flow::FrontendError> + Send,
    ) -> Result<(), super::flow::FrontendError> {
        match self {}
    }
}
