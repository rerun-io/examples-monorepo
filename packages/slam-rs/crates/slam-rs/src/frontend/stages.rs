//! Frame preparation and tracking phases owned by the frontend.

use crate::calib::Calibration;
use crate::camera::RigCamera;
use crate::config::VioConfig;
use crate::duration_ns;
use crate::frontend::detect::{CellSelect, DetectorScratch};
use crate::frontend::parallel::WorkPool;
use crate::frontend::tracker::{PatchTracker, TrackInput, TrackerError};
use crate::image::ImageU16;
use crate::pyramid::{CpuPyramidBuilder, PyramidBuilder, PyramidU16, ensure_pyramids};

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
    type Tracker: PatchTracker;
    type Scanner: crate::frontend::detect::CornerScan + ?Sized;

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
        images: &[ImageU16],
        levels: usize,
        pool: &WorkPool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError>;

    fn prepare_detection(
        &mut self,
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
        context: StereoContext<'_>,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError>;

    fn temporal(
        &mut self,
        inputs: &mut [TrackInput],
        timings: &mut FlowTimings,
    ) -> Result<(), TrackerError>;

    fn stereo(
        &mut self,
        inputs: &mut [TrackInput],
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
        nonoverlap: bool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError>;

    fn finish(&mut self) -> Result<(), FrontendError>;
    fn discard(&mut self);
}

/// CPU frame storage. The previous pyramid changes only when a frame commits.
#[derive(Debug)]
pub struct CpuStages<T: PatchTracker<Pyramid = PyramidU16>> {
    builder: CpuPyramidBuilder,
    previous: Vec<PyramidU16>,
    current: Vec<PyramidU16>,
    detector: DetectorScratch,
    tracker: T,
    patches: T::Patches,
}

impl<T: PatchTracker<Pyramid = PyramidU16>> CpuStages<T> {
    pub fn new(
        builder: CpuPyramidBuilder,
        tracker: T,
        detector: DetectorScratch,
    ) -> Result<Self, TrackerError> {
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

impl<T: PatchTracker<Pyramid = PyramidU16>> FrameStages for CpuStages<T> {
    type Tracker = T;
    type Scanner = dyn crate::frontend::detect::CornerScan;

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
        (1..cameras).map(|_| self.detector.fork()).collect()
    }

    fn prepare(
        &mut self,
        _t_ns: i64,
        images: &[ImageU16],
        levels: usize,
        pool: &WorkPool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        let mark = std::time::Instant::now();
        ensure_pyramids(&self.builder, &mut self.current, images, levels)?;
        self.builder.build_frames(images, &mut self.current, pool)?;
        timings.pyramid_ns = duration_ns(mark);
        Ok(())
    }

    fn prepare_detection(
        &mut self,
        _images: &[ImageU16],
        _selects: &[Option<CellSelect>],
        _context: StereoContext<'_>,
        _timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        Ok(())
    }

    fn temporal(
        &mut self,
        inputs: &mut [TrackInput],
        _timings: &mut FlowTimings,
    ) -> Result<(), TrackerError> {
        self.tracker.submit_batch(
            &self.previous,
            &self.current,
            inputs,
            &mut self.patches,
            true,
        )?;
        self.tracker.collect()
    }

    fn stereo(
        &mut self,
        inputs: &mut [TrackInput],
        _images: &[ImageU16],
        _selects: &[Option<CellSelect>],
        _nonoverlap: bool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        let mark = std::time::Instant::now();
        if !inputs.is_empty() {
            self.tracker.submit_batch(
                &self.current,
                &self.current,
                inputs,
                &mut self.patches,
                false,
            )?;
        }
        timings.stereo_ns += duration_ns(mark);
        let mark = std::time::Instant::now();
        if !inputs.is_empty() {
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
