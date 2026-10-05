//! GPU frame resources and the current/next-frame submission schedule.

use super::{
    GpuCornerScan, GpuError, GpuPatchSources, GpuPatchTracker, GpuPyramid, GpuPyramidBuilder,
    guarded, submission,
};
use crate::duration_ns;
use crate::frontend::detect::{CellSelect, DetectorScratch};
use crate::frontend::flow::{FlowTimings, FrontendError};
use crate::frontend::parallel::WorkPool;
use crate::frontend::patterns::Pattern;
use crate::frontend::stages::FrameStages;
use crate::frontend::tracker::{PatchTracker, TrackInput, TrackerError};
use crate::image::ImageU16;
use crate::pyramid::ensure_pyramids;
use cubecl::prelude::*;

struct GpuFrame<R: Runtime> {
    builder: GpuPyramidBuilder<R>,
    pyramids: Vec<GpuPyramid<R>>,
    detector: DetectorScratch<GpuCornerScan<R>>,
    selects: Vec<Option<CellSelect>>,
}

impl<R: Runtime> GpuFrame<R> {
    fn new(
        client: ComputeClient<R>,
        launches: &submission::LaunchList,
    ) -> Result<Self, TrackerError> {
        let builder = GpuPyramidBuilder::new(client.clone(), launches.clone());
        let scanner = GpuCornerScan::new(client, launches.clone())?;
        Ok(Self {
            builder,
            pyramids: Vec::new(),
            detector: DetectorScratch::with_scanner(Box::new(scanner)),
            selects: Vec::new(),
        })
    }

    fn build(&mut self, images: &[ImageU16], levels: usize) -> Result<(), FrontendError> {
        ensure_pyramids(&self.builder, &mut self.pyramids, images, levels)?;
        self.builder.build_images(images, &mut self.pyramids)?;
        self.detector.scanner.use_level0(&mut self.builder);
        Ok(())
    }
}

/// Frame storage and scheduling stay here; the KLT tracker owns only point work.
pub struct GpuStages<P: Pattern, R: Runtime> {
    client: ComputeClient<R>,
    current: GpuFrame<R>,
    previous: Vec<GpuPyramid<R>>,
    tracker: GpuPatchTracker<P, R>,
    patches: GpuPatchSources<P, R>,
    guard: Option<submission::FrameBatch>,
    launches: submission::LaunchList,
}

impl<P: Pattern, R: Runtime> GpuStages<P, R> {
    pub(super) fn new(
        client: ComputeClient<R>,
        tracker: GpuPatchTracker<P, R>,
        launches: submission::LaunchList,
    ) -> Result<Self, TrackerError> {
        let patches = tracker.make_patches()?;
        Ok(Self {
            current: GpuFrame::new(client.clone(), &launches)?,
            client,
            tracker,
            patches,
            previous: Vec::new(),
            guard: None,
            launches,
        })
    }

    fn collect(&mut self) -> Result<(), TrackerError> {
        let outcome = guarded(GpuError::DeviceLost { what: "tracker" }, || {
            let mut reads = self.tracker.read_handles();
            let lanes = reads.len();
            let staged = self.current.detector.scanner.take_staged();
            let selected = staged.is_some();
            if let Some(handles) = staged {
                reads.extend(handles);
            }
            let mut bytes = if reads.is_empty() {
                Vec::new()
            } else {
                submission::read_blocking(
                    &self.client,
                    &self.launches,
                    reads,
                    "the tracker result",
                )?
            };
            let outputs = lanes;
            if selected && bytes.len() >= outputs {
                self.current
                    .detector
                    .scanner
                    .deliver(bytes.split_off(outputs));
            }
            self.tracker.decode_results(&bytes)
        });
        self.tracker.discard();
        outcome
    }
}

impl<P: Pattern, R: Runtime> FrameStages for GpuStages<P, R> {
    type Tracker = GpuPatchTracker<P, R>;
    type Scanner = GpuCornerScan<R>;
    fn tracker(&self) -> &Self::Tracker {
        &self.tracker
    }
    fn tracker_mut(&mut self) -> &mut Self::Tracker {
        &mut self.tracker
    }
    fn detector(&mut self) -> &mut DetectorScratch<Self::Scanner> {
        &mut self.current.detector
    }
    fn side_detectors(&self, _cameras: usize) -> Option<Vec<DetectorScratch<Self::Scanner>>> {
        None
    }
    fn executor(&self) -> Option<impl crate::frontend::stages::FrameExecutor + use<P, R>> {
        Some(submission::FrameExecutor {
            client: self.client.clone(),
        })
    }
    fn prepare(
        &mut self,
        _t_ns: i64,
        images: &[ImageU16],
        levels: usize,
        _pool: &WorkPool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        self.guard = Some(self.launches.begin().map_err(TrackerError::from)?);
        let mark = std::time::Instant::now();
        if let Err(error) = self.current.build(images, levels) {
            self.guard.take();
            return Err(error);
        }
        timings.pyramid_ns = duration_ns(mark);
        Ok(())
    }
    fn prepare_detection(
        &mut self,
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        let mark = std::time::Instant::now();
        self.current.selects.clear();
        self.current.selects.extend(
            selects.iter().enumerate().map(
                |(camera, select)| {
                    if camera == 0 { *select } else { None }
                },
            ),
        );
        self.current
            .detector
            .submit_cells(images, &self.current.selects)?;
        timings.detect_ns += duration_ns(mark);
        Ok(())
    }
    fn temporal(
        &mut self,
        inputs: &mut [TrackInput],
        _timings: &mut FlowTimings,
    ) -> Result<(), TrackerError> {
        self.tracker.submit_batch(
            &self.previous,
            &self.current.pyramids,
            inputs,
            &mut self.patches,
            true,
        )?;
        self.collect()
    }
    fn stereo(
        &mut self,
        inputs: &mut [TrackInput],
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
        nonoverlap: bool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        let mark = std::time::Instant::now();
        if !inputs.is_empty() {
            self.tracker.submit_batch(
                &self.current.pyramids,
                &self.current.pyramids,
                inputs,
                &mut self.patches,
                false,
            )?;
        }
        timings.stereo_ns += duration_ns(mark);
        if !inputs.is_empty() && nonoverlap {
            let mark = std::time::Instant::now();
            self.current.selects.copy_from_slice(selects);
            self.current.selects[0] = None;
            self.current
                .detector
                .submit_cells(images, &self.current.selects)?;
            timings.detect_ns += duration_ns(mark);
        }
        let mark = std::time::Instant::now();
        if !inputs.is_empty() {
            self.collect()?;
        }
        timings.stereo_ns += duration_ns(mark);
        let mark = std::time::Instant::now();
        self.current.detector.take_cells()?;
        timings.detect_ns += duration_ns(mark);
        Ok(())
    }
    fn finish(&mut self) -> Result<(), FrontendError> {
        if let Some(guard) = self.guard.take() {
            guard.finish(&self.client)?;
        }
        std::mem::swap(&mut self.previous, &mut self.current.pyramids);
        Ok(())
    }
    fn discard(&mut self) {
        self.guard.take();
        self.tracker.discard();
    }
}

impl<P: Pattern, R: Runtime> std::fmt::Debug for GpuStages<P, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuStages")
            .field("tracker", &self.tracker)
            .finish_non_exhaustive()
    }
}
