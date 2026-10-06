//! GPU frame resources and the current/next-frame submission schedule.

pub(super) mod onewait;

use super::{
    GpuCornerScan, GpuError, GpuPatchSources, GpuPatchTracker, GpuPyramid, GpuPyramidBuilder,
    guarded, submission,
};
use crate::frontend::detect::FrameCornerScan;
use crate::frontend::flow::{FlowTimings, FrontendError};
use crate::frontend::input::{FrameImages, PackedImages};
use crate::frontend::parallel::WorkPool;
use crate::frontend::stages::{FrameStages, StereoContext};
use crate::pyramid::ensure_pyramid_sizes;
use crate::{VioError, duration_ns};
use cubecl::prelude::*;
use kornia_staging_imgproc::features::{CellSelect, DetectorScratch};
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;
use kornia_staging_slam::tracking::optical_flow::{PatchTracker, TrackInput, TrackPhase};

#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameInput {
    Queued(i64),
    Ready(i64),
}

struct GpuFrame<R: Runtime> {
    builder: GpuPyramidBuilder<R>,
    pyramids: Vec<GpuPyramid<R>>,
    detector: DetectorScratch<GpuCornerScan<R>>,
    selects: Vec<Option<CellSelect>>,
    input: Option<FrameInput>,
    images: PackedImages,
}

impl<R: Runtime> GpuFrame<R> {
    fn new(
        client: ComputeClient<R>,
        launches: &submission::LaunchList,
    ) -> Result<Self, FrontendError> {
        let builder = GpuPyramidBuilder::new(client.clone(), launches.clone());
        let scanner = GpuCornerScan::new(client, launches.clone())?;
        Ok(Self {
            builder,
            pyramids: Vec::new(),
            detector: DetectorScratch::with_scanner(Box::new(scanner)),
            selects: Vec::new(),
            input: None,
            images: PackedImages::default(),
        })
    }

    fn build(&mut self, images: FrameImages<'_>, levels: usize) -> Result<(), FrontendError> {
        ensure_pyramid_sizes(
            &self.builder,
            &mut self.pyramids,
            images.iter().map(|image| image.size()),
            levels,
        )?;
        match images {
            FrameImages::Dense(images) => self.builder.build_images(images, &mut self.pyramids)?,
            FrameImages::Packed(images) => self.builder.build_packed(images, &mut self.pyramids)?,
        }
        self.detector.scanner_mut().use_level0(&mut self.builder);
        Ok(())
    }
}

/// Frame storage and scheduling stay here; the KLT tracker owns only point work.
pub struct GpuStages<P: Pattern, R: Runtime> {
    client: ComputeClient<R>,
    current: GpuFrame<R>,
    next: Option<GpuFrame<R>>,
    previous: Vec<GpuPyramid<R>>,
    tracker: GpuPatchTracker<P, R>,
    patches: GpuPatchSources<P, R>,
    one_wait: Option<onewait::OneWait>,
    geometry: Vec<u32>,
    guard: Option<submission::FrameBatch>,
    launches: submission::LaunchList,
}

impl<P: Pattern, R: Runtime> GpuStages<P, R> {
    pub(super) fn new(
        client: ComputeClient<R>,
        tracker: GpuPatchTracker<P, R>,
        launches: submission::LaunchList,
    ) -> Result<Self, FrontendError> {
        let patches = tracker.make_patches()?;
        Ok(Self {
            current: GpuFrame::new(client.clone(), &launches)?,
            client,
            tracker,
            patches,
            previous: Vec::new(),
            next: None,
            one_wait: None,
            geometry: Vec::new(),
            guard: None,
            launches,
        })
    }

    pub(crate) fn queue_lookahead(
        &mut self,
        t_ns: i64,
        images: &mut PackedImages,
        selects: &[Option<CellSelect>],
    ) -> Result<(), VioError> {
        if self.next.is_none() {
            self.next = Some(
                self.client
                    .exclusive(|| GpuFrame::new(self.client.clone(), &self.launches))
                    .map_err(|error| {
                        FrontendError::from(submission::read_failed(
                            "lookahead construction",
                            &error,
                        ))
                    })??,
            );
        }
        if let Some(next) = &mut self.next {
            std::mem::swap(&mut next.images, images);
            next.selects.clear();
            next.selects.extend_from_slice(selects);
            next.input = Some(FrameInput::Queued(t_ns));
        }
        Ok(())
    }

    pub(crate) fn discard_lookahead(&mut self) {
        self.current.input = None;
        if let Some(next) = &mut self.next {
            next.input = None;
        }
    }

    fn submit_lookahead(&mut self) -> Result<(), FrontendError> {
        let Some(next) = &mut self.next else {
            return Ok(());
        };
        let Some(FrameInput::Queued(t_ns)) = next.input else {
            return Ok(());
        };
        next.input = None;
        // Move the owned input aside while building through the same frame method.
        let images = std::mem::take(&mut next.images);
        let outcome = next.build(FrameImages::Packed(&images), self.tracker.num_levels() - 1);
        next.images = images;
        outcome?;
        next.detector
            .scanner_mut()
            .submit_cells(FrameImages::Packed(&next.images), &next.selects)?;
        guarded(
            GpuError::DeviceLost {
                what: "lookahead submission",
            },
            || {
                self.launches.flush(&self.client);
                self.client.flush().map_err(|error| {
                    FrontendError::from(submission::read_failed("lookahead submission", &error))
                })?;

                Ok::<(), FrontendError>(())
            },
        )?;
        next.input = Some(FrameInput::Ready(t_ns));
        Ok(())
    }

    fn collect(
        &mut self,
        timings: &mut FlowTimings,
        overlap_lookahead: bool,
    ) -> Result<(), FrontendError> {
        let outcome = guarded(GpuError::DeviceLost { what: "tracker" }, || {
            let mut reads = self.tracker.read_handles();
            let lanes = reads.len();
            let stereo_read = self
                .one_wait
                .as_ref()
                .is_some_and(|state| matches!(state.phase, onewait::Phase::Submitted));
            if let Some(state) = self.one_wait.as_ref().filter(|_| stereo_read) {
                reads.push(state.io.clone());
            }
            let staged = self.current.detector.scanner_mut().take_staged();
            let selected = staged.is_some();
            if let Some(handles) = staged {
                reads.extend(handles);
            }
            let mut bytes = if reads.is_empty() {
                Vec::new()
            } else {
                submission::read_with_lookahead(
                    &self.client.clone(),
                    &self.launches.clone(),
                    reads,
                    "the tracker result",
                    || {
                        if stereo_read || overlap_lookahead {
                            self.submit_lookahead().map_err(|error| {
                                log::warn!("lookahead preparation failed: {error}");
                                GpuError::DeviceLost {
                                    what: "lookahead preparation",
                                }
                            })?;
                        }
                        Ok(())
                    },
                )?
            };
            let outputs = lanes + usize::from(stereo_read);
            if selected && bytes.len() >= outputs {
                self.current
                    .detector
                    .scanner_mut()
                    .deliver(bytes.split_off(outputs));
            }
            if stereo_read && bytes.len() == outputs {
                timings.gpu_one_wait = true;
                if let Some(state) = &mut self.one_wait
                    && let Some(bytes) = bytes.pop()
                {
                    state.phase = onewait::Phase::Ready(bytes);
                }
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
        t_ns: i64,
        images: FrameImages<'_>,
        levels: usize,
        _pool: &WorkPool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        self.guard = Some(self.launches.begin()?);
        let mark = std::time::Instant::now();
        // A hint is reusable only for the exact retained byte input.
        if self.current.input == Some(FrameInput::Ready(t_ns))
            && matches!(images, FrameImages::Packed(images) if self.current.images.same_pixels(images))
        {
            timings.gpu_lookahead = true;
        } else {
            self.current.input = None;
            if let Err(error) = self.current.build(images, levels) {
                self.guard.take();
                return Err(error);
            }
        }
        timings.pyramid_ns = duration_ns(mark);
        Ok(())
    }
    fn prepare_detection(
        &mut self,
        images: FrameImages<'_>,
        selects: &[Option<CellSelect>],
        context: StereoContext<'_>,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        let one_wait = self.prepare_one_wait(context, selects)?;
        let mark = std::time::Instant::now();
        self.current.selects.clear();
        self.current
            .selects
            .extend(selects.iter().enumerate().map(|(camera, select)| {
                if one_wait || camera == 0 {
                    *select
                } else {
                    None
                }
            }));
        if self.current.input.take().is_none() {
            self.current
                .detector
                .scanner_mut()
                .submit_cells(images, &self.current.selects)?;
        }
        timings.detect_ns += duration_ns(mark);
        Ok(())
    }
    fn temporal(
        &mut self,
        inputs: &[TrackInput],
        slots: &mut [usize],
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        self.tracker.submit_batch(
            &self.previous,
            &self.current.pyramids,
            TrackPhase::Temporal(inputs),
            &mut self.patches,
            slots,
        )?;
        self.submit_stereo()?;
        self.collect(timings, false)
    }
    fn stereo(
        &mut self,
        phase: TrackPhase<'_>,
        slots: &mut [usize],
        images: FrameImages<'_>,
        selects: &[Option<CellSelect>],
        nonoverlap: bool,
        timings: &mut FlowTimings,
    ) -> Result<(), FrontendError> {
        let mark = std::time::Instant::now();
        if self.take_stereo(phase, slots)? {
            timings.stereo_ns += duration_ns(mark);
            return Ok(());
        }
        if !slots.is_empty() {
            self.tracker.submit_batch(
                &self.current.pyramids,
                &self.current.pyramids,
                phase,
                &mut self.patches,
                slots,
            )?;
        }
        timings.stereo_ns += duration_ns(mark);
        if !slots.is_empty() && nonoverlap {
            let mark = std::time::Instant::now();
            self.current.selects.copy_from_slice(selects);
            self.current.selects[0] = None;
            self.current
                .detector
                .scanner_mut()
                .submit_cells(images, &self.current.selects)?;
            timings.detect_ns += duration_ns(mark);
        }
        let mark = std::time::Instant::now();
        if !slots.is_empty() {
            self.collect(timings, true)?;
        }
        timings.stereo_ns += duration_ns(mark);
        let mark = std::time::Instant::now();
        self.current.detector.scanner_mut().take_cells()?;
        timings.detect_ns += duration_ns(mark);
        Ok(())
    }
    fn finish(&mut self) -> Result<(), FrontendError> {
        if let Some(guard) = self.guard.take() {
            guard.finish(&self.client)?;
        }
        self.submit_lookahead()?;
        std::mem::swap(&mut self.previous, &mut self.current.pyramids);
        // Keep the completed hint in current while the next call queues its successor.
        if let Some(next) = &mut self.next
            && matches!(next.input, Some(FrameInput::Ready(_)))
        {
            std::mem::swap(&mut self.current, next);
        }
        Ok(())
    }
    fn discard(&mut self) {
        self.guard.take();
        self.tracker.discard();
        if let Some(state) = &mut self.one_wait {
            state.phase = onewait::Phase::Off;
        }
    }
}

impl<P: Pattern, R: Runtime> std::fmt::Debug for GpuStages<P, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuStages")
            .field("tracker", &self.tracker)
            .finish_non_exhaustive()
    }
}
