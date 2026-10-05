//! The flow frontend's shared fixtures: the frontend on the synthetic rig, a
//! CPU tracker, and a tracker that fails on a chosen call.

use slam_rs::config::VioConfig;
use slam_rs::frontend::detect::CpuCornerScan;
use slam_rs::frontend::flow::{FrameToFrameOpticalFlow, FrontendOptions};
use slam_rs::frontend::parallel::WorkPool;
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::frontend::tracker::{
    CpuPatchTracker, FlowTransforms, PatchSoA, PatchTracker, TrackerError,
};
use slam_rs::pyramid::{CpuPyramidBuilder, PyramidU16};

use super::{flow_config as config, flow_rig as rig};

/// The CPU frontend on the synthetic rig of `cameras` cameras.
#[allow(
    dead_code,
    reason = "used by the flow_* tests; other binaries compile a subset"
)]
pub fn frontend(cameras: usize, options: FrontendOptions) -> FrameToFrameOpticalFlow<Pattern51> {
    FrameToFrameOpticalFlow::new(config(), &rig(cameras), options).unwrap()
}

/// A one-worker CPU tracker shaped by `config`, for `capacity` keypoints.
#[allow(
    dead_code,
    reason = "used by the flow_* tests; other binaries compile a subset"
)]
pub fn cpu_tracker(config: &VioConfig, capacity: usize) -> CpuPatchTracker<Pattern51> {
    CpuPatchTracker::new(
        capacity,
        config.optical_flow_levels as usize + 1,
        config.optical_flow_max_iterations as usize,
        config.optical_flow_max_recovered_dist2,
        WorkPool::new(1).unwrap(),
    )
    .unwrap()
}

/// A tracker that forwards to the CPU one and refuses the *n*-th call.
///
/// The point is a failure from inside a pluggable backend, in the middle of
/// `processFrame`, after the frame's pyramids are built and after some of the
/// cameras have already been tracked. Nothing else can produce that.
#[allow(
    dead_code,
    reason = "used by the flow_* tests; other binaries compile a subset"
)]
#[derive(Debug)]
pub struct FailingTracker {
    inner: CpuPatchTracker<Pattern51>,
    calls: std::cell::Cell<usize>,
    failures: std::ops::RangeInclusive<usize>,
}

impl FailingTracker {
    #[allow(
        dead_code,
        reason = "used by frame_allocations; other binaries compile a subset"
    )]
    pub fn fail_from(inner: CpuPatchTracker<Pattern51>, call: usize) -> Self {
        Self {
            inner,
            calls: std::cell::Cell::new(0),
            failures: call..=usize::MAX,
        }
    }
}

impl PatchTracker for FailingTracker {
    fn set_klt_exit_step_px(&mut self, threshold: Option<f32>) {
        self.inner.set_klt_exit_step_px(threshold)
    }

    fn batch(&self) -> &slam_rs::frontend::tracker::TrackBatch {
        self.inner.batch()
    }
    fn batch_mut(&mut self) -> &mut slam_rs::frontend::tracker::TrackBatch {
        self.inner.batch_mut()
    }

    type Pattern = Pattern51;
    type Pyramid = PyramidU16;
    type Patches = PatchSoA<Pattern51>;

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    fn make_patches(&self) -> Result<PatchSoA<Pattern51>, TrackerError> {
        self.inner.make_patches()
    }

    fn submit(
        &mut self,
        prev: &PyramidU16,
        next: &PyramidU16,
        patches: &PatchSoA<Pattern51>,
        transforms_in: &FlowTransforms,
    ) -> Result<usize, TrackerError> {
        self.calls.set(self.calls.get() + 1);
        if self.failures.contains(&self.calls.get()) {
            return Err(TrackerError::CapacityExceeded {
                offered: usize::MAX,
                capacity: 0,
            });
        }
        self.inner.submit(prev, next, patches, transforms_in)
    }
}

/// [`failing_frontend_with_ratio`] at the default survivor ratio.
#[allow(
    dead_code,
    reason = "used by the flow_* tests; other binaries compile a subset"
)]
pub fn failing_frontend(
    fail_on: usize,
) -> FrameToFrameOpticalFlow<Pattern51, slam_rs::frontend::stages::CpuStages<FailingTracker>> {
    failing_frontend_with_ratio(fail_on, 0.0)
}

/// A frontend on two cameras whose tracker refuses its `fail_on`-th call.
#[allow(
    dead_code,
    reason = "used by the flow_* tests; other binaries compile a subset"
)]
pub fn failing_frontend_with_ratio(
    fail_on: usize,
    ratio: f32,
) -> FrameToFrameOpticalFlow<Pattern51, slam_rs::frontend::stages::CpuStages<FailingTracker>> {
    let config = VioConfig {
        port_redetect_survivor_ratio: ratio,
        ..config()
    };
    let options: FrontendOptions = FrontendOptions::default();
    let inner: CpuPatchTracker<Pattern51> = cpu_tracker(&config, options.max_keypoints);
    FrameToFrameOpticalFlow::with_stages(
        config,
        &rig(2),
        options,
        slam_rs::frontend::stages::CpuStages::new(
            CpuPyramidBuilder::new(),
            FailingTracker {
                inner,
                calls: std::cell::Cell::new(0),
                failures: fail_on..=fail_on,
            },
            slam_rs::frontend::detect::DetectorScratch::with_scanner(Box::new(
                CpuCornerScan::default(),
            )),
        )
        .unwrap(),
        WorkPool::new(1).unwrap(),
    )
    .unwrap()
}
