//! Stateful KLT batching and template reuse.
//!
//! Create tracker-owned source storage and choose an explicit execution policy.
//! Reuse the tracker across frames to retain identity-keyed templates.
//!
//! ```
//! use kornia_staging_imgproc::optical_flow::patch_se2::Pattern51;
//! use kornia_staging_slam::tracking::optical_flow::{CpuPatchTracker, PatchTracker};
//! let tracker = CpuPatchTracker::<Pattern51>::new(128, 4, 5, 0.04, None)?;
//! let patches = tracker.make_patches()?;
//! assert_eq!(tracker.threads(), 1);
//! assert_eq!(patches.capacity(), 128);
//! assert_eq!(patches.num_levels(), 4);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
mod cpu;
pub use cpu::CpuPatchTracker;
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;
use kornia_staging_imgproc::optical_flow::patch_tracker::{
    FlowResult, FlowTransforms, PatchSoA, PointsSoA, TrackerError,
};
use kornia_staging_imgproc::pyramid::PyramidPlanU16;
/// One temporal camera's source points and destination guesses.
#[derive(Debug, Default)]
pub struct TrackInput {
    /// Keypoint identities, ascending, after source masking.
    pub ids: Vec<u64>,
    /// Source template positions at level zero.
    pub positions: PointsSoA,
    /// Source linear transforms with predicted destination positions.
    pub guesses: FlowTransforms,
}

/// A tracking phase in camera order. Matching owns one camera-zero source;
/// destination guesses correspond to cameras one onward.
#[derive(Clone, Copy, Debug)]
pub enum TrackPhase<'a> {
    /// Each camera uses its own previous image and source points.
    Temporal(&'a [TrackInput]),
    /// Every destination uses the same camera-zero identities and positions.
    Matching {
        /// Shared source identities.
        ids: &'a [u64],
        /// Shared source positions.
        positions: &'a PointsSoA,
        /// Destination guesses in camera order, starting at camera one.
        destinations: &'a [FlowTransforms],
    },
}

/// One lane of a validated phase. Source data is borrowed, never duplicated.
#[derive(Clone, Copy)]
pub struct TrackLane<'a> {
    /// Source camera index.
    pub source: usize,
    /// Destination camera index.
    pub destination: usize,
    /// Source identities.
    pub ids: &'a [u64],
    /// Source positions.
    pub positions: &'a PointsSoA,
    /// Destination guesses.
    pub guesses: &'a FlowTransforms,
}

/// Geometry-checked phase shared by CPU, packed GPU and fallback submission.
#[derive(Clone, Copy)]
pub struct ValidatedPhase<'a> {
    phase: TrackPhase<'a>,
}
impl<'a> TrackPhase<'a> {
    /// Check the complete phase before mutating caches or launching work.
    /// Pyramid callbacks return None for an absent camera, or its level count.
    /// # Errors
    /// Rejects missing cameras, unequal point counts, depth or capacity mismatches.
    pub fn validate(
        self,
        capacity: usize,
        levels: usize,
        slots: usize,
        patch_levels: Option<usize>,
        previous_levels: impl Fn(usize) -> Option<usize>,
        next_levels: impl Fn(usize) -> Option<usize>,
    ) -> Result<ValidatedPhase<'a>, TrackerError> {
        let view = ValidatedPhase { phase: self };
        if slots != view.len() {
            return Err(TrackerError::LengthMismatch {
                first_name: "result slots",
                first: slots,
                second_name: "phase lanes",
                second: view.len(),
            });
        }
        for lane in view.iter() {
            let previous = previous_levels(lane.source).ok_or(TrackerError::MissingCamera {
                role: "source",
                camera: lane.source,
            })?;
            let next = next_levels(lane.destination).ok_or(TrackerError::MissingCamera {
                role: "destination",
                camera: lane.destination,
            })?;
            kornia_staging_imgproc::optical_flow::patch_tracker::limits::check_track_inputs(
                lane.guesses.len(),
                lane.positions.len(),
                patch_levels.unwrap_or(levels),
                previous,
                next,
                capacity,
                levels,
            )?;
            if lane.ids.len() != lane.positions.len() {
                return Err(TrackerError::LengthMismatch {
                    first_name: "keypoint ids",
                    first: lane.ids.len(),
                    second_name: "positions",
                    second: lane.positions.len(),
                });
            }
        }
        Ok(view)
    }
}
impl<'a> ValidatedPhase<'a> {
    /// Number of destination lanes.
    pub fn len(self) -> usize {
        match self.phase {
            TrackPhase::Temporal(inputs) => inputs.len(),
            TrackPhase::Matching { destinations, .. } => destinations.len(),
        }
    }
    /// Whether the phase has no destinations.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    /// Whether sources belong to the previous frame.
    pub fn is_temporal(self) -> bool {
        matches!(self.phase, TrackPhase::Temporal(_))
    }
    /// Borrow a lane, in destination camera order.
    /// # Panics
    /// If index is outside the phase.
    pub fn lane(self, index: usize) -> TrackLane<'a> {
        match self.phase {
            TrackPhase::Temporal(inputs) => {
                let input = &inputs[index];
                TrackLane {
                    source: index,
                    destination: index,
                    ids: &input.ids,
                    positions: &input.positions,
                    guesses: &input.guesses,
                }
            }
            TrackPhase::Matching {
                ids,
                positions,
                destinations,
            } => TrackLane {
                source: 0,
                destination: index + 1,
                ids,
                positions,
                guesses: &destinations[index],
            },
        }
    }
    /// Iterate in camera order without allocating lane records.
    pub fn iter(self) -> impl Iterator<Item = TrackLane<'a>> {
        (0..self.len()).map(move |i| self.lane(i))
    }
}

/// The source patches of one camera, whatever holds them.
///
/// Split from [`PatchTracker`] so a backend can pair its own patch storage with
/// its own pyramid: `build` is the "sample every patch at every level" stage the
/// GPU wants as one kernel, and the tracker consumes the result.
pub trait SourcePatches {
    /// Backend failure, including shared input validation errors.
    type Error: std::error::Error + Send + Sync + From<TrackerError>;

    /// The pyramid representation these patches are sampled from.
    type Pyramid;

    /// Build one patch per entry of `positions`, at every level.
    ///
    /// # Arguments
    /// * `pyramid` - Immutable source levels for this build.
    /// * `positions` - Centres in full-resolution pixels, preserving input order.
    /// * `selected` - Optional mask; false slots are not sampled.
    ///
    /// `selected` may switch patches off; a patch that is switched off is marked
    /// invalid at every level and never sampled.
    ///
    /// # Errors
    ///
    /// [`TrackerError`] when the positions do not fit, the selection mask is
    /// shorter than the positions, or the pyramid is the wrong depth.
    fn build(
        &mut self,
        pyramid: &Self::Pyramid,
        positions: &PointsSoA,
        selected: Option<&[bool]>,
    ) -> Result<(), Self::Error>;
}

/// Reusable output slots owned by a tracking batch on either backend.
#[derive(Debug, Default)]
pub struct TrackBatch {
    pub(crate) slots: Vec<FlowResult>,
    submitted: usize,
}

impl TrackBatch {
    /// Reserve a reusable result slot for a synchronous backend.
    pub fn submit_slot(&mut self, capacity: usize) -> (usize, &mut FlowResult) {
        let pass = self.submitted;
        self.submitted += 1;
        (pass, self.slot_mut(pass, capacity))
    }

    /// Allocate or reuse a slot in submission order; skip no slots.
    pub fn slot_mut(&mut self, lane: usize, capacity: usize) -> &mut FlowResult {
        if lane == self.slots.len() {
            self.slots.push(FlowResult::with_capacity(capacity));
        }
        &mut self.slots[lane]
    }

    /// Mutable access to an allocated result slot.
    ///
    /// # Panics
    /// If `pass` has not been reserved.
    pub fn result_mut(&mut self, pass: usize) -> &mut FlowResult {
        &mut self.slots[pass]
    }

    /// Slot `pass` remains readable until the next submission reuses it.
    pub fn result(&self, pass: usize) -> &FlowResult {
        &self.slots[pass]
    }
}

/// The frontend's tracking stage: one call moves a whole camera's patches.
///
/// The trait takes the entire keypoint set, never one point, and names no
/// concrete pyramid or patch storage, so the CPU implementation here and a later
/// CubeCL one can be swapped without touching the driver.
pub trait PatchTracker {
    /// Backend failure, including shared input validation errors.
    type Error: std::error::Error + Send + Sync + From<TrackerError>;

    /// Set a positive, finite per-level convergence threshold, or disable early exit.
    ///
    /// # Errors
    /// Returns [`TrackerError::InvalidExitStep`] for invalid thresholds.
    fn set_klt_exit_step_px(&mut self, threshold: Option<f32>) -> Result<(), Self::Error>;

    /// Reusable result storage owned by this backend.
    fn batch(&self) -> &TrackBatch;
    /// Mutable result storage used by the synchronous submission default.
    fn batch_mut(&mut self) -> &mut TrackBatch;

    /// Submit all cameras of one phase, retaining camera order for results.
    /// Temporal calls use each camera's previous pyramid. Matching calls use
    /// the same camera-zero templates for every destination. The default keeps
    /// device launch/collect batching; CPU trackers can run each phase jointly.
    ///
    /// For template reuse, pass the last committed temporal batch's `next` as
    /// `prev`; newly matched points use the matching batch's destination pyramid.
    /// Call [`Self::discard`] after any failed or abandoned batch, including in
    /// wrappers that forward submission. The CPU cache also checks the source
    /// pyramid's build generation, so a different or rebuilt `prev` is a miss.
    fn submit_batch(
        &mut self,
        prev: &[Self::Pyramid],
        next: &[Self::Pyramid],
        phase: TrackPhase<'_>,
        patches: &mut Self::Patches,
        slots: &mut [usize],
    ) -> Result<(), Self::Error>;

    /// Read a result by the slot returned from submission, after collection.
    fn result(&self, pass: usize) -> &FlowResult {
        self.batch().result(pass)
    }

    /// Finish every submitted pass, retaining results in the batch's slots.
    ///
    /// # Errors
    /// A device download can fail.
    fn collect(&mut self) -> Result<(), Self::Error> {
        self.batch_mut().submitted = 0;
        Ok(())
    }

    /// Drop the current batch while retaining all result allocations.
    fn discard(&mut self) {
        self.batch_mut().submitted = 0;
    }

    /// The sampling pattern this tracker was built for.
    type Pattern: Pattern;

    /// The pyramid representation it reads.
    type Pyramid;

    /// The source-patch storage it consumes.
    type Patches: SourcePatches<Pyramid = Self::Pyramid, Error = Self::Error>;

    /// Keypoints this tracker can carry in one call.
    fn capacity(&self) -> usize;

    /// Number of pyramid levels, including the full-resolution level.
    fn num_levels(&self) -> usize;

    /// Fresh source-patch storage matching this tracker's capacity and depth.
    ///
    /// The driver cannot name the concrete type, so the tracker makes it.
    ///
    /// # Errors
    ///
    /// [`TrackerError`] when the storage cannot be sized — the same shape checks
    /// the tracker's own constructor made.
    fn make_patches(&self) -> Result<Self::Patches, Self::Error>;
}

/// Submit each validated lane without introducing a wait between cameras.
/// The callback owns backend dispatch; output slots remain separate from inputs.
pub fn submit_each<E>(
    phase: ValidatedPhase<'_>,
    slots: &mut [usize],
    mut submit: impl FnMut(TrackLane<'_>) -> Result<usize, E>,
) -> Result<(), E> {
    for (slot, lane) in slots.iter_mut().zip(phase.iter()) {
        *slot = submit(lane)?;
    }
    Ok(())
}

impl<P: Pattern> SourcePatches for PatchSoA<P> {
    type Error = TrackerError;
    type Pyramid = PyramidPlanU16;
    fn build(
        &mut self,
        pyramid: &Self::Pyramid,
        positions: &PointsSoA,
        selected: Option<&[bool]>,
    ) -> Result<(), Self::Error> {
        self.build(pyramid, positions, selected)
    }
}
#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;
