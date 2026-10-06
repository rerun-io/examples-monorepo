//! Inverse-compositional SE(2) KLT tracking.
//! Run bounded Gauss-Newton steps per level, coarse to fine, then track backward
//! and keep points that return close enough to their sources.
//!
//! [`PatchTracker`] and [`SourcePatches`] use an associated pyramid type so CPU
//! and GPU backends share the driver. Buffers use structure-of-arrays storage
//! with patch index varying fastest and allocate only above their high-water mark.
//!
//! Source patches depend only on the previous pyramid, position and level, so
//! build them before tracking. Backward patches depend on forward results and
//! are built between passes. Invalid points idle through fixed loop bounds to
//! keep GPU execution uniform. Masks and depth guesses stay in the driver;
//! the tracker receives guesses and recovers their source-position offsets.

mod cpu;
mod patch_soa;
pub use cpu::CpuPatchTracker;
mod storage;
pub use patch_soa::PatchSoA;
pub use storage::{FlowTransforms, PointsSoA, TrackInput};

use nalgebra::Vector2;

use crate::pyramid::Pyramid;
use kornia_staging_imgproc::optical_flow::patch_se2::AffineCompact2f;
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;

/// Upper bound for a valid increment.
///
/// `pub(crate)` so the GPU lane's kernels alias it rather than re-declaring the
/// number: the two lanes have no compiler coupling otherwise.
pub(crate) const MAX_INCREMENT_INFINITY_NORM: f32 = 1e6;

/// `const int filter_margin = 2`.
///
/// `pub(crate)` for the same reason as [`MAX_INCREMENT_INFINITY_NORM`].
pub(crate) const FILTER_MARGIN: f32 = 2.0;

/// Maximum tracker capacity, bounding caller-controlled preallocation.
/// A million keypoints already implies roughly 7 GB across two patch sets with
/// four Pattern51 levels, far above the shipped grid's needs. Rejecting larger
/// requests prevents capacity arithmetic overflow at the public boundary.
pub const MAX_CAPACITY: usize = 1 << 20;

/// Maximum pyramid level count, limiting the capacity multiplier.
/// Each level halves image dimensions. Unbounded counts could request storage
/// large enough to abort allocation instead of returning an input error.
pub const MAX_LEVELS: usize = 24;

/// What the tracker can refuse.
///
/// Every public entry point in this module validates its inputs and returns one
/// of these rather than indexing past the end of a buffer: a panic on a rayon
/// worker inside the released-GIL region aborts the process (decision D32).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TrackerError {
    /// Invalid sampling pattern geometry.
    #[error(transparent)]
    Patch(#[from] kornia_staging_imgproc::optical_flow::patch_se2::PatchError),
    /// A convergence threshold must be finite and positive.
    #[error("port.klt_exit_step_px must be finite and positive, or null")]
    InvalidExitStep,
    /// More keypoints were offered than the preallocated buffers hold.
    #[error("{offered} keypoints do not fit the tracker's capacity of {capacity}")]
    CapacityExceeded {
        /// Keypoints offered.
        offered: usize,
        /// Slots the tracker was built for.
        capacity: usize,
    },
    /// Two of the inputs disagree on how many keypoints there are.
    #[error("{first} {first_name} against {second} {second_name}")]
    LengthMismatch {
        /// What the first input is.
        first_name: &'static str,
        /// How many it holds.
        first: usize,
        /// What the second input is.
        second_name: &'static str,
        /// How many it holds.
        second: usize,
    },
    /// A tracker was asked for more keypoints than [`MAX_CAPACITY`].
    #[error("a capacity of {capacity} keypoints is over the ceiling of {ceiling}")]
    CapacityTooLarge {
        /// Keypoints asked for.
        capacity: usize,
        /// [`MAX_CAPACITY`].
        ceiling: usize,
    },
    /// A GPU backend refused to come up.
    ///
    /// Carried here rather than returned separately because
    /// `FrameToFrameOpticalFlow::with_stages` takes an already-built tracker,
    /// so the construction of a device backend has one error path (decision D32).
    #[cfg(feature = "gpu-core")]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
    /// More passes were put in flight at once than the tracker has result slots.
    #[error("{submitted} tracking passes in flight against {lanes} result slots")]
    TooManyPasses {
        /// Passes submitted since the last collect, this one included.
        submitted: usize,
        /// Slots the tracker was built for, which is the rig's camera count.
        lanes: usize,
    },
    /// A tracker was asked for more pyramid levels than [`MAX_LEVELS`].
    #[error("a patch buffer over {num_levels} pyramid levels is over the ceiling of {ceiling}")]
    TooManyLevels {
        /// Levels asked for, which is `optical_flow_levels + 1`.
        num_levels: usize,
        /// [`MAX_LEVELS`].
        ceiling: usize,
    },
    /// A buffer shape does not fit in a `usize`.
    ///
    /// Checked product by product rather than after the fact: a wrapped
    /// multiplication would have turned an impossible shape into a plausible
    /// allocation. [`MAX_CAPACITY`] and [`MAX_LEVELS`] together bound the largest
    /// product at 24 x 2^20 x 52 x 3 = 3,925,868,544 elements, which is 91% of
    /// `u32::MAX` — inside a 32-bit `usize`, but not by much, so this cannot fire
    /// today and would as soon as either ceiling rose. The test
    /// `the_ceilings_bound_every_buffer_product` is the arithmetic that says so,
    /// and this variant is what keeps raising a ceiling from silently
    /// reintroducing a wrapped allocation.
    #[error(
        "a {capacity}-patch buffer over {num_levels} levels of {taps} taps does not fit in a usize"
    )]
    BufferShapeOverflow {
        /// Patches the buffer is sized for.
        capacity: usize,
        /// Pyramid levels it is sized for.
        num_levels: usize,
        /// Pattern taps per patch and level.
        taps: usize,
    },
    /// A pyramid or patch set does not carry the levels the tracker was built for.
    #[error("{what} holds {actual} levels, the tracker needs {expected}")]
    LevelMismatch {
        /// What was too shallow.
        what: &'static str,
        /// Levels the tracker runs over.
        expected: usize,
        /// Levels the input holds.
        actual: usize,
    },
}

/// The four preconditions of [`PatchTracker::track`], checked before any
/// mutation.
///
/// Both lanes call this rather than each spelling the four out: the seam exists
/// to keep them interchangeable, and a fifth check added to one lane and not the
/// other would be invisible.
///
/// # Errors
///
/// [`TrackerError::LengthMismatch`] when the patch set and the guesses disagree,
/// [`TrackerError::CapacityExceeded`] above the tracker's capacity, and
/// [`TrackerError::LevelMismatch`] when the patch set or either pyramid is
/// shallower than the tracker. The patch set is built by the caller, so its
/// depth is an input like any other: a one-level `PatchSoA` in a two-level
/// tracker used to index past the end of `valid`.
pub(crate) fn check_track_inputs(
    count: usize,
    patches_len: usize,
    patch_levels: usize,
    prev_levels: usize,
    next_levels: usize,
    capacity: usize,
    num_levels: usize,
) -> Result<(), TrackerError> {
    if count != patches_len {
        return Err(TrackerError::LengthMismatch {
            first_name: "patches",
            first: patches_len,
            second_name: "transforms",
            second: count,
        });
    }
    if count > capacity {
        return Err(TrackerError::CapacityExceeded {
            offered: count,
            capacity,
        });
    }
    if patch_levels < num_levels {
        return Err(TrackerError::LevelMismatch {
            what: "the patch set",
            expected: num_levels,
            actual: patch_levels,
        });
    }
    for (what, levels) in [
        ("the previous pyramid", prev_levels),
        ("the next pyramid", next_levels),
    ] {
        if levels < num_levels {
            return Err(TrackerError::LevelMismatch {
                what,
                expected: num_levels,
                actual: levels,
            });
        }
    }
    Ok(())
}

/// The element counts a patch set of this shape needs: `(flags, taps)`.
///
/// `flags` is one entry per (level, patch) and `taps` is `flags * P::SIZE`; each
/// constructor forms its own last product from them, which is the part the two
/// lanes do differently (the CPU one wants three Jacobian arrays, the GPU one
/// stores point coordinates and samples templates in registers). The ceilings and the
/// `checked_mul` ladder are the part that must not drift.
///
/// # Errors
///
/// [`TrackerError::CapacityTooLarge`] above [`MAX_CAPACITY`],
/// [`TrackerError::TooManyLevels`] above [`MAX_LEVELS`], and
/// [`TrackerError::BufferShapeOverflow`] when a count does not fit a `usize`.
pub(crate) fn checked_patch_shape(
    capacity: usize,
    num_levels: usize,
    taps_per_patch: usize,
) -> Result<(usize, usize), TrackerError> {
    if capacity > MAX_CAPACITY {
        return Err(TrackerError::CapacityTooLarge {
            capacity,
            ceiling: MAX_CAPACITY,
        });
    }
    if num_levels > MAX_LEVELS {
        return Err(TrackerError::TooManyLevels {
            num_levels,
            ceiling: MAX_LEVELS,
        });
    }
    let overflow = || TrackerError::BufferShapeOverflow {
        capacity,
        num_levels,
        taps: taps_per_patch,
    };
    let flags: usize = num_levels.checked_mul(capacity).ok_or_else(overflow)?;
    let taps: usize = flags.checked_mul(taps_per_patch).ok_or_else(overflow)?;
    Ok((flags, taps))
}

/// The three preconditions of [`SourcePatches::build`], checked before any
/// mutation, on both lanes for the same reason as [`check_track_inputs`].
///
/// # Errors
///
/// [`TrackerError::CapacityExceeded`] when the positions do not fit,
/// [`TrackerError::LengthMismatch`] when the selection mask is shorter than the
/// positions, and [`TrackerError::LevelMismatch`] when the pyramid is shallower
/// than the patch set.
pub(crate) fn check_patch_inputs(
    count: usize,
    capacity: usize,
    selected: Option<&[bool]>,
    pyramid_levels: usize,
    num_levels: usize,
) -> Result<(), TrackerError> {
    if count > capacity {
        return Err(TrackerError::CapacityExceeded {
            offered: count,
            capacity,
        });
    }
    if let Some(flags) = selected
        && flags.len() < count
    {
        return Err(TrackerError::LengthMismatch {
            first_name: "positions",
            first: count,
            second_name: "selection flags",
            second: flags.len(),
        });
    }
    if pyramid_levels < num_levels {
        return Err(TrackerError::LevelMismatch {
            what: "the pyramid",
            expected: num_levels,
            actual: pyramid_levels,
        });
    }
    Ok(())
}

/// The source patches of one camera, whatever holds them.
///
/// Split from [`PatchTracker`] so a backend can pair its own patch storage with
/// its own pyramid: `build` is the "sample every patch at every level" stage the
/// GPU wants as one kernel, and the tracker consumes the result.
#[allow(clippy::len_without_is_empty)]
pub trait SourcePatches {
    /// The pyramid representation these patches are sampled from.
    type Pyramid: Pyramid;

    /// Build one patch per entry of `positions`, at every level.
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
    ) -> Result<(), TrackerError>;

    /// Patches currently filled.
    fn len(&self) -> usize;

    /// The level-0 source position of one patch.
    ///
    /// # Panics
    ///
    /// If `patch` is past the end.
    fn position(&self, patch: usize) -> Vector2<f32>;
}

/// What one call to [`PatchTracker::track`] produced.
///
/// Dense per-input arrays plus a compacted list of the inputs that survived, all
/// preallocated and all structure-of-arrays: there is no map keyed by keypoint id
/// anywhere on this path, and no `push` inside the tracking loop (§12.2, §12.3).
#[derive(Debug, Default, PartialEq)]
pub struct FlowResult {
    valid: Vec<bool>,
    transforms: FlowTransforms,
    tracked: Vec<u32>,
}

/// `Clone` by hand, for the `clone_from` reason on [`PointsSoA`].
impl Clone for FlowResult {
    fn clone(&self) -> Self {
        Self {
            valid: self.valid.clone(),
            transforms: self.transforms.clone(),
            tracked: self.tracked.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.valid.clone_from(&source.valid);
        self.transforms.clone_from(&source.transforms);
        self.tracked.clone_from(&source.tracked);
    }
}

impl FlowResult {
    /// Room for `capacity` inputs.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut transforms: FlowTransforms = FlowTransforms::with_capacity(capacity);
        transforms.resize(capacity);
        Self {
            valid: vec![false; capacity],
            transforms,
            tracked: Vec::with_capacity(capacity),
        }
    }

    /// Whether input `index` was tracked.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn is_valid(&self, index: usize) -> bool {
        self.valid[index]
    }

    /// The tracked warp of input `index`; meaningless unless [`FlowResult::is_valid`].
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn transform(&self, index: usize) -> AffineCompact2f {
        self.transforms.get(index)
    }

    /// The input indices that survived, ascending.
    pub fn tracked(&self) -> &[u32] {
        &self.tracked
    }

    /// Size this result for `len` inputs and mark them all untracked.
    ///
    /// The first call of the write sequence a [`PatchTracker`] uses to publish:
    /// `reset`, then [`FlowResult::set_track`] or
    /// [`FlowResult::parts_mut`] per input, then [`FlowResult::finish`]. The
    /// allocation is kept, so a backend that resets every frame never allocates.
    pub fn reset(&mut self, len: usize) {
        if self.valid.len() < len {
            self.valid.resize(len, false);
        }
        if self.transforms.len() < len {
            self.transforms.resize(len);
        }
        self.valid[..len].fill(false);
        self.tracked.clear();
    }

    /// Publish one input's outcome.
    ///
    /// # Panics
    ///
    /// If `index` is past the length [`FlowResult::reset`] was given.
    pub fn set_track(&mut self, index: usize, valid: bool, transform: &AffineCompact2f) {
        self.valid[index] = valid;
        self.transforms.set(index, transform);
    }

    /// The validity flags and the warps, mutably, for a backend that writes them
    /// in bulk rather than one at a time.
    ///
    /// Pair it with [`FlowTransforms::coefficients_mut`] and
    /// [`crate::frontend::parallel::WorkPool::for_each_warp`]; [`CpuPatchTracker`]
    /// publishes through exactly this, so the seam is exercised by the shipped
    /// backend and not only by a test.
    pub fn parts_mut(&mut self) -> (&mut [bool], &mut FlowTransforms) {
        (&mut self.valid, &mut self.transforms)
    }

    /// Rebuild the compacted survivor list from the validity flags.
    ///
    /// The last call of the write sequence. `len` is what
    /// [`FlowResult::reset`] was given; entries past it are ignored.
    pub fn finish(&mut self, len: usize) {
        self.tracked.clear();
        for index in 0..len.min(self.valid.len()) {
            if self.valid[index] {
                self.tracked.push(index as u32);
            }
        }
    }

    /// How many inputs survived.
    pub fn len(&self) -> usize {
        self.tracked.len()
    }

    /// Whether nothing survived.
    pub fn is_empty(&self) -> bool {
        self.tracked.is_empty()
    }
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

    pub(crate) fn slot_mut(&mut self, lane: usize, capacity: usize) -> &mut FlowResult {
        if lane == self.slots.len() {
            self.slots.push(FlowResult::with_capacity(capacity));
        }
        &mut self.slots[lane]
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
/// CubeCL one can be swapped without touching the driver (§12.1).
pub trait PatchTracker {
    /// Set the per-level convergence threshold validated by the frontend.
    fn set_klt_exit_step_px(&mut self, threshold: Option<f32>);

    /// Reusable result storage owned by this backend.
    fn batch(&self) -> &TrackBatch;
    /// Mutable result storage used by the synchronous submission default.
    fn batch_mut(&mut self) -> &mut TrackBatch;

    /// Submit a pass and return its result slot. CPU backends fill it now;
    /// device backends fill the same slot at collection.
    ///
    /// # Errors
    /// As [`PatchTracker::track`], plus a backend's pass capacity limit.
    fn submit(
        &mut self,
        prev: &Self::Pyramid,
        next: &Self::Pyramid,
        patches: &Self::Patches,
        transforms_in: &FlowTransforms,
    ) -> Result<usize, TrackerError>;

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
        inputs: &mut [TrackInput],
        patches: &mut Self::Patches,
        temporal: bool,
    ) -> Result<(), TrackerError> {
        submit_each(self, prev, next, inputs, patches, temporal)
    }

    /// Read a result by the slot returned from submission, after collection.
    fn result(&self, pass: usize) -> &FlowResult {
        self.batch().result(pass)
    }

    /// Finish every submitted pass, retaining results in the batch's slots.
    ///
    /// # Errors
    /// A device download can fail.
    fn collect(&mut self) -> Result<(), TrackerError> {
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
    type Pyramid: Pyramid;

    /// The source-patch storage it consumes.
    type Patches: SourcePatches<Pyramid = Self::Pyramid>;

    /// Track source patches from `prev` to `next`.
    /// `transforms_in` supplies source linear parts and guessed translations;
    /// source positions come from `patches`.
    ///
    /// # Errors
    /// [`TrackerError`] if inputs do not match the allocated tracker geometry.
    fn track(
        &mut self,
        prev: &Self::Pyramid,
        next: &Self::Pyramid,
        patches: &Self::Patches,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        let pass = self.submit(prev, next, patches, transforms_in)?;
        self.collect()?;
        out.clone_from(self.result(pass));
        Ok(())
    }

    /// Keypoints this tracker can carry in one call.
    fn capacity(&self) -> usize;

    /// Pyramid levels it runs over, which is `optical_flow_levels + 1`.
    fn num_levels(&self) -> usize;

    /// Fresh source-patch storage matching this tracker's capacity and depth.
    ///
    /// The driver cannot name the concrete type, so the tracker makes it.
    ///
    /// # Errors
    ///
    /// [`TrackerError`] when the storage cannot be sized — the same shape checks
    /// the tracker's own constructor made.
    fn make_patches(&self) -> Result<Self::Patches, TrackerError>;
}

/// Default camera-order submission, also used by the GPU's mixed-arena fallback.
pub(crate) fn submit_each<T: PatchTracker + ?Sized>(
    tracker: &mut T,
    prev: &[T::Pyramid],
    next: &[T::Pyramid],
    inputs: &mut [TrackInput],
    patches: &mut T::Patches,
    temporal: bool,
) -> Result<(), TrackerError> {
    for (index, input) in inputs.iter_mut().enumerate() {
        if temporal || index == 0 {
            patches.build(&prev[input.source], &input.positions, None)?;
        }
        input.result = tracker.submit(
            &prev[input.source],
            &next[input.destination],
            patches,
            &input.guesses,
        )?;
    }
    Ok(())
}
