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

mod patch_soa;
mod storage;
pub use patch_soa::PatchSoA;
pub use storage::{FlowTransforms, PointsSoA, TrackInput};

use nalgebra::{Matrix2, Vector2, Vector3};

use crate::frontend::parallel::WorkPool;
use crate::frontend::patch::{patch_increment_rows, patch_residual_taps};
use crate::frontend::patterns::{MAX_PATTERN_SIZE, Pattern};
use crate::frontend::se2::{AffineCompact2f, se2_exp};
use crate::image::ImageU16;
use crate::pyramid::{Pyramid, PyramidU16};

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
    /// `FrameToFrameOpticalFlow::with_backends` takes an already-built tracker,
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
/// folds `4 * taps + flags` into a single buffer). The ceilings and the
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
    /// Prepare source storage before tracker inputs are uploaded. A synchronous
    /// backend builds immediately; a device tracker may defer the patch kernel.
    fn prepare(
        &mut self,
        pyramid: &Self::Pyramid,
        positions: &PointsSoA,
        selected: Option<&[bool]>,
    ) -> Result<(), TrackerError> {
        self.build(pyramid, positions, selected)
    }

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
        if pass == self.slots.len() {
            self.slots.push(FlowResult::with_capacity(capacity));
        }
        self.submitted += 1;
        (pass, &mut self.slots[pass])
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
    /// Track source patches prepared by [`SourcePatches::prepare`].
    /// Device implementations can upload all inputs before launching the source
    /// patch kernel; synchronous implementations keep their ordinary path.
    fn track_prepared(
        &mut self,
        prev: &Self::Pyramid,
        next: &Self::Pyramid,
        patches: &Self::Patches,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        self.track(prev, next, patches, transforms_in, out)
    }

    /// Reusable result storage owned by this backend.
    fn batch(&self) -> &TrackBatch;
    /// Mutable result storage used by the synchronous submission default.
    fn batch_mut(&mut self) -> &mut TrackBatch;

    /// Submit a pass and return its result slot. CPU backends fill it now;
    /// device backends fill the same slot at collection.
    ///
    /// # Errors
    /// As [`PatchTracker::track`], plus a backend's pass capacity limit.
    fn submit_prepared(
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
        for (index, input) in inputs.iter_mut().enumerate() {
            if temporal || index == 0 {
                patches.prepare(&prev[input.source], &input.positions, None)?;
            }
            input.result = self.submit_prepared(
                &prev[input.source],
                &next[input.destination],
                patches,
                &input.guesses,
            )?;
        }
        Ok(())
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
        let pass = self.submit_prepared(prev, next, patches, transforms_in)?;
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

/// The CPU tracker: `trackPoints` with the same arithmetic and a fixed thread budget.
///
/// Template reuse follows [`PatchTracker::submit_batch`]'s caller rules: carry
/// committed pyramids forward and discard failed or abandoned batches. Cache
/// hits require the id, position bits, and source pyramid build generation.
#[derive(Debug)]
pub struct CpuPatchTracker<P: Pattern> {
    capacity: usize,
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
    pool: WorkPool,
    /// Backward source patches, built from `next` at the forward result.
    backward: PatchSoA<P>,
    /// Forward result per input, before the backward check.
    forward: FlowTransforms,
    /// Whether the forward track succeeded, per input.
    forward_valid: Vec<bool>,
    /// The positions the backward patches are built at.
    backward_positions: PointsSoA,
    batch: TrackBatch,
    /// Temporal source templates and the last backward templates per camera.
    cameras: Vec<CpuCameraBuffers<P>>,
    temporal: Vec<CpuBackwardCache<P>>,
    /// Stereo backward templates for newly detected ids in the same frame.
    /// Those ids are disjoint from the temporal cache's surviving ids.
    stereo: Vec<CpuBackwardCache<P>>,
    offsets: Vec<usize>,
    batch_result: FlowResult,
}

#[derive(Debug)]
struct CpuBackwardCache<P: Pattern> {
    patches: PatchSoA<P>,
    generation: crate::pyramid::PyramidGeneration,
    ids: Vec<crate::types::KeypointId>,
    positions: PointsSoA,
    valid: Vec<bool>,
}

#[derive(Debug)]
struct CpuCameraBuffers<P: Pattern> {
    source: PatchSoA<P>,
    build: Vec<bool>,
    temporal_columns: Vec<(usize, usize)>,
    stereo_columns: Vec<(usize, usize)>,
}

impl<P: Pattern> CpuPatchTracker<P> {
    /// A tracker sized for `capacity` keypoints over `num_levels` pyramid levels.
    ///
    /// `max_iterations` is `optical_flow_max_iterations`,
    /// `max_recovered_dist2` is `optical_flow_max_recovered_dist2`, and `pool`
    /// carries the explicit thread budget (decision D31).
    ///
    /// # Errors
    ///
    /// As [`PatchSoA::new`]: the capacity is checked against [`MAX_CAPACITY`] and
    /// every buffer product against the `usize` range before anything is
    /// allocated.
    pub fn new(
        capacity: usize,
        num_levels: usize,
        max_iterations: usize,
        max_recovered_dist2: f32,
        pool: WorkPool,
    ) -> Result<Self, TrackerError> {
        // First, so the smaller buffers below cannot be sized from a capacity the
        // patch storage would have refused.
        let backward: PatchSoA<P> = PatchSoA::new(capacity, num_levels)?.with_pool(pool.clone());
        let mut forward: FlowTransforms = FlowTransforms::with_capacity(capacity);
        forward.resize(capacity);
        let mut backward_positions: PointsSoA = PointsSoA::with_capacity(capacity);
        backward_positions.resize(capacity);
        Ok(Self {
            capacity,
            num_levels,
            max_iterations,
            max_recovered_dist2,
            pool,
            backward,
            forward,
            forward_valid: vec![false; capacity],
            backward_positions,
            batch: TrackBatch::default(),
            cameras: Vec::new(),
            temporal: Vec::new(),
            stereo: Vec::new(),
            offsets: Vec::new(),
            batch_result: FlowResult::default(),
        })
    }

    /// Workers the tracking passes run on.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    fn steps(&self) -> TrackingSteps {
        TrackingSteps {
            num_levels: self.num_levels,
            max_iterations: self.max_iterations,
            max_recovered_dist2: self.max_recovered_dist2,
        }
    }
}

impl<P: Pattern> PatchTracker for CpuPatchTracker<P> {
    fn batch(&self) -> &TrackBatch {
        &self.batch
    }
    fn batch_mut(&mut self) -> &mut TrackBatch {
        &mut self.batch
    }

    type Pattern = P;
    type Pyramid = PyramidU16;
    type Patches = PatchSoA<P>;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn make_patches(&self) -> Result<PatchSoA<P>, TrackerError> {
        Ok(PatchSoA::new(self.capacity, self.num_levels)?.with_pool(self.pool.clone()))
    }

    fn submit_prepared(
        &mut self,
        prev: &Self::Pyramid,
        next: &Self::Pyramid,
        patches: &Self::Patches,
        transforms_in: &FlowTransforms,
    ) -> Result<usize, TrackerError> {
        let capacity = self.capacity();
        let batch = self.batch_mut();
        let pass = batch.submitted;
        if pass == batch.slots.len() {
            batch.slots.push(FlowResult::with_capacity(capacity));
        }
        let mut result = std::mem::take(&mut batch.slots[pass]);
        let outcome = self.track_into(prev, next, patches, transforms_in, &mut result);
        self.batch_mut().slots[pass] = result;
        outcome?;
        self.batch_mut().submitted += 1;
        Ok(pass)
    }

    fn discard(&mut self) {
        self.batch.submitted = 0;
        // A refused frame may have replaced some backward stores. Rebuild from
        // the committed pyramids on retry instead of reusing uncommitted data.
        for camera in self.temporal.iter_mut().chain(&mut self.stereo) {
            camera.ids.clear();
        }
    }

    fn submit_batch(
        &mut self,
        prev: &[PyramidU16],
        next: &[PyramidU16],
        inputs: &mut [TrackInput],
        patches: &mut PatchSoA<P>,
        temporal: bool,
    ) -> Result<(), TrackerError> {
        if inputs.is_empty() {
            return Ok(());
        }
        // Validate the whole batch before modifying either cache. Source and
        // destination indices come from the driver, but are checked at this
        // public boundary just like the ordinary single-pass inputs.
        for (camera, input) in inputs.iter().enumerate() {
            let destination = camera + usize::from(!temporal);
            let source = if temporal { camera } else { 0 };
            for (actual, expected, name) in [
                (input.source, source, "source camera"),
                (input.destination, destination, "destination camera"),
            ] {
                if actual != expected {
                    return Err(TrackerError::LengthMismatch {
                        first_name: name,
                        first: actual,
                        second_name: "camera in batch order",
                        second: expected,
                    });
                }
            }
            for (index, len, name) in [
                (input.source, prev.len(), "source camera"),
                (input.destination, next.len(), "destination camera"),
            ] {
                if index >= len {
                    return Err(TrackerError::LengthMismatch {
                        first_name: name,
                        first: index,
                        second_name: "cameras",
                        second: len,
                    });
                }
            }
            check_track_inputs(
                input.guesses.len(),
                input.positions.len(),
                self.num_levels,
                prev[input.source].num_levels(),
                next[input.destination].num_levels(),
                self.capacity,
                self.num_levels,
            )?;
            if input.ids.len() != input.positions.len() {
                return Err(TrackerError::LengthMismatch {
                    first_name: "keypoint ids",
                    first: input.ids.len(),
                    second_name: "positions",
                    second: input.positions.len(),
                });
            }
            if !temporal {
                if input.positions.len() != inputs[0].positions.len() {
                    return Err(TrackerError::LengthMismatch {
                        first_name: "patches",
                        first: inputs[0].positions.len(),
                        second_name: "transforms",
                        second: input.guesses.len(),
                    });
                }
                debug_assert_eq!(input.ids, inputs[0].ids);
                debug_assert_eq!(input.positions, inputs[0].positions);
            }
        }
        while self.cameras.len() < next.len() {
            self.cameras.push(CpuCameraBuffers {
                source: self.make_patches()?,
                build: Vec::new(),
                temporal_columns: Vec::new(),
                stereo_columns: Vec::new(),
            });
            self.temporal.push(CpuBackwardCache {
                patches: self.make_patches()?,
                generation: crate::pyramid::PyramidGeneration::default(),
                ids: Vec::new(),
                positions: PointsSoA::default(),
                valid: Vec::new(),
            });
            self.stereo.push(CpuBackwardCache {
                patches: self.make_patches()?,
                generation: crate::pyramid::PyramidGeneration::default(),
                ids: Vec::new(),
                positions: PointsSoA::default(),
                valid: Vec::new(),
            });
        }
        self.offsets.clear();
        self.offsets.push(0);
        let mut total = 0;
        for input in inputs.iter() {
            total += input.guesses.len();
            self.offsets.push(total);
        }
        self.forward.resize(total);
        self.forward_valid.resize(total, false);
        self.batch_result.reset(total);

        // Forward builds: copy surviving columns from the last frame, and
        // sample detections with no matching template. The cache key is id,
        // position bits, and the committed source pyramid's build generation.
        if temporal {
            let prepare = |(camera, buffers): (usize, &mut CpuCameraBuffers<P>)| {
                let input = &inputs[camera];
                let temporal = &self.temporal[input.destination];
                let stereo = &self.stereo[input.destination];
                buffers.build.clear();
                buffers.temporal_columns.clear();
                buffers.stereo_columns.clear();
                for (slot, id) in input.ids.iter().enumerate() {
                    let position = input.positions.get(slot);
                    let cached = |cache: &CpuBackwardCache<P>| {
                        if cache.generation != prev[input.source].generation() {
                            return None;
                        }
                        cache.ids.binary_search(id).ok().filter(|&column| {
                            let old = cache.patches.position(column);
                            cache.valid[column]
                                && old.x.to_bits() == position.x.to_bits()
                                && old.y.to_bits() == position.y.to_bits()
                        })
                    };
                    if let Some(column) = cached(temporal) {
                        buffers.temporal_columns.push((slot, column));
                        buffers.build.push(false);
                    } else if let Some(column) = cached(stereo) {
                        buffers.stereo_columns.push((slot, column));
                        buffers.build.push(false);
                    } else {
                        buffers.build.push(true);
                    }
                }
                buffers.source.build(
                    &prev[input.source],
                    &input.positions,
                    Some(&buffers.build),
                )?;
                buffers
                    .source
                    .copy_columns_from(&temporal.patches, &buffers.temporal_columns)?;
                buffers
                    .source
                    .copy_columns_from(&stereo.patches, &buffers.stereo_columns)
            };
            if let Some(result) = self.pool.install(|| {
                use rayon::prelude::*;
                self.cameras[..inputs.len()]
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(prepare)
            }) {
                result?;
            } else {
                self.cameras[..inputs.len()]
                    .iter_mut()
                    .enumerate()
                    .try_for_each(prepare)?;
            }
            // Stereo caches are valid for exactly the next temporal call.
            // A frame that skips detection must never leave old templates live.
            for cache in &mut self.stereo {
                cache.ids.clear();
            }
        } else {
            patches.prepare(&prev[inputs[0].source], &inputs[0].positions, None)?;
        }

        let steps = self.steps();
        let offsets = &self.offsets;
        let cameras = &self.cameras;
        let source = |camera: usize| {
            if temporal {
                &cameras[camera].source
            } else {
                &*patches
            }
        };
        // One point range across all cameras. Empty lanes have equal offsets;
        // partition_point skips them without scheduling empty work.
        self.pool.for_each_warp(
            self.forward.coefficients_prefix_mut(total),
            &mut self.forward_valid[..total],
            |slot| {
                let camera = offsets.partition_point(|&start| start <= slot) - 1;
                let index = slot - offsets[camera];
                let input = &inputs[camera];
                steps.forward_slot(
                    source(camera),
                    index,
                    &next[input.destination],
                    input.guesses.get(index),
                )
            },
        );

        // Build all backward stores before the backward KLT region starts.
        let forward = &self.forward;
        let forward_valid = &self.forward_valid;
        let build_backward = |camera: usize, cache: &mut CpuBackwardCache<P>| {
            let input = &inputs[camera];
            let start = offsets[camera];
            let count = input.guesses.len();
            cache.positions.resize(count);
            for index in 0..count {
                cache
                    .positions
                    .set(index, forward.translation(start + index));
            }
            cache.patches.build(
                &next[input.destination],
                &cache.positions,
                Some(&forward_valid[start..start + count]),
            )?;
            cache.ids.clone_from(&input.ids);
            cache.generation = next[input.destination].generation();
            Ok::<(), TrackerError>(())
        };
        let caches = if temporal {
            &mut self.temporal[..inputs.len()]
        } else {
            &mut self.stereo[1..1 + inputs.len()]
        };
        if let Some(result) = self.pool.install(|| {
            use rayon::prelude::*;
            caches
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(camera, cache)| build_backward(camera, cache))
        }) {
            result?;
        } else {
            for (camera, cache) in caches.iter_mut().enumerate() {
                build_backward(camera, cache)?;
            }
        }

        let (valid, transforms) = self.batch_result.parts_mut();
        self.pool.for_each_warp(
            transforms.coefficients_prefix_mut(total),
            &mut valid[..total],
            |slot| {
                let camera = offsets.partition_point(|&start| start <= slot) - 1;
                let index = slot - offsets[camera];
                let input = &inputs[camera];
                let backward = if temporal {
                    &self.temporal[camera].patches
                } else {
                    &self.stereo[input.destination].patches
                };
                steps.backward_slot(
                    backward,
                    index,
                    &prev[input.source],
                    input.positions.get(index),
                    input.guesses.translation(index),
                    (forward.get(slot), forward_valid[slot]),
                )
            },
        );
        // Publish in camera order, independent of which worker finished first.
        for (camera, input) in inputs.iter_mut().enumerate() {
            let (pass, result) = self.batch.submit_slot(self.capacity);
            input.result = pass;
            result.reset(input.guesses.len());
            let cache = if temporal {
                &mut self.temporal[camera]
            } else {
                &mut self.stereo[input.destination]
            };
            cache.valid.resize(input.guesses.len(), false);
            for index in 0..input.guesses.len() {
                let slot = offsets[camera] + index;
                let valid = self.batch_result.is_valid(slot);
                cache.valid[index] = valid;
                result.set_track(index, valid, &forward.get(slot));
            }
            result.finish(input.guesses.len());
        }
        Ok(())
    }
}

impl<P: Pattern> CpuPatchTracker<P> {
    fn track_into(
        &mut self,
        prev: &PyramidU16,
        next: &PyramidU16,
        patches: &PatchSoA<P>,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        let count: usize = transforms_in.len();
        check_track_inputs(
            count,
            patches.len(),
            patches.num_levels(),
            prev.num_levels(),
            next.num_levels(),
            self.capacity,
            self.num_levels,
        )?;

        out.reset(count);
        // A previous batch may have left these at a different live length.
        self.forward.resize(count);
        self.forward_valid.resize(count, false);

        // ── forward: `trackPoint(pyr_1, pyr_2, transform_1, transform_2)`
        let steps = self.steps();
        {
            self.pool.for_each_warp(
                self.forward.coefficients_prefix_mut(count),
                &mut self.forward_valid[..count],
                |index| steps.forward_slot(patches, index, next, transforms_in.get(index)),
            );
        }

        // ── the backward source patches, from `next` at the forward result
        {
            // `build` wants `&mut self.backward` and `&self.backward_positions`
            // at once, which is what the destructure is for; `resize(count)` is
            // what `PatchSoA::build` reads as the patch count, and it comes
            // before the writes so the next call's may be longer.
            let Self {
                backward,
                backward_positions,
                forward,
                forward_valid,
                ..
            } = self;
            backward_positions.resize(count);
            for index in 0..count {
                backward_positions.set(index, forward.translation(index));
            }
            backward.build(next, backward_positions, Some(&forward_valid[..count]))?;
        }

        // ── backward: `trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered)`
        let backward: &PatchSoA<P> = &self.backward;
        let forward: &FlowTransforms = &self.forward;
        let forward_valid: &[bool] = &self.forward_valid[..count];
        {
            let (valid, transforms) = out.parts_mut();
            self.pool.for_each_warp(
                transforms.coefficients_prefix_mut(count),
                &mut valid[..count],
                |index| {
                    steps.backward_slot(
                        backward,
                        index,
                        prev,
                        patches.position(index),
                        transforms_in.translation(index),
                        (forward.get(index), forward_valid[index]),
                    )
                },
            );
        }

        out.finish(count);
        Ok(())
    }
}

/// Per-point KLT arithmetic shared by single-pass and cached batch tracking.
struct TrackingSteps {
    num_levels: usize,
    max_iterations: usize,
    max_recovered_dist2: f32,
}

impl TrackingSteps {
    fn forward_slot<P: Pattern>(
        &self,
        patches: &PatchSoA<P>,
        index: usize,
        target: &PyramidU16,
        guess: AffineCompact2f,
    ) -> ([f32; 6], bool) {
        let (width, height) = level0_size(target);
        let position = guess.translation;
        if position.x < 0.0 || position.y < 0.0 || position.x >= width || position.y >= height {
            return (AffineCompact2f::identity().coefficients(), false);
        }
        let (tracked, ok) = track_point::<P>(
            patches,
            index,
            target,
            guess.linear,
            position,
            self.num_levels,
            self.max_iterations,
        );
        (tracked.coefficients(), ok)
    }

    fn backward_slot<P: Pattern>(
        &self,
        patches: &PatchSoA<P>,
        index: usize,
        target: &PyramidU16,
        source: Vector2<f32>,
        guess: Vector2<f32>,
        (forward, valid): (AffineCompact2f, bool),
    ) -> ([f32; 6], bool) {
        let kept = forward.coefficients();
        if !valid {
            return (kept, false);
        }
        // `off = source position - guess`; `t1_recovered += off`.
        let offset = source - guess;
        let recovered_guess = forward.translation + offset;
        let (recovered, ok) = track_point::<P>(
            patches,
            index,
            target,
            forward.linear,
            recovered_guess,
            self.num_levels,
            self.max_iterations,
        );
        if !ok {
            return (kept, false);
        }
        let dist2 = (source - recovered.translation).norm_squared();
        (kept, dist2 < self.max_recovered_dist2)
    }
}

/// Target camera's level-zero dimensions as floats, supporting mixed-resolution rigs.
fn level0_size(pyramid: &PyramidU16) -> (f32, f32) {
    // `check_track_inputs` refuses a pyramid with fewer levels than the patch
    // set, and the patch set always has at least one, so level 0 is there. The
    // `(0.0, 0.0)` would fail every keypoint's bounds test silently, so the
    // debug build says so instead (decision D32).
    debug_assert!(pyramid.level_size(0).is_some(), "no level 0 to size");
    match pyramid.level_size(0) {
        Some((width, height, _)) => (width as f32, height as f32),
        None => (0.0, 0.0),
    }
}

/// `trackPoint`.
///
/// Coarse to fine, with the translation divided by `1 << level` on the way in and
/// multiplied back on the way out — both exact in `f32`, since
/// the scale is a power of two. The linear part starts at the identity
/// and is composed with the source's at the end, so the SE(2) rotation
/// is re-estimated from scratch at every frame pair and never warm-started
/// (`papers-part2.md` §13 deviation D7).
fn track_point<P: Pattern>(
    patches: &PatchSoA<P>,
    index: usize,
    target: &PyramidU16,
    old_linear: Matrix2<f32>,
    guess: Vector2<f32>,
    num_levels: usize,
    max_iterations: usize,
) -> (AffineCompact2f, bool) {
    let mut transform: AffineCompact2f = AffineCompact2f {
        linear: Matrix2::identity(),
        translation: guess,
    };
    let mut patch_valid: bool = true;

    for level in (0..num_levels).rev() {
        if !patch_valid {
            // Idle through remaining levels after failure to keep loop bounds fixed.
            continue;
        }
        let scale: f32 = (1u32 << level) as f32;
        transform.translation /= scale;

        patch_valid &= patches.valid(level, index);
        if patch_valid && let Some(image) = target.level(level) {
            patch_valid &= track_point_at_level::<P>(
                image,
                patches,
                level,
                index,
                &mut transform,
                max_iterations,
            );
        }

        transform.translation *= scale;
    }

    transform.linear = old_linear * transform.linear;
    (transform, patch_valid)
}

/// `trackPointAtLevel`.
///
/// One Gauss-Newton step is: warp the pattern, take the mean-normalised residual,
/// `inc = -H_se2^-1 J_se2^T r`, reject a non-finite or huge increment
/// (because `SE2::exp` crashes on NaN), apply it on the right
/// (`transform *= SE2::exp(inc)`) and require the new centre to stay two
/// pixels inside the image.
fn track_point_at_level<P: Pattern>(
    image: &ImageU16,
    patches: &PatchSoA<P>,
    level: usize,
    index: usize,
    transform: &mut AffineCompact2f,
    max_iterations: usize,
) -> bool {
    let mut residual: [f32; MAX_PATTERN_SIZE] = [0.0; MAX_PATTERN_SIZE];
    let data_offset: usize = patches.data_offset(level, index);
    let jacobian_offset: usize = patches.jacobian_offset(level, index);
    let element_stride: usize = 4;
    let row_stride: usize = P::SIZE * element_stride;
    let mut patch_valid: bool = true;

    for _ in 0..max_iterations {
        if !patch_valid {
            // `for (iteration = 0; patch_valid &&...)`, as a no-op.
            continue;
        }

        patch_valid &= patch_residual_taps::<P>(
            &patches.data[data_offset..],
            element_stride,
            image,
            transform,
            &mut residual,
        );

        if patch_valid {
            let increment: Vector3<f32> = -patch_increment_rows::<P>(
                &patches.h_inv_jt[jacobian_offset..],
                element_stride,
                row_stride,
                &residual,
            );

            patch_valid &= increment.iter().all(|value| value.is_finite());
            // Fold absolute coefficients from element zero, retaining a left-hand NaN.
            let mut infinity_norm: f32 = increment[0].abs();
            for row in 1..3 {
                let candidate: f32 = increment[row].abs();
                if infinity_norm < candidate {
                    infinity_norm = candidate;
                }
            }
            patch_valid &= infinity_norm < MAX_INCREMENT_INFINITY_NORM;

            if patch_valid {
                *transform = transform.compose(&se2_exp(&increment));
                patch_valid &= image.in_bounds(
                    transform.translation.x,
                    transform.translation.y,
                    FILTER_MARGIN,
                );
            }
        }
    }

    patch_valid
}
