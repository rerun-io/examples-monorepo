//! Stateless forward/backward pyramidal SE(2) tracking with reusable scratch.
//!
//! ```
//! use kornia_image::{Image, ImageSize};
//! use kornia_staging_imgproc::{pyramid::PyramidPlanU16, optical_flow::{patch_se2::{Pattern51, AffineCompact2f}, patch_tracker::*}};
//! let image = Image::from_size_val(ImageSize { width: 32, height: 32 }, 0u16)?;
//! let mut pyramid = PyramidPlanU16::new(image.size(), 0)?;
//! pyramid.run(&image)?;
//! let mut points = PointsSoA::default();
//! points.push([16.0, 16.0]);
//! let mut patches = PatchSoA::<Pattern51>::new(1, 1)?;
//! patches.build(&pyramid, &points, None)?;
//! let mut guesses = FlowTransforms::default();
//! guesses.push(&AffineCompact2f::at([16.0, 16.0]));
//! let mut result = FlowResult::default();
//! let mut plan = PatchTrackerPlan::<Pattern51>::new(1, 1, 5, 0.04, None)?;
//! plan.track(&pyramid, &pyramid, &patches, &guesses, &mut result)?;
//! assert!(result.is_empty());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
mod cpu;
mod patch_soa;
mod storage;
use crate::optical_flow::patch_se2::AffineCompact2f;
pub use cpu::{PatchTrackerPlan, TrackingSteps, TrackingTarget};
pub use patch_soa::PatchSoA;
pub use storage::{FlowTransforms, PointsSoA};
/// What the tracker can refuse.
///
/// Checked constructors and tracking passes validate inputs before mutation.
/// Unchecked per-point kernels require those validated shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TrackerError {
    /// A phase was submitted before the previous phase was collected or discarded.
    #[error("collect or discard the previous tracking phase before submitting another")]
    PendingBatch,
    /// The phase names a camera without a pyramid.
    #[error("{role} camera {camera} has no pyramid")]
    MissingCamera {
        /// Whether the missing pyramid is a source or destination.
        role: &'static str,
        /// Requested camera index.
        camera: usize,
    },

    /// A tracker parameter is outside its supported range.
    #[error("invalid tracker parameter: {0}")]
    InvalidParameter(&'static str),
    /// Invalid sampling pattern geometry.
    #[error(transparent)]
    Patch(#[from] crate::optical_flow::patch_se2::PatchError),
    /// A convergence threshold must be finite and positive.
    #[error("convergence threshold must be finite and positive, or absent")]
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
        /// Requested level count, including the full-resolution level.
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

/// What one call to [`PatchTrackerPlan::track`] produced.
///
/// Dense per-input arrays plus a compacted list of the inputs that survived, all
/// preallocated and all structure-of-arrays: there is no map keyed by keypoint id
/// anywhere on this path, and no `push` inside the tracking loop.
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
        let transforms = FlowTransforms::with_capacity(capacity);
        Self {
            valid: Vec::with_capacity(capacity),
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
    #[inline]
    pub fn transform(&self, index: usize) -> AffineCompact2f {
        self.transforms.get(index)
    }

    /// The input indices that survived, ascending.
    pub fn tracked(&self) -> &[u32] {
        &self.tracked
    }

    /// Size this result for `len` inputs and mark them all untracked.
    ///
    /// The first call of the write sequence a tracking pass uses to publish:
    /// `reset`, then [`FlowResult::set_track`] or
    /// [`FlowResult::parts_mut`] per input, then [`FlowResult::finish`]. The
    /// allocation is kept, so a backend that resets every frame never allocates.
    pub fn reset(&mut self, len: usize) {
        self.valid.resize(len, false);
        self.transforms.resize(len);
        self.valid.fill(false);
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
    /// Each input slot may be written independently.
    pub fn parts_mut(&mut self) -> (&mut [bool], [&mut [f32]; 6]) {
        (&mut self.valid, self.transforms.coefficients_mut())
    }

    /// Fill all active slots using the caller's worker pool.
    /// Each slot is evaluated once; the output view cannot resize retained storage.
    pub fn fill_with<S>(
        &mut self,
        pool: Option<&rayon::ThreadPool>,
        init: impl Fn() -> S + Sync + Send,
        body: impl Fn(&mut S, usize) -> ([f32; 6], bool) + Sync + Send,
    ) {
        self.transforms
            .fill_with(pool, self.valid.len(), &mut self.valid, init, body);
    }

    /// Clear the active inputs and survivors, retaining all allocations.
    pub fn clear(&mut self) {
        self.reset(0);
    }

    /// Rebuild the compacted survivor list from the validity flags.
    ///
    /// The last call of the write sequence, using the extent established by reset.
    pub fn finish(&mut self) {
        self.tracked.clear();
        for index in 0..self.valid.len() {
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

/// Shared CPU/device validation and limits.
pub mod limits;
pub(crate) use limits::*;

#[cfg(test)]
mod result_tests {
    use super::*;

    #[test]
    fn result_extent_shrinks_clears_and_regrows_without_stale_tracks() {
        let mut result = FlowResult::with_capacity(8);
        result.reset(8);
        result.set_track(7, true, &AffineCompact2f::at([7.0, 8.0]));
        result.finish();
        assert_eq!(result.tracked(), &[7]);
        result.reset(2);
        assert_eq!(result.parts_mut().0.len(), 2);
        result.set_track(1, true, &AffineCompact2f::at([1.0, 2.0]));
        result.finish();
        assert_eq!(result.tracked(), &[1]);
        assert_eq!(result.transform(1).translation, [1.0, 2.0]);
        result.clear();
        assert!(result.parts_mut().0.is_empty());
        result.reset(8);
        result.finish();
        assert!(result.is_empty());
        assert!(result.valid.capacity() >= 8);
        assert!(result.tracked.capacity() >= 8);
    }
}
