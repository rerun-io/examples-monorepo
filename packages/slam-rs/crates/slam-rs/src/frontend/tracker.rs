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

use nalgebra::{Matrix2, Vector2, Vector3};

use crate::frontend::parallel::WorkPool;
use crate::frontend::patch::{build_patch, patch_increment, patch_residual};
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

/// A list of 2-D points with the coordinates in two flat arrays.
///
/// `Vec<Vector2<f32>>` would give one coordinate a stride of two floats; here a
/// warp reading every patch's `x` reads consecutive addresses (§12.2 item 1).
#[derive(Debug, Default, PartialEq)]
pub struct PointsSoA {
    x: Vec<f32>,
    y: Vec<f32>,
}

/// `Clone` by hand for the sake of `clone_from`.
///
/// `#[derive(Clone)]` only writes `clone`; `clone_from` then falls back to
/// `*self = source.clone()`, which drops both buffers and allocates two more.
/// Copying field by field lets `Vec::clone_from` overwrite in place, which is
/// what makes the frontend's per-frame snapshot allocation-free (see
/// [`super::flow::FrameToFrameOpticalFlow::process_frame`]).
impl Clone for PointsSoA {
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.x.clone_from(&source.x);
        self.y.clone_from(&source.y);
    }
}

// Nothing asks a point list, a warp list or a patch set whether it is empty:
// they are sized to a capacity at construction and read by index.
#[allow(clippy::len_without_is_empty)]
impl PointsSoA {
    /// An empty list with room for `capacity` points.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
        }
    }

    /// Points held.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Drop every point, keeping the allocation.
    pub fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
    }

    /// Append one point.
    pub fn push(&mut self, point: Vector2<f32>) {
        self.x.push(point.x);
        self.y.push(point.y);
    }

    /// Overwrite point `index`.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn set(&mut self, index: usize, point: Vector2<f32>) {
        self.x[index] = point.x;
        self.y[index] = point.y;
    }

    /// Grow to `len` points, filling with the origin.
    pub fn resize(&mut self, len: usize) {
        self.x.resize(len, 0.0);
        self.y.resize(len, 0.0);
    }

    /// Point `index`.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn get(&self, index: usize) -> Vector2<f32> {
        Vector2::new(self.x[index], self.y[index])
    }

    /// Every `x` coordinate, patch index fast-varying.
    pub fn xs(&self) -> &[f32] {
        &self.x
    }

    /// Every `y` coordinate, patch index fast-varying.
    pub fn ys(&self) -> &[f32] {
        &self.y
    }
}

/// Affine warps in six flat coefficient arrays, with keypoint index varying fastest.
/// [`FlowTransforms::get`] reconstructs one warp with six loads.
#[derive(Debug, Default, PartialEq)]
pub struct FlowTransforms {
    m00: Vec<f32>,
    m01: Vec<f32>,
    m10: Vec<f32>,
    m11: Vec<f32>,
    tx: Vec<f32>,
    ty: Vec<f32>,
}

/// `Clone` by hand, for the `clone_from` reason on [`PointsSoA`].
impl Clone for FlowTransforms {
    fn clone(&self) -> Self {
        Self {
            m00: self.m00.clone(),
            m01: self.m01.clone(),
            m10: self.m10.clone(),
            m11: self.m11.clone(),
            tx: self.tx.clone(),
            ty: self.ty.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.m00.clone_from(&source.m00);
        self.m01.clone_from(&source.m01);
        self.m10.clone_from(&source.m10);
        self.m11.clone_from(&source.m11);
        self.tx.clone_from(&source.tx);
        self.ty.clone_from(&source.ty);
    }
}

#[allow(clippy::len_without_is_empty)]
impl FlowTransforms {
    /// An empty list with room for `capacity` warps.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            m00: Vec::with_capacity(capacity),
            m01: Vec::with_capacity(capacity),
            m10: Vec::with_capacity(capacity),
            m11: Vec::with_capacity(capacity),
            tx: Vec::with_capacity(capacity),
            ty: Vec::with_capacity(capacity),
        }
    }

    /// Warps held.
    pub fn len(&self) -> usize {
        self.m00.len()
    }

    /// Drop every warp, keeping the allocation.
    pub fn clear(&mut self) {
        self.m00.clear();
        self.m01.clear();
        self.m10.clear();
        self.m11.clear();
        self.tx.clear();
        self.ty.clear();
    }

    /// Grow to `len` warps, filling with the identity.
    pub fn resize(&mut self, len: usize) {
        self.m00.resize(len, 1.0);
        self.m01.resize(len, 0.0);
        self.m10.resize(len, 0.0);
        self.m11.resize(len, 1.0);
        self.tx.resize(len, 0.0);
        self.ty.resize(len, 0.0);
    }

    /// Append one warp.
    pub fn push(&mut self, warp: &AffineCompact2f) {
        self.m00.push(warp.linear[(0, 0)]);
        self.m01.push(warp.linear[(0, 1)]);
        self.m10.push(warp.linear[(1, 0)]);
        self.m11.push(warp.linear[(1, 1)]);
        self.tx.push(warp.translation.x);
        self.ty.push(warp.translation.y);
    }

    /// Insert one warp at `index`, shifting the rest up.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn insert(&mut self, index: usize, warp: &AffineCompact2f) {
        self.m00.insert(index, warp.linear[(0, 0)]);
        self.m01.insert(index, warp.linear[(0, 1)]);
        self.m10.insert(index, warp.linear[(1, 0)]);
        self.m11.insert(index, warp.linear[(1, 1)]);
        self.tx.insert(index, warp.translation.x);
        self.ty.insert(index, warp.translation.y);
    }

    /// Remove the warp at `index`, shifting the rest down.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn remove(&mut self, index: usize) {
        self.m00.remove(index);
        self.m01.remove(index);
        self.m10.remove(index);
        self.m11.remove(index);
        self.tx.remove(index);
        self.ty.remove(index);
    }

    /// Overwrite the warp at `index`.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn set(&mut self, index: usize, warp: &AffineCompact2f) {
        self.m00[index] = warp.linear[(0, 0)];
        self.m01[index] = warp.linear[(0, 1)];
        self.m10[index] = warp.linear[(1, 0)];
        self.m11[index] = warp.linear[(1, 1)];
        self.tx[index] = warp.translation.x;
        self.ty[index] = warp.translation.y;
    }

    /// The warp at `index`, reassembled.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn get(&self, index: usize) -> AffineCompact2f {
        AffineCompact2f {
            linear: Matrix2::new(
                self.m00[index],
                self.m01[index],
                self.m10[index],
                self.m11[index],
            ),
            translation: Vector2::new(self.tx[index], self.ty[index]),
        }
    }

    /// The translation at `index`, without reassembling the linear part.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn translation(&self, index: usize) -> Vector2<f32> {
        Vector2::new(self.tx[index], self.ty[index])
    }

    /// The six coefficients at `index`, in the order
    /// [`AffineCompact2f::coefficients`] uses.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn coefficients(&self, index: usize) -> [f32; 6] {
        [
            self.m00[index],
            self.m01[index],
            self.m10[index],
            self.m11[index],
            self.tx[index],
            self.ty[index],
        ]
    }

    /// The six coefficient arrays, mutably, in the order
    /// [`AffineCompact2f::coefficients`] uses.
    ///
    /// This is how a backend writes a whole camera's warps without going through
    /// one warp at a time: each array is contiguous with the patch index
    /// fast-varying, which is what a device copy and a rayon `par_chunks_mut`
    /// both want. [`crate::frontend::parallel::WorkPool::for_each_warp`] takes
    /// exactly this shape.
    pub fn coefficients_mut(&mut self) -> [&mut [f32]; 6] {
        [
            &mut self.m00,
            &mut self.m01,
            &mut self.m10,
            &mut self.m11,
            &mut self.tx,
            &mut self.ty,
        ]
    }

    /// The first `len` entries of the six coefficient arrays, mutably.
    ///
    /// The prefix [`crate::frontend::parallel::WorkPool::for_each_warp`] wants
    /// when a capacity-sized buffer is carrying `len` live warps, which is the
    /// tracker's shape on both of its passes.
    ///
    /// # Panics
    ///
    /// If `len` is past the end of the arrays.
    pub fn coefficients_prefix_mut(&mut self, len: usize) -> [&mut [f32]; 6] {
        [
            &mut self.m00[..len],
            &mut self.m01[..len],
            &mut self.m10[..len],
            &mut self.m11[..len],
            &mut self.tx[..len],
            &mut self.ty[..len],
        ]
    }

    /// Every translation `x`, patch index fast-varying.
    pub fn translations_x(&self) -> &[f32] {
        &self.tx
    }

    /// Every translation `y`, patch index fast-varying.
    pub fn translations_y(&self) -> &[f32] {
        &self.ty
    }
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

/// One camera's source patches for every pyramid level, in structure-of-arrays form.
///
/// The layout puts the **patch index fast-varying** in every array, so a GPU
/// thread per patch reads consecutive addresses (§12.2). Capacity is fixed at
/// construction and the per-frame path never allocates.
#[derive(Debug, Clone)]
pub struct PatchSoA<P: Pattern> {
    capacity: usize,
    num_levels: usize,
    len: usize,
    /// Source position at level 0, one per patch.
    positions: PointsSoA,
    /// `data[(level * P::SIZE + tap) * capacity + patch]`.
    data: Vec<f32>,
    /// `h_inv_jt[((level * 3 + row) * P::SIZE + tap) * capacity + patch]`.
    h_inv_jt: Vec<f32>,
    /// `valid[level * capacity + patch]`.
    valid: Vec<bool>,
    pattern: std::marker::PhantomData<P>,
}

impl<P: Pattern> PatchSoA<P> {
    /// Storage for `capacity` patches over `num_levels` pyramid levels.
    ///
    /// `num_levels` is `optical_flow_levels + 1`, matching
    /// [`crate::pyramid::Pyramid::num_levels`].
    ///
    /// # Errors
    ///
    /// Whatever the crate's `checked_patch_shape` refuses. All of it is checked before
    /// anything is allocated: the products below reach `Vec` as a length, and a
    /// `Vec` too long to exist panics rather than returning (decision D32).
    pub fn new(capacity: usize, num_levels: usize) -> Result<Self, TrackerError> {
        let (flags, taps): (usize, usize) = checked_patch_shape(capacity, num_levels, P::SIZE)?;
        let jacobians: usize = taps
            .checked_mul(3)
            .ok_or(TrackerError::BufferShapeOverflow {
                capacity,
                num_levels,
                taps: P::SIZE,
            })?;
        let mut positions: PointsSoA = PointsSoA::with_capacity(capacity);
        positions.resize(capacity);
        Ok(Self {
            capacity,
            num_levels,
            len: 0,
            positions,
            data: vec![0.0; taps],
            h_inv_jt: vec![0.0; jacobians],
            valid: vec![false; flags],
            pattern: std::marker::PhantomData,
        })
    }

    /// Patches this set can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Pyramid levels each patch is built at.
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Whether one patch at one level may be tracked.
    ///
    /// # Panics
    ///
    /// If `level` or `patch` is past the end.
    pub fn valid(&self, level: usize, patch: usize) -> bool {
        self.valid[level * self.capacity + patch]
    }

    /// Offset of tap 0 of one patch's `data` at one level; taps are `capacity` apart.
    #[inline]
    fn data_offset(&self, level: usize, patch: usize) -> usize {
        level * P::SIZE * self.capacity + patch
    }

    /// Offset of row 0, tap 0 of one patch's `H^-1 J^T`; rows are
    /// `P::SIZE * capacity` apart and taps `capacity` apart.
    #[inline]
    fn jacobian_offset(&self, level: usize, patch: usize) -> usize {
        level * 3 * P::SIZE * self.capacity + patch
    }
}

impl<P: Pattern> SourcePatches for PatchSoA<P> {
    type Pyramid = PyramidU16;

    /// Build every patch at every level from `pyramid`.
    ///
    /// One patch per entry of `positions`, at `position / (1 << level)` — the
    /// Source position divided by the pyramid scale.
    /// [`build_patch`] writes straight into this structure's arrays, so no packed
    /// per-patch record is ever built (§12.2).
    ///
    /// This runs on the calling thread. It is a pure per-patch map, so moving it
    /// onto [`WorkPool`] later cannot change a value; it is left sequential in V0
    /// because the patch-fast-varying layout has no chunk-contiguous mutable
    /// split, and the tracking passes are the ones the thread budget is spent on.
    fn build(
        &mut self,
        pyramid: &PyramidU16,
        positions: &PointsSoA,
        selected: Option<&[bool]>,
    ) -> Result<(), TrackerError> {
        let count: usize = positions.len();
        check_patch_inputs(
            count,
            self.capacity,
            selected,
            pyramid.num_levels(),
            self.num_levels,
        )?;
        self.len = count;

        for level in 0..self.num_levels {
            let Some(image) = pyramid.level(level) else {
                return Err(TrackerError::LevelMismatch {
                    what: "the pyramid",
                    expected: self.num_levels,
                    actual: pyramid.num_levels(),
                });
            };
            // `const Scalar scale = 1 << level`.
            let scale: f32 = (1u32 << level) as f32;
            for index in 0..count {
                if !selected.is_none_or(|flags| flags[index]) {
                    self.valid[level * self.capacity + index] = false;
                    continue;
                }
                let position: Vector2<f32> = positions.get(index) / scale;
                let data_offset: usize = self.data_offset(level, index);
                let jacobian_offset: usize = self.jacobian_offset(level, index);
                let (_mean, valid) = build_patch::<P, ImageU16>(
                    image,
                    &position,
                    &mut self.data[data_offset..],
                    self.capacity,
                    &mut self.h_inv_jt[jacobian_offset..],
                    self.capacity,
                    P::SIZE * self.capacity,
                );
                self.valid[level * self.capacity + index] = valid;
            }
        }

        for index in 0..count {
            self.positions.set(index, positions.get(index));
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.len
    }

    fn position(&self, patch: usize) -> Vector2<f32> {
        self.positions.get(patch)
    }
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
        let backward: PatchSoA<P> = PatchSoA::new(capacity, num_levels)?;
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
        })
    }

    /// Workers the tracking passes run on.
    pub fn threads(&self) -> usize {
        self.pool.threads()
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
        PatchSoA::new(self.capacity, self.num_levels)
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

        // ── forward: `trackPoint(pyr_1, pyr_2, transform_1, transform_2)`
        let max_iterations: usize = self.max_iterations;
        let num_levels: usize = self.num_levels;
        let (target_width, target_height): (f32, f32) = level0_size(next);
        {
            self.pool.for_each_warp(
                self.forward.coefficients_prefix_mut(count),
                &mut self.forward_valid[..count],
                |index| {
                    let guess: Vector2<f32> = transforms_in.translation(index);
                    // `valid = t2(0) >= 0 && t2(1) >= 0 && t2(0) < w && t2(1) < h`.
                    if guess.x < 0.0
                        || guess.y < 0.0
                        || guess.x >= target_width
                        || guess.y >= target_height
                    {
                        return (AffineCompact2f::identity().coefficients(), false);
                    }
                    let (tracked, ok) = track_point::<P>(
                        patches,
                        index,
                        next,
                        transforms_in.get(index).linear,
                        guess,
                        num_levels,
                        max_iterations,
                    );
                    (tracked.coefficients(), ok)
                },
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
        let max_recovered_dist2: f32 = self.max_recovered_dist2;
        {
            let (valid, transforms) = out.parts_mut();
            self.pool.for_each_warp(
                transforms.coefficients_prefix_mut(count),
                &mut valid[..count],
                |index| {
                    let kept: [f32; 6] = forward.coefficients(index);
                    if !forward_valid[index] {
                        return (kept, false);
                    }
                    // `off = t2 - t2_guess` with `t2 == t1` at that point,
                    // so `off == source position - guess`; `t1_recovered += off`.
                    let source: Vector2<f32> = patches.position(index);
                    let offset: Vector2<f32> = source - transforms_in.translation(index);
                    let recovered_guess: Vector2<f32> = forward.translation(index) + offset;
                    let (recovered, ok) = track_point::<P>(
                        backward,
                        index,
                        prev,
                        forward.get(index).linear,
                        recovered_guess,
                        num_levels,
                        max_iterations,
                    );
                    if !ok {
                        return (kept, false);
                    }
                    // `dist2 = (t1 - t1_recovered).squaredNorm()`.
                    let dist2: f32 = (source - recovered.translation).norm_squared();
                    (kept, dist2 < max_recovered_dist2)
                },
            );
        }

        out.finish(count);
        Ok(())
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
    let capacity: usize = patches.capacity;
    let row_stride: usize = P::SIZE * capacity;
    let mut patch_valid: bool = true;

    for _ in 0..max_iterations {
        if !patch_valid {
            // `for (iteration = 0; patch_valid &&...)`, as a no-op.
            continue;
        }

        patch_valid &= patch_residual::<P, ImageU16>(
            &patches.data[data_offset..],
            capacity,
            image,
            transform,
            &mut residual,
        );

        if patch_valid {
            let increment: Vector3<f32> = -patch_increment::<P>(
                &patches.h_inv_jt[jacobian_offset..],
                capacity,
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
