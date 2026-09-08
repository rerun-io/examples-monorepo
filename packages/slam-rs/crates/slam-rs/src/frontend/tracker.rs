//! The SE(2) inverse-compositional KLT tracker, ported from
//! `frame_to_frame_optical_flow.h:294-438`.
//!
//! Three C++ functions come across:
//!
//! * `trackPointAtLevel` (`:404-438`) — up to `optical_flow_max_iterations`
//!   Gauss-Newton steps on one pyramid level;
//! * `trackPoint` (`:377-402`) — the coarse-to-fine sweep, with the source patch
//!   rebuilt from the previous frame's pyramid at every level;
//! * `trackPoints` (`:294-375`) — the whole keypoint set forward, then backward,
//!   keeping only the tracks that come back to where they started.
//!
//! ## The stage seam
//!
//! [`PatchTracker`] and [`SourcePatches`] are the trait pair the frontend is
//! generic over (`cubecl-portability.md` §12.1). Neither names a concrete
//! pyramid: the pyramid type is associated, bounded by
//! [`crate::pyramid::Pyramid`], so a CubeCL backend supplies its own device-side
//! pyramid, patch storage and tracker and the driver compiles against it
//! unchanged. [`CpuPatchTracker`] and [`PatchSoA`] are the CPU pair.
//!
//! ## Data layout
//!
//! Every per-patch buffer here is structure-of-arrays with the **patch index
//! fast-varying**, so thread *i* of a warp reads element *i*: the patch taps and
//! Jacobians in [`PatchSoA`], the positions in [`PointsSoA`], and the 2x3 warps
//! in [`FlowTransforms`], whose six coefficients live in six flat arrays rather
//! than one array of six-float records (§12.2). Nothing on this path is a map,
//! and nothing on it allocates once the buffers have reached their high-water
//! mark.
//!
//! ## What the port moves, and why nothing moves numerically
//!
//! **The source patches are hoisted into a [`PatchSoA`].** `trackPoint` builds
//! `PatchT p(old_pyr.lvl(level), old_transform.translation() / scale)` inside its
//! level loop (`:388`). That patch depends only on the previous pyramid, the
//! source position and the level — never on anything the tracker computes — so
//! building all of them up front is the same arithmetic in a different order, and
//! it is the split a GPU wants: one kernel over (patch, level), then one over
//! patches. The forward patches come in as an argument; the backward ones are
//! built inside [`CpuPatchTracker`], because their positions are the forward
//! result.
//!
//! **The three passes are separated.** basalt does forward track, mask test and
//! backward track inside one per-point body. Here the forward pass runs over all
//! points, then the backward patches are built, then the backward pass runs.
//! Points are independent, so per-point values are identical; the mask test that
//! sat between the two passes moves to the caller, which drops the same points
//! one step later (see [`super::flow`]).
//!
//! **Loops have fixed bounds.** The C++ conditions its `for` on `patch_valid`
//! (`:383`, `:408`). Breaking out early and running to the end with a no-op give
//! the same numbers here, and only the second form ports to a GPU where the whole
//! warp runs the maximum count anyway (§12.2). Invalid points therefore idle
//! instead of exiting.
//!
//! **The masks and the depth guess stay outside.** `trackPoints` reads
//! `masks1`/`masks2` and calls `calib.projectBetweenCams` (`:329`, `:335-342`),
//! which would drag the calibration into the tracker. The driver applies both and
//! hands over the already-offset guesses; the offset itself is recovered here as
//! `source position - guess`, exactly the `off` the C++ adds back before the
//! backward track (`:357`).

use nalgebra::{Matrix2, Vector2, Vector3};

use crate::frontend::parallel::WorkPool;
use crate::frontend::patch::{build_patch, patch_increment, patch_residual};
use crate::frontend::patterns::{MAX_PATTERN_SIZE, Pattern};
use crate::frontend::se2::{AffineCompact2f, se2_exp};
use crate::image::ImageU16;
use crate::pyramid::{Pyramid, PyramidU16};

/// The increment guard at `frame_to_frame_optical_flow.h:425`.
const MAX_INCREMENT_INFINITY_NORM: f32 = 1e6;

/// `const int filter_margin = 2` (`frame_to_frame_optical_flow.h:430`).
const FILTER_MARGIN: f32 = 2.0;

/// The most keypoints a tracker may be sized for.
///
/// basalt needs no such ceiling: its keypoint maps grow, so its memory follows
/// the scene. The port preallocates every per-patch buffer
/// ([`FrontendOptions::max_keypoints`](super::flow::FrontendOptions::max_keypoints)),
/// which turns the budget into a memory request that arrives from outside — over
/// the Python boundary among other places — and `2^63` keypoints panicked
/// `Vec::with_capacity` with a capacity overflow before this existed.
///
/// A million keypoints is about 3.4 kB of patch storage each at pattern 51 over
/// four pyramid levels, so roughly 7 GB across the two [`PatchSoA`] a
/// [`CpuPatchTracker`] holds: far more than any rig this port runs (the shipped
/// 50-pixel grid on a 960x960 frame produces about 400) and far below the point
/// where the products in [`PatchSoA::new`] leave the `usize` range.
pub const MAX_CAPACITY: usize = 1 << 20;

/// The most pyramid levels a tracker may be sized for.
///
/// `num_levels` is `optical_flow_levels + 1` and every per-patch buffer is sized
/// with it, so it multiplies the capacity above. Each level halves both sides of
/// the image: at 24 levels the top of the pyramid is one pixel of a 16-million
/// pixel-wide frame, and basalt ships 3. Without a ceiling here a config asking
/// for `10^12` levels turned into a 600-petabyte `Vec`, and a `Vec` that cannot be
/// allocated aborts the process rather than returning.
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

/// A list of `Eigen::AffineCompact2f` warps in six flat arrays.
///
/// The C++ keeps them as 2x3 matrices in a map (`optical_flow.h:66-68`), which
/// gives one coefficient a stride of six floats across keypoints. Splitting the
/// six coefficients into six arrays is the layout §12.2 asks for; callers that
/// want one warp back get it through [`FlowTransforms::get`], which costs six
/// loads and no indirection.
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

    /// Every translation `x`, patch index fast-varying.
    pub fn translations_x(&self) -> &[f32] {
        &self.tx
    }

    /// Every translation `y`, patch index fast-varying.
    pub fn translations_y(&self) -> &[f32] {
        &self.ty
    }
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
    /// [`TrackerError::CapacityTooLarge`] above [`MAX_CAPACITY`],
    /// [`TrackerError::TooManyLevels`] above [`MAX_LEVELS`], and
    /// [`TrackerError::BufferShapeOverflow`] when a buffer's element count does
    /// not fit in a `usize`. All three are checked before anything is allocated:
    /// the products below reach `Vec` as a length, and a `Vec` too long to exist
    /// panics rather than returning (decision D32).
    pub fn new(capacity: usize, num_levels: usize) -> Result<Self, TrackerError> {
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
            taps: P::SIZE,
        };
        let flags: usize = num_levels.checked_mul(capacity).ok_or_else(overflow)?;
        let taps: usize = flags.checked_mul(P::SIZE).ok_or_else(overflow)?;
        let jacobians: usize = taps.checked_mul(3).ok_or_else(overflow)?;
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

    /// Whether one patch at one level may be tracked (`patch.h:164-165`).
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
    /// `old_transform.translation() / scale` of `frame_to_frame_optical_flow.h:388`.
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
        if count > self.capacity {
            return Err(TrackerError::CapacityExceeded {
                offered: count,
                capacity: self.capacity,
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
        if pyramid.num_levels() < self.num_levels {
            return Err(TrackerError::LevelMismatch {
                what: "the pyramid",
                expected: self.num_levels,
                actual: pyramid.num_levels(),
            });
        }
        self.len = count;

        for level in 0..self.num_levels {
            let Some(image) = pyramid.level(level) else {
                return Err(TrackerError::LevelMismatch {
                    what: "the pyramid",
                    expected: self.num_levels,
                    actual: pyramid.num_levels(),
                });
            };
            // `const Scalar scale = 1 << level` (`frame_to_frame_optical_flow.h:384`).
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

/// The frontend's tracking stage: one call moves a whole camera's patches.
///
/// The trait takes the entire keypoint set, never one point, and names no
/// concrete pyramid or patch storage, so the CPU implementation here and a later
/// CubeCL one can be swapped without touching the driver (§12.1).
pub trait PatchTracker {
    /// The sampling pattern this tracker was built for.
    type Pattern: Pattern;

    /// The pyramid representation it reads.
    type Pyramid: Pyramid;

    /// The source-patch storage it consumes.
    type Patches: SourcePatches<Pyramid = Self::Pyramid>;

    /// Track every patch of `patches` from `prev` into `next`.
    ///
    /// `transforms_in[i]` is basalt's `transform_2` before tracking: the linear
    /// part of the source keypoint and the translation the caller guesses. The
    /// source position itself is `patches.position(i)`.
    ///
    /// # Errors
    ///
    /// [`TrackerError`] when the inputs do not match the shape the tracker was
    /// built for.
    fn track(
        &mut self,
        prev: &Self::Pyramid,
        next: &Self::Pyramid,
        patches: &Self::Patches,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError>;

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
        })
    }

    /// Workers the tracking passes run on.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }
}

impl<P: Pattern> PatchTracker for CpuPatchTracker<P> {
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

    fn track(
        &mut self,
        prev: &PyramidU16,
        next: &PyramidU16,
        patches: &PatchSoA<P>,
        transforms_in: &FlowTransforms,
        out: &mut FlowResult,
    ) -> Result<(), TrackerError> {
        let count: usize = transforms_in.len();
        if count != patches.len() {
            return Err(TrackerError::LengthMismatch {
                first_name: "patches",
                first: patches.len(),
                second_name: "transforms",
                second: count,
            });
        }
        if count > self.capacity {
            return Err(TrackerError::CapacityExceeded {
                offered: count,
                capacity: self.capacity,
            });
        }
        // The patch set is built by the caller, so its depth is an input like any
        // other: a one-level `PatchSoA` in a two-level tracker used to index past
        // the end of `valid`.
        if patches.num_levels() < self.num_levels {
            return Err(TrackerError::LevelMismatch {
                what: "the patch set",
                expected: self.num_levels,
                actual: patches.num_levels(),
            });
        }
        for (what, pyramid) in [("the previous pyramid", prev), ("the next pyramid", next)] {
            if pyramid.num_levels() < self.num_levels {
                return Err(TrackerError::LevelMismatch {
                    what,
                    expected: self.num_levels,
                    actual: pyramid.num_levels(),
                });
            }
        }

        out.reset(count);

        // ── forward: `trackPoint(pyr_1, pyr_2, transform_1, transform_2)` (`:349`)
        let max_iterations: usize = self.max_iterations;
        let num_levels: usize = self.num_levels;
        let (target_width, target_height): (f32, f32) = level0_size(next);
        {
            let [m00, m01, m10, m11, tx, ty] = self.forward.coefficients_mut();
            self.pool.for_each_warp(
                [
                    &mut m00[..count],
                    &mut m01[..count],
                    &mut m10[..count],
                    &mut m11[..count],
                    &mut tx[..count],
                    &mut ty[..count],
                ],
                &mut self.forward_valid[..count],
                |index| {
                    let guess: Vector2<f32> = transforms_in.translation(index);
                    // `valid = t2(0) >= 0 && t2(1) >= 0 && t2(0) < w && t2(1) < h` (`:346`).
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
        for index in 0..count {
            self.backward_positions
                .set(index, self.forward.translation(index));
        }
        let mut backward_positions: PointsSoA = std::mem::take(&mut self.backward_positions);
        backward_positions.resize(count);
        let build: Result<(), TrackerError> = self.backward.build(
            next,
            &backward_positions,
            Some(&self.forward_valid[..count]),
        );
        backward_positions.resize(self.capacity);
        self.backward_positions = backward_positions;
        build?;

        // ── backward: `trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered)` (`:359`)
        let backward: &PatchSoA<P> = &self.backward;
        let forward: &FlowTransforms = &self.forward;
        let forward_valid: &[bool] = &self.forward_valid[..count];
        let max_recovered_dist2: f32 = self.max_recovered_dist2;
        {
            let (valid, transforms) = out.parts_mut();
            let [m00, m01, m10, m11, tx, ty] = transforms.coefficients_mut();
            self.pool.for_each_warp(
                [
                    &mut m00[..count],
                    &mut m01[..count],
                    &mut m10[..count],
                    &mut m11[..count],
                    &mut tx[..count],
                    &mut ty[..count],
                ],
                &mut valid[..count],
                |index| {
                    let kept: [f32; 6] = forward.coefficients(index);
                    if !forward_valid[index] {
                        return (kept, false);
                    }
                    // `off = t2 - t2_guess` with `t2 == t1` at that point (`:339`),
                    // so `off == source position - guess`; `t1_recovered += off` (`:357`).
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
                    // `dist2 = (t1 - t1_recovered).squaredNorm()` (`:362`).
                    let dist2: f32 = (source - recovered.translation).norm_squared();
                    (kept, dist2 < max_recovered_dist2)
                },
            );
        }

        out.finish(count);
        Ok(())
    }
}

/// Level 0's `(width, height)` as floats, standing in for basalt's `w`, `h`.
///
/// basalt reads `calib.resolution.at(0)` for every camera
/// (`frame_to_frame_optical_flow.h:108-109`, trap 16); the port reads the target
/// camera's own level-0 size, which is the same number for a rig whose cameras
/// share a resolution and the right one for msd-g2, whose cameras do not
/// (decision D30).
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

/// `trackPoint` (`frame_to_frame_optical_flow.h:377-402`).
///
/// Coarse to fine, with the translation divided by `1 << level` on the way in and
/// multiplied back on the way out (`:386`, `:396`) — both exact in `f32`, since
/// the scale is a power of two. The linear part starts at the identity (`:381`)
/// and is composed with the source's at the end (`:399`), so the SE(2) rotation
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
            // The C++ `for` exits here (`:383`); running the remaining levels as
            // a no-op keeps the bound fixed and the numbers identical.
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

/// `trackPointAtLevel` (`frame_to_frame_optical_flow.h:404-438`).
///
/// One Gauss-Newton step is: warp the pattern, take the mean-normalised residual,
/// `inc = -H_se2^-1 J_se2^T r`, reject a non-finite or huge increment
/// (`:422-425`, because `SE2::exp` crashes on NaN), apply it on the right
/// (`transform *= SE2::exp(inc)`, `:428`) and require the new centre to stay two
/// pixels inside the image (`:430-432`).
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
            // `for (iteration = 0; patch_valid && ...)` (`:408`), as a no-op.
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
            // `inc.lpNorm<Eigen::Infinity>()` is `cwiseAbs().maxCoeff()`, whose
            // reduction is `(a < b) ? b : a` from element 0 — spelled out so a
            // NaN takes the same branch it takes in C++.
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::frontend::patterns::Pattern51;
    use crate::pyramid::{CpuPyramidBuilder, PyramidBuilder};
    use proptest::prelude::*;

    /// A band-limited texture: twelve plane waves with wavelengths between 16
    /// and 56 pixels, in fixed pseudo-random directions and phases.
    ///
    /// Band-limited matters twice over. Below the Nyquist of the finest pyramid
    /// level the `[1,4,6,4,1]` subsample does not alias, so the coarse levels
    /// really do carry the same shift; and away from the sampling limit bilinear
    /// interpolation reconstructs the field closely, so the residual's fixed
    /// point sits near the true shift rather than a fraction of a pixel off it.
    /// Twelve components in different directions also keep every patch's `H_se2`
    /// well conditioned: a single wave, or a field that is locally almost affine,
    /// is the aperture problem and no tracker recovers a shift from it.
    fn texture(x: f64, y: f64) -> f64 {
        // (wavelength, direction in turns, phase in turns)
        const WAVES: [(f64, f64, f64); 16] = [
            (22.0000, 0.000000, 0.000000),
            (23.5218, 0.381966, 0.618034),
            (25.1489, 0.763932, 0.236068),
            (26.8886, 0.145898, 0.854102),
            (28.7486, 0.527864, 0.472136),
            (30.7373, 0.909830, 0.090170),
            (32.8635, 0.291796, 0.708204),
            (35.1368, 0.673762, 0.326238),
            (37.5674, 0.055728, 0.944272),
            (40.1661, 0.437694, 0.562306),
            (42.9446, 0.819660, 0.180340),
            (45.9153, 0.201626, 0.798374),
            (49.0914, 0.583592, 0.416408),
            (52.4873, 0.965558, 0.034442),
            (56.1181, 0.347524, 0.652476),
            (60.0000, 0.729490, 0.270510),
        ];
        let tau: f64 = std::f64::consts::TAU;
        let mut sum: f64 = 0.0;
        for (wavelength, direction, phase) in WAVES {
            let angle: f64 = tau * direction;
            let projection: f64 = x * angle.cos() + y * angle.sin();
            sum += (tau * (projection / wavelength + phase)).sin();
        }
        sum / WAVES.len() as f64
    }

    /// A textured frame, shifted by `(dx, dy)`: the same continuous field
    /// resampled at `(x - dx, y - dy)`, so the shift is exact by construction.
    fn shifted_image(width: usize, height: usize, dx: f32, dy: f32) -> ImageU16 {
        let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
        for y in 0..height {
            for x in 0..width {
                let fx: f64 = f64::from(x as f32 - dx);
                let fy: f64 = f64::from(y as f32 - dy);
                let value: f64 = 32_000.0 + 28_000.0 * texture(fx, fy);
                image.set(x, y, value as u16);
            }
        }
        image
    }

    fn pyramid_of(image: &ImageU16, levels: usize) -> PyramidU16 {
        let mut pyramid: PyramidU16 =
            PyramidU16::with_capacity(image.width(), image.height(), levels).unwrap();
        CpuPyramidBuilder::new()
            .build(0, image, &mut pyramid)
            .unwrap();
        pyramid
    }

    struct Fixture {
        prev: PyramidU16,
        next: PyramidU16,
        patches: PatchSoA<Pattern51>,
        transforms: FlowTransforms,
        positions: PointsSoA,
    }

    fn fixture(dx: f32, dy: f32, levels: usize) -> Fixture {
        let base: ImageU16 = shifted_image(160, 160, 0.0, 0.0);
        let moved: ImageU16 = shifted_image(160, 160, dx, dy);
        let prev: PyramidU16 = pyramid_of(&base, levels);
        let next: PyramidU16 = pyramid_of(&moved, levels);

        let mut positions: PointsSoA = PointsSoA::default();
        for y in (40..120).step_by(16) {
            for x in (40..120).step_by(16) {
                positions.push(Vector2::new(x as f32, y as f32));
            }
        }
        let mut transforms: FlowTransforms = FlowTransforms::default();
        for index in 0..positions.len() {
            transforms.push(&AffineCompact2f::at(positions.get(index)));
        }

        let mut patches: PatchSoA<Pattern51> = PatchSoA::new(positions.len(), levels + 1).unwrap();
        patches.build(&prev, &positions, None).unwrap();

        Fixture {
            prev,
            next,
            patches,
            transforms,
            positions,
        }
    }

    fn tracker(capacity: usize, levels: usize, threads: usize) -> CpuPatchTracker<Pattern51> {
        CpuPatchTracker::new(
            capacity,
            levels + 1,
            5,
            0.04,
            WorkPool::new(threads).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn an_integer_shift_is_recovered() {
        let levels: usize = 3;
        let scene: Fixture = fixture(2.0, -1.0, levels);
        let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut out,
            )
            .unwrap();

        assert!(
            out.len() >= scene.positions.len() / 2,
            "tracked {}",
            out.len()
        );
        for index in out.tracked() {
            let index: usize = *index as usize;
            let moved: Vector2<f32> = out.transform(index).translation - scene.positions.get(index);
            // An integer shift moves the samples themselves, so the two
            // bilinear reconstructions are exact translates of each other and
            // the residual's fixed point is the true shift.
            assert!(
                (moved.x - 2.0).abs() < 0.01 && (moved.y + 1.0).abs() < 0.01,
                "patch {index} moved by {moved:?}, expected (2, -1)"
            );
        }
    }

    /// A sub-pixel shift, up to the pattern's own radius.
    ///
    /// The tolerance is not the tracker's convergence — it converges to five
    /// decimal places in three iterations — but the **bias of the fixed point
    /// itself**. `interp` reconstructs the image bilinearly and `interpGrad`
    /// differentiates that reconstruction by central differences
    /// (`image.h:396-469`), so for a shift that is not a whole number of samples
    /// the residual vanishes not at the true shift but a little beside it.
    ///
    /// The size of that displacement depends only on the **fractional** part of
    /// the shift, not on its magnitude: on this texture an exactly integer shift
    /// is recovered to `0.0000` px, a shift of 0.02 px to 0.0004, and a shift of
    /// half a pixel to 0.035 on the median patch and 0.13 on the worst — the same
    /// numbers whether the shift is 0.5 or 3.5 pixels. Shortening the texture's
    /// wavelengths raises the floor and lengthening them makes the patches
    /// ill-conditioned instead; basalt's C++ has the same property, because this
    /// is its arithmetic. The gate is therefore the median, with a cap on the tail.
    fn sub_pixel_shift_error(dx: f32, dy: f32) -> (f32, f32, usize, usize) {
        let levels: usize = 3;
        let scene: Fixture = fixture(dx, dy, levels);
        let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut out,
            )
            .unwrap();

        let mut errors: Vec<f32> = Vec::new();
        for index in out.tracked() {
            let index: usize = *index as usize;
            let moved: Vector2<f32> = out.transform(index).translation - scene.positions.get(index);
            errors.push((moved.x - dx).abs().max((moved.y - dy).abs()));
        }
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median: f32 = errors
            .get(errors.len() / 2)
            .copied()
            .unwrap_or(f32::INFINITY);
        let worst: f32 = errors.last().copied().unwrap_or(f32::INFINITY);
        (median, worst, out.len(), scene.positions.len())
    }

    #[test]
    fn a_sub_pixel_shift_is_recovered() {
        let (median, worst, tracked, total) = sub_pixel_shift_error(0.6, 1.4);
        assert_eq!(tracked, total);
        assert!(median < 0.05, "median error {median}");
        assert!(worst < 0.2, "worst error {worst}");
    }

    /// The forward-backward gate (`frame_to_frame_optical_flow.h:362-364`) is
    /// what rejects a track onto an unrelated image.
    #[test]
    fn a_mismatched_pair_is_rejected() {
        let levels: usize = 3;
        let scene: Fixture = fixture(0.0, 0.0, levels);
        // A different texture entirely, not a shift of the first.
        let mut other: ImageU16 = ImageU16::zeros(160, 160).unwrap();
        for y in 0..160 {
            for x in 0..160 {
                let value: f64 = 25_000.0
                    + 9_000.0 * ((x as f64) * 0.61).cos()
                    + 6_000.0 * ((y as f64) * 0.47).sin();
                other.set(x, y, value as u16);
            }
        }
        let unrelated: PyramidU16 = pyramid_of(&other, levels);

        let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker
            .track(
                &scene.prev,
                &unrelated,
                &scene.patches,
                &scene.transforms,
                &mut out,
            )
            .unwrap();
        assert!(
            out.len() * 4 < scene.positions.len(),
            "{} of {} tracks survived an unrelated image",
            out.len(),
            scene.positions.len()
        );
    }

    #[test]
    fn one_thread_and_four_threads_agree_exactly() {
        let levels: usize = 3;
        let scene: Fixture = fixture(1.3, -0.7, levels);

        let mut single: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker(scene.positions.len(), levels, 1)
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut single,
            )
            .unwrap();

        let mut wide: FlowResult = FlowResult::with_capacity(scene.positions.len());
        tracker(scene.positions.len(), levels, 4)
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut wide,
            )
            .unwrap();

        assert_eq!(single.tracked(), wide.tracked());
        assert!(!single.is_empty());
        for index in single.tracked() {
            let index: usize = *index as usize;
            assert_eq!(single.transform(index), wide.transform(index));
        }
    }

    #[test]
    fn two_runs_of_the_same_tracker_agree_exactly() {
        let levels: usize = 3;
        let scene: Fixture = fixture(0.9, 0.4, levels);
        let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 4);

        let mut first: FlowResult = FlowResult::with_capacity(scene.positions.len());
        let mut second: FlowResult = FlowResult::with_capacity(scene.positions.len());
        for out in [&mut first, &mut second] {
            tracker
                .track(
                    &scene.prev,
                    &scene.next,
                    &scene.patches,
                    &scene.transforms,
                    out,
                )
                .unwrap();
        }
        assert_eq!(first.tracked(), second.tracked());
        for index in first.tracked() {
            let index: usize = *index as usize;
            assert_eq!(first.transform(index), second.transform(index));
        }
    }

    #[test]
    fn more_keypoints_than_capacity_is_refused() {
        let levels: usize = 1;
        let scene: Fixture = fixture(0.0, 0.0, levels);
        let mut tracker: CpuPatchTracker<Pattern51> = tracker(2, levels, 1);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        let error = tracker
            .track(
                &scene.prev,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut out,
            )
            .unwrap_err();
        assert_eq!(
            error,
            TrackerError::CapacityExceeded {
                offered: scene.positions.len(),
                capacity: 2
            }
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(12))]

        /// Any shift up to the pattern's radius (3.5 px for `Pattern51`) is
        /// recovered, with the interpolation bias documented on
        /// [`sub_pixel_shift_error`] as the tolerance.
        #[test]
        fn any_shift_within_the_pattern_radius_is_recovered(
            dx in -3.5f32..3.5,
            dy in -3.5f32..3.5,
        ) {
            let (median, worst, tracked, total) = sub_pixel_shift_error(dx, dy);
            prop_assert!(tracked * 4 >= total * 3, "tracked {tracked} of {total}");
            prop_assert!(median < 0.05, "median error {median} for ({dx}, {dy})");
            prop_assert!(worst < 0.2, "worst error {worst} for ({dx}, {dy})");
        }
    }
    /// A patch set shallower than the tracker used to index past the end of its
    /// validity array; it is a typed error now (decision D32).
    #[test]
    fn a_shallow_patch_set_is_refused() {
        for threads in [1, 4] {
            let levels: usize = 3;
            let scene: Fixture = fixture(0.0, 0.0, levels);
            let mut shallow: PatchSoA<Pattern51> = PatchSoA::new(scene.positions.len(), 1).unwrap();
            shallow.build(&scene.prev, &scene.positions, None).unwrap();

            let mut tracker: CpuPatchTracker<Pattern51> =
                tracker(scene.positions.len(), levels, threads);
            let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
            let error = tracker
                .track(
                    &scene.prev,
                    &scene.next,
                    &shallow,
                    &scene.transforms,
                    &mut out,
                )
                .unwrap_err();
            assert_eq!(
                error,
                TrackerError::LevelMismatch {
                    what: "the patch set",
                    expected: levels + 1,
                    actual: 1
                }
            );
        }
    }

    /// A pyramid shallower than the tracker is refused too, on either side.
    #[test]
    fn a_shallow_pyramid_is_refused() {
        let levels: usize = 3;
        let scene: Fixture = fixture(0.0, 0.0, levels);
        let shallow: PyramidU16 = pyramid_of(&shifted_image(160, 160, 0.0, 0.0), 1);
        let mut tracker: CpuPatchTracker<Pattern51> = tracker(scene.positions.len(), levels, 1);
        let mut out: FlowResult = FlowResult::with_capacity(scene.positions.len());
        let error = tracker
            .track(
                &shallow,
                &scene.next,
                &scene.patches,
                &scene.transforms,
                &mut out,
            )
            .unwrap_err();
        assert_eq!(
            error,
            TrackerError::LevelMismatch {
                what: "the previous pyramid",
                expected: levels + 1,
                actual: 2
            }
        );
    }

    /// A selection mask shorter than the positions used to index past its end.
    #[test]
    fn a_short_selection_mask_is_refused() {
        let levels: usize = 1;
        let scene: Fixture = fixture(0.0, 0.0, levels);
        let mut patches: PatchSoA<Pattern51> =
            PatchSoA::new(scene.positions.len(), levels + 1).unwrap();
        let short: Vec<bool> = vec![true; 2];
        let error = patches
            .build(&scene.prev, &scene.positions, Some(&short))
            .unwrap_err();
        assert_eq!(
            error,
            TrackerError::LengthMismatch {
                first_name: "positions",
                first: scene.positions.len(),
                second_name: "selection flags",
                second: 2,
            }
        );
    }

    /// More positions than the patch storage holds is a typed error too.
    #[test]
    fn more_positions_than_patch_capacity_is_refused() {
        let levels: usize = 1;
        let scene: Fixture = fixture(0.0, 0.0, levels);
        let mut patches: PatchSoA<Pattern51> = PatchSoA::new(2, levels + 1).unwrap();
        let error = patches
            .build(&scene.prev, &scene.positions, None)
            .unwrap_err();
        assert_eq!(
            error,
            TrackerError::CapacityExceeded {
                offered: scene.positions.len(),
                capacity: 2
            }
        );
    }

    /// The warp storage is six flat arrays; the round trip through them is exact.
    /// A capacity past the ceiling is refused before a byte is allocated.
    ///
    /// `Vec::with_capacity(2^63)` panics with `capacity overflow`, and a panic in
    /// here reaches Python as a `PanicException` that ordinary `except Exception`
    /// handlers do not catch (decision D32). The ceiling is what makes the
    /// request answerable instead.
    #[test]
    fn a_capacity_over_the_ceiling_is_refused() {
        for capacity in [MAX_CAPACITY + 1, 1 << 40, usize::MAX / 2, usize::MAX] {
            assert_eq!(
                PatchSoA::<Pattern51>::new(capacity, 4).unwrap_err(),
                TrackerError::CapacityTooLarge {
                    capacity,
                    ceiling: MAX_CAPACITY,
                }
            );
            assert_eq!(
                CpuPatchTracker::<Pattern51>::new(capacity, 4, 5, 0.04, WorkPool::new(1).unwrap())
                    .unwrap_err(),
                TrackerError::CapacityTooLarge {
                    capacity,
                    ceiling: MAX_CAPACITY,
                }
            );
        }
    }

    /// A pyramid deeper than the ceiling is refused before the allocation.
    ///
    /// `optical_flow_levels = 10^12` sized a `Vec` of 6e17 floats. A `Vec` whose
    /// length fits in a `usize` but whose bytes do not exist does not panic — the
    /// allocator handler **aborts** the process, taking the Python interpreter
    /// with it, so this is checked rather than attempted.
    #[test]
    fn more_levels_than_the_ceiling_is_refused() {
        for num_levels in [MAX_LEVELS + 1, 1_000_000_000_001, usize::MAX] {
            assert_eq!(
                PatchSoA::<Pattern51>::new(3000, num_levels).unwrap_err(),
                TrackerError::TooManyLevels {
                    num_levels,
                    ceiling: MAX_LEVELS,
                }
            );
        }
    }

    /// The two ceilings bound every buffer product, in `usize` and in `u32`.
    ///
    /// This is what makes [`TrackerError::BufferShapeOverflow`] unreachable
    /// today: it is the guard that fires if either ceiling is ever raised past
    /// the point where a product wraps, and this test is the proof that it does
    /// not have to fire now.
    #[test]
    fn the_ceilings_bound_every_buffer_product() {
        let flags: usize = MAX_LEVELS.checked_mul(MAX_CAPACITY).unwrap();
        let taps: usize = flags.checked_mul(Pattern51::SIZE).unwrap();
        let jacobians: usize = taps.checked_mul(3).unwrap();
        assert!(
            jacobians <= u32::MAX as usize,
            "{jacobians} elements would wrap a 32-bit usize"
        );
    }

    /// The ceiling is well clear of anything the port runs.
    #[test]
    fn the_default_budget_is_far_under_the_ceiling() {
        // `FrontendOptions::default().max_keypoints` is 3000.
        const { assert!(MAX_CAPACITY > 300 * 3000) };
        let patches: PatchSoA<Pattern51> = PatchSoA::new(3000, 4).unwrap();
        assert_eq!(patches.capacity(), 3000);
        assert_eq!(patches.num_levels(), 4);
    }

    #[test]
    fn flow_transforms_round_trip_through_the_soa_arrays() {
        let mut transforms: FlowTransforms = FlowTransforms::default();
        let warps: [AffineCompact2f; 3] = [
            AffineCompact2f::at(Vector2::new(1.0, 2.0)),
            AffineCompact2f {
                linear: Matrix2::new(0.5, -0.25, 0.25, 0.5),
                translation: Vector2::new(-3.0, 4.5),
            },
            AffineCompact2f::identity(),
        ];
        for warp in &warps {
            transforms.push(warp);
        }
        assert_eq!(transforms.len(), 3);
        for (index, warp) in warps.iter().enumerate() {
            assert_eq!(transforms.get(index), *warp);
            assert_eq!(transforms.translation(index), warp.translation);
            assert_eq!(transforms.coefficients(index), warp.coefficients());
        }
        // One coefficient of every warp is contiguous, which is the point.
        assert_eq!(transforms.translations_x(), &[1.0, -3.0, 0.0]);
        assert_eq!(transforms.translations_y(), &[2.0, 4.5, 0.0]);

        transforms.remove(1);
        assert_eq!(transforms.len(), 2);
        assert_eq!(transforms.get(1), warps[2]);
        transforms.insert(1, &warps[1]);
        assert_eq!(transforms.get(1), warps[1]);
        transforms.set(0, &warps[2]);
        assert_eq!(transforms.get(0), warps[2]);
    }
    /// A second [`PatchTracker`] implementation, written only against the public
    /// API, proving a backend outside this module can publish results.
    ///
    /// It reports every input as tracked, at the guess it was given, through
    /// [`FlowResult::reset`], [`FlowResult::set_track`] and
    /// [`FlowResult::finish`]; the bulk path through [`FlowResult::parts_mut`]
    /// and [`FlowTransforms::coefficients_mut`] is what [`CpuPatchTracker`]
    /// itself uses, so both halves of the writing surface are exercised.
    #[derive(Debug, Default)]
    struct EchoTracker {
        capacity: usize,
        num_levels: usize,
    }

    impl PatchTracker for EchoTracker {
        type Pattern = Pattern51;
        type Pyramid = PyramidU16;
        type Patches = PatchSoA<Pattern51>;

        fn capacity(&self) -> usize {
            self.capacity
        }

        fn num_levels(&self) -> usize {
            self.num_levels
        }

        fn make_patches(&self) -> Result<PatchSoA<Pattern51>, TrackerError> {
            PatchSoA::new(self.capacity, self.num_levels)
        }

        fn track(
            &mut self,
            _prev: &PyramidU16,
            _next: &PyramidU16,
            patches: &PatchSoA<Pattern51>,
            transforms_in: &FlowTransforms,
            out: &mut FlowResult,
        ) -> Result<(), TrackerError> {
            let count: usize = transforms_in.len();
            if count != patches.len() {
                return Err(TrackerError::LengthMismatch {
                    first_name: "patches",
                    first: patches.len(),
                    second_name: "transforms",
                    second: count,
                });
            }
            out.reset(count);
            for index in 0..count {
                out.set_track(index, true, &transforms_in.get(index));
            }
            // The bulk path over the same buffers, to prove it is reachable.
            let (valid, transforms) = out.parts_mut();
            let [m00, ..] = transforms.coefficients_mut();
            assert_eq!(m00.len(), valid.len().max(m00.len()));
            out.finish(count);
            Ok(())
        }
    }

    #[test]
    fn a_second_backend_can_publish_results_through_the_public_api() {
        let levels: usize = 3;
        let scene: Fixture = fixture(0.7, -1.2, levels);
        let mut echo: EchoTracker = EchoTracker {
            capacity: scene.positions.len(),
            num_levels: levels + 1,
        };
        let mut out: FlowResult = FlowResult::default();
        echo.track(
            &scene.prev,
            &scene.next,
            &scene.patches,
            &scene.transforms,
            &mut out,
        )
        .unwrap();

        assert_eq!(out.len(), scene.positions.len());
        for index in out.tracked() {
            let index: usize = *index as usize;
            assert!(out.is_valid(index));
            assert_eq!(out.transform(index), scene.transforms.get(index));
        }
    }

    /// The write sequence is usable on its own: reset, set, finish.
    #[test]
    fn the_flow_result_writing_surface_compacts_what_it_is_given() {
        let mut out: FlowResult = FlowResult::default();
        out.reset(5);
        assert!(out.is_empty());
        out.set_track(1, true, &AffineCompact2f::at(Vector2::new(3.0, 4.0)));
        out.set_track(4, true, &AffineCompact2f::at(Vector2::new(-1.0, 0.5)));
        out.set_track(2, false, &AffineCompact2f::identity());
        out.finish(5);

        assert_eq!(out.tracked(), &[1, 4]);
        assert_eq!(out.transform(1).translation, Vector2::new(3.0, 4.0));
        assert_eq!(out.transform(4).translation, Vector2::new(-1.0, 0.5));
        assert!(!out.is_valid(0));
        assert!(!out.is_valid(2));

        // A reset clears the survivors without dropping the allocation.
        out.reset(3);
        assert!(out.tracked().is_empty());
        assert!(!out.is_valid(1));
    }
}
