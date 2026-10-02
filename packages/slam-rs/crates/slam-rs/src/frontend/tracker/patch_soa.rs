//! Source patches with the patch index varying fastest in each array.

use nalgebra::Vector2;

use super::{
    MAX_LEVELS, Pattern, PointsSoA, SourcePatches, TrackerError, WorkPool, check_patch_inputs,
    checked_patch_shape,
};
use crate::frontend::patch::build_patch;
use crate::image::ImageU16;
use crate::pyramid::{Pyramid, PyramidU16};

/// One camera's source patches for every pyramid level, in structure-of-arrays form.
///
/// The layout puts the **patch index fast-varying** in every array, so a GPU
/// thread per patch reads consecutive addresses (§12.2). Capacity is fixed at
/// construction and the per-frame path never allocates.
#[derive(Debug, Clone)]
pub struct PatchSoA<P: Pattern> {
    pub(super) capacity: usize,
    num_levels: usize,
    len: usize,
    /// Source position at level 0, one per patch.
    positions: PointsSoA,
    /// `data[(level * P::SIZE + tap) * capacity + patch]`.
    pub(super) data: Vec<f32>,
    /// `h_inv_jt[((level * 3 + row) * P::SIZE + tap) * capacity + patch]`.
    pub(super) h_inv_jt: Vec<f32>,
    /// `valid[level * capacity + patch]`.
    valid: Vec<bool>,
    /// Workers [`SourcePatches::build`] spreads the levels over; `None` builds
    /// on the calling thread.
    pool: Option<WorkPool>,
    pattern: std::marker::PhantomData<P>,
}

impl<P: Pattern> PatchSoA<P> {
    /// Build the levels on `pool`'s workers, one level per task. Every level is
    /// its own contiguous block of each array and every patch a pure function
    /// of its position and level, so the values are those of the sequential
    /// build.
    pub fn with_pool(mut self, pool: WorkPool) -> Self {
        self.pool = (pool.threads() > 1).then_some(pool);
        self
    }

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
            pool: None,
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
    pub(super) fn data_offset(&self, level: usize, patch: usize) -> usize {
        level * P::SIZE * self.capacity + patch
    }

    /// Offset of row 0, tap 0 of one patch's `H^-1 J^T`; rows are
    /// `P::SIZE * capacity` apart and taps `capacity` apart.
    #[inline]
    pub(super) fn jacobian_offset(&self, level: usize, patch: usize) -> usize {
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
    /// A pure per-patch map. The patch-fast-varying layout has no contiguous
    /// split by patch, but each level is one contiguous block of every array, so
    /// with a pool ([`PatchSoA::with_pool`]) the levels are built in parallel;
    /// without one, on the calling thread.
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

        let mut images: [Option<&ImageU16>; MAX_LEVELS] = [None; MAX_LEVELS];
        for level in 0..self.num_levels {
            let image: Option<&ImageU16> = pyramid.level(level);
            match (images.get_mut(level), image) {
                (Some(slot), Some(image)) => *slot = Some(image),
                _ => {
                    return Err(TrackerError::LevelMismatch {
                        what: "the pyramid",
                        expected: self.num_levels,
                        actual: pyramid.num_levels(),
                    });
                }
            }
        }

        // One level: patch `index` sits at `index` of the level's block of each
        // array, taps `capacity` apart, Jacobian rows `P::SIZE * capacity` apart.
        let capacity: usize = self.capacity;
        if capacity == 0 {
            // No patches (count <= capacity), and the per-level chunks below would be zero-sized.
            return Ok(());
        }
        let build_level =
            |level: usize, data: &mut [f32], h_inv_jt: &mut [f32], valid: &mut [bool]| {
                let Some(image) = images[level] else {
                    return;
                };
                // `const Scalar scale = 1 << level`.
                let scale: f32 = (1u32 << level) as f32;
                for index in 0..count {
                    if !selected.is_none_or(|flags| flags[index]) {
                        valid[index] = false;
                        continue;
                    }
                    let position: Vector2<f32> = positions.get(index) / scale;
                    let (_mean, ok) = build_patch::<P, ImageU16>(
                        image,
                        &position,
                        &mut data[index..],
                        capacity,
                        &mut h_inv_jt[index..],
                        capacity,
                        P::SIZE * capacity,
                    );
                    valid[index] = ok;
                }
            };
        let levels: usize = self.num_levels;
        let (data_block, jacobian_block): (usize, usize) =
            (P::SIZE * capacity, 3 * P::SIZE * capacity);
        let data: &mut [f32] = &mut self.data[..levels * data_block];
        let h_inv_jt: &mut [f32] = &mut self.h_inv_jt[..levels * jacobian_block];
        let valid: &mut [bool] = &mut self.valid[..levels * capacity];
        let parallel: Option<()> = self.pool.as_ref().and_then(|pool| {
            pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks_mut(data_block)
                    .zip(h_inv_jt.par_chunks_mut(jacobian_block))
                    .zip(valid.par_chunks_mut(capacity))
                    .enumerate()
                    .for_each(|(level, ((data, h_inv_jt), valid))| {
                        build_level(level, data, h_inv_jt, valid)
                    });
            })
        });
        if parallel.is_none() {
            for (level, ((data, h_inv_jt), valid)) in data
                .chunks_mut(data_block)
                .zip(h_inv_jt.chunks_mut(jacobian_block))
                .zip(valid.chunks_mut(capacity))
                .enumerate()
            {
                build_level(level, data, h_inv_jt, valid);
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
