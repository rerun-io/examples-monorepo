//! Source patches in groups of four: `[group][level][row][tap][lane]`.

use nalgebra::Vector2;

use super::{
    MAX_LEVELS, Pattern, PointsSoA, SourcePatches, TrackerError, WorkPool, check_patch_inputs,
    checked_patch_shape,
};
use crate::frontend::patch::build_patch_group;
use crate::image::ImageU16;
use crate::pyramid::{Pyramid, PyramidU16};

/// One camera's source patches for every pyramid level, in structure-of-arrays form.
///
/// Four adjacent patches share contiguous taps. All levels of a group are
/// contiguous, so builds split over point groups and tracking reuses nearby
/// cache lines. Capacity is fixed; the per-frame path never allocates.
#[derive(Debug, Clone)]
pub struct PatchSoA<P: Pattern> {
    pub(super) capacity: usize,
    num_levels: usize,
    len: usize,
    /// Source position at level 0, one per patch.
    positions: PointsSoA,
    /// `data[((group * num_levels + level) * P::SIZE + tap) * 4 + lane]`.
    pub(super) data: Vec<f32>,
    /// `h_inv_jt[(((group * num_levels + level) * 3 + row) * P::SIZE + tap) * 4 + lane]`.
    pub(super) h_inv_jt: Vec<f32>,
    /// `valid[(group * num_levels + level) * 4 + lane]`.
    valid: Vec<bool>,
    /// Workers [`SourcePatches::build`] spreads the groups over; `None` builds
    /// on the calling thread.
    pool: Option<WorkPool>,
    pattern: std::marker::PhantomData<P>,
}

impl<P: Pattern> PatchSoA<P> {
    /// Copy selected columns from a previous patch build without recomputing
    /// their floating-point values. Each pair is `(destination, source)`;
    /// capacities may differ, but both stores must already contain the slots.
    ///
    /// # Errors
    /// Returns a level or length mismatch before writing if a column is absent.
    pub fn copy_columns_from(
        &mut self,
        source: &Self,
        columns: &[(usize, usize)],
    ) -> Result<(), TrackerError> {
        if self.num_levels != source.num_levels {
            return Err(TrackerError::LevelMismatch {
                what: "the cached patch set",
                expected: self.num_levels,
                actual: source.num_levels,
            });
        }
        for &(destination, cached) in columns {
            for (index, len) in [(destination, self.len), (cached, source.len)] {
                if index >= len {
                    return Err(TrackerError::LengthMismatch {
                        first_name: "column index",
                        first: index,
                        second_name: "patches",
                        second: len,
                    });
                }
            }
        }
        for &(destination, cached) in columns {
            self.positions
                .set(destination, source.positions.get(cached));
            for level in 0..self.num_levels {
                self.valid[(destination / 4 * self.num_levels + level) * 4 + destination % 4] =
                    source.valid(level, cached);
                let dst_data = self.data_offset(level, destination);
                let src_data = source.data_offset(level, cached);
                for tap in 0..P::SIZE {
                    self.data[dst_data + tap * 4] = source.data[src_data + tap * 4];
                }
                let dst_jacobian = self.jacobian_offset(level, destination);
                let src_jacobian = source.jacobian_offset(level, cached);
                for tap in 0..3 * P::SIZE {
                    self.h_inv_jt[dst_jacobian + tap * 4] = source.h_inv_jt[src_jacobian + tap * 4];
                }
            }
        }
        Ok(())
    }

    /// Build point groups on `pool`'s workers. Every group is
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
        checked_patch_shape(capacity, num_levels, P::SIZE)?;
        let padded_capacity = capacity.div_ceil(4) * 4;
        let (flags, taps) = checked_patch_shape(padded_capacity, num_levels, P::SIZE)?;
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
        self.valid[(patch / 4 * self.num_levels + level) * 4 + patch % 4]
    }

    /// Offset of tap 0 of one patch's data; taps are four floats apart.
    #[inline]
    pub(super) fn data_offset(&self, level: usize, patch: usize) -> usize {
        (patch / 4 * self.num_levels + level) * P::SIZE * 4 + patch % 4
    }

    /// Offset of row 0, tap 0 of one patch's `H^-1 J^T`; rows are
    /// `P::SIZE * 4` apart and taps four floats apart.
    #[inline]
    pub(super) fn jacobian_offset(&self, level: usize, patch: usize) -> usize {
        (patch / 4 * self.num_levels + level) * 3 * P::SIZE * 4 + patch % 4
    }
}

impl<P: Pattern> SourcePatches for PatchSoA<P> {
    type Pyramid = PyramidU16;

    /// Build every patch at every level from `pyramid`.
    ///
    /// One patch per entry of `positions`, at `position / (1 << level)` — the
    /// Source position divided by the pyramid scale.
    /// [`build_patch_group`] writes directly into these arrays. With a pool,
    /// groups are built in parallel; otherwise the caller builds them.
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

        if count == 0 {
            return Ok(());
        }
        let levels: usize = self.num_levels;
        let build_group =
            |group: usize, data: &mut [f32], h_inv_jt: &mut [f32], valid: &mut [bool]| {
                let active: [bool; 4] = std::array::from_fn(|lane| {
                    let index = group * 4 + lane;
                    index < count && selected.is_none_or(|flags| flags[index])
                });
                if !active.iter().any(|flag| *flag) {
                    valid.fill(false);
                    return;
                }
                for level in 0..levels {
                    let Some(image) = images[level] else { continue };
                    let scale = (1u32 << level) as f32;
                    let points = std::array::from_fn(|lane| {
                        if active[lane] {
                            positions.get(group * 4 + lane) / scale
                        } else {
                            Vector2::new(-100.0, -100.0)
                        }
                    });
                    let (_, ok) = build_patch_group::<P>(
                        image,
                        points,
                        &mut data[level * 4 * P::SIZE..],
                        &mut h_inv_jt[level * 12 * P::SIZE..],
                    );
                    for lane in 0..4 {
                        valid[level * 4 + lane] = active[lane] && ok[lane];
                    }
                }
            };
        let (data_block, jacobian_block): (usize, usize) =
            (levels * P::SIZE * 4, levels * 3 * P::SIZE * 4);
        let groups = count.div_ceil(4);
        let data = &mut self.data[..groups * data_block];
        let h_inv_jt = &mut self.h_inv_jt[..groups * jacobian_block];
        let valid = &mut self.valid[..groups * levels * 4];
        let parallel: Option<()> = self.pool.as_ref().and_then(|pool| {
            pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks_mut(data_block)
                    .zip(h_inv_jt.par_chunks_mut(jacobian_block))
                    .zip(valid.par_chunks_mut(levels * 4))
                    .enumerate()
                    .for_each(|(group, ((data, h_inv_jt), valid))| {
                        build_group(group, data, h_inv_jt, valid)
                    });
            })
        });
        if parallel.is_none() {
            for (group, ((data, h_inv_jt), valid)) in data
                .chunks_mut(data_block)
                .zip(h_inv_jt.chunks_mut(jacobian_block))
                .zip(valid.chunks_mut(levels * 4))
                .enumerate()
            {
                build_group(group, data, h_inv_jt, valid);
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
