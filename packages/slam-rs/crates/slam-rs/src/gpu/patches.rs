//! Source positions for templates built in registers by fused KLT.

use cubecl::prelude::Runtime;
use nalgebra::Vector2;

use super::pyramid::GpuPyramid;
use crate::frontend::tracker::{
    PointsSoA, SourcePatches, TrackerError, check_patch_inputs, checked_patch_shape,
};
use crate::pyramid::Pyramid;
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;

/// Host-side inputs for tracking against separately allocated pyramids.
/// The shared-arena path reads its positions directly from `TrackInput`.
#[derive(Debug)]
pub struct GpuPatchSources<P: Pattern, R: Runtime> {
    capacity: usize,
    num_levels: usize,
    positions: PointsSoA,
    selected: Vec<bool>,
    marker: std::marker::PhantomData<(P, R)>,
}

impl<P: Pattern, R: Runtime> GpuPatchSources<P, R> {
    pub fn new(capacity: usize, num_levels: usize) -> Result<Self, TrackerError> {
        kornia_staging_imgproc::optical_flow::patch_se2::validate_pattern::<P>()?;
        checked_patch_shape(capacity, num_levels, P::SIZE)?;
        Ok(Self {
            capacity,
            num_levels,
            positions: PointsSoA::with_capacity(capacity),
            selected: Vec::with_capacity(capacity),
            marker: std::marker::PhantomData,
        })
    }

    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    pub(super) fn selected(&self, index: usize) -> f32 {
        f32::from(u8::from(self.selected[index]))
    }
}

impl<P: Pattern, R: Runtime> SourcePatches for GpuPatchSources<P, R> {
    type Pyramid = GpuPyramid<R>;

    fn build(
        &mut self,
        pyramid: &GpuPyramid<R>,
        positions: &PointsSoA,
        selected: Option<&[bool]>,
    ) -> Result<(), TrackerError> {
        check_patch_inputs(
            positions.len(),
            self.capacity,
            selected,
            pyramid.num_levels(),
            self.num_levels,
        )?;
        self.positions.clone_from(positions);
        self.selected.clear();
        self.selected
            .extend((0..positions.len()).map(|index| selected.is_none_or(|flags| flags[index])));
        Ok(())
    }

    fn len(&self) -> usize {
        self.positions.len()
    }

    fn position(&self, patch: usize) -> Vector2<f32> {
        self.positions.get(patch)
    }
}
