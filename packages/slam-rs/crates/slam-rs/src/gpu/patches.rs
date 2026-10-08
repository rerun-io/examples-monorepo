//! Source positions for templates built in registers by fused KLT.

use crate::frontend::flow::FrontendError;
use cubecl::prelude::Runtime;

use super::pyramid::GpuPyramid;
use crate::pyramid::Pyramid;
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern;
use kornia_staging_slam::tracking::optical_flow::{SourcePatches};
use kornia_staging_imgproc::optical_flow::patch_tracker::{PointsSoA};

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
    pub fn new(capacity: usize, num_levels: usize) -> Result<Self, FrontendError> {
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

    pub(super) fn len(&self) -> usize {
        self.positions.len()
    }

    pub(super) fn position(&self, patch: usize) -> [f32; 2] {
        self.positions.get(patch)
    }
    pub(super) fn selected(&self, index: usize) -> f32 {
        f32::from(u8::from(self.selected[index]))
    }
}

impl<P: Pattern, R: Runtime> SourcePatches for GpuPatchSources<P, R> {
    type Error = FrontendError;
    type Pyramid = GpuPyramid<R>;

    fn build(
        &mut self,
        pyramid: &GpuPyramid<R>,
        positions: &PointsSoA,
        selected: Option<&[bool]>,
    ) -> Result<(), FrontendError> {
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


}

use kornia_staging_imgproc::optical_flow::patch_tracker::limits::{check_patch_inputs, checked_patch_shape};
