//! Packed cell selection over contiguous cameras of a shared pyramid arena.

use std::sync::Arc;

use cubecl::prelude::*;

use super::{GpuCornerScan, PendingSelection, SelectionLayout};
use crate::kernels;
use kornia_image::ImageSize;
use kornia_staging_imgproc::features::CellSelect;

#[derive(Clone)]
pub(super) struct BatchScanBuffers {
    keys: cubecl::server::Handle,
    width: usize,
    height: usize,
    cameras: usize,
    key_stride: usize,
}

impl<R: Runtime> GpuCornerScan<R> {
    /// Return None for mixed grids, noncontiguous camera sets, or a builder
    /// without a shared arena. Those retain the general selection path.
    ///
    /// # Arguments
    /// * `sizes` - Camera sizes in the batch begun by `begin_cells`.
    /// * `selects` - Optional cell policy for each camera.
    ///
    /// # Errors
    /// Rejects missing batch state, mismatched camera counts, and device failures.
    pub fn prepare_batch(
        &mut self,
        sizes: impl ExactSizeIterator<Item = ImageSize> + Clone,
        selects: &[Option<CellSelect>],
    ) -> Result<Option<CornerLaunch>, super::ScanError> {
        let outcome = crate::runtime::guarded(
            crate::runtime::GpuError::DeviceLost {
                what: "corner cell selection",
            },
            || {
                if !matches!(self.reads, super::SelectionReads::Pending(_)) {
                    return Err(super::ScanError::BatchNotBegun);
                }
                if sizes.len() != self.cameras.len() {
                    return Err(super::ScanError::BatchSizeMismatch {
                        inputs: sizes.len(),
                        outputs: self.cameras.len(),
                    });
                }
                Ok(self.prepare_batch_inner(sizes, selects))
            },
        );
        if outcome.is_err() {
            self.abort_selection();
        }
        outcome
    }

    fn prepare_batch_inner(
        &mut self,
        sizes: impl ExactSizeIterator<Item = ImageSize> + Clone,
        selects: &[Option<CellSelect>],
    ) -> Option<CornerLaunch> {
        let first = selects.iter().take(sizes.len()).position(Option::is_some)?;
        let select = selects[first]?;
        let end = selects
            .iter()
            .take(sizes.len())
            .rposition(Option::is_some)?
            + 1;
        if selects[first..end]
            .iter()
            .any(|value| *value != Some(select))
        {
            return None;
        }
        let image = sizes.clone().nth(first)?;
        let (width, height) = (image.width, image.height);
        let geometry = kernels::CellSelectGeometry::new(width, height, &select)?;
        let count = end - first;
        if count > kernels::MAX_CUBES_PER_DIM as usize
            || !kernels::uses_cell_kernel(select.grid.cell, width, &self.client)
        {
            return None;
        }
        let table = &self.level0;
        let frame = table.get(first)?.as_ref()?;
        let arena = frame.arena.as_ref()?.clone();
        if sizes.len() != arena.cameras {
            return None;
        }
        for (camera, image) in sizes.clone().enumerate().take(end).skip(first) {
            let frame = table.get(camera)?.as_ref()?;
            if image.width != width
                || image.height != height
                || frame.width != width
                || frame.height != height
                || frame.camera != camera
                || !frame
                    .arena
                    .as_ref()
                    .is_some_and(|other| Arc::ptr_eq(other, &arena))
            {
                return None;
            }
        }

        let cells = geometry.cells_x * geometry.cells_y;
        let (alignment, limit) = crate::transfer::binding_limits(&self.client);
        let key_stride = (cells * size_of::<u32>()).next_multiple_of(alignment) / size_of::<u32>();
        let key_len = key_stride.checked_mul(sizes.len())?;
        if key_len > limit / size_of::<u32>() {
            return None;
        }
        let fits = self.batch.as_ref().is_some_and(|batch| {
            batch.width == width
                && batch.height == height
                && batch.cameras == sizes.len()
                && batch.key_stride == key_stride
        });
        let batch = match &mut self.batch {
            Some(existing) if fits => existing,
            slot => {
                self.buffer_allocations += 1;
                slot.insert(BatchScanBuffers {
                    keys: self.client.empty(key_len * size_of::<u32>()),
                    width,
                    height,
                    cameras: sizes.len(),
                    key_stride,
                })
            }
        };
        let result = batch
            .keys
            .clone()
            .offset_end(((key_len - key_stride * count) * size_of::<u32>()) as u64);
        let work = CornerLaunch {
            arena,
            keys: batch.keys.clone(),
            key_len,
            geometry,
            first,
            key_stride,
            count,
        };
        self.reads = super::SelectionReads::Pending(PendingSelection {
            generation: self.selection_generation,
            entries: (first..end).map(|camera| (camera, select)).collect(),
            layout: SelectionLayout::Packed {
                stride: key_stride * size_of::<u32>(),
            },
            handles: vec![result],
        });
        for _camera in first..end {
            #[cfg(all(test, feature = "wgpu"))]
            super::fire_if_armed("selection camera submitted");
        }
        Some(work)
    }
}

/// Prepared packed cell-selection work with retained input/output allocations.
pub struct CornerLaunch {
    arena: Arc<crate::pyramid::FrameArena>,
    keys: cubecl::server::Handle,
    key_len: usize,
    geometry: kernels::CellSelectGeometry,
    first: usize,
    key_stride: usize,
    count: usize,
}
impl CornerLaunch {
    /// Enqueue the prepared cell-selection kernel.
    ///
    /// # Safety
    /// Use the preparation client and stream. Source pyramids must still contain
    /// the prepared frame, and no input/output buffer may be accessed concurrently.
    pub unsafe fn run<R: Runtime>(self, client: &ComputeClient<R>) {
        let Self {
            arena,
            keys,
            key_len,
            geometry,
            first,
            key_stride,
            count,
        } = self;
        kernels::launch_fast_cell_batch(
            client,
            arena.bindings()[0],
            (&keys, key_len),
            geometry,
            first * arena.strides[0],
            arena.strides[0],
            key_stride,
            count,
        );
    }
}
