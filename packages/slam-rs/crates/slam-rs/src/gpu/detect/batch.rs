//! Packed cell selection over contiguous cameras of a shared pyramid arena.

use std::sync::Arc;

use cubecl::prelude::*;

use super::{GpuCornerScan, Selection};
use crate::frontend::detect::CellSelect;
use crate::gpu::kernels;
use kornia_image::Image;

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
    pub(super) fn launch_selection_batch(
        &mut self,
        images: &[Image<u16, 1>],
        selects: &[Option<CellSelect>],
    ) -> Option<cubecl::server::Handle> {
        let first = selects
            .iter()
            .take(images.len())
            .position(Option::is_some)?;
        let select = selects[first]?;
        let end = selects
            .iter()
            .take(images.len())
            .rposition(Option::is_some)?
            + 1;
        if selects[first..end]
            .iter()
            .any(|value| *value != Some(select))
        {
            return None;
        }
        let image = &images[first];
        let (width, height) = (image.width(), image.height());
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
        if images.len() != arena.cameras {
            return None;
        }
        for (camera, image) in images.iter().enumerate().take(end).skip(first) {
            let frame = table.get(camera)?.as_ref()?;
            if image.width() != width
                || image.height() != height
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
        let (alignment, limit) = crate::gpu::submission::binding_limits(&self.client);
        let key_stride = (cells * size_of::<u32>()).next_multiple_of(alignment) / size_of::<u32>();
        let key_len = key_stride.checked_mul(images.len())?;
        if key_len > limit / size_of::<u32>() {
            return None;
        }
        let fits = self.batch.as_ref().is_some_and(|batch| {
            batch.width == width
                && batch.height == height
                && batch.cameras == images.len()
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
                    cameras: images.len(),
                    key_stride,
                })
            }
        };
        let result = batch
            .keys
            .clone()
            .offset_end(((key_len - key_stride * count) * size_of::<u32>()) as u64);
        self.launches.dispatch(
            &self.client,
            crate::gpu::submission::Launch::Corners(CornerLaunch {
                arena,
                keys: batch.keys.clone(),
                key_len,
                geometry,
                first,
                key_stride,
                count,
            }),
        );
        self.selection_stride = Some(key_stride * size_of::<u32>());
        for camera in first..end {
            self.cameras[camera].selection = Selection::Pending(select);
            #[cfg(test)]
            crate::gpu::fire_if_armed("selection camera submitted");
        }
        Some(result)
    }
}

pub(in crate::gpu) struct CornerLaunch {
    arena: Arc<crate::gpu::pyramid::FrameArena>,
    keys: cubecl::server::Handle,
    key_len: usize,
    geometry: kernels::CellSelectGeometry,
    first: usize,
    key_stride: usize,
    count: usize,
}
impl CornerLaunch {
    pub(in crate::gpu) fn run<R: Runtime>(self, client: &ComputeClient<R>) {
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
