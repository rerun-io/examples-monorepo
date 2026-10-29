//! Cached CPU conversion for the stock texture-backed fallback.
use crate::cache::{BatchKey, GpuCache};
use gsplat_core::native::{NativeSplats, at_or_last};
use re_viewer::external::{re_log_types, re_renderer, re_viewer_context};
use re_viewer_context::ViewSystemExecutionError;

pub(crate) struct FallbackArrays {
    pub(crate) centers: Vec<glam::Vec3>,
    pub(crate) scales: Vec<glam::Vec3>,
    pub(crate) rotations: Vec<glam::Quat>,
    pub(crate) colors: Vec<re_renderer::Rgba32Unmul>,
    pub(crate) sh: Vec<[re_renderer::GaussianShCoefficient; 15]>,
    pub(crate) sh_count: usize,
    pub(crate) picking_ids: Vec<re_renderer::PickingLayerInstanceId>,
    pub(crate) bounds: re_renderer::RobustBounds,
}

impl FallbackArrays {
    pub(crate) fn new(cloud: &NativeSplats<'_>) -> Self {
        let centers: Vec<_> = cloud
            .centers
            .iter()
            .copied()
            .map(glam::Vec3::from_array)
            .collect();
        // The stock builder zips centers with scales before clamping its packed data.
        // Expand scales first so a broadcast scale cannot repeat the first center.
        let scales: Vec<_> = (0..cloud.centers.len())
            .map(|index| {
                glam::Vec3::from_array(at_or_last(cloud.scales, index, [0.01; 3]))
                    .max(glam::Vec3::splat(1e-6))
            })
            .collect();
        let rotations: Vec<_> = cloud
            .quaternions
            .iter()
            .copied()
            .map(glam::Quat::from_array)
            .collect();
        let colors: Vec<_> = cloud
            .colors
            .iter()
            .map(|rgba| re_renderer::Rgba32Unmul::from_rgba_unmul_array(rgba.to_be_bytes()))
            .collect();
        let sh: Vec<_> = cloud
            .sh
            .iter()
            .map(|coefficients| coefficients.map(re_renderer::GaussianShCoefficient::from_rgb))
            .collect();
        Self {
            centers,
            scales,
            rotations,
            colors,
            sh,
            sh_count: cloud.coefficient_count() - 1,
            picking_ids: (0..cloud.centers.len() as u64)
                .map(re_renderer::PickingLayerInstanceId)
                .collect(),
            bounds: crate::bounds::from_splats(cloud),
        }
    }

    /// Heap bytes these arrays keep alive between frames, for the cache's memory report.
    pub(crate) fn heap_size_bytes(&self) -> u64 {
        (self.centers.capacity() * size_of::<glam::Vec3>()
            + self.scales.capacity() * size_of::<glam::Vec3>()
            + self.rotations.capacity() * size_of::<glam::Quat>()
            + self.colors.capacity() * size_of::<re_renderer::Rgba32Unmul>()
            + self.sh.capacity() * size_of::<[re_renderer::GaussianShCoefficient; 15]>()
            + self.picking_ids.capacity() * size_of::<re_renderer::PickingLayerInstanceId>())
            as u64
    }
}

/// The stock texture-backed path remains available when compute storage cannot hold a scene;
/// it keeps the stock per-splat picking ids and outlines.
pub(crate) fn native_fallback(
    ctx: &re_viewer_context::ViewContext<'_>,
    query: &re_viewer_context::ViewQuery<'_>,
    key: BatchKey,
    cloud: &NativeSplats<'_>,
    world_from_local: glam::Affine3A,
    generation: re_log_types::hash::Hash64,
) -> Result<re_renderer::QueueableDrawData, ViewSystemExecutionError> {
    re_tracing::profile_function!();
    let masks = query.highlights.entity_outline_mask(key.entity.hash());
    let object = re_renderer::PickingLayerObjectId(key.entity.hash64());
    ctx.viewer_ctx
        .store_context
        .memoizer::<GpuCache, _>(|cache| {
            let (arrays, sort, changed) =
                cache.fallback(query.view_id, key, generation, cloud, world_from_local);
            if changed {
                ctx.egui_ctx().request_repaint();
            }
            let mut builder = re_renderer::GaussianSplatBuilder::new(ctx.render_ctx());
            let mut batch = builder
                .batch("Gaussian compute fallback")
                .world_from_obj(world_from_local)
                .outline_mask_ids(masks.overall)
                .picking_object_id(object)
                .sort_order(sort)
                .add_gaussians(
                    &arrays.centers,
                    &arrays.scales,
                    &arrays.rotations,
                    &arrays.colors,
                    &arrays.sh,
                    arrays.sh_count,
                    &arrays.picking_ids,
                );
            #[expect(clippy::iter_over_hash_type)] // Ranges of one row each never overlap.
            for (instance, mask) in &masks.instances {
                if let Ok(row) = u32::try_from(instance.get())
                    && (row as usize) < arrays.centers.len()
                {
                    batch = batch.push_additional_outline_mask_ids_for_range(
                        re_renderer::Span::from_start_len(row, 1),
                        *mask,
                    );
                }
            }
            drop(batch);
            Ok(builder.into_draw_data()?.into())
        })
}
