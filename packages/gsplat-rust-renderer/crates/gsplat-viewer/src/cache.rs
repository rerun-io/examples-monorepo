//! Store-owned uploads and per-view lifetimes; viewer identities stop at this boundary.
use crate::renderer::{GaussianDrawData, GaussianRenderer, Prepared, RenderView};
use ahash::HashMap;
use gsplat_core::{Camera, RenderOptions, native::NativeSplats};
use parking_lot::Mutex;
use re_viewer::external::re_log_types::hash::Hash64;
use re_viewer::external::{
    parking_lot, re_log, re_log_types, re_renderer, re_sdk_types, re_viewer_context,
};
use re_viewer_context::external::nohash_hasher::IntMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

struct CachedView {
    last_frame: u64,
    generation: Hash64,
    render: Arc<Mutex<RenderView>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct BatchKey {
    pub(crate) view_id: re_viewer_context::ViewId,
    pub(crate) entity: re_log_types::EntityPath,
    pub(crate) instruction: re_sdk_types::blueprint::components::VisualizerInstructionId,
    pub(crate) instance: usize,
    pub(crate) row: (re_log_types::TimeInt, re_sdk_types::RowId),
}

struct CachedScene {
    last_frame: u64,
    scene: Arc<gsplat_core::Scene>,
    bounds: re_renderer::RobustBounds,
}

#[derive(Default)]
pub(crate) struct GpuCache {
    frame: u64,
    fallbacks: IntMap<Hash64, (crate::fallback::FallbackArrays, u64)>,
    scenes: IntMap<Hash64, CachedScene>,
    views: HashMap<BatchKey, CachedView>,
    native_sorts: HashMap<BatchKey, (re_renderer::SortOrderCache, u64)>,
    bounds: HashMap<re_viewer_context::ViewId, (macaw::BoundingBox, u64)>,
}

fn record_bounds(
    views: &mut HashMap<re_viewer_context::ViewId, (macaw::BoundingBox, u64)>,
    frame: u64,
    view_id: re_viewer_context::ViewId,
    bounds: macaw::BoundingBox,
) -> bool {
    if !bounds.min.is_finite() || !bounds.max.is_finite() {
        return false;
    }
    let entry = match views.entry(view_id) {
        Entry::Vacant(entry) => {
            entry.insert((bounds, frame));
            return true;
        }
        Entry::Occupied(entry) => entry.into_mut(),
    };
    let previous = entry.0;
    entry.0 = if entry.1 == frame {
        previous.union(bounds)
    } else {
        bounds
    };
    entry.1 = frame;
    entry.0 != previous
}

impl re_viewer_context::Cache for GpuCache {
    fn name(&self) -> &'static str {
        "ComputeGaussianSplats3D GPU"
    }

    fn begin_frame(&mut self) {
        self.fallbacks
            .retain(|_, (_, seen)| *seen + 1 >= self.frame);
        self.views.retain(|_, entry| entry.last_frame == self.frame);
        // Blueprint activation can omit all compute instructions for one frame.
        // Keep shared uploads across that transition, then evict genuinely unused data.
        self.scenes
            .retain(|_, entry| entry.last_frame + 1 >= self.frame);
        self.bounds.retain(|_, (_, frame)| *frame == self.frame);
        self.native_sorts.retain(|_, (_, seen)| *seen == self.frame);
        self.frame += 1;
    }

    fn vram_usage(&self) -> re_byte_size::MemUsageTree {
        let mut usage = re_byte_size::MemUsageNode::new();

        for (generation, entry) in &self.scenes {
            usage.add(format!("Scene {generation:?}"), entry.scene.gpu_byte_size());
        }

        for (key, entry) in &self.views {
            usage.add(
                format!(
                    "View {:?} / {} / {:?} / {} / {:?}",
                    key.view_id, key.entity, key.instruction, key.instance, key.row
                ),
                entry.render.lock().gpu_byte_size(),
            );
        }
        re_byte_size::MemUsageTree::Node(usage)
    }

    fn purge_memory(&mut self) {
        *self = Self::default();
    }
}

impl re_byte_size::MemUsageTreeCapture for GpuCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        // GPU allocations are reported separately by Cache::vram_usage. Native sort orders live inside
        // re_renderer::SortOrderCache, which does not expose its size.
        let records = (self.scenes.len() * size_of::<CachedScene>()
            + self.views.len() * size_of::<CachedView>()) as u64;
        let fallback_arrays = self
            .fallbacks
            .values()
            .map(|(arrays, _)| arrays.heap_size_bytes())
            .sum::<u64>();
        re_byte_size::MemUsageNode::new()
            .with_child("Scene and view records", records)
            .with_child("Fallback CPU arrays", fallback_arrays)
            .into_tree()
    }
}

impl GpuCache {
    pub(crate) fn view_bounds(&self, id: re_viewer_context::ViewId) -> Option<macaw::BoundingBox> {
        self.bounds.get(&id).map(|(bounds, _)| *bounds)
    }

    pub(crate) fn fallback(
        &mut self,
        view_id: re_viewer_context::ViewId,
        key: BatchKey,
        generation: Hash64,
        cloud: &NativeSplats<'_>,
        world_from_local: glam::Affine3A,
    ) -> (
        &crate::fallback::FallbackArrays,
        re_renderer::SortOrderCache,
        bool,
    ) {
        let (arrays, seen) = self
            .fallbacks
            .entry(generation)
            .or_insert_with(|| (crate::fallback::FallbackArrays::new(cloud), self.frame));
        *seen = self.frame;
        let changed = record_bounds(
            &mut self.bounds,
            self.frame,
            view_id,
            arrays
                .bounds
                .region_of_interest
                .transform_affine3(&world_from_local),
        );
        let (sort, seen) = self.native_sorts.entry(key).or_default();
        *seen = self.frame;
        (arrays, sort.clone(), changed)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        ctx: &re_renderer::RenderContext,
        renderer: &GaussianRenderer,
        key: BatchKey,
        cloud: &NativeSplats<'_>,
        generation: Hash64,
        camera: &Camera,
        options: RenderOptions,
    ) -> Result<(GaussianDrawData, bool), gsplat_core::Error> {
        re_tracing::profile_function!();
        let core = renderer.core()?;
        core.check_capacity(
            cloud.centers.len(),
            cloud.coefficient_count().isqrt() as u32 - 1,
        )?;
        let frame = self.frame;
        let shared = match self.scenes.entry(generation) {
            Entry::Vacant(entry) => {
                let scene = {
                    re_tracing::profile_scope!("upload");
                    core.upload(&cloud.to_core())?
                };
                re_log::debug!(
                    "Uploaded {} Gaussian splats",
                    re_format::format_uint(cloud.centers.len())
                );
                entry.insert(CachedScene {
                    last_frame: frame,
                    bounds: crate::bounds::from_splats(cloud),
                    scene,
                })
            }
            Entry::Occupied(entry) => entry.into_mut(),
        };
        shared.last_frame = frame;
        let bounds = shared
            .bounds
            .region_of_interest
            .transform_affine3(&options.world_from_local);
        record_bounds(&mut self.bounds, frame, key.view_id, bounds);
        let entry = match self.views.entry(key) {
            Entry::Vacant(entry) => entry.insert(CachedView {
                last_frame: frame,
                generation,
                render: Arc::new(Mutex::new(renderer.create_view(
                    ctx,
                    &shared.scene,
                    camera.size,
                )?)),
            }),
            Entry::Occupied(entry) => entry.into_mut(),
        };
        entry.last_frame = frame;
        let mut render = entry.render.lock();
        if entry.generation != generation {
            render.replace_scene(core, &shared.scene)?;
            entry.generation = generation;
        }
        let prepared = render.prepare(renderer, ctx, camera, options)?;
        Ok((
            GaussianDrawData {
                view: entry.render.clone(),
                camera: *camera,
                options,
                center: bounds.center().into(),
                prepared,
            },
            !matches!(prepared, Prepared::Current),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_byte_size::MemUsageTreeCapture as _;
    use re_viewer_context::Cache as _;

    #[test]
    fn memory_report_counts_fallback_arrays_until_purge() {
        let centers = vec![[0.0_f32; 3]; 1000];
        let sh = vec![[[half::f16::ZERO; 3]; 15]; 1000];
        let cloud = NativeSplats {
            centers: &centers,
            scales: &[],
            quaternions: &[[0.0, 0.0, 0.0, 1.0]; 1000],
            colors: &[u32::MAX],
            sh: &sh,
            degree: 3,
        };
        let arrays = crate::fallback::FallbackArrays::new(&cloud);
        let arrays_bytes = arrays.heap_size_bytes();
        // Centers, scales and rotations alone take 40 bytes per splat.
        assert!(arrays_bytes >= 1000 * 40);
        let mut cache = GpuCache::default();
        cache.fallbacks.insert(Hash64::from_u64(1), (arrays, 0));

        assert_eq!(cache.capture_mem_usage_tree().size_bytes(), arrays_bytes);
        cache.purge_memory();
        assert_eq!(cache.capture_mem_usage_tree().size_bytes(), 0);
    }
}
