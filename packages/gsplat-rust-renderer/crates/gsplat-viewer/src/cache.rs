//! Store-owned uploads and per-view lifetimes; viewer identities stop at this boundary.
use crate::renderer::{GaussianDrawData, GaussianRenderer, RenderView};
use gsplat_core::{Camera, RenderOptions, native::NativeSplats};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

struct CachedView {
    last_frame: u64,
    generation: u64,
    render: Arc<Mutex<RenderView>>,
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct BatchKey {
    pub view_id: re_viewer_context::ViewId,
    pub entity: re_log_types::EntityPath,
    pub instruction: re_sdk_types::blueprint::components::VisualizerInstructionId,
    pub instance: usize,
    pub row: (re_log_types::TimeInt, re_sdk_types::RowId),
}
pub(crate) struct Batch<'a> {
    pub key: BatchKey,
    pub cloud: &'a NativeSplats<'a>,
    pub generation: u64,
    pub camera: &'a Camera,
    pub options: RenderOptions,
}
struct CachedScene {
    last_frame: u64,
    scene: Arc<gsplat_core::Scene>,
    count: usize,
    bounds: [glam::Vec3; 2],
}
impl CachedScene {
    fn upload(
        core: &gsplat_core::Renderer,
        cloud: &NativeSplats<'_>,
        frame: u64,
    ) -> Result<Self, gsplat_core::Error> {
        let bounds = crate::bounds::from_splats(cloud);
        let scene = core.upload(&cloud.to_core())?;
        re_log::debug!("Uploaded {} Gaussian splats", cloud.centers.len());
        Ok(Self {
            last_frame: frame,
            scene,
            count: cloud.centers.len(),
            bounds,
        })
    }
}
#[derive(Default)]
struct Batches {
    frame: u64,
    scenes: HashMap<u64, CachedScene>,
    views: HashMap<BatchKey, CachedView>,
    native_sorts: HashMap<BatchKey, (re_renderer::SortOrderCache, u64)>,
    bounds: HashMap<re_viewer_context::ViewId, ([glam::Vec3; 2], u64)>,
}
fn record_bounds(
    views: &mut HashMap<re_viewer_context::ViewId, ([glam::Vec3; 2], u64)>,
    frame: u64,
    view_id: re_viewer_context::ViewId,
    bounds: [glam::Vec3; 2],
) -> bool {
    let entry = match views.entry(view_id) {
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert((bounds, frame));
            return true;
        }
        std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
    };
    let previous = entry.0;
    entry.0 = if entry.1 == frame {
        [previous[0].min(bounds[0]), previous[1].max(bounds[1])]
    } else {
        bounds
    };
    entry.1 = frame;
    entry.0 != previous
}
/// Store-owned cache: begin_frame also runs when the entity/view disappears.
#[derive(Default)]
pub(crate) struct GpuCache(Mutex<Batches>);
impl GpuCache {
    pub(crate) fn view_bounds(&self, id: re_viewer_context::ViewId) -> Option<[glam::Vec3; 2]> {
        self.0
            .lock()
            .expect("GPU cache")
            .bounds
            .get(&id)
            .map(|(bounds, _)| *bounds)
    }
}
impl re_viewer_context::Cache for GpuCache {
    fn name(&self) -> &'static str {
        "ComputeGaussianSplats3D GPU"
    }
    fn begin_frame(&mut self) {
        let cache = self.0.get_mut().expect("GPU cache");
        cache
            .views
            .retain(|_, entry| entry.last_frame == cache.frame);
        // Blueprint activation can omit all compute instructions for one frame.
        // Keep shared uploads across that transition, then evict genuinely unused data.
        cache
            .scenes
            .retain(|_, entry| entry.last_frame + 1 >= cache.frame);
        cache.bounds.retain(|_, (_, frame)| *frame == cache.frame);
        cache
            .native_sorts
            .retain(|_, (_, seen)| *seen == cache.frame);
        cache.frame += 1;
    }
    fn purge_memory(&mut self) {
        *self.0.get_mut().expect("GPU cache") = Batches::default();
    }
}
impl re_byte_size::MemUsageTreeCapture for GpuCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        let cache = self.0.lock().expect("GPU cache");
        // CPU handles only; GPU allocations are accounted by wgpu.
        re_byte_size::MemUsageTree::Bytes(
            (cache.scenes.len() * size_of::<CachedScene>()
                + cache.views.len() * size_of::<CachedView>()) as u64,
        )
    }
}

impl GpuCache {
    pub(crate) fn fallback_bounds(
        &mut self,
        view_id: re_viewer_context::ViewId,
        cloud: &NativeSplats<'_>,
        world_from_local: glam::Affine3A,
    ) -> bool {
        let cache = self.0.get_mut().expect("GPU cache");
        record_bounds(
            &mut cache.bounds,
            cache.frame,
            view_id,
            crate::bounds::transformed(crate::bounds::from_splats(cloud), world_from_local),
        )
    }
    pub(crate) fn native_sort(&mut self, key: BatchKey) -> re_renderer::SortOrderCache {
        let cache = self.0.get_mut().expect("GPU cache");
        let (sort, seen) = cache.native_sorts.entry(key).or_default();
        *seen = cache.frame;
        sort.clone()
    }
    pub(crate) fn prepare(
        &mut self,
        ctx: &re_renderer::RenderContext,
        batch: Batch<'_>,
    ) -> Result<(GaussianDrawData, bool), gsplat_core::Error> {
        let Batch {
            key,
            cloud,
            generation,
            camera,
            options,
        } = batch;
        let renderer = ctx
            .renderer::<GaussianRenderer>()
            .expect("registered renderer");
        let core = renderer
            .core
            .as_ref()
            .ok_or(gsplat_core::Error::Capabilities)?;
        // Check the wire dimensions before allocating a potentially large f32 conversion.
        let coefficients = if cloud.sh.is_empty() {
            1
        } else {
            (cloud.degree.min(3) + 1).pow(2)
        };
        let required = cloud.centers.len() as u64 * u64::from((12 * coefficients).max(40));
        let limit = ctx
            .device
            .limits()
            .max_storage_buffer_binding_size
            .min(ctx.device.limits().max_buffer_size);
        if required > limit {
            return Err(gsplat_core::Error::Capacity { required, limit });
        }
        let cache = self.0.get_mut().expect("GPU cache");
        let frame = cache.frame;
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.scenes.entry(generation) {
            entry.insert(CachedScene::upload(core, cloud, frame)?);
        }
        let shared = cache.scenes.get_mut(&generation).expect("uploaded scene");
        shared.last_frame = frame;
        let bounds = crate::bounds::transformed(shared.bounds, options.world_from_local);
        record_bounds(&mut cache.bounds, frame, key.view_id, bounds);
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.views.entry(key.clone()) {
            entry.insert(CachedView {
                last_frame: frame,
                generation,
                render: Arc::new(Mutex::new(renderer.create_view(
                    ctx,
                    &shared.scene,
                    shared.count,
                    camera.size,
                )?)),
            });
        }
        let entry = cache.views.get_mut(&key).expect("inserted view");
        entry.last_frame = frame;
        if entry.generation != generation {
            entry.render.lock().expect("render view").replace_scene(
                core,
                &shared.scene,
                shared.count,
            )?;
            entry.generation = generation;
        }
        let retry = entry
            .render
            .lock()
            .expect("render view")
            .prepare(renderer, ctx, camera, options)?;
        Ok((
            GaussianDrawData::new(
                entry.render.clone(),
                *camera,
                options,
                ((bounds[0] + bounds[1]) * 0.5).into(),
            ),
            retry,
        ))
    }
}
