//! Store-owned uploads and per-view lifetimes; viewer identities stop at this boundary.
use crate::renderer::{GaussianDrawData, GaussianRenderer, RenderView};
use gsplat_core::{Camera, RenderOptions, native::NativeSplats};
use std::collections::{HashMap, hash_map::Entry};
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
struct CachedScene {
    last_frame: u64,
    scene: Arc<gsplat_core::Scene>,
    count: usize,
    bounds: [glam::Vec3; 2],
}
#[derive(Default)]
pub(crate) struct GpuCache {
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
        Entry::Vacant(entry) => {
            entry.insert((bounds, frame));
            return true;
        }
        Entry::Occupied(entry) => entry.into_mut(),
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
impl re_viewer_context::Cache for GpuCache {
    fn name(&self) -> &'static str {
        "ComputeGaussianSplats3D GPU"
    }
    fn begin_frame(&mut self) {
        self.views.retain(|_, entry| entry.last_frame == self.frame);
        // Blueprint activation can omit all compute instructions for one frame.
        // Keep shared uploads across that transition, then evict genuinely unused data.
        self.scenes
            .retain(|_, entry| entry.last_frame + 1 >= self.frame);
        self.bounds.retain(|_, (_, frame)| *frame == self.frame);
        self.native_sorts.retain(|_, (_, seen)| *seen == self.frame);
        self.frame += 1;
    }
    fn purge_memory(&mut self) {
        *self = Self::default();
    }
}
impl re_byte_size::MemUsageTreeCapture for GpuCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        // CPU handles only; GPU allocations are accounted by wgpu.
        re_byte_size::MemUsageTree::Bytes(
            (self.scenes.len() * size_of::<CachedScene>()
                + self.views.len() * size_of::<CachedView>()) as u64,
        )
    }
}

impl GpuCache {
    pub(crate) fn view_bounds(&self, id: re_viewer_context::ViewId) -> Option<[glam::Vec3; 2]> {
        self.bounds.get(&id).map(|(bounds, _)| *bounds)
    }
    pub(crate) fn fallback_bounds(
        &mut self,
        view_id: re_viewer_context::ViewId,
        cloud: &NativeSplats<'_>,
        world_from_local: glam::Affine3A,
    ) -> bool {
        record_bounds(
            &mut self.bounds,
            self.frame,
            view_id,
            crate::bounds::transformed(crate::bounds::from_splats(cloud), world_from_local),
        )
    }
    pub(crate) fn native_sort(&mut self, key: BatchKey) -> re_renderer::SortOrderCache {
        let (sort, seen) = self.native_sorts.entry(key).or_default();
        *seen = self.frame;
        sort.clone()
    }
    pub(crate) fn prepare(
        &mut self,
        ctx: &re_renderer::RenderContext,
        key: BatchKey,
        cloud: &NativeSplats<'_>,
        generation: u64,
        camera: &Camera,
        options: RenderOptions,
    ) -> Result<(GaussianDrawData, bool), gsplat_core::Error> {
        let renderer = ctx
            .renderer::<GaussianRenderer>()
            .expect("registered renderer");
        let core = renderer.core()?;
        // Check the wire dimensions before allocating a potentially large f32 conversion.
        let required = gsplat_core::Splats::gpu_bytes(
            cloud.centers.len(),
            cloud.coefficient_count().isqrt() as u32 - 1,
        );
        let limit = ctx
            .device
            .limits()
            .max_storage_buffer_binding_size
            .min(ctx.device.limits().max_buffer_size);
        if required > limit {
            return Err(gsplat_core::Error::Capacity { required, limit });
        }
        let frame = self.frame;
        let shared = match self.scenes.entry(generation) {
            Entry::Vacant(entry) => {
                let scene = core.upload(&cloud.to_core())?;
                re_log::debug!("Uploaded {} Gaussian splats", cloud.centers.len());
                entry.insert(CachedScene {
                    last_frame: frame,
                    bounds: crate::bounds::from_splats(cloud),
                    scene,
                    count: cloud.centers.len(),
                })
            }
            Entry::Occupied(entry) => entry.into_mut(),
        };
        shared.last_frame = frame;
        let bounds = crate::bounds::transformed(shared.bounds, options.world_from_local);
        record_bounds(&mut self.bounds, frame, key.view_id, bounds);
        let entry = match self.views.entry(key) {
            Entry::Vacant(entry) => entry.insert(CachedView {
                last_frame: frame,
                generation,
                render: Arc::new(Mutex::new(renderer.create_view(
                    ctx,
                    &shared.scene,
                    shared.count,
                    camera.size,
                )?)),
            }),
            Entry::Occupied(entry) => entry.into_mut(),
        };
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
            GaussianDrawData {
                view: entry.render.clone(),
                camera: *camera,
                options,
                center: ((bounds[0] + bounds[1]) * 0.5).into(),
            },
            retry,
        ))
    }
}
