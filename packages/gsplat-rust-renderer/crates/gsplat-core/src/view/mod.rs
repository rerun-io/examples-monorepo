//! Per-view scratch, reusable frame bindings, and bounded asynchronous feedback.
use crate::gpu::{self, bind, storage, uniform};
mod encode;
use crate::kernels::{CHECKS_PER_ITER, Kernels, WG};
use crate::primitives::dispatch::{CountSlot, DispatchPlan, Dispatches};
use crate::primitives::{RadixSort, Scan};
use crate::{Error, FrameStats, Scene, Target};
use encode::Uniforms;
use std::collections::VecDeque;
use std::sync::Arc;

struct Intersections {
    capacity: u32,
    sort: RadixSort,
}

impl Intersections {
    fn new(device: &wgpu::Device, kernels: &Kernels, counts: &wgpu::Buffer, capacity: u32) -> Self {
        let keys = storage(device, "intersection tile keys", u64::from(capacity) * 4);
        let ids = storage(device, "intersection global IDs", u64::from(capacity) * 4);
        let sort = RadixSort::new(
            device,
            kernels,
            capacity,
            CountSlot::Intersections,
            (&keys, &ids),
            counts,
            TILE_SORT,
        );
        Self { capacity, sort }
    }
}

struct MappingGroups {
    tiles: wgpu::BindGroup,
    offsets: wgpu::BindGroup,
}

struct FrameSlot {
    uniform: wgpu::Buffer,
    /// The level-of-detail run list read by the splat forward pass (`lod::RUN_UNIFORM_BYTES`; 16 B for hexels).
    runs: wgpu::Buffer,
    readback: wgpu::Buffer,
    forward: wgpu::BindGroup,
    mapping: MappingGroups,
    raster: Option<(Target, wgpu::BindGroup)>,
}

struct PendingFrame {
    slot: usize,
    receiver: gpu::Feedback,
    capacity: u32,
}

struct Scratch {
    counts: wgpu::Buffer,
    projected: wgpu::Buffer,
    depth_sort: RadixSort,
    scan: Scan,
    gather: wgpu::BindGroup,
    intersections: Intersections,
    offsets: wgpu::Buffer,
    bits: u32,
}

impl Scratch {
    fn mapping(
        &self,
        device: &wgpu::Device,
        kernels: &Kernels,
        scene: &Scene,
        uniform: &wgpu::Buffer,
    ) -> MappingGroups {
        MappingGroups {
            tiles: bind(
                device,
                &kernels.stages(scene).map_tiles.layout,
                &[
                    (0, uniform.as_entire_binding()),
                    (1, self.counts.as_entire_binding()),
                    (
                        2,
                        self.depth_sort.output(DEPTH_KEY_BITS).1.as_entire_binding(),
                    ),
                    (5, self.projected.as_entire_binding()),
                    (6, self.scan.output().as_entire_binding()),
                    (7, self.intersections.sort.output(0).0.as_entire_binding()),
                    (8, self.intersections.sort.output(0).1.as_entire_binding()),
                ],
            ),
            offsets: bind(
                device,
                &kernels.tile_offsets.layout,
                &[
                    (0, uniform.as_entire_binding()),
                    (1, self.counts.as_entire_binding()),
                    (
                        7,
                        self.intersections
                            .sort
                            .output(self.bits)
                            .0
                            .as_entire_binding(),
                    ),
                    (9, self.offsets.as_entire_binding()),
                ],
            ),
        }
    }
}

/// Scratch for one scene/view pair. Several views may encode before one submit.
/// Poll feedback before reuse; at most three unconsumed frames may be in flight.
pub struct ViewState {
    device: wgpu::Device,
    scene: Arc<Scene>,
    scratch: Scratch,
    tile_count: u32,
    dispatches: Dispatches,
    frames: [FrameSlot; 3],
    pending: VecDeque<PendingFrame>,
    required_capacity: u32,
    limit: u64,
    timing: Option<wgpu::QuerySet>,
    overflow_events: u32,
    /// This view's evaluated transforms + opacity when the scene is animated.
    deform: Option<crate::motion_gpu::Deform>,
    /// Rows the forward pass projects (`None` = all rows); see [`ViewState::set_runs`].
    runs: Option<Vec<crate::lod::Run>>,
}

/// Byte offset of the per-primitive tile counts in the counts buffer (`TILE_HITS` in counts.wgsl).
const TILE_HITS: u64 = 256;

fn tile_hits(counts: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: counts,
        offset: TILE_HITS,
        size: None,
    })
}

const VISIBLE: u32 = 0;

const INTERSECTIONS: u32 = 1;

const DEPTH_SORT: u32 = 2;

/// Depth key width. Gather and `map_tiles` read the global ids in depth order from
/// `depth_sort.output(DEPTH_KEY_BITS).1`, whose buffer depends on the pass count.
const DEPTH_KEY_BITS: u32 = 32;

const TILE_SORT: u32 = 4;

const SCAN: u32 = 6;

fn plans(n: u32, capacity: u32) -> Vec<DispatchPlan> {
    let mut plans = vec![
        DispatchPlan::new(CountSlot::Visible, n, WG, 1),
        DispatchPlan::new(CountSlot::Intersections, capacity, WG * CHECKS_PER_ITER, 1),
    ];
    plans.extend(RadixSort::plans(CountSlot::Visible, n));
    plans.extend(RadixSort::plans(CountSlot::Intersections, capacity));
    plans.extend(Scan::plans(CountSlot::Visible, n));
    plans
}

impl ViewState {
    /// Retained scratch bytes, excluding scene buffers and buffers held only by bind groups.
    pub fn gpu_byte_size(&self) -> u64 {
        let scratch = &self.scratch;
        [
            &scratch.counts,
            &scratch.projected,
            &scratch.offsets,
            scratch.scan.output(),
        ]
        .into_iter()
        .map(wgpu::Buffer::size)
        .sum::<u64>()
            + scratch.depth_sort.gpu_byte_size()
            + scratch.intersections.sort.gpu_byte_size()
            + self.dispatches.gpu_byte_size()
            + self
                .frames
                .iter()
                .map(|frame| frame.uniform.size() + frame.runs.size() + frame.readback.size())
                .sum::<u64>()
            + self
                .deform
                .as_ref()
                .map_or(0, crate::motion_gpu::Deform::gpu_byte_size)
    }

    pub(crate) fn new(
        device: &wgpu::Device,
        kernels: &Kernels,
        scene: &Arc<Scene>,
        initial_capacity: u32,
        limit: u64,
    ) -> Result<Self, Error> {
        let capacity = initial_capacity.max(1);
        let n = scene.n;
        let projected = u64::from(n.max(1)) * scene.projected_bytes();
        let counts_bytes = TILE_HITS + u64::from(n.max(1)) * 4;
        for required in [u64::from(capacity) * 4, projected, counts_bytes] {
            if required > limit {
                return Err(Error::Capacity { required, limit });
            }
        }
        let counts = storage(
            device,
            "visible and intersection counts, global tile counts",
            counts_bytes,
        );
        let ids = storage(device, "compact IDs", u64::from(n) * 4);
        let depths = storage(device, "depth keys", u64::from(n) * 4);
        let gathered = storage(device, "sorted tile counts", u64::from(n) * 4);
        let visible = CountSlot::Visible;
        let depth_sort = RadixSort::new(
            device,
            kernels,
            n,
            visible,
            (&depths, &ids),
            &counts,
            DEPTH_SORT,
        );
        let gather = bind(
            device,
            &kernels.gather.layout,
            &[
                (1, counts.as_entire_binding()),
                (2, depth_sort.output(DEPTH_KEY_BITS).1.as_entire_binding()),
                (3, tile_hits(&counts)),
                (4, gathered.as_entire_binding()),
            ],
        );
        let scratch = Scratch {
            projected: storage(device, "projected primitives", projected),
            depth_sort,
            scan: Scan::new(device, kernels, n, visible, &gathered, &counts, SCAN),
            gather,
            intersections: Intersections::new(device, kernels, &counts, capacity),
            offsets: storage(device, "tile ranges", 8),
            bits: 1,
            counts,
        };
        let deform = match (scene.motion(), scene.transforms()) {
            (Some(motion), Some(canonical)) => Some(crate::motion_gpu::Deform::new(
                device,
                &kernels.trigs_deform,
                motion,
                canonical,
                n,
                limit,
            )?),
            _ => None,
        };
        let stages = kernels.stages(scene);
        let frames = std::array::from_fn(|_| {
            let uniform = uniform(device, &[0; size_of::<Uniforms>() / 4]);
            let runs = if scene.hexels() {
                gpu::uniform(device, &[0; 4])
            } else {
                gpu::uniform(device, &vec![0; crate::lod::RUN_UNIFORM_BYTES / 4])
            };
            let forward = {
                let mut bindings = scene.bindings();
                if !scene.hexels() {
                    bindings.push((10, runs.as_entire_binding()));
                }
                if let Some(deform) = &deform {
                    // The projector reads this view's evaluated rows instead of the canonical ones.
                    for (binding, resource) in &mut bindings {
                        match binding {
                            1 => *resource = deform.transforms.as_entire_binding(),
                            2 => *resource = deform.opacity.as_entire_binding(),
                            _ => {}
                        }
                    }
                }
                bindings.extend([
                    (0, uniform.as_entire_binding()),
                    (4, ids.as_entire_binding()),
                    (5, depths.as_entire_binding()),
                    (6, scratch.counts.as_entire_binding()),
                    (8, scratch.projected.as_entire_binding()),
                ]);
                bind(device, &stages.forward.layout, &bindings)
            };
            let mapping = scratch.mapping(device, kernels, scene, &uniform);
            FrameSlot {
                uniform,
                runs,
                forward,
                mapping,
                raster: None,
                readback: gpu::feedback_buffer(device),
            }
        });
        Ok(Self {
            device: device.clone(),
            scene: scene.clone(),
            dispatches: Dispatches::new(
                device,
                &kernels.prepare,
                &scratch.counts,
                &plans(n, capacity),
            ),
            scratch,
            tile_count: 1,
            frames,
            pending: VecDeque::new(),
            required_capacity: capacity,
            limit,
            timing: None,
            overflow_events: 0,
            deform,
            runs: None,
        })
    }

    /// Rows the forward pass projects from the next encode on: `Some(runs)` = only these global rows (sorted,
    /// disjoint, inside the scene, at most [`crate::lod::MAX_RUNS`]), `None` = all rows. Picking ids stay global rows.
    pub fn set_runs(&mut self, runs: Option<&[crate::lod::Run]>) -> Result<(), Error> {
        if let Some(runs) = runs {
            if self.scene.hexels() {
                return Err(Error::Input("run lists select splat rows only"));
            }
            if runs.len() > crate::lod::MAX_RUNS {
                return Err(Error::Input("run list exceeds lod::MAX_RUNS"));
            }
            let mut end = 0u64;
            for run in runs {
                let first = u64::from(run.first);
                if first < end || first + u64::from(run.count) > u64::from(self.scene.n) {
                    return Err(Error::Input(
                        "run list must be sorted, disjoint and inside the scene",
                    ));
                }
                end = first + u64::from(run.count);
            }
        }
        self.runs = runs.map(<[_]>::to_vec);
        Ok(())
    }

    /// Rows the forward pass projects (all rows without a run list).
    pub fn selected_rows(&self) -> u32 {
        self.runs
            .as_ref()
            .map_or(self.scene.n, |runs| runs.iter().map(|r| r.count).sum())
    }

    fn resize(
        &mut self,
        queue: &wgpu::Queue,
        kernels: &Kernels,
        tile_count: u32,
    ) -> Result<(), Error> {
        let scratch = &mut self.scratch;
        let grow = self.required_capacity > scratch.intersections.capacity;
        if !grow && tile_count == self.tile_count {
            return Ok(());
        }
        if !self.pending.is_empty() {
            return Err(Error::Busy(
                "consume pending feedback before resizing a view",
            ));
        }
        if u64::from(tile_count) * 8 > self.limit {
            return Err(Error::Capacity {
                required: u64::from(tile_count) * 8,
                limit: self.limit,
            });
        }
        if grow {
            scratch.intersections = Intersections::new(
                &self.device,
                kernels,
                &scratch.counts,
                self.required_capacity,
            );
        }
        let resize_offsets = u64::from(tile_count) * 8 > scratch.offsets.size();
        if resize_offsets {
            let tiles = u64::from(tile_count) + u64::from(tile_count.div_ceil(4));
            scratch.offsets = storage(
                &self.device,
                "tile ranges",
                (tiles * 8).min(self.limit / 8 * 8),
            );
        }
        let bits = 32 - tile_count.leading_zeros();
        let rebind = grow || resize_offsets || bits.div_ceil(4) % 2 != scratch.bits.div_ceil(4) % 2;
        self.tile_count = tile_count;
        scratch.bits = bits;
        let capacity = scratch.intersections.capacity;
        self.dispatches
            .update(queue, &plans(self.scene.n, capacity));
        if !rebind {
            return Ok(());
        }
        for frame in &mut self.frames {
            frame.mapping = scratch.mapping(&self.device, kernels, &self.scene, &frame.uniform);
            frame.raster = None;
        }
        Ok(())
    }

    /// Whether this view still has submitted feedback to consume.
    pub fn has_pending_frames(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Whether a frame slot is free to encode another frame before feedback arrives.
    pub fn has_free_frame(&self) -> bool {
        self.pending.len() < self.frames.len()
    }

    /// Nonblocking completed-frame counts. An overflow preserves the target and requests a rerender.
    pub fn poll_feedback(&mut self) -> Result<Option<FrameStats>, Error> {
        self.device.poll(wgpu::PollType::Poll)?;
        let mut latest = None;
        while let Some(pending) = self.pending.front() {
            let frame = &self.frames[pending.slot];
            let Some(counts) = gpu::read_feedback(&frame.readback, &pending.receiver) else {
                break;
            };
            let capacity = pending.capacity;
            self.pending.pop_front();
            let counts = counts?;
            if counts[0] & 0x8000_0000 != 0 {
                return Err(Error::IntersectionOverflow);
            }
            let overflow = counts[1] > capacity;
            self.overflow_events = self.overflow_events.saturating_add(u32::from(overflow));
            if u64::from(counts[1]) * 4 > self.limit {
                return Err(Error::Capacity {
                    required: u64::from(counts[1]) * 4,
                    limit: self.limit,
                });
            }
            let maximum = (self.limit / 4).min(u64::from(u32::MAX)) as u32;
            self.required_capacity = self
                .required_capacity
                .max(counts[1].saturating_add(counts[1].div_ceil(4)).min(maximum));
            latest = Some(FrameStats {
                visible: counts[0],
                intersections: counts[1],
                intersection_capacity: capacity,
                overflow_events: self.overflow_events,
                needs_rerender: overflow,
            });
        }
        Ok(latest)
    }

    /// Diagnostic timestamps need ten slots; wait for completion before resolving on Metal.
    pub fn set_timestamp_queries(&mut self, queries: Option<wgpu::QuerySet>) -> Result<(), Error> {
        if let Some(q) = &queries
            && (!self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                || !matches!(q.ty(), wgpu::QueryType::Timestamp)
                || q.count() < crate::QUERY_COUNT)
        {
            return Err(Error::Input(
                "stage profiling requires TIMESTAMP_QUERY and ten timestamp slots",
            ));
        }
        self.timing = queries;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{gpu, upload};
    use crate::{Error, RenderOptions, Renderer, Target};

    #[test]
    fn tile_hits_offset_matches_counts_wgsl() {
        assert_eq!(
            super::TILE_HITS % 256,
            0,
            "the portable storage offset alignment"
        );
        let words = super::TILE_HITS / 4;
        assert!(
            include_str!("../../shader/counts.wgsl")
                .contains(&format!("const TILE_HITS = {words}u;"))
        );
    }

    #[test]
    #[ignore = "integration: GPU"]
    fn feedback_recovers_after_capacity_error_and_bounds_pending_frames() {
        let (device, queue) = &gpu();
        let renderer = Renderer::new(device).unwrap();
        let scene = renderer
            .upload(&crate::test_utils::splat([0.0, 0.0, 2.0], -1.0, 2.0))
            .unwrap();
        let mut view = renderer.create_view(&scene, 1).unwrap();
        let mut camera = crate::test_utils::pinhole_camera(64);
        let target = upload(device, &vec![[0.0f32; 4]; 64 * 64]);
        let render = |encoder: &mut wgpu::CommandEncoder,
                      view: &mut crate::ViewState,
                      camera: &crate::Camera| {
            renderer.render(
                queue,
                encoder,
                view,
                camera,
                &RenderOptions::default(),
                Target::Float(target.clone()),
            )
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        render(&mut encoder, &mut view, &camera).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let limit = view.limit;
        view.limit = 4; // Reproduce an oversized completed frame without allocating GB of splats.
        assert!(matches!(view.poll_feedback(), Err(Error::Capacity { .. })));
        view.limit = limit;
        assert!(view.poll_feedback().unwrap().is_none());
        camera.position.z = 10.0;
        let mut encoder = device.create_command_encoder(&Default::default());
        for z in [10.0, 0.0, 10.0] {
            camera.position.z = z;
            assert!(view.has_free_frame());
            render(&mut encoder, &mut view, &camera).unwrap();
        }
        assert!(!view.has_free_frame());
        assert!(render(&mut encoder, &mut view, &camera).is_err());
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(view.poll_feedback().unwrap().unwrap().intersections, 0);
        assert!(view.poll_feedback().unwrap().is_none());
        // Inject callback failures after the real mapping completes. Both errors
        // must release the pending record and its slot before returning.
        for map_error in [true, false] {
            let mut encoder = device.create_command_encoder(&Default::default());
            render(&mut encoder, &mut view, &camera).unwrap();
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            if map_error {
                sender.send(Err(wgpu::BufferAsyncError)).unwrap();
            }
            drop(sender);
            view.pending.front_mut().unwrap().receiver = receiver;
            let error = view.poll_feedback().unwrap_err();
            if map_error {
                assert!(matches!(error, Error::Readback(_)));
            } else {
                assert!(matches!(error, Error::Disconnected(_)));
            }
            assert!(!view.has_pending_frames());
        }
        // All slots are reusable after draining, including the earlier failed frame.
        let mut encoder = device.create_command_encoder(&Default::default());
        render(&mut encoder, &mut view, &camera).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(view.poll_feedback().unwrap().unwrap().visible, 0);
    }

    #[test]
    #[ignore = "integration: GPU"]
    fn projected_hexels_above_the_binding_limit_are_a_capacity_error() {
        let (device, _) = &gpu();
        let kernels = crate::kernels::Kernels::new(device);
        let hexels = crate::Hexels {
            variant: crate::HexelVariant::V1,
            params: vec![0.0; 10 * 20],
        };
        // 800 parameter bytes fit an 800-byte binding; 10 x 84 projected bytes do not.
        let scene = std::sync::Arc::new(crate::Scene::upload_hexels(device, &hexels, 800).unwrap());
        assert!(matches!(
            super::ViewState::new(device, &kernels, &scene, 1, 800),
            Err(Error::Capacity {
                required: 840,
                limit: 800
            })
        ));
    }

    #[test]
    #[ignore = "integration: GPU"]
    fn raster_dispatch_rejects_tile_axes_above_the_portable_limit() {
        let (device, queue) = &gpu();
        let renderer = Renderer::new(device).unwrap();
        let scene = renderer
            .upload(&crate::test_utils::splat([0.0, 0.0, 2.0], -1.0, 2.0))
            .unwrap();
        let mut view = renderer.create_view(&scene, 1).unwrap();
        let mut camera = crate::test_utils::pinhole_camera(64);
        for size in [
            glam::UVec2::new(65_535 * 16, 1),
            glam::UVec2::new(1, 65_535 * 16),
        ] {
            camera.size = size;
            renderer.prepare_view(queue, &mut view, &camera).unwrap();
            camera.size = size + glam::UVec2::ONE;
            assert!(matches!(
                renderer.prepare_view(queue, &mut view, &camera),
                Err(Error::Input(_))
            ));
        }
    }
}
