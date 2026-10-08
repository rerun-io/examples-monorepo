//! Per-view scratch, reusable frame bindings, and bounded asynchronous feedback.
use crate::gpu::{self, bind, storage, uniform};
mod encode;
use crate::kernels::Kernels;
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
        let ids = storage(device, "intersection compact IDs", u64::from(capacity) * 4);
        let sort = RadixSort::new(
            device,
            kernels,
            capacity,
            CountSlot::Intersections,
            &keys,
            &ids,
            counts,
        );
        Self { capacity, sort }
    }
}

struct ProjectionGroups {
    forward: wgpu::BindGroup,
    visible: wgpu::BindGroup,
}
struct MappingGroups {
    tiles: wgpu::BindGroup,
    offsets: wgpu::BindGroup,
}
struct FrameSlot {
    uniform: wgpu::Buffer,
    readback: wgpu::Buffer,
    projection: ProjectionGroups,
    mapping: MappingGroups,
    raster: Option<(Target, wgpu::BindGroup)>,
    receiver: Option<gpu::Feedback>,
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
        uniform: &wgpu::Buffer,
    ) -> MappingGroups {
        MappingGroups {
            tiles: bind(
                device,
                &kernels.map_tiles.layout,
                &[
                    (0, uniform.as_entire_binding()),
                    (1, self.counts.as_entire_binding()),
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
    pending: VecDeque<usize>,
    required_capacity: u32,
    limit: u64,
    timing: Option<wgpu::QuerySet>,
    overflow_events: u32,
}
const VISIBLE: u32 = 0;
const INTERSECTIONS: u32 = 1;
fn plans(n: u32, capacity: u32) -> [DispatchPlan; 2] {
    [
        DispatchPlan::new(CountSlot::Visible, n, 256, 1),
        DispatchPlan::new(CountSlot::Intersections, capacity, 2048, 1),
    ]
}
impl ViewState {
    pub(crate) fn new(
        device: &wgpu::Device,
        kernels: &Kernels,
        scene: &Arc<Scene>,
        initial_capacity: u32,
        limit: u64,
    ) -> Result<Self, Error> {
        let capacity = initial_capacity.max(1);
        if u64::from(capacity) * 4 > limit {
            return Err(Error::Capacity {
                required: u64::from(capacity) * 4,
                limit,
            });
        }
        let n = scene.n;
        let counts = storage(device, "visible and intersection counts", 8);
        let ids = storage(device, "compact IDs", u64::from(n) * 4);
        let depths = storage(device, "depth keys", u64::from(n) * 4);
        let hits = storage(device, "global tile counts", u64::from(n) * 4);
        let gathered = storage(device, "sorted tile counts", u64::from(n) * 4);
        let visible = CountSlot::Visible;
        let scratch = Scratch {
            projected: storage(device, "projected splats", u64::from(n.max(1)) * 40),
            depth_sort: RadixSort::new(device, kernels, n, visible, &depths, &ids, &counts),
            scan: Scan::new(device, kernels, n, visible, &gathered, &counts),
            gather: bind(
                device,
                &kernels.gather.layout,
                &[
                    (1, counts.as_entire_binding()),
                    (2, ids.as_entire_binding()),
                    (3, hits.as_entire_binding()),
                    (4, gathered.as_entire_binding()),
                ],
            ),
            intersections: Intersections::new(device, kernels, &counts, capacity),
            offsets: storage(device, "tile ranges", 8),
            bits: 1,
            counts,
        };
        let frames = std::array::from_fn(|_| {
            let uniform = uniform(device, &vec![0; size_of::<Uniforms>() / 4]);
            let projection = ProjectionGroups {
                forward: bind(
                    device,
                    &kernels.project_forward.layout,
                    &[
                        (0, uniform.as_entire_binding()),
                        (1, scene.transforms.as_entire_binding()),
                        (2, scene.opacity.as_entire_binding()),
                        (3, scene.min_scale.as_entire_binding()),
                        (4, ids.as_entire_binding()),
                        (5, depths.as_entire_binding()),
                        (6, scratch.counts.as_entire_binding()),
                        (7, hits.as_entire_binding()),
                    ],
                ),
                visible: bind(
                    device,
                    &kernels.project_visible.layout,
                    &[
                        (0, uniform.as_entire_binding()),
                        (1, scene.transforms.as_entire_binding()),
                        (2, scene.opacity.as_entire_binding()),
                        (3, scene.min_scale.as_entire_binding()),
                        (4, ids.as_entire_binding()),
                        (6, scratch.counts.as_entire_binding()),
                        (8, scratch.projected.as_entire_binding()),
                        (9, scene.sh.as_entire_binding()),
                    ],
                ),
            };
            let mapping = scratch.mapping(device, kernels, &uniform);
            FrameSlot {
                uniform,
                projection,
                mapping,
                raster: None,
                receiver: None,
                capacity,
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
        })
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
            frame.mapping = scratch.mapping(&self.device, kernels, &frame.uniform);
            frame.raster = None;
        }
        Ok(())
    }
    /// Whether this view still has submitted feedback to consume.
    pub fn has_pending_frames(&self) -> bool {
        !self.pending.is_empty()
    }
    /// Nonblocking completed-frame counts. An overflow preserves the target and requests a rerender.
    pub fn poll_feedback(&mut self) -> Result<Option<FrameStats>, Error> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| Error::Readback(e.to_string()))?;
        let mut latest = None;
        while let Some(&index) = self.pending.front() {
            let frame = &mut self.frames[index];
            let Some(counts) = gpu::read_feedback(
                &frame.readback,
                frame.receiver.as_ref().expect("pending receiver"),
            ) else {
                break;
            };
            self.pending.pop_front();
            frame.receiver = None;
            let counts = counts?;
            if counts[0] & 0x8000_0000 != 0 {
                return Err(Error::IntersectionOverflow);
            }
            let overflow = counts[1] > frame.capacity;
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
                intersection_capacity: frame.capacity,
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
        for _ in 0..3 {
            render(&mut encoder, &mut view, &camera).unwrap();
        }
        assert!(render(&mut encoder, &mut view, &camera).is_err());
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(view.poll_feedback().unwrap().unwrap().intersections, 0);
        assert!(view.poll_feedback().unwrap().is_none());
        // All slots are reusable after draining, including the earlier failed frame.
        let mut encoder = device.create_command_encoder(&Default::default());
        render(&mut encoder, &mut view, &camera).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(view.poll_feedback().unwrap().unwrap().visible, 0);
    }
}
