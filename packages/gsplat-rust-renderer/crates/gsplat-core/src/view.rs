//! Per-view scratch, reusable frame bindings, and bounded asynchronous feedback.
use crate::gpu::{Dispatches, bind, storage, uniform};
use crate::kernels::Kernels;
use crate::primitives::{RadixSort, Scan};
use crate::{Error, FrameStats, Scene, Target};
use std::collections::VecDeque;
use std::sync::{Arc, mpsc};

pub(crate) struct Intersections {
    pub capacity: u32,
    pub keys: wgpu::Buffer,
    pub ids: wgpu::Buffer,
    pub sort: RadixSort,
}
impl Intersections {
    pub fn new(
        device: &wgpu::Device,
        kernels: &Kernels,
        counts: &wgpu::Buffer,
        capacity: u32,
    ) -> Self {
        let keys = storage(device, "intersection tile keys", u64::from(capacity) * 4);
        let ids = storage(device, "intersection compact IDs", u64::from(capacity) * 4);
        let sort = RadixSort::new(device, kernels, capacity, 1, &keys, &ids, counts);
        Self {
            capacity,
            keys,
            ids,
            sort,
        }
    }
}

pub(crate) enum TargetHandle {
    Float(wgpu::Buffer),
    Packed(wgpu::Buffer),
    Texture(wgpu::TextureView),
}
impl TargetHandle {
    pub fn matches(&self, target: &Target<'_>) -> bool {
        match (self, target) {
            (Self::Float(a), Target::Float(b)) | (Self::Packed(a), Target::Packed(b)) => a == *b,
            (Self::Texture(a), Target::Texture(b)) => a == *b,
            _ => false,
        }
    }
}
pub(crate) struct FrameSlot {
    pub uniform: wgpu::Buffer,
    pub readback: wgpu::Buffer,
    pub projection: [[wgpu::BindGroup; 2]; 5],
    pub mapping: [wgpu::BindGroup; 2],
    pub raster: Option<(TargetHandle, wgpu::BindGroup)>,
    pub receiver: Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    pub capacity: u32,
}

/// Scratch for one scene/view pair. Several views may encode before one submit.
/// Poll feedback before reuse; at most three unconsumed frames may be in flight.
pub struct ViewState {
    pub(crate) device: wgpu::Device,
    pub(crate) scene: Arc<Scene>,
    pub(crate) counts: wgpu::Buffer,
    pub(crate) projected: wgpu::Buffer,
    pub(crate) depth_sort: RadixSort,
    pub(crate) scan: Scan,
    pub(crate) gather: wgpu::BindGroup,
    pub(crate) intersections: Intersections,
    pub(crate) offsets: wgpu::Buffer,
    pub(crate) tile_count: u32,
    pub(crate) bits: u32,
    pub(crate) dispatches: Dispatches,
    pub(crate) frames: [FrameSlot; 3],
    pub(crate) pending: VecDeque<usize>,
    pub(crate) required_capacity: u32,
    pub(crate) limit: u64,
    overflow_events: u32,
}
impl ViewState {
    pub(crate) fn new(
        device: &wgpu::Device,
        kernels: &Kernels,
        scene: &Arc<Scene>,
        initial_capacity: u32,
        limit: u64,
        uniform_words: usize,
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
        let projected = storage(device, "projected splats", u64::from(n.max(1)) * 36);
        let gathered = storage(device, "sorted tile counts", u64::from(n) * 4);
        let depth_sort = RadixSort::new(device, kernels, n, 0, &depths, &ids, &counts);
        let scan = Scan::new(device, kernels, n, 0, &gathered, &counts);
        let gather = bind(
            device,
            &kernels.mapping[0],
            &[
                (1, counts.as_entire_binding()),
                (2, ids.as_entire_binding()),
                (3, hits.as_entire_binding()),
                (4, gathered.as_entire_binding()),
            ],
        );
        let intersections = Intersections::new(device, kernels, &counts, capacity);
        let offsets = storage(device, "tile ranges", 8);
        let frames = std::array::from_fn(|_| {
            let uniform = uniform(device, &vec![0; uniform_words]);
            let projection = std::array::from_fn(|model| {
                [
                    bind(
                        device,
                        &kernels.projection[model][0],
                        &[
                            (0, uniform.as_entire_binding()),
                            (1, scene.transforms.as_entire_binding()),
                            (2, scene.opacity.as_entire_binding()),
                            (3, scene.min_scale.as_entire_binding()),
                            (4, ids.as_entire_binding()),
                            (5, depths.as_entire_binding()),
                            (6, counts.as_entire_binding()),
                            (7, hits.as_entire_binding()),
                        ],
                    ),
                    bind(
                        device,
                        &kernels.projection[model][1],
                        &[
                            (0, uniform.as_entire_binding()),
                            (1, scene.transforms.as_entire_binding()),
                            (2, scene.opacity.as_entire_binding()),
                            (3, scene.min_scale.as_entire_binding()),
                            (4, ids.as_entire_binding()),
                            (6, counts.as_entire_binding()),
                            (8, projected.as_entire_binding()),
                            (9, scene.sh.as_entire_binding()),
                        ],
                    ),
                ]
            });
            let mapping = Self::mapping(
                device,
                kernels,
                &uniform,
                &counts,
                &projected,
                &scan,
                &intersections,
                &offsets,
                1,
            );
            FrameSlot {
                uniform,
                projection,
                mapping,
                raster: None,
                receiver: None,
                capacity,
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("frame feedback"),
                    size: 8,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            }
        });
        Ok(Self {
            device: device.clone(),
            scene: scene.clone(),
            counts: counts.clone(),
            projected,
            depth_sort,
            scan,
            gather,
            dispatches: Dispatches::new(
                device,
                &kernels.prepare,
                &counts,
                &[
                    [0, n, 256, 1],
                    [1, capacity, 2048, 1],
                    [u32::MAX, capacity, 1, 0],
                ],
            ),
            intersections,
            offsets,
            tile_count: 1,
            bits: 1,
            frames,
            pending: VecDeque::new(),
            required_capacity: capacity,
            limit,
            overflow_events: 0,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn mapping(
        device: &wgpu::Device,
        kernels: &Kernels,
        uniform: &wgpu::Buffer,
        counts: &wgpu::Buffer,
        projected: &wgpu::Buffer,
        scan: &Scan,
        intersections: &Intersections,
        offsets: &wgpu::Buffer,
        bits: u32,
    ) -> [wgpu::BindGroup; 2] {
        [
            bind(
                device,
                &kernels.mapping[1],
                &[
                    (0, uniform.as_entire_binding()),
                    (1, counts.as_entire_binding()),
                    (5, projected.as_entire_binding()),
                    (6, scan.output().as_entire_binding()),
                    (7, intersections.keys.as_entire_binding()),
                    (8, intersections.ids.as_entire_binding()),
                ],
            ),
            bind(
                device,
                &kernels.mapping[2],
                &[
                    (0, uniform.as_entire_binding()),
                    (1, counts.as_entire_binding()),
                    (7, intersections.sort.output(bits).0.as_entire_binding()),
                    (9, offsets.as_entire_binding()),
                ],
            ),
        ]
    }
    pub(crate) fn resize(
        &mut self,
        queue: &wgpu::Queue,
        kernels: &Kernels,
        tile_count: u32,
    ) -> Result<(), Error> {
        let grow = self.required_capacity > self.intersections.capacity;
        if !grow && tile_count == self.tile_count {
            return Ok(());
        }
        if !self.pending.is_empty() {
            return Err(Error::Input(
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
            self.intersections =
                Intersections::new(&self.device, kernels, &self.counts, self.required_capacity);
        }
        let resize_offsets = u64::from(tile_count) * 8 > self.offsets.size();
        if resize_offsets {
            let tiles = u64::from(tile_count) + u64::from(tile_count.div_ceil(4));
            self.offsets = storage(
                &self.device,
                "tile ranges",
                (tiles * 8).min(self.limit / 8 * 8),
            );
        }
        let bits = 32 - tile_count.leading_zeros();
        let rebind = grow || resize_offsets || bits.div_ceil(4) % 2 != self.bits.div_ceil(4) % 2;
        self.tile_count = tile_count;
        self.bits = bits;
        let capacity = self.intersections.capacity;
        self.dispatches.update(
            queue,
            &[
                [0, self.scene.n, 256, 1],
                [1, capacity, 2048, 1],
                [u32::MAX, capacity, tile_count, 0],
            ],
        );
        if !rebind {
            return Ok(());
        }
        for frame in &mut self.frames {
            frame.mapping = Self::mapping(
                &self.device,
                kernels,
                &frame.uniform,
                &self.counts,
                &self.projected,
                &self.scan,
                &self.intersections,
                &self.offsets,
                self.bits,
            );
            frame.raster = None;
        }
        Ok(())
    }
    /// Nonblocking completed-frame counts. An overflow preserves the target and requests a rerender.
    pub fn poll_feedback(&mut self) -> Result<Option<FrameStats>, Error> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| Error::Readback(e.to_string()))?;
        let mut latest = None;
        while let Some(&index) = self.pending.front() {
            let frame = &mut self.frames[index];
            let status = frame
                .receiver
                .as_ref()
                .expect("pending receiver")
                .try_recv();
            if matches!(status, Err(mpsc::TryRecvError::Empty)) {
                break;
            }
            self.pending.pop_front();
            frame.receiver = None;
            let result = (|| {
                status
                    .map_err(|e| Error::Readback(e.to_string()))?
                    .map_err(|e| Error::Readback(e.to_string()))?;
                let data = frame
                    .readback
                    .get_mapped_range(..)
                    .map_err(|e| Error::Readback(e.to_string()))?;
                Ok(*bytemuck::from_bytes::<[u32; 2]>(&data))
            })();
            frame.readback.unmap();
            let counts = result?;
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
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{gpu, upload};
    use crate::{Camera, CameraModel, Error, RenderMode, RenderOptions, Renderer, Splats, Target};
    use glam::{Quat, UVec2, Vec2, Vec3};

    #[test]
    #[ignore = "integration: GPU"]
    fn feedback_recovers_after_capacity_error_and_bounds_pending_frames() {
        let (device, queue) = gpu();
        let renderer = Renderer::new(&device, &queue).unwrap();
        let scene = renderer
            .upload(
                &Splats {
                    transforms: vec![[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0]],
                    raw_opacities: vec![2.0],
                    sh_coefficients: vec![[0.0; 3]],
                    sh_degree: 0,
                    min_scale: None,
                },
                RenderMode::Default,
            )
            .unwrap();
        let mut view = renderer.create_view(&scene, 1).unwrap();
        let mut camera = Camera {
            model: CameraModel::Pinhole,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            fov_x: 1.0,
            fov_y: 1.0,
            center_uv: Vec2::splat(0.5),
            size: UVec2::splat(64),
        };
        let target = upload(&device, &vec![[0.0f32; 4]; 64 * 64]);
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(
                &mut encoder,
                &mut view,
                &camera,
                &RenderOptions::default(),
                Target::Float(&target),
            )
            .unwrap();
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
            renderer
                .render(
                    &mut encoder,
                    &mut view,
                    &camera,
                    &RenderOptions::default(),
                    Target::Float(&target),
                )
                .unwrap();
        }
        assert!(
            renderer
                .render(
                    &mut encoder,
                    &mut view,
                    &camera,
                    &RenderOptions::default(),
                    Target::Float(&target)
                )
                .is_err()
        );
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(view.poll_feedback().unwrap().unwrap().intersections, 0);
        assert!(view.poll_feedback().unwrap().is_none());
        // All slots are reusable after draining, including the earlier failed frame.
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(
                &mut encoder,
                &mut view,
                &camera,
                &RenderOptions::default(),
                Target::Float(&target),
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(view.poll_feedback().unwrap().unwrap().visible, 0);
    }
}
