//! Own GPU resources once, encode the same forward pipeline for all consumers.
use crate::gpu::{bind, dispatch, pipeline, storage};
use crate::primitives::{RadixSort, Scan};
use crate::{Camera, Capabilities, Error, FrameStats, RenderOptions, Splats, Target};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt as _;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view: [[f32; 4]; 4],
    camera: [f32; 4],
    pinhole: [f32; 4],
    clamp_limits: [f32; 4],
    image: [u32; 4],
    scene: [u32; 4],
    background: [f32; 4],
    options: [f32; 4],
    coeff0: [f32; 4],
    coeff1: [f32; 4],
    lens: [u32; 4],
    camera_limits: [f32; 4],
}
struct Scene {
    transforms: wgpu::Buffer,
    opacity: wgpu::Buffer,
    sh: wgpu::Buffer,
    min_scale: wgpu::Buffer,
    has_min_scale: bool,
    n: u32,
    degree: u32,
    ids: wgpu::Buffer,
    depths: wgpu::Buffer,
    hits: wgpu::Buffer,
    projected: wgpu::Buffer,
    gathered: wgpu::Buffer,
    depth_sort: RadixSort,
    scan: Scan,
}
struct Intersections {
    capacity: u32,
    tile_count: u32,
    keys: wgpu::Buffer,
    ids: wgpu::Buffer,
    offsets: wgpu::Buffer,
    sort: RadixSort,
}

/// A renderer belongs to one wgpu device. Upload once and reuse across frames.
/// Each call captures its uniforms in a new buffer; already-encoded frames keep their values.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    caps: Capabilities,
    projection: [[wgpu::ComputePipeline; 2]; 5],
    gather: wgpu::ComputePipeline,
    map: wgpu::ComputePipeline,
    offsets: wgpu::ComputePipeline,
    raster: [wgpu::ComputePipeline; 3],
    counts: wgpu::Buffer,
    readback: wgpu::Buffer,
    scene: Option<Scene>,
    intersections: Option<Intersections>,
}
impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        caps: Capabilities,
    ) -> Result<Self, Error> {
        let actual = Capabilities::from_device(device)?;
        let caps = Capabilities {
            max_storage_buffer_bytes: caps
                .max_storage_buffer_bytes
                .min(actual.max_storage_buffer_bytes),
        };
        let common = include_str!("../shaders/render_common.wgsl");
        let projection = format!(
            "{common}\n{}\n{}",
            include_str!("../shaders/camera.wgsl"),
            include_str!("../shaders/project.wgsl")
        );
        let mapping = format!("{common}\n{}", include_str!("../shaders/map.wgsl"));
        let raster = format!("{common}\n{}", include_str!("../shaders/raster.wgsl"));
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            caps,
            projection: std::array::from_fn(|i| {
                let kind = if i == 4 { u32::MAX } else { i as u32 };
                ["project_forward", "project_visible"].map(|entry| {
                    pipeline(
                        device,
                        &projection,
                        entry,
                        &[("CAMERA_MODEL", f64::from(kind))],
                    )
                })
            }),
            gather: pipeline(device, &mapping, "gather", &[]),
            map: pipeline(device, &mapping, "map_tiles", &[]),
            offsets: pipeline(device, &mapping, "tile_offsets", &[]),
            raster: ["raster_float", "raster_packed", "raster_texture"]
                .map(|entry| pipeline(device, &raster, entry, &[])),
            counts: storage(device, "visible and intersection counts", 8),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame counts readback"),
                size: 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            scene: None,
            intersections: None,
        })
    }
    /// Validate shapes and upload the raw parameter layout. Invalid numeric splats are culled on GPU.
    pub fn upload(&mut self, splats: &Splats) -> Result<(), Error> {
        let n =
            u32::try_from(splats.transforms.len()).map_err(|_| Error::Input("too many splats"))?;
        if splats.sh_degree > 4
            || splats.raw_opacities.len() != n as usize
            || splats.sh_coefficients.len() != n as usize * (splats.sh_degree as usize + 1).pow(2)
            || splats
                .min_scale
                .as_ref()
                .is_some_and(|s| s.len() != n as usize)
        {
            return Err(Error::Input(
                "inconsistent transform, opacity, SH, or scale-floor lengths",
            ));
        }
        let upload = |label, bytes: &[u8]| -> Result<wgpu::Buffer, Error> {
            if bytes.len() as u64 > self.caps.max_storage_buffer_bytes {
                return Err(Error::Capacity {
                    required: bytes.len() as u64,
                    limit: self.caps.max_storage_buffer_bytes,
                });
            }
            Ok(self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: if bytes.is_empty() { &[0; 4] } else { bytes },
                    usage: wgpu::BufferUsages::STORAGE,
                }))
        };
        let scene = Scene {
            min_scale: upload(
                "3D scale floor",
                bytemuck::cast_slice(splats.min_scale.as_deref().unwrap_or(&[0.0])),
            )?,
            has_min_scale: splats.min_scale.is_some(),
            transforms: upload("raw transforms", bytemuck::cast_slice(&splats.transforms))?,
            opacity: upload("raw opacity", bytemuck::cast_slice(&splats.raw_opacities))?,
            sh: upload(
                "SH RGB coefficients",
                bytemuck::cast_slice(&splats.sh_coefficients),
            )?,
            n,
            degree: splats.sh_degree,
            ids: storage(&self.device, "compact IDs", u64::from(n) * 4),
            depths: storage(&self.device, "depth keys", u64::from(n) * 4),
            hits: storage(&self.device, "global tile counts", u64::from(n) * 4),
            projected: storage(
                &self.device,
                "projected splats (36 bytes)",
                u64::from(n) * 36,
            ),
            gathered: storage(&self.device, "sorted tile counts", u64::from(n) * 4),
            depth_sort: RadixSort::new(&self.device, n, 0, 32),
            scan: Scan::new(&self.device, n, 0),
        };
        self.scene = Some(scene);
        Ok(())
    }
    /// Exact-count mode submits the projection pass and waits for its eight-byte count readback,
    /// then encodes the remaining stages into the caller's encoder. No pixels cross the CPU.
    /// The caller submits that encoder, and owns the lifetime and synchronization of its target.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        options: &RenderOptions,
        target: Target<'_>,
    ) -> Result<FrameStats, Error> {
        if camera.size.x == 0
            || camera.size.y == 0
            || !camera.position.is_finite()
            || !camera.rotation.is_finite()
            || !camera.center_uv.is_finite()
            || !(camera.fov_x > 0.0 && camera.fov_x < std::f64::consts::TAU)
            || !(camera.fov_y > 0.0 && camera.fov_y < std::f64::consts::TAU)
            || !camera.model.coefficients().iter().all(|x| x.is_finite())
            || !options.background.is_finite()
            || !options.splat_scale.is_finite()
            || options.splat_scale <= 0.0
        {
            return Err(Error::Input("invalid camera or background"));
        }
        let scene = self
            .scene
            .as_ref()
            .ok_or(Error::Input("upload splats before rendering"))?;
        let tiles = glam::UVec2::new(camera.size.x.div_ceil(16), camera.size.y.div_ceil(16));
        let focal = camera.focal();
        let center = camera.center_uv * camera.size.as_vec2();
        let (clamps, radial_limit) = camera.clamp_limits();
        let coefficients = camera.model.coefficients();
        let projection_index = if options.specialize_camera {
            camera.model.kind() as usize
        } else {
            4
        };
        let [project, visible] = &self.projection[projection_index];
        let mut uniforms = Uniforms {
            view: Mat4::from(
                glam::Affine3A::from_rotation_translation(camera.rotation, camera.position)
                    .inverse(),
            )
            .to_cols_array_2d(),
            camera: camera.position.extend(0.0).to_array(),
            pinhole: [focal.x, focal.y, center.x, center.y],
            clamp_limits: clamps.to_array(),
            image: [camera.size.x, camera.size.y, tiles.x, tiles.y],
            scene: [scene.n, scene.degree, (scene.degree + 1).pow(2), 0],
            background: options.background.extend(0.0).to_array(),
            options: [
                options.splat_scale.ln(),
                f32::from(options.render_mode == crate::RenderMode::Mip),
                f32::from(scene.has_min_scale),
                0.0,
            ],
            coeff0: coefficients[..4].try_into().expect("four coefficients"),
            coeff1: coefficients[4..].try_into().expect("four coefficients"),
            lens: [camera.model.kind(), 0, 0, 0],
            camera_limits: [camera.half_max_render_fov(), radial_limit, 0.0, 0.0],
        };
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("projection uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut projection = self.device.create_command_encoder(&Default::default());
        projection.clear_buffer(&self.counts, 0, None);
        let group = bind(
            &self.device,
            project,
            &[
                (0, &uniform),
                (1, &scene.transforms),
                (2, &scene.opacity),
                (3, &scene.min_scale),
                (4, &scene.ids),
                (5, &scene.depths),
                (6, &self.counts),
                (7, &scene.hits),
            ],
        );
        dispatch(&mut projection, project, &group, scene.n.div_ceil(256));
        projection.copy_buffer_to_buffer(&self.counts, 0, &self.readback, 0, 8);
        self.queue.submit([projection.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Readback(e.to_string()))?;
        rx.recv()
            .map_err(|e| Error::Readback(e.to_string()))?
            .map_err(|e| Error::Readback(e.to_string()))?;
        let counts: [u32; 2] = {
            let data = self
                .readback
                .get_mapped_range(..)
                .map_err(|e| Error::Readback(e.to_string()))?;
            *bytemuck::from_bytes(&data)
        };
        self.readback.unmap();
        let capacity = counts[1].max(1);
        let tile_count = tiles
            .x
            .checked_mul(tiles.y)
            .ok_or(Error::Input("too many tiles"))?;
        if u64::from(capacity) * 4 > self.caps.max_storage_buffer_bytes {
            return Err(Error::Capacity {
                required: u64::from(capacity) * 4,
                limit: self.caps.max_storage_buffer_bytes,
            });
        }
        if self
            .intersections
            .as_ref()
            .is_none_or(|i| i.capacity < capacity || i.tile_count != tile_count)
        {
            self.intersections = Some(Intersections {
                capacity,
                tile_count,
                keys: storage(
                    &self.device,
                    "intersection tile keys",
                    u64::from(capacity) * 4,
                ),
                ids: storage(
                    &self.device,
                    "intersection compact IDs",
                    u64::from(capacity) * 4,
                ),
                offsets: storage(&self.device, "tile ranges", u64::from(tile_count) * 8),
                sort: RadixSort::new(&self.device, capacity, 1, 32 - tile_count.leading_zeros()),
            });
        }
        let intersections = self
            .intersections
            .as_ref()
            .expect("allocated intersections");
        uniforms.scene[3] = intersections.capacity;
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("render uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        scene
            .depth_sort
            .encode(encoder, &scene.depths, &scene.ids, &self.counts);
        let group = bind(
            &self.device,
            &self.gather,
            &[
                (1, &self.counts),
                (2, &scene.ids),
                (3, &scene.hits),
                (4, &scene.gathered),
            ],
        );
        dispatch(encoder, &self.gather, &group, counts[0].div_ceil(256));
        scene.scan.encode(encoder, &scene.gathered, &self.counts);
        let group = bind(
            &self.device,
            visible,
            &[
                (0, &uniform),
                (1, &scene.transforms),
                (2, &scene.opacity),
                (3, &scene.min_scale),
                (4, &scene.ids),
                (6, &self.counts),
                (8, &scene.projected),
                (9, &scene.sh),
            ],
        );
        dispatch(encoder, visible, &group, counts[0].div_ceil(256));
        let group = bind(
            &self.device,
            &self.map,
            &[
                (0, &uniform),
                (1, &self.counts),
                (5, &scene.projected),
                (6, scene.scan.output()),
                (7, &intersections.keys),
                (8, &intersections.ids),
            ],
        );
        dispatch(encoder, &self.map, &group, counts[0].div_ceil(256));
        intersections.sort.encode(
            encoder,
            &intersections.keys,
            &intersections.ids,
            &self.counts,
        );
        encoder.clear_buffer(&intersections.offsets, 0, None);
        let group = bind(
            &self.device,
            &self.offsets,
            &[
                (0, &uniform),
                (1, &self.counts),
                (7, &intersections.keys),
                (9, &intersections.offsets),
            ],
        );
        dispatch(encoder, &self.offsets, &group, counts[1].div_ceil(2048));
        let (kernel, binding, resource) = match target {
            Target::Float(buffer) => (0, 4, buffer.as_entire_binding()),
            Target::Packed(buffer) => (1, 5, buffer.as_entire_binding()),
            Target::Texture(view) => (2, 6, wgpu::BindingResource::TextureView(view)),
        };
        let kernel = &self.raster[kernel];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster target"),
            layout: &kernel.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: intersections.ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: intersections.offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene.projected.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding, resource },
            ],
        });
        dispatch(encoder, kernel, &group, tile_count);
        Ok(FrameStats {
            visible: counts[0],
            intersections: counts[1],
            intersection_capacity: intersections.capacity,
            overflow_events: 0,
        })
    }
}
