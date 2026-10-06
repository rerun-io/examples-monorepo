//! Encode one GPU-driven forward path against shared scenes and independent views.
use crate::gpu::{bind, dispatch};
use crate::kernels::Kernels;
use crate::view::TargetHandle;
use crate::{Camera, Error, RenderMode, RenderOptions, Scene, Splats, Target, ViewState};
use bytemuck::{Pod, Zeroable};
use std::sync::{Arc, mpsc};

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

/// Device-lifetime pipelines. Scene uploads and per-view scratch have independent lifetimes.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limit: u64,
    kernels: Kernels,
}
impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, Error> {
        let limits = device.limits();
        if !device.features().contains(wgpu::Features::SUBGROUP)
            || limits.max_storage_buffers_per_shader_stage < 8
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_workgroup_storage_size < 10_240
        {
            return Err(Error::Capabilities);
        }
        let kernels = Kernels::new(device);
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            limit: limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size),
            kernels,
        })
    }
    pub fn upload(&self, splats: &Splats, mode: RenderMode) -> Result<Arc<Scene>, Error> {
        Ok(Arc::new(Scene::upload(
            &self.device,
            splats,
            mode,
            self.limit,
        )?))
    }
    pub fn create_view(
        &self,
        scene: &Arc<Scene>,
        initial_capacity: u32,
    ) -> Result<ViewState, Error> {
        ViewState::new(
            &self.device,
            &self.kernels,
            scene,
            initial_capacity,
            self.limit,
            size_of::<Uniforms>() / 4,
        )
    }

    /// Encode without submission or CPU waits. Submit, then poll this view's feedback.
    /// Overflow leaves the target intact; rerender after feedback requests more capacity.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &mut ViewState,
        camera: &Camera,
        options: &RenderOptions,
        target: Target<'_>,
    ) -> Result<(), Error> {
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
            return Err(Error::Input("invalid camera or render options"));
        }
        let pixels = u64::from(camera.size.x) * u64::from(camera.size.y);
        let (raster_kind, binding, resource) = match &target {
            Target::Float(buffer) | Target::Packed(buffer) => {
                let float = matches!(target, Target::Float(_));
                let bytes = pixels
                    .checked_mul(if float { 16 } else { 4 })
                    .ok_or(Error::Input("target size overflow"))?;
                if buffer.size() < bytes || !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
                    return Err(Error::Input(
                        "target buffer is too small or lacks STORAGE usage",
                    ));
                }
                (
                    usize::from(!float),
                    if float { 4 } else { 5 },
                    buffer.as_entire_binding(),
                )
            }
            Target::Texture(texture_view) => {
                let texture = texture_view.texture();
                if texture.width() != camera.size.x
                    || texture.height() != camera.size.y
                    || texture.depth_or_array_layers() != 1
                    || texture.mip_level_count() != 1
                    || texture.sample_count() != 1
                    || texture.dimension() != wgpu::TextureDimension::D2
                    || texture.format() != wgpu::TextureFormat::Rgba8Unorm
                    || !texture
                        .usage()
                        .contains(wgpu::TextureUsages::STORAGE_BINDING)
                {
                    return Err(Error::Input(
                        "target requires a matching single-mip rgba8unorm storage texture",
                    ));
                }
                (2, 6, wgpu::BindingResource::TextureView(texture_view))
            }
        };
        let tiles = glam::UVec2::new(camera.size.x.div_ceil(16), camera.size.y.div_ceil(16));
        let tile_count = tiles
            .x
            .checked_mul(tiles.y)
            .ok_or(Error::Input("too many tiles"))?;
        view.resize(&self.queue, &self.kernels, tile_count)?;
        let index = view
            .frames
            .iter()
            .position(|frame| frame.receiver.is_none())
            .ok_or(Error::Input(
                "three frames are pending; poll view feedback before rendering again",
            ))?;
        let frame = &mut view.frames[index];
        let scene = &view.scene;
        let focal = camera.focal();
        let center = camera.center_uv * camera.size.as_vec2();
        let (clamps, radial_limit) = camera.clamp_limits();
        let coefficients = camera.model.coefficients();
        let model = if options.specialize_camera {
            camera.model.kind() as usize
        } else {
            4
        };
        let uniforms = Uniforms {
            view: camera.world_to_local().to_cols_array_2d(),
            camera: camera.position.extend(0.0).to_array(),
            pinhole: [focal.x, focal.y, center.x, center.y],
            clamp_limits: clamps.to_array(),
            image: [camera.size.x, camera.size.y, tiles.x, tiles.y],
            scene: [scene.n, scene.degree, (scene.degree + 1).pow(2), 0],
            background: options.background.extend(0.0).to_array(),
            options: [
                options.splat_scale.ln(),
                f32::from(scene.mode == RenderMode::Mip),
                f32::from(scene.has_min_scale),
                0.0,
            ],
            coeff0: coefficients[..4].try_into().unwrap(),
            coeff1: coefficients[4..].try_into().unwrap(),
            lens: [camera.model.kind(), 0, 0, 0],
            camera_limits: [camera.half_max_render_fov(), radial_limit, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&frame.uniform, 0, bytemuck::bytes_of(&uniforms));
        let raster = &self.kernels.raster[raster_kind];
        if frame
            .raster
            .as_ref()
            .is_none_or(|(cached, _)| !cached.matches(&target))
        {
            let group = bind(
                &self.device,
                raster,
                &[
                    (0, frame.uniform.as_entire_binding()),
                    (
                        1,
                        view.intersections
                            .sort
                            .output(view.bits)
                            .1
                            .as_entire_binding(),
                    ),
                    (2, view.offsets.as_entire_binding()),
                    (3, view.projected.as_entire_binding()),
                    (binding, resource),
                ],
            );
            let handle = match target {
                Target::Float(b) => TargetHandle::Float(b.clone()),
                Target::Packed(b) => TargetHandle::Packed(b.clone()),
                Target::Texture(v) => TargetHandle::Texture(v.clone()),
            };
            frame.raster = Some((handle, group));
        }
        encoder.clear_buffer(&view.counts, 0, None);
        dispatch(
            encoder,
            &self.kernels.projection[model][0],
            &frame.projection[model][0],
            scene.n.div_ceil(256),
        );
        view.dispatches.prepare(encoder, &self.kernels.prepare);
        view.depth_sort.encode(encoder, &self.kernels, 32);
        view.dispatches
            .dispatch(encoder, 0, &self.kernels.mapping[0], &view.gather);
        view.scan.encode(encoder, &self.kernels);
        view.dispatches.dispatch(
            encoder,
            0,
            &self.kernels.projection[model][1],
            &frame.projection[model][1],
        );
        view.dispatches
            .dispatch(encoder, 0, &self.kernels.mapping[1], &frame.mapping[0]);
        view.intersections
            .sort
            .encode(encoder, &self.kernels, view.bits);
        encoder.clear_buffer(&view.offsets, 0, None);
        view.dispatches
            .dispatch(encoder, 1, &self.kernels.mapping[2], &frame.mapping[1]);
        view.dispatches
            .dispatch(encoder, 2, raster, &frame.raster.as_ref().unwrap().1);
        encoder.copy_buffer_to_buffer(&view.counts, 0, &frame.readback, 0, 8);
        let (tx, rx) = mpsc::channel();
        encoder.map_buffer_on_submit(&frame.readback, wgpu::MapMode::Read, .., move |result| {
            let _ = tx.send(result);
        });
        frame.receiver = Some(rx);
        frame.capacity = view.intersections.capacity;
        view.pending.push_back(index);
        Ok(())
    }
}
