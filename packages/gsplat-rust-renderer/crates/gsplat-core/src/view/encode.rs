//! Frame uniforms and the ordered forward compute passes.
use super::{ViewDispatch, ViewState};
use crate::gpu::{self, bind};
use crate::kernels::Kernels;
use crate::primitives::dispatch::dispatch;
use crate::{Camera, Error, RenderMode, RenderOptions, Scene, Target};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Uniforms {
    view: [[f32; 4]; 4],
    camera: [f32; 4],
    pinhole: [f32; 4],
    clamp_limits: [f32; 4],
    image: [u32; 4],
    scene: [u32; 4],
    background: [f32; 4],
    options: Flags,
    coeff0: [f32; 4],
    coeff1: [f32; 4],
    lens: [u32; 4],
    camera_limits: [f32; 4],
    padding: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Flags {
    log_splat_scale: f32,
    mip: u32,
    has_min_scale: u32,
    padding: u32,
}
impl Uniforms {
    fn new(camera: &Camera, options: &RenderOptions, scene: &Scene, tiles: glam::UVec2) -> Self {
        let focal = camera.focal();
        let center = camera.center_uv * camera.size.as_vec2();
        let (clamps, radial_limit) = camera.clamp_limits();
        let coefficients = camera.model.coefficients();
        Self {
            view: (camera.world_to_local() * glam::Mat4::from(options.world_from_local))
                .to_cols_array_2d(),
            camera: options
                .world_from_local
                .inverse()
                .transform_point3(camera.position)
                .extend(0.0)
                .to_array(),
            pinhole: [focal.x, focal.y, center.x, center.y],
            clamp_limits: clamps.to_array(),
            image: [camera.size.x, camera.size.y, tiles.x, tiles.y],
            scene: [scene.n, scene.degree, (scene.degree + 1).pow(2), 0],
            background: options.background.extend(0.0).to_array(),
            options: Flags {
                log_splat_scale: options.splat_scale.ln(),
                mip: u32::from(options.render_mode == RenderMode::Mip),
                has_min_scale: u32::from(scene.has_min_scale),
                padding: 0,
            },
            coeff0: coefficients[..4].try_into().unwrap(),
            coeff1: coefficients[4..].try_into().unwrap(),
            lens: [camera.model.kind(), 0, 0, 0],
            camera_limits: [camera.half_max_render_fov(), radial_limit, 0.0, 0.0],
            padding: [0; 4],
        }
    }
}
impl ViewState {
    pub(crate) fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        kernels: &Kernels,
        camera: &Camera,
        options: &RenderOptions,
        target: Target,
    ) -> Result<(), Error> {
        let raster_kind = target.layout(camera.size)?;
        let tiles = glam::UVec2::new(camera.size.x.div_ceil(16), camera.size.y.div_ceil(16));
        let tile_count = tiles
            .x
            .checked_mul(tiles.y)
            .ok_or(Error::Input("too many tiles"))?;
        self.resize(queue, kernels, tile_count)?;
        let index = self
            .frames
            .iter()
            .position(|frame| frame.receiver.is_none())
            .ok_or(Error::Busy(
                "three frames are pending; poll view feedback before rendering again",
            ))?;
        let frame = &mut self.frames[index];
        let scene = &self.scene;
        let uniforms = Uniforms::new(camera, options, scene, tiles);
        gpu::write(queue, &frame.uniform, bytemuck::bytes_of(&uniforms));
        let raster = kernels.raster(raster_kind);
        if frame
            .raster
            .as_ref()
            .is_none_or(|(cached, _)| cached != &target)
        {
            let mut bindings = vec![
                (0, frame.uniform.as_entire_binding()),
                (
                    1,
                    self.intersections
                        .sort
                        .output(self.bits)
                        .1
                        .as_entire_binding(),
                ),
                (2, self.offsets.as_entire_binding()),
                (3, self.projected.as_entire_binding()),
                (raster_kind.binding(), target.resource()),
            ];
            if let Target::TextureDepth { depth, .. } = &target {
                bindings.push((7, self.depth_sort.output(32).0.as_entire_binding()));
                bindings.push((8, wgpu::BindingResource::TextureView(depth)));
            }
            let group = bind(&self.device, raster, &bindings);
            frame.raster = Some((target, group));
        }
        let timestamps = |start, end| self.timing.pass(start, end);
        encoder.clear_buffer(&self.counts, 0, None);
        dispatch(
            encoder,
            &kernels.project_forward,
            &frame.projection.forward,
            scene.n.div_ceil(256),
            timestamps(Some(0), 1),
        );
        self.dispatches
            .prepare(encoder, &kernels.prepare, timestamps(None, 2));
        self.depth_sort
            .encode(encoder, kernels, 32, timestamps(None, 3));
        self.dispatches.dispatch(
            encoder,
            ViewDispatch::Visible,
            &kernels.gather,
            &self.gather,
            None,
        );
        self.scan.encode(encoder, kernels, timestamps(None, 4));
        self.dispatches.dispatch(
            encoder,
            ViewDispatch::Visible,
            &kernels.project_visible,
            &frame.projection.visible,
            timestamps(None, 5),
        );
        self.dispatches.dispatch(
            encoder,
            ViewDispatch::Visible,
            &kernels.map_tiles,
            &frame.mapping.tiles,
            timestamps(None, 6),
        );
        self.intersections
            .sort
            .encode(encoder, kernels, self.bits, timestamps(None, 7));
        encoder.clear_buffer(&self.offsets, 0, None);
        self.dispatches.dispatch(
            encoder,
            ViewDispatch::Intersections,
            &kernels.tile_offsets,
            &frame.mapping.offsets,
            timestamps(None, 8),
        );
        self.dispatches.dispatch(
            encoder,
            ViewDispatch::Raster,
            raster,
            &frame.raster.as_ref().unwrap().1,
            timestamps(None, 9),
        );
        encoder.copy_buffer_to_buffer(&self.counts, 0, &frame.readback, 0, 8);
        frame.receiver = Some(gpu::feedback(encoder, &frame.readback));
        frame.capacity = self.intersections.capacity;
        self.pending.push_back(index);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn frame_uniforms_match_renderer_upload_alignment() {
        assert_eq!(std::mem::size_of::<super::Uniforms>(), 256);
    }
}
