//! Frame uniforms and the ordered forward compute passes.
use super::{INTERSECTIONS, VISIBLE, ViewState};
use crate::gpu::{self, bind};
use crate::kernels::Kernels;
use crate::primitives::dispatch::dispatch;
use crate::{Camera, Error, RenderMode, RenderOptions, Scene, Target};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ImageInfo {
    width: u32,
    height: u32,
    tiles_x: u32,
    tiles_y: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneInfo {
    splats: u32,
    sh_degree: u32,
    coefficients: u32,
    padding: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LensInfo {
    kind: u32,
    padding: [u32; 3],
    half_fov: f32,
    radial_limit: f32,
    padding2: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Uniforms {
    view: [[f32; 4]; 4],
    camera: [f32; 4],
    pinhole: [f32; 4],
    clamp_limits: [f32; 4],
    image: ImageInfo,
    scene: SceneInfo,
    background: [f32; 4],
    options: Flags,
    coeff0: [f32; 4],
    coeff1: [f32; 4],
    lens: LensInfo,
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
            image: ImageInfo {
                width: camera.size.x,
                height: camera.size.y,
                tiles_x: tiles.x,
                tiles_y: tiles.y,
            },
            scene: SceneInfo {
                splats: scene.n,
                sh_degree: scene.degree,
                coefficients: (scene.degree + 1).pow(2),
                padding: 0,
            },
            background: options.background.extend(0.0).to_array(),
            options: Flags {
                log_splat_scale: options.splat_scale.ln(),
                mip: u32::from(options.render_mode == RenderMode::Mip),
                has_min_scale: u32::from(scene.has_min_scale),
                padding: 0,
            },
            coeff0: coefficients[..4].try_into().unwrap(),
            coeff1: coefficients[4..].try_into().unwrap(),
            lens: LensInfo {
                kind: camera.model.kind(),
                padding: [0; 3],
                half_fov: camera.half_max_render_fov(),
                radial_limit,
                padding2: [0.0; 2],
            },
            padding: [0; 4],
        }
    }
}
impl ViewState {
    pub(crate) fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        kernels: &Kernels,
        size: glam::UVec2,
    ) -> Result<glam::UVec2, Error> {
        let tiles = glam::UVec2::new(size.x.div_ceil(16), size.y.div_ceil(16));
        let tile_count = tiles
            .x
            .checked_mul(tiles.y)
            .ok_or(Error::Input("too many tiles"))?;
        self.resize(queue, kernels, tile_count)?;
        Ok(tiles)
    }
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
        let tiles = self.prepare(queue, kernels, camera.size)?;
        let scratch = &self.scratch;
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
        queue.write_buffer(&frame.uniform, 0, bytemuck::bytes_of(&uniforms));
        let raster = kernels.raster(raster_kind);
        if frame
            .raster
            .as_ref()
            .is_none_or(|(cached, _)| cached != &target)
        {
            let (_, ids) = scratch.intersections.sort.output(scratch.bits);
            let mut bindings = vec![
                (0, frame.uniform.as_entire_binding()),
                (1, ids.as_entire_binding()),
                (2, scratch.offsets.as_entire_binding()),
                (3, scratch.projected.as_entire_binding()),
                (7, scratch.counts.as_entire_binding()),
                (raster_kind.binding(), target.resource()),
            ];
            if let Target::TextureDepth { depth, .. } = &target {
                bindings.push((8, wgpu::BindingResource::TextureView(depth)));
            }
            let group = bind(&self.device, &raster.layout, &bindings);
            frame.raster = Some((target, group));
        }
        let timestamps = |start, end| {
            self.timing
                .as_ref()
                .map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: start,
                    end_of_pass_write_index: Some(end),
                })
        };
        encoder.clear_buffer(&scratch.counts, 0, None);
        dispatch(
            encoder,
            &kernels.project_forward,
            &frame.projection.forward,
            scene.n.div_ceil(256),
            timestamps(Some(0), 1),
        );
        self.dispatches
            .prepare(encoder, &kernels.prepare, timestamps(None, 2));
        scratch
            .depth_sort
            .encode(encoder, kernels, 32, timestamps(None, 3));
        self.dispatches
            .dispatch(encoder, VISIBLE, &kernels.gather, &scratch.gather, None);
        scratch.scan.encode(encoder, kernels, timestamps(None, 4));
        self.dispatches.dispatch(
            encoder,
            VISIBLE,
            &kernels.project_visible,
            &frame.projection.visible,
            timestamps(None, 5),
        );
        self.dispatches.dispatch(
            encoder,
            VISIBLE,
            &kernels.map_tiles,
            &frame.mapping.tiles,
            timestamps(None, 6),
        );
        scratch
            .intersections
            .sort
            .encode(encoder, kernels, scratch.bits, timestamps(None, 7));
        encoder.clear_buffer(&scratch.offsets, 0, None);
        self.dispatches.dispatch(
            encoder,
            INTERSECTIONS,
            &kernels.tile_offsets,
            &frame.mapping.offsets,
            timestamps(None, 8),
        );
        dispatch(
            encoder,
            raster,
            &frame.raster.as_ref().unwrap().1,
            self.tile_count,
            timestamps(None, 9),
        );
        encoder.copy_buffer_to_buffer(&scratch.counts, 0, &frame.readback, 0, 8);
        frame.receiver = Some(gpu::feedback(encoder, &frame.readback));
        frame.capacity = scratch.intersections.capacity;
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
