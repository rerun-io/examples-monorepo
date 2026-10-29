//! Frame uniforms and the ordered forward compute passes.
use super::{INTERSECTIONS, PendingFrame, VISIBLE, ViewState};
use crate::gpu::{self, bind};
use crate::kernels::{Kernels, TILE_WIDTH, WG};
use crate::primitives::dispatch::dispatch_direct;
use crate::scene::Inputs;
use crate::{Camera, CameraModel, Error, RenderMode, RenderOptions, Scene, Target};
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
    padding: u32,
    padding2: [u32; 2],
    half_max_render_fov: f32,
    radial_limit: f32,
    padding3: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct Uniforms {
    view_from_obj: [[f32; 4]; 4],
    camera_position_in_obj: [f32; 4],
    pinhole_params: [f32; 4],
    jacobian_clamp_limits: [f32; 4],
    image: ImageInfo,
    scene: SceneInfo,
    background: [f32; 4],
    options: RenderFlags,
    coefficients: [f32; 8],
    lens: LensInfo,
    /// Camera-to-world rotation (xyzw) for world-space hexel normals.
    camera_rotation: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RenderFlags {
    log_scale_offset: f32,
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
            view_from_obj: (camera.view_from_world() * glam::Mat4::from(options.world_from_local))
                .to_cols_array_2d(),
            camera_position_in_obj: options
                .world_from_local
                .inverse()
                .transform_point3(camera.position)
                .extend(0.0)
                .to_array(),
            pinhole_params: [focal.x, focal.y, center.x, center.y],
            jacobian_clamp_limits: clamps.to_array(),
            image: ImageInfo {
                width: camera.size.x,
                height: camera.size.y,
                tiles_x: tiles.x,
                tiles_y: tiles.y,
            },
            scene: match scene.inputs {
                Inputs::Splats { degree, .. } => SceneInfo {
                    splats: scene.n,
                    sh_degree: degree,
                    coefficients: (degree + 1).pow(2),
                    padding: 0,
                },
                // Hexel shaders read the variant (1-3) and the per-hexel stride here.
                Inputs::Hexels { variant, .. } => SceneInfo {
                    splats: scene.n,
                    sh_degree: variant as u32 + 1,
                    coefficients: variant.stride() as u32,
                    padding: 0,
                },
            },
            background: options.background.extend(0.0).to_array(),
            options: RenderFlags {
                log_scale_offset: options.splat_scale.ln(),
                mip: u32::from(options.render_mode == RenderMode::Mip),
                has_min_scale: u32::from(matches!(
                    scene.inputs,
                    Inputs::Splats {
                        has_min_scale: true,
                        ..
                    }
                )),
                padding: 0,
            },
            coefficients,
            lens: LensInfo {
                kind: camera.model.kind(),
                padding: 0,
                padding2: [0; 2],
                half_max_render_fov: camera.half_max_render_fov(),
                radial_limit,
                padding3: [0.0; 2],
            },
            camera_rotation: camera.rotation.to_array(),
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
        let tiles = glam::UVec2::new(size.x.div_ceil(TILE_WIDTH), size.y.div_ceil(TILE_WIDTH));
        if tiles.x > 65_535 || tiles.y > 65_535 {
            return Err(Error::Input("raster tile axis exceeds 65535 workgroups"));
        }
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
        if self.scene.hexels() && camera.model != CameraModel::Pinhole {
            return Err(Error::Input("hexel scenes render with a pinhole camera"));
        }
        let stages = kernels.stages(&self.scene);
        let raster = stages.raster(raster_kind)?;
        let tiles = self.prepare(queue, kernels, camera.size)?;
        let scratch = &self.scratch;
        let index = self
            .frames
            .iter()
            .enumerate()
            .find_map(|(slot, _)| {
                (!self.pending.iter().any(|pending| pending.slot == slot)).then_some(slot)
            })
            .ok_or(Error::Busy(
                "three frames are pending; poll view feedback before rendering again",
            ))?;
        let frame = &mut self.frames[index];
        let scene = &self.scene;
        let uniforms = Uniforms::new(camera, options, scene, tiles);
        queue.write_buffer(&frame.uniform, 0, bytemuck::bytes_of(&uniforms));
        frame.raster.take_if(|(cached, _)| cached != &target);
        let (_, raster_group) = frame.raster.get_or_insert_with(|| {
            let (_, ids) = scratch.intersections.sort.output(scratch.bits);
            let mut bindings = vec![
                (0, frame.uniform.as_entire_binding()),
                (1, ids.as_entire_binding()),
                (2, scratch.offsets.as_entire_binding()),
                (3, scratch.projected.as_entire_binding()),
                (7, scratch.counts.as_entire_binding()),
                (raster_kind.binding(), target.resource()),
            ];
            match &target {
                Target::TextureDepth { depth, .. } => {
                    bindings.push((8, wgpu::BindingResource::TextureView(depth)));
                }
                Target::TextureDepthId { depth, id, .. } => {
                    bindings.extend([
                        (8, wgpu::BindingResource::TextureView(depth)),
                        // The 32-bit depth sort ends in its input buffers: global rows in depth order.
                        (11, scratch.depth_sort.output(32).1.as_entire_binding()),
                        (12, wgpu::BindingResource::TextureView(id)),
                    ]);
                }
                Target::Surface { depth, normal, .. } => {
                    bindings.extend([
                        (9, depth.as_entire_binding()),
                        (10, normal.as_entire_binding()),
                    ]);
                }
                _ => {}
            }
            let group = bind(&self.device, &raster.layout, &bindings);
            (target, group)
        });
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
        dispatch_direct(
            encoder,
            &stages.forward,
            &frame.projection.forward,
            scene.n.div_ceil(WG),
            timestamps(Some(0), 1),
        );
        self.dispatches
            .prepare(encoder, &kernels.prepare, timestamps(None, 2));
        scratch
            .depth_sort
            .encode(encoder, kernels, &self.dispatches, 32, timestamps(None, 3));
        self.dispatches
            .dispatch_indirect(encoder, VISIBLE, &kernels.gather, &scratch.gather, None);
        scratch
            .scan
            .encode(encoder, kernels, &self.dispatches, timestamps(None, 4));
        self.dispatches.dispatch_indirect(
            encoder,
            VISIBLE,
            &stages.visible,
            &frame.projection.visible,
            timestamps(None, 5),
        );
        self.dispatches.dispatch_indirect(
            encoder,
            VISIBLE,
            &stages.map_tiles,
            &frame.mapping.tiles,
            timestamps(None, 6),
        );
        scratch.intersections.sort.encode(
            encoder,
            kernels,
            &self.dispatches,
            scratch.bits,
            timestamps(None, 7),
        );
        encoder.clear_buffer(&scratch.offsets, 0, Some(u64::from(self.tile_count) * 8));
        self.dispatches.dispatch_indirect(
            encoder,
            INTERSECTIONS,
            &kernels.tile_offsets,
            &frame.mapping.offsets,
            timestamps(None, 8),
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gsplat raster"),
                timestamp_writes: timestamps(None, 9),
            });
            pass.set_pipeline(&raster.pipeline);
            pass.set_bind_group(0, &*raster_group, &[]);
            pass.dispatch_workgroups(tiles.x, tiles.y, 1);
        }
        encoder.copy_buffer_to_buffer(&scratch.counts, 0, &frame.readback, 0, 8);
        self.pending.push_back(PendingFrame {
            slot: index,
            receiver: gpu::feedback(encoder, &frame.readback),
            capacity: scratch.intersections.capacity,
        });
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
