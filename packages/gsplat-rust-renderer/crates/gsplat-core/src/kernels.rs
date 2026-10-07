//! Device-lifetime modules and pipelines; allocation paths never compile shaders.
use crate::gpu::{module, pipeline};

pub(crate) fn sources() -> [String; 6] {
    let common = include_str!("../shaders/render_common.wgsl");
    let scan = include_str!("../shaders/scan_common.wgsl");
    [
        format!(
            "{common}\n{}\n{}\n{}",
            include_str!("../shaders/counts.wgsl"),
            include_str!("../shaders/camera.wgsl"),
            include_str!("../shaders/project.wgsl")
        ),
        format!("{common}\n{}", include_str!("../shaders/map.wgsl")),
        format!("{common}\n{}", include_str!("../shaders/raster.wgsl")),
        format!("{scan}\n{}", include_str!("../shaders/scan.wgsl")),
        format!("{scan}\n{}", include_str!("../shaders/sort.wgsl")),
        include_str!("../shaders/dispatch.wgsl").into(),
    ]
}

pub(crate) struct Kernels {
    pub project_forward: wgpu::ComputePipeline,
    pub project_visible: wgpu::ComputePipeline,
    pub gather: wgpu::ComputePipeline,
    pub map_tiles: wgpu::ComputePipeline,
    pub tile_offsets: wgpu::ComputePipeline,
    pub float: wgpu::ComputePipeline,
    pub packed: wgpu::ComputePipeline,
    pub texture: wgpu::ComputePipeline,
    pub texture_depth: wgpu::ComputePipeline,
    pub scan: wgpu::ComputePipeline,
    pub add_offsets: wgpu::ComputePipeline,
    pub count_keys: wgpu::ComputePipeline,
    pub reduce_counts: wgpu::ComputePipeline,
    pub scan_counts: wgpu::ComputePipeline,
    pub scan_add: wgpu::ComputePipeline,
    pub scatter: wgpu::ComputePipeline,
    pub prepare: wgpu::ComputePipeline,
}
impl Kernels {
    pub fn new(device: &wgpu::Device) -> Self {
        let [projection, mapping, raster, scan, sort, prepare] =
            sources().map(|s| module(device, &s));
        let depth = module(device, &depth_source());
        Self {
            project_forward: pipeline(device, &projection, "project_forward"),
            project_visible: pipeline(device, &projection, "project_visible"),
            gather: pipeline(device, &mapping, "gather"),
            map_tiles: pipeline(device, &mapping, "map_tiles"),
            tile_offsets: pipeline(device, &mapping, "tile_offsets"),
            float: pipeline(device, &raster, "raster_float"),
            packed: pipeline(device, &raster, "raster_packed"),
            texture: pipeline(device, &raster, "raster_texture"),
            texture_depth: pipeline(device, &depth, "raster_texture_depth"),
            scan: pipeline(device, &scan, "scan"),
            add_offsets: pipeline(device, &scan, "add_offsets"),
            count_keys: pipeline(device, &sort, "count_keys"),
            reduce_counts: pipeline(device, &sort, "reduce_counts"),
            scan_counts: pipeline(device, &sort, "scan_counts"),
            scan_add: pipeline(device, &sort, "scan_add"),
            scatter: pipeline(device, &sort, "scatter"),
            prepare: pipeline(device, &prepare, "prepare"),
        }
    }
    pub fn raster(&self, kind: crate::types::RasterKind) -> &wgpu::ComputePipeline {
        use crate::types::RasterKind;
        match kind {
            RasterKind::Float => &self.float,
            RasterKind::Packed => &self.packed,
            RasterKind::Texture => &self.texture,
            RasterKind::TextureDepth => &self.texture_depth,
        }
    }
}
fn depth_source() -> String {
    format!(
        "{}\n{}",
        include_str!("../shaders/render_common.wgsl"),
        include_str!("../shaders/raster_depth.wgsl")
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_native_shader_validates_with_naga() {
        for source in super::sources().into_iter().chain([super::depth_source()]) {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let mut layout = naga::proc::Layouter::default();
            layout.update(module.to_ctx()).unwrap();
            let workgroup_bytes: u32 = module
                .global_variables
                .iter()
                .filter(|(_, variable)| variable.space == naga::AddressSpace::WorkGroup)
                .map(|(_, variable)| layout[variable.ty].size)
                .sum();
            assert!(
                workgroup_bytes <= crate::REQUIRED_WORKGROUP_STORAGE_BYTES,
                "shader needs {workgroup_bytes} workgroup bytes, exceeding the device guard"
            );
        }
    }
}
