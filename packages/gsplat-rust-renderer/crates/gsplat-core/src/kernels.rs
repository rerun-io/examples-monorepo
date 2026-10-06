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
    pub projection: [wgpu::ComputePipeline; 2],
    pub mapping: [wgpu::ComputePipeline; 3],
    pub raster: [wgpu::ComputePipeline; 4],
    pub scan: [wgpu::ComputePipeline; 2],
    pub sort: [wgpu::ComputePipeline; 5],
    pub prepare: wgpu::ComputePipeline,
}
impl Kernels {
    pub fn new(device: &wgpu::Device) -> Self {
        let [projection, mapping, raster, scan, sort, prepare] =
            sources().map(|s| module(device, &s));
        let depth = module(device, &depth_source());
        Self {
            projection: ["project_forward", "project_visible"]
                .map(|entry| pipeline(device, &projection, entry)),
            mapping: ["gather", "map_tiles", "tile_offsets"]
                .map(|entry| pipeline(device, &mapping, entry)),
            raster: [
                pipeline(device, &raster, "raster_float"),
                pipeline(device, &raster, "raster_packed"),
                pipeline(device, &raster, "raster_texture"),
                pipeline(device, &depth, "raster_texture"),
            ],
            scan: ["scan", "add_offsets"].map(|entry| pipeline(device, &scan, entry)),
            sort: [
                "count_keys",
                "reduce_counts",
                "scan_counts",
                "scan_add",
                "scatter",
            ]
            .map(|entry| pipeline(device, &sort, entry)),
            prepare: pipeline(device, &prepare, "prepare"),
        }
    }
}

// Specialize only the optional viewer path; normal raster entry points do not bind
// or read depth, and retain their original workgroup storage and arithmetic.
fn depth_source() -> String {
    format!("{}\n{}", include_str!("../shaders/render_common.wgsl"), include_str!("../shaders/raster.wgsl"))
        .replace("// OPTIONAL_DEPTH_DECLARATIONS", "@group(0) @binding(7) var<storage, read> sorted_depth: array<u32>;\n@group(0) @binding(8) var out_depth: texture_storage_2d<r32float, write>;\nvar<workgroup> batch_depth: array<f32, 256>;")
        .replace("// OPTIONAL_DEPTH_LOAD", "batch_depth[lid] = bitcast<f32>(sorted_depth[isect_ids[start + lid]]);")
        .replace("// OPTIONAL_DEPTH_INIT", "var expected_depth = 0.0;")
        .replace("// OPTIONAL_DEPTH_ACCUMULATE", "expected_depth += batch_depth[t] * alpha * transmittance;")
        .replace("// OPTIONAL_DEPTH_STORE", "if inside { textureStore(out_depth, pix, vec4f(expected_depth / max(1.0 - transmittance, 1e-8))); }")
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
                workgroup_bytes <= crate::renderer::REQUIRED_WORKGROUP_STORAGE_BYTES,
                "shader needs {workgroup_bytes} workgroup bytes, exceeding the device guard"
            );
        }
    }
}
