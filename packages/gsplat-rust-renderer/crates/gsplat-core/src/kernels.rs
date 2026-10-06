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
    pub projection: [[wgpu::ComputePipeline; 2]; 5],
    pub mapping: [wgpu::ComputePipeline; 3],
    pub raster: [wgpu::ComputePipeline; 3],
    pub scan: [wgpu::ComputePipeline; 2],
    pub sort: [wgpu::ComputePipeline; 5],
    pub prepare: wgpu::ComputePipeline,
}
impl Kernels {
    pub fn new(device: &wgpu::Device) -> Self {
        let [projection, mapping, raster, scan, sort, prepare] =
            sources().map(|s| module(device, &s));
        Self {
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
            mapping: ["gather", "map_tiles", "tile_offsets"]
                .map(|entry| pipeline(device, &mapping, entry, &[])),
            raster: ["raster_float", "raster_packed", "raster_texture"]
                .map(|entry| pipeline(device, &raster, entry, &[])),
            scan: ["scan", "add_offsets"].map(|entry| pipeline(device, &scan, entry, &[])),
            sort: [
                "count_keys",
                "reduce_counts",
                "scan_counts",
                "scan_add",
                "scatter",
            ]
            .map(|entry| pipeline(device, &sort, entry, &[])),
            prepare: pipeline(device, &prepare, "prepare", &[]),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_native_shader_validates_with_naga() {
        for source in super::sources() {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
