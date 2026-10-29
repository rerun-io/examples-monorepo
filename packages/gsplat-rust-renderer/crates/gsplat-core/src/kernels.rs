//! Device-lifetime modules and pipelines; allocation paths never compile shaders.
use crate::gpu::{Kernel, module, pipeline};

pub(crate) const TILE_WIDTH: u32 = 16;

pub(crate) const WG: u32 = 256;

pub(crate) const ELEMENTS_PER_THREAD: u32 = 4;

pub(crate) const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;

pub(crate) const BITS_PER_PASS: u32 = 4;

pub(crate) const BIN_COUNT: u32 = 1 << BITS_PER_PASS;

pub(crate) const CHECKS_PER_ITER: u32 = 8;

pub(crate) fn sources() -> [String; 12] {
    [
        "project.wgsl",
        "map.wgsl",
        "raster.wgsl",
        "scan.wgsl",
        "sort.wgsl",
        "dispatch.wgsl",
        "hexel_project.wgsl",
        "hexel_map.wgsl",
        "hexel_raster.wgsl",
        "image_project.wgsl",
        "image_map.wgsl",
        "image_raster.wgsl",
    ]
    .map(crate::shader::resolve)
}

const U: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Uniform,
    has_dynamic_offset: false,
    min_binding_size: None,
};

const R: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Storage { read_only: true },
    has_dynamic_offset: false,
    min_binding_size: None,
};

pub(crate) const W: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Storage { read_only: false },
    has_dynamic_offset: false,
    min_binding_size: None,
};

const COLOR: wgpu::BindingType = wgpu::BindingType::StorageTexture {
    access: wgpu::StorageTextureAccess::WriteOnly,
    format: wgpu::TextureFormat::Rgba8Unorm,
    view_dimension: wgpu::TextureViewDimension::D2,
};

const DEPTH: wgpu::BindingType = wgpu::BindingType::StorageTexture {
    access: wgpu::StorageTextureAccess::WriteOnly,
    format: wgpu::TextureFormat::R32Float,
    view_dimension: wgpu::TextureViewDimension::D2,
};

const ID: wgpu::BindingType = wgpu::BindingType::StorageTexture {
    access: wgpu::StorageTextureAccess::WriteOnly,
    format: wgpu::TextureFormat::R32Uint,
    view_dimension: wgpu::TextureViewDimension::D2,
};

pub(crate) fn bindings(entry: &str) -> &'static [(u32, wgpu::BindingType)] {
    match entry {
        "project_forward" => &[
            (0, U),
            (1, R),
            (2, R),
            (3, R),
            (4, W),
            (5, W),
            (6, W),
            (7, W),
        ],
        "project_visible" => &[
            (0, U),
            (1, R),
            (2, R),
            (3, R),
            (4, W),
            (6, W),
            (8, W),
            (9, R),
        ],
        "gather" => &[(1, R), (2, R), (3, R), (4, W)],
        "map_tiles" => &[(0, U), (1, R), (5, R), (6, R), (7, W), (8, W)],
        "tile_offsets" => &[(0, U), (1, R), (7, W), (9, W)],
        "raster_float" => &[(0, U), (1, R), (2, R), (3, R), (4, W), (7, R)],
        "raster_packed" => &[(0, U), (1, R), (2, R), (3, R), (5, W), (7, R)],
        "raster_texture" => &[(0, U), (1, R), (2, R), (3, R), (6, COLOR), (7, R)],
        "raster_texture_depth" => &[
            (0, U),
            (1, R),
            (2, R),
            (3, R),
            (6, COLOR),
            (7, R),
            (8, DEPTH),
        ],
        "raster_texture_depth_id" => &[
            (0, U),
            (1, R),
            (2, R),
            (3, R),
            (6, COLOR),
            (7, R),
            (8, DEPTH),
            (11, R),
            (12, ID),
        ],
        "hexel_forward" | "image_forward" => &[(0, U), (1, R), (4, W), (5, W), (6, W), (7, W)],
        "hexel_visible" | "image_visible" => &[(0, U), (1, R), (4, W), (6, W), (8, W)],
        "raster_surface" => &[
            (0, U),
            (1, R),
            (2, R),
            (3, R),
            (4, W),
            (7, R),
            (9, W),
            (10, W),
        ],
        "scan" => &[(0, R), (1, R), (2, W), (3, W), (4, U)],
        "add_offsets" => &[(0, R), (1, R), (2, W), (4, U)],
        "count_keys" => &[(0, R), (1, R), (3, W), (7, U)],
        "reduce_counts" | "scan_add" => &[(0, R), (3, W), (4, W), (7, U)],
        "scan_counts" => &[(0, R), (4, W), (7, U)],
        "scatter" => &[(0, R), (1, R), (2, R), (3, W), (5, W), (6, W), (7, U)],
        "prepare" => &[(0, R), (1, R), (2, W)],
        _ => unreachable!("unknown gsplat kernel: {entry}"),
    }
}

/// Kernels that depend on the primitive; the sort, scan and tile-range kernels are shared.
pub(crate) struct Stages {
    pub forward: Kernel,
    pub visible: Kernel,
    pub map_tiles: Kernel,
    /// Rasterizers in `RasterKind` order; `None` where unsupported.
    rasters: [Option<Kernel>; 6],
}

impl Stages {
    pub fn raster(&self, kind: crate::output::RasterKind) -> Result<&Kernel, crate::Error> {
        self.rasters[kind as usize]
            .as_ref()
            .ok_or(crate::Error::Input(
                "this primitive has no such render target",
            ))
    }
}

pub(crate) struct Kernels {
    pub splats: Stages,
    pub hexels: Stages,
    pub images: Stages,
    pub gather: Kernel,
    pub tile_offsets: Kernel,
    pub scan: Kernel,
    pub add_offsets: Kernel,
    pub count_keys: Kernel,
    pub reduce_counts: Kernel,
    pub scan_counts: Kernel,
    pub scan_add: Kernel,
    pub scatter: Kernel,
    pub prepare: Kernel,
}

/// Raster entry points in `RasterKind` order; every primitive module declares all of them.
const RASTERS: [&str; 6] = [
    "raster_float",
    "raster_packed",
    "raster_texture",
    "raster_texture_depth",
    "raster_surface",
    "raster_texture_depth_id",
];

impl Kernels {
    pub fn new(device: &wgpu::Device) -> Self {
        let [
            projection,
            mapping,
            raster,
            scan,
            sort,
            prepare,
            hexel_projection,
            hexel_mapping,
            hexel_raster,
            image_projection,
            image_mapping,
            image_raster,
        ] = sources().map(|s| module(device, &s));
        let kernel = |module: &wgpu::ShaderModule, entry: &str| {
            pipeline(device, module, entry, bindings(entry))
        };
        // 2D fits keep only the float and packed targets.
        let stages = |project,
                      [forward, visible]: [&str; 2],
                      map,
                      raster: &wgpu::ShaderModule,
                      targets: usize| Stages {
            forward: kernel(project, forward),
            visible: kernel(project, visible),
            map_tiles: kernel(map, "map_tiles"),
            rasters: std::array::from_fn(|i| (i < targets).then(|| kernel(raster, RASTERS[i]))),
        };
        let splat = ["project_forward", "project_visible"];
        let params = ["hexel_forward", "hexel_visible"];
        let image = ["image_forward", "image_visible"];
        Self {
            splats: stages(&projection, splat, &mapping, &raster, 6),
            hexels: stages(&hexel_projection, params, &hexel_mapping, &hexel_raster, 6),
            images: stages(&image_projection, image, &image_mapping, &image_raster, 2),
            gather: kernel(&mapping, "gather"),
            tile_offsets: kernel(&mapping, "tile_offsets"),
            scan: kernel(&scan, "scan"),
            add_offsets: kernel(&scan, "add_offsets"),
            count_keys: kernel(&sort, "count_keys"),
            reduce_counts: kernel(&sort, "reduce_counts"),
            scan_counts: kernel(&sort, "scan_counts"),
            scan_add: kernel(&sort, "scan_add"),
            scatter: kernel(&sort, "scatter"),
            prepare: kernel(&prepare, "prepare"),
        }
    }

    pub fn stages(&self, scene: &crate::Scene) -> &Stages {
        use crate::scene::Inputs;
        match scene.inputs {
            Inputs::Splats { .. } => &self.splats,
            Inputs::Hexels {
                variant: crate::HexelVariant::Image,
                ..
            } => &self.images,
            Inputs::Hexels { .. } => &self.hexels,
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
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            for (index, entry) in module.entry_points.iter().enumerate() {
                let mut used: Vec<_> = module
                    .global_variables
                    .iter()
                    .filter_map(|(handle, variable)| {
                        variable
                            .binding
                            .as_ref()
                            .filter(|_| !info.get_entry_point(index)[handle].is_empty())
                            .map(|binding| binding.binding)
                    })
                    .collect();
                used.sort_unstable();
                let declared: Vec<_> = super::bindings(&entry.name)
                    .iter()
                    .map(|(binding, _)| *binding)
                    .collect();
                assert_eq!(used, declared, "{} layout bindings", entry.name);
            }
            let mut layout = naga::proc::Layouter::default();
            layout.update(module.to_ctx()).unwrap();
            for (handle, ty) in module.types.iter() {
                if ty.name.as_deref() == Some("Uniforms") {
                    assert_eq!(layout[handle].size, 256);
                }
                if ty.name.as_deref() == Some("Splat") {
                    assert_eq!(layout[handle].size, 40);
                }
                if ty.name.as_deref() == Some("Hexel") {
                    assert_eq!(layout[handle].size, 84);
                }
                if ty.name.as_deref() == Some("ImageHexel") {
                    assert_eq!(layout[handle].size, 64);
                }
            }
            let workgroup_bytes: u32 = module
                .global_variables
                .iter()
                .filter(|(_, variable)| variable.space == naga::AddressSpace::WorkGroup)
                .map(|(_, variable)| layout[variable.ty].size)
                .sum();
            assert!(
                workgroup_bytes <= crate::device_caps::REQUIRED_WORKGROUP_STORAGE_BYTES,
                "shader needs {workgroup_bytes} workgroup bytes, exceeding the device guard"
            );
        }
    }
}
