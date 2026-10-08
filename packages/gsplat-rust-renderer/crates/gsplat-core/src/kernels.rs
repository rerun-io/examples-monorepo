//! Device-lifetime modules and pipelines; allocation paths never compile shaders.
use crate::gpu::{Kernel, module, pipeline};

pub(crate) fn sources() -> [String; 3] {
    ["scan.wgsl", "sort.wgsl", "dispatch.wgsl"].map(crate::shader::resolve)
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

pub(crate) fn bindings(entry: &str) -> &'static [(u32, wgpu::BindingType)] {
    match entry {
        "scan" => &[(0, R), (1, R), (2, W), (3, W), (4, U)],
        "add_offsets" => &[(0, R), (1, R), (2, W), (4, U)],
        "count_keys" => &[(0, R), (1, R), (3, W), (7, U)],
        "reduce_counts" => &[(0, R), (3, W), (4, W), (7, U)],
        "scan_counts" => &[(0, R), (4, W), (7, U)],
        "scan_add" => &[(0, R), (3, W), (4, W), (7, U)],
        "scatter" => &[(0, R), (1, R), (2, R), (3, W), (5, W), (6, W), (7, U)],
        "prepare" => &[(0, R), (1, R), (2, W)],
        _ => unreachable!("unknown gsplat kernel: {entry}"),
    }
}

pub(crate) struct Kernels {
    pub scan: Kernel,
    pub add_offsets: Kernel,
    pub count_keys: Kernel,
    pub reduce_counts: Kernel,
    pub scan_counts: Kernel,
    pub scan_add: Kernel,
    pub scatter: Kernel,
    pub prepare: Kernel,
}
impl Kernels {
    pub fn new(device: &wgpu::Device) -> Self {
        let [scan, sort, prepare] = sources().map(|s| module(device, &s));
        let kernel = |module, entry| pipeline(device, module, entry, bindings(entry));
        Self {
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
            }
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
