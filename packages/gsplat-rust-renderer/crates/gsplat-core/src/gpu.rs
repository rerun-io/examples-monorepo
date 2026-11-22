//! Raw resource operations; replace this file with `re_renderer` pools and belts upstream.
use wgpu::util::DeviceExt as _;

pub(crate) type Feedback = std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>;

pub(crate) fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::INDIRECT,
    })
}

pub(crate) fn upload(
    device: &wgpu::Device,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: if bytes.is_empty() { &[0; 4] } else { bytes },
        usage,
    })
}

pub(crate) fn uniform(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gsplat parameters"),
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Whether WGSL code (comments ignored) names a subgroup built-in: a `subgroupXxx` function or the
/// `subgroup_invocation_id` / `subgroup_size` built-in values. Decides the web-only `enable subgroups;`.
#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) fn uses_subgroups(source: &str) -> bool {
    let bytes = source.as_bytes();
    let mut code = String::with_capacity(source.len());
    let (mut i, mut depth) = (0, 0usize);
    while i < bytes.len() {
        let next = bytes.get(i + 1).copied();
        if depth == 0 && bytes[i] == b'/' && next == Some(b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if bytes[i] == b'/' && next == Some(b'*') {
            depth += 1;
            i += 2;
        } else if depth > 0 && bytes[i] == b'*' && next == Some(b'/') {
            depth -= 1;
            i += 2;
            code.push(' ');
        } else {
            if depth == 0 {
                code.push(bytes[i] as char);
            }
            i += 1;
        }
    }
    code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|token| {
            matches!(token, "subgroup_invocation_id" | "subgroup_size")
                || token
                    .strip_prefix("subgroup")
                    .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_uppercase()))
        })
}

pub(crate) fn module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
    // WebGPU passes WGSL to the browser, which requires the extension directive;
    // native naga 30 rejects it (gfx-rs/wgpu#5555), so it is added on the web only.
    #[cfg(target_arch = "wasm32")]
    let source = &if uses_subgroups(source) {
        format!("enable subgroups;\n{source}")
    } else {
        source.to_owned()
    };
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gsplat shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

pub(crate) struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
}

pub(crate) fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    bindings: &[(u32, wgpu::BindingType)],
) -> Kernel {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(entry),
        entries: &bindings
            .iter()
            .map(|&(binding, ty)| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions {
            // CPU-reference tests poison shared memory to verify initialization before reads.
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
    });
    Kernel { pipeline, layout }
}

pub(crate) fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gsplat stage"),
        layout,
        entries: &resources
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.clone(),
            })
            .collect::<Vec<_>>(),
    })
}

pub(crate) fn feedback_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame feedback"),
        size: 8,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    })
}

pub(crate) fn feedback(encoder: &wgpu::CommandEncoder, buffer: &wgpu::Buffer) -> Feedback {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    encoder.map_buffer_on_submit(buffer, wgpu::MapMode::Read, .., move |result| {
        tx.send(result).ok();
    });
    rx
}

pub(crate) fn read_feedback(
    buffer: &wgpu::Buffer,
    receiver: &Feedback,
) -> Option<Result<[u32; 2], crate::Error>> {
    let status = match receiver.try_recv() {
        Err(std::sync::mpsc::TryRecvError::Empty) => return None,
        Err(error) => Err(error.into()),
        Ok(status) => status.map_err(crate::Error::from),
    };
    let result = status.and_then(|()| {
        buffer
            .get_mapped_range(..)
            .map(|data| *bytemuck::from_bytes::<[u32; 2]>(&data))
            .map_err(crate::Error::from)
    });
    buffer.unmap();
    Some(result)
}

#[cfg(test)]
mod subgroup_detection_tests {
    use super::uses_subgroups;

    #[test]
    fn comments_do_not_enable_subgroups_and_builtins_do() {
        assert!(!uses_subgroups(
            "// no subgroup built-ins here\n/* subgroupAdd /* nested subgroup_size */ still comment */ fn f() {}"
        ));
        assert!(!uses_subgroups("var<private> my_subgroup_count: u32;"));
        assert!(uses_subgroups("let x = subgroupInclusiveAdd(v);"));
        assert!(uses_subgroups(
            "@compute fn f(@builtin(subgroup_size) width: u32) {} // comment"
        ));
        assert!(uses_subgroups(
            "/* a */ let s = subgroupExclusiveAdd(1u); /* b */"
        ));
    }

    #[test]
    fn module_variants_get_the_extension_only_when_they_use_subgroups() {
        assert!(
            crate::kernels::sources(false)
                .iter()
                .all(|s| !uses_subgroups(s))
        );
        let with = crate::kernels::sources(true);
        assert_eq!(with.iter().filter(|s| uses_subgroups(s)).count(), 2);
    }
}
