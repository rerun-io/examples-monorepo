//! Raw resource operations; replace this file with re_renderer pools and belts upstream.
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
pub(crate) fn module(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
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
pub(crate) fn feedback(encoder: &mut wgpu::CommandEncoder, buffer: &wgpu::Buffer) -> Feedback {
    let (tx, rx) = std::sync::mpsc::channel();
    encoder.map_buffer_on_submit(buffer, wgpu::MapMode::Read, .., move |result| {
        let _ = tx.send(result);
    });
    rx
}
pub(crate) fn read_feedback(
    buffer: &wgpu::Buffer,
    receiver: &Feedback,
) -> Option<Result<[u32; 2], crate::Error>> {
    let status = match receiver.try_recv() {
        Err(std::sync::mpsc::TryRecvError::Empty) => return None,
        Err(error) => Err(crate::Error::Readback(error.to_string())),
        Ok(status) => status.map_err(|error| crate::Error::Readback(error.to_string())),
    };
    let result = status.and_then(|()| {
        buffer
            .get_mapped_range(..)
            .map(|data| *bytemuck::from_bytes::<[u32; 2]>(&data))
            .map_err(|error| crate::Error::Readback(error.to_string()))
    });
    buffer.unmap();
    Some(result)
}
