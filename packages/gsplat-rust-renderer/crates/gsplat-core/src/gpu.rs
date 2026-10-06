//! Resource helpers and cached GPU-written dispatch arguments.
use wgpu::util::DeviceExt as _;

pub(crate) fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
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
pub(crate) fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            // Scan/sort initialize their scratch before barriers; raster writes
            // each live batch element and its counters before any shared read.
            // Primitive CPU-reference tests poison shared memory to enforce this.
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    })
}
pub(crate) fn bind(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    resources: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gsplat stage"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &resources
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.clone(),
            })
            .collect::<Vec<_>>(),
    })
}
pub(crate) fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    groups: u32,
    timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
) {
    if groups == 0 && timestamp_writes.is_none() {
        return;
    }
    let groups = groups.max(1);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    let x = groups.min(65535);
    pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
}

/// Plans are [count index, capacity, divisor, multiplier]; MAX selects guarded raster work.
pub(crate) struct Dispatches {
    plans: wgpu::Buffer,
    args: wgpu::Buffer,
    group: wgpu::BindGroup,
    len: u32,
}
impl Dispatches {
    pub fn new(
        device: &wgpu::Device,
        kernel: &wgpu::ComputePipeline,
        counts: &wgpu::Buffer,
        plans: &[[u32; 4]],
    ) -> Self {
        let plan_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dispatch plans"),
            contents: bytemuck::cast_slice(plans),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let args = storage(device, "indirect dispatches", plans.len() as u64 * 16);
        let group = bind(
            device,
            kernel,
            &[
                (0, counts.as_entire_binding()),
                (1, plan_buffer.as_entire_binding()),
                (2, args.as_entire_binding()),
            ],
        );
        Self {
            plans: plan_buffer,
            args,
            group,
            len: plans.len() as u32,
        }
    }
    pub fn update(&self, queue: &wgpu::Queue, plans: &[[u32; 4]]) {
        assert_eq!(plans.len(), self.len as usize);
        queue.write_buffer(&self.plans, 0, bytemuck::cast_slice(plans));
    }
    pub fn prepare(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernel: &wgpu::ComputePipeline,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        dispatch(
            encoder,
            kernel,
            &self.group,
            self.len.div_ceil(64),
            timestamp_writes,
        );
    }
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        index: u32,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes,
        });
        self.dispatch_in_pass(&mut pass, index, pipeline, group);
    }
    pub fn dispatch_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        index: u32,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups_indirect(&self.args, u64::from(index) * 16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{gpu, read, upload};
    #[test]
    #[ignore = "integration: GPU"]
    fn wrapped_intersection_count_cannot_enable_raster() {
        let (device, queue) = gpu();
        let counts = upload(&device, &[0u32, u32::MAX - 4]);
        let source = format!(
            "{}\n@compute @workgroup_size(1) fn test_counter() {{ add_intersections(8u); }}",
            include_str!("../shaders/counts.wgsl")
        );
        let kernel = pipeline(&device, &module(&device, &source), "test_counter");
        let group = bind(&device, &kernel, &[(6, counts.as_entire_binding())]);
        let prepare = pipeline(
            &device,
            &module(&device, include_str!("../shaders/dispatch.wgsl")),
            "prepare",
        );
        let dispatches = Dispatches::new(&device, &prepare, &counts, &[[u32::MAX, 16, 1, 0]]);
        let mut encoder = device.create_command_encoder(&Default::default());
        dispatch(&mut encoder, &kernel, &group, 1, None);
        dispatches.prepare(&mut encoder, &prepare, None);
        queue.submit([encoder.finish()]);
        assert_eq!(read::<u32>(&device, &queue, &counts, 2), [0x8000_0000, 3]);
        assert_eq!(
            read::<u32>(&device, &queue, &dispatches.args, 4),
            [1, 0, 1, 0]
        );
    }
}
