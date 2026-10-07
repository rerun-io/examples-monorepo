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

#[derive(Clone, Copy)]
pub(crate) enum CountSlot {
    Visible,
    Intersections,
    Raster,
}
impl CountSlot {
    pub fn index(self) -> u32 {
        match self {
            Self::Visible => 0,
            Self::Intersections => 1,
            Self::Raster => u32::MAX,
        }
    }
}
/// A count source and the workgroup arithmetic consumed by dispatch.wgsl.
pub(crate) struct DispatchPlan {
    count: CountSlot,
    max: u32,
    per_group: u32,
    multiplier: u32,
}
impl DispatchPlan {
    pub fn new(count: CountSlot, max: u32, per_group: u32, multiplier: u32) -> Self {
        Self {
            count,
            max,
            per_group,
            multiplier,
        }
    }
    fn words(&self) -> [u32; 4] {
        [
            self.count.index(),
            self.max,
            self.per_group,
            self.multiplier,
        ]
    }
}
pub(crate) trait DispatchSlot: Copy {
    fn index(self) -> u32;
}
pub(crate) struct Dispatches<S: DispatchSlot> {
    slot: std::marker::PhantomData<S>,
    plans: wgpu::Buffer,
    args: wgpu::Buffer,
    group: wgpu::BindGroup,
    len: u32,
}
impl<S: DispatchSlot> Dispatches<S> {
    pub fn new(
        device: &wgpu::Device,
        kernel: &wgpu::ComputePipeline,
        counts: &wgpu::Buffer,
        plans: &[DispatchPlan],
    ) -> Self {
        let plan_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dispatch plans"),
            contents: bytemuck::cast_slice(
                &plans.iter().map(DispatchPlan::words).collect::<Vec<_>>(),
            ),
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
            slot: std::marker::PhantomData,
            plans: plan_buffer,
            args,
            group,
            len: plans.len() as u32,
        }
    }
    pub fn update(&self, queue: &wgpu::Queue, plans: &[DispatchPlan]) {
        assert_eq!(plans.len(), self.len as usize);
        queue.write_buffer(
            &self.plans,
            0,
            bytemuck::cast_slice(&plans.iter().map(DispatchPlan::words).collect::<Vec<_>>()),
        );
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
        index: S,
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
        index: S,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups_indirect(&self.args, u64::from(index.index()) * 16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone, Copy)]
    struct Raster;
    impl DispatchSlot for Raster {
        fn index(self) -> u32 {
            0
        }
    }
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
        let dispatches = Dispatches::<Raster>::new(
            &device,
            &prepare,
            &counts,
            &[DispatchPlan::new(CountSlot::Raster, 16, 1, 0)],
        );
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
