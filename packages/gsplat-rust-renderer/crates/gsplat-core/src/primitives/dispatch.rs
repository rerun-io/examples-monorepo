//! GPU count sources and cached two-dimensional indirect dispatch plans.
use crate::gpu::{bind, storage, upload};

pub(crate) fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &crate::gpu::Kernel,
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
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, group, &[]);
    let x = groups.min(65535);
    pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
}

#[derive(Clone, Copy)]
pub(crate) enum CountSlot {
    Visible,
    Intersections,
}
impl CountSlot {
    pub fn index(self) -> u32 {
        match self {
            Self::Visible => 0,
            Self::Intersections => 1,
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
pub(crate) struct Dispatches {
    plans: wgpu::Buffer,
    args: wgpu::Buffer,
    group: wgpu::BindGroup,
    len: u32,
}
impl Dispatches {
    pub fn new(
        device: &wgpu::Device,
        kernel: &crate::gpu::Kernel,
        counts: &wgpu::Buffer,
        plans: &[DispatchPlan],
    ) -> Self {
        let plan_buffer = upload(
            device,
            "dispatch plans",
            bytemuck::cast_slice(&plans.iter().map(DispatchPlan::words).collect::<Vec<_>>()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let args = storage(device, "indirect dispatches", plans.len() as u64 * 16);
        let group = bind(
            device,
            &kernel.layout,
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
        kernel: &crate::gpu::Kernel,
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
        pipeline: &crate::gpu::Kernel,
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
        pipeline: &crate::gpu::Kernel,
        group: &wgpu::BindGroup,
    ) {
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups_indirect(&self.args, u64::from(index) * 16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{module, pipeline};
    use crate::test_utils::{gpu, read, upload};
    #[test]
    #[ignore = "integration: GPU"]
    fn wrapped_intersection_count_cannot_enable_raster() {
        let (device, queue) = &gpu();
        let counts = upload(device, &[0u32, u32::MAX - 4]);
        let source = format!(
            "{}\n@compute @workgroup_size(1) fn test_counter() {{ add_intersections(8u); }}",
            include_str!("../../shader/counts.wgsl")
        );
        let kernel = pipeline(
            device,
            &module(device, &source),
            "test_counter",
            &[(6, crate::kernels::W)],
        );
        let group = bind(device, &kernel.layout, &[(6, counts.as_entire_binding())]);
        let kernels = crate::kernels::Kernels::new(device);
        let raster = &kernels.float;
        let mut words = [0u32; 64];
        words[28..32].fill(1);
        let uniform = crate::gpu::uniform(device, &words);
        let scratch = upload(device, &[0u32; 10]);
        let target = upload(device, &[[0.25f32; 4]]);
        let raster_group = bind(
            device,
            &raster.layout,
            &[
                (0, uniform.as_entire_binding()),
                (1, scratch.as_entire_binding()),
                (2, scratch.as_entire_binding()),
                (3, scratch.as_entire_binding()),
                (4, target.as_entire_binding()),
                (7, counts.as_entire_binding()),
            ],
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        dispatch(&mut encoder, &kernel, &group, 1, None);
        dispatch(&mut encoder, raster, &raster_group, 1, None);
        queue.submit([encoder.finish()]);
        assert_eq!(read::<u32>(device, queue, &counts, 2), [0x8000_0000, 3]);
        assert_eq!(read::<[f32; 4]>(device, queue, &target, 1), [[0.25; 4]]);
    }
}
