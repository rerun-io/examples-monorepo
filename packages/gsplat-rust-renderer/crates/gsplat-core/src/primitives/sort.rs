//! Stable four-bit radix sort with GPU-resident counts.
use super::dispatch::{CountSlot, DispatchPlan, Dispatches};
use crate::gpu::{bind, storage, uniform};
use crate::kernels::Kernels;

const BLOCKS: u32 = 0;
const REDUCED: u32 = 1;
struct SortGroups {
    count_keys: wgpu::BindGroup,
    reduce_counts: wgpu::BindGroup,
    scan_counts: wgpu::BindGroup,
    scan_add: wgpu::BindGroup,
    scatter: wgpu::BindGroup,
}
pub(crate) struct RadixSort {
    keys: [wgpu::Buffer; 2],
    values: [wgpu::Buffer; 2],
    groups: [SortGroups; 8],
    dispatches: Dispatches,
}
impl RadixSort {
    pub fn new(
        device: &wgpu::Device,
        kernels: &Kernels,
        capacity: u32,
        count: CountSlot,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        counts: &wgpu::Buffer,
    ) -> Self {
        let keys = [
            keys.clone(),
            storage(device, "sort keys", u64::from(capacity) * 4),
        ];
        let values = [
            values.clone(),
            storage(device, "sort values", u64::from(capacity) * 4),
        ];
        let blocks = capacity.div_ceil(1024).max(1);
        let histogram = storage(device, "sort histogram", u64::from(blocks) * 64);
        let reduced = storage(
            device,
            "sort reduced histogram",
            u64::from(blocks.div_ceil(1024)) * 64,
        );
        let groups = std::array::from_fn(|digit| {
            let params = uniform(device, &[count.index(), digit as u32 * 4, capacity, 0]);
            let source = digit % 2;
            let dest = 1 - source;
            let group = |kernel: &wgpu::ComputePipeline, bindings: &[(u32, &wgpu::Buffer)]| {
                bind(
                    device,
                    kernel,
                    &bindings
                        .iter()
                        .map(|(slot, buffer)| (*slot, buffer.as_entire_binding()))
                        .collect::<Vec<_>>(),
                )
            };
            SortGroups {
                count_keys: group(
                    &kernels.count_keys,
                    &[
                        (0, counts),
                        (1, &keys[source]),
                        (3, &histogram),
                        (7, &params),
                    ],
                ),
                reduce_counts: group(
                    &kernels.reduce_counts,
                    &[(0, counts), (3, &histogram), (4, &reduced), (7, &params)],
                ),
                scan_counts: group(
                    &kernels.scan_counts,
                    &[(0, counts), (4, &reduced), (7, &params)],
                ),
                scan_add: group(
                    &kernels.scan_add,
                    &[(0, counts), (3, &histogram), (4, &reduced), (7, &params)],
                ),
                scatter: group(
                    &kernels.scatter,
                    &[
                        (0, counts),
                        (1, &keys[source]),
                        (2, &values[source]),
                        (3, &histogram),
                        (5, &keys[dest]),
                        (6, &values[dest]),
                        (7, &params),
                    ],
                ),
            }
        });
        Self {
            keys,
            values,
            groups,
            dispatches: Dispatches::new(
                device,
                &kernels.prepare,
                counts,
                &[
                    DispatchPlan::new(count, capacity, 1024, 1),
                    DispatchPlan::new(count, capacity, 1024 * 1024, 16),
                ],
            ),
        }
    }
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernels: &Kernels,
        bits: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        assert!((1..=32).contains(&bits));
        self.dispatches.prepare(encoder, &kernels.prepare, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gsplat radix sort"),
            timestamp_writes,
        });
        for groups in &self.groups[..bits.div_ceil(4) as usize] {
            self.dispatches.dispatch_in_pass(
                &mut pass,
                BLOCKS,
                &kernels.count_keys,
                &groups.count_keys,
            );
            self.dispatches.dispatch_in_pass(
                &mut pass,
                REDUCED,
                &kernels.reduce_counts,
                &groups.reduce_counts,
            );
            pass.set_pipeline(&kernels.scan_counts);
            pass.set_bind_group(0, &groups.scan_counts, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            self.dispatches.dispatch_in_pass(
                &mut pass,
                REDUCED,
                &kernels.scan_add,
                &groups.scan_add,
            );
            self.dispatches
                .dispatch_in_pass(&mut pass, BLOCKS, &kernels.scatter, &groups.scatter);
        }
    }
    pub fn output(&self, bits: u32) -> (&wgpu::Buffer, &wgpu::Buffer) {
        let index = bits.div_ceil(4) as usize % 2;
        (&self.keys[index], &self.values[index])
    }
}
