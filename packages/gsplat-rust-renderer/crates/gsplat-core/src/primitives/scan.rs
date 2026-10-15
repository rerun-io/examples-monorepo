//! Hierarchical inclusive scan with GPU-resident counts.
use super::dispatch::{CountSlot, DispatchPlan, Dispatches};
use crate::gpu::{bind, storage, uniform};
use crate::kernels::{BLOCK_SIZE, Kernels, WG};

const fn scan_level(level: usize) -> u32 {
    level as u32 * 2
}
const ADD_OFFSETS: u32 = 1;
pub(crate) struct Scan {
    output: wgpu::Buffer,
    groups: Vec<(wgpu::BindGroup, Option<wgpu::BindGroup>)>,
    dispatch_base: u32,
}
impl Scan {
    pub fn new(
        device: &wgpu::Device,
        kernels: &Kernels,
        capacity: u32,
        count: CountSlot,
        input: &wgpu::Buffer,
        counts: &wgpu::Buffer,
        dispatch_base: u32,
    ) -> Self {
        let mut levels = Vec::new();
        let mut n = capacity.max(1);
        let mut divisor = 1;
        loop {
            let blocks = n.div_ceil(BLOCK_SIZE);
            levels.push((
                storage(device, "scan output", u64::from(n) * 4),
                storage(device, "scan block sums", u64::from(blocks) * 4),
                uniform(device, &[count.index(), divisor, capacity, 0]),
            ));
            if blocks == 1 {
                break;
            }
            n = blocks;
            divisor *= BLOCK_SIZE;
        }
        let groups = levels
            .iter()
            .enumerate()
            .map(|(i, (output, sums, params))| {
                let source = if i == 0 { input } else { &levels[i - 1].1 };
                let scan = bind(
                    device,
                    &kernels.scan.layout,
                    &[
                        (0, counts.as_entire_binding()),
                        (1, source.as_entire_binding()),
                        (2, output.as_entire_binding()),
                        (3, sums.as_entire_binding()),
                        (4, params.as_entire_binding()),
                    ],
                );
                let add = levels.get(i + 1).map(|next| {
                    bind(
                        device,
                        &kernels.add_offsets.layout,
                        &[
                            (0, counts.as_entire_binding()),
                            (1, next.0.as_entire_binding()),
                            (2, output.as_entire_binding()),
                            (4, params.as_entire_binding()),
                        ],
                    )
                });
                (scan, add)
            })
            .collect();
        Self {
            output: levels[0].0.clone(),
            groups,
            dispatch_base,
        }
    }
    pub fn plans(count: CountSlot, capacity: u32) -> Vec<DispatchPlan> {
        let mut plans = Vec::new();
        let mut n = capacity.max(1);
        let mut divisor = 1;
        loop {
            let blocks = n.div_ceil(BLOCK_SIZE);
            plans.push(DispatchPlan::new(count, capacity, divisor * BLOCK_SIZE, 1));
            plans.push(DispatchPlan::new(count, capacity, divisor * WG, 1));
            if blocks == 1 {
                break;
            }
            n = blocks;
            divisor *= BLOCK_SIZE;
        }
        plans
    }
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernels: &Kernels,
        dispatches: &Dispatches,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gsplat scan"),
            timestamp_writes,
        });
        for (i, groups) in self.groups.iter().enumerate() {
            dispatches.dispatch_in_pass(
                &mut pass,
                self.dispatch_base + scan_level(i),
                &kernels.scan,
                &groups.0,
            );
        }
        for (i, groups) in self.groups.iter().enumerate().rev() {
            if let Some(group) = &groups.1 {
                dispatches.dispatch_in_pass(
                    &mut pass,
                    self.dispatch_base + scan_level(i) + ADD_OFFSETS,
                    &kernels.add_offsets,
                    group,
                );
            }
        }
    }
    pub fn output(&self) -> &wgpu::Buffer {
        &self.output
    }
}
