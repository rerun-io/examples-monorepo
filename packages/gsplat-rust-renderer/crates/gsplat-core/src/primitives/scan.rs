//! Hierarchical inclusive scan with GPU-resident counts.
use super::dispatch::{CountSlot, DispatchPlan, Dispatches};
use crate::gpu::{bind, storage, uniform};
use crate::kernels::Kernels;

const fn scan_level(level: usize) -> u32 {
    level as u32 * 2
}
const ADD_OFFSETS: u32 = 1;
pub(crate) struct Scan {
    output: wgpu::Buffer,
    groups: Vec<(wgpu::BindGroup, Option<wgpu::BindGroup>)>,
    dispatches: Dispatches,
}
impl Scan {
    pub fn new(
        device: &wgpu::Device,
        kernels: &Kernels,
        capacity: u32,
        count: CountSlot,
        input: &wgpu::Buffer,
        counts: &wgpu::Buffer,
    ) -> Self {
        let mut levels = Vec::new();
        let mut n = capacity.max(1);
        let mut divisor = 1;
        let mut plans = Vec::new();
        loop {
            let blocks = n.div_ceil(1024);
            levels.push((
                storage(device, "scan output", u64::from(n) * 4),
                storage(device, "scan block sums", u64::from(blocks) * 4),
                uniform(device, &[count.index(), divisor, capacity, 0]),
            ));
            plans.push(DispatchPlan::new(count, capacity, divisor * 1024, 1));
            plans.push(DispatchPlan::new(count, capacity, divisor * 256, 1));
            if blocks == 1 {
                break;
            }
            n = blocks;
            divisor *= 1024;
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
            dispatches: Dispatches::new(device, &kernels.prepare, counts, &plans),
        }
    }
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernels: &Kernels,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        self.dispatches.prepare(encoder, &kernels.prepare, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gsplat scan"),
            timestamp_writes,
        });
        for (i, groups) in self.groups.iter().enumerate() {
            self.dispatches
                .dispatch_in_pass(&mut pass, scan_level(i), &kernels.scan, &groups.0);
        }
        for (i, groups) in self.groups.iter().enumerate().rev() {
            if let Some(group) = &groups.1 {
                self.dispatches.dispatch_in_pass(
                    &mut pass,
                    scan_level(i) + ADD_OFFSETS,
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
