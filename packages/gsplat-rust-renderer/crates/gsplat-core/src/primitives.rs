//! Cached scan and stable 4-bit radix sort; counts stay on the GPU.
use crate::gpu::{Dispatches, bind, storage, uniform};
use crate::kernels::Kernels;

pub(crate) struct Scan {
    output: wgpu::Buffer,
    groups: Vec<[Option<wgpu::BindGroup>; 2]>,
    dispatches: Dispatches,
}
impl Scan {
    pub fn new(
        device: &wgpu::Device,
        kernels: &Kernels,
        capacity: u32,
        count_index: u32,
        input: &wgpu::Buffer,
        count: &wgpu::Buffer,
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
                uniform(device, &[count_index, divisor, capacity, 0]),
            ));
            plans.push([count_index, capacity, divisor * 1024, 1]);
            plans.push([count_index, capacity, divisor * 256, 1]);
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
                    &kernels.scan[0],
                    &[
                        (0, count.as_entire_binding()),
                        (1, source.as_entire_binding()),
                        (2, output.as_entire_binding()),
                        (3, sums.as_entire_binding()),
                        (4, params.as_entire_binding()),
                    ],
                );
                let add = levels.get(i + 1).map(|next| {
                    bind(
                        device,
                        &kernels.scan[1],
                        &[
                            (0, count.as_entire_binding()),
                            (1, next.0.as_entire_binding()),
                            (2, output.as_entire_binding()),
                            (4, params.as_entire_binding()),
                        ],
                    )
                });
                [Some(scan), add]
            })
            .collect();
        Self {
            output: levels[0].0.clone(),
            groups,
            dispatches: Dispatches::new(device, &kernels.prepare, count, &plans),
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
            self.dispatches.dispatch_in_pass(
                &mut pass,
                i as u32 * 2,
                &kernels.scan[0],
                groups[0].as_ref().unwrap(),
            );
        }
        for (i, groups) in self.groups.iter().enumerate().rev() {
            if let Some(group) = &groups[1] {
                self.dispatches.dispatch_in_pass(
                    &mut pass,
                    i as u32 * 2 + 1,
                    &kernels.scan[1],
                    group,
                );
            }
        }
    }
    pub fn output(&self) -> &wgpu::Buffer {
        &self.output
    }
}

pub(crate) struct RadixSort {
    keys: [wgpu::Buffer; 2],
    values: [wgpu::Buffer; 2],
    groups: [[wgpu::BindGroup; 5]; 8],
    dispatches: Dispatches,
}
impl RadixSort {
    pub fn new(
        device: &wgpu::Device,
        kernels: &Kernels,
        capacity: u32,
        count_index: u32,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
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
            let params = uniform(device, &[count_index, digit as u32 * 4, capacity, 0]);
            let source = digit % 2;
            let dest = 1 - source;
            let bindings = [
                vec![
                    (0, count),
                    (1, &keys[source]),
                    (3, &histogram),
                    (7, &params),
                ],
                vec![(0, count), (3, &histogram), (4, &reduced), (7, &params)],
                vec![(0, count), (4, &reduced), (7, &params)],
                vec![(0, count), (3, &histogram), (4, &reduced), (7, &params)],
                vec![
                    (0, count),
                    (1, &keys[source]),
                    (2, &values[source]),
                    (3, &histogram),
                    (5, &keys[dest]),
                    (6, &values[dest]),
                    (7, &params),
                ],
            ];
            std::array::from_fn(|stage| {
                bind(
                    device,
                    &kernels.sort[stage],
                    &bindings[stage]
                        .iter()
                        .map(|(i, b)| (*i, b.as_entire_binding()))
                        .collect::<Vec<_>>(),
                )
            })
        });
        Self {
            keys,
            values,
            groups,
            dispatches: Dispatches::new(
                device,
                &kernels.prepare,
                count,
                &[
                    [count_index, capacity, 1024, 1],
                    [count_index, capacity, 1024 * 1024, 16],
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
            for (stage, plan) in [Some(0), Some(1), None, Some(1), Some(0)]
                .into_iter()
                .enumerate()
            {
                if let Some(plan) = plan {
                    self.dispatches.dispatch_in_pass(
                        &mut pass,
                        plan,
                        &kernels.sort[stage],
                        &groups[stage],
                    );
                } else {
                    pass.set_pipeline(&kernels.sort[stage]);
                    pass.set_bind_group(0, &groups[stage], &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
        }
    }
    pub fn output(&self, bits: u32) -> (&wgpu::Buffer, &wgpu::Buffer) {
        let index = bits.div_ceil(4) as usize % 2;
        (&self.keys[index], &self.values[index])
    }
}
