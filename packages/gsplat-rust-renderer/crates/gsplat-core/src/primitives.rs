//! Stable radix sorting and inclusive prefix sums, with 1024 elements per block.
use crate::gpu::{bind, dispatch, pipeline, storage, uniform};

struct ScanLevel {
    output: wgpu::Buffer,
    sums: wgpu::Buffer,
    params: wgpu::Buffer,
    capacity: u32,
}

/// Recursive inclusive u32 scan. The count stays on the GPU.
/// Allocate once for a capacity, then reuse for each frame.
/// Values and their sum must fit in u32.
pub struct Scan {
    device: wgpu::Device,
    scan: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    levels: Vec<ScanLevel>,
}
impl Scan {
    /// `count_index` selects a u32 in the count buffer passed to [`Self::encode`].
    pub fn new(device: &wgpu::Device, capacity: u32, count_index: u32) -> Self {
        let source = format!(
            "{}\n{}",
            include_str!("../shaders/scan_common.wgsl"),
            include_str!("../shaders/scan.wgsl")
        );
        let mut levels = Vec::new();
        let mut n = capacity.max(1);
        let mut divisor = 1;
        loop {
            let blocks = n.div_ceil(1024);
            levels.push(ScanLevel {
                output: storage(device, "scan output", u64::from(n) * 4),
                sums: storage(device, "scan block sums", u64::from(blocks) * 4),
                params: uniform(device, &[count_index, divisor, capacity, 0]),
                capacity: n,
            });
            if blocks == 1 {
                break;
            }
            n = blocks;
            divisor *= 1024;
        }
        Self {
            device: device.clone(),
            scan: pipeline(device, &source, "scan", &[]),
            add: pipeline(device, &source, "add_offsets", &[]),
            levels,
        }
    }
    /// Encode without submission or host synchronization; read [`Self::output`] after completion.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        count: &wgpu::Buffer,
    ) {
        for (i, level) in self.levels.iter().enumerate() {
            let source = if i == 0 {
                input
            } else {
                &self.levels[i - 1].sums
            };
            let group = bind(
                &self.device,
                &self.scan,
                &[
                    (0, count),
                    (1, source),
                    (2, &level.output),
                    (3, &level.sums),
                    (4, &level.params),
                ],
            );
            dispatch(encoder, &self.scan, &group, level.capacity.div_ceil(1024));
        }
        for i in (0..self.levels.len() - 1).rev() {
            let level = &self.levels[i];
            let group = bind(
                &self.device,
                &self.add,
                &[
                    (0, count),
                    (1, &self.levels[i + 1].output),
                    (2, &level.output),
                    (4, &level.params),
                ],
            );
            dispatch(encoder, &self.add, &group, level.capacity.div_ceil(256));
        }
    }
    /// Inclusive sums, valid up to the input count (at most the capacity).
    pub fn output(&self) -> &wgpu::Buffer {
        &self.levels[0].output
    }
}

/// Stable 4-bit radix sort. Keys and values are sorted in place; no CPU count readback.
/// Each digit uses count, reduce, scan, scan-add, and scatter kernels.
pub struct RadixSort {
    device: wgpu::Device,
    kernels: [wgpu::ComputePipeline; 5],
    temporary_keys: wgpu::Buffer,
    temporary_values: wgpu::Buffer,
    histogram: wgpu::Buffer,
    reduced: wgpu::Buffer,
    params: Vec<wgpu::Buffer>,
    capacity: u32,
}
impl RadixSort {
    pub fn new(device: &wgpu::Device, capacity: u32, count_index: u32, bits: u32) -> Self {
        assert!((1..=32).contains(&bits));
        let source = format!(
            "{}\n{}",
            include_str!("../shaders/scan_common.wgsl"),
            include_str!("../shaders/sort.wgsl")
        );
        let blocks = capacity.div_ceil(1024).max(1);
        Self {
            device: device.clone(),
            kernels: [
                "count_keys",
                "reduce_counts",
                "scan_counts",
                "scan_add",
                "scatter",
            ]
            .map(|entry| pipeline(device, &source, entry, &[])),
            temporary_keys: storage(device, "sort keys", u64::from(capacity) * 4),
            temporary_values: storage(device, "sort values", u64::from(capacity) * 4),
            histogram: storage(device, "sort histogram", u64::from(blocks) * 16 * 4),
            reduced: storage(
                device,
                "sort reduced histogram",
                u64::from(blocks.div_ceil(1024)) * 16 * 4,
            ),
            params: (0..bits.div_ceil(4))
                .map(|digit| uniform(device, &[count_index, digit * 4, capacity, 0]))
                .collect(),
            capacity,
        }
    }
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
    ) {
        let blocks = self.capacity.div_ceil(1024);
        let reduced_blocks = 16 * blocks.div_ceil(1024);
        for (digit, params) in self.params.iter().enumerate() {
            let (src, vals, dst, out_vals) = if digit % 2 == 0 {
                (keys, values, &self.temporary_keys, &self.temporary_values)
            } else {
                (&self.temporary_keys, &self.temporary_values, keys, values)
            };
            let bindings = [
                vec![(0, count), (1, src), (3, &self.histogram), (7, params)],
                vec![
                    (0, count),
                    (3, &self.histogram),
                    (4, &self.reduced),
                    (7, params),
                ],
                vec![(0, count), (4, &self.reduced), (7, params)],
                vec![
                    (0, count),
                    (3, &self.histogram),
                    (4, &self.reduced),
                    (7, params),
                ],
                vec![
                    (0, count),
                    (1, src),
                    (2, vals),
                    (3, &self.histogram),
                    (5, dst),
                    (6, out_vals),
                    (7, params),
                ],
            ];
            for ((kernel, bindings), groups) in self.kernels.iter().zip(bindings).zip([
                blocks,
                reduced_blocks,
                1,
                reduced_blocks,
                blocks,
            ]) {
                let group = bind(&self.device, kernel, &bindings);
                dispatch(encoder, kernel, &group, groups);
            }
        }
        if self.params.len() % 2 == 1 && self.capacity > 0 {
            let size = u64::from(self.capacity) * 4;
            encoder.copy_buffer_to_buffer(&self.temporary_keys, 0, keys, 0, size);
            encoder.copy_buffer_to_buffer(&self.temporary_values, 0, values, 0, size);
        }
    }
}
