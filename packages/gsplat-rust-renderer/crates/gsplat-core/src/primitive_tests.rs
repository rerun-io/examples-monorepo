//! Primitive contracts against independent CPU references.
use crate::gpu::CountSlot;
use crate::kernels::Kernels;
use crate::primitives::{RadixSort, Scan};
use crate::test_utils::{gpu, read, upload};

// Start each primitive with nonzero shared memory. CPU-reference comparisons
// must still pass when pipeline compilation omits workgroup zero initialization.
fn poisoned_kernels(device: &wgpu::Device) -> Kernels {
    use crate::gpu::{module, pipeline};
    let mut kernels = Kernels::new(device);
    let sources = crate::kernels::sources();
    let poison_scan = "
        if lid < 64u { partials[lid] = 0xdeadbeefu; }
        if lid == 0u { cube_total = 0xdeadbeefu; length = 0xdeadbeefu; }
        for (var j = 0u; j < 4u; j++) { lds[j * 256u + lid] = 0xdeadbeefu; }
        workgroupBarrier();
    ";
    let scan = sources[3].replacen(
        "    if lid == 0u {\n        length =",
        &format!("{poison_scan}\n    if lid == 0u {{\n        length ="),
        1,
    );
    let scan = module(device, &scan);
    kernels.scan = pipeline(device, &scan, "scan");
    kernels.add_offsets = pipeline(device, &scan, "add_offsets");
    let sort = sources[4].replace(
        "    let n = sort_length(lid);",
        &format!(
            "{poison_scan}
            local_keys[lid] = 0xdeadbeefu;
            local_values[lid] = 0xdeadbeefu;
            if lid < 16u {{
                atomicStore(&histogram[lid], 0xdeadbeefu);
                bin_offsets[lid] = 0xdeadbeefu;
                bin_prefix[lid] = 0xdeadbeefu;
            }}
            workgroupBarrier();
            let n = sort_length(lid);"
        ),
    );
    let sort = module(device, &sort);
    kernels.count_keys = pipeline(device, &sort, "count_keys");
    kernels.reduce_counts = pipeline(device, &sort, "reduce_counts");
    kernels.scan_counts = pipeline(device, &sort, "scan_counts");
    kernels.scan_add = pipeline(device, &sort, "scan_add");
    kernels.scatter = pipeline(device, &sort, "scatter");
    kernels
}

#[test]
#[ignore = "integration: GPU"]
fn inclusive_scan_crosses_recursive_block_boundaries() {
    let (device, queue) = gpu();
    let kernels = poisoned_kernels(&device);
    for n in [0, 1, 1023, 1024, 1025, 1_048_577] {
        let input: Vec<u32> = (0..n).map(|i| (i % 7) as u32).collect();
        let expected: Vec<u32> = input
            .iter()
            .scan(0, |sum, v| {
                *sum += v;
                Some(*sum)
            })
            .collect();
        let values = upload(&device, &input);
        let count = upload(&device, &[n as u32, 0]);
        let scan = Scan::new(
            &device,
            &kernels,
            n as u32,
            CountSlot::Visible,
            &values,
            &count,
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        scan.encode(&mut encoder, &kernels, None);
        queue.submit([encoder.finish()]);
        assert_eq!(
            read::<u32>(&device, &queue, scan.output(), n),
            expected,
            "length {n}"
        );
    }
}

#[test]
#[ignore = "integration: GPU"]
fn radix_sort_is_stable_for_duplicates_and_partial_blocks() {
    let (device, queue) = gpu();
    let kernels = poisoned_kernels(&device);
    for n in [1u32, 255, 256, 257, 1023, 1024, 1025, 65537, 1_048_577] {
        let input: Vec<u32> = (0..n)
            .map(|i| i.wrapping_mul(1664525).wrapping_add(1013904223) % 65537)
            .collect();
        let mut expected: Vec<u32> = (0..n).collect();
        expected.sort_by_key(|i| input[*i as usize]);
        let keys = upload(&device, &input);
        let values = upload(&device, &(0..n).collect::<Vec<_>>());
        let count = upload(&device, &[n, 0]);
        let sort = RadixSort::new(
            &device,
            &kernels,
            n,
            CountSlot::Visible,
            &keys,
            &values,
            &count,
        );
        // Odd and even pass counts must expose the correct ping-pong output without copies.
        for bits in [20, 32] {
            queue.write_buffer(&keys, 0, bytemuck::cast_slice(&input));
            queue.write_buffer(
                &values,
                0,
                bytemuck::cast_slice(&(0..n).collect::<Vec<_>>()),
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            sort.encode(&mut encoder, &kernels, bits, None);
            queue.submit([encoder.finish()]);
            let (keys, values) = sort.output(bits);
            assert_eq!(
                read::<u32>(&device, &queue, values, n as usize),
                expected,
                "length {n}, {bits} bits"
            );
            assert_eq!(
                read::<u32>(&device, &queue, keys, n as usize),
                expected
                    .iter()
                    .map(|i| input[*i as usize])
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
#[ignore = "integration: GPU"]
fn gpu_counts_cross_recursive_boundaries_and_reuse_scratch() {
    let (device, queue) = gpu();
    let kernels = poisoned_kernels(&device);
    let capacity = 1_048_577u32;
    let input: Vec<u32> = (0..capacity)
        .map(|i| i.wrapping_mul(1664525) % 17)
        .collect();
    let ids: Vec<u32> = (0..capacity).collect();
    let keys = upload(&device, &input);
    let values = upload(&device, &ids);
    let count = upload(&device, &[0, capacity]);
    let sort = RadixSort::new(
        &device,
        &kernels,
        capacity,
        CountSlot::Intersections,
        &keys,
        &values,
        &count,
    );
    let scan = Scan::new(
        &device,
        &kernels,
        capacity,
        CountSlot::Intersections,
        &keys,
        &count,
    );
    for n in [capacity, 0, 17, 1025, 1] {
        queue.write_buffer(&keys, 0, bytemuck::cast_slice(&input));
        queue.write_buffer(&values, 0, bytemuck::cast_slice(&ids));
        queue.write_buffer(&count, 4, bytemuck::bytes_of(&n));
        let mut encoder = device.create_command_encoder(&Default::default());
        scan.encode(&mut encoder, &kernels, None);
        sort.encode(&mut encoder, &kernels, 8, None);
        queue.submit([encoder.finish()]);
        let expected_scan: Vec<u32> = input[..n as usize]
            .iter()
            .scan(0, |sum, v| {
                *sum += v;
                Some(*sum)
            })
            .collect();
        assert_eq!(
            read::<u32>(&device, &queue, scan.output(), n as usize),
            expected_scan
        );
        let mut expected: Vec<u32> = (0..n).collect();
        expected.sort_by_key(|i| input[*i as usize]);
        assert_eq!(
            read::<u32>(&device, &queue, sort.output(8).1, n as usize),
            expected
        );
    }
}

#[test]
#[ignore = "integration: GPU with >= 280 MB storage bindings"]
fn radix_sort_crosses_the_70m_reduced_histogram_boundary() {
    let n = 70_000_001u32;
    let (device, queue) = gpu();
    if device.limits().max_storage_buffer_binding_size < u64::from(n) * 4 {
        eprintln!("SKIP: adapter storage binding limit cannot hold 70M keys");
        return;
    }
    let kernels = Kernels::new(&device);
    let input: Vec<u32> = (0..n)
        .map(|i| i.wrapping_mul(1664525).wrapping_add(1013904223) >> 28)
        .collect();
    let keys = upload(&device, &input);
    let values = upload(&device, &(0..n).collect::<Vec<_>>());
    let count = upload(&device, &[n, 0]);
    let sort = RadixSort::new(
        &device,
        &kernels,
        n,
        CountSlot::Visible,
        &keys,
        &values,
        &count,
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    sort.encode(&mut encoder, &kernels, 4, None);
    queue.submit([encoder.finish()]);
    let result = read::<u32>(&device, &queue, sort.output(4).1, n as usize);
    let mut expected = (0..n).collect::<Vec<_>>();
    expected.sort_by_key(|i| input[*i as usize]);
    assert_eq!(result.iter().zip(&expected).position(|(a, b)| a != b), None);
    let output = read::<u32>(&device, &queue, sort.output(4).0, n as usize);
    assert!(
        output
            .iter()
            .zip(&result)
            .all(|(key, idx)| *key == input[*idx as usize])
    );
}
