//! Primitive contracts against independent CPU references.
use crate::kernels::Kernels;
use crate::primitives::dispatch::CountSlot;
use crate::primitives::{RadixSort, Scan};
use crate::test_utils::{gpu, read, upload};

// Start each primitive with nonzero shared memory. CPU-reference comparisons
// must still pass when pipeline compilation omits workgroup zero initialization.
fn poisoned_kernels(device: &wgpu::Device) -> Kernels {
    use crate::gpu::{module, pipeline};
    let mut kernels = Kernels::new(device);
    let kernel = |module, entry| pipeline(device, module, entry, crate::kernels::bindings(entry));
    let poison_scan = "
        if lid < 64u { partials[lid] = 0xdeadbeefu; }
        if lid == 0u { cube_total = 0xdeadbeefu; length = 0xdeadbeefu; }
        for (var j = 0u; j < 4u; j++) { lds[j * 256u + lid] = 0xdeadbeefu; }
        workgroupBarrier();
    ";
    let scan = crate::shader::resolve("scan.wgsl");
    assert_eq!(
        scan.matches("    if lid == 0u {\n        length =").count(),
        1
    );
    let scan = scan.replacen(
        "    if lid == 0u {\n        length =",
        &format!("{poison_scan}\n    if lid == 0u {{\n        length ="),
        1,
    );
    assert_ne!(scan, crate::shader::resolve("scan.wgsl"));
    let scan = module(device, &scan);
    kernels.scan = kernel(&scan, "scan");
    kernels.add_offsets = kernel(&scan, "add_offsets");
    let sort = crate::shader::resolve("sort.wgsl");
    assert_eq!(sort.matches("    let n = sort_length(lid);").count(), 5);
    let sort = sort.replace(
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
    assert_ne!(sort, crate::shader::resolve("sort.wgsl"));
    let sort = module(device, &sort);
    kernels.count_keys = kernel(&sort, "count_keys");
    kernels.reduce_counts = kernel(&sort, "reduce_counts");
    kernels.scan_counts = kernel(&sort, "scan_counts");
    kernels.scan_add = kernel(&sort, "scan_add");
    kernels.scatter = kernel(&sort, "scatter");
    kernels
}

#[test]
#[ignore = "integration: GPU"]
fn inclusive_scan_crosses_recursive_block_boundaries() {
    let (device, queue) = &gpu();
    let kernels = poisoned_kernels(device);
    for n in [0, 1, 1023, 1024, 1025, 1_048_577] {
        let input: Vec<u32> = (0..n).map(|i| (i % 7) as u32).collect();
        let expected: Vec<u32> = input
            .iter()
            .scan(0, |sum, v| {
                *sum += v;
                Some(*sum)
            })
            .collect();
        let values = upload(device, &input);
        let count = upload(device, &[n as u32, 0]);
        let scan = Scan::new(
            device,
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
            read::<u32>(device, queue, scan.output(), n),
            expected,
            "length {n}"
        );
    }
}

#[test]
#[ignore = "integration: GPU"]
fn radix_sort_is_stable_for_duplicates_and_partial_blocks() {
    let (device, queue) = &gpu();
    let kernels = poisoned_kernels(device);
    for n in [1u32, 255, 256, 257, 1023, 1024, 1025, 65537, 1_048_577] {
        let input: Vec<u32> = (0..n)
            .map(|i| i.wrapping_mul(1664525).wrapping_add(1013904223) % 65537)
            .collect();
        let mut expected: Vec<u32> = (0..n).collect();
        expected.sort_by_key(|i| input[*i as usize]);
        let keys = upload(device, &input);
        let values = upload(device, &(0..n).collect::<Vec<_>>());
        let count = upload(device, &[n, 0]);
        let sort = RadixSort::new(
            device,
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
                read::<u32>(device, queue, values, n as usize),
                expected,
                "length {n}, {bits} bits"
            );
            assert_eq!(
                read::<u32>(device, queue, keys, n as usize),
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
    let (device, queue) = &gpu();
    let kernels = poisoned_kernels(device);
    let capacity = 1_048_577u32;
    let input: Vec<u32> = (0..capacity)
        .map(|i| i.wrapping_mul(1664525) % 17)
        .collect();
    let ids: Vec<u32> = (0..capacity).collect();
    let keys = upload(device, &input);
    let values = upload(device, &ids);
    let count = upload(device, &[0, capacity]);
    let sort = RadixSort::new(
        device,
        &kernels,
        capacity,
        CountSlot::Intersections,
        &keys,
        &values,
        &count,
    );
    let scan = Scan::new(
        device,
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
            read::<u32>(device, queue, scan.output(), n as usize),
            expected_scan
        );
        let mut expected: Vec<u32> = (0..n).collect();
        expected.sort_by_key(|i| input[*i as usize]);
        assert_eq!(
            read::<u32>(device, queue, sort.output(8).1, n as usize),
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

#[test]
fn poison_injection_anchors_are_present() {
    assert_eq!(
        crate::shader::resolve("scan.wgsl")
            .matches("    if lid == 0u {\n        length =")
            .count(),
        1
    );
    assert_eq!(
        crate::shader::resolve("sort.wgsl")
            .matches("    let n = sort_length(lid);")
            .count(),
        5
    );
}

#[test]
#[ignore = "integration: GPU"]
fn sentinel_closes_the_last_real_tile_range() {
    use crate::gpu::bind;
    use crate::primitives::dispatch::dispatch_direct;
    let (device, queue) = &gpu();
    let kernels = Kernels::new(device);
    let keys = upload(device, &[0u32, 0, 1, 2, 2]);
    let counts = upload(device, &[0u32, 5]);
    let offsets = upload(device, &[0u32; 4]);
    let mut words = [0u32; 64];
    words[30] = 2; // ImageInfo.tiles_x: two real tiles, sentinel = 2.
    words[31] = 1;
    let uniform = crate::gpu::uniform(device, &words);
    let kernel = &kernels.tile_offsets;
    let group = bind(
        device,
        &kernel.layout,
        &[
            (0, uniform.as_entire_binding()),
            (1, counts.as_entire_binding()),
            (7, keys.as_entire_binding()),
            (9, offsets.as_entire_binding()),
        ],
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    dispatch_direct(&mut encoder, kernel, &group, 1, None);
    queue.submit([encoder.finish()]);
    assert_eq!(read::<u32>(device, queue, &offsets, 4), [0, 2, 2, 3]);
}

#[test]
#[ignore = "integration: GPU"]
fn float_depth_keys_sort_negative_depths_before_positive_depths() {
    use crate::gpu::{bind, module, pipeline};
    use crate::primitives::dispatch::dispatch_direct;
    let (device, queue) = &gpu();
    let kernels = poisoned_kernels(device);
    let depths = [4.0f32, -1.0, -4.0, 0.0, 1.0, -0.0];
    let values = upload(device, &[0u32, 1, 2, 3, 4, 5]);
    let keys = upload(device, &[0u32; 6]);
    let input = upload(device, &depths);
    let counts = upload(device, &[6u32, 0]);
    let source = format!("{}
        @group(0) @binding(10) var<storage, read_write> test_depths: array<f32>;
        @group(0) @binding(11) var<storage, read_write> test_keys: array<u32>;
        @compute @workgroup_size(1) fn test_depth_key(@builtin(global_invocation_id) id: vec3u) {{
            test_keys[id.x] = depth_key(test_depths[id.x]);
        }}", crate::shader::resolve("project.wgsl"));
    let key_kernel = pipeline(device, &module(device, &source), "test_depth_key", &[
        (10, crate::kernels::W), (11, crate::kernels::W),
    ]);
    let group = bind(device, &key_kernel.layout, &[
        (10, input.as_entire_binding()), (11, keys.as_entire_binding()),
    ]);
    let sort = RadixSort::new(device, &kernels, 6, CountSlot::Visible, &keys, &values, &counts);
    let mut encoder = device.create_command_encoder(&Default::default());
    dispatch_direct(&mut encoder, &key_kernel, &group, 6, None);
    sort.encode(&mut encoder, &kernels, 32, None);
    queue.submit([encoder.finish()]);
    // Independent CPU total order includes negative zero immediately before positive zero.
    let mut expected = [0u32, 1, 2, 3, 4, 5];
    expected.sort_by(|a, b| depths[*a as usize].total_cmp(&depths[*b as usize]));
    assert_eq!(expected, [2, 1, 5, 3, 4, 0]);
    assert_eq!(read::<u32>(device, queue, sort.output(32).1, 6), expected);
}
