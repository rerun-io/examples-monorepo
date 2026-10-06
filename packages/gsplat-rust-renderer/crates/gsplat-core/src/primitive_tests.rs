//! Primitive contracts against independent CPU references.
use crate::kernels::Kernels;
use crate::primitives::{RadixSort, Scan};
use crate::test_utils::{gpu, read, upload};

#[test]
#[ignore = "integration: GPU"]
fn inclusive_scan_crosses_recursive_block_boundaries() {
    let (device, queue) = gpu();
    let kernels = Kernels::new(&device);
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
        let scan = Scan::new(&device, &kernels, n as u32, 0, &values, &count);
        let mut encoder = device.create_command_encoder(&Default::default());
        scan.encode(&mut encoder, &kernels);
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
    let kernels = Kernels::new(&device);
    for n in [1u32, 255, 256, 257, 1023, 1024, 1025, 65537, 1_048_577] {
        let input: Vec<u32> = (0..n)
            .map(|i| i.wrapping_mul(1664525).wrapping_add(1013904223) % 65537)
            .collect();
        let mut expected: Vec<u32> = (0..n).collect();
        expected.sort_by_key(|i| input[*i as usize]);
        let keys = upload(&device, &input);
        let values = upload(&device, &(0..n).collect::<Vec<_>>());
        let count = upload(&device, &[n, 0]);
        let sort = RadixSort::new(&device, &kernels, n, 0, &keys, &values, &count);
        // Odd and even pass counts must expose the correct ping-pong output without copies.
        for bits in [20, 32] {
            queue.write_buffer(&keys, 0, bytemuck::cast_slice(&input));
            queue.write_buffer(
                &values,
                0,
                bytemuck::cast_slice(&(0..n).collect::<Vec<_>>()),
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            sort.encode(&mut encoder, &kernels, bits);
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
    let kernels = Kernels::new(&device);
    let capacity = 1_048_577u32;
    let input: Vec<u32> = (0..capacity)
        .map(|i| i.wrapping_mul(1664525) % 17)
        .collect();
    let ids: Vec<u32> = (0..capacity).collect();
    let keys = upload(&device, &input);
    let values = upload(&device, &ids);
    let count = upload(&device, &[0, capacity]);
    let sort = RadixSort::new(&device, &kernels, capacity, 1, &keys, &values, &count);
    let scan = Scan::new(&device, &kernels, capacity, 1, &keys, &count);
    for n in [capacity, 0, 17, 1025, 1] {
        queue.write_buffer(&keys, 0, bytemuck::cast_slice(&input));
        queue.write_buffer(&values, 0, bytemuck::cast_slice(&ids));
        queue.write_buffer(&count, 4, bytemuck::bytes_of(&n));
        let mut encoder = device.create_command_encoder(&Default::default());
        scan.encode(&mut encoder, &kernels);
        sort.encode(&mut encoder, &kernels, 8);
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
    let sort = RadixSort::new(&device, &kernels, n, 0, &keys, &values, &count);
    let mut encoder = device.create_command_encoder(&Default::default());
    sort.encode(&mut encoder, &kernels, 4);
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
