//! GPU integration tests against independent CPU references.
use wgpu::util::DeviceExt as _;

fn upload(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(if words.is_empty() { &[0] } else { words }),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}
fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, n: usize) -> Vec<u32> {
    if n == 0 {
        return Vec::new();
    }
    let bytes = (n * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&staging.get_mapped_range(..).unwrap()).to_vec()
}

#[test]
#[ignore = "integration: WebGPU adapter with subgroups"]
fn inclusive_scan_crosses_recursive_block_boundaries() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SUBGROUP,
        ..Default::default()
    }))
    .unwrap();
    for n in [0, 1, 1023, 1024, 1025, 1_048_577] {
        let input: Vec<u32> = (0..n).map(|i| (i % 7) as u32).collect();
        let expected: Vec<u32> = input
            .iter()
            .scan(0u32, |sum, x| {
                *sum += x;
                Some(*sum)
            })
            .collect();
        let scan = crate::primitives::Scan::new(&device, n as u32, 0);
        let mut encoder = device.create_command_encoder(&Default::default());
        scan.encode(
            &mut encoder,
            &upload(&device, &input),
            &upload(&device, &[n as u32]),
        );
        queue.submit([encoder.finish()]);
        assert_eq!(
            read(&device, &queue, scan.output(), n),
            expected,
            "length {n}"
        );
    }
}

#[test]
#[ignore = "integration: WebGPU adapter with subgroups"]
fn radix_sort_is_stable_for_duplicates_and_partial_blocks() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SUBGROUP,
        ..Default::default()
    }))
    .unwrap();
    for n in [1u32, 255, 256, 257, 1023, 1024, 1025, 65537, 1_048_577] {
        let input: Vec<u32> = (0..n)
            .map(|i| i.wrapping_mul(1664525).wrapping_add(1013904223) % 65537)
            .collect();
        let mut expected: Vec<u32> = (0..n).collect();
        expected.sort_by_key(|i| input[*i as usize]);
        let keys = upload(&device, &input);
        let values = upload(&device, &(0..n).collect::<Vec<_>>());
        let sort = crate::primitives::RadixSort::new(&device, n, 0, 32);
        let mut encoder = device.create_command_encoder(&Default::default());
        sort.encode(&mut encoder, &keys, &values, &upload(&device, &[n]));
        queue.submit([encoder.finish()]);
        assert_eq!(
            read(&device, &queue, &values, n as usize),
            expected,
            "length {n}"
        );
        assert_eq!(
            read(&device, &queue, &keys, n as usize),
            expected
                .iter()
                .map(|i| input[*i as usize])
                .collect::<Vec<_>>()
        );
    }
}

#[test]
#[ignore = "integration: GPU with >= 280 MB storage bindings, 70M-key sort regression"]
fn radix_sort_crosses_the_70m_reduced_histogram_boundary() {
    let n = 70_000_001u32;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    if adapter.limits().max_storage_buffer_binding_size < u64::from(n) * 4 {
        eprintln!("SKIP: adapter storage binding limit cannot hold 70M keys");
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SUBGROUP,
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: u64::from(n) * 4,
            max_buffer_size: u64::from(n) * 4,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
    let input: Vec<u32> = (0..n)
        .map(|i| i.wrapping_mul(1664525).wrapping_add(1013904223) >> 28)
        .collect();
    let keys = upload(&device, &input);
    let values = upload(&device, &(0..n).collect::<Vec<_>>());
    let sort = crate::primitives::RadixSort::new(&device, n, 0, 4);
    let mut encoder = device.create_command_encoder(&Default::default());
    sort.encode(&mut encoder, &keys, &values, &upload(&device, &[n]));
    queue.submit([encoder.finish()]);
    let result = read(&device, &queue, &values, n as usize);
    let mut expected = (0..n).collect::<Vec<_>>();
    expected.sort_by_key(|i| input[*i as usize]);
    assert_eq!(result.iter().zip(&expected).position(|(a, b)| a != b), None);
    let output = read(&device, &queue, &keys, n as usize);
    assert!(
        output
            .iter()
            .zip(&result)
            .all(|(key, idx)| *key == input[*idx as usize])
    );
}
