//! GPU contract helpers shared by unit and integration tests.
#![allow(dead_code)]
use wgpu::util::DeviceExt as _;

pub fn gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SUBGROUP
            | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY),
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
            max_buffer_size: adapter.limits().max_buffer_size,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap()
}
pub fn upload<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    let bytes = bytemuck::cast_slice(values);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: if bytes.is_empty() { &[0; 4] } else { bytes },
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}
pub fn read<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    let bytes = (count * size_of::<T>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    encoder.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
        tx.send(result).unwrap()
    });
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&staging.get_mapped_range(..).unwrap()).to_vec()
}
