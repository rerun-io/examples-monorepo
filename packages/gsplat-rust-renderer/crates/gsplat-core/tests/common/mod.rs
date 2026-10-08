//! GPU contract helpers shared by unit and integration tests.
#![allow(dead_code)]
use wgpu::util::DeviceExt as _;

pub fn gpu() -> (wgpu::Device, wgpu::Queue) {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    println!("core test adapter: {:?}", adapter.get_info());
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SUBGROUP
            | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY),
        required_limits: adapter.limits(),
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

pub fn pinhole_camera(size: u32) -> crate::Camera {
    crate::Camera {
        model: crate::CameraModel::Pinhole,
        position: glam::Vec3::ZERO,
        rotation: glam::Quat::IDENTITY,
        fov_x: 1.0,
        fov_y: 1.0,
        center_uv: glam::Vec2::splat(0.5),
        size: glam::UVec2::splat(size),
    }
}

pub fn splat(pos: [f32; 3], log_scale: f32, raw_opacity: f32) -> crate::Splats {
    let [x, y, z] = pos;
    crate::Splats {
        transforms: vec![[x, y, z, 1.0, 0.0, 0.0, 0.0, log_scale, log_scale, log_scale]],
        raw_opacities: vec![raw_opacity],
        sh_coefficients: vec![[0.0; 3]],
        sh_degree: 0,
        min_scale: None,
    }
}
