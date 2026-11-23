//! GPU contract helpers shared by unit and integration tests.
#![allow(dead_code)]
use wgpu::util::DeviceExt as _;

/// The test device; `GSPLAT_NO_SUBGROUPS=1` withholds `SUBGROUP`, so the whole GPU suite also
/// checks the workgroup-memory scan and sort kernels.
pub fn gpu() -> (wgpu::Device, wgpu::Queue) {
    gpu_with(crate::subgroups_allowed())
}

/// The test device with or without `SUBGROUP` (which the adapter must offer when requested).
pub fn gpu_with(subgroups: bool) -> (wgpu::Device, wgpu::Queue) {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("request GPU test adapter");
    println!(
        "core test adapter: {:?}, subgroups: {subgroups}",
        adapter.get_info()
    );
    let subgroup = if subgroups {
        wgpu::Features::SUBGROUP
    } else {
        wgpu::Features::empty()
    };
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: subgroup | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("request GPU test device")
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
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    encoder.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
        tx.send(result).expect("send test readback status");
    });
    queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("wait for test readback");
    rx.recv()
        .expect("receive test readback status")
        .expect("map test readback buffer");
    bytemuck::cast_slice(
        &staging
            .get_mapped_range(..)
            .expect("access test readback buffer"),
    )
    .to_vec()
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

/// What [`asog_gpu_load`] reads back.
pub struct AsogGpuLoad {
    pub gpu_rows: Vec<f32>,
    pub cpu_rows: Vec<f32>,
    /// Per time: (transforms, raw opacity).
    pub evaluated: Vec<(Vec<f32>, Vec<f32>)>,
}

/// The GPU load path of an entropy-decoded `.asog`, read back next to the CPU oracle.
///
/// `Renderer::upload_asog` (`sog_dequant.wgsl` + `asog_motion.wgsl`) gives (GPU motion rows, CPU `AsogParts::motion` rows packed the
/// same way, `ASOG_STRIDE` f32 each), then for each time in `times` the GPU motion pass on those rows as (transforms
/// f32 x 10 per row, raw opacity per row).
pub fn asog_gpu_load(parts: &crate::asog::AsogParts, times: &[f32]) -> AsogGpuLoad {
    use crate::motion_gpu::{ASOG_STRIDE, Deform};
    let (device, queue) = gpu();
    let renderer = crate::Renderer::new(&device).expect("renderer");
    let scene = renderer.upload_asog(&queue, parts).expect("upload_asog");
    let n = scene.len() as usize;
    let motion = scene.motion().expect("animated scene");
    let gpu_rows = read(&device, &queue, &motion.params, n * ASOG_STRIDE);
    let cpu_rows = crate::motion_gpu::pack(&parts.motion());
    let kernels = crate::kernels::Kernels::new(&device);
    let mut deform = Deform::new(
        &device,
        &kernels.trigs_deform,
        motion,
        scene.transforms().expect("splat scene"),
        scene.len(),
        u64::MAX,
    )
    .expect("deform buffers");
    let evaluated = times
        .iter()
        .map(|&t| {
            let mut encoder = device.create_command_encoder(&Default::default());
            deform.encode(&mut encoder, &queue, &kernels.trigs_deform, 0, t, None);
            queue.submit([encoder.finish()]);
            (
                read(&device, &queue, &deform.transforms, n * 10),
                read(&device, &queue, &deform.opacity, n),
            )
        })
        .collect();
    AsogGpuLoad {
        gpu_rows,
        cpu_rows,
        evaluated,
    }
}
