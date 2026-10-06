//! Analytic public-API checks; scene parity tests use the separate Brush oracle.
use glam::{Quat, UVec2, Vec2, Vec3};
use gsplat_core::{Camera, Capabilities, RenderOptions, Renderer, Splats, Target};

#[test]
#[ignore = "integration: GPU"]
fn centered_gaussian_has_analytic_color_alpha_and_background() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SUBGROUP,
        ..Default::default()
    }))
    .unwrap();
    let mut renderer =
        Renderer::new(&device, &queue, Capabilities::from_device(&device).unwrap()).unwrap();
    renderer
        .upload(&Splats {
            transforms: vec![[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0]],
            raw_opacities: vec![0.0],
            sh_coefficients: vec![[0.0; 3]],
            sh_degree: 0,
            min_scale: None,
        })
        .unwrap();
    let camera = Camera {
        model: gsplat_core::CameraModel::Pinhole,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        fov_x: 1.0,
        fov_y: 1.0,
        center_uv: Vec2::splat(0.5),
        size: UVec2::splat(33),
    };
    let target = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 33 * 33 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: target.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer
        .render(
            &mut encoder,
            &camera,
            &RenderOptions {
                background: Vec3::new(0.2, 0.4, 0.6),
                ..Default::default()
            },
            Target::Float(&target),
        )
        .unwrap();
    encoder.copy_buffer_to_buffer(&target, 0, &staging, 0, target.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = staging.get_mapped_range(..).unwrap();
    let rgba: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    for (actual, expected) in rgba[16 * 33 + 16].iter().zip([0.35, 0.45, 0.55, 0.5]) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    assert_eq!(rgba[0], [0.2, 0.4, 0.6, 0.0]);
    drop(mapped);
    staging.unmap();

    let packed = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 33 * 33 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 33,
            height: 33,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    for texture_output in [false, true] {
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(
                &mut encoder,
                &camera,
                &RenderOptions {
                    background: Vec3::new(0.2, 0.4, 0.6),
                    ..Default::default()
                },
                if texture_output {
                    Target::Texture(&view)
                } else {
                    Target::Packed(&packed)
                },
            )
            .unwrap();
        if texture_output {
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(33),
                    },
                },
                texture.size(),
            );
        } else {
            encoder.copy_buffer_to_buffer(&packed, 0, &staging, 0, packed.size());
        }
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = staging.get_mapped_range(..).unwrap();
        let stride = if texture_output { 256 } else { 33 * 4 };
        let center = &mapped[16 * stride + 16 * 4..16 * stride + 17 * 4];
        assert_eq!(
            &center[..3],
            if texture_output {
                &[89, 115, 140]
            } else {
                &[89, 114, 140]
            }
        );
        // exp() rounding can put the exactly-half alpha on either side of an UNORM tie.
        assert!(center[3].abs_diff(if texture_output { 128 } else { 127 }) <= 1);
        drop(mapped);
        staging.unmap();
    }
}
