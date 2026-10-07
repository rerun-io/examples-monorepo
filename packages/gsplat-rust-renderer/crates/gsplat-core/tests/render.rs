//! Analytic output formats and target validation through the public renderer API.
mod common;
use glam::{Quat, Vec3};
use gsplat_core::{Camera, CameraModel, RenderMode, RenderOptions, Renderer, Splats, Target};

#[test]
#[ignore = "integration: GPU"]
fn centered_gaussian_has_analytic_color_alpha_and_background() {
    analytic_render(glam::Affine3A::IDENTITY, 0.182_996_84);
}
#[test]
#[ignore = "integration: GPU"]
fn instance_affine_preserves_projection_and_covariance() {
    analytic_render(
        glam::Affine3A::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 1.0),
            Quat::from_rotation_z(0.7),
            Vec3::new(0.4, -0.3, 1.0),
        ),
        0.106_753_71,
    );
}
fn analytic_render(world_from_local: glam::Affine3A, alpha_three_pixels_right: f32) {
    let (device, queue) = common::gpu();
    let renderer = Renderer::new(&device, &queue).unwrap();
    let p = world_from_local
        .inverse()
        .transform_point3(Vec3::new(0.0, 0.0, 2.0));
    let scene = renderer
        .upload(&Splats {
            transforms: vec![[p.x, p.y, p.z, 1.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0]],
            raw_opacities: vec![0.0],
            sh_coefficients: vec![[0.0; 3]],
            sh_degree: 0,
            min_scale: None,
        })
        .unwrap();
    let mut view = renderer.create_view(&scene, 64).unwrap();
    let queries = device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
        .then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("render contract timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 10,
            })
        });
    view.set_timestamp_queries(queries.clone()).unwrap();
    let camera = common::pinhole_camera(33);
    let options = RenderOptions {
        world_from_local,
        background: Vec3::new(0.2, 0.4, 0.6),
        ..Default::default()
    };
    let float = common::upload(&device, &vec![[0.0f32; 4]; 33 * 33]);
    let packed = common::upload(&device, &vec![0u32; 33 * 33]);
    let texture = texture(&device, wgpu::TextureFormat::Rgba8Unorm);
    let texture_view = texture.create_view(&Default::default());
    let texture_copy = common::upload(&device, &vec![0u8; 256 * 33]);
    for target in [
        Target::Float(float.clone()),
        Target::Packed(packed.clone()),
        Target::Texture(texture_view.clone()),
    ] {
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(&mut encoder, &mut view, &camera, &options, target)
            .unwrap();
        copy_texture(&mut encoder, &texture, &texture_copy);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(!view.poll_feedback().unwrap().unwrap().needs_rerender);
        if let Some(queries) = &queries {
            let resolved = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 80,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.resolve_query_set(queries, 0..10, &resolved, 0);
            queue.submit([encoder.finish()]);
            let ticks = common::read::<u64>(&device, &queue, &resolved, 10);
            assert!(ticks[0] > 0 && ticks[9] > ticks[0]);
            assert!(ticks.windows(2).all(|pair| pair[1] >= pair[0]), "{ticks:?}");
            let total: u64 = (0..gsplat_core::STAGE_NAMES.len())
                .map(gsplat_core::stage_queries)
                .map(|(start, end)| ticks[end] - ticks[start])
                .sum();
            assert_eq!(total, ticks[9] - ticks[0]);
        }
    }
    let rgba = common::read::<[f32; 4]>(&device, &queue, &float, 33 * 33);
    for (actual, expected) in rgba[16 * 33 + 16].iter().zip([0.35, 0.45, 0.55, 0.5]) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    assert_eq!(rgba[0], [0.2, 0.4, 0.6, 0.0]);
    // Independent Gaussian covariance: (f/2)^2 exp(-4) A A^T + 0.3 I.
    assert!((rgba[16 * 33 + 19][3] - alpha_three_pixels_right).abs() < 1e-5);
    let bytes = common::read::<u8>(&device, &queue, &packed, 33 * 33 * 4);
    assert_eq!(
        &bytes[(16 * 33 + 16) * 4..(16 * 33 + 17) * 4],
        &[89, 114, 140, 127]
    );
    let bytes = common::read::<u8>(&device, &queue, &texture_copy, 256 * 33);
    let center = &bytes[16 * 256 + 16 * 4..16 * 256 + 17 * 4];
    assert_eq!(&center[..3], &[89, 115, 140]);
    assert!(center[3].abs_diff(128) <= 1);

    if world_from_local == glam::Affine3A::IDENTITY {
        // The same uploaded scene supports two blueprint modes. For this
        // isotropic Gaussian mip compensation is v/(v+0.1), v=4.1769916755.
        for (mode, alpha) in [(RenderMode::Mip, 0.488_309_54), (RenderMode::Default, 0.5)] {
            let mut encoder = device.create_command_encoder(&Default::default());
            renderer
                .render(
                    &mut encoder,
                    &mut view,
                    &camera,
                    &RenderOptions {
                        render_mode: mode,
                        ..options
                    },
                    Target::Float(float.clone()),
                )
                .unwrap();
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            assert!(!view.poll_feedback().unwrap().unwrap().needs_rerender);
            let rgba = common::read::<[f32; 4]>(&device, &queue, &float, 33 * 33);
            assert!((rgba[16 * 33 + 16][3] - alpha).abs() < 1e-5);
        }
    }

    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(
        renderer
            .render(
                &mut encoder,
                &mut view,
                &camera,
                &options,
                Target::Float(packed.clone())
            )
            .is_err()
    );
    let mut wrong_camera = camera;
    wrong_camera.size.x += 1;
    assert!(
        renderer
            .render(
                &mut encoder,
                &mut view,
                &wrong_camera,
                &options,
                Target::Texture(texture_view.clone())
            )
            .is_err()
    );
}

#[test]
#[ignore = "integration: GPU"]
fn tiny_invertible_instances_are_valid() {
    analytic_render(glam::Affine3A::from_scale(Vec3::splat(0.0001)), 0.0);
}

#[test]
#[ignore = "integration: GPU"]
fn optional_depth_is_alpha_weighted_and_normal_color_is_preserved() {
    let (device, queue) = common::gpu();
    let renderer = Renderer::new(&device, &queue).unwrap();
    let scene = renderer
        .upload(&Splats {
            transforms: [2.0, 4.0]
                .map(|z| [0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0])
                .to_vec(),
            raw_opacities: vec![0.0; 2],
            sh_coefficients: vec![[0.0; 3]; 2],
            sh_degree: 0,
            min_scale: None,
        })
        .unwrap();
    let mut view = renderer.create_view(&scene, 64).unwrap();
    let camera = common::pinhole_camera(33);
    let color = texture(&device, wgpu::TextureFormat::Rgba8Unorm);
    let depth = texture(&device, wgpu::TextureFormat::R32Float);
    let output = common::upload(&device, &vec![0.0f32; 64 * 33]);
    let color_output = common::upload(&device, &vec![0u8; 256 * 33]);
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer
        .render(
            &mut encoder,
            &mut view,
            &camera,
            &RenderOptions::default(),
            Target::TextureDepth {
                color: color.create_view(&Default::default()),
                depth: depth.create_view(&Default::default()),
            },
        )
        .unwrap();
    copy_texture(&mut encoder, &depth, &output);
    copy_texture(&mut encoder, &color, &color_output);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert!(!view.poll_feedback().unwrap().unwrap().needs_rerender);
    let values: Vec<f32> = common::read(&device, &queue, &output, 64 * 33);
    assert!((values[16 * 64 + 16] - 8.0 / 3.0).abs() < 1e-5);
    assert_eq!(values[0], 0.0);
    let rgba: Vec<u8> = common::read(&device, &queue, &color_output, 256 * 33);
    for (actual, expected) in rgba[16 * 256 + 16 * 4..16 * 256 + 17 * 4]
        .iter()
        .zip([96, 96, 96, 191])
    {
        assert!(actual.abs_diff(expected) <= 1);
    }
}

fn texture(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 33,
            height: 33,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
fn copy_texture(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    destination: &wgpu::Buffer,
) {
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: destination,
            // Both formats use four-byte texels; 33-pixel rows require padding.
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(33),
            },
        },
        texture.size(),
    );
}
