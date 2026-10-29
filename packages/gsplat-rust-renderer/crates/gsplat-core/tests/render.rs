//! Analytic output formats and target validation through the public renderer API.
#![cfg(feature = "testing")]
use glam::{Quat, Vec3};
use gsplat_core::test_utils as common;
use gsplat_core::{HexelVariant, Hexels, RenderMode, RenderOptions, Renderer, Splats, Target};

#[test]
#[ignore = "integration: GPU"]
fn centered_gaussian_has_analytic_color_alpha_and_background() {
    let (device, queue) = common::gpu();
    analytic_render(&device, &queue, glam::Affine3A::IDENTITY, 0.182_996_84);
}

#[test]
#[ignore = "integration: GPU"]
fn instance_affine_preserves_projection_and_covariance() {
    let (device, queue) = common::gpu();
    analytic_render(
        &device,
        &queue,
        glam::Affine3A::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 1.0),
            Quat::from_rotation_z(0.7),
            Vec3::new(0.4, -0.3, 1.0),
        ),
        0.106_753_71,
    );
}

fn analytic_render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    world_from_local: glam::Affine3A,
    alpha_three_pixels_right: f32,
) {
    let renderer = Renderer::new(device).expect("create analytic renderer");
    let p = world_from_local
        .inverse()
        .transform_point3(Vec3::new(0.0, 0.0, 2.0));
    let scene = renderer
        .upload(&common::splat(p.to_array(), -2.0, 0.0))
        .expect("upload analytic splat");
    let mut view = renderer
        .create_view(&scene, 64)
        .expect("create analytic view");
    let queries = device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
        .then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("render contract timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: gsplat_core::QUERY_COUNT,
            })
        });
    view.set_timestamp_queries(queries.clone())
        .expect("set analytic timestamp queries");
    let camera = common::pinhole_camera(33);
    let options = RenderOptions {
        world_from_local,
        background: Vec3::new(0.2, 0.4, 0.6),
        ..Default::default()
    };
    let float = common::upload(device, &vec![[0.0f32; 4]; 33 * 33]);
    let packed = common::upload(device, &vec![0u32; 33 * 33]);
    let texture = texture(device, wgpu::TextureFormat::Rgba8Unorm, 33);
    let texture_view = texture.create_view(&Default::default());
    let texture_copy = common::upload(device, &vec![0u8; 256 * 33]);
    for target in [
        Target::Float(float.clone()),
        Target::Packed(packed.clone()),
        Target::Texture(texture_view.clone()),
    ] {
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(queue, &mut encoder, &mut view, &camera, &options, target)
            .expect("render analytic target");
        copy_texture(&mut encoder, &texture, &texture_copy);
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for analytic render");
        assert!(
            !view
                .poll_feedback()
                .expect("receive analytic feedback")
                .expect("read analytic feedback")
                .needs_rerender
        );
        if let Some(queries) = &queries {
            let resolved = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(gsplat_core::QUERY_COUNT) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.resolve_query_set(queries, 0..gsplat_core::QUERY_COUNT, &resolved, 0);
            queue.submit([encoder.finish()]);
            let ticks =
                common::read::<u64>(device, queue, &resolved, gsplat_core::QUERY_COUNT as usize);
            let stages = gsplat_core::stage_ms(&ticks, queue.get_timestamp_period())
                .expect("decode analytic stage timings");
            let total =
                (ticks[9] - ticks[0]) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
            assert!((stages.iter().sum::<f64>() - total).abs() < 1e-8);
        }
    }
    let rgba = common::read::<[f32; 4]>(device, queue, &float, 33 * 33);
    for (actual, expected) in rgba[16 * 33 + 16].iter().zip([0.35, 0.45, 0.55, 0.5]) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    assert_eq!(rgba[0], [0.2, 0.4, 0.6, 0.0]);
    // Independent Gaussian covariance: (f/2)^2 exp(-4) A A^T + 0.3 I.
    assert!((rgba[16 * 33 + 19][3] - alpha_three_pixels_right).abs() < 1e-5);
    let bytes = common::read::<u8>(device, queue, &packed, 33 * 33 * 4);
    assert_eq!(
        &bytes[(16 * 33 + 16) * 4..(16 * 33 + 17) * 4],
        &[89, 114, 140, 127]
    );
    let bytes = common::read::<u8>(device, queue, &texture_copy, 256 * 33);
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
                    queue,
                    &mut encoder,
                    &mut view,
                    &camera,
                    &RenderOptions {
                        render_mode: mode,
                        ..options
                    },
                    Target::Float(float.clone()),
                )
                .expect("render analytic mode");
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("wait for analytic mode");
            assert!(
                !view
                    .poll_feedback()
                    .expect("receive analytic mode feedback")
                    .expect("read analytic mode feedback")
                    .needs_rerender
            );
            let rgba = common::read::<[f32; 4]>(device, queue, &float, 33 * 33);
            assert!((rgba[16 * 33 + 16][3] - alpha).abs() < 1e-5);
        }
    }

    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(
        renderer
            .render(
                queue,
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
                queue,
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
    let (device, queue) = common::gpu();
    analytic_render(
        &device,
        &queue,
        glam::Affine3A::from_scale(Vec3::splat(0.0001)),
        0.0,
    );
}

#[test]
#[ignore = "integration: GPU"]
fn optional_depth_is_alpha_weighted_and_normal_color_is_preserved() {
    let (device, queue) = &common::gpu();
    let renderer = Renderer::new(device).unwrap();
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
    let color = texture(device, wgpu::TextureFormat::Rgba8Unorm, 33);
    let depth = texture(device, wgpu::TextureFormat::R32Float, 33);
    let id = texture(device, wgpu::TextureFormat::R32Uint, 33);
    let output = common::upload(device, &vec![0.0f32; 64 * 33]);
    let color_output = common::upload(device, &vec![0u8; 256 * 33]);
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer
        .render(
            queue,
            &mut encoder,
            &mut view,
            &camera,
            &RenderOptions::default(),
            Target::TextureDepth {
                color: color.create_view(&Default::default()),
                depth: depth.create_view(&Default::default()),
                id: id.create_view(&Default::default()),
            },
        )
        .unwrap();
    copy_texture(&mut encoder, &depth, &output);
    copy_texture(&mut encoder, &color, &color_output);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert!(!view.poll_feedback().unwrap().unwrap().needs_rerender);
    let values: Vec<f32> = common::read(device, queue, &output, 64 * 33);
    assert!((values[16 * 64 + 16] - 8.0 / 3.0).abs() < 1e-5);
    assert_eq!(values[0], 0.0);
    let rgba: Vec<u8> = common::read(device, queue, &color_output, 256 * 33);
    for (actual, expected) in rgba[16 * 256 + 16 * 4..16 * 256 + 17 * 4]
        .iter()
        .zip([96, 96, 96, 191])
    {
        assert!(actual.abs_diff(expected) <= 1);
    }
}

/// Logged rows (global ids) 0..5: one behind the camera, then the left pixel's strong back layer, the
/// right pixel's back layer, the left pixel's weak front layer and the right pixel's opaque front layer.
/// The depth sort reorders them, so a compact or intersection index would not match these ids.
const PICK_ROWS: [([f32; 3], f32); 5] = [
    ([0.0, 0.0, -3.0], 0.95),
    ([-0.2647 * 4.0, 0.0, 4.0], 0.95),
    ([0.2818 * 3.0, 0.0, 3.0], 0.95),
    ([-0.2647 * 2.0, 0.0, 2.0], 0.4),
    ([0.2818 * 2.5, 0.0, 2.5], 0.95),
];

/// The id texture of `TextureDepth` stores the global row of the median contributor (the last one
/// composited while transmittance exceeds one half).
#[test]
#[ignore = "integration: GPU"]
fn median_contributor_ids_are_global_rows_for_splats_and_hexels() {
    let (device, queue) = &common::gpu();
    let renderer = Renderer::new(device).unwrap();
    let splats = Splats {
        transforms: PICK_ROWS
            .iter()
            .map(|&([x, y, z], _)| {
                let log_scale = (0.0512 * z.abs()).ln();
                [x, y, z, 1.0, 0.0, 0.0, 0.0, log_scale, log_scale, log_scale]
            })
            .collect(),
        raw_opacities: PICK_ROWS
            .iter()
            .map(|&(_, o)| (o / (1.0 - o)).ln())
            .collect(),
        sh_coefficients: vec![[0.0; 3]; PICK_ROWS.len()],
        sh_degree: 0,
        min_scale: None,
    };
    let hexels = Hexels {
        variant: HexelVariant::V1,
        params: PICK_ROWS
            .iter()
            .flat_map(|&(centre, opacity)| {
                let scale = 0.12 * centre[2].abs();
                let mut row = centre.to_vec();
                row.extend([1.0, 0.0, 0.0, 0.0, scale, scale, opacity, 200.0]);
                row.extend([0.5; 3].into_iter().chain([0.0; 6]));
                row
            })
            .collect(),
    };
    let camera = common::pinhole_camera(64);
    let options = RenderOptions::default();
    for scene in [
        renderer.upload(&splats).unwrap(),
        renderer.upload_hexels(&hexels).unwrap(),
    ] {
        let mut view = renderer.create_view(&scene, 256).unwrap();
        let [color, depth, id] = [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureFormat::R32Uint,
        ]
        .map(|format| texture(device, format, 64));
        let view_of = |t: &wgpu::Texture| t.create_view(&Default::default());
        let target = Target::TextureDepth {
            color: view_of(&color),
            depth: view_of(&depth),
            id: view_of(&id),
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(queue, &mut encoder, &mut view, &camera, &options, target)
            .unwrap();
        let ids = common::upload(device, &vec![0u8; 256 * 64]);
        copy_texture(&mut encoder, &id, &ids);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(!view.poll_feedback().unwrap().unwrap().needs_rerender);
        let ids = common::read::<u32>(device, queue, &ids, 64 * 64);
        let at = |x: usize, y: usize| ids[y * 64 + x];
        assert_eq!(at(16, 32), 1, "weak front over strong back: the back row");
        assert_eq!(at(48, 32), 4, "opaque front: the front row");
        assert_eq!(at(32, 2), u32::MAX, "empty pixel");
        assert_eq!(at(2, 62), u32::MAX, "empty corner");
    }
}

fn texture(device: &wgpu::Device, format: wgpu::TextureFormat, size: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size,
            height: size,
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
            // Every format here uses four-byte texels; rows of at most 64 pixels pad to 256 bytes.
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(texture.height()),
            },
        },
        texture.size(),
    );
}
