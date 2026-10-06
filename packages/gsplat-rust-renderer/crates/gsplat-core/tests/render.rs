//! Analytic output formats and target validation through the public renderer API.
mod common;
use glam::{Quat, UVec2, Vec2, Vec3};
use gsplat_core::{Camera, CameraModel, RenderMode, RenderOptions, Renderer, Splats, Target};

#[test]
#[ignore = "integration: GPU"]
fn centered_gaussian_has_analytic_color_alpha_and_background() {
    let (device, queue) = common::gpu();
    let renderer = Renderer::new(&device, &queue).unwrap();
    let scene = renderer
        .upload(
            &Splats {
                transforms: vec![[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0]],
                raw_opacities: vec![0.0],
                sh_coefficients: vec![[0.0; 3]],
                sh_degree: 0,
                min_scale: None,
            },
            RenderMode::Default,
        )
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
    let camera = Camera {
        model: CameraModel::Pinhole,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        fov_x: 1.0,
        fov_y: 1.0,
        center_uv: Vec2::splat(0.5),
        size: UVec2::splat(33),
    };
    let options = RenderOptions {
        background: Vec3::new(0.2, 0.4, 0.6),
        ..Default::default()
    };
    let float = common::upload(&device, &vec![[0.0f32; 4]; 33 * 33]);
    let packed = common::upload(&device, &vec![0u32; 33 * 33]);
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
    let texture_view = texture.create_view(&Default::default());
    let texture_copy = common::upload(&device, &vec![0u8; 256 * 33]);
    for target in [
        Target::Float(&float),
        Target::Packed(&packed),
        Target::Texture(&texture_view),
    ] {
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(&mut encoder, &mut view, &camera, &options, target)
            .unwrap();
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &texture_copy,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(33),
                },
            },
            texture.size(),
        );
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
            let total: u64 = gsplat_core::STAGE_QUERIES
                .iter()
                .map(|&(start, end)| ticks[end] - ticks[start])
                .sum();
            assert_eq!(total, ticks[9] - ticks[0]);
        }
    }
    let rgba = common::read::<[f32; 4]>(&device, &queue, &float, 33 * 33);
    for (actual, expected) in rgba[16 * 33 + 16].iter().zip([0.35, 0.45, 0.55, 0.5]) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    assert_eq!(rgba[0], [0.2, 0.4, 0.6, 0.0]);
    let bytes = common::read::<u8>(&device, &queue, &packed, 33 * 33 * 4);
    assert_eq!(
        &bytes[(16 * 33 + 16) * 4..(16 * 33 + 17) * 4],
        &[89, 114, 140, 127]
    );
    let bytes = common::read::<u8>(&device, &queue, &texture_copy, 256 * 33);
    let center = &bytes[16 * 256 + 16 * 4..16 * 256 + 17 * 4];
    assert_eq!(&center[..3], &[89, 115, 140]);
    assert!(center[3].abs_diff(128) <= 1);

    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(
        renderer
            .render(
                &mut encoder,
                &mut view,
                &camera,
                &options,
                Target::Float(&packed)
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
                Target::Texture(&texture_view)
            )
            .is_err()
    );
}
