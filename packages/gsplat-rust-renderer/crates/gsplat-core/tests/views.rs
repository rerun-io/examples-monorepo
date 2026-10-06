//! A shared upload must render independent views in one queue submission.
mod common;
use glam::{Quat, UVec2, Vec2, Vec3};
use gsplat_core::{Camera, CameraModel, RenderMode, RenderOptions, Renderer, Splats, Target};

#[test]
#[ignore = "integration: GPU"]
fn one_scene_renders_two_views_in_one_submit() {
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
    let mut views = [
        renderer.create_view(&scene, 64).unwrap(),
        renderer.create_view(&scene, 64).unwrap(),
    ];
    let targets = std::array::from_fn::<_, 2, _>(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 33 * 33 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let camera = Camera {
        model: CameraModel::Pinhole,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        fov_x: 1.0,
        fov_y: 1.0,
        center_uv: Vec2::splat(0.5),
        size: UVec2::splat(33),
    };
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    for _ in 0..8 {
        let mut encoder = device.create_command_encoder(&Default::default());
        for (index, view) in views.iter_mut().enumerate() {
            renderer
                .render(
                    &mut encoder,
                    view,
                    &camera,
                    &RenderOptions {
                        background: [Vec3::X, Vec3::Z][index],
                        ..Default::default()
                    },
                    Target::Float(&targets[index]),
                )
                .unwrap();
            encoder.copy_buffer_to_buffer(
                &targets[index],
                (16 * 33 + 16) * 16,
                &read,
                index as u64 * 16,
                16,
            );
        }
        let (tx, rx) = std::sync::mpsc::channel();
        encoder.map_buffer_on_submit(&read, wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap()
        });
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = read.get_mapped_range(..).unwrap();
        let pixels: &[[f32; 4]] = bytemuck::cast_slice(&data);
        assert_eq!(pixels, &[[0.75, 0.25, 0.25, 0.5], [0.25, 0.25, 0.75, 0.5]]);
        drop(data);
        read.unmap();
        for view in &mut views {
            let stats = view.poll_feedback().unwrap().unwrap();
            assert_eq!(stats.visible, 1);
            assert!(!stats.needs_rerender);
        }
    }
}
