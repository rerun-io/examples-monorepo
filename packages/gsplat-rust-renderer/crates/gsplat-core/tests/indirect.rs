//! Overflow preserves the old image, then feedback grows and rerenders it exactly.
mod common;
use glam::{Quat, UVec2, Vec2, Vec3};
use gsplat_core::{Camera, CameraModel, RenderMode, RenderOptions, Renderer, Splats, Target};

#[test]
#[ignore = "integration: GPU"]
fn overflow_preserves_target_then_grows_and_rerenders_exactly() {
    let (device, queue) = common::gpu();
    let renderer = Renderer::new(&device, &queue).unwrap();
    let scene = renderer
        .upload(
            &Splats {
                transforms: vec![[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0]],
                raw_opacities: vec![2.0],
                sh_coefficients: vec![[0.0; 3]],
                sh_degree: 0,
                min_scale: None,
            },
            RenderMode::Default,
        )
        .unwrap();
    let mut small = renderer.create_view(&scene, 1).unwrap();
    let mut reference = renderer.create_view(&scene, 64).unwrap();
    let camera = Camera {
        model: CameraModel::Pinhole,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        fov_x: 1.0,
        fov_y: 1.0,
        center_uv: Vec2::splat(0.5),
        size: UVec2::splat(64),
    };
    let target = common::upload(&device, &vec![[0.0f32; 4]; 64 * 64]);
    let mut outputs = Vec::new();
    for frame in 0..3 {
        let view = if frame == 2 {
            &mut reference
        } else {
            &mut small
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(
                &mut encoder,
                view,
                &camera,
                &RenderOptions::default(),
                Target::Float(&target),
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        outputs.push(common::read::<[f32; 4]>(&device, &queue, &target, 64 * 64));
        let stats = view.poll_feedback().unwrap().unwrap();
        assert_eq!(stats.needs_rerender, frame == 0);
        if frame != 2 {
            assert_eq!(stats.overflow_events, 1);
        }
    }
    assert!(outputs[0].iter().all(|pixel| *pixel == [0.0; 4]));
    assert_eq!(outputs[1], outputs[2]);
    assert!(outputs[1].iter().any(|pixel| pixel[3] > 0.0));
    // Empty views reuse the high-water allocation; resize changes no sort buffers.
    let mut empty = camera;
    empty.position.z = 10.0;
    empty.size = UVec2::splat(32);
    let mut previous_capacity = 0;
    for camera in [camera, empty, camera] {
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(
                &mut encoder,
                &mut small,
                &camera,
                &RenderOptions::default(),
                Target::Float(&target),
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let stats = small.poll_feedback().unwrap().unwrap();
        assert!(!stats.needs_rerender);
        assert!(stats.intersection_capacity >= previous_capacity);
        previous_capacity = stats.intersection_capacity;
    }
}
