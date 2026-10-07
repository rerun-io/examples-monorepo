//! Synthetic contracts for branches absent from the stored SH3 scenes.
use super::*;
use glam::Mat4;
use gsplat_core::CameraModel;

fn synthetic_scene(degree: u32) -> Scene {
    let n = 12;
    Scene {
        data: brush_serde::import::SplatData {
            means: (0..n)
                .flat_map(|i| {
                    [
                        (i % 4) as f32 * 0.3 - 0.45,
                        (i / 4) as f32 * 0.3 - 0.3,
                        2.0 + i as f32 * 0.07,
                    ]
                })
                .collect(),
            rotations: None,
            log_scales: Some((0..n).flat_map(|_| [-2.7, -2.3, -3.0]).collect()),
            sh_coeffs: Some(
                (0..n * (degree as usize + 1).pow(2) * 3)
                    .map(|i| ((i * 37 % 101) as f32 - 50.0) / 100.0)
                    .collect(),
            ),
            raw_opacities: Some((0..n).map(|i| i as f32 * 0.1).collect()),
        },
        mode: gsplat_core::RenderMode::Default,
        center: Vec3::new(0.0, 0.0, 2.0),
        extent: 1.0,
    }
}
fn camera(width: u32, height: u32) -> CameraSpec {
    CameraSpec {
        world_from_camera: Mat4::IDENTITY.to_cols_array_2d(),
        width,
        height,
        fx: width as f32 * 0.8,
        fy: height as f32 * 0.8,
        cx: width as f32 * 0.5,
        cy: height as f32 * 0.5,
        model: CameraModel::Pinhole,
    }
}
async fn compare(
    scene: &Scene,
    camera: &CameraSpec,
    floor: Option<Vec<f32>>,
) -> (Vec<f32>, Counts) {
    let mut reference = Brush::new(scene, &Default::default()).await;
    let mut raw = gsplat_render::raw_splats(&scene.data).unwrap();
    raw.min_scale = floor.clone();
    if let Some(floor) = floor {
        reference.splats = reference.splats.with_min_scale(Tensor::from_data(
            burn::tensor::TensorData::new(floor, [scene.data.num_splats()]),
            &reference.device,
        ));
    }
    let mut ours = gsplat_render::Renderer::new(
        &raw,
        gsplat_core::RenderOptions {
            render_mode: scene.mode,
            ..Default::default()
        },
        glam::uvec2(camera.width, camera.height),
        64,
    )
    .await
    .unwrap();
    let stats = RenderEngine::render(&mut ours, camera, true).await.unwrap();
    let oracle_stats = reference.render(camera, true).await.unwrap();
    reference.finish().unwrap();
    let actual = ours.read_rgba().unwrap();
    let expected = reference.read_rgba_f32().await.unwrap();
    assert_eq!(stats.visible, oracle_stats.visible);
    let maximum = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum < 1e-3, "maximum unclipped RGBA error {maximum}");
    assert!(actual.as_chunks::<4>().0.iter().any(|pixel| pixel[3] > 0.1));
    (actual, stats)
}

#[tokio::test]
#[ignore = "integration: GPU"]
async fn sh_1_2_4_and_varying_scale_floors_match_brush() {
    for degree in [1, 2, 4] {
        let mut scene = synthetic_scene(degree);
        compare(&scene, &camera(192, 128), None).await;
        scene.mode = gsplat_core::RenderMode::Mip;
        let floor = (0..scene.data.num_splats())
            .map(|i| i as f32 * 0.003)
            .collect();
        compare(&scene, &camera(192, 128), Some(floor)).await;
    }
}

#[tokio::test]
#[ignore = "integration: GPU"]
async fn fisheye_keeps_visible_splats_behind_the_camera() {
    let mut scene = synthetic_scene(1);
    scene.data.means = vec![2.0, 0.0, -0.35, 2.0, 0.0, 0.35];
    scene.data.log_scales = Some(vec![-3.0; 6]);
    scene.data.sh_coeffs = Some(vec![0.1; 2 * 4 * 3]);
    scene.data.raw_opacities = Some(vec![2.0; 2]);
    let mut camera = camera(256, 192);
    camera.model = CameraModel::KannalaBrandt4 {
        k1: 0.01,
        k2: 0.0,
        k3: 0.0,
        k4: 0.0,
    };
    camera.fx = camera.model.fov_to_focal(220.0_f64.to_radians(), 256) as f32;
    camera.fy = camera.model.fov_to_focal(160.0_f64.to_radians(), 192) as f32;
    let (_, stats) = compare(&scene, &camera, None).await;
    assert_eq!(
        stats.visible,
        Some(2),
        "the negative-z splat must remain visible"
    );
}

#[tokio::test]
#[ignore = "integration: GPU with an 8K float output binding"]
async fn eight_k_sorts_five_digits_and_dispatches_beyond_65535_tiles() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    if adapter.limits().max_storage_buffer_binding_size < 7680 * 4320 * 16 {
        eprintln!("SKIP: adapter storage binding limit cannot hold an 8K float target");
        return;
    }
    let mut scene = synthetic_scene(0);
    scene.data.means = vec![0.0, 0.0, 2.0];
    scene.data.log_scales = Some(vec![-7.0; 3]);
    scene.data.sh_coeffs = Some(vec![0.1; 3]);
    scene.data.raw_opacities = Some(vec![2.0]);
    let mut camera = camera(7680, 4320);
    camera.cx = 7600.5;
    camera.cy = 4200.5;
    let (pixels, stats) = compare(&scene, &camera, None).await;
    assert_eq!(stats.visible, Some(1));
    // This pixel lies in tile 126235, beyond the first 65535-workgroup row.
    assert!(pixels[(4200 * 7680 + 7600) * 4 + 3] > 0.8);
}
