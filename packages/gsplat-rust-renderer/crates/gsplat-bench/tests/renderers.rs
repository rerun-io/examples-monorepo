//! GPU/asset contracts, selected by the pytest integration runner with real skips.
use glam::{Mat4, Vec3};
use gsplat_bench::{
    camera::{self, CameraSpec},
    renderers::{Brush, Native, Old, RenderEngine, Scene},
};
use gsplat_eval::Evaluator;
use std::path::PathBuf;

fn one_splat(log_scale: f32) -> Scene {
    Scene {
        data: brush_serde::import::SplatData {
            means: vec![0.0; 3],
            rotations: None,
            log_scales: Some(vec![log_scale; 3]),
            sh_coeffs: Some(vec![0.1, 0.2, 0.3]),
            raw_opacities: Some(vec![0.0]),
        },
        mode: brush_render::gaussian_splats::SplatRenderMode::Default,
        center: Vec3::ZERO,
        extent: 1.0,
    }
}
async fn capture<R: RenderEngine>(renderer: &mut R, camera: &CameraSpec) -> Vec<f32> {
    println!("adapter: {:?}", renderer.adapter());
    assert!(!renderer.adapter().name.to_lowercase().contains("llvmpipe"));
    renderer.render(camera, true).await.unwrap();
    renderer.finish().unwrap();
    let pixels = renderer.read_rgba_f32().await.unwrap();
    assert_eq!(
        pixels.len(),
        camera.width as usize * camera.height as usize * 4
    );
    assert!(pixels.iter().all(|v| v.is_finite()));
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[..3].iter().any(|v| *v > 0.05))
            .count()
            > 10
    );
    pixels
}
#[tokio::test]
#[ignore = "integration: GPU and Lego assets; tests-integration handles missing-asset skips"]
async fn all_renderers_nonblack_and_brush_identity() {
    let ply = PathBuf::from(
        std::env::var("GSPLAT_TEST_PLY").expect("missing GSPLAT_TEST_PLY; use tests-integration"),
    );
    let cameras =
        PathBuf::from(std::env::var("GSPLAT_TEST_CAMERAS").expect("missing GSPLAT_TEST_CAMERAS"));
    let scene = Scene::load(&ply).await.unwrap();
    let mut camera = camera::test_views(&cameras).unwrap()[0].resized(256, 256);
    // Exercise principal-point mapping away from the image centre.
    camera.cx -= 13.0;
    camera.cy += 9.0;
    camera.validate().unwrap();
    let mut brush = Brush::new(&scene).await;
    let reference = capture(&mut brush, &camera).await;
    brush.render(&camera, false).await.unwrap();
    brush.finish().unwrap();
    let packed = brush.read_rgba_f32().await.unwrap();
    let packed_score = Evaluator::new(false)
        .evaluate_renders(&packed, &reference, 256, 256)
        .await
        .unwrap();
    println!("Brush Packed versus Float: {packed_score:?}");
    assert!(packed_score.minimum_psnr() > 35.0);
    let mut old = Old::new(&scene, 256, 256).unwrap();
    let pixels = capture(&mut old, &camera).await;
    let evaluator = Evaluator::new(false);
    let old_score = evaluator
        .evaluate_renders(&pixels, &reference, 256, 256)
        .await
        .unwrap();
    println!("old off-centre parity: {old_score:?}");
    assert!(old_score.rgb.psnr > 35.0);
    let stages = old.stages(&camera).await.unwrap().unwrap();
    assert_eq!(stages.len(), 6);
    assert!(stages.iter().all(|s| s.ms > 0.0));
    let stages = brush.stages(&camera).await.unwrap().unwrap();
    assert_eq!(stages.len(), 1);
    assert!(stages[0].ms > 0.0);
    println!("Brush device window: {} ms", stages[0].ms);
    let mut native = Native::new(&ply, scene.data.num_splats()).await.unwrap();
    let pixels = capture(&mut native, &camera).await;
    let native_score = evaluator
        .evaluate_renders(&pixels, &reference, 256, 256)
        .await
        .unwrap();
    println!("native off-centre parity: {native_score:?}");
    assert!(native_score.rgb.psnr > 20.0);
    assert!(
        native_score.alpha_psnr > 20.0,
        "native screenshot must retain scene alpha"
    );
    let mut second = Brush::new(&scene).await;
    let pixels = capture(&mut second, &camera).await;
    let repeat = evaluator
        .evaluate_renders(&pixels, &reference, 256, 256)
        .await
        .unwrap();
    println!("independent Lego float repeat: {repeat:?}");
    assert!(repeat.minimum_psnr() > 80.0);
}
#[tokio::test]
#[ignore = "integration: GPU"]
async fn independent_brush_renders_have_exact_identity_on_one_splat() {
    let scene = one_splat(-1.5);
    let camera = CameraSpec::from_nerf(Mat4::from_translation(Vec3::Z * 3.0), 0.8, 64, 64);
    let mut a = Brush::new(&scene).await;
    let mut b = Brush::new(&scene).await;
    let a = capture(&mut a, &camera).await;
    let b = capture(&mut b, &camera).await;
    assert_eq!(a, b);
    let score = Evaluator::new(false)
        .evaluate_renders(&a, &b, 64, 64)
        .await
        .unwrap();
    assert!((score.minimum_psnr() - 100.0).abs() < 1e-4);
    assert!((score.rgb.ssim - 1.0).abs() < 1e-5);
}
#[tokio::test]
#[ignore = "integration: GPU 4K target"]
async fn old_core_renders_at_4k() {
    let scene = one_splat(-6.0);
    let camera = CameraSpec::from_nerf(Mat4::from_translation(Vec3::Z * 3.0), 0.8, 3840, 2160);
    let mut renderer = Old::new(&scene, 3840, 2160).unwrap();
    capture(&mut renderer, &camera).await;
}
#[tokio::test]
#[ignore = "integration: garden sparse/0 assets; tests-integration handles skips"]
async fn garden_colmap_projects_observed_points() {
    use tokio::io::BufReader;
    let path = PathBuf::from(
        std::env::var("GSPLAT_TEST_COLMAP").expect("missing garden GSPLAT_TEST_COLMAP"),
    );
    let cameras = camera::colmap_views(&path).await.unwrap();
    assert_eq!(cameras.len(), 185);
    let mut images = colmap_reader::read_images(
        BufReader::new(
            tokio::fs::File::open(path.join("images.bin"))
                .await
                .unwrap(),
        ),
        true,
        true,
    )
    .await
    .unwrap();
    images.sort_by_key(|i| i.id);
    let points = colmap_reader::read_points3d(
        BufReader::new(
            tokio::fs::File::open(path.join("points3D.bin"))
                .await
                .unwrap(),
        ),
        true,
        false,
    )
    .await
    .unwrap();
    let points: std::collections::HashMap<_, _> =
        points.into_iter().map(|p| (p.id, p.xyz)).collect();
    let mut errors = Vec::new();
    for (camera, image) in cameras.iter().zip(images).take(5) {
        camera.validate().unwrap();
        let data = image.points.unwrap();
        for (xy, id) in data.xys.iter().zip(data.point3d_ids) {
            if let Some(point) = points.get(&id) {
                errors.push(camera.project(*point).distance(*xy) as f64);
            }
        }
    }
    assert!(errors.len() > 1000);
    let error = gsplat_bench::statistics::median(&errors);
    println!(
        "garden COLMAP median residual {error} px, {} landmarks",
        errors.len()
    );
    assert!(error < 2.0);
}
