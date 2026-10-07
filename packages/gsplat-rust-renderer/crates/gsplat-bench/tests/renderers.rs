//! GPU/asset contracts, selected by the pytest integration runner with real skips.
use glam::{Mat4, Vec3};
use gsplat_bench::renderers::{Brush, Native, RenderEngine, Scene};
use gsplat_eval::Evaluator;
use gsplat_render::camera::{self, CameraSpec};
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
        mode: gsplat_core::RenderMode::Default,
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
fn assert_rgba8(pixels: &[f32]) {
    assert!(
        pixels.iter().all(|value| {
            (0.0..=1.0).contains(value) && (value * 255.0 - (value * 255.0).round()).abs() < 1e-5
        }),
        "timed packed output must contain decoded RGBA8 values"
    );
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
    let mut camera = camera::load_frames(&cameras, None).await.unwrap()[0]
        .camera
        .resized(256, 256);
    // Exercise principal-point mapping away from the image centre.
    camera.cx -= 13.0;
    camera.cy += 9.0;
    camera.validate().unwrap();
    let mut brush = Brush::new(&scene, &Default::default()).await;
    let reference = capture(&mut brush, &camera).await;
    brush.render(&camera, false).await.unwrap();
    brush.finish().unwrap();
    let packed = brush.read_rgba_f32().await.unwrap();
    assert_rgba8(&packed);
    let packed_score = Evaluator::new(false)
        .evaluate_renders(&packed, &reference, 256, 256)
        .await
        .unwrap();
    println!("Brush Packed versus Float: {packed_score:?}");
    assert!(packed_score.minimum_psnr() > 35.0);
    let mut ours = gsplat_bench::renderers::ours(&scene, 256, 256, &Default::default())
        .await
        .unwrap();
    // Speed renders into the packed target; readback must not score an untouched float buffer.
    RenderEngine::render(&mut ours, &camera, false)
        .await
        .unwrap();
    RenderEngine::finish(&ours).unwrap();
    let packed_pixels = ours.read_rgba_f32().await.unwrap();
    assert_rgba8(&packed_pixels);
    let packed_core_score = Evaluator::new(false)
        .evaluate_renders(&packed_pixels, &reference, 256, 256)
        .await
        .unwrap();
    assert!(
        packed_core_score.minimum_psnr() > 35.0,
        "packed core readback: {packed_core_score:?}"
    );
    let pixels = capture(&mut ours, &camera).await;
    let score = Evaluator::new(false)
        .evaluate_renders(&pixels, &reference, 256, 256)
        .await
        .unwrap();
    println!("ours off-centre float parity: {score:?}");
    assert!(score.minimum_psnr() >= 40.0);
    let stages = ours
        .stages(&camera)
        .await
        .unwrap()
        .expect("ours GPU stage timings");
    assert_eq!(stages.len(), 8);
    assert!(stages.iter().all(|s| s.ms.is_finite() && s.ms > 0.0));
    let stages = brush.stages(&camera).await.unwrap().unwrap();
    assert_eq!(stages.len(), 1);
    assert!(stages[0].ms > 0.0);
    println!("Brush device window: {} ms", stages[0].ms);
    let evaluator = Evaluator::new(false);
    let mut native = Native::new(&ply, scene.data.num_splats()).await.unwrap();
    native.capture_next_frame();
    native.render(&camera, false).await.unwrap();
    native.finish().unwrap();
    let native_timed = native.read_rgba_f32().await.unwrap();
    assert!(
        native_timed
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| pixel[3] == 1.0),
        "native speed validation must read the opaque timed target"
    );
    let timed_score = evaluator
        .evaluate_renders(&native_timed, &reference, 256, 256)
        .await
        .unwrap();
    println!("native timed target: {timed_score:?}");
    assert!(
        timed_score.rgb.psnr > 20.0,
        "native timed target: {timed_score:?}"
    );
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
    let mut second = Brush::new(&scene, &Default::default()).await;
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
    let mut a = Brush::new(&scene, &Default::default()).await;
    let mut b = Brush::new(&scene, &Default::default()).await;
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
#[ignore = "integration: garden sparse/0 assets; tests-integration handles skips"]
async fn garden_colmap_projects_observed_points() {
    use tokio::io::BufReader;
    let path = PathBuf::from(
        std::env::var("GSPLAT_TEST_COLMAP").expect("missing garden GSPLAT_TEST_COLMAP"),
    );
    let cameras: Vec<_> = camera::load_frames(&path, None)
        .await
        .unwrap()
        .into_iter()
        .map(|frame| frame.camera)
        .collect();
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
