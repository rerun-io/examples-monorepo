//! Public evaluator contracts, including optional GPU checks.
use gsplat_cli::metrics::published::rgb as published_rgb;
use gsplat_cli::{Convention, Evaluator, pair_directories};
use image::{DynamicImage, Rgba, RgbaImage};

#[test]
fn white_composition_truncates_and_checks_alpha() {
    let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([0, 128, 255, 128])));
    assert_eq!(
        published_rgb(&image),
        vec![126.0 / 255.0, 191.0 / 255.0, 1.0]
    );
}

#[test]
fn strict_directory_pairing_rejects_different_names() {
    let root = std::env::temp_dir().join(format!("gsplat-pairs-{}", std::process::id()));
    let render = root.join("render");
    let gt = root.join("gt");
    std::fs::create_dir_all(&render).unwrap();
    std::fs::create_dir_all(&gt).unwrap();
    assert!(pair_directories(&render, &gt).is_err());
    std::fs::write(render.join("one.png"), []).unwrap();
    std::fs::write(gt.join("two.png"), []).unwrap();
    assert!(pair_directories(&render, &gt).is_err());
    std::fs::rename(gt.join("two.png"), gt.join("one.png")).unwrap();
    assert_eq!(
        pair_directories(&render, &gt).unwrap(),
        vec![std::path::PathBuf::from("one.png")]
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
#[ignore = "integration: GPU Brush metric contracts"]
async fn brush_quantization_premultiplication_and_identity() {
    let evaluator = Evaluator::new(false);
    let gt = DynamicImage::ImageRgba8(RgbaImage::from_pixel(16, 16, Rgba([255, 0, 0, 128])));
    let rendered =
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(16, 16, image::Rgb([128, 0, 0])));
    let identity = evaluator
        .evaluate_pair(&rendered, &gt, Convention::Brush)
        .await
        .unwrap();
    assert!((identity.psnr - 100.0).abs() < 1e-4, "{identity:?}");
    assert!((identity.ssim - 1.0).abs() < 1e-5, "{identity:?}");
    let black = DynamicImage::ImageRgb8(image::RgbImage::new(16, 16));
    let score = evaluator
        .evaluate_pair(&rendered, &black, Convention::Brush)
        .await
        .unwrap();
    let expected = 10.0 * (3.0 / (128.0_f64 / 255.0).powi(2)).log10();
    assert!((score.psnr - expected).abs() < 1e-4, "{score:?}");
    let reverse = evaluator
        .evaluate_pair(&black, &rendered, Convention::Brush)
        .await
        .unwrap();
    assert!((score.ssim - reverse.ssim).abs() < 1e-5);
}

#[tokio::test]
#[ignore = "golden: GPU LPIPS reference through evaluator API"]
async fn evaluator_lpips_matches_reference() {
    let a = image::load_from_memory(include_bytes!("fixtures/apple.png")).unwrap();
    let b = image::load_from_memory(include_bytes!("fixtures/pear.png")).unwrap();
    let score = Evaluator::new(true)
        .evaluate_pair(&a, &b, Convention::Brush)
        .await
        .unwrap();
    assert!(
        (score.lpips.unwrap() - 0.6571019887924194).abs() < 1e-4,
        "{score:?}"
    );
}

#[tokio::test]
#[ignore = "integration: GPU"]
async fn float_parity_preserves_highlights_and_detects_alpha() {
    let evaluator = Evaluator::new(false);
    let a = [1.2, 1.2, 1.2, 0.5].repeat(16 * 16);
    let b = [1.1, 1.1, 1.1, 1.0].repeat(16 * 16);
    let score = evaluator.evaluate_renders(&a, &b, 16, 16).await.unwrap();
    assert!((score.rgb.psnr - 20.0).abs() < 1e-4);
    assert!((score.alpha_psnr - 6.020599913).abs() < 1e-4);
    assert!((score.white_psnr - 4.43697499).abs() < 1e-4);
    assert_eq!(score.minimum_psnr(), score.white_psnr);
}

#[tokio::test]
#[ignore = "integration: GPU and Lego PLY/cameras/GT assets (selected by pytest)"]
async fn lego_float_evaluation_matches_brush_eval_stats() {
    let ply = std::env::var("GSPLAT_TEST_PLY")
        .expect("asset GSPLAT_TEST_PLY is required; use tests-integration for skip handling");
    let cameras = std::env::var("GSPLAT_TEST_CAMERAS").expect("asset GSPLAT_TEST_CAMERAS required");
    let gt = std::env::var("GSPLAT_TEST_GT").expect("asset GSPLAT_TEST_GT required");
    let loaded =
        brush_serde::import::load_splat_from_ply(tokio::fs::File::open(ply).await.unwrap(), None)
            .await
            .unwrap();
    let frames = gsplat_cli::camera::load_frames(std::path::Path::new(&cameras), Some((256, 256)))
        .await
        .unwrap();
    let camera = frames[0].camera.brush_camera();
    let gt = image::open(gt)
        .unwrap()
        .resize_exact(256, 256, image::imageops::FilterType::Triangle);
    let device = burn::tensor::Device::default();
    let splats = loaded.data.into_splats(
        &device,
        loaded
            .meta
            .render_mode
            .unwrap_or(brush_render::gaussian_splats::SplatRenderMode::Default),
    );
    let reference = brush_train::eval::eval_stats(
        splats,
        &camera,
        gt.clone(),
        brush_render::AlphaMode::Transparent,
        &device,
    )
    .await
    .unwrap();
    // Score exactly the tensor returned by the reference, with no PNG boundary.
    let pixels = reference
        .rendered
        .into_data_async()
        .await
        .unwrap()
        .try_into_vec::<f32>()
        .unwrap();
    let rendered =
        DynamicImage::ImageRgb32F(image::Rgb32FImage::from_raw(256, 256, pixels).unwrap());
    let psnr = reference.psnr.into_scalar_async::<f32>().await.unwrap();
    let ssim = reference.ssim.into_scalar_async::<f32>().await.unwrap();
    let actual = Evaluator::new(false)
        .evaluate_pair(&rendered, &gt, Convention::Brush)
        .await
        .unwrap();
    assert!(
        (actual.psnr - f64::from(psnr)).abs() <= 1e-6,
        "{actual:?} vs {psnr}"
    );
    assert!(
        (actual.ssim - f64::from(ssim)).abs() <= 1e-7,
        "{actual:?} vs {ssim}"
    );
    println!(
        "Brush eval_stats equality: PSNR {} SSIM {}",
        actual.psnr, actual.ssim
    );
}

#[tokio::test]
#[ignore = "integration: GPU"]
async fn float_ssim_agrees_with_brush_on_byte_exact_inputs() {
    let a: Vec<f32> = (0..16 * 16)
        .flat_map(|i| [if i % 7 == 0 { 1.0 } else { 0.0 }, 0.0, 0.0, 1.0])
        .collect();
    let b: Vec<f32> = (0..16 * 16)
        .flat_map(|i| [if i % 9 == 0 { 1.0 } else { 0.0 }, 0.0, 0.0, 1.0])
        .collect();
    let evaluator = Evaluator::new(false);
    let float = evaluator.evaluate_renders(&a, &b, 16, 16).await.unwrap();
    let image =
        |p: Vec<f32>| DynamicImage::ImageRgba32F(image::Rgba32FImage::from_raw(16, 16, p).unwrap());
    let brush = evaluator
        .evaluate_pair(&image(a), &image(b), Convention::Brush)
        .await
        .unwrap();
    assert!((float.rgb.psnr - brush.psnr).abs() < 1e-4);
    assert!(
        (float.rgb.ssim - brush.ssim).abs() < 1e-5,
        "{} != {}",
        float.rgb.ssim,
        brush.ssim
    );
}
