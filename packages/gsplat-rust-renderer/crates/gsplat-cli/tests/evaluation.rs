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
