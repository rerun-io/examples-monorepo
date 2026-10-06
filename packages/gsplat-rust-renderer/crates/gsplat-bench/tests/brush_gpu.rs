//! Compatibility checks for Brush on the workspace's native wgpu version.
//! Run explicitly with `cargo test -p gsplat-bench --test brush_gpu -- --ignored`.

use brush_render::camera::Camera;
use brush_render::gaussian_splats::SplatRenderMode;
use brush_render::kernels::camera_model::CameraModel;
use brush_render::{TextureMode, render_splats};
use burn::tensor::{Device, Tensor, TensorData};
use glam::{Mat3, Mat4, Quat, Vec2, Vec3, uvec2};

#[tokio::test]
#[ignore = "integration: GPU and GSPLAT_TEST_PLY Lego asset"]
async fn brush_renders_lego() {
    let Ok(path) = std::env::var("GSPLAT_TEST_PLY") else {
        eprintln!("SKIP: GSPLAT_TEST_PLY must name the pretrained Lego PLY");
        return;
    };
    let file = tokio::fs::File::open(path).await.expect("Lego PLY asset");
    let loaded = brush_serde::import::load_splat_from_ply(file, None)
        .await
        .expect("Brush PLY loader");
    let device = Device::default();
    let splats = loaded.data.into_splats(
        &device,
        loaded.meta.render_mode.unwrap_or(SplatRenderMode::Default),
    );
    let position = Vec3::new(4.0, -4.0, 2.4);
    let view = Mat4::look_at_lh(position, Vec3::ZERO, -Vec3::Z);
    let camera = Camera::new(
        position,
        Quat::from_mat3(&Mat3::from_mat4(view)).conjugate(),
        0.6911112,
        0.6911112,
        Vec2::splat(0.5),
        CameraModel::Pinhole,
    );
    let (render, _) = render_splats(
        splats,
        &camera,
        uvec2(512, 512),
        Vec3::ZERO,
        None,
        TextureMode::Float,
    )
    .await;
    let pixels = render
        .into_data_async()
        .await
        .expect("render readback")
        .try_into_vec::<f32>()
        .expect("f32 render");
    assert!(pixels.iter().all(|x| x.is_finite()));
    let lit = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[..3].iter().any(|c| *c > 0.1))
        .count();
    assert!(lit > 1000, "only {lit} nonblack pixels");
    if let Ok(output) = std::env::var("GSPLAT_TEST_OUTPUT") {
        let image = image::Rgba32FImage::from_raw(512, 512, pixels).expect("RGBA dimensions");
        image::DynamicImage::ImageRgba32F(image)
            .to_rgba8()
            .save(output)
            .expect("save evidence");
    }
}

#[tokio::test]
#[ignore = "golden: GPU LPIPS apple/pear reference"]
async fn lpips_matches_apple_pear_reference() {
    let device = Device::default();
    let tensor = |bytes: &[u8]| {
        let img = image::load_from_memory(bytes)
            .expect("LPIPS fixture")
            .to_rgb32f();
        let (w, h) = img.dimensions();
        Tensor::<4>::from_data(
            TensorData::new(img.into_raw(), [1, h as usize, w as usize, 3]),
            &device,
        )
    };
    let model = lpips::load_vgg_lpips(&device);
    let score = model
        .lpips(
            tensor(include_bytes!("fixtures/apple.png")),
            tensor(include_bytes!("fixtures/pear.png")),
        )
        .into_scalar_async::<f32>()
        .await
        .expect("LPIPS scalar");
    println!("LPIPS apple/pear = {score}");
    assert!((f64::from(score) - 0.6571019887924194).abs() <= 1e-4);
}
