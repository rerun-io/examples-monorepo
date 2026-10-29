//! Held-out views rendered once after training and kept as 16-bit RGB PNGs for external scoring.
use anyhow::Context as _;
use brush_dataset::scene::SceneView;
use brush_render::gaussian_splats::Splats;
use std::path::Path;

/// Recorded in the summary so scores name the renderer that produced them.
pub const RENDERER: &str = "brush-render render_splats (wgpu), black background, float RGB clamped to [0,1] and rounded to 16 bits";

/// Render every view at its training resolution and write `<dir>/<image stem>.png`.
pub async fn write(splats: &Splats, views: &[SceneView], dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    for view in views {
        // Loading applies Brush's resolution cap, so renders match what training saw.
        let target = view.image.load().await?;
        let (width, height) = (target.width(), target.height());
        let (render, _) = Box::pin(brush_render::render_splats(
            splats.clone(),
            &view.camera,
            glam::uvec2(width, height),
            glam::Vec3::ZERO,
            None,
            brush_render::TextureMode::Float,
        ))
        .await;
        let pixels = render.into_data_async().await?.try_into_vec::<f32>()?;
        let name = view
            .image
            .path()
            .file_stem()
            .context("eval image has no file name")?;
        rgb16(width, height, pixels)?.save(dir.join(format!("{}.png", name.to_string_lossy())))?;
    }
    Ok(())
}

/// Drop alpha (the render is already composited on black), clamp to [0,1] and round to 16 bits.
fn rgb16(
    width: u32,
    height: u32,
    rgba: Vec<f32>,
) -> anyhow::Result<image::ImageBuffer<image::Rgb<u16>, Vec<u16>>> {
    let render =
        image::Rgba32FImage::from_raw(width, height, rgba).context("render size mismatch")?;
    Ok(image::DynamicImage::ImageRgba32F(render).to_rgb16())
}

#[cfg(test)]
mod tests {
    #[test]
    fn float_render_becomes_clamped_rounded_rgb16() {
        let image = super::rgb16(
            2,
            1,
            vec![
                0.5,
                -0.25,
                1.5,
                0.0,
                1.0 / 65535.0,
                0.4999 / 65535.0,
                1.0,
                1.0,
            ],
        )
        .unwrap();
        assert_eq!(image.into_raw(), vec![32768, 0, 65535, 1, 0, 65535]);
    }
}
