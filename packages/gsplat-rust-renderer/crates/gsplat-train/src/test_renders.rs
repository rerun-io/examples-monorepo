//! Held-out views rendered once after training and kept as 16-bit RGB PNGs for external scoring.
use anyhow::Context as _;
use brush_dataset::scene::SceneView;
use brush_render::gaussian_splats::Splats;
use std::path::Path;

/// Recorded in the summary so scores name the renderer that produced them.
pub const RENDERER: &str = "brush-render render_splats (wgpu), black background, float RGB clamped to [0,1] and rounded to 16 bits";

/// Render every view at its training resolution and write `<dir>/<image stem>.png`.
pub async fn write(splats: &Splats, views: &[SceneView], dir: &Path) -> anyhow::Result<()> {
    let names = output_names(views.iter().map(|view| view.image.path()))?;
    std::fs::create_dir_all(dir)?;
    for (view, name) in views.iter().zip(names) {
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
        rgb16(width, height, pixels)?.save(dir.join(name))?;
    }
    Ok(())
}

/// One `<image stem>.png` per view; two views with the same stem would overwrite each other.
fn output_names<'a>(paths: impl IntoIterator<Item = &'a Path>) -> anyhow::Result<Vec<String>> {
    let mut seen = std::collections::HashMap::new();
    paths
        .into_iter()
        .map(|path| {
            let stem = path.file_stem().context("eval image has no file name")?;
            let name = format!("{}.png", stem.to_string_lossy());
            if let Some(first) = seen.insert(name.clone(), path) {
                anyhow::bail!(
                    "eval images {} and {} would both write {name}",
                    first.display(),
                    path.display()
                );
            }
            Ok(name)
        })
        .collect()
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

    #[test]
    fn eval_views_sharing_a_file_stem_are_rejected_before_rendering() {
        use std::path::Path;
        let names =
            super::output_names([Path::new("left/0001.jpg"), Path::new("left/0002.jpg")]).unwrap();
        assert_eq!(names, ["0001.png", "0002.png"]);
        let error = super::output_names([Path::new("left/0001.jpg"), Path::new("right/0001.jpg")])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("left/0001.jpg") && error.contains("right/0001.jpg"),
            "{error}"
        );
    }
}
