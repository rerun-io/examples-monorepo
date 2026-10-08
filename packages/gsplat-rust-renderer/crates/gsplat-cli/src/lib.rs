//! Camera and image boundaries shared by rendering, benchmarks, and evaluation.
pub mod camera;
mod camera_files;
pub mod engines;
pub mod metrics;
mod provenance;
mod renderer;
pub mod settings;
pub mod statistics;
pub use metrics::{
    Convention, Evaluation, Evaluator, Metrics, RenderMetrics, ViewMetrics, evaluate_directories,
    mean, pair_directories,
};
pub use provenance::Provenance;
pub use renderer::{Output, PlyScene, Renderer, raw_splats};

pub use anyhow::{Error, Result};

#[derive(Debug, thiserror::Error)]
#[error("unsupported: {0}")]
pub struct Unsupported(pub String);

/// Preserve the scored float values; conversion to PNG belongs only at export.
pub fn parity_image(pixels: Vec<f32>, width: u32, height: u32) -> Result<image::DynamicImage> {
    if pixels.len() != width as usize * height as usize * 4 || !pixels.iter().all(|v| v.is_finite())
    {
        anyhow::bail!("invalid finite RGBA dimensions");
    }
    Ok(image::DynamicImage::ImageRgba32F(
        image::Rgba32FImage::from_raw(width, height, pixels).expect("validated dimensions"),
    ))
}
#[cfg(test)]
mod tests {
    #[test]
    fn evidence_preserves_scored_hdr_and_alpha() {
        let image = super::parity_image(vec![1.17, 0.5, -0.2, 0.1], 1, 1).unwrap();
        assert_eq!(image.to_rgba32f().as_raw(), &[1.17, 0.5, -0.2, 0.1]);
        assert!(super::parity_image(vec![f32::NAN; 4], 1, 1).is_err());
    }
}

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod common;
