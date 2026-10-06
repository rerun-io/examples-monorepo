//! Reproducible camera, renderer, and measurement boundaries for splat benchmarks.
pub mod camera;
pub mod renderers;
pub mod settings;
pub mod statistics;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Render(#[from] gsplat_render::Error),
    #[error("{0}")]
    Invalid(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("GPU: {0}")]
    Gpu(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
}
pub type Result<T> = std::result::Result<T, Error>;

fn gpu(e: impl std::fmt::Display) -> Error {
    Error::Gpu(e.to_string())
}
fn wait(device: &wgpu::Device) -> Result<()> {
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map(|_| ())
        .map_err(gpu)
}
/// Preserve the scored float values; conversion to PNG belongs only at export.
pub fn parity_image(pixels: Vec<f32>, width: u32, height: u32) -> Result<image::DynamicImage> {
    if pixels.len() != width as usize * height as usize * 4 || !pixels.iter().all(|v| v.is_finite())
    {
        return Err(Error::Invalid("invalid finite RGBA dimensions".into()));
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
