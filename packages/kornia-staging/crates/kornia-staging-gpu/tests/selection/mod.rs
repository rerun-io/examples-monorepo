use kornia_image::{Image, ImageError, ImageSize};
pub use kornia_staging_imgproc::test_fixtures::cornered_image;
pub use kornia_staging_imgproc::test_fixtures::zeros;
pub fn from_u8_strided(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Image<u16, 1>, ImageError> {
    let mut image = Image::from_size_val(ImageSize { width, height }, 0u16)?;
    kornia_staging_imgproc::color::widen_u8_shift8_strided(bytes, stride, &mut image)?;
    Ok(image)
}
pub fn mio10_frame(frame: usize, camera: usize) -> Image<u16, 1> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../fixtures/frames/frame_{frame:03}_cam{camera}.pgm"
    ));
    let pgm = kornia_staging_imgproc::test_fixtures::read_pgm(&path);
    from_u8_strided(&pgm.pixels, pgm.width, pgm.height, pgm.width).unwrap()
}
