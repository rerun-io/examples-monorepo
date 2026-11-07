#![deny(missing_docs)]
//! Lossless PNG encoding and decoding of single-channel depth images.
//!
//! [`encode_u16`] and [`encode_u8`] write an 8- or 16-bit greyscale PNG with the `Up` row
//! filter: depth maps change little from one row to the next, so `Up` compresses as well as
//! adaptive filtering, costs nothing to choose, and is about three times faster to undo than
//! the `Paeth` rows adaptive filtering tends to pick. Compression is zlib (the pure-Rust
//! `zlib-rs` port), so a level means what it means to libpng, OpenCV or Pillow.
//!
//! [`decode`] reads any 8- or 16-bit greyscale PNG, whichever encoder wrote it.
//!
//! Images are row-major, `width * height` samples, in native byte order.

/// Errors returned by [`encode_u16`], [`encode_u8`] and [`decode`].
#[derive(Debug, thiserror::Error)]
pub enum DepthPngError {
    /// The compression level is outside 0-9.
    #[error("compression level must be 0-9; got {0}")]
    InvalidLevel(u8),

    /// The sample count does not match the image size.
    #[error("a {width}x{height} image has {expected} samples; got {actual}")]
    SampleCount {
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// `width * height`.
        expected: usize,
        /// Samples supplied.
        actual: usize,
    },

    /// The PNG is not 8- or 16-bit greyscale.
    #[error("not an 8- or 16-bit greyscale PNG: colour type {color:?}, bit depth {depth:?}")]
    NotGreyscale {
        /// Colour type in the PNG header.
        color: png::ColorType,
        /// Bit depth in the PNG header.
        depth: png::BitDepth,
    },

    /// The `png` crate failed to write the image.
    #[error(transparent)]
    Encoding(#[from] png::EncodingError),

    /// The `png` crate failed to read the image.
    #[error(transparent)]
    Decoding(#[from] png::DecodingError),
}

/// A decoded depth image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DepthImage {
    /// From an 8-bit greyscale PNG.
    U8 {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Row-major samples.
        data: Vec<u8>,
    },
    /// From a 16-bit greyscale PNG.
    U16 {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Row-major samples, native byte order.
        data: Vec<u16>,
    },
}

/// Encode a 16-bit depth image as a greyscale PNG.
///
/// `level` is the zlib level: 0 stores the rows uncompressed, 1 is fastest, 9 smallest.
/// Level 4 is a good default for depth: within a few percent of level 6 in size at about a
/// third of the time.
///
/// # Example
///
/// ```
/// let depth: Vec<u16> = vec![0, 450, 451, 0, 452, 453];
/// let png = depth_png::encode_u16(&depth, [3, 2], 4).unwrap();
/// assert_eq!(depth_png::decode(&png).unwrap(), depth_png::DepthImage::U16 { width: 3, height: 2, data: depth });
/// ```
pub fn encode_u16(
    depth: &[u16],
    [width, height]: [u32; 2],
    level: u8,
) -> Result<Vec<u8>, DepthPngError> {
    check_sample_count(depth.len(), width, height)?;
    // PNG stores 16-bit samples big-endian.
    let big_endian: Vec<u8> = depth
        .iter()
        .flat_map(|sample| sample.to_be_bytes())
        .collect();
    encode_rows(&big_endian, [width, height], png::BitDepth::Sixteen, level)
}

/// Encode an 8-bit depth image as a greyscale PNG; `level` as in [`encode_u16`].
pub fn encode_u8(
    depth: &[u8],
    [width, height]: [u32; 2],
    level: u8,
) -> Result<Vec<u8>, DepthPngError> {
    check_sample_count(depth.len(), width, height)?;
    encode_rows(depth, [width, height], png::BitDepth::Eight, level)
}

/// Decode an 8- or 16-bit greyscale PNG from any encoder.
pub fn decode(png_bytes: &[u8]) -> Result<DepthImage, DepthPngError> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info()?;
    let (color, depth) = reader.output_color_type();
    let png::Info { width, height, .. } = *reader.info();
    let samples = width as usize * height as usize;
    match (color, depth) {
        (png::ColorType::Grayscale, png::BitDepth::Eight) => {
            let mut data = vec![0u8; samples];
            reader.next_frame(&mut data)?;
            Ok(DepthImage::U8 {
                width,
                height,
                data,
            })
        }
        (png::ColorType::Grayscale, png::BitDepth::Sixteen) => {
            // Decode straight into the output buffer, then swap each big-endian sample in place.
            let mut data = vec![0u16; samples];
            reader.next_frame(bytemuck::cast_slice_mut(&mut data))?;
            for sample in &mut data {
                *sample = u16::from_be(*sample);
            }
            Ok(DepthImage::U16 {
                width,
                height,
                data,
            })
        }
        _ => Err(DepthPngError::NotGreyscale { color, depth }),
    }
}

fn check_sample_count(actual: usize, width: u32, height: u32) -> Result<(), DepthPngError> {
    let expected = width as usize * height as usize;
    if actual == expected {
        Ok(())
    } else {
        Err(DepthPngError::SampleCount {
            width,
            height,
            expected,
            actual,
        })
    }
}

fn encode_rows(
    rows: &[u8],
    [width, height]: [u32; 2],
    bit_depth: png::BitDepth,
    level: u8,
) -> Result<Vec<u8>, DepthPngError> {
    if level > 9 {
        return Err(DepthPngError::InvalidLevel(level));
    }
    // zlib level 0 writes stored blocks; unlike `DeflateCompression::NoCompression` it keeps the
    // Up filter, so every level meets the same contract.
    let compression = png::DeflateCompression::Level(level);
    let mut png_bytes = Vec::with_capacity(rows.len() / 2);
    let mut encoder = png::Encoder::new(&mut png_bytes, width, height);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(bit_depth);
    encoder.set_deflate_compression(compression);
    encoder.set_filter(png::Filter::Up);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rows)?;
    writer.finish()?;
    Ok(png_bytes)
}
