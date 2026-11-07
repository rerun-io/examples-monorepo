//! PNGs written by OpenCV and Pillow (scripts/make_fixtures.py) decode to the pixels they were written from.

use depth_png::{decode, DepthImage, DepthPngError};

const WIDTH: u32 = 97;
const HEIGHT: u32 = 61;

/// The pixel formula of scripts/make_fixtures.py: a 32-bit hash of the position, cut to `bits`.
fn sample(x: u32, y: u32, bits: u32) -> u32 {
    if (x + 2 * y).is_multiple_of(11) {
        return 0;
    }
    let mut h = x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5BD1_E995);
    h ^= h >> 15;
    h & ((1 << bits) - 1)
}

fn expected_u16() -> DepthImage {
    let data = (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| sample(x, y, 16) as u16))
        .collect();
    DepthImage::U16 {
        width: WIDTH,
        height: HEIGHT,
        data,
    }
}

fn expected_u8() -> DepthImage {
    let data = (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| sample(x, y, 8) as u8))
        .collect();
    DepthImage::U8 {
        width: WIDTH,
        height: HEIGHT,
        data,
    }
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

#[test]
fn opencv_16_bit_pngs_in_several_idat_chunks_decode_exactly() {
    for name in ["opencv_u16_level4_up.png", "opencv_u16_default.png"] {
        assert_eq!(decode(&fixture(name), 0).unwrap(), expected_u16(), "{name}");
    }
}

#[test]
fn pillow_pngs_with_mixed_row_filters_decode_exactly() {
    assert_eq!(
        decode(&fixture("pillow_u16_level6.png"), 0).unwrap(),
        expected_u16()
    );
    assert_eq!(
        decode(&fixture("pillow_u8_level6.png"), 0).unwrap(),
        expected_u8()
    );
}

#[test]
fn opencv_8_bit_png_decodes_exactly() {
    assert_eq!(
        decode(&fixture("opencv_u8_default.png"), 0).unwrap(),
        expected_u8()
    );
}

#[test]
fn colour_pngs_are_refused() {
    assert!(matches!(
        decode(&fixture("pillow_rgb8.png"), 0),
        Err(DepthPngError::NotGreyscale {
            color: png::ColorType::Rgb,
            depth: png::BitDepth::Eight
        })
    ));
}
