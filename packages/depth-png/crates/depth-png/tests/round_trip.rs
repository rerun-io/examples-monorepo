//! Encode then decode returns the input exactly, and the bytes are a standard PNG.

use depth_png::{decode, encode_u16, encode_u8, DepthImage, DepthPngError, EncodeOptions};

/// A depth-like ramp with a hole pattern.
fn ramp(width: u32, height: u32) -> Vec<u16> {
    (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            if (x + y).is_multiple_of(7) {
                0
            } else {
                (300 + 37 * x + 101 * y) as u16
            }
        })
        .collect()
}

#[test]
fn round_trip_is_exact_at_odd_sizes_and_every_level() {
    for [width, height] in [[1, 1], [1, 9], [3, 5], [17, 9], [640, 480]] {
        let u16_depth = ramp(width, height);
        let u8_depth: Vec<u8> = u16_depth.iter().map(|&d| d as u8).collect();
        for level in 0..=9 {
            let decoded = decode(
                &encode_u16(&u16_depth, [width, height], &at_level(level)).unwrap(),
                0,
            )
            .unwrap();
            assert_eq!(
                decoded,
                DepthImage::U16 {
                    width,
                    height,
                    data: u16_depth.clone()
                },
                "u16 {width}x{height} level {level}"
            );
            let decoded = decode(
                &encode_u8(&u8_depth, [width, height], &at_level(level)).unwrap(),
                0,
            )
            .unwrap();
            assert_eq!(
                decoded,
                DepthImage::U8 {
                    width,
                    height,
                    data: u8_depth.clone()
                },
                "u8 {width}x{height} level {level}"
            );
        }
    }
}

#[test]
fn samples_are_stored_big_endian_behind_up_filter_bytes() {
    // 2x2 image: rows [0x0102, 0xA0B0] and [0x0103, 0xA0B0].
    let png_bytes = encode_u16(&[0x0102, 0xA0B0, 0x0103, 0xA0B0], [2, 2], &at_level(0)).unwrap();
    // Level 0 stores the filtered rows verbatim in one stored deflate block: find them in the IDAT payload.
    let filtered_rows = [
        2, 0x01, 0x02, 0xA0, 0xB0, // Up filter on the first row: the row above is zero
        2, 0x00, 0x01, 0x00, 0x00, // Up filter: big-endian differences to the row above
    ];
    assert!(
        png_bytes
            .windows(filtered_rows.len())
            .any(|window| window == filtered_rows),
        "{png_bytes:02x?}"
    );

    let mut reader = png::Decoder::new(std::io::Cursor::new(&png_bytes))
        .read_info()
        .unwrap();
    let mut raw = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut raw).unwrap();
    assert_eq!(raw, [0x01, 0x02, 0xA0, 0xB0, 0x01, 0x03, 0xA0, 0xB0]);
}

#[test]
fn every_row_uses_the_up_filter_at_every_level() {
    let (width, height) = (33, 21);
    for level in 0..=9 {
        let png_bytes =
            encode_u16(&ramp(width, height), [width, height], &at_level(level)).unwrap();
        let rows = miniz_oxide::inflate::decompress_to_vec_zlib(&idat_payload(&png_bytes)).unwrap();
        // Each row is its filter-type byte followed by the filtered samples.
        let filters: Vec<u8> = rows
            .chunks_exact(1 + 2 * width as usize)
            .map(|row| row[0])
            .collect();
        assert_eq!(filters, vec![2; height as usize], "level {level}");
    }
}

#[test]
fn bad_inputs_are_refused() {
    assert!(matches!(
        encode_u16(&[0; 4], [2, 2], &at_level(10)),
        Err(DepthPngError::InvalidLevel(10))
    ));
    assert!(matches!(
        encode_u8(&[0; 3], [2, 2], &at_level(4)),
        Err(DepthPngError::SampleCount {
            expected: 4,
            actual: 3,
            ..
        })
    ));
    assert!(matches!(
        decode(b"not a png", 0),
        Err(DepthPngError::Decoding(_))
    ));
}

/// The concatenated IDAT chunk data of a PNG.
fn idat_payload(png_bytes: &[u8]) -> Vec<u8> {
    let mut payload = Vec::new();
    let mut at = 8;
    while at < png_bytes.len() {
        let length = u32::from_be_bytes(png_bytes[at..at + 4].try_into().unwrap()) as usize;
        if &png_bytes[at + 4..at + 8] == b"IDAT" {
            payload.extend_from_slice(&png_bytes[at + 8..at + 8 + length]);
        }
        at += 12 + length;
    }
    payload
}

/// Default options at `level`.
fn at_level(level: u8) -> EncodeOptions {
    EncodeOptions {
        level,
        ..EncodeOptions::default()
    }
}

#[test]
fn banded_pngs_read_as_one_stream_in_the_png_crate() {
    // The png crate is the PNG decoder behind the image crate, which the Rerun viewer uses.
    let (width, height) = (300, 97);
    let depth = ramp(width, height);
    let options = EncodeOptions {
        band_rows: Some(10),
        threads: 4,
        ..EncodeOptions::default()
    };
    let png_bytes = encode_u16(&depth, [width, height], &options).unwrap();
    assert!(png_bytes.windows(4).any(|kind| kind == b"dpBN"));
    let mut reader = png::Decoder::new(std::io::Cursor::new(&png_bytes))
        .read_info()
        .unwrap();
    let mut raw = vec![0; reader.output_buffer_size().unwrap()];
    reader.next_frame(&mut raw).unwrap();
    let samples: Vec<u16> = raw
        .chunks_exact(2)
        .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
        .collect();
    assert_eq!(samples, depth);
}
