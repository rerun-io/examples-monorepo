//! Hand-built 16-bit PNGs that probe the edges of the decoder's Up-filter fast path.

use depth_png::{decode, DepthImage};

const WIDTH: u32 = 5;
const HEIGHT: u32 = 4;

fn depth() -> Vec<u16> {
    (0..WIDTH * HEIGHT)
        .map(|i| (1000 + 257 * i) as u16)
        .collect()
}

/// Filtered rows of `depth()` with the given filter type per row (only None=0, Sub=1 and Up=2).
fn filtered_rows(filters: [u8; HEIGHT as usize]) -> Vec<u8> {
    let stride = 2 * WIDTH as usize;
    let raw: Vec<u8> = depth().iter().flat_map(|d| d.to_be_bytes()).collect();
    let mut out = Vec::new();
    for (row, &filter) in filters.iter().enumerate() {
        let current = &raw[row * stride..(row + 1) * stride];
        out.push(filter);
        for i in 0..stride {
            let predictor = match filter {
                0 => 0,
                1 if i >= 2 => current[i - 2],
                1 => 0,
                2 if row > 0 => raw[(row - 1) * stride + i],
                2 => 0,
                _ => unreachable!(),
            };
            out.push(current[i].wrapping_sub(predictor));
        }
    }
    out
}

/// A PNG around a zlib stream, split into IDAT chunks of `idat_size` bytes.
fn png(zlib: &[u8], idat_size: usize) -> Vec<u8> {
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let crc = crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    }
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&WIDTH.to_be_bytes());
    ihdr.extend_from_slice(&HEIGHT.to_be_bytes());
    ihdr.extend_from_slice(&[16, 0, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    for part in zlib.chunks(idat_size) {
        chunk(&mut out, b"IDAT", part);
    }
    chunk(&mut out, b"IEND", &[]);
    out
}

/// CRC-32 as PNG defines it (bitwise, so the test does not share the decoder's implementation).
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn expected() -> DepthImage {
    DepthImage::U16 {
        width: WIDTH,
        height: HEIGHT,
        data: depth(),
    }
}

#[test]
fn up_rows_split_over_odd_idat_boundaries_decode() {
    let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&filtered_rows([2, 2, 2, 2]), 6);
    for idat_size in [1, 3, 7, zlib.len()] {
        assert_eq!(
            decode(&png(&zlib, idat_size), 0).unwrap(),
            expected(),
            "IDAT size {idat_size}"
        );
    }
}

#[test]
fn a_non_up_row_after_an_up_first_row_still_decodes() {
    for filters in [[2, 2, 1, 2], [2, 0, 0, 0], [2, 2, 2, 1]] {
        let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&filtered_rows(filters), 6);
        assert_eq!(
            decode(&png(&zlib, 64), 0).unwrap(),
            expected(),
            "filters {filters:?}"
        );
    }
}

#[test]
fn corrupt_and_truncated_pngs_are_errors() {
    let zlib = miniz_oxide::deflate::compress_to_vec_zlib(&filtered_rows([2, 2, 2, 2]), 6);
    let good = png(&zlib, zlib.len());

    let mut flipped = good.clone();
    let idat_data = good.windows(4).position(|w| w == b"IDAT").unwrap() + 4;
    flipped[idat_data + 3] ^= 0x40; // CRC no longer matches
    assert!(decode(&flipped, 0).is_err());

    let truncated = png(&zlib[..zlib.len() - 6], 64); // valid chunks, zlib stream cut short
    assert!(decode(&truncated, 0).is_err());
}
