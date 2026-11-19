//! Builders of synthetic metadata records and trailers for the reader's tests.
use super::trailer::MAGIC;

/// The real X4 Air strings and window crop, from the clip VID_20261116_125200_00_003.insv.
mod x4air {
    #![allow(dead_code)]
    use kornia_staging_3d::camera::formats::insta360::WindowCrop;
    include!("../../tests/common/x4air_lens.rs");
    pub(crate) const V3_STRING: &str = V3;
    pub(crate) const V6_STRING: &str = V6;
    pub(crate) const WINDOW_CROP: WindowCrop = CROP;
}
pub(super) use x4air::{V3_STRING as V3, V6_STRING as V6, WINDOW_CROP as CROP};

pub(super) fn key(field: u64, wire: u64, out: &mut Vec<u8>) {
    varint((field << 3) | wire, out);
}

pub(super) fn varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

pub(super) fn bytes_field(field: u64, b: &[u8], out: &mut Vec<u8>) {
    key(field, 2, out);
    varint(b.len() as u64, out);
    out.extend_from_slice(b);
}

pub(super) fn varint_field(field: u64, v: u64, out: &mut Vec<u8>) {
    key(field, 0, out);
    varint(v, out);
}

pub(super) fn metadata(raw: bool) -> Vec<u8> {
    metadata_with_lens(raw, 54, V3)
}

/// [`metadata`] with one lens string in `field` (54 V3, 111 V6) instead of the V3 string.
pub(super) fn metadata_with_lens(raw: bool, field: u64, lens: &str) -> Vec<u8> {
    let mut m = Vec::new();
    bytes_field(2, b"Insta360 X4 Air", &mut m);
    bytes_field(3, b"v1.2.7_build1", &mut m);
    let mut res = Vec::new();
    varint_field(1, 1920, &mut res);
    varint_field(2, 1920, &mut res);
    bytes_field(19, &res, &mut m);
    varint_field(20, 30, &mut m);
    varint_field(24, 1_000_200, &mut m);
    key(25, 1, &mut m);
    m.extend_from_slice(&15.882978f64.to_le_bytes());
    let mut crop = Vec::new();
    for (i, v) in [7680u64, 7680, 7424, 7424, 0, 0].into_iter().enumerate() {
        varint_field(i as u64 + 1, v, &mut crop);
    }
    bytes_field(27, &crop, &mut m);
    key(28, 1, &mut m);
    m.extend_from_slice(&1.6f64.to_le_bytes());
    bytes_field(field, lens.as_bytes(), &mut m);
    varint_field(62, u64::from(raw), &mut m);
    let mut ranges = Vec::new();
    varint_field(1, 32, &mut ranges);
    varint_field(2, 2000, &mut ranges);
    bytes_field(65, &ranges, &mut m);
    varint_field(80, 1, &mut m);
    varint_field(131, 4, &mut m);
    m
}

/// A file of `prefix` junk bytes then a trailer holding the given records in 256-byte slots.
pub(super) fn trailer_file(records: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut body = vec![0u8; 64];
    let mut index = vec![0u8; 310];
    for (id, record) in records {
        let offset = body.len() as u32;
        body.extend_from_slice(record);
        body.resize(body.len().div_ceil(256) * 256, 0xee);
        let slot = &mut index[10 * usize::from(*id)..10 * usize::from(*id) + 10];
        slot[0] = *id;
        slot[2..6].copy_from_slice(&(record.len() as u32).to_le_bytes());
        slot[6..10].copy_from_slice(&offset.to_le_bytes());
    }
    body.extend_from_slice(&index);
    body.extend_from_slice(&[0, 0]);
    body.extend_from_slice(&310u32.to_le_bytes());
    close_trailer(body)
}

/// The same records as [`trailer_file`] in the layout of the bodies before the X4 (ONE X2, ONE R/RS, X3): no index,
/// each record followed by its descriptor `[format][id][u32 size]`, the last one right before the footer.
pub(super) fn chained_trailer_file(records: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut body = vec![0u8; 64];
    for (id, record) in records {
        body.extend_from_slice(record);
        body.extend_from_slice(&[u8::from(*id == 1), *id]);
        body.extend_from_slice(&(record.len() as u32).to_le_bytes());
    }
    close_trailer(body)
}

/// `1000` junk bytes, then `body` closed by the trailer footer (size, version 3, magic).
fn close_trailer(mut body: Vec<u8>) -> Vec<u8> {
    let size = (body.len() + 72) as u32;
    body.extend_from_slice(&[0u8; 32]);
    body.extend_from_slice(&size.to_le_bytes());
    body.extend_from_slice(&3u32.to_le_bytes());
    body.extend_from_slice(MAGIC);
    let mut file = vec![0x55u8; 1000];
    file.extend(body);
    file
}

pub(super) fn exposures(times: &[u64]) -> Vec<u8> {
    times
        .iter()
        .flat_map(|&t| {
            let mut item = t.to_le_bytes().to_vec();
            item.extend(((t % 7) as f64 / 100.0).to_le_bytes());
            item
        })
        .collect()
}
