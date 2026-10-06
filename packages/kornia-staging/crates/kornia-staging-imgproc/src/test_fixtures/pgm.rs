//! Parser for the committed binary PGM fixtures.
use std::path::Path;

/// An 8-bit PGM fixture.
pub struct Pgm {
    /// Visible columns.
    pub width: usize,
    /// Visible rows.
    pub height: usize,
    /// Row-major raster bytes.
    pub pixels: Vec<u8>,
}
/// Read the controlled `P5` fixture format (three header fields, no comments).
/// Panics on missing or malformed test data.
pub fn read_pgm(path: &Path) -> Pgm {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(&bytes[..2], b"P5");
    let mut cursor = 2;
    let mut fields = Vec::new();
    while fields.len() < 3 {
        while bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let start = cursor;
        while !bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        fields.push(
            std::str::from_utf8(&bytes[start..cursor])
                .unwrap()
                .parse::<usize>()
                .unwrap(),
        );
    }
    assert_eq!(fields[2], 255);
    let pixels = bytes[cursor + 1..].to_vec();
    assert_eq!(pixels.len(), fields[0] * fields[1]);
    Pgm {
        width: fields[0],
        height: fields[1],
        pixels,
    }
}
