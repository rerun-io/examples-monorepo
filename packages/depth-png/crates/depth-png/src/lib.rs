#![deny(missing_docs)]
//! Lossless PNG encoding and decoding of single-channel depth images, in parallel row bands.
//!
//! The encoder writes an 8- or 16-bit greyscale PNG with the `Up` row filter on every row: depth
//! maps change little from one row to the next, so `Up` compresses as well as adaptive filtering,
//! costs nothing to choose, and is cheap to undo. The image is cut into bands of rows, each
//! compressed on its own thread into one segment of the single zlib stream; every segment but the
//! last ends with a full flush (byte-aligned, no back-references across the cut). The PNG is
//! standard: any reader decodes it as one stream. A private ancillary chunk, `dpBN` (before
//! `IDAT`), records where each band's segment starts, so this decoder inflates the bands in
//! parallel too; readers that do not know the chunk skip it.
//!
//! [`decode`] reads any 8- or 16-bit greyscale PNG. PNGs in the encoder's layout (greyscale, not
//! interlaced, `Up` on every row, with or without `dpBN`; OpenCV writes this with
//! `IMWRITE_PNG_FILTER_UP`) are inflated with zlib-rs and unfiltered row by row as they inflate.
//! Any other PNG goes through the `png` crate.
//!
//! Images are row-major, `width * height` samples, in native byte order.

use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::sync::Mutex;

/// Errors returned by the encoders and decoders.
#[derive(Debug, thiserror::Error)]
pub enum DepthPngError {
    /// The compression level is outside 0-9.
    #[error("compression level must be 0-9; got {0}")]
    InvalidLevel(u8),

    /// The image has no pixels.
    #[error("a depth image needs at least one pixel; got {width}x{height}")]
    Empty {
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },

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

    /// The output buffer of a `decode_into_*` call does not fit the PNG.
    #[error(
        "the PNG is a {width}x{height} {bits}-bit image; the output holds {capacity} {out_bits}-bit samples"
    )]
    OutputMismatch {
        /// PNG width in pixels.
        width: u32,
        /// PNG height in pixels.
        height: u32,
        /// PNG bit depth.
        bits: u8,
        /// Samples in the output buffer.
        capacity: usize,
        /// Bits per sample of the output buffer.
        out_bits: u8,
    },

    /// zlib-rs failed to compress.
    #[error("compression failed: {0}")]
    Compression(String),

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

/// How [`encode_u16`] and [`encode_u8`] compress.
#[derive(Clone, Copy, Debug)]
pub struct EncodeOptions {
    /// zlib level: 0 stores the rows uncompressed, 1 is fastest, 9 smallest.
    pub level: u8,
    /// Worker threads; 0 picks the available parallelism (never more than the bands).
    pub threads: usize,
    /// Rows per band; `None` cuts bands of at most [`BAND_BYTES`] of filtered rows, equal in height.
    pub band_rows: Option<u32>,
}

impl Default for EncodeOptions {
    /// Level 4: within a few percent of level 6 in size on depth maps at about a third of the time.
    fn default() -> Self {
        Self {
            level: 4,
            threads: 0,
            band_rows: None,
        }
    }
}

/// Filtered bytes per band when [`EncodeOptions::band_rows`] is `None`: a 256x192 or 512x416
/// 16-bit depth map is one band, a 1920x1440 one twelve bands of 120 rows.
pub const BAND_BYTES: usize = 480 * 1024;

/// Encode a 16-bit depth image as a greyscale PNG.
///
/// # Example
///
/// ```
/// use depth_png::{decode, encode_u16, DepthImage, EncodeOptions};
///
/// let depth: Vec<u16> = vec![0, 450, 451, 0, 452, 453];
/// let png = encode_u16(&depth, [3, 2], &EncodeOptions::default()).unwrap();
/// assert_eq!(decode(&png, 0).unwrap(), DepthImage::U16 { width: 3, height: 2, data: depth });
/// ```
pub fn encode_u16(
    depth: &[u16],
    size: [u32; 2],
    options: &EncodeOptions,
) -> Result<Vec<u8>, DepthPngError> {
    encode(Samples::U16(depth), size, options)
}

/// Encode an 8-bit depth image as a greyscale PNG.
pub fn encode_u8(
    depth: &[u8],
    size: [u32; 2],
    options: &EncodeOptions,
) -> Result<Vec<u8>, DepthPngError> {
    encode(Samples::U8(depth), size, options)
}

/// Decode an 8- or 16-bit greyscale PNG from any encoder; `threads` as in [`EncodeOptions`].
pub fn decode(png_bytes: &[u8], threads: usize) -> Result<DepthImage, DepthPngError> {
    let (width, height, sixteen_bit) = header(png_bytes)?;
    let samples = width as usize * height as usize;
    if sixteen_bit {
        let mut data = vec![0u16; samples];
        decode_into(png_bytes, Output::U16(&mut data), threads)?;
        Ok(DepthImage::U16 {
            width,
            height,
            data,
        })
    } else {
        let mut data = vec![0u8; samples];
        decode_into(png_bytes, Output::U8(&mut data), threads)?;
        Ok(DepthImage::U8 {
            width,
            height,
            data,
        })
    }
}

/// Decode a 16-bit greyscale PNG into `out`, which holds exactly `width * height` samples.
/// Returns `[width, height]`.
pub fn decode_into_u16(
    png_bytes: &[u8],
    out: &mut [u16],
    threads: usize,
) -> Result<[u32; 2], DepthPngError> {
    decode_into(png_bytes, Output::U16(out), threads)
}

/// Decode an 8-bit greyscale PNG into `out`, which holds exactly `width * height` samples.
/// Returns `[width, height]`.
pub fn decode_into_u8(
    png_bytes: &[u8],
    out: &mut [u8],
    threads: usize,
) -> Result<[u32; 2], DepthPngError> {
    decode_into(png_bytes, Output::U8(out), threads)
}

/// `(width, height, sixteen_bit)` of an 8- or 16-bit greyscale PNG, from its header alone.
pub fn header(png_bytes: &[u8]) -> Result<(u32, u32, bool), DepthPngError> {
    let reader = png::Decoder::new(std::io::Cursor::new(png_bytes)).read_info()?;
    let info = reader.info();
    match (info.color_type, info.bit_depth) {
        (png::ColorType::Grayscale, png::BitDepth::Eight) => Ok((info.width, info.height, false)),
        (png::ColorType::Grayscale, png::BitDepth::Sixteen) => Ok((info.width, info.height, true)),
        (color, depth) => Err(DepthPngError::NotGreyscale { color, depth }),
    }
}

#[derive(Clone, Copy)]
enum Samples<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

impl Samples<'_> {
    fn len(self) -> usize {
        match self {
            Samples::U8(samples) => samples.len(),
            Samples::U16(samples) => samples.len(),
        }
    }

    fn bytes_per_sample(self) -> usize {
        match self {
            Samples::U8(_) => 1,
            Samples::U16(_) => 2,
        }
    }

    /// Appends row `row` Up-filtered against the row above (zero above row 0): its filter byte,
    /// then the big-endian byte differences.
    fn filter_row(self, row: usize, width: usize, out: &mut Vec<u8>) {
        out.push(2);
        match self {
            Samples::U8(samples) => {
                let current = &samples[row * width..(row + 1) * width];
                if row == 0 {
                    out.extend_from_slice(current);
                } else {
                    let above = &samples[(row - 1) * width..row * width];
                    out.extend(current.iter().zip(above).map(|(&c, &a)| c.wrapping_sub(a)));
                }
            }
            Samples::U16(samples) => {
                let current = &samples[row * width..(row + 1) * width];
                if row == 0 {
                    out.extend(current.iter().flat_map(|sample| sample.to_be_bytes()));
                } else {
                    // PNG filters bytes, not samples: the high and low bytes are subtracted apart.
                    let above = &samples[(row - 1) * width..row * width];
                    out.extend(current.iter().zip(above).flat_map(|(&c, &a)| {
                        let [ch, cl] = c.to_be_bytes();
                        let [ah, al] = a.to_be_bytes();
                        [ch.wrapping_sub(ah), cl.wrapping_sub(al)]
                    }));
                }
            }
        }
    }
}

enum Output<'a> {
    U8(&'a mut [u8]),
    U16(&'a mut [u16]),
}

fn worker_count(threads: usize, bands: usize) -> usize {
    let available = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let wanted = if threads == 0 { available } else { threads };
    wanted.clamp(1, bands.max(1))
}

/// Runs `work(band)` for every band on `workers` scoped threads (inline for one) and returns the
/// results in band order.
fn for_each_band<T: Send>(
    bands: usize,
    workers: usize,
    work: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    if workers <= 1 {
        return (0..bands).map(work).collect();
    }
    let work = &work;
    let mut results: Vec<(usize, T)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (1..workers)
            .map(|worker| {
                scope.spawn(move || {
                    (worker..bands)
                        .step_by(workers)
                        .map(|band| (band, work(band)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut results: Vec<(usize, T)> = (0..bands)
            .step_by(workers)
            .map(|band| (band, work(band)))
            .collect();
        for handle in handles {
            results.extend(handle.join().expect("a depth-png worker panicked"));
        }
        results
    });
    results.sort_unstable_by_key(|(band, _)| *band);
    results.into_iter().map(|(_, result)| result).collect()
}

const BANDS_CHUNK: &[u8; 4] = b"dpBN";
/// Rows inflated per call before they are unfiltered: fewer calls than row by row, while the
/// scratch (122 KiB at 1920x1440x16) stays in cache; 1.25x faster than row by row there.
const INFLATE_ROWS: usize = 32;
const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

fn encode(
    samples: Samples<'_>,
    [width, height]: [u32; 2],
    options: &EncodeOptions,
) -> Result<Vec<u8>, DepthPngError> {
    if options.level > 9 {
        return Err(DepthPngError::InvalidLevel(options.level));
    }
    if width == 0 || height == 0 {
        return Err(DepthPngError::Empty { width, height });
    }
    let expected = width as usize * height as usize;
    if samples.len() != expected {
        return Err(DepthPngError::SampleCount {
            width,
            height,
            expected,
            actual: samples.len(),
        });
    }
    let (w, h) = (width as usize, height as usize);
    let row_len = 1 + w * samples.bytes_per_sample();
    let band_rows = match options.band_rows {
        Some(rows) => (rows.max(1) as usize).min(h),
        None => h.div_ceil((h * row_len).div_ceil(BAND_BYTES)),
    };
    let bands = h.div_ceil(band_rows);
    let level = flate2::Compression::new(u32::from(options.level));

    // Each band: its Up-filtered rows as raw deflate ending byte-aligned (a full flush, or the
    // final block for the last band), and the Adler-32 and length of those filtered bytes.
    let segments = for_each_band(bands, worker_count(options.threads, bands), |band| {
        let rows = band * band_rows..((band + 1) * band_rows).min(h);
        let mut filtered = Vec::with_capacity(rows.len() * row_len);
        for row in rows {
            samples.filter_row(row, w, &mut filtered);
        }
        let flush = if band + 1 == bands {
            flate2::FlushCompress::Finish
        } else {
            flate2::FlushCompress::Full
        };
        let adler = zlib_rs::adler32::adler32(1, &filtered);
        deflate_raw(&filtered, level, flush).map(|deflated| (deflated, adler, filtered.len()))
    })
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;

    let mut zlib = Vec::with_capacity(6 + segments.iter().map(|s| s.0.len()).sum::<usize>());
    zlib.extend_from_slice(&zlib_header(options.level));
    let mut table = Vec::with_capacity(4 + 4 * bands);
    table.extend_from_slice(&(band_rows as u32).to_be_bytes());
    let mut adler = 1;
    for (deflated, band_adler, filtered_len) in &segments {
        table.extend_from_slice(&(zlib.len() as u32).to_be_bytes());
        zlib.extend_from_slice(deflated);
        adler = zlib_rs::adler32::adler32_combine(adler, *band_adler, *filtered_len as u64);
    }
    zlib.extend_from_slice(&adler.to_be_bytes());

    let mut png_bytes = Vec::with_capacity(zlib.len() + table.len() + 64);
    png_bytes.extend_from_slice(PNG_SIGNATURE);
    let mut ihdr = [0u8; 13]; // colour type, compression, filter method and interlace all 0
    ihdr[..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8 * samples.bytes_per_sample() as u8;
    write_chunk(&mut png_bytes, b"IHDR", &ihdr);
    if bands > 1 {
        // u32 rows per band, then each band's offset into the zlib stream (the IDAT data), big-endian.
        write_chunk(&mut png_bytes, BANDS_CHUNK, &table);
    }
    write_chunk(&mut png_bytes, b"IDAT", &zlib);
    write_chunk(&mut png_bytes, b"IEND", &[]);
    Ok(png_bytes)
}

fn deflate_raw(
    data: &[u8],
    level: flate2::Compression,
    flush: flate2::FlushCompress,
) -> Result<Vec<u8>, DepthPngError> {
    let mut compressor = flate2::Compress::new(level, false);
    let mut out = Vec::with_capacity(data.len() / 2 + 1024);
    loop {
        if out.len() == out.capacity() {
            out.reserve(out.capacity());
        }
        let consumed = compressor.total_in() as usize;
        let status = compressor
            .compress_vec(&data[consumed..], &mut out, flush)
            .map_err(|error| DepthPngError::Compression(error.to_string()))?;
        let all_in = compressor.total_in() as usize == data.len();
        let flushed = match flush {
            flate2::FlushCompress::Finish => status == flate2::Status::StreamEnd,
            // A flush is complete once a call returns with room to spare in the output.
            _ => out.len() < out.capacity(),
        };
        if all_in && flushed {
            return Ok(out);
        }
    }
}

/// The 2-byte zlib header: deflate, 32 KiB window, no dictionary, the level's FLEVEL hint.
fn zlib_header(level: u8) -> [u8; 2] {
    let flevel: u16 = match level {
        0 | 1 => 0,
        2..=5 => 1,
        6 => 2,
        _ => 3,
    };
    let cmf_flg = 0x7800 | (flevel << 6);
    (cmf_flg + 31 - cmf_flg % 31).to_be_bytes()
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32fast::hash(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn decode_into(
    png_bytes: &[u8],
    mut out: Output<'_>,
    threads: usize,
) -> Result<[u32; 2], DepthPngError> {
    if let Some(layout) = UpLayout::parse(png_bytes) {
        let bytes: Option<&mut [u8]> = match &mut out {
            Output::U16(samples) if layout.sixteen_bit && samples.len() == layout.samples() => {
                Some(bytemuck::cast_slice_mut(samples))
            }
            Output::U8(samples) if !layout.sixteen_bit && samples.len() == layout.samples() => {
                Some(&mut **samples)
            }
            _ => None,
        };
        // A stream that fails the fast path (bad data, a row that is not Up) gets the png crate's verdict.
        if bytes.is_some_and(|bytes| layout.decode(bytes, threads).is_some()) {
            return Ok([layout.width, layout.height]);
        }
    }
    decode_any(png_bytes, out)
}

/// Any 8- or 16-bit greyscale PNG, through the `png` crate, into `out`.
fn decode_any(png_bytes: &[u8], out: Output<'_>) -> Result<[u32; 2], DepthPngError> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info()?;
    let (color, depth) = reader.output_color_type();
    let png::Info { width, height, .. } = *reader.info();
    let samples = width as usize * height as usize;
    let bits = match (color, depth) {
        (png::ColorType::Grayscale, png::BitDepth::Eight) => 8,
        (png::ColorType::Grayscale, png::BitDepth::Sixteen) => 16,
        _ => return Err(DepthPngError::NotGreyscale { color, depth }),
    };
    match out {
        Output::U8(out) if bits == 8 && out.len() == samples => {
            reader.next_frame(out)?;
        }
        Output::U16(out) if bits == 16 && out.len() == samples => {
            reader.next_frame(bytemuck::cast_slice_mut(out))?;
            for sample in out.iter_mut() {
                *sample = u16::from_be(*sample);
            }
        }
        out => {
            let (capacity, out_bits) = match out {
                Output::U8(out) => (out.len(), 8),
                Output::U16(out) => (out.len(), 16),
            };
            return Err(DepthPngError::OutputMismatch {
                width,
                height,
                bits,
                capacity,
                out_bits,
            });
        }
    }
    Ok([width, height])
}

/// Header facts, band table and zlib stream of a PNG that may be in the encoder's layout.
struct UpLayout<'a> {
    width: u32,
    height: u32,
    sixteen_bit: bool,
    /// Rows per band; the whole image without a `dpBN` chunk.
    band_rows: usize,
    /// Each band's offset into `zlib`.
    offsets: Vec<usize>,
    /// The IDAT payloads in order: borrowed when there is one chunk, joined otherwise.
    zlib: Cow<'a, [u8]>,
}

impl<'a> UpLayout<'a> {
    /// Walks the chunks and checks their CRCs; `None` unless the IHDR says greyscale, 8 or 16
    /// bits, not interlaced, the zlib stream has a plain header, and a band table is consistent.
    fn parse(png_bytes: &'a [u8]) -> Option<Self> {
        let mut rest = png_bytes.strip_prefix(PNG_SIGNATURE)?;
        let mut header: Option<(u32, u32, bool)> = None;
        let mut table: Option<&'a [u8]> = None;
        let mut idat: Vec<&'a [u8]> = Vec::new();
        while rest.len() >= 12 {
            let length = u32::from_be_bytes(rest[..4].try_into().ok()?) as usize;
            let end = 8usize.checked_add(length)?;
            let body = rest.get(4..end)?; // chunk type and data
            let crc = u32::from_be_bytes(rest.get(end..end + 4)?.try_into().ok()?);
            if crc32fast::hash(body) != crc {
                return None;
            }
            let (kind, data) = body.split_at(4);
            match kind {
                b"IHDR" => {
                    let ihdr: [u8; 13] = data.try_into().ok()?;
                    let [.., depth, color, compression, filter, interlace] = ihdr;
                    if color != 0
                        || compression != 0
                        || filter != 0
                        || interlace != 0
                        || !matches!(depth, 8 | 16)
                    {
                        return None;
                    }
                    let width = u32::from_be_bytes(ihdr[..4].try_into().ok()?);
                    let height = u32::from_be_bytes(ihdr[4..8].try_into().ok()?);
                    if width == 0 || height == 0 {
                        return None;
                    }
                    header = Some((width, height, depth == 16));
                }
                b"IDAT" => idat.push(data),
                b"IEND" => break,
                kind if kind == BANDS_CHUNK => table = Some(data),
                _ => {}
            }
            rest = &rest[end + 4..];
        }
        let (width, height, sixteen_bit) = header?;
        let zlib: Cow<'a, [u8]> = match idat.as_slice() {
            [] => return None,
            [only] => Cow::Borrowed(*only),
            several => Cow::Owned(several.concat()),
        };
        // CM 8 (deflate), valid FCHECK, no preset dictionary, room for the Adler-32 trailer.
        if zlib.len() < 6
            || zlib[0] & 0x0F != 8
            || !u16::from_be_bytes([zlib[0], zlib[1]]).is_multiple_of(31)
            || zlib[1] & 0x20 != 0
        {
            return None;
        }
        let h = height as usize;
        let (band_rows, offsets) = match table {
            None => (h, vec![2]),
            Some(table) => {
                let band_rows = u32::from_be_bytes(table.get(..4)?.try_into().ok()?) as usize;
                let offsets: Vec<usize> = table[4..]
                    .chunks_exact(4)
                    .map(|offset| {
                        u32::from_be_bytes([offset[0], offset[1], offset[2], offset[3]]) as usize
                    })
                    .collect();
                let consistent = band_rows > 0
                    && table.len() % 4 == 0
                    && offsets.len() == h.div_ceil(band_rows)
                    && offsets.first() == Some(&2)
                    && offsets.windows(2).all(|pair| pair[0] < pair[1])
                    && offsets.last().is_some_and(|&last| last < zlib.len() - 4);
                if !consistent {
                    return None;
                }
                (band_rows, offsets)
            }
        };
        Some(Self {
            width,
            height,
            sixteen_bit,
            band_rows,
            offsets,
            zlib,
        })
    }

    fn samples(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Decodes into `out` (the image's bytes, native-endian samples once done); `None` if a band
    /// fails to inflate, has a row that is not Up, or the Adler-32 does not match.
    fn decode(&self, out: &mut [u8], threads: usize) -> Option<()> {
        let sample_bytes = if self.sixteen_bit { 2 } else { 1 };
        let stride = self.width as usize * sample_bytes;
        let bands = self.offsets.len();
        let workers = worker_count(threads, bands);
        let stream_end = self.zlib.len() - 4;
        // Each worker locks only its own bands; the mutexes hand the disjoint slices across threads.
        let band_rows: Vec<Mutex<&mut [u8]>> = out
            .chunks_mut(self.band_rows * stride)
            .map(Mutex::new)
            .collect();

        // Pass 1, per band: inflate row by row and undo Up within the band (taking the row above
        // the band as zero), writing big-endian bytes; the Adler-32 of the inflated bytes.
        let adlers = for_each_band(bands, workers, |band| {
            let mut rows = band_rows[band].lock().ok()?;
            let end = self.offsets.get(band + 1).copied().unwrap_or(stream_end);
            let segment = &self.zlib[self.offsets[band]..end];
            inflate_band(segment, &mut rows, stride, band + 1 == bands)
        });
        let mut adler = 1;
        for (band, band_adler) in adlers.into_iter().enumerate() {
            let filtered_len = band_rows[band].lock().ok()?.len() / stride * (1 + stride);
            adler = zlib_rs::adler32::adler32_combine(adler, band_adler?, filtered_len as u64);
        }
        if adler.to_be_bytes() != self.zlib[stream_end..] {
            return None;
        }

        // Each band still lacks the row above it: the true last row of the band before, that is
        // the previous carry plus that band's local last row. Serial, one row per band.
        let mut carries = vec![vec![0u8; stride]; bands];
        for band in 1..bands {
            let previous = band_rows[band - 1].lock().ok()?;
            let local_last = &previous[previous.len() - stride..];
            let carry: Vec<u8> = carries[band - 1]
                .iter()
                .zip(local_last)
                .map(|(&c, &l)| c.wrapping_add(l))
                .collect();
            carries[band] = carry;
        }

        // Pass 2, per band: add the carry to every row and turn big-endian samples native.
        for_each_band(bands, workers, |band| {
            if let Ok(mut rows) = band_rows[band].lock() {
                finish_band(&mut rows, &carries[band], band > 0, self.sixteen_bit);
            }
        });
        Some(())
    }
}

/// Inflates one band's raw deflate segment row by row into `rows` (`stride` bytes each, filter
/// byte dropped), undoing Up within the band. Returns the Adler-32 of the inflated bytes.
fn inflate_band(segment: &[u8], rows: &mut [u8], stride: usize, last: bool) -> Option<u32> {
    let mut inflater = flate2::Decompress::new(false);
    let row_len = 1 + stride;
    let row_count = rows.len() / stride;
    let mut scratch = vec![0u8; INFLATE_ROWS.min(row_count) * row_len];
    let mut adler = 1;
    for first in (0..row_count).step_by(INFLATE_ROWS) {
        let count = INFLATE_ROWS.min(row_count - first);
        let scratch = &mut scratch[..count * row_len];
        let produced = inflater.total_out();
        let consumed = inflater.total_in() as usize;
        inflater
            .decompress(&segment[consumed..], scratch, flate2::FlushDecompress::None)
            .ok()?;
        if inflater.total_out() - produced != scratch.len() as u64 {
            return None;
        }
        adler = zlib_rs::adler32::adler32(adler, scratch);
        for (offset, row) in scratch.chunks_exact(row_len).enumerate() {
            if row[0] != 2 {
                return None;
            }
            let index = first + offset;
            let (above, current) = rows.split_at_mut(index * stride);
            let current = &mut current[..stride];
            if index == 0 {
                current.copy_from_slice(&row[1..]);
            } else {
                let previous = &above[(index - 1) * stride..];
                for ((out, &delta), &up) in current.iter_mut().zip(&row[1..]).zip(previous) {
                    *out = delta.wrapping_add(up);
                }
            }
        }
    }
    // Consume the segment's tail: the empty stored block of a full flush, or the final block's end.
    let consumed = inflater.total_in() as usize;
    let flush = if last {
        flate2::FlushDecompress::Finish
    } else {
        flate2::FlushDecompress::Sync
    };
    let status = inflater
        .decompress(&segment[consumed..], &mut [], flush)
        .ok()?;
    let ended = if last {
        status == flate2::Status::StreamEnd
    } else {
        inflater.total_in() as usize == segment.len()
    };
    ended.then_some(adler)
}

/// Adds `carry` to every row (byte-wise, modulo 256) when `add_carry`, and turns 16-bit samples
/// from big-endian to native order.
fn finish_band(rows: &mut [u8], carry: &[u8], add_carry: bool, sixteen_bit: bool) {
    if sixteen_bit {
        let samples: &mut [u16] = bytemuck::cast_slice_mut(rows);
        let carry: &[u16] = bytemuck::cast_slice(carry);
        for row in samples.chunks_exact_mut(carry.len()) {
            for (sample, &c) in row.iter_mut().zip(carry) {
                let value = if add_carry {
                    // Byte-wise: no carry from one byte into the other.
                    ((*sample & 0xFF00).wrapping_add(c & 0xFF00) & 0xFF00)
                        | (sample.wrapping_add(c) & 0x00FF)
                } else {
                    *sample
                };
                *sample = u16::from_be(value);
            }
        }
    } else if add_carry {
        for row in rows.chunks_exact_mut(carry.len()) {
            for (byte, &c) in row.iter_mut().zip(carry) {
                *byte = byte.wrapping_add(c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(width: u32, height: u32) -> Vec<u16> {
        (0..width * height)
            .map(|i| (i * 7 + i / width * 300) as u16)
            .collect()
    }

    #[test]
    fn bands_follow_the_image_size() {
        for ([width, height], bands) in [([256, 192], 1), ([512, 416], 1), ([1920, 1440], 12)] {
            let png_bytes = encode_u16(
                &ramp(width, height),
                [width, height],
                &EncodeOptions::default(),
            )
            .unwrap();
            let layout = UpLayout::parse(&png_bytes).expect("in the encoder's layout");
            assert_eq!(layout.offsets.len(), bands, "{width}x{height}");
        }
    }

    #[test]
    fn every_band_count_and_thread_count_decodes_through_the_fast_path() {
        let (width, height) = (37, 23);
        let depth = ramp(width, height);
        for band_rows in [1, 2, 5, 22, 23] {
            for level in [0, 1, 4, 9] {
                let options = EncodeOptions {
                    level,
                    threads: 3,
                    band_rows: Some(band_rows),
                };
                let png_bytes = encode_u16(&depth, [width, height], &options).unwrap();
                let layout = UpLayout::parse(&png_bytes).expect("in the encoder's layout");
                assert_eq!(layout.offsets.len(), height.div_ceil(band_rows) as usize);
                for threads in [1, 2, 8] {
                    let mut out = vec![0u16; depth.len()];
                    layout
                        .decode(bytemuck::cast_slice_mut(&mut out), threads)
                        .expect("fast path");
                    assert_eq!(
                        out, depth,
                        "{band_rows} rows per band, level {level}, {threads} threads"
                    );
                }
            }
        }
    }

    #[test]
    fn other_layouts_leave_the_fast_path() {
        let fixture = |name: &str| {
            std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures")
                    .join(name),
            )
            .unwrap()
        };
        let up = fixture("opencv_u16_level4_up.png");
        let layout = UpLayout::parse(&up).expect("OpenCV with the Up filter, two IDAT chunks");
        assert!(layout
            .decode(&mut vec![0u8; layout.samples() * 2], 1)
            .is_some());
        for name in [
            "opencv_u16_default.png",
            "pillow_u16_level6.png",
            "pillow_u8_level6.png",
        ] {
            let png_bytes = fixture(name);
            let layout = UpLayout::parse(&png_bytes).expect("greyscale");
            let bytes = layout.samples() * if layout.sixteen_bit { 2 } else { 1 };
            assert!(layout.decode(&mut vec![0u8; bytes], 1).is_none(), "{name}");
        }
        assert!(UpLayout::parse(&fixture("pillow_rgb8.png")).is_none());
    }
}
