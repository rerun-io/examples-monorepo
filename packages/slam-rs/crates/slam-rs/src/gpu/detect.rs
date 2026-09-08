//! The GPU [`CornerScan`]: FAST-9 scores and kornia's local-maximum filter on
//! the device, one download per frame, every rung of the ladder derived from it.

use cubecl::prelude::*;
use kornia_imgproc::features::FastCorner;

use super::kernels::{self, MASK_BITS, RING_BIAS};
use crate::frontend::detect::{CornerScan, DetectError, opencv_corner_score};
use crate::image::ImageU16;

/// The ring margin `fast_detect_rect_u8` clamps to on every side.
const MARGIN: usize = 3;

/// kornia's Bresenham ring, `fast.rs:489-490`.
const RING_ROW: [i32; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
/// The column half of the same ring.
const RING_COLUMN: [i32; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

/// The width at which kornia turns its in-block local-maximum filter on
/// (`fast.rs:517`).
const FILTER_WIDTH: usize = 800;

/// One row band at one threshold, already filtered out of the candidate image.
#[derive(Debug)]
struct Band {
    /// First row of the band.
    y: usize,
    /// Threshold it was filtered at.
    threshold: i32,
    /// Candidates over the whole width, row-major, carrying OpenCV's score.
    corners: Vec<FastCorner>,
}

/// The GPU corner scanner.
///
/// One frame costs three launches and **one** synchronisation: the dense FAST-9
/// score image, kornia's local-maximum filter over it, and a bitmask of the
/// columns that survived. Both the candidate image and the bitmask come back in
/// one read. Every band the detector then asks for is a row range walked through
/// the bitmask — one word per thirty-two columns, and the score is only touched
/// where a bit is set — filtered by `> threshold`, which is exact for every rung
/// because the local-maximum filter is threshold-independent and the candidate
/// test *is* `corner_score_9 > threshold` (`tests/fast_model.rs`).
///
/// The bitmask is the difference between this being faster and not. The first
/// version downloaded only the candidate image and walked it byte by byte; the
/// detector asks for up to eighty whole-width bands per camera per frame, and
/// that walk measured 1.53 ms of host time against the 1.81 ms of kornia sweeps
/// it replaced — a kernel that was exact and not faster. The report carries both
/// rows.
pub struct GpuCornerScan<R: Runtime> {
    client: ComputeClient<R>,
    /// The biased ring, uploaded once.
    ring: cubecl::server::Handle,
    /// The candidate image, one byte per pixel, as it came back.
    kept: Vec<u8>,
    /// One bit per column of `kept`, thirty-two to a word.
    mask: Vec<u32>,
    /// Words per row of `mask`.
    words: usize,
    width: usize,
    height: usize,
    /// Bands already filtered out of `kept`, in the order they were asked for.
    bands: Vec<Band>,
}

impl<R: Runtime> GpuCornerScan<R> {
    /// A scanner on `client`, with the ring uploaded.
    pub fn new(client: ComputeClient<R>) -> Self {
        let mut ring: Vec<u32> = Vec::with_capacity(32);
        for offsets in [RING_ROW, RING_COLUMN] {
            for offset in offsets {
                ring.push((offset + RING_BIAS as i32) as u32);
            }
        }
        Self {
            ring: client.create_from_slice(u32::as_bytes(&ring)),
            kept: Vec::new(),
            mask: Vec::new(),
            words: 0,
            width: 0,
            height: 0,
            bands: Vec::new(),
            client,
        }
    }

    /// One row's candidates over `threshold`, appended in column order.
    ///
    /// The bitmask says where to look: one word per thirty-two columns, and
    /// `trailing_zeros` walks only the bits that are set, so a row of 960
    /// columns costs thirty word loads plus one score load per candidate.
    fn filter_row(&self, y: usize, threshold: u8, out: &mut Vec<FastCorner>) {
        let row: &[u8] = &self.kept[y * self.width..(y + 1) * self.width];
        let words: &[u32] = &self.mask[y * self.words..(y + 1) * self.words];
        for (index, word) in words.iter().enumerate() {
            let mut bits: u32 = *word;
            while bits != 0 {
                let bit: usize = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let x: usize = index * MASK_BITS + bit;
                let score: u8 = row[x];
                if score > threshold {
                    out.push(FastCorner {
                        xy: [x as f32, y as f32],
                        // The response kornia reports is `score / 255`, which
                        // `opencv_corner_score` turns back into `score - 1`.
                        response: opencv_corner_score(f32::from(score) / 255.0),
                    });
                }
            }
        }
    }
}

impl<R: Runtime> CornerScan for GpuCornerScan<R> {
    fn scan(&mut self, image: &ImageU16) -> Result<(), DetectError> {
        self.bands.clear();
        self.width = image.width();
        self.height = image.height();
        let pixels: usize = self.width * self.height;
        self.words = self.width.div_ceil(MASK_BITS);
        let mask_len: usize = self.words * self.height;

        // The frame goes up as `u16` and the `>> 8` the detector reads happens on
        // the device: the extra 0.9 MB over the bus costs less than a
        // whole-frame narrowing pass on the host. A frame whose stride exceeds
        // its width is repacked row by row, as the pyramid's staging does.
        let handle: cubecl::server::Handle = if image.stride() == self.width {
            self.client
                .create_from_slice(u16::as_bytes(&image.data()[..pixels]))
        } else {
            let mut packed: Vec<u16> = Vec::with_capacity(pixels);
            for y in 0..self.height {
                packed.extend_from_slice(image.row(y));
            }
            self.client.create_from_slice(u16::as_bytes(&packed))
        };
        let score: cubecl::server::Handle = self.client.empty(pixels);
        let kept: cubecl::server::Handle = self.client.empty(pixels);
        let mask: cubecl::server::Handle = self.client.empty(mask_len * size_of::<u32>());
        kernels::launch_fast_score::<R>(
            &self.client,
            (&handle, pixels),
            (&self.ring, 32),
            (&score, pixels),
            self.width,
            self.height,
            MARGIN,
        );
        // kornia's SIMD loop runs while `x + 16 <= width - margin`, so the whole
        // blocks end here and the scalar tail past it is unfiltered.
        let blocks: usize = self.width.saturating_sub(2 * MARGIN) / 16;
        kernels::launch_fast_localmax::<R>(
            &self.client,
            (&score, pixels),
            (&kept, pixels),
            self.width,
            self.height,
            MARGIN,
            MARGIN + blocks * 16,
            self.width >= FILTER_WIDTH,
        );
        kernels::launch_fast_mask::<R>(
            &self.client,
            (&kept, pixels),
            (&mask, mask_len),
            self.width,
            self.height,
            self.words,
        );

        // One read for both, so one synchronisation for the frame.
        let mut buffers = self.client.read(vec![kept, mask]);
        let Some(mask_bytes) = buffers.pop() else {
            return Err(DetectError::DeviceRead {
                expected: mask_len * size_of::<u32>(),
                actual: 0,
            });
        };
        let Some(kept_bytes) = buffers.pop() else {
            return Err(DetectError::DeviceRead {
                expected: pixels,
                actual: 0,
            });
        };
        if kept_bytes.len() != pixels || mask_bytes.len() != mask_len * size_of::<u32>() {
            return Err(DetectError::DeviceRead {
                expected: pixels + mask_len * size_of::<u32>(),
                actual: kept_bytes.len() + mask_bytes.len(),
            });
        }
        self.kept.clear();
        self.kept.extend_from_slice(&kept_bytes);
        self.mask.clear();
        self.mask.extend_from_slice(u32::from_bytes(&mask_bytes));
        Ok(())
    }

    fn band(
        &mut self,
        y: usize,
        rows: usize,
        threshold: i32,
    ) -> Result<&[FastCorner], DetectError> {
        if let Some(index) = self
            .bands
            .iter()
            .position(|band| band.y == y && band.threshold == threshold)
        {
            return Ok(&self.bands[index].corners);
        }
        // `row_start = rows.start.max(margin)`, `row_end = rows.end.min(height -
        // margin)` (`fast.rs:495-498`).
        let first: usize = y.max(MARGIN);
        let last: usize = (y + rows).min(self.height.saturating_sub(MARGIN));
        let mut corners: Vec<FastCorner> = Vec::new();
        // A threshold at or over 255 admits nothing: the score is a `u8`.
        if let Ok(bound) = u8::try_from(threshold.max(0)) {
            for row in first..last {
                self.filter_row(row, bound, &mut corners);
            }
        }
        self.bands.push(Band {
            y,
            threshold,
            corners,
        });
        // Either the band the search found or the one just pushed.
        let last_index: usize = self.bands.len() - 1;
        Ok(&self.bands[last_index].corners)
    }
}

impl<R: Runtime> std::fmt::Debug for GpuCornerScan<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuCornerScan")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bands", &self.bands.len())
            .finish()
    }
}
