//! The GPU [`CornerScan`]: FAST-9 scores and kornia's local-maximum filter on
//! the device, one download per frame, every rung of the ladder derived from it.

use cubecl::prelude::*;
use kornia_imgproc::features::FastCorner;

use super::kernels::{self, MASK_BITS, RING_BIAS};
use super::pyramid::{Level0, Level0Table};
use crate::frontend::detect::{
    CornerScan, DetectError, FAST_BORDER, FAST_RING_COLUMN, FAST_RING_ROW, block_filter_end,
    opencv_corner_score,
};
use crate::image::ImageU16;

/// The three device buffers one frame geometry needs, kept between frames.
struct ScanBuffers {
    /// The dense FAST-9 score, one byte per pixel.
    score: cubecl::server::Handle,
    /// The score where the local-maximum filter kept it.
    kept: cubecl::server::Handle,
    /// One bit per column of `kept`, thirty-two to a word.
    mask: cubecl::server::Handle,
    /// Pixels these were sized for.
    pixels: usize,
    /// Mask words these were sized for.
    mask_len: usize,
}

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
    /// Level 0 of each camera's pyramid, as the builder published it. When the
    /// entry for the camera being scanned matches the frame's geometry the
    /// score kernel reads it and this stage uploads nothing at all; otherwise
    /// — a scanner with no builder beside it, which is how the tolerance tests
    /// drive it — the frame goes up here.
    level0: Level0Table,
    /// Frames this scanner has uploaded itself, which the shared level-0 path
    /// is meant to keep at zero.
    uploads: usize,
    /// A repack buffer for a frame whose stride exceeds its width, reused
    /// between frames so this stage reaches the **host** allocator not at all
    /// (D49). The device side is a different promise and a weaker one: see
    /// [`super`]'s "Residency".
    packed: Vec<u16>,
    /// The dense score image, the candidate image and the column bitmask, per
    /// camera: allocated once per camera geometry rather than once per frame.
    ///
    /// Indexed by camera because `scan` runs once per camera per frameset, so
    /// one slot only holds for a rig whose cameras are all the same size — on a
    /// mixed-geometry rig, which the port supports deliberately, every scan
    /// missed and the three `client.empty` calls came back every camera every
    /// frameset. `the_gpu_corner_scan_reads_the_pyramid_and_uploads_nothing`
    /// drives 960x240 next to 512x192 and asserts the allocation count does not
    /// move across framesets.
    ///
    /// Every one of the three kernels writes every element of its output in
    /// range — the score kernel zeroes the margin, the filter writes each pixel
    /// and the packer each word — so a reused buffer cannot carry a previous
    /// frame's candidate, and `a_reused_corner_scan_carries_only_the_newest_frame`
    /// is the test that says so. Three `client.empty` calls per camera per
    /// frameset were the largest source of pool churn in the lane.
    buffers: Vec<Option<ScanBuffers>>,
    /// Times the three device buffers have been allocated, which a rig of one
    /// geometry keeps at one.
    buffer_allocations: usize,
    /// The candidate image, one byte per pixel, as it came back — the device
    /// read's own buffer, not a copy of it.
    ///
    /// `cubecl::bytes::Bytes` is owned, `Send + Sync` and derefs to `[u8]`, so
    /// there is nothing a `Vec` adds except the copy: `extend_from_slice` of
    /// the candidate image and the bitmask was 0.92 + 0.115 MB per camera per
    /// frameset — **2.07 MB per two-camera frameset** — on top of a read that
    /// had already handed back owned host memory. `None` before the first scan.
    kept: Option<cubecl::bytes::Bytes>,
    /// One bit per column of `kept`, thirty-two to a word; likewise the read's
    /// own buffer, cast to `u32` once per band rather than copied.
    mask: Option<cubecl::bytes::Bytes>,
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
        for offsets in [FAST_RING_ROW, FAST_RING_COLUMN] {
            for offset in offsets {
                ring.push((offset + RING_BIAS as i32) as u32);
            }
        }
        Self {
            ring: client.create_from_slice(u32::as_bytes(&ring)),
            level0: Level0Table::default(),
            uploads: 0,
            packed: Vec::new(),
            buffers: Vec::new(),
            buffer_allocations: 0,
            kept: None,
            mask: None,
            words: 0,
            width: 0,
            height: 0,
            bands: Vec::new(),
            client,
        }
    }

    /// Read level 0 out of `table` rather than uploading the frame.
    ///
    /// Wired by [`super::gpu_backends`], which builds the scanner and the
    /// pyramid builder on one client: they are handed the same pixels, so
    /// sharing them is the difference between one upload per camera per
    /// frameset and two.
    pub fn share_level0(&mut self, table: Level0Table) {
        self.level0 = table;
    }

    /// Frames this scanner uploaded itself. Zero once
    /// [`GpuCornerScan::share_level0`] is wired to a builder that runs first.
    pub fn frame_uploads(&self) -> usize {
        self.uploads
    }

    /// Times the score, candidate and bitmask buffers have been allocated.
    ///
    /// One per camera geometry for the life of the scanner; anything that grows
    /// with the frameset count is the pool churn step 1b removed coming back.
    pub fn buffer_allocations(&self) -> usize {
        self.buffer_allocations
    }

    /// The buffer the score kernel reads for camera `camera`, and its length.
    ///
    /// The pyramid's own level 0 when the builder published one of this
    /// geometry; a fresh upload otherwise. The geometry check is what makes the
    /// fallback safe rather than hopeful: a stale entry from another frame size
    /// is refused instead of read.
    fn frame(&mut self, camera: usize, image: &ImageU16) -> (cubecl::server::Handle, usize) {
        // `image` is the authority on the geometry, not `self.width`/`self.height`:
        // those are the same frame's, set by `scan` two lines up, and one fact
        // with two sources inside one call is how they come apart.
        let (width, height): (usize, usize) = (image.width(), image.height());
        let pixels: usize = width * height;
        let shared: Option<Level0> = self
            .level0
            .lock()
            .ok()
            .and_then(|table| table.get(camera).cloned().flatten())
            .filter(|level0| level0.width == width && level0.height == height);
        if let Some(level0) = shared {
            return (level0.handle, pixels);
        }

        self.uploads += 1;
        // The frame goes up as `u16` and the `>> 8` the detector reads happens
        // on the device: the extra 0.9 MB over the bus costs less than a
        // whole-frame narrowing pass on the host. A frame whose stride exceeds
        // its width is repacked row by row, as the pyramid's staging does.
        if image.stride() == width {
            return (
                self.client
                    .create_from_slice(u16::as_bytes(&image.data()[..pixels])),
                pixels,
            );
        }
        self.packed.clear();
        self.packed.reserve(pixels);
        for y in 0..height {
            self.packed.extend_from_slice(image.row(y));
        }
        (
            self.client.create_from_slice(u16::as_bytes(&self.packed)),
            pixels,
        )
    }

    /// One row's candidates over `threshold`, appended in column order.
    ///
    /// The bitmask says where to look: one word per thirty-two columns, and
    /// `trailing_zeros` walks only the bits that are set, so a row of 960
    /// columns costs thirty word loads plus one score load per candidate.
    ///
    /// `scores` and `bits` are the downloaded buffers, sliced by the caller
    /// once per band rather than re-cast per row.
    fn filter_row(
        &self,
        scores: &[u8],
        bits: &[u32],
        y: usize,
        threshold: u8,
        out: &mut Vec<FastCorner>,
    ) {
        let row: &[u8] = &scores[y * self.width..(y + 1) * self.width];
        let words: &[u32] = &bits[y * self.words..(y + 1) * self.words];
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
    fn scan(&mut self, camera: usize, image: &ImageU16) -> Result<(), DetectError> {
        self.bands.clear();
        self.width = image.width();
        self.height = image.height();
        let pixels: usize = self.width * self.height;
        self.words = self.width.div_ceil(MASK_BITS);
        let mask_len: usize = self.words * self.height;

        let (handle, handle_len): (cubecl::server::Handle, usize) = self.frame(camera, image);
        if self.buffers.len() <= camera {
            self.buffers.resize_with(camera + 1, || None);
        }
        let slot: &mut Option<ScanBuffers> = &mut self.buffers[camera];
        let fits: bool = slot
            .as_ref()
            .is_some_and(|buffers| buffers.pixels == pixels && buffers.mask_len == mask_len);
        let buffers: &ScanBuffers = match slot {
            Some(existing) if fits => existing,
            slot => {
                self.buffer_allocations += 1;
                slot.insert(ScanBuffers {
                    score: self.client.empty(pixels),
                    kept: self.client.empty(pixels),
                    mask: self.client.empty(mask_len * size_of::<u32>()),
                    pixels,
                    mask_len,
                })
            }
        };
        let (score, kept, mask) = (
            buffers.score.clone(),
            buffers.kept.clone(),
            buffers.mask.clone(),
        );
        kernels::launch_fast_score::<R>(
            &self.client,
            (&handle, handle_len),
            (&self.ring, 32),
            (&score, pixels),
            self.width,
            self.height,
            FAST_BORDER,
        );
        let (filtered_end, use_filter): (usize, bool) = block_filter_end(self.width);
        kernels::launch_fast_localmax::<R>(
            &self.client,
            (&score, pixels),
            (&kept, pixels),
            self.width,
            self.height,
            FAST_BORDER,
            filtered_end,
            use_filter,
        );
        kernels::launch_fast_mask::<R>(
            &self.client,
            (&kept, pixels),
            (&mask, mask_len),
            self.width,
            self.height,
            self.words,
        );

        // One read for both, so one synchronisation for the frame — and the
        // fallible form of it: `client.read` is `read_sync(..).expect("TODO")`,
        // and a panic here would unwind out of the frontend with the GIL
        // detached (decision D32).
        let reads: Vec<cubecl::bytes::Bytes> = cubecl::reader::read_sync(
            self.client.read_async(vec![kept, mask]),
        )
        .map_err(|error| super::read_failed("the candidate image and its bitmask", &error))?;
        // One buffer per handle, in the order they were asked for; anything else
        // is the runtime breaking its own contract rather than short data.
        let Ok([kept_bytes, mask_bytes]) = <[cubecl::bytes::Bytes; 2]>::try_from(reads) else {
            return Err(super::GpuError::DeviceReadFailed {
                what: "the corner scan's two buffers",
            }
            .into());
        };
        // Checked one buffer at a time, so a short read says which one was
        // short: summing the two lengths made that unsayable.
        for (what, actual, expected) in [
            ("the candidate image", kept_bytes.len(), pixels),
            (
                "the candidate bitmask",
                mask_bytes.len(),
                mask_len * size_of::<u32>(),
            ),
        ] {
            if actual != expected {
                return Err(super::GpuError::ShortRead {
                    what,
                    actual,
                    expected,
                }
                .into());
            }
        }
        // Held, not copied: the read already owns host memory of exactly this
        // length, and on CUDA it may be pinned, which is where the band walk
        // wants to read from anyway.
        self.kept = Some(kept_bytes);
        self.mask = Some(mask_bytes);
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
        let first: usize = y.max(FAST_BORDER);
        let last: usize = (y + rows).min(self.height.saturating_sub(FAST_BORDER));
        let mut corners: Vec<FastCorner> = Vec::new();
        // A threshold at or over 255 admits nothing: the score is a `u8`. And
        // no scan has run means no rows to walk, which `first..last` already
        // says on a zero geometry.
        if let (Ok(bound), Some(kept), Some(mask)) = (
            u8::try_from(threshold.max(0)),
            self.kept.as_ref(),
            self.mask.as_ref(),
        ) {
            let scores: &[u8] = kept;
            let bits: &[u32] = u32::from_bytes(mask);
            for row in first..last {
                self.filter_row(scores, bits, row, bound, &mut corners);
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
            .field("uploads", &self.uploads)
            .finish()
    }
}
