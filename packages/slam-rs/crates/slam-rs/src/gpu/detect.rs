//! The GPU [`CornerScan`]: FAST-9 scores and kornia's local-maximum filter on
//! the device, one download per frame, every rung of the ladder derived from it.

use cubecl::prelude::*;
use kornia_imgproc::features::FastCorner;

use super::kernels::{self, MASK_BITS, RING_BIAS};
use super::pyramid::{Level0, Level0Table};
use super::{GpuError, guarded};
use crate::frontend::cell::SelectionStatus;
use crate::frontend::detect::{
    BandCache, BandRequest, CellSelect, CornerScan, DetectError, FAST_BORDER, FAST_RING_COLUMN,
    FAST_RING_ROW, block_filter_end, opencv_corner_score,
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

/// What one frame's two shared kernels leave on the device.
struct ScanHandles {
    /// The candidate image: the score where the local-maximum filter kept it.
    kept: cubecl::server::Handle,
    /// Its column bitmask, which only the band path goes on to fill.
    mask: cubecl::server::Handle,
    /// Pixels in the frame.
    pixels: usize,
    /// Words in the bitmask.
    mask_len: usize,
}

#[derive(Default)]
enum Selection {
    #[default]
    Empty,
    Pending(CellSelect),
    Ready(CellSelect),
}

#[derive(Default)]
struct CameraWorkspace {
    buffers: Option<ScanBuffers>,
    keys: Option<(cubecl::server::Handle, usize)>,
    host_keys: Vec<u32>,
    selection: Selection,
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
    /// Reusable allocations and explicit selection readiness, indexed by camera.
    cameras: Vec<CameraWorkspace>,
    reads: super::selection_batch::Producer,
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
    bands: BandCache,
}

impl<R: Runtime> GpuCornerScan<R> {
    /// A scanner on `client`, with the ring uploaded.
    ///
    /// Fallible because the upload is a device operation like any other:
    /// `create_from_slice` **panics** inside CubeCL's own client when the worker
    /// submission fails, so a constructor with no error channel is a path from a
    /// dying device to an unwind through the caller (decision D32).
    ///
    /// # Errors
    ///
    /// [`GpuError::DeviceLost`] when the upload panics instead of returning.
    pub fn new(client: ComputeClient<R>) -> Result<Self, GpuError> {
        guarded(
            GpuError::DeviceLost {
                what: "corner scan setup",
            },
            || {
                let mut ring: Vec<u32> = Vec::with_capacity(32);
                for offsets in [FAST_RING_ROW, FAST_RING_COLUMN] {
                    for offset in offsets {
                        ring.push((offset + RING_BIAS as i32) as u32);
                    }
                }
                Ok(Self {
                    ring: super::submission::upload(&client, u32::as_bytes(&ring)),
                    level0: Level0Table::default(),
                    uploads: 0,
                    packed: Vec::new(),
                    cameras: Vec::new(),
                    reads: super::selection_batch::endpoints().0,
                    buffer_allocations: 0,
                    kept: None,
                    mask: None,
                    words: 0,
                    width: 0,
                    height: 0,
                    bands: BandCache::default(),
                    client,
                })
            },
        )
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

    /// Carry this scanner's keys in a tracker's download on the same client.
    ///
    /// The private endpoints cannot be cloned or shared with another scanner.
    ///
    /// # Panics
    /// Panics if the tracker belongs to another device client.
    pub fn share_reads<P: crate::frontend::patterns::Pattern>(
        &mut self,
        tracker: &mut super::GpuPatchTracker<P, R>,
    ) {
        assert!(tracker.uses_client(&self.client));
        let (producer, consumer) = super::selection_batch::endpoints();
        self.reads = producer;
        tracker.share_reads(consumer);
    }

    fn abort_selection(&mut self) {
        self.reads.abort();
        for camera in &mut self.cameras {
            camera.selection = Selection::Empty;
        }
    }

    /// Frames this scanner uploaded itself. Zero once
    /// [`GpuCornerScan::share_level0`] is wired to a builder that runs first.
    pub fn frame_uploads(&self) -> usize {
        self.uploads
    }

    /// Times the score, candidate and bitmask buffers have been allocated.
    ///
    /// One per camera geometry for the life of the scanner; anything that grows
    /// with the frameset count is per-frame pool churn coming back.
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
        let locked = self.level0.lock();
        if locked.is_err() {
            log::warn!(
                "the shared level-0 table is poisoned: camera {camera} uploads its frame twice from here on"
            );
        }
        let shared: Option<Level0> = locked
            .ok()
            .and_then(|table| table.get(camera).cloned().flatten())
            .filter(|level0| level0.width == width && level0.height == height);
        if let Some(level0) = shared {
            return (level0.handle, pixels);
        }

        self.uploads += 1;
        // The frame goes up as `u16` and the `>> 8` the detector reads happens
        // on the device: the extra 0.9 MB over the bus costs less than a
        // whole-frame narrowing pass on the host.
        super::upload_frame(&self.client, image, &mut self.packed)
    }

    /// This camera's candidate kernels and its cell selection, launched into the
    /// camera's own key buffer; the handle and the cell count to read back.
    ///
    /// `None` when the grid needs more cubes in one dispatch dimension than a
    /// WebGPU implementation must allow, which is the caller's signal to leave
    /// the frame alone and let the band walk answer.
    fn launch_selection(
        &mut self,
        camera: usize,
        image: &ImageU16,
        select: &CellSelect,
    ) -> Option<(cubecl::server::Handle, usize)> {
        let grid: &crate::frontend::detect::CellGrid = &select.grid;
        let (cells_x, cells_y) = grid.dimensions();
        let ceiling: usize = kernels::MAX_CUBES_PER_DIM as usize;
        if cells_x > ceiling || cells_y > ceiling {
            return None;
        }
        let cells: usize = cells_x * cells_y;
        let (width, height): (usize, usize) = (image.width(), image.height());
        let handles: ScanHandles = self.candidates(camera, image);

        if self.cameras.len() <= camera {
            self.cameras
                .resize_with(camera + 1, CameraWorkspace::default);
        }
        let slot: &mut Option<(cubecl::server::Handle, usize)> = &mut self.cameras[camera].keys;
        let fits: bool = slot.as_ref().is_some_and(|(_, sized)| *sized == cells);
        let (best, best_len): (cubecl::server::Handle, usize) = match slot {
            Some(existing) if fits => existing.clone(),
            slot => {
                self.buffer_allocations += 1;
                slot.insert((super::empty(&self.client, cells * size_of::<u32>()), cells))
                    .clone()
            }
        };

        kernels::launch_fast_cell_select::<R>(
            &self.client,
            (&handles.kept, handles.pixels),
            (&best, best_len),
            kernels::CellSelectGeometry {
                width,
                height,
                cell: grid.cell,
                x_start: grid.x_start,
                y_start: grid.y_start,
                cells_x,
                cells_y,
                // The score is a `u8`, so a rung at or over 255 admits nothing
                // and one under zero is every candidate.
                threshold: select.threshold.clamp(0, 255) as u32,
                safe_radius: select.safe_radius,
                // `img_raw.w / 2` is an integer halving.
                centre_x: (width / 2) as f32,
                centre_y: (height / 2) as f32,
            },
        );
        Some((best, cells))
    }

    /// Level 0 in, the candidate image out: the two kernels both entry points
    /// run, this camera's buffers sized and reused.
    ///
    /// The previous frame's readback is dropped **first**, and the geometry is
    /// recorded before anything can fail, so a scan that dies further on leaves
    /// a scanner that refuses a band rather than one that answers with the last
    /// frame's corners under this frame's width (decision D32).
    fn candidates(&mut self, camera: usize, image: &ImageU16) -> ScanHandles {
        self.bands.clear();
        self.kept = None;
        self.mask = None;
        self.width = image.width();
        self.height = image.height();
        let pixels: usize = self.width * self.height;
        self.words = self.width.div_ceil(MASK_BITS);
        let mask_len: usize = self.words * self.height;

        let (handle, handle_len): (cubecl::server::Handle, usize) = self.frame(camera, image);
        if self.cameras.len() <= camera {
            self.cameras
                .resize_with(camera + 1, CameraWorkspace::default);
        }
        let slot: &mut Option<ScanBuffers> = &mut self.cameras[camera].buffers;
        let fits: bool = slot
            .as_ref()
            .is_some_and(|buffers| buffers.pixels == pixels && buffers.mask_len == mask_len);
        let buffers: &ScanBuffers = match slot {
            Some(existing) if fits => existing,
            slot => {
                self.buffer_allocations += 1;
                slot.insert(ScanBuffers {
                    score: super::empty(&self.client, pixels),
                    kept: super::empty(&self.client, pixels),
                    mask: super::empty(&self.client, mask_len * size_of::<u32>()),
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
        ScanHandles {
            kept,
            mask,
            pixels,
            mask_len,
        }
    }
}

/// A downloaded key buffer as `u32`, refused when it is not the grid's length.
fn checked_keys(keys: &cubecl::bytes::Bytes, cells: usize) -> Result<&[u32], DetectError> {
    let expected: usize = cells * size_of::<u32>();
    if keys.len() != expected {
        return Err(super::GpuError::ShortRead {
            what: "the cell winner keys",
            actual: keys.len(),
            expected,
        }
        .into());
    }
    Ok(u32::from_bytes(keys))
}

/// One row's candidates over `threshold`, appended in column order.
///
/// The bitmask says where to look: one word per thirty-two columns, and
/// `trailing_zeros` walks only the bits that are set, so a row of 960 columns
/// costs thirty word loads plus one score load per candidate.
///
/// `scores` and `bits` are the downloaded buffers, sliced by the caller once per
/// band rather than re-cast per row; `width` and `stride` are the frame's row
/// length in pixels and in mask words. Free rather than a method because
/// [`CornerScan::band`] calls it with the band cache mutably borrowed.
fn filter_row(
    scores: &[u8],
    bits: &[u32],
    width: usize,
    stride: usize,
    y: usize,
    threshold: u8,
    out: &mut Vec<FastCorner>,
) {
    let row: &[u8] = &scores[y * width..(y + 1) * width];
    let words: &[u32] = &bits[y * stride..(y + 1) * stride];
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

impl<R: Runtime> CornerScan for GpuCornerScan<R> {
    fn scan(&mut self, camera: usize, image: &ImageU16) -> Result<(), DetectError> {
        guarded(
            GpuError::DeviceLost {
                what: "corner scan",
            },
            || {
                let handles: ScanHandles = self.candidates(camera, image);
                kernels::launch_fast_mask::<R>(
                    &self.client,
                    (&handles.kept, handles.pixels),
                    (&handles.mask, handles.mask_len),
                    self.width,
                    self.height,
                    self.words,
                );

                // Where a test makes this scan fail as a lost device would, on
                // the far side of the geometry above (test-only).
                #[cfg(test)]
                super::fire_if_armed(super::CORNER_SCAN_READ);

                // One read for both, so one synchronisation for the frame.
                let reads: Vec<cubecl::bytes::Bytes> = super::read_blocking(
                    &self.client,
                    vec![handles.kept, handles.mask],
                    "the candidate image and its bitmask",
                    &super::seam::READ_DETECT,
                )?;
                // One buffer per handle, in the order they were asked for; anything else
                // is the runtime breaking its own contract rather than short data.
                let Ok([kept_bytes, mask_bytes]) = <[cubecl::bytes::Bytes; 2]>::try_from(reads)
                else {
                    return Err(super::GpuError::DeviceReadFailed {
                        what: "the corner scan's two buffers",
                    }
                    .into());
                };
                // Checked one buffer at a time, so a short read says which one was
                // short: summing the two lengths made that unsayable.
                for (what, actual, expected) in [
                    ("the candidate image", kept_bytes.len(), handles.pixels),
                    (
                        "the candidate bitmask",
                        mask_bytes.len(),
                        handles.mask_len * size_of::<u32>(),
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
                // length, which is where the band walk
                // wants to read from anyway.
                self.kept = Some(kept_bytes);
                self.mask = Some(mask_bytes);
                Ok(())
            },
        )
    }

    /// One packed key per grid cell, and no candidate image at all.
    ///
    /// This is where the 2.07 MB the band path downloads per two-camera frameset
    /// goes away: the selection kernel walks each cell's own window, suppresses
    /// non-maxima against the same zero rim `suppress_non_maxima` sees, applies
    /// `safe_radius` and the edge margin, and reduces the survivors under the
    /// host's own total order. What comes back is 361 x 4 B on the 960x960 index
    /// rig. The band path stays for the shapes the trait's contract excludes and
    /// for [`CpuCornerScan`](crate::frontend::detect::CpuCornerScan), which is
    /// the reference the equality tests measure this against.
    ///
    /// `out` is left empty — and the frame untouched — when the grid needs more
    /// cubes in one dispatch dimension than a WebGPU implementation must allow.
    fn select_cells(
        &mut self,
        camera: usize,
        image: &ImageU16,
        select: &CellSelect,
        out: &mut Vec<u32>,
    ) -> Result<SelectionStatus, DetectError> {
        out.clear();
        // Spent, not read twice: an entry left behind would answer a later
        // frameset with this one's corners.
        if let Some(workspace) = self.cameras.get_mut(camera) {
            if matches!(std::mem::take(&mut workspace.selection), Selection::Ready(ready) if ready == *select)
            {
                out.extend_from_slice(&workspace.host_keys);
                return Ok(SelectionStatus::Selected);
            }
        }
        guarded(
            GpuError::DeviceLost {
                what: "corner cell selection",
            },
            || {
                let Some((best, cells)) = self.launch_selection(camera, image, select) else {
                    return Ok(SelectionStatus::Unsupported);
                };

                #[cfg(test)]
                super::fire_if_armed(super::CORNER_SCAN_READ);

                let reads: Vec<cubecl::bytes::Bytes> = super::read_blocking(
                    &self.client,
                    vec![best],
                    "the cell winner keys",
                    &super::seam::READ_DETECT,
                )?;
                let Ok([keys]) = <[cubecl::bytes::Bytes; 1]>::try_from(reads) else {
                    return Err(super::GpuError::DeviceReadFailed {
                        what: "the cell winner buffer",
                    }
                    .into());
                };
                // Copied rather than held, unlike the candidate image: 1.4 kB
                // into a buffer the detector owns and reuses, against a
                // `Bytes` the next frame would replace anyway.
                out.extend_from_slice(checked_keys(&keys, cells)?);
                Ok(SelectionStatus::Selected)
            },
        )
    }

    /// The named cameras' selections launched together and downloaded by
    /// nobody: the handles go on the relay, and whichever stage reads next
    /// carries them (D78).
    ///
    /// On this lane the wait is what a frameset pays for, not the 1.4 kB each
    /// camera brings back, so this stage's whole job is to be launched early
    /// enough that another stage's read can absorb it.
    fn submit_cells(
        &mut self,
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
    ) -> Result<(), DetectError> {
        self.abort_selection();
        self.cameras
            .resize_with(images.len(), CameraWorkspace::default);
        let outcome = guarded(
            GpuError::DeviceLost {
                what: "corner cell selection",
            },
            || {
                let mut handles = Vec::new();
                for (camera, image) in images.iter().enumerate() {
                    let Some(select) = selects.get(camera).copied().flatten() else {
                        continue;
                    };
                    if let Some((best, _)) = self.launch_selection(camera, image, &select) {
                        self.cameras[camera].selection = Selection::Pending(select);
                        handles.push(best);
                        #[cfg(test)]
                        super::fire_if_armed("selection camera submitted");
                    }
                }
                self.reads.stage(handles);
                Ok(())
            },
        );
        if outcome.is_err() {
            self.abort_selection();
        }
        outcome
    }

    fn take_cells(&mut self) -> Result<(), DetectError> {
        let outcome = guarded(
            GpuError::DeviceLost {
                what: "corner cell selection",
            },
            || {
                use super::selection_batch::TransferResult;
                let reads = match self.reads.take() {
                    TransferResult::Empty => return Ok(()),
                    TransferResult::Ready(bytes) => bytes,
                    TransferResult::Pending(handles)
                        if handles.is_empty()
                            && !self.cameras.iter().any(|camera| {
                                matches!(camera.selection, Selection::Pending(_))
                            }) =>
                    {
                        return Ok(());
                    }
                    TransferResult::Pending(handles) => {
                        #[cfg(test)]
                        super::fire_if_armed(super::CORNER_SCAN_READ);
                        super::read_blocking(
                            &self.client,
                            handles,
                            "the cell winner keys",
                            &super::seam::READ_DETECT,
                        )?
                    }
                };
                let pending = self
                    .cameras
                    .iter()
                    .filter(|camera| matches!(camera.selection, Selection::Pending(_)));
                if pending.clone().count() != reads.len() {
                    return Err(super::GpuError::DeviceReadFailed {
                        what: "the cell winner buffers",
                    }
                    .into());
                }
                // Validate the whole batch before publishing any camera as ready.
                for (camera, keys) in pending.zip(&reads) {
                    checked_keys(keys, camera.keys.as_ref().map_or(0, |(_, cells)| *cells))?;
                }
                for (camera, keys) in self
                    .cameras
                    .iter_mut()
                    .filter(|camera| matches!(camera.selection, Selection::Pending(_)))
                    .zip(&reads)
                {
                    camera.host_keys.clear();
                    camera.host_keys.extend_from_slice(u32::from_bytes(keys));
                    if let Selection::Pending(select) = camera.selection {
                        camera.selection = Selection::Ready(select);
                    }
                }
                Ok(())
            },
        );
        if outcome.is_err() {
            self.abort_selection();
        }
        outcome
    }

    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], DetectError> {
        // The same refusal the CPU lane returns, and asked in the same place: a
        // band before a scan is a programming error, not an empty frame.
        let (Some(kept), Some(mask)) = (self.kept.as_ref(), self.mask.as_ref()) else {
            return Err(DetectError::NotScanned);
        };
        // `row_start = rows.start.max(margin)`, `row_end = rows.end.min(height -
        // margin)` (`fast.rs).
        let first: usize = request.y.max(FAST_BORDER);
        let last: usize = (request.y + request.rows).min(self.height.saturating_sub(FAST_BORDER));
        let (width, stride): (usize, usize) = (self.width, self.words);
        let threshold: i32 = request.threshold;
        Ok(self
            .bands
            .get_or_insert_with(request.row, request.rung, || {
                let mut corners: Vec<FastCorner> = Vec::new();
                // A threshold at or over 255 admits nothing: the score is a `u8`.
                if let Ok(bound) = u8::try_from(threshold.max(0)) {
                    let scores: &[u8] = kept;
                    let bits: &[u32] = u32::from_bytes(mask);
                    for row in first..last {
                        filter_row(scores, bits, width, stride, row, bound, &mut corners);
                    }
                }
                corners
            }))
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

#[cfg(test)]
mod tests;
