//! The GPU [`CornerScan`]: FAST-9 scores and kornia's local-maximum filter on
//! the device, one download per frame, every rung of the ladder derived from it.

mod batch;
pub use batch::CornerLaunch;

use cubecl::prelude::*;
use kornia_imgproc::features::FastCorner;

use crate::kernels::{self, MASK_BITS, RING_BIAS};
use crate::pyramid::Level0;
use crate::runtime::{guarded, GpuError};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::features::backend::{
    block_filter_end, decode_cell_key, BandCache, FAST_RING_COLUMN, FAST_RING_ROW,
};
use kornia_staging_imgproc::features::SelectionStatus;
use kornia_staging_imgproc::features::{
    opencv_corner_score, BandRequest, CellSelect, CenteredCellError, CornerScan, FAST_BORDER,
};
use std::mem::size_of;

/// Failure to scan an image or collect its cell selections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ScanError {
    /// A published camera selection must be consumed before a new batch begins.
    #[error("camera {camera} has an unread selection")]
    UnreadSelection {
        /// Camera whose selection is still pending consumption.
        camera: usize,
    },
    /// The delivered record belongs to a cancelled or replaced selection batch.
    #[error("selection batch was cancelled before read delivery")]
    SelectionCancelled,
    /// No selection batch was begun.
    #[error("begin a cell-selection batch before submitting cameras")]
    BatchNotBegun,
    /// Camera submissions must be unique, ascending and inside the batch.
    #[error("camera {camera} is out of range or out of order for this batch")]
    InvalidBatchCamera {
        /// Rejected index.
        camera: usize,
    },
    /// Input and scanner camera counts differ.
    #[error("batch has {inputs} inputs but {outputs} camera slots")]
    BatchSizeMismatch {
        /// Input cameras.
        inputs: usize,
        /// Scanner slots.
        outputs: usize,
    },
    /// Device failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// Invalid cell/band request.
    #[error(transparent)]
    Detect(#[from] CenteredCellError),
    /// No published device view matches this input.
    #[error("camera {camera} has no uploaded {width}x{height} image")]
    MissingUpload {
        /// Camera index.
        camera: usize,
        /// Requested width.
        width: usize,
        /// Requested height.
        height: usize,
    },
}

/// Dense host pixels or a published level-zero view of the requested geometry.
#[derive(Clone, Copy)]
pub enum ScanInput<'a> {
    /// Dense host pixels; an existing device view is reused when available.
    Dense(&'a Image<u16, 1>),
    /// A view previously supplied through `GpuCornerScan::use_level0`.
    Uploaded(ImageSize),
}
impl ScanInput<'_> {
    fn size(self) -> ImageSize {
        match self {
            Self::Dense(image) => image.size(),
            Self::Uploaded(size) => size,
        }
    }
}

/// Ordered cameras and policy, retained with their readback layout through delivery.
#[derive(Default)]
pub struct PendingSelection {
    generation: u64,
    entries: Vec<(usize, CellSelect)>,
    layout: SelectionLayout,
    handles: Vec<cubecl::server::Handle>,
}
#[derive(Default)]
enum SelectionLayout {
    #[default]
    PerCamera,
    Packed {
        stride: usize,
    },
}
impl PendingSelection {
    /// Remove handles for a caller-owned combined read. Keep this record and
    /// return it with the resulting bytes to `GpuCornerScan::deliver`.
    pub fn take_handles(&mut self) -> Vec<cubecl::server::Handle> {
        std::mem::take(&mut self.handles)
    }
}
#[derive(Default)]
enum SelectionReads {
    #[default]
    Empty,
    Pending(PendingSelection),
    Ready {
        selection: PendingSelection,
        bytes: Vec<cubecl::bytes::Bytes>,
    },
}

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
struct CameraWorkspace {
    buffers: Option<ScanBuffers>,
    keys: Option<(cubecl::server::Handle, usize)>,
    host_keys: Vec<u32>,
    selection: Option<CellSelect>,
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
/// test *is* `corner_score_9 > threshold`; the staged
/// `the_gpu_corner_scan_is_exact_against_kornia` test checks every rung.
///
/// The bitmask skips columns without candidates during repeated band queries.
pub struct GpuCornerScan<R: Runtime> {
    client: ComputeClient<R>,
    /// The biased ring, uploaded once.
    ring: cubecl::server::Handle,
    /// Level 0 of each camera's pyramid, as the builder published it. When the
    /// entry for the camera being scanned matches the frame's geometry the
    /// score kernel reads it and this stage uploads nothing at all; otherwise
    /// — a scanner with no builder beside it, which is how the tolerance tests
    /// drive it — the frame goes up here.
    level0: Vec<Option<Level0>>,
    /// Frames this scanner has uploaded itself, which the shared level-0 path
    /// is meant to keep at zero.
    uploads: usize,
    /// Reusable allocations and explicit selection readiness, indexed by camera.
    cameras: Vec<CameraWorkspace>,
    batch: Option<batch::BatchScanBuffers>,
    reads: SelectionReads,
    selection_generation: u64,
    /// Times the three device buffers have been allocated, which a rig of one
    /// geometry keeps at one.
    buffer_allocations: usize,
    /// The candidate image, one byte per pixel, as it came back — the device
    /// read's own buffer, not a copy of it.
    ///
    /// Owned readback bytes avoid another host copy. Empty before the first scan.
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
    /// ```no_run
    /// # #[cfg(feature = "wgpu")] {
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_staging_gpu::{features::GpuCornerScan, runtime::gpu_client};
    /// use kornia_staging_imgproc::features::CornerScan;
    /// let image = Image::from_size_val(ImageSize { width: 64, height: 64 }, 0u16)?;
    /// let mut scan = GpuCornerScan::new(gpu_client()?)?;
    /// scan.scan(0, &image)?;
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Arguments
    /// * `client` - Device client shared by this scanner and any pyramid builder.
    ///
    /// Fallible because the upload is a device operation like any other:
    /// `create_from_slice` **panics** inside CubeCL's own client when the worker
    /// submission fails, so a constructor with no error channel is a path from a
    /// dying device to an unwind through the caller.
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
                    ring: crate::transfer::upload_inner(&client, u32::as_bytes(&ring)),
                    level0: Vec::new(),
                    uploads: 0,
                    cameras: Vec::new(),
                    batch: None,
                    reads: SelectionReads::Empty,
                    selection_generation: 0,
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

    /// Take the builder's published level-zero views after building the frame.
    /// Both stages must use the same client. Reuse the two table allocations.
    pub fn use_level0(&mut self, builder: &mut crate::pyramid::GpuPyramidBuilder<R>) {
        builder.take_level0(&mut self.level0);
    }

    /// Whether a published level-zero view matches the camera and input size.
    pub fn has_level0(&self, camera: usize, size: ImageSize) -> bool {
        self.level0
            .get(camera)
            .and_then(Option::as_ref)
            .is_some_and(|view| view.width == size.width && view.height == size.height)
    }

    /// Pending downloads, in camera order, for a caller-owned readback batch.
    pub fn staged_handles(&self) -> Option<&[cubecl::server::Handle]> {
        match &self.reads {
            SelectionReads::Pending(selection) => Some(&selection.handles),
            _ => None,
        }
    }

    /// Transfer pending downloads to a caller that will supply them with `deliver`.
    pub fn take_staged(&mut self) -> Option<PendingSelection> {
        match std::mem::take(&mut self.reads) {
            SelectionReads::Pending(selection) => Some(selection),
            other => {
                self.reads = other;
                None
            }
        }
    }

    /// Return the owned selection record with downloads in its handle order.
    /// `take_cells` validates the entire record before publishing any camera.
    ///
    /// # Errors
    /// Returns [`ScanError::SelectionCancelled`] for a cancelled or replaced batch,
    /// leaving the current batch unchanged.
    pub fn deliver(
        &mut self,
        selection: PendingSelection,
        bytes: Vec<cubecl::bytes::Bytes>,
    ) -> Result<(), ScanError> {
        if selection.generation != self.selection_generation {
            return Err(ScanError::SelectionCancelled);
        }
        self.reads = SelectionReads::Ready { selection, bytes };
        Ok(())
    }

    /// Discard downloaded bands after a failed or cancelled scan.
    pub fn abort_scan(&mut self) {
        self.bands.clear();
        self.kept = None;
        self.mask = None;
    }

    /// Discard partial selections after a failed or cancelled batch.
    pub fn abort_selection(&mut self) {
        self.selection_generation = self.selection_generation.wrapping_add(1);
        self.reads = SelectionReads::Empty;
        for camera in &mut self.cameras {
            camera.selection = None;
        }
    }

    /// Frames this scanner uploaded itself. Zero once
    /// [`GpuCornerScan::use_level0`] is wired to a builder that runs first.
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
    fn frame(
        &mut self,
        camera: usize,
        image: ScanInput<'_>,
    ) -> Result<(cubecl::server::Handle, usize), ScanError> {
        // `image` is the authority on the geometry, not `self.width`/`self.height`:
        // those are the same frame's, set by `scan` two lines up, and one fact
        // with two sources inside one call is how they come apart.
        let (width, height): (usize, usize) = (image.size().width, image.size().height);
        let pixels: usize = width * height;
        let shared = self
            .level0
            .get(camera)
            .cloned()
            .flatten()
            .filter(|level0| level0.width == width && level0.height == height);
        if let Some(level0) = shared {
            return Ok((level0.handle, pixels));
        }

        // The frame goes up as `u16` and the `>> 8` the detector reads happens
        // on the device: the extra 0.9 MB over the bus costs less than a
        // whole-frame narrowing pass on the host.
        match image {
            ScanInput::Dense(image) => {
                self.uploads += 1;
                Ok((
                    crate::transfer::upload_inner(&self.client, u16::as_bytes(image.as_slice())),
                    pixels,
                ))
            }
            ScanInput::Uploaded(_) => Err(ScanError::MissingUpload {
                camera,
                width,
                height,
            }),
        }
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
        image: ScanInput<'_>,
        select: &CellSelect,
    ) -> Result<Option<(cubecl::server::Handle, usize)>, ScanError> {
        // A sparse or mixed-grid selection uses the immediate per-camera path.
        // Its level-zero input may still be in the frame's deferred dispatches.
        let Some(geometry) =
            kernels::CellSelectGeometry::new(image.size().width, image.size().height, select)
        else {
            return Ok(None);
        };
        let cells = geometry.cells_x * geometry.cells_y;
        let fused = kernels::uses_cell_kernel(select.grid.cell, image.size().width, &self.client);
        let handles = if fused {
            None
        } else {
            Some(self.candidates(camera, image)?)
        };

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
                slot.insert((self.client.empty(cells * size_of::<u32>()), cells))
                    .clone()
            }
        };

        if let Some(handles) = handles {
            kernels::launch_fast_cell_select::<R>(
                &self.client,
                (&handles.kept, handles.pixels),
                (&best, best_len),
                geometry,
            );
        } else {
            let (frame, pixels) = self.frame(camera, image)?;
            kernels::launch_fast_cell::<R>(
                &self.client,
                (&frame, pixels),
                (&best, best_len),
                geometry,
            );
        }
        Ok(Some((best, cells)))
    }

    /// Level 0 in, the candidate image out: the two kernels both entry points
    /// run, this camera's buffers sized and reused.
    ///
    /// The previous frame's readback is dropped **first**, and the geometry is
    /// recorded before anything can fail, so a scan that dies further on leaves
    /// a scanner that refuses a band rather than one that answers with the last
    /// frame's corners under this frame's width.
    fn candidates(
        &mut self,
        camera: usize,
        image: ScanInput<'_>,
    ) -> Result<ScanHandles, ScanError> {
        self.bands.clear();
        self.kept = None;
        self.mask = None;
        self.width = image.size().width;
        self.height = image.size().height;
        let pixels: usize = self.width * self.height;
        self.words = self.width.div_ceil(MASK_BITS);
        let mask_len: usize = self.words * self.height;

        let (handle, handle_len): (cubecl::server::Handle, usize) = self.frame(camera, image)?;
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
        Ok(ScanHandles {
            kept,
            mask,
            pixels,
            mask_len,
        })
    }
}

/// A downloaded key buffer as `u32`, refused when it is not the grid's length.
fn checked_keys(keys: &[u8], cells: usize) -> Result<&[u32], ScanError> {
    let expected: usize = cells * size_of::<u32>();
    if keys.len() != expected {
        return Err(GpuError::ShortRead {
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

impl<R: Runtime> GpuCornerScan<R> {
    /// Scan all FAST bands of one dense or uploaded camera image.
    ///
    /// # Arguments
    /// * `camera` - Index in the published pyramid batch.
    /// * `image` - Dense pixels or the geometry of a published upload.
    ///
    /// # Errors
    /// Rejects missing uploads, invalid geometry, malformed reads and device failures.
    pub fn scan_input(&mut self, camera: usize, image: ScanInput<'_>) -> Result<(), ScanError> {
        guarded(
            GpuError::DeviceLost {
                what: "corner scan",
            },
            || {
                let handles: ScanHandles = self.candidates(camera, image)?;
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
                #[cfg(all(test, feature = "wgpu"))]
                fire_if_armed(CORNER_SCAN_READ);

                // One read for both, so one synchronisation for the frame.
                let reads: Vec<cubecl::bytes::Bytes> = crate::transfer::read_inner(
                    &self.client,
                    vec![handles.kept, handles.mask],
                    "the candidate image and its bitmask",
                    || Ok(()),
                )?;
                // One buffer per handle, in the order they were asked for; anything else
                // is the runtime breaking its own contract rather than short data.
                let Ok([kept_bytes, mask_bytes]) = <[cubecl::bytes::Bytes; 2]>::try_from(reads)
                else {
                    return Err(GpuError::DeviceReadFailed {
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
                        return Err(GpuError::ShortRead {
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
    /// Select the highest-scoring eligible corner per cell in stable row-major
    /// tie order, transferring only packed winner keys. Unsupported geometries
    /// retain the band-selection path.
    ///
    /// # Errors
    /// Rejects missing uploads, malformed reads and device failures.
    ///
    /// `out` is left empty — and the frame untouched — when the grid needs more
    /// cubes in one dispatch dimension than a WebGPU implementation must allow.
    pub fn select_input(
        &mut self,
        camera: usize,
        image: ScanInput<'_>,
        select: &CellSelect,
        out: &mut Vec<Option<FastCorner>>,
    ) -> Result<SelectionStatus, ScanError> {
        out.clear();
        if !select.supports(image.size().width, image.size().height) {
            return Ok(SelectionStatus::Unsupported);
        }
        // Spent, not read twice: an entry left behind would answer a later
        // frameset with this one's corners.
        if let Some(workspace) = self.cameras.get_mut(camera) {
            if matches!(workspace.selection.take(), Some(ready) if ready == *select) {
                out.extend(workspace.host_keys.iter().copied().map(decode_cell_key));
                return Ok(SelectionStatus::Selected);
            }
        }
        guarded(
            GpuError::DeviceLost {
                what: "corner cell selection",
            },
            || {
                let Some((best, cells)) = self.launch_selection(camera, image, select)? else {
                    return Ok(SelectionStatus::Unsupported);
                };

                #[cfg(all(test, feature = "wgpu"))]
                fire_if_armed(CORNER_SCAN_READ);

                let reads: Vec<cubecl::bytes::Bytes> = crate::transfer::read_inner(
                    &self.client,
                    vec![best],
                    "the cell winner keys",
                    || Ok(()),
                )?;
                let Ok([keys]) = <[cubecl::bytes::Bytes; 1]>::try_from(reads) else {
                    return Err(GpuError::DeviceReadFailed {
                        what: "the cell winner buffer",
                    }
                    .into());
                };
                // Copied rather than held, unlike the candidate image: 1.4 kB
                // into a buffer the detector owns and reuses, against a
                // `Bytes` the next frame would replace anyway.
                out.extend(
                    checked_keys(&keys, cells)?
                        .iter()
                        .copied()
                        .map(decode_cell_key),
                );
                Ok(SelectionStatus::Selected)
            },
        )
    }

    /// Reset selection state for a batch of `cameras` inputs.
    /// # Errors
    /// Refuses to discard an unread camera selection.
    pub fn begin_cells(&mut self, cameras: usize) -> Result<(), ScanError> {
        if let Some(camera) = self
            .cameras
            .iter()
            .position(|workspace| workspace.selection.is_some())
        {
            return Err(ScanError::UnreadSelection { camera });
        }
        self.abort_selection();
        self.cameras.resize_with(cameras, CameraWorkspace::default);
        self.reads = SelectionReads::Pending(PendingSelection {
            generation: self.selection_generation,
            ..PendingSelection::default()
        });
        Ok(())
    }

    /// Submit one camera after `begin_cells`, retaining its readback handle.
    /// Indices must be unique, ascending and below the begun camera count.
    ///
    /// # Errors
    /// Returns a typed device or input failure without publishing ready results.
    pub fn submit_input(
        &mut self,
        camera: usize,
        image: ScanInput<'_>,
        select: &CellSelect,
    ) -> Result<(), ScanError> {
        let outcome = guarded(
            GpuError::DeviceLost {
                what: "corner cell selection",
            },
            || {
                if !matches!(self.reads, SelectionReads::Pending(_)) {
                    return Err(ScanError::BatchNotBegun);
                }
                if camera >= self.cameras.len()
                    || matches!(&self.reads, SelectionReads::Pending(pending) if pending.entries.last().is_some_and(|(last, _)| *last >= camera))
                {
                    return Err(ScanError::InvalidBatchCamera { camera });
                }
                if let Some((best, _)) = self.launch_selection(camera, image, select)? {
                    if let SelectionReads::Pending(pending) = &mut self.reads {
                        pending.entries.push((camera, *select));
                        pending.handles.push(best);
                    }
                    #[cfg(all(test, feature = "wgpu"))]
                    fire_if_armed("selection camera submitted");
                }
                Ok(())
            },
        );
        if outcome.is_err() {
            self.abort_selection();
        }
        outcome
    }

    /// Prepare a camera batch, dispatching packed work or submitting individual inputs.
    ///
    /// `submit` supplies each fallback input lazily; `run` schedules the owned
    /// packed launch on this scanner's client stream before its buffers are reused.
    ///
    /// # Errors
    /// Returns input, scheduling or device failures and invalidates partial results.
    pub fn submit_cells<E: From<ScanError>>(
        &mut self,
        sizes: impl ExactSizeIterator<Item = ImageSize> + Clone,
        selects: &[Option<CellSelect>],
        mut submit: impl FnMut(&mut Self, usize, &CellSelect) -> Result<(), E>,
        run: impl FnOnce(CornerLaunch) -> Result<(), E>,
    ) -> Result<(), E> {
        let cameras = sizes.len();
        self.begin_cells(cameras)?;
        let outcome = (|| {
            if let Some(work) = self.prepare_batch(sizes, selects)? {
                run(work)?;
            } else {
                for camera in 0..cameras {
                    if let Some(select) = selects.get(camera).copied().flatten() {
                        submit(self, camera, &select)?;
                    }
                }
            }
            Ok(())
        })();
        if outcome.is_err() {
            self.abort_selection();
        }
        outcome
    }

    /// Download pending cell keys, checking the whole batch before publication.
    ///
    /// # Errors
    /// Rejects malformed reads and device failures; invalidates the entire batch.
    pub fn take_cells(&mut self) -> Result<(), ScanError> {
        let outcome = guarded(
            GpuError::DeviceLost {
                what: "corner cell selection",
            },
            || {
                let (selection, reads) = match std::mem::take(&mut self.reads) {
                    SelectionReads::Empty => return Ok(()),
                    SelectionReads::Ready { selection, bytes } => (selection, bytes),
                    SelectionReads::Pending(mut selection) => {
                        if selection.entries.is_empty() {
                            return Ok(());
                        }
                        #[cfg(all(test, feature = "wgpu"))]
                        fire_if_armed(CORNER_SCAN_READ);
                        let bytes = crate::transfer::read_inner(
                            &self.client,
                            selection.take_handles(),
                            "the cell winner keys",
                            || Ok(()),
                        )?;
                        (selection, bytes)
                    }
                };
                let cells = |select: &CellSelect| {
                    let (x, y) = select.grid.dimensions();
                    x * y
                };
                let keys: Vec<&[u8]> = match selection.layout {
                    SelectionLayout::Packed { stride } => {
                        if reads.len() != 1 || reads[0].len() != selection.entries.len() * stride {
                            return Err(GpuError::DeviceReadFailed {
                                what: "the packed cell winner buffer",
                            }
                            .into());
                        }
                        selection
                            .entries
                            .iter()
                            .enumerate()
                            .map(|(index, (_, select))| {
                                reads[0]
                                    .get(
                                        index * stride
                                            ..index * stride + cells(select) * size_of::<u32>(),
                                    )
                                    .ok_or(GpuError::DeviceReadFailed {
                                        what: "the packed cell winner extent",
                                    })
                            })
                            .collect::<Result<_, _>>()?
                    }
                    SelectionLayout::PerCamera => {
                        reads.iter().map(|bytes| bytes.as_ref()).collect()
                    }
                };
                if selection.entries.len() != keys.len() {
                    return Err(GpuError::DeviceReadFailed {
                        what: "the cell winner buffers",
                    }
                    .into());
                }
                // Validate every extent and key before publishing any result.
                for ((camera, select), keys) in selection.entries.iter().zip(&keys) {
                    if *camera >= self.cameras.len() {
                        return Err(ScanError::InvalidBatchCamera { camera: *camera });
                    }
                    checked_keys(keys, cells(select))?;
                }
                for ((camera, select), keys) in selection.entries.into_iter().zip(keys) {
                    let workspace = &mut self.cameras[camera];
                    workspace.host_keys.clear();
                    workspace.host_keys.extend_from_slice(u32::from_bytes(keys));
                    workspace.selection = Some(select);
                }
                Ok(())
            },
        );
        if outcome.is_err() {
            self.abort_selection();
        }
        outcome
    }
}
impl<R: Runtime> CornerScan for GpuCornerScan<R> {
    type Error = ScanError;
    fn scan(&mut self, camera: usize, image: &Image<u16, 1>) -> Result<(), Self::Error> {
        self.scan_input(camera, ScanInput::Dense(image))
    }
    fn select_cells(
        &mut self,
        camera: usize,
        image: &Image<u16, 1>,
        select: &CellSelect,
        _eligibility: Option<(&kornia_staging_imgproc::features::Occupancy<'_>, &[bool])>,
        out: &mut Vec<Option<FastCorner>>,
    ) -> Result<SelectionStatus, Self::Error> {
        self.select_input(camera, ScanInput::Dense(image), select, out)
    }
    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], ScanError> {
        // The same refusal the CPU lane returns, and asked in the same place: a
        // band before a scan is a programming error, not an empty frame.
        let (Some(kept), Some(mask)) = (self.kept.as_ref(), self.mask.as_ref()) else {
            return Err(CenteredCellError::NotScanned.into());
        };
        // `row_start = rows.start.max(margin)`, `row_end = rows.end.min(height -
        // margin)` (`fast.rs`).
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

#[cfg(all(test, feature = "wgpu"))]
mod tests;

#[cfg(all(test, feature = "wgpu"))]
use crate::fault::{arm as arm_fault_at, fire as fire_if_armed};
#[cfg(all(test, feature = "wgpu"))]
const CORNER_SCAN_READ: &str = "the corner scan's read";
