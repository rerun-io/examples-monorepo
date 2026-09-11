//! CPU band scans, candidate caching, and host suppression.
use super::{BandRequest, CornerScan, DetectError, FastCorner};
use crate::image::ImageU16;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::{Rect as KorniaRect, fast_detect_rect_u8};

/// The Bresenham radius `cv::FAST` and kornia both skip at a border
/// (`cells.rs, `fast.rs).
pub const FAST_BORDER: usize = 3;

/// kornia's Bresenham ring: ring point `k` sits `FAST_RING_ROW[k]` rows and
/// [`FAST_RING_COLUMN`]`[k]` columns from the centre (`fast.rs).
pub const FAST_RING_ROW: [i32; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
/// The column half of [`FAST_RING_ROW`]'s ring.
pub const FAST_RING_COLUMN: [i32; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

/// Lanes in one block of kornia's in-block local-maximum filter (`fast.rs).
pub const FAST_FILTER_LANES: usize = 16;

/// The width at which kornia turns that filter on (`fast.rs).
const FAST_FILTER_WIDTH: usize = 800;

/// Where kornia's local-maximum filter stops, and whether it runs at all.
///
/// The filter keeps a candidate only when its score beats both neighbours
/// *inside its own sixteen-lane block*, the blocks are aligned to the image's
/// own left margin, and the scalar tail past the last whole block is
/// unfiltered — kornia's SIMD loop runs while `x + 16 <= width - margin`
/// (`fast.rs). So the alignment and the tail are part of the corner set,
/// not an implementation detail, and this is the one place that arithmetic
/// lives: the CPU sweep gets it from kornia itself, and the GPU kernel and
/// `tests/fast_model.rs` read it here rather than each spelling it out.
pub fn block_filter_end(width: usize) -> (usize, bool) {
    let blocks: usize = width.saturating_sub(2 * FAST_BORDER) / FAST_FILTER_LANES;
    (
        FAST_BORDER + blocks * FAST_FILTER_LANES,
        width >= FAST_FILTER_WIDTH,
    )
}

/// `cv::FAST`'s default segment length, `FastFeatureDetector::TYPE_9_16`.
const FAST_ARC_LENGTH: usize = 9;

/// One cell row's raw FAST candidates at one threshold, over the whole image width.
///
/// `fast_detect_rect_u8` detects over **whole rows** and filters the result by
/// column (`cells.rs), so the scan a cell asks for depends only on its
/// row band and its threshold — the nineteen cells of one grid row at one rung
/// all pay for the same 960-wide sweep. Scanning it once and filtering each
/// cell's columns out of it is the same call with the same arguments, so the
/// candidates and their order are the ones the per-cell calls produced.
///
/// The whole-image width is what the scan has to keep: kornia's kernel turns its
/// in-block local-maximum filter on at `width >= 800` and aligns its sixteen-lane
/// blocks to the image's own left margin (`fast.rs), so a cell-sized
/// copy would detect a different set.
///
/// Both scanners hold one of these, which is why it lives here beside the trait
/// that documents the key rather than twice in the two backends.
#[derive(Debug, Default)]
pub(crate) struct BandCache {
    /// `slots[rung][row]`, `None` until that band has been produced.
    ///
    /// Indexed rather than searched: `detect_keypoints_with_cells` asks 361
    /// cells x up to 5 rungs per camera per frameset, and a linear scan of a
    /// list that grows to `rows x rungs` cost about 86,000 comparisons for
    /// nothing on a 960x960 frame.
    slots: Vec<Vec<Option<Vec<FastCorner>>>>,
}

impl BandCache {
    /// Drop every band, keeping the two outer allocations.
    pub(crate) fn clear(&mut self) {
        for rows in &mut self.slots {
            for slot in rows.iter_mut() {
                *slot = None;
            }
        }
    }

    /// Bands currently held, for `Debug`.
    pub(crate) fn len(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|slot| slot.is_some())
            .count()
    }

    /// The band at `(row, rung)`, produced by `scan` the first time it is asked
    /// for and read out of the cache afterwards.
    pub(crate) fn get_or_insert_with(
        &mut self,
        row: usize,
        rung: usize,
        scan: impl FnOnce() -> Vec<FastCorner>,
    ) -> &[FastCorner] {
        if self.slots.len() <= rung {
            self.slots.resize_with(rung + 1, Vec::new);
        }
        let rows: &mut Vec<Option<Vec<FastCorner>>> = &mut self.slots[rung];
        if rows.len() <= row {
            rows.resize_with(row + 1, || None);
        }
        rows[row].get_or_insert_with(scan)
    }
}

/// CPU [`CornerScan`]: one kornia rectangle sweep per row band and threshold.
/// Narrow the u16 frame to u8 once per frame; all cells then use zero-copy rectangles.
/// Rebuilding per band copied the whole frame repeatedly and measured 40.1 ms
/// against 9.4 ms on MIO07/1500. Write directly into kornia's image storage to
/// avoid a second full-frame copy; reallocate only when geometry changes.
#[derive(Default)]
pub struct CpuCornerScan {
    /// The narrowed frame, `None` until the first [`CornerScan::scan`], and
    /// reallocated only for a new geometry.
    gray: Option<Image<u8, 1>>,
    width: usize,
    height: usize,
    /// The [`Band`]s this frame has already scanned, in the order they were
    /// first asked for.
    bands: BandCache,
}

/// `kornia_image::Image` is not `Debug`, so the geometry is what this prints.
impl std::fmt::Debug for CpuCornerScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuCornerScan")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bands", &self.bands.len())
            .finish()
    }
}

impl CornerScan for CpuCornerScan {
    /// `_camera` is unused: the caller holds the frame and nothing here is
    /// shared between cameras.
    fn scan(&mut self, _camera: usize, image: &ImageU16) -> Result<(), DetectError> {
        // The bands are this image's; the previous frame's are stale.
        self.bands.clear();
        let (width, height): (usize, usize) = (image.width(), image.height());
        self.width = width;
        self.height = height;
        let fits: bool = self
            .gray
            .as_ref()
            .is_some_and(|gray| gray.width() == width && gray.height() == height);
        let gray: &mut Image<u8, 1> = match &mut self.gray {
            Some(existing) if fits => existing,
            slot => slot.insert(
                Image::from_size_val(ImageSize { width, height }, 0u8).map_err(|_| {
                    DetectError::GrayViewRefused {
                        width,
                        height,
                        actual: 0,
                    }
                })?,
            ),
        };
        // `sub_ptr[x] = (sub_img_raw(x, y) >> 8)`, once,
        // straight into the pixels the detector reads. One row's slice at a
        // time, not one `push` per pixel: the capacity check a `push` carries is
        // what stops the narrowing from vectorising.
        let pixels: &mut [u8] = gray.as_slice_mut();
        if image.stride() == width {
            // One pass over the whole frame when it is unstrided, which every
            // frame from the port's own decode path is: a single long loop
            // vectorises where 960 short ones each pay their own prologue.
            for (narrowed, wide) in pixels.iter_mut().zip(&image.data()[..width * height]) {
                *narrowed = (*wide >> 8) as u8;
            }
        } else {
            for y in 0..height {
                let row: &mut [u8] = &mut pixels[y * width..(y + 1) * width];
                for (narrowed, wide) in row.iter_mut().zip(image.row(y)) {
                    *narrowed = (*wide >> 8) as u8;
                }
            }
        }
        Ok(())
    }

    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], DetectError> {
        let Some(gray) = self.gray.as_ref() else {
            return Err(DetectError::NotScanned);
        };
        let width: usize = self.width;
        Ok(self
            .bands
            .get_or_insert_with(request.row, request.rung, || {
                fast_detect_rect_u8(
                    gray,
                    KorniaRect {
                        x: 0,
                        y: request.y,
                        w: width,
                        h: request.rows,
                    },
                    request.threshold as f32,
                    FAST_ARC_LENGTH,
                    FAST_BORDER,
                )
                .into_iter()
                .map(|corner| FastCorner {
                    xy: corner.xy,
                    response: opencv_corner_score(corner.response),
                })
                .collect()
            }))
    }
}

/// Convert normalized kornia FAST scores to OpenCV integer corner scores.
/// Kornia reports the first threshold that rejects the corner; OpenCV reports
/// the last that accepts it, one less. Convert before suppression so comparison
/// against a non-candidate's zero uses the same score convention.
/// Candidates at threshold `t` score at least `t`, and the ladder has a positive floor.
#[inline]
pub fn opencv_corner_score(normalized: f32) -> f32 {
    (normalized * 255.0).round() - 1.0
}

/// OpenCV's `cv::FAST` non-maximum suppression, over one cell's candidates.
///
/// `scores` is a `side` x `side` scratch grid in the cell's own coordinates,
/// zero everywhere there is no candidate — which is exactly what OpenCV's row
/// ring holds, since it `memset`s each row and only fills the columns it tested.
/// A candidate survives when its score is **strictly greater** than all eight
/// neighbours'. Two neighbouring candidates with equal scores therefore both
/// die; that is OpenCV's rule, not an approximation of it.
///
/// `corners` is filtered in place and left in its original scan order. The grid
/// is left zeroed, so only the entries this cell wrote are ever touched and the
/// cost is linear in the candidate count rather than in the cell's area.
pub(super) fn suppress_non_maxima(
    corners: &mut Vec<FastCorner>,
    scores: &mut Vec<f32>,
    keep: &mut Vec<bool>,
    origin_x: usize,
    origin_y: usize,
    side: usize,
) {
    if scores.len() < side * side {
        scores.resize(side * side, 0.0);
    }
    let local = |corner: &FastCorner| -> (usize, usize) {
        (
            corner.xy[0] as usize - origin_x,
            corner.xy[1] as usize - origin_y,
        )
    };

    for corner in corners.iter() {
        let (lx, ly) = local(corner);
        scores[ly * side + lx] = corner.response;
    }

    keep.clear();
    keep.reserve(corners.len());
    for corner in corners.iter() {
        let (lx, ly) = local(corner);
        let score: f32 = corner.response;
        let mut peak: bool = true;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx: i32 = lx as i32 + dx;
                let ny: i32 = ly as i32 + dy;
                let neighbour: f32 = if nx < 0 || ny < 0 || nx >= side as i32 || ny >= side as i32 {
                    0.0
                } else {
                    scores[ny as usize * side + nx as usize]
                };
                peak &= score > neighbour;
            }
        }
        keep.push(peak);
    }

    for corner in corners.iter() {
        let (lx, ly) = local(corner);
        scores[ly * side + lx] = 0.0;
    }

    let mut index: usize = 0;
    corners.retain(|_| {
        let survives: bool = keep[index];
        index += 1;
        survives
    });
}
