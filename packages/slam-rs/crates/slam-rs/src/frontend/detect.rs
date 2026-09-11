//! Grid-cell FAST detection over a centered grid.
//!
//! Skip occupied cells, walk a halving threshold ladder, then keep the strongest
//! corners that pass masks, safe radius and edge margins. Kornia's rectangle
//! entry point lets this module own the cell geometry. Shrink the candidate
//! rectangle by the three-pixel FAST ring radius on every side.
//!
//! The rectangle scanner scans whole rows, so call it once per row band and
//! threshold and filter columns per cell. Cropping changes kornia's width-dependent
//! local-maximum filter, so it is not interchangeable with this scan.
//!
//! Convert normalized scores to integer corner scores by multiplying by 255 and
//! subtracting one. Suppression requires a score strictly greater than all eight
//! neighbors, with zero for non-candidates. Equal-score plateaus yield no corners.
//! Stable sorting retains scan order among tied scores (D09, trap 2).

mod band;
pub use super::cell::{
    CELL_KEY_LIMIT, CellGrid, CellSelect, DetectorConfig, EDGE_THRESHOLD, LOWEST_THRESHOLD_RUNG,
    MAX_CELLS, Masks, NO_CELL_WINNER, Occupancy, Rect, SelectionStatus, cell_select,
    threshold_rungs,
};
use super::cell::{cell_masks, decode_key};
#[cfg(feature = "gpu-core")]
pub(crate) use band::BandCache;
use band::suppress_non_maxima;
pub use band::{
    CpuCornerScan, FAST_BORDER, FAST_FILTER_LANES, FAST_RING_COLUMN, FAST_RING_ROW,
    block_filter_end, opencv_corner_score,
};

/// kornia's FAST corner, re-exported because [`CornerScan::band`] hands it back:
/// a backend outside this crate cannot implement the trait without naming it.
pub use kornia_imgproc::features::FastCorner;

use crate::image::ImageU16;

/// Detector input failures, including invalid caller-supplied occupancy dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DetectError {
    /// A backend claimed selection but returned a malformed key vector.
    #[error("selection returned {actual} keys, expected {expected}")]
    SelectionLength {
        /// Keys returned by the backend.
        actual: usize,
        /// Cells in the requested grid.
        expected: usize,
    },
    /// The declared occupancy shape does not fit in a `usize`.
    ///
    /// Checked before the multiplication, not after: `rows * columns` wrapping
    /// would have turned an impossible shape into a plausible one.
    #[error("an occupancy shape of {rows}x{columns} does not fit in a usize")]
    OccupancyShapeOverflow {
        /// Rows the shape declares.
        rows: usize,
        /// Columns the shape declares.
        columns: usize,
    },
    /// The occupancy buffer is smaller than the shape it was declared with.
    #[error("occupancy is {actual} cells, the {rows}x{columns} grid needs {expected}")]
    OccupancyTooSmall {
        /// Rows the shape declares.
        rows: usize,
        /// Columns the shape declares.
        columns: usize,
        /// Cells that shape needs.
        expected: usize,
        /// Cells the buffer holds.
        actual: usize,
    },
    /// A GPU scanner's device read failed, or came back the wrong length.
    ///
    /// Only a GPU scanner produces this, and it is one variant rather than
    /// several because an incomplete CubeCL runtime fails every way at once: it
    /// panics on its own worker thread, the launch reports success, and the
    /// download comes back short or as zeros. The [`crate::gpu::GpuError`]
    /// inside names which buffer and whether the read failed or was short;
    /// reading either as a candidate image would quietly detect nothing
    /// (decision D32).
    #[cfg(feature = "gpu-core")]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
    /// [`CornerScan::band`] was asked for corners before [`CornerScan::scan`]
    /// ran on this frame.
    ///
    /// Unreachable: [`detect_keypoints_with_cells`] scans the frame before it
    /// walks a single band. It is a typed error on **both** lanes rather than
    /// an empty band because a detector that reports success and finds nothing
    /// is exactly the failure decision D32 exists to make impossible.
    #[error("a corner band was asked for before the frame was scanned")]
    NotScanned,
    /// The 8-bit view could not be built over the bytes written for it.
    ///
    /// Unreachable: the loop that fills those bytes writes exactly
    /// `width * height` of them. It is a typed error rather than an early `Ok`
    /// because reporting success with no keypoints would turn a future geometry
    /// slip into an empty detector instead of a loud one (decision D32).
    #[error("a {width}x{height} 8-bit view over {actual} bytes was refused")]
    GrayViewRefused {
        /// Row length the view was asked for.
        width: usize,
        /// Row count the view was asked for.
        height: usize,
        /// Bytes offered.
        actual: usize,
    },
}

/// Detected corners and responses in parallel arrays.
/// Buffers are cleared and filled in place, allocating only above the previous
/// high-water mark (`cubecl-portability.md` §12.2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeypointsData {
    /// Corners in cell scan order.
    pub corners: Vec<[f32; 2]>,
    /// `kd.corner_responses`, one per corner.
    pub responses: Vec<f32>,
}

impl KeypointsData {
    /// How many corners were detected.
    pub fn len(&self) -> usize {
        self.corners.len()
    }

    /// Whether nothing was detected.
    pub fn is_empty(&self) -> bool {
        self.corners.is_empty()
    }
}

/// Which band the detector wants, and where the cache keeps it.
///
/// `row` and `rung` are the cell-grid row index and the threshold-ladder rung
/// index — two loop counters [`detect_keypoints_with_cells`] already has — and
/// they are the cache's key, so a lookup is an index and not a search. `y`,
/// `rows` and `threshold` are what producing the band costs; they are a function
/// of the key for as long as one frame's grid stands, which is the invariant
/// [`CornerScan::band`] states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandRequest {
    /// Cell-grid row index, the cache's first key.
    pub row: usize,
    /// Threshold-ladder rung index, the cache's second key.
    pub rung: usize,
    /// First row of the band, in image coordinates.
    pub y: usize,
    /// Rows the band covers.
    pub rows: usize,
    /// FAST threshold the band is scanned at.
    pub threshold: i32,
}

/// The frontend's corner-candidate stage: one frame in, row bands of FAST
/// candidates out.
///
/// The fourth stage seam, alongside the pyramid builder, the patch tracker and
/// the residual accumulator. It is cut here because this is the only place the
/// detector touches an image at all: everything above it —  the cell grid, the
/// threshold ladder, the per-cell budget, the non-maximum suppression, the
/// response ordering and the three filters — is arithmetic on the candidate
/// list, and [`detect_keypoints_with_cells`] keeps all of it.
///
/// `band` is `&mut self` because an implementation caches: the nineteen cells of
/// one grid row at one rung all ask for the same sweep. `Send + Sync` because
/// the PyO3 wrapper holds a whole pipeline in a `pyclass`, which is shared
/// across the interpreter's threads even though nothing here runs on more than
/// one.
pub trait CornerScan: std::fmt::Debug + Send + Sync {
    /// Take camera `camera`'s frame, discarding whatever the last one left.
    ///
    /// Called once per camera per frameset, before any [`CornerScan::band`].
    ///
    /// `camera` is the frame's place in the frameset. It is here for the same
    /// reason it is on [`crate::pyramid::PyramidBuilder::build`]: these are the
    /// only two stages that read level 0, and on a device backend they read the
    /// *same* pixels, so a scanner that knows which camera it was handed can
    /// read the pyramid's own level 0 instead of uploading the frame a second
    /// time. A host backend ignores it.
    ///
    /// # Errors
    ///
    /// [`DetectError`] when the geometry cannot be viewed as 8-bit, or a device
    /// backend cannot size its buffers.
    fn scan(&mut self, camera: usize, image: &ImageU16) -> Result<(), DetectError>;

    /// The candidates of `request`'s band.
    ///
    /// Row-major over the **whole image width**, each carrying OpenCV's integer
    /// `cornerScore` as its response — the ordering and the score
    /// [`detect_keypoints_with_cells`] then reads.
    ///
    /// Both implementations key their band cache on `(row, rung)` and **not**
    /// on `rows`, which also decides the result. That is sound only because
    /// `rows` is the cell grid's row height, constant between two
    /// [`CornerScan::scan`] calls: [`detect_keypoints_with_cells`] derives the
    /// grid once per frame and clears the cache at the entry. An implementation
    /// that wants to be asked for two different `rows` under one key has to put
    /// `rows` in the key.
    ///
    /// # Errors
    ///
    /// [`DetectError::NotScanned`] when no frame has been scanned: a band before
    /// a scan is a programming error on both lanes, not an empty frame
    /// (decision D32). A device backend can also fail on the read.
    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], DetectError>;

    /// Pick the winner of every grid cell where the pixels already are, filling
    /// `out` with one packed key per cell and returning `true`.
    ///
    /// The default is `false`: a scanner with no device behind it has nothing to
    /// gain from the shape, and [`detect_keypoints_with_cells`] then runs its own
    /// walk. [`CpuCornerScan`] takes the default and stays the reference the GPU
    /// lane is checked against.
    ///
    /// `out` is `cells_y * cells_x` keys, row-major over the grid the caller
    /// walks — `(x_stop - x_start) / cell + 1` across and the same down, which
    /// is one **less** than the occupancy matrix's shape. A key packs
    /// `((255 - score) << 24) | (y << 12) | x`, so the winner is the key's
    /// minimum and its fields are the corner the host walk would have chosen;
    /// [`NO_CELL_WINNER`] means the cell has none.
    ///
    /// **Only the ladder's last rung is passed, and that is exact rather than
    /// an approximation.** A candidate at rung `t` is `kept > t`; suppression kills
    /// `p` only through a neighbour whose score is at least `p`'s, and such a
    /// neighbour is a candidate at every rung `p` is — so surviving suppression
    /// does not depend on the rung, and the ladder only admits survivors in
    /// descending score. With `num_points_cell == 1` the whole ladder is
    /// therefore "the best survivor over the last rung it visits", which is what
    /// this asks for — and that rung is [`threshold_rungs`]'s last value, not
    /// `min_threshold`, which the ladder can step straight past. A caller with a
    /// larger budget per cell must use [`CornerScan::band`].
    ///
    /// The masks are the caller's: a backend applies `safe_radius` and the edge
    /// margin, and the caller drops the cells its masks cover.
    ///
    /// A backend with no device path returns `Unsupported`. A `Selected`
    /// result must contain exactly one key per visited cell; malformed lengths
    /// are refused at the detector boundary.
    ///
    /// # Errors
    ///
    /// Whatever the backend's own scan can fail with; a backend that leaves
    /// `out` empty must not have consumed the frame.
    fn select_cells(
        &mut self,
        camera: usize,
        image: &ImageU16,
        select: &CellSelect,
        out: &mut Vec<u32>,
    ) -> Result<SelectionStatus, DetectError> {
        let _ = (camera, image, select);
        out.clear();
        Ok(SelectionStatus::Unsupported)
    }

    /// Launch the [`CornerScan::select_cells`] of every camera `selects` names,
    /// and download none of them.
    ///
    /// `selects[camera]` is [`cell_select`]'s answer for that camera, `None`
    /// where the shape cannot take the device path — or where the caller does
    /// not want that camera's selection yet. What this saves is a wait, not
    /// arithmetic: the selection kernels read the frame and nothing else, so
    /// they can be launched before the frameset has decided anything at all,
    /// and the answer picked up by [`CornerScan::take_cells`] behind whatever
    /// download comes next. A backend that takes it must answer the matching
    /// [`CornerScan::select_cells`] with the same keys, and one that does
    /// nothing leaves every `select_cells` to answer for itself.
    ///
    /// Called once or twice per frameset, and it **clears** what a previous
    /// call left: whatever a `submit_cells` prepares is spent by the
    /// `select_cells` calls of its own frameset and must not outlive them.
    ///
    /// # Errors
    ///
    /// Whatever the backend's own scan can fail with.
    fn submit_cells(
        &mut self,
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
    ) -> Result<(), DetectError> {
        let _ = (images, selects);
        Ok(())
    }

    /// Take what [`CornerScan::submit_cells`] launched, downloading it here only
    /// if nothing else already has.
    ///
    /// Called once after every `submit_cells`, and after the download the
    /// caller expected to carry it. A no-op for a backend whose `submit_cells`
    /// is one.
    ///
    /// # Errors
    ///
    /// Whatever the backend's own download can fail with.
    fn take_cells(&mut self) -> Result<(), DetectError> {
        Ok(())
    }
}

/// The detector's per-frame working set: the corner scanner plus the buffers
/// [`detect_keypoints_with_cells`] filters and suppresses in.
#[derive(Debug)]
pub struct DetectorScratch {
    scanner: Box<dyn CornerScan>,
    corners: Vec<FastCorner>,
    /// One cell's FAST scores, local coordinates, zero where there is no
    /// candidate. Kept zero between calls so only the entries a cell writes are
    /// touched, rather than the whole grid.
    scores: Vec<f32>,
    /// Which of a cell's candidates survived suppression, in candidate order.
    keep: Vec<bool>,
    /// Which cells this frame's masks cover whole, row-major over the grid the
    /// cell loop walks. Only the device path reads it.
    masked: Vec<bool>,
    /// One packed winner key per cell, as a device backend filled it.
    winners: Vec<u32>,
}

impl Default for DetectorScratch {
    fn default() -> Self {
        Self::with_scanner(Box::new(CpuCornerScan::default()))
    }
}

impl DetectorScratch {
    /// A working set over a caller-supplied corner scanner.
    ///
    /// This is how a GPU backend enters the detector: the scanner is the only
    /// part of it that reads pixels.
    pub fn with_scanner(scanner: Box<dyn CornerScan>) -> Self {
        Self {
            scanner,
            corners: Vec::new(),
            scores: Vec::new(),
            keep: Vec::new(),
            masked: Vec::new(),
            winners: Vec::new(),
        }
    }

    /// [`CornerScan::submit_cells`] on the scanner this holds, which is the
    /// only way to it from outside this module.
    ///
    /// # Errors
    ///
    /// Whatever the scanner's own preparation can fail with.
    pub fn submit_cells(
        &mut self,
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
    ) -> Result<(), DetectError> {
        self.scanner.submit_cells(images, selects)
    }

    /// [`CornerScan::take_cells`] on the scanner this holds.
    ///
    /// # Errors
    ///
    /// Whatever the scanner's own download can fail with.
    pub fn take_cells(&mut self) -> Result<(), DetectError> {
        self.scanner.take_cells()
    }
}

/// Detect corners with the image's own grid and a separate shared occupancy matrix.
/// The shapes can differ for mixed-resolution rigs. Skip full cells and cells
/// outside occupancy. Halve the threshold until the minimum or cell budget is
/// reached, never below [`LOWEST_THRESHOLD_RUNG`]. Sort survivors by descending
/// response and apply safe radius, masks and edge margin.
/// `max_corners` caps the call in scan order to protect fixed-capacity buffers.
///
/// # Errors
/// Returns typed errors for occupancy overflow, a short count buffer or an
/// invalid gray image view.
#[allow(clippy::too_many_arguments)]
pub fn detect_keypoints_with_cells(
    image: &ImageU16,
    camera: usize,
    grid: &CellGrid,
    occupancy: &Occupancy<'_>,
    config: &DetectorConfig,
    masks: &Masks,
    max_corners: usize,
    scratch: &mut DetectorScratch,
    out: &mut KeypointsData,
) -> Result<(), DetectError> {
    out.corners.clear();
    out.responses.clear();

    let needed: usize = occupancy.rows.checked_mul(occupancy.columns).ok_or(
        DetectError::OccupancyShapeOverflow {
            rows: occupancy.rows,
            columns: occupancy.columns,
        },
    )?;
    if occupancy.counts.len() < needed {
        return Err(DetectError::OccupancyTooSmall {
            rows: occupancy.rows,
            columns: occupancy.columns,
            expected: needed,
            actual: occupancy.counts.len(),
        });
    }

    // The last rung the cell walk below visits, which is also what the device
    // path is handed (`threshold_rungs`). A ladder with no rung at all detects
    // nothing: the walk's own loop would not run once, so returning here is the
    // answer it would give, without touching the frame.
    if threshold_rungs(config).last().is_none() {
        return Ok(());
    }

    let width: usize = image.width();
    let height: usize = image.height();
    if width < grid.cell || height < grid.cell || max_corners == 0 {
        return Ok(());
    }

    // Every field is taken apart here because the band the scanner lends is
    // borrowed while the candidate list it feeds is written.
    let DetectorScratch {
        scanner,
        corners: candidates,
        scores,
        keep,
        masked,
        winners,
    } = scratch;

    // `float dist_to_center = {full_x - img_raw.w / 2, ...}.norm()` — an integer
    // halving of the size, then a float subtraction.
    let centre_x: f32 = (width / 2) as f32;
    let centre_y: f32 = (height / 2) as f32;

    // Cells the loop below walks: `x_stop` is the **last** cell's left edge, so
    // this is one less each way than the occupancy matrix's shape.
    let (cells_x, cells_y) = grid.dimensions();

    // The device path: the scanner picks each cell's winner where the pixels
    // already are, and what comes back is one packed key per cell instead of the
    // candidate image. Four things have to hold — a budget of one point per cell,
    // for which the threshold ladder carries no information
    // ([`CornerScan::select_cells`]); a cell wide enough to hold a rectangle at
    // all, which is also the only case the loop below detects in; a frame a
    // packed key can name; and masks this grid's cells decompose
    // ([`cell_masks`]). Anything else takes the band walk, which stays the
    // reference.
    let shaped: Option<CellSelect> = cell_select(image, grid, config)
        .filter(|_| cell_masks(masks, grid, cells_x, cells_y, masked));
    let selected = if let Some(select) = shaped {
        scanner.select_cells(camera, image, &select, winners)? == SelectionStatus::Selected
    } else {
        false
    };
    if selected && winners.len() != cells_x * cells_y {
        return Err(DetectError::SelectionLength {
            actual: winners.len(),
            expected: cells_x * cells_y,
        });
    }
    if !selected {
        scanner.scan(camera, image)?;
    }

    // Both sources share capacity, occupancy, and column-major output ordering.
    for (column, row) in grid.cells() {
        if out.corners.len() >= max_corners {
            return Ok(());
        }
        if row >= occupancy.rows || column >= occupancy.columns {
            continue;
        }
        if occupancy.counts[row * occupancy.columns + column] >= config.num_points_cell as i32 {
            continue;
        }
        if selected {
            if !masked[row * cells_x + column] {
                if let Some((point, score)) = decode_key(winners[row * cells_x + column]) {
                    out.corners.push(point);
                    out.responses
                        .push(opencv_corner_score(score as f32 / 255.0));
                }
            }
            continue;
        }
        let x = grid.x_start + column * grid.cell;
        let y = grid.y_start + row * grid.cell;
        let mut points_added: usize = 0;
        // `rung` is the ladder's position, which with `row` is the band
        // cache's key.
        for (rung, threshold) in threshold_rungs(config).enumerate() {
            if points_added >= config.num_points_cell {
                break;
            }
            // `cv::FAST` on the `PATCH_SIZE` sub-image detects at
            // sub-coordinates `[3, PATCH_SIZE - 3)`; the same rectangle in
            // whole-image coordinates is the cell shrunk by the ring radius.
            candidates.clear();
            if grid.cell > 2 * FAST_BORDER {
                // `fast_detect_rect_u8` clamps the rectangle to the ring
                // margin on every side (`cells.rs); the columns this
                // cell keeps out of its row band are that clamp.
                let first: f32 = (x + FAST_BORDER) as f32;
                // The right clamp is the one a caller-supplied grid can
                // need: `x + cell` may reach past the image, where the left
                // edge cannot — `x + 3` is a `usize`.
                let last: f32 =
                    (x + grid.cell - FAST_BORDER).min(width.saturating_sub(FAST_BORDER)) as f32;
                // The row band a cell's own rectangle would have clamped
                // to: `{x + 3, y + 3, cell - 6, cell - 6}` keeps rows
                // `[y + 3, y + cell - 3)` whatever `x` is, and the kernel
                // only ever emits columns `[3, width - 3)`.
                let band: &[FastCorner] = scanner.band(BandRequest {
                    row,
                    rung,
                    y: y + FAST_BORDER,
                    rows: grid.cell - 2 * FAST_BORDER,
                    threshold,
                })?;
                candidates.extend(
                    band.iter()
                        .filter(|corner| corner.xy[0] >= first && corner.xy[0] < last)
                        .copied(),
                );
                suppress_non_maxima(candidates, scores, keep, x, y, grid.cell);
            }
            candidates.sort_by(|a, b| {
                b.response
                    .partial_cmp(&a.response)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for corner in candidates.iter() {
                if points_added >= config.num_points_cell || out.corners.len() >= max_corners {
                    break;
                }
                let full_x: f32 = corner.xy[0];
                let full_y: f32 = corner.xy[1];
                let dx: f32 = full_x - centre_x;
                let dy: f32 = full_y - centre_y;
                // Distance is `sqrt(dx*dx + dy*dy)`.
                let dist_to_center: f32 = (dx * dx + dy * dy).sqrt();

                if config.safe_radius != 0.0 && dist_to_center >= config.safe_radius {
                    continue;
                }
                if masks.in_bounds(full_x, full_y) {
                    continue;
                }
                if !image.in_bounds(full_x, full_y, EDGE_THRESHOLD) {
                    continue;
                }

                out.corners.push([full_x, full_y]);
                // Already OpenCV's integer `cornerScore`; see
                // `opencv_corner_score`.
                out.responses.push(corner.response);
                points_added += 1;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
