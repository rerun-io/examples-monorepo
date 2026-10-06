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
//! Stable sorting retains scan order among tied scores.

mod band;
mod grid;
mod scores;
use self::grid::cell_masks;
use self::grid::NO_CELL_WINNER;
pub use self::grid::{
    cell_select, threshold_rungs, CellGrid, CellGridError, CellMasks, CellSelect,
    CenteredCellConfig, MaskRect, Occupancy, SelectionStatus, EDGE_THRESHOLD,
    LOWEST_THRESHOLD_RUNG, MAX_CELLS,
};
use band::suppress_non_maxima;
pub use band::BandCache;
use band::{block_filter_end, FAST_FILTER_LANES, FAST_RING_COLUMN, FAST_RING_ROW};
pub use band::{opencv_corner_score, CpuCornerScan, FAST_BORDER};

/// kornia's FAST corner, re-exported because [`CornerScan::band`] hands it back:
/// a backend outside this crate cannot implement the trait without naming it.
pub use kornia_imgproc::features::FastCorner;

use kornia_image::Image;

/// Detector input failures, including invalid caller-supplied occupancy dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CenteredCellError {
    /// The scanner could not allocate its grayscale image.
    #[error("cannot allocate a {width}x{height} grayscale image")]
    GrayAllocation {
        /// Requested columns.
        width: usize,
        /// Requested rows.
        height: usize,
    },
    /// A backend claimed selection but returned a malformed winner vector.
    #[error("selection returned {actual} winners, expected {expected}")]
    SelectionLength {
        /// Winners returned by the backend.
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
    /// [`CornerScan::band`] was asked for corners before [`CornerScan::scan`]
    /// ran on this frame.
    ///
    /// [`detect_keypoints_with_cells`] scans before requesting bands. Returning
    /// an error here prevents an unprepared scanner from reporting an empty frame.
    #[error("a corner band was asked for before the frame was scanned")]
    NotScanned,
    /// The 8-bit view could not be built over the bytes written for it.
    ///
    /// Unreachable: the loop that fills those bytes writes exactly
    /// `width * height` of them. It is a typed error rather than an early `Ok`
    /// because reporting success with no keypoints would turn a future geometry
    /// slip into an empty detection result.
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
/// high-water mark.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CenteredCellKeypoints {
    /// Corners in cell scan order.
    pub corners: Vec<[f32; 2]>,
    /// One response per corner.
    pub responses: Vec<f32>,
}

impl CenteredCellKeypoints {
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

/// A frame scanner that supplies FAST candidates in row bands.
///
/// The detector applies the grid, threshold ladder, suppression and masks to
/// these candidates. Implementations may cache each `(row, rung)` band until
/// the next scan. `Send + Sync` lets callers share scanner ownership across threads.
pub trait CornerScan: std::fmt::Debug + Send + Sync {
    /// Backend failures include the common detector input errors.
    type Error: std::error::Error + Send + Sync + From<CenteredCellError>;

    /// Make an independent scanner with the same corner-selection behavior.
    ///
    /// # Returns
    ///
    /// A scanner that can scan any camera concurrently with this one, or
    /// `None` when callers must use this scanner serially. Implementations
    /// must preserve configuration and must not depend on shared per-frame
    /// preparation performed on the original scanner.
    fn fork(&self) -> Option<Box<dyn CornerScan<Error = Self::Error>>> {
        None
    }

    /// Take camera `camera`'s frame, discarding whatever the last one left.
    ///
    /// Called once per camera per frameset, before any [`CornerScan::band`].
    ///
    /// `camera` identifies the source within a frame set. The CPU scanner ignores it.
    ///
    /// # Errors
    ///
    /// Backend errors, including geometry that cannot be viewed as 8-bit.
    ///
    /// # Arguments
    /// * `camera` - Source index within the frame set.
    /// * `image` - Dense current source image.
    fn scan(&mut self, camera: usize, image: &Image<u16, 1>) -> Result<(), Self::Error>;

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
    /// [`CenteredCellError::NotScanned`] when no frame has been scanned: a band before
    /// a scan is a programming error, not an empty frame.
    ///
    /// # Arguments
    /// * `request` - Cell row, threshold rung and pixel band.
    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], Self::Error>;

    /// Select one corner per visited grid cell into `out`.
    ///
    /// Return [`SelectionStatus::Unsupported`] to use the band walk instead.
    /// On success, `out` holds `cells_y * cells_x` entries in row-major order;
    /// each is a corner or `None` when its cell has no winner. The visited grid
    /// has one fewer row and column than the occupancy matrix.
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
    /// A backend applies `safe_radius` and the edge margin. Optional eligibility
    /// supplies occupancy and whole-cell masks so CPU scanners can skip cells.
    /// Scanners may ignore eligibility when selection predates occupancy; the
    /// caller always filters returned corners by occupancy and masks.
    ///
    /// A backend with no cell-selection path returns `Unsupported`. A `Selected`
    /// result must contain exactly one entry per visited cell; malformed lengths
    /// are refused at the detector boundary.
    ///
    /// # Errors
    ///
    /// Whatever the backend's own scan can fail with; a backend that leaves
    /// `out` empty must not have consumed the frame.
    ///
    /// # Arguments
    /// * `camera`, `image` - Frame position and source pixels.
    /// * `select` - Cell selection geometry and policy.
    /// * `_eligibility` - Optional occupancy counts and whole-cell masks.
    /// * `out` - Reusable output, one optional corner per visited cell.
    fn select_cells(
        &mut self,
        camera: usize,
        image: &Image<u16, 1>,
        select: &CellSelect,
        _eligibility: Option<(&Occupancy<'_>, &[bool])>,
        out: &mut Vec<Option<FastCorner>>,
    ) -> Result<SelectionStatus, Self::Error> {
        let _ = (camera, image, select);
        out.clear();
        Ok(SelectionStatus::Unsupported)
    }
}

/// The detector's per-frame working set: the corner scanner plus the buffers
/// [`detect_keypoints_with_cells`] filters and suppresses in.
#[derive(Debug)]
pub struct DetectorScratch<S: CornerScan + ?Sized = dyn CornerScan<Error = CenteredCellError>> {
    scanner: Box<S>,
    corners: Vec<FastCorner>,
    /// One cell's FAST scores, local coordinates, zero where there is no
    /// candidate. Kept zero between calls so only the entries a cell writes are
    /// touched, rather than the whole grid.
    scores: Vec<f32>,
    /// Which of a cell's candidates survived suppression, in candidate order.
    keep: Vec<bool>,
    /// Which cells this frame's masks cover whole, row-major over the grid the
    /// cell loop walks. Only cell selection reads it.
    masked: Vec<bool>,
    /// One optional winner per visited cell.
    winners: Vec<Option<FastCorner>>,
}

impl Default for DetectorScratch {
    fn default() -> Self {
        Self::with_scanner(Box::new(CpuCornerScan::default()))
    }
}

impl<S: CornerScan + ?Sized> DetectorScratch<S> {
    /// A working set over a caller-supplied corner scanner.
    ///
    /// The scanner supplies candidates; this workspace stores filtering scratch.
    ///
    /// # Arguments
    /// * `scanner` - Owned backend providing row bands and optional cell selection.
    pub fn with_scanner(scanner: Box<S>) -> Self {
        Self {
            scanner,
            corners: Vec::new(),
            scores: Vec::new(),
            keep: Vec::new(),
            masked: Vec::new(),
            winners: Vec::new(),
        }
    }

    /// Borrow the scanner for read-only backend operations.
    pub fn scanner(&self) -> &S {
        &self.scanner
    }

    /// Access the owned backend for backend-specific frame preparation.
    pub fn scanner_mut(&mut self) -> &mut S {
        &mut self.scanner
    }

    /// Independent detector scratch for side-camera CPU work.
    pub fn fork(&self) -> Option<DetectorScratch<dyn CornerScan<Error = S::Error>>> {
        self.scanner.fork().map(DetectorScratch::with_scanner)
    }
}

/// Detect corners with the image's own grid and a separate shared occupancy matrix.
/// The shapes can differ for mixed-resolution rigs. Skip full cells and cells
/// outside occupancy. Halve the threshold until the minimum or cell budget is
/// reached, never below [`LOWEST_THRESHOLD_RUNG`]. Sort survivors by descending
/// response and apply safe radius, masks and edge margin.
/// `max_corners` caps the call in scan order to protect fixed-capacity buffers.
///
/// # Arguments
/// * `image`, `camera` - Dense image and its index in a caller-owned frame set.
/// * `grid`, `occupancy` - Centered detection geometry and shared feature counts.
/// * `config`, `masks` - Selection policy and excluded image regions.
/// * `max_corners` - Maximum number of corners emitted in scan order.
/// * `scratch`, `out` - Reusable working storage and output arrays.
///
/// # Examples
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_staging_imgproc::features::{CellGrid, CenteredCellConfig, DetectorScratch,
///     CenteredCellKeypoints, CellMasks, Occupancy, detect_keypoints_with_cells};
/// let image = Image::from_size_val(ImageSize { width: 100, height: 100 }, 4000u16)?;
/// let grid = CellGrid::new(100, 100, 25)?;
/// let counts = vec![0; grid.rows * grid.columns];
/// let occupancy = Occupancy { counts: &counts, rows: grid.rows, columns: grid.columns };
/// let config = CenteredCellConfig { num_points_cell: 1, min_threshold: 5,
///     max_threshold: 40, safe_radius: 0.0 };
/// let mut scratch = DetectorScratch::default();
/// let mut out = CenteredCellKeypoints::default();
/// detect_keypoints_with_cells(&image, 0, &grid, &occupancy, &config,
///     &CellMasks::default(), 100, &mut scratch, &mut out)?;
/// assert!(out.is_empty()); // A constant image has no FAST corners.
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
/// Returns typed errors for occupancy overflow, a short count buffer or an
/// invalid gray image view.
#[allow(clippy::too_many_arguments)]
pub fn detect_keypoints_with_cells<S: CornerScan + ?Sized>(
    image: &Image<u16, 1>,
    camera: usize,
    grid: &CellGrid,
    occupancy: &Occupancy<'_>,
    config: &CenteredCellConfig,
    masks: &CellMasks,
    max_corners: usize,
    scratch: &mut DetectorScratch<S>,
    out: &mut CenteredCellKeypoints,
) -> Result<(), S::Error> {
    detect_prepared_keypoints_with_cells(
        image.size(),
        grid,
        occupancy,
        config,
        masks,
        max_corners,
        scratch,
        out,
        |scanner, select, winners| {
            let selected = if let Some((select, occupancy, masked)) = select {
                scanner.select_cells(camera, image, select, Some((occupancy, masked)), winners)?
                    == SelectionStatus::Selected
            } else {
                false
            };
            if !selected {
                scanner.scan(camera, image)?;
            }
            Ok(selected)
        },
    )
}

/// Apply grid, occupancy, masks and suppression to a prepared corner source.
/// The preparation callback receives eligible cell-selection geometry, or None
/// when the source must provide row bands. It returns true only when it filled
/// one optional corner per cell. Otherwise it must prepare the scanner for
/// subsequent band requests. Corner responses use integer FAST scores.
///
/// # Errors
/// Rejects invalid occupancy or winner lengths; forwards scanner failures.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn detect_prepared_keypoints_with_cells<
    S: CornerScan<Error: Into<E>> + ?Sized,
    E: From<CenteredCellError>,
>(
    size: kornia_image::ImageSize,
    grid: &CellGrid,
    occupancy: &Occupancy<'_>,
    config: &CenteredCellConfig,
    masks: &CellMasks,
    max_corners: usize,
    scratch: &mut DetectorScratch<S>,
    out: &mut CenteredCellKeypoints,
    prepare: impl FnOnce(
        &mut S,
        Option<(&CellSelect, &Occupancy<'_>, &[bool])>,
        &mut Vec<Option<FastCorner>>,
    ) -> Result<bool, E>,
) -> Result<(), E> {
    out.corners.clear();
    out.responses.clear();

    let needed: usize = occupancy.rows.checked_mul(occupancy.columns).ok_or(
        CenteredCellError::OccupancyShapeOverflow {
            rows: occupancy.rows,
            columns: occupancy.columns,
        },
    )?;
    if occupancy.counts.len() < needed {
        return Err(CenteredCellError::OccupancyTooSmall {
            rows: occupancy.rows,
            columns: occupancy.columns,
            expected: needed,
            actual: occupancy.counts.len(),
        }
        .into());
    }

    // The last rung the cell walk and one-corner selection both use. A ladder with no rung at all detects
    // nothing: the walk's own loop would not run once, so returning here is the
    // answer it would give, without touching the frame.
    if threshold_rungs(config).last().is_none() {
        return Ok(());
    }

    let width = size.width;
    let height = size.height;
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

    // One-corner selection requires a one-point budget, supported image and
    // cell dimensions, and masks aligned to whole cells. Other cases use bands.
    let shaped: Option<CellSelect> = cell_select(size, grid, config)
        .filter(|_| cell_masks(masks, grid, cells_x, cells_y, masked));
    let selected = prepare(
        scanner,
        shaped
            .as_ref()
            .map(|select| (select, occupancy, masked.as_slice())),
        winners,
    )?;
    if selected && winners.len() != cells_x * cells_y {
        return Err(CenteredCellError::SelectionLength {
            actual: winners.len(),
            expected: cells_x * cells_y,
        }
        .into());
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
                if let Some(corner) = winners[row * cells_x + column] {
                    out.corners.push(corner.xy);
                    out.responses.push(corner.response);
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
                // margin on every side (`cells.rs`); the columns this
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
                let band: &[FastCorner] = scanner
                    .band(BandRequest {
                        row,
                        rung,
                        y: y + FAST_BORDER,
                        rows: grid.cell - 2 * FAST_BORDER,
                        threshold,
                    })
                    .map_err(Into::into)?;
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
                if !(full_x >= EDGE_THRESHOLD
                    && full_y >= EDGE_THRESHOLD
                    && full_x < (width as f32 - EDGE_THRESHOLD - 1.0)
                    && full_y < (height as f32 - EDGE_THRESHOLD - 1.0))
                {
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
