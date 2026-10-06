//! Detection-cell geometry, eligibility, masks, and private winner keys.
use super::FAST_BORDER;
use kornia_image::ImageSize;

/// Margin excluded from the final keypoint set, in pixels.
pub const EDGE_THRESHOLD: f32 = 19.0;

/// Shared feature counts with an explicit allocated shape.
/// Explicit dimensions let images with different resolutions share occupancy
/// counts and skip cells outside the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occupancy<'a> {
    /// Feature counts, row-major over `rows` x `columns`.
    pub counts: &'a [i32],
    /// Rows the matrix was allocated with.
    pub rows: usize,
    /// Columns the matrix was allocated with.
    pub columns: usize,
}

/// A half-open rectangle in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskRect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl MaskRect {
    /// Whether a point lies within this half-open rectangle.
    ///
    /// # Arguments
    /// * `x`, `y` - Pixel coordinates in the image frame.
    #[inline]
    pub fn in_bounds(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

/// Image regions to ignore.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CellMasks {
    /// The rectangles; a point inside any of them is masked.
    pub masks: Vec<MaskRect>,
}

impl CellMasks {
    /// Whether a point lies inside any rectangle.
    ///
    /// # Arguments
    /// * `x`, `y` - Pixel coordinates in the image frame.
    #[inline]
    pub fn in_bounds(&self, x: f32, y: f32) -> bool {
        self.masks.iter().any(|mask| mask.in_bounds(x, y))
    }

    /// Append rectangles without merging them.
    ///
    /// # Arguments
    /// * `other` - Rectangles to append without merging.
    pub fn extend(&mut self, other: &CellMasks) {
        self.masks.extend_from_slice(&other.masks);
    }
}

/// Maximum entries in a grid's occupancy matrix.
/// This independent limit bounds i32 occupancy storage to 4 MiB.
pub const MAX_CELLS: usize = 1 << 20;

/// Centered detection grid: `x_start = (w % cell) / 2` and
/// `x_stop = x_start + cell * (w / cell - 1)`. Shared by detection and cell counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellGrid {
    /// Side length of each square cell in pixels.
    pub cell: usize,
    /// First cell's left edge.
    pub x_start: usize,
    /// Last cell's left edge.
    pub x_stop: usize,
    /// First cell's top edge.
    pub y_start: usize,
    /// Last cell's top edge.
    pub y_stop: usize,
    /// `w / cell + 1`, the occupancy matrix's column count.
    pub columns: usize,
    /// `h / cell + 1`, the occupancy matrix's row count.
    pub rows: usize,
}

/// Invalid geometry or excessive occupancy storage for a centered grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CellGridError {
    /// At least one cell must fit along each image side.
    #[error("a {width}x{height} image cannot hold cells of side {cell}")]
    InvalidGeometry {
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
        /// Requested cell side.
        cell: usize,
    },
    /// An occupancy dimension cannot be represented.
    #[error("the grid occupancy dimensions exceed usize")]
    ShapeOverflow,
    /// Occupancy would exceed the independent detector storage limit.
    #[error("the {rows}x{columns} occupancy grid exceeds {ceiling} cells")]
    TooManyCells {
        /// Occupancy rows, including the final boundary row.
        rows: usize,
        /// Occupancy columns, including the final boundary column.
        columns: usize,
        /// Maximum occupancy entries.
        ceiling: usize,
    },
}

impl CellGrid {
    /// Number of visited cells, distinct from the occupancy matrix shape.
    pub fn dimensions(&self) -> (usize, usize) {
        (
            (self.x_stop - self.x_start) / self.cell + 1,
            (self.y_stop - self.y_start) / self.cell + 1,
        )
    }

    /// Column-major (column, row) order used to assign corner IDs.
    pub fn cells(&self) -> impl Iterator<Item = (usize, usize)> {
        let (columns, rows) = self.dimensions();
        (0..columns).flat_map(move |column| (0..rows).map(move |row| (column, row)))
    }

    /// Build a centered grid and cache its validated occupancy bounds.
    ///
    /// # Arguments
    /// * `width`, `height` - Image dimensions in pixels.
    /// * `cell` - Positive side length of a square detection cell.
    ///
    /// # Errors
    /// Rejects zero or oversized cells, unrepresentable dimensions and occupancy
    /// above [`MAX_CELLS`]. No allocation is performed.
    pub fn new(width: usize, height: usize, cell: usize) -> Result<Self, CellGridError> {
        if cell == 0 || width < cell || height < cell {
            return Err(CellGridError::InvalidGeometry {
                width,
                height,
                cell,
            });
        }
        let columns = (width / cell)
            .checked_add(1)
            .ok_or(CellGridError::ShapeOverflow)?;
        let rows = (height / cell)
            .checked_add(1)
            .ok_or(CellGridError::ShapeOverflow)?;
        if rows.saturating_mul(columns) > MAX_CELLS {
            return Err(CellGridError::TooManyCells {
                rows,
                columns,
                ceiling: MAX_CELLS,
            });
        }
        let x_start: usize = (width % cell) / 2;
        let y_start: usize = (height % cell) / 2;
        Ok(Self {
            cell,
            x_start,
            x_stop: x_start + cell * (width / cell - 1),
            y_start,
            y_stop: y_start + cell * (height / cell - 1),
            columns,
            rows,
        })
    }

    /// Map a keypoint to an occupancy cell.
    /// The f32 quotient is truncated towards zero: a point just left of the origin
    /// maps to column zero. Saturating conversion prevents an invalid index.
    ///
    /// # Arguments
    /// * `x`, `y` - Pixel coordinates in the image frame.
    #[inline]
    pub fn cell_of(&self, x: f32, y: f32) -> (usize, usize) {
        let column: i32 = ((x - self.x_start as f32) / self.cell as f32) as i32;
        let row: i32 = ((y - self.y_start as f32) / self.cell as f32) as i32;
        (
            row.clamp(0, self.rows as i32 - 1) as usize,
            column.clamp(0, self.columns as i32 - 1) as usize,
        )
    }

    /// Whether a keypoint is inside the grid at all.
    ///
    /// # Arguments
    /// * `x`, `y` - Pixel coordinates in the image frame.
    #[inline]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x_start as f32
            && y >= self.y_start as f32
            && x < (self.x_stop + self.cell) as f32
            && y < (self.y_stop + self.cell) as f32
    }
}

/// Thresholds, point budget and safe radius for centered-cell detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CenteredCellConfig {
    /// Maximum accepted corners per visited cell.
    pub num_points_cell: usize,
    /// Minimum FAST threshold, floored at
    /// [`LOWEST_THRESHOLD_RUNG`] by the ladder below.
    pub min_threshold: i32,
    /// First threshold in the halving ladder.
    pub max_threshold: i32,
    /// Allowed radial fraction around the image center; zero disables this filter.
    pub safe_radius: f32,
}

/// Lowest rung of the detector's halving threshold ladder.
/// Integer division leaves zero unchanged, so allowing a non-positive lower
/// threshold could loop forever. This floor guarantees termination.
pub const LOWEST_THRESHOLD_RUNG: i32 = 1;

/// The thresholds [`super::detect_keypoints_with_cells`]'s ladder visits, in order.
///
/// `max_threshold`, then halved by integer division for as long as the value
/// stays at or above the floor, which is `min_threshold` but never under
/// [`LOWEST_THRESHOLD_RUNG`]. For example, limits 40/5 give 40, 20, 10, 5; 40/6
/// gives 40, 20, 10 and **not** 6, because the next halving is 5 and the ladder
/// never visits the floor itself unless a halving happens to land on it. The
/// sequence is empty when `max_threshold` is already under the floor.
///
/// One iterator rather than two expressions because both callers need it and
/// they must not drift: the cell walk steps through it, and
/// [`super::CornerScan::select_cells`] is handed its **last** value, which is the only
/// rung that decides a one-point-per-cell winner. A device handed
/// `min_threshold` instead would admit corners scoring between the two, which
/// the walk never sees.
///
/// # Arguments
/// * `config` - Integer threshold ladder limits.
pub fn threshold_rungs(config: &CenteredCellConfig) -> impl Iterator<Item = i32> {
    let floor: i32 = config.min_threshold.max(LOWEST_THRESHOLD_RUNG);
    std::iter::successors(Some(config.max_threshold), |threshold| Some(threshold / 2))
        .take_while(move |threshold| *threshold >= floor)
}

/// Geometry and threshold for selecting one corner per grid cell.
///
/// The other half of [`super::CornerScan`], and the reason it is a separate call
/// rather than a flag on [`super::BandRequest`]: a backend that takes it does the whole
/// of the inner loop — the candidate walk, the suppression, the ordering and the
/// three filters — and hands back one answer per cell, so nothing about a band
/// is meaningful afterwards.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSelect {
    /// The detected image's own grid, as [`super::detect_keypoints_with_cells`] derives it.
    pub grid: CellGrid,
    /// The **last rung the ladder visits** ([`threshold_rungs`]), which is the
    /// only one that decides the winner (see [`super::CornerScan::select_cells`]).
    /// Not `min_threshold`: the halving ladder need never reach it.
    pub threshold: i32,
    /// Allowed radial fraction around the image center; zero disables this filter.
    pub safe_radius: f32,
}

/// The key a cell with no winner reports.
///
/// Unreachable as a real key: a winner scores at least `threshold + 1 >= 2`, so
/// its score field is at most 253 where this is 255.
pub const NO_CELL_WINNER: u32 = u32::MAX;

/// Where a packed cell key keeps `255 - score`.
pub const KEY_SCORE_SHIFT: u32 = 24;
/// Where a packed cell key keeps the row.
pub const KEY_ROW_SHIFT: u32 = 12;
/// A packed cell key's column field, which is also its row field's width.
pub const KEY_FIELD_MASK: u32 = 0xFFF;

/// The frame size a packed cell key stops describing.
///
/// Twelve bits each for row and column; larger images use the band path.
pub const CELL_KEY_LIMIT: usize = 1 << KEY_ROW_SHIFT;

/// Cell-selection geometry for a one-point-per-cell detector, when supported.
///
/// Mask-independent: whole-cell eligibility is checked by the detector after
/// geometry is prepared. Unsupported sizes and point budgets use the band path.
///
/// # Arguments
/// * `size` - Source dimensions used to bound cell selection.
/// * `grid` - Centered selection grid.
/// * `config` - Threshold, point-budget and radius policy.
#[must_use]
pub fn cell_select(
    size: ImageSize,
    grid: &CellGrid,
    config: &CenteredCellConfig,
) -> Option<CellSelect> {
    let (width, height): (usize, usize) = (size.width, size.height);
    let takes_cell_selection: bool = config.num_points_cell == 1
        && grid.cell > 2 * FAST_BORDER
        && width >= grid.cell
        && height >= grid.cell
        && width < CELL_KEY_LIMIT
        && height < CELL_KEY_LIMIT;
    if !takes_cell_selection {
        return None;
    }
    Some(CellSelect {
        grid: *grid,
        // The **last** rung the cell walk visits, which is the only one that
        // decides a one-point-per-cell winner.
        threshold: threshold_rungs(config).last()?,
        safe_radius: config.safe_radius,
    })
}

impl CellSelect {
    /// Whether the cell scorer supports these dimensions and threshold.
    ///
    /// # Arguments
    /// * `width`, `height` - Image dimensions for the proposed cell selection.
    pub fn supports(&self, width: usize, height: usize) -> bool {
        width < CELL_KEY_LIMIT && height < CELL_KEY_LIMIT && self.threshold >= LOWEST_THRESHOLD_RUNG
    }
}

/// One cell of `grid` a rectangle's edge names, if it names one exactly.
fn cell_index(edge: f32, start: usize, cell: f32) -> Option<usize> {
    let offset: f32 = edge - start as f32;
    if offset < 0.0 {
        return None;
    }
    let index: f32 = offset / cell;
    if index.fract() != 0.0 {
        return None;
    }
    Some(index as usize)
}

/// Fold `masks` into one flag per cell, or refuse the whole camera.
///
/// Cell selection picks one corner per cell and cannot ask the masks about the
/// runner-up, so it is only sound where a mask covers a cell **whole**: then the
/// cell has no unmasked candidate at all and dropping its key is the same answer
/// the band walk gives. Each mask is checked against this image's grid;
/// a rectangle that straddles a cell boundary requires the band path.
///
/// `false` is that refusal. A rectangle past the last cell is not one: the cell
/// loop never looks there, and the candidates of the last cell stop at its own
/// right edge.
pub(crate) fn cell_masks(
    masks: &CellMasks,
    grid: &CellGrid,
    cells_x: usize,
    cells_y: usize,
    out: &mut Vec<bool>,
) -> bool {
    out.clear();
    out.resize(cells_x * cells_y, false);
    let cell: f32 = grid.cell as f32;
    for rect in &masks.masks {
        if rect.w != cell || rect.h != cell {
            return false;
        }
        let (Some(column), Some(row)) = (
            cell_index(rect.x, grid.x_start, cell),
            cell_index(rect.y, grid.y_start, cell),
        ) else {
            return false;
        };
        if column < cells_x && row < cells_y {
            out[row * cells_x + column] = true;
        }
    }
    true
}

/// A backend either supports selection or leaves the frame for band scanning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStatus {
    /// Shape, mask, or backend limits require the band path.
    Unsupported,
    /// The backend returned one optional corner per visited grid cell.
    Selected,
}

/// Decode one packed winner, retaining the sentinel as absence.
pub fn decode_cell_key(key: u32) -> Option<kornia_imgproc::features::FastCorner> {
    if key == NO_CELL_WINNER {
        return None;
    }
    Some(kornia_imgproc::features::FastCorner {
        xy: [
            (key & KEY_FIELD_MASK) as f32,
            ((key >> KEY_ROW_SHIFT) & KEY_FIELD_MASK) as f32,
        ],
        response: super::opencv_corner_score((255 - (key >> KEY_SCORE_SHIFT)) as f32 / 255.0),
    })
}
