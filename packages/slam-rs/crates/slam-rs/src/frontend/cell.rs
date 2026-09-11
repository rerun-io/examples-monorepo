//! Shared detection-cell geometry, eligibility, masks, and packed winner keys.
use super::detect::FAST_BORDER;
use crate::image::ImageU16;

/// `const int EDGE_THRESHOLD = 19`.
pub const EDGE_THRESHOLD: f32 = 19.0;

/// Shared feature counts with an explicit allocated shape.
/// The frontend sizes occupancy from camera 0 while detection uses each camera's
/// own grid. Explicit dimensions let mixed-resolution rigs skip cells outside
/// the shared matrix instead of indexing beyond it.
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
pub struct Rect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    pub w: f32,
    /// Height.
    pub h: f32,
}

impl Rect {
    /// `Rect::inBounds`.
    #[inline]
    pub fn in_bounds(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

/// Image regions to ignore.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Masks {
    /// The rectangles; a point inside any of them is masked.
    pub masks: Vec<Rect>,
}

impl Masks {
    /// `Masks::inBounds` : inside *any* rectangle.
    #[inline]
    pub fn in_bounds(&self, x: f32, y: f32) -> bool {
        self.masks.iter().any(|mask| mask.in_bounds(x, y))
    }

    /// `Masks::operator+=` : append, never merge.
    pub fn extend(&mut self, other: &Masks) {
        self.masks.extend_from_slice(&other.masks);
    }
}

/// The most cells one camera's detection grid may carry.
///
/// The shape comes from the calibrated resolution over
/// `optical_flow_detection_grid_size`, so a calibration sizes a buffer as much
/// as `max_keypoints` does: the occupancy counts are one `i32` per cell per
/// camera, and a one-pixel grid over a 4,294,967,294-pixel-square frame asked
/// for 2^64 of them — `vec![0; rows * columns]` answers that with a `capacity
/// overflow` panic rather than an error (decision D32).
///
/// It is the keypoint budget's own ceiling because every keypoint occupies a
/// cell, so no detection could fill a grid larger than the budget; 4 MiB of
/// counts per camera.
pub const MAX_CELLS: usize = crate::frontend::tracker::MAX_CAPACITY;

/// Centered detection grid: `x_start = (w % cell) / 2` and
/// `x_stop = x_start + cell * (w / cell - 1)`. Shared by detection and cell counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellGrid {
    /// `PATCH_SIZE`, i.e. `optical_flow_detection_grid_size`.
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

    /// Build a cell grid, returning `None` if either image side is smaller than one cell.
    /// This prevents underflow in the final-cell index.
    pub fn new(width: usize, height: usize, cell: usize) -> Option<Self> {
        if cell == 0 || width < cell || height < cell {
            return None;
        }
        let x_start: usize = (width % cell) / 2;
        let y_start: usize = (height % cell) / 2;
        Some(Self {
            cell,
            x_start,
            x_stop: x_start + cell * (width / cell - 1),
            y_start,
            y_stop: y_start + cell * (height / cell - 1),
            columns: width / cell + 1,
            rows: height / cell + 1,
        })
    }

    /// Map a keypoint to an occupancy cell.
    /// The f32 quotient is truncated towards zero: a point just left of the origin
    /// maps to column zero. Saturating conversion prevents an invalid index (trap 15).
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
    #[inline]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x_start as f32
            && y >= self.y_start as f32
            && x < (self.x_stop + self.cell) as f32
            && y < (self.y_stop + self.cell) as f32
    }
}

/// Everything `detectKeypointsWithCells` reads out of the config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectorConfig {
    /// `optical_flow_detection_num_points_cell`.
    pub num_points_cell: usize,
    /// `optical_flow_detection_min_threshold`, floored at
    /// [`LOWEST_THRESHOLD_RUNG`] by the ladder below.
    pub min_threshold: i32,
    /// `optical_flow_detection_max_threshold`.
    pub max_threshold: i32,
    /// `optical_flow_image_safe_radius`; `0` switches the gate off.
    pub safe_radius: f32,
}

/// Lowest rung of the detector's halving threshold ladder.
/// Integer division leaves zero unchanged, so allowing a non-positive lower
/// threshold could loop forever. The frontend refuses such configurations, and
/// this floor also protects direct detector callers (D32).
pub const LOWEST_THRESHOLD_RUNG: i32 = 1;

/// The thresholds [`super::detect::detect_keypoints_with_cells`]'s ladder visits, in order.
///
/// `max_threshold`, then halved by integer division for as long as the value
/// stays at or above the floor, which is `min_threshold` but never under
/// [`LOWEST_THRESHOLD_RUNG`]. The shipped 40/5 configs give 40, 20, 10, 5; 40/6
/// gives 40, 20, 10 and **not** 6, because the next halving is 5 and the ladder
/// never visits the floor itself unless a halving happens to land on it. The
/// sequence is empty when `max_threshold` is already under the floor.
///
/// One iterator rather than two expressions because both callers need it and
/// they must not drift: the cell walk steps through it, and
/// [`super::detect::CornerScan::select_cells`] is handed its **last** value, which is the only
/// rung that decides a one-point-per-cell winner. A device handed
/// `min_threshold` instead would admit corners scoring between the two, which
/// the walk never sees.
pub fn threshold_rungs(config: &DetectorConfig) -> impl Iterator<Item = i32> {
    let floor: i32 = config.min_threshold.max(LOWEST_THRESHOLD_RUNG);
    std::iter::successors(Some(config.max_threshold), |threshold| Some(threshold / 2))
        .take_while(move |threshold| *threshold >= floor)
}

/// What a device backend needs to pick one corner per grid cell itself.
///
/// The other half of [`super::detect::CornerScan`], and the reason it is a separate call
/// rather than a flag on [`super::detect::BandRequest`]: a backend that takes it does the whole
/// of the inner loop — the candidate walk, the suppression, the ordering and the
/// three filters — and hands back one answer per cell, so nothing about a band
/// is meaningful afterwards.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSelect {
    /// The detected image's own grid, as [`super::detect::detect_keypoints_with_cells`] derives it.
    pub grid: CellGrid,
    /// The **last rung the ladder visits** ([`threshold_rungs`]), which is the
    /// only one that decides the winner (see [`super::detect::CornerScan::select_cells`]).
    /// Not `min_threshold`: the halving ladder need never reach it.
    pub threshold: i32,
    /// `optical_flow_image_safe_radius`; `0` switches the gate off.
    pub safe_radius: f32,
}

/// The key a cell with no winner reports.
///
/// Unreachable as a real key: a winner scores at least `threshold + 1 >= 2`, so
/// its score field is at most 253 where this is 255.
pub const NO_CELL_WINNER: u32 = u32::MAX;

/// Where a packed cell key keeps `255 - score`.
pub(crate) const KEY_SCORE_SHIFT: u32 = 24;
/// Where a packed cell key keeps the row.
pub(crate) const KEY_ROW_SHIFT: u32 = 12;
/// A packed cell key's column field, which is also its row field's width.
const KEY_FIELD_MASK: u32 = 0xFFF;

/// The frame size a packed cell key stops describing.
///
/// Twelve bits each for the row and the column, which every calibrated frame in
/// the reference set is two orders of magnitude inside; a larger one takes the
/// band path instead of losing a coordinate.
pub const CELL_KEY_LIMIT: usize = 1 << KEY_ROW_SHIFT;

/// The device selection [`super::detect::detect_keypoints_with_cells`] would ask `camera` for,
/// or `None` when nothing about the shape can take the device path.
///
/// Mask-independent on purpose: `cell_masks` decides the rest of the gate and is
/// not known until the frameset has masked the camera, while the kernels this
/// describes read the frame and nothing else. So this is what
/// [`super::detect::CornerScan::submit_cells`] can be handed before the frameset has run, and
/// [`super::detect::detect_keypoints_with_cells`] applies the mask half itself.
#[must_use]
pub fn cell_select(
    image: &ImageU16,
    grid: &CellGrid,
    config: &DetectorConfig,
) -> Option<CellSelect> {
    let (width, height): (usize, usize) = (image.width(), image.height());
    let takes_device: bool = config.num_points_cell == 1
        && grid.cell > 2 * FAST_BORDER
        && width >= grid.cell
        && height >= grid.cell
        && width < CELL_KEY_LIMIT
        && height < CELL_KEY_LIMIT;
    if !takes_device {
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
/// The device path picks one corner per cell and cannot ask the masks about the
/// runner-up, so it is only sound where a mask covers a cell **whole**: then the
/// cell has no unmasked candidate at all and dropping its key is the same answer
/// the host walk gives. That is what the frontend's masks are —
/// `cam0OverlapCellsMasksForCam` pushes `cell` x `cell` rectangles at the cell
/// origins — but only while the camera being detected shares camera 0's grid,
/// which the mixed-geometry rigs the port supports deliberately do not. Rather
/// than assume it, every rectangle is checked against this grid and a camera
/// with one that straddles a cell boundary takes the band path.
///
/// `false` is that refusal. A rectangle past the last cell is not one: the cell
/// loop never looks there, and the candidates of the last cell stop at its own
/// right edge.
pub(crate) fn cell_masks(
    masks: &Masks,
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
    /// The backend returned one key per visited grid cell.
    Selected,
}

/// Decode one packed winner, retaining the sentinel as absence.
pub(crate) fn decode_key(key: u32) -> Option<([f32; 2], u32)> {
    if key == NO_CELL_WINNER {
        return None;
    }
    Some((
        [
            (key & KEY_FIELD_MASK) as f32,
            ((key >> KEY_ROW_SHIFT) & KEY_FIELD_MASK) as f32,
        ],
        255 - (key >> KEY_SCORE_SHIFT),
    ))
}
