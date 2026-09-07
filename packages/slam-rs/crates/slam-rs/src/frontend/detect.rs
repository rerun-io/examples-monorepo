//! Grid-cell FAST detection, ported from `src/utils/keypoints.cpp:132-205`.
//!
//! `detectKeypointsWithCells` walks a centred grid of `PATCH_SIZE` cells, skips
//! the cells that already hold enough features, copies each cell down to 8 bits
//! and runs `cv::FAST` on it with a halving threshold ladder, keeping the
//! strongest corners that survive a distance-to-centre gate, the rectangle masks
//! and an edge margin.
//!
//! ## What comes from kornia-rs, and the one thing that cannot
//!
//! kornia's grid detector was written against this exact function
//! (`kornia-imgproc/src/features/cells.rs:92-99`), but its grid starts at `(0, 0)`
//! with ceil-division cells while basalt centres the grid and stops one cell
//! early. Rather than accept different cell boundaries, this wrapper keeps
//! basalt's geometry and calls kornia's **rectangle** entry point
//! [`fast_detect_rect_u8`] once per cell (`cells.rs:141`), which is the layer the
//! inventory recommends for "basalt-style consumers that already own a grid
//! walker" (`cells.rs:20-22`). The rectangle is shrunk by the FAST ring radius on
//! every side, because `cv::FAST` on a `PATCH_SIZE`-square sub-image detects only
//! at sub-coordinates `[3, PATCH_SIZE - 3)`; a rectangle over the whole cell
//! would detect in a three-pixel band the C++ never looks at.
//!
//! The remaining difference is the detector itself and is accepted, not worked
//! around (decision D09, trap 2): `cv::FAST`'s response is the largest threshold
//! at which a pixel is still a corner, kornia's is the sum of absolute ring
//! differences over 255 (`cells.rs:57-62`). The ordering the two induce is
//! similar but not equal, so the two implementations select different corners in
//! a cell that offers more than the budget, and the absolute thresholds do not
//! transfer either. That is why the C++ parity gate seeds the tracker with the
//! C++ keypoints instead of comparing detector output. Two smaller deltas ride
//! along: `std::sort` (`keypoints.cpp:166`) is not stable while the sort here is,
//! so ties break differently, and kornia's threshold is an `f32` in `[0, 255]`
//! that it rounds back to a `u8` internally (`fast.rs:485`), which reproduces the
//! integer ladder 40, 20, 10, 5 exactly.

use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::{FastCorner, Rect as KorniaRect, fast_detect_rect_u8};

use crate::image::ImageU16;

/// `const int EDGE_THRESHOLD = 19` (`keypoints.cpp:55`).
pub const EDGE_THRESHOLD: f32 = 19.0;

/// The Bresenham radius `cv::FAST` and kornia both skip at a border
/// (`cells.rs:151`, `fast.rs:494`).
const FAST_BORDER: usize = 3;

/// `cv::FAST`'s default segment length, `FastFeatureDetector::TYPE_9_16`.
const FAST_ARC_LENGTH: usize = 9;

/// `basalt::Rect` (`utils/keypoints.h:55-61`): a half-open rectangle in pixels.
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
    /// `Rect::inBounds` (`keypoints.h:60`).
    #[inline]
    pub fn in_bounds(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

/// `basalt::Masks` (`utils/keypoints.h:63-79`): regions of the image to ignore.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Masks {
    /// The rectangles; a point inside any of them is masked.
    pub masks: Vec<Rect>,
}

impl Masks {
    /// `Masks::inBounds` (`keypoints.h:66-68`): inside *any* rectangle.
    #[inline]
    pub fn in_bounds(&self, x: f32, y: f32) -> bool {
        self.masks.iter().any(|mask| mask.in_bounds(x, y))
    }

    /// `Masks::operator+=` (`keypoints.h:70-73`): append, never merge.
    pub fn extend(&mut self, other: &Masks) {
        self.masks.extend_from_slice(&other.masks);
    }
}

/// `basalt::KeypointsData`'s two frontend fields (`utils/common_types.h`).
///
/// Parallel arrays rather than a vector of structs, for the same reason every
/// other per-keypoint buffer here is (`cubecl-portability.md` §12.2), and cleared
/// and refilled in place so the per-frame path allocates only when a frame beats
/// the previous high-water mark.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeypointsData {
    /// `kd.corners`, in the C++'s cell scan order.
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

/// basalt's centred detection grid (`keypoints.cpp:140-144`).
///
/// `x_start = (w % cell) / 2` and `x_stop = x_start + cell * (w / cell - 1)`, so
/// the grid is centred in the image and its last cell ends one cell short of the
/// right edge. The same arithmetic drives `updateCellCounts` and `addKeypoint`
/// (`frame_to_frame_optical_flow.h:110-113`), which is why it lives in one place.
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
    /// The grid an image of `width` x `height` gets for a `cell`-pixel grid.
    ///
    /// `None` when the image is narrower or shorter than one cell: the C++
    /// `x_start + PATCH_SIZE * (w / PATCH_SIZE - 1)` underflows `size_t` there
    /// and the loop reads far outside the image.
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

    /// The occupancy cell a keypoint falls in (`frame_to_frame_optical_flow.h:727-728`).
    ///
    /// The C++ computes `(kp.x - x_start) / c` in the estimator's **float**
    /// scalar and casts to `int`, so a coordinate just left of `x_start` gives a
    /// quotient in `(-1, 0]` that truncates to `0` rather than a negative index.
    /// The saturating cast here does the same for the values the frontend can
    /// produce and, unlike the C++, cannot index out of range on the ones it
    /// cannot (trap 15).
    #[inline]
    pub fn cell_of(&self, x: f32, y: f32) -> (usize, usize) {
        let column: i32 = ((x - self.x_start as f32) / self.cell as f32) as i32;
        let row: i32 = ((y - self.y_start as f32) / self.cell as f32) as i32;
        (
            row.clamp(0, self.rows as i32 - 1) as usize,
            column.clamp(0, self.columns as i32 - 1) as usize,
        )
    }

    /// Whether a keypoint is inside the grid at all (`frame_to_frame_optical_flow.h:711`).
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
    /// `optical_flow_detection_min_threshold`.
    pub min_threshold: i32,
    /// `optical_flow_detection_max_threshold`.
    pub max_threshold: i32,
    /// `optical_flow_image_safe_radius`; `0` switches the gate off (`keypoints.cpp:178`).
    pub safe_radius: f32,
}

/// The 8-bit view of a level-0 pyramid image, reused between frames.
///
/// basalt copies each cell into its own `cv::Mat` with `sub_img_raw(x, y) >> 8`
/// (`keypoints.cpp:152-157`); one whole-image shift produces the same bytes and
/// lets every cell be a zero-copy rectangle over it.
#[derive(Debug, Default)]
pub struct DetectorScratch {
    bytes: Vec<u8>,
    corners: Vec<FastCorner>,
}

/// `detectKeypointsWithCells` (`keypoints.cpp:132-205`).
///
/// `cells[row * grid.columns + column]` is basalt's `Eigen::MatrixXi cells`: the
/// number of features already in that cell. Cells at or over `num_points_cell`
/// are skipped whole (`:148`).
///
/// The threshold ladder is `max_threshold`, then repeatedly halved by integer
/// division until it drops below `min_threshold` — 40, 20, 10, 5 for the shipped
/// configs (`:160-188`) — and it stops early as soon as the cell's budget is full.
/// Within one threshold the corners are ordered by descending response (`:166-167`)
/// and taken until the budget is met, each having to clear the safe radius
/// (`:178`), the masks (`:179`) and `EDGE_THRESHOLD` (`:180`).
///
/// Returns without detecting anything when the image is smaller than one cell.
pub fn detect_keypoints_with_cells(
    image: &ImageU16,
    grid: &CellGrid,
    cells: &[i32],
    config: &DetectorConfig,
    masks: &Masks,
    scratch: &mut DetectorScratch,
    out: &mut KeypointsData,
) {
    out.corners.clear();
    out.responses.clear();

    let width: usize = image.width();
    let height: usize = image.height();
    if width < grid.cell || height < grid.cell {
        return;
    }

    // `sub_ptr[x] = (sub_img_raw(x, y) >> 8)` (`keypoints.cpp:156`), once.
    scratch.bytes.clear();
    scratch.bytes.reserve(width * height);
    for y in 0..height {
        for pixel in image.row(y) {
            scratch.bytes.push((*pixel >> 8) as u8);
        }
    }
    let Ok(gray) = Image::<u8, 1>::from_size_slice(ImageSize { width, height }, &scratch.bytes)
    else {
        return;
    };

    // `float dist_to_center = {full_x - img_raw.w / 2, ...}.norm()` — an integer
    // halving of the size, then a float subtraction (`keypoints.cpp:176`).
    let centre_x: f32 = (width / 2) as f32;
    let centre_y: f32 = (height / 2) as f32;

    // `for (x = x_start; x <= x_stop; x += PATCH_SIZE) for (y = y_start; ...)`
    // — x outer, y inner, which fixes the order corners are numbered in.
    let mut x: usize = grid.x_start;
    while x <= grid.x_stop {
        let mut y: usize = grid.y_start;
        while y <= grid.y_stop {
            let column: usize = (x - grid.x_start) / grid.cell;
            let row: usize = (y - grid.y_start) / grid.cell;
            if cells[row * grid.columns + column] >= config.num_points_cell as i32 {
                y += grid.cell;
                continue;
            }

            let mut points_added: usize = 0;
            let mut threshold: i32 = config.max_threshold;
            while points_added < config.num_points_cell && threshold >= config.min_threshold {
                // `cv::FAST` on the `PATCH_SIZE` sub-image detects at
                // sub-coordinates `[3, PATCH_SIZE - 3)`; the same rectangle in
                // whole-image coordinates is the cell shrunk by the ring radius.
                scratch.corners.clear();
                if grid.cell > 2 * FAST_BORDER {
                    scratch.corners.extend(fast_detect_rect_u8(
                        &gray,
                        KorniaRect {
                            x: x + FAST_BORDER,
                            y: y + FAST_BORDER,
                            w: grid.cell - 2 * FAST_BORDER,
                            h: grid.cell - 2 * FAST_BORDER,
                        },
                        threshold as f32,
                        FAST_ARC_LENGTH,
                        FAST_BORDER,
                    ));
                }
                scratch.corners.sort_by(|a, b| {
                    b.response
                        .partial_cmp(&a.response)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for corner in &scratch.corners {
                    if points_added >= config.num_points_cell {
                        break;
                    }
                    let full_x: f32 = corner.xy[0];
                    let full_y: f32 = corner.xy[1];
                    let dx: f32 = full_x - centre_x;
                    let dy: f32 = full_y - centre_y;
                    // `Eigen::Vector2f{...}.norm()` is `sqrt(dx*dx + dy*dy)`,
                    // not `hypot` (`keypoints.cpp:176`).
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
                    out.responses.push(corner.response);
                    points_added += 1;
                }

                threshold /= 2;
            }

            y += grid.cell;
        }
        x += grid.cell;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn config() -> DetectorConfig {
        DetectorConfig {
            num_points_cell: 1,
            min_threshold: 5,
            max_threshold: 40,
            safe_radius: 0.0,
        }
    }

    /// A field of bright 5x5 squares: four strong FAST corners each.
    fn dotted_image(width: usize, height: usize, spacing: usize) -> ImageU16 {
        let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
        let mut cy: usize = spacing;
        while cy + 5 < height {
            let mut cx: usize = spacing;
            while cx + 5 < width {
                for dy in 0..5 {
                    for dx in 0..5 {
                        image.set(cx + dx, cy + dy, 0xFF00);
                    }
                }
                cx += spacing;
            }
            cy += spacing;
        }
        image
    }

    /// `keypoints.cpp:140-144` on a 960x960 frame with `grid_size = 50`.
    #[test]
    fn the_grid_is_centred_and_stops_one_cell_early() {
        let grid: CellGrid = CellGrid::new(960, 960, 50).unwrap();
        assert_eq!(grid.x_start, (960 % 50) / 2);
        assert_eq!(grid.x_start, 5);
        assert_eq!(grid.x_stop, 5 + 50 * (960 / 50 - 1));
        assert_eq!(grid.x_stop, 905);
        assert_eq!(grid.x_stop + grid.cell, 955);
        assert_eq!(grid.columns, 960 / 50 + 1);
        assert_eq!(grid.rows, 960 / 50 + 1);
    }

    /// The C++ `x_start + PATCH_SIZE * (w / PATCH_SIZE - 1)` underflows when the
    /// image is narrower than one cell; the port refuses the geometry instead.
    #[test]
    fn an_image_narrower_than_a_cell_has_no_grid() {
        assert!(CellGrid::new(30, 200, 50).is_none());
        assert!(CellGrid::new(200, 30, 50).is_none());
        assert!(CellGrid::new(200, 200, 0).is_none());
    }

    /// A coordinate left of `x_start` truncates to column 0 in C++, not to -1.
    #[test]
    fn a_coordinate_before_the_grid_lands_in_the_first_cell() {
        let grid: CellGrid = CellGrid::new(960, 960, 50).unwrap();
        assert_eq!(grid.cell_of(2.0, 2.0), (0, 0));
        assert_eq!(grid.cell_of(5.0, 5.0), (0, 0));
        assert_eq!(grid.cell_of(55.0, 5.0), (0, 1));
        assert_eq!(grid.cell_of(959.0, 959.0), (19, 19));
        assert!(!grid.contains(2.0, 2.0));
        assert!(grid.contains(5.0, 5.0));
        assert!(!grid.contains(955.0, 5.0));
    }

    #[test]
    fn corners_are_found_and_stay_inside_the_edge_threshold() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; grid.rows * grid.columns];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        detect_keypoints_with_cells(
            &image,
            &grid,
            &cells,
            &config(),
            &Masks::default(),
            &mut scratch,
            &mut out,
        );

        assert!(!out.is_empty());
        assert_eq!(out.corners.len(), out.responses.len());
        for corner in &out.corners {
            assert!(
                image.in_bounds(corner[0], corner[1], EDGE_THRESHOLD),
                "corner {corner:?} is inside the edge margin"
            );
        }
    }

    /// The per-cell budget (`keypoints.cpp:173`) is `num_points_cell`, and the
    /// grid holds `(x_stop - x_start) / cell + 1` cells per axis.
    #[test]
    fn no_cell_yields_more_than_its_budget() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; grid.rows * grid.columns];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        let mut config: DetectorConfig = config();
        config.num_points_cell = 2;
        detect_keypoints_with_cells(
            &image,
            &grid,
            &cells,
            &config,
            &Masks::default(),
            &mut scratch,
            &mut out,
        );

        let mut per_cell: Vec<usize> = vec![0; grid.rows * grid.columns];
        for corner in &out.corners {
            let (row, column) = grid.cell_of(corner[0], corner[1]);
            per_cell[row * grid.columns + column] += 1;
        }
        assert!(
            per_cell
                .iter()
                .all(|count| *count <= config.num_points_cell)
        );
    }

    /// `keypoints.cpp:148`: an occupied cell is skipped whole.
    #[test]
    fn an_occupied_cell_is_skipped() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();

        let empty: Vec<i32> = vec![0; grid.rows * grid.columns];
        detect_keypoints_with_cells(
            &image,
            &grid,
            &empty,
            &config(),
            &Masks::default(),
            &mut scratch,
            &mut out,
        );
        let with_empty_cells: usize = out.len();

        let full: Vec<i32> = vec![1; grid.rows * grid.columns];
        detect_keypoints_with_cells(
            &image,
            &grid,
            &full,
            &config(),
            &Masks::default(),
            &mut scratch,
            &mut out,
        );
        assert!(with_empty_cells > 0);
        assert_eq!(out.len(), 0);
    }

    /// `keypoints.cpp:179`: a masked corner is dropped.
    #[test]
    fn a_mask_over_the_whole_image_drops_everything() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; grid.rows * grid.columns];
        let masks: Masks = Masks {
            masks: vec![Rect {
                x: 0.0,
                y: 0.0,
                w: 200.0,
                h: 200.0,
            }],
        };
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        detect_keypoints_with_cells(
            &image,
            &grid,
            &cells,
            &config(),
            &masks,
            &mut scratch,
            &mut out,
        );
        assert!(out.is_empty());
    }

    /// `keypoints.cpp:178`: outside `safe_radius` of the image centre, nothing is kept.
    #[test]
    fn the_safe_radius_gate_keeps_only_the_middle() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; grid.rows * grid.columns];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        let mut config: DetectorConfig = config();
        config.safe_radius = 40.0;
        detect_keypoints_with_cells(
            &image,
            &grid,
            &cells,
            &config,
            &Masks::default(),
            &mut scratch,
            &mut out,
        );
        for corner in &out.corners {
            let distance: f32 = (corner[0] - 100.0).hypot(corner[1] - 100.0);
            assert!(
                distance < 40.0,
                "corner {corner:?} is outside the safe radius"
            );
        }
        assert!(!out.is_empty());
    }

    /// `keypoints.h:66-68`: masked means inside *any* rectangle, half-open.
    #[test]
    fn masks_are_half_open_and_disjunctive() {
        let masks: Masks = Masks {
            masks: vec![
                Rect {
                    x: 0.0,
                    y: 0.0,
                    w: 10.0,
                    h: 10.0,
                },
                Rect {
                    x: 50.0,
                    y: 50.0,
                    w: 5.0,
                    h: 5.0,
                },
            ],
        };
        assert!(masks.in_bounds(0.0, 0.0));
        assert!(masks.in_bounds(9.99, 9.99));
        assert!(!masks.in_bounds(10.0, 5.0));
        assert!(masks.in_bounds(52.0, 52.0));
        assert!(!masks.in_bounds(20.0, 20.0));
        assert!(!Masks::default().in_bounds(1.0, 1.0));
    }
}
