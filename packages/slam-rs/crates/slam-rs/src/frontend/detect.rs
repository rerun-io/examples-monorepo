//! Grid-cell FAST detection, ported from `src/utils/keypoints.cpp:132-205`.
//!
//! `detectKeypointsWithCells` walks a centred grid of `PATCH_SIZE` cells, skips
//! the cells that already hold enough features, copies each cell down to 8 bits
//! and runs `cv::FAST` on it with a halving threshold ladder, keeping the
//! strongest corners that survive a distance-to-centre gate, the rectangle masks
//! and an edge margin.
//!
//! ## What comes from kornia-rs, and what this wrapper adds
//!
//! kornia's grid detector was written against this exact function
//! (`kornia-imgproc/src/features/cells.rs:92-99`), but its grid starts at `(0, 0)`
//! with ceil-division cells while basalt centres the grid and stops one cell
//! early. Rather than accept different cell boundaries, this wrapper keeps
//! basalt's geometry and calls kornia's **rectangle** entry point
//! [`fast_detect_rect_u8`] (`cells.rs:141`), which is the layer the inventory
//! recommends for "basalt-style consumers that already own a grid walker"
//! (`cells.rs:20-22`). The rectangle is shrunk by the FAST ring radius on every
//! side, because `cv::FAST` on a `PATCH_SIZE`-square sub-image detects only at
//! sub-coordinates `[3, PATCH_SIZE - 3)`; a rectangle over the whole cell would
//! detect in a three-pixel band the C++ never looks at.
//!
//! That entry point scans **whole rows** whatever columns the rectangle asks
//! for, so the call is made once per `(cell row, threshold)` over the whole
//! width and each cell filters its own columns out of the result — see `Band`,
//! which also records why a cell-sized crop is not the same detection
//! (kornia turns its in-block local-maximum filter on at `width >= 800`).
//!
//! **The scores are the same quantity, off by one.** At `arc_length == 9` kornia
//! returns `corner_score_9_scalar(...) / 255.0` (`fast.rs:705-711`, `:838-873`):
//! the max over the sixteen arc starts of the min saturating difference along the
//! arc, which is the smallest threshold at which the pixel stops being a corner.
//! OpenCV's `cornerScore` returns `max(a0, -b0) - 1`
//! (`modules/features2d/src/fast_score.cpp`), i.e. the largest threshold at which
//! it is **still** a corner — one less. [`opencv_corner_score`] applies that
//! subtraction the moment a candidate comes back, so suppression, ranking and
//! [`KeypointsData::responses`] all carry the integer `cv::FAST` reports.
//!
//! **Non-maximum suppression is this wrapper's job.** `cv::FAST`'s third argument
//! defaults to `nonmaxSuppression = true`, and `fast_detect_rect_u8` performs
//! none (`cells.rs:138`). Without it the port emits every pixel along a strong
//! edge where the C++ emits only the local peaks — on a synthetic image of flat
//! bright squares the port produced sixteen corners where `cv::FAST` produces
//! **none**, because every candidate there ties with its neighbour.
//! `suppress_non_maxima` reproduces OpenCV's rule exactly: a candidate survives
//! only when its score is **strictly greater** than all eight neighbours', a
//! neighbour that is not itself a candidate scoring zero. Strictness on both
//! sides is why a plateau of equal scores yields nothing, which is OpenCV's
//! behaviour and the reason for that sixteen-versus-zero.
//!
//! What is left is genuinely different and is accepted, not worked around
//! (decision D09, trap 2): `cv::FAST` walks the whole cell in one pass with its
//! own three-row score ring, and `std::sort` (`keypoints.cpp:166`) is not stable
//! while the sort here is, so ties inside a cell break differently. The C++
//! parity gate seeds the tracker with the C++ keypoints for that reason.

use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::{Rect as KorniaRect, fast_detect_rect_u8};

/// kornia's FAST corner, re-exported because [`CornerScan::band`] hands it back:
/// a backend outside this crate cannot implement the trait without naming it.
pub use kornia_imgproc::features::FastCorner;

use crate::image::ImageU16;

/// `const int EDGE_THRESHOLD = 19` (`keypoints.cpp:55`).
pub const EDGE_THRESHOLD: f32 = 19.0;

/// The Bresenham radius `cv::FAST` and kornia both skip at a border
/// (`cells.rs:151`, `fast.rs:494`).
pub const FAST_BORDER: usize = 3;

/// kornia's Bresenham ring: ring point `k` sits `FAST_RING_ROW[k]` rows and
/// [`FAST_RING_COLUMN`]`[k]` columns from the centre (`fast.rs:489-490`).
pub const FAST_RING_ROW: [i32; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
/// The column half of [`FAST_RING_ROW`]'s ring.
pub const FAST_RING_COLUMN: [i32; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

/// Lanes in one block of kornia's in-block local-maximum filter (`fast.rs:539`).
pub const FAST_FILTER_LANES: usize = 16;

/// The width at which kornia turns that filter on (`fast.rs:517`).
const FAST_FILTER_WIDTH: usize = 800;

/// Where kornia's local-maximum filter stops, and whether it runs at all.
///
/// The filter keeps a candidate only when its score beats both neighbours
/// *inside its own sixteen-lane block*, the blocks are aligned to the image's
/// own left margin, and the scalar tail past the last whole block is
/// unfiltered — kornia's SIMD loop runs while `x + 16 <= width - margin`
/// (`fast.rs:524`). So the alignment and the tail are part of the corner set,
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

/// What the detector can refuse.
///
/// The occupancy matrix is a caller-supplied buffer, so its size is an input
/// like any other: the C++ indexes it unchecked (`keypoints.cpp:148`, trap 15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DetectError {
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

/// The shared feature-count matrix, with the shape the caller allocated it for.
///
/// basalt sizes `cells` from **camera 0** (`frame_to_frame_optical_flow.h:119`)
/// and then lets `detectKeypointsWithCells` index it with the *detected* image's
/// own grid arithmetic (`keypoints.cpp:148`). For a rig whose cameras differ in
/// size those two disagree, and the C++ reads out of range; carrying the shape
/// explicitly is what lets the port skip such a cell instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occupancy<'a> {
    /// Feature counts, row-major over `rows` x `columns`.
    pub counts: &'a [i32],
    /// Rows the matrix was allocated with.
    pub rows: usize,
    /// Columns the matrix was allocated with.
    pub columns: usize,
}

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
    /// `optical_flow_detection_min_threshold`, floored at
    /// [`LOWEST_THRESHOLD_RUNG`] by the ladder below.
    pub min_threshold: i32,
    /// `optical_flow_detection_max_threshold`.
    pub max_threshold: i32,
    /// `optical_flow_image_safe_radius`; `0` switches the gate off (`keypoints.cpp:178`).
    pub safe_radius: f32,
}

/// The rung the halving threshold ladder stops at, whatever the config says.
///
/// basalt's ladder is `while (points_added < num_points_cell && threshold >=
/// min_threshold) { ...; threshold /= 2; }` (`keypoints.cpp:162`, `:187`), and
/// integer division halves 1 to 0 and then 0 to 0 for ever: a
/// `min_threshold` of `0` or less makes **the C++ loop non-terminating too**, on
/// the same cell, and the port reproduced that hang. The frontend refuses such a
/// config up front ([`crate::frontend::flow::FrontendError::ThresholdLadderNeverEnds`]),
/// and this floor is the second line: the detector is public, so a caller that
/// builds a [`DetectorConfig`] by hand gets the ladder run down to a threshold of
/// 1 and no further, rather than a wedged process (decision D32).
pub const LOWEST_THRESHOLD_RUNG: i32 = 1;

/// The thresholds [`detect_keypoints_with_cells`]'s ladder visits, in order.
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
/// [`CornerScan::select_cells`] is handed its **last** value, which is the only
/// rung that decides a one-point-per-cell winner. A device handed
/// `min_threshold` instead would admit corners scoring between the two, which
/// the walk never sees.
pub fn threshold_rungs(config: &DetectorConfig) -> impl Iterator<Item = i32> {
    let floor: i32 = config.min_threshold.max(LOWEST_THRESHOLD_RUNG);
    std::iter::successors(Some(config.max_threshold), |threshold| Some(threshold / 2))
        .take_while(move |threshold| *threshold >= floor)
}

/// One cell row's raw FAST candidates at one threshold, over the whole image width.
///
/// `fast_detect_rect_u8` detects over **whole rows** and filters the result by
/// column (`cells.rs:20-35`), so the scan a cell asks for depends only on its
/// row band and its threshold — the nineteen cells of one grid row at one rung
/// all pay for the same 960-wide sweep. Scanning it once and filtering each
/// cell's columns out of it is the same call with the same arguments, so the
/// candidates and their order are the ones the per-cell calls produced.
///
/// The whole-image width is what the scan has to keep: kornia's kernel turns its
/// in-block local-maximum filter on at `width >= 800` and aligns its sixteen-lane
/// blocks to the image's own left margin (`fast.rs:517`, `:524`), so a cell-sized
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

/// What a device backend needs to pick one corner per grid cell itself.
///
/// The other half of [`CornerScan`], and the reason it is a separate call
/// rather than a flag on [`BandRequest`]: a backend that takes it does the whole
/// of the inner loop — the candidate walk, the suppression, the ordering and the
/// three filters — and hands back one answer per cell, so nothing about a band
/// is meaningful afterwards.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSelect {
    /// The detected image's own grid, as [`detect_keypoints_with_cells`] derives it.
    pub grid: CellGrid,
    /// The **last rung the ladder visits** ([`threshold_rungs`]), which is the
    /// only one that decides the winner (see [`CornerScan::select_cells`]).
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
    /// A backend with no device path leaves `out` **empty**, which is also how
    /// the caller decides: it takes the keys only when there are exactly as many
    /// as the grid has cells, so a backend that fills the wrong number is
    /// ignored rather than indexed.
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
    ) -> Result<(), DetectError> {
        let _ = (camera, image, select);
        out.clear();
        Ok(())
    }

    /// Answer every camera of a frameset's [`CornerScan::select_cells`] in one
    /// device round trip, before the frameset asks for any of them.
    ///
    /// `selects[camera]` is [`cell_select`]'s answer for that camera, `None`
    /// where the shape cannot take the device path at all. What this saves is a
    /// wait, not arithmetic: the selection kernels read the frame and nothing
    /// else, so every camera's can be launched together and downloaded once,
    /// where the per-camera call synchronises once per camera. A backend that
    /// takes it must answer the matching [`CornerScan::select_cells`] with the
    /// same keys it downloaded here, and one that does nothing leaves every
    /// `select_cells` to answer for itself.
    ///
    /// Called once per detecting frameset. Whatever it prepared is spent by the
    /// `select_cells` calls that follow and must not outlive them.
    ///
    /// # Errors
    ///
    /// Whatever the backend's own scan can fail with.
    fn prepare_cells(
        &mut self,
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
    ) -> Result<(), DetectError> {
        let _ = (images, selects);
        Ok(())
    }
}

/// The device selection [`detect_keypoints_with_cells`] would ask `camera` for,
/// or `None` when nothing about the shape can take the device path.
///
/// Mask-independent on purpose: `cell_masks` decides the rest of the gate and is
/// not known until the frameset has masked the camera, while the kernels this
/// describes read the frame and nothing else. So this is what
/// [`CornerScan::prepare_cells`] can be handed before the frameset has run, and
/// [`detect_keypoints_with_cells`] applies the mask half itself.
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

/// The CPU [`CornerScan`]: kornia's `fast_detect_rect_u8`, one sweep per
/// `(row band, threshold)`.
///
/// basalt copies each cell into its own `cv::Mat` with `sub_img_raw(x, y) >> 8`
/// (`keypoints.cpp:152-157`); one whole-image shift produces the same bytes and
/// lets every cell be a zero-copy rectangle over it. The narrowed frame is
/// built **once per frame** and kept: rebuilding the view per band call instead
/// copies the whole frame forty times a frameset, which measured 40.1 ms against
/// 9.4 ms on MIO07/1500 with every value unchanged.
///
/// And the frame is narrowed **into** that image rather than into a staging
/// buffer it is then built from. `Image::from_size_slice` is `data.to_vec()`, so
/// the old shape kept the frame twice — a `Vec<u8>` and the image's own copy —
/// and paid a whole-frame `memcpy` per camera per frameset (0.9 MB at 960x960)
/// for the second. kornia's `Image` derefs to its tensor, which has
/// `as_slice_mut`, so the narrowing pass can write straight into the pixels the
/// detector will read, and the image is reallocated only when the geometry
/// changes.
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
        // `sub_ptr[x] = (sub_img_raw(x, y) >> 8)` (`keypoints.cpp:156`), once,
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

    /// [`CornerScan::prepare_cells`] on the scanner this holds, which is the
    /// only way to it from outside this module.
    ///
    /// # Errors
    ///
    /// Whatever the scanner's own preparation can fail with.
    pub fn prepare_cells(
        &mut self,
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
    ) -> Result<(), DetectError> {
        self.scanner.prepare_cells(images, selects)
    }
}

/// kornia's normalised FAST score as OpenCV's integer `cornerScore`.
///
/// kornia returns `corner_score_9_scalar(...) / 255.0` (`fast.rs:710`), the
/// smallest threshold at which the pixel stops being a corner. OpenCV returns
/// that minus one — the largest threshold at which it is still a corner
/// (`fast_score.cpp`, `threshold = std::max(a0, -b0) - 1`). The subtraction is a
/// constant shift, so it cannot change an ordering, but it is what makes the
/// number the port reports equal to the one `cv::FAST` writes into
/// `cv::KeyPoint::response` and basalt copies into `keypoint_responses`.
///
/// It is applied before suppression rather than after, so the scores compared
/// against a non-candidate's zero are OpenCV's own. Nothing reaches zero: a
/// candidate found at threshold `t` scores at least `t`, and the ladder's floor
/// is `optical_flow_detection_min_threshold`.
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
fn suppress_non_maxima(
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
fn cell_masks(
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

/// `detectKeypointsWithCells` (`keypoints.cpp:132-205`).
///
/// `grid` is **the detected image's own** geometry, as the C++ derives it from
/// `img_raw.w`/`.h` (`:140-144`); `occupancy` is the shared feature-count matrix,
/// which basalt shapes from camera 0. The two agree for a rig whose cameras share
/// a resolution and differ otherwise, which is why they are separate arguments.
/// Cells at or over `num_points_cell` are skipped whole (`:148`); a cell whose
/// index falls outside `occupancy` is skipped too, where the C++ reads out of
/// range.
///
/// The threshold ladder is [`threshold_rungs`]: `max_threshold`, then repeatedly
/// halved by integer division until it would drop below `min_threshold` — 40, 20,
/// 10, 5 for the shipped configs (`:160-188`) — and it stops early as soon as the
/// cell's budget is full. The last rung is never below [`LOWEST_THRESHOLD_RUNG`],
/// which is what makes the ladder finite for every `min_threshold`; basalt's own
/// is not.
/// Within one threshold the surviving corners are ordered by descending response
/// (`:166-167`) and taken until the budget is met, each having to clear the safe
/// radius (`:178`), the masks (`:179`) and `EDGE_THRESHOLD` (`:180`).
///
/// `max_corners` caps the whole call. basalt has no such cap; the port needs one
/// because every downstream buffer is fixed-capacity, and truncating in the
/// detector's own scan order is what keeps a frame processable without ever
/// discarding a keypoint that already exists.
///
/// # Errors
///
/// [`DetectError::OccupancyShapeOverflow`] when the declared shape does not fit
/// in a `usize`, [`DetectError::OccupancyTooSmall`] when the counts buffer is
/// shorter than the shape it was declared with, or
/// [`DetectError::GrayViewRefused`] on an 8-bit view the image geometry should
/// have made impossible.
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
    // halving of the size, then a float subtraction (`keypoints.cpp:176`).
    let centre_x: f32 = (width / 2) as f32;
    let centre_y: f32 = (height / 2) as f32;

    // Cells the loop below walks: `x_stop` is the **last** cell's left edge, so
    // this is one less each way than the occupancy matrix's shape.
    let cells_x: usize = (grid.x_stop - grid.x_start) / grid.cell + 1;
    let cells_y: usize = (grid.y_stop - grid.y_start) / grid.cell + 1;

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
    if let Some(select) = shaped {
        scanner.select_cells(camera, image, &select, winners)?;
        if winners.len() == cells_x * cells_y {
            // `for (x = x_start; x <= x_stop; x += PATCH_SIZE) for (y = ...)`
            // — x outer, y inner, which fixes the order corners are numbered in,
            // and the cap is checked where the C++ checks it.
            for column in 0..cells_x {
                for row in 0..cells_y {
                    if out.corners.len() >= max_corners {
                        return Ok(());
                    }
                    if row >= occupancy.rows || column >= occupancy.columns {
                        continue;
                    }
                    if occupancy.counts[row * occupancy.columns + column]
                        >= config.num_points_cell as i32
                    {
                        continue;
                    }
                    // A wholly masked cell has no unmasked candidate, so the
                    // host walk finds nothing in it either.
                    if masked[row * cells_x + column] {
                        continue;
                    }
                    let key: u32 = winners[row * cells_x + column];
                    if key == NO_CELL_WINNER {
                        continue;
                    }
                    let score: u32 = 255 - (key >> KEY_SCORE_SHIFT);
                    out.corners.push([
                        (key & KEY_FIELD_MASK) as f32,
                        ((key >> KEY_ROW_SHIFT) & KEY_FIELD_MASK) as f32,
                    ]);
                    // The same arithmetic the band walk applies to the same
                    // `u8`, so the response is the one `cv::FAST` reports.
                    out.responses
                        .push(opencv_corner_score(score as f32 / 255.0));
                }
            }
            return Ok(());
        }
    }

    scanner.scan(camera, image)?;

    // `for (x = x_start; x <= x_stop; x += PATCH_SIZE) for (y = y_start; ...)`
    // — x outer, y inner, which fixes the order corners are numbered in.
    let mut x: usize = grid.x_start;
    while x <= grid.x_stop {
        let mut y: usize = grid.y_start;
        while y <= grid.y_stop {
            if out.corners.len() >= max_corners {
                return Ok(());
            }
            let column: usize = (x - grid.x_start) / grid.cell;
            let row: usize = (y - grid.y_start) / grid.cell;
            if row >= occupancy.rows || column >= occupancy.columns {
                y += grid.cell;
                continue;
            }
            if occupancy.counts[row * occupancy.columns + column] >= config.num_points_cell as i32 {
                y += grid.cell;
                continue;
            }

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
                    // margin on every side (`cells.rs:12-15`); the columns this
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
                    // Already OpenCV's integer `cornerScore`; see
                    // `opencv_corner_score`.
                    out.responses.push(corner.response);
                    points_added += 1;
                }
            }

            y += grid.cell;
        }
        x += grid.cell;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    const BUDGET: usize = 4096;

    fn config() -> DetectorConfig {
        DetectorConfig {
            num_points_cell: 1,
            min_threshold: 5,
            max_threshold: 40,
            safe_radius: 0.0,
        }
    }

    /// Bright 5x5 squares on a **gently varying** background.
    ///
    /// The variation matters: with a flat background every candidate along a
    /// square's edge scores the same 255, and OpenCV's strictly-greater
    /// suppression then kills the whole plateau — a real image has no such ties,
    /// and neither does this one.
    /// A band asked for before the frame was scanned is a programming error on
    /// both lanes, and it is the same typed error on both:
    /// `detect_keypoints_with_cells` scans first, so an empty band here would
    /// be a detector that reports success and finds nothing (D32).
    #[test]
    fn a_band_before_a_scan_is_refused() {
        let mut scanner: CpuCornerScan = CpuCornerScan::default();
        let request: BandRequest = BandRequest {
            row: 0,
            rung: 0,
            y: 0,
            rows: 32,
            threshold: 5,
        };
        assert_eq!(scanner.band(request).unwrap_err(), DetectError::NotScanned);
    }

    fn dotted_image(width: usize, height: usize, spacing: usize) -> ImageU16 {
        let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
        for y in 0..height {
            for x in 0..width {
                let background: f64 =
                    60.0 + 25.0 * (x as f64 * 0.09).sin() * (y as f64 * 0.07).cos();
                image.set(x, y, (background as u16) << 8);
            }
        }
        let mut cy: usize = spacing;
        while cy + 5 < height {
            let mut cx: usize = spacing;
            while cx + 5 < width {
                for dy in 0..5 {
                    for dx in 0..5 {
                        image.set(cx + dx, cy + dy, 200u16 << 8);
                    }
                }
                cx += spacing;
            }
            cy += spacing;
        }
        image
    }

    fn occupancy<'a>(counts: &'a [i32], grid: &CellGrid) -> Occupancy<'a> {
        Occupancy {
            counts,
            rows: grid.rows,
            columns: grid.columns,
        }
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

    /// Two cameras of different sizes get different grid **starts** even when
    /// their occupancy matrices come out the same shape.
    #[test]
    fn a_different_image_size_gives_a_different_grid_start() {
        let small: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let large: CellGrid = CellGrid::new(240, 240, 50).unwrap();
        assert_eq!(small.x_start, 0);
        assert_eq!(large.x_start, 20);
        assert_eq!(small.rows, large.rows);
        assert_eq!(small.columns, large.columns);
        // A corner near (210, 80) is inside the 240-wide camera's own grid and
        // outside the one the 200-wide camera would impose.
        assert!(large.contains(210.0, 80.0));
        assert!(!small.contains(210.0, 80.0));
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
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config(),
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        assert!(!out.is_empty());
        assert_eq!(out.corners.len(), out.responses.len());
        for corner in &out.corners {
            assert!(
                image.in_bounds(corner[0], corner[1], EDGE_THRESHOLD),
                "corner {corner:?} is inside the edge margin"
            );
        }
        // OpenCV reports `max(a0, -b0) - 1`, and so does this.
        for response in &out.responses {
            assert!(*response >= 0.0 && *response <= 254.0 && response.fract() == 0.0);
        }
    }

    /// The per-cell budget (`keypoints.cpp:173`) is `num_points_cell`.
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
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config,
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();

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

    /// A `min_threshold` the halving ladder can never reach must still terminate.
    ///
    /// basalt hangs here: `threshold /= 2` reaches 1, then 0, and `0 >= 0` keeps
    /// the loop alive for ever on the first empty cell (`keypoints.cpp:162`,
    /// `:187`). A blank frame is the worst case, because no cell ever fills its
    /// budget and every rung of the ladder is walked. The test finishing at all is
    /// the assertion; the counts are the same as a `min_threshold` of 1 gives,
    /// which is what the floor makes the ladder run.
    #[test]
    fn a_non_positive_min_threshold_still_terminates() {
        let blank: ImageU16 = ImageU16::zeros(200, 200).unwrap();
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; grid.rows * grid.columns];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();

        let mut floored: DetectorConfig = config();
        floored.min_threshold = LOWEST_THRESHOLD_RUNG;
        detect_keypoints_with_cells(
            &blank,
            0,
            &grid,
            &occupancy(&cells, &grid),
            &floored,
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let expected: usize = out.corners.len();

        for min_threshold in [0, -1, i32::MIN] {
            let mut bad: DetectorConfig = config();
            bad.min_threshold = min_threshold;
            detect_keypoints_with_cells(
                &blank,
                0,
                &grid,
                &occupancy(&cells, &grid),
                &bad,
                &Masks::default(),
                BUDGET,
                &mut scratch,
                &mut out,
            )
            .unwrap();
            assert_eq!(
                out.corners.len(),
                expected,
                "min_threshold {min_threshold} detected a different number of corners than the floor"
            );
        }
    }

    /// The rungs the ladder visits, and the last of them.
    ///
    /// The last rung is **not** `min_threshold`: halving lands on 5 from 40 and
    /// steps straight past 6, and a different maximum moves every rung. That
    /// last value is what a device path is handed, so the two must be one
    /// computation ([`threshold_rungs`]).
    #[test]
    fn the_ladder_halves_the_maximum_and_need_never_reach_the_minimum() {
        let rungs = |max_threshold: i32, min_threshold: i32| -> Vec<i32> {
            threshold_rungs(&DetectorConfig {
                max_threshold,
                min_threshold,
                ..config()
            })
            .collect()
        };
        assert_eq!(rungs(40, 5), vec![40, 20, 10, 5]);
        assert_eq!(rungs(40, 6), vec![40, 20, 10]);
        assert_eq!(rungs(32, 5), vec![32, 16, 8]);
        // The floor holds at `LOWEST_THRESHOLD_RUNG` whatever the config asks
        // for, which is what makes the ladder finite.
        assert_eq!(rungs(8, i32::MIN), vec![8, 4, 2, 1]);
        // A maximum already under the floor is a ladder with no rung at all.
        assert!(rungs(4, 5).is_empty());
    }

    /// A ladder with no rung detects nothing rather than detecting at some rung
    /// nobody asked for.
    #[test]
    fn a_maximum_under_the_minimum_detects_nothing() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; grid.rows * grid.columns];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        let mut empty_ladder: DetectorConfig = config();
        empty_ladder.max_threshold = config().min_threshold - 1;
        detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &occupancy(&cells, &grid),
            &empty_ladder,
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert!(out.is_empty());
    }

    /// The floor changes nothing for a config whose ladder already ends.
    #[test]
    fn a_valid_min_threshold_is_left_alone() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; grid.rows * grid.columns];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config(),
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert!(
            !out.is_empty(),
            "the shipped 5..40 ladder should still detect on a textured frame"
        );
        assert!(config().min_threshold > LOWEST_THRESHOLD_RUNG);
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
            0,
            &grid,
            &occupancy(&empty, &grid),
            &config(),
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let with_empty_cells: usize = out.len();

        let full: Vec<i32> = vec![1; grid.rows * grid.columns];
        detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &occupancy(&full, &grid),
            &config(),
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();
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
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config(),
            &masks,
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();
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
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config,
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        for corner in &out.corners {
            let distance: f32 = (corner[0] - 100.0).hypot(corner[1] - 100.0);
            assert!(
                distance < 40.0,
                "corner {corner:?} is outside the safe radius"
            );
        }
        assert!(!out.is_empty());
    }

    /// `max_corners` is the port's own cap and stops the walk mid-grid.
    #[test]
    fn the_corner_budget_truncates_in_scan_order() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; grid.rows * grid.columns];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut full: KeypointsData = KeypointsData::default();
        detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config(),
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut full,
        )
        .unwrap();
        assert!(full.len() > 2);

        let mut capped: KeypointsData = KeypointsData::default();
        detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config(),
            &Masks::default(),
            2,
            &mut scratch,
            &mut capped,
        )
        .unwrap();
        assert_eq!(capped.len(), 2);
        assert_eq!(capped.corners, full.corners[..2]);

        let mut none: KeypointsData = KeypointsData::default();
        detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config(),
            &Masks::default(),
            0,
            &mut scratch,
            &mut none,
        )
        .unwrap();
        assert!(none.is_empty());
    }

    /// The occupancy buffer is a caller input, so a short one is a typed error
    /// rather than a read past the end (decision D32).
    #[test]
    fn a_short_occupancy_buffer_is_refused() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; 3];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        let error = detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config(),
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap_err();
        assert_eq!(
            error,
            DetectError::OccupancyTooSmall {
                rows: grid.rows,
                columns: grid.columns,
                expected: grid.rows * grid.columns,
                actual: 3,
            }
        );
    }

    /// A detection grid wider than the occupancy matrix skips the cells that
    /// fall outside it, where the C++ indexes out of range (trap 15).
    #[test]
    fn a_cell_outside_the_occupancy_matrix_is_skipped() {
        let image: ImageU16 = dotted_image(300, 200, 16);
        let grid: CellGrid = CellGrid::new(300, 200, 50).unwrap();
        // The occupancy matrix of a 200-wide camera 0.
        let narrow: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; narrow.rows * narrow.columns];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &occupancy(&cells, &narrow),
            &config(),
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        for corner in &out.corners {
            let (_, column) = grid.cell_of(corner[0], corner[1]);
            assert!(
                column < narrow.columns,
                "corner {corner:?} came from a skipped cell"
            );
        }
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

    /// OpenCV's rule, spelled out: strictly greater than all eight neighbours,
    /// a non-candidate scoring zero. A tie kills both sides.
    #[test]
    fn suppression_keeps_a_strict_peak_and_kills_a_plateau() {
        let mut scores: Vec<f32> = Vec::new();
        let mut keep: Vec<bool> = Vec::new();
        let corner = |x: usize, y: usize, response: f32| FastCorner {
            xy: [x as f32, y as f32],
            response,
        };

        let mut peak: Vec<FastCorner> = vec![corner(10, 10, 9.0), corner(11, 10, 5.0)];
        suppress_non_maxima(&mut peak, &mut scores, &mut keep, 8, 8, 16);
        assert_eq!(peak.len(), 1);
        assert_eq!(peak[0].xy, [10.0, 10.0]);

        let mut plateau: Vec<FastCorner> = vec![corner(10, 10, 9.0), corner(11, 10, 9.0)];
        suppress_non_maxima(&mut plateau, &mut scores, &mut keep, 8, 8, 16);
        assert!(plateau.is_empty(), "OpenCV suppresses both sides of a tie");

        // Two peaks two pixels apart do not see each other.
        let mut apart: Vec<FastCorner> = vec![corner(10, 10, 9.0), corner(12, 10, 9.0)];
        suppress_non_maxima(&mut apart, &mut scores, &mut keep, 8, 8, 16);
        assert_eq!(apart.len(), 2);

        // A lone candidate survives: every neighbour scores zero.
        let mut lone: Vec<FastCorner> = vec![corner(10, 10, 1.0)];
        suppress_non_maxima(&mut lone, &mut scores, &mut keep, 8, 8, 16);
        assert_eq!(lone.len(), 1);
    }

    /// Without suppression a strong edge emits every pixel along it; with
    /// OpenCV's rule only the peaks survive, which is what `cv::FAST` returns.
    #[test]
    fn suppression_thins_a_run_of_candidates() {
        let mut scores: Vec<f32> = Vec::new();
        let mut keep: Vec<bool> = Vec::new();
        let mut run: Vec<FastCorner> = (0..8)
            .map(|k| FastCorner {
                xy: [(10 + k) as f32, 10.0],
                response: [3.0, 7.0, 4.0, 2.0, 9.0, 1.0, 6.0, 2.0][k as usize],
            })
            .collect();
        suppress_non_maxima(&mut run, &mut scores, &mut keep, 8, 8, 24);
        let kept: Vec<f32> = run.iter().map(|corner| corner.xy[0]).collect();
        assert_eq!(kept, vec![11.0, 14.0, 16.0]);
    }
    /// OpenCV's `cornerScore` is one less than the smallest threshold at which
    /// the pixel stops being a corner (`fast_score.cpp`), which is what kornia
    /// returns. An isolated maximum-contrast peak therefore scores 254, not 255.
    #[test]
    fn the_response_is_opencvs_corner_score() {
        assert_eq!(opencv_corner_score(1.0), 254.0);
        assert_eq!(opencv_corner_score(40.0 / 255.0), 39.0);
        // A constant shift cannot reorder two candidates.
        assert!(opencv_corner_score(0.5) < opencv_corner_score(0.75));

        // On a real isolated peak, the whole pipeline reports 254.
        let mut image: ImageU16 = ImageU16::zeros(120, 120).unwrap();
        for y in 0..120 {
            for x in 0..120 {
                image.set(x, y, 0);
            }
        }
        // Inside cell (60, 60)'s detection band `[63, 107)` and clear of the
        // 19-pixel edge margin.
        image.set(80, 80, 255u16 << 8);
        let grid: CellGrid = CellGrid::new(120, 120, 50).unwrap();
        let cells: Vec<i32> = vec![0; grid.rows * grid.columns];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &occupancy(&cells, &grid),
            &config(),
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert_eq!(out.corners, vec![[80.0, 80.0]]);
        assert_eq!(out.responses, vec![254.0]);
    }

    /// `rows * columns` used to wrap before it was compared with the buffer.
    #[test]
    fn an_occupancy_shape_that_overflows_is_refused() {
        let image: ImageU16 = dotted_image(200, 200, 16);
        let grid: CellGrid = CellGrid::new(200, 200, 50).unwrap();
        let cells: Vec<i32> = vec![0; 32];
        let mut scratch: DetectorScratch = DetectorScratch::default();
        let mut out: KeypointsData = KeypointsData::default();
        let error = detect_keypoints_with_cells(
            &image,
            0,
            &grid,
            &Occupancy {
                counts: &cells,
                rows: usize::MAX,
                columns: 2,
            },
            &config(),
            &Masks::default(),
            BUDGET,
            &mut scratch,
            &mut out,
        )
        .unwrap_err();
        assert_eq!(
            error,
            DetectError::OccupancyShapeOverflow {
                rows: usize::MAX,
                columns: 2
            }
        );
    }
}
