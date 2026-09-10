//! The GPU detector against the host band walk, cell for cell.
//!
//! A binary of its own rather than three more tests in `gpu_kernels.rs`, and for
//! a measurable reason: `the_whole_gpu_path_holds_the_pool_flat` asserts that
//! CubeCL's pool is **exactly** flat frame after frame, and every test in one
//! binary shares one client and one pool. These drive two 960x960 scanners at a
//! time, which made that assertion fail about one run in three. Cargo runs test
//! binaries one after another, so the separation is what makes both
//! deterministic.
//!
//! What is checked here is the exactness argument behind
//! [`slam_rs::frontend::detect::CornerScan::select_cells`]: the whole of
//! `detectKeypointsWithCells` runs twice over the same frame — once through
//! `CpuCornerScan`, which walks the threshold ladder band by band, and once
//! through `GpuCornerScan`, which picks one winner per cell on the device — and
//! the two results have to be equal corner for corner, response for response, in
//! the same cell scan order. Ties are included rather than excused: the packed
//! key's row and column fields break them the way the row-major band walk and a
//! stable sort do.
//!
//! These run only under `--features gpu-wgpu` and need a working CubeCL runtime.
#![cfg(feature = "gpu-core")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use slam_rs::frontend::detect::{
    BandRequest, CELL_KEY_LIMIT, CellGrid, CellSelect, CornerScan, CpuCornerScan, DetectError,
    DetectorConfig, DetectorScratch, FAST_BORDER, FastCorner, KeypointsData, Masks, Occupancy,
    Rect, detect_keypoints_with_cells, threshold_rungs,
};
use slam_rs::frontend::patterns::Pattern51;
use slam_rs::frontend::tracker::PatchTracker;
use slam_rs::gpu::{GpuCornerScan, GpuPatchTracker, ReadRelay, gpu_client};
use slam_rs::image::ImageU16;

mod common;

use common::cornered_image;

/// The msd-index detector, which is `num_points_cell = 1` and the 40/20/10/5
/// ladder every shipped config runs.
fn detector_config(safe_radius: f32) -> DetectorConfig {
    DetectorConfig {
        num_points_cell: 1,
        min_threshold: 5,
        max_threshold: 40,
        safe_radius,
    }
}

/// A scanner that records which of the detector's two paths it was asked for.
///
/// Equality on its own cannot tell them apart: a device path that quietly never
/// engaged agrees with the host walk perfectly, and every check here would pass
/// while measuring nothing. So each one says which path it meant.
#[derive(Debug)]
struct CountingScan {
    inner: Box<dyn CornerScan>,
    bands: Arc<AtomicUsize>,
    selections: Arc<AtomicUsize>,
}

impl CornerScan for CountingScan {
    fn scan(&mut self, camera: usize, image: &ImageU16) -> Result<(), DetectError> {
        self.inner.scan(camera, image)
    }

    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], DetectError> {
        self.bands.fetch_add(1, Ordering::Relaxed);
        self.inner.band(request)
    }

    fn select_cells(
        &mut self,
        camera: usize,
        image: &ImageU16,
        select: &CellSelect,
        out: &mut Vec<u32>,
    ) -> Result<(), DetectError> {
        self.selections.fetch_add(1, Ordering::Relaxed);
        self.inner.select_cells(camera, image, select, out)
    }

    /// Forwarded, not defaulted: the trait's default prepares nothing, so a
    /// decorator that forgot this would silently take the device lane off the
    /// batched path and every equality below would still pass.
    fn submit_cells(
        &mut self,
        images: &[ImageU16],
        selects: &[Option<CellSelect>],
    ) -> Result<(), DetectError> {
        self.inner.submit_cells(images, selects)
    }

    /// Forwarded for the reason above: this is the half that downloads.
    fn take_cells(&mut self) -> Result<(), DetectError> {
        self.inner.take_cells()
    }
}

/// What one equality check saw.
struct Agreement {
    /// Corners both lanes produced, cell scan order for cell scan order.
    corners: usize,
    /// Band sweeps the GPU scanner was asked for: nonzero means the walk ran.
    bands: usize,
    /// Cell selections it was asked for: nonzero means the device path ran.
    selections: usize,
}

/// One camera's detection, from a scanner of the caller's choosing.
fn detect_with(
    scanner: Box<dyn CornerScan>,
    image: &ImageU16,
    grid: &CellGrid,
    counts: &[i32],
    config: &DetectorConfig,
    masks: &Masks,
    budget: usize,
) -> KeypointsData {
    let mut scratch: DetectorScratch = DetectorScratch::with_scanner(scanner);
    let mut out: KeypointsData = KeypointsData::default();
    detect_keypoints_with_cells(
        image,
        0,
        grid,
        &Occupancy {
            counts,
            rows: grid.rows,
            columns: grid.columns,
        },
        config,
        masks,
        budget,
        &mut scratch,
        &mut out,
    )
    .unwrap();
    out
}

/// The GPU's per-cell winners are the ones the host band walk chooses.
///
/// The whole of `detectKeypointsWithCells` runs twice over the same frame — once
/// through `CpuCornerScan`, which takes the trait's default and walks bands, and
/// once through `GpuCornerScan`, which picks each cell's winner on the device —
/// and the two `KeypointsData` must be equal, corner for corner and response for
/// response, in the same cell scan order.
///
/// That equality is the exactness argument, not the A/B run: the ladder carries
/// no information at `num_points_cell = 1`, suppression is threshold-independent,
/// and the packed key orders candidates the way the host's stable sort does. Ties
/// are included rather than excused — the key's row and column fields break them
/// the way the row-major band walk does.
fn detection_agrees(
    image: &ImageU16,
    grid: &CellGrid,
    counts: &[i32],
    config: &DetectorConfig,
    masks: &Masks,
    budget: usize,
    label: &str,
) -> Agreement {
    let want: KeypointsData = detect_with(
        Box::new(CpuCornerScan::default()),
        image,
        grid,
        counts,
        config,
        masks,
        budget,
    );
    let bands: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let selections: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let got: KeypointsData = detect_with(
        Box::new(CountingScan {
            inner: Box::new(GpuCornerScan::new(gpu_client().unwrap()).unwrap()),
            bands: Arc::clone(&bands),
            selections: Arc::clone(&selections),
        }),
        image,
        grid,
        counts,
        config,
        masks,
        budget,
    );
    assert_eq!(
        got.corners.len(),
        want.corners.len(),
        "{label}: {} corners against {}",
        got.corners.len(),
        want.corners.len()
    );
    for (index, (got, want)) in got.corners.iter().zip(want.corners.iter()).enumerate() {
        assert_eq!(got, want, "{label}: corner {index}");
    }
    assert_eq!(got.responses, want.responses, "{label}: responses");
    Agreement {
        corners: want.corners.len(),
        bands: bands.load(Ordering::Relaxed),
        selections: selections.load(Ordering::Relaxed),
    }
}

/// The three framesets of MIO10 the flow fixtures carry, as the detector sees
/// them: 960x960, the geometry the msd-index rig runs.
fn mio10_frame(frame: usize, camera: usize) -> ImageU16 {
    let pgm: common::Pgm = common::read_pgm(&common::fixtures().join("flow/frames"), frame, camera);
    let mut image: ImageU16 = ImageU16::zeros(pgm.width, pgm.height).unwrap();
    for y in 0..pgm.height {
        for x in 0..pgm.width {
            image.set(x, y, u16::from(pgm.pixels[y * pgm.width + x]) << 8);
        }
    }
    image
}

/// Every cell of a real MIO10 frameset, both cameras, empty and half full.
#[test]
fn the_gpu_cell_selection_matches_the_host_walk_on_a_real_frameset() {
    let config: DetectorConfig = detector_config(472.0);
    for camera in 0..2 {
        let image: ImageU16 = mio10_frame(0, camera);
        let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
        let cells: usize = grid.rows * grid.columns;

        // Nothing tracked yet: every cell of the grid is detected in.
        let empty: Vec<i32> = vec![0; cells];
        let agreed: Agreement = detection_agrees(
            &image,
            &grid,
            &empty,
            &config,
            &Masks::default(),
            4096,
            &format!("cam{camera} empty"),
        );
        assert!(
            agreed.corners > 30,
            "cam{camera} found only {} corners, which would make the equality vacuous",
            agreed.corners
        );
        assert!(
            agreed.selections > 0 && agreed.bands == 0,
            "cam{camera} did not take the device path: {} selections, {} bands",
            agreed.selections,
            agreed.bands
        );

        // Half the cells already hold a feature, which is the steady state: the
        // skip has to land on the same cells on both lanes.
        let mut busy: Vec<i32> = vec![0; cells];
        for (index, count) in busy.iter_mut().enumerate() {
            *count = i32::from(index % 3 == 0);
        }
        detection_agrees(
            &image,
            &grid,
            &busy,
            &config,
            &Masks::default(),
            4096,
            &format!("cam{camera} occupied"),
        );
    }
}

/// The gates the kernel took over, one at a time, and the budget the host keeps.
#[test]
fn the_gpu_cell_selection_applies_the_same_gates() {
    let image: ImageU16 = mio10_frame(1, 0);
    let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
    let counts: Vec<i32> = vec![0; grid.rows * grid.columns];

    // `safe_radius = 0` switches the distance gate off entirely, and 200 makes
    // it bite on a 960x960 frame where 472 leaves most of the grid alone.
    for radius in [0.0f32, 200.0, 472.0] {
        detection_agrees(
            &image,
            &grid,
            &counts,
            &detector_config(radius),
            &Masks::default(),
            4096,
            &format!("safe radius {radius}"),
        );
    }

    // `cam0OverlapCellsMasksForCam`'s own shape: `cell` x `cell` rectangles at
    // the cell origins. Every third one, so masked and clear cells interleave.
    let mut masks: Masks = Masks::default();
    let mut index: usize = 0;
    let mut y: usize = grid.y_start;
    while y <= grid.y_stop {
        let mut x: usize = grid.x_start;
        while x <= grid.x_stop {
            if index % 3 == 0 {
                masks.masks.push(Rect {
                    x: x as f32,
                    y: y as f32,
                    w: grid.cell as f32,
                    h: grid.cell as f32,
                });
            }
            index += 1;
            x += grid.cell;
        }
        y += grid.cell;
    }
    detection_agrees(
        &image,
        &grid,
        &counts,
        &detector_config(472.0),
        &masks,
        4096,
        "cell-aligned masks",
    );

    // A rectangle that straddles a cell boundary is the mixed-geometry rig's
    // shape: the device path has to refuse it and the band walk has to answer,
    // which is the same answer either way.
    let mut straddling: Masks = Masks::default();
    straddling.masks.push(Rect {
        x: (grid.x_start + 17) as f32,
        y: (grid.y_start + 21) as f32,
        w: grid.cell as f32,
        h: grid.cell as f32,
    });
    let refused: Agreement = detection_agrees(
        &image,
        &grid,
        &counts,
        &detector_config(472.0),
        &straddling,
        4096,
        "a mask across a cell boundary",
    );
    assert_eq!(
        refused.selections, 0,
        "a straddling mask has to send the whole camera down the band walk"
    );
    assert!(refused.bands > 0, "and the band walk has to have run");

    // The port's own cap, checked where the C++ checks its cell budget: the
    // truncation must fall in the same place in the same scan order.
    for budget in [1usize, 7, 40] {
        detection_agrees(
            &image,
            &grid,
            &counts,
            &detector_config(472.0),
            &Masks::default(),
            budget,
            &format!("budget {budget}"),
        );
    }
}

/// A frame whose width is not a whole number of cells, and one shorter than it
/// is wide.
///
/// Not the clamps — [`CellGrid::new`] floors and centres, so every grid it
/// derives ends inside its image and the kernel's two strict clamps never run.
/// What these three do exercise is the other half of the geometry: kornia's
/// in-block local-maximum filter is on at 960 and off at 512, and a width that
/// is not a whole number of cells moves the grid's own start, both of which
/// change the candidate set the selection reads.
#[test]
fn the_gpu_cell_selection_matches_the_host_walk_on_uneven_frames() {
    for (width, height, cell) in [
        (960usize, 240usize, 50usize),
        (512, 192, 32),
        (517, 193, 50),
    ] {
        let image: ImageU16 = cornered_image(width, height);
        let grid: CellGrid = CellGrid::new(width, height, cell).unwrap();
        // The precondition the clamp test below needs and this one does not
        // have: `CellGrid::new` cannot produce a cell that runs past the image.
        assert!(grid.x_stop + cell <= width && grid.y_stop + cell <= height);
        let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
        let agreed: Agreement = detection_agrees(
            &image,
            &grid,
            &counts,
            &detector_config(0.0),
            &Masks::default(),
            4096,
            &format!("{width}x{height} cell {cell}"),
        );
        assert!(
            agreed.corners > 0,
            "{width}x{height} found nothing to compare"
        );
        assert!(agreed.selections > 0, "{width}x{height} took the band walk");
    }
}

/// A grid the detector's **caller** supplies, rather than one `CellGrid::new`
/// derives from an image.
///
/// The occupancy matrix `detect_with` sizes from `rows` and `columns` has to
/// cover every cell the walk visits, or the cells fall out of it instead of
/// being detected in.
fn caller_grid(
    x_start: usize,
    x_stop: usize,
    y_start: usize,
    y_stop: usize,
    cell: usize,
) -> CellGrid {
    CellGrid {
        cell,
        x_start,
        x_stop,
        y_start,
        y_stop,
        columns: (x_stop - x_start) / cell + 1,
        rows: (y_stop - y_start) / cell + 1,
    }
}

/// Grids whose last cell runs past the image, which is where the clamps bite.
///
/// The frontend derives its grid with `CellGrid::new`, whose last cell always
/// ends inside the frame, so `min(x + cell - 3, width - 3)` and the same down
/// the side are dead there. The detector takes the grid from its caller, and
/// these are the two shapes that reach them: a last column and a last row that
/// overhang while still holding pixels inside `EDGE_THRESHOLD`, and a last
/// column whose whole window is empty, which the device has to report as the
/// no-winner sentinel and the host as nothing at all.
///
/// What the equality proves is that the device stays inside the image and sees
/// the same zero rim; the clamped columns themselves are past
/// `width - EDGE_THRESHOLD - 1`, so no corner can come out of them on either
/// lane whatever the clamp does.
#[test]
fn the_gpu_cell_selection_matches_the_host_walk_on_overhanging_cells() {
    let (width, height, cell): (usize, usize, usize) = (200, 150, 50);
    let image: ImageU16 = cornered_image(width, height);

    // Cells at x = 20, 70, 120, 170 and y = 10, 60, 110: the last column ends at
    // 220 and the last row at 160, both past the frame.
    let overhanging: CellGrid = caller_grid(20, 170, 10, 110, cell);
    assert!(
        overhanging.x_stop + cell > width,
        "the last column has to run past the right edge"
    );
    assert!(
        overhanging.y_stop + cell > height,
        "the last row has to run past the bottom edge"
    );
    assert!(
        overhanging.x_stop + FAST_BORDER < width - FAST_BORDER,
        "and its window still has to hold candidates"
    );

    // Cells at x = 47, 97, 147, 197: the last column's window is [200, 197),
    // which is empty, and the cell has to come back as the sentinel.
    let empty_window: CellGrid = caller_grid(47, 197, 0, 100, cell);
    assert!(
        empty_window.x_stop + FAST_BORDER >= width - FAST_BORDER,
        "the last column's candidate window has to be empty"
    );

    for (grid, label) in [
        (overhanging, "an overhanging last column and row"),
        (empty_window, "an empty last candidate window"),
    ] {
        let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
        let agreed: Agreement = detection_agrees(
            &image,
            &grid,
            &counts,
            &detector_config(0.0),
            &Masks::default(),
            4096,
            label,
        );
        assert!(agreed.corners > 0, "{label} found nothing to compare");
        assert!(agreed.selections > 0, "{label} took the band walk");
    }
}

/// The ladder's last rung is what the device is handed, not `min_threshold`.
///
/// `40/6` visits 40, 20, 10 and stops, because the next halving is 5 and the
/// floor is 6; `32/5` visits 32, 16, 8. A device handed the configured minimum
/// would admit a cell's best survivor scoring between the two — a corner the
/// host walk never sees. The `admitted` run below is exactly that ladder, and
/// asserting it finds strictly more corners is what stops this from passing
/// vacuously.
#[test]
fn the_gpu_cell_selection_stops_at_the_last_rung_the_walk_visits() {
    let image: ImageU16 = mio10_frame(0, 0);
    let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
    let counts: Vec<i32> = vec![0; grid.rows * grid.columns];

    for (max_threshold, min_threshold, last_rung) in [(40i32, 6i32, 10i32), (32, 5, 8)] {
        let config: DetectorConfig = DetectorConfig {
            max_threshold,
            min_threshold,
            ..detector_config(472.0)
        };
        assert_eq!(threshold_rungs(&config).last(), Some(last_rung));

        // The ladder that stops at the configured minimum instead: one rung, at
        // `min_threshold`. This is what a device given the wrong bound detects.
        let at_the_minimum: DetectorConfig = DetectorConfig {
            max_threshold: min_threshold,
            ..config
        };
        assert_eq!(threshold_rungs(&at_the_minimum).last(), Some(min_threshold));
        let admitted: usize = detect_with(
            Box::new(CpuCornerScan::default()),
            &image,
            &grid,
            &counts,
            &at_the_minimum,
            &Masks::default(),
            4096,
        )
        .corners
        .len();

        let label: String = format!("ladder {max_threshold}/{min_threshold}");
        let agreed: Agreement = detection_agrees(
            &image,
            &grid,
            &counts,
            &config,
            &Masks::default(),
            4096,
            &label,
        );
        assert!(agreed.selections > 0, "{label} took the band walk");
        assert!(
            admitted > agreed.corners,
            "{label}: a rung at {min_threshold} admits {admitted} corners against the last \
             rung's {}, so the two bounds are not separable on this frame",
            agreed.corners
        );
    }
}

/// A frame a packed key cannot name takes the band walk.
///
/// The key keeps twelve bits each for the row and the column, so
/// [`CELL_KEY_LIMIT`] pixels on a side is where the device path has to give up
/// rather than lose a coordinate. The guard is on the frame, not on the grid, so
/// a wide short frame is enough to reach it.
#[test]
fn a_frame_at_the_key_limit_takes_the_band_walk() {
    let (width, height, cell): (usize, usize, usize) = (CELL_KEY_LIMIT, 96, 32);
    let image: ImageU16 = cornered_image(width, height);
    let grid: CellGrid = CellGrid::new(width, height, cell).unwrap();
    let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
    let agreed: Agreement = detection_agrees(
        &image,
        &grid,
        &counts,
        &detector_config(0.0),
        &Masks::default(),
        4096,
        "a frame at the key limit",
    );
    assert_eq!(
        agreed.selections, 0,
        "{width} pixels wide is past what a packed key can name"
    );
    assert!(agreed.bands > 0, "so the band walk has to have run");
    assert!(agreed.corners > 0, "and it has to have found something");
}

/// More than one point per cell takes the band walk.
///
/// The exactness argument is a one-point-per-cell argument: with a larger budget
/// the ladder decides how many corners a cell contributes and one key cannot say.
#[test]
fn a_budget_over_one_point_per_cell_takes_the_band_walk() {
    let image: ImageU16 = mio10_frame(1, 0);
    let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
    let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
    let config: DetectorConfig = DetectorConfig {
        num_points_cell: 2,
        ..detector_config(472.0)
    };
    let agreed: Agreement = detection_agrees(
        &image,
        &grid,
        &counts,
        &config,
        &Masks::default(),
        4096,
        "two points per cell",
    );
    assert_eq!(agreed.selections, 0, "two points per cell is the band walk");
    assert!(agreed.bands > 0, "so the band walk has to have run");
    assert!(agreed.corners > 0, "and it has to have found something");
}

/// One scanner, four camera slots.
///
/// The device path keeps a key buffer per camera beside the three the band path
/// keeps, and the frontend calls the detector with the frameset's own camera
/// index — so a rig with four cameras reaches slot 3. The scratch is reused
/// across the four calls, which is what the frontend does and what makes the
/// per-camera buffers a thing that can be got wrong.
#[test]
fn the_gpu_cell_selection_holds_for_every_camera_slot() {
    let config: DetectorConfig = detector_config(472.0);
    let mut host: DetectorScratch = DetectorScratch::default();
    let mut device: DetectorScratch =
        DetectorScratch::with_scanner(Box::new(GpuCornerScan::new(gpu_client().unwrap()).unwrap()));

    for camera in 0..4 {
        // Two frames alternating, so consecutive slots hold different pixels.
        let image: ImageU16 = mio10_frame(camera % 2, camera % 2);
        let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
        let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
        let occupancy: Occupancy<'_> = Occupancy {
            counts: &counts,
            rows: grid.rows,
            columns: grid.columns,
        };
        let mut want: KeypointsData = KeypointsData::default();
        let mut got: KeypointsData = KeypointsData::default();
        for (scratch, out) in [(&mut host, &mut want), (&mut device, &mut got)] {
            detect_keypoints_with_cells(
                &image,
                camera,
                &grid,
                &occupancy,
                &config,
                &Masks::default(),
                4096,
                scratch,
                out,
            )
            .unwrap();
        }
        assert!(!want.corners.is_empty(), "camera {camera} found nothing");
        assert_eq!(got.corners, want.corners, "camera {camera}: corners");
        assert_eq!(got.responses, want.responses, "camera {camera}: responses");
    }
}

/// The batched preparation answers exactly what the per-camera call does.
///
/// [`CornerScan::submit_cells`] launches every camera's selection at once and
/// [`CornerScan::take_cells`] downloads them together, which is a scheduling
/// change and must be nothing
/// else: the keys it hands each camera have to be the ones that camera's own
/// `select_cells` would have read. Two different MIO10 frames in the two camera
/// slots, so a batch that crossed its cameras over would be caught.
#[test]
fn the_batched_preparation_answers_what_the_per_camera_call_does() {
    let config: DetectorConfig = detector_config(472.0);
    let images: [ImageU16; 2] = [mio10_frame(0, 0), mio10_frame(1, 1)];
    let grid: CellGrid = CellGrid::new(images[0].width(), images[0].height(), 50).unwrap();
    let selects: Vec<Option<CellSelect>> = images
        .iter()
        .map(|image| slam_rs::frontend::detect::cell_select(image, &grid, &config))
        .collect();
    assert!(
        selects.iter().all(Option::is_some),
        "the rig is device shaped"
    );

    // The selection's grid is the cells the walk visits, one less each way than
    // the occupancy matrix: `x_stop` is the last cell's left edge.
    let cells: usize = ((grid.x_stop - grid.x_start) / grid.cell + 1)
        * ((grid.y_stop - grid.y_start) / grid.cell + 1);

    let mut scanner: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let mut alone: Vec<Vec<u32>> = Vec::new();
    for (camera, image) in images.iter().enumerate() {
        let mut keys: Vec<u32> = Vec::new();
        scanner
            .select_cells(camera, image, &selects[camera].unwrap(), &mut keys)
            .unwrap();
        assert_eq!(keys.len(), cells, "camera {camera}");
        alone.push(keys);
    }

    scanner.submit_cells(&images, &selects).unwrap();
    scanner.take_cells().unwrap();
    for (camera, image) in images.iter().enumerate() {
        let mut keys: Vec<u32> = Vec::new();
        scanner
            .select_cells(camera, image, &selects[camera].unwrap(), &mut keys)
            .unwrap();
        assert_eq!(keys, alone[camera], "camera {camera} out of the batch");
    }
    assert_ne!(alone[0], alone[1], "the two cameras hold the same frame");
}

/// The scanner's keys come home inside the tracker's download.
///
/// The relay is the whole of D78: `submit_cells` launches and downloads
/// nothing, and the next stage to synchronise is what brings the keys back —
/// here a `collect` with no tracking pass in flight at all, which is the
/// weakest form of the claim and so the sharpest test of it. A relay that
/// dropped them would be invisible from the values alone, because `take_cells`
/// reads for itself when nothing was delivered; so what is asserted is that the
/// scanner made no read of its own, and that the keys are still the ones its
/// own download would have given.
#[test]
fn the_tracker_download_carries_the_scanner_keys() {
    let config: DetectorConfig = detector_config(472.0);
    let images: [ImageU16; 2] = [mio10_frame(0, 0), mio10_frame(1, 1)];
    let grid: CellGrid = CellGrid::new(images[0].width(), images[0].height(), 50).unwrap();
    let selects: Vec<Option<CellSelect>> = images
        .iter()
        .map(|image| slam_rs::frontend::detect::cell_select(image, &grid, &config))
        .collect();

    // What the scanner answers when it downloads for itself: no relay wired.
    let mut alone: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    alone.submit_cells(&images, &selects).unwrap();
    alone.take_cells().unwrap();
    let mut want: Vec<Vec<u32>> = Vec::new();
    for (camera, image) in images.iter().enumerate() {
        let mut keys: Vec<u32> = Vec::new();
        alone
            .select_cells(camera, image, &selects[camera].unwrap(), &mut keys)
            .unwrap();
        assert!(!keys.is_empty(), "camera {camera} took the device path");
        want.push(keys);
    }

    // And the same two cameras with the tracker's `collect` in between.
    let relay: ReadRelay = ReadRelay::default();
    let mut scanner: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let mut tracker: GpuPatchTracker<Pattern51, _> =
        GpuPatchTracker::new(gpu_client().unwrap(), 512, 4, 5, 4.0, 2).unwrap();
    scanner.share_reads(relay.clone());
    tracker.share_reads(relay.clone());

    scanner.submit_cells(&images, &selects).unwrap();
    assert_eq!(relay.waiting(), 2, "both cameras are launched and unread");
    assert_eq!(relay.carried(), 0, "nothing has downloaded them yet");
    tracker.collect(&mut []).unwrap();
    assert_eq!(relay.waiting(), 0, "the collect took the staged handles");
    assert_eq!(relay.carried(), 2, "and left both cameras' keys behind");
    scanner.take_cells().unwrap();
    assert_eq!(relay.carried(), 0, "which the scanner then took");
    for (camera, image) in images.iter().enumerate() {
        let mut keys: Vec<u32> = Vec::new();
        scanner
            .select_cells(camera, image, &selects[camera].unwrap(), &mut keys)
            .unwrap();
        assert_eq!(keys, want[camera], "camera {camera} through the relay");
    }
}

/// A prepared selection is spent by the call that reads it, and a camera the
/// batch skipped still answers for itself.
///
/// The keys are cached on the scanner between `take_cells` and the
/// `select_cells` that takes them, so the thing that must not happen is a
/// leftover answering a later frameset. Reading twice, and preparing a rig where
/// only one camera is offered, are the two ways that could happen.
#[test]
fn a_prepared_selection_is_spent_once() {
    let config: DetectorConfig = detector_config(472.0);
    let images: [ImageU16; 2] = [mio10_frame(0, 0), mio10_frame(1, 1)];
    let grid: CellGrid = CellGrid::new(images[0].width(), images[0].height(), 50).unwrap();
    let select: CellSelect =
        slam_rs::frontend::detect::cell_select(&images[0], &grid, &config).unwrap();
    let cells: usize = ((grid.x_stop - grid.x_start) / grid.cell + 1)
        * ((grid.y_stop - grid.y_start) / grid.cell + 1);

    let mut scanner: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    // Only camera 0 is offered, so camera 1 has nothing prepared for it.
    scanner
        .submit_cells(&images, &[Some(select), None])
        .unwrap();
    scanner.take_cells().unwrap();

    let mut first: Vec<u32> = Vec::new();
    scanner
        .select_cells(0, &images[0], &select, &mut first)
        .unwrap();
    let mut again: Vec<u32> = Vec::new();
    scanner
        .select_cells(0, &images[1], &select, &mut again)
        .unwrap();
    let mut independent = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let mut expected = Vec::new();
    independent
        .select_cells(0, &images[1], &select, &mut expected)
        .unwrap();
    assert_ne!(expected, first, "the second image must have different keys");
    assert_eq!(again, expected, "the second read must scan the new image");

    let mut second: Vec<u32> = Vec::new();
    scanner
        .select_cells(1, &images[1], &select, &mut second)
        .unwrap();
    assert_eq!(second.len(), cells);
    assert_ne!(second, first, "camera 1 answered with camera 0's frame");
}
