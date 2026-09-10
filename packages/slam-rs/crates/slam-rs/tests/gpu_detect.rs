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

use slam_rs::frontend::detect::{
    CellGrid, CornerScan, CpuCornerScan, DetectorConfig, DetectorScratch, KeypointsData, Masks,
    Occupancy, Rect, detect_keypoints_with_cells,
};
use slam_rs::gpu::{GpuCornerScan, gpu_client};
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
) -> usize {
    let want: KeypointsData = detect_with(
        Box::new(CpuCornerScan::default()),
        image,
        grid,
        counts,
        config,
        masks,
        budget,
    );
    let got: KeypointsData = detect_with(
        Box::new(GpuCornerScan::new(gpu_client().unwrap()).unwrap()),
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
    want.corners.len()
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
        let found: usize = detection_agrees(
            &image,
            &grid,
            &empty,
            &config,
            &Masks::default(),
            4096,
            &format!("cam{camera} empty"),
        );
        assert!(
            found > 30,
            "cam{camera} found only {found} corners, which would make the equality vacuous"
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
    detection_agrees(
        &image,
        &grid,
        &counts,
        &detector_config(472.0),
        &straddling,
        4096,
        "a mask across a cell boundary",
    );

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
/// The right and bottom clamps are the two the cell window carries and the two
/// a square 960x960 frame never exercises: `min(x + cell - 3, width - 3)` bites
/// only where the last cell runs past the image, and kornia's in-block filter is
/// on at 960 and off at 512, which changes the candidate set the selection reads.
#[test]
fn the_gpu_cell_selection_matches_the_host_walk_on_clamped_grids() {
    for (width, height, cell) in [
        (960usize, 240usize, 50usize),
        (512, 192, 32),
        (517, 193, 50),
    ] {
        let image: ImageU16 = cornered_image(width, height);
        let grid: CellGrid = CellGrid::new(width, height, cell).unwrap();
        let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
        let found: usize = detection_agrees(
            &image,
            &grid,
            &counts,
            &detector_config(0.0),
            &Masks::default(),
            4096,
            &format!("{width}x{height} cell {cell}"),
        );
        assert!(found > 0, "{width}x{height} found nothing to compare");
    }
}
