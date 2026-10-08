use super::*;

/// Run this binary both with and without KORNIA_FAST_NEON=0 on aarch64.
#[test]
#[cfg(target_arch = "aarch64")]
fn default_cell_selection_follows_kornias_neon_gate() {
    use kornia_staging_imgproc::features::SelectionStatus;
    let image = slam_rs::image::zeros(64, 64).unwrap();
    let select = CellSelect {
        grid: CellGrid::new(64, 64, 32).unwrap(),
        threshold: 5,
        safe_radius: 0.0,
    };
    let expected = if std::env::var("KORNIA_FAST_NEON").map_or(true, |value| value != "0") {
        SelectionStatus::Selected
    } else {
        SelectionStatus::Unsupported
    };
    assert_eq!(
        CpuCornerScan::default()
            .select_cells(0, &image, &select, None, &mut Vec::new())
            .unwrap(),
        expected
    );
}

#[test]
fn cell_selection_refuses_unrepresentable_keys_and_thresholds() {
    use kornia_staging_imgproc::features::{LOWEST_THRESHOLD_RUNG, SelectionStatus};
    for (width, height, threshold) in [
        (CELL_KEY_LIMIT, 64, 5),
        (64, CELL_KEY_LIMIT, 5),
        (64, 64, LOWEST_THRESHOLD_RUNG - 1),
    ] {
        let image = slam_rs::image::zeros(width, height).unwrap();
        let grid = CellGrid::new(width, height, 32).unwrap();
        let select = CellSelect {
            grid,
            threshold,
            safe_radius: 0.0,
        };
        let counts = vec![0; grid.rows * grid.columns];
        let occupancy = Occupancy {
            counts: &counts,
            rows: grid.rows,
            columns: grid.columns,
        };
        let masked = vec![false; grid.dimensions().0 * grid.dimensions().1];
        let mut scanner = CpuCornerScan::with_cell_selection(true);
        let mut keys = Vec::new();
        assert_eq!(
            scanner
                .select_cells(0, &image, &select, None, &mut keys)
                .unwrap(),
            SelectionStatus::Unsupported
        );
        assert_eq!(
            scanner
                .select_cells(0, &image, &select, Some((&occupancy, &masked)), &mut keys)
                .unwrap(),
            SelectionStatus::Unsupported
        );
    }
}

#[test]
fn cell_selection_matches_the_band_walk_on_random_strided_images() {
    let mut state = 0x8912_ab34u32;
    for (width, height, cell) in [
        (97, 83, 7),
        (128, 95, 17),
        (157, 97, 21),
        (167, 97, 22),
        (181, 97, 23),
        (640, 123, 50),
        (799, 129, 37),
        (800, 129, 50),
        (817, 127, 60),
        (960, 129, 50),
    ] {
        let mut image = slam_rs::image::from_u8_strided(
            &vec![0; (width + 13) * height],
            width,
            height,
            width + 13,
        )
        .unwrap();
        for y in 0..height {
            for x in 0..width {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                image.set_pixel(x, y, 0, (state >> 16) as u16).unwrap();
            }
        }
        let grid = CellGrid::new(width, height, cell).unwrap();
        let counts: Vec<i32> = (0..grid.rows * grid.columns)
            .map(|i| i32::from(i % 5 == 0))
            .collect();
        let masks = CellMasks {
            masks: grid
                .cells()
                .enumerate()
                .filter(|(i, _)| i % 3 == 0)
                .map(|(_, (column, row))| MaskRect {
                    x: (grid.x_start + column * cell) as f32,
                    y: (grid.y_start + row * cell) as f32,
                    w: cell as f32,
                    h: cell as f32,
                })
                .collect(),
        };
        for threshold in [1, 5, 10, 40, 127, 254, 255, 300] {
            detection_agrees(
                DetectionCase {
                    image: &image,
                    grid: &grid,
                    counts: &counts,
                    config: &CenteredCellConfig {
                        min_threshold: threshold,
                        max_threshold: threshold,
                        ..detector_config(0.0)
                    },
                    masks: &masks,
                    budget: 4096,
                    label: &format!("random {width}x{height}, cell {cell}, threshold {threshold}"),
                },
                ExpectedPath::CellSelection,
            );
        }
    }
}

#[test]
fn cell_selection_keeps_scan_order_for_ties_and_rejects_plateaus() {
    let mut image = slam_rs::image::zeros(100, 100).unwrap();
    for y in 0..100 {
        for x in 0..100 {
            image.set_pixel(x, y, 0, 128 << 8).unwrap();
        }
    }
    for (x, y) in [(35, 35), (65, 35), (35, 65)] {
        image.set_pixel(x, y, 0, 0).unwrap();
    }
    let grid = CellGrid::new(100, 100, 100).unwrap();
    let counts = vec![0; grid.rows * grid.columns];
    let detect = |image: &Image<u16, 1>| {
        detect_with(
            Box::new(AppScan(CpuCornerScan::with_cell_selection(true))),
            image,
            &grid,
            &counts,
            &detector_config(0.0),
            &CellMasks::default(),
            1,
        )
    };
    let tied = detect(&image);
    assert_eq!(tied.corners, [[35.0, 35.0]]);
    assert_eq!(tied.responses, [127.0]);
    image.set_pixel(36, 35, 0, 0).unwrap();
    let plateau = detect(&image);
    assert_eq!(plateau.corners, [[65.0, 35.0]]);
    assert_eq!(plateau.responses, [127.0]);
}

/// Every cell of a real MIO10 frameset, both cameras, empty and half full.
#[test]
fn the_cell_selection_matches_the_host_walk_on_a_real_frameset() {
    let config: CenteredCellConfig = detector_config(472.0);
    for camera in 0..2 {
        let image: Image<u16, 1> = common::mio10_frame(0, camera);
        let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
        let cells: usize = grid.rows * grid.columns;

        // Nothing tracked yet: every cell of the grid is detected in.
        let empty: Vec<i32> = vec![0; cells];
        let agreed: usize = detection_agrees(
            DetectionCase {
                image: &image,
                grid: &grid,
                counts: &empty,
                config: &config,
                masks: &CellMasks::default(),
                budget: 4096,
                label: &format!("cam{camera} empty"),
            },
            ExpectedPath::CellSelection,
        );
        assert!(
            agreed > 30,
            "cam{camera} found only {} corners, which would make the equality vacuous",
            agreed
        );

        // Half the cells already hold a feature, which is the steady state: the
        // skip has to land on the same cells on both lanes.
        let mut busy: Vec<i32> = vec![0; cells];
        for (index, count) in busy.iter_mut().enumerate() {
            *count = i32::from(index.is_multiple_of(3));
        }
        detection_agrees(
            DetectionCase {
                image: &image,
                grid: &grid,
                counts: &busy,
                config: &config,
                masks: &CellMasks::default(),
                budget: 4096,
                label: &format!("cam{camera} occupied"),
            },
            ExpectedPath::CellSelection,
        );
    }
}

/// The gates the kernel took over, one at a time, and the budget the host keeps.
#[test]
fn the_cell_selection_applies_the_same_gates() {
    let image: Image<u16, 1> = common::mio10_frame(1, 0);
    let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
    let counts: Vec<i32> = vec![0; grid.rows * grid.columns];

    // `safe_radius = 0` switches the distance gate off entirely, and 200 makes
    // it bite on a 960x960 frame where 472 leaves most of the grid alone.
    for radius in [0.0f32, 200.0, 472.0] {
        detection_agrees(
            DetectionCase {
                image: &image,
                grid: &grid,
                counts: &counts,
                config: &detector_config(radius),
                masks: &CellMasks::default(),
                budget: 4096,
                label: &format!("safe radius {radius}"),
            },
            ExpectedPath::CellSelection,
        );
    }

    // `cam0OverlapCellsMasksForCam`'s own shape: `cell` x `cell` rectangles at
    // the cell origins. Every third one, so masked and clear cells interleave.
    let mut masks: CellMasks = CellMasks::default();
    let mut index: usize = 0;
    let mut y: usize = grid.y_start;
    while y <= grid.y_stop {
        let mut x: usize = grid.x_start;
        while x <= grid.x_stop {
            if index.is_multiple_of(3) {
                masks.masks.push(MaskRect {
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
        DetectionCase {
            image: &image,
            grid: &grid,
            counts: &counts,
            config: &detector_config(472.0),
            masks: &masks,
            budget: 4096,
            label: "cell-aligned masks",
        },
        ExpectedPath::CellSelection,
    );

    // A rectangle that straddles a cell boundary is the mixed-geometry rig's
    // shape: the device path has to refuse it and the band walk has to answer,
    // which is the same answer either way.
    let mut straddling: CellMasks = CellMasks::default();
    straddling.masks.push(MaskRect {
        x: (grid.x_start + 17) as f32,
        y: (grid.y_start + 21) as f32,
        w: grid.cell as f32,
        h: grid.cell as f32,
    });
    detection_agrees(
        DetectionCase {
            image: &image,
            grid: &grid,
            counts: &counts,
            config: &detector_config(472.0),
            masks: &straddling,
            budget: 4096,
            label: "a mask across a cell boundary",
        },
        ExpectedPath::BandWalk,
    );

    // Capacity truncation must retain detector scan order.
    for budget in [1usize, 7, 40] {
        detection_agrees(
            DetectionCase {
                image: &image,
                grid: &grid,
                counts: &counts,
                config: &detector_config(472.0),
                masks: &CellMasks::default(),
                budget,
                label: &format!("budget {budget}"),
            },
            ExpectedPath::CellSelection,
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
fn the_cell_selection_matches_the_host_walk_on_uneven_frames() {
    for (width, height, cell) in [
        (960usize, 240usize, 50usize),
        (512, 192, 32),
        (517, 193, 50),
        (640, 480, 50),
        (641, 479, 64),
    ] {
        let image: Image<u16, 1> = cornered_image(width, height);
        let grid: CellGrid = CellGrid::new(width, height, cell).unwrap();
        // The precondition the clamp test below needs and this one does not
        // have: `CellGrid::new` cannot produce a cell that runs past the image.
        assert!(grid.x_stop + cell <= width && grid.y_stop + cell <= height);
        let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
        let agreed: usize = detection_agrees(
            DetectionCase {
                image: &image,
                grid: &grid,
                counts: &counts,
                config: &detector_config(0.0),
                masks: &CellMasks::default(),
                budget: 4096,
                label: &format!("{width}x{height} cell {cell}"),
            },
            ExpectedPath::CellSelection,
        );
        assert!(agreed > 0, "{width}x{height} found nothing to compare");
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
fn the_cell_selection_matches_the_host_walk_on_overhanging_cells() {
    let (width, height, cell): (usize, usize, usize) = (200, 150, 50);
    let image: Image<u16, 1> = cornered_image(width, height);

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
        let agreed: usize = detection_agrees(
            DetectionCase {
                image: &image,
                grid: &grid,
                counts: &counts,
                config: &detector_config(0.0),
                masks: &CellMasks::default(),
                budget: 4096,
                label,
            },
            ExpectedPath::CellSelection,
        );
        assert!(agreed > 0, "{label} found nothing to compare");
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
fn the_cell_selection_stops_at_the_last_rung_the_walk_visits() {
    let image: Image<u16, 1> = common::mio10_frame(0, 0);
    let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
    let counts: Vec<i32> = vec![0; grid.rows * grid.columns];

    for (max_threshold, min_threshold, last_rung) in [(40i32, 6i32, 10i32), (32, 5, 8)] {
        let config: CenteredCellConfig = CenteredCellConfig {
            max_threshold,
            min_threshold,
            ..detector_config(472.0)
        };
        assert_eq!(threshold_rungs(&config).last(), Some(last_rung));

        // The ladder that stops at the configured minimum instead: one rung, at
        // `min_threshold`. This is what a device given the wrong bound detects.
        let at_the_minimum: CenteredCellConfig = CenteredCellConfig {
            max_threshold: min_threshold,
            ..config
        };
        assert_eq!(threshold_rungs(&at_the_minimum).last(), Some(min_threshold));
        let admitted: usize = detect_with(
            Box::new(AppScan(CpuCornerScan::with_cell_selection(true))),
            &image,
            &grid,
            &counts,
            &at_the_minimum,
            &CellMasks::default(),
            4096,
        )
        .corners
        .len();

        let label: String = format!("ladder {max_threshold}/{min_threshold}");
        let agreed: usize = detection_agrees(
            DetectionCase {
                image: &image,
                grid: &grid,
                counts: &counts,
                config: &config,
                masks: &CellMasks::default(),
                budget: 4096,
                label: &label,
            },
            ExpectedPath::CellSelection,
        );
        assert!(
            admitted > agreed,
            "{label}: a rung at {min_threshold} admits {admitted} corners against the last \
         rung's {}, so the two bounds are not separable on this frame",
            agreed
        );
    }
}

/// A frame a packed key cannot name takes the band walk.
///
/// The key keeps twelve bits each for the row and the column, so
/// [`CELL_KEY_LIMIT`] pixels on a side is where cell selection has to give up
/// rather than lose a coordinate. The guard is on the frame, not on the grid, so
/// a wide short frame is enough to reach it.
#[test]
fn a_frame_at_the_key_limit_takes_the_band_walk() {
    let (width, height, cell): (usize, usize, usize) = (CELL_KEY_LIMIT, 96, 32);
    let image: Image<u16, 1> = cornered_image(width, height);
    let grid: CellGrid = CellGrid::new(width, height, cell).unwrap();
    let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
    let agreed: usize = detection_agrees(
        DetectionCase {
            image: &image,
            grid: &grid,
            counts: &counts,
            config: &detector_config(0.0),
            masks: &CellMasks::default(),
            budget: 4096,
            label: "a frame at the key limit",
        },
        ExpectedPath::BandWalk,
    );
    assert!(agreed > 0, "and it has to have found something");
}

/// More than one point per cell takes the band walk.
///
/// The exactness argument is a one-point-per-cell argument: with a larger budget
/// the ladder decides how many corners a cell contributes and one key cannot say.
#[test]
fn a_budget_over_one_point_per_cell_takes_the_band_walk() {
    let image: Image<u16, 1> = common::mio10_frame(1, 0);
    let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
    let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
    let config: CenteredCellConfig = CenteredCellConfig {
        num_points_cell: 2,
        ..detector_config(472.0)
    };
    let agreed: usize = detection_agrees(
        DetectionCase {
            image: &image,
            grid: &grid,
            counts: &counts,
            config: &config,
            masks: &CellMasks::default(),
            budget: 4096,
            label: "two points per cell",
        },
        ExpectedPath::BandWalk,
    );
    assert!(agreed > 0, "and it has to have found something");
}
