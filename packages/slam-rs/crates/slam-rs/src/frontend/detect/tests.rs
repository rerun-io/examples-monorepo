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
            let background: f64 = 60.0 + 25.0 * (x as f64 * 0.09).sin() * (y as f64 * 0.07).cos();
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

///  on a 960x960 frame with `grid_size = 50`.
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

/// An image smaller than one cell must be refused before grid arithmetic underflows.
#[test]
fn an_image_narrower_than_a_cell_has_no_grid() {
    assert!(CellGrid::new(30, 200, 50).is_none());
    assert!(CellGrid::new(200, 30, 50).is_none());
    assert!(CellGrid::new(200, 200, 0).is_none());
}

/// A coordinate just left of `x_start` truncates to column zero.
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

/// The per-cell budget is `num_points_cell`.
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

/// The threshold ladder must terminate even with a non-positive requested minimum.
/// A blank frame walks every rung because no cell fills. The result must match
/// a minimum of one, which is the enforced floor.
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

/// an occupied cell is skipped whole.
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

/// a masked corner is dropped.
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

/// outside `safe_radius` of the image centre, nothing is kept.
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

/// Skip detection cells outside the allocated occupancy shape (trap 15).
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

/// masked means inside *any* rectangle, half-open.
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
/// the pixel stops being a corner, which is what kornia
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

#[derive(Debug)]
struct MalformedSelection(usize);

impl CornerScan for MalformedSelection {
    fn scan(&mut self, _: usize, _: &ImageU16) -> Result<(), DetectError> {
        panic!("a malformed selection must not fall back to scanning")
    }

    fn band(&mut self, _: BandRequest) -> Result<&[FastCorner], DetectError> {
        panic!("a malformed selection must not fall back to a band")
    }

    fn select_cells(
        &mut self,
        _: usize,
        _: &ImageU16,
        _: &CellSelect,
        out: &mut Vec<u32>,
    ) -> Result<SelectionStatus, DetectError> {
        out.resize(self.0, NO_CELL_WINNER);
        Ok(SelectionStatus::Selected)
    }
}

#[test]
fn malformed_selection_lengths_are_refused_without_fallback() {
    let image = ImageU16::zeros(200, 200).unwrap();
    let grid = CellGrid::new(200, 200, 50).unwrap();
    let counts = vec![0; grid.rows * grid.columns];
    let occupancy = Occupancy {
        counts: &counts,
        rows: grid.rows,
        columns: grid.columns,
    };
    for actual in [0, 15, 17] {
        let mut scratch = DetectorScratch::with_scanner(Box::new(MalformedSelection(actual)));
        let mut out = KeypointsData::default();
        assert_eq!(
            detect_keypoints_with_cells(
                &image,
                0,
                &grid,
                &occupancy,
                &config(),
                &Masks::default(),
                BUDGET,
                &mut scratch,
                &mut out
            ),
            Err(DetectError::SelectionLength {
                actual,
                expected: 16
            })
        );
    }
}
