#![allow(clippy::unwrap_used)]

use super::*;
use crate::gpu::{CORNER_SCAN_READ, GpuRuntime, arm_fault_at, gpu_client};

/// Bright squares on a flat background: a frame FAST finds corners in,
/// which is what `frontend::detect`'s own fixture is for the CPU lane.
fn dotted_image(width: usize, height: usize) -> ImageU16 {
    let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
    for y in 0..height {
        for x in 0..width {
            image.set(x, y, 60u16 << 8);
        }
    }
    let mut cy: usize = 20;
    while cy + 5 < height {
        let mut cx: usize = 20;
        while cx + 5 < width {
            for dy in 0..5 {
                for dx in 0..5 {
                    image.set(cx + dx, cy + dy, 200u16 << 8);
                }
            }
            cx += 20;
        }
        cy += 20;
    }
    image
}

/// One band of the grid the detector walks, at the first rung.
fn band(y: usize) -> BandRequest {
    BandRequest {
        row: 0,
        rung: 0,
        y,
        rows: 44,
        threshold: 5,
    }
}

/// A failed scan leaves nothing readable, not the previous frame's corners.
///
/// The new buffers replace the old ones only where the scan succeeds, so a
/// scan that fails on the far side of the geometry it records would answer
/// the next band with the *last* frame's corners under this frame's request.
/// The fault is armed at the download, which is where a lost device really
/// lands, and what the band returns afterwards is the refusal a band before
/// any scan returns.
#[test]
fn a_failed_scan_leaves_no_band_readable() {
    let mut scanner: GpuCornerScan<GpuRuntime> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    scanner.scan(0, &dotted_image(512, 128)).unwrap();
    assert!(
        !scanner.band(band(3)).unwrap().is_empty(),
        "the dotted frame has corners, so the failed scan below has something to leak"
    );

    arm_fault_at(CORNER_SCAN_READ);
    assert_eq!(
        scanner.scan(0, &dotted_image(512, 128)).unwrap_err(),
        DetectError::Gpu(GpuError::DeviceLost {
            what: "corner scan"
        })
    );
    assert_eq!(scanner.band(band(3)).unwrap_err(), DetectError::NotScanned);
}

/// A failed scan that changes the geometry refuses rather than indexes.
///
/// The dangerous half of the same state: a rig whose cameras differ in size
/// — which the port supports — would walk 512-wide rows of the last frame
/// with the 960-wide stride of this one, and a band deep in the taller frame
/// runs off the end of the buffer.
#[test]
fn a_failed_scan_that_changes_the_geometry_does_not_panic() {
    let mut scanner: GpuCornerScan<GpuRuntime> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    scanner.scan(0, &dotted_image(512, 128)).unwrap();

    arm_fault_at(CORNER_SCAN_READ);
    assert!(scanner.scan(1, &dotted_image(960, 240)).is_err());
    assert_eq!(
        scanner.band(band(150)).unwrap_err(),
        DetectError::NotScanned
    );
}
#[test]
fn selection_failures_invalidate_the_batch_and_allow_retry() {
    use crate::frontend::detect::{CellGrid, DetectorConfig, cell_select};
    let images = [dotted_image(512, 128), dotted_image(512, 128)];
    let config = DetectorConfig {
        num_points_cell: 1,
        min_threshold: 5,
        max_threshold: 40,
        safe_radius: 0.0,
    };
    let grid = CellGrid::new(512, 128, 50).unwrap();
    let selection = cell_select(&images[0], &grid, &config).unwrap();
    let selects = [Some(selection), Some(selection)];
    let mut scanner = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    for site in ["selection camera submitted", super::super::BLOCKING_READ] {
        if site == "selection camera submitted" {
            arm_fault_at(site);
            assert!(scanner.submit_cells(&images, &selects).is_err());
        } else {
            scanner.submit_cells(&images, &selects).unwrap();
            arm_fault_at(site);
            assert!(scanner.take_cells().is_err());
        }
        assert!(
            scanner
                .cameras
                .iter()
                .all(|camera| matches!(camera.selection, Selection::Empty))
        );
        let allocations = scanner.buffer_allocations();
        scanner.submit_cells(&images, &selects).unwrap();
        scanner.take_cells().unwrap();
        assert!(
            scanner
                .cameras
                .iter()
                .all(|camera| matches!(camera.selection, Selection::Ready(_)))
        );
        scanner.submit_cells(&images, &selects).unwrap();
        scanner.take_cells().unwrap();
        assert!(scanner.buffer_allocations() <= allocations + 2);
    }
}
