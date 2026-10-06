#![allow(clippy::unwrap_used)]

use super::*;
use crate::gpu::{CORNER_SCAN_READ, GpuRuntime, arm_fault_at, gpu_client};

/// Shared-arena batches preserve exact winners across camera subsets and refills.
#[test]
fn packed_camera_selection_is_exact_in_one_dispatch() {
    use crate::frontend::input::PackedImages;
    use crate::gpu::GpuPyramidBuilder;
    use crate::pyramid::PyramidBuilder;
    use kornia_staging_imgproc::features::{CellGrid, CpuCornerScan};

    let client = gpu_client().unwrap();
    let mut builder = GpuPyramidBuilder::new(client.clone(), Default::default());
    let mut scanner = GpuCornerScan::new(client, Default::default()).unwrap();
    let mut cpu = CpuCornerScan::with_cell_selection(true);
    let mut packed = PackedImages::default();
    let mut state = 0x8912_ab34u32;
    for (width, height, cell) in [(640, 480, 50), (517, 193, 37)] {
        let mut pyramids: Vec<_> = (0..4)
            .map(|_| builder.allocate(width, height, 3).unwrap())
            .collect();
        let select = CellSelect {
            grid: CellGrid::new(width, height, cell).unwrap(),
            threshold: 5,
            safe_radius: 0.0,
        };
        for frame in 0..2 {
            let cameras: Vec<Vec<u8>> = (0..4)
                .map(|_| {
                    let mut bytes = vec![0; (width + 13) * height];
                    for byte in &mut bytes {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        *byte = (state >> 24) as u8;
                    }
                    bytes
                })
                .collect();
            let views: Vec<_> = cameras
                .iter()
                .map(|bytes| crate::ImageView {
                    data: bytes,
                    width,
                    height,
                    stride: width + 13,
                })
                .collect();
            let images: Vec<_> = views
                .iter()
                .map(|view| {
                    let mut image = crate::image::empty();
                    crate::image::fill_from_u8_strided(
                        &mut image,
                        view.data,
                        width,
                        height,
                        view.stride,
                    )
                    .unwrap();
                    image
                })
                .collect();
            packed.fill(&views);
            builder.build_packed(&packed, &mut pyramids).unwrap();
            scanner.use_level0(&mut builder);
            for (first, end) in [(0, 1), (1, 4), (0, 4)] {
                let selects: Vec<_> = (0..4)
                    .map(|camera| (first..end).contains(&camera).then_some(select))
                    .collect();
                scanner
                    .submit_cells(FrameImages::Packed(&packed), &selects)
                    .unwrap();
                assert!(
                    scanner.selection_stride.is_some(),
                    "shared-arena batch required: {width}x{height}, frame {frame}, {first}..{end}"
                );
                assert!(
                    matches!(&scanner.reads, SelectionReads::Pending(handles) if handles.len() == 1)
                );
                scanner.take_cells().unwrap();
                for (camera, image) in images.iter().enumerate().take(end).skip(first) {
                    assert!(
                        matches!(scanner.cameras[camera].selection, Selection::Ready(ready) if ready == select)
                    );
                    let mut got = Vec::new();
                    let mut want = Vec::new();
                    scanner
                        .select_cells(camera, image, &select, None, &mut got)
                        .unwrap();
                    cpu.select_cells(camera, image, &select, None, &mut want)
                        .unwrap();
                    assert!(want.iter().any(Option::is_some));
                    assert_eq!(
                        got, want,
                        "{width}x{height}, frame {frame}, camera {camera}, batch {first}..{end}"
                    );
                }
            }
        }
    }
}

/// Bright squares on a flat background: a frame FAST finds corners in,
/// which is what `frontend::detect`'s own fixture is for the CPU lane.
fn dotted_image(width: usize, height: usize) -> Image<u16, 1> {
    let mut image: Image<u16, 1> = crate::image::zeros(width, height).unwrap();
    for y in 0..height {
        for x in 0..width {
            image.set_pixel(x, y, 0, 60u16 << 8).unwrap();
        }
    }
    let mut cy: usize = 20;
    while cy + 5 < height {
        let mut cx: usize = 20;
        while cx + 5 < width {
            for dy in 0..5 {
                for dx in 0..5 {
                    image.set_pixel(cx + dx, cy + dy, 0, 200u16 << 8).unwrap();
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
    let mut scanner: GpuCornerScan<GpuRuntime> =
        GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
    scanner.scan(0, &dotted_image(512, 128)).unwrap();
    assert!(
        !scanner.band(band(3)).unwrap().is_empty(),
        "the dotted frame has corners, so the failed scan below has something to leak"
    );

    arm_fault_at(CORNER_SCAN_READ);
    assert_eq!(
        scanner.scan(0, &dotted_image(512, 128)).unwrap_err(),
        FrontendError::Gpu(GpuError::DeviceLost {
            what: "corner scan"
        })
    );
    assert_eq!(
        scanner.band(band(3)).unwrap_err(),
        FrontendError::Detect(CenteredCellError::NotScanned)
    );
}

/// A failed scan that changes the geometry refuses rather than indexes.
///
/// The dangerous half of the same state: a rig whose cameras differ in size
/// — which the port supports — would walk 512-wide rows of the last frame
/// with the 960-wide stride of this one, and a band deep in the taller frame
/// runs off the end of the buffer.
#[test]
fn a_failed_scan_that_changes_the_geometry_does_not_panic() {
    let mut scanner: GpuCornerScan<GpuRuntime> =
        GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
    scanner.scan(0, &dotted_image(512, 128)).unwrap();

    arm_fault_at(CORNER_SCAN_READ);
    assert!(scanner.scan(1, &dotted_image(960, 240)).is_err());
    assert_eq!(
        scanner.band(band(150)).unwrap_err(),
        FrontendError::Detect(CenteredCellError::NotScanned)
    );
}
#[test]
fn selection_failures_invalidate_the_batch_and_allow_retry() {
    use kornia_staging_imgproc::features::{CellGrid, CenteredCellConfig, cell_select};
    let images = [dotted_image(512, 128), dotted_image(512, 128)];
    let config = CenteredCellConfig {
        num_points_cell: 1,
        min_threshold: 5,
        max_threshold: 40,
        safe_radius: 0.0,
    };
    let grid = CellGrid::new(512, 128, 50).unwrap();
    let selection = cell_select(images[0].size(), &grid, &config).unwrap();
    let selects = [Some(selection), Some(selection)];
    let mut scanner = GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
    for site in ["selection camera submitted", super::super::BLOCKING_READ] {
        if site == "selection camera submitted" {
            arm_fault_at(site);
            assert!(
                scanner
                    .submit_cells(FrameImages::Dense(&images), &selects)
                    .is_err()
            );
        } else {
            scanner
                .submit_cells(FrameImages::Dense(&images), &selects)
                .unwrap();
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
        scanner
            .submit_cells(FrameImages::Dense(&images), &selects)
            .unwrap();
        scanner.take_cells().unwrap();
        assert!(
            scanner
                .cameras
                .iter()
                .all(|camera| matches!(camera.selection, Selection::Ready(_)))
        );
        scanner
            .submit_cells(FrameImages::Dense(&images), &selects)
            .unwrap();
        scanner.take_cells().unwrap();
        assert!(scanner.buffer_allocations() <= allocations + 2);
    }
}
