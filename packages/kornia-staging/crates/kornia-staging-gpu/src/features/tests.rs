#![allow(clippy::unwrap_used)]

use super::*;
use crate::{runtime::gpu_client, GpuRuntime};

/// Shared-arena batches preserve exact winners across camera subsets and refills.
#[test]
fn packed_camera_selection_is_exact_in_one_dispatch() {
    use crate::pyramid::GpuPyramidBuilder;
    use kornia_staging_imgproc::features::{CellGrid, CpuCornerScan};

    let client = gpu_client().unwrap();
    let builder_client = client.clone();
    let mut builder = GpuPyramidBuilder::new(client.clone());
    let mut scanner = GpuCornerScan::new(client).unwrap();
    let mut cpu = CpuCornerScan::with_cell_selection(true);
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
                    kornia_staging_imgproc::test_fixtures::lcg_bytes(&mut state, &mut bytes);
                    bytes
                })
                .collect();
            let size = ImageSize { width, height };
            let images: Vec<_> = cameras
                .iter()
                .map(|bytes| {
                    Image::new(
                        size,
                        bytes
                            .chunks_exact(width + 13)
                            .flat_map(|row| row[..width].iter().map(|&v| u16::from(v) << 8))
                            .collect(),
                    )
                    .unwrap()
                })
                .collect();
            let mut packed: Vec<u8> = images
                .iter()
                .flat_map(|image| image.as_slice().iter().map(|&v| (v >> 8) as u8))
                .collect();
            packed.resize(packed.len().next_multiple_of(4), 0);
            let work = builder
                .prepare_packed(
                    &bytes::Bytes::from(packed),
                    images.iter().map(Image::size),
                    &mut pyramids,
                )
                .unwrap()
                .unwrap();
            // SAFETY: These operations share the current client stream.
            unsafe {
                work.run(&builder_client);
            }
            scanner.use_level0(&mut builder);
            for (first, end) in [(0, 1), (1, 4), (0, 4)] {
                let selects: Vec<_> = (0..4)
                    .map(|camera| (first..end).contains(&camera).then_some(select))
                    .collect();
                submit_cells(&mut scanner, &images, &selects).unwrap();
                assert!(
                    matches!(&scanner.reads, SelectionReads::Pending(selection) if matches!(selection.layout, SelectionLayout::Packed { .. })),
                    "shared-arena batch required: {width}x{height}, frame {frame}, {first}..{end}"
                );
                assert!(
                    matches!(&scanner.reads, SelectionReads::Pending(selection) if selection.handles.len() == 1)
                );
                scanner.take_cells().unwrap();
                assert_eq!(
                    scanner.begin_cells(4),
                    Err(ScanError::UnreadSelection { camera: first })
                );
                for (camera, image) in images.iter().enumerate().take(end).skip(first) {
                    assert!(
                        matches!(scanner.cameras[camera].selection, Some(ready) if ready == select)
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
    let mut image: Image<u16, 1> = Image::from_size_val(ImageSize { width, height }, 0u16).unwrap();
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
    let mut scanner: GpuCornerScan<GpuRuntime> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    scanner.scan(0, &dotted_image(512, 128)).unwrap();
    assert!(
        !scanner.band(band(3)).unwrap().is_empty(),
        "the dotted frame has corners, so the failed scan below has something to leak"
    );

    arm_fault_at(CORNER_SCAN_READ);
    assert!(matches!(
        scanner.scan(0, &dotted_image(512, 128)).unwrap_err(),
        ScanError::Gpu(GpuError::DeviceLost {
            what: "corner scan"
        })
    ));
    assert!(matches!(
        scanner.band(band(3)).unwrap_err(),
        ScanError::Detect(CenteredCellError::NotScanned)
    ));
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
    assert!(matches!(
        scanner.band(band(150)).unwrap_err(),
        ScanError::Detect(CenteredCellError::NotScanned)
    ));
}
#[test]
fn selection_failures_invalidate_the_batch_and_allow_retry() {
    use kornia_staging_imgproc::features::{cell_select, CellGrid, CenteredCellConfig};
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
    let mut scanner = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    for site in ["selection camera submitted", "blocking read"] {
        if site == "selection camera submitted" {
            arm_fault_at(site);
            assert!(submit_cells(&mut scanner, &images, &selects).is_err());
        } else {
            submit_cells(&mut scanner, &images, &selects).unwrap();
            arm_fault_at(site);
            assert!(scanner.take_cells().is_err());
        }
        assert!(scanner
            .cameras
            .iter()
            .all(|camera| camera.selection.is_none()));
        let allocations = scanner.buffer_allocations();
        submit_cells(&mut scanner, &images, &selects).unwrap();
        scanner.take_cells().unwrap();
        assert!(scanner
            .cameras
            .iter()
            .all(|camera| camera.selection.is_some()));
        scanner.abort_selection();
        submit_cells(&mut scanner, &images, &selects).unwrap();
        scanner.take_cells().unwrap();
        assert!(scanner.buffer_allocations() <= allocations + 2);
        scanner.abort_selection();
    }
}

fn submit_cells<R: cubecl::prelude::Runtime>(
    scanner: &mut GpuCornerScan<R>,
    images: &[Image<u16, 1>],
    selects: &[Option<CellSelect>],
) -> Result<(), ScanError> {
    let client = scanner.client.clone();
    scanner.submit_cells(
        images.iter().map(Image::size),
        selects,
        |scanner, camera, select| {
            scanner.submit_input(camera, ScanInput::Dense(&images[camera]), select)
        },
        |work| {
            unsafe {
                work.run(&client);
            }
            Ok(())
        },
    )
}

#[test]
fn owned_selection_delivery_is_atomic_and_cancellation_retains_buffers() {
    use kornia_staging_imgproc::features::CellGrid;
    let client = gpu_client().unwrap();
    let mut scanner = GpuCornerScan::new(client.clone()).unwrap();
    let images = [dotted_image(512, 128), dotted_image(512, 128)];
    let policy = CellSelect {
        grid: CellGrid::new(512, 128, 50).unwrap(),
        threshold: 5,
        safe_radius: 0.0,
    };
    let selects = [Some(policy); 2];
    submit_cells(&mut scanner, &images, &selects).unwrap();
    let allocations = scanner.buffer_allocations();
    let mut pending = scanner.take_staged().unwrap();
    assert_eq!(pending.entries, vec![(0, policy), (1, policy)]);
    let mut bytes =
        crate::transfer::read_inner(&client, pending.take_handles(), "test keys", || Ok(()))
            .unwrap();
    bytes.pop(); // A later camera fails; the first must not be published.
    scanner.deliver(pending, bytes).unwrap();
    assert!(scanner.take_cells().is_err());
    assert!(scanner
        .cameras
        .iter()
        .all(|camera| camera.selection.is_none()));
    submit_cells(&mut scanner, &images, &selects).unwrap();
    drop(scanner.take_staged().unwrap());
    scanner.abort_selection();
    submit_cells(&mut scanner, &images, &selects).unwrap();
    scanner.take_cells().unwrap();
    assert_eq!(scanner.buffer_allocations(), allocations);
    assert!(scanner
        .cameras
        .iter()
        .all(|camera| camera.selection.is_some()));
}

#[test]
fn cancelled_selection_delivery_cannot_publish_or_replace_a_new_batch() {
    use kornia_staging_imgproc::features::CellGrid;
    let client = gpu_client().unwrap();
    let mut scanner = GpuCornerScan::new(client.clone()).unwrap();
    let images = [dotted_image(512, 128)];
    let policy = CellSelect {
        grid: CellGrid::new(512, 128, 50).unwrap(),
        threshold: 5,
        safe_radius: 0.0,
    };
    let selects = [Some(policy)];
    for start_new_batch in [false, true] {
        submit_cells(&mut scanner, &images, &selects).unwrap();
        let allocations = scanner.buffer_allocations();
        let mut cancelled = scanner.take_staged().unwrap();
        let bytes = crate::transfer::read_inner(
            &client,
            cancelled.take_handles(),
            "cancelled test keys",
            || Ok(()),
        )
        .unwrap();
        scanner.abort_selection();
        if start_new_batch {
            scanner.begin_cells(1).unwrap();
        }
        assert_eq!(
            scanner.deliver(cancelled, bytes),
            Err(ScanError::SelectionCancelled)
        );
        assert_eq!(scanner.staged_handles().is_some(), start_new_batch);
        scanner.take_cells().unwrap();
        assert!(scanner
            .cameras
            .iter()
            .all(|camera| camera.selection.is_none()));
        assert_eq!(scanner.buffer_allocations(), allocations);
        submit_cells(&mut scanner, &images, &selects).unwrap();
        scanner.take_cells().unwrap();
        assert!(scanner
            .cameras
            .iter()
            .all(|camera| camera.selection.is_some()));
        scanner.abort_selection();
    }
}
