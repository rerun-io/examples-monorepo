//! Exact GPU FAST bands and shared-upload contracts.
#![cfg(feature = "wgpu")]
use bands::band_at;
use images::cornered_image;
use kornia_image::{Image, ImageSize};
use kornia_staging_gpu::{
    features::{GpuCornerScan, ScanError},
    pyramid::GpuPyramidBuilder,
    runtime::gpu_client,
};
use kornia_staging_imgproc::features::{
    BandRequest, CenteredCellError, CornerScan, CpuCornerScan, FastCorner,
};
use kornia_staging_imgproc::test_fixtures as bands;
use kornia_staging_imgproc::test_fixtures as images;

/// Every band the detector asks for, at every rung of the shipped ladder, from
/// two scanners — and the count of corners they agreed on.
///
/// One per grid row of a 50-pixel cell, which is the shape
/// `detect_keypoints_with_cells` drives. Written once because both corner tests
/// need exactly it: the "exact against kornia" one and the "reads the pyramid"
/// one, which would otherwise drift into checking different amounts.
fn bands_agree(
    reference: &mut impl CornerScan,
    actual: &mut impl CornerScan,
    height: usize,
    label: &str,
) -> usize {
    let mut total: usize = 0;
    for (row, band_y) in (3..height - 3).step_by(50).enumerate() {
        for (rung, threshold) in [40i32, 20, 10, 5, 1].into_iter().enumerate() {
            let request: BandRequest = band_at(row, rung, band_y, 44, threshold);
            let want: Vec<FastCorner> = reference.band(request).unwrap().to_vec();
            let got: &[FastCorner] = actual.band(request).unwrap();
            assert_eq!(
                got.len(),
                want.len(),
                "{label} band {band_y} threshold {threshold}: {} corners against {}",
                got.len(),
                want.len()
            );
            for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    (got.xy, got.response),
                    (want.xy, want.response),
                    "{label} band {band_y} threshold {threshold}, corner {index}"
                );
            }
            total += want.len();
        }
    }
    total
}

/// The GPU corner scanner is **exact**, not within a tolerance.
///
/// kornia's candidate test at threshold `t` is the same statement as
/// `corner_score_9 > t`, and its in-block local-maximum filter compares raw
/// scores, so one dense score image answers every rung of the ladder.
/// This test checks that the two CubeCL kernels implement that contract, corner for
/// corner, response for response, and in the same row-major order.
#[test]
fn the_gpu_corner_scan_is_exact_against_kornia() {
    for (width, height) in [(960usize, 240usize), (512, 192)] {
        let image: Image<u16, 1> = cornered_image(width, height);
        let mut cpu: CpuCornerScan = CpuCornerScan::default();
        let mut gpu: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
        cpu.scan(0, &image).unwrap();
        gpu.scan(0, &image).unwrap();

        let total: usize = bands_agree(&mut cpu, &mut gpu, height, &format!("{width}x{height}"));
        println!("{width}x{height}: {total} corners over every band and rung, identical");
    }
}

/// The GPU lane refuses a band before a scan with the same typed error the CPU
/// lane returns, rather than caching an empty one and reporting success.
#[test]
fn a_gpu_band_before_a_scan_is_refused() {
    let mut gpu: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    assert!(matches!(
        gpu.band(band_at(0, 0, 0, 32, 5)).unwrap_err(),
        ScanError::Detect(CenteredCellError::NotScanned)
    ));
}

/// The scanner is reused frame after frame, so the second frame's bands must be
/// the second frame's.
#[test]
fn a_reused_corner_scan_carries_only_the_newest_frame() {
    let first: Image<u16, 1> = cornered_image(512, 128);
    let second: Image<u16, 1> = Image::from_size_val(
        ImageSize {
            width: 512,
            height: 128,
        },
        0u16,
    )
    .unwrap();

    let mut gpu: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    gpu.scan(0, &first).unwrap();
    assert!(
        !gpu.band(band_at(0, 0, 3, 44, 5)).unwrap().is_empty(),
        "the textured frame has corners"
    );
    gpu.scan(0, &second).unwrap();
    assert!(
        gpu.band(band_at(0, 0, 3, 44, 5)).unwrap().is_empty(),
        "a black frame has none"
    );
}

/// The detector reads the pyramid's level 0, and uploads nothing.
///
/// The corner scanner and the pyramid builder are handed the same pixels once
/// per camera per frameset — the detector's input *is* level 0
///  — and until the camera index reached
/// both seams the GPU scanner had no way to know that, so it uploaded the frame
/// a second time. This is the test that the sharing is exact rather than merely
/// cheaper: the same corners at every rung, from a scanner that uploaded zero
/// frames, on two cameras of different geometry so the per-camera table is
/// actually indexed.
#[test]
fn the_gpu_corner_scan_reads_the_pyramid_and_uploads_nothing() {
    let frames: [Image<u16, 1>; 2] = [cornered_image(960, 240), cornered_image(512, 192)];
    let client = gpu_client().unwrap();

    // The lane the frontend runs: one builder, one scanner, one client, the
    // level-0 table between them.
    let mut builder = GpuPyramidBuilder::new(client.clone());
    let mut shared: GpuCornerScan<_> = GpuCornerScan::new(client.clone()).unwrap();

    // The lane before this change: the scanner uploads its own copy.
    let mut alone: GpuCornerScan<_> = GpuCornerScan::new(client).unwrap();

    let mut pyramids: Vec<_> = frames
        .iter()
        .map(|frame| builder.allocate(frame.width(), frame.height(), 3).unwrap())
        .collect();
    for (camera, frame) in frames.iter().enumerate() {
        builder.build(camera, frame, &mut pyramids[camera]).unwrap();
    }
    shared.use_level0(&mut builder);
    for (camera, frame) in frames.iter().enumerate() {
        shared.scan(camera, frame).unwrap();
        alone.scan(camera, frame).unwrap();
        let total: usize = bands_agree(
            &mut alone,
            &mut shared,
            frame.height(),
            &format!("camera {camera}"),
        );
        println!(
            "camera {camera} ({}x{}): {total} corners identical, uploads shared {} / alone {}",
            frame.width(),
            frame.height(),
            shared.frame_uploads(),
            alone.frame_uploads()
        );
    }
    assert_eq!(
        shared.frame_uploads(),
        0,
        "the shared scanner uploaded a frame the pyramid had already put on the device"
    );
    assert_eq!(alone.frame_uploads(), frames.len());

    // And each camera's three device buffers are allocated once for the life of
    // the scanner, not once per frameset. A one-slot geometry cache holds only
    // for a rig whose cameras are all the same size; on this one — 960x240 next
    // to 512x192, which is why the test drives two — a scanner without the
    // cache misses on every scan and re-allocates 4 MB.
    let allocations: usize = shared.buffer_allocations();
    assert_eq!(allocations, frames.len(), "one geometry, one allocation");
    for _ in 0..3 {
        for (camera, frame) in frames.iter().enumerate() {
            shared.scan(camera, frame).unwrap();
        }
    }
    assert_eq!(
        shared.buffer_allocations(),
        allocations,
        "three more framesets of the same rig re-allocated the scan buffers"
    );
    assert_eq!(shared.frame_uploads(), 0);

    // A frame whose geometry does not match the published entry is refused
    // rather than read: the fallback upload is what keeps a stale table safe.
    let odd: Image<u16, 1> = cornered_image(256, 128);
    shared.scan(0, &odd).unwrap();
    assert_eq!(shared.frame_uploads(), 1);
}

#[test]
fn submitting_without_a_batch_is_refused() {
    use kornia_staging_gpu::features::ScanInput;
    use kornia_staging_imgproc::features::{CellGrid, CellSelect};
    let image = cornered_image(64, 64);
    let select = CellSelect {
        grid: CellGrid::new(64, 64, 32).unwrap(),
        threshold: 5,
        safe_radius: 0.0,
    };
    let mut scan = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    assert!(scan
        .submit_input(0, ScanInput::Dense(&image), &select)
        .is_err());
}

#[test]
fn camera_submissions_refuse_reversed_duplicate_and_out_of_range_indices() {
    use kornia_staging_gpu::features::ScanInput;
    use kornia_staging_imgproc::features::{CellGrid, CellSelect};
    let image = cornered_image(64, 64);
    let select = CellSelect {
        grid: CellGrid::new(64, 64, 32).unwrap(),
        threshold: 5,
        safe_radius: 0.0,
    };
    let mut scan = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    for (first, next) in [(1, 0), (0, 0), (0, 2)] {
        scan.begin_cells(2).unwrap();
        scan.submit_input(first, ScanInput::Dense(&image), &select)
            .unwrap();
        assert!(
            scan.submit_input(next, ScanInput::Dense(&image), &select)
                .is_err(),
            "{first} then {next}"
        );
        assert!(
            scan.staged_handles().is_none(),
            "the failed batch must be discarded"
        );
    }
}
