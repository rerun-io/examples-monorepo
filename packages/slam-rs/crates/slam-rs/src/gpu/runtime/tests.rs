#![allow(clippy::unwrap_used)]

use super::*;
use crate::gpu::GpuCornerScan;
use crate::gpu::submission::read_failed;
use kornia_staging_gpu::runtime::GpuError;
use kornia_staging_gpu::runtime::gpu_client;

/// A read that fails is a typed error at every boundary, never a panic.
///
/// The failure this is about cannot be forced from a test: a `ServerError`
/// on a download means a lost device or a staging allocation refused under
/// memory pressure, and neither is reachable from a healthy card. What is
/// testable — and what the per-frame path actually depends on — is that the
/// mapping produces a variant the stage errors carry, so the unwinding a
/// panicking read would do through the released GIL (decision D32) cannot
/// happen.
#[test]
fn a_failed_device_read_is_a_typed_error_at_every_stage() {
    let what: &str = "the tracker result";
    let error: cubecl::server::ServerError = cubecl::server::ServerError::Generic {
        reason: "the device is gone".to_owned(),
        backtrace: Default::default(),
    };
    assert_eq!(
        read_failed(what, &error),
        GpuError::DeviceReadFailed { what }
    );

    // The three stages a download sits in each carry it, so the error
    // reaches the Python boundary as the documented `ValueError`.
    let tracker: crate::frontend::flow::FrontendError = GpuError::DeviceReadFailed { what }.into();
    let pyramid: crate::pyramid::PyramidError = GpuError::DeviceReadFailed { what }.into();
    let detect: crate::frontend::flow::FrontendError = GpuError::DeviceReadFailed { what }.into();
    for message in [tracker.to_string(), pyramid.to_string(), detect.to_string()] {
        assert!(
            message.contains(what),
            "the stage error dropped what failed: {message}"
        );
    }
}

#[test]
fn a_failed_corner_queue_flush_is_typed_and_invalidates_old_bands() {
    use kornia_image::{Image, ImageSize};
    use kornia_staging_imgproc::features::{BandRequest, CornerScan};
    let mut scan = GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
    let image = Image::from_size_val(
        ImageSize {
            width: 64,
            height: 64,
        },
        0u16,
    )
    .unwrap();
    let band = BandRequest {
        row: 0,
        rung: 0,
        y: 3,
        rows: 44,
        threshold: 5,
    };
    scan.scan(0, &image).unwrap();
    assert!(scan.band(band).is_ok());
    arm_fault_at("queued launch");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scan.scan(0, &image)));
    assert!(outcome.is_ok(), "the device panic must be converted");
    assert!(outcome.unwrap().is_err());
    assert!(scan.band(band).is_err(), "old bands must be invalidated");
}

#[test]
fn a_failed_cell_queue_flush_cannot_publish_previous_frame_keys() {
    use crate::frontend::input::PackedImages;
    use crate::frontend::{detect::FrameCornerScan, input::FrameImages};
    use crate::gpu::pyramid::GpuPyramidBuilder;
    use crate::pyramid::PyramidBuilder;
    use kornia_image::{Image, ImageSize};
    use kornia_staging_imgproc::features::{CellGrid, CellSelect, CornerScan};
    use kornia_staging_imgproc::test_fixtures::cornered_image;

    let client = gpu_client().unwrap();
    let launches = crate::gpu::LaunchList::default();
    let mut scan = GpuCornerScan::new(client.clone(), launches.clone()).unwrap();
    let mut builder = GpuPyramidBuilder::new(client.clone(), launches.clone());
    let mut pyramids = vec![builder.allocate(640, 480, 1).unwrap()];
    let frame_a = [cornered_image(640, 480)];
    let frame_b = [Image::from_size_val(
        ImageSize {
            width: 640,
            height: 480,
        },
        0u16,
    )
    .unwrap()];
    let select = CellSelect {
        grid: CellGrid::new(640, 480, 50).unwrap(),
        threshold: 5,
        safe_radius: 0.0,
    };
    let bytes_a: Vec<u8> = frame_a[0]
        .as_slice()
        .iter()
        .map(|pixel| (pixel >> 8) as u8)
        .collect();
    let bytes_b = vec![0u8; 640 * 480];
    let mut packed_a = PackedImages::default();
    packed_a.fill(&[crate::ImageView {
        data: &bytes_a,
        width: 640,
        height: 480,
        stride: 640,
    }]);
    let mut packed_b = PackedImages::default();
    packed_b.fill(&[crate::ImageView {
        data: &bytes_b,
        width: 640,
        height: 480,
        stride: 640,
    }]);
    builder.build_packed(&packed_a, &mut pyramids).unwrap();
    scan.inner.use_level0(&mut builder.inner);
    scan.submit_cells(FrameImages::Packed(&packed_a), &[Some(select)])
        .unwrap();
    scan.take_cells().unwrap();
    let mut previous = Vec::new();
    scan.select_cells(0, &frame_a[0], &select, None, &mut previous)
        .unwrap();
    assert!(previous.iter().any(Option::is_some));

    let batch = launches.begin().unwrap();
    builder.build_packed(&packed_b, &mut pyramids).unwrap();
    scan.inner.use_level0(&mut builder.inner);
    scan.submit_cells(FrameImages::Packed(&packed_b), &[Some(select)])
        .unwrap();
    arm_fault_at("queued launch");
    assert!(scan.take_cells().is_err());
    scan.take_cells().unwrap();
    batch.finish(&client).unwrap();
    // Re-submit the dropped pyramid, without submitting a fresh selection. A stale
    // Ready selection must not mask these new pixels after the second take_cells.
    builder.build_packed(&packed_b, &mut pyramids).unwrap();
    scan.inner.use_level0(&mut builder.inner);
    let mut current = Vec::new();
    scan.select_cells(0, &frame_b[0], &select, None, &mut current)
        .unwrap();
    assert!(
        current.iter().all(Option::is_none),
        "failed frame B published frame A's keys"
    );
    scan.submit_cells(FrameImages::Packed(&packed_b), &[Some(select)])
        .unwrap();
    scan.take_cells().unwrap();
    scan.select_cells(0, &frame_b[0], &select, None, &mut current)
        .unwrap();
    assert!(current.iter().all(Option::is_none));
}
