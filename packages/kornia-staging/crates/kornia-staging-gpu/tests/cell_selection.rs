//! CPU and GPU cell selectors against the kornia band walk, cell for cell.
#![allow(unsafe_code)]
//!
//! What is checked here is the exactness argument behind
//! [`kornia_staging_imgproc::features::CornerScan::select_cells`]: the whole of
//! `detectKeypointsWithCells` runs twice over the same frame — once through
//! a band-only `CpuCornerScan`, and once through each cell selector — and
//! the two results have to be equal corner for corner, response for response, in
//! the same cell scan order. Ties are included rather than excused: the packed
//! key's row and column fields break them the way the row-major band walk and a
//! stable sort do.
//!
//! These parity and lifecycle tests need `--features wgpu` and a working device.
#![cfg(feature = "wgpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kornia_image::Image;
use kornia_staging_gpu::{features::GpuCornerScan, runtime::gpu_client};
use kornia_staging_imgproc::features::{
    detect_keypoints_with_cells, threshold_rungs, BandRequest, CellGrid, CellMasks, CellSelect,
    CenteredCellConfig, CenteredCellError, CenteredCellKeypoints, CornerScan, CpuCornerScan,
    DetectorScratch, FastCorner, MaskRect, Occupancy, FAST_BORDER,
};

mod selection;
use selection as common;

use common::cornered_image;

/// Keep kornia's threshold ladder as an independent reference after the CPU
/// scanner learns cell selection.
#[derive(Debug, Default)]
struct BandScan(CpuCornerScan);

impl CornerScan for BandScan {
    type Error = CenteredCellError;
    fn scan(&mut self, camera: usize, image: &Image<u16, 1>) -> Result<(), CenteredCellError> {
        self.0.scan(camera, image)
    }

    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], CenteredCellError> {
        self.0.band(request)
    }
}

/// The msd-index detector, which is `num_points_cell = 1` and the 40/20/10/5
/// ladder every shipped config runs.
fn detector_config(safe_radius: f32) -> CenteredCellConfig {
    CenteredCellConfig {
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
struct CountingScan<S> {
    inner: S,
    bands: Arc<AtomicUsize>,
    selections: Arc<AtomicUsize>,
}

impl<S: CornerScan> CornerScan for CountingScan<S> {
    type Error = S::Error;
    fn scan(&mut self, camera: usize, image: &Image<u16, 1>) -> Result<(), Self::Error> {
        self.inner.scan(camera, image)
    }

    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], Self::Error> {
        self.bands.fetch_add(1, Ordering::Relaxed);
        self.inner.band(request)
    }

    fn select_cells(
        &mut self,
        camera: usize,
        image: &Image<u16, 1>,
        select: &CellSelect,
        eligibility: Option<(&Occupancy<'_>, &[bool])>,
        out: &mut Vec<Option<FastCorner>>,
    ) -> Result<kornia_staging_imgproc::features::SelectionStatus, Self::Error> {
        self.selections.fetch_add(1, Ordering::Relaxed);
        self.inner
            .select_cells(camera, image, select, eligibility, out)
    }
}

/// The execution path each numerical case requires.
#[derive(Clone, Copy)]
enum ExpectedPath {
    CellSelection,
    BandWalk,
}

impl ExpectedPath {
    fn assert(self, bands: usize, selections: usize, label: &str) {
        match self {
            Self::CellSelection => assert!(
                selections > 0 && bands == 0,
                "{label}: expected cell selection, got {selections} selections and {bands} bands"
            ),
            Self::BandWalk => assert!(
                selections == 0 && bands > 0,
                "{label}: expected band walk, got {selections} selections and {bands} bands"
            ),
        }
    }
}

/// One camera's detection, from a scanner of the caller's choosing.
fn detect_with<S: CornerScan + 'static>(
    scanner: S,
    image: &Image<u16, 1>,
    grid: &CellGrid,
    counts: &[i32],
    config: &CenteredCellConfig,
    masks: &CellMasks,
    budget: usize,
) -> CenteredCellKeypoints {
    let mut scratch = DetectorScratch::with_scanner(Box::new(scanner));
    let mut out: CenteredCellKeypoints = CenteredCellKeypoints::default();
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

/// Inputs for one numerical selection comparison.
#[derive(Clone, Copy)]
struct DetectionCase<'a> {
    image: &'a Image<u16, 1>,
    grid: &'a CellGrid,
    counts: &'a [i32],
    config: &'a CenteredCellConfig,
    masks: &'a CellMasks,
    budget: usize,
    label: &'a str,
}

/// Every selector's per-cell winners are the ones the kornia band walk chooses.
///
/// The whole of `detectKeypointsWithCells` runs twice over the same frame — once
/// through `BandScan`, which walks bands, and once through each selector —
/// and the two `CenteredCellKeypoints` must be equal, corner for corner and response for
/// response, in the same cell scan order.
///
/// That equality is the exactness argument, not the A/B run: the ladder carries
/// no information at `num_points_cell = 1`, suppression is threshold-independent,
/// and the packed key orders candidates the way the host's stable sort does. Ties
/// are included rather than excused — the key's row and column fields break them
/// the way the row-major band walk does.
fn detection_agrees(case: DetectionCase<'_>, expected: ExpectedPath) -> usize {
    let DetectionCase {
        image,
        grid,
        counts,
        config,
        masks,
        budget,
        label: _,
    } = case;
    let want: CenteredCellKeypoints = detect_with(
        BandScan::default(),
        image,
        grid,
        counts,
        config,
        masks,
        budget,
    );
    compare_detection(
        CpuCornerScan::with_cell_selection(true),
        "CPU",
        case,
        expected,
        &want,
    );
    compare_detection(
        GpuCornerScan::new(gpu_client().unwrap()).unwrap(),
        "GPU",
        case,
        expected,
        &want,
    );
    want.corners.len()
}

fn compare_detection<S: CornerScan + 'static>(
    scanner: S,
    selector: &str,
    case: DetectionCase<'_>,
    expected: ExpectedPath,
    want: &CenteredCellKeypoints,
) {
    let label = &format!("{} ({selector})", case.label);
    let bands = Arc::new(AtomicUsize::new(0));
    let selections = Arc::new(AtomicUsize::new(0));
    let got = detect_with(
        CountingScan {
            inner: scanner,
            bands: bands.clone(),
            selections: selections.clone(),
        },
        case.image,
        case.grid,
        case.counts,
        case.config,
        case.masks,
        case.budget,
    );
    assert_eq!(&got, want, "{label}: corners and responses");
    expected.assert(
        bands.load(Ordering::Relaxed),
        selections.load(Ordering::Relaxed),
        label,
    );
}
#[path = "selection/batch_lifecycle.rs"]
mod batch_lifecycle;
#[path = "selection/numerical_selection.rs"]
mod numerical_selection;
use kornia_staging_imgproc::features::backend::CELL_KEY_LIMIT;

fn submit_cells<R: cubecl::prelude::Runtime>(
    scan: &mut GpuCornerScan<R>,
    client: &cubecl::prelude::ComputeClient<R>,
    images: &[Image<u16, 1>],
    selects: &[Option<CellSelect>],
) -> Result<(), kornia_staging_gpu::features::ScanError> {
    scan.submit_cells(
        images.iter().map(Image::size),
        selects,
        |scan, camera, select| {
            scan.submit_input(
                camera,
                kornia_staging_gpu::features::ScanInput::Dense(&images[camera]),
                select,
            )
        },
        |work| {
            unsafe {
                work.run(client);
            }
            Ok(())
        },
    )
}
