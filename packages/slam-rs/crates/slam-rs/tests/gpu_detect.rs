//! CPU and GPU cell selectors against the kornia band walk, cell for cell.
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
//! a band-only `CpuCornerScan`, and once through each cell selector — and
//! the two results have to be equal corner for corner, response for response, in
//! the same cell scan order. Ties are included rather than excused: the packed
//! key's row and column fields break them the way the row-major band walk and a
//! stable sort do.
//!
//! CPU comparisons always run. GPU comparisons and lifecycle tests need
//! `--features gpu-wgpu` and a working CubeCL runtime.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use slam_rs::frontend::detect::{
    BandRequest, CELL_KEY_LIMIT, CellGrid, CellSelect, CornerScan, CpuCornerScan, DetectError,
    DetectorConfig, DetectorScratch, FAST_BORDER, FastCorner, KeypointsData, Masks, Occupancy,
    Rect, detect_keypoints_with_cells, threshold_rungs,
};
#[cfg(feature = "gpu-core")]
use slam_rs::gpu::{GpuCornerScan, gpu_client};
use slam_rs::image::ImageU16;

mod common;

use common::cornered_image;

/// Keep kornia's threshold ladder as an independent reference after the CPU
/// scanner learns cell selection.
#[derive(Debug, Default)]
struct BandScan(CpuCornerScan);

impl CornerScan for BandScan {
    fn scan(&mut self, camera: usize, image: &ImageU16) -> Result<(), DetectError> {
        self.0.scan(camera, image)
    }

    fn band(&mut self, request: BandRequest) -> Result<&[FastCorner], DetectError> {
        self.0.band(request)
    }
}

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
        eligibility: Option<(&Occupancy<'_>, &[bool])>,
        out: &mut Vec<u32>,
    ) -> Result<slam_rs::frontend::cell::SelectionStatus, DetectError> {
        self.selections.fetch_add(1, Ordering::Relaxed);
        self.inner
            .select_cells(camera, image, select, eligibility, out)
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

/// Inputs for one numerical selection comparison.
struct DetectionCase<'a> {
    image: &'a ImageU16,
    grid: &'a CellGrid,
    counts: &'a [i32],
    config: &'a DetectorConfig,
    masks: &'a Masks,
    budget: usize,
    label: &'a str,
}

/// Every selector's per-cell winners are the ones the kornia band walk chooses.
///
/// The whole of `detectKeypointsWithCells` runs twice over the same frame — once
/// through `BandScan`, which walks bands, and once through each selector —
/// and the two `KeypointsData` must be equal, corner for corner and response for
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
        label,
    } = case;
    let want: KeypointsData = detect_with(
        Box::new(BandScan::default()),
        image,
        grid,
        counts,
        config,
        masks,
        budget,
    );
    let scanners: Vec<(&str, Box<dyn CornerScan>)> = vec![
        ("CPU", Box::new(CpuCornerScan::with_cell_selection(true))),
        #[cfg(feature = "gpu-core")]
        (
            "GPU",
            Box::new(GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap()),
        ),
    ];
    for (selector, scanner) in scanners {
        let label = &format!("{label} ({selector})");
        let bands: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
        let selections: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
        let got: KeypointsData = detect_with(
            Box::new(CountingScan {
                inner: scanner,
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
        assert_eq!(got, want, "{label}: corners and responses");
        expected.assert(
            bands.load(Ordering::Relaxed),
            selections.load(Ordering::Relaxed),
            label,
        );
    }
    want.corners.len()
}

#[path = "gpu_detect/numerical_selection.rs"]
mod numerical_selection;

#[cfg(feature = "gpu-core")]
#[path = "gpu_detect/batch_lifecycle.rs"]
mod batch_lifecycle;
