use super::*;

/// One scanner, four camera slots.
///
/// The device path keeps a key buffer per camera beside the three the band path
/// keeps, and the frontend calls the detector with the frameset's own camera
/// index — so a rig with four cameras reaches slot 3. The scratch is reused
/// across the four calls, which is what the frontend does and what makes the
/// per-camera buffers a thing that can be got wrong.
#[test]
fn the_gpu_cell_selection_holds_for_every_camera_slot() {
    let config: DetectorConfig = detector_config(472.0);
    let mut host: DetectorScratch = DetectorScratch::default();
    let bands = Arc::new(AtomicUsize::new(0));
    let selections = Arc::new(AtomicUsize::new(0));
    let mut device: DetectorScratch = DetectorScratch::with_scanner(Box::new(CountingScan {
        inner: Box::new(GpuCornerScan::new(gpu_client().unwrap()).unwrap()),
        bands: Arc::clone(&bands),
        selections: Arc::clone(&selections),
    }));

    for camera in 0..4 {
        let before_bands = bands.load(Ordering::Relaxed);
        let before_selections = selections.load(Ordering::Relaxed);
        // Two frames alternating, so consecutive slots hold different pixels.
        let image: ImageU16 = common::mio10_frame(camera % 2, camera % 2);
        let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
        let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
        let occupancy: Occupancy<'_> = Occupancy {
            counts: &counts,
            rows: grid.rows,
            columns: grid.columns,
        };
        let mut want: KeypointsData = KeypointsData::default();
        let mut got: KeypointsData = KeypointsData::default();
        for (scratch, out) in [(&mut host, &mut want), (&mut device, &mut got)] {
            detect_keypoints_with_cells(
                &image,
                camera,
                &grid,
                &occupancy,
                &config,
                &Masks::default(),
                4096,
                scratch,
                out,
            )
            .unwrap();
        }
        ExpectedPath::CellSelection.assert(
            bands.load(Ordering::Relaxed) - before_bands,
            selections.load(Ordering::Relaxed) - before_selections,
            &format!("camera {camera}"),
        );
        assert!(!want.corners.is_empty(), "camera {camera} found nothing");
        assert_eq!(got.corners, want.corners, "camera {camera}: corners");
        assert_eq!(got.responses, want.responses, "camera {camera}: responses");
    }
}

/// Two different camera inputs with device-shaped selection geometry.
struct SelectionFixture {
    images: [ImageU16; 2],
    selects: Vec<Option<CellSelect>>,
    cells: usize,
}

impl SelectionFixture {
    fn new(images: [ImageU16; 2]) -> Self {
        let config = detector_config(472.0);
        let grid = CellGrid::new(images[0].width(), images[0].height(), 50).unwrap();
        let selects: Vec<_> = images
            .iter()
            .map(|image| slam_rs::frontend::detect::cell_select(image, &grid, &config))
            .collect();
        assert!(
            selects.iter().all(Option::is_some),
            "the rig is device shaped"
        );
        // Selection visits one fewer cell each way than the occupancy matrix.
        let cells = ((grid.x_stop - grid.x_start) / grid.cell + 1)
            * ((grid.y_stop - grid.y_start) / grid.cell + 1);
        Self {
            images,
            selects,
            cells,
        }
    }

    fn keys(&self, scanner: &mut dyn CornerScan) -> Vec<Vec<u32>> {
        self.images
            .iter()
            .enumerate()
            .map(|(camera, image)| {
                let mut keys = Vec::new();
                scanner
                    .select_cells(
                        camera,
                        image,
                        &self.selects[camera].unwrap(),
                        None,
                        &mut keys,
                    )
                    .unwrap();
                assert_eq!(keys.len(), self.cells, "camera {camera}: device selection");
                keys
            })
            .collect()
    }

    fn batch_keys(&self, scanner: &mut dyn CornerScan) -> Vec<Vec<u32>> {
        scanner.submit_cells(&self.images, &self.selects).unwrap();
        scanner.take_cells().unwrap();
        self.keys(scanner)
    }
}

/// The batched preparation answers exactly what the per-camera call does.
///
/// [`CornerScan::submit_cells`] launches every camera's selection at once and
/// [`CornerScan::take_cells`] downloads them together, which is a scheduling
/// change and must be nothing
/// else: the keys it hands each camera have to be the ones that camera's own
/// `select_cells` would have read. Two different MIO10 frames in the two camera
/// slots, so a batch that crossed its cameras over would be caught.
#[test]
fn the_batched_preparation_answers_what_the_per_camera_call_does() {
    let fixture = SelectionFixture::new([common::mio10_frame(0, 0), common::mio10_frame(1, 1)]);
    let images = &fixture.images;
    let selects = &fixture.selects;

    let mut scanner: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let alone = fixture.keys(&mut scanner);

    scanner.submit_cells(images, selects).unwrap();
    scanner.take_cells().unwrap();
    assert_eq!(fixture.keys(&mut scanner), alone, "out of the batch");
    assert_ne!(alone[0], alone[1], "the two cameras hold the same frame");
}

/// The scanner's keys come home inside the tracker's download.
///
/// The relay is the whole of D78: `submit_cells` launches and downloads
/// nothing, and the next stage to synchronise is what brings the keys back —
/// here a `collect` with no tracking pass in flight at all, which is the
/// weakest form of the claim and so the sharpest test of it. A relay that
/// dropped them would be invisible from the values alone, because `take_cells`
/// reads for itself when nothing was delivered; so what is asserted is that the
/// scanner made no read of its own, and that the keys are still the ones its
/// own download would have given.
#[test]
fn the_tracker_download_carries_the_scanner_keys() {
    let fixture = SelectionFixture::new([common::mio10_frame(0, 0), common::mio10_frame(1, 1)]);
    let images = &fixture.images;
    let selects = &fixture.selects;

    // What the scanner answers when it downloads for itself: no relay wired.
    let mut alone: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let want = fixture.batch_keys(&mut alone);

    // And the same two cameras with the tracker's `collect` in between.
    let mut scanner: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let mut tracker: GpuPatchTracker<Pattern51, _> =
        GpuPatchTracker::new(gpu_client().unwrap(), 512, 4, 5, 4.0, 2).unwrap();
    scanner.share_reads(&mut tracker);

    scanner.submit_cells(images, selects).unwrap();
    let before = slam_rs::gpu::seam::snapshot();
    tracker.collect().unwrap();
    scanner.take_cells().unwrap();
    let reads = slam_rs::gpu::seam::snapshot().delta(before);
    assert_eq!(reads.read_track.calls, 1);
    assert_eq!(reads.read_detect.calls, 0);
    assert_eq!(fixture.keys(&mut scanner), want, "through the relay");
}

/// Reusing a scanner discards the prior delivered generation.
#[test]
fn a_retry_uses_only_the_new_generation() {
    let first = SelectionFixture::new([common::mio10_frame(0, 0), common::mio10_frame(1, 1)]);
    let second = SelectionFixture::new([common::mio10_frame(2, 1), common::mio10_frame(2, 0)]);
    let mut alone = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let want = second.batch_keys(&mut alone);
    let mut scanner = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let mut tracker: GpuPatchTracker<Pattern51, _> =
        GpuPatchTracker::new(gpu_client().unwrap(), 512, 4, 5, 4.0, 2).unwrap();
    scanner.share_reads(&mut tracker);
    scanner.submit_cells(&first.images, &first.selects).unwrap();
    tracker.collect().unwrap();
    scanner
        .submit_cells(&second.images, &second.selects)
        .unwrap();
    tracker.collect().unwrap();
    scanner.take_cells().unwrap();
    assert_eq!(second.keys(&mut scanner), want);
}

/// A prepared selection is spent by the call that reads it, and a camera the
/// batch skipped still answers for itself.
///
/// The keys are cached on the scanner between `take_cells` and the
/// `select_cells` that takes them, so the thing that must not happen is a
/// leftover answering a later frameset. Reading twice, and preparing a rig where
/// only one camera is offered, are the two ways that could happen.
#[test]
fn a_prepared_selection_is_spent_once() {
    let config: DetectorConfig = detector_config(472.0);
    let images: [ImageU16; 2] = [common::mio10_frame(0, 0), common::mio10_frame(1, 1)];
    let grid: CellGrid = CellGrid::new(images[0].width(), images[0].height(), 50).unwrap();
    let select: CellSelect =
        slam_rs::frontend::detect::cell_select(&images[0], &grid, &config).unwrap();
    let cells: usize = ((grid.x_stop - grid.x_start) / grid.cell + 1)
        * ((grid.y_stop - grid.y_start) / grid.cell + 1);

    let mut scanner: GpuCornerScan<_> = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    // Only camera 0 is offered, so camera 1 has nothing prepared for it.
    scanner
        .submit_cells(&images, &[Some(select), None])
        .unwrap();
    scanner.take_cells().unwrap();

    let mut first: Vec<u32> = Vec::new();
    scanner
        .select_cells(0, &images[0], &select, None, &mut first)
        .unwrap();
    let mut again: Vec<u32> = Vec::new();
    scanner
        .select_cells(0, &images[1], &select, None, &mut again)
        .unwrap();
    let mut independent = GpuCornerScan::new(gpu_client().unwrap()).unwrap();
    let mut expected = Vec::new();
    independent
        .select_cells(0, &images[1], &select, None, &mut expected)
        .unwrap();
    assert_ne!(expected, first, "the second image must have different keys");
    assert_eq!(again, expected, "the second read must scan the new image");

    let mut second: Vec<u32> = Vec::new();
    scanner
        .select_cells(1, &images[1], &select, None, &mut second)
        .unwrap();
    assert_eq!(second.len(), cells);
    assert_ne!(second, first, "camera 1 answered with camera 0's frame");
}
