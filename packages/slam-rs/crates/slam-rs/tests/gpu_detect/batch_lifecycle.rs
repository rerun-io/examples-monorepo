use super::*;

/// Dense camera uploads preserve exact cell selection for side-camera batches
/// and later framesets, without entering the packed-byte ingest path.
#[test]
fn dense_camera_selection_is_exact_after_pyramid_upload() {
    use slam_rs::frontend::parallel::WorkPool;
    use slam_rs::gpu::GpuPyramidBuilder;
    use slam_rs::pyramid::PyramidBuilder;

    let client = gpu_client().unwrap();
    let mut builder = GpuPyramidBuilder::new(client.clone(), Default::default());
    let mut scanner = GpuCornerScan::new(client, Default::default()).unwrap();

    let pool = WorkPool::new(1).unwrap();
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
        for _ in 0..2 {
            let images: Vec<_> = (0..4)
                .map(|_| {
                    let mut bytes = vec![0; (width + 13) * height];
                    for byte in &mut bytes {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        *byte = (state >> 24) as u8;
                    }
                    let mut image = slam_rs::image::empty();
                    slam_rs::image::fill_from_u8_strided(
                        &mut image,
                        &bytes,
                        width,
                        height,
                        width + 13,
                    )
                    .unwrap();
                    image
                })
                .collect();
            builder.build_frames(&images, &mut pyramids, &pool).unwrap();
            scanner.use_level0(&mut builder);
            for (first, end) in [(0, 1), (1, 4), (0, 4)] {
                let selects: Vec<_> = (0..4)
                    .map(|camera| (first..end).contains(&camera).then_some(select))
                    .collect();
                scanner
                    .submit_cells(FrameImages::Dense(&images), &selects)
                    .unwrap();
                scanner.take_cells().unwrap();
                for (camera, image) in images.iter().enumerate().take(end).skip(first) {
                    let mut got = Vec::new();
                    let mut want = Vec::new();
                    scanner
                        .select_cells(camera, image, &select, None, &mut got)
                        .unwrap();
                    cpu.select_cells(camera, image, &select, None, &mut want)
                        .unwrap();
                    assert_eq!(got, want, "camera {camera}, batch {first}..{end}");
                }
            }
        }
    }
}

/// One scanner, four camera slots.
///
/// The device path keeps a key buffer per camera beside the three the band path
/// keeps, and the frontend calls the detector with the frameset's own camera
/// index — so a rig with four cameras reaches slot 3. The scratch is reused
/// across the four calls, which is what the frontend does and what makes the
/// per-camera buffers a thing that can be got wrong.
#[test]
fn the_gpu_cell_selection_holds_for_every_camera_slot() {
    let config: CenteredCellConfig = detector_config(472.0);
    let mut host: DetectorScratch<dyn CornerScan<Error = FrontendError>> =
        DetectorScratch::with_scanner(Box::new(AppScan(CpuCornerScan::default())));
    let bands = Arc::new(AtomicUsize::new(0));
    let selections = Arc::new(AtomicUsize::new(0));
    let mut device: DetectorScratch<dyn CornerScan<Error = FrontendError>> =
        DetectorScratch::with_scanner(Box::new(CountingScan {
            inner: Box::new(GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap()),
            bands: Arc::clone(&bands),
            selections: Arc::clone(&selections),
        }));

    for camera in 0..4 {
        let before_bands = bands.load(Ordering::Relaxed);
        let before_selections = selections.load(Ordering::Relaxed);
        // Two frames alternating, so consecutive slots hold different pixels.
        let image: Image<u16, 1> = common::mio10_frame(camera % 2, camera % 2);
        let grid: CellGrid = CellGrid::new(image.width(), image.height(), 50).unwrap();
        let counts: Vec<i32> = vec![0; grid.rows * grid.columns];
        let occupancy: Occupancy<'_> = Occupancy {
            counts: &counts,
            rows: grid.rows,
            columns: grid.columns,
        };
        let mut want: CenteredCellKeypoints = CenteredCellKeypoints::default();
        let mut got: CenteredCellKeypoints = CenteredCellKeypoints::default();
        for (scratch, out) in [(&mut host, &mut want), (&mut device, &mut got)] {
            detect_keypoints_with_cells(
                &image,
                camera,
                &grid,
                &occupancy,
                &config,
                &CellMasks::default(),
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
    images: [Image<u16, 1>; 2],
    selects: Vec<Option<CellSelect>>,
    cells: usize,
}

impl SelectionFixture {
    fn new(images: [Image<u16, 1>; 2]) -> Self {
        let config = detector_config(472.0);
        let grid = CellGrid::new(images[0].width(), images[0].height(), 50).unwrap();
        let selects: Vec<_> = images
            .iter()
            .map(|image| {
                kornia_staging_imgproc::features::cell_select(image.size(), &grid, &config)
            })
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

    fn keys(&self, scanner: &mut impl CornerScan) -> Vec<Vec<Option<FastCorner>>> {
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
}

/// The batched preparation answers exactly what the per-camera call does.
///
/// [`FrameCornerScan::submit_cells`] launches every camera's selection at once and
/// [`FrameCornerScan::take_cells`] downloads them together, which is a scheduling
/// change and must be nothing
/// else: the keys it hands each camera have to be the ones that camera's own
/// `select_cells` would have read. Two different MIO10 frames in the two camera
/// slots, so a batch that crossed its cameras over would be caught.
#[test]
fn the_batched_preparation_answers_what_the_per_camera_call_does() {
    let fixture = SelectionFixture::new([common::mio10_frame(0, 0), common::mio10_frame(1, 1)]);
    let images = &fixture.images;
    let selects = &fixture.selects;

    let mut scanner: GpuCornerScan<_> =
        GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
    let alone = fixture.keys(&mut scanner);

    scanner
        .submit_cells(FrameImages::Dense(images), selects)
        .unwrap();
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
/// A prepared selection is spent by the call that reads it, and a camera the
/// batch skipped still answers for itself.
///
/// The keys are cached on the scanner between `take_cells` and the
/// `select_cells` that takes them, so the thing that must not happen is a
/// leftover answering a later frameset. Reading twice, and preparing a rig where
/// only one camera is offered, are the two ways that could happen.
#[test]
fn a_prepared_selection_is_spent_once() {
    let config: CenteredCellConfig = detector_config(472.0);
    let images: [Image<u16, 1>; 2] = [common::mio10_frame(0, 0), common::mio10_frame(1, 1)];
    let grid: CellGrid = CellGrid::new(images[0].width(), images[0].height(), 50).unwrap();
    let select: CellSelect =
        kornia_staging_imgproc::features::cell_select(images[0].size(), &grid, &config).unwrap();
    let cells: usize = ((grid.x_stop - grid.x_start) / grid.cell + 1)
        * ((grid.y_stop - grid.y_start) / grid.cell + 1);

    let mut scanner: GpuCornerScan<_> =
        GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
    // Only camera 0 is offered, so camera 1 has nothing prepared for it.
    scanner
        .submit_cells(FrameImages::Dense(&images), &[Some(select), None])
        .unwrap();
    scanner.take_cells().unwrap();

    let mut first: Vec<Option<FastCorner>> = Vec::new();
    scanner
        .select_cells(0, &images[0], &select, None, &mut first)
        .unwrap();
    let mut again: Vec<Option<FastCorner>> = Vec::new();
    scanner
        .select_cells(0, &images[1], &select, None, &mut again)
        .unwrap();
    let mut independent = GpuCornerScan::new(gpu_client().unwrap(), Default::default()).unwrap();
    let mut expected = Vec::new();
    independent
        .select_cells(0, &images[1], &select, None, &mut expected)
        .unwrap();
    assert_ne!(expected, first, "the second image must have different keys");
    assert_eq!(again, expected, "the second read must scan the new image");

    let mut second: Vec<Option<FastCorner>> = Vec::new();
    scanner
        .select_cells(1, &images[1], &select, None, &mut second)
        .unwrap();
    assert_eq!(second.len(), cells);
    assert_ne!(second, first, "camera 1 answered with camera 0's frame");
}
