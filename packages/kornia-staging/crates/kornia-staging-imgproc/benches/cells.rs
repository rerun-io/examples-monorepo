use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::features::{
    detect_keypoints_with_cells, CellGrid, CpuCornerScan, DetectorConfig, DetectorScratch,
    KeypointsData, Masks, Occupancy,
};
use std::hint::black_box;
fn bench(c: &mut Criterion) {
    let image = Image::new(
        ImageSize {
            width: 960,
            height: 720,
        },
        (0u32..960 * 720)
            .map(|v| (v.wrapping_mul(7919) as u16) & 0xff00)
            .collect(),
    )
    .unwrap();
    let grid = CellGrid::new(960, 720, 50).unwrap();
    let counts = vec![0; grid.rows * grid.columns];
    let occupancy = Occupancy {
        counts: &counts,
        rows: grid.rows,
        columns: grid.columns,
    };
    let config = DetectorConfig {
        num_points_cell: 1,
        min_threshold: 5,
        max_threshold: 40,
        safe_radius: 0.0,
    };
    let masks = Masks::default();
    for select in [false, true] {
        let mut scratch =
            DetectorScratch::with_scanner(Box::new(CpuCornerScan::with_cell_selection(select)));
        let mut out = KeypointsData::default();
        c.bench_function(
            if select {
                "centered_fast_cells_960x720"
            } else {
                "centered_fast_bands_960x720"
            },
            |b| {
                b.iter(|| {
                    detect_keypoints_with_cells(
                        black_box(&image),
                        0,
                        &grid,
                        &occupancy,
                        &config,
                        &masks,
                        4096,
                        &mut scratch,
                        black_box(&mut out),
                    )
                    .unwrap()
                })
            },
        );
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
