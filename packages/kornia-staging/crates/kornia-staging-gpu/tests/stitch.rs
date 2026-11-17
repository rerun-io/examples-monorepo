//! The device stitch against the CPU remap, bit for bit: `lens_level1` + `remap_tile` and the CPU pyramid of the
//! CPU tile.
//!
//! The clip tests are asset-gated and decode with an `ffmpeg` on `PATH`: run
//! `cargo test -p kornia-staging-gpu --features wgpu --test stitch -- --ignored` (`KORNIA_STAGING_INSV` names the
//! clip).
#![allow(unsafe_code)]
#![cfg(feature = "wgpu")]
use kornia_image::{Image, ImageSize};
use kornia_staging_gpu::{
    pyramid::{GpuPyramid, GpuPyramidBuilder},
    runtime::gpu_client,
    stitch::TileTables,
    GpuRuntime,
};
use kornia_staging_imgproc::panorama::{lens_level1, remap, LensStitch, RemapTable, TileGeometry};
use kornia_staging_imgproc::pyramid::PyramidPlanU16;
use kornia_staging_io::insta360::{parse_lens_string, WindowCrop};

const LEVELS: usize = 3;
include!("../../kornia-staging-io/tests/common/x4air_lens.rs");
include!("../../kornia-staging-io/tests/common/x4air_clip.rs");

/// The 1088x960 tiles of a 1920x960 panorama.
const FACTORY_TILE: TileGeometry = TileGeometry {
    hemisphere_width: 960,
    height: 960,
    margin: 64,
};

/// The CPU stitch of one lens: `lens_level1` then `remap`.
fn cpu_tile(lens: &Image<u8, 1>, table: &RemapTable) -> Image<u16, 1> {
    let mut tile = Image::from_size_val(table.dst_size(), 0u16).unwrap();
    remap(&lens_level1(lens).unwrap(), table, &mut tile).unwrap();
    tile
}

/// Lens 0's then lens 1's bytes, padded to whole words, as `prepare_lenses` takes them.
fn packed(lenses: &[Image<u8, 1>; 2]) -> bytes::Bytes {
    let mut bytes: Vec<u8> = lenses
        .iter()
        .flat_map(|lens| lens.as_slice().iter().copied())
        .collect();
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes::Bytes::from(bytes)
}

/// Run the device stitch for one frameset and return the two tile pyramids.
fn device_pyramids(
    builder: &mut GpuPyramidBuilder<GpuRuntime>,
    tables: &TileTables<GpuRuntime>,
    lenses: &[Image<u8, 1>; 2],
    pyramids: &mut [GpuPyramid<GpuRuntime>],
) {
    let client = gpu_client().unwrap();
    let launch = builder
        .prepare_lenses(&packed(lenses), lenses[0].size(), tables, pyramids)
        .unwrap()
        .expect("two tiles fit one arena");
    // SAFETY: prepared on this process's shared client.
    unsafe { launch.run(&client) };
}

/// Every level of the device tile pyramid equals the CPU pyramid of the CPU tile; the message names the worst
/// pixel.
fn assert_tile_pyramid(expected: &Image<u16, 1>, gpu: &GpuPyramid<GpuRuntime>, label: &str) {
    let mut cpu = PyramidPlanU16::new(expected.size(), LEVELS).unwrap();
    cpu.run(expected).unwrap();
    assert_eq!(gpu.num_levels(), cpu.levels().len(), "{label}");
    let mut actual = Image::from_size_val(expected.size(), 0u16).unwrap();
    for (level, want) in cpu.levels().iter().enumerate() {
        gpu.read_level_into(level, &mut actual).unwrap();
        assert_eq!(actual.size(), want.size(), "{label}, level {level}");
        let worst = actual
            .as_slice()
            .iter()
            .zip(want.as_slice())
            .enumerate()
            .find(|(_, (a, b))| a != b);
        assert!(
            worst.is_none(),
            "{label}, level {level}: first difference at pixel {:?} of {}x{}; integer arithmetic must be exact",
            worst.map(|(i, (a, b))| (i % want.width(), i / want.width(), *a, *b)),
            want.width(),
            want.height()
        );
    }
}

/// Stitch `lenses` on both lanes and compare every tile level.
fn assert_stitch_matches(stitch: &LensStitch, lenses: &[Image<u8, 1>; 2], label: &str) {
    let client = gpu_client().unwrap();
    let tables = TileTables::upload(&client, stitch).unwrap();
    let mut builder = GpuPyramidBuilder::new(client);
    let tile = tables.tile_size();
    let mut pyramids: Vec<_> = (0..2)
        .map(|_| builder.allocate(tile.width, tile.height, LEVELS).unwrap())
        .collect();
    device_pyramids(&mut builder, &tables, lenses, &mut pyramids);
    for k in 0..2 {
        let expected = cpu_tile(&lenses[k], &stitch.tables()[k]);
        assert_tile_pyramid(&expected, &pyramids[k], &format!("{label}, tile {k}"));
    }
}

/// A deterministic full-range byte image.
fn noise(size: ImageSize, seed: u32) -> Image<u8, 1> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let pixels = (0..size.width * size.height)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Saturated runs exercise the largest products of the integer rule.
            match i % 97 {
                0..=5 => 255,
                6..=9 => 0,
                _ => (state >> 24) as u8,
            }
        })
        .collect();
    Image::new(size, pixels).unwrap()
}

/// A 256 x 264 tile over a 100 x 90 level-1 source (200x180 lenses): the first 256 rows hold every `(ax, ay)` phase
/// pair once, at scattered `(x0, y0)`; the last 8 rows hold the right and bottom edges (`x0 = w - 1`, `y0 = h - 1`,
/// phase 0, where the clamp matters) and invalid entries.
fn synthetic_stitch() -> LensStitch {
    let geometry = TileGeometry {
        hemisphere_width: 248,
        height: 264,
        margin: 4,
    };
    let source_size = ImageSize {
        width: 100,
        height: 90,
    };
    let (w, h) = (source_size.width as u32, source_size.height as u32);
    let table = |tile: u32| {
        let sample = (0..geometry.width() * geometry.height)
            .map(|p| {
                let p = p as u32;
                let (column, row) = (p % 256, p / 256);
                if row < 256 {
                    let (ax, ay) = (column, row);
                    let x0 = (p * 7 + tile * 13) % (w - 1);
                    let y0 = (p * 11 + tile * 5) % (h - 1);
                    [256 * x0 + ax, 256 * y0 + ay]
                } else {
                    match column % 4 {
                        0 => [256 * (w - 1), 256 * (column % h)],
                        1 => [256 * (column % w), 256 * (h - 1)],
                        2 => [256 * (w - 1), 256 * (h - 1)],
                        _ => RemapTable::INVALID,
                    }
                }
            })
            .collect();
        RemapTable::from_raw(geometry.size(), source_size, sample).unwrap()
    };
    let lens = ImageSize {
        width: 200,
        height: 180,
    };
    LensStitch::from_tables(geometry, [table(0), table(1)], lens).unwrap()
}

#[test]
fn synthetic_tables_cover_every_bilinear_phase_bit_exactly() {
    let stitch = synthetic_stitch();
    assert!(stitch.tables()[0].samples().contains(&RemapTable::INVALID));
    // 200x180 lenses reduce to the 100x90 source; 201x181 does too, through the odd-geometry ingest path.
    for (size, label) in [
        (
            ImageSize {
                width: 200,
                height: 180,
            },
            "even lens",
        ),
        (
            ImageSize {
                width: 201,
                height: 181,
            },
            "odd lens",
        ),
    ] {
        let lenses = [noise(size, 1), noise(size, 2)];
        assert_stitch_matches(&stitch, &lenses, label);
    }
}

#[test]
fn factory_tables_stitch_bit_exactly() {
    for (name, text) in [("V6", V6), ("V3", V3)] {
        let lenses = parse_lens_string(text, Some(CROP), [1920, 1920]).unwrap();
        let stitch = LensStitch::new(FACTORY_TILE, LENS, |k, d| {
            lenses[k].project_from_panorama(d)
        })
        .unwrap();
        assert!(stitch
            .tables()
            .iter()
            .all(|table| table.samples().contains(&RemapTable::INVALID)));
        let images = [noise(LENS, 3), noise(LENS, 4)];
        assert_stitch_matches(&stitch, &images, name);
    }
}

#[test]
fn a_reused_stitch_carries_only_the_newest_frameset() {
    let stitch = synthetic_stitch();
    let client = gpu_client().unwrap();
    let tables = TileTables::upload(&client, &stitch).unwrap();
    let mut builder = GpuPyramidBuilder::new(client);
    let mut pyramids: Vec<_> = (0..2)
        .map(|_| builder.allocate(256, 264, LEVELS).unwrap())
        .collect();
    let size = ImageSize {
        width: 200,
        height: 180,
    };
    let first = [noise(size, 5), noise(size, 6)];
    let second = [noise(size, 7), noise(size, 8)];
    device_pyramids(&mut builder, &tables, &first, &mut pyramids);
    device_pyramids(&mut builder, &tables, &second, &mut pyramids);
    for k in 0..2 {
        assert_tile_pyramid(
            &cpu_tile(&second[k], &stitch.tables()[k]),
            &pyramids[k],
            "second frameset",
        );
    }
}

/// A valid table uploads (the table type refuses malformed ones on the CPU, see `kornia-staging-imgproc`); lens
/// images of another size are refused at the stitch.
#[test]
fn a_valid_table_uploads_and_other_lens_sizes_are_refused() {
    let client = gpu_client().unwrap();
    let stitch = synthetic_stitch();
    let tables = TileTables::upload(&client, &stitch).unwrap();
    assert_eq!(tables.tile_size(), stitch.tile_size());
    let mut builder = GpuPyramidBuilder::new(client);
    let mut pyramids: Vec<_> = (0..2)
        .map(|_| builder.allocate(256, 264, LEVELS).unwrap())
        .collect();
    let wrong = ImageSize {
        width: 198,
        height: 180,
    };
    assert!(builder
        .prepare_lenses(
            &packed(&[noise(wrong, 1), noise(wrong, 2)]),
            wrong,
            &tables,
            &mut pyramids
        )
        .is_err());
}

/// The clip's tables and frame `frame`'s two lens images in lens order, or `None` with the missing asset named.
fn clip_frame(frame: usize) -> Option<(LensStitch, [Image<u8, 1>; 2])> {
    let (path, clip) = clip()?;
    let stitch = LensStitch::new(FACTORY_TILE, LENS, |k, d| {
        clip.lenses_v6[k].project_from_panorama(d)
    })
    .unwrap();
    let lenses = [0, 1].map(|k| decode(&path, clip.lens_track[k], frame));
    Some((stitch, lenses))
}

#[test]
#[ignore = "needs the X4 Air clip VID_20261116_125200_00_003.insv (KORNIA_STAGING_INSV) and ffmpeg"]
fn a_real_frame_pair_stitches_bit_exactly() {
    let Some((stitch, lenses)) = clip_frame(3862) else {
        return;
    };
    assert_stitch_matches(&stitch, &lenses, "frame 3862");
}
