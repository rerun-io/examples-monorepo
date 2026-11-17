//! The device stitch through the GPU frontend: lens images in give the same frontend frames as the CPU-remapped
//! tiles uploaded as dense images.
#![allow(clippy::unwrap_used)]

use super::{GpuStages, gpu_stages};
use crate::ImageView;
use crate::calib::{BasaltCamera, Calibration, EquirectangularParams};
use crate::config::VioConfig;
use crate::frontend::flow::{FrameToFrameOpticalFlow, FrontendOptions, PosePrediction};
use crate::frontend::input::{FrameImages, FrameStorage, PackedImages};
use crate::frontend::parallel::WorkPool;
use crate::pipeline::PanoramaRig;
use kornia_image::{Image, ImageSize};
use kornia_staging_gpu::GpuRuntime;
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern51;
use kornia_staging_imgproc::panorama::{LensStitch, RemapTable, TileGeometry, lens_level1, remap};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

type GpuFlow = FrameToFrameOpticalFlow<Pattern51, GpuStages<Pattern51, GpuRuntime>>;

const LENS: ImageSize = ImageSize {
    width: 1920,
    height: 1920,
};

/// The two 1088x960 tile cameras of a 1920x960 panorama (margin 64), with stereo matching off as in the panorama
/// lane.
fn tile_rig() -> (VioConfig, Calibration<f64>) {
    let config = VioConfig {
        optical_flow_detection_grid_size: 64,
        optical_flow_stereo_matching: false,
        ..VioConfig::default()
    };
    let mut calibration =
        Calibration::from_json_str(include_str!("../../tests/fixtures/msdmg_calib.json")).unwrap();
    calibration.intrinsics = vec![BasaltCamera::Equirectangular(EquirectangularParams {
        fx: 1920.0 / TAU,
        fy: 960.0 / PI,
        cx: 959.5,
        cy: 479.5,
        lon_max: PI,
    })];
    calibration.resolution = vec![[1920, 960]];
    calibration.t_i_c.truncate(1);
    calibration.vignette.clear();
    let (rig, tiles) = PanoramaRig::expand(&config, calibration).unwrap();
    assert!(rig.is_some());
    assert_eq!(tiles.resolution, vec![[1088, 960]; 2]);
    (config, tiles)
}

fn gpu_flow(config: &VioConfig, tiles: &Calibration<f64>) -> GpuFlow {
    let stages = gpu_stages::<Pattern51>(
        FrontendOptions::default().max_keypoints,
        config.optical_flow_levels as usize + 1,
        config.optical_flow_max_iterations as usize,
        config.optical_flow_max_recovered_dist2,
        2,
    )
    .unwrap();
    FrameToFrameOpticalFlow::with_stages(
        config.clone(),
        tiles,
        FrontendOptions::default(),
        stages,
        WorkPool::new(1).unwrap(),
    )
    .unwrap()
}

/// The stitch of two synthetic back-to-back equidistant lenses (560 px/rad, 100 deg half field) into the 1088x960
/// tiles.
fn lens_stitch() -> LensStitch {
    let geometry = TileGeometry {
        hemisphere_width: 960,
        height: 960,
        margin: 64,
    };
    LensStitch::new(geometry, LENS, |tile, d| {
        // Lens 1 looks along -Z: its frame is P turned 180 deg about Y.
        let [x, y, z] = if tile == 0 { d } else { [-d[0], d[1], -d[2]] };
        let theta = z.clamp(-1.0, 1.0).acos();
        let rho = x.hypot(y);
        if theta > 100f64.to_radians() || rho == 0.0 {
            return (theta == 0.0).then_some([959.5, 959.5]);
        }
        let r = 560.0 * theta / rho;
        Some([959.5 + r * x, 959.5 + r * y])
    })
    .unwrap()
}

/// A blocky texture, moved by `shift` pixels per frame.
fn lens_image(lens: usize, shift: usize) -> Vec<u8> {
    (0..LENS.width * LENS.height)
        .map(|i| {
            let (x, y) = (
                (i % LENS.width + shift) as u64,
                (i / LENS.width + shift / 2) as u64,
            );
            let block = (x / 9) * 7919 + (y / 9) * 104_729 + lens as u64 * 31;
            40 + ((block.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8) / 2
        })
        .collect()
}

fn cpu_tiles(lenses: &[Vec<u8>; 2], luts: &[RemapTable; 2]) -> Vec<Image<u16, 1>> {
    (0..2)
        .map(|k| {
            let lens = Image::new(LENS, lenses[k].clone()).unwrap();
            let mut tile = Image::from_size_val(
                ImageSize {
                    width: 1088,
                    height: 960,
                },
                0u16,
            )
            .unwrap();
            remap(&lens_level1(&lens).unwrap(), &luts[k], &mut tile).unwrap();
            tile
        })
        .collect()
}

#[test]
fn lens_images_through_the_device_stitch_track_like_the_cpu_remapped_tiles() {
    let (config, tiles) = tile_rig();
    let stitch = Arc::new(lens_stitch());
    let mut lens_flow = gpu_flow(&config, &tiles);
    let mut tile_flow = gpu_flow(&config, &tiles);
    let mut packed = PackedImages::default();
    let mut dense = FrameStorage::new(false);
    for frame in 0..4usize {
        let lenses = [lens_image(0, 2 * frame), lens_image(1, 3 * frame)];
        let views: Vec<ImageView<'_>> = lenses
            .iter()
            .map(|data| ImageView {
                data,
                width: LENS.width,
                height: LENS.height,
                stride: LENS.width,
            })
            .collect();
        packed.fill_lenses(&views, &stitch).unwrap();
        let expected = cpu_tiles(&lenses, stitch.tables());

        // The CPU lane's storage and the GPU fallback's host tiles are the CPU remap.
        dense.fill_lenses(&views, &stitch).unwrap();
        let FrameImages::Dense(cpu_lane) = dense.view() else {
            panic!("dense storage");
        };
        for (k, tile) in expected.iter().enumerate() {
            assert_eq!(
                cpu_lane[k].as_slice(),
                tile.as_slice(),
                "CPU lane, tile {k}"
            );
            FrameImages::Packed(&packed)
                .get(k)
                .with_dense(|host| {
                    assert_eq!(host.as_slice(), tile.as_slice(), "host fallback, tile {k}")
                })
                .unwrap();
        }

        let t_ns = (frame as i64 + 1) * 100_000_000;
        let prediction = PosePrediction::default();
        lens_flow
            .process_frame_input(t_ns, FrameImages::Packed(&packed), &prediction, &[])
            .unwrap();
        tile_flow
            .process_frame(t_ns, &expected, &prediction, &[])
            .unwrap();
        assert_eq!(lens_flow.frame(), tile_flow.frame(), "frame {frame}");
        for camera in &lens_flow.frame().cameras {
            assert!(
                camera.ids.len() > 20,
                "frame {frame}: {} keypoints",
                camera.ids.len()
            );
        }
    }
}

#[test]
fn lens_images_of_the_wrong_size_or_count_are_refused() {
    let stitch = Arc::new(lens_stitch());
    let data = vec![0u8; 1918 * 1920];
    let small = ImageView {
        data: &data,
        width: 1918,
        height: 1920,
        stride: 1918,
    };
    let mut packed = PackedImages::default();
    assert!(packed.fill_lenses(&[small, small], &stitch).is_err());
    let full = vec![0u8; 1920 * 1920];
    let view = ImageView {
        data: &full,
        width: 1920,
        height: 1920,
        stride: 1920,
    };
    assert!(packed.fill_lenses(&[view], &stitch).is_err());
}
