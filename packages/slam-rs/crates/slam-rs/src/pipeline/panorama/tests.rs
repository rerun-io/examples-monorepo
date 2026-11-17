#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::frontend::flow::{FrameToFrameOpticalFlow, FrontendOptions, PosePrediction};
use kornia_image::Image;
use kornia_staging_3d::camera::{CameraModel, Equirectangular};
use kornia_staging_imgproc::optical_flow::patch_se2::Pattern51;
use std::collections::BTreeSet;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// A one-camera rig whose camera is a full `width x height` panorama.
fn panorama_calibration(width: u32, height: u32) -> Calibration<f64> {
    let mut calibration =
        Calibration::from_json_str(include_str!("../../../tests/fixtures/msdmg_calib.json"))
            .unwrap();
    let (w, h) = (f64::from(width), f64::from(height));
    calibration.intrinsics = vec![BasaltCamera::Equirectangular(EquirectangularParams {
        fx: w / TAU,
        fy: h / PI,
        cx: w / 2.0 - 0.5,
        cy: h / 2.0 - 0.5,
        lon_max: PI,
    })];
    calibration.resolution = vec![[width, height]];
    calibration.t_i_c.truncate(1);
    calibration.vignette.clear();
    calibration
}

fn panorama_config(stereo_matching: bool) -> VioConfig {
    VioConfig {
        optical_flow_detection_grid_size: 64,
        optical_flow_stereo_matching: stereo_matching,
        ..VioConfig::default()
    }
}

fn expanded(width: u32, height: u32) -> (PanoramaRig, Calibration<f64>) {
    let (rig, tiles) = PanoramaRig::expand(panorama_calibration(width, height)).unwrap();
    (rig.expect("a full panorama expands"), tiles)
}

/// Tile `tile`'s camera model; the dual-fisheye tests share it.
pub(super) fn tile_model(tiles: &Calibration<f64>, tile: usize) -> Equirectangular<f64> {
    let BasaltCamera::Equirectangular(p) = tiles.intrinsics[tile] else {
        panic!("tile cameras are equirectangular");
    };
    Equirectangular::new([p.fx, p.fy, p.cx, p.cy, p.lon_max]).unwrap()
}

#[test]
fn tiles_have_the_designed_sizes_origins_and_cameras() {
    for (width, height, tile_width, origins, cells) in [
        (2048, 1024, 1152, [448, 1472], (18, 16)),
        (1920, 960, 1088, [416, 1376], (17, 15)),
    ] {
        let (rig, tiles) = expanded(width, height);
        assert_eq!(rig.tile_size(), (tile_width, height as usize));
        assert_eq!(rig.tile_origins(), origins);
        assert!(rig.tile_lut(0).is_none());
        assert_eq!(tiles.resolution, vec![[tile_width as u32, height]; 2]);
        let BasaltCamera::Equirectangular(p) = tiles.intrinsics[1] else {
            panic!("equirectangular tiles");
        };
        assert_eq!(p.fx, f64::from(width) / TAU);
        assert_eq!(p.fy, f64::from(height) / PI);
        assert_eq!(p.cx, (tile_width as f64 - 1.0) / 2.0);
        assert_eq!(p.cy, (f64::from(height) - 1.0) / 2.0);
        assert!((p.lon_max - (FRAC_PI_2 + 64.0 * TAU / f64::from(width))).abs() < 1e-12);
        // Same centre, turned half a turn about the vertical axis.
        let relative: Se3<f64> = tiles.t_i_c[0].inverse() * tiles.t_i_c[1];
        assert!(relative.translation.norm() < 1e-12);
        let turned: Vector3<f64> = relative.rotation * Vector3::new(0.0, 0.0, 1.0);
        assert!((turned - Vector3::new(0.0, 0.0, -1.0)).norm() < 1e-12);
        let up: Vector3<f64> = relative.rotation * Vector3::new(0.0, 1.0, 0.0);
        assert!((up - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-12);
        let grid = CellGrid::new(tile_width, height as usize, 64).unwrap();
        assert_eq!(grid.dimensions(), cells);
    }
}

#[test]
fn a_rig_without_a_full_panorama_is_returned_unchanged() {
    let calibration =
        Calibration::from_json_str(include_str!("../../../tests/fixtures/msdmg_calib.json"))
            .unwrap();
    let (rig, same) = PanoramaRig::expand(calibration.clone()).unwrap();
    assert!(rig.is_none());
    assert_eq!(same.t_i_c, calibration.t_i_c);
    assert_eq!(same.intrinsics, calibration.intrinsics);
}

#[test]
fn a_panorama_beside_other_cameras_is_refused() {
    let mut calibration = panorama_calibration(2048, 1024);
    calibration.intrinsics.push(calibration.intrinsics[0]);
    calibration.resolution.push(calibration.resolution[0]);
    calibration.t_i_c.push(calibration.t_i_c[0]);
    let error = PanoramaRig::expand(calibration).unwrap_err();
    assert!(matches!(error, VioError::Panorama(_)), "{error}");
}

#[test]
fn a_width_that_puts_an_outer_tile_pixel_on_lon_max_is_refused() {
    // `W % 4 == 2`: the rounded tile origin puts one outer pixel centre exactly
    // at `lon_max`.
    for width in [1922, 2050] {
        let error = PanoramaRig::expand(panorama_calibration(width, 960))
            .unwrap_err();
        assert!(matches!(error, VioError::Panorama(_)), "{error}");
    }
    // A multiple of 4 keeps both outer pixel centres half a pixel inside.
    for width in [1920, 2048] {
        let (_, tiles) = expanded(width, 960);
        let BasaltCamera::Equirectangular(p) = tiles.intrinsics[0] else {
            panic!("equirectangular tiles");
        };
        let tile_width: f64 = f64::from(tiles.resolution[0][0]);
        for x in [0.0, tile_width - 1.0] {
            let slack: f64 = p.lon_max * p.fx - (x - p.cx).abs();
            assert!((slack - 0.5).abs() < 1e-9, "{width}: {slack}");
        }
    }
}

#[test]
fn the_crop_is_a_pure_copy_with_horizontal_wrap() {
    let (width, height): (usize, usize) = (1920, 960);
    let (mut rig, _) = expanded(width as u32, height as u32);
    let stride: usize = width + 3;
    let value = |x: usize, y: usize| ((x * 7 + y * 13) % 251) as u8;
    let mut data: Vec<u8> = vec![0; stride * height];
    for y in 0..height {
        for x in 0..width {
            data[y * stride + x] = value(x, y);
        }
    }
    let image = ImageView {
        width,
        height,
        stride,
        data: &data,
    };
    rig.crop(&[image]).unwrap();
    let (tile_width, _) = rig.tile_size();
    for (tile, view) in rig.tile_views().iter().enumerate() {
        assert_eq!(
            (view.width, view.height, view.stride),
            (tile_width, height, tile_width)
        );
        for y in 0..height {
            for x in 0..tile_width {
                let source: usize =
                    (rig.tile_origins()[tile] + x as i64).rem_euclid(width as i64) as usize;
                assert_eq!(
                    view.data[y * tile_width + x],
                    value(source, y),
                    "tile {tile} ({x}, {y})"
                );
            }
        }
    }
    // Tile 1 runs off the right edge: its column 544 is panorama column 0.
    let wrap: usize = width - 1376;
    assert_eq!(wrap, 544);
    assert_eq!(rig.tile_views()[1].data[wrap], value(0, 0));
    assert_eq!(rig.tile_views()[1].data[wrap - 1], value(width - 1, 0));
    assert_eq!(rig.panorama_x(1, wrap as f32), 0.0);
    assert_eq!(rig.panorama_x(1, wrap as f32 - 0.5), width as f32 - 0.5);
    assert_eq!(rig.panorama_x(0, 0.0), 416.0);
}

#[test]
fn a_frameset_that_is_not_one_panorama_is_refused() {
    let (mut rig, _) = expanded(2048, 1024);
    let data: Vec<u8> = vec![0; 2048 * 1024];
    let wrong = ImageView {
        width: 1024,
        height: 2048,
        stride: 1024,
        data: &data,
    };
    assert!(rig.crop(&[wrong]).is_err());
    let right = ImageView {
        width: 2048,
        height: 1024,
        stride: 2048,
        data: &data,
    };
    assert!(rig.crop(&[right, right]).is_err());
    assert!(rig.crop(&[right]).is_ok());
}

#[test]
fn a_bearing_lands_in_its_tile_where_the_panorama_model_puts_it() {
    let (width, height) = (2048_u32, 1024_u32);
    let (rig, tiles) = expanded(width, height);
    let pano = Equirectangular::<f64>::panorama(width, height).unwrap();
    let models = [tile_model(&tiles, 0), tile_model(&tiles, 1)];
    // Rotation taking a panorama-frame bearing into each tile's frame.
    let tile_from_pano: [Se3<f64>; 2] =
        [0, 1].map(|tile| tiles.t_i_c[tile].inverse() * tiles.t_i_c[0]);
    let lon_max: f64 = models[0].lon_max();
    for step in 0..=360 {
        let lon: f64 = -PI + TAU * f64::from(step) / 360.0 + 1e-7;
        for lat in [-1.2_f64, -0.4, 0.0, 0.7, 1.25] {
            let bearing = Vector3::new(lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos());
            let Ok(expected) = pano.project([bearing.x, bearing.y, bearing.z]) else {
                continue;
            };
            for tile in 0..2 {
                let local: Vector3<f64> = tile_from_pano[tile].rotation * bearing;
                let tile_lon: f64 = local.x.atan2(local.z);
                match models[tile].project([local.x, local.y, local.z]) {
                    Ok([u, v]) => {
                        assert!(tile_lon.abs() <= lon_max);
                        let x: f64 =
                            (u + rig.tile_origins()[tile] as f64).rem_euclid(f64::from(width));
                        let dx: f64 = (x - expected[0]).abs();
                        assert!(
                            dx.min(f64::from(width) - dx) < 1e-6,
                            "tile {tile} lon {lon}: {x} vs {}",
                            expected[0]
                        );
                        assert!((v - expected[1]).abs() < 1e-6);
                    }
                    Err(_) => assert!(
                        tile_lon.abs() > lon_max - 1e-9,
                        "tile {tile} rejects lon {tile_lon}"
                    ),
                }
            }
        }
    }
    // The window edge: valid just inside in the owning tile, refused just past
    // it; a bearing at the opposite tile's centre is outside the other window.
    for (lon, tile0, tile1) in [
        (lon_max - 1e-6, true, true),
        (lon_max + 1e-3, false, true),
        (FRAC_PI_2 - 64.0 * TAU / 2048.0 - 1e-3, true, false),
        (0.0, true, false),
        (PI - 1e-3, false, true),
    ] {
        let bearing = Vector3::new(lon.sin(), 0.0, lon.cos());
        for (tile, valid) in [(0, tile0), (1, tile1)] {
            let local: Vector3<f64> = tile_from_pano[tile].rotation * bearing;
            assert_eq!(
                models[tile].project([local.x, local.y, local.z]).is_ok(),
                valid,
                "lon {lon} tile {tile}"
            );
        }
    }
}

#[test]
fn detection_masks_are_whole_cells_over_the_margins_and_the_polar_rows() {
    let config = panorama_config(false);
    for (width, height, polar_rows) in [(2048_u32, 1024_u32, [0, 15]), (1920, 960, [0, 14])] {
        let (rig, tiles) = expanded(width, height);
        let BasaltCamera::Equirectangular(p) = tiles.intrinsics[0] else {
            panic!("equirectangular tiles");
        };
        let masks = rig.detection_masks(&config, p.fy, p.cy).unwrap();
        let (tile_width, tile_height) = rig.tile_size();
        let grid = CellGrid::new(tile_width, tile_height, 64).unwrap();
        let (columns, rows) = grid.dimensions();
        let mut masked: BTreeSet<(usize, usize)> = BTreeSet::new();
        for rect in &masks[0].masks {
            assert_eq!((rect.w, rect.h), (64.0, 64.0));
            assert_eq!(rect.x % 64.0, 0.0);
            assert_eq!(rect.y % 64.0, 0.0);
            masked.insert((rect.x as usize / 64, rect.y as usize / 64));
        }
        let expected: BTreeSet<(usize, usize)> = grid
            .cells()
            .filter(|(column, row)| {
                *column == 0 || *column == columns - 1 || polar_rows.contains(row)
            })
            .collect();
        assert_eq!(masked, expected);
        assert_eq!(masks[1], masks[0]);
        assert!(rows > 2);
        // The frontend takes them as whole cells.
        let mut flow = FrameToFrameOpticalFlow::<Pattern51>::new(
            config.clone(),
            &tiles,
            FrontendOptions::default(),
        )
        .unwrap();
        flow.set_detection_masks(masks).unwrap();
        let misaligned = CellMasks {
            masks: vec![MaskRect {
                x: 10.0,
                y: 0.0,
                w: 64.0,
                h: 64.0,
            }],
        };
        assert!(
            flow.set_detection_masks(vec![misaligned, CellMasks::default()])
                .is_err()
        );
    }
}

/// A blocky pseudo-random panorama shifted right by `shift` columns, with wrap.
fn textured_panorama(width: usize, height: usize, shift: usize) -> Vec<u8> {
    let mut data: Vec<u8> = vec![0; width * height];
    for y in 0..height {
        for x in 0..width {
            let block: u64 =
                (((x + width - shift) % width) / 7) as u64 * 7919 + (y / 7) as u64 * 104_729;
            let hashed: u64 = block.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56;
            data[y * width + x] = 40 + (hashed as u8) / 2;
        }
    }
    data
}

/// Track two framesets of a shifted textured panorama through the tile frontend.
fn track_tiles(stereo_matching: bool) -> FrameToFrameOpticalFlow<Pattern51> {
    let (width, height): (usize, usize) = (1024, 512);
    let config = panorama_config(stereo_matching);
    let (mut rig, tiles) = expanded(width as u32, height as u32);
    let mut flow = FrameToFrameOpticalFlow::<Pattern51>::new(
        config.clone(),
        &tiles,
        FrontendOptions::default(),
    )
    .unwrap();
    let BasaltCamera::Equirectangular(p) = tiles.intrinsics[0] else {
        panic!("equirectangular tiles");
    };
    flow.set_detection_masks(rig.detection_masks(&config, p.fy, p.cy).unwrap())
        .unwrap();
    for (frame, shift) in [0, 2].into_iter().enumerate() {
        let data = textured_panorama(width, height, shift);
        rig.crop(&[ImageView {
            width,
            height,
            stride: width,
            data: &data,
        }])
        .unwrap();
        let images: Vec<Image<u16, 1>> = rig
            .tile_views()
            .iter()
            .map(|view| {
                crate::image::from_u8_strided(view.data, view.width, view.height, view.stride)
                    .unwrap()
            })
            .collect();
        flow.process_frame(
            frame as i64 * 100_000_000,
            &images,
            &PosePrediction::default(),
            &[],
        )
        .unwrap();
    }
    flow
}

fn shared_ids(flow: &FrameToFrameOpticalFlow<Pattern51>) -> usize {
    let zero: BTreeSet<_> = flow.frame().cameras[0].ids.iter().collect();
    flow.frame().cameras[1]
        .ids
        .iter()
        .filter(|id| zero.contains(id))
        .count()
}

#[test]
fn with_stereo_matching_off_both_tiles_detect_and_share_no_id() {
    let flow = track_tiles(false);
    assert!(!flow.matches_into(1));
    assert_eq!(shared_ids(&flow), 0);
    for camera in 0..2 {
        let keypoints = &flow.frame().cameras[camera];
        assert!(
            keypoints.len() > 20,
            "camera {camera} has {} keypoints",
            keypoints.len()
        );
    }
    // Something was tracked from the first frameset into the second.
    assert!(flow.last_keypoint_id_before_frame() > 0);
}

#[test]
fn a_shared_centre_is_never_matched_even_with_stereo_matching_on() {
    let flow = track_tiles(true);
    assert!(!flow.matches_into(1));
    assert!(flow.essential(1).iter().all(|value| *value == 0.0));
    assert_eq!(shared_ids(&flow), 0);
    assert!(flow.frame().cameras[1].len() > 20);
}

#[test]
fn tracks_above_the_latitude_limit_are_dropped() {
    let (width, height): (usize, usize) = (1024, 512);
    let config = panorama_config(false);
    let (mut rig, tiles) = expanded(width as u32, height as u32);
    let BasaltCamera::Equirectangular(p) = tiles.intrinsics[0] else {
        panic!("equirectangular tiles");
    };
    let limit: f64 = 40_f64.to_radians();
    let kill_masks = rig.track_kill_masks(p.fy, p.cy, limit);
    let mut run = |limited: bool| {
        let mut flow = FrameToFrameOpticalFlow::<Pattern51>::new(
            config.clone(),
            &tiles,
            FrontendOptions::default(),
        )
        .unwrap();
        if limited {
            flow.set_track_kill_masks(kill_masks.iter().cloned().map(Some).collect())
                .unwrap();
        }
        for (frame, shift) in [0, 1].into_iter().enumerate() {
            let data = textured_panorama(width, height, shift);
            rig.crop(&[ImageView {
                width,
                height,
                stride: width,
                data: &data,
            }])
            .unwrap();
            let images: Vec<Image<u16, 1>> = rig
                .tile_views()
                .iter()
                .map(|view| {
                    crate::image::from_u8_strided(view.data, view.width, view.height, view.stride)
                        .unwrap()
                })
                .collect();
            flow.process_frame(
                frame as i64 * 100_000_000,
                &images,
                &PosePrediction::default(),
                &[],
            )
            .unwrap();
        }
        flow
    };
    let free = run(false);
    let limited = run(true);
    // The mask holds whole rows: a keypoint is judged by its rounded row.
    let latitude = |y: f32| ((f64::from(y.round()) - p.cy) / p.fy).abs();
    let tracked_above = |flow: &FrameToFrameOpticalFlow<Pattern51>| {
        let watermark = flow.last_keypoint_id_before_frame();
        (0..flow.frame().cameras[0].len())
            .filter(|index| flow.frame().cameras[0].ids[*index].0 < watermark)
            .filter(|index| {
                latitude(flow.frame().cameras[0].transforms.translation(*index)[1]) > limit
            })
            .count()
    };
    assert!(tracked_above(&free) > 0);
    assert_eq!(tracked_above(&limited), 0);
}
