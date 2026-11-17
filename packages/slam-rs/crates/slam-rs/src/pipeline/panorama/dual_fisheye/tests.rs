#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::ImageView;
use crate::frontend::input::FrameStorage;
use crate::pipeline::panorama::{PanoramaRig, PanoramaSource};
use kornia_image::Image;
use kornia_staging_imgproc::panorama::{assemble_panorama, lens_level1, remap_tile};
use std::f64::consts::PI;

/// The X4 Air clip's V6 factory string (recon/09), as `kornia-staging-io`'s tests hold it.
const V6: &str = "2_2.314940_7096.760_7095.390_3834.830_3864.120_0.119_0.407_90.629_0.000000_0.000000_0.000000_0.93861496_-1.68402135_2.19166017_3.85762072_0.00000000_-0.00495962_0.00156076_-0.00330722_0.00397486_0.00827898_-0.00125315_0.02416405_-0.00143547_15360_7680_131_2.314940_7149.220_7149.700_11543.300_3858.430_-0.212_0.339_88.279_0.000132_-0.000043_-0.033291_0.95675713_-1.80496192_2.88544273_1.76948690_0.00000000_0.00131924_0.00023973_0.00279902_-0.00570679_0.00069985_0.00064811_0.00799088_0.02677096_15360_7680_131_394240";
const CROP: WindowCrop = WindowCrop {
    src: [7680, 7680],
    dst: [7424, 7424],
    offset: [0, 0],
};
const READOUT_S: f64 = 0.015883;

fn lenses() -> DualFisheyeLenses {
    DualFisheyeLenses::parse(V6, Some(CROP), [1920, 1920], READOUT_S).unwrap()
}

/// `R_imu_P` of recon/09 (the nearest permutation): `x_P = -z_imu, y_P = x_imu, z_P = -y_imu`.
fn imu_from_pano() -> Se3<f64> {
    let p_from_imu = Matrix3::new(0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0);
    let r: So3<f64> = So3::from_unit_quaternion(UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(p_from_imu.transpose()),
    ));
    Se3::new(r, Vector3::new(0.001, -0.002, 0.003))
}

/// A two-lens rig as the converter writes it: `T_imu_lens = T_imu_P * T_P_lens`.
fn lens_calibration(lenses: &DualFisheyeLenses) -> Calibration<f64> {
    let mut calibration =
        Calibration::from_json_str(include_str!("../../../../tests/fixtures/msdmg_calib.json"))
            .unwrap();
    calibration.t_i_c = (0..2)
        .map(|k| imu_from_pano() * lenses.pano_from_lens(k))
        .collect();
    calibration.resolution = vec![[1920, 1920]; 2];
    calibration.intrinsics.truncate(2);
    calibration.vignette.clear();
    calibration
}

fn config(two_centre: bool) -> VioConfig {
    VioConfig {
        optical_flow_detection_grid_size: 64,
        optical_flow_stereo_matching: false,
        panorama_dual_fisheye: true,
        panorama_two_centre: two_centre,
        panorama_rolling_shutter: true,
        ..VioConfig::default()
    }
}

fn rig(two_centre: bool) -> (PanoramaRig, Calibration<f64>) {
    let lenses = lenses();
    PanoramaRig::expand_dual_fisheye(&config(two_centre), lens_calibration(&lenses), lenses)
        .unwrap()
}

fn tile_model(tiles: &Calibration<f64>, tile: usize) -> Equirectangular<f64> {
    let BasaltCamera::Equirectangular(p) = tiles.intrinsics[tile] else {
        panic!("tile cameras are equirectangular");
    };
    Equirectangular::new([p.fx, p.fy, p.cx, p.cy, p.lon_max]).unwrap()
}

#[test]
fn tiles_are_the_designed_hemisphere_cameras() {
    let (rig, tiles) = rig(true);
    assert_eq!(rig.source(), PanoramaSource::DualFisheye);
    assert_eq!(rig.tile_size(), (1088, 960));
    assert_eq!(rig.panorama_size(), (1920, 960));
    assert_eq!(rig.tile_origins(), [-64, 896]);
    let BasaltCamera::Equirectangular(p) = tiles.intrinsics[1] else {
        panic!("equirectangular");
    };
    assert!((p.fx - 960.0 / PI).abs() < 1e-12 && p.fx == p.fy);
    assert_eq!((p.cx, p.cy), (543.5, 479.5));
    assert!((p.lon_max - 1088.0 / (2.0 * 960.0 / PI)).abs() < 1e-12);
    assert_eq!(tiles.resolution, vec![[1088, 960]; 2]);
    // Two centres 33.1 mm apart; the central lane puts both at the midpoint.
    let baseline = (tiles.t_i_c[0].translation - tiles.t_i_c[1].translation).norm();
    assert!((baseline - 0.0331).abs() < 3e-4, "{baseline}");
    let (_, central) = super::super::PanoramaRig::expand_dual_fisheye(
        &config(false),
        lens_calibration(&lenses()),
        lenses(),
    )
    .unwrap();
    assert!((central.t_i_c[0].translation - central.t_i_c[1].translation).norm() < 1e-12);
    assert!((central.t_i_c[0].translation - imu_from_pano().translation).norm() < 1e-12);
}

#[test]
fn a_far_point_lands_where_the_tile_camera_and_the_table_agree() {
    let lenses = lenses();
    for two_centre in [true, false] {
        let (rig, tiles) = rig(two_centre);
        for tile in 0..2 {
            let model = tile_model(&tiles, tile);
            let lut = rig.tile_lut(tile).unwrap();
            let lens_from_imu: Se3<f64> = (imu_from_pano() * lenses.pano_from_lens(tile)).inverse();
            let mut checked = 0;
            for tv in (40..920).step_by(97) {
                for tu in (70..1018).step_by(89) {
                    let index = tv * 1088 + tu;
                    if lut.sample[index] == LUT_INVALID {
                        continue;
                    }
                    let bearing = Vector3::from(model.unproject([tu as f64, tv as f64]).unwrap());
                    // 10 km along the tile pixel's ray, in the IMU frame, seen through the lens's factory model.
                    let point: Vector3<f64> = tiles.t_i_c[tile] * (bearing * 1.0e4);
                    let ray: Vector3<f64> = lens_from_imu * point;
                    let [u, v] = lenses.lenses[tile].project([ray.x, ray.y, ray.z]).unwrap();
                    let [qx, qy] = lut.sample[index];
                    let (tu_lens, tv_lens) = (f64::from(qx) / 128.0, f64::from(qy) / 128.0);
                    assert!(
                        (tu_lens - u).abs() < 0.05 && (tv_lens - v).abs() < 0.05,
                        "tile {tile} ({tu}, {tv}): table ({tu_lens}, {tv_lens}) vs model ({u}, {v})"
                    );
                    checked += 1;
                }
            }
            assert!(checked > 80, "{checked}");
        }
    }
}

#[test]
fn a_near_point_shows_the_lens_offset_parallax_between_the_two_lanes() {
    let (_, two) = rig(true);
    let (_, central) = rig(false);
    // 1 m from the midpoint, perpendicular to the lens axis (longitude 90 deg in P, on the equator).
    let point: Vector3<f64> = imu_from_pano() * Vector3::new(1.0, 0.0, 0.0);
    let model = tile_model(&two, 0);
    let pixel = |tiles: &Calibration<f64>| {
        let local: Vector3<f64> = tiles.t_i_c[0].inverse() * point;
        model.project([local.x, local.y, local.z]).unwrap()
    };
    let (a, b) = (pixel(&two), pixel(&central));
    let offset = lenses().lenses[0].pano_centre_m;
    // Lens 0 sits +16.56 mm along Z, so the point is seen 0.0166 rad further from the lens axis.
    let expected = 960.0 / PI * (offset[2] / (1.0 - offset[0])).atan();
    assert!(
        (a[0] - b[0] - expected).abs() < 0.05,
        "{} vs {expected}",
        a[0] - b[0]
    );
    assert!(expected > 5.0);
}

fn textured_lens(seed: u32) -> Vec<u8> {
    (0..1920 * 1920)
        .map(|index: u32| {
            let (x, y) = (index % 1920, index / 1920);
            ((x * 7 + y * 13 + seed * 31) ^ (x / 9 * 5 + y / 11 * 3)).wrapping_mul(2_654_435_761)
                as u8
                >> 1
                | ((x + y) % 2) as u8
        })
        .collect()
}

#[test]
fn both_lanes_hand_the_lens_images_to_the_storage_and_see_the_kornia_staging_remap() {
    let (mut rig, _) = rig(true);
    // A padded stride: the CPU lane reads the rows in place, no dense copy first.
    let stride: usize = 1924;
    let lenses_px: [Vec<u8>; 2] = [textured_lens(1), textured_lens(2)];
    let padded: Vec<Vec<u8>> = lenses_px
        .iter()
        .map(|dense| {
            dense
                .chunks_exact(1920)
                .flat_map(|row| row.iter().copied().chain([7u8; 4]))
                .collect()
        })
        .collect();
    let views: Vec<ImageView<'_>> = padded
        .iter()
        .map(|data| ImageView {
            width: 1920,
            height: 1920,
            stride,
            data,
        })
        .collect();
    let references: Vec<Image<u16, 1>> = (0..2)
        .map(|tile| {
            let lens = Image::<u8, 1>::new(
                ImageSize {
                    width: 1920,
                    height: 1920,
                },
                lenses_px[tile].clone(),
            )
            .unwrap();
            let mut reference = Image::from_size_val(
                ImageSize {
                    width: 1088,
                    height: 960,
                },
                0u16,
            )
            .unwrap();
            remap_tile(
                &lens_level1(&lens).unwrap(),
                rig.tile_lut(tile).unwrap(),
                &mut reference,
            )
            .unwrap();
            reference
        })
        .collect();
    let geometry: TileGeometry = rig.tile_lut(0).unwrap().geometry;
    let panorama: Image<u8, 1> =
        assemble_panorama([&references[0], &references[1]], geometry).unwrap();
    // The CPU lane (dense) remaps on the host; the GPU lane (packed) keeps the lens bytes for the device
    // stitch, and its host view (detector fallback, panorama logging) is the same remap, never a u8 round trip.
    for gpu in [false, true] {
        let mut storage = FrameStorage::new(gpu);
        for _ in 0..2 {
            // Twice: the reused scratch must not leak the previous frameset.
            rig.ingest(&views, &mut storage).unwrap();
        }
        let frames = storage.view();
        assert_eq!(frames.len(), 2);
        for (tile, reference) in references.iter().enumerate() {
            assert_eq!(frames.get(tile).size(), reference.size());
            frames
                .get(tile)
                .with_dense(|image| {
                    assert_eq!(
                        image.as_slice(),
                        reference.as_slice(),
                        "gpu {gpu}, tile {tile}"
                    );
                })
                .unwrap();
        }
        let logged = rig.panorama_image(&storage).unwrap().unwrap();
        assert_eq!(logged.as_slice(), panorama.as_slice(), "gpu {gpu}");
    }
    // A frameset that is not two lens images is refused before anything moves.
    let mut dense = FrameStorage::new(false);
    assert!(rig.ingest(&views[..1], &mut dense).is_err());
}

#[test]
fn masks_are_whole_cells_over_margins_poles_and_every_invalid_pixel() {
    let (rig, tiles) = rig(true);
    let BasaltCamera::Equirectangular(p) = tiles.intrinsics[0] else {
        panic!("equirectangular")
    };
    let masks = rig.detection_masks(&config(true), p.fy, p.cy).unwrap();
    let masked_cells: usize = masks[0].masks.len();
    for (tile, cells) in masks.iter().enumerate() {
        let lut = rig.tile_lut(tile).unwrap();
        for rect in &cells.masks {
            assert_eq!((rect.w, rect.h), (64.0, 64.0));
            assert_eq!(rect.x % 64.0, 0.0);
            assert_eq!(rect.y % 64.0, 0.0);
        }
        let mut invalid = 0;
        for (index, sample) in lut.sample.iter().enumerate() {
            if *sample != LUT_INVALID {
                continue;
            }
            invalid += 1;
            let (x, y) = ((index % 1088) as f32, (index / 1088) as f32);
            assert!(
                cells
                    .masks
                    .iter()
                    .any(|r| x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h),
                "tile {tile}: invalid pixel ({x}, {y}) outside every masked cell"
            );
        }
        assert!(invalid > 0, "margin 64 reaches past the lens coverage");
    }
    assert!(
        masked_cells >= 2 * 15 + 2 * (17 - 2),
        "margin columns and polar rows: {masked_cells}"
    );
}

#[test]
fn the_kill_mask_covers_invalid_pixels_and_their_patch_radius() {
    let (rig, _) = rig(true);
    let dual = rig.dual.as_deref().unwrap();
    for tile in 0..2 {
        let mask = dual.kill_mask(tile);
        let invalid = dual.invalid(tile);
        let index = invalid.iter().position(|&flag| flag).unwrap();
        let (x, y) = ((index % 1088) as f32, (index / 1088) as f32);
        assert!(mask.kills(x, y));
        assert!(mask.kills((x + KILL_RADIUS_PX as f32).min(1087.0), y));
        assert!(
            !mask.kills(543.0, 479.0),
            "the tile centre is inside the lens"
        );
        assert!(mask.kills(-1.0, 10.0), "off the image");
    }
}

#[test]
fn the_key_and_the_lenses_must_agree() {
    let lenses = lenses();
    let off = VioConfig {
        panorama_dual_fisheye: false,
        ..config(true)
    };
    assert!(
        PanoramaRig::expand_dual_fisheye(&off, lens_calibration(&lenses), lenses.clone()).is_err()
    );
    let mut swapped = lens_calibration(&lenses);
    swapped.t_i_c.swap(0, 1);
    assert!(
        PanoramaRig::expand_dual_fisheye(&config(true), swapped, lenses).is_err(),
        "lens order is checked"
    );
}

/// An equidistant test lens looking along tile `k`'s axis: `r = 560 theta` about the image centre.
fn equidistant(tile: usize, ray_p: [f64; 3]) -> Option<[f64; 2]> {
    let sign = if tile == 0 { 1.0 } else { -1.0 };
    let [x, y, z] = [sign * ray_p[0], ray_p[1], sign * ray_p[2]];
    let norm = (x * x + y * y + z * z).sqrt();
    let theta = (z / norm).clamp(-1.0, 1.0).acos();
    if theta > 100f64.to_radians() {
        return None;
    }
    let phi = y.atan2(x);
    Some([
        959.5 + 560.0 * theta * phi.cos(),
        959.5 + 560.0 * theta * phi.sin(),
    ])
}

#[test]
fn the_rolling_shutter_correction_recovers_the_global_shutter_pixel_at_one_rad_per_second() {
    let geometry = TileGeometry::default();
    let focal = geometry.focal();
    let [cx, cy] = geometry.principal();
    let camera = Equirectangular::new([focal, focal, cx, cy, geometry.lon_max()]).unwrap();
    let lens = ImageSize {
        width: 1920,
        height: 1920,
    };
    for tile in 0..2 {
        let lut = build_tile_lut(geometry, tile, lens, |ray| equidistant(tile, ray));
        // Tile camera frame = P turned by k pi about Y.
        let pano_from_tile = So3::exp(&Vector3::new(0.0, PI * tile as f64, 0.0));
        for omega in [
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.6, -0.48, 0.64),
            Vector3::new(1.0, 0.0, 0.0),
        ] {
            let mut worst: f64 = 0.0;
            let mut worst_flipped: f64 = 0.0;
            for lat_deg in [-70.0_f64, -40.0, -10.0, 0.0, 20.0, 50.0, 72.0] {
                for lon_deg in [-85.0_f64, -50.0, -20.0, 0.0, 30.0, 60.0, 88.0] {
                    let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
                    // A static far point; the camera's frame at the frame time is the world frame.
                    let world =
                        Vector3::new(lat.cos() * lon.sin(), -lat.sin(), lat.cos() * lon.cos());
                    let truth = camera.project([world.x, world.y, world.z]).unwrap();
                    // Row-by-row exposure: solve tau = readout (row(p(tau)) - 959.5) / 1920.
                    let mut tau = 0.0;
                    let mut observed = truth;
                    for _ in 0..20 {
                        let seen: Vector3<f64> = So3::exp(&(omega * tau)).inverse() * world;
                        observed = camera.project([seen.x, seen.y, seen.z]).unwrap();
                        let ray_p =
                            pano_from_tile * Vector3::from(camera.unproject(observed).unwrap());
                        let row = equidistant(tile, [ray_p.x, ray_p.y, ray_p.z]).unwrap()[1];
                        tau = READOUT_S * (row - 959.5) / 1920.0;
                    }
                    let corrected =
                        rolling_shutter_correct(observed, &camera, &lut, 1920, READOUT_S, &omega)
                            .unwrap();
                    let flipped =
                        rolling_shutter_correct(observed, &camera, &lut, 1920, -READOUT_S, &omega)
                            .unwrap();
                    let error = |p: [f64; 2]| (p[0] - truth[0]).hypot(p[1] - truth[1]);
                    worst = worst.max(error(corrected));
                    worst_flipped = worst_flipped.max(error(flipped));
                }
            }
            assert!(worst < 0.05, "tile {tile} omega {omega:?}: {worst} px");
            assert!(
                worst_flipped > 1.0,
                "the wrong sign doubles the error: {worst_flipped} px"
            );
        }
    }
}

#[test]
fn the_rig_rotates_the_gyro_into_each_tile_and_skips_without_a_rate() {
    let (mut rig, tiles) = rig(true);
    let mut observations = FlowObservations::new(1_000_000_000, 2);
    observations.cameras[0].insert(
        crate::types::KeypointId(1),
        nalgebra::Vector2::new(300.0, 200.0),
    );
    observations.cameras[1].insert(
        crate::types::KeypointId(2),
        nalgebra::Vector2::new(700.0, 800.0),
    );
    let raw = observations.clone();
    rig.correct_rolling_shutter(&mut observations, None);
    assert_eq!(observations, raw, "no gyro sample, no correction");
    let omega_imu = Vector3::new(0.4, -0.8, 0.3);
    for step in -20..=20 {
        rig.push_gyro(1_000_000_000 + step * 1_000_000, omega_imu.into());
    }
    rig.correct_rolling_shutter(&mut observations, None);
    let dual = rig.dual.as_deref().unwrap();
    for tile in 0..2 {
        let (_, raw_pixel) = raw.cameras[tile].iter().next().unwrap();
        let omega_tile = tiles.t_i_c[tile].rotation.inverse() * omega_imu;
        let expected = rolling_shutter_correct(
            [f64::from(raw_pixel.x), f64::from(raw_pixel.y)],
            &tile_model(&tiles, tile),
            dual.lut(tile),
            1920,
            READOUT_S,
            &omega_tile,
        )
        .unwrap();
        let (_, got) = observations.cameras[tile].iter().next().unwrap();
        assert!(
            (f64::from(got.x) - expected[0]).abs() < 1e-3
                && (f64::from(got.y) - expected[1]).abs() < 1e-3
        );
        assert!(
            (got - raw_pixel).norm() > 0.1,
            "the correction moved tile {tile}'s keypoint"
        );
    }
}
