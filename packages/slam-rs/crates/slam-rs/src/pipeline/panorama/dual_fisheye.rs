//! The dual-fisheye source (the Insta360 lane): two lens images in, the two
//! hemisphere tiles out, through fixed-point lookup tables built once from the
//! factory lens calibration; and the rolling-shutter correction of the tiles'
//! keypoints (the readout convention is [`DualFisheyeLenses::readout_s`]'s).
//!
//! Frames: `P` is the panorama frame of `kornia_staging_io::insta360` (`+Z` the
//! lens-0 axis, `+Y` toward the handle). Tile `k`'s camera frame is `P` turned
//! by `k * pi` about `Y`; its origin is lens `k`'s centre (two-centre lane) or
//! the lens midpoint (central lane).

use super::super::VioError;
use crate::calib::{BasaltCamera, Calibration, EquirectangularParams};
use crate::config::VioConfig;
use crate::estimator::FlowObservations;
use crate::frontend::flow::TrackKillMask;
use crate::frontend::input::LensStitch;
use crate::lie::{Se3, So3};
use kornia_image::ImageSize;
use kornia_staging_3d::camera::{CameraModel, Equirectangular};
use kornia_staging_imgproc::panorama::{LUT_INVALID, TileGeometry, TileLut, build_tile_lut};
use kornia_staging_io::insta360::{LensCalibration, WindowCrop, parse_lens_string};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use std::collections::VecDeque;
use std::sync::Arc;

/// Pixels around an invalid table pixel where a tracked keypoint is dropped:
/// the radius of the 51-point patch at level 0 plus one pixel of rounding.
pub const KILL_RADIUS_PX: usize = 4;

/// How long gyro samples are kept for the rolling-shutter rate, nanoseconds.
const GYRO_KEEP_NS: i64 = 200_000_000;

/// The factory calibration the dual-fisheye lane is built from.
#[derive(Debug, Clone)]
pub struct DualFisheyeLenses {
    /// The two lenses, in lens order (not track order), at the lens image's pixels.
    pub lenses: [LensCalibration; 2],
    /// Rolling-shutter readout time of one lens image, seconds: row `v` of an
    /// `H`-row image is read at `t + readout_s * (v - (H - 1) / 2) / H` relative
    /// to the frame time `t`. Positive reads top to bottom; negative bottom to top.
    pub readout_s: f64,
}

impl DualFisheyeLenses {
    /// Parse a factory lens string (V3 or V6) for `lens_size` lens images.
    ///
    /// # Errors
    ///
    /// [`VioError::Panorama`] when the string does not parse or holds other than two lenses.
    pub fn parse(
        lens_string: &str,
        window_crop: Option<WindowCrop>,
        lens_size: [u32; 2],
        readout_s: f64,
    ) -> Result<Self, VioError> {
        let lenses: Vec<LensCalibration> =
            parse_lens_string(lens_string, window_crop, lens_size)
                .map_err(|error| VioError::Panorama(format!("factory lens string: {error}")))?;
        let lenses: [LensCalibration; 2] = lenses.try_into().map_err(|all: Vec<_>| {
            VioError::Panorama(format!(
                "a dual fisheye has two lenses; the string holds {}",
                all.len()
            ))
        })?;
        if !readout_s.is_finite() {
            return Err(VioError::Panorama(format!(
                "readout time {readout_s} s is not finite"
            )));
        }
        Ok(Self { lenses, readout_s })
    }

    /// `T_P_lens`: lens `k`'s pose in the panorama frame (factory rotation and centre).
    pub fn pano_from_lens(&self, lens: usize) -> Se3<f64> {
        let lens: &LensCalibration = &self.lenses[lens];
        Se3::new(
            rotation(&lens.pano_from_lens_rotation),
            Vector3::from(lens.pano_centre_m),
        )
    }

    /// The tables that stitch `lens` images into the two 960-row hemisphere tiles
    /// with `margin_px` overlap columns: the one stitch the lane and the viewing
    /// panorama ([`LensStitch::panorama`]) share. Tile `k` reads lens `k`.
    ///
    /// # Errors
    ///
    /// [`VioError::Panorama`] for a margin wider than half a hemisphere (480 px)
    /// or a lens image below 4x4, before any table is built.
    pub fn stitch(&self, lens: ImageSize, margin_px: usize) -> Result<LensStitch, VioError> {
        let geometry: TileGeometry = TileGeometry {
            hemisphere_width: 960,
            height: 960,
            margin: margin_px,
        };
        if margin_px > geometry.hemisphere_width / 2 {
            return Err(VioError::Panorama(format!(
                "a {margin_px} px margin is wider than half the {} px hemisphere",
                geometry.hemisphere_width
            )));
        }
        if lens.width < 4 || lens.height < 4 {
            return Err(VioError::Panorama(format!(
                "a {}x{} lens image is below 4x4",
                lens.width, lens.height
            )));
        }
        let luts: [TileLut; 2] = std::array::from_fn(|k| {
            build_tile_lut(geometry, k, lens, |ray| {
                self.lenses[k].project_from_panorama(ray)
            })
        });
        Ok(LensStitch::new(luts, lens)?)
    }
}

fn rotation(rows: &[[f64; 3]; 3]) -> So3<f64> {
    let matrix: Matrix3<f64> = Matrix3::from_fn(|row, column| rows[row][column]);
    So3::from_unit_quaternion(UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix(&matrix),
    ))
}

/// The tile cameras, extrinsics and tables of one dual-fisheye rig.
pub(super) struct Expanded {
    pub ingest: DualFisheyeIngest,
    pub calibration: Calibration<f64>,
    pub geometry: TileGeometry,
}

/// Turn a two-lens calibration (lens cameras posed in the IMU frame) into the
/// two tile cameras and build the lookup tables.
///
/// `T_imu_P` comes from lens 0: `T_imu_lens0 * T_P_lens0^-1`; lens 1 must agree.
pub(super) fn expand(
    config: &VioConfig,
    calibration: Calibration<f64>,
    lenses: DualFisheyeLenses,
) -> Result<Expanded, VioError> {
    if !config.panorama_dual_fisheye {
        return Err(VioError::Panorama(
            "factory lenses were given but config.panorama_dual_fisheye is false".to_owned(),
        ));
    }
    if calibration.intrinsics.len() != 2 || calibration.resolution.len() != 2 {
        return Err(VioError::Panorama(format!(
            "a dual fisheye rig has two lens cameras; this rig has {}",
            calibration.intrinsics.len()
        )));
    }
    let [lens_w, lens_h] = calibration.resolution[0];
    if calibration.resolution[1] != [lens_w, lens_h] || lens_w % 2 != 0 || lens_h % 2 != 0 {
        return Err(VioError::Panorama(format!(
            "both lens images must share one even size; got {:?}",
            calibration.resolution
        )));
    }
    let imu_from_pano: Se3<f64> = calibration.t_i_c[0] * lenses.pano_from_lens(0).inverse();
    let check: Se3<f64> =
        (imu_from_pano * lenses.pano_from_lens(1)).inverse() * calibration.t_i_c[1];
    if check.rotation.log().norm() > 1e-3 || check.translation.norm() > 1e-3 {
        return Err(VioError::Panorama(format!(
            "lens 1's extrinsics disagree with lens 0's and the factory rig by {:.2e} rad, {:.2e} m",
            check.rotation.log().norm(),
            check.translation.norm()
        )));
    }
    let lens_size: ImageSize = ImageSize {
        width: lens_w as usize,
        height: lens_h as usize,
    };
    // Built once per rig: the GPU lane uploads these tables once and the lookahead
    // recognises the stitch by identity.
    let stitch: Arc<LensStitch> =
        Arc::new(lenses.stitch(lens_size, config.panorama_tile_margin_px as usize)?);
    let geometry: TileGeometry = stitch.luts()[0].geometry;
    let focal: f64 = geometry.focal();
    let [cx, cy] = geometry.principal();
    let tile: EquirectangularParams<f64> = EquirectangularParams {
        fx: focal,
        fy: focal,
        cx,
        cy,
        lon_max: geometry.lon_max(),
    };
    let t_i_c: Vec<Se3<f64>> = (0..2)
        .map(|k| {
            let origin: Vector3<f64> = if config.panorama_two_centre {
                Vector3::from(lenses.lenses[k].pano_centre_m)
            } else {
                Vector3::zeros()
            };
            let turn: So3<f64> = So3::exp(&Vector3::new(0.0, std::f64::consts::PI * k as f64, 0.0));
            imu_from_pano * Se3::new(turn, origin)
        })
        .collect();
    let model: Equirectangular<f64> =
        Equirectangular::new([tile.fx, tile.fy, tile.cx, tile.cy, tile.lon_max])
            .map_err(|error| VioError::Panorama(format!("tile camera: {error:?}")))?;
    let ingest: DualFisheyeIngest = DualFisheyeIngest {
        tile_from_imu: [t_i_c[0].rotation.inverse(), t_i_c[1].rotation.inverse()],
        stitch,
        readout_s: lenses.readout_s,
        rolling_shutter: config.panorama_rolling_shutter,
        gyro: VecDeque::new(),
        tile_camera: model,
        last_ns: 0,
    };
    let mut tiles: Calibration<f64> = calibration;
    tiles.intrinsics = vec![BasaltCamera::Equirectangular(tile); 2];
    tiles.t_i_c = t_i_c;
    tiles.resolution = vec![[geometry.width() as u32, geometry.height as u32]; 2];
    Ok(Expanded {
        ingest,
        calibration: tiles,
        geometry,
    })
}

/// Per-rig state of the dual-fisheye ingest; every buffer is reused.
pub(super) struct DualFisheyeIngest {
    /// The two tables, shared with the frame storage that stitches through them.
    stitch: Arc<LensStitch>,
    readout_s: f64,
    rolling_shutter: bool,
    /// Recent raw gyro samples (rig = IMU frame), oldest first.
    gyro: VecDeque<(i64, Vector3<f64>)>,
    /// Both tiles share one camera model; only their extrinsics differ.
    tile_camera: Equirectangular<f64>,
    /// `R_tile_imu` per tile.
    tile_from_imu: [So3<f64>; 2],
    /// Wall time of the last host-side ingest ([`super::PanoramaRig::ingest`]), ns.
    last_ns: u64,
}

impl std::fmt::Debug for DualFisheyeIngest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DualFisheyeIngest")
            .field("lens_size", &self.stitch.lens_size())
            .field("readout_s", &self.readout_s)
            .field("rolling_shutter", &self.rolling_shutter)
            .finish_non_exhaustive()
    }
}

impl DualFisheyeIngest {
    /// Lens image `(width, height)` each frameset must carry twice.
    pub fn lens_size(&self) -> ImageSize {
        self.stitch.lens_size()
    }

    /// The lookup table of tile `k`.
    pub fn lut(&self, tile: usize) -> &TileLut {
        &self.stitch.luts()[tile]
    }

    /// The stitch the frame storage takes the lens images through.
    pub fn stitch(&self) -> &Arc<LensStitch> {
        &self.stitch
    }

    /// Record the wall time of the last host-side ingest.
    pub fn set_last_ns(&mut self, ns: u64) {
        self.last_ns = ns;
    }

    /// Wall time of the last host-side ingest, ns: the CPU remap on the CPU lane, the lens byte copy on the GPU
    /// lane (whose stitch runs on the device inside the frontend's pyramid build).
    pub fn last_ns(&self) -> u64 {
        self.last_ns
    }

    /// Per tile, whether each pixel's table entry is invalid (outside the lens).
    pub fn invalid(&self, tile: usize) -> Vec<bool> {
        self.lut(tile)
            .sample
            .iter()
            .map(|&sample| sample == LUT_INVALID)
            .collect()
    }

    /// Per tile, the invalid pixels grown by [`KILL_RADIUS_PX`] (Chebyshev).
    pub fn kill_mask(&self, tile: usize) -> Arc<TrackKillMask> {
        let geometry: &TileGeometry = &self.lut(tile).geometry;
        let (width, height) = (geometry.width(), geometry.height);
        let invalid: Vec<bool> = self.invalid(tile);
        let mut killed: Vec<bool> = vec![false; width * height];
        let radius: usize = KILL_RADIUS_PX;
        for row in 0..height {
            for column in 0..width {
                if !invalid[row * width + column] {
                    continue;
                }
                for y in row.saturating_sub(radius)..(row + radius + 1).min(height) {
                    killed[y * width + column.saturating_sub(radius)
                        ..y * width + (column + radius + 1).min(width)]
                        .fill(true);
                }
            }
        }
        Arc::new(TrackKillMask {
            width,
            height,
            killed,
        })
    }

    /// Keep a raw gyro sample for the rolling-shutter rate.
    pub fn push_gyro(&mut self, t_ns: i64, gyro: [f64; 3]) {
        self.gyro.push_back((t_ns, Vector3::from(gyro)));
        while self
            .gyro
            .front()
            .is_some_and(|&(t, _)| t < t_ns - GYRO_KEEP_NS)
        {
            self.gyro.pop_front();
        }
    }

    /// The gyro rate at frame time `t_ns` (IMU frame): the mean of the samples
    /// inside the readout window around it, else the nearest within 20 ms.
    fn gyro_rate(&self, t_ns: i64) -> Option<Vector3<f64>> {
        let half: i64 = (0.5 * self.readout_s.abs() * 1e9) as i64;
        let inside: Vec<Vector3<f64>> = self
            .gyro
            .iter()
            .filter(|&&(t, _)| (t - t_ns).abs() <= half)
            .map(|&(_, rate)| rate)
            .collect();
        if !inside.is_empty() {
            return Some(inside.iter().sum::<Vector3<f64>>() / inside.len() as f64);
        }
        self.gyro
            .iter()
            .min_by_key(|&&(t, _)| (t - t_ns).abs())
            .filter(|&&(t, _)| (t - t_ns).abs() <= 20_000_000)
            .map(|&(_, rate)| rate)
    }

    /// The estimator's copy of one frameset's keypoints with the rolling shutter
    /// removed, when `config.panorama_rolling_shutter` is on and a gyro rate
    /// exists at the frame time; otherwise unchanged. `gyro_bias` is the
    /// estimator's latest bias (IMU frame), subtracted when known. A keypoint
    /// whose lens row is unknown keeps its raw position.
    pub fn correct_rolling_shutter(
        &self,
        observations: &mut FlowObservations,
        gyro_bias: Option<Vector3<f64>>,
    ) {
        if !self.rolling_shutter {
            return;
        }
        let Some(rate) = self.gyro_rate(observations.t_ns) else {
            return;
        };
        let rate: Vector3<f64> = rate - gyro_bias.unwrap_or_else(Vector3::zeros);
        for (tile, keypoints) in observations.cameras.iter_mut().enumerate() {
            let omega: Vector3<f64> = self.tile_from_imu[tile] * rate;
            for pixel in keypoints.values_mut() {
                if let Some([u, v]) = rolling_shutter_correct(
                    [f64::from(pixel.x), f64::from(pixel.y)],
                    &self.tile_camera,
                    self.lut(tile),
                    self.lens_size().height,
                    self.readout_s,
                    &omega,
                ) {
                    *pixel = nalgebra::Vector2::new(u as f32, v as f32);
                }
            }
        }
    }
}

/// Lens row (full-resolution) under a tile pixel: bilinear over the four table
/// neighbours when all are valid, else the nearest one's, else `None`.
pub fn lens_row(lut: &TileLut, pixel: [f64; 2]) -> Option<f64> {
    let (width, height) = (lut.geometry.width(), lut.geometry.height);
    let [u, v] = pixel;
    if !(u >= 0.0 && v >= 0.0 && u <= (width - 1) as f64 && v <= (height - 1) as f64) {
        return None;
    }
    let (x0, y0) = (u.floor() as usize, v.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (ax, ay) = (u - x0 as f64, v - y0 as f64);
    let row = |x: usize, y: usize| f64::from(lut.source_row[y * width + x]);
    let corners: [f64; 4] = [row(x0, y0), row(x1, y0), row(x0, y1), row(x1, y1)];
    if corners.iter().all(|value| value.is_finite()) {
        let top: f64 = corners[0] * (1.0 - ax) + corners[1] * ax;
        let bottom: f64 = corners[2] * (1.0 - ax) + corners[3] * ax;
        return Some(top * (1.0 - ay) + bottom * ay);
    }
    let nearest: f64 = row(u.round() as usize, v.round() as usize);
    nearest.is_finite().then_some(nearest)
}

/// One keypoint observed by a rolling-shutter lens, moved to where a global
/// shutter at the frame time (the middle row) would have seen it.
///
/// The keypoint's lens row gives `dt = readout_s * (row - (H - 1) / 2) / H`.
/// The camera turned by `exp(omega * dt)` (`omega` in the tile camera's frame,
/// rad/s) between the frame time and that row, so the frame-time bearing of a
/// static point is `exp(omega * dt) * b_observed`; it is projected back with
/// the tile camera. Translation during the readout is ignored.
///
/// `None` when the row is unknown or the bearing leaves the camera.
pub fn rolling_shutter_correct(
    pixel: [f64; 2],
    camera: &Equirectangular<f64>,
    lut: &TileLut,
    lens_height: usize,
    readout_s: f64,
    omega: &Vector3<f64>,
) -> Option<[f64; 2]> {
    let row: f64 = lens_row(lut, pixel)?;
    let height: f64 = lens_height as f64;
    let dt: f64 = readout_s * (row - 0.5 * (height - 1.0)) / height;
    let bearing: Vector3<f64> = Vector3::from(camera.unproject(pixel).ok()?);
    let turned: Vector3<f64> = So3::exp(&(omega * dt)) * bearing;
    camera.project([turned.x, turned.y, turned.z]).ok()
}

#[cfg(test)]
mod tests;
