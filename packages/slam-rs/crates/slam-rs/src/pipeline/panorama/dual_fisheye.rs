//! The dual-fisheye source: two back-to-back cameras' images in, the two
//! hemisphere tiles out, through fixed-point lookup tables built once from the
//! calibration's two camera models; and the rolling-shutter correction of the
//! tiles' keypoints. Readout: row `v` of an `H`-row lens image is read at
//! `t + readout_s * (v - (H - 1) / 2) / H` relative to the frame time `t`.
//!
//! Frames: `P` has camera 0's axes and its origin at the midpoint of the two
//! camera centres. Tile `k`'s camera frame is `P` turned by `k * pi` about `Y`;
//! its origin is camera `k`'s centre.

use super::super::VioError;
use crate::calib::{BasaltCamera, Calibration, EquirectangularParams};
use crate::config::VioConfig;
use crate::estimator::FlowObservations;
use crate::frontend::flow::TrackKillMask;
use crate::lie::{Se3, So3};
use kornia_image::ImageSize;
use kornia_staging_3d::camera::{CameraModel, CameraModelKind, Equirectangular};
use kornia_staging_imgproc::panorama::{LensStitch, RemapTable, TileGeometry};
use nalgebra::Vector3;
use kornia_staging_slam::init::inertial::{ImuSample, sample_at};
use std::sync::Arc;

/// Pixels around an invalid table pixel where a tracked keypoint is dropped:
/// the radius of the 51-point patch at level 0 plus one pixel of rounding.
pub const KILL_RADIUS_PX: usize = 4;

/// Columns of one hemisphere tile's 180 deg, and its rows (pole to pole): the tiles of a 1920x960 panorama.
const HEMISPHERE_PX: usize = 960;


/// The stitch of two back-to-back cameras into the two 960-row hemisphere tiles of
/// the panorama frame `P` with `margin_px` overlap columns: the one stitch the lane
/// and a viewing panorama ([`LensStitch::panorama`]) share. Tile `k` reads camera `k`.
///
/// `P` has camera 0's axes; tile 0 looks along camera 0's `+Z`, tile 1 along `-Z`.
/// `cam0_from_cam1` is camera 1's pose in camera 0; only its rotation enters the
/// tables (the tiles see the far field).
///
/// # Errors
///
/// [`VioError::Panorama`] for a margin wider than half a hemisphere (480 px)
/// or a lens image below 4x4, before any table is built.
pub fn stitch(
    cameras: [&CameraModelKind<f64>; 2],
    cam0_from_cam1: &Se3<f64>,
    lens: ImageSize,
    margin_px: usize,
) -> Result<LensStitch, VioError> {
    let geometry: TileGeometry = TileGeometry {
        hemisphere_width: HEMISPHERE_PX,
        height: HEMISPHERE_PX,
        margin: margin_px,
    };
    // The arithmetic of dataforge's panorama layer, so both stitch the same pixels.
    let relative: nalgebra::Rotation3<f64> =
        nalgebra::Rotation3::from_matrix_unchecked(cam0_from_cam1.rotation.matrix());
    LensStitch::new(geometry, lens, |k, ray| {
        let local: Vector3<f64> = if k == 0 {
            Vector3::from(ray)
        } else {
            relative.inverse_transform_vector(&Vector3::from(ray))
        };
        cameras[k].project(local.into()).ok()
    })
    .map_err(|error| VioError::Panorama(error.to_string()))
}

/// The tile cameras, extrinsics and tables of one dual-fisheye rig.
pub(super) struct Expanded {
    pub ingest: DualFisheyeIngest,
    pub rolling_shutter: Option<RollingShutter>,
    pub calibration: Calibration<f64>,
    pub geometry: TileGeometry,
}

/// Turn a two-camera calibration (back-to-back cameras posed in the IMU frame)
/// into the two tile cameras and build the lookup tables.
///
/// `P`: camera 0's axes, origin at the midpoint of the two camera centres.
pub(super) fn expand(
    config: &VioConfig,
    calibration: Calibration<f64>,
) -> Result<Expanded, VioError> {
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
    let readout_s: f64 = match calibration.readout_s.as_slice() {
        [first, second] if first == second => first.unwrap_or(0.0),
        [] => 0.0,
        other => {
            return Err(VioError::Panorama(format!(
                "the two lenses must share one rolling-shutter readout; got {other:?}"
            )));
        }
    };
    let models: Vec<CameraModelKind<f64>> = calibration
        .intrinsics
        .iter()
        .map(|camera| {
            camera
                .to_staged()
                .map_err(|error| VioError::Panorama(format!("lens camera: {error}")))
        })
        .collect::<Result<_, _>>()?;
    let imu_from_pano: Se3<f64> = Se3::new(
        calibration.t_i_c[0].rotation,
        (calibration.t_i_c[0].translation + calibration.t_i_c[1].translation) * 0.5,
    );
    let lens_size: ImageSize = ImageSize {
        width: lens_w as usize,
        height: lens_h as usize,
    };
    // Built once per rig: the GPU lane uploads these tables once and the lookahead
    // recognises the stitch by identity.
    let cam0_from_cam1: Se3<f64> = calibration.t_i_c[0].inverse() * calibration.t_i_c[1];
    let stitch: Arc<LensStitch> = Arc::new(stitch(
        [&models[0], &models[1]],
        &cam0_from_cam1,
        lens_size,
        super::TILE_MARGIN_PX,
    )?);
    let geometry: TileGeometry = stitch.geometry();
    let model: Equirectangular<f64> = geometry
        .camera()
        .map_err(|error| VioError::Panorama(format!("tile camera: {error:?}")))?;
    let [fx, fy, cx, cy, lon_max] = model.params();
    let tile: EquirectangularParams<f64> = EquirectangularParams {
        fx,
        fy,
        cx,
        cy,
        lon_max,
    };
    let t_i_c: Vec<Se3<f64>> = (0..2)
        .map(|k| {
            let origin: Vector3<f64> = (imu_from_pano.inverse() * calibration.t_i_c[k]).translation;
            let turn: So3<f64> = So3::exp(&Vector3::new(0.0, std::f64::consts::PI * k as f64, 0.0));
            imu_from_pano * Se3::new(turn, origin)
        })
        .collect();
    let rolling_shutter: Option<RollingShutter> =
        (config.panorama_rolling_shutter && readout_s != 0.0).then(|| RollingShutter {
            stitch: Arc::clone(&stitch),
            readout_s,
            tile_camera: model,
            tile_from_imu: [t_i_c[0].rotation.inverse(), t_i_c[1].rotation.inverse()],
            gyro: Vec::new(),
            corrected: 0,
            no_coverage: 0,
        });
    let ingest: DualFisheyeIngest = DualFisheyeIngest { stitch, last_ns: 0 };
    let mut tiles: Calibration<f64> = calibration;
    tiles.intrinsics = vec![BasaltCamera::Equirectangular(tile); 2];
    tiles.t_i_c = t_i_c;
    tiles.resolution = vec![[geometry.width() as u32, geometry.height as u32]; 2];
    Ok(Expanded {
        ingest,
        rolling_shutter,
        calibration: tiles,
        geometry,
    })
}

/// Per-rig state of the dual-fisheye ingest; every buffer is reused.
pub(super) struct DualFisheyeIngest {
    /// The two tables, shared with the frame storage that stitches through them.
    stitch: Arc<LensStitch>,
    /// Wall time of the last host-side ingest ([`super::PanoramaRig::ingest`]), ns.
    last_ns: u64,
}

impl std::fmt::Debug for DualFisheyeIngest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DualFisheyeIngest")
            .field("lens_size", &self.stitch.lens_size())
            .finish_non_exhaustive()
    }
}

impl DualFisheyeIngest {
    /// Lens image `(width, height)` each frameset must carry twice.
    pub fn lens_size(&self) -> ImageSize {
        self.stitch.lens_size()
    }

    /// The lookup table of tile `k`.
    pub fn table(&self, tile: usize) -> &RemapTable {
        &self.stitch.tables()[tile]
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
        self.table(tile)
            .samples()
            .iter()
            .map(|&sample| sample == RemapTable::INVALID)
            .collect()
    }

    /// Per tile, the invalid pixels grown by [`KILL_RADIUS_PX`] (Chebyshev).
    pub fn kill_mask(&self, tile: usize) -> Arc<TrackKillMask> {
        let ImageSize { width, height } = self.stitch.tile_size();
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
}

/// The rolling-shutter correction of a two-lens rig's tile keypoints (`config.panorama_rolling_shutter` with a
/// non-zero readout): the rate of a frame is the calibrated gyro interpolated at the frame time, less the
/// estimator's bias. The gyro buffer is trimmed by the frame times consumed, so how the IMU is batched does not
/// change a correction.
pub struct RollingShutter {
    /// The tables whose fixed-point entries give each tile pixel's lens row.
    stitch: Arc<LensStitch>,
    readout_s: f64,
    /// Both tiles share one camera model; only their extrinsics differ.
    tile_camera: Equirectangular<f64>,
    /// `R_tile_imu` per tile.
    tile_from_imu: [So3<f64>; 2],
    /// Calibrated gyro samples (IMU frame), oldest first, from the last one at or before the last corrected frame.
    gyro: Vec<ImuSample>,
    corrected: usize,
    no_coverage: usize,
}

impl std::fmt::Debug for RollingShutter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RollingShutter")
            .field("readout_s", &self.readout_s)
            .field("corrected", &self.corrected)
            .field("no_coverage", &self.no_coverage)
            .finish_non_exhaustive()
    }
}

impl RollingShutter {
    /// Keep one calibrated gyro sample (IMU frame).
    pub fn push_gyro(&mut self, t_ns: i64, gyro: Vector3<f64>) {
        self.gyro.push(ImuSample {
            t_ns,
            gyro,
            accel: Vector3::zeros(),
        });
    }

    /// Frames corrected so far, and frames left raw because the gyro did not cover their time.
    pub fn counts(&self) -> (usize, usize) {
        (self.corrected, self.no_coverage)
    }

    /// The estimator's copy of one frameset's keypoints with the rolling shutter removed. `gyro_bias` is the
    /// estimator's latest bias (IMU frame), subtracted when known. A frame the gyro does not cover on both sides
    /// stays raw and is counted ([`Self::counts`]); a keypoint whose lens row is unknown keeps its raw position.
    pub fn correct(&mut self, observations: &mut FlowObservations, gyro_bias: Option<Vector3<f64>>) {
        let t_ns: i64 = observations.t_ns;
        let before: usize = self.gyro.partition_point(|sample| sample.t_ns <= t_ns);
        if before == 0 || self.gyro.last().is_none_or(|sample| sample.t_ns < t_ns) {
            self.no_coverage += 1;
            return;
        }
        let (rate, _) = sample_at(&self.gyro, t_ns);
        // Frames arrive in time order: what precedes this frame's left neighbour is never read again.
        self.gyro.drain(..before - 1);
        self.corrected += 1;
        let rate: Vector3<f64> = rate - gyro_bias.unwrap_or_else(Vector3::zeros);
        let lens_height: usize = self.stitch.lens_size().height;
        for (tile, keypoints) in observations.cameras.iter_mut().enumerate() {
            let omega: Vector3<f64> = self.tile_from_imu[tile] * rate;
            let table: &RemapTable = &self.stitch.tables()[tile];
            for pixel in keypoints.values_mut() {
                if let Some([u, v]) = rolling_shutter_correct(
                    [f64::from(pixel.x), f64::from(pixel.y)],
                    &self.tile_camera,
                    table,
                    lens_height,
                    self.readout_s,
                    &omega,
                ) {
                    *pixel = nalgebra::Vector2::new(u as f32, v as f32);
                }
            }
        }
    }
}

/// Lens row (full-resolution) under a tile pixel, from the table's fixed-point
/// entries (`qy / 128`: an entry is the level-1 row in 1/256 px): bilinear over
/// the four table neighbours when all are valid, else the nearest one's, else
/// `None`.
pub fn lens_row(table: &RemapTable, pixel: [f64; 2]) -> Option<f64> {
    let ImageSize { width, height } = table.dst_size();
    let [u, v] = pixel;
    if !(u >= 0.0 && v >= 0.0 && u <= (width - 1) as f64 && v <= (height - 1) as f64) {
        return None;
    }
    let (x0, y0) = (u.floor() as usize, v.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (ax, ay) = (u - x0 as f64, v - y0 as f64);
    let row = |x: usize, y: usize| {
        let q: [u32; 2] = table.samples()[y * width + x];
        (q != RemapTable::INVALID).then(|| f64::from(q[1]) / 128.0)
    };
    if let (Some(r00), Some(r10), Some(r01), Some(r11)) =
        (row(x0, y0), row(x1, y0), row(x0, y1), row(x1, y1))
    {
        let top: f64 = r00 * (1.0 - ax) + r10 * ax;
        let bottom: f64 = r01 * (1.0 - ax) + r11 * ax;
        return Some(top * (1.0 - ay) + bottom * ay);
    }
    row(u.round() as usize, v.round() as usize)
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
    table: &RemapTable,
    lens_height: usize,
    readout_s: f64,
    omega: &Vector3<f64>,
) -> Option<[f64; 2]> {
    let row: f64 = lens_row(table, pixel)?;
    let height: f64 = lens_height as f64;
    let dt: f64 = readout_s * (row - 0.5 * (height - 1.0)) / height;
    let bearing: Vector3<f64> = Vector3::from(camera.unproject(pixel).ok()?);
    let turned: Vector3<f64> = So3::exp(&(omega * dt)) * bearing;
    camera.project([turned.x, turned.y, turned.z]).ok()
}

#[cfg(test)]
mod tests;
