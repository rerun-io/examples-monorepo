//! The panorama lane: one 360-degree source in, two hemisphere tiles out.
//!
//! The frontend and the estimator keep seeing "N cameras, N images". This module
//! is the one place where a rig whose calibration holds a full panorama becomes
//! two hemisphere-tile cameras, and each frameset's panorama becomes two tile
//! images. Tile `k` is centred on panorama longitude
//! `k * pi`, is `W / 2 + 2 * margin` pixels wide, and its camera is the panorama
//! camera turned by `k * 180` degrees about its own vertical axis.
//!
//! The tile cut answers two kernel limits at once: KLT, the pyramid and FAST
//! never wrap, and a tile never crosses a lens seam. A track that leaves its tile
//! ends; the other tile detects the point again.

mod dual_fisheye;

pub use dual_fisheye::DualFisheyeLenses;

use super::{ImageView, VioError};
use crate::calib::{BasaltCamera, Calibration, EquirectangularParams};
use crate::config::VioConfig;
use crate::estimator::FlowObservations;
use crate::frontend::flow::TrackKillMask;
use crate::frontend::input::{FrameStorage, PackedImages};
use crate::lie::{Se3, So3};
use dual_fisheye::DualFisheyeIngest;
use kornia_staging_imgproc::features::{CellGrid, CellMasks, MaskRect};
use nalgebra::Vector3;
use std::sync::Arc;

/// Columns each hemisphere tile reaches past its 180-degree half of the
/// panorama, on each side, in panorama pixels: one 64 px detection cell.
pub(super) const TILE_MARGIN_PX: usize = 64;

/// Tracked keypoints above this absolute latitude, degrees, are dropped, and
/// detection skips every cell row whose centre lies above it.
const TRACK_MAX_LATITUDE_DEG: f64 = 75.0;

/// One panorama camera expanded into two hemisphere tiles.
#[derive(Debug)]
pub struct PanoramaRig {
    /// Panorama size in pixels.
    width: usize,
    height: usize,
    /// Tile width in pixels; both tiles are `tile_width x height`.
    tile_width: usize,
    /// Panorama column of each tile's column 0 (may be negative: it wraps).
    origins: [i64; 2],
    /// The two tiles of the last frameset as bytes (equirect crop; unused by
    /// the dual fisheye, whose storage takes the lens images).
    tiles: [Vec<u8>; 2],
    /// Tables, buffers and rolling-shutter state of the dual-fisheye source.
    dual: Option<Box<DualFisheyeIngest>>,
}

impl PanoramaRig {
    /// Expand a calibration that holds one full equirectangular panorama
    /// (`lon_max = pi`) into the two tile cameras; any other rig is returned
    /// unchanged with `None`.
    ///
    /// # Errors
    ///
    /// [`VioError::Panorama`] when a full panorama shares the rig with other
    /// cameras, is not `fx = W / 2pi` with an even width, when the margin
    /// leaves a tile wider than the panorama, or when a tile's outer pixel
    /// centre is not half a pixel inside its `lon_max` (a width that is not a
    /// multiple of 4, for `cx = (W - 1) / 2`).
    pub fn expand(
        calibration: Calibration<f64>,
    ) -> Result<(Option<Self>, Calibration<f64>), VioError> {
        let full: Vec<EquirectangularParams<f64>> = calibration
            .intrinsics
            .iter()
            .filter_map(|camera| match camera {
                BasaltCamera::Equirectangular(params)
                    if params.lon_max >= std::f64::consts::PI - 1e-9 =>
                {
                    Some(*params)
                }
                _ => None,
            })
            .collect();
        let Some(pano) = full.first().copied() else {
            return Ok((None, calibration));
        };
        if calibration.intrinsics.len() != 1 {
            return Err(VioError::Panorama(format!(
                "a full panorama must be the rig's only camera; this rig has {}",
                calibration.intrinsics.len()
            )));
        }
        let [width, height] = calibration.resolution[0].map(|value| value as usize);
        let tau: f64 = std::f64::consts::TAU;
        if width % 2 != 0 || (pano.fx * tau - width as f64).abs() > 1e-6 * width as f64 {
            return Err(VioError::Panorama(format!(
                "a {width}-wide panorama needs an even width and fx = W / 2pi = {}; got fx = {}",
                width as f64 / tau,
                pano.fx
            )));
        }
        let margin: usize = TILE_MARGIN_PX;
        let tile_width: usize = width / 2 + 2 * margin;
        if tile_width > width {
            return Err(VioError::Panorama(format!(
                "a {margin} px margin makes {tile_width} px tiles, wider than the {width} px panorama"
            )));
        }
        // Tile 0's column 0 sits at the integer panorama column nearest the
        // centred position, and the tile's principal point absorbs the rest, so
        // the crop stays a pure copy for any panorama `cx`.
        let origin: i64 = (pano.cx - (tile_width as f64 - 1.0) / 2.0).round() as i64;
        // `lon_max` is `W / 4 + margin` columns from the tile's principal point.
        // Each outer pixel centre must stay at least half a pixel inside it, or
        // its rounding decides on which side of `lon_max` it lands. With
        // `cx = (W - 1) / 2` that holds exactly when `W` is a multiple of 4.
        let tile_cx: f64 = pano.cx - origin as f64;
        let reach: f64 = tile_cx.max(tile_width as f64 - 1.0 - tile_cx) + 0.5;
        if reach > width as f64 / 4.0 + margin as f64 + 1e-9 {
            return Err(VioError::Panorama(format!(
                "a {width}-wide panorama with cx = {} puts a tile's outer pixel on its lon_max; \
                 use a width that is a multiple of 4 with cx = (W - 1) / 2",
                pano.cx
            )));
        }
        let tile = EquirectangularParams {
            fx: pano.fx,
            fy: pano.fy,
            cx: tile_cx,
            cy: pano.cy,
            lon_max: std::f64::consts::FRAC_PI_2 + margin as f64 / pano.fx,
        };
        let half_turn: Se3<f64> = Se3::new(
            So3::exp(&Vector3::new(0.0, std::f64::consts::PI, 0.0)),
            Vector3::zeros(),
        );
        let imu_t_pano: Se3<f64> = calibration.t_i_c[0];
        let mut tiles: Calibration<f64> = calibration;
        tiles.intrinsics = vec![BasaltCamera::Equirectangular(tile); 2];
        tiles.t_i_c = vec![imu_t_pano, imu_t_pano * half_turn];
        tiles.resolution = vec![[tile_width as u32, height as u32]; 2];
        tiles.vignette = std::iter::repeat_n(tiles.vignette.first().cloned(), 2)
            .flatten()
            .collect();
        Ok((
            Some(Self {
                width,
                height,
                tile_width,
                origins: [origin, origin + (width / 2) as i64],
                tiles: [vec![0; tile_width * height], vec![0; tile_width * height]],
                dual: None,
            }),
            tiles,
        ))
    }

    /// Expand a two-lens rig (lens cameras posed in the IMU frame, lens order)
    /// into two equirectangular hemisphere tiles stitched from the lens images
    /// through the factory model; see [`DualFisheyeLenses`].
    ///
    /// Tile `k`: `fx = fy = 960 / pi`, principal point at the tile centre,
    /// `lon_max` to the tile's outer pixel edge; `T_imu_tile_k = T_imu_P *
    /// (Ry(k pi), c_k)` with `c_k` lens `k`'s centre in `P`. The panorama they
    /// form is 1920x960: tile columns `[margin, margin + 960)` of tile 0, then
    /// of tile 1.
    ///
    /// # Errors
    ///
    /// [`VioError::Panorama`] when `config.panorama_dual_fisheye` is off, the
    /// rig is not two equal even-sized lens images, or lens 1's extrinsics
    /// disagree with lens 0's and the factory rig.
    pub fn expand_dual_fisheye(
        config: &VioConfig,
        calibration: Calibration<f64>,
        lenses: DualFisheyeLenses,
    ) -> Result<(Self, Calibration<f64>), VioError> {
        let expanded = dual_fisheye::expand(config, calibration, lenses)?;
        let geometry = expanded.geometry;
        let (tile_width, height) = (geometry.width(), geometry.height);
        let margin: i64 = geometry.margin as i64;
        Ok((
            Self {
                width: 2 * geometry.hemisphere_width,
                height,
                tile_width,
                origins: [-margin, geometry.hemisphere_width as i64 - margin],
                tiles: [Vec::new(), Vec::new()],
                dual: Some(Box::new(expanded.ingest)),
            },
            expanded.calibration,
        ))
    }

    /// Tile `(width, height)`, the same for both tiles.
    pub fn tile_size(&self) -> (usize, usize) {
        (self.tile_width, self.height)
    }

    /// Panorama column of each tile's column 0, before wrapping.
    pub fn tile_origins(&self) -> [i64; 2] {
        self.origins
    }

    /// A tile pixel's column in the panorama, wrapped into `[0, W)`.
    pub fn panorama_x(&self, tile: usize, x: f32) -> f32 {
        (x + self.origins[tile] as f32).rem_euclid(self.width as f32)
    }

    /// Detection cells for both tiles: the margin columns (the other
    /// hemisphere's half of the seam) and every cell row whose centre lies above
    /// [`TRACK_MAX_LATITUDE_DEG`]. Whole grid-aligned cells.
    ///
    /// # Errors
    ///
    /// When the cell size does not fit the tile.
    pub fn detection_masks(
        &self,
        config: &VioConfig,
        fy: f64,
        cy: f64,
    ) -> Result<Vec<CellMasks>, VioError> {
        let cell: usize = config.optical_flow_detection_grid_size as usize;
        let grid: CellGrid = CellGrid::new(self.tile_width, self.height, cell)
            .map_err(|error| VioError::Panorama(error.to_string()))?;
        let margin: usize = TILE_MARGIN_PX;
        let limit: f64 = TRACK_MAX_LATITUDE_DEG.to_radians();
        let mut masks: CellMasks = CellMasks::default();
        for (column, row) in grid.cells() {
            let x: usize = grid.x_start + column * cell;
            let y: usize = grid.y_start + row * cell;
            let in_margin: bool = x < margin || x + cell > self.tile_width - margin;
            let centre_latitude: f64 = (y as f64 + (cell as f64 - 1.0) / 2.0 - cy) / fy;
            if in_margin || centre_latitude.abs() > limit {
                masks.masks.push(MaskRect {
                    x: x as f32,
                    y: y as f32,
                    w: cell as f32,
                    h: cell as f32,
                });
            }
        }
        let Some(dual) = &self.dual else {
            return Ok(vec![masks.clone(), masks]);
        };
        // Dual fisheye: also every cell holding a pixel the lens does not cover.
        Ok((0..2)
            .map(|tile| {
                let invalid: Vec<bool> = dual.invalid(tile);
                let mut tile_masks: CellMasks = masks.clone();
                for (column, row) in grid.cells() {
                    let (x, y) = (grid.x_start + column * cell, grid.y_start + row * cell);
                    let rect = MaskRect {
                        x: x as f32,
                        y: y as f32,
                        w: cell as f32,
                        h: cell as f32,
                    };
                    let holds_invalid: bool = (y..y + cell).any(|r| {
                        invalid[r * self.tile_width + x..r * self.tile_width + x + cell]
                            .contains(&true)
                    });
                    if holds_invalid && !tile_masks.masks.contains(&rect) {
                        tile_masks.masks.push(rect);
                    }
                }
                tile_masks
            })
            .collect())
    }

    /// Track kill masks for both tiles: every row whose latitude
    /// `(v - cy) / fy` lies above `limit_rad`, plus, for stitched tiles, the
    /// pixels whose patch would read outside the lens.
    pub fn track_kill_masks(&self, fy: f64, cy: f64, limit_rad: f64) -> Vec<Arc<TrackKillMask>> {
        (0..2)
            .map(|tile| {
                let mut mask: TrackKillMask = match &self.dual {
                    Some(dual) => Arc::unwrap_or_clone(dual.kill_mask(tile)),
                    None => TrackKillMask {
                        width: self.tile_width,
                        height: self.height,
                        killed: vec![false; self.tile_width * self.height],
                    },
                };
                for (row, killed) in mask.killed.chunks_exact_mut(self.tile_width).enumerate() {
                    if ((row as f64 - cy) / fy).abs() > limit_rad {
                        killed.fill(true);
                    }
                }
                Arc::new(mask)
            })
            .collect()
    }

    /// Cut the panorama into the two tiles; a pure copy with horizontal wrap.
    ///
    /// # Errors
    ///
    /// When the frameset is not exactly one image of the panorama's size.
    pub fn crop(&mut self, images: &[ImageView<'_>]) -> Result<(), VioError> {
        self.check(images)?;
        let image: &ImageView<'_> = &images[0];
        for (tile, origin) in self.tiles.iter_mut().zip(self.origins) {
            let start: usize = origin.rem_euclid(self.width as i64) as usize;
            let first: usize = (self.width - start).min(self.tile_width);
            for (row, out) in tile.chunks_exact_mut(self.tile_width).enumerate() {
                let source: &[u8] =
                    &image.data[row * image.stride..row * image.stride + self.width];
                out[..first].copy_from_slice(&source[start..start + first]);
                out[first..].copy_from_slice(&source[..self.tile_width - first]);
            }
        }
        Ok(())
    }

    /// The tiles [`Self::crop`] last wrote, as the frontend's two images.
    pub fn tile_views(&self) -> [ImageView<'_>; 2] {
        self.tiles.each_ref().map(|data| ImageView {
            width: self.tile_width,
            height: self.height,
            stride: self.tile_width,
            data,
        })
    }

    /// The frameset rules for the panorama input: one image, panorama-sized,
    /// with its rows in bounds (equirect), or the two lens images (dual fisheye).
    ///
    /// # Errors
    ///
    /// The same errors the per-camera checks give.
    pub fn check(&self, images: &[ImageView<'_>]) -> Result<(), VioError> {
        if let Some(dual) = &self.dual {
            super::check_frameset(images, 2)?;
            let size = dual.lens_size();
            for (camera, image) in images.iter().enumerate() {
                if (image.width, image.height) != (size.width, size.height) {
                    return Err(crate::frontend::flow::FrontendError::FrameSizeMismatch {
                        camera,
                        expected_width: size.width,
                        expected_height: size.height,
                        actual_width: image.width,
                        actual_height: image.height,
                    }
                    .into());
                }
            }
            return Ok(());
        }
        super::check_frameset(images, 1)?;
        let image: &ImageView<'_> = &images[0];
        if (image.width, image.height) != (self.width, self.height) {
            return Err(crate::frontend::flow::FrontendError::FrameSizeMismatch {
                camera: 0,
                expected_width: self.width,
                expected_height: self.height,
                actual_width: image.width,
                actual_height: image.height,
            }
            .into());
        }
        Ok(())
    }
}

impl PanoramaRig {
    /// Turn a checked frameset into the frontend's two tile images in `frames`.
    ///
    /// Equirect: the crop, widened by the normal fill. Dual fisheye: the two lens
    /// images go to the storage with the rig's one [`crate::frontend::input::LensStitch`]
    /// (`fill_lenses`): dense (CPU) storage remaps on the host, packed (GPU)
    /// storage keeps the lens bytes and the pyramid builder stitches on the device.
    ///
    /// # Errors
    ///
    /// When the frameset is refused or an image operation fails.
    pub(super) fn ingest(
        &mut self,
        images: &[ImageView<'_>],
        frames: &mut FrameStorage,
    ) -> Result<(), VioError> {
        self.check(images)?;
        let Some(dual) = &mut self.dual else {
            self.crop(images)?;
            frames.fill(&self.tile_views())?;
            return Ok(());
        };
        let mark: std::time::Instant = std::time::Instant::now();
        frames.fill_lenses(images, dual.stitch())?;
        dual.set_last_ns(mark.elapsed().as_nanos() as u64);
        Ok(())
    }

    /// The same for the GPU lane's lookahead upload.
    ///
    /// # Errors
    ///
    /// When the frameset is refused or an image operation fails.
    pub(super) fn ingest_packed(
        &mut self,
        images: &[ImageView<'_>],
        packed: &mut PackedImages,
    ) -> Result<(), VioError> {
        self.check(images)?;
        match &self.dual {
            Some(dual) => packed.fill_lenses(images, dual.stitch())?,
            None => {
                self.crop(images)?;
                packed.fill(&self.tile_views());
            }
        }
        Ok(())
    }

    /// The stitched panorama (tile centres side by side, `u8`) of the frameset
    /// `frames` holds, for logging; `None` for an equirect rig. Only a caller
    /// that asks pays for it: on the GPU lane the tiles are remapped on the host
    /// from the retained lens bytes (the same integer rule as the device), never
    /// read back from the device.
    ///
    /// # Errors
    ///
    /// When the storage does not hold this rig's two tiles.
    pub fn panorama_image(
        &self,
        frames: &FrameStorage,
    ) -> Result<Option<kornia_image::Image<u8, 1>>, VioError> {
        let Some(dual) = &self.dual else {
            return Ok(None);
        };
        let view = frames.view();
        if view.len() != 2 {
            return Ok(None);
        }
        let geometry = dual.lut(0).geometry;
        let to_vio = |error: kornia_image::ImageError| VioError::Panorama(error.to_string());
        let panorama = view.get(0).with_dense(|tile0| {
            view.get(1).with_dense(|tile1| {
                kornia_staging_imgproc::panorama::assemble_panorama([tile0, tile1], geometry)
            })
        })??;
        Ok(Some(panorama.map_err(to_vio)?))
    }

    /// Keep a raw gyro sample (IMU frame) for the rolling-shutter correction.
    pub(super) fn push_gyro(&mut self, t_ns: i64, gyro: [f64; 3]) {
        if let Some(dual) = &mut self.dual {
            dual.push_gyro(t_ns, gyro);
        }
    }

    /// Remove the rolling shutter from the estimator's copy of the keypoints
    /// (dual fisheye with `config.panorama_rolling_shutter`); a no-op otherwise.
    pub(super) fn correct_rolling_shutter(
        &self,
        observations: &mut FlowObservations,
        gyro_bias: Option<Vector3<f64>>,
    ) {
        if let Some(dual) = &self.dual {
            dual.correct_rolling_shutter(observations, gyro_bias);
        }
    }

    /// Wall time of the last dual-fisheye host ingest, ns: the CPU remap of
    /// both lenses (CPU lane) or the lens byte copy (GPU lane, which stitches on
    /// the device inside the frontend); `None` for equirect.
    pub fn dual_fisheye_ingest_ns(&self) -> Option<u64> {
        self.dual.as_deref().map(DualFisheyeIngest::last_ns)
    }

    /// The lookup table of tile `k` (dual fisheye), `None` for equirect.
    pub fn tile_lut(&self, tile: usize) -> Option<&kornia_staging_imgproc::panorama::TileLut> {
        self.dual.as_deref().map(|dual| dual.lut(tile))
    }
}

/// The two tiles' detection cells and track kill masks on a freshly built frontend.
///
/// # Errors
///
/// When the frontend refuses them.
pub(super) fn configure(
    panorama: &PanoramaRig,
    config: &VioConfig,
    calibration: &Calibration<f64>,
    frontend: &mut super::FrontendLane,
) -> Result<(), VioError> {
    let BasaltCamera::Equirectangular(tile) = calibration.intrinsics[0] else {
        return Err(VioError::Panorama(
            "tile cameras must be equirectangular".to_owned(),
        ));
    };
    frontend.set_detection_masks(panorama.detection_masks(config, tile.fy, tile.cy)?)?;
    let limit: f64 = TRACK_MAX_LATITUDE_DEG.to_radians();
    frontend.set_track_kill_masks(
        panorama
            .track_kill_masks(tile.fy, tile.cy, limit)
            .into_iter()
            .map(Some)
            .collect(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
