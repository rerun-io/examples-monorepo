//! The panorama lane: one 360-degree source in, two hemisphere tiles out.
//!
//! The frontend and the estimator keep seeing "N cameras, N images". This module
//! is the one place where a rig whose calibration holds a full panorama becomes
//! two hemisphere-tile cameras, and each frameset's panorama becomes two tile
//! images (DESIGN §4.1, §13 M1). Tile `k` is centred on panorama longitude
//! `k * pi`, is `W / 2 + 2 * margin` pixels wide, and its camera is the panorama
//! camera turned by `k * 180` degrees about its own vertical axis.
//!
//! The tile cut answers two kernel limits at once: KLT, the pyramid and FAST
//! never wrap, and a tile never crosses a lens seam. A track that leaves its tile
//! ends; the other tile detects the point again.

use super::{ImageView, VioError};
use crate::calib::{BasaltCamera, Calibration, EquirectangularParams};
use crate::config::VioConfig;
use crate::lie::{Se3, So3};
use kornia_staging_imgproc::features::{CellGrid, CellMasks, MaskRect};
use nalgebra::Vector3;

/// Where the two tiles' pixels come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanoramaSource {
    /// A central equirectangular panorama (TartanAir `lcam_equirect`): each tile
    /// is a crop with horizontal wrap, a pure pixel copy; both tile cameras share
    /// the panorama's centre.
    Equirect,
    /// Two fisheye lens images remapped through a prebuilt lookup table; the
    /// tile cameras sit at the two lens centres. Planned for the next phase:
    /// [`PanoramaRig::expand`] does not select it yet.
    DualFisheye,
}

/// One panorama camera expanded into two hemisphere tiles.
#[derive(Debug, Clone)]
pub struct PanoramaRig {
    source: PanoramaSource,
    /// Panorama size in pixels.
    width: usize,
    height: usize,
    /// Tile width in pixels; both tiles are `tile_width x height`.
    tile_width: usize,
    /// Panorama column of each tile's column 0 (may be negative: it wraps).
    origins: [i64; 2],
    /// The two tiles of the last cropped frameset.
    tiles: [Vec<u8>; 2],
}

impl PanoramaRig {
    /// Expand a calibration that holds one full equirectangular panorama
    /// (`lon_max = pi`) into the two tile cameras; any other rig is returned
    /// unchanged with `None`.
    ///
    /// # Errors
    ///
    /// [`VioError::Panorama`] when a full panorama shares the rig with other
    /// cameras, is not `fx = W / 2pi` with an even width, or when the margin
    /// leaves a tile wider than the panorama.
    pub fn expand(
        config: &VioConfig,
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
        let margin: usize = config.panorama_tile_margin_px as usize;
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
        let tile = EquirectangularParams {
            fx: pano.fx,
            fy: pano.fy,
            cx: pano.cx - origin as f64,
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
                source: PanoramaSource::Equirect,
                width,
                height,
                tile_width,
                origins: [origin, origin + (width / 2) as i64],
                tiles: [vec![0; tile_width * height], vec![0; tile_width * height]],
            }),
            tiles,
        ))
    }

    /// Where the tiles come from.
    pub fn source(&self) -> PanoramaSource {
        self.source
    }

    /// Panorama `(width, height)` the input frameset must have.
    pub fn panorama_size(&self) -> (usize, usize) {
        (self.width, self.height)
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
    /// `config.panorama_track_max_latitude_deg`. Whole grid-aligned cells.
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
        let margin: usize = config.panorama_tile_margin_px as usize;
        let limit: f64 = f64::from(config.panorama_track_max_latitude_deg).to_radians();
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
        Ok(vec![masks.clone(), masks])
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
    /// with its rows in bounds.
    ///
    /// # Errors
    ///
    /// The same errors the per-camera checks give.
    pub fn check(&self, images: &[ImageView<'_>]) -> Result<(), VioError> {
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

/// The two tiles' cells and latitude limits on a freshly built frontend.
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
    let limit: f32 = config.panorama_track_max_latitude_deg.to_radians();
    frontend.set_detection_masks(panorama.detection_masks(config, tile.fy, tile.cy)?)?;
    frontend.set_track_latitude_limits(vec![Some(limit); 2])?;
    Ok(())
}

#[cfg(test)]
mod tests;
