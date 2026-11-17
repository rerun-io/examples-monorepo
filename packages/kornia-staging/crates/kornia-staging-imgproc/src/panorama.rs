//! Dual-fisheye lens images to the two equirectangular hemisphere tiles of a 360 panorama, in fixed point.
//!
//! Panorama frame `P`: `+Z` = the lens-0 axis, `+Y` = down (toward the handle on a 360 body), `+X` right. Tile `k`
//! (lens `k`) is an equirectangular image centred at longitude `k pi`: pixel `(tu, tv)` has
//! `lon = (tu - cx) / F + k pi`, `lat = (cy - tv) / F` (`lat > 0` up) and bearing
//! `(cos lat sin lon, -sin lat, cos lat cos lon)` in `P`, with `F = hemisphere_width / pi` px per radian,
//! `cx = (width - 1) / 2`, `cy = (height - 1) / 2`. Its camera frame is `T_k = Ry(k pi)` applied to `P`. A tile holds the
//! hemisphere's `hemisphere_width` columns plus `margin` columns per side, all filled from the same lens.
//!
//! The table is built once on the CPU in f64 and then holds integers only, so a device kernel that follows
//! [`remap_tile`]'s integer rule reproduces it bit for bit from the same table.

use crate::color::widen_u8_shift8_strided_unchecked;
use crate::pyramid::pyrdown_floor_u16;
use kornia_image::{Image, ImageError, ImageSize};

/// Table entry of a tile pixel with no source sample (outside the lens picture or beyond the lens coverage).
pub const LUT_INVALID: [u32; 2] = [u32::MAX; 2];

/// Size and resolution of a hemisphere tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGeometry {
    /// Columns that span the hemisphere's 180 deg of longitude (960 for a 1920x960 panorama).
    pub hemisphere_width: usize,
    /// Rows; `height / F` radians of latitude (960 = pole to pole).
    pub height: usize,
    /// Extra same-lens columns on each side of the hemisphere.
    pub margin: usize,
}

impl Default for TileGeometry {
    /// 960 + 2 x 64 = 1088 by 960.
    fn default() -> Self {
        Self {
            hemisphere_width: 960,
            height: 960,
            margin: 64,
        }
    }
}

impl TileGeometry {
    /// Tile width, `hemisphere_width + 2 margin`.
    pub fn width(&self) -> usize {
        self.hemisphere_width + 2 * self.margin
    }

    /// `F = fx = fy`, pixels per radian.
    pub fn focal(&self) -> f64 {
        self.hemisphere_width as f64 / std::f64::consts::PI
    }

    /// `(cx, cy) = ((width - 1) / 2, (height - 1) / 2)`, pixel centres at integers.
    pub fn principal(&self) -> [f64; 2] {
        [
            (self.width() as f64 - 1.0) / 2.0,
            (self.height as f64 - 1.0) / 2.0,
        ]
    }

    /// Longitude from the tile centre to the tile's outer pixel edge, `width / (2 F)` radians.
    pub fn lon_max(&self) -> f64 {
        self.width() as f64 / (2.0 * self.focal())
    }

    /// Unit bearing in `P` of tile `tile`'s pixel `(tu, tv)`.
    pub fn bearing(&self, tile: usize, tu: f64, tv: f64) -> [f64; 3] {
        let [cx, cy] = self.principal();
        let lon = (tu - cx) / self.focal() + tile as f64 * std::f64::consts::PI;
        let lat = (cy - tv) / self.focal();
        let (sin_lon, cos_lon) = lon.sin_cos();
        let (sin_lat, cos_lat) = lat.sin_cos();
        [cos_lat * sin_lon, -sin_lat, cos_lat * cos_lon]
    }
}

/// Fixed-point lookup table from one tile into its lens's level-1 (2x reduced) image.
#[derive(Clone, Debug, PartialEq)]
pub struct TileLut {
    /// Tile size.
    pub geometry: TileGeometry,
    /// Tile index, 0 or 1 (the lens it reads).
    pub tile: usize,
    /// Size of the level-1 source image, the floor-halved lens image.
    pub source_size: ImageSize,
    /// Per tile pixel, row-major: the level-1 source position `(x, y)` in 1/256 px,
    /// `q = floor(128 u + 0.5)` for the full-resolution lens coordinate `u` (level-1 pixel `j` is lens pixel `2j`, the
    /// centre of [`pyrdown_floor_u16`]'s filter), or [`LUT_INVALID`].
    pub sample: Vec<[u32; 2]>,
    /// Per tile pixel: the full-resolution lens row `v` (the rolling-shutter row); NaN where the ray does not project.
    pub source_row: Vec<f32>,
}

impl TileLut {
    /// Number of invalid tile pixels.
    pub fn invalid_count(&self) -> usize {
        self.sample.iter().filter(|&&s| s == LUT_INVALID).count()
    }
}

/// Build tile `tile`'s table.
///
/// # Arguments
/// * `geometry` - Tile size.
/// * `tile` - 0 or 1: the tile centre is at longitude `tile * pi`.
/// * `lens_size` - The full-resolution lens image (e.g. 1920x1920); the source is its floor-halved level 1.
/// * `project` - Bearing in `P` to a full-resolution lens pixel (pixel centres at integers), `None` beyond the lens
///   coverage; for an Insta360 lens, `LensCalibration::project_from_panorama`.
///
/// A pixel is valid when its fixed-point position lies in `[0, 256 (w1 - 1)] x [0, 256 (h1 - 1)]` of the level-1
/// image: every sample it reads is a real pixel.
pub fn build_tile_lut(
    geometry: TileGeometry,
    tile: usize,
    lens_size: ImageSize,
    project: impl Fn([f64; 3]) -> Option<[f64; 2]>,
) -> TileLut {
    let source_size = ImageSize {
        width: lens_size.width / 2,
        height: lens_size.height / 2,
    };
    let limit = [
        256.0 * (source_size.width as f64 - 1.0),
        256.0 * (source_size.height as f64 - 1.0),
    ];
    let pixels = geometry.width() * geometry.height;
    let mut sample = Vec::with_capacity(pixels);
    let mut source_row = Vec::with_capacity(pixels);
    for tv in 0..geometry.height {
        for tu in 0..geometry.width() {
            let Some(uv) = project(geometry.bearing(tile, tu as f64, tv as f64)) else {
                sample.push(LUT_INVALID);
                source_row.push(f32::NAN);
                continue;
            };
            source_row.push(uv[1] as f32);
            let q = uv.map(|c| (128.0 * c + 0.5).floor());
            let inside = (0..2).all(|i| q[i] >= 0.0 && q[i] <= limit[i]);
            sample.push(if inside {
                q.map(|c| c as u32)
            } else {
                LUT_INVALID
            });
        }
    }
    TileLut {
        geometry,
        tile,
        source_size,
        sample,
        source_row,
    }
}

/// The lens image as the pyramid ingests it, then reduced once: `byte << 8`
/// ([`widen_u8_shift8_strided_unchecked`]), then [`pyrdown_floor_u16`] (`[1 4 6 4 1]`, reflect-101, `(sum + 128) >> 8`).
///
/// # Errors
/// When the lens image is smaller than 3x3.
pub fn lens_level1(lens: &Image<u8, 1>) -> Result<Image<u16, 1>, ImageError> {
    let mut level0 = Image::from_size_val(lens.size(), 0u16)?;
    let mut level1 = Image::from_size_val(
        ImageSize {
            width: lens.width() / 2,
            height: lens.height() / 2,
        },
        0u16,
    )?;
    lens_level1_into(lens.as_slice(), lens.width(), &mut level0, &mut level1)?;
    Ok(level1)
}

/// [`lens_level1`] into caller-owned images, without allocating: `level0` takes the lens size, `level1` its floor half.
///
/// # Arguments
/// * `lens` - Lens bytes, `stride` per row, covering `stride * level0.height()` (checked by the caller, as for
///   [`widen_u8_shift8_strided_unchecked`]).
/// * `stride` - Row pitch in bytes, at least `level0.width()`.
///
/// # Errors
/// When `level1` is not `level0`'s floor half or `level0` is smaller than 3x3.
pub fn lens_level1_into(
    lens: &[u8],
    stride: usize,
    level0: &mut Image<u16, 1>,
    level1: &mut Image<u16, 1>,
) -> Result<(), ImageError> {
    widen_u8_shift8_strided_unchecked(lens, stride, level0);
    pyrdown_floor_u16(level0, level1)
}

/// Gather tile level 0 from a lens's level-1 image through `lut`.
///
/// Integer rule, per tile pixel with entry `(qx, qy)`: `x0 = qx >> 8`, `ax = qx & 255`, `x1 = min(x0 + 1, w1 - 1)`, the same
/// for `y`; `top = p(x0, y0) (256 - ax) + p(x1, y0) ax`, `bottom` likewise on row `y1`; the result is
/// `(top (256 - ay) + bottom ay + 32768) >> 16`, all in u32 (the largest sum, `65535 * 65536 + 32768`, fits).
/// An invalid entry gives 0.
///
/// # Errors
/// When `level1` is not `lut.source_size` or `dst` is not the tile size.
pub fn remap_tile(
    level1: &Image<u16, 1>,
    lut: &TileLut,
    dst: &mut Image<u16, 1>,
) -> Result<(), ImageError> {
    let (w, h) = (lut.source_size.width, lut.source_size.height);
    if level1.width() != w || level1.height() != h {
        return Err(ImageError::InvalidImageSize(
            level1.width(),
            level1.height(),
            w,
            h,
        ));
    }
    let tile = &lut.geometry;
    if dst.width() != tile.width() || dst.height() != tile.height {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            tile.width(),
            tile.height,
        ));
    }
    let src = level1.as_slice();
    for (out, &[qx, qy]) in dst.as_slice_mut().iter_mut().zip(&lut.sample) {
        if qx == u32::MAX {
            *out = 0;
            continue;
        }
        let (x0, y0, ax, ay) = ((qx >> 8) as usize, (qy >> 8) as usize, qx & 255, qy & 255);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let p = |x: usize, y: usize| u32::from(src[y * w + x]);
        let top = p(x0, y0) * (256 - ax) + p(x1, y0) * ax;
        let bottom = p(x0, y1) * (256 - ax) + p(x1, y1) * ax;
        *out = ((top * (256 - ay) + bottom * ay + 32768) >> 16) as u16;
    }
    Ok(())
}

/// The viewing panorama, `2 hemisphere_width x height` u8: tile 0's centre columns `margin .. margin + hemisphere_width`
/// then tile 1's, each value `(v + 128) >> 8` (at most 255).
///
/// # Errors
/// When a tile is not the geometry's size.
pub fn assemble_panorama(
    tiles: [&Image<u16, 1>; 2],
    geometry: TileGeometry,
) -> Result<Image<u8, 1>, ImageError> {
    let (hw, h, m) = (geometry.hemisphere_width, geometry.height, geometry.margin);
    for t in tiles {
        if t.width() != geometry.width() || t.height() != h {
            return Err(ImageError::InvalidImageSize(
                t.width(),
                t.height(),
                geometry.width(),
                h,
            ));
        }
    }
    let mut pano = Image::from_size_val(
        ImageSize {
            width: 2 * hw,
            height: h,
        },
        0u8,
    )?;
    for (row, out) in pano.as_slice_mut().chunks_exact_mut(2 * hw).enumerate() {
        for (k, t) in tiles.iter().enumerate() {
            let src = &t.as_slice()[row * geometry.width() + m..][..hw];
            for (o, &v) in out[k * hw..][..hw].iter_mut().zip(src) {
                *o = ((u32::from(v) + 128) >> 8).min(255) as u8;
            }
        }
    }
    Ok(pano)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Synthetic lens `k`: equidistant fisheye `r = f theta` up to 110 deg, looking along `Ry(k pi + yaw_k)` of `P`, 600x600.
    struct Lens {
        k: usize,
        yaw: f64,
        f: f64,
        c: f64,
    }

    impl Lens {
        /// `P` vector into the lens frame: `Ry(-(k pi + yaw))`.
        fn to_lens(&self, d: [f64; 3]) -> [f64; 3] {
            let (s, c) = (-(self.k as f64 * PI + self.yaw)).sin_cos();
            [c * d[0] + s * d[2], d[1], -s * d[0] + c * d[2]]
        }

        fn project(&self, d_p: [f64; 3]) -> Option<[f64; 2]> {
            let [x, y, z] = self.to_lens(d_p);
            let rho = (x * x + y * y).sqrt();
            let theta = rho.atan2(z);
            if theta > 110f64.to_radians() {
                return None;
            }
            let s = if rho > 0.0 {
                self.f * theta / rho
            } else {
                self.f
            };
            Some([self.c + s * x, self.c + s * y])
        }

        /// The `P` bearing of lens pixel `(u, v)`.
        fn unproject(&self, u: f64, v: f64) -> [f64; 3] {
            let (x, y) = ((u - self.c) / self.f, (v - self.c) / self.f);
            let theta = (x * x + y * y).sqrt();
            let s = if theta > 0.0 {
                theta.sin() / theta
            } else {
                1.0
            };
            let l = [x * s, y * s, theta.cos()];
            let (sn, cs) = (self.k as f64 * PI + self.yaw).sin_cos();
            [cs * l[0] + sn * l[2], l[1], -sn * l[0] + cs * l[2]]
        }
    }

    const SMALL: TileGeometry = TileGeometry {
        hemisphere_width: 240,
        height: 240,
        margin: 32,
    };
    const LENS: ImageSize = ImageSize {
        width: 600,
        height: 600,
    };

    fn lens(k: usize) -> Lens {
        Lens {
            k,
            yaw: [0.02, -0.03][k],
            f: 140.0,
            c: 299.5,
        }
    }

    /// Gaussian spots of 4 deg at the given P bearings, drawn into lens `k` as u8.
    fn render(l: &Lens, spots: &[[f64; 3]]) -> Image<u8, 1> {
        let sigma = 4f64.to_radians();
        let data = (0..LENS.width * LENS.height)
            .map(|i| {
                let d = l.unproject((i % LENS.width) as f64, (i / LENS.width) as f64);
                let v: f64 = spots
                    .iter()
                    .map(|s| {
                        let a = (d[0] * s[0] + d[1] * s[1] + d[2] * s[2])
                            .clamp(-1.0, 1.0)
                            .acos();
                        (-0.5 * (a / sigma).powi(2)).exp()
                    })
                    .sum();
                (20.0 + 220.0 * v).round().min(255.0) as u8
            })
            .collect();
        Image::new(LENS, data).unwrap()
    }

    fn tile_of(k: usize, img: &Image<u8, 1>, lut: &TileLut) -> Image<u16, 1> {
        let mut out = Image::from_size_val(
            ImageSize {
                width: SMALL.width(),
                height: SMALL.height,
            },
            0u16,
        )
        .unwrap();
        assert_eq!(lut.tile, k);
        remap_tile(&lens_level1(img).unwrap(), lut, &mut out).unwrap();
        out
    }

    #[test]
    fn geometry_numbers_follow_the_design() {
        let g = TileGeometry::default();
        assert_eq!((g.width(), g.height), (1088, 960));
        assert_eq!(g.principal(), [543.5, 479.5]);
        assert!((g.focal() - 960.0 / PI).abs() < 1e-12);
        assert!((g.lon_max() - 544.0 * PI / 960.0).abs() < 1e-12);
        // Tile 0's centre looks along +Z, tile 1's along -Z; the top row is near the north pole (-Y).
        let b0 = g.bearing(0, 543.5, 479.5);
        let b1 = g.bearing(1, 543.5, 479.5);
        assert!((b0[2] - 1.0).abs() < 1e-12 && (b1[2] + 1.0).abs() < 1e-12);
        assert!(g.bearing(0, 543.5, 0.0)[1] < -0.99);
    }

    #[test]
    fn spots_land_at_their_predicted_tile_pixels() {
        // Spots on the equator (a symmetric footprint), at least 30 deg apart: in each hemisphere at the centre, 50-60 deg
        // off and 7 deg into the margin.
        let lons = [0.0f64, 50.0, -60.0, 97.0, 180.0, 240.0, 125.0, 277.0];
        let spots: Vec<[f64; 3]> = lons
            .iter()
            .map(|l| {
                let l = l.to_radians();
                [l.sin(), 0.0, l.cos()]
            })
            .collect();
        let f = SMALL.focal();
        let [cx, cy] = SMALL.principal();
        for k in 0..2 {
            let l = lens(k);
            let lut = build_tile_lut(SMALL, k, LENS, |d| l.project(d));
            let tile = tile_of(k, &render(&l, &spots), &lut);
            for lon in lons {
                let mut dl = lon.to_radians() - k as f64 * PI;
                dl = (dl + PI).rem_euclid(2.0 * PI) - PI;
                let (pu, pv) = (cx + dl * f, cy);
                if dl.abs() * f > cx - 14.0 {
                    continue;
                }
                let (mut sw, mut su, mut sv) = (0.0, 0.0, 0.0);
                for tv in (pv as i64 - 12)..=(pv as i64 + 13) {
                    for tu in (pu as i64 - 12)..=(pu as i64 + 13) {
                        let w =
                            f64::from(tile.as_slice()[tv as usize * SMALL.width() + tu as usize])
                                - f64::from(20u16 << 8);
                        let w = w.max(0.0);
                        (sw, su, sv) = (sw + w, su + w * tu as f64, sv + w * tv as f64);
                    }
                }
                let (eu, ev) = (su / sw - pu, sv / sw - pv);
                assert!(
                    eu.abs() < 0.1 && ev.abs() < 0.1,
                    "tile {k} lon {lon}: centroid off by ({eu:.3}, {ev:.3}) px"
                );
            }
        }
    }

    #[test]
    fn the_integer_rule_and_invalid_pixels() {
        // A level-1 image with p(x, y) = 1000 x + 7 y; the lens maps every tile pixel to a fixed full-res point.
        let src = Image::new(
            ImageSize {
                width: 8,
                height: 6,
            },
            (0..48)
                .map(|i| (1000 * (i % 8) + 7 * (i / 8)) as u16)
                .collect(),
        )
        .unwrap();
        let geometry = TileGeometry {
            hemisphere_width: 4,
            height: 2,
            margin: 1,
        };
        let size = ImageSize {
            width: 16,
            height: 12,
        };
        // Full-res (5.3, 4.1) -> q = floor(128 (5.3, 4.1) + 0.5) = (678, 525). Rays with d_x >= 0.5 do not project.
        let lut = build_tile_lut(geometry, 0, size, |d| (d[0] < 0.5).then_some([5.3, 4.1]));
        assert_eq!(lut.sample[0], [678, 525]);
        assert!((lut.source_row[0] - 4.1).abs() < 1e-6);
        let mut dst = Image::from_size_val(
            ImageSize {
                width: 6,
                height: 2,
            },
            9u16,
        )
        .unwrap();
        remap_tile(&src, &lut, &mut dst).unwrap();
        // x0 = 2, ax = 166, y0 = 2, ay = 13: top = 2014 * 90 + 3014 * 166, bottom = 2021 * 90 + 3021 * 166.
        let (top, bottom) = (2014 * 90 + 3014 * 166u32, 2021 * 90 + 3021 * 166u32);
        let expected = ((top * 243 + bottom * 13 + 32768) >> 16) as u16;
        assert_eq!(dst.as_slice()[0], expected);
        let invalid: Vec<usize> = (0..12).filter(|&i| lut.sample[i] == LUT_INVALID).collect();
        assert_eq!(lut.invalid_count(), invalid.len());
        assert!(
            !invalid.is_empty()
                && invalid
                    .iter()
                    .all(|&i| dst.as_slice()[i] == 0 && lut.source_row[i].is_nan())
        );
        // Outside the picture is invalid too; the last level-1 sample position (w1 - 1) is valid.
        let edge = build_tile_lut(geometry, 0, size, |_| Some([14.0, 10.0]));
        assert_eq!(edge.sample[0], [7 * 256, 5 * 256]);
        let past = build_tile_lut(geometry, 0, size, |_| Some([14.01, -0.01]));
        assert_eq!(past.invalid_count(), 12);
        remap_tile(&src, &edge, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[0], 7035);
    }

    #[test]
    fn the_same_input_gives_the_same_bytes() {
        let l = lens(1);
        let spots = [[0.3f64, 0.1, -0.95]];
        let img = render(&l, &spots);
        let a = build_tile_lut(SMALL, 1, LENS, |d| l.project(d));
        let b = build_tile_lut(SMALL, 1, LENS, |d| l.project(d));
        assert_eq!(a.sample, b.sample);
        assert_eq!(
            tile_of(1, &img, &a).as_slice(),
            tile_of(1, &img, &b).as_slice()
        );
    }

    #[test]
    fn the_panorama_is_the_two_tile_centres() {
        let size = ImageSize {
            width: SMALL.width(),
            height: SMALL.height,
        };
        let ramp = (0..size.width * size.height).map(|i| ((i % size.width).min(255) as u16) << 8);
        let t0 = Image::new(size, ramp.collect()).unwrap();
        let t1 = Image::from_size_val(size, 65280u16).unwrap();
        let pano = assemble_panorama([&t0, &t1], SMALL).unwrap();
        assert_eq!((pano.width(), pano.height()), (480, 240));
        let row = &pano.as_slice()[..480];
        assert_eq!((row[0], row[239], row[240], row[479]), (32, 255, 255, 255));
        assert_eq!(row[100], 132);
        assert!(assemble_panorama(
            [&t0, &t0],
            TileGeometry {
                margin: 31,
                ..SMALL
            }
        )
        .is_err());
    }
}
