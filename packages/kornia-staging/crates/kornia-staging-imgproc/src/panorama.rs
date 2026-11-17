//! Fixed-point remap tables, and the dual-fisheye stitch of two lens images into the two equirectangular
//! hemisphere tiles of a 360 panorama.
//!
//! A [`RemapTable`] is built once on the CPU in f64 and then holds integers only, so a device kernel that follows
//! [`remap`]'s integer rule reproduces it bit for bit from the same table.
//!
//! Panorama frame `P`: `+Z` = the lens-0 axis, `+Y` = down (toward the handle on a 360 body), `+X` right. Tile `k`
//! (lens `k`) is the [`Equirectangular::hemisphere_tile`] camera of a [`TileGeometry`] (see [`TileGeometry::camera`]),
//! turned by `Ry(k pi)` in `P`. A tile holds the hemisphere's `hemisphere_width` columns plus `margin` columns per
//! side, all filled from the same lens.

use std::sync::Mutex;

use crate::color::widen_u8_shift8_strided_unchecked;
use crate::pyramid::pyrdown_floor_u16;
use kornia_image::{Image, ImageError, ImageSize};
use kornia_staging_3d::camera::{CameraModel, Equirectangular, InvalidCalibration};

/// A raw table that is not a valid [`RemapTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RemapTableError {
    /// The entry count is not one per destination pixel.
    #[error("{actual} entries for {expected} destination pixels")]
    Length {
        /// Destination pixels.
        expected: usize,
        /// Entries given.
        actual: usize,
    },
    /// An entry is neither [`RemapTable::INVALID`] nor inside the source image.
    #[error(
        "entry {pixel} is neither the invalid sentinel nor inside the {width}x{height} source"
    )]
    OutOfRange {
        /// Row-major destination pixel.
        pixel: usize,
        /// Source width.
        width: usize,
        /// Source height.
        height: usize,
    },
}

/// Fixed-point lookup table from a destination image into a source image, valid by construction: one entry per
/// destination pixel (row-major), each either [`RemapTable::INVALID`] or a source position `(qx, qy)` in 1/256 px
/// inside `[0, 256 (w - 1)] x [0, 256 (h - 1)]`, so every sample [`remap`] reads is a real source pixel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemapTable {
    dst: ImageSize,
    src: ImageSize,
    sample: Vec<[u32; 2]>,
}

impl RemapTable {
    /// Entry of a destination pixel with no source sample.
    pub const INVALID: [u32; 2] = [u32::MAX; 2];

    /// Build a table from a map of destination pixel `(x, y)` to source position (pixel centres at integers), `None`
    /// where it has none. The entry is `q = floor(256 p + 0.5)` per coordinate; a position outside
    /// `[0, w - 1] x [0, h - 1]` after rounding (or NaN) gives [`Self::INVALID`].
    pub fn build(
        dst: ImageSize,
        src: ImageSize,
        map: impl Fn(usize, usize) -> Option<[f64; 2]>,
    ) -> Self {
        let limit = [
            256.0 * (src.width as f64 - 1.0),
            256.0 * (src.height as f64 - 1.0),
        ];
        let mut sample = Vec::with_capacity(dst.width * dst.height);
        for y in 0..dst.height {
            for x in 0..dst.width {
                let entry = map(x, y).and_then(|position| {
                    let q = position.map(|c| (256.0 * c + 0.5).floor());
                    (0..2)
                        .all(|i| q[i] >= 0.0 && q[i] <= limit[i])
                        .then(|| q.map(|c| c as u32))
                });
                sample.push(entry.unwrap_or(Self::INVALID));
            }
        }
        Self { dst, src, sample }
    }

    /// A table from raw parts, checked once: one entry per destination pixel, each [`Self::INVALID`] (both words) or
    /// inside the source.
    ///
    /// # Errors
    /// [`RemapTableError::Length`] or [`RemapTableError::OutOfRange`].
    pub fn from_raw(
        dst: ImageSize,
        src: ImageSize,
        sample: Vec<[u32; 2]>,
    ) -> Result<Self, RemapTableError> {
        if sample.len() != dst.width * dst.height {
            return Err(RemapTableError::Length {
                expected: dst.width * dst.height,
                actual: sample.len(),
            });
        }
        let inside = |q: u32, n: usize| n > 0 && u64::from(q) <= 256 * (n as u64 - 1);
        if let Some(pixel) = sample.iter().position(|&q| {
            q != Self::INVALID && !(inside(q[0], src.width) && inside(q[1], src.height))
        }) {
            return Err(RemapTableError::OutOfRange {
                pixel,
                width: src.width,
                height: src.height,
            });
        }
        Ok(Self { dst, src, sample })
    }

    /// Destination size.
    pub fn dst_size(&self) -> ImageSize {
        self.dst
    }

    /// Source size.
    pub fn src_size(&self) -> ImageSize {
        self.src
    }

    /// The entries, row-major over the destination.
    pub fn samples(&self) -> &[[u32; 2]] {
        &self.sample
    }
}

/// Size of a hemisphere tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGeometry {
    /// Columns that span the hemisphere's 180 deg of longitude (960 for a 1920x960 panorama).
    pub hemisphere_width: usize,
    /// Rows; `height / F` radians of latitude (`height = hemisphere_width` is pole to pole).
    pub height: usize,
    /// Extra same-lens columns on each side of the hemisphere.
    pub margin: usize,
}

impl TileGeometry {
    /// Tile width, `hemisphere_width + 2 margin`.
    pub fn width(&self) -> usize {
        self.hemisphere_width + 2 * self.margin
    }

    /// Tile size, `width() x height`.
    pub fn size(&self) -> ImageSize {
        ImageSize {
            width: self.width(),
            height: self.height,
        }
    }

    /// The tile camera: [`Equirectangular::hemisphere_tile`] at `F = hemisphere_width / pi` px per radian, its
    /// longitude window reaching the tile's outer pixel edges (`width / (2 F)`).
    ///
    /// # Errors
    /// An empty hemisphere or a margin wider than half the hemisphere (the window would pass `pi`).
    pub fn camera(&self) -> Result<Equirectangular<f64>, InvalidCalibration> {
        let focal = self.hemisphere_width as f64 / std::f64::consts::PI;
        let (width, height) = (u32::try_from(self.width()), u32::try_from(self.height));
        let (Ok(width), Ok(height)) = (width, height) else {
            return Err(InvalidCalibration);
        };
        Equirectangular::hemisphere_tile(width, height, focal, f64::from(width) / (2.0 * focal))
    }
}

/// Tile `tile`'s table over its lens's level-1 (floor-halved) image: tile pixel -> tile camera bearing
/// ([`TileGeometry::camera`]) -> `Ry(tile pi)` into `P` -> `project` -> level-1 position (half the lens pixel; level-1
/// pixel `j` is lens pixel `2j`, the centre of [`pyrdown_floor_u16`]'s filter).
///
/// # Arguments
/// * `geometry` - Tile size.
/// * `tile` - 0 or 1: the tile looks along `+Z` or `-Z` of `P`.
/// * `lens_size` - The full-resolution lens image (e.g. 1920x1920).
/// * `project` - Bearing in `P` to a full-resolution lens pixel (pixel centres at integers), `None` beyond the lens
///   coverage.
///
/// # Errors
/// When [`TileGeometry::camera`] refuses the geometry.
pub fn hemisphere_tile_table(
    geometry: TileGeometry,
    tile: usize,
    lens_size: ImageSize,
    project: impl Fn([f64; 3]) -> Option<[f64; 2]>,
) -> Result<RemapTable, InvalidCalibration> {
    let camera = geometry.camera()?;
    let level1 = ImageSize {
        width: lens_size.width / 2,
        height: lens_size.height / 2,
    };
    Ok(RemapTable::build(geometry.size(), level1, |x, y| {
        let [bx, by, bz] = camera.unproject([x as f64, y as f64]).ok()?;
        let in_p = if tile % 2 == 0 {
            [bx, by, bz]
        } else {
            [-bx, by, -bz]
        };
        project(in_p).map(|uv| uv.map(|c| 0.5 * c))
    }))
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

/// Gather `dst` from `src` through `table`.
///
/// Integer rule, per destination pixel with entry `(qx, qy)`: `x0 = qx >> 8`, `ax = qx & 255`, `x1 = min(x0 + 1, w - 1)`,
/// the same for `y`; `top = p(x0, y0) (256 - ax) + p(x1, y0) ax`, `bottom` likewise on row `y1`; the result is
/// `(top (256 - ay) + bottom ay + 32768) >> 16`, all in u32 (the largest sum, `65535 * 65536 + 32768`, fits).
/// An invalid entry gives 0.
///
/// # Errors
/// When `src` is not the table's source size or `dst` not its destination size.
pub fn remap(
    src: &Image<u16, 1>,
    table: &RemapTable,
    dst: &mut Image<u16, 1>,
) -> Result<(), ImageError> {
    for (image, size) in [(src.size(), table.src), (dst.size(), table.dst)] {
        if image != size {
            return Err(ImageError::InvalidImageSize(
                image.width,
                image.height,
                size.width,
                size.height,
            ));
        }
    }
    let w = table.src.width;
    let (h, src) = (table.src.height, src.as_slice());
    for (out, &[qx, qy]) in dst.as_slice_mut().iter_mut().zip(&table.sample) {
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

/// A stitch the plan cannot run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StitchError {
    /// The lens image is below 4x4 (its level 1 must be at least 2x2).
    #[error("a {width}x{height} lens image is below 4x4")]
    LensTooSmall {
        /// Lens width.
        width: usize,
        /// Lens height.
        height: usize,
    },
    /// Table `tile` does not map the geometry's tile from the lens's level 1.
    #[error(
        "table {tile} maps {dst:?} from {src:?}; the stitch needs {tile_size:?} from {level1:?}"
    )]
    Tables {
        /// Tile index.
        tile: usize,
        /// The table's destination size.
        dst: ImageSize,
        /// The table's source size.
        src: ImageSize,
        /// The geometry's tile size.
        tile_size: ImageSize,
        /// The lens's level-1 size.
        level1: ImageSize,
    },
    /// Lens bytes that do not hold a lens image at the given stride.
    #[error("{len} bytes at stride {stride} do not hold a {width}x{height} lens image")]
    LensBuffer {
        /// Bytes given.
        len: usize,
        /// Row pitch given.
        stride: usize,
        /// Lens width.
        width: usize,
        /// Lens height.
        height: usize,
    },
    /// The tile geometry has no tile camera.
    #[error("the tile geometry {0:?} has no tile camera (empty, or a margin wider than half the hemisphere)")]
    Geometry(TileGeometry),
    /// An image of the plan's own scratch or output has the wrong size (an image operation refused it).
    #[error("an image of the stitch has the wrong size")]
    Image,
}

impl From<ImageError> for StitchError {
    fn from(_: ImageError) -> Self {
        Self::Image
    }
}

/// The reusable dual-fisheye stitch: two tile tables (tile `k` reads lens `k`), the lens size, and the host scratch.
///
/// Built once per rig; the host stitch of tile `k` is [`lens_level1_into`] then [`remap`] through table `k`. A device
/// stitch uploads [`Self::tables`] once and follows the same integer rule, so both give the same pixels.
pub struct LensStitch {
    geometry: TileGeometry,
    tables: [RemapTable; 2],
    lens: ImageSize,
    /// Host remap scratch, reused every frameset: one lens widened to `u16`, then its level 1.
    scratch: Mutex<[Image<u16, 1>; 2]>,
}

impl std::fmt::Debug for LensStitch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LensStitch")
            .field("geometry", &self.geometry)
            .field("lens", &self.lens)
            .finish_non_exhaustive()
    }
}

impl LensStitch {
    /// The stitch of `lens`-sized lens images into `geometry`'s two tiles: [`hemisphere_tile_table`] of each lens.
    ///
    /// # Arguments
    /// * `project` - Lens `k`'s map from a bearing in `P` to its full-resolution pixel, `project(k, bearing)`.
    ///
    /// # Errors
    /// [`StitchError::Geometry`] or [`StitchError::LensTooSmall`], before any table is built.
    pub fn new(
        geometry: TileGeometry,
        lens: ImageSize,
        project: impl Fn(usize, [f64; 3]) -> Option<[f64; 2]>,
    ) -> Result<Self, StitchError> {
        geometry
            .camera()
            .map_err(|_| StitchError::Geometry(geometry))?;
        check_lens(lens)?;
        let table = |k: usize| {
            hemisphere_tile_table(geometry, k, lens, |ray| project(k, ray))
                .map_err(|_| StitchError::Geometry(geometry))
        };
        Self::from_tables(geometry, [table(0)?, table(1)?], lens)
    }

    /// The stitch through given tables, checked once: both map `geometry`'s tile from `lens`'s level 1.
    ///
    /// # Errors
    /// [`StitchError::LensTooSmall`] or [`StitchError::Tables`].
    pub fn from_tables(
        geometry: TileGeometry,
        tables: [RemapTable; 2],
        lens: ImageSize,
    ) -> Result<Self, StitchError> {
        check_lens(lens)?;
        let level1 = ImageSize {
            width: lens.width / 2,
            height: lens.height / 2,
        };
        if let Some((tile, table)) = tables
            .iter()
            .enumerate()
            .find(|(_, table)| table.dst != geometry.size() || table.src != level1)
        {
            return Err(StitchError::Tables {
                tile,
                dst: table.dst,
                src: table.src,
                tile_size: geometry.size(),
                level1,
            });
        }
        let scratch = Mutex::new([
            Image::from_size_val(lens, 0u16)?,
            Image::from_size_val(level1, 0u16)?,
        ]);
        Ok(Self {
            geometry,
            tables,
            lens,
            scratch,
        })
    }

    /// The tile geometry.
    pub fn geometry(&self) -> TileGeometry {
        self.geometry
    }

    /// The two tables, tile 0 then tile 1.
    pub fn tables(&self) -> &[RemapTable; 2] {
        &self.tables
    }

    /// Each lens image's size.
    pub fn lens_size(&self) -> ImageSize {
        self.lens
    }

    /// Each tile's size.
    pub fn tile_size(&self) -> ImageSize {
        self.geometry.size()
    }

    /// Tile `tile` from lens `tile`'s bytes (`stride` bytes per row) on the host, through the reused scratch and
    /// without copying the lens first; `out` is resized to the tile when it is not.
    ///
    /// # Errors
    /// [`StitchError::LensBuffer`] when the bytes do not hold a lens image at `stride`.
    pub fn remap(
        &self,
        tile: usize,
        lens: &[u8],
        stride: usize,
        out: &mut Image<u16, 1>,
    ) -> Result<(), StitchError> {
        let (width, height) = (self.lens.width, self.lens.height);
        if stride < width || lens.len() < stride * (height - 1) + width {
            return Err(StitchError::LensBuffer {
                len: lens.len(),
                stride,
                width,
                height,
            });
        }
        if out.size() != self.tile_size() {
            *out = Image::from_size_val(self.tile_size(), 0u16)?;
        }
        let mut scratch = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let [level0, level1] = &mut *scratch;
        lens_level1_into(lens, stride, level0, level1)?;
        Ok(remap(level1, &self.tables[tile], out)?)
    }

    /// The viewing panorama of two lens images ([`Self::remap`] of each tile, then [`assemble_panorama`]).
    ///
    /// # Arguments
    /// * `lenses` - Lens 0's and lens 1's bytes, each `stride` bytes per row.
    ///
    /// # Errors
    /// As [`Self::remap`].
    pub fn panorama(&self, lenses: [&[u8]; 2], stride: usize) -> Result<Image<u8, 1>, StitchError> {
        let mut tiles = [
            Image::from_size_val(self.tile_size(), 0u16)?,
            Image::from_size_val(self.tile_size(), 0u16)?,
        ];
        for (tile, (out, lens)) in tiles.iter_mut().zip(lenses).enumerate() {
            self.remap(tile, lens, stride, out)?;
        }
        Ok(assemble_panorama([&tiles[0], &tiles[1]], self.geometry)?)
    }
}

fn check_lens(lens: ImageSize) -> Result<(), StitchError> {
    if lens.width < 4 || lens.height < 4 {
        return Err(StitchError::LensTooSmall {
            width: lens.width,
            height: lens.height,
        });
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

    fn tile_of(img: &Image<u8, 1>, table: &RemapTable) -> Image<u16, 1> {
        let mut out = Image::from_size_val(SMALL.size(), 0u16).unwrap();
        remap(&lens_level1(img).unwrap(), table, &mut out).unwrap();
        out
    }

    fn table(k: usize, l: &Lens) -> RemapTable {
        hemisphere_tile_table(SMALL, k, LENS, |d| l.project(d)).unwrap()
    }

    #[test]
    fn the_tile_camera_follows_the_design() {
        let g = TileGeometry {
            hemisphere_width: 960,
            height: 960,
            margin: 64,
        };
        assert_eq!(
            g.size(),
            ImageSize {
                width: 1088,
                height: 960
            }
        );
        let [fx, fy, cx, cy, lon_max] = g.camera().unwrap().params();
        assert!((fx - 960.0 / PI).abs() < 1e-12 && fx == fy);
        assert_eq!([cx, cy], [543.5, 479.5]);
        assert!((lon_max - 544.0 * PI / 960.0).abs() < 1e-12);
        // The window passes pi past half a hemisphere of margin.
        assert!(TileGeometry { margin: 480, ..g }.camera().is_ok());
        assert!(TileGeometry { margin: 481, ..g }.camera().is_err());
        // Tile 0's centre looks along +Z, tile 1's along -Z; the top row is near the north pole (-Y).
        let centre = |k: usize| {
            let t = hemisphere_tile_table(
                g,
                k,
                ImageSize {
                    width: 8,
                    height: 8,
                },
                |d| Some([d[2] + 3.0, d[1] + 3.0]),
            )
            .unwrap();
            t.samples()[479 * 1088 + 543]
        };
        // Pixel (543, 479) is half a pixel off the centre: d_z = 1 - O(1e-6), d_y = O(1e-3).
        assert_eq!(centre(0)[0], (256.0f64 * 2.0 + 0.5).floor() as u32);
        assert_eq!(centre(1)[0], (256.0f64 * 1.0 + 0.5).floor() as u32);
        let top = hemisphere_tile_table(
            g,
            0,
            ImageSize {
                width: 8,
                height: 8,
            },
            |d| Some([0.0, 2.0 * (d[1] + 1.0)]),
        )
        .unwrap();
        assert!(top.samples()[543][1] < 3);
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
        let [f, _, cx, cy, _] = SMALL.camera().unwrap().params();
        for k in 0..2 {
            let l = lens(k);
            let tile = tile_of(&render(&l, &spots), &table(k, &l));
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
        // An 8x6 source with p(x, y) = 1000 x + 7 y; the map sends every destination pixel left of x = 3 to one point.
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
        let dst_size = ImageSize {
            width: 6,
            height: 2,
        };
        // (2.65, 2.05) -> q = floor(256 (2.65, 2.05) + 0.5) = (678, 525).
        let table = RemapTable::build(dst_size, src.size(), |x, _| (x < 3).then_some([2.65, 2.05]));
        assert_eq!(table.samples()[0], [678, 525]);
        let mut dst = Image::from_size_val(dst_size, 9u16).unwrap();
        remap(&src, &table, &mut dst).unwrap();
        // x0 = 2, ax = 166, y0 = 2, ay = 13: top = 2014 * 90 + 3014 * 166, bottom = 2021 * 90 + 3021 * 166.
        let (top, bottom) = (2014 * 90 + 3014 * 166u32, 2021 * 90 + 3021 * 166u32);
        let expected = ((top * 243 + bottom * 13 + 32768) >> 16) as u16;
        assert_eq!(dst.as_slice()[0], expected);
        let invalid: Vec<usize> = (0..12)
            .filter(|&i| table.samples()[i] == RemapTable::INVALID)
            .collect();
        assert_eq!(invalid, [3, 4, 5, 9, 10, 11]);
        assert!(invalid.iter().all(|&i| dst.as_slice()[i] == 0));
        // Outside the source is invalid too; the last sample position (w - 1) is valid.
        let edge = RemapTable::build(dst_size, src.size(), |_, _| Some([7.0, 5.0]));
        assert_eq!(edge.samples()[0], [7 * 256, 5 * 256]);
        let past = RemapTable::build(dst_size, src.size(), |_, _| Some([7.005, -0.005]));
        assert!(past.samples().iter().all(|&q| q == RemapTable::INVALID));
        remap(&src, &edge, &mut dst).unwrap();
        assert_eq!(dst.as_slice()[0], 7035);
        // Images of the wrong size are refused.
        let mut wrong = Image::from_size_val(src.size(), 0u16).unwrap();
        assert!(remap(&src, &edge, &mut wrong).is_err());
    }

    #[test]
    fn raw_tables_are_checked_once() {
        let (dst, src) = (
            ImageSize {
                width: 3,
                height: 2,
            },
            ImageSize {
                width: 8,
                height: 6,
            },
        );
        let good = vec![
            [0, 0],
            [7 * 256, 5 * 256],
            RemapTable::INVALID,
            [100, 200],
            [1, 1],
            [256, 256],
        ];
        let table = RemapTable::from_raw(dst, src, good.clone()).unwrap();
        assert_eq!(
            table,
            RemapTable::build(dst, src, |x, y| {
                [
                    Some([0.0, 0.0]),
                    Some([7.0, 5.0]),
                    None,
                    Some([100.0 / 256.0, 200.0 / 256.0]),
                    Some([1.0 / 256.0, 1.0 / 256.0]),
                    Some([1.0, 1.0]),
                ][y * 3 + x]
            })
        );
        let with = |pixel: usize, entry: [u32; 2]| {
            let mut sample = good.clone();
            sample[pixel] = entry;
            RemapTable::from_raw(dst, src, sample)
        };
        // One past the last column or row, a half sentinel in either word.
        for (pixel, entry) in [
            (1, [7 * 256 + 1, 0]),
            (4, [0, 5 * 256 + 1]),
            (2, [u32::MAX, 0]),
            (5, [0, u32::MAX]),
        ] {
            assert_eq!(
                with(pixel, entry),
                Err(RemapTableError::OutOfRange {
                    pixel,
                    width: 8,
                    height: 6
                })
            );
        }
        let short = RemapTable::from_raw(dst, src, good[..5].to_vec());
        assert_eq!(
            short,
            Err(RemapTableError::Length {
                expected: 6,
                actual: 5
            })
        );
        let wrong_geometry = RemapTable::from_raw(
            ImageSize {
                width: 2,
                height: 2,
            },
            src,
            good.clone(),
        );
        assert_eq!(
            wrong_geometry,
            Err(RemapTableError::Length {
                expected: 4,
                actual: 6
            })
        );
        // An empty source holds no valid position.
        assert!(RemapTable::from_raw(
            dst,
            ImageSize {
                width: 0,
                height: 6
            },
            good
        )
        .is_err());
    }

    #[test]
    fn the_stitch_takes_only_tables_of_its_geometry_and_lens() {
        let tables = [0, 1].map(|k| table(k, &lens(k)));
        let wide = TileGeometry {
            margin: 33,
            ..SMALL
        };
        let refused = LensStitch::from_tables(wide, tables.clone(), LENS);
        assert!(
            matches!(refused, Err(StitchError::Tables { tile: 0, .. })),
            "{refused:?}"
        );
        let other_lens = ImageSize {
            width: 602,
            height: 600,
        };
        assert!(matches!(
            LensStitch::from_tables(SMALL, tables.clone(), other_lens),
            Err(StitchError::Tables { .. })
        ));
        let mut mixed = tables.clone();
        mixed[1] = hemisphere_tile_table(wide, 1, LENS, |d| lens(1).project(d)).unwrap();
        assert!(matches!(
            LensStitch::from_tables(SMALL, mixed, LENS),
            Err(StitchError::Tables { tile: 1, .. })
        ));
        assert!(matches!(
            LensStitch::new(
                SMALL,
                ImageSize {
                    width: 3,
                    height: 600
                },
                |_, _| None
            ),
            Err(StitchError::LensTooSmall { .. })
        ));
        assert!(matches!(
            LensStitch::new(
                TileGeometry {
                    margin: 121,
                    ..SMALL
                },
                LENS,
                |_, _| None
            ),
            Err(StitchError::Geometry(_))
        ));
        // The plan is the tables plus the host remap: the same tiles as lens_level1 + remap.
        let stitch = LensStitch::from_tables(SMALL, tables.clone(), LENS).unwrap();
        let img = render(&lens(1), &[[0.3f64, 0.1, -0.95]]);
        let mut out = Image::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0u16,
        )
        .unwrap();
        stitch
            .remap(1, img.as_slice(), LENS.width, &mut out)
            .unwrap();
        assert_eq!(out.as_slice(), tile_of(&img, &tables[1]).as_slice());
        assert!(matches!(
            stitch.remap(1, &img.as_slice()[1..], LENS.width, &mut out),
            Err(StitchError::LensBuffer { .. })
        ));
    }

    #[test]
    fn the_same_input_gives_the_same_bytes() {
        let l = lens(1);
        let spots = [[0.3f64, 0.1, -0.95]];
        let img = render(&l, &spots);
        let (a, b) = (table(1, &l), table(1, &l));
        assert_eq!(a, b);
        assert_eq!(tile_of(&img, &a).as_slice(), tile_of(&img, &b).as_slice());
    }

    #[test]
    fn the_panorama_is_the_two_tile_centres() {
        let size = SMALL.size();
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
