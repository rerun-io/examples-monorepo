//! The dual-fisheye stitch on the device: two lens images in, two hemisphere tiles' pyramids out.
//!
//! Per frameset, [`crate::pyramid::GpuPyramidBuilder::prepare_lenses`] uploads the two lens images (u8), reduces
//! each once exactly as `kornia_staging_imgproc::panorama::lens_level1` does (`byte << 8`, then `[1 4 6 4 1]`,
//! reflect-101, `(sum + 128) >> 8`, through the pyramid's own fused ingest kernel with level 0 not written), gathers
//! each tile's level 0 with `remap_tile`'s integer rule from the tables uploaded here, and reduces the tile
//! pyramids with the pyramid's subsample kernel. Every step is integer arithmetic, so the device tiles and their
//! pyramids equal the CPU's bit for bit.
//!
//! The tables are the CPU-built [`TileLut`]s, uploaded once as they are; the device never rebuilds them.

use std::mem::size_of;

use cubecl::prelude::*;
use kornia_image::ImageSize;
use kornia_staging_imgproc::panorama::{TileLut, LUT_INVALID};

use crate::runtime::{guarded, GpuError};
use crate::transfer::upload_inner;

/// A pair of tile tables the device cannot use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TableError {
    /// Table `index` is not tile `index`: tile `k` reads lens `k`, so the order is fixed.
    #[error("table {index} is for tile {tile}; tables come in tile order 0, 1")]
    TileOrder {
        /// Position in the pair.
        index: usize,
        /// The table's own tile index.
        tile: usize,
    },
    /// The two tables disagree on the tile or the source size.
    #[error("the two tables have different tile geometry or source size")]
    Mismatch,
    /// A table does not hold one entry per tile pixel.
    #[error("table {tile} holds {actual} entries, the tile has {expected} pixels")]
    Length {
        /// Tile index.
        tile: usize,
        /// Entries in the table.
        actual: usize,
        /// Tile pixels.
        expected: usize,
    },
    /// An entry is neither `LUT_INVALID` nor inside the level-1 image, so the gather would read outside it.
    #[error("table {tile} entry {pixel} lies outside the {width}x{height} level-1 image")]
    OutOfRange {
        /// Tile index.
        tile: usize,
        /// Row-major tile pixel.
        pixel: usize,
        /// Level-1 width.
        width: usize,
        /// Level-1 height.
        height: usize,
    },
    /// The level-1 image is below the bilinear gather's 2x2 reach.
    #[error("a {width}x{height} level-1 source is too small")]
    SourceTooSmall {
        /// Level-1 width.
        width: usize,
        /// Level-1 height.
        height: usize,
    },
    /// Upload failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// The two tiles' tables and the lens level-1 scratch, resident on one device.
///
/// Built once per rig (the tables are about 16.7 MB for two 1088x960 tiles); every frameset reuses them.
pub struct TileTables<R: Runtime> {
    /// `2 * tiles * tile pixels` u32: tile 0's `(qx, qy)` pairs, then tile 1's.
    pub(crate) table: cubecl::server::Handle,
    pub(crate) table_len: usize,
    /// Both lenses' level 1, `level1_stride` u16 apart.
    pub(crate) level1: cubecl::server::Handle,
    pub(crate) level1_stride: usize,
    /// A two-element binding for the fused ingest's level-0 output, which the lens reduction never writes.
    pub(crate) unused_level0: cubecl::server::Handle,
    /// Level-1 size, `lens / 2` floored.
    pub(crate) source: ImageSize,
    /// Tile size, level 0 of each tile pyramid.
    pub(crate) tile: ImageSize,
    _runtime: std::marker::PhantomData<R>,
}

/// Lenses and tiles in a stitch: tile `k` reads lens `k`.
pub const TILES: usize = 2;

impl<R: Runtime> TileTables<R> {
    /// Validate and upload the two CPU-built tables.
    ///
    /// # Arguments
    /// * `client` - The client the tile pyramids live on.
    /// * `tables` - Tile 0's and tile 1's tables, from `kornia_staging_imgproc::panorama::build_tile_lut`.
    ///
    /// # Errors
    /// A table out of order, of the wrong length, with an entry outside the level-1 image, two tables that
    /// disagree, or a device failure.
    pub fn upload(
        client: &ComputeClient<R>,
        tables: [&TileLut; TILES],
    ) -> Result<Self, TableError> {
        let first = tables[0];
        let source = first.source_size;
        if source.width < 2 || source.height < 2 {
            return Err(TableError::SourceTooSmall {
                width: source.width,
                height: source.height,
            });
        }
        let tile = ImageSize {
            width: first.geometry.width(),
            height: first.geometry.height,
        };
        let limit = [
            256 * (source.width as u32 - 1),
            256 * (source.height as u32 - 1),
        ];
        for (index, table) in tables.iter().enumerate() {
            if table.tile != index {
                return Err(TableError::TileOrder {
                    index,
                    tile: table.tile,
                });
            }
            if table.geometry != first.geometry || table.source_size != source {
                return Err(TableError::Mismatch);
            }
            if table.sample.len() != tile.width * tile.height {
                return Err(TableError::Length {
                    tile: index,
                    actual: table.sample.len(),
                    expected: tile.width * tile.height,
                });
            }
            if let Some(pixel) = table
                .sample
                .iter()
                .position(|&q| q != LUT_INVALID && (q[0] > limit[0] || q[1] > limit[1]))
            {
                return Err(TableError::OutOfRange {
                    tile: index,
                    pixel,
                    width: source.width,
                    height: source.height,
                });
            }
        }
        let flat: Vec<u32> = tables
            .iter()
            .flat_map(|table| table.sample.iter().flatten().copied())
            .collect();
        let level1_stride = source.width * source.height;
        guarded(
            GpuError::DeviceLost {
                what: "stitch table upload",
            },
            || {
                Ok(Self {
                    table: upload_inner(client, u32::as_bytes(&flat)),
                    table_len: flat.len(),
                    level1: client.empty(TILES * level1_stride * size_of::<u16>()),
                    level1_stride,
                    unused_level0: client.empty(2 * size_of::<u16>()),
                    source,
                    tile,
                    _runtime: std::marker::PhantomData,
                })
            },
        )
    }

    /// Level 0 of each tile pyramid.
    pub fn tile_size(&self) -> ImageSize {
        self.tile
    }

    /// Whether a lens image of `size` reduces to the tables' level-1 source.
    pub fn accepts_lens(&self, size: ImageSize) -> bool {
        size.width / 2 == self.source.width && size.height / 2 == self.source.height
    }
}

impl<R: Runtime> std::fmt::Debug for TileTables<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileTables")
            .field("source", &self.source)
            .field("tile", &self.tile)
            .finish()
    }
}
