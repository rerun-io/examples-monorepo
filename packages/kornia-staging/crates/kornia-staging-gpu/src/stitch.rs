//! The dual-fisheye stitch on the device: two lens images in, two hemisphere tiles' pyramids out.
//!
//! Per frameset, [`crate::pyramid::GpuPyramidBuilder::prepare_lenses`] uploads the two lens images (u8), reduces
//! each once exactly as `kornia_staging_imgproc::panorama::lens_level1` does (`byte << 8`, then `[1 4 6 4 1]`,
//! reflect-101, `(sum + 128) >> 8`, through the pyramid's own fused ingest kernel with level 0 not written), gathers
//! each tile's level 0 with `remap`'s integer rule from the tables uploaded here, and reduces the tile
//! pyramids with the pyramid's subsample kernel. Every step is integer arithmetic, so the device tiles and their
//! pyramids equal the CPU's bit for bit.
//!
//! The tables are the CPU-built [`LensStitch`] tables, valid by construction and uploaded once as they are; the device
//! never rebuilds or re-checks them.

use std::mem::size_of;

use cubecl::prelude::*;
use kornia_image::ImageSize;
use kornia_staging_imgproc::panorama::LensStitch;

use crate::runtime::{guarded, GpuError};
use crate::transfer::upload_inner;

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
    /// Upload the stitch's two tables.
    ///
    /// # Arguments
    /// * `client` - The client the tile pyramids live on.
    /// * `stitch` - The host stitch plan; its tables map the tile from the lens's level 1 (2x2 or larger).
    ///
    /// # Errors
    /// A device failure.
    pub fn upload(client: &ComputeClient<R>, stitch: &LensStitch) -> Result<Self, GpuError> {
        let tables = stitch.tables();
        let source = tables[0].src_size();
        let tile = stitch.tile_size();
        let flat: Vec<u32> = tables
            .iter()
            .flat_map(|table| table.samples().iter().flatten().copied())
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
