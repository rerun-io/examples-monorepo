//! The dual-fisheye stitch gather: lens level 1 through a fixed-point table into tile level 0.
use super::layout::*;
use cubecl::prelude::*;

/// `kornia_staging_imgproc::panorama::remap_tile` on the device, one unit per tile pixel and one cube row of
/// `CUBE_POS_Z` per tile. The table holds `(qx, qy)` pairs in 1/256 px of the level-1 lens image; the host refuses
/// any entry that is neither `LUT_INVALID` nor inside `[0, 256 (w - 1)] x [0, 256 (h - 1)]`, so `qx > max_qx` is
/// exactly the invalid test. All arithmetic is `u32`, as on the CPU: the largest sum, `65535 * 65536 + 32768`,
/// fits.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gather_kernel(
    src: &[u16],
    table: &[u32],
    dst: &mut [u16],
    src_width: usize,
    src_height: usize,
    src_stride: usize,
    tile_width: usize,
    tile_height: usize,
    dst_stride: usize,
) {
    let c = usize::cast_from(ABSOLUTE_POS_X);
    let r = usize::cast_from(ABSOLUTE_POS_Y);
    let tile = usize::cast_from(CUBE_POS_Z);
    if c < tile_width && r < tile_height {
        let pixel = r * tile_width + c;
        let entry = 2usize * (tile * tile_width * tile_height + pixel);
        let qx = table[entry];
        let qy = table[entry + 1usize];
        let mut out = 0u32;
        if qx <= 256u32 * u32::cast_from(src_width - 1usize) {
            let x0 = usize::cast_from(qx >> 8u32);
            let y0 = usize::cast_from(qy >> 8u32);
            let ax = qx & 255u32;
            let ay = qy & 255u32;
            let x1 = min(x0 + 1usize, src_width - 1usize);
            let y1 = min(y0 + 1usize, src_height - 1usize);
            let base = tile * src_stride;
            let row0 = base + y0 * src_width;
            let row1 = base + y1 * src_width;
            let top = u32::cast_from(src[row0 + x0]) * (256u32 - ax)
                + u32::cast_from(src[row0 + x1]) * ax;
            let bottom = u32::cast_from(src[row1 + x0]) * (256u32 - ax)
                + u32::cast_from(src[row1 + x1]) * ax;
            out = (top * (256u32 - ay) + bottom * ay + 32_768u32) >> 16u32;
        }
        dst[tile * dst_stride + pixel] = u16::cast_from(out);
    }
}

/// Gather every tile's level 0 from its lens's level 1.
///
/// # Safety
/// All handles belong to `client` and hold their declared element counts. `src` holds `tiles` level-1 images of
/// `src_size` at `src_stride`, `table` holds `2 * tiles * tile pixels` validated entries, and `dst` holds `tiles`
/// slots of at least `tile pixels` at `dst_stride`. `dst` is a different allocation from `src` and `table`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_gather<R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    table: Buffer<'_>,
    dst: Buffer<'_>,
    src_size: [usize; 3],
    tile_size: [usize; 2],
    dst_stride: usize,
    tiles: usize,
) {
    let [src_width, src_height, src_stride] = src_size;
    let [tile_width, tile_height] = tile_size;
    unsafe {
        gather_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(
                (tile_width as u32).div_ceil(TILE_W),
                (tile_height as u32).div_ceil(TILE_H),
                tiles as u32,
            ),
            CubeDim::new_3d(TILE_W, TILE_H, 1),
            BufferArg::from_raw_parts(src.0.clone(), src.1),
            BufferArg::from_raw_parts(table.0.clone(), table.1),
            BufferArg::from_raw_parts(dst.0.clone(), dst.1),
            src_width,
            src_height,
            src_stride,
            tile_width,
            tile_height,
            dst_stride,
        );
    }
}
