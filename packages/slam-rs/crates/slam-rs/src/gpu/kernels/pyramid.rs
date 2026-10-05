//! pyramid kernels and their launchers.
use super::layout::*;
use cubecl::prelude::*;

// Per-frame launchers use launch_unchecked. Stage shape checks establish
// the binding lengths; each raw binding keeps its handle and element count
// together. The storage probe alone retains checked launch mode.
// ── the pyramid ──────────────────────────────────────────────────────────────

/// `border101` for an index past the high end.
///
/// `h - 1 - |h - 1 - x|`, written without an absolute value so it needs no
/// signed intermediate: below `h` it is the identity, at or above it the
/// reflection `2h - 2 - x`. Callers only ever reach `x <= h`.
#[cube]
fn reflect_high(x: usize, h: usize) -> usize {
    if x < h { x } else { h + h - 2usize - x }
}

/// `std::abs(2 * r - k)`, which is the *other*
/// reflection: about zero rather than about the far edge. Trap 3 of the
/// architecture dossier is that these two are not one function.
// `usize::abs_diff` is not in CubeCL's kernel language, so the subtraction is
// spelled out.
#[allow(clippy::manual_abs_diff)]
#[cube]
fn reflect_low(twice: usize, k: usize) -> usize {
    if twice >= k { twice - k } else { k - twice }
}

/// One row's five horizontal taps, `[1, 4, 6, 4, 1]`, as an exact `usize` sum.
#[cube]
#[allow(clippy::too_many_arguments)]
fn subsample_band(
    src: &[u16],
    row_base: usize,
    c0: usize,
    c1: usize,
    c2: usize,
    c3: usize,
    c4: usize,
) -> usize {
    usize::cast_from(src[row_base + c0])
        + 4usize * usize::cast_from(src[row_base + c1])
        + 6usize * usize::cast_from(src[row_base + c2])
        + 4usize * usize::cast_from(src[row_base + c3])
        + usize::cast_from(src[row_base + c4])
}

#[cube]
fn input_byte(src: &[u32], index: usize) -> u32 {
    (src[index / 4usize] >> u32::cast_from(8usize * (index % 4usize))) & 255u32
}

/// Widen level zero and reduce level one in the same dispatch. All inputs are
/// bytes, so the first filter's factor of 256 cancels its final division exactly.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn ingest_kernel(
    src: &[u32],
    even: &mut [u16],
    odd: &mut [u16],
    #[comptime] width: usize,
    #[comptime] height: usize,
    even_stride: usize,
    odd_stride: usize,
) {
    let c = usize::cast_from(ABSOLUTE_POS_X);
    let r = usize::cast_from(ABSOLUTE_POS_Y);
    let camera = usize::cast_from(CUBE_POS_Z);
    let src_base = camera * width * height;
    let dst_width = width / 2usize;
    let dst_height = height / 2usize;
    let x = usize::cast_from(UNIT_POS_X);
    let y = usize::cast_from(UNIT_POS_Y);
    let first_row = usize::cast_from(CUBE_POS_Y) * 2usize * SUBSAMPLE_H;
    let col2 = 2usize * c;
    let mut horizontal = Shared::<[u32]>::new_slice(SUBSAMPLE_W * SOURCE_ROWS);
    for sy in range_stepped(y, SOURCE_ROWS, SUBSAMPLE_H) {
        let raw_row = first_row + sy;
        let mut value = 0u32;
        if c < dst_width && raw_row <= 2usize * dst_height + 2usize {
            let source_row = reflect_high(reflect_low(raw_row, 2usize), height);
            let base = src_base + source_row * width;
            if comptime!(width % 4 == 0) && c > 0usize && c + 1usize < dst_width {
                // Five adjacent bytes span two words. Two coalesced loads
                // replace five scalar byte loads on the load/store-bound Mali.
                let start = base + col2 - 2usize;
                let first = src[start / 4usize];
                let second = src[start / 4usize + 1usize];
                let shift = u32::cast_from(8usize * (start % 4usize));
                let a = first >> shift;
                let b = select(shift == 0u32, first >> 16u32, second);
                let last = select(shift == 0u32, second, second >> 16u32);
                value = (a & 255u32)
                    + 4u32 * ((a >> 8u32) & 255u32)
                    + 6u32 * (b & 255u32)
                    + 4u32 * ((b >> 8u32) & 255u32)
                    + (last & 255u32);
            } else {
                value = input_byte(src, base + reflect_low(col2, 2usize))
                    + 4u32 * input_byte(src, base + reflect_low(col2, 1usize))
                    + 6u32 * input_byte(src, base + col2)
                    + 4u32 * input_byte(src, base + reflect_high(col2 + 1usize, width))
                    + input_byte(src, base + reflect_high(col2 + 2usize, width));
            }
        }
        horizontal[sy * SUBSAMPLE_W + x] = value;
    }
    if c < dst_width && r < dst_height {
        if comptime!(width % 4 == 0 && height % 2 == 0) {
            #[unroll]
            for row in 0..2usize {
                let slot = (2usize * r + row) * width + col2;
                let bits =
                    src[(src_base + slot) / 4usize] >> u32::cast_from(8usize * (slot % 4usize));
                even[camera * even_stride + slot] = u16::cast_from((bits & 255u32) << 8u32);
                even[camera * even_stride + slot + 1usize] = u16::cast_from(bits & 65_280u32);
            }
        } else {
            // The final output owns an extra input row/column for odd geometry.
            let end_x = min(col2 + 2usize, width);
            let end_y = min(2usize * r + 2usize, height);
            let end_x = select(c + 1usize == dst_width, width, end_x);
            let end_y = select(r + 1usize == dst_height, height, end_y);
            for sy in 2usize * r..end_y {
                for sx in col2..end_x {
                    let slot = sy * width + sx;
                    even[camera * even_stride + slot] =
                        u16::cast_from(input_byte(src, src_base + slot)) << 8u16;
                }
            }
        }
    }
    sync_cube();
    if c < dst_width && r < dst_height {
        let slot = 2usize * y * SUBSAMPLE_W + x;
        let acc = vertical(&horizontal, slot);
        odd[camera * odd_stride + r * dst_width + c] = u16::cast_from(acc);
    }
}

#[cube]
fn vertical(horizontal: &Shared<[u32]>, slot: usize) -> u32 {
    horizontal[slot]
        + 4u32 * horizontal[slot + SUBSAMPLE_W]
        + 6u32 * horizontal[slot + 2usize * SUBSAMPLE_W]
        + 4u32 * horizontal[slot + 3usize * SUBSAMPLE_W]
        + horizontal[slot + 4usize * SUBSAMPLE_W]
}

const SUBSAMPLE_W: usize = 16;
const SUBSAMPLE_H: usize = 8;
const SOURCE_ROWS: usize = 2 * SUBSAMPLE_H + 4;

/// Separable tiled counterpart of [`crate::pyramid::subsample`].
/// Exact integer sums and one final rounding make it equal to the separable CPU
/// filter. Reflect rows about source height and columns about source width.
/// The maximum accumulator is `65535 * 16 * 16`, which fits the working integer.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn subsample_kernel(
    src: &[u16],
    dst: &mut [u16],
    src_base: usize,
    src_width: usize,
    src_height: usize,
    dst_base: usize,
    dst_width: usize,
    dst_height: usize,
    src_camera_stride: usize,
    dst_camera_stride: usize,
) {
    let c = usize::cast_from(ABSOLUTE_POS_X);
    let r = usize::cast_from(ABSOLUTE_POS_Y);
    let camera = usize::cast_from(CUBE_POS_Z);
    let src_base = src_base + camera * src_camera_stride;
    let dst_base = dst_base + camera * dst_camera_stride;
    let col2 = 2usize * c;
    let x = usize::cast_from(UNIT_POS_X);
    let y = usize::cast_from(UNIT_POS_Y);
    let first_row = usize::cast_from(CUBE_POS_Y) * 2usize * SUBSAMPLE_H;
    let mut horizontal = Shared::<[u32]>::new_slice(SUBSAMPLE_W * SOURCE_ROWS);
    // Each horizontal result serves as many as three output rows. Preserve the
    // full integer sum here; rounding belongs only after the vertical filter.
    for sy in range_stepped(y, SOURCE_ROWS, SUBSAMPLE_H) {
        let raw_row = first_row + sy;
        let mut value = 0u32;
        if c < dst_width && raw_row <= 2usize * dst_height + 2usize {
            let source_row = reflect_high(reflect_low(raw_row, 2usize), src_height);
            value = u32::cast_from(subsample_band(
                src,
                src_base + source_row * src_width,
                reflect_low(col2, 2usize),
                reflect_low(col2, 1usize),
                col2,
                reflect_high(col2 + 1usize, src_width),
                reflect_high(col2 + 2usize, src_width),
            ));
        }
        horizontal[sy * SUBSAMPLE_W + x] = value;
    }
    sync_cube();
    if c < dst_width && r < dst_height {
        let slot = 2usize * y * SUBSAMPLE_W + x;
        let acc = vertical(&horizontal, slot);
        dst[dst_base + r * dst_width + c] = u16::cast_from((acc + 128u32) >> 8u32);
    }
}

/// One level of one pyramid: `source` down into `target` at half its size.
pub(crate) fn launch_subsample<R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    dst: Buffer<'_>,
    source: super::super::pyramid::Level,
    target: super::super::pyramid::Level,
) {
    launch_subsample_batch(client, src, dst, source, target, 0, 0, 1);
}

/// Equal-geometry cameras share a dispatch, with independent borders and tiles.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_subsample_batch<R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    dst: Buffer<'_>,
    source: super::super::pyramid::Level,
    target: super::super::pyramid::Level,
    src_camera_stride: usize,
    dst_camera_stride: usize,
    cameras: usize,
) {
    let cubes = CubeCount::Static(
        target.width.div_ceil(SUBSAMPLE_W) as u32,
        target.height.div_ceil(SUBSAMPLE_H) as u32,
        cameras as u32,
    );
    let units = CubeDim::new_3d(SUBSAMPLE_W as u32, SUBSAMPLE_H as u32, 1);

    // SAFETY: The builder validates level extents and allocates every camera stride.
    // Each group owns one target tile; the kernel clips edge writes and reflects source
    // borders.
    unsafe {
        subsample_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            BufferArg::from_raw_parts(src.0.clone(), src.1),
            BufferArg::from_raw_parts(dst.0.clone(), dst.1),
            source.base,
            source.width,
            source.height,
            target.base,
            target.width,
            target.height,
            src_camera_stride,
            dst_camera_stride,
        );
    }
}

/// Ingest dense byte frames into level zero and the first reduced level.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_ingest<R: Runtime>(
    client: &ComputeClient<R>,
    bytes: Buffer<'_>,
    even: Buffer<'_>,
    odd: Buffer<'_>,
    width: usize,
    height: usize,
    odd_stride: usize,
    cameras: usize,
) {
    // SAFETY: Dense u8 uploads are padded to whole u32 words. The builder allocated
    // even/odd arenas for every camera and these strides; the kernel bounds the edge
    // tiles.
    unsafe {
        ingest_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(
                (width / 2).div_ceil(SUBSAMPLE_W) as u32,
                (height / 2).div_ceil(SUBSAMPLE_H) as u32,
                cameras as u32,
            ),
            CubeDim::new_3d(SUBSAMPLE_W as u32, SUBSAMPLE_H as u32, 1),
            BufferArg::from_raw_parts(bytes.0.clone(), bytes.1 / 4),
            BufferArg::from_raw_parts(even.0.clone(), even.1),
            BufferArg::from_raw_parts(odd.0.clone(), odd.1),
            width,
            height,
            even.1 / cameras,
            odd_stride,
        );
    }
}
