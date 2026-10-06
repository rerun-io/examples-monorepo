//! sampling kernels and their launchers.
use super::layout::{linear_1d, Buffer};
use cubecl::prelude::*;

// Per-frame launchers use launch_unchecked. Stage shape checks establish
// the binding lengths; each raw binding keeps its handle and element count
// together. The storage probe alone retains checked launch mode.
// ── sampling ─────────────────────────────────────────────────────────────────

/// `Image<u16, 1>::in_bounds`.
#[cube]
pub fn in_bounds(x: f32, y: f32, border: f32, width: usize, height: usize) -> bool {
    border <= x
        && x < (f32::cast_from(width) - border - 1.0f32)
        && border <= y
        && y < (f32::cast_from(height) - border - 1.0f32)
}

/// One pixel of a level, as `f32`.
#[cube]
pub fn at(image: &[u16], base: usize, stride: usize, x: usize, y: usize) -> f32 {
    f32::cast_from(image[base + y * stride + x])
}

/// Bilinear sampling with the CPU implementation's multiplication grouping and sum order.
#[cube]
pub fn interp(image: &[u16], base: usize, stride: usize, x: f32, y: f32) -> f32 {
    let ix = usize::cast_from(x);
    let iy = usize::cast_from(y);
    let dx = x - f32::cast_from(ix);
    let dy = y - f32::cast_from(iy);
    let ddx = 1.0f32 - dx;
    let ddy = 1.0f32 - dy;
    ddx * ddy * at(image, base, stride, ix, iy)
        + ddx * dy * at(image, base, stride, ix, iy + 1usize)
        + dx * ddy * at(image, base, stride, ix + 1usize, iy)
        + dx * dy * at(image, base, stride, ix + 1usize, iy + 1usize)
}

/// Copy a typed buffer for the storage probe or a persistent pyramid's level zero.
#[cube(launch, launch_unchecked)]
fn copy_kernel<N: Numeric>(src: &[N], dst: &mut [N], count: usize) {
    let index = usize::cast_from(ABSOLUTE_POS);
    if index >= count {
        terminate!();
    }
    dst[index] = src[index];
}

/// Copy `count` elements of type `N` from `src` to `dst`.
///
/// The one launcher here that keeps CubeCL's bounds checks — `launch`, not
/// `launch_unchecked`. It runs once per process, off the per-frame path, and it
/// is the kernel whose whole job is to prove the runtime is sound: bounds checks
/// off is the wrong shape for that, and 256 elements cost nothing either way.
/// The `unsafe` that remains is `BufferArg::from_raw_parts`, which every launcher
/// needs to name a handle's element count, not the launch.
pub(crate) fn launch_probe<N: Numeric, R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    dst: Buffer<'_>,
    count: usize,
) {
    let (cubes, units) = linear_1d(count);

    unsafe {
        copy_kernel::launch::<N, R>(
            client,
            cubes,
            units,
            BufferArg::from_raw_parts(src.0.clone(), src.1),
            BufferArg::from_raw_parts(dst.0.clone(), dst.1),
            count,
        );
    }
}

/// Copy `count` pixels from the upload buffer to the front of `dst`.
///
/// `copy_kernel` instantiated at `u16`, not a second kernel: the body was
/// the same three lines, so a copy kernel of its own meant two
/// SPIR-V modules compiled for one copy. `launch_unchecked` here where
/// `launch_probe` takes the checked one — this is the per-frame path.
///
/// # Safety
/// Both handles must belong to `client` and hold their declared number of u16
/// elements. Each declared length must be at least `count`. The source and
/// destination ranges must not overlap.
pub(crate) unsafe fn launch_copy_level0<R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    dst: Buffer<'_>,
    count: usize,
) {
    let (cubes, units) = linear_1d(count);

    unsafe {
        copy_kernel::launch_unchecked::<u16, R>(
            client,
            cubes,
            units,
            BufferArg::from_raw_parts(src.0.clone(), src.1),
            BufferArg::from_raw_parts(dst.0.clone(), dst.1),
            count,
        );
    }
}
