//! sampling kernels and their launchers.
use super::layout::{Buffer, linear_1d};
use cubecl::prelude::*;

// Per-frame launchers use launch_unchecked. Stage shape checks establish
// the binding lengths; each raw binding keeps its handle and element count
// together. The storage probe alone retains checked launch mode.
// ── sampling ─────────────────────────────────────────────────────────────────

/// `ImageU16::in_bounds`.
#[cube]
pub(crate) fn in_bounds(x: f32, y: f32, border: f32, width: usize, height: usize) -> bool {
    border <= x
        && x < (f32::cast_from(width) - border - 1.0f32)
        && border <= y
        && y < (f32::cast_from(height) - border - 1.0f32)
}

/// One pixel of a level, as `f32`.
#[cube]
fn at(image: &[u16], base: usize, stride: usize, x: usize, y: usize) -> f32 {
    f32::cast_from(image[base + y * stride + x])
}

/// Bilinear sampling with the CPU implementation's multiplication grouping and sum order.
#[cube]
pub(crate) fn interp(image: &[u16], base: usize, stride: usize, x: f32, y: f32) -> f32 {
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

/// `ImageU16::interp_grad` writing its three results into
/// three shared arrays at `slot`.
///
/// The value is the bilinear surface; the gradient is the *central difference*
/// of that surface at unit spacing, not its analytic derivative, which is why
/// twelve pixels are read (`ix - 1` through `ix + 2`, `iy - 1` through
/// `iy + 2`) and why the caller must have satisfied `in_bounds(x, y, 1)`.
///
/// Three `&mut Shared` parameters rather than a returned triple: a
/// `#[cube]` function returns one value, and the caller wants these in shared
/// memory anyway so unit 0 can reduce over them.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn interp_grad_into(
    image: &[u16],
    base: usize,
    stride: usize,
    x: f32,
    y: f32,
    values: &mut Shared<[f32]>,
    grad_x: &mut Shared<[f32]>,
    grad_y: &mut Shared<[f32]>,
    slot: usize,
) {
    let ix = usize::cast_from(x);
    let iy = usize::cast_from(y);
    let dx = x - f32::cast_from(ix);
    let dy = y - f32::cast_from(iy);
    let ddx = 1.0f32 - dx;
    let ddy = 1.0f32 - dy;

    let px0y0 = at(image, base, stride, ix, iy);
    let px1y0 = at(image, base, stride, ix + 1usize, iy);
    let px0y1 = at(image, base, stride, ix, iy + 1usize);
    let px1y1 = at(image, base, stride, ix + 1usize, iy + 1usize);
    values[slot] = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;

    let pxm1y0 = at(image, base, stride, ix - 1usize, iy);
    let pxm1y1 = at(image, base, stride, ix - 1usize, iy + 1usize);
    let res_mx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;
    let px2y0 = at(image, base, stride, ix + 2usize, iy);
    let px2y1 = at(image, base, stride, ix + 2usize, iy + 1usize);
    let res_px = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;
    grad_x[slot] = 0.5f32 * (res_px - res_mx);

    let px0ym1 = at(image, base, stride, ix, iy - 1usize);
    let px1ym1 = at(image, base, stride, ix + 1usize, iy - 1usize);
    let res_my = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;
    let px0y2 = at(image, base, stride, ix, iy + 2usize);
    let px1y2 = at(image, base, stride, ix + 1usize, iy + 2usize);
    let res_py = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;
    grad_y[slot] = 0.5f32 * (res_py - res_my);
}

/// A device copy of `count` elements of one width.
///
/// Two callers, and the second is why the kernel is generic rather than fixed
/// to the width the probe needs. [`super::super::probe_storage`] copies a known
/// pattern of four widths to refuse a runtime that cannot store one of them;
/// [`launch_copy_level0`] copies the frame at `u16`, because the upload and the
/// pyramid want different shapes. The upload wants to be exactly as long as the
/// frame — `create_from_slice` is CubeCL 0.10's only host-to-device write and it
/// copies the payload on the host before the bus sees it, so every byte over
/// the frame is paid for twice — while the pyramid wants level 0 at offset zero
/// of the buffer that also holds levels 2 and 4, because the per-patch kernels
/// reach a level through two bindings split by parity. One device copy of the
/// frame, a few microseconds at this card's bandwidth, buys both, and leaves
/// the even allocation to be made once at `allocate` rather than replaced every
/// frame.
#[cube(launch, launch_unchecked)]
fn probe_kernel<N: Numeric>(src: &[N], dst: &mut [N], count: usize) {
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
    super::super::submission::launch(client);
    unsafe {
        probe_kernel::launch::<N, R>(
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
/// [`probe_kernel`] instantiated at `u16`, not a second kernel: the body was
/// the same three lines, so a copy kernel of its own meant two
/// SPIR-V modules compiled for one copy. `launch_unchecked` here where
/// [`launch_probe`] takes the checked one — this is the per-frame path.
pub(crate) fn launch_copy_level0<R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    dst: Buffer<'_>,
    count: usize,
) {
    let (cubes, units) = linear_1d(count);
    super::super::submission::launch(client);
    unsafe {
        probe_kernel::launch_unchecked::<u16, R>(
            client,
            cubes,
            units,
            BufferArg::from_raw_parts(src.0.clone(), src.1),
            BufferArg::from_raw_parts(dst.0.clone(), dst.1),
            count,
        );
    }
}
