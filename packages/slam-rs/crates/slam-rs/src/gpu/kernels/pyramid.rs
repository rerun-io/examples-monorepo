//! pyramid kernels and their launchers.
use super::layout::*;
use cubecl::prelude::*;

// Per-frame launchers use launch_unchecked. Stage shape checks establish
// the binding lengths; each raw binding keeps its handle and element count
// together. The storage probe alone retains checked launch mode.
// ── the pyramid ──────────────────────────────────────────────────────────────

/// `border101` for an index past the high end (`image_pyr.h:83`).
///
/// `h - 1 - |h - 1 - x|`, written without an absolute value so it needs no
/// signed intermediate: below `h` it is the identity, at or above it the
/// reflection `2h - 2 - x`. Callers only ever reach `x <= h`.
#[cube]
fn reflect_high(x: usize, h: usize) -> usize {
    if x < h { x } else { h + h - 2usize - x }
}

/// `std::abs(2 * r - k)` (`image_pyr.h:110-111`), which is the *other*
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
    src: &Array<u16>,
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

/// [`crate::pyramid::subsample`] (`image_pyr.h:99-140`) as one fused 5x5 pass.
///
/// The CPU port runs the separable form: a vertical pass into an `i32`
/// accumulator, then a horizontal pass with one rounding at the very end. Both
/// passes are exact integer sums, so the fused 5x5 here is **bit-exact** with
/// it — which is why this kernel's tolerance test asserts equality rather than
/// a bound. Row indices reflect about the source height and column indices
/// about the source width, exactly as the transposed C++ accumulator makes
/// them (see the CPU docstring). Every pixel is non-negative and the
/// accumulator peaks at `65535 * 16 * 16 = 16,776,960`, so `usize` carries it and
/// `>> 8` is the C++ shift rather than a division that would differ on a
/// negative value.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn subsample_kernel(
    src: &Array<u16>,
    dst: &mut Array<u16>,
    src_base: usize,
    src_width: usize,
    src_height: usize,
    dst_base: usize,
    dst_width: usize,
    dst_height: usize,
) {
    let c = usize::cast_from(ABSOLUTE_POS_X);
    let r = usize::cast_from(ABSOLUTE_POS_Y);
    if c >= dst_width || r >= dst_height {
        terminate!();
    }

    let row2 = 2usize * r;
    let col2 = 2usize * c;
    let r0 = reflect_low(row2, 2usize);
    let r1 = reflect_low(row2, 1usize);
    let r3 = reflect_high(row2 + 1usize, src_height);
    let r4 = reflect_high(row2 + 2usize, src_height);
    let c0 = reflect_low(col2, 2usize);
    let c1 = reflect_low(col2, 1usize);
    let c3 = reflect_high(col2 + 1usize, src_width);
    let c4 = reflect_high(col2 + 2usize, src_width);

    let acc = subsample_band(src, src_base + r0 * src_width, c0, c1, col2, c3, c4)
        + 4usize * subsample_band(src, src_base + r1 * src_width, c0, c1, col2, c3, c4)
        + 6usize * subsample_band(src, src_base + row2 * src_width, c0, c1, col2, c3, c4)
        + 4usize * subsample_band(src, src_base + r3 * src_width, c0, c1, col2, c3, c4)
        + subsample_band(src, src_base + r4 * src_width, c0, c1, col2, c3, c4);

    // `T val = ((val_int + (1 << 7)) >> 8)` (`image_pyr.h:135`).
    dst[dst_base + r * dst_width + c] = u16::cast_from((acc + 128usize) >> 8usize);
}

/// One level of one pyramid: `source` down into `target` at half its size.
pub(crate) fn launch_subsample<R: Runtime>(
    client: &ComputeClient<R>,
    src: Buffer<'_>,
    dst: Buffer<'_>,
    source: super::super::pyramid::Level,
    target: super::super::pyramid::Level,
) {
    let (cubes, units) = tile_2d(target.width, target.height);
    super::super::submission::launch(client);
    unsafe {
        subsample_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(src.0.clone(), src.1),
            ArrayArg::from_raw_parts(dst.0.clone(), dst.1),
            source.base,
            source.width,
            source.height,
            target.base,
            target.width,
            target.height,
        );
    }
}
