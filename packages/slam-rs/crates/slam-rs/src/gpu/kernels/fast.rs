//! fast kernels and their launchers.
use super::layout::*;
use cubecl::prelude::*;

// Per-frame launchers use launch_unchecked. Stage shape checks establish
// the binding lengths; each raw binding keeps its handle and element count
// together. The storage probe alone retains checked launch mode.
// ── FAST detection ───────────────────────────────────────────────────────────

/// The Bresenham ring's row and column offsets, biased by 3 so they are
/// unsigned: `ring[k]` is the row and `ring[16 + k]` the column of ring point
/// `k`, each plus 3. Every caller is at least 3 pixels inside the image, so the
/// bias cancels without a signed intermediate.
pub(crate) const RING_BIAS: usize = 3;

/// kornia's `corner_score_9_scalar` (`fast.rs) over every pixel.
///
/// The score is FAST-9's own: the maximum over the sixteen arc starts of the
/// minimum over nine consecutive saturating differences, on the bright and the
/// dark side. It is written for **every** pixel, corner or not, because the
/// in-block local-maximum filter below compares raw scores.
///
/// This is what makes a GPU detector exact rather than approximate. kornia's
/// candidate test at threshold `t` is `p > center + t` or `p < center - t` for
/// nine consecutive ring points, which is the same statement as
/// `corner_score_9 > t` — including the saturation, since a saturated bound
/// forces the corresponding difference under `t` too. So one dense score image
/// answers every rung of the detector's threshold ladder, and
/// `tests/fast_model.rs` checks that claim against kornia itself at five rungs
/// and two widths before any of this runs.
#[cube(launch, launch_unchecked)]
fn fast_score_kernel(
    frame: &Array<u16>,
    ring: &Array<u32>,
    score: &mut Array<u8>,
    width: usize,
    height: usize,
    margin: usize,
) {
    let x = usize::cast_from(ABSOLUTE_POS_X);
    let y = usize::cast_from(ABSOLUTE_POS_Y);
    if x >= width || y >= height {
        terminate!();
    }
    let slot = y * width + x;
    if x < margin || y < margin || x + margin >= width || y + margin >= height {
        score[slot] = 0u8;
        terminate!();
    }

    // `sub_ptr[x] = (sub_img_raw(x, y) >> 8)`, on the
    // device: uploading the `u16` frame and shifting here costs 0.9 MB more over
    // the bus and saves a whole-frame pass on the host, which measured the
    // larger of the two.
    let center = u32::cast_from(frame[slot]) >> 8u32;
    // `#[unroll]` on all three loops below, and not for the sake of this card.
    // `dark` and `bright` are indexed dynamically (`dark[(k + i) % 16]`) from
    // loops with comptime bounds, so a rolled kernel cannot keep them in
    // registers: 16 x 4 B x 2 arrays x 256 units is 32 kB of local memory per
    // cube in the frame's hottest kernel, plus seventeen uncoalesced global
    // loads per pixel. Unrolled, every index is a literal and both arrays fit
    // in registers. On a 5090 at 7-11 % utilisation this changes nothing
    // measurable and nothing should be concluded from measuring it here; it is
    // for the shared-LPDDR targets the portable lane exists for, where
    // `Robocap.md`'s Pi 5 lesson is that a memory-bound kernel loses to the CPU
    // outright. The corner-scan equality tests on both lanes are what say it
    // changed no value.
    let mut dark = Array::<u32>::new(16usize);
    let mut bright = Array::<u32>::new(16usize);
    #[unroll]
    for k in 0..16usize {
        let row = y + usize::cast_from(ring[k]) - RING_BIAS;
        let column = x + usize::cast_from(ring[16usize + k]) - RING_BIAS;
        let p = u32::cast_from(frame[row * width + column]) >> 8u32;
        let mut below: u32 = 0u32;
        if center > p {
            below = center - p;
        }
        let mut above: u32 = 0u32;
        if p > center {
            above = p - center;
        }
        dark[k] = below;
        bright[k] = above;
    }

    // Step by 2: each iteration shares the seven-element core `[k+1 .. k+8]`
    // between the arc starting at `k` and the one starting at `k+1`, which is
    // the canonical FAST-9 structure and the order kornia sums it in.
    let mut dark_score: u32 = 0u32;
    let mut bright_score: u32 = 0u32;
    #[unroll]
    for step in 0..8usize {
        let k = 2usize * step;
        let mut core_dark = dark[(k + 1usize) % 16usize];
        let mut core_bright = bright[(k + 1usize) % 16usize];
        #[unroll]
        for i in 2..9usize {
            core_dark = min(core_dark, dark[(k + i) % 16usize]);
            core_bright = min(core_bright, bright[(k + i) % 16usize]);
        }
        dark_score = max(dark_score, min(core_dark, dark[k]));
        dark_score = max(dark_score, min(core_dark, dark[(k + 9usize) % 16usize]));
        bright_score = max(bright_score, min(core_bright, bright[k]));
        bright_score = max(
            bright_score,
            min(core_bright, bright[(k + 9usize) % 16usize]),
        );
    }
    score[slot] = u8::cast_from(max(dark_score, bright_score));
}

/// Lanes per block of kornia's local-maximum filter, off the CPU detector's own
/// constant so the kernel and the host cannot disagree about the block
/// alignment that decides the corner set.
const FILTER_LANES: usize = crate::frontend::detect::FAST_FILTER_LANES;
/// The last lane of a block, which has no right-hand neighbour to beat.
const FILTER_LAST: usize = FILTER_LANES - 1;

/// kornia's in-block local-maximum filter (`fast.rs).
///
/// Turned on at `width >= 800`, where dense-corner images emit so many
/// candidates that `Vec::push` dominates the CPU kernel; kornia keeps a lane
/// only when its score beats its two neighbours **inside its own sixteen-lane
/// block**, with zero outside the block. The blocks are aligned to the image's
/// own left margin and the tail past the last whole block takes the scalar path,
/// which has no filter — so the alignment and the tail are part of the corner
/// set, not an implementation detail, and both are reproduced here.
///
/// The output is the score where a candidate survives and zero elsewhere, which
/// makes `kept > t` the exact candidate set for every threshold `t >= 0`.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn fast_localmax_kernel(
    score: &Array<u8>,
    kept: &mut Array<u8>,
    width: usize,
    height: usize,
    margin: usize,
    filtered_end: usize,
    use_filter: usize,
) {
    let x = usize::cast_from(ABSOLUTE_POS_X);
    let y = usize::cast_from(ABSOLUTE_POS_Y);
    if x >= width || y >= height {
        terminate!();
    }
    let slot = y * width + x;
    let value = u32::cast_from(score[slot]);
    let mut survivor = value;
    if use_filter == 1usize && x >= margin && x < filtered_end {
        let lane = (x - margin) % FILTER_LANES;
        let mut left: u32 = 0u32;
        if lane != 0usize {
            left = u32::cast_from(score[slot - 1usize]);
        }
        let mut right: u32 = 0u32;
        if lane != FILTER_LAST {
            right = u32::cast_from(score[slot + 1usize]);
        }
        if value <= left || value <= right {
            survivor = 0u32;
        }
    }
    kept[slot] = u8::cast_from(survivor);
}

/// The dense FAST-9 score image of one frame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_fast_score<R: Runtime>(
    client: &ComputeClient<R>,
    frame: Buffer<'_>,
    ring: Buffer<'_>,
    score: Buffer<'_>,
    width: usize,
    height: usize,
    margin: usize,
) {
    let (cubes, units) = tile_2d(width, height);
    super::super::submission::launch(client);
    unsafe {
        fast_score_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(frame.0.clone(), frame.1),
            ArrayArg::from_raw_parts(ring.0.clone(), ring.1),
            ArrayArg::from_raw_parts(score.0.clone(), score.1),
            width,
            height,
            margin,
        );
    }
}

/// The candidate image: the score where the local-maximum filter keeps it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_fast_localmax<R: Runtime>(
    client: &ComputeClient<R>,
    score: Buffer<'_>,
    kept: Buffer<'_>,
    width: usize,
    height: usize,
    margin: usize,
    filtered_end: usize,
    use_filter: bool,
) {
    let (cubes, units) = tile_2d(width, height);
    super::super::submission::launch(client);
    unsafe {
        fast_localmax_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(score.0.clone(), score.1),
            ArrayArg::from_raw_parts(kept.0.clone(), kept.1),
            width,
            height,
            margin,
            filtered_end,
            usize::from(use_filter),
        );
    }
}

/// Words of [`MASK_BITS`] columns each in one row of the candidate bitmask.
pub(crate) const MASK_BITS: usize = 32;

/// One bit per column of the candidate image, packed thirty-two to a word.
///
/// The detector asks for up to eighty row bands per camera per frame and each
/// one is the whole image width, so walking the candidate image byte by byte
/// costs more than the FAST sweep it replaced: a byte-by-byte host walk measured
/// 1.53 ms against kornia's 1.81 ms, so the whole GPU scan was barely ahead of
/// the CPU one. With a bitmask the host reads one word per thirty-two columns
/// and only touches the score where a bit is set.
#[cube(launch, launch_unchecked)]
fn fast_mask_kernel(
    kept: &Array<u8>,
    mask: &mut Array<u32>,
    width: usize,
    height: usize,
    words: usize,
) {
    let word = usize::cast_from(ABSOLUTE_POS_X);
    let y = usize::cast_from(ABSOLUTE_POS_Y);
    if word >= words || y >= height {
        terminate!();
    }
    let first = word * MASK_BITS;
    let mut bits: u32 = 0u32;
    for bit in 0..MASK_BITS {
        let x = first + bit;
        if x < width && kept[y * width + x] != 0u8 {
            bits |= 1u32 << u32::cast_from(bit);
        }
    }
    mask[y * words + word] = bits;
}

/// Pack the candidate image into one bit per column.
pub(crate) fn launch_fast_mask<R: Runtime>(
    client: &ComputeClient<R>,
    kept: Buffer<'_>,
    mask: Buffer<'_>,
    width: usize,
    height: usize,
    words: usize,
) {
    let (cubes, units) = tile_2d(words, height);
    super::super::submission::launch(client);
    unsafe {
        fast_mask_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(kept.0.clone(), kept.1),
            ArrayArg::from_raw_parts(mask.0.clone(), mask.1),
            width,
            height,
            words,
        );
    }
}
