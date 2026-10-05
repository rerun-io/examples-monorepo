//! Row kernels of [`super::resize_area_u8`]: a NEON path for the 3x3 mono case on aarch64, scalar elsewhere.

/// One output row of the generic box downscale. `block` holds the `ky` input rows (stride `src_stride`).
pub(super) fn area_row_generic<const C: usize>(
    block: &[u8],
    src_stride: usize,
    kx: usize,
    ky: usize,
    out: &mut [u8],
) {
    let area = (kx * ky) as u32;
    for (x, pixel) in out.chunks_exact_mut(C).enumerate() {
        for (channel, value) in pixel.iter_mut().enumerate() {
            let mut sum = 0u32;
            for dy in 0..ky {
                let row = &block[dy * src_stride..];
                for dx in 0..kx {
                    sum += u32::from(row[(x * kx + dx) * C + channel]);
                }
            }
            *value = ((sum + area / 2) / area) as u8;
        }
    }
}

/// One output row of the 3x3 mono case, scalar: `out[x] = (sum of the 3x3 block + 4) / 9`.
pub(super) fn area3_row_scalar(r0: &[u8], r1: &[u8], r2: &[u8], out: &mut [u8]) {
    for (x, value) in out.iter_mut().enumerate() {
        let at = 3 * x;
        let column = |i: usize| u16::from(r0[i]) + u16::from(r1[i]) + u16::from(r2[i]);
        let sum = column(at) + column(at + 1) + column(at + 2);
        *value = ((sum + 4) / 9) as u8;
    }
}

/// One output row of the 3x3 mono case: NEON on aarch64 (16 outputs per step via `vld3q_u8` deinterleave), scalar tail.
#[cfg(target_arch = "aarch64")]
pub(super) fn area3_row(r0: &[u8], r1: &[u8], r2: &[u8], out: &mut [u8]) {
    let n = out.len();
    if r0.len() < 3 * n || r1.len() < 3 * n || r2.len() < 3 * n {
        return area3_row_scalar(r0, r1, r2, out);
    }
    let full = n / 16 * 16;
    // SAFETY: NEON is part of the aarch64 baseline. Every load reads 48 bytes at 3*x with x + 16 <= full <= n, so it stays
    // inside r0/r1/r2 (each at least 3n long, checked above); every store writes 16 bytes at x + 16 <= n inside `out`.
    unsafe {
        area3_row_neon(
            r0.as_ptr(),
            r1.as_ptr(),
            r2.as_ptr(),
            out.as_mut_ptr(),
            full,
        )
    };
    area3_row_scalar(
        &r0[3 * full..],
        &r1[3 * full..],
        &r2[3 * full..],
        &mut out[full..],
    );
}

/// Scalar build of [`area3_row`] on other architectures (the compiler vectorises it).
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn area3_row(r0: &[u8], r1: &[u8], r2: &[u8], out: &mut [u8]) {
    area3_row_scalar(r0, r1, r2, out)
}

/// `(v + 4) / 9` for every u16 lane `v <= 2295`, as `((v + 4) * 7282) >> 16` (exact for `v + 4 <= 2299`; tested).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn div9_round(sum: core::arch::aarch64::uint16x8_t) -> core::arch::aarch64::uint8x8_t {
    use core::arch::aarch64::*;
    let v = vaddq_u16(sum, vdupq_n_u16(4));
    let lo = vmull_u16(vget_low_u16(v), vdup_n_u16(7282));
    let hi = vmull_high_u16(v, vdupq_n_u16(7282));
    vmovn_u16(vcombine_u16(vshrn_n_u32::<16>(lo), vshrn_n_u32::<16>(hi)))
}

/// How far ahead of the loads the kernel prefetches each input row.
#[cfg(target_arch = "aarch64")]
const PREFETCH_BYTES: usize = 320;

/// # Safety
///
/// `r0`, `r1`, `r2` must be readable for `3 * full` bytes and `out` writable for `full` bytes; `full` is a multiple of 16.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn area3_row_neon(r0: *const u8, r1: *const u8, r2: *const u8, out: *mut u8, full: usize) {
    use core::arch::aarch64::*;
    let mut x = 0;
    while x < full {
        // SAFETY: in bounds per the function contract (x + 16 <= full); `prfm` is a hint that never faults, even past the
        // end of a row.
        unsafe {
            // The A55's prefetcher does not follow three interleaved row streams (measured: 0.28 GB/s without this hint,
            // against 2.8 GB/s memcpy); ask for each row ~10 cache lines ahead.
            core::arch::asm!(
                "prfm pldl1strm, [{a}]",
                "prfm pldl1strm, [{b}]",
                "prfm pldl1strm, [{c}]",
                a = in(reg) r0.wrapping_add(3 * x + PREFETCH_BYTES),
                b = in(reg) r1.wrapping_add(3 * x + PREFETCH_BYTES),
                c = in(reg) r2.wrapping_add(3 * x + PREFETCH_BYTES),
                options(nostack, readonly, preserves_flags)
            );
            let a = vld3q_u8(r0.add(3 * x));
            let b = vld3q_u8(r1.add(3 * x));
            let c = vld3q_u8(r2.add(3 * x));
            let mut lo = vaddl_u8(vget_low_u8(a.0), vget_low_u8(a.1));
            lo = vaddw_u8(lo, vget_low_u8(a.2));
            lo = vaddw_u8(lo, vget_low_u8(b.0));
            lo = vaddw_u8(lo, vget_low_u8(b.1));
            lo = vaddw_u8(lo, vget_low_u8(b.2));
            lo = vaddw_u8(lo, vget_low_u8(c.0));
            lo = vaddw_u8(lo, vget_low_u8(c.1));
            lo = vaddw_u8(lo, vget_low_u8(c.2));
            let mut hi = vaddl_high_u8(a.0, a.1);
            hi = vaddw_high_u8(hi, a.2);
            hi = vaddw_high_u8(hi, b.0);
            hi = vaddw_high_u8(hi, b.1);
            hi = vaddw_high_u8(hi, b.2);
            hi = vaddw_high_u8(hi, c.0);
            hi = vaddw_high_u8(hi, c.1);
            hi = vaddw_high_u8(hi, c.2);
            vst1q_u8(out.add(x), vcombine_u8(div9_round(lo), div9_round(hi)));
        }
        x += 16;
    }
}

#[cfg(test)]
pub(super) mod tests {
    #[test]
    fn the_multiply_shift_division_is_exact_over_the_whole_range() {
        for v in 0u32..=2295 {
            assert_eq!(((v + 4) * 7282) >> 16, (v + 4) / 9, "v = {v}");
        }
    }
}
