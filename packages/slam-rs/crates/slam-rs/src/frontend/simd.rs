//! Four independent KLT points per vector. No horizontal reductions or FMA.
//!
//! NEON and SSE use the same IEEE add/multiply/divide sequence. Other targets
//! retain the scalar sequence so the CPU frontend stays portable.

use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub(crate) struct F32x4(pub [f32; 4]);

impl F32x4 {
    pub const ZERO: Self = Self([0.0; 4]);

    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self([value; 4])
    }

    #[inline]
    pub fn load(values: &[f32]) -> Self {
        Self([values[0], values[1], values[2], values[3]])
    }

    #[inline]
    pub fn store(self, values: &mut [f32]) {
        values[..4].copy_from_slice(&self.0);
    }

    #[inline]
    pub fn select(mask: [bool; 4], yes: Self, no: Self) -> Self {
        Self(std::array::from_fn(|lane| {
            if mask[lane] { yes.0[lane] } else { no.0[lane] }
        }))
    }
}

macro_rules! binary_op {
    ($trait:ident, $method:ident, $op:tt, $neon:ident, $sse:ident) => {
        impl $trait for F32x4 {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                #[cfg(target_arch = "aarch64")]
                {
                    use core::arch::aarch64::*;
                    // SAFETY: NEON is part of AArch64; loads/stores cover four f32s.
                    unsafe {
                        let mut out = [0.0; 4];
                        vst1q_f32(out.as_mut_ptr(), $neon(vld1q_f32(self.0.as_ptr()), vld1q_f32(rhs.0.as_ptr())));
                        Self(out)
                    }
                }
                #[cfg(target_arch = "x86_64")]
                {
                    use core::arch::x86_64::*;
                    // SAFETY: SSE2 is part of x86-64; unaligned loads/stores cover four f32s.
                    unsafe {
                        let mut out = [0.0; 4];
                        _mm_storeu_ps(out.as_mut_ptr(), $sse(_mm_loadu_ps(self.0.as_ptr()), _mm_loadu_ps(rhs.0.as_ptr())));
                        Self(out)
                    }
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
                Self(std::array::from_fn(|lane| self.0[lane] $op rhs.0[lane]))
            }
        }
    };
}

binary_op!(Add, add, +, vaddq_f32, _mm_add_ps);
binary_op!(Sub, sub, -, vsubq_f32, _mm_sub_ps);
binary_op!(Mul, mul, *, vmulq_f32, _mm_mul_ps);
binary_op!(Div, div, /, vdivq_f32, _mm_div_ps);
