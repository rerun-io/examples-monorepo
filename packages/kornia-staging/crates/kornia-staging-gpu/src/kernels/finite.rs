//! Float classification shared by the kernels and their device regression probe.
// CubeCL generates undocumented expansion modules for inline device functions.
#![allow(missing_docs)]
use cubecl::prelude::*;

/// Exponent bits classify NaN and infinity without float arithmetic.
/// CubeCL can fold `value * 0.0 == 0.0` to true even for nonfinite values.
#[cube]
pub fn is_finite(value: f32) -> bool {
    (u32::reinterpret(value) & 0x7f800000u32) != 0x7f800000u32
}
