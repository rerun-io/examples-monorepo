//! Float classification shared by the kernels and their device regression probe.
use cubecl::prelude::*;

/// Exponent bits classify NaN and infinity without float arithmetic.
/// CubeCL can fold `value * 0.0 == 0.0` to true even for nonfinite values.
#[cube]
pub(crate) fn is_finite(value: f32) -> bool {
    (u32::reinterpret(value) & 0x7f800000u32) != 0x7f800000u32
}
