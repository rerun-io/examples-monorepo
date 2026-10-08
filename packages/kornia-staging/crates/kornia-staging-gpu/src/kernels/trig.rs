//! Small-angle sine for the SE(2) translation factors.
//! Division by theta amplifies native sine's measured 257-ULP error; native
//! cosine stays within 2 ULP and needs no polynomial. MIO14 GT ATE was 9.48 cm
//! with sin+cos polynomials and 9.72 cm with sine only (10.63 cm allowed).
use cubecl::prelude::*;

/// Preserve relative accuracy near zero, where the SE(2) update divides by theta.
#[cube]
pub(crate) fn sin(theta: f32) -> f32 {
    if f32::abs(theta) <= 0.5f32 {
        let square = theta * theta;
        let polynomial = -1.0f32 / 6.0f32
            + square
                * (1.0f32 / 120.0f32
                    + square * (-1.0f32 / 5040.0f32 + square * (1.0f32 / 362880.0f32)));
        theta + theta * square * polynomial
    } else {
        f32::sin(theta)
    }
}
