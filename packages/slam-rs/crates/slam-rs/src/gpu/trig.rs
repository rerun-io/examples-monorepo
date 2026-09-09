//! Small-angle trigonometry for the SE(2) translation factors.
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

/// Taylor remainder is below 6e-13 on this interval, below f32 rounding.
#[cube]
pub(crate) fn cos(theta: f32) -> f32 {
    if f32::abs(theta) <= 0.5f32 {
        let square = theta * theta;
        let polynomial = -0.5f32
            + square
                * (1.0f32 / 24.0f32
                    + square
                        * (-1.0f32 / 720.0f32
                            + square * (1.0f32 / 40320.0f32 - square * (1.0f32 / 3628800.0f32))));
        1.0f32 + square * polynomial
    } else {
        f32::cos(theta)
    }
}
