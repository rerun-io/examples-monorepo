//! Shared deterministic image fixtures.
#![allow(clippy::unwrap_used)]

use kornia_image::{Image, ImageSize};

pub(crate) fn zeros(width: usize, height: usize) -> Image<u16, 1> {
    Image::from_size_val(ImageSize { width, height }, 0).unwrap()
}

pub(crate) fn random_image(width: usize, height: usize, seed: u64) -> Image<u16, 1> {
    let mut image = zeros(width, height);
    let mut state = seed | 1;
    for pixel in image.as_slice_mut() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *pixel = (state >> 32) as u16;
    }
    image
}
