//! Shared synthetic inputs for CPU/GPU contracts.
use kornia_image::Image;

/// Twelve plane waves between 16 and 56 pixels, in fixed pseudo-random
/// directions and phases: band-limited, so a shift really does survive down a
/// pyramid and every patch's `H_se2` is well conditioned.
#[allow(
    dead_code,
    reason = "used by gpu_kernels; other binaries compile a subset"
)]
pub fn texture(x: f64, y: f64) -> f64 {
    const WAVES: [(f64, f64, f64); 12] = [
        (16.0, 0.031, 0.11),
        (19.0, 0.187, 0.37),
        (23.0, 0.311, 0.63),
        (27.0, 0.451, 0.05),
        (31.0, 0.077, 0.81),
        (35.0, 0.229, 0.29),
        (39.0, 0.383, 0.55),
        (43.0, 0.497, 0.73),
        (47.0, 0.143, 0.19),
        (51.0, 0.271, 0.91),
        (54.0, 0.419, 0.43),
        (56.0, 0.353, 0.67),
    ];
    let mut total: f64 = 0.0;
    for (wavelength, direction, phase) in WAVES {
        let angle: f64 = std::f64::consts::TAU * direction;
        let projection: f64 = x * angle.cos() + y * angle.sin();
        total += (std::f64::consts::TAU * (projection / wavelength + phase)).sin();
    }
    total / WAVES.len() as f64
}

/// [`texture`] rendered into a `u16` image, shifted by `(dx, dy)`.
#[allow(
    dead_code,
    reason = "used by gpu_kernels; other binaries compile a subset"
)]
pub fn textured_image(width: usize, height: usize, dx: f32, dy: f32) -> Image<u16, 1> {
    let mut image: Image<u16, 1> =
        Image::from_size_val(kornia_image::ImageSize { width, height }, 0u16)
            .expect("a valid image geometry");
    for y in 0..height {
        for x in 0..width {
            let value: f64 = texture(x as f64 - f64::from(dx), y as f64 - f64::from(dy));
            let scaled: f64 = (value * 0.4 + 0.5) * 65535.0;
            image
                .set_pixel(x, y, 0, scaled.clamp(0.0, 65535.0) as u16)
                .unwrap();
        }
    }
    image
}

/// A textured 8-bit field with fine detail, so FAST has plenty to find: the
/// smooth plane-wave texture above gives almost no corners.
///
/// One LCG plus a `sin`/`cos` wave, so it is the same field on every machine.
#[allow(
    dead_code,
    reason = "used by fast_model; other binaries compile a subset"
)]
pub fn cornered_bytes(width: usize, height: usize) -> Vec<u8> {
    let mut out: Vec<u8> = vec![0u8; width * height];
    let mut state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let wave: i32 = ((x as f64 / 11.0).sin() * 60.0 + (y as f64 / 7.0).cos() * 50.0) as i32;
            let noise: i32 = (state >> 24) as i32 / 4;
            out[y * width + x] = (128 + wave + noise).clamp(0, 255) as u8;
        }
    }
    out
}

/// [`cornered_bytes`] as a `u16` image, the byte in the high half.
///
/// The detector reads `pixel >> 8`, so this is the same field the CPU sweep and
/// the GPU score kernel see.
#[allow(
    dead_code,
    reason = "used by gpu_detect; other binaries compile a subset"
)]
pub fn cornered_image(width: usize, height: usize) -> Image<u16, 1> {
    let bytes: Vec<u8> = cornered_bytes(width, height);
    let mut image: Image<u16, 1> =
        Image::from_size_val(kornia_image::ImageSize { width, height }, 0u16)
            .expect("a valid image geometry");
    for y in 0..height {
        for x in 0..width {
            image
                .set_pixel(x, y, 0, u16::from(bytes[y * width + x]) << 8)
                .unwrap();
        }
    }
    image
}
