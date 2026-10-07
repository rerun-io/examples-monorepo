//! Historical published NeRF checkpoint arithmetic.
use image::DynamicImage;

/// Published-convention PSNR with NumPy-compatible f32 arithmetic.
pub fn psnr(rendered: &[f32], gt: &[f32]) -> f64 {
    assert_eq!(rendered.len(), gt.len());
    assert!(!rendered.is_empty());
    let errors: Vec<f32> = rendered
        .iter()
        .zip(gt)
        .map(|(a, b)| (a - b) * (a - b))
        .collect();
    let mse = f64::from(numpy_sum(&errors)) / errors.len() as f64;
    if mse < 1e-10 {
        100.0
    } else {
        10.0 * mse.recip().log10()
    }
}

// NumPy's contiguous float pairwise reduction uses eight accumulators and
// 128-element leaves. Order matters for the checkpoint's 1e-6 dB tolerance.
// Reference: numpy/_core/src/umath/loops_utils.h.src, FLOAT_pairwise_sum.
fn numpy_sum(values: &[f32]) -> f32 {
    if values.len() < 8 {
        return values.iter().fold(-0.0, |sum, value| sum + value);
    }
    if values.len() > 128 {
        let midpoint = (values.len() / 2) & !7;
        return numpy_sum(&values[..midpoint]) + numpy_sum(&values[midpoint..]);
    }
    let mut lanes: [f32; 8] = values[..8].try_into().expect("eight values");
    let end = values.len() & !7;
    for chunk in values[8..end].as_chunks::<8>().0 {
        for (sum, value) in lanes.iter_mut().zip(chunk) {
            *sum += value;
        }
    }
    let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    values[end..].iter().fold(sum, |sum, value| sum + value)
}

/// Valid-window SSIM matching the published Python checkpoint evaluator.
pub fn ssim(a: &[f32], b: &[f32], width: usize, height: usize) -> f64 {
    assert_eq!(a.len(), width * height * 3);
    assert_eq!(a.len(), b.len());
    assert!(width >= 11 && height >= 11);
    let mut kernel: [f64; 11] =
        std::array::from_fn(|i| (-0.5 * ((i as f64 - 5.0) / 1.5).powi(2)).exp());
    let total: f64 = kernel.iter().sum();
    for weight in &mut kernel {
        *weight /= total;
    }
    let blur = |values: &[f32]| {
        let row = (width - 10) * 3;
        let mut horizontal = vec![0.0; height * row];
        for y in 0..height {
            for (tap, weight) in kernel.iter().enumerate() {
                for x in 0..row {
                    horizontal[y * row + x] +=
                        f64::from(values[y * width * 3 + tap * 3 + x]) * weight;
                }
            }
        }
        let mut output = vec![0.0; (height - 10) * row];
        for (tap, weight) in kernel.iter().enumerate() {
            for (i, result) in output.iter_mut().enumerate() {
                *result += horizontal[tap * row + i] * weight;
            }
        }
        output
    };
    let mu_a = blur(a);
    let mu_b = blur(b);
    // Python multiplies in f32 before the blur promotes each sample to f64.
    let second_a = blur(&a.iter().map(|x| x * x).collect::<Vec<_>>());
    let second_b = blur(&b.iter().map(|x| x * x).collect::<Vec<_>>());
    let cross = blur(&a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<_>>());
    let epsilon = f64::from(f32::EPSILON).powi(2);
    let sum: f64 = (0..mu_a.len())
        .map(|i| {
            let aa = mu_a[i] * mu_a[i];
            let bb = mu_b[i] * mu_b[i];
            let ab = mu_a[i] * mu_b[i];
            let var_a = (second_a[i] - aa).max(epsilon);
            let var_b = (second_b[i] - bb).max(epsilon);
            let raw_covariance = cross[i] - ab;
            let covariance =
                raw_covariance.signum() * raw_covariance.abs().min((var_a * var_b).sqrt());
            ((2.0 * ab + 0.0001) * (2.0 * covariance + 0.0009))
                / ((aa + bb + 0.0001) * (var_a + var_b + 0.0009))
        })
        .sum();
    sum / mu_a.len() as f64
}

/// Historical white composition: f32 operations, truncation to bytes, then
/// conversion back to f32. Alpha-free images bypass the composition.
pub fn rgb(image: &DynamicImage) -> Vec<f32> {
    if !image.color().has_alpha() {
        return image
            .to_rgb8()
            .into_raw()
            .into_iter()
            .map(|c| f32::from(c) / 255.0)
            .collect();
    }
    image
        .to_rgba8()
        .pixels()
        .flat_map(|p| {
            let alpha = f32::from(p[3]) / 255.0;
            [p[0], p[1], p[2]].map(|c| {
                let composited = f32::from(c) / 255.0 * alpha + (1.0 - alpha);
                f32::from((composited * 255.0).clamp(0.0, 255.0) as u8) / 255.0
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_identity_and_known_error() {
        assert_eq!(psnr(&[0.25; 768], &[0.25; 768]), 100.0);
        assert!((psnr(&[0.5; 768], &[0.0; 768]) - 6.020599913279624).abs() < 1e-12);
    }

    #[test]
    fn valid_window_ssim_identity_known_constant_and_symmetry() {
        let a = vec![0.0; 16 * 16 * 3];
        let b = vec![1.0; a.len()];
        assert!((ssim(&a, &a, 16, 16) - 1.0).abs() < 1e-9);
        assert!((ssim(&a, &b, 16, 16) - 0.0001 / 1.0001).abs() < 1e-10);
        assert_eq!(ssim(&a, &b, 16, 16), ssim(&b, &a, 16, 16));
    }
}
