//! CPU software rasterizer for correctness testing.
//!
//! Renders a [`RenderGaussianCloud`] from a given camera viewpoint by
//! projecting each visible Gaussian to 2D and performing per-pixel alpha
//! blending.  The math matches `gaussian_splat.wgsl` (the GPU CPU-fallback
//! shader) exactly: Mahalanobis distance, 3-sigma cutoff, back-to-front
//! blending with early termination.
//!
//! **This is NOT fast** — it's O(pixels × visible_splats).  It exists purely
//! for correctness testing against ground truth images.  For interactive
//! rendering, use the GPU pipeline via the Rerun viewer.

use super::culling::rebuild_visible_indices;
use super::projection::{build_prepared_splat, project_gaussian_to_ndc};
use super::sh::evaluate_sh_rgb;
use super::types::{CameraApproximation, PreparedSplat, RenderGaussianCloud, SortedSplatIndex};

/// Output of the CPU software rasterizer.
pub struct RenderOutput {
    /// RGBA pixels in row-major order (top-left origin), float32 per channel.
    /// Each pixel is `[R, G, B, A]` with values in `[0, 1]`.
    pub pixels: Vec<[f32; 4]>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl RenderOutput {
    /// Composite over a background color and convert to 8-bit RGB for PNG saving.
    ///
    /// Returns a flat byte array of length `width * height * 3` in row-major RGB order.
    pub fn to_rgb8(&self, background: [f32; 3]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.width as usize * self.height as usize * 3);
        for pixel in &self.pixels {
            let alpha = pixel[3];
            let r = pixel[0] + background[0] * (1.0 - alpha);
            let g = pixel[1] + background[1] * (1.0 - alpha);
            let b = pixel[2] + background[2] * (1.0 - alpha);
            bytes.push((r.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            bytes.push((g.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            bytes.push((b.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        bytes
    }
}

/// Render a Gaussian cloud from a camera viewpoint using CPU software rasterization.
///
/// # Algorithm
///
/// 1. Run frustum culling + depth sort to get visible splats (back-to-front).
/// 2. Project each visible splat to 2D (center, covariance, color, opacity).
/// 3. For each pixel, iterate back-to-front over splats:
///    - Compute Mahalanobis distance from the pixel to the splat center.
///    - Skip if beyond 3σ (mahalanobis > 9.0).
///    - Accumulate `color += splat.color * alpha * transmittance`.
///    - Update `transmittance *= (1 - alpha)`.
///    - Early terminate if transmittance < 1e-4 (pixel is fully opaque).
///
/// The output has premultiplied alpha: `pixel.rgb` already contains the
/// accumulated color, and `pixel.a = 1 - transmittance`.  To get the final
/// image, composite over a background using [`RenderOutput::to_rgb8`].
pub fn software_render(
    cloud: &RenderGaussianCloud,
    camera: &CameraApproximation,
    _background: [f32; 3],
) -> RenderOutput {
    let width = camera.viewport_size_px.x.max(1.0) as u32;
    let height = camera.viewport_size_px.y.max(1.0) as u32;
    let num_pixels = (width as usize) * (height as usize);

    // Step 1: Cull and sort (back-to-front).
    let mut visible: Vec<SortedSplatIndex> = Vec::new();
    rebuild_visible_indices(&mut visible, cloud, camera);

    // Step 2: Project each visible splat.
    let prepared: Vec<PreparedSplat> = visible
        .iter()
        .filter_map(|sorted| {
            let idx = sorted.splat_index as usize;
            let mean = cloud.means_world[idx];
            let quat = cloud.quats[idx];
            let scale = cloud.scales[idx];
            let opacity = cloud.opacities[idx];

            let projected = project_gaussian_to_ndc(camera, mean, quat, scale, opacity)?;

            // Evaluate SH for view-dependent color if available.
            let view_direction = (mean - camera.world_position).normalize_or_zero();
            let color = if let Some(sh) = &cloud.sh_coeffs {
                let offset = idx * sh.coeffs_per_channel * 3;
                let end = offset + sh.coeffs_per_channel * 3;
                if end <= sh.coefficients.len() {
                    evaluate_sh_rgb(
                        &sh.coefficients[offset..end],
                        sh.coeffs_per_channel,
                        view_direction,
                    )
                    .unwrap_or(cloud.colors_dc[idx])
                } else {
                    cloud.colors_dc[idx]
                }
            } else {
                cloud.colors_dc[idx]
            };

            Some(build_prepared_splat(projected, color, opacity))
        })
        .collect();

    // Step 3: Rasterize — per-pixel back-to-front alpha blending.
    let mut pixels = vec![[0.0_f32; 4]; num_pixels];

    // Pre-compute bounding boxes in pixel coordinates for each prepared splat
    // so we can skip splats that don't overlap a given pixel.
    let half_vp = glam::Vec2::new(width as f32 * 0.5, height as f32 * 0.5);

    // Splat-centric: iterate splats, accumulate into pixels that overlap.
    // This is more cache-friendly for the splat data and matches the GPU approach.
    //
    // We accumulate premultiplied alpha back-to-front:
    //   color_accum += splat_color * splat_alpha * transmittance
    //   transmittance *= (1 - splat_alpha)
    //
    // To do this splat-centric (back-to-front), we iterate in the order
    // provided by `prepared` (which comes from `visible`, already sorted
    // back-to-front by `rebuild_visible_indices`).

    // We need per-pixel transmittance tracking.
    let mut transmittance = vec![1.0_f32; num_pixels];

    for splat in &prepared {
        // Convert NDC center to pixel coordinates.
        let cx_px = (splat.center_ndc.x * 0.5 + 0.5) * width as f32;
        let cy_px = (splat.center_ndc.y * -0.5 + 0.5) * height as f32;

        // Compute bounding box in pixels from the NDC radius.
        let radius_px_x = splat.radius_ndc * half_vp.x;
        let radius_px_y = splat.radius_ndc * half_vp.y;

        let x_min = ((cx_px - radius_px_x).floor() as i32).max(0) as u32;
        let x_max = ((cx_px + radius_px_x).ceil() as i32).min(width as i32) as u32;
        let y_min = ((cy_px - radius_px_y).floor() as i32).max(0) as u32;
        let y_max = ((cy_px + radius_px_y).ceil() as i32).min(height as i32) as u32;

        if x_min >= x_max || y_min >= y_max {
            continue;
        }

        let inv_cov_xx = splat.inv_cov_ndc_xx_xy_yy[0];
        let inv_cov_xy = splat.inv_cov_ndc_xx_xy_yy[1];
        let inv_cov_yy = splat.inv_cov_ndc_xx_xy_yy[2];

        for y in y_min..y_max {
            // Pixel center in NDC.
            let py_ndc = -((y as f32 + 0.5) / height as f32 * 2.0 - 1.0);
            let dy = py_ndc - splat.center_ndc.y;

            let row_offset = y as usize * width as usize;

            for x in x_min..x_max {
                let pixel_idx = row_offset + x as usize;

                // Early skip if this pixel is already fully opaque.
                if transmittance[pixel_idx] < 1e-4 {
                    continue;
                }

                let px_ndc = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
                let dx = px_ndc - splat.center_ndc.x;

                // Mahalanobis distance: Δᵀ * Σ⁻¹ * Δ
                let mahalanobis =
                    dx * dx * inv_cov_xx + 2.0 * dx * dy * inv_cov_xy + dy * dy * inv_cov_yy;

                if mahalanobis > 9.0 {
                    continue;
                }

                let alpha = (-0.5 * mahalanobis).exp() * splat.opacity;
                if alpha < 1.0 / 255.0 {
                    continue;
                }

                let t = transmittance[pixel_idx];
                let pixel = &mut pixels[pixel_idx];
                pixel[0] += splat.color_rgb[0] * alpha * t;
                pixel[1] += splat.color_rgb[1] * alpha * t;
                pixel[2] += splat.color_rgb[2] * alpha * t;
                pixel[3] += alpha * t;
                transmittance[pixel_idx] = t * (1.0 - alpha);
            }
        }
    }

    RenderOutput {
        pixels,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec2, Vec3};

    use super::*;
    use crate::gsplat_core::camera::make_camera_approximation;
    use crate::gsplat_core::types::RenderGaussianCloud;

    #[test]
    fn empty_cloud_renders_transparent() {
        let cloud = RenderGaussianCloud::from_raw(vec![], vec![], vec![], vec![], vec![], None);
        let camera = make_camera_approximation(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0_f32.to_radians(),
            Vec2::new(64.0, 64.0),
            0.01,
        );
        let output = software_render(&cloud, &camera, [0.0, 0.0, 0.0]);
        assert_eq!(output.width, 64);
        assert_eq!(output.height, 64);
        // All pixels should be fully transparent.
        for pixel in &output.pixels {
            assert_eq!(*pixel, [0.0, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn single_splat_produces_nonzero_alpha_at_center() {
        // Place a single white Gaussian at the origin, camera looking at it.
        let cloud = RenderGaussianCloud::from_raw(
            vec![Vec3::ZERO],
            vec![Quat::IDENTITY],
            vec![Vec3::splat(0.1)],
            vec![0.99],
            vec![[1.0, 1.0, 1.0]],
            None,
        );
        let camera = make_camera_approximation(
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0_f32.to_radians(),
            Vec2::new(64.0, 64.0),
            0.01,
        );
        let output = software_render(&cloud, &camera, [0.0, 0.0, 0.0]);

        // The center pixel should have some alpha contribution.
        let center_idx = 32 * 64 + 32;
        let pixel = output.pixels[center_idx];
        assert!(
            pixel[3] > 0.0,
            "Center pixel should have nonzero alpha, got {pixel:?}"
        );
    }

    #[test]
    fn to_rgb8_composites_over_background() {
        let output = RenderOutput {
            pixels: vec![[0.5, 0.0, 0.0, 0.5]], // half-transparent red
            width: 1,
            height: 1,
        };
        let bytes = output.to_rgb8([1.0, 1.0, 1.0]); // white background
        // Expected: 0.5 (red) + 1.0 * 0.5 (background) = 1.0 → 255
        assert_eq!(bytes[0], 255);
        // Green: 0.0 + 1.0 * 0.5 = 0.5 → 128
        assert_eq!(bytes[1], 128);
        // Blue: 0.0 + 1.0 * 0.5 = 0.5 → 128
        assert_eq!(bytes[2], 128);
    }
}
