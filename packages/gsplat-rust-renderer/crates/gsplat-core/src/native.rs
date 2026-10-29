//! The native `GaussianSplats3D` and `Hexels3D` upload boundary, shared by viewer and parity harness.
use crate::{HexelVariant, Hexels, Splats};
use glam::Quat;
use half::f16;

const SH_C0: f32 = 0.282_094_8;

/// Borrowed native component arrays. Short optional arrays repeat their last item.
pub struct NativeSplats<'a> {
    pub centers: &'a [[f32; 3]],
    pub scales: &'a [[f32; 3]],
    pub quaternions: &'a [[f32; 4]],
    pub colors: &'a [u32],
    pub sh: &'a [[[f16; 3]; 15]],
    pub degree: u32,
}

/// Read an optional component, repeating its last item when the array is short.
pub fn at_or_last<T: Copy>(items: &[T], i: usize, default: T) -> T {
    items
        .get(i)
        .or_else(|| items.last())
        .copied()
        .unwrap_or(default)
}

impl NativeSplats<'_> {
    /// Native SH supports at most degree three; absent SH leaves only DC.
    pub fn coefficient_count(&self) -> usize {
        if self.sh.is_empty() {
            1
        } else {
            (self.degree.min(3) as usize + 1).pow(2)
        }
    }

    /// Undo the native activations without further quantization. Alpha endpoints
    /// use saturated finite logits so the core finite-input guard accepts them.
    pub fn to_core(&self) -> Splats {
        let coefficients = self.coefficient_count();
        let n = self.centers.len();
        let mut splats = Splats {
            transforms: Vec::with_capacity(n),
            raw_opacities: Vec::with_capacity(n),
            sh_coefficients: Vec::with_capacity(n * coefficients),
            sh_degree: coefficients.isqrt() as u32 - 1,
            min_scale: None,
        };
        for (i, center) in self.centers.iter().enumerate() {
            let scale = at_or_last(self.scales, i, [0.01; 3]);
            let q = at_or_last(self.quaternions, i, [0.0, 0.0, 0.0, 1.0]);
            let q = Quat::from_array(q);
            let q = if q.is_finite() && q.length_squared() > 1e-12 {
                q.normalize()
            } else {
                Quat::IDENTITY
            };
            splats.transforms.push([
                center[0],
                center[1],
                center[2],
                q.w,
                q.x,
                q.y,
                q.z,
                scale[0].max(1e-6).ln(),
                scale[1].max(1e-6).ln(),
                scale[2].max(1e-6).ln(),
            ]);
            let rgba = at_or_last(self.colors, i, u32::MAX).to_be_bytes();
            let alpha = rgba[3] as f32 / 255.0;
            splats
                .raw_opacities
                .push((alpha / (1.0 - alpha)).ln().clamp(-100.0, 100.0));
            splats.sh_coefficients.push(std::array::from_fn(|c| {
                (rgba[c] as f32 / 255.0 - 0.5) / SH_C0
            }));
            let rest = at_or_last(self.sh, i, [[f16::ZERO; 3]; 15]);
            for coefficient in &rest[..coefficients - 1] {
                splats.sh_coefficients.push(coefficient.map(f16::to_f32));
            }
        }
        splats
    }
}

/// Borrowed `Hexels3D` component arrays (f32, activated). Rotations are Rerun xyzw.
/// Short optional arrays repeat their last item; `view_linear` selects V2, `sh` V3.
pub struct NativeHexels<'a> {
    pub centers: &'a [[f32; 3]],
    pub quaternions: &'a [[f32; 4]],
    pub scales: &'a [[f32; 2]],
    pub opacities: &'a [f32],
    pub sharpness: &'a [f32],
    pub color_field: &'a [[f32; 9]],
    pub view_linear: &'a [[f32; 9]],
    pub sh: &'a [[f32; 45]],
}

impl NativeHexels<'_> {
    pub fn variant(&self) -> HexelVariant {
        if !self.sh.is_empty() {
            HexelVariant::V3
        } else if !self.view_linear.is_empty() {
            HexelVariant::V2
        } else {
            HexelVariant::V1
        }
    }

    /// Core rows in the hexel PLY order: centre, wxyz rotation, scales, opacity, sharpness,
    /// colour field, then the V2/V3 view term. Values pass through unchanged.
    pub fn to_core(&self) -> Hexels {
        let variant = self.variant();
        let mut params = Vec::with_capacity(self.centers.len() * variant.stride());
        for (i, center) in self.centers.iter().enumerate() {
            let [x, y, z, w] = at_or_last(self.quaternions, i, [0.0, 0.0, 0.0, 1.0]);
            params.extend_from_slice(center);
            params.extend_from_slice(&[w, x, y, z]);
            params.extend_from_slice(&at_or_last(self.scales, i, [0.01; 2]));
            params.push(at_or_last(self.opacities, i, 1.0));
            params.push(at_or_last(self.sharpness, i, 10.0));
            params.extend_from_slice(&at_or_last(
                self.color_field,
                i,
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ));
            match variant {
                HexelVariant::V2 => {
                    params.extend_from_slice(&at_or_last(self.view_linear, i, [0.0; 9]));
                }
                HexelVariant::V3 => params.extend_from_slice(&at_or_last(self.sh, i, [0.0; 45])),
                HexelVariant::V1 | HexelVariant::Image => {}
            }
        }
        Hexels { variant, params }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hexel_rotation_returns_to_core_wxyz_order() {
        // Rerun xyzw for a 90-degree turn about x; the core rotates t_v = R e_y onto +z.
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let hexels = NativeHexels {
            centers: &[[1.0, 2.0, 3.0]],
            quaternions: &[[h, 0.0, 0.0, h]],
            scales: &[[0.2, 0.4]],
            opacities: &[0.5],
            sharpness: &[7.0],
            color_field: &[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
            view_linear: &[],
            sh: &[],
        }
        .to_core();
        assert_eq!(hexels.variant, HexelVariant::V1);
        assert_eq!(
            hexels.params,
            [
                1.0, 2.0, 3.0, h, h, 0.0, 0.0, 0.2, 0.4, 0.5, 7.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                0.7, 0.8, 0.9
            ]
        );
        let q = &hexels.params[3..7];
        let rotation = glam::Quat::from_xyzw(q[1], q[2], q[3], q[0]);
        assert!((rotation * glam::Vec3::Y).abs_diff_eq(glam::Vec3::Z, 1e-6));
    }

    #[test]
    fn hexel_view_terms_select_the_variant_and_short_arrays_repeat() {
        let sh: Vec<[f32; 45]> = vec![std::array::from_fn(|i| i as f32)];
        let hexels = NativeHexels {
            centers: &[[0.0; 3], [1.0; 3]],
            quaternions: &[],
            scales: &[[0.3, 0.3]],
            opacities: &[],
            sharpness: &[],
            color_field: &[],
            view_linear: &[],
            sh: &sh,
        }
        .to_core();
        assert_eq!(hexels.variant, HexelVariant::V3);
        assert_eq!(hexels.len(), 2);
        let second = &hexels.params[65..];
        assert_eq!(
            &second[..11],
            &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.3, 0.3, 1.0, 10.0]
        );
        assert_eq!(second[20..], sh[0]);
        let linear = NativeHexels {
            centers: &[[0.0; 3]],
            quaternions: &[],
            scales: &[],
            opacities: &[],
            sharpness: &[],
            color_field: &[],
            view_linear: &[[2.0; 9]],
            sh: &[],
        };
        assert_eq!(linear.variant(), HexelVariant::V2);
        assert_eq!(linear.to_core().params.len(), 29);
    }
}
