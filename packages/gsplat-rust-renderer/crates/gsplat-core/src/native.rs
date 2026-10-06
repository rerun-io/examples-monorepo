//! The native GaussianSplats3D upload boundary, shared by viewer and parity harness.
use crate::Splats;
use glam::Quat;
use half::f16;

pub const SH_C0: f32 = 0.282_094_8;

/// Borrowed native component arrays. Short optional arrays repeat their last item.
pub struct NativeSplats<'a> {
    pub centers: &'a [[f32; 3]],
    pub scales: &'a [[f32; 3]],
    pub quaternions: &'a [[f32; 4]],
    pub colors: &'a [u32],
    pub sh: &'a [[[f16; 3]; 15]],
    pub degree: u32,
}

impl NativeSplats<'_> {
    /// Undo the native activations without further quantization. Alpha endpoints
    /// use saturated finite logits so the core finite-input guard accepts them.
    pub fn to_core(&self) -> Result<Splats, crate::Error> {
        let degree = if self.sh.is_empty() {
            0
        } else {
            self.degree.min(3)
        };
        let n = self.centers.len();
        let coefficients = (degree as usize + 1).pow(2);
        let mut splats = Splats {
            transforms: Vec::with_capacity(n),
            raw_opacities: Vec::with_capacity(n),
            sh_coefficients: Vec::with_capacity(n * coefficients),
            sh_degree: degree,
            min_scale: None,
        };
        for (i, center) in self.centers.iter().enumerate() {
            let scale = self
                .scales
                .get(i)
                .or_else(|| self.scales.last())
                .copied()
                .unwrap_or([0.01; 3]);
            let q = self
                .quaternions
                .get(i)
                .or_else(|| self.quaternions.last())
                .copied()
                .unwrap_or([0.0, 0.0, 0.0, 1.0]);
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
            let rgba = self
                .colors
                .get(i)
                .or_else(|| self.colors.last())
                .copied()
                .unwrap_or(u32::MAX)
                .to_be_bytes();
            let alpha = rgba[3] as f32 / 255.0;
            splats
                .raw_opacities
                .push((alpha / (1.0 - alpha)).ln().clamp(-100.0, 100.0));
            splats.sh_coefficients.push(std::array::from_fn(|c| {
                (rgba[c] as f32 / 255.0 - 0.5) / SH_C0
            }));
            let rest = self.sh.get(i).or_else(|| self.sh.last());
            for k in 0..coefficients - 1 {
                splats
                    .sh_coefficients
                    .push(rest.map_or([0.0; 3], |sh| sh[k].map(f16::to_f32)));
            }
        }
        Ok(splats)
    }
}
