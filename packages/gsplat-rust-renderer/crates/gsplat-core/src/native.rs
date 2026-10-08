//! The native GaussianSplats3D upload boundary, shared by viewer and parity harness.
use crate::Splats;
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
    items.get(i).or(items.last()).copied().unwrap_or(default)
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
