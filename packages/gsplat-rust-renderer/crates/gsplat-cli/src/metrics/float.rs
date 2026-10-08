//! Symmetric, unclipped float-render comparison. Inputs are premultiplied RGBA.
use crate::{Error, Evaluator, Metrics};
use burn::tensor::{Tensor, TensorData};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RenderMetrics {
    pub rgb: Metrics,
    pub alpha_psnr: f64,
    pub white_psnr: f64,
}
impl RenderMetrics {
    /// The quality gate must see alpha and both backgrounds.
    pub fn minimum_psnr(&self) -> f64 {
        self.rgb.psnr.min(self.alpha_psnr).min(self.white_psnr)
    }
}
impl Evaluator {
    /// Compare in-memory renderer outputs without clipping or PNG conversion.
    /// PSNR calls Brush's public float helper; SSIM ports its zero padding,
    /// Gaussian taps, nonnegative variances, and [-1,1] output clamp.
    pub async fn evaluate_renders(
        &self,
        a: &[f32],
        b: &[f32],
        width: u32,
        height: u32,
    ) -> Result<RenderMetrics, Error> {
        let count = width as usize * height as usize;
        if width < 11
            || height < 11
            || a.len() != count * 4
            || b.len() != a.len()
            || !a.iter().chain(b).all(|v| v.is_finite())
        {
            anyhow::bail!("expected equal finite RGBA renders, at least 11x11");
        }
        let maps = [
            (|p: &[f32; 4]| [p[0], p[1], p[2]]) as fn(&[f32; 4]) -> [f32; 3],
            |p| [p[3]; 3],
            |p| [p[0] + 1.0 - p[3], p[1] + 1.0 - p[3], p[2] + 1.0 - p[3]],
        ];
        let mut psnr = [0.0; 3];
        let mut ssim = 0.0;
        for (i, map) in maps.into_iter().enumerate() {
            let ra: Vec<_> = a.as_chunks::<4>().0.iter().flat_map(map).collect();
            let rb: Vec<_> = b.as_chunks::<4>().0.iter().flat_map(map).collect();
            if i == 0 {
                ssim = float_ssim(&ra, &rb, width as usize, height as usize);
            }
            let tensor = |data| {
                Tensor::<3>::from_data(
                    TensorData::new(data, [height as usize, width as usize, 3]),
                    &self.device,
                )
            };
            psnr[i] = brush_loss::psnr(tensor(ra), tensor(rb))
                .into_scalar_async::<f32>()
                .await?;
        }
        let [psnr, alpha_psnr, white_psnr] = psnr;
        Ok(RenderMetrics {
            rgb: Metrics {
                psnr: psnr.into(),
                ssim,
                lpips: None,
            },
            alpha_psnr: alpha_psnr.into(),
            white_psnr: white_psnr.into(),
        })
    }
}

fn float_ssim(a: &[f32], b: &[f32], width: usize, height: usize) -> f64 {
    let mut weights: [f32; 11] =
        std::array::from_fn(|i| (-((i as f32 - 5.0).powi(2)) / (2.0 * 1.5 * 1.5)).exp());
    let total: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }
    let mut horizontal = vec![[0.0f32; 5]; a.len()];
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let mut sums = [0.0; 5];
                for (tap, w) in weights.iter().enumerate() {
                    let xx = x as isize + tap as isize - 5;
                    if !(0..width as isize).contains(&xx) {
                        continue;
                    }
                    let i = (y * width + xx as usize) * 3 + c;
                    let av = a[i];
                    let bv = b[i];
                    for (sum, v) in sums.iter_mut().zip([av, av * av, bv, bv * bv, av * bv]) {
                        *sum += w * v;
                    }
                }
                horizontal[(y * width + x) * 3 + c] = sums;
            }
        }
    }
    let mut total = 0.0f64;
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let mut sums = [0.0f32; 5];
                for (tap, w) in weights.iter().enumerate() {
                    let yy = y as isize + tap as isize - 5;
                    if !(0..height as isize).contains(&yy) {
                        continue;
                    }
                    for (sum, v) in sums
                        .iter_mut()
                        .zip(horizontal[(yy as usize * width + x) * 3 + c])
                    {
                        *sum += w * v;
                    }
                }
                let [ma, aa, mb, bb, ab] = sums;
                let raw = ((2.0 * ma * mb + 0.0001) * (2.0 * (ab - ma * mb) + 0.0009))
                    / ((ma * ma + mb * mb + 0.0001)
                        * ((aa - ma * ma).max(0.0) + (bb - mb * mb).max(0.0) + 0.0009));
                total += f64::from(raw.clamp(-1.0, 1.0));
            }
        }
    }
    total / a.len() as f64
}
