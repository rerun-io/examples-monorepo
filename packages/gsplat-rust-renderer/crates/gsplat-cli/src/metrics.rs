//! Offline image metrics. Brush is the default; `Published` preserves the
//! white-background NeRF checkpoint guard's historical Python arithmetic.

mod float;
pub mod published;
pub use crate::Provenance;
use crate::{Error, gpu};
pub use float::RenderMetrics;
pub use published::rgb as published_rgb;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use brush_dataset::scene::view_to_packed_data;
use brush_loss::{ImageLossConfig, image_loss_eval, psnr_from_mse, unpack_gt_rgb};
use brush_render::AlphaMode;
use burn::tensor::{Device, Int, Tensor, TensorData};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Convention {
    #[default]
    Brush,
    Published,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub psnr: f64,
    pub ssim: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lpips: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ViewMetrics {
    pub name: String,
    #[serde(flatten)]
    pub metrics: Metrics,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub views: Vec<ViewMetrics>,
    pub mean: Metrics,
    pub convention: Convention,
    pub provenance: Provenance,
}

/// Holds a device and (when requested) one reusable VGG model.
pub struct Evaluator {
    device: Device,
    lpips: Option<lpips::LpipsModel>,
}

impl Evaluator {
    pub fn new(with_lpips: bool) -> Self {
        let device = Device::default();
        let lpips = with_lpips.then(|| lpips::load_vgg_lpips(&device));
        Self { device, lpips }
    }

    /// Score equal-size images. Brush renders are already on black; their
    /// alpha channel is ignored. Brush GT is byte-premultiplied by Brush itself.
    pub async fn evaluate_pair(
        &self,
        rendered: &DynamicImage,
        gt: &DynamicImage,
        convention: Convention,
    ) -> Result<Metrics, Error> {
        let (w, h) = (rendered.width(), rendered.height());
        if (w, h) != (gt.width(), gt.height()) || w < 11 || h < 11 {
            return Err(Error::Invalid(
                "dimensions must match and be at least 11x11".into(),
            ));
        }
        let shape = [h as usize, w as usize, 3];
        let rgb = rendered.to_rgb32f().into_raw();
        if !rgb.iter().all(|v| v.is_finite()) {
            return Err(Error::Invalid("render contains non-finite channels".into()));
        }
        let (psnr, ssim, lpips_inputs) = match convention {
            Convention::Brush => {
                let render = (Tensor::<3>::from_data(TensorData::new(rgb, shape), &self.device)
                    * 255.0)
                    .round()
                    / 255.0;
                let (packed, _) = view_to_packed_data(gt.clone(), AlphaMode::Transparent);
                let packed: Tensor<2, Int> = Tensor::from_data(packed, &self.device);
                let cfg = |l1_weight, ssim_weight| ImageLossConfig {
                    l1_weight,
                    ssim_weight,
                    composite_bg: None,
                    mask: false,
                    alpha_weight: 0.0,
                };
                let mse = image_loss_eval(render.clone(), packed.clone(), cfg(1.0, 0.0))
                    .powi_scalar(2)
                    .mean();
                let psnr = psnr_from_mse(mse)
                    .into_scalar_async::<f32>()
                    .await
                    .map_err(gpu)?;
                let ssim = image_loss_eval(render.clone(), packed.clone(), cfg(0.0, 1.0))
                    .mean()
                    .into_scalar_async::<f32>()
                    .await
                    .map_err(gpu)?;
                let inputs = self
                    .lpips
                    .as_ref()
                    .map(|_| (render, unpack_gt_rgb(packed, None)));
                (f64::from(psnr), f64::from(ssim), inputs)
            }
            Convention::Published => {
                let a = published_rgb(rendered);
                let b = published_rgb(gt);
                let psnr = published::psnr(&a, &b);
                let ssim = published::ssim(&a, &b, w as usize, h as usize);
                let inputs = self.lpips.as_ref().map(|_| {
                    (
                        Tensor::<3>::from_data(TensorData::new(a, shape), &self.device),
                        Tensor::<3>::from_data(TensorData::new(b, shape), &self.device),
                    )
                });
                (psnr, ssim, inputs)
            }
        };
        let lpips = if let (Some(model), Some((a, b))) = (&self.lpips, lpips_inputs) {
            Some(f64::from(
                model
                    .lpips(a.unsqueeze::<4>(), b.unsqueeze::<4>())
                    .into_scalar_async::<f32>()
                    .await
                    .map_err(gpu)?,
            ))
        } else {
            None
        };
        Ok(Metrics { psnr, ssim, lpips })
    }
}

/// Recursively pair PNGs by identical relative path sets; reject omissions and
/// empty sets before scoring any image. Symlinks are not traversed.
pub fn pair_directories(render: &Path, gt: &Path) -> Result<Vec<PathBuf>, Error> {
    fn collect(root: &Path, current: &Path, names: &mut BTreeSet<PathBuf>) -> Result<(), Error> {
        for entry in std::fs::read_dir(current)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            let path = entry.path();
            if kind.is_dir() {
                collect(root, &path, names)?;
            } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "png") {
                names.insert(path.strip_prefix(root).expect("descendant").to_owned());
            }
        }
        Ok(())
    }
    let mut renders = BTreeSet::new();
    let mut truths = BTreeSet::new();
    collect(render, render, &mut renders)?;
    collect(gt, gt, &mut truths)?;
    if renders.is_empty() || renders != truths {
        return Err(Error::Invalid(format!(
            "PNG paths differ or are empty: render-only {:?}, GT-only {:?}",
            renders.difference(&truths).collect::<Vec<_>>(),
            truths.difference(&renders).collect::<Vec<_>>()
        )));
    }
    Ok(renders.into_iter().collect())
}

pub async fn evaluate_directories(
    render: &Path,
    gt: &Path,
    convention: Convention,
    lpips: bool,
) -> Result<Evaluation, Error> {
    let paths = pair_directories(render, gt)?;
    let evaluator = Evaluator::new(lpips);
    let mut views = Vec::with_capacity(paths.len());
    for path in paths {
        let metrics = evaluator
            .evaluate_pair(
                &image::open(render.join(&path))?,
                &image::open(gt.join(&path))?,
                convention,
            )
            .await?;
        views.push(ViewMetrics {
            name: path.to_string_lossy().replace('\\', "/"),
            metrics,
        });
    }
    let mean = mean(&views);
    Ok(Evaluation {
        views,
        mean,
        convention,
        provenance: Provenance::default(),
    })
}

pub fn mean(views: &[ViewMetrics]) -> Metrics {
    assert!(!views.is_empty(), "empty metric set");
    let n = views.len() as f64;
    Metrics {
        psnr: views.iter().map(|v| v.metrics.psnr).sum::<f64>() / n,
        ssim: views.iter().map(|v| v.metrics.ssim).sum::<f64>() / n,
        lpips: views.iter().all(|v| v.metrics.lpips.is_some()).then(|| {
            views
                .iter()
                .map(|v| v.metrics.lpips.expect("requested LPIPS"))
                .sum::<f64>()
                / n
        }),
    }
}
