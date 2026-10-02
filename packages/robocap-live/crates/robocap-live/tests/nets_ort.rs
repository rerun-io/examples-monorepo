//! ONNX Runtime backend vs the PyTorch golden files (FP32: tight tolerances). Runs with `--features ort` where `ORT_DYLIB_PATH`
//! names a `libonnxruntime.so` and the `.onnx` models exist; skips otherwise. Device: `ROBOCAP_LIVE_ORT_DEVICE` = `cpu` | `cuda`
//! | `auto` (default `auto`).
#![cfg(feature = "ort")]

use std::path::{Path, PathBuf};

use robocap_live::nets::golden::{Comparison, Golden, default_dir};
use robocap_live::nets::ort::{DETNET_ONNX, KEYNET_ONNX, OrtDevice, OrtNets, OrtConfig};
use robocap_live::nets::{HandNets, KeyNetRaw, NetFrame};

#[test]
fn ort_matches_pytorch_golden() -> Result<(), Box<dyn std::error::Error>> {
    let models: PathBuf =
        std::env::var_os("ROBOCAP_LIVE_MODELS").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"), PathBuf::from);
    let Some(dylib) = std::env::var_os("ORT_DYLIB_PATH").filter(|p| !p.is_empty()).map(PathBuf::from) else {
        eprintln!("SKIP ort_matches_pytorch_golden: ORT_DYLIB_PATH is not set (point it at libonnxruntime.so, see nets/ort.rs)");
        return Ok(());
    };
    for path in [&dylib, &models.join(DETNET_ONNX), &models.join(KEYNET_ONNX)] {
        if !path.exists() {
            eprintln!("SKIP ort_matches_pytorch_golden: {} not found", path.display());
            return Ok(());
        }
    }
    let device: OrtDevice = match std::env::var("ROBOCAP_LIVE_ORT_DEVICE").as_deref() {
        Ok("cpu") => OrtDevice::Cpu,
        Ok("cuda") => OrtDevice::Cuda(0),
        _ => OrtDevice::Auto,
    };
    let golden: Golden = Golden::load(std::env::var_os("ROBOCAP_LIVE_NETS_GOLDEN").map_or_else(default_dir, PathBuf::from))?;
    let mut nets: OrtNets = OrtNets::new(&models, &OrtConfig { device, dylib: Some(dylib), intra_threads: 0 })?;
    let frames: Vec<Vec<u8>> = golden.detnet_frames();
    let frame_refs: Vec<NetFrame<'_>> = frames.iter().map(|frame| NetFrame { pixels: frame, top: 0 }).collect();
    let detnet_out = nets.detnet(&frame_refs)?;
    let crops: Vec<&[f32]> = golden.keynet_crops.iter().map(Vec::as_slice).collect();
    let keynet_out: Vec<KeyNetRaw> = nets.keynet(&crops, &golden.keynet_keypoints)?;
    let c: Comparison = Comparison::of(&golden, &detnet_out, &keynet_out)?;
    eprintln!("{} vs PyTorch: {c:?}", nets.describe());
    assert!(c.detnet_centre_px_max < 0.01, "detnet centre {}", c.detnet_centre_px_max);
    assert!(c.detnet_presence_logit_max < 1e-3, "detnet presence {}", c.detnet_presence_logit_max);
    assert!(c.keynet_heatmap_max < 1e-4, "heatmaps {}", c.keynet_heatmap_max);
    assert!(c.keynet_distance_max < 1e-4, "distance {}", c.keynet_distance_max);
    assert!(c.keynet_keypoint_px_max < 0.01, "keypoints {}", c.keynet_keypoint_px_max);
    assert!(c.keynet_presence_logit_max < 1e-3, "presence {}", c.keynet_presence_logit_max);
    assert!(c.keynet_pinch_probability_max < 1e-4, "pinch {}", c.keynet_pinch_probability_max);
    Ok(())
}
