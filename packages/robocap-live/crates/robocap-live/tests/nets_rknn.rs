//! RKNN backend vs the PyTorch golden files. Runs where `librknnrt.so` and the `.rknn` models exist (Cap A); skips elsewhere.
//!
//! Models directory: `ROBOCAP_LIVE_MODELS` (default `packages/robocap-live/models`); library: `RKNN_LIB` (default
//! `/usr/lib/librknnrt.so`); golden files: `ROBOCAP_LIVE_NETS_GOLDEN` (default the crate's `tests/data/nets`).

use std::path::{Path, PathBuf};

use robocap_live::nets::golden::{Comparison, Golden, default_dir};
use robocap_live::nets::rknn::{DEFAULT_DETNET, DEFAULT_KEYNET, DEFAULT_LIBRARY, RknnNets};
use robocap_live::nets::{HandNets, KeyNetRaw, NetFrame};

fn models_dir() -> PathBuf {
    std::env::var_os("ROBOCAP_LIVE_MODELS").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"), PathBuf::from)
}

fn golden_dir() -> PathBuf {
    std::env::var_os("ROBOCAP_LIVE_NETS_GOLDEN").map_or_else(default_dir, PathBuf::from)
}

#[test]
fn rknn_matches_pytorch_golden() -> Result<(), Box<dyn std::error::Error>> {
    let library: PathBuf = std::env::var_os("RKNN_LIB").map_or_else(|| PathBuf::from(DEFAULT_LIBRARY), PathBuf::from);
    let models: PathBuf = models_dir();
    let (detnet, keynet) = (models.join(DEFAULT_DETNET), models.join(DEFAULT_KEYNET));
    for path in [&library, &detnet, &keynet] {
        if !path.exists() {
            eprintln!("SKIP rknn_matches_pytorch_golden: {} not found (this test runs on the RK3588 cap)", path.display());
            return Ok(());
        }
    }
    let golden: Golden = Golden::load(golden_dir())?;
    let mut nets: RknnNets = RknnNets::with_files(&library, &detnet, &keynet, 3, 2)?;
    let frames: Vec<Vec<u8>> = golden.detnet_frames();
    let frame_refs: Vec<NetFrame<'_>> = frames.iter().map(|frame| NetFrame { pixels: frame, top: 0 }).collect();
    let detnet_out = nets.detnet(&frame_refs)?;
    let crops: Vec<&[f32]> = golden.keynet_crops.iter().map(Vec::as_slice).collect();
    let keynet_out: Vec<KeyNetRaw> = nets.keynet(&crops, &golden.keynet_keypoints)?;
    let c: Comparison = Comparison::of(&golden, &detnet_out, &keynet_out)?;
    eprintln!("{} vs PyTorch: {c:?}", nets.describe());
    // Bands from the simulator over the full held-out sets (MODELS.md): FP16 DetNet <= 2.5 px centre, INT8 KeyNet mean
    // keypoint shift ~0.3 crop px. A layout or normalisation error gives tens of pixels.
    assert!(c.detnet_centre_px_max < 6.0, "detnet centre {}", c.detnet_centre_px_max);
    assert_eq!(c.detnet_presence_flips, 0);
    assert!(c.keynet_keypoint_px_mean < 1.5, "keynet keypoints {}", c.keynet_keypoint_px_mean);
    assert!(c.keynet_pinch_probability_max < 0.2, "pinch {}", c.keynet_pinch_probability_max);
    assert!(c.keynet_presence_logit_max < 2.0, "presence {}", c.keynet_presence_logit_max);
    // Two contexts (cores 1 and 2) must give the same answer as one-by-one calls.
    for (crop, (prior, batched)) in crops.iter().zip(golden.keynet_keypoints.iter().zip(&keynet_out)) {
        let single: Vec<KeyNetRaw> = nets.keynet(&[crop], std::slice::from_ref(prior))?;
        assert_eq!(&single[0], batched);
    }
    Ok(())
}
