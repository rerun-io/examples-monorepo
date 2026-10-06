//! The PyTorch reference for the hand nets (`tests/data/nets/`, written by handtrack's `export_nets_onnx`) and the comparisons
//! that the backend tests and `examples/nets_bench.rs` print.
//!
//! The files are little-endian raw arrays: DetNet pooled frames (u8 [k, 120, 160]) with their outputs (f32 [k, 8] = center
//! (l.x, l.y, r.x, r.y), radius (l, r), presence logit (l, r)), and KeyNet crops (u8 [k, 96, 96], the net sees u8 / 255),
//! priors (f32 [k, 63]) and outputs (f32 [k, 7184] = heatmaps, distance, presence logit, pinch logit), plus the decoded
//! keypoints (f32 [k, 21, 2], crop pixels).

use std::path::{Path, PathBuf};

use super::rknn::POOLED_LEN;
use super::{
    CROP_LEN, DETNET_HEIGHT, DETNET_WIDTH, DISTANCE_LEN, DetNetRaw, HEATMAP_LEN, KeyNetRaw,
    NUM_LANDMARKS, NetsError,
};
use crate::hands::detect::sigmoid;
use crate::hands::heatmaps::decode_heatmaps;

const KEYNET_OUT_LEN: usize = HEATMAP_LEN + DISTANCE_LEN + 2;

/// The crate's own golden directory (`crates/robocap-live/tests/data/nets`).
pub fn default_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/nets")
}

/// The golden inputs and PyTorch FP32 outputs.
#[derive(Clone, Debug)]
pub struct Golden {
    /// DetNet pooled frames, u8 120x160 each.
    pub detnet_pooled: Vec<Vec<u8>>,
    /// PyTorch DetNet outputs, one per frame.
    pub detnet_out: Vec<DetNetRaw>,
    /// KeyNet crops as the trait takes them (u8 / 255).
    pub keynet_crops: Vec<Vec<f32>>,
    /// KeyNet priors.
    pub keynet_keypoints: Vec<[f32; 3 * NUM_LANDMARKS]>,
    /// PyTorch KeyNet outputs, one per crop.
    pub keynet_out: Vec<KeyNetRaw>,
    /// handtrack's `decode_heatmaps` of the PyTorch heatmaps (crop px, (x, y) per landmark).
    pub keynet_points: Vec<[[f32; 2]; NUM_LANDMARKS]>,
}

fn read(path: &Path) -> Result<Vec<u8>, NetsError> {
    std::fs::read(path).map_err(|error| NetsError::Load {
        what: path.display().to_string(),
        message: error.to_string(),
    })
}

/// Little-endian f32 values from raw bytes (the golden files' format).
///
/// # Arguments
///
/// * `bytes` - Whole 4-byte little-endian values.
///
/// # Returns
///
/// The values; `None` when the length is not a multiple of 4.
pub fn f32_values(bytes: &[u8]) -> Option<Vec<f32>> {
    bytes.len().is_multiple_of(4).then(|| bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn floats(path: &Path) -> Result<Vec<f32>, NetsError> {
    let bytes = read(path)?;
    f32_values(&bytes).ok_or_else(|| {
        invalid(
            path,
            format!("{} bytes is not a whole number of f32 values", bytes.len()),
        )
    })
}

fn invalid(path: &Path, message: String) -> NetsError {
    NetsError::Load {
        what: path.display().to_string(),
        message,
    }
}

/// Expands a 120x160 pooled frame to the 640x480 frame the [`super::HandNets`] trait takes: each pixel becomes a 4x4 block, so
/// the 4x4 average pool gives back exactly the pooled frame.
pub fn expand_pooled(pooled: &[u8]) -> Vec<u8> {
    let width: usize = DETNET_WIDTH / 4;
    let mut frame: Vec<u8> = vec![0; DETNET_WIDTH * DETNET_HEIGHT];
    for (y, row) in frame.as_chunks_mut::<DETNET_WIDTH>().0.iter_mut().enumerate() {
        for (x, value) in row.iter_mut().enumerate() {
            *value = pooled[(y / 4) * width + x / 4];
        }
    }
    frame
}

/// Splits raw KeyNet output rows (f32 [k, 7184]) into [`KeyNetRaw`]s.
pub fn keynet_from_rows(values: &[f32]) -> Vec<KeyNetRaw> {
    values
        .as_chunks::<KEYNET_OUT_LEN>().0.iter()
        .map(|row| KeyNetRaw {
            heatmaps: row[..HEATMAP_LEN].to_vec(),
            distance: row[HEATMAP_LEN..HEATMAP_LEN + DISTANCE_LEN].to_vec(),
            presence_logit: row[HEATMAP_LEN + DISTANCE_LEN],
            pinch_logit: Some(row[HEATMAP_LEN + DISTANCE_LEN + 1]),
        })
        .collect()
}

/// Splits raw DetNet output rows (f32 [k, 8]) into [`DetNetRaw`]s.
pub fn detnet_from_rows(values: &[f32]) -> Vec<DetNetRaw> {
    values
        .as_chunks::<8>().0.iter()
        .map(|r| DetNetRaw { center: [[r[0], r[1]], [r[2], r[3]]], radius: [r[4], r[5]], presence_logit: [r[6], r[7]] })
        .collect()
}

/// The rows [`detnet_from_rows`] reads, for writing device outputs.
pub fn detnet_row(raw: &DetNetRaw) -> [f32; 8] {
    [
        raw.center[0][0],
        raw.center[0][1],
        raw.center[1][0],
        raw.center[1][1],
        raw.radius[0],
        raw.radius[1],
        raw.presence_logit[0],
        raw.presence_logit[1],
    ]
}

/// The row [`keynet_from_rows`] reads (a missing pinch logit is written as 0).
pub fn keynet_row(raw: &KeyNetRaw) -> Vec<f32> {
    let mut row: Vec<f32> = Vec::with_capacity(KEYNET_OUT_LEN);
    row.extend_from_slice(&raw.heatmaps);
    row.extend_from_slice(&raw.distance);
    row.push(raw.presence_logit);
    row.push(raw.pinch_logit.unwrap_or(0.0));
    row
}

impl Golden {
    /// Reads the golden files from `dir` (see the module docs).
    ///
    /// # Errors
    ///
    /// [`NetsError::Load`] when a file is missing or its size does not fit the others.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, NetsError> {
        let dir: &Path = dir.as_ref();
        let pooled: Vec<u8> = read(&dir.join("detnet_pooled_u8.bin"))?;
        let detnet_out: Vec<f32> = floats(&dir.join("detnet_out_f32.bin"))?;
        let crops: Vec<u8> = read(&dir.join("keynet_crops_u8.bin"))?;
        let priors: Vec<f32> = floats(&dir.join("keynet_keypoints_f32.bin"))?;
        let keynet_out: Vec<f32> = floats(&dir.join("keynet_out_f32.bin"))?;
        let points: Vec<f32> = floats(&dir.join("keynet_points_crop_f32.bin"))?;
        let frames: usize = pooled.len() / POOLED_LEN;
        let samples: usize = crops.len() / CROP_LEN;
        if pooled.len() != frames * POOLED_LEN || detnet_out.len() != frames * 8 {
            return Err(invalid(
                dir,
                format!(
                    "{} pooled bytes and {} DetNet outputs do not agree",
                    pooled.len(),
                    detnet_out.len()
                ),
            ));
        }
        if crops.len() != samples * CROP_LEN
            || priors.len() != samples * 63
            || keynet_out.len() != samples * KEYNET_OUT_LEN
            || points.len() != samples * 42
        {
            return Err(invalid(dir, "KeyNet golden file sizes do not agree".into()));
        }
        Ok(Self {
            detnet_pooled: pooled.as_chunks::<POOLED_LEN>().0.iter().map(|row| row.to_vec()).collect(),
            detnet_out: detnet_from_rows(&detnet_out),
            keynet_crops: crops.as_chunks::<CROP_LEN>().0.iter().map(|crop| crop.iter().map(|&v| f32::from(v) / 255.0).collect()).collect(),
            keynet_keypoints: priors.as_chunks::<63>().0.iter().map(|p| std::array::from_fn(|i| p[i])).collect(),
            keynet_out: keynet_from_rows(&keynet_out),
            keynet_points: points.as_chunks::<42>().0.iter().map(|p| std::array::from_fn(|k| [p[2 * k], p[2 * k + 1]])).collect(),
        })
    }

    /// The DetNet frames expanded to 640x480 (see [`expand_pooled`]).
    pub fn detnet_frames(&self) -> Vec<Vec<u8>> {
        self.detnet_pooled
            .iter()
            .map(|pooled| expand_pooled(pooled))
            .collect()
    }
}

/// How far a backend's outputs are from PyTorch's.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Comparison {
    /// DetNet: largest centre error, net px (640 x 480), over slots PyTorch reports present.
    pub detnet_centre_px_max: f32,
    /// DetNet: largest radius error, px.
    pub detnet_radius_px_max: f32,
    /// DetNet: largest presence-logit difference.
    pub detnet_presence_logit_max: f32,
    /// DetNet: slots whose presence (logit > 0) differs from PyTorch.
    pub detnet_presence_flips: usize,
    /// KeyNet: largest raw heatmap difference.
    pub keynet_heatmap_max: f32,
    /// KeyNet: largest raw distance difference.
    pub keynet_distance_max: f32,
    /// KeyNet: mean decoded keypoint shift, crop px.
    pub keynet_keypoint_px_mean: f32,
    /// KeyNet: largest decoded keypoint shift, crop px.
    pub keynet_keypoint_px_max: f32,
    /// KeyNet: largest presence-logit difference.
    pub keynet_presence_logit_max: f32,
    /// KeyNet: largest pinch-probability difference.
    pub keynet_pinch_probability_max: f32,
}

impl Comparison {
    /// Compares a backend's outputs on the golden inputs with PyTorch's.
    ///
    /// # Errors
    ///
    /// [`NetsError::Input`] for mismatched counts or shapes, missing heads, or non-finite values.
    pub fn of(
        golden: &Golden,
        detnet: &[DetNetRaw],
        keynet: &[KeyNetRaw],
    ) -> Result<Self, NetsError> {
        let invalid = |net, message: &str| NetsError::Input {
            net,
            message: message.into(),
        };
        if detnet.len() != golden.detnet_out.len() {
            return Err(invalid("detnet", "output count differs from golden"));
        }
        if keynet.len() != golden.keynet_out.len() || keynet.len() != golden.keynet_points.len() {
            return Err(invalid(
                "keynet",
                "output or decoded-point count differs from golden",
            ));
        }
        if detnet
            .iter()
            .chain(&golden.detnet_out)
            .any(|raw| detnet_row(raw).iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("detnet", "non-finite output"));
        }
        for raw in keynet.iter().chain(&golden.keynet_out) {
            if raw.heatmaps.len() != HEATMAP_LEN || raw.distance.len() != DISTANCE_LEN {
                return Err(invalid("keynet", "wrong heatmap or distance tensor length"));
            }
            if raw
                .heatmaps
                .iter()
                .chain(&raw.distance)
                .chain(std::iter::once(&raw.presence_logit))
                .chain(raw.pinch_logit.iter())
                .any(|v| !v.is_finite())
            {
                return Err(invalid("keynet", "non-finite output"));
            }
        }
        for (ours, theirs) in keynet.iter().zip(&golden.keynet_out) {
            if theirs.pinch_logit.is_some() && ours.pinch_logit.is_none() {
                return Err(invalid("keynet", "missing head present in golden"));
            }
        }
        if golden
            .keynet_points
            .iter()
            .flatten()
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err(invalid("keynet", "non-finite golden decoded point"));
        }
        let mut result: Comparison = Comparison::default();
        for (ours, theirs) in detnet.iter().zip(&golden.detnet_out) {
            for slot in 0..2 {
                if theirs.presence_logit[slot] > 0.0 {
                    let dx: f32 =
                        (ours.center[slot][0] - theirs.center[slot][0]) * DETNET_WIDTH as f32;
                    let dy: f32 =
                        (ours.center[slot][1] - theirs.center[slot][1]) * DETNET_HEIGHT as f32;
                    result.detnet_centre_px_max = result.detnet_centre_px_max.max(dx.hypot(dy));
                    result.detnet_radius_px_max = result
                        .detnet_radius_px_max
                        .max((ours.radius[slot] - theirs.radius[slot]).abs() * DETNET_WIDTH as f32);
                }
                result.detnet_presence_logit_max = result
                    .detnet_presence_logit_max
                    .max((ours.presence_logit[slot] - theirs.presence_logit[slot]).abs());
                result.detnet_presence_flips += usize::from(
                    (ours.presence_logit[slot] > 0.0) != (theirs.presence_logit[slot] > 0.0),
                );
            }
        }
        let mut shift_sum: f32 = 0.0;
        let mut shifts: usize = 0;
        for ((ours, theirs), points) in keynet
            .iter()
            .zip(&golden.keynet_out)
            .zip(&golden.keynet_points)
        {
            let max_diff = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
            };
            result.keynet_heatmap_max = result
                .keynet_heatmap_max
                .max(max_diff(&ours.heatmaps, &theirs.heatmaps));
            result.keynet_distance_max = result
                .keynet_distance_max
                .max(max_diff(&ours.distance, &theirs.distance));
            let (decoded, _) = decode_heatmaps(&ours.heatmaps)
                .ok_or_else(|| invalid("keynet", "heatmap decoding failed"))?;
            if decoded.iter().flatten().any(|v| !v.is_finite()) {
                return Err(invalid("keynet", "non-finite decoded point"));
            }
            for (decoded, reference) in decoded.iter().zip(points) {
                let shift: f32 = (decoded[0] - reference[0]).hypot(decoded[1] - reference[1]);
                shift_sum += shift;
                shifts += 1;
                result.keynet_keypoint_px_max = result.keynet_keypoint_px_max.max(shift);
            }
            result.keynet_presence_logit_max = result
                .keynet_presence_logit_max
                .max((ours.presence_logit - theirs.presence_logit).abs());
            if let (Some(a), Some(b)) = (ours.pinch_logit, theirs.pinch_logit) {
                result.keynet_pinch_probability_max = result
                    .keynet_pinch_probability_max
                    .max((sigmoid(a) - sigmoid(b)).abs());
            }
        }
        result.keynet_keypoint_px_mean = if shifts > 0 {
            shift_sum / shifts as f32
        } else {
            0.0
        };
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_rejects_empty_outputs() {
        let golden = Golden {
            detnet_pooled: vec![vec![0; POOLED_LEN]],
            detnet_out: detnet_from_rows(&[0.0; 8]),
            keynet_crops: vec![vec![0.0; CROP_LEN]],
            keynet_keypoints: vec![[0.0; 63]],
            keynet_out: keynet_from_rows(&[0.0; KEYNET_OUT_LEN]),
            keynet_points: vec![[[0.0; 2]; NUM_LANDMARKS]],
        };
        assert!(matches!(
            Comparison::of(&golden, &[], &[]),
            Err(NetsError::Input { .. })
        ));
        assert!(Comparison::of(&golden, &[], &golden.keynet_out).is_err());
        assert!(Comparison::of(&golden, &golden.detnet_out, &[]).is_err());
    }

    #[test]
    fn comparison_rejects_malformed_outputs_and_references() {
        let golden = Golden::load(default_dir()).expect("golden fixtures");
        for change in 0..8 {
            let mut keynet = golden.keynet_out.clone();
            match change {
                0 => {
                    keynet[0].heatmaps.pop();
                }
                1 => {
                    keynet[0].distance.pop();
                }
                2 => keynet[0].heatmaps[0] = f32::NAN,
                3 => keynet[0].distance[0] = f32::INFINITY,
                4 => keynet[0].presence_logit = f32::NAN,
                5 => keynet[0].pinch_logit = Some(f32::NAN),
                6 => keynet[0].pinch_logit = None,
                _ => {
                    keynet.pop();
                }
            }
            assert!(
                Comparison::of(&golden, &golden.detnet_out, &keynet).is_err(),
                "change {change}"
            );
            let mut reference = golden.clone();
            reference.keynet_out = keynet;
            assert_eq!(
                Comparison::of(&reference, &golden.detnet_out, &golden.keynet_out).is_err(),
                change != 6,
                "reference change {change}"
            );
        }
        let mut detnet = golden.detnet_out.clone();
        detnet[0].center[0][0] = f32::NAN;
        assert!(Comparison::of(&golden, &detnet, &golden.keynet_out).is_err());
        let mut reference = golden.clone();
        reference.detnet_out = detnet;
        assert!(Comparison::of(&reference, &golden.detnet_out, &golden.keynet_out).is_err());
        reference = golden.clone();
        reference.keynet_points[0][0][0] = f32::NAN;
        assert!(Comparison::of(&reference, &golden.detnet_out, &golden.keynet_out).is_err());
        reference = golden.clone();
        reference.keynet_points.pop();
        assert!(Comparison::of(&reference, &golden.detnet_out, &golden.keynet_out).is_err());
    }

    #[test]
    fn golden_rejects_trailing_partial_floats() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!(
            "robocap-live-golden-partial-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir)?;
        for entry in std::fs::read_dir(default_dir())? {
            let entry = entry?;
            std::fs::copy(entry.path(), dir.join(entry.file_name()))?;
        }
        assert!(Golden::load(&dir).is_ok());
        for name in [
            "detnet_out_f32.bin",
            "keynet_keypoints_f32.bin",
            "keynet_out_f32.bin",
            "keynet_points_crop_f32.bin",
        ] {
            let path = dir.join(name);
            let bytes = std::fs::read(&path)?;
            for trailing in 1..4 {
                let mut malformed = bytes.clone();
                malformed.extend(vec![0; trailing]);
                std::fs::write(&path, malformed)?;
                assert!(
                    matches!(Golden::load(&dir), Err(NetsError::Load { .. })),
                    "{name}: {trailing} trailing bytes"
                );
            }
            std::fs::write(&path, bytes)?;
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn golden_files_load_and_decode_like_handtrack() {
        let golden: Result<Golden, NetsError> = Golden::load(default_dir());
        assert!(golden.is_ok(), "{:?}", golden.err());
        let Ok(golden) = golden else { return };
        assert_eq!(golden.detnet_pooled.len(), golden.detnet_out.len());
        assert!(!golden.keynet_out.is_empty());
        for (raw, points) in golden.keynet_out.iter().zip(&golden.keynet_points) {
            let decoded = decode_heatmaps(&raw.heatmaps).map(|(points, _)| points);
            assert!(decoded.is_some(), "21 x 18 x 18 heatmaps");
            for (ours, theirs) in decoded.iter().flatten().zip(points) {
                assert!(
                    (ours[0] - theirs[0]).abs() < 1e-3 && (ours[1] - theirs[1]).abs() < 1e-3,
                    "{ours:?} vs {theirs:?}"
                );
            }
        }
        let comparison: Comparison =
            Comparison::of(&golden, &golden.detnet_out, &golden.keynet_out).expect("valid golden");
        assert_eq!(comparison.keynet_heatmap_max, 0.0);
        assert!(comparison.keynet_keypoint_px_max < 1e-3);
    }

    #[test]
    fn expanded_frames_pool_back_exactly() {
        let pooled: Vec<u8> = (0..POOLED_LEN).map(|i| (i % 251) as u8).collect();
        let frame: Vec<u8> = expand_pooled(&pooled);
        let mut back: Vec<u8> = vec![0; POOLED_LEN];
        assert!(
            kornia_staging_imgproc::resize::pool4_u8(
                &frame,
                DETNET_WIDTH,
                DETNET_HEIGHT,
                &mut back
            )
            .is_ok()
        );
        assert_eq!(back, pooled);
    }
}
