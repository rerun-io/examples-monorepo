//! The two hand networks behind one trait: the RK3588 NPU (librknnrt, dlopened) on the cap, ONNX Runtime (`ort`) on hosts.
//!
//! Inputs and outputs are the raw tensors of handtrack's `DetNetF` and `KeyNetF` (decoding lives in `hands`), so both backends are
//! checked against the same PyTorch reference.
#![deny(missing_docs)]

/// DetNet's input frame (BarLetterbox of the 640x360 small image), width x height.
pub const DETNET_WIDTH: usize = 640;
/// DetNet's input frame height.
pub const DETNET_HEIGHT: usize = 480;
/// KeyNet's crop side.
pub const KEYNET_CROP: usize = 96;
/// KeyNet heatmap side.
pub const HEATMAP_SIDE: usize = 18;
/// KeyNet relative-distance bins.
pub const DISTANCE_BINS: usize = 18;
/// Hand landmarks (UmeTrack `LANDMARK` order: five fingertips, the wrist at 5, ..., the palm centre at 20).
pub const NUM_LANDMARKS: usize = 21;
/// Values in one KeyNet crop, 96 x 96.
pub const CROP_LEN: usize = KEYNET_CROP * KEYNET_CROP;
/// Values in one crop's KeyNet heatmaps, 21 x 18 x 18.
pub const HEATMAP_LEN: usize = NUM_LANDMARKS * HEATMAP_SIDE * HEATMAP_SIDE;
/// Values in one crop's KeyNet distance heatmaps, 21 x 18.
pub const DISTANCE_LEN: usize = NUM_LANDMARKS * DISTANCE_BINS;

/// Errors of a network backend.
#[derive(Debug, thiserror::Error)]
pub enum NetsError {
    /// The runtime library or a model could not be loaded.
    #[error("loading {what}: {message}")]
    Load {
        /// The library or model file.
        what: String,
        /// Why it failed.
        message: String,
    },
    /// Inference failed.
    #[error("running {net}: {message}")]
    Run {
        /// `detnet` or `keynet`.
        net: &'static str,
        /// The runtime's error.
        message: String,
    },
    /// An input had the wrong size or count.
    #[error("bad input for {net}: {message}")]
    Input {
        /// `detnet` or `keynet`.
        net: &'static str,
        /// What did not match.
        message: String,
    },
}

/// DetNet output for one frame (handtrack `DetNetOutput`); slot 0 = left hand, 1 = right.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetNetRaw {
    /// (cx / 640, cy / 480) in the net frame.
    pub center: [[f32; 2]; 2],
    /// Circle radius / 640.
    pub radius: [f32; 2],
    /// Unbounded presence logits.
    pub presence_logit: [f32; 2],
}

/// KeyNet output for one crop (handtrack `KeyNetOutput`), crop in left-hand orientation.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyNetRaw {
    /// 21 x 18 x 18, row-major (landmark, row, column).
    pub heatmaps: Vec<f32>,
    /// 21 x 18 relative-distance heatmaps.
    pub distance: Vec<f32>,
    /// Unbounded crop-presence logit.
    pub presence_logit: f32,
    /// Thumb-index contact logit when the model has the pinch head (keynet-strong-pinch does).
    pub pinch_logit: Option<f32>,
}

/// One DetNet input frame without its padding: 640-wide u8 rows (row-major) at net rows `top..top + rows`, black (0) above and
/// below in the 640x480 frame. A full 640x480 frame has `top` 0; a RoboCap 640x360 small image has `top` 60 (the BarLetterbox
/// bars). Backends write their input straight from these rows, so nobody builds the padded frame.
#[derive(Clone, Copy, Debug)]
pub struct NetFrame<'a> {
    /// `640 * rows` bytes.
    pub pixels: &'a [u8],
    /// The net row of the first pixel row.
    pub top: usize,
}

impl NetFrame<'_> {
    /// The number of pixel rows.
    ///
    /// # Errors
    ///
    /// [`NetsError::Input`] when the pixels are not whole 640-wide rows or do not fit in the 480 rows below `top`.
    pub fn rows(&self) -> Result<usize, NetsError> {
        let rows: usize = self.pixels.len() / DETNET_WIDTH;
        if self.pixels.len() % DETNET_WIDTH != 0 || self.top + rows > DETNET_HEIGHT {
            let message: String = format!("{} bytes at row {} do not fit the 640x480 frame", self.pixels.len(), self.top);
            return Err(NetsError::Input { net: "detnet", message });
        }
        Ok(rows)
    }

    /// Writes the whole 640x480 frame as `u8 / 255` floats (ONNX DetNet's input; the bars are 0).
    ///
    /// # Arguments
    ///
    /// * `dst` - 640 x 480 values, overwritten.
    ///
    /// # Errors
    ///
    /// As [`NetFrame::rows`]; [`NetsError::Input`] when `dst` has another length.
    pub fn write_unit_f32(&self, dst: &mut [f32]) -> Result<(), NetsError> {
        let rows: usize = self.rows()?;
        if dst.len() != DETNET_WIDTH * DETNET_HEIGHT {
            return Err(NetsError::Input { net: "detnet", message: format!("{} values for a 640x480 frame", dst.len()) });
        }
        let (above, rest) = dst.split_at_mut(self.top * DETNET_WIDTH);
        let (image, below) = rest.split_at_mut(rows * DETNET_WIDTH);
        above.fill(0.0);
        for (out, &value) in image.iter_mut().zip(self.pixels) {
            *out = f32::from(value) / 255.0;
        }
        below.fill(0.0);
        Ok(())
    }
}

/// A backend running DetNet and KeyNet. Batches may be any size; outputs come back in input order.
pub trait HandNets: Send {
    /// `frames`: 640x480 BarLetterbox frames, given without their black bars; the net sees `u8 / 255`.
    ///
    /// # Errors
    ///
    /// [`NetsError::Input`] for a frame that does not fit the 640x480 net frame, [`NetsError::Run`] when inference fails.
    fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError>;
    /// `crops`: 96x96 values in [0, 1], already mirrored to left-hand orientation; `keypoints`: the 63-value prior
    /// (21 x (u, v, relative distance), zeros when untracked), as handtrack's `KeyNetEstimator` builds it.
    ///
    /// # Errors
    ///
    /// [`NetsError::Input`] for a crop of another size or mismatched counts, [`NetsError::Run`] when inference fails.
    fn keynet(&mut self, crops: &[&[f32]], keypoints: &[[f32; 3 * NUM_LANDMARKS]]) -> Result<Vec<KeyNetRaw>, NetsError>;
    /// A short description for logs ("rknn int8 3 cores", "ort cuda").
    fn describe(&self) -> String;
}

/// No networks: DetNet sees no hand and KeyNet is never asked (`--nets none`; the tracker then runs but tracks nothing).
pub struct NoNets;

impl HandNets for NoNets {
    fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        Ok(frames.iter().map(|_| DetNetRaw { center: [[0.5, 0.5]; 2], radius: [0.0; 2], presence_logit: [-20.0; 2] }).collect())
    }

    fn keynet(&mut self, crops: &[&[f32]], _: &[[f32; 3 * NUM_LANDMARKS]]) -> Result<Vec<KeyNetRaw>, NetsError> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        Err(NetsError::Run { net: "keynet", message: "no networks: there is no KeyNet".into() })
    }

    fn describe(&self) -> String {
        "none (no hands)".into()
    }
}

// Backends.
pub mod golden;
#[cfg(feature = "ort")]
pub mod ort;
pub mod rknn;
