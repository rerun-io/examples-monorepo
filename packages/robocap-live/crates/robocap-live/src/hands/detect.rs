//! DetNet as the tracker's detector: letterbox the small images, run the net, decode (handtrack `DetNetDetector` and
//! `models/detnet.py::decode_detections`).

use kornia_image::Image;

use super::HandsError;
use super::letterbox::BarLetterbox;
use crate::nets::{DETNET_HEIGHT, DETNET_WIDTH, DetNetRaw, HandNets, NetFrame};

/// Decoded detections of one frame in the 640x480 net frame; slot 0 = left hand.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Detections {
    /// (cx, cy, radius) in net pixels; the radius uses the frame width (640).
    pub circle_net: [[f32; 3]; 2],
    /// Presence probabilities (sigmoid of the logits).
    pub probability: [f32; 2],
}

impl Detections {
    /// handtrack's `present`: the probability strictly exceeds `threshold` (ROBUST_TRACKER_CONFIG uses 0.8).
    pub fn present(&self, side: usize, threshold: f32) -> bool {
        self.probability.get(side).is_some_and(|p| *p > threshold)
    }

    /// The square box `(x0, y0, x1, y1)` enclosing a slot's circle, unclipped.
    pub fn box_net(&self, side: usize) -> Option<[f32; 4]> {
        self.circle_net.get(side).map(|[cx, cy, r]| [cx - r, cy - r, cx + r, cy + r])
    }
}

/// The logistic function in f32, as torch's `sigmoid`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Scale DetNet's normalised circles to net pixels (`decode_detections`): centre x by 640, y by 480, radius by 640.
pub fn decode_detections(raw: &DetNetRaw) -> Detections {
    let mut out = Detections::default();
    for side in 0..2 {
        out.circle_net[side] =
            [raw.center[side][0] * DETNET_WIDTH as f32, raw.center[side][1] * DETNET_HEIGHT as f32, raw.radius[side] * DETNET_WIDTH as f32];
        out.probability[side] = sigmoid(raw.presence_logit[side]);
    }
    out
}

/// The circles DetNet found in a frame turned 180 degrees, in the unturned net frame: pixel centres (x, y) -> (639 - x, 479 - y)
/// (the RoboCap letterbox's bars are equal above and below, so the turned image keeps its rows); radii and slots stay.
pub fn turn_back(detections: Detections) -> Detections {
    let mut out = detections;
    for circle in &mut out.circle_net {
        circle[0] = (DETNET_WIDTH - 1) as f32 - circle[0];
        circle[1] = (DETNET_HEIGHT - 1) as f32 - circle[1];
    }
    out
}

/// Letterbox each camera's 640x360 small image, run DetNet on the batch and decode; one [`Detections`] per input, in order.
/// The backend reads the small images' rows directly (no padded frame is built). An image marked in `half_turn` is fed turned
/// 180 degrees (an upright model on a camera mounted upside down) and its circles come back in the unturned net frame.
///
/// # Arguments
///
/// * `nets` - The DetNet backend.
/// * `letterbox` - The small image's placement in the 640x480 net frame.
/// * `small` - One 640x360 image per camera.
/// * `half_turn` - Per image: feed it turned 180 degrees.
///
/// # Errors
///
/// [`HandsError::Invalid`] for a small image of the wrong size, [`HandsError::Nets`] when the backend fails or returns the
/// wrong number of outputs.
pub fn detect(
    nets: &mut dyn HandNets,
    letterbox: &BarLetterbox,
    small: &[&Image<u8, 1>],
    half_turn: &[bool],
) -> Result<Vec<Detections>, HandsError> {
    if half_turn.len() != small.len() {
        return Err(HandsError::Invalid(format!("{} half-turn flags for {} images", half_turn.len(), small.len())));
    }
    let invalid = |error: kornia_image::ImageError| HandsError::Invalid(format!("DetNet letterbox: {error}"));
    let inputs: Vec<NetFrame<'_>> = small
        .iter()
        .zip(half_turn)
        .map(|(image, &turned)| letterbox.net_frame(image).map(|frame| frame.with_half_turn(turned)))
        .collect::<Result<_, _>>()
        .map_err(invalid)?;
    let raw = nets.detnet(&inputs)?;
    if raw.len() != inputs.len() {
        return Err(HandsError::Invalid(format!("DetNet returned {} outputs for {} frames", raw.len(), inputs.len())));
    }
    Ok(raw
        .iter()
        .zip(half_turn)
        .map(|(raw, &turned)| if turned { turn_back(decode_detections(raw)) } else { decode_detections(raw) })
        .collect())
}
