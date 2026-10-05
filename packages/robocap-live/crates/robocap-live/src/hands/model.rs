//! The generic UmeTrack hand model (handtrack's `generic_hand_model()`) as the handfit fit takes it, scaled by phi.
//!
//! `assets/generic_hand_model.json` holds the arrays `handtrack.fit.native` passes to handfit (float32 values, joints 0-19,
//! the topology), exported once by `tools/golden_tracker.py`. It is compiled into the binary, so the cap needs no extra file.
//! Scaling follows `handtrack/fit/scale.py::scaled_hand_model`: rest joints and rest landmarks times phi, in float32.

use handfit::model::Step;
use handfit::nalgebra::{SMatrix, SVector};
use handfit::{Model, Pose};
use serde::Deserialize;

use super::{HandsError, LEFT};
use crate::nets::NUM_LANDMARKS;

/// The exported generic model (see the module docs).
const GENERIC_HAND_MODEL_JSON: &str = include_str!("../../assets/generic_hand_model.json");

/// Skinned joint angles; the pose stores 22, angles 20 and 21 are not skinned.
pub const FIT_JOINTS: usize = 20;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HandModelArrays {
    /// The asset's provenance: required by the format, not used.
    #[serde(rename = "source")]
    _source: String,
    joint_rotation_axes: Vec<[f64; 3]>,
    joint_rest_positions: Vec<[f64; 3]>,
    landmark_rest_positions: Vec<[f64; 3]>,
    landmark_rest_bone_weights: Vec<[f64; 3]>,
    landmark_rest_bone_indices: Vec<[i64; 3]>,
    joint_limits: Vec<[f64; 2]>,
    topology: Vec<Vec<i64>>,
}

/// The generic hand model at scale 1 (handfit's `Model`, millimetres).
#[derive(Clone)]
pub struct GenericHandModel {
    model: Model,
}

impl GenericHandModel {
    /// Parse and validate the compiled-in generic model.
    ///
    /// # Errors
    ///
    /// [`HandsError::Model`] when the asset is malformed or its topology differs from handfit's generated kernel.
    pub fn load() -> Result<Self, HandsError> {
        Self::from_json(GENERIC_HAND_MODEL_JSON)
    }

    /// Parse a model document in the asset's format.
    ///
    /// # Errors
    ///
    /// [`HandsError::Model`] on malformed JSON, wrong array sizes, or a model [`Model::new`] refuses (non-finite values, reversed
    /// limits, duplicate bone weights, a topology that handfit's generated kernel does not implement).
    pub fn from_json(text: &str) -> Result<Self, HandsError> {
        let arrays: HandModelArrays = serde_json::from_str(text).map_err(|e| HandsError::Model(format!("generic hand model: {e}")))?;
        let sizes = [
            ("joint_rotation_axes", arrays.joint_rotation_axes.len(), FIT_JOINTS),
            ("joint_rest_positions", arrays.joint_rest_positions.len(), FIT_JOINTS),
            ("landmark_rest_positions", arrays.landmark_rest_positions.len(), NUM_LANDMARKS),
            ("landmark_rest_bone_weights", arrays.landmark_rest_bone_weights.len(), NUM_LANDMARKS),
            ("landmark_rest_bone_indices", arrays.landmark_rest_bone_indices.len(), NUM_LANDMARKS),
            ("joint_limits", arrays.joint_limits.len(), FIT_JOINTS),
            ("topology", arrays.topology.len(), 4),
        ];
        for (name, got, want) in sizes {
            if got != want {
                return Err(HandsError::Model(format!("generic hand model: {name} has {got} rows, expected {want}")));
            }
        }
        let indices: Vec<i64> = arrays.landmark_rest_bone_indices.iter().flatten().copied().collect();
        let topology: Vec<i64> = arrays.topology.iter().flatten().copied().collect();
        let model = Model::new(
            SMatrix::from_fn(|i, k| arrays.joint_rotation_axes[i][k]),
            SMatrix::from_fn(|i, k| arrays.joint_rest_positions[i][k]),
            SMatrix::from_fn(|i, k| arrays.landmark_rest_positions[i][k]),
            SMatrix::from_fn(|i, k| arrays.landmark_rest_bone_weights[i][k]),
            SMatrix::from_fn(|i, k| arrays.joint_limits[i][k]),
            &indices,
            &topology,
        )
        .map_err(|e| HandsError::Model(format!("generic hand model: {e}")))?;
        Ok(Self { model })
    }

    /// The model at scale 1.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The hand enlarged by `phi` about its wrist (handtrack `scaled_hand_model`): rest joints and rest landmarks times phi,
    /// multiplied in float32 as torch does.
    pub fn scaled(&self, phi: f64) -> Model {
        let phi32 = phi as f32;
        let scale = |x: f64| (x as f32 * phi32) as f64;
        Model { pivots: self.model.pivots.map(scale), rest: self.model.rest.map(scale), ..self.model.clone() }
    }
}

/// The skinning mirror of a hand slot: +1 for the left hand (the model's own), -1 for the right (x axis mirrored).
pub fn mirror(side: usize) -> f64 {
    if side == LEFT { 1.0 } else { -1.0 }
}

/// The 21 skinned landmarks of `pose` in world metres (handtrack `hand.pose.landmarks`).
pub fn landmarks_world(model: &Model, pose: &Pose, side: usize) -> [[f64; 3]; NUM_LANDMARKS] {
    let points: SVector<f64, 63> = model.landmarks(pose, mirror(side), &Step::zeros(), None);
    std::array::from_fn(|i| [points[3 * i], points[3 * i + 1], points[3 * i + 2]])
}

/// The neutral pose (identity wrist at the origin, zero joint angles): handtrack's placeholder prior of an acquisition.
#[cfg(test)]
pub fn identity_pose() -> Pose {
    Pose { rotation: handfit::nalgebra::Matrix3::identity(), translation: handfit::nalgebra::Vector3::zeros(), angles: SVector::zeros() }
}

/// Every pose value rounded to float32, as handfit's Python binding returns poses and the Python tracker stores them.
pub fn pose_f32(pose: &Pose) -> Pose {
    let round = |x: f64| x as f32 as f64;
    Pose { rotation: pose.rotation.map(round), translation: pose.translation.map(round), angles: pose.angles.map(round) }
}

/// Whether every value of `pose` is finite (handtrack `tracker._finite`).
pub fn pose_finite(pose: &Pose) -> bool {
    pose.rotation.iter().chain(pose.translation.iter()).chain(pose.angles.iter()).all(|x| x.is_finite())
}

#[cfg(test)]
mod tests {
    use handfit::nalgebra::Vector3;

    use super::*;

    #[test]
    fn the_asset_loads_and_scales_about_the_wrist() -> Result<(), HandsError> {
        let generic = GenericHandModel::load()?;
        let pose = Pose { translation: Vector3::new(0.1, -0.2, 0.4), ..identity_pose() };
        let unit = landmarks_world(generic.model(), &pose, LEFT);
        let half = landmarks_world(&generic.scaled(0.5), &pose, LEFT);
        for i in 0..NUM_LANDMARKS {
            for k in 0..3 {
                let t = pose.translation[k];
                assert!(((half[i][k] - t) - 0.5 * (unit[i][k] - t)).abs() < 1e-6, "landmark {i} axis {k}");
            }
        }
        // A hand is a few centimetres across, not millimetres or metres.
        let span = (0..NUM_LANDMARKS).map(|i| (unit[i][1] - unit[5][1]).abs().max((unit[i][0] - unit[5][0]).abs())).fold(0.0, f64::max);
        assert!(span > 0.05 && span < 0.3, "span {span}");
        Ok(())
    }

    #[test]
    fn the_right_hand_mirrors_the_left_in_the_wrist_frame() -> Result<(), HandsError> {
        let generic = GenericHandModel::load()?;
        let pose = identity_pose();
        let left = landmarks_world(generic.model(), &pose, LEFT);
        let right = landmarks_world(generic.model(), &pose, super::super::RIGHT);
        for i in 0..NUM_LANDMARKS {
            assert!((left[i][0] + right[i][0]).abs() < 1e-9 && (left[i][1] - right[i][1]).abs() < 1e-9);
        }
        Ok(())
    }

    #[test]
    fn a_malformed_model_is_an_error() {
        assert!(GenericHandModel::from_json("{}").is_err());
    }
}
