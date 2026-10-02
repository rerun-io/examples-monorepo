//! The UmeTrack generic hand mesh skinned on a fitted pose, for the viewer (simplecv `skin_mesh`, what show3d and handtrack draw).
//!
//! `assets/generic_hand_mesh.json` holds UmeTrack's generic mesh (788 vertices in millimetres in the left hand's wrist frame,
//! 1544 triangles, dense weights over the 17 skinning frames); the joints come from the fit's own model, so the mesh follows
//! exactly the hand the fit solved. The frames are simplecv's `_hand_skinning_transform`: the root and the wrist, then per finger
//! a chain of four rotations about the joint pivots whose first two share a frame. handfit's generated kernel skins the
//! landmarks through the same frames (the tests check that).

use std::sync::Arc;

use handfit::nalgebra::{Matrix4, Rotation3, Vector3};
use handfit::{Model, Pose};
use serde::Deserialize;

use super::HandsError;
use super::model::GenericHandModel;

/// The extracted mesh (see the module docs).
const GENERIC_HAND_MESH_JSON: &str = include_str!("../../assets/generic_hand_mesh.json");
/// Skinning frames: root, wrist, three per finger.
pub const SKINNING_FRAMES: usize = 17;
/// Mesh vertices.
pub const MESH_VERTICES: usize = 788;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MeshArrays {
    #[serde(rename = "source")]
    _source: String,
    mesh_vertices: Vec<[f64; 3]>,
    mesh_triangles: Vec<[u32; 3]>,
    dense_bone_weights: Vec<[f64; SKINNING_FRAMES]>,
}

/// The generic hand mesh at scale 1.
#[derive(Clone, Debug)]
pub struct HandMesh {
    vertices: Vec<Vector3<f64>>,
    triangles: Arc<[[u32; 3]]>,
    weights: Vec<[f64; SKINNING_FRAMES]>,
}

impl HandMesh {
    /// Parse and check the compiled-in mesh.
    ///
    /// # Errors
    ///
    /// [`HandsError::Model`] when the asset is malformed: wrong sizes, a triangle past the last vertex, non-finite values.
    pub fn load() -> Result<Self, HandsError> {
        let arrays: MeshArrays = serde_json::from_str(GENERIC_HAND_MESH_JSON).map_err(|e| HandsError::Model(format!("generic hand mesh: {e}")))?;
        let vertices: Vec<Vector3<f64>> = arrays.mesh_vertices.iter().map(|v| Vector3::new(v[0], v[1], v[2])).collect();
        let finite = vertices.iter().flat_map(|v| v.iter()).chain(arrays.dense_bone_weights.iter().flatten()).all(|x| x.is_finite());
        if vertices.len() != MESH_VERTICES || arrays.dense_bone_weights.len() != MESH_VERTICES || !finite {
            return Err(HandsError::Model(format!("generic hand mesh: {} vertices, {} weight rows", vertices.len(), arrays.dense_bone_weights.len())));
        }
        if arrays.mesh_triangles.iter().flatten().any(|&i| i as usize >= MESH_VERTICES) {
            return Err(HandsError::Model("generic hand mesh: a triangle past the last vertex".into()));
        }
        Ok(Self { vertices, triangles: arrays.mesh_triangles.into(), weights: arrays.dense_bone_weights })
    }

    /// The triangle list (shared: a clone copies no indices).
    pub fn triangles(&self) -> &Arc<[[u32; 3]]> {
        &self.triangles
    }

    /// The mesh on `pose` in world metres, for the generic hand at scale `phi`: the joints from the fit's model
    /// (`generic.scaled(phi)`), the vertices scaled by `phi` about the wrist as the landmarks are; `mirror` is -1 for the
    /// right hand.
    pub fn skin(&self, generic: &GenericHandModel, phi: f64, pose: &Pose, mirror: f64) -> Vec<[f32; 3]> {
        let scaled: Vec<Vector3<f64>> = self.vertices.iter().map(|v| v * phi).collect();
        skin_points(&generic.scaled(phi), pose, mirror, &scaled, &self.weights).iter().map(|v| [v.x as f32, v.y as f32, v.z as f32]).collect()
    }
}

/// A joint's rotation by `angle` about its axis through its pivot (simplecv `_joint_local_transform`, its `so3_exp_map`).
fn joint_local(axis: &Vector3<f64>, pivot: &Vector3<f64>, angle: f64) -> Matrix4<f64> {
    let rotation = Rotation3::from_scaled_axis(axis * angle).into_inner();
    let mut local = Matrix4::identity();
    local.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    local.fixed_view_mut::<3, 1>(0, 3).copy_from(&(pivot - rotation * pivot));
    local
}

/// The 17 skinning frames of `pose` in the model's millimetres, rotated into the world but not yet moved (simplecv
/// `_hand_skinning_transform` with the wrist's rotation; `wrist_for_hand` mirrors its x column for the right hand).
pub fn skinning_frames(model: &Model, pose: &Pose, mirror: f64) -> [Matrix4<f64>; SKINNING_FRAMES] {
    let mut wrist = Matrix4::identity();
    let mut rotation = pose.rotation;
    rotation.column_mut(0).scale_mut(mirror);
    wrist.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    let mut frames = [wrist; SKINNING_FRAMES];
    for finger in 0..5 {
        let mut chain = wrist;
        for dof in 0..4 {
            let joint = 4 * finger + dof;
            let axis: Vector3<f64> = model.axes.row(joint).transpose();
            let pivot: Vector3<f64> = model.pivots.row(joint).transpose();
            chain *= joint_local(&axis, &pivot, pose.angles[joint]);
            // The finger's first two joints share a frame.
            if dof >= 1 {
                frames[2 + 3 * finger + dof - 1] = chain;
            }
        }
    }
    frames
}

/// Points (millimetres, in the model's wrist frame) blended from the skinning frames by dense weights, in world metres.
pub fn skin_points(model: &Model, pose: &Pose, mirror: f64, points_mm: &[Vector3<f64>], weights: &[[f64; SKINNING_FRAMES]]) -> Vec<Vector3<f64>> {
    let frames = skinning_frames(model, pose, mirror);
    points_mm
        .iter()
        .zip(weights)
        .map(|(point, row)| {
            let homogeneous = point.push(1.0);
            let blended: Vector3<f64> = frames.iter().zip(row).filter(|(_, w)| **w != 0.0).map(|(frame, w)| (frame * homogeneous).xyz() * *w).sum();
            blended * 1e-3 + pose.translation
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use handfit::nalgebra::SVector;

    use super::*;
    use crate::hands::model::{GenericHandModel, landmarks_world, mirror};
    use crate::nets::NUM_LANDMARKS;
    use crate::hands::{LEFT, RIGHT};

    /// A few poses inside the joint limits, turned and moved away from the origin.
    fn poses(model: &Model) -> Vec<Pose> {
        (0..4)
            .map(|k| {
                let t = k as f64;
                let angles = SVector::<f64, 22>::from_fn(|i, _| {
                    if i >= 20 {
                        return 0.0;
                    }
                    let (lo, hi) = (model.limits[(i, 0)], model.limits[(i, 1)]);
                    lo + (hi - lo) * (0.2 + 0.15 * t + 0.03 * i as f64).fract()
                });
                Pose {
                    rotation: Rotation3::from_euler_angles(0.3 * t - 0.4, 1.1 - 0.2 * t, 0.5 * t).into_inner(),
                    translation: Vector3::new(0.1 * t - 0.15, 0.05, 0.4 + 0.02 * t),
                    angles,
                }
            })
            .collect()
    }

    #[test]
    fn skinning_the_landmarks_by_their_bone_weights_reproduces_handfits_landmarks() -> Result<(), HandsError> {
        let generic = GenericHandModel::load()?;
        for phi in [1.0, 0.93] {
            let model = generic.scaled(phi);
            let rest: Vec<Vector3<f64>> = (0..NUM_LANDMARKS).map(|i| model.rest.row(i).transpose()).collect();
            let weights: Vec<[f64; SKINNING_FRAMES]> = (0..NUM_LANDMARKS)
                .map(|i| {
                    let mut row = [0.0; SKINNING_FRAMES];
                    for (slot, &bone) in handfit::generated::BONE_INDICES[i].iter().enumerate() {
                        row[bone as usize] += model.weights[(i, slot)];
                    }
                    row
                })
                .collect();
            for pose in poses(&model) {
                for side in [LEFT, RIGHT] {
                    let ours = skin_points(&model, &pose, mirror(side), &rest, &weights);
                    let handfit = landmarks_world(&model, &pose, side);
                    for i in 0..NUM_LANDMARKS {
                        let error = (ours[i] - Vector3::from(handfit[i])).norm();
                        assert!(error < 1e-9, "phi {phi} side {side} landmark {i}: {error} m");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn the_mesh_wraps_the_fitted_hand() -> Result<(), HandsError> {
        let generic = GenericHandModel::load()?;
        let mesh = HandMesh::load()?;
        assert_eq!((mesh.triangles().len(), mesh.vertices.len()), (1544, MESH_VERTICES));
        let phi = 0.97;
        let model = generic.scaled(phi);
        for pose in poses(&model) {
            for side in [LEFT, RIGHT] {
                let vertices = mesh.skin(&generic, phi, &pose, mirror(side));
                // In the rest pose the fingertips (0-4) and the palm centre (20) are mesh vertices, the joints lie 6-11 mm
                // under the skin and the wrist 21 mm: posed, the first stay on the skin and the rest inside it.
                for (i, landmark) in landmarks_world(&model, &pose, side).iter().enumerate() {
                    let nearest = vertices
                        .iter()
                        .map(|v| ((f64::from(v[0]) - landmark[0]).powi(2) + (f64::from(v[1]) - landmark[1]).powi(2) + (f64::from(v[2]) - landmark[2]).powi(2)).sqrt())
                        .fold(f64::INFINITY, f64::min);
                    let bound = if i < 5 || i == 20 { 1e-3 } else { 0.025 };
                    assert!(nearest < bound, "side {side} landmark {i}: nearest vertex {nearest} m");
                }
            }
        }
        Ok(())
    }
}
