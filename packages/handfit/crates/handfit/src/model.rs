use crate::generated::{self, landmarks_with_jacobian::sym::landmarks_with_jacobian};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

pub type Step = SVector<f64, 26>;
pub type LandmarkJacobian = SMatrix<f64, 63, 26>;

#[derive(Clone, Debug)]
pub struct Pose {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub angles: SVector<f64, 22>,
}

#[derive(Clone)]
pub struct Model {
    pub axes: SMatrix<f64, 20, 3>,
    pub pivots: SMatrix<f64, 20, 3>,
    pub rest: SMatrix<f64, 21, 3>,
    pub weights: SMatrix<f64, 21, 3>,
    pub limits: SMatrix<f64, 20, 2>,
}

/// Why a hand model cannot be fitted.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ModelError {
    /// The bone indices or weights are not 21x3.
    #[error("hand model bone indices/weights must be 21x3")]
    BoneShape,
    /// The topology is not the one baked into the generated kernels.
    #[error("hand model topology differs from the generated UmeTrack topology")]
    Topology,
    /// A non-finite value, or a joint whose lower limit lies above its upper one.
    #[error("non-finite model or reversed joint limits")]
    NonFinite,
    /// Two non-zero weights of one landmark name the same bone.
    #[error("topology: duplicate nonzero bone weights")]
    DuplicateBone,
}

impl Model {
    /// A checked model: the topology of the generated kernels ([`Model::validate_topology`]), finite values, ordered joint
    /// limits, and no bone named by two non-zero weights of one landmark.
    ///
    /// # Arguments
    ///
    /// * `indices` - The landmarks' bone slots, 21x3 row-major.
    /// * `topology` - The joint parent, frame index, first child and next sibling tables, 4x22 row-major.
    ///
    /// # Errors
    ///
    /// [`ModelError`] naming the first rule the model breaks.
    pub fn new(
        axes: SMatrix<f64, 20, 3>,
        pivots: SMatrix<f64, 20, 3>,
        rest: SMatrix<f64, 21, 3>,
        weights: SMatrix<f64, 21, 3>,
        limits: SMatrix<f64, 20, 2>,
        indices: &[i64],
        topology: &[i64],
    ) -> Result<Self, ModelError> {
        // The transpose's column-major storage is the weights in row-major order, as `indices`.
        Self::validate_topology(indices, weights.transpose().as_slice(), topology)?;
        if axes
            .iter()
            .chain(pivots.iter())
            .chain(rest.iter())
            .chain(weights.iter())
            .chain(limits.iter())
            .any(|x| !x.is_finite())
            || (0..20).any(|i| limits[(i, 0)] > limits[(i, 1)])
        {
            return Err(ModelError::NonFinite);
        }
        // Torch assigns nonzero slots, rather than summing duplicate bone indices.
        // Keep this contract explicit instead of silently accepting another blend.
        for i in 0..21 {
            for k in 0..3 {
                for l in 0..k {
                    if indices[3 * i + k] == indices[3 * i + l]
                        && weights[(i, k)] != 0.0
                        && weights[(i, l)] != 0.0
                    {
                        return Err(ModelError::DuplicateBone);
                    }
                }
            }
        }
        Ok(Self {
            axes,
            pivots,
            rest,
            weights,
            limits,
        })
    }

    /// Check every topology field baked into code generation, including unused wrist joints.
    ///
    /// A landmark's bone slot must name the generated bone only where its weight is non-zero: a zero-weight slot
    /// contributes nothing whatever bone it names (UmeTrack's synthetic profiles put bone 8 in landmark 20's third
    /// slot at weight 0, where the generic model has bone 5).
    pub fn validate_topology(
        indices: &[i64],
        weights: &[f64],
        topology: &[i64],
    ) -> Result<(), ModelError> {
        let expected: Vec<i64> = generated::BONE_INDICES.iter().flatten().copied().collect();
        let expected_topology: Vec<i64> = [
            generated::JOINT_PARENT,
            generated::JOINT_FRAME_INDEX,
            generated::JOINT_FIRST_CHILD,
            generated::JOINT_NEXT_SIBLING,
        ]
        .concat();
        if indices.len() != expected.len() || weights.len() != expected.len() {
            return Err(ModelError::BoneShape);
        }
        let bones_match: bool = indices
            .iter()
            .zip(weights)
            .zip(&expected)
            .all(|((&index, &weight), &want)| weight == 0.0 || index == want);
        if !bones_match || topology != expected_topology {
            return Err(ModelError::Topology);
        }
        Ok(())
    }

    /// World landmarks and tangent Jacobian. Numeric model geometry is never baked into the kernel.
    pub fn landmarks(
        &self,
        pose: &Pose,
        mirror: f64,
        delta: &Step,
        jacobian: Option<&mut LandmarkJacobian>,
    ) -> SVector<f64, 63> {
        landmarks_with_jacobian(
            &pose.rotation,
            &pose.translation,
            &pose.angles.fixed_rows::<20>(0).into_owned(),
            &self.axes,
            &self.pivots,
            &self.rest,
            &self.weights,
            mirror,
            delta,
            jacobian,
        )
    }
}

/// The nearest rotation (polar decomposition).
pub(crate) fn orthonormalize(rotation: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = rotation.svd(true, true);
    match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => {
            let mut fix = Matrix3::identity();
            fix[(2, 2)] = (u * v_t).determinant().signum();
            u * fix * v_t
        }
        _ => *rotation,
    }
}

/// A pose moved by a tangent step: the wrist rotation by the exponential map of `step[0..3]` (right-multiplied), the
/// translation by `step[3..6]`, each joint angle by its step and clamped to `limits`. The step is rounded to `f32` first, as
/// the reference does.
///
/// # Arguments
///
/// * `pose` - The pose to move.
/// * `step` - The 26-coordinate tangent step.
/// * `limits` - Per joint, the lower and upper angle.
///
/// # Returns
///
/// The moved pose; its two extra angles are `pose`'s.
pub fn retract(pose: &Pose, step: &Step, limits: &SMatrix<f64, 20, 2>) -> Pose {
    // Keep the reference's step quantization before the manifold retraction.
    let step = step.map(|x| x as f32 as f64);
    let v = step.fixed_rows::<3>(0).into_owned();
    let sq = v.norm_squared();
    let angle = sq.sqrt();
    let (a, b) = if angle < 1e-4 {
        (1.0 - sq / 6.0, 0.5 - sq / 24.0)
    } else {
        (angle.sin() / angle, (1.0 - angle.cos()) / sq)
    };
    let k = v.cross_matrix();
    let mut next = pose.clone();
    next.rotation = pose.rotation * (Matrix3::identity() + a * k + b * k * k);
    next.translation += step.fixed_rows::<3>(3);
    for j in 0..20 {
        next.angles[j] = (pose.angles[j] + step[j + 6]).clamp(limits[(j, 0)], limits[(j, 1)]);
    }
    next
}
