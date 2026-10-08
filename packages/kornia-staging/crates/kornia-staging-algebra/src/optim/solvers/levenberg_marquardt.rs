use crate::Scalar;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix, OVector};

/// Floors used to scale a normal-equation diagonal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalingFloor<S> {
    /// Floor relative to the largest diagonal entry.
    pub relative: S,
    /// Lower bound on that largest entry before relative scaling.
    pub absolute: S,
}

impl<S: Scalar> Default for ScalingFloor<S> {
    fn default() -> Self {
        Self {
            relative: S::from_literal(1e-9),
            absolute: S::from_literal(1e-12),
        }
    }
}

/// Numerical policy for an accepted-step Nielsen damping update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NielsenPolicy<S> {
    /// Smallest multiplier applied to damping.
    pub min_ratio: S,
    /// Positive floor on the predicted energy decrease.
    pub predicted_floor: S,
    /// Multiplier on the actual-to-predicted reduction ratio.
    pub growth: S,
}

impl<S: Scalar> Default for NielsenPolicy<S> {
    fn default() -> Self {
        Self {
            min_ratio: S::from_literal(1.0 / 3.0),
            predicted_floor: S::from_literal(1e-30),
            growth: S::from_literal(2.0),
        }
    }
}

/// Floor a normal-equation diagonal for Marquardt damping, before applying a variable mask.
/// # Arguments
/// * `diagonal` - Diagonal entries replaced in place by their floored values.
/// * `policy` - Relative and absolute scaling floors.
/// ```
/// use kornia_staging_algebra::optim::solvers::marquardt_scaling;
/// let mut d = nalgebra::DVector::from_vec(vec![2.0_f64, 0.0]);
/// marquardt_scaling(&mut d, Default::default());
/// assert_eq!(d[1], 2e-9);
/// ```
#[inline]
pub fn marquardt_scaling<S: Scalar, D: Dim>(diagonal: &mut OVector<S, D>, policy: ScalingFloor<S>)
where
    DefaultAllocator: Allocator<D>,
{
    let floor = policy.relative * diagonal.max().max(policy.absolute);
    diagonal
        .iter_mut()
        .for_each(|value| *value = value.max(floor));
}

/// Quadratic energy decrease for an already projected effective step.
/// # Arguments
/// * `step` - Applied tangent step.
/// * `h` - Normal-equation Hessian.
/// * `g` - Normal-equation gradient.
#[inline]
pub fn predicted_reduction<S: Scalar, D: Dim>(
    step: &OVector<S, D>,
    h: &OMatrix<S, D, D>,
    g: &OVector<S, D>,
) -> S
where
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    -(S::from_literal(2.0) * g.dot(step) + step.dot(&(h * step)))
}

/// Nielsen damping update after an accepted step.
/// # Arguments
/// * `damping` - Current nonnegative damping.
/// * `reduction` - Actual energy decrease.
/// * `predicted` - Predicted energy decrease.
/// * `policy` - Reduction floor and damping multiplier policy.
#[inline]
pub fn nielsen_damping<S: Scalar>(
    damping: S,
    reduction: S,
    predicted: S,
    policy: NielsenPolicy<S>,
) -> S {
    let ratio = reduction / predicted.max(policy.predicted_floor);
    damping * (S::one() - (policy.growth * ratio - S::one()).powi(3)).max(policy.min_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector, SVector};

    #[test]
    fn default_policy_preserves_numeric_floors() {
        let scaling = ScalingFloor::<f64>::default();
        assert_eq!(scaling.relative.to_bits(), 1e-9_f64.to_bits());
        assert_eq!(scaling.absolute.to_bits(), 1e-12_f64.to_bits());
        let damping = NielsenPolicy::<f64>::default();
        assert_eq!(damping.predicted_floor.to_bits(), 1e-30_f64.to_bits());
        assert_eq!(damping.min_ratio.to_bits(), (1.0_f64 / 3.0).to_bits());
        assert_eq!(damping.growth.to_bits(), 2.0_f64.to_bits());
    }

    #[test]
    fn diagonal_floor_and_quadratic_prediction() {
        let mut diagonal = SVector::<f64, 3>::new(4.0, 0.0, 2.0);
        marquardt_scaling(&mut diagonal, Default::default());
        assert_eq!(diagonal.as_slice(), &[4.0, 4e-9, 2.0]);
        let h = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 4.0]));
        let g = DVector::from_vec(vec![-2.0, -4.0]);
        let step = DVector::from_vec(vec![1.0, 1.0]);
        assert_eq!(predicted_reduction(&step, &h, &g), 6.0);
        assert_eq!(nielsen_damping(3.0_f64, 2.0, 2.0, Default::default()), 1.0);
    }
}
