//! Scalar-preserving chart and triangulation arithmetic.
use kornia_staging_algebra::Scalar;
use nalgebra::{Matrix3, Matrix3x4, Matrix4, Vector3, Vector4};

const SVD_MAX_ITERATIONS: usize = 64;

pub(super) fn triangulate<S: Scalar>(
    f0: &Vector3<S>,
    f1: &Vector3<S>,
    f1_in_0: &Vector3<S>,
    p2: Matrix3x4<S>,
) -> Option<Vector4<S>> {
    if f0.cross(f1_in_0).iter().all(|value| *value == S::zero()) {
        return None;
    }
    let p1: Matrix3x4<S> = {
        let mut m = Matrix3x4::zeros();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        m
    };
    let mut a: Matrix4<S> = Matrix4::zeros();
    a.row_mut(0)
        .copy_from(&(p1.row(2) * f0[0] - p1.row(0) * f0[2]));
    a.row_mut(1)
        .copy_from(&(p1.row(2) * f0[1] - p1.row(1) * f0[2]));
    a.row_mut(2)
        .copy_from(&(p2.row(2) * f1[0] - p2.row(0) * f1[2]));
    a.row_mut(3)
        .copy_from(&(p2.row(2) * f1[1] - p2.row(1) * f1[2]));

    let wide: Matrix4<f64> = a.map(|value| value.to_f64());
    if wide.iter().any(|value| !value.is_finite()) {
        return None;
    }

    // `max_niter` bounds the total implicit-shift sweeps: a 4x4 that has not
    // converged in `SVD_MAX_ITERATIONS` is not going to, and a landmark that
    // does not exist is a better answer than an unbounded loop in the tracker.
    let svd: nalgebra::SVD<f64, nalgebra::U4, nalgebra::U4> =
        nalgebra::SVD::try_new_unordered(wide, false, true, f64::EPSILON, SVD_MAX_ITERATIONS)?;
    if svd.singular_values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    // A DLT whose largest singular value is zero carries no constraint at all —
    // two zero bearing vectors build a zero `A` — so *every* direction is a null
    // direction and the one the decomposition happens to return is fabricated.
    if svd.singular_values.max() <= 0.0 {
        return None;
    }
    // The null vector is the right-singular vector of the *smallest* singular
    // value, and `v_t` holds the right-singular vectors as its **rows**. The
    // decomposition is unordered, so the row is found rather than assumed.
    let mut smallest: usize = 0;
    for i in 1..4 {
        if svd.singular_values[i] < svd.singular_values[smallest] {
            smallest = i;
        }
    }
    let v_t: nalgebra::Matrix4<f64> = svd.v_t?;

    let mut world_point: Vector4<S> = v_t
        .row(smallest)
        .transpose()
        .map(|value| crate::camera::c::<S>(value));
    let norm: S = world_point.fixed_rows::<3>(0).norm();
    // A homogeneous vector with no spatial part has no direction: dividing by
    // its norm used to hand the caller `[NaN, NaN, NaN, inf]`.
    if norm <= S::zero() {
        return None;
    }
    for i in 0..4 {
        world_point[i] /= norm;
    }

    // `if (f0.dot(worldPoint.head<3>()) < 0) worldPoint *= -1`.
    let dot: S = f0[0] * world_point[0] + f0[1] * world_point[1] + f0[2] * world_point[2];
    if dot < S::zero() {
        world_point = -world_point;
    }
    Some(world_point)
}
