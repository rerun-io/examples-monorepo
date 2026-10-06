//! Hosted visual factors, with target-from-host poses and raw-pixel Huber loss.
//!
//! Absolute world-body increments are decoupled: translation adds in the world
//! frame and rotation left-multiplies. Relative-transform increments in the
//! reprojection Jacobian are coupled left SE(3), translation first.
//! Homogeneous landmarks use stereographic direction
//! and inverse distance. Residuals are projection minus observation.
use kornia_staging_algebra::{
    lie::{RigidTransform, Rotation3},
    Scalar,
};
use nalgebra::{
    Matrix2x3, Matrix2x4, Matrix2x6, Matrix3, Matrix4, Matrix4x2, Matrix4x3, Matrix6, Vector2,
    Vector3, Vector4,
};

/// The host-camera to target-camera transform and its two 6x6 Jacobians.
///
/// The composition is decoupled: rotation is a product, while translation is
/// `R_t^-1 (t_h - t_t)`. Its Jacobians use the decoupled, left-multiplied increment.
pub fn compute_rel_pose<S: Scalar>(
    t_w_i_h: &RigidTransform<S>,
    t_i_c_h: &RigidTransform<S>,
    t_w_i_t: &RigidTransform<S>,
    t_i_c_t: &RigidTransform<S>,
    d_rel_d_h: Option<&mut Matrix6<S>>,
    d_rel_d_t: Option<&mut Matrix6<S>>,
) -> RigidTransform<S> {
    let tmp2: RigidTransform<S> = t_i_c_t.inverse();

    // `T_t_i_h_i`.
    let t_t_i_h_i: RigidTransform<S> = RigidTransform::new(
        t_w_i_t.rotation.inverse() * t_w_i_h.rotation,
        t_w_i_t.rotation.inverse() * (t_w_i_h.translation - t_w_i_t.translation),
    );

    let tmp: RigidTransform<S> = tmp2 * t_t_i_h_i;
    let res: RigidTransform<S> = tmp * *t_i_c_h;

    let block_diagonal = |r: Matrix3<S>| {
        let mut rr = Matrix6::zeros();
        rr.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        rr.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        rr
    };

    if let Some(out) = d_rel_d_h {
        // `RR = blkdiag(R, R)` with `R = T_w_i_h.so3().inverse().matrix()`
        let r: Matrix3<S> = t_w_i_h.rotation.inverse().matrix();
        let rr = block_diagonal(r);
        *out = tmp.adjoint() * rr;
    }

    if let Some(out) = d_rel_d_t {
        // `-T_i_c_t.inverse().Adj() * RR`.
        let r: Matrix3<S> = t_w_i_t.rotation.inverse().matrix();
        let rr = block_diagonal(r);
        *out = -(tmp2.adjoint() * rr);
    }

    res
}

/// Optional borrowed column-major Jacobians and visualization output.
#[derive(Debug)]
pub struct LinearizePointOut<'a, S: Scalar> {
    /// Residual derivative against translation-first, coupled left SE(3)
    /// increments of the target-from-host transform.
    pub d_res_d_xi: Option<&'a mut Matrix2x6<S>>,
    /// Residual derivative against [direction x, direction y, inverse distance].
    pub d_res_d_p: Option<&'a mut Matrix2x3<S>>,
    /// [u,v,inverse distance in target,unchanged].
    pub proj: Option<&'a mut Vector4<S>>,
}
impl<S: Scalar> Default for LinearizePointOut<'_, S> {
    fn default() -> Self {
        Self {
            d_res_d_xi: None,
            d_res_d_p: None,
            proj: None,
        }
    }
}
/// Evaluate a hosted pixel residual and requested Jacobians with a staged camera.
/// Huber weighting is separate and acts on the returned pixel residual.
///
/// ```
/// use kornia_staging_slam::factors::{linearize_point, LinearizePointOut};
/// use kornia_staging_3d::camera::{CameraModelKind, Pinhole};
/// use nalgebra::{Matrix4, Vector2};
/// let camera = CameraModelKind::Pinhole(Pinhole::new([1.0,1.0,0.0,0.0]).unwrap());
/// let mut residual = Vector2::zeros();
/// linearize_point(&Vector2::new(2.0,3.0), &Vector2::zeros(), 0.5,
///     &Matrix4::identity(), &camera, &mut residual, &mut LinearizePointOut::default())?;
/// assert_eq!(residual, Vector2::new(-2.0,-3.0));
/// # Ok::<(), kornia_staging_3d::camera::ProjectionReject>(())
/// ```
///
/// # Errors
/// Returns the camera rejection with its unchecked pixel in `residual` and the first two `proj` entries.
/// Jacobian outputs are written only on success.
#[inline]
pub fn linearize_point<S: Scalar>(
    observation: &Vector2<S>,
    direction: &Vector2<S>,
    inverse_distance: S,
    target_from_host: &Matrix4<S>,
    camera: &kornia_staging_3d::camera::CameraModelKind<S>,
    residual: &mut Vector2<S>,
    out: &mut LinearizePointOut<'_, S>,
) -> Result<(), kornia_staging_3d::camera::ProjectionReject> {
    let (point, chart_jacobian) = if out.d_res_d_p.is_some() {
        let (point, jacobian) =
            kornia_staging_3d::pose::stereographic_unproject_with_jacobian((*direction).into());
        (point, Some(jacobian))
    } else {
        (
            kornia_staging_3d::pose::stereographic_unproject((*direction).into()),
            None,
        )
    };
    let mut p_h_3d = Vector4::from(point);
    p_h_3d[3] = inverse_distance;
    let t_t_h = target_from_host;
    let p_t_3d = t_t_h * p_h_3d;
    let mut point_j = [[S::zero(); 3]; 2];
    let needs_jacobian = out.d_res_d_xi.is_some() || out.d_res_d_p.is_some();
    let (pixel, status) = camera.project_with_status(
        [p_t_3d[0], p_t_3d[1], p_t_3d[2]],
        needs_jacobian.then_some(&mut point_j),
    );
    *residual = Vector2::from(pixel);
    if let Some(proj) = out.proj.as_deref_mut() {
        proj[0] = pixel[0];
        proj[1] = pixel[1];
    }
    status?;
    let mut jp = Matrix2x4::zeros();
    if needs_jacobian {
        jp.fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&Matrix2x3::from_row_slice(point_j.as_flattened()));
    }
    if let Some(proj) = out.proj.as_deref_mut() {
        proj[0] = residual[0];
        proj[1] = residual[1];
        proj[2] = p_t_3d[3] / p_t_3d.fixed_rows::<3>(0).norm();
    }
    *residual -= observation;
    if let Some(d_res_d_xi) = out.d_res_d_xi.as_deref_mut() {
        let mut d_point_d_xi = nalgebra::Matrix4x6::zeros();
        let mut ident = Matrix3::identity();
        ident *= inverse_distance;
        d_point_d_xi.fixed_view_mut::<3, 3>(0, 0).copy_from(&ident);
        d_point_d_xi
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-Rotation3::hat(&Vector3::new(p_t_3d[0], p_t_3d[1], p_t_3d[2]))));
        *d_res_d_xi = jp * d_point_d_xi;
    }
    if let (Some(d_res_d_p), Some(chart_jacobian)) = (out.d_res_d_p.as_deref_mut(), chart_jacobian)
    {
        let jup = Matrix4x2::from(chart_jacobian);
        let mut jpp = Matrix4x3::zeros();
        let top = t_t_h.fixed_view::<3, 4>(0, 0);
        jpp.fixed_view_mut::<3, 2>(0, 0).copy_from(&(top * jup));
        jpp.set_column(2, &t_t_h.column(3));
        *d_res_d_p = jp * jpp;
    }
    Ok(())
}
/// The robust weight and cost of one observation, accumulated in fixed order.
///
/// ```text
/// huber_weight = e < huber_thresh ? 1 : huber_thresh / e
/// obs_weight   = huber_weight / (obs_std_dev * obs_std_dev)
/// cost         = 0.5 * (2 - huber_weight) * obs_weight * res^T * res
/// ```
///
/// `e` is the norm in raw pixels. Huber weighting precedes noise scaling, so
/// 1 px at a 0.5 px deviation is a 2-sigma threshold.
/// The scalar factor multiplies each row coefficient before contraction with
/// the unscaled residual. Reassociating to scale the dot product changes rounding
/// and can change the LM acceptance test near its threshold.
#[inline]
pub fn irls_huber_cost<S: Scalar>(
    res: &Vector2<S>,
    e: S,
    huber_thresh: S,
    obs_std_dev: S,
) -> (S, S) {
    let huber_weight: S = if e < huber_thresh {
        S::one()
    } else {
        huber_thresh / e
    };
    let obs_weight: S = huber_weight / (obs_std_dev * obs_std_dev);
    // `Scalar(0.5) * (2 - huber_weight) * obs_weight` folds left into one
    // scalar, which then scales the row.
    let factor: S = S::from_literal(0.5) * (S::from_literal(2.0) - huber_weight) * obs_weight;
    let cost: S = (factor * res[0]) * res[0] + (factor * res[1]) * res[1];
    (huber_weight, cost)
}

#[cfg(test)]
mod tests;
