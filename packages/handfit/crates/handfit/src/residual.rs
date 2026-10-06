use crate::{
    model::{LandmarkJacobian, Model, Pose, Step},
    Config, JacobianMode,
};
pub use kornia_staging_3d::camera::CameraModelKind;
use kornia_staging_3d::camera::{Fisheye624, InvalidCalibration, Pinhole};
use nalgebra::{Matrix3, Matrix4, RowVector3, SMatrix, SVector, Vector2, Vector3};

pub type Residual = SVector<f64, 158>;
/// A Jacobian stored by residual row: column `r` holds row `r` of `J` over the first `N` tangent coordinates (that is, `Jᵀ`).
/// Writing a row and reading it back for `JᵀJ` touch contiguous memory.
pub type JacobianRows<const N: usize> = SMatrix<f64, N, 158>;
/// The residual Jacobian over the 26 tangent coordinates, by row.
pub type Jacobian = JacobianRows<26>;
/// The residual Jacobian over the six rigid coordinates (wrist rotation, translation) of the palm-only stage, by row.
pub type RigidJacobian = JacobianRows<6>;

#[derive(Clone)]
pub struct View {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    /// Lens validated once when the view is constructed.
    pub camera: CameraModelKind<f64>,
    pub pixels: SMatrix<f64, 21, 2>,
    pub weights: SVector<f64, 21>,
    pub distances: SVector<f64, 21>,
}

/// The most views of one hand a fit takes: the residual has pixel and distance rows for two.
pub const MAX_VIEWS: usize = 2;

/// Rejection when validating a hand's view collection.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum ViewValidationError {
    /// The fixed residual cannot hold this many views.
    #[error("{views} views exceed the limit of {MAX_VIEWS}")]
    TooManyViews { views: usize },
    /// Invalid camera calibration, retaining the source view and cause.
    #[error("view {view}: {source}")]
    InvalidCalibration {
        view: usize,
        source: InvalidCalibration,
    },
}

/// The views of one hand, at most [`MAX_VIEWS`]: view `v` owns pixel rows `42 v..42 v + 42` and distance rows
/// `84 + 21 v..84 + 21 v + 21` of the [`Residual`]. No views is a valid input: a warm fit then runs on its prior alone.
#[derive(Clone, Copy)]
pub struct Views<'a> {
    views: &'a [View],
}

impl<'a> Views<'a> {
    /// The checked views of one hand.
    ///
    /// # Arguments
    ///
    /// * `views` - The hand's views, in residual order.
    ///
    /// # Errors
    /// Rejects excess views. Each view already holds a validated lens.
    pub fn new(views: &'a [View]) -> Result<Self, ViewValidationError> {
        if views.len() > MAX_VIEWS {
            return Err(ViewValidationError::TooManyViews { views: views.len() });
        }
        Ok(Self { views })
    }
    /// The validated lens cached for one view.
    pub fn camera(&self, index: usize) -> &CameraModelKind<f64> {
        &self.views[index].camera
    }
}

impl std::ops::Deref for Views<'_> {
    type Target = [View];

    fn deref(&self) -> &[View] {
        self.views
    }
}

/// The `rotation` and `translation` of a rig camera's [`View`]: camera-from-world = `cam_from_rig · world_from_rig⁻¹`, the rig
/// pose inverted rigidly (`Rᵀ`, `−Rᵀ t`).
pub fn cam_from_world(
    cam_from_rig: &Matrix4<f64>,
    world_from_rig: &Matrix4<f64>,
) -> (Matrix3<f64>, Vector3<f64>) {
    let rig_rotation = world_from_rig.fixed_view::<3, 3>(0, 0).transpose();
    let mut inverse = Matrix4::identity();
    inverse
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rig_rotation);
    inverse
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&(-rig_rotation * world_from_rig.fixed_view::<3, 1>(0, 3)));
    let cam_world = cam_from_rig * inverse;
    (
        cam_world.fixed_view::<3, 3>(0, 0).into_owned(),
        cam_world.fixed_view::<3, 1>(0, 3).into_owned(),
    )
}

/// The weighted residual of a hand pose: [84 pixel, 42 distance, 32 temporal] rows; the rows of missing views and of
/// unobserved keypoints stay zero.
///
/// # Arguments
///
/// * `model` - The hand model.
/// * `config` - `phi` and the residual weights (`dist_weight`, `temporal_weight`, `translation_unit`).
/// * `pose` - The pose to evaluate.
/// * `prior` - The temporal prior.
/// * `mirror` - +1 left hand, −1 right hand.
/// * `views` - The hand's views.
/// * `delta` - A tangent step applied to `pose` first (zero for `pose` itself).
/// * `jacobian` - Receives `Jᵀ` over the 26 tangent coordinates when given.
///
/// # Returns
///
/// The residual.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    model: &Model,
    config: &Config,
    pose: &Pose,
    prior: &Pose,
    mirror: f64,
    views: Views<'_>,
    delta: &Step,
    jacobian: Option<&mut Jacobian>,
) -> Residual {
    let Some(rows) = jacobian else {
        let points = model.landmarks(pose, mirror, delta, None);
        return assemble::<26>(config, pose, prior, views, delta, &points, None);
    };
    // `assemble` writes only the rows of observed points, distances and the prior: the rest must be zero.
    rows.fill(0.0);
    let mut landmarks_jac = LandmarkJacobian::zeros();
    let points = model.landmarks(pose, mirror, delta, Some(&mut landmarks_jac));
    assemble(
        config,
        pose,
        prior,
        views,
        delta,
        &points,
        Some((rows, &landmarks_jac)),
    )
}

/// `evaluate` at `delta = 0` with the Jacobian into buffers that one solve reuses.
///
/// `rows` must be zero outside the rows this call writes. It stays so when it starts zero and every call gets the same
/// `views`: the written rows (observed points, distance rows, temporal rows) depend only on the views' weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_into(
    model: &Model,
    config: &Config,
    pose: &Pose,
    prior: &Pose,
    mirror: f64,
    views: Views<'_>,
    landmarks_jac: &mut LandmarkJacobian,
    rows: &mut JacobianRows<26>,
) -> Residual {
    let zero = Step::zeros();
    let points = model.landmarks(pose, mirror, &zero, Some(landmarks_jac));
    assemble(
        config,
        pose,
        prior,
        views,
        &zero,
        &points,
        Some((rows, landmarks_jac)),
    )
}

/// The hand-frame landmarks (metres, mirrored) of `pose`'s finger angles: what the rigid stage moves with the wrist.
pub fn rigid_landmarks(model: &Model, pose: &Pose, mirror: f64) -> SVector<f64, 63> {
    let hand = Pose {
        rotation: Matrix3::identity(),
        translation: Vector3::zeros(),
        angles: pose.angles,
    };
    model.landmarks(&hand, mirror, &Step::zeros(), None)
}

/// `evaluate` with the fingers held: the landmarks are `pose.rotation * local + pose.translation` for the fixed hand-frame
/// points `local` (`rigid_landmarks`), so only the six rigid Jacobian columns exist and no forward kinematics run.
pub fn evaluate_rigid(
    config: &Config,
    pose: &Pose,
    prior: &Pose,
    local: &SVector<f64, 63>,
    views: Views<'_>,
    jacobian: Option<&mut RigidJacobian>,
) -> Residual {
    let Some(rows) = jacobian else {
        let points = rigid_points(pose, local, views);
        return assemble::<6>(config, pose, prior, views, &Step::zeros(), &points, None);
    };
    rows.fill(0.0);
    evaluate_rigid_into(config, pose, prior, local, views, rows)
}

/// `evaluate_rigid` into a Jacobian that one solve reuses (the contract of `evaluate_into`).
pub(crate) fn evaluate_rigid_into(
    config: &Config,
    pose: &Pose,
    prior: &Pose,
    local: &SVector<f64, 63>,
    views: Views<'_>,
    rows: &mut JacobianRows<6>,
) -> Residual {
    let points = rigid_points(pose, local, views);
    let mut landmarks_jac = SMatrix::<f64, 63, 6>::zeros();
    for i in (0..21).filter(|&i| observed(views, i)) {
        // R (I + hat(w)) p + t + dt: d/dw = -R hat(p), d/dt = I.
        let p = local.fixed_rows::<3>(3 * i).into_owned();
        landmarks_jac
            .fixed_view_mut::<3, 3>(3 * i, 0)
            .copy_from(&(-pose.rotation * p.cross_matrix()));
        landmarks_jac
            .fixed_view_mut::<3, 3>(3 * i, 3)
            .copy_from(&Matrix3::identity());
    }
    assemble(
        config,
        pose,
        prior,
        views,
        &Step::zeros(),
        &points,
        Some((rows, &landmarks_jac)),
    )
}

/// A landmark some view observes; the residual ignores the others.
fn observed(views: Views<'_>, i: usize) -> bool {
    views.iter().any(|view| view.weights[i] > 0.0)
}

/// World landmarks of the observed points (the rest stay zero and unused).
fn rigid_points(pose: &Pose, local: &SVector<f64, 63>, views: Views<'_>) -> SVector<f64, 63> {
    let mut points = SVector::<f64, 63>::zeros();
    for i in (0..21).filter(|&i| observed(views, i)) {
        points
            .fixed_rows_mut::<3>(3 * i)
            .copy_from(&(pose.rotation * local.fixed_rows::<3>(3 * i) + pose.translation));
    }
    points
}

/// Residual rows from world landmarks; the Jacobian over the first `N` tangent coordinates from the landmarks' Jacobian.
///
/// Writes only the Jacobian rows of observed points and the temporal entries; the caller provides zeros elsewhere.
fn assemble<const N: usize>(
    config: &Config,
    pose: &Pose,
    prior: &Pose,
    views: Views<'_>,
    delta: &Step,
    points: &SVector<f64, 63>,
    mut jacobian: Option<(&mut JacobianRows<N>, &SMatrix<f64, 63, N>)>,
) -> Residual {
    let mut result = Residual::zeros();
    let want = jacobian.is_some();
    for (v, view) in views.iter().enumerate() {
        let camera = |i: usize| view.rotation * points.fixed_rows::<3>(3 * i) + view.translation;
        // d(distance mm)/d(world point m), zero at the camera centre.
        let gradient = |point: &Vector3<f64>, norm: f64| {
            if norm > 0.0 {
                point.transpose() * view.rotation * (1000.0 / norm)
            } else {
                RowVector3::zeros()
            }
        };
        // Unobserved points cannot contribute, including as a distance reference.
        let reference = if view.weights[5] > 0.0 { 5 } else { 20 };
        let distance_rows = view.weights[reference] > 0.0 && config.dist_weight > 0.0;
        let reference_point = camera(reference);
        let reference_norm = reference_point.norm();
        let reference_distance = reference_norm * 1000.0;
        // The reference's distance Jacobian row, subtracted from every distance row.
        let mut reference_row = [0.0_f64; N];
        if let (true, Some((_, landmarks_jac))) = (distance_rows, jacobian.as_ref()) {
            let gradient = gradient(&reference_point, reference_norm);
            for (k, out) in reference_row.iter_mut().enumerate() {
                let column = landmarks_jac.fixed_view::<3, 1>(3 * reference, k);
                *out = gradient.dot(&column.transpose());
            }
        }
        for i in 0..21 {
            if view.weights[i] <= 0.0 {
                continue;
            }
            let point = camera(i);
            let norm = point.norm();
            let weight = view.weights[i].sqrt();
            let mut proj_jac = SMatrix::<f64, 2, 3>::zeros();
            let pixels = project_camera(views.camera(v), &point, want.then_some(&mut proj_jac));
            let pixel_row = 42 * v + 2 * i;
            result
                .fixed_rows_mut::<2>(pixel_row)
                .copy_from(&((pixels - view.pixels.row(i).transpose()) * weight));
            let distance_row = 84 + 21 * v + i;
            let distance_weight =
                (config.dist_weight * view.weights[i] * view.weights[reference]).sqrt();
            if distance_rows {
                result[distance_row] = distance_weight
                    * (norm * 1000.0
                        - reference_distance
                        - config.phi * (view.distances[i] - view.distances[reference]));
            }
            if let Some((j, landmarks_jac)) = jacobian.as_mut() {
                // Chain through the camera rotation once per point, then one pass over the N tangent columns.
                let pixel = proj_jac * view.rotation * weight;
                let gradient = gradient(&point, norm);
                for k in 0..N {
                    let column = landmarks_jac.fixed_view::<3, 1>(3 * i, k).into_owned();
                    j[(k, pixel_row)] = pixel.row(0).dot(&column.transpose());
                    j[(k, pixel_row + 1)] = pixel.row(1).dot(&column.transpose());
                    if distance_rows {
                        j[(k, distance_row)] = (gradient.dot(&column.transpose())
                            - reference_row[k])
                            * distance_weight;
                    }
                }
            }
        }
    }
    let weight = config.temporal_weight.sqrt();
    let rotation = pose.rotation * (Matrix3::identity() + delta.fixed_rows::<3>(0).cross_matrix());
    for row in 0..3 {
        for col in 0..3 {
            result[126 + 3 * row + col] =
                (rotation[(row, col)] - prior.rotation[(row, col)]) * weight / 2.0_f64.sqrt();
        }
    }
    for i in 0..3 {
        result[135 + i] = (pose.translation[i] + delta[3 + i] - prior.translation[i]) * weight
            / config.translation_unit;
    }
    for i in 0..20 {
        result[138 + i] = (pose.angles[i] + delta[6 + i] - prior.angles[i]) * weight;
    }
    if let Some((j, _)) = jacobian {
        for k in 0..3 {
            let mut axis = Vector3::zeros();
            axis[k] = 1.0;
            let derivative = pose.rotation * axis.cross_matrix() * weight / 2.0_f64.sqrt();
            for row in 0..3 {
                for col in 0..3 {
                    j[(k, 126 + 3 * row + col)] = derivative[(row, col)];
                }
            }
            j[(3 + k, 135 + k)] = weight / config.translation_unit;
        }
        for i in 0..N.saturating_sub(6) {
            j[(6 + i, 138 + i)] = weight;
        }
    }
    result
}

/// `J^T J` and `J^T r` over the rows that can be non-zero: each view's pixel and distance rows of its observed landmarks,
/// and the temporal rows when they are weighted. The rest of `J` is zero by construction (`assemble`).
///
/// A landmark's three rows in a view (two pixel rows, one distance row) touch the same few columns: the six rigid ones and the
/// landmark's finger chain, about ten of the 26. They are added together over those columns only, several times fewer
/// products than the dense `J^T J`, which they equal up to rounding.
pub fn normal_equations<const N: usize>(
    config: &Config,
    views: Views<'_>,
    r: &Residual,
    rows: &JacobianRows<N>,
) -> (SMatrix<f64, N, N>, SVector<f64, N>) {
    // Upper triangle, row-major.
    let mut upper = [[0.0_f64; N]; N];
    let mut g = SVector::<f64, N>::zeros();
    let mut add = |group: [usize; 3], size: usize| {
        let mut support = [0_usize; N];
        let mut values = [[0.0_f64; 3]; N];
        let mut count = 0;
        for k in 0..N {
            let mut column = [0.0; 3];
            for (slot, &row) in group[..size].iter().enumerate() {
                column[slot] = rows[(k, row)];
            }
            if column != [0.0; 3] {
                support[count] = k;
                values[count] = column;
                count += 1;
            }
        }
        let residual = [
            r[group[0]],
            if size > 1 { r[group[1]] } else { 0.0 },
            if size > 2 { r[group[2]] } else { 0.0 },
        ];
        for p in 0..count {
            let (a, x) = (support[p], values[p]);
            g[a] += x[0] * residual[0] + x[1] * residual[1] + x[2] * residual[2];
            let line = &mut upper[a];
            for q in p..count {
                let y = values[q];
                line[support[q]] += x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
            }
        }
    };
    for (v, view) in views.iter().enumerate() {
        // An unobserved landmark's rows are zero.
        for i in (0..21).filter(|&i| view.weights[i] > 0.0) {
            add([42 * v + 2 * i, 42 * v + 2 * i + 1, 84 + 21 * v + i], 3);
        }
    }
    if config.temporal_weight != 0.0 {
        for row in 126..158 {
            add([row; 3], 1);
        }
    }
    let h = SMatrix::<f64, N, N>::from_fn(|a, b| if a <= b { upper[a][b] } else { upper[b][a] });
    (h, g)
}

/// The residual at `pose` and its Jacobian over the 26 tangent coordinates.
///
/// # Arguments
///
/// * `model`, `config`, `pose`, `prior`, `mirror`, `views` - As [`evaluate`].
/// * `mode` - The analytic Jacobian, or central differences with handtrack's steps (3e-4 for the translation, 3e-3 for
///   the rest, rounded to `f32`).
///
/// # Returns
///
/// The residual and the Jacobian, stored by row (`Jᵀ`, see [`JacobianRows`]).
pub fn linearize(
    model: &Model,
    config: &Config,
    pose: &Pose,
    prior: &Pose,
    mirror: f64,
    views: Views<'_>,
    mode: JacobianMode,
) -> (Residual, Jacobian) {
    let mut jac = Jacobian::zeros();
    let zero = Step::zeros();
    if mode == JacobianMode::Analytic {
        let residual = evaluate(
            model,
            config,
            pose,
            prior,
            mirror,
            views,
            &zero,
            Some(&mut jac),
        );
        return (residual, jac);
    }
    let residual = evaluate(model, config, pose, prior, mirror, views, &zero, None);
    for k in 0..26 {
        let h = if (3..6).contains(&k) {
            3e-4_f32 as f64
        } else {
            3e-3_f32 as f64
        };
        let mut step = zero;
        step[k] = h;
        let plus = evaluate(model, config, pose, prior, mirror, views, &step, None);
        step[k] = -h;
        let minus = evaluate(model, config, pose, prior, mirror, views, &step, None);
        jac.row_mut(k)
            .copy_from(&((plus - minus) / (2.0 * h)).transpose());
    }
    (residual, jac)
}

/// Construct the validated Fisheye62 or pinhole lens used by hand fitting.
/// # Arguments
/// * `focal`, `principal` - Focal lengths and principal point in pixels.
/// * `distortion` - Fisheye62 radial and tangential coefficients, or pinhole.
/// # Errors
/// Returns the staged camera's calibration error.
pub fn camera_model(
    focal: &Vector2<f64>,
    principal: &Vector2<f64>,
    distortion: Option<&SVector<f64, 8>>,
) -> Result<CameraModelKind<f64>, InvalidCalibration> {
    let head = [focal.x, focal.y, principal.x, principal.y];
    Ok(if let Some(d) = distortion {
        CameraModelKind::Fisheye624(Fisheye624::fisheye62(
            head,
            [d[0], d[1], d[2], d[3], d[4], d[5]],
            [d[6], d[7]],
        )?)
    } else {
        CameraModelKind::Pinhole(Pinhole::new(head)?)
    })
}

/// Unclipped projection under handtrack's lens policy, using a prevalidated camera.
/// # Arguments
/// * `camera` - Immutable lens constructed once for the calibration.
/// * `point` - Camera-frame point in metres.
/// * `jacobian` - Optional pixel derivative with respect to the point.
pub fn project_camera(
    camera: &CameraModelKind<f64>,
    point: &Vector3<f64>,
    jacobian: Option<&mut SMatrix<f64, 2, 3>>,
) -> Vector2<f64> {
    if let CameraModelKind::Pinhole(pinhole) = camera {
        return handtrack_pinhole_project(pinhole, point, jacobian);
    }
    let mut derivative = [[0.0; 3]; 2];
    let output = camera.project_unchecked_with_point_jacobian(
        [point.x, point.y, point.z],
        jacobian.as_ref().map(|_| &mut derivative),
    );
    if let Some(j) = jacobian {
        *j = SMatrix::from_row_slice(derivative.as_flattened());
    }
    Vector2::from(output)
}

/// Pinhole projection with handtrack's positive depth clamp and flat depth derivative.
/// # Arguments
/// * `camera` - Validated pinhole calibration.
/// * `point` - Camera-frame point; depths with magnitude below 1e-9 are clamped to 1e-9.
/// * `jacobian` - Optional pixel derivative; its depth column is zero when clamped.
pub fn handtrack_pinhole_project(
    camera: &Pinhole<f64>,
    point: &Vector3<f64>,
    jacobian: Option<&mut SMatrix<f64, 2, 3>>,
) -> Vector2<f64> {
    use kornia_staging_3d::camera::CameraModel;
    let clamped = point.z.abs() < 1e-9;
    let z = if clamped { 1e-9 } else { point.z };
    let mut derivative = [[0.0; 3]; 2];
    let output = camera.project_unchecked_with_jacobians(
        [point.x, point.y, z],
        jacobian.as_ref().map(|_| &mut derivative),
        None,
    );
    if let Some(j) = jacobian {
        if clamped {
            derivative[0][2] = 0.0;
            derivative[1][2] = 0.0;
        }
        *j = SMatrix::from_row_slice(derivative.as_flattened());
    }
    Vector2::from(output)
}
