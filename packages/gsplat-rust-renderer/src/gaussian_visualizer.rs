//! Custom visualizer glue.
//!
//! This file is the Rerun-facing middle layer:
//! query the `GaussianSplats3D` archetype, rebuild a renderer-facing cloud once, compute a visible
//! candidate set from the active 3D camera, and hand that batch to the custom renderer.

use std::collections::HashMap;
use std::sync::Arc;

use crate::gaussian_renderer::GaussianDrawData;
use glam::{Affine3A, Mat2, Mat3, Mat4, Quat, Vec2, Vec3};
use rayon::prelude::*;
use re_view::{DataResultQuery as _, VisualizerInstructionQueryResults};
use re_view_spatial::{SpatialViewState, TransformTreeContext};
use re_viewer_context::{
    AppOptions, IdentifiedViewSystem, ViewContext, ViewContextCollection, ViewQuery,
    ViewSystemExecutionError, ViewSystemIdentifier, VisualizerExecutionOutput, VisualizerQueryInfo,
    VisualizerSystem,
};
use rerun::{Archetype as _, Component as _};
const MAX_SPLATS_RENDERED: usize = 200_000;
const MIN_RADIUS_PX: f32 = 0.35;
const OPACITY_SCALE: f32 = 1.0;
const SIGMA_COVERAGE: f32 = 3.0;
const PARALLEL_SPLAT_THRESHOLD: usize = 16_384;
const SH_C0: f32 = 0.282_094_8;
const BRUSH_COVARIANCE_BLUR_PX: f32 = 0.3;
const BRUSH_VISIBILITY_ALPHA_THRESHOLD: f32 = 1.0 / 255.0;

struct GaussianSplats3D;

impl rerun::Archetype for GaussianSplats3D {
    fn name() -> rerun::ArchetypeName {
        "GaussianSplats3D".into()
    }

    fn display_name() -> &'static str {
        "Gaussian Splats 3D"
    }

    fn required_components() -> std::borrow::Cow<'static, [rerun::ComponentDescriptor]> {
        vec![
            Self::descriptor_centers(),
            Self::descriptor_quaternions(),
            Self::descriptor_scales(),
            Self::descriptor_opacities(),
            Self::descriptor_colors(),
        ]
        .into()
    }

    fn optional_components() -> std::borrow::Cow<'static, [rerun::ComponentDescriptor]> {
        vec![Self::descriptor_sh_coefficients()].into()
    }
}

impl GaussianSplats3D {
    fn descriptor_centers() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:centers".into(),
            component_type: Some(rerun::components::Translation3D::name()),
        }
    }

    fn descriptor_quaternions() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:quaternions".into(),
            component_type: Some(rerun::components::RotationQuat::name()),
        }
    }

    fn descriptor_scales() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:scales".into(),
            component_type: Some(rerun::components::Scale3D::name()),
        }
    }

    fn descriptor_opacities() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:opacities".into(),
            component_type: Some(rerun::components::Opacity::name()),
        }
    }

    fn descriptor_colors() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:colors".into(),
            component_type: Some(rerun::components::Color::name()),
        }
    }

    fn descriptor_sh_coefficients() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:sh_coefficients".into(),
            component_type: Some(rerun::components::TensorData::name()),
        }
    }
}

#[derive(Default)]
pub struct GaussianSplatVisualizer {
    // One cached render cloud per entity path. Unlike the earlier fixed-path demo, the visualizer
    // now renders any entity that logs the Gaussian component contract.
    clouds: HashMap<String, CachedCloud>,
}

struct CachedCloud {
    signature: CloudSignature,
    cloud: Arc<RenderGaussianCloud>,
}

#[derive(Clone, PartialEq, Eq)]
struct CloudSignature {
    expected_splats: usize,
    sh_coeffs_per_channel: Option<usize>,
    transform_bits: [u32; 12],
}

#[derive(Clone, Debug)]
pub(crate) struct RenderShCoefficients {
    /// Number of SH coefficients per channel, including DC.
    pub coeffs_per_channel: usize,
    /// Flat `[splat][coefficient][channel]` SH storage.
    pub coefficients: Arc<[f32]>,
}

#[derive(Clone, Debug)]
pub(crate) struct RenderGaussianCloud {
    /// Transformed world-space means used by both CPU and compute paths.
    pub means_world: Arc<[Vec3]>,
    pub quats: Arc<[Quat]>,
    pub scales: Arc<[Vec3]>,
    pub opacities: Arc<[f32]>,
    pub colors_dc: Arc<[[f32; 3]]>,
    pub sh_coeffs: Option<RenderShCoefficients>,
    /// Loose bounds only used for the startup / fallback camera path.
    pub bounds_world: Option<(Vec3, Vec3)>,
}

impl RenderGaussianCloud {
    pub(crate) fn len(&self) -> usize {
        self.means_world.len()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CameraApproximation {
    /// Right-handed view transform expected by the splat math.
    pub view_from_world: Affine3A,
    /// Perspective projection used for candidate culling and CPU fallback projection.
    pub projection_from_view: Mat4,
    pub world_position: Vec3,
    pub viewport_size_px: Vec2,
    pub near_plane: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SortedSplatIndex {
    /// Index back into the cached render cloud arrays.
    pub splat_index: u32,
    /// Positive distance from camera used for back-to-front ordering.
    pub camera_depth: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProjectedGaussian {
    pub center_ndc: Vec2,
    pub ndc_depth: f32,
    pub inv_cov_ndc_xx_xy_yy: [f32; 3],
    pub radius_ndc: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedSplat {
    pub center_ndc: Vec2,
    pub ndc_depth: f32,
    pub inv_cov_ndc_xx_xy_yy: [f32; 3],
    pub radius_ndc: f32,
    pub color_rgb: [f32; 3],
    pub opacity: f32,
}

impl IdentifiedViewSystem for GaussianSplatVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        "GaussianSplats3D".into()
    }
}

impl VisualizerSystem for GaussianSplatVisualizer {
    fn visualizer_query_info(&self, _app_options: &AppOptions) -> VisualizerQueryInfo {
        VisualizerQueryInfo::from_archetype::<GaussianSplats3D>()
    }

    fn execute(
        &mut self,
        ctx: &ViewContext<'_>,
        query: &ViewQuery<'_>,
        context_systems: &ViewContextCollection,
    ) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
        let mut output = VisualizerExecutionOutput::default();
        let transforms = context_systems.get::<TransformTreeContext>(&output)?;
        let mut draw_data = GaussianDrawData::new(ctx.render_ctx());
        let mut extra_draw_data = Vec::new();
        let mut active_labels = Vec::new();

        for (data_result, instruction) in query.iter_visualizer_instruction_for(Self::identifier())
        {
            // Query any logged `GaussianSplats3D` entity and rebuild the renderer-facing cloud
            // only when the recorded data or transform signature changes.

            let results = data_result.query_archetype_with_history::<GaussianSplats3D>(
                ctx,
                query,
                instruction,
            );
            let results = VisualizerInstructionQueryResults::new(instruction, &results, &output);

            let centers = results.iter_required(GaussianSplats3D::descriptor_centers().component);
            if centers.is_empty() {
                continue;
            }

            let quaternions =
                results.iter_required(GaussianSplats3D::descriptor_quaternions().component);
            let scales = results.iter_required(GaussianSplats3D::descriptor_scales().component);
            let opacities =
                results.iter_required(GaussianSplats3D::descriptor_opacities().component);
            let colors = results.iter_required(GaussianSplats3D::descriptor_colors().component);
            let expected_splats = count_splats_in_results(centers.slice::<[f32; 3]>());

            let latest_at_query = query.latest_at_query();
            let latest_at_results = data_result
                .latest_at_with_blueprint_resolved_data::<GaussianSplats3D>(
                    ctx,
                    &latest_at_query,
                    Some(instruction),
                );
            let sh_coefficients = extract_sh_coefficients(&latest_at_results, expected_splats)
                .map_err(|err| {
                    ViewSystemExecutionError::DrawDataCreationError(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        err,
                    )))
                })?;

            let transform = transforms
                .target_from_entity_path(data_result.entity_path.hash())
                .and_then(|result| result.as_ref().ok())
                .map(|transform_info| {
                    transform_info
                        .single_transform_required_for_entity(
                            &data_result.entity_path,
                            GaussianSplats3D::name(),
                        )
                        .as_affine3a()
                })
                .unwrap_or(Affine3A::IDENTITY);

            let label = format!("gaussian_splats::{}", data_result.entity_path);
            let signature = CloudSignature {
                expected_splats,
                sh_coeffs_per_channel: sh_coefficients.as_ref().map(|sh| sh.coeffs_per_channel),
                transform_bits: transform.to_cols_array().map(f32::to_bits),
            };
            active_labels.push(label.clone());

            let cache_entry = self
                .clouds
                .entry(label.clone())
                .or_insert_with(|| CachedCloud {
                    signature: signature.clone(),
                    cloud: Arc::new(build_render_cloud(
                        centers.slice::<[f32; 3]>(),
                        quaternions.slice::<[f32; 4]>(),
                        scales.slice::<[f32; 3]>(),
                        opacities.slice::<f32>(),
                        colors.slice::<u32>(),
                        transform,
                        sh_coefficients.clone(),
                    )),
                });

            if cache_entry.signature != signature {
                cache_entry.signature = signature;
                cache_entry.cloud = Arc::new(build_render_cloud(
                    centers.slice::<[f32; 3]>(),
                    quaternions.slice::<[f32; 4]>(),
                    scales.slice::<[f32; 3]>(),
                    opacities.slice::<f32>(),
                    colors.slice::<u32>(),
                    transform,
                    sh_coefficients,
                ));
            }

            let cloud = cache_entry.cloud.clone();
            // Camera extraction prefers the active 3D view state, then a simple bounds-based
            // fallback so the example still works before the user interacts with the camera.
            let camera =
                camera_from_view(ctx, query).unwrap_or_else(|| fallback_camera(cloud.bounds_world));
            let mut visible = Vec::new();
            // Visibility is conservative: quickly reject obviously invisible splats, then keep the
            // surviving candidates sorted back-to-front for the renderer.
            rebuild_visible_indices(&mut visible, &cloud, &camera);
            let farthest_depth = visible.first().map_or(0.0, |visible| visible.camera_depth);

            let submission = draw_data.add_batch(
                ctx.render_ctx(),
                &label,
                &cloud,
                &camera,
                &visible,
                farthest_depth,
            );
            if let Some(extra) = submission.extra_draw_data {
                extra_draw_data.push(extra);
            }
        }

        self.clouds.retain(|label, _| active_labels.contains(label));
        output.draw_data = vec![draw_data.into()];
        output.draw_data.extend(extra_draw_data);
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_render_cloud<'a, Idx, ICenters, IQuaternions, IScales, IOpacities, IColors>(
    centers: ICenters,
    quaternions: IQuaternions,
    scales: IScales,
    opacities: IOpacities,
    colors: IColors,
    transform: Affine3A,
    sh_coefficients: Option<RenderShCoefficients>,
) -> RenderGaussianCloud
where
    Idx: Ord,
    ICenters: IntoIterator<Item = (Idx, &'a [[f32; 3]])>,
    IQuaternions: IntoIterator<Item = (Idx, &'a [[f32; 4]])>,
    IScales: IntoIterator<Item = (Idx, &'a [[f32; 3]])>,
    IOpacities: IntoIterator<Item = (Idx, &'a [f32])>,
    IColors: IntoIterator<Item = (Idx, &'a [u32])>,
{
    // Convert the recorded component batches back into the single packed cloud layout the
    // renderer expects. The visualizer is the only place that knows about Rerun query results.
    let mut means_world = Vec::new();
    let mut quats_world = Vec::new();
    let mut scales_world = Vec::new();
    let mut opacities_world = Vec::new();
    let mut colors_world = Vec::new();

    for (_index, centers, quats, scales, opacities, colors) in
        re_query::range_zip_1x4(centers, quaternions, scales, opacities, colors)
    {
        let quats = quats.unwrap_or_default();
        let scales = scales.unwrap_or_default();
        let opacities = opacities.unwrap_or_default();
        let colors = colors.unwrap_or_default();

        let count = centers
            .len()
            .min(quats.len())
            .min(scales.len())
            .min(opacities.len())
            .min(colors.len());

        for row_index in 0..count {
            means_world.push(transform.transform_point3(Vec3::from_array(centers[row_index])));
            quats_world.push(normalize_quat_or_identity(Quat::from_xyzw(
                quats[row_index][0],
                quats[row_index][1],
                quats[row_index][2],
                quats[row_index][3],
            )));
            scales_world.push(Vec3::from_array(scales[row_index]).max(Vec3::splat(1e-4)));
            opacities_world.push(opacities[row_index].clamp(0.0, 1.0));
            colors_world.push(rerun_color_to_rgb(colors[row_index]));
        }
    }

    let bounds_world = approximate_bounds_from_points(&means_world);

    RenderGaussianCloud {
        means_world: Arc::from(means_world),
        quats: Arc::from(quats_world),
        scales: Arc::from(scales_world),
        opacities: Arc::from(opacities_world),
        colors_dc: Arc::from(colors_world),
        sh_coeffs: sh_coefficients,
        bounds_world,
    }
}

fn rerun_color_to_rgb(color: u32) -> [f32; 3] {
    let [r, g, b, _a] = re_sdk_types::datatypes::Rgba32::from_u32(color).to_array();
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}

fn approximate_bounds_from_points(points: &[Vec3]) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for point in points {
        min = min.min(*point);
        max = max.max(*point);
    }
    min.is_finite().then_some((min, max))
}

fn rebuild_visible_indices(
    visible_indices: &mut Vec<SortedSplatIndex>,
    cloud: &RenderGaussianCloud,
    camera: &CameraApproximation,
) {
    // This is intentionally an approximate first pass. The compute renderer does the exact
    // projection later; here we only want a sorted candidate list that is cheap to build.
    visible_indices.clear();
    if cloud.len() >= PARALLEL_SPLAT_THRESHOLD {
        *visible_indices = cloud
            .means_world
            .par_iter()
            .enumerate()
            .filter_map(|(index, mean_world)| {
                build_compute_candidate(index, *mean_world, cloud, camera)
            })
            .collect();
    } else {
        visible_indices.extend(cloud.means_world.iter().enumerate().filter_map(
            |(index, mean_world)| build_compute_candidate(index, *mean_world, cloud, camera),
        ));
    }

    if visible_indices.len() > MAX_SPLATS_RENDERED {
        let nth = MAX_SPLATS_RENDERED.saturating_sub(1);
        visible_indices.select_nth_unstable_by(nth, |left, right| {
            left.camera_depth
                .partial_cmp(&right.camera_depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        visible_indices.truncate(MAX_SPLATS_RENDERED);
    }

    if visible_indices.len() >= PARALLEL_SPLAT_THRESHOLD {
        visible_indices.par_sort_unstable_by(|left, right| {
            right
                .camera_depth
                .partial_cmp(&left.camera_depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        visible_indices.sort_by(|left, right| {
            right
                .camera_depth
                .partial_cmp(&left.camera_depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

fn build_compute_candidate(
    index: usize,
    mean_world: Vec3,
    cloud: &RenderGaussianCloud,
    camera: &CameraApproximation,
) -> Option<SortedSplatIndex> {
    // Reject splats that are clearly invisible before the renderer spends time projecting them.
    let opacity = cloud.opacities[index] * OPACITY_SCALE;
    if opacity <= 0.0 {
        return None;
    }

    let mean_view = camera.view_from_world.transform_point3(mean_world);
    let camera_depth = -mean_view.z;
    if !camera_depth.is_finite() || camera_depth <= camera.near_plane {
        return None;
    }

    let clip = camera.projection_from_view * mean_view.extend(1.0);
    if !clip.is_finite() || clip.w.abs() <= 1e-6 {
        return None;
    }

    let center_ndc = clip.truncate() / clip.w;
    if center_ndc.x.abs() > 1.5 || center_ndc.y.abs() > 1.5 {
        return None;
    }

    Some(SortedSplatIndex {
        splat_index: index as u32,
        camera_depth,
    })
}

fn count_splats_in_results<'a, I, Idx>(centers: I) -> usize
where
    I: IntoIterator<Item = (Idx, &'a [[f32; 3]])>,
{
    centers
        .into_iter()
        .map(|(_index, positions)| positions.len())
        .sum()
}

fn extract_sh_coefficients(
    latest_at_results: &re_view::BlueprintResolvedLatestAtResults<'_>,
    expected_splats: usize,
) -> Result<Option<RenderShCoefficients>, String> {
    let Some(tensor) = latest_at_results.get_mono::<re_sdk_types::components::TensorData>(
        GaussianSplats3D::descriptor_sh_coefficients().component,
    ) else {
        return Ok(None);
    };

    let shape = tensor.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(format!(
            "invalid SH tensor shape {:?}: expected [num_splats, coeffs_per_channel, 3]",
            shape
        ));
    }

    let tensor_splats = usize::try_from(shape[0])
        .map_err(|_| format!("SH tensor splat count {} does not fit into usize", shape[0]))?;
    if tensor_splats != expected_splats {
        return Err(format!(
            "invalid SH tensor shape {:?}: expected {} splats, got {}",
            shape, expected_splats, tensor_splats
        ));
    }

    let coeffs_per_channel = usize::try_from(shape[1]).map_err(|_| {
        format!(
            "SH tensor coefficient count {} does not fit into usize",
            shape[1]
        )
    })?;
    if sh_degree_from_coeffs(coeffs_per_channel).is_none() {
        return Err(format!(
            "unsupported SH tensor coefficient count {coeffs_per_channel}"
        ));
    }

    let coefficients: Arc<[f32]> = match &tensor.buffer {
        re_sdk_types::datatypes::TensorBuffer::F32(values) => Arc::from(values.as_ref()),
        other => {
            return Err(format!(
                "invalid SH tensor dtype {:?}: expected Float32",
                other.dtype()
            ));
        }
    };

    let expected_len = expected_splats
        .checked_mul(coeffs_per_channel)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "SH tensor size overflow".to_owned())?;
    if coefficients.len() != expected_len {
        return Err(format!(
            "invalid SH tensor payload: expected {} floats, got {}",
            expected_len,
            coefficients.len()
        ));
    }

    Ok(Some(RenderShCoefficients {
        coeffs_per_channel,
        coefficients,
    }))
}

fn camera_from_view(ctx: &ViewContext<'_>, query: &ViewQuery<'_>) -> Option<CameraApproximation> {
    camera_from_spatial_view_state(ctx, query)
}

fn camera_from_spatial_view_state(
    ctx: &ViewContext<'_>,
    query: &ViewQuery<'_>,
) -> Option<CameraApproximation> {
    let spatial_view_state = ctx.view_state.as_any().downcast_ref::<SpatialViewState>()?;
    let eye = spatial_view_state.state_3d.eye_state.last_eye?;
    let vertical_fov = eye.fov_y?;
    let viewport_size_px = published_viewport_size_px(ctx, query)?;
    let aspect_ratio = (viewport_size_px.x / viewport_size_px.y.max(1.0)).max(1e-4);
    let near_plane = eye.near();

    Some(CameraApproximation {
        view_from_world: Affine3A::from_mat4(eye.world_from_rub_view.inverse().to_mat4()),
        projection_from_view: Mat4::perspective_infinite_rh(vertical_fov, aspect_ratio, near_plane),
        world_position: eye.pos_in_world(),
        viewport_size_px,
        near_plane,
    })
}

fn published_viewport_size_px(ctx: &ViewContext<'_>, query: &ViewQuery<'_>) -> Option<Vec2> {
    let view_info = ctx.egui_ctx().memory_mut(|memory| {
        memory
            .caches
            .cache::<re_viewer_context::ViewRectPublisher>()
            .get(&query.view_id)
            .cloned()
    })?;
    let rect = view_info.rect.shrink(2.5);
    if !rect.is_positive() {
        return None;
    }
    let viewport_size_px = rect.size() * ctx.egui_ctx().pixels_per_point();
    Some(Vec2::new(
        viewport_size_px.x.max(1.0),
        viewport_size_px.y.max(1.0),
    ))
}

fn fallback_camera(bounds: Option<(Vec3, Vec3)>) -> CameraApproximation {
    let (center, extent) = bounds
        .map(|(min, max)| (0.5 * (min + max), max - min))
        .unwrap_or((Vec3::ZERO, Vec3::ONE));
    let distance = extent.length().max(1.0) * 1.5;
    make_camera_approximation(
        center + Vec3::new(distance, distance * 0.5, distance),
        center,
        Vec3::Y,
        55.0_f32.to_radians(),
        Vec2::new(1600.0, 900.0),
        0.01,
    )
}

fn make_camera_approximation(
    world_position: Vec3,
    look_at: Vec3,
    up: Vec3,
    vertical_fov: f32,
    viewport_size_px: Vec2,
    near_plane: f32,
) -> CameraApproximation {
    let view_from_world_mat = Mat4::look_at_rh(world_position, look_at, up);
    let aspect_ratio = (viewport_size_px.x / viewport_size_px.y.max(1.0)).max(1e-4);
    let projection_from_view =
        Mat4::perspective_infinite_rh(vertical_fov, aspect_ratio, near_plane);
    CameraApproximation {
        view_from_world: Affine3A::from_mat4(view_from_world_mat),
        projection_from_view,
        world_position,
        viewport_size_px,
        near_plane,
    }
}

pub(crate) fn normalize_quat_or_identity(quat: Quat) -> Quat {
    if quat.length_squared() > 1e-12 {
        quat.normalize()
    } else {
        Quat::IDENTITY
    }
}

pub(crate) fn project_gaussian_to_ndc(
    camera: &CameraApproximation,
    mean_world: Vec3,
    quat_world: Quat,
    scales_world: Vec3,
    opacity: f32,
) -> Option<ProjectedGaussian> {
    // CPU projection mirrors the Brush-style local linearization used in the working repo:
    // build a 3D covariance from rotation+scale, project it into pixel space around the mean,
    // derive a conservative footprint, then convert that back into the NDC representation used by
    // the instanced-quad draw path.
    let mean_view = camera.view_from_world.transform_point3(mean_world);
    let camera_depth = -mean_view.z;
    if !camera_depth.is_finite() || camera_depth <= camera.near_plane {
        return None;
    }

    let opacity = opacity.clamp(0.0, 1.0);
    if opacity < BRUSH_VISIBILITY_ALPHA_THRESHOLD {
        return None;
    }

    let viewport_size_px = camera.viewport_size_px.max(Vec2::ONE);
    let pixel_center = viewport_size_px * 0.5;
    let focal_ndc = Vec2::new(
        camera.projection_from_view.x_axis.x,
        camera.projection_from_view.y_axis.y,
    );
    let focal_px = focal_ndc * pixel_center;
    let mean_camera = Vec3::new(mean_view.x, mean_view.y, camera_depth);
    let mean_px = focal_px * mean_camera.truncate() / camera_depth + pixel_center;
    if !mean_px.is_finite() {
        return None;
    }

    let rotation = Mat3::from_quat(normalize_quat_or_identity(quat_world));
    let scale_diag = Mat3::from_diagonal(scales_world.max(Vec3::splat(1e-6)).powf(2.0));
    let covariance_world = rotation * scale_diag * rotation.transpose();
    let covariance_px = compensate_covariance_px(brush_covariance_in_pixels(
        camera,
        covariance_world,
        mean_camera,
        focal_px,
        pixel_center,
    ));

    let power_threshold = (opacity * 255.0).ln();
    if !power_threshold.is_finite() || power_threshold <= 0.0 {
        return None;
    }

    let extent_px = brush_bbox_extent_px(covariance_px, power_threshold) * (SIGMA_COVERAGE / 3.0);
    let px_radius = extent_px.max_element();
    if !px_radius.is_finite() || px_radius < MIN_RADIUS_PX {
        return None;
    }

    if mean_px.x + extent_px.x <= 0.0
        || mean_px.x - extent_px.x >= viewport_size_px.x
        || mean_px.y + extent_px.y <= 0.0
        || mean_px.y - extent_px.y >= viewport_size_px.y
    {
        return None;
    }

    let ndc_per_pixel = Vec2::new(2.0 / viewport_size_px.x, 2.0 / viewport_size_px.y);
    let center_ndc = (mean_px - pixel_center) * ndc_per_pixel;
    let covariance_ndc =
        regularize_covariance(pixel_covariance_to_ndc(covariance_px, viewport_size_px));
    let extent_ndc = extent_px * ndc_per_pixel;
    let clip = camera.projection_from_view * mean_view.extend(1.0);
    let inv_cov = covariance_ndc.inverse();

    Some(ProjectedGaussian {
        center_ndc,
        ndc_depth: clip.z / clip.w,
        inv_cov_ndc_xx_xy_yy: [inv_cov.x_axis.x, inv_cov.y_axis.x, inv_cov.y_axis.y],
        radius_ndc: extent_ndc.max_element(),
    })
}

pub(crate) fn build_prepared_splat(
    projected: ProjectedGaussian,
    color_rgb: [f32; 3],
    opacity: f32,
) -> PreparedSplat {
    PreparedSplat {
        center_ndc: projected.center_ndc,
        ndc_depth: projected.ndc_depth,
        inv_cov_ndc_xx_xy_yy: projected.inv_cov_ndc_xx_xy_yy,
        radius_ndc: projected.radius_ndc,
        color_rgb,
        opacity,
    }
}

pub(crate) fn sh_degree_from_coeffs(coeffs_per_channel: usize) -> Option<u32> {
    match coeffs_per_channel {
        1 => Some(0),
        4 => Some(1),
        9 => Some(2),
        16 => Some(3),
        25 => Some(4),
        _ => None,
    }
}

pub(crate) fn evaluate_sh_rgb(
    coefficients: &[f32],
    coeffs_per_channel: usize,
    view_direction: Vec3,
) -> Option<[f32; 3]> {
    evaluate_sh_rgb_raw(coefficients, coeffs_per_channel, view_direction).map(activate_sh_rgb)
}

fn activate_sh_rgb(raw_rgb: [f32; 3]) -> [f32; 3] {
    // Match the same SH activation used in the larger working repo and the minimal loader.
    [
        (raw_rgb[0] + 0.5).max(0.0),
        (raw_rgb[1] + 0.5).max(0.0),
        (raw_rgb[2] + 0.5).max(0.0),
    ]
}

fn evaluate_sh_rgb_raw(
    coefficients: &[f32],
    coeffs_per_channel: usize,
    view_direction: Vec3,
) -> Option<[f32; 3]> {
    // SH coefficients are packed channel-major. `coeff(i)` reconstructs the RGB triplet for one
    // basis coefficient so the code below can read like the mathematical expansion.
    let degree = sh_degree_from_coeffs(coeffs_per_channel)?;
    if coefficients.len() < coeffs_per_channel * 3 {
        return None;
    }

    let viewdir = if view_direction.length_squared() > 1e-12 {
        view_direction.normalize()
    } else {
        Vec3::Z
    };
    let coeff = |basis_index: usize| -> Vec3 {
        let offset = basis_index * 3;
        Vec3::new(
            coefficients[offset],
            coefficients[offset + 1],
            coefficients[offset + 2],
        )
    };

    let mut color = SH_C0 * coeff(0);
    if degree == 0 {
        return Some(color.to_array());
    }

    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    let f_tmp_0a = 0.488_602_52_f32;
    color += f_tmp_0a * (-y * coeff(1) + z * coeff(2) - x * coeff(3));
    if degree == 1 {
        return Some(color.to_array());
    }

    let z2 = z * z;

    let f_tmp_0b = -1.092_548_5_f32 * z;
    let f_tmp_1a = 0.546_274_24_f32;
    let f_c1 = x * x - y * y;
    let f_s1 = 2.0 * x * y;
    let p_sh_6 = 0.946_174_7_f32 * z2 - 0.315_391_57_f32;
    let p_sh_7 = f_tmp_0b * x;
    let p_sh_5 = f_tmp_0b * y;
    let p_sh_8 = f_tmp_1a * f_c1;
    let p_sh_4 = f_tmp_1a * f_s1;

    color += p_sh_4 * coeff(4)
        + p_sh_5 * coeff(5)
        + p_sh_6 * coeff(6)
        + p_sh_7 * coeff(7)
        + p_sh_8 * coeff(8);
    if degree == 2 {
        return Some(color.to_array());
    }

    let f_tmp_0c = -2.285_229_f32 * z2 + 0.457_045_8_f32;
    let f_tmp_1b = 1.445_305_7_f32 * z;
    let f_tmp_2a = -0.590_043_6_f32;
    let f_c2 = x * f_c1 - y * f_s1;
    let f_s2 = x * f_s1 + y * f_c1;
    let p_sh_12 = z * (1.865_881_7_f32 * z2 - 1.119_529_f32);
    let p_sh_13 = f_tmp_0c * x;
    let p_sh_11 = f_tmp_0c * y;
    let p_sh_14 = f_tmp_1b * f_c1;
    let p_sh_10 = f_tmp_1b * f_s1;
    let p_sh_15 = f_tmp_2a * f_c2;
    let p_sh_9 = f_tmp_2a * f_s2;

    color += p_sh_9 * coeff(9)
        + p_sh_10 * coeff(10)
        + p_sh_11 * coeff(11)
        + p_sh_12 * coeff(12)
        + p_sh_13 * coeff(13)
        + p_sh_14 * coeff(14)
        + p_sh_15 * coeff(15);
    if degree == 3 {
        return Some(color.to_array());
    }

    let f_tmp_0d = z * (-4.683_326_f32 * z2 + 2.007_139_7_f32);
    let f_tmp_1c = 3.311_611_4_f32 * z2 - 0.473_087_34_f32;
    let f_tmp_2b = -1.770_130_8_f32 * z;
    let f_tmp_3a = 0.625_835_7_f32;
    let f_c3 = x * f_c2 - y * f_s2;
    let f_s3 = x * f_s2 + y * f_c2;
    let p_sh_20 = 1.984_313_5_f32 * z * p_sh_12 - 1.006_230_6_f32 * p_sh_6;
    let p_sh_21 = f_tmp_0d * x;
    let p_sh_19 = f_tmp_0d * y;
    let p_sh_22 = f_tmp_1c * f_c1;
    let p_sh_18 = f_tmp_1c * f_s1;
    let p_sh_23 = f_tmp_2b * f_c2;
    let p_sh_17 = f_tmp_2b * f_s2;
    let p_sh_24 = f_tmp_3a * f_c3;
    let p_sh_16 = f_tmp_3a * f_s3;

    color += p_sh_16 * coeff(16)
        + p_sh_17 * coeff(17)
        + p_sh_18 * coeff(18)
        + p_sh_19 * coeff(19)
        + p_sh_20 * coeff(20)
        + p_sh_21 * coeff(21)
        + p_sh_22 * coeff(22)
        + p_sh_23 * coeff(23)
        + p_sh_24 * coeff(24);
    Some(color.to_array())
}

fn regularize_covariance(covariance: Mat2) -> Mat2 {
    let off_diagonal = 0.5 * (covariance.x_axis.y + covariance.y_axis.x);
    Mat2::from_cols(
        Vec2::new(covariance.x_axis.x.max(1e-8), off_diagonal),
        Vec2::new(off_diagonal, covariance.y_axis.y.max(1e-8)),
    )
}

fn mat3_from_affine(affine: Affine3A) -> Mat3 {
    Mat3::from_cols(
        affine.matrix3.x_axis.into(),
        affine.matrix3.y_axis.into(),
        affine.matrix3.z_axis.into(),
    )
}

fn brush_covariance_in_pixels(
    camera: &CameraApproximation,
    covariance_world: Mat3,
    mean_camera: Vec3,
    focal_px: Vec2,
    pixel_center: Vec2,
) -> Mat2 {
    let view_linear = mat3_from_affine(camera.view_from_world);
    let covariance_view = view_linear * covariance_world * view_linear.transpose();
    let [row0, row1] = brush_camera_jacobian_rows(
        mean_camera,
        focal_px,
        camera.viewport_size_px.max(Vec2::ONE),
        pixel_center,
    );
    let cov_times_row0 = covariance_view * row0;
    let cov_times_row1 = covariance_view * row1;
    regularize_covariance(Mat2::from_cols(
        Vec2::new(row0.dot(cov_times_row0), row0.dot(cov_times_row1)),
        Vec2::new(row1.dot(cov_times_row0), row1.dot(cov_times_row1)),
    ))
}

fn brush_camera_jacobian_rows(
    mean_camera: Vec3,
    focal_px: Vec2,
    viewport_size_px: Vec2,
    pixel_center: Vec2,
) -> [Vec3; 2] {
    let lims_pos = (1.15 * viewport_size_px - pixel_center) / focal_px.max(Vec2::splat(1e-6));
    let lims_neg = (-0.15 * viewport_size_px - pixel_center) / focal_px.max(Vec2::splat(1e-6));
    let uv = mean_camera.truncate() / mean_camera.z.max(1e-6);
    let uv_clipped = uv.clamp(lims_neg, lims_pos);
    let duv_dxy = focal_px / mean_camera.z.max(1e-6);
    [
        Vec3::new(duv_dxy.x, 0.0, -duv_dxy.x * uv_clipped.x),
        Vec3::new(0.0, duv_dxy.y, -duv_dxy.y * uv_clipped.y),
    ]
}

fn compensate_covariance_px(covariance_px: Mat2) -> Mat2 {
    let mut compensated = regularize_covariance(covariance_px);
    compensated.x_axis.x += BRUSH_COVARIANCE_BLUR_PX;
    compensated.y_axis.y += BRUSH_COVARIANCE_BLUR_PX;
    regularize_covariance(compensated)
}

fn brush_bbox_extent_px(covariance_px: Mat2, power_threshold: f32) -> Vec2 {
    Vec2::new(
        (2.0 * power_threshold * covariance_px.x_axis.x)
            .max(0.0)
            .sqrt(),
        (2.0 * power_threshold * covariance_px.y_axis.y)
            .max(0.0)
            .sqrt(),
    )
}

fn pixel_covariance_to_ndc(covariance_px: Mat2, viewport_size_px: Vec2) -> Mat2 {
    let scale = Vec2::new(
        2.0 / viewport_size_px.x.max(1.0),
        2.0 / viewport_size_px.y.max(1.0),
    );
    let xx = covariance_px.x_axis.x * scale.x * scale.x;
    let xy = covariance_px.x_axis.y * scale.x * scale.y;
    let yy = covariance_px.y_axis.y * scale.y * scale.y;
    Mat2::from_cols(Vec2::new(xx, xy), Vec2::new(xy, yy))
}
