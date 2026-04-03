//! Custom Gaussian splat visualizer — the Rerun-facing middle layer.
//!
//! # Role in the Pipeline
//!
//! This module sits between Rerun's data store and the GPU renderer
//! ([`crate::gaussian_renderer`]).  Each frame it:
//!
//! 1. **Queries** the data store for any entity that matches the
//!    `GaussianSplats3D` archetype (centers, quaternions, scales, opacities,
//!    colors, and optionally spherical-harmonic coefficients).
//!
//! 2. **Builds or reuses** a [`RenderGaussianCloud`] — a packed, renderer-ready
//!    representation of the Gaussian data.  Clouds are cached per entity path
//!    and only rebuilt when the data or transform signature changes.
//!
//! 3. **Culls** splats that are behind the camera or outside the frustum using
//!    a fast approximate test on the CPU.  This is intentionally conservative;
//!    the GPU shader does the exact projection later.
//!
//! 4. **Sorts** the surviving candidates back-to-front by camera depth.  Alpha
//!    blending requires this ordering for correct compositing.
//!
//! 5. **Submits** the sorted candidate list to [`GaussianDrawData`] which
//!    drives the actual GPU render pass.
//!
//! # Rerun Extension Points
//!
//! The two traits that make this a Rerun visualizer are:
//!
//! - [`IdentifiedViewSystem`] — provides the string identifier
//!   `"GaussianSplats3D"` that the blueprint uses to bind an entity to this
//!   visualizer (e.g. `overrides={entity: rrb.Visualizer("GaussianSplats3D")}`).
//!
//! - [`VisualizerSystem`] — the `execute()` method is called once per frame by
//!   the Rerun viewer with the current view context and query.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::gaussian_renderer::GaussianDrawData;
use glam::{Affine3A, Quat, Vec2, Vec3};
use re_view::{DataResultQuery as _, VisualizerInstructionQueryResults};
use re_view_spatial::{SpatialViewState, TransformTreeContext};
use re_viewer_context::{
    AppOptions, IdentifiedViewSystem, ViewContext, ViewContextCollection, ViewQuery,
    ViewSystemExecutionError, ViewSystemIdentifier, VisualizerExecutionOutput, VisualizerQueryInfo,
    VisualizerSystem,
};
use rerun::{Archetype as _, Component as _};

// ── Imports from gsplat_core (the Rerun-free algorithm module) ───────────
use crate::gsplat_core::{
    CameraApproximation, RenderGaussianCloud, RenderShCoefficients, approximate_bounds_from_points,
    fallback_camera, normalize_quat_or_identity, rebuild_visible_indices, sh_degree_from_coeffs,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Archetype definition
// ═══════════════════════════════════════════════════════════════════════════════
//
// The `GaussianSplats3D` archetype defines the **component contract** between
// the Python logger and this Rust visualizer.  Both sides must agree on the
// archetype name and component descriptors.  The Python side
// (`gsplat_rust_renderer.gaussians3d.Gaussians3D`) implements `rr.AsComponents`
// and produces the exact same descriptors.

/// Marker type implementing the Rerun `Archetype` trait.  This tells the
/// viewer which components an entity needs in order to be rendered by our
/// custom visualizer.
struct GaussianSplats3D;

impl rerun::Archetype for GaussianSplats3D {
    fn name() -> rerun::ArchetypeName {
        "GaussianSplats3D".into()
    }

    fn display_name() -> &'static str {
        "Gaussian Splats 3D"
    }

    /// The five components that every Gaussian splat entity must provide.
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

    /// Spherical harmonic coefficients are optional — when absent, only the
    /// DC color is used (no view-dependent effects).
    fn optional_components() -> std::borrow::Cow<'static, [rerun::ComponentDescriptor]> {
        vec![Self::descriptor_sh_coefficients()].into()
    }
}

/// Component descriptor builders.  Each descriptor specifies the archetype
/// name, a unique component name within that archetype, and the underlying
/// Rerun component type that carries the actual data.
impl GaussianSplats3D {
    /// World-space Gaussian center positions — `[N, 3]` float32.
    fn descriptor_centers() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:centers".into(),
            component_type: Some(rerun::components::Translation3D::name()),
        }
    }

    /// Per-splat rotation quaternions in `[x, y, z, w]` order — `[N, 4]` float32.
    fn descriptor_quaternions() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:quaternions".into(),
            component_type: Some(rerun::components::RotationQuat::name()),
        }
    }

    /// Per-axis scale factors (already exponentiated) — `[N, 3]` float32.
    fn descriptor_scales() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:scales".into(),
            component_type: Some(rerun::components::Scale3D::name()),
        }
    }

    /// Per-splat opacity in `[0, 1]` — `[N]` float32.
    fn descriptor_opacities() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:opacities".into(),
            component_type: Some(rerun::components::Opacity::name()),
        }
    }

    /// Base RGB color derived from the zeroth SH coefficient — `[N]` packed RGBA32.
    fn descriptor_colors() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:colors".into(),
            component_type: Some(rerun::components::Color::name()),
        }
    }

    /// Optional higher-order SH coefficients — `[N, C, 3]` float32 tensor.
    /// `C` is the number of coefficients per channel (1, 4, 9, 16, or 25 for
    /// SH degrees 0–4).
    fn descriptor_sh_coefficients() -> rerun::ComponentDescriptor {
        rerun::ComponentDescriptor {
            archetype: Some("GaussianSplats3D".into()),
            component: "GaussianSplats3D:sh_coefficients".into(),
            component_type: Some(rerun::components::TensorData::name()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Visualizer state and cache
// ═══════════════════════════════════════════════════════════════════════════════

/// The visualizer that Rerun instantiates once and calls `execute()` on every
/// frame.  It maintains a cache of [`RenderGaussianCloud`]s keyed by entity
/// path so that unchanged data isn't rebuilt every frame.
#[derive(Default)]
pub struct GaussianSplatVisualizer {
    /// One cached render cloud per entity path.  The key is
    /// `"gaussian_splats::{entity_path}"`.
    clouds: HashMap<String, CachedCloud>,
}

/// A cached cloud together with the signature that was used to build it.
/// When the signature changes (different splat count, SH shape, or transform),
/// the cloud is rebuilt from the current query results.
struct CachedCloud {
    signature: CloudSignature,
    cloud: Arc<RenderGaussianCloud>,
}

/// Lightweight fingerprint of a cloud's configuration.  Two signatures are
/// equal if and only if the cloud data can be reused without rebuilding.
#[derive(Clone, PartialEq, Eq)]
struct CloudSignature {
    /// Total number of Gaussian splats.
    expected_splats: usize,
    /// Number of SH coefficients per color channel, or `None` if no SH data.
    sh_coeffs_per_channel: Option<usize>,
    /// Bit-exact representation of the 3×4 entity transform.  Using raw bits
    /// avoids floating-point comparison issues.
    transform_bits: [u32; 12],
}

// ═══════════════════════════════════════════════════════════════════════════════
// Rerun trait implementations
// ═══════════════════════════════════════════════════════════════════════════════

impl IdentifiedViewSystem for GaussianSplatVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        "GaussianSplats3D".into()
    }
}

impl VisualizerSystem for GaussianSplatVisualizer {
    /// Tell Rerun which archetype this visualizer handles.
    fn visualizer_query_info(&self, _app_options: &AppOptions) -> VisualizerQueryInfo {
        VisualizerQueryInfo::from_archetype::<GaussianSplats3D>()
    }

    /// Called once per frame.  This is the main entry point for the visualizer.
    ///
    /// # Per-frame flow
    ///
    /// For each entity that matches the `GaussianSplats3D` archetype:
    /// 1. Query the five required components + optional SH tensor
    /// 2. Compute a cache signature (splat count + SH shape + transform)
    /// 3. Build or reuse the `RenderGaussianCloud`
    /// 4. Extract the current camera from the 3D view state
    /// 5. Cull splats outside the frustum (approximate, on CPU)
    /// 6. Sort survivors back-to-front by depth
    /// 7. Submit to `GaussianDrawData` for GPU rendering
    fn execute(
        &mut self,
        ctx: &ViewContext<'_>,
        query: &ViewQuery<'_>,
        context_systems: &ViewContextCollection,
    ) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
        let mut output = VisualizerExecutionOutput::default();
        // The transform tree tells us how each entity's coordinate frame
        // relates to the view's coordinate frame.
        let transforms = context_systems.get::<TransformTreeContext>(&output)?;
        let mut draw_data = GaussianDrawData::new(ctx.render_ctx());
        let mut extra_draw_data = Vec::new();
        let mut active_labels = HashSet::new();

        // Iterate over every entity in the current view that has been assigned
        // to this visualizer (via blueprint override or automatic matching).
        for (data_result, instruction) in query.iter_visualizer_instruction_for(Self::identifier())
        {
            // ── Step 1: Query components from the data store ──────────
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

            // SH coefficients are fetched separately as a single tensor (not
            // per-row like the other components).
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

            // ── Step 2: Resolve entity transform ──────────────────────
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

            // ── Step 3: Build or reuse the render cloud ───────────────
            let label = format!("gaussian_splats::{}", data_result.entity_path);
            let signature = CloudSignature {
                expected_splats,
                sh_coeffs_per_channel: sh_coefficients.as_ref().map(|sh| sh.coeffs_per_channel),
                transform_bits: transform.to_cols_array().map(f32::to_bits),
            };
            active_labels.insert(label.clone());

            // `or_insert_with` only builds the cloud if this is the first time
            // we've seen this entity path.
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

            // If the signature changed (e.g. different splat count after
            // re-logging), rebuild the cloud from the new data.
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

            // ── Step 4: Extract camera ────────────────────────────────
            let cloud = cache_entry.cloud.clone();
            // Prefer the interactive 3D camera from the view state.  Fall back
            // to a synthetic camera positioned around the cloud's bounding box
            // (so the splats are visible even before the user orbits).
            let camera =
                camera_from_view(ctx, query).unwrap_or_else(|| fallback_camera(cloud.bounds_world));

            // ── Step 5 & 6: Cull and sort ─────────────────────────────
            let mut visible = Vec::new();
            rebuild_visible_indices(&mut visible, &cloud, &camera);
            let farthest_depth = visible.first().map_or(0.0, |visible| visible.camera_depth);

            // ── Step 7: Submit to the GPU renderer ────────────────────
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

        // Evict cached clouds for entities that are no longer in the view.
        self.clouds.retain(|label, _| active_labels.contains(label));
        output.draw_data = vec![draw_data.into()];
        output.draw_data.extend(extra_draw_data);
        Ok(output)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cloud construction (Rerun-specific)
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert Rerun query results into a packed [`RenderGaussianCloud`].
///
/// This is the only place that knows about Rerun's query result format.  The
/// renderer only sees the flat arrays in `RenderGaussianCloud`.
///
/// The entity `transform` is baked into the positions here so the GPU shaders
/// don't need to carry a per-entity transform matrix.
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
    let mut means_world = Vec::new();
    let mut quats_world = Vec::new();
    let mut scales_world = Vec::new();
    let mut opacities_world = Vec::new();
    let mut colors_world = Vec::new();

    // `range_zip_1x4` iterates the five component arrays in lockstep, yielding
    // one row at a time.  Missing optional components get a default empty slice.
    for (_index, centers, quats, scales, opacities, colors) in
        re_query::range_zip_1x4(centers, quaternions, scales, opacities, colors)
    {
        let quats = quats.unwrap_or_default();
        let scales = scales.unwrap_or_default();
        let opacities = opacities.unwrap_or_default();
        let colors = colors.unwrap_or_default();

        // Use the minimum length across all components to avoid out-of-bounds.
        let count = centers
            .len()
            .min(quats.len())
            .min(scales.len())
            .min(opacities.len())
            .min(colors.len());

        for row_index in 0..count {
            // Only positions are transformed here — quaternions and scales stay
            // in entity-local space.  This is intentional and matches Brush:
            // the GPU projection shader builds the 3D covariance from the
            // untransformed quat + scale, then applies the view matrix during
            // the Jacobian-based 2D projection.  Applying the entity transform
            // to the covariance would require decomposing rotation and non-uniform
            // scale from the affine, which is fragile for arbitrary transforms.
            means_world.push(transform.transform_point3(Vec3::from_array(centers[row_index])));
            quats_world.push(normalize_quat_or_identity(Quat::from_xyzw(
                quats[row_index][0],
                quats[row_index][1],
                quats[row_index][2],
                quats[row_index][3],
            )));
            // Clamp scales to a small positive minimum to avoid degenerate
            // (zero-volume) Gaussians.  1e-6 matches the Python side, the GPU
            // shader, and Brush.
            scales_world.push(Vec3::from_array(scales[row_index]).max(Vec3::splat(1e-6)));
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

/// Unpack a Rerun RGBA32 color (packed as a `u32`) into `[r, g, b]` in `[0, 1]`.
fn rerun_color_to_rgb(color: u32) -> [f32; 3] {
    let [r, g, b, _a] = re_sdk_types::datatypes::Rgba32::from_u32(color).to_array();
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}

/// Count the total number of splats across all component batches.
fn count_splats_in_results<'a, I, Idx>(centers: I) -> usize
where
    I: IntoIterator<Item = (Idx, &'a [[f32; 3]])>,
{
    centers
        .into_iter()
        .map(|(_index, positions)| positions.len())
        .sum()
}

// ═══════════════════════════════════════════════════════════════════════════════
// SH coefficient extraction (Rerun-specific)
// ═══════════════════════════════════════════════════════════════════════════════

/// Extract spherical harmonic coefficients from the data store.
///
/// The SH tensor is expected to have shape `[N, coeffs_per_channel, 3]` where
/// `N` matches the number of splats and `coeffs_per_channel` is one of
/// `{1, 4, 9, 16, 25}` (corresponding to SH degrees 0–4).
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

// ═══════════════════════════════════════════════════════════════════════════════
// Camera extraction (Rerun-specific)
// ═══════════════════════════════════════════════════════════════════════════════

/// Try to extract camera parameters from the Rerun 3D view's interactive
/// orbit camera.  Returns `None` if the view hasn't been set up yet.
fn camera_from_view(ctx: &ViewContext<'_>, query: &ViewQuery<'_>) -> Option<CameraApproximation> {
    camera_from_spatial_view_state(ctx, query)
}

/// Read the eye state from the Spatial3DView and convert it into our
/// simplified camera representation.
fn camera_from_spatial_view_state(
    ctx: &ViewContext<'_>,
    query: &ViewQuery<'_>,
) -> Option<CameraApproximation> {
    // Downcast the generic view state to the 3D-specific one.
    let spatial_view_state = ctx.view_state.as_any().downcast_ref::<SpatialViewState>()?;
    let eye = spatial_view_state.state_3d.eye_state.last_eye?;
    let vertical_fov = eye.fov_y?;
    let viewport_size_px = published_viewport_size_px(ctx, query)?;
    let aspect_ratio = (viewport_size_px.x / viewport_size_px.y.max(1.0)).max(1e-4);
    let near_plane = eye.near();

    Some(CameraApproximation {
        // Rerun stores `world_from_rub_view` (world ← view); we need the
        // inverse (world → view).
        view_from_world: Affine3A::from_mat4(eye.world_from_rub_view.inverse().to_mat4()),
        projection_from_view: glam::Mat4::perspective_infinite_rh(
            vertical_fov,
            aspect_ratio,
            near_plane,
        ),
        world_position: eye.pos_in_world(),
        viewport_size_px,
        near_plane,
    })
}

/// Read the viewport rectangle from the egui cache and convert to physical pixels.
fn published_viewport_size_px(ctx: &ViewContext<'_>, query: &ViewQuery<'_>) -> Option<Vec2> {
    let view_info = ctx.egui_ctx().memory_mut(|memory| {
        memory
            .caches
            .cache::<re_viewer_context::ViewRectPublisher>()
            .get(&query.view_id)
            .cloned()
    })?;
    // Shrink slightly to avoid edge artifacts.
    let rect = view_info.rect.shrink(2.5);
    if !rect.is_positive() {
        return None;
    }
    // Convert from logical UI points to physical pixels.
    let viewport_size_px = rect.size() * ctx.egui_ctx().pixels_per_point();
    Some(Vec2::new(
        viewport_size_px.x.max(1.0),
        viewport_size_px.y.max(1.0),
    ))
}
