//! Native archetype queries, memoized uploads, instance transforms and blueprint properties.
use crate::cache::{BatchKey, GpuCache, Primitives};
use crate::renderer::GaussianRenderer;
use glam::UVec2;
use gsplat_core::{Camera, RenderMode, RenderOptions, native::NativeSplats};
use half::f16;
use re_sdk_types::{Archetype as _, Component as _, archetypes::GaussianSplats3D};
use re_view::{
    BlueprintResolvedResultsExt as _, DataResultQuery as _, VisualizerInstructionQueryResults,
};
use re_view_spatial::{SpatialViewState, TransformTreeContext};
use re_viewer::external::re_viewport_blueprint::ViewProperty;
use re_viewer::external::{
    re_log, re_log_types, re_query, re_renderer, re_sdk_types, re_view, re_view_spatial,
    re_viewer_context,
};
use re_viewer_context::{
    AppOptions, DataResult, IdentifiedViewSystem, ViewContext, ViewContextCollection, ViewQuery,
    ViewSystemExecutionError, ViewSystemIdentifier, ViewerReportSeverity,
    VisualizerExecutionOutput, VisualizerInstruction, VisualizerQueryInfo, VisualizerSystem,
};
use std::hash::{DefaultHasher, Hash as _, Hasher as _};

/// Blueprint-only field on a `ComputeGaussianSplats3D` visualizer instruction.
/// The data recording still contains only native `GaussianSplats3D` descriptors.
pub(crate) fn render_mode_descriptor() -> re_sdk_types::ComponentDescriptor {
    re_sdk_types::ComponentDescriptor {
        archetype: None,
        component: "ComputeGaussianSplats3D:render_mode".into(),
        component_type: Some(re_sdk_types::components::Text::name()),
    }
}

#[derive(Default)]
pub(crate) struct GaussianSplatVisualizer;

impl IdentifiedViewSystem for GaussianSplatVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        crate::selection::COMPUTE.into()
    }
}

impl VisualizerSystem for GaussianSplatVisualizer {
    fn affinity(&self) -> Option<re_sdk_types::ViewClassIdentifier> {
        use re_sdk_types::View as _;
        Some(re_sdk_types::blueprint::views::Spatial3DView::identifier())
    }

    fn visualizer_query_info(&self, _: &AppOptions) -> VisualizerQueryInfo {
        let mut components = GaussianSplats3D::all_components().to_vec();
        components.push(render_mode_descriptor());
        VisualizerQueryInfo::single_required_component::<re_sdk_types::components::Position3D>(
            &GaussianSplats3D::descriptor_centers(),
            &components,
        )
    }

    fn execute(
        &self,
        ctx: &ViewContext<'_>,
        query: &ViewQuery<'_>,
        systems: &ViewContextCollection,
    ) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
        execute_compute(ctx, query, systems, Self::identifier(), draw_splat_entity)
    }
}

/// Query and queue one entity instruction; returns whether the view needs another frame.
pub(crate) type DrawEntity = fn(
    &ViewContext<'_>,
    &ViewQuery<'_>,
    &DataResult,
    &VisualizerInstruction,
    &[glam::DAffine3],
    &GaussianRenderer,
    &VisualizerExecutionOutput,
    &dyn Fn(ViewerReportSeverity, String),
    &mut Vec<re_renderer::QueueableDrawData>,
    &Camera,
) -> Result<bool, ViewSystemExecutionError>;

/// Shared splat and hexel execution: one eye per view, explicit instructions, instance transforms.
pub(crate) fn execute_compute(
    ctx: &ViewContext<'_>,
    query: &ViewQuery<'_>,
    systems: &ViewContextCollection,
    identifier: ViewSystemIdentifier,
    draw_entity: DrawEntity,
) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
    re_tracing::profile_function!();
    let renderer = ctx.render_ctx().renderer::<GaussianRenderer>()?;
    let output = VisualizerExecutionOutput::default();
    let Some(camera) = camera_from_view(ctx, query) else {
        // The spatial view publishes its first eye on the next frame.
        ctx.egui_ctx().request_repaint();
        return Ok(output);
    };
    let transforms = systems.get::<TransformTreeContext>(&output)?;
    let mut draw = Vec::new();
    for (data, instruction) in query.iter_visualizer_instruction_for(identifier) {
        let report = |severity, message: String| {
            output.report_unspecified_source(instruction.id, severity, message);
        };
        if !crate::selection::has_explicit_visualizers(ctx.viewer_ctx, data) {
            continue;
        }
        let instances = match transforms.target_from_entity_path(data.entity_path.hash()) {
            Some(Ok(transform)) => &transform.target_from_instances()[..],
            Some(Err(error)) => {
                report(
                    ViewerReportSeverity::Error,
                    format!("Invalid entity transform: {error:#}"),
                );
                continue;
            }
            None => continue,
        };
        match draw_entity(
            ctx,
            query,
            data,
            instruction,
            instances,
            renderer,
            &output,
            &report,
            &mut draw,
            &camera,
        ) {
            Ok(true) => ctx.egui_ctx().request_repaint(),
            Ok(false) => {}
            Err(error) => report(ViewerReportSeverity::Error, format!("{error:#}")),
        }
    }
    Ok(output.with_draw_data(draw))
}

fn draw_splat_entity(
    ctx: &ViewContext<'_>,
    query: &ViewQuery<'_>,
    data: &DataResult,
    instruction: &VisualizerInstruction,
    instances: &[glam::DAffine3],
    renderer: &GaussianRenderer,
    output: &VisualizerExecutionOutput,
    report: &dyn Fn(ViewerReportSeverity, String),
    draw: &mut Vec<re_renderer::QueueableDrawData>,
    camera: &Camera,
) -> Result<bool, ViewSystemExecutionError> {
    re_tracing::profile_function!();
    let components =
        GaussianSplats3D::all_component_identifiers().chain([render_mode_descriptor().component]);
    let query_results =
        data.query_components_with_history(ctx, query, components, instruction, None);
    let results = VisualizerInstructionQueryResults::new(instruction, &query_results, output);
    let centers = results.iter_required(GaussianSplats3D::descriptor_centers().component);
    let scales = results.iter_optional(GaussianSplats3D::descriptor_scales().component);
    let rotations = results.iter_optional(GaussianSplats3D::descriptor_quaternions().component);
    let colors = results.iter_optional(GaussianSplats3D::descriptor_colors().component);
    let sh = results.iter_optional(GaussianSplats3D::descriptor_sh_coefficients().component);
    let degree =
        results.iter_optional(GaussianSplats3D::descriptor_spherical_harmonics_degree().component);
    let modes = results.iter_optional(render_mode_descriptor().component);
    let mode = modes
        .slice::<String>()
        .last()
        .and_then(|(_, values)| values.first().cloned());
    let render_mode = match mode.as_deref() {
        Some("mip") => RenderMode::Mip,
        None | Some("default") => RenderMode::Default,
        Some(unknown) => {
            report(
                ViewerReportSeverity::Warning,
                format!("Unknown Gaussian render mode {unknown:?}; using default"),
            );
            RenderMode::Default
        }
    };
    let attribute_generation = attribute_generation(
        instruction,
        &query_results,
        GaussianSplats3D::all_component_identifiers(),
    );
    let query_ctx = ctx.query_context(data, query.latest_at_query(), instruction.id);
    let color: re_sdk_types::components::Color = re_viewer_context::typed_fallback_for(
        &query_ctx,
        GaussianSplats3D::descriptor_colors().component,
    );
    let fallback = [u32::from_be_bytes(color.to_array())];
    let fallback_degree: re_sdk_types::components::SphericalHarmonicsDegree =
        re_viewer_context::typed_fallback_for(
            &query_ctx,
            GaussianSplats3D::descriptor_spherical_harmonics_degree().component,
        );
    let mut retry = false;
    for (row, centers, scales, rotations, colors, sh, degree) in re_query::range_zip_1x5(
        centers.slice::<[f32; 3]>(),
        scales.slice::<[f32; 3]>(),
        rotations.slice::<[f32; 4]>(),
        colors.slice::<u32>(),
        sh.slice::<[[f16; 3]; 15]>(),
        degree.slice::<u32>(),
    ) {
        if centers.is_empty() {
            continue;
        }
        let mut hasher = DefaultHasher::new();
        (attribute_generation, row).hash(&mut hasher);
        if colors.is_none_or(|values| values.is_empty()) {
            fallback.hash(&mut hasher);
        }
        let signature = re_log_types::hash::Hash64::from_u64(hasher.finish());
        let native = NativeSplats {
            centers,
            scales: scales.unwrap_or(&[]),
            quaternions: rotations.unwrap_or(&[]),
            colors: colors.filter(|c| !c.is_empty()).unwrap_or(&fallback),
            sh: sh.unwrap_or(&[]),
            degree: degree
                .and_then(|d| d.first().copied())
                .unwrap_or(*fallback_degree.0),
        };
        retry |= draw_instances(
            ctx,
            query,
            data,
            instruction,
            instances,
            renderer,
            report,
            draw,
            camera,
            row,
            signature,
            Primitives::Splats(&native),
            render_mode,
        )?;
    }
    Ok(retry)
}

/// Queue one compute batch per instance; capacity or capability failures draw the stock fallback.
pub(crate) fn draw_instances(
    ctx: &ViewContext<'_>,
    query: &ViewQuery<'_>,
    data: &DataResult,
    instruction: &VisualizerInstruction,
    instances: &[glam::DAffine3],
    renderer: &GaussianRenderer,
    report: &dyn Fn(ViewerReportSeverity, String),
    draw: &mut Vec<re_renderer::QueueableDrawData>,
    camera: &Camera,
    row: (re_log_types::TimeInt, re_sdk_types::RowId),
    signature: re_log_types::hash::Hash64,
    primitives: Primitives<'_>,
    render_mode: RenderMode,
) -> Result<bool, ViewSystemExecutionError> {
    let mut retry = false;
    let background = solid_background(ctx);
    for (index, world_from_local) in instances.iter().enumerate() {
        let world_from_local = world_from_local.as_affine3a();
        if !world_from_local.inverse().is_finite() {
            report(
                ViewerReportSeverity::Error,
                "Non-invertible instance transform".into(),
            );
            continue;
        }
        let key = BatchKey {
            view_id: query.view_id,
            entity: data.entity_path.clone(),
            instruction: instruction.id,
            instance: index,
            row,
        };
        let options = RenderOptions {
            render_mode,
            world_from_local,
            ..Default::default()
        };
        let prepared = ctx
            .viewer_ctx
            .store_context
            .memoizer::<GpuCache, _>(|cache| {
                cache.prepare(
                    ctx.render_ctx(),
                    renderer,
                    key.clone(),
                    primitives,
                    signature,
                    camera,
                    options,
                    background,
                )
            });
        match prepared {
            Ok((batch, needs_repaint)) => {
                draw.push(batch.into());
                retry |= needs_repaint;
            }
            Err(
                error @ (gsplat_core::Error::Capacity { .. } | gsplat_core::Error::Capabilities),
            ) => {
                // The pixel contracts use these machine-visible markers to distinguish fallback from compute.
                draw.push(match primitives {
                    Primitives::Splats(native) => {
                        report(
                            ViewerReportSeverity::Warning,
                            format!("{error}; using the native Gaussian renderer"),
                        );
                        re_log::warn_once!("Gaussian compute fallback: {error:#}");
                        crate::fallback::native_fallback(
                            ctx,
                            query,
                            key,
                            native,
                            world_from_local,
                            signature,
                        )?
                    }
                    Primitives::Hexels(hexels) => {
                        report(
                            ViewerReportSeverity::Warning,
                            format!("{error}; drawing hexel centres as points"),
                        );
                        re_log::warn_once!("Hexel compute fallback: {error:#}");
                        crate::hexels::points_fallback(ctx, query, hexels, world_from_local)?
                    }
                });
            }
            Err(error) => {
                return Err(ViewSystemExecutionError::DrawDataCreationError(
                    std::sync::Arc::new(error),
                ));
            }
        }
    }
    Ok(retry)
}

/// The view's solid background colour in display space; gradients and query failures give none.
fn solid_background(ctx: &ViewContext<'_>) -> Option<glam::Vec3> {
    use re_sdk_types::blueprint::{archetypes::Background, components::BackgroundKind};
    let property = ViewProperty::from_archetype::<Background>(ctx);
    let kind: BackgroundKind = property
        .component_or_fallback(ctx, Background::descriptor_kind().component)
        .ok()?;
    if kind != BackgroundKind::SolidColor {
        return None;
    }
    let color: re_sdk_types::components::Color = property
        .component_or_fallback(ctx, Background::descriptor_color().component)
        .ok()?;
    let [r, g, b, _] = color.to_array();
    Some(glam::Vec3::new(r.into(), g.into(), b.into()) / 255.0)
}

pub(crate) fn attribute_generation(
    instruction: &VisualizerInstruction,
    query_results: &re_view::BlueprintResolvedResults<'_>,
    components: impl Iterator<Item = re_sdk_types::ComponentIdentifier>,
) -> u64 {
    // Rerun's combined hash includes every view default, even camera properties.
    // Hash only resolved primitive components so views with different eyes/modes share uploads.
    let mut attribute_hasher = DefaultHasher::new();
    for component in components {
        std::hash::Hash::hash(&component, &mut attribute_hasher);
        instruction
            .component_mappings
            .get(&component)
            .hash(&mut attribute_hasher);
        if let Ok(chunks) =
            re_view::ChunksWithComponent::try_from(query_results.get_required_chunks(component))
        {
            for chunk in &*chunks.chunks {
                for row in chunk.component_row_ids(component) {
                    row.hash(&mut attribute_hasher);
                    // Blueprint overrides have zeroed row ids; hash their actual values.
                    if row == re_sdk_types::RowId::ZERO
                        && let Some(array) = chunk.raw_component_array(component)
                    {
                        hash_arrow(&array.to_data(), &mut attribute_hasher);
                    }
                }
            }
        }
    }
    attribute_hasher.finish()
}

/// Read the last published eye and viewport for the compute renderer and probe.
pub fn camera_from_view(ctx: &ViewContext<'_>, query: &ViewQuery<'_>) -> Option<Camera> {
    let state = ctx.view_state.as_any().downcast_ref::<SpatialViewState>()?;
    let eye = state.state_3d.eye_state.last_eye?;
    let fov_y = eye.fov_y?;
    let info = ctx.egui_ctx().memory_mut(|m| {
        m.caches
            .cache::<re_viewer_context::ViewRectPublisher>()
            .get(&query.view_id)
            .cloned()
    })?;
    let rect = info.rect;
    if !rect.is_positive() {
        return None;
    }
    let size = UVec2::from_array(
        re_viewer_context::gpu_bridge::viewport_resolution_in_pixels(
            rect,
            ctx.egui_ctx().pixels_per_point(),
        ),
    );
    Some(Camera::from_view(
        glam::Affine3A::from_mat4(eye.world_from_rub_view.to_mat4()),
        f64::from(fov_y),
        size,
    ))
}

// Used only for blueprint-valued components, whose source row ids Rerun deliberately clears.
fn hash_arrow(data: &re_sdk_types::external::arrow::array::ArrayData, hasher: &mut DefaultHasher) {
    data.len().hash(hasher);
    data.offset().hash(hasher);
    for buffer in data.buffers() {
        buffer.as_slice().hash(hasher);
    }
    for child in data.child_data() {
        hash_arrow(child, hasher);
    }
}
