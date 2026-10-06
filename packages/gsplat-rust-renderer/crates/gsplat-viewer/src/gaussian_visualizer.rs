//! Native archetype queries, memoized uploads, instance transforms and blueprint properties.
use crate::gaussian_renderer::{GaussianDrawData, GpuCache};
use glam::{Quat, UVec2, Vec2};
use gsplat_core::{Camera, RenderMode, RenderOptions, native::NativeSplats};
use half::f16;
use re_sdk_types::{Archetype as _, Component as _, archetypes::GaussianSplats3D};
use re_view::{
    BlueprintResolvedResultsExt as _, DataResultQuery as _, VisualizerInstructionQueryResults,
};
use re_view_spatial::{SpatialViewState, TransformTreeContext};
use re_viewer_context::{
    AppOptions, IdentifiedViewSystem, ViewContext, ViewContextCollection, ViewQuery,
    ViewSystemExecutionError, ViewSystemIdentifier, VisualizerExecutionOutput, VisualizerQueryInfo,
    VisualizerSystem,
};
use std::hash::{DefaultHasher, Hash as _, Hasher as _};
/// Blueprint-only field on a ComputeGaussianSplats3D visualizer instruction.
/// The data recording still contains only native GaussianSplats3D descriptors.
pub fn render_mode_descriptor() -> re_sdk_types::ComponentDescriptor {
    re_sdk_types::ComponentDescriptor {
        archetype: None,
        component: "ComputeGaussianSplats3D:render_mode".into(),
        component_type: Some(re_sdk_types::components::Text::name()),
    }
}

#[derive(Default)]
pub struct GaussianSplatVisualizer;
impl IdentifiedViewSystem for GaussianSplatVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        "ComputeGaussianSplats3D".into()
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
        let output = VisualizerExecutionOutput::default();
        let camera = camera_from_view(ctx, query);
        let transforms = systems.get::<TransformTreeContext>(&output)?;
        let mut draw = GaussianDrawData::default();
        for (data, instruction) in query.iter_visualizer_instruction_for(Self::identifier()) {
            // Until the runtime default reaches the blueprint store, only native draws.
            // Explicit instructions (including our runtime default) bypass heuristics.
            let active =
                re_sdk_types::blueprint::archetypes::ActiveVisualizers::descriptor_instruction_ids(
                )
                .component;
            if ctx
                .viewer_ctx
                .store_context
                .blueprint
                .latest_at(
                    ctx.viewer_ctx.blueprint_query,
                    &data.override_base_path,
                    [active],
                )
                .get(active)
                .is_none()
            {
                continue;
            }
            let components = GaussianSplats3D::all_component_identifiers();
            let query_results =
                data.query_components_with_history(ctx, query, components, instruction, None);
            let results =
                VisualizerInstructionQueryResults::new(instruction, &query_results, &output);
            let centers = results.iter_required(GaussianSplats3D::descriptor_centers().component);
            let scales = results.iter_optional(GaussianSplats3D::descriptor_scales().component);
            let rotations =
                results.iter_optional(GaussianSplats3D::descriptor_quaternions().component);
            let colors = results.iter_optional(GaussianSplats3D::descriptor_colors().component);
            let sh =
                results.iter_optional(GaussianSplats3D::descriptor_sh_coefficients().component);
            let degree = results
                .iter_optional(GaussianSplats3D::descriptor_spherical_harmonics_degree().component);
            let mode_query = data.query_components_with_history(
                ctx,
                query,
                [render_mode_descriptor().component],
                instruction,
                None,
            );
            let mode_results =
                VisualizerInstructionQueryResults::new(instruction, &mode_query, &output);
            let modes = mode_results.iter_optional(render_mode_descriptor().component);
            let mode = modes
                .slice::<String>()
                .last()
                .and_then(|(_, values)| values.first().cloned());
            let render_mode = match mode.as_deref() {
                Some("mip") => RenderMode::Mip,
                _ => RenderMode::Default,
            };
            let transform = match transforms.target_from_entity_path(data.entity_path.hash()) {
                Some(Ok(transform)) => transform,
                Some(Err(error)) => {
                    output.report_unspecified_source(
                        instruction.id,
                        re_viewer_context::ViewerReportSeverity::Error,
                        format!("Invalid entity transform: {error:?}"),
                    );
                    continue;
                }
                None => continue,
            };
            // Rerun's combined hash includes every view default, even camera properties.
            // Hash only resolved splat components so views with different eyes/modes share uploads.
            let mut attribute_hasher = DefaultHasher::new();
            for component in GaussianSplats3D::all_component_identifiers() {
                std::hash::Hash::hash(&component, &mut attribute_hasher);
                instruction
                    .component_mappings
                    .get(&component)
                    .hash(&mut attribute_hasher);
                if let Ok(chunks) = re_view::ChunksWithComponent::try_from(
                    query_results.get_required_chunks(component),
                ) {
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
            let attribute_generation = attribute_hasher.finish();
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
                let fallback: re_sdk_types::components::Color =
                    re_viewer_context::typed_fallback_for(
                        &ctx.query_context(data, query.latest_at_query(), instruction.id),
                        GaussianSplats3D::descriptor_colors().component,
                    );
                let fallback = [u32::from_be_bytes(fallback.to_array())];
                if colors.is_none_or(|values| values.is_empty()) {
                    fallback.hash(&mut hasher);
                }
                let signature = hasher.finish();
                let native = NativeSplats {
                    centers,
                    scales: scales.unwrap_or(&[]),
                    quaternions: rotations.unwrap_or(&[]),
                    colors: colors.filter(|c| !c.is_empty()).unwrap_or(&fallback),
                    sh: sh.unwrap_or(&[]),
                    degree: degree.and_then(|d| d.first().copied()).unwrap_or(3),
                };
                for (index, world_from_local) in
                    transform.target_from_instances().iter().enumerate()
                {
                    let world_from_local = world_from_local.as_affine3a();
                    if !world_from_local.inverse().is_finite() {
                        output.report_unspecified_source(
                            instruction.id,
                            re_viewer_context::ViewerReportSeverity::Error,
                            "Non-invertible splat instance transform",
                        );
                        continue;
                    }
                    if let Some(camera) = &camera {
                        let key = format!(
                            "{:?}/{:?}/{}/{:?}/{index}/{row:?}",
                            ctx.viewer_ctx.store_context.recording.store_id(),
                            query.view_id,
                            data.entity_path,
                            instruction.id
                        );
                        let retry = ctx
                            .viewer_ctx
                            .store_context
                            .memoizer::<GpuCache, _>(|cache| {
                                draw.add_batch(
                                    ctx.render_ctx(),
                                    cache,
                                    key,
                                    query.view_id,
                                    &native,
                                    signature,
                                    camera,
                                    RenderOptions {
                                        render_mode: Some(render_mode),
                                        world_from_local,
                                        ..Default::default()
                                    },
                                )
                            });
                        let (retry, _submitted) = match retry {
                            Ok(retry) => retry,
                            Err(error) => {
                                output.report_unspecified_source(
                                    instruction.id,
                                    re_viewer_context::ViewerReportSeverity::Error,
                                    error.to_string(),
                                );
                                continue;
                            }
                        };
                        if retry {
                            ctx.egui_ctx().request_repaint();
                        }
                    }
                }
            }
        }
        if camera.is_none() {
            ctx.egui_ctx()
                .request_repaint_after(std::time::Duration::from_millis(50));
        }
        Ok(output.with_draw_data([draw.into()]))
    }
}
fn camera_from_view(ctx: &ViewContext<'_>, query: &ViewQuery<'_>) -> Option<Camera> {
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
    let world_from_rdf = eye.world_from_rub_view.to_mat4()
        * glam::Mat4::from_quat(Quat::from_xyzw(1.0, 0.0, 0.0, 0.0));
    let (_, rotation, position) = world_from_rdf.to_scale_rotation_translation();
    Some(Camera {
        model: gsplat_core::CameraModel::Pinhole,
        position,
        rotation,
        fov_y: fov_y as f64,
        fov_x: 2.0 * ((fov_y as f64 * 0.5).tan() * size.x as f64 / size.y as f64).atan(),
        center_uv: Vec2::splat(0.5),
        size,
    })
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
