//! `Hexels3D` (hexel PLY contract, f32 fields) through the shared compute path.
//! Stock viewers draw the `Points3D` fallback logged on the same entity instead.
use crate::cache::{GpuCache, Primitives};
use crate::renderer::GaussianRenderer;
use crate::visualizer::{attribute_generation, draw_instances, execute_compute};
use gsplat_core::native::{NativeHexels, at_or_last};
use gsplat_core::{Camera, RenderMode};
use re_sdk_types::{Component as _, ComponentDescriptor, components};
use re_view::{DataResultQuery as _, VisualizerInstructionQueryResults};
use re_viewer::external::{
    re_log_types, re_query, re_renderer, re_sdk_types, re_view, re_viewer_context,
};
use re_viewer_context::{
    AppOptions, DataResult, IdentifiedViewSystem, ViewContext, ViewContextCollection, ViewQuery,
    ViewSystemExecutionError, ViewSystemIdentifier, ViewerReportSeverity,
    VisualizerExecutionOutput, VisualizerInstruction, VisualizerQueryInfo, VisualizerSystem,
};
use std::hash::{DefaultHasher, Hash as _, Hasher as _};

fn descriptor(
    field: &str,
    component_type: Option<re_sdk_types::ComponentType>,
) -> ComponentDescriptor {
    let archetype: re_sdk_types::ArchetypeName = "Hexels3D".into();
    ComponentDescriptor {
        archetype: Some(archetype),
        component: re_sdk_types::ComponentIdentifier::from_archetype_field(archetype, field),
        component_type,
    }
}

/// Centres, rotations (xyzw), scales, opacities, sharpness, colour field, V2 and V3 view terms.
fn descriptors() -> [ComponentDescriptor; 8] {
    [
        descriptor("centers", Some(components::Position3D::name())),
        descriptor("quaternions", Some(components::RotationQuat::name())),
        descriptor("scales", Some(components::Vector2D::name())),
        descriptor("opacities", Some(components::Opacity::name())),
        descriptor("sharpness", None),
        descriptor("color_field", None),
        descriptor("view_linear", None),
        descriptor("sh_coefficients", None),
    ]
}

#[derive(Default)]
pub(crate) struct HexelVisualizer;

impl IdentifiedViewSystem for HexelVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        crate::selection::COMPUTE_HEXELS.into()
    }
}

impl VisualizerSystem for HexelVisualizer {
    fn affinity(&self) -> Option<re_sdk_types::ViewClassIdentifier> {
        use re_sdk_types::View as _;
        Some(re_sdk_types::blueprint::views::Spatial3DView::identifier())
    }

    fn visualizer_query_info(&self, _: &AppOptions) -> VisualizerQueryInfo {
        let all = descriptors();
        VisualizerQueryInfo::single_required_component::<components::Position3D>(&all[0], &all)
    }

    fn execute(
        &self,
        ctx: &ViewContext<'_>,
        query: &ViewQuery<'_>,
        systems: &ViewContextCollection,
    ) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
        execute_compute(ctx, query, systems, Self::identifier(), draw_hexel_entity)
    }
}

fn draw_hexel_entity(
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
    let identifiers = descriptors().map(|d| d.component);
    let query_results = data.query_components_with_history(
        ctx,
        query,
        identifiers.iter().copied(),
        instruction,
        None,
    );
    let results = VisualizerInstructionQueryResults::new(instruction, &query_results, output);
    let [
        centers,
        quaternions,
        scales,
        opacities,
        sharpness,
        color_field,
        view_linear,
        sh,
    ] = identifiers;
    let centers = results.iter_required(centers);
    let quaternions = results.iter_optional(quaternions);
    let scales = results.iter_optional(scales);
    let opacities = results.iter_optional(opacities);
    let sharpness = results.iter_optional(sharpness);
    let color_field = results.iter_optional(color_field);
    let view_linear = results.iter_optional(view_linear);
    let sh = results.iter_optional(sh);
    let generation = attribute_generation(instruction, &query_results, identifiers.into_iter());
    let mut retry = false;
    for (row, centers, quaternions, scales, opacities, sharpness, color_field, view_linear, sh) in
        re_query::range_zip_1x7(
            centers.slice::<[f32; 3]>(),
            quaternions.slice::<[f32; 4]>(),
            scales.slice::<[f32; 2]>(),
            opacities.slice::<f32>(),
            sharpness.slice::<f32>(),
            color_field.slice::<[f32; 9]>(),
            view_linear.slice::<[f32; 9]>(),
            sh.slice::<[f32; 45]>(),
        )
    {
        if centers.is_empty() {
            continue;
        }
        let mut hasher = DefaultHasher::new();
        (generation, row).hash(&mut hasher);
        let hexels = NativeHexels {
            centers,
            quaternions: quaternions.unwrap_or(&[]),
            scales: scales.unwrap_or(&[]),
            opacities: opacities.unwrap_or(&[]),
            sharpness: sharpness.unwrap_or(&[]),
            color_field: color_field.unwrap_or(&[]),
            view_linear: view_linear.unwrap_or(&[]),
            sh: sh.unwrap_or(&[]),
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
            re_log_types::hash::Hash64::from_u64(hasher.finish()),
            Primitives::Hexels(&hexels),
            RenderMode::Default,
        )?;
    }
    Ok(retry)
}

/// The `Points3D` fallback, drawn by the compute visualizer when the core cannot hold the scene:
/// centres in their base colour clamped to [0, 1], 1.5 UI points (as `hexels3d.py` logs it).
pub(crate) fn points_fallback(
    ctx: &ViewContext<'_>,
    query: &ViewQuery<'_>,
    hexels: &NativeHexels<'_>,
    world_from_local: glam::Affine3A,
) -> Result<re_renderer::QueueableDrawData, ViewSystemExecutionError> {
    re_tracing::profile_function!();
    let count = hexels.centers.len();
    let points: Vec<_> = hexels
        .centers
        .iter()
        .map(|center| re_renderer::PositionRadius {
            pos: glam::Vec3::from_array(*center),
            radius: re_renderer::Size::new_ui_points(1.5),
        })
        .collect();
    let colors: Vec<_> = (0..count)
        .map(|i| {
            let c = at_or_last(hexels.color_field, i, [1.0; 9]);
            let channel = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            re_renderer::Color32::from_rgb(channel(c[0]), channel(c[1]), channel(c[2]))
        })
        .collect();
    let bounds = crate::bounds::from_hexels(hexels)
        .region_of_interest
        .transform_affine3(&world_from_local);
    if ctx
        .viewer_ctx
        .store_context
        .memoizer::<GpuCache, _>(|cache| cache.record_view_bounds(query.view_id, bounds))
    {
        ctx.egui_ctx().request_repaint();
    }
    let mut builder = re_renderer::PointCloudBuilder::new(ctx.render_ctx());
    builder
        .batch("Hexel compute fallback")
        .world_from_obj(world_from_local)
        .add_points(&points, &colors, &[]);
    Ok(builder.into_draw_data()?.into())
}
