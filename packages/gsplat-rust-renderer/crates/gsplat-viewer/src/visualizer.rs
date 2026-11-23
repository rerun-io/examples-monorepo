//! Native archetype queries, memoized uploads, instance transforms and blueprint properties.
use crate::cache::{BatchKey, GpuCache, Primitives};
use crate::renderer::GaussianRenderer;
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

/// Blueprint-only field on a `ComputeGaussianSplats3D` visualizer instruction ("default" or "mip").
/// The data recording still contains only native `GaussianSplats3D` descriptors.
pub(crate) fn render_mode_descriptor() -> re_sdk_types::ComponentDescriptor {
    let archetype = crate::selection::COMPUTE.into();
    re_sdk_types::ComponentDescriptor {
        archetype: Some(archetype),
        component: re_sdk_types::ComponentIdentifier::from_archetype_field(
            archetype,
            "render_mode",
        ),
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

/// One entity instruction in one view: what each of its rows and instances draws with.
pub(crate) struct EntityDraw<'a> {
    pub(crate) ctx: &'a ViewContext<'a>,
    pub(crate) query: &'a ViewQuery<'a>,
    pub(crate) data: &'a DataResult,
    pub(crate) instruction: &'a VisualizerInstruction,
    instances: &'a [glam::DAffine3],
    renderer: &'a GaussianRenderer,
    pub(crate) output: &'a VisualizerExecutionOutput,
    report: &'a dyn Fn(ViewerReportSeverity, String),
    camera: &'a Camera,
    /// The view's solid background, queried once per view.
    background: Option<glam::Vec3>,
}

/// Query and queue one entity instruction; returns whether the view needs another frame.
pub(crate) type DrawEntity = fn(
    &EntityDraw<'_>,
    &mut Vec<re_renderer::QueueableDrawData>,
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
    let background = solid_background(ctx);
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
        let entity = EntityDraw {
            ctx,
            query,
            data,
            instruction,
            instances,
            renderer,
            output: &output,
            report: &report,
            camera: &camera,
            background,
        };
        match draw_entity(&entity, &mut draw) {
            Ok(true) => ctx.egui_ctx().request_repaint(),
            Ok(false) => {}
            Err(error) => report(ViewerReportSeverity::Error, format!("{error:#}")),
        }
    }
    Ok(output.with_draw_data(draw))
}

fn draw_splat_entity(
    entity: &EntityDraw<'_>,
    draw: &mut Vec<re_renderer::QueueableDrawData>,
) -> Result<bool, ViewSystemExecutionError> {
    re_tracing::profile_function!();
    let &EntityDraw {
        ctx,
        query,
        data,
        instruction,
        output,
        report,
        ..
    } = entity;
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
    // TRiGS motion fields (static, untyped f32/f64): evaluated at the time cursor when all are present.
    let motion_ids = crate::trigs::components();
    let motion_query = data.query_components_with_history(
        ctx,
        query,
        motion_ids.iter().copied(),
        instruction,
        None,
    );
    let motion_results = VisualizerInstructionQueryResults::new(instruction, &motion_query, output);
    let [
        m_op,
        m_times,
        m_logdur,
        m_base,
        m_c0,
        m_c1,
        m_c2,
        m_anchors,
        m_static,
        m_fps,
        m_origin,
        m_threshold,
    ] = motion_ids.map(|id| motion_results.iter_optional(id));
    let motion = match (
        last_slice::<f32>(&m_op),
        last_slice::<f32>(&m_times),
        last_slice::<f32>(&m_logdur),
        last_slice::<[f32; 6]>(&m_base),
        last_slice::<[f32; 6]>(&m_c0),
        last_slice::<[f32; 6]>(&m_c1),
        last_slice::<[f32; 6]>(&m_c2),
        last_slice::<[f32; 3]>(&m_anchors),
    ) {
        (
            Some(opacities),
            Some(times),
            Some(log_durations),
            Some(base),
            Some(c0),
            Some(c1),
            Some(c2),
            Some(anchors),
        ) => {
            let fps = last_scalar::<f64>(&m_fps, 1.0);
            let origin = last_scalar::<f64>(&m_origin, 0.0);
            let t = crate::trigs::clip_seconds(
                ctx.recording().timeline_type(&query.timeline),
                query.latest_at,
                fps,
                origin,
            );
            Some((
                gsplat_core::trigs::TrigsMotion {
                    opacities,
                    times,
                    log_durations,
                    twist_base: base,
                    twist_c0: c0,
                    twist_c1: c1,
                    twist_c2: c2,
                    anchors,
                    is_static: last_slice::<f32>(&m_static).unwrap_or(&[]),
                    active_threshold: last_scalar::<f32>(&m_threshold, 0.0),
                },
                t,
            ))
        }
        _ => None,
    };
    // `.asog` payload (static, untyped): drawn instead of the preview rows when present.
    let asog_ids = crate::asog::components();
    let asog_query =
        data.query_components_with_history(ctx, query, asog_ids.iter().copied(), instruction, None);
    let asog_results = VisualizerInstructionQueryResults::new(instruction, &asog_query, output);
    let [a_bytes, a_origin, a_threshold] = asog_ids.map(|id| asog_results.iter_optional(id));
    let asog = last_slice::<u8>(&a_bytes).map(|bytes| {
        let threshold = last_scalar::<f32>(&a_threshold, gsplat_core::trigs::ASOG_ACTIVE_THRESHOLD);
        let origin = last_scalar::<f64>(&a_origin, 0.0);
        (
            bytes,
            threshold,
            origin,
            hash64((
                self::attribute_generation(instruction, &asog_query, asog_ids.into_iter()),
                threshold.to_bits(),
            ))
            .hash64(),
        )
    });
    let time_type = ctx.recording().timeline_type(&query.timeline);
    let motion_generation =
        self::attribute_generation(instruction, &motion_query, motion_ids.iter().copied());
    // SOG payload fields (static, untyped Utf8 + u8): drawn instead of the preview rows when present.
    let sog_ids = crate::sog::components();
    let sog_query =
        data.query_components_with_history(ctx, query, sog_ids.iter().copied(), instruction, None);
    let sog_generation =
        self::attribute_generation(instruction, &sog_query, sog_ids.iter().copied());
    let sog_results = VisualizerInstructionQueryResults::new(instruction, &sog_query, output);
    let sog_iters = sog_ids.map(|id| sog_results.iter_optional(id));
    // meta.json is parsed once here, not once per row.
    let payload =
        sog_webp(&sog_iters).map(|(meta, webp)| (gsplat_core::sog::Meta::parse(&meta), webp));
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
        if let Some((meta, webp)) = &payload {
            // The payload replaces the preview rows; it decodes once per payload (row ids), never per time.
            let meta = match meta {
                Ok(meta) => Some(meta),
                Err(error) => {
                    report(
                        ViewerReportSeverity::Warning,
                        format!("{error}; drawing the GaussianSplats3D preview rows"),
                    );
                    None
                }
            };
            if let Some(meta) = meta {
                let sog = crate::sog::Payload {
                    meta,
                    webp: *webp,
                    preview: &native,
                };
                retry |= draw_instances(
                    entity,
                    draw,
                    row,
                    hash64((signature, sog_generation)),
                    Primitives::Sog(&sog),
                    render_mode,
                    0.0,
                )?;
                continue;
            }
        }
        if let Some((bytes, threshold, origin, generation)) = asog {
            let info = ctx
                .viewer_ctx
                .store_context
                .memoizer::<crate::asog::AsogCache, _>(|cache| cache.info(generation, bytes));
            match info {
                Ok(info) => {
                    // Web: the browser inflates the streams first; nothing of ours draws until they arrive.
                    let egui = ctx.egui_ctx().clone();
                    let browser = ctx
                        .viewer_ctx
                        .store_context
                        .memoizer::<crate::asog::AsogCache, _>(|cache| {
                            cache.browser(generation, bytes, threshold, move || {
                                egui.request_repaint();
                            })
                        });
                    let inflated = match browser {
                        crate::asog::Browser::Pending => continue,
                        crate::asog::Browser::Ready(inflated) => Some(inflated),
                        crate::asog::Browser::Unused => None,
                    };
                    // Decoded once per payload; a time change is one motion dispatch in the view (time not in the key).
                    let t =
                        crate::trigs::clip_seconds(time_type, query.latest_at, info.fps, origin);
                    let payload = crate::asog::Payload {
                        bytes,
                        active_threshold: threshold,
                        info,
                        preview: &native,
                        inflated: inflated.as_deref(),
                    };
                    retry |= draw_instances(
                        entity,
                        draw,
                        row,
                        hash64((signature, generation)),
                        Primitives::Asog(&payload),
                        render_mode,
                        t,
                    )?;
                    if inflated.is_some() {
                        ctx.viewer_ctx
                            .store_context
                            .memoizer::<crate::asog::AsogCache, _>(|cache| {
                                cache.uploaded(generation);
                            });
                    }
                    continue;
                }
                Err(error) => report(
                    ViewerReportSeverity::Warning,
                    format!(
                        "AnimatedSogSplats3D: {error}; drawing the GaussianSplats3D preview rows"
                    ),
                ),
            }
        }
        if let Some((motion, t)) = motion
            .as_ref()
            .filter(|(m, _)| m.is_consistent(centers.len()))
        {
            // Canonical rows + motion uploaded once; a time change is one motion dispatch in the view.
            retry |= draw_instances(
                entity,
                draw,
                row,
                hash64((signature, motion_generation)),
                Primitives::Animated {
                    splats: &native,
                    motion,
                },
                render_mode,
                *t,
            )?;
            continue;
        }
        retry |= draw_instances(
            entity,
            draw,
            row,
            signature,
            Primitives::Splats(&native),
            render_mode,
            0.0,
        )?;
    }
    Ok(retry)
}

/// meta.json and the stored WebP planes of `SogSplats3D`, when meta and the five required planes are present.
fn sog_webp<'a>(
    iters: &'a [re_view::HybridResultsChunkIter<'a>; 8],
) -> Option<(String, gsplat_core::sog::SogWebp<'a>)> {
    let [
        meta,
        means_l,
        means_u,
        quats,
        scales,
        sh0,
        centroids,
        labels,
    ] = iters;
    let meta = meta.slice::<String>().last()?.1.first()?.to_string();
    let webp = gsplat_core::sog::SogWebp {
        means_l: last_slice::<u8>(means_l)?,
        means_u: last_slice::<u8>(means_u)?,
        quats: last_slice::<u8>(quats)?,
        scales: last_slice::<u8>(scales)?,
        sh0: last_slice::<u8>(sh0)?,
        sh_centroids: last_slice::<u8>(centroids),
        sh_labels: last_slice::<u8>(labels),
    };
    Some((meta, webp))
}

/// A cache key from `value` (the std hasher, as `attribute_generation`).
fn hash64(value: impl std::hash::Hash) -> re_log_types::hash::Hash64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    re_log_types::hash::Hash64::from_u64(hasher.finish())
}

/// The first value of the last row of a (static, one-value) optional component, or `default`.
fn last_scalar<'a, T>(iter: &'a re_view::HybridResultsChunkIter<'a>, default: T) -> T
where
    T: 'a + Copy + re_viewer::external::re_chunk::ChunkComponentSlicer<Item<'a> = &'a [T]>,
{
    last_slice(iter)
        .and_then(|v| v.first().copied())
        .unwrap_or(default)
}

/// The values of the last row of a (static, one-row) optional component.
fn last_slice<'a, T>(iter: &'a re_view::HybridResultsChunkIter<'a>) -> Option<&'a [T]>
where
    T: 'a + re_viewer::external::re_chunk::ChunkComponentSlicer<Item<'a> = &'a [T]>,
{
    iter.slice::<T>().last().map(|(_, values)| values)
}

/// Queue one compute batch per instance; capacity or capability failures draw the stock fallback.
pub(crate) fn draw_instances(
    entity: &EntityDraw<'_>,
    draw: &mut Vec<re_renderer::QueueableDrawData>,
    row: (re_log_types::TimeInt, re_sdk_types::RowId),
    signature: re_log_types::hash::Hash64,
    primitives: Primitives<'_>,
    render_mode: RenderMode,
    time: f32,
) -> Result<bool, ViewSystemExecutionError> {
    let &EntityDraw {
        ctx,
        query,
        data,
        instruction,
        instances,
        renderer,
        report,
        camera,
        background,
        ..
    } = entity;
    let mut retry = false;
    let masks = query
        .highlights
        .entity_outline_mask(data.entity_path.hash());
    let object = re_renderer::PickingLayerObjectId(data.entity_path.hash64());
    let uniform = crate::renderer::DrawUniform::new(background, object, masks, primitives.len());
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
            world_from_local: to_core(world_from_local),
            time,
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
                    uniform,
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
                    Primitives::Animated { splats, motion } => {
                        // The rows evaluated on the CPU at this time (never the canonical rows: a ghost trail).
                        report(
                            ViewerReportSeverity::Warning,
                            format!("{error}; using the native Gaussian renderer"),
                        );
                        re_log::warn_once!("Gaussian compute fallback: {error:#}");
                        let evaluated = crate::trigs::Evaluated::new(splats, motion, time);
                        crate::fallback::native_fallback(
                            ctx,
                            query,
                            key,
                            &evaluated.native(splats),
                            world_from_local,
                            hash64((signature, time.to_bits())),
                            masks,
                            object,
                        )?
                    }
                    Primitives::Splats(native)
                    | Primitives::Sog(&crate::sog::Payload {
                        preview: native, ..
                    })
                    | Primitives::Asog(&crate::asog::Payload {
                        preview: native, ..
                    }) => {
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
                            masks,
                            object,
                        )?
                    }
                    Primitives::Hexels(hexels) => {
                        report(
                            ViewerReportSeverity::Warning,
                            format!("{error}; drawing hexel centres as points"),
                        );
                        re_log::warn_once!("Hexel compute fallback: {error:#}");
                        crate::hexels::points_fallback(
                            ctx,
                            query,
                            hexels,
                            world_from_local,
                            masks,
                            object,
                        )?
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
    let size = gsplat_core::glam::UVec2::from_array(
        re_viewer_context::gpu_bridge::viewport_resolution_in_pixels(
            rect,
            ctx.egui_ctx().pixels_per_point(),
        ),
    );
    Some(Camera::from_view(
        to_core(glam::Affine3A::from_mat4(eye.world_from_rub_view.to_mat4())),
        f64::from(fov_y),
        size,
    ))
}

/// Rerun's glam 0.33 transform as gsplat-core's (Brush's) glam 0.30 one.
fn to_core(transform: glam::Affine3A) -> gsplat_core::glam::Affine3A {
    gsplat_core::glam::Affine3A::from_cols_array(&transform.to_cols_array())
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
