//! The `DepthVideoStream` visualizer: decode the sample at the time cursor, then draw it like a depth image.
//!
//! The decoded codes become an `ImageInfo` of kind depth and go through the same public upload as
//! `DepthImage` (`gpu_bridge::image_to_gpu`, an `R16Uint` texture with the colormap), so the pixels match
//! a PNG `EncodedDepthImage` of the same codes. In 2D the texture is a colormapped rectangle; in 3D a
//! `DepthCloud` under the entity's pinhole (copy of `re_view_spatial`'s private
//! `process_entity_view_as_depth_cloud`).

use crate::archetype;
use crate::decoder::{LumaFrame, Sample, StreamDecoder};
use re_renderer::renderer::{
    ColormappedTexture, DepthCloud, DepthCloudDrawData, DepthClouds, RectangleDrawData,
    RectangleOptions, TextureFilterMag, TextureFilterMin, TexturedRect,
};
use re_sdk_types::components::{
    Colormap, DepthMeter, FillRatio, ValueRange, VideoCodec, VideoSample,
};
use re_sdk_types::{RowId, View as _};
use re_view_spatial::{EntityDepthOffsets, TransformTreeContext};
use re_viewer::external::{
    re_chunk_store::LatestAtQuery,
    re_log_types::{EntityPath, EntityPathHash, TimeInt},
    re_renderer, re_sdk_types, re_view, re_view_spatial, re_viewer_context,
};
use re_viewer_context::{
    ColormapWithRange, DataResult, IdentifiedViewSystem, ImageInfo, ImageStatsCache, ViewContext,
    ViewContextCollection, ViewQuery, ViewSystemExecutionError, ViewSystemIdentifier,
    ViewerReportSeverity, VisualizerExecutionOutput, VisualizerInstruction, VisualizerQueryInfo,
    VisualizerSystem, gpu_bridge,
};
use std::collections::HashMap;
use std::sync::Arc;

/// (time, row id) of a sample on the queried timeline.
type SampleKey = (TimeInt, RowId);

#[derive(Default)]
pub struct DepthVideoStreamVisualizer;

impl IdentifiedViewSystem for DepthVideoStreamVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        "DepthVideoStream".into()
    }
}

impl VisualizerSystem for DepthVideoStreamVisualizer {
    fn affinity(&self) -> Option<re_sdk_types::ViewClassIdentifier> {
        Some(re_sdk_types::blueprint::views::Spatial2DView::identifier())
    }

    fn visualizer_query_info(&self, _: &re_viewer_context::AppOptions) -> VisualizerQueryInfo {
        VisualizerQueryInfo::single_required_component::<VideoCodec>(
            &archetype::descriptor_codec(),
            &archetype::all_components(),
        )
    }

    fn execute(
        &self,
        ctx: &ViewContext<'_>,
        query: &ViewQuery<'_>,
        systems: &ViewContextCollection,
    ) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
        re_tracing::profile_function!();
        let output = VisualizerExecutionOutput::default();
        let transforms = systems.get::<TransformTreeContext>(&output)?;
        let depth_offsets = systems.get::<EntityDepthOffsets>(&output)?;
        let is_3d = ctx.view_class_identifier
            == re_sdk_types::blueprint::views::Spatial3DView::identifier();

        let mut clouds = Vec::new();
        let mut rects = Vec::new();
        for (data, instruction) in query.iter_visualizer_instruction_for(Self::identifier()) {
            let entity = data.entity_path.hash();
            let drawn = match depth_texture(ctx, query, data, instruction) {
                Ok(Some(drawn)) => drawn,
                Ok(None) => continue, // Nothing logged at or before the time cursor.
                Err(message) => {
                    output.report_unspecified_source(
                        instruction.id,
                        ViewerReportSeverity::Error,
                        message,
                    );
                    continue;
                }
            };
            let Some(Ok(transform)) = transforms.target_from_entity_path(entity) else {
                continue;
            };
            let outline_mask = query.highlights.entity_outline_mask(entity).overall;
            if is_3d {
                // A depth cloud needs a pinhole at or above the entity.
                let root = transform.tree_root();
                let (Some(pinhole), Some(world_from_view)) = (
                    transforms.pinhole_tree_root_info(root),
                    transforms.target_from_pinhole_root(root),
                ) else {
                    output.report_unspecified_source(
                        instruction.id,
                        ViewerReportSeverity::Error,
                        "Cannot draw depth video as a 3D point cloud: it is not under a pinhole camera."
                            .to_owned(),
                    );
                    continue;
                };
                let pinhole = &pinhole.pinhole_projection;
                let world_from_rdf = world_from_view.as_affine3a()
                    * glam::Affine3A::from_mat3(pinhole.view_coordinates.from_rdf());
                let texture = &drawn.texture;
                let dimensions = glam::UVec2::from_array(texture.texture.width_height());
                let world_depth_from_texture_depth = 1.0 / drawn.meter;
                let fov_y = pinhole
                    .image_from_camera
                    .fov_y(pinhole.resolution.unwrap_or_else(|| [1.0, 1.0].into()));
                let pixel_width_from_depth = (0.5 * fov_y).tan() / (0.5 * dimensions.y as f32);
                clouds.push(DepthCloud {
                    world_from_rdf,
                    depth_camera_intrinsics: pinhole.image_from_camera.0.into(),
                    world_depth_from_texture_depth,
                    point_radius_from_world_depth: drawn.fill_ratio * pixel_width_from_depth,
                    min_max_depth_in_world: texture
                        .range
                        .map(|d| world_depth_from_texture_depth * d),
                    depth_dimensions: dimensions,
                    depth_texture: texture.texture.clone(),
                    colormap: match texture.color_mapper {
                        re_renderer::renderer::ColorMapper::Function(colormap) => colormap,
                        _ => re_renderer::Colormap::Grayscale,
                    },
                    outline_mask_id: outline_mask,
                    picking_object_id: re_renderer::PickingLayerObjectId(data.entity_path.hash64()),
                });
            } else {
                let world_from_entity = transform.target_from_instances().first().as_affine3a();
                let [width, height] = drawn.texture.texture.width_height();
                rects.push(TexturedRect {
                    top_left_corner_position: world_from_entity.transform_point3(glam::Vec3::ZERO),
                    extent_u: world_from_entity.transform_vector3(glam::Vec3::X * width as f32),
                    extent_v: world_from_entity.transform_vector3(glam::Vec3::Y * height as f32),
                    colormapped_texture: drawn.texture,
                    // Nearest both ways, as for colormapped depth images: interpolating codes before
                    // the colormap invents depths.
                    options: RectangleOptions {
                        texture_filter_magnification: TextureFilterMag::Nearest,
                        texture_filter_minification: TextureFilterMin::Nearest,
                        multiplicative_tint: re_renderer::Rgba::WHITE,
                        depth_offset: depth_offsets
                            .per_entity_and_visualizer
                            .get(&(Self::identifier(), entity))
                            .copied()
                            .unwrap_or_default(),
                        outline_mask,
                    },
                });
            }
        }

        let render_ctx = ctx.render_ctx();
        let mut draw_data: Vec<re_renderer::QueueableDrawData> = Vec::new();
        if !clouds.is_empty() {
            let clouds = DepthClouds {
                clouds,
                radius_boost_in_ui_points_for_outlines:
                    re_view::SIZE_BOOST_IN_POINTS_FOR_POINT_OUTLINES,
            };
            let cloud = DepthCloudDrawData::new(render_ctx, &clouds)
                .map_err(|err| ViewSystemExecutionError::DrawDataCreationError(Arc::new(err)))?;
            draw_data.push(cloud.into());
        }
        if !rects.is_empty() {
            let rects = RectangleDrawData::new(render_ctx, &rects)
                .map_err(|err| ViewSystemExecutionError::DrawDataCreationError(Arc::new(err)))?;
            draw_data.push(rects.into());
        }
        Ok(output.with_draw_data(draw_data))
    }
}

/// The decoded frame at the time cursor as a colormapped GPU texture, with the cloud parameters.
struct DrawnDepth {
    texture: ColormappedTexture,
    meter: f32,
    fill_ratio: f32,
}

fn depth_texture(
    ctx: &ViewContext<'_>,
    query: &ViewQuery<'_>,
    data: &DataResult,
    instruction: &VisualizerInstruction,
) -> Result<Option<DrawnDepth>, String> {
    let latest_at = query.latest_at_query();
    let codec_c = archetype::descriptor_codec().component;
    let meter_c = archetype::descriptor_meter().component;
    let colormap_c = archetype::descriptor_colormap().component;
    let range_c = archetype::descriptor_depth_range().component;
    let fill_c = archetype::descriptor_point_fill_ratio().component;
    let sample_c = archetype::descriptor_sample().component;

    // Properties may be overridden in the blueprint; samples are read straight from the store.
    let properties = re_view::latest_at_with_blueprint_resolved_data(
        ctx,
        None,
        &latest_at,
        data,
        [codec_c, meter_c, colormap_c, range_c, fill_c],
        Some(instruction),
    );
    match properties.get_mono::<VideoCodec>(codec_c) {
        Some(VideoCodec::AV1) => {}
        Some(codec) => return Err(format!("DepthVideoStream supports AV1 only, got {codec:?}")),
        None => return Err("DepthVideoStream has no codec".to_owned()),
    }

    let entity_path = &data.entity_path;
    let read_sample = |at: &LatestAtQuery| -> Option<Sample<SampleKey>> {
        let results = ctx.recording().latest_at(at, entity_path, [sample_c]);
        let VideoSample(blob) = results.component_mono_quiet::<VideoSample>(sample_c)?;
        let key = results.max_index();
        Some(Sample {
            key,
            is_keyframe: re_video::is_start_of_gop(&blob, re_video::VideoCodec::AV1)
                .unwrap_or(false),
            data: blob.to_vec(),
        })
    };
    let Some(target) = read_sample(&latest_at) else {
        return Ok(None);
    };
    let target_key = target.key;
    let previous = |(time, _): SampleKey| {
        // One sample per time point, as for `VideoStream`; a static sample has no predecessor.
        if time.is_static() || time == TimeInt::MIN {
            return None;
        }
        read_sample(&LatestAtQuery::new(query.timeline, time.as_i64() - 1))
    };

    let image = ctx
        .viewer_ctx
        .store_context
        .memoizer::<DecoderCache, _>(|cache| cache.image(entity_path, target, previous))
        .map_err(|err| format!("Failed to decode depth video: {err}"))?;
    let frame_max = ((1_u32 << image.bit_depth) - 1) as f32;

    let meter = properties
        .get_mono::<DepthMeter>(meter_c)
        .map_or(archetype::DEFAULT_METER, |m| *m.0);
    let colormap = properties
        .get_mono::<Colormap>(colormap_c)
        .unwrap_or(archetype::DEFAULT_COLORMAP);
    let value_range = properties
        .get_mono::<ValueRange>(range_c)
        .map_or([0.0, frame_max], |r| [r.0.0[0] as f32, r.0.0[1] as f32]);
    let fill_ratio = properties
        .get_mono::<FillRatio>(fill_c)
        .map_or(1.0, |f| *f.0);

    let image_info = image.info;
    let stats = ctx
        .viewer_ctx
        .store_context
        .memoizer_read_or_compute::<ImageStatsCache, _, _>(&image_info);
    let texture = gpu_bridge::image_to_gpu(
        ctx.render_ctx(),
        &format!("{entity_path} @ {target_key:?}"),
        &image_info,
        &stats,
        None,
        Some(&ColormapWithRange {
            colormap,
            value_range,
        }),
    )
    .map_err(|err| format!("{err:#}"))?;
    Ok(Some(DrawnDepth {
        texture,
        meter,
        fill_ratio,
    }))
}

/// One decoded frame as a depth image, keyed by its sample's row id for the texture cache.
#[derive(Clone)]
struct DecodedImage {
    info: ImageInfo,
    bit_depth: u8,
}

impl DecodedImage {
    fn new(row_id: RowId, frame: &LumaFrame) -> Self {
        let bytes: Vec<u8> = frame.codes.iter().flat_map(|c| c.to_le_bytes()).collect();
        Self {
            info: ImageInfo::from_stored_blob(
                row_id,
                archetype::descriptor_sample().component,
                bytes.into(),
                re_sdk_types::components::ImageFormat::depth(
                    [frame.width, frame.height],
                    re_sdk_types::datatypes::ChannelDatatype::U16,
                )
                .0,
                re_sdk_types::image::ImageKind::Depth,
            ),
            bit_depth: frame.bit_depth,
        }
    }
}

/// Per-recording decoder state, one stream decoder per entity.
#[derive(Default)]
struct DecoderCache {
    frame: u64,
    streams: HashMap<EntityPathHash, CachedStream>,
}

struct CachedStream {
    decoder: StreamDecoder<SampleKey>,
    shown: Option<(SampleKey, DecodedImage)>,
    last_used: u64,
}

impl DecoderCache {
    fn image(
        &mut self,
        entity: &EntityPath,
        target: Sample<SampleKey>,
        previous: impl FnMut(SampleKey) -> Option<Sample<SampleKey>>,
    ) -> Result<DecodedImage, crate::decoder::DecodeError> {
        let stream = match self.streams.entry(entity.hash()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(CachedStream {
                decoder: StreamDecoder::new()?,
                shown: None,
                last_used: 0,
            }),
        };
        stream.last_used = self.frame;
        if let Some((key, image)) = &stream.shown
            && *key == target.key
        {
            return Ok(image.clone());
        }
        let key = target.key;
        let frame = stream.decoder.frame(target, previous)?;
        let image = DecodedImage::new(key.1, &frame);
        stream.shown = Some((key, image.clone()));
        Ok(image)
    }
}

impl re_viewer_context::Cache for DecoderCache {
    fn name(&self) -> &'static str {
        "DepthVideoStream decoders"
    }

    fn begin_frame(&mut self) {
        // Keep a stream's decoder while it is on screen, plus a second for blueprint transitions.
        self.streams
            .retain(|_, stream| stream.last_used + 60 >= self.frame);
        self.frame += 1;
    }

    fn purge_memory(&mut self) {
        self.streams.clear();
    }
}

impl re_byte_size::MemUsageTreeCapture for DecoderCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        let bytes = self
            .streams
            .values()
            .filter_map(|stream| stream.shown.as_ref())
            .map(|(_, image)| image.info.buffer.len() as u64)
            .sum();
        re_byte_size::MemUsageTree::Bytes(bytes)
    }
}
