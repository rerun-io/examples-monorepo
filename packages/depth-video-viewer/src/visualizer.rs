//! The `DepthVideoStream` visualizer: decode the sample at the time cursor, then draw it like a depth image.
//!
//! Out-of-tree scaffolding, deleted once `DepthVideoStream` ships in Rerun; there the visualizer would be a thin
//! adapter over the built-in video stream path instead of this per-entity decoder cache.
//!
//! The decoded codes become an `ImageInfo` of kind depth and go through the same public upload as
//! `DepthImage` (`gpu_bridge::image_to_gpu`, an `R16Uint` texture with the colormap), so the pixels match
//! a PNG `EncodedDepthImage` of the same codes. In 2D the texture is a colormapped rectangle; in 3D a
//! `DepthCloud` under the entity's pinhole (copy of `re_view_spatial`'s private
//! `process_entity_view_as_depth_cloud`).

use crate::archetype::{self, Field};
use crate::decoder::{DecodeError, LumaFrame, Sample, StreamDecoder, latest_at};
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
    arrow::buffer::ScalarBuffer,
    re_chunk_store::{ChunkStoreEvent, ChunkTrackingMode, RangeQuery},
    re_entity_db::EntityDb,
    re_log_types::{AbsoluteTimeRange, EntityPath, EntityPathHash, TimeInt, TimelineName},
    re_renderer, re_sdk_types, re_view, re_view_spatial, re_viewer_context,
};
use re_viewer_context::{
    ColormapWithRange, DataResult, IdentifiedViewSystem, ImageInfo, ImageStats, ViewContext,
    ViewContextCollection, ViewQuery, ViewSystemExecutionError, ViewSystemIdentifier,
    ViewerReportSeverity, VisualizerExecutionOutput, VisualizerInstruction, VisualizerQueryInfo,
    VisualizerSystem, gpu_bridge,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// (time, row id) of a sample on the queried timeline: the order samples decode in.
type SampleKey = (TimeInt, RowId);

/// A sample as stored: the blob shares the store's buffer.
type StoredSample = Sample<SampleKey, ScalarBuffer<u8>>;

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
            &Field::Codec.descriptor(),
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
            let report = |message: String| {
                output.report_unspecified_source(
                    instruction.id,
                    ViewerReportSeverity::Error,
                    message,
                );
            };

            // Where to draw comes first: nothing is decoded for an entity that cannot be placed.
            let transform = match transforms.target_from_entity_path(entity) {
                Some(Ok(transform)) => transform,
                Some(Err(err)) => {
                    report(format!("Invalid transform: {err}"));
                    continue;
                }
                None => {
                    report("No transform from this entity to the view's origin.".to_owned());
                    continue;
                }
            };
            // A depth cloud needs a pinhole at or above the entity.
            let pinhole = if is_3d {
                let root = transform.tree_root();
                let (Some(pinhole), Some(world_from_view)) = (
                    transforms.pinhole_tree_root_info(root),
                    transforms.target_from_pinhole_root(root),
                ) else {
                    report(
                        "Cannot draw depth video as a 3D point cloud: it is not under a pinhole camera."
                            .to_owned(),
                    );
                    continue;
                };
                Some((&pinhole.pinhole_projection, world_from_view))
            } else {
                None
            };

            let drawn = match depth_texture(ctx, query, data, instruction, &output) {
                Ok(Some(drawn)) => drawn,
                Ok(None) => continue, // Nothing at or before the time cursor, or chunks still loading.
                Err(message) => {
                    report(message);
                    continue;
                }
            };
            let outline_mask = query.highlights.entity_outline_mask(entity).overall;
            if let Some((pinhole, world_from_view)) = pinhole {
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
    output: &VisualizerExecutionOutput,
) -> Result<Option<DrawnDepth>, String> {
    let [codec_c, meter_c, colormap_c, range_c, fill_c] = [
        Field::Codec,
        Field::Meter,
        Field::Colormap,
        Field::DepthRange,
        Field::PointFillRatio,
    ]
    .map(Field::component);

    // Properties may be overridden in the blueprint; samples are read straight from the store.
    let properties = re_view::latest_at_with_blueprint_resolved_data(
        ctx,
        None,
        &query.latest_at_query(),
        data,
        [codec_c, meter_c, colormap_c, range_c, fill_c],
        Some(instruction),
    );
    if properties.any_missing_chunks() {
        output.set_missing_chunks();
        return Ok(None);
    }
    match properties.get_mono::<VideoCodec>(codec_c) {
        Some(VideoCodec::AV1) => {}
        Some(codec) => return Err(format!("DepthVideoStream supports AV1 only, got {codec:?}")),
        None => return Err("DepthVideoStream has no codec".to_owned()),
    }

    let entity_path = &data.entity_path;
    let mut partial = false;
    let decoded = ctx
        .viewer_ctx
        .store_context
        .memoizer::<DecoderCache, _>(|cache| {
            cache.frame(
                (entity_path.hash(), query.timeline),
                query.latest_at,
                || {
                    let samples = read_samples(ctx.recording(), entity_path, query.timeline);
                    partial = samples.is_none();
                    samples
                },
            )
        })
        .map_err(|err| format!("Failed to decode depth video: {err}"))?;
    if partial {
        output.set_missing_chunks();
    }
    let Some((row_id, frame)) = decoded else {
        return Ok(None);
    };

    let meter = properties
        .get_mono::<DepthMeter>(meter_c)
        .map_or(archetype::DEFAULT_METER, |m| *m.0);
    let colormap = properties
        .get_mono::<Colormap>(colormap_c)
        .unwrap_or(ColormapWithRange::DEFAULT_DEPTH_COLORMAP);
    let value_range = properties.get_mono::<ValueRange>(range_c).map_or_else(
        || [0.0, ((1_u32 << frame.bit_depth) - 1) as f32],
        |r| [r.0.0[0] as f32, r.0.0[1] as f32],
    );
    let fill_ratio = properties
        .get_mono::<FillRatio>(fill_c)
        .map_or(1.0, |f| *f.0);

    // Keyed by the sample's row id, so the texture manager uploads each decoded frame once.
    let image = ImageInfo::from_stored_blob(
        row_id,
        Field::Sample.component(),
        frame.data.clone(),
        re_sdk_types::components::ImageFormat::depth(
            [frame.width, frame.height],
            re_sdk_types::datatypes::ChannelDatatype::U16,
        )
        .0,
        re_sdk_types::image::ImageKind::Depth,
    );
    // With an explicit colormap range the depth upload never reads the image statistics.
    let no_stats = ImageStats {
        range: None,
        finite_range: (0.0, 0.0),
    };
    let texture = gpu_bridge::image_to_gpu(
        ctx.render_ctx(),
        &format!("{entity_path} @ {row_id}"),
        &image,
        &no_stats,
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

/// All samples of an entity on a timeline in decode order, or `None` while some of its chunks are not loaded.
fn read_samples(
    recording: &EntityDb,
    entity_path: &EntityPath,
    timeline: TimelineName,
) -> Option<Vec<StoredSample>> {
    re_tracing::profile_function!();
    let component = Field::Sample.component();
    let results = recording.storage_engine().cache().range(
        ChunkTrackingMode::Report,
        &RangeQuery::new(timeline, AbsoluteTimeRange::EVERYTHING),
        entity_path,
        [component],
    );
    if results.is_partial() {
        return None;
    }
    let mut samples: Vec<_> = results
        .get(component)
        .unwrap_or(&[])
        .iter()
        .flat_map(|chunk| {
            chunk
                .iter_component_indices(timeline, component)
                .zip(chunk.iter_component::<VideoSample>(component))
                .filter_map(|(key, batch)| {
                    Some(Sample {
                        key,
                        data: batch.first()?.0.0.clone(),
                    })
                })
        })
        .collect();
    samples.sort_by_key(|sample| sample.key);
    Some(samples)
}

/// Per-recording decoder state: a sample index and a decoder per entity and timeline.
#[derive(Default)]
struct DecoderCache {
    streams: BTreeMap<(EntityPathHash, TimelineName), CachedStream>,
}

struct CachedStream {
    /// Rebuilt after the entity's chunks change.
    samples: Option<Vec<StoredSample>>,
    decoder: StreamDecoder<SampleKey>,
    used: bool,
}

impl DecoderCache {
    /// The frame shown at `time` and its sample's row id; builds the sample index with `read_samples` if needed.
    fn frame(
        &mut self,
        key: (EntityPathHash, TimelineName),
        time: TimeInt,
        read_samples: impl FnOnce() -> Option<Vec<StoredSample>>,
    ) -> Result<Option<(RowId, Arc<LumaFrame>)>, DecodeError> {
        let stream = match self.streams.entry(key) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => entry.insert(CachedStream {
                samples: None,
                decoder: StreamDecoder::new()?,
                used: false,
            }),
        };
        stream.used = true;
        if stream.samples.is_none() {
            stream.samples = read_samples();
        }
        let Some(samples) = &stream.samples else {
            return Ok(None);
        };
        let Some(target) = latest_at(samples, time) else {
            return Ok(None);
        };
        let frame = stream.decoder.frame(samples, target)?;
        Ok(Some((samples[target].key.1, frame)))
    }
}

impl re_viewer_context::Cache for DecoderCache {
    fn name(&self) -> &'static str {
        "DepthVideoStream decoders"
    }

    fn begin_frame(&mut self) {
        for stream in self.streams.values_mut() {
            stream.used = false;
        }
    }

    fn purge_memory(&mut self) {
        // Called before `begin_frame`: keep what the last frame drew.
        self.streams.retain(|_, stream| stream.used);
    }

    fn on_store_events(&mut self, events: &[&ChunkStoreEvent], _: &EntityDb) {
        for event in events {
            if event.is_virtual_addition() {
                self.streams.clear();
            } else if let Some(chunk) = event.delta_chunk() {
                let entity = chunk.entity_path().hash();
                for ((stream_entity, _), stream) in &mut self.streams {
                    if *stream_entity == entity {
                        stream.samples = None;
                    }
                }
            }
        }
    }
}

impl re_byte_size::MemUsageTreeCapture for DecoderCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        // The sample index shares the store's buffers; the decoded frame is ours.
        let bytes = self
            .streams
            .values()
            .filter_map(|stream| stream.decoder.last())
            .map(|frame| frame.data.len() as u64)
            .sum();
        re_byte_size::MemUsageTree::Bytes(bytes)
    }
}
