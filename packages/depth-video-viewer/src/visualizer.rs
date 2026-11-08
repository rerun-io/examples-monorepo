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
use crate::decoder::{DecodeError, LumaFrame, MAX_GOP, Sample, StreamDecoder, latest_at};
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

/// A row of the sample component as stored: its blob, which shares the store's buffer, or `None` for an empty
/// batch, which clears the entity from that time on.
type StoredRow = Sample<SampleKey, Option<ScalarBuffer<u8>>>;

/// What a loaded row holds. The index keeps only this, no sample bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowKind {
    Clear,
    Keyframe,
    Frame,
}

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
    let components = [
        Field::Codec,
        Field::Meter,
        Field::Colormap,
        Field::DepthRange,
        Field::PointFillRatio,
    ]
    .map(Field::component);
    let [codec_c, meter_c, colormap_c, range_c, fill_c] = components;

    // Properties may be overridden in the blueprint; samples are read straight from the store.
    let properties = re_view::latest_at_with_blueprint_resolved_data(
        ctx,
        None,
        &query.latest_at_query(),
        data,
        components,
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
    let shown = ctx
        .viewer_ctx
        .store_context
        .memoizer::<DecoderCache, _>(|cache| {
            cache.frame(
                ctx.recording(),
                entity_path,
                query.timeline,
                query.latest_at,
            )
        })
        .map_err(|err| format!("Failed to decode depth video: {err}"))?;
    let (row_id, frame) = match shown {
        Shown::Frame(row_id, frame) => (row_id, frame),
        Shown::Nothing => return Ok(None),
        Shown::Loading => {
            output.set_missing_chunks();
            return Ok(None);
        }
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
        .map_or(archetype::DEFAULT_POINT_FILL_RATIO, |f| *f.0);

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

/// The rows of the sample component in `range`, in decode order, and whether some of their chunks are not loaded.
fn read_rows(
    recording: &EntityDb,
    entity_path: &EntityPath,
    timeline: TimelineName,
    range: AbsoluteTimeRange,
    tracking: ChunkTrackingMode,
) -> (Vec<StoredRow>, bool) {
    re_tracing::profile_function!();
    let component = Field::Sample.component();
    let results = recording.storage_engine().cache().range(
        tracking,
        &RangeQuery::new(timeline, range),
        entity_path,
        [component],
    );
    let mut rows: Vec<_> = results
        .get(component)
        .unwrap_or(&[])
        .iter()
        .flat_map(|chunk| {
            chunk
                .iter_component_indices(timeline, component)
                .zip(chunk.iter_component::<VideoSample>(component))
                .map(|(key, batch)| Sample {
                    key,
                    data: batch.first().map(|sample| sample.0.0.clone()),
                })
        })
        .collect();
    rows.sort_by_key(|row| row.key);
    (rows, results.is_partial())
}

/// The loaded rows of the sample component on a timeline, without their bytes.
fn read_index(
    recording: &EntityDb,
    entity_path: &EntityPath,
    timeline: TimelineName,
) -> Vec<Sample<SampleKey, RowKind>> {
    // Ignore: the index only bounds the span a decode reads, and that read reports what is missing.
    let (rows, _) = read_rows(
        recording,
        entity_path,
        timeline,
        AbsoluteTimeRange::EVERYTHING,
        ChunkTrackingMode::Ignore,
    );
    rows.into_iter()
        .map(|row| Sample {
            key: row.key,
            data: match row.data {
                None => RowKind::Clear,
                Some(blob)
                    if re_video::is_start_of_gop(&blob, re_video::VideoCodec::AV1)
                        .unwrap_or(false) =>
                {
                    RowKind::Keyframe
                }
                Some(_) => RowKind::Frame,
            },
        })
        .collect()
}

/// The first time a decode for `time` reads: the keyframe before the loaded row shown at `time`, or that row if it is
/// a clear. Without either among the loaded rows, the start of the timeline: the read then reports what to load.
fn span_start(index: &[Sample<SampleKey, RowKind>], time: TimeInt) -> TimeInt {
    let Some(target) = latest_at(index, time) else {
        return TimeInt::MIN;
    };
    if index[target].data == RowKind::Clear {
        return index[target].key.0;
    }
    index[..=target]
        .iter()
        .rev()
        .filter(|row| row.data != RowKind::Clear)
        .take(MAX_GOP)
        .find(|row| row.data == RowKind::Keyframe)
        .map_or(TimeInt::MIN, |row| row.key.0)
}

/// What an entity shows at the time cursor.
enum Shown {
    /// Chunks the decode needs are not loaded yet; they have been requested.
    Loading,
    /// No sample at or before the cursor, or a clear.
    Nothing,
    /// The decoded frame and its sample's row id.
    Frame(RowId, Arc<LumaFrame>),
}

/// Per-recording decoder state: a loaded-row index and a decoder per entity and timeline.
#[derive(Default)]
struct DecoderCache {
    streams: BTreeMap<(EntityPathHash, TimelineName), CachedStream>,
}

struct CachedStream {
    /// Rebuilt after the entity's sample chunks change.
    index: Option<Vec<Sample<SampleKey, RowKind>>>,
    decoder: StreamDecoder<SampleKey>,
    used: bool,
}

impl DecoderCache {
    /// The frame `entity_path` shows at `time`. Only the rows from the keyframe to `time` are read (and their chunks
    /// requested); clears never reach the decoder, which keeps its reference state across them.
    fn frame(
        &mut self,
        recording: &EntityDb,
        entity_path: &EntityPath,
        timeline: TimelineName,
        time: TimeInt,
    ) -> Result<Shown, DecodeError> {
        let stream = match self.streams.entry((entity_path.hash(), timeline)) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => entry.insert(CachedStream {
                index: None,
                decoder: StreamDecoder::new()?,
                used: false,
            }),
        };
        stream.used = true;
        let index = stream
            .index
            .get_or_insert_with(|| read_index(recording, entity_path, timeline));
        let span = AbsoluteTimeRange::new(span_start(index, time), time);
        let (rows, partial) = read_rows(
            recording,
            entity_path,
            timeline,
            span,
            ChunkTrackingMode::Report,
        );
        if partial {
            return Ok(Shown::Loading);
        }
        if rows.last().is_none_or(|row| row.data.is_none()) {
            return Ok(Shown::Nothing);
        }
        let samples: Vec<_> = rows
            .into_iter()
            .filter_map(|row| {
                Some(Sample {
                    key: row.key,
                    data: row.data?,
                })
            })
            .collect();
        let target = samples.len() - 1;
        let frame = stream.decoder.frame(&samples, target)?;
        Ok(Shown::Frame(samples[target].key.1, frame))
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
        let component = Field::Sample.component();
        for event in events {
            if event.is_virtual_addition() {
                self.streams.clear();
            } else if let Some(chunk) = event.delta_chunk()
                && chunk.components().contains_component(component)
            {
                let entity = chunk.entity_path().hash();
                for ((stream_entity, _), stream) in &mut self.streams {
                    if *stream_entity == entity {
                        stream.index = None;
                    }
                }
            }
        }
    }
}

impl re_byte_size::MemUsageTreeCapture for DecoderCache {
    fn capture_mem_usage_tree(&self) -> re_byte_size::MemUsageTree {
        // The index holds no sample bytes; the decoded frame is ours.
        let bytes = self
            .streams
            .values()
            .filter_map(|stream| stream.decoder.last())
            .map(|frame| frame.data.len() as u64)
            .sum();
        re_byte_size::MemUsageTree::Bytes(bytes)
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_wrap,
    reason = "frame indices of a 24-frame fixture"
)]
mod tests {
    use super::*;
    use crate::decoder::testdata::{FRAMES, codes, expected, units};
    use re_viewer::external::re_chunk::{Chunk, ChunkBuilder, ChunkId, TimePoint, Timeline};
    use re_viewer::external::re_log_types::{StoreId, StoreKind};

    const ENTITY: &str = "depth";

    fn timeline() -> Timeline {
        Timeline::new_sequence("frame")
    }

    fn new_db() -> EntityDb {
        EntityDb::new(StoreId::random(StoreKind::Recording, "depth_video_test"))
    }

    /// One chunk with a row per `(time, samples)`; an empty list is a clear.
    fn chunk(rows: &[(i64, Vec<VideoSample>)]) -> Arc<Chunk> {
        let mut builder = ChunkBuilder::new(ChunkId::new(), ENTITY.into());
        for (time, samples) in rows {
            builder = builder.with_component_batch(
                RowId::new(),
                TimePoint::from_iter([(timeline(), *time)]),
                (Field::Sample.descriptor(), samples as _),
            );
        }
        Arc::new(builder.build().unwrap())
    }

    fn frame_rows(
        frames: std::ops::Range<usize>,
        time: impl Fn(usize) -> i64,
    ) -> Vec<(i64, Vec<VideoSample>)> {
        let units = units();
        frames
            .map(|f| (time(f), vec![VideoSample::from(units[f].clone())]))
            .collect()
    }

    fn add(db: &mut EntityDb, cache: &mut DecoderCache, chunk: &Arc<Chunk>) {
        let events = db.add_chunk(chunk).unwrap();
        re_viewer_context::Cache::on_store_events(cache, &events.iter().collect::<Vec<_>>(), db);
    }

    fn shown(db: &EntityDb, cache: &mut DecoderCache, time: i64) -> Result<Shown, DecodeError> {
        cache.frame(
            db,
            &ENTITY.into(),
            *timeline().name(),
            TimeInt::new_temporal(time),
        )
    }

    fn assert_frame(db: &EntityDb, cache: &mut DecoderCache, time: i64, f: usize) {
        match shown(db, cache, time) {
            Ok(Shown::Frame(_, frame)) => assert_eq!(codes(&frame), expected(f), "time {time}"),
            Ok(Shown::Loading) => panic!("time {time}: loading, expected frame {f}"),
            Ok(Shown::Nothing) => panic!("time {time}: nothing, expected frame {f}"),
            Err(err) => panic!("time {time}: {err}, expected frame {f}"),
        }
    }

    /// Frames 0-4 at times 0-4, a clear at 5, frames 5-9 at 6-10 (one GOP): the clear hides the depth until the next
    /// frame, and later frames still decode from the history before it, after seeks in either direction.
    #[test]
    fn a_clear_hides_the_depth_until_the_next_frame() {
        let mut rows = frame_rows(0..5, |f| f as i64);
        rows.push((5, vec![]));
        rows.extend(frame_rows(5..10, |f| f as i64 + 1));
        let mut db = new_db();
        let mut cache = DecoderCache::default();
        add(&mut db, &mut cache, &chunk(&rows));

        assert_frame(&db, &mut cache, 3, 3);
        assert!(matches!(shown(&db, &mut cache, 5), Ok(Shown::Nothing)));
        assert_frame(&db, &mut cache, 6, 5);
        assert_frame(&db, &mut cache, 2, 2);
        assert!(matches!(shown(&db, &mut cache, 5), Ok(Shown::Nothing)));
        assert_frame(&db, &mut cache, 10, 9);
        assert_frame(&db, &mut cache, 4, 4);
    }

    /// A row holding one zero-byte sample is a sample, not a clear: the decoder rejects it (rav1d would abort).
    #[test]
    fn a_zero_byte_sample_is_not_a_clear() {
        let mut rows = frame_rows(0..3, |f| f as i64);
        rows.push((3, vec![VideoSample::from(Vec::<u8>::new())]));
        let mut db = new_db();
        let mut cache = DecoderCache::default();
        add(&mut db, &mut cache, &chunk(&rows));
        assert_frame(&db, &mut cache, 2, 2);
        assert!(matches!(
            shown(&db, &mut cache, 3),
            Err(DecodeError::EmptySample)
        ));
        assert_frame(&db, &mut cache, 1, 1);
    }

    /// Both chunks are in the manifest, only the first GOP is loaded: its frames draw, a later time waits for its
    /// chunk, and draws once that chunk arrives.
    #[test]
    fn an_unloaded_later_chunk_does_not_block_the_current_gop() {
        let first = chunk(&frame_rows(0..10, |f| f as i64));
        let later = chunk(&frame_rows(10..FRAMES, |f| f as i64));
        let mut db = new_db();
        let mut builder = re_chunk_index::RrdManifestBuilder::default();
        for (i, chunk) in [&first, &later].into_iter().enumerate() {
            let span = re_span::Span {
                start: 100 * i as u64,
                len: 100,
            };
            builder
                .append(&chunk.to_chunk_batch().unwrap(), span, 100)
                .unwrap();
        }
        let manifest = builder.build(db.store_id().clone()).unwrap();
        let manifest = re_chunk_index::RrdManifest::try_new(&manifest).unwrap();
        let mut cache = DecoderCache::default();
        let events = db.add_rrd_manifest_message(Arc::new(manifest));
        re_viewer_context::Cache::on_store_events(
            &mut cache,
            &events.iter().collect::<Vec<_>>(),
            &db,
        );
        assert!(matches!(shown(&db, &mut cache, 5), Ok(Shown::Loading)));

        add(&mut db, &mut cache, &first);
        assert_frame(&db, &mut cache, 5, 5);
        assert_frame(&db, &mut cache, 9, 9);
        assert!(matches!(shown(&db, &mut cache, 15), Ok(Shown::Loading)));

        add(&mut db, &mut cache, &later);
        assert_frame(&db, &mut cache, 15, 15);
        assert_frame(&db, &mut cache, 3, 3);
    }
}
