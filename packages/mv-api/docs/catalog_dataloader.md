# Catalog dataloader: native-rate stock sampling

The dataloader in `src/mv_api/api/catalog_prediction_layer.py` pulls exo-camera AV1
frames from a catalog segment and runs pose detection on them. It uses Rerun's
**stock** dataloader — `RerunIterableDataset` + `Field(VideoFrameDecoder)` +
`FixedRateSampling` — sampled at each segment's **native frame rate**. There is no
bespoke sample index, no exact-packet targeting, and no `Field.window`.

This note records *why* native rate is the load-bearing choice, because an earlier
version of this code (and this doc) argued the opposite.

## Why native rate

Reliable AV1 decode from the catalog needs the decoder to see a **contiguous packet
run** from a keyframe forward. Sampling `video_time` at the stream's native fps requests
every packet in order, so the decoder always gets that contiguous run and decodes cleanly.

Measured live on the `73ce701` build (reality main, incl. #2073), stock
`RerunIterableDataset` + `FixedRateSampling`, no bespoke code:

| sampling | decode result |
| --- | --- |
| **native fps** (this code) | **OK — 26/26 stream-checks, 12/12 frames each, 0 skips** |
| native ÷ 2 (2:1 decimation) | assembly101 OK *warm* but THROW *cold*; hot3d/aria `InvalidDataError` |
| native ÷ 3 (3:1 decimation) | `InvalidDataError` everywhere |

Native rate decoded every stream across assembly101 / hot3d-quest3 / aria-gen2. Live
verify on the actual mv-api path: all 8 assembly101 exo cams (C10095…C10404, uniform
60 fps) decode 20/20 through one `FixedRateSampling(rate_hz=60)` grid.

**Decimation is graded and flaky.** The same assembly101 stream at 30 Hz decoded in a
warm process but threw `InvalidDataError` in a cold one — an order/cache-dependent state
in the decoder. Sub-native sampling is also *non-monotonic*: 54 Hz (0.9×) throws while
45 Hz, 30 Hz and every oversample (1.5–3×) decode. So the safe invariant is
**`rate_hz >= native_fps`**; never decimate AV1 here.

## What reality #2073 changed (and didn't)

`video_time` is a **duration** timeline (a stopwatch), not a **timestamp**. Older builds
crashed when fixed-rate-sampling a duration timeline (`int(Timedelta)` /
`Duration(ns) <= Int64`). **reality #2073** fixed that **sampler** crash — which is what
makes stock `FixedRateSampling` over `video_time` viable at all — and we are pinned to a
build that includes it (`73ce701`, which also carries #2496's fast catalog register).

#2073 did **not** touch the **decoder**. Sub-native decimated AV1 still throws (table
above); that residual decoder bug is filed upstream at
`rerun-io/rerun-av1-duration-decode-repro` (`ISSUE.md`). Native-rate sampling sidesteps it
entirely, so no upstream fix is needed for this pipeline.

## Native-fps detection

`build_rerun_iterable_dataset` calls `detect_uniform_native_fps`, which:

- reads each exo stream's `video_time` packet timestamps and derives its rate from the
  **median** inter-packet gap (`native_fps_from_packet_ns`);
- **rejects irregular spacing** (`max_gap / median_gap >= 1.5`) so a dropped-frame artifact
  can't under-estimate the rate into the sub-native throw zone;
- **hard-asserts every exo stream shares one native fps**. A single `FixedRateSampling`
  grid drives all camera fields at one shared timestamp, which keeps the cameras in
  multiview lock-step. If the cameras had *different* native rates, `fill_latest_at` would
  silently duplicate the slower camera's frames as fresh instants and corrupt
  triangulation with no error — so mismatched rates raise instead.

assembly101's exo cams are uniform 60 fps, so one grid aligns them all. A future
mixed-fps exo rig would need its own shared-grid design; the assertion fails loudly rather
than corrupt silently. `config.native_fps_override` is the manual escape hatch.

Switching from the old fixed 30 Hz to native 60 fps is a **behavioral change, not a
byte-identical swap**: ~2× as many frames, so ~2× pose-detection compute and prediction
density, and the tracker's constant-velocity extrapolator sees a denser (better) grid.
`max_frames` still bounds capped runs.

## Load-bearing vs deleted

- **Load-bearing (keep):** `discover_exo_camera_streams`, `detect_uniform_native_fps` /
  `native_fps_from_packet_ns` / `_catalog_entity_packet_ns`, `_arrow_time_column_to_ns`,
  `build_rerun_iterable_dataset` (now the stock 4-liner), `build_torch_loader`,
  `none_decoded_exo_stream_names` (frame-0 pre-keyframe None safety net),
  `index_value_to_time_ns`, `rgb_chw_to_bgr_hwc`, the calibration / screenshot /
  registration stack.
- **Deleted (native-rate simplification):** `DurationTimelineSampleIndex`,
  `DurationTimelineSegment`, `CatalogTimeRange`, `intersect_time_ranges`,
  `align_time_range_to_sample_grid`, `sample_rate_hz_to_ns`,
  `build_duration_video_time_sample_index`, `select_video_time_target_values`,
  `catalog_exo_video_time_range`, `catalog_common_exo_video_time_values`,
  `catalog_segment_time_range`, `_index_range_columns`, `_catalog_entity_time_range`, the
  `RerunIterableDataset.__new__` private-attr injection, `Field.window`, and the
  `sample_rate_hz` config field (replaced by `native_fps_override`).
