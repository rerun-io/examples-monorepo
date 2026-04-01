# pyvrs-viewer Development Learnings

## pyvrs API (PyPI package: `vrs`)

### Stream ID format
VRS stream IDs are strings like `"1201-1"` where:
- The first number is the **RecordableTypeId** — a numeric enum from [vrs/StreamId.h](https://github.com/facebookresearch/vrs/blob/main/vrs/StreamId.h) that identifies the sensor type (e.g., 1201 = SLAM camera, 1202 = IMU, 214 = RGB camera)
- The second number is the **instance ID** — differentiates multiple sensors of the same type (e.g., `1201-1` = left SLAM camera, `1201-2` = right SLAM camera)

### SyncVRSReader key attributes/methods
- `reader.stream_ids` — `set[str]`, each string like `"1201-1"` (type_id-instance_id)
- `reader.get_stream_info(stream_id)` — dict with `device_name`, `flavor`, `configuration_records_count`, `data_records_count`, etc.
- `reader.might_contain_images(stream_id)` — bool, checks if stream has image blocks
- `reader.might_contain_audio(stream_id)` — bool
- `reader.stream_tags` — dict of stream_id → tags dict
- `reader.n_records` — total record count (for tqdm progress bars)
- Iteration: `for record in reader:` yields records in timestamp order

### Record structure
- `record.stream_id` — str like `"1201-1"`
- `record.record_type` — str: `"configuration"`, `"state"`, or `"data"`
- `record.timestamp` — float, seconds
- `record.n_image_blocks` / `record.n_metadata_blocks` / `record.n_audio_blocks`
- `record.metadata_blocks[j]` — returns a Python dict with typed fields
- `record.image_blocks[j]` — returns 1D uint8 numpy array (raw bytes for jpg/video, decoded for raw)
- `record.image_specs[j]` — `ImageSpec` object with `.image_format`, `.codec_name`, `.width`, `.height`

### Image format handling
- For `image_format == "jpg"`: `image_blocks[j]` returns **raw JPEG bytes** (not decoded). Width/height are 0 in the spec since JPEG is self-describing.
- For `image_format == "video"`: returns raw codec frame bytes. Check `codec_name` for H264/H265.
- For `image_format == "raw"`: returns decoded pixel data. Width/height available in spec.

## Rerun Integration

### RerunTyroConfig (from simplecv)
- `__post_init__` calls `rr.init()` automatically — no separate init needed
- When `save` is set, calls `rr.save(path)` automatically
- Exposes `--rr-config.save`, `--rr-config.connect`, `--rr-config.headless` via tyro CLI

### Image archetype strategy
1. **`rr.VideoStream`** (default, encode_video=True) — AV1/H265 encoded, 13-42x smaller RRDs
2. **`rr.EncodedImage`** (encode_video=False) — raw JPEG passthrough, fastest but large RRDs
3. **`rr.Image`** — decoded pixels, only for RAW format fallback

### IMU batch logging
- `rr.send_columns()` is **282x faster** than row-by-row `rr.log()` for IMU data
- Aria VRS has 224k IMU records — batch logging saves ~4s
- Accumulate timestamps + sensor arrays during VRS read, flush once at the end

### Dynamic Blueprint
- `rerun.blueprint` module: `Grid`, `Horizontal`, `Vertical`, `Tabs` containers
- `rrb.Spatial2DView(origin=entity)` for cameras
- `rrb.TimeSeriesView(origin=entity)` for IMU data
- `rrb.TextDocumentView(origin=entity)` for config/metadata
- Send early: `rr.send_blueprint(blueprint, make_active=True, make_default=True)`

## Video Encoding

### Codec defaults
- **AV1** is the default codec (best compression, wide NVENC support)
- Encoder preference: NVENC hardware first → CPU fallback (libsvtav1 or libx265)
- NVENC default rate control gives best size/quality tradeoff (don't set constqp — it's a different scale than CRF)
- Software encoders use CRF=30 (validated by HuggingFace/LeRobot benchmarks)
- GOP=30 (1-second keyframe interval for Rerun viewer scrubbing; GOP=2 produces 10x larger files)

### PyAV encoding setup
- `av.CodecContext.create(codec_name, 'w')` for containerless encoding
- `from fractions import Fraction` (NOT `av.Fraction` — doesn't exist)
- `ctx.max_b_frames = 0` required by Rerun VideoStream
- `ctx.pix_fmt = 'yuv420p'` required by all video encoders

### Encoder packet buffering (critical)
- **NVENC buffers 2 frames**: submit frame 2 → get packet for frame 0
- **SVT-AV1 buffers ALL frames**: every packet comes from `flush()` only
- Packets have PTS that maps back to the original frame submission order
- Must track `PTS → (source_timestamp, frame_number)` to log packets at correct times
- Without this fix, video timestamps shift by 1-2 frames (NVENC) or collapse entirely (SVT-AV1)

### turbojpeg YUV decode (performance critical)
- JPEG internally stores YCbCr — `turbojpeg.decode_to_yuv_planes()` extracts it without RGB conversion
- Grayscale JPEG → 1 plane (Y only), color JPEG → 3 planes (Y, U, V)
- For grayscale: fill U/V with 128 (neutral chroma), cache the allocation
- 1.8x faster than `cv2.imdecode` + `av.VideoFrame.reformat()`

## Pipeline Architecture (encode_video=True)

### Streaming parallel pipeline
```
Phase 1: Read VRS + collect JPEG bytes (IMU accumulated, config logged inline)
Phase 2+3: pool.map iterator — parallel decode overlapped with serial encode
Phase 4: Batch IMU logging via rr.send_columns()
```

Key insight: `pool.map()` returns an **iterator** (not a list). By iterating directly, decode and encode overlap — the encoder starts as soon as the first decode finishes.

### Performance breakdown (Quest VRS, 7962 frames @ 1280x1024)

| Stage | Time |
|-------|------|
| VRS read + collect | 0.8s |
| JPEG decode (8 threads, turbojpeg YUV) | 2.6s |
| AV1 NVENC encode | ~3.8s (GPU throughput limit) |
| Rerun logging | 0.2s |
| **Total (overlapped)** | **~5s** |

Theoretical minimum: 3.8s (NVENC throughput). Current ~5s is close to optimal.

## Benchmark Results (5 Quest + 5 Aria, RTX 5090)

| Device | VRS Size | AV1 Time | AV1 RRD | Compression | JPEG Time | JPEG RRD |
|--------|----------|----------|---------|-------------|-----------|----------|
| Quest | 0.8-2.7 GB | 2-5s | 31-66 MB | 21-41x | 0.3-0.9s | 0.8-2.7 GB |
| Aria | 0.8-1.8 GB | 4-9s | 50-110 MB | 15-16x | 5-11s | 0.8-1.8 GB |

- AV1 is **faster than JPEG passthrough** on Aria (send_columns eliminates IMU overhead)
- C++ reference (rerun-io/cpp-example-vrs) takes ~19s for Quest (decodes every JPEG frame)

## Stream Entity Naming
- Quest VRS has unique `flavor` values per stream → used as Rerun entity path
- Aria VRS has shared `flavor` ("device/ariane") for all streams → falls back to stream_id string
- Detection: count flavor occurrences, use flavor only when count == 1

## Gotchas
- `SyncVRSReader` does NOT have a `.streams` attribute — use `.stream_ids` (set of strings)
- `ImageSpec` objects stringify to the format name (e.g., `"jpg"`) — access attributes via `.image_format`, `.width` etc.
- Configuration record timestamps are near-zero (5e-324) — sentinel values, not real timestamps
- `VRSBlocks` is not iterable directly — use `blocks[j]` with range(n_blocks)
- pyvrs is on PyPI as `vrs`, NOT on conda-forge
- NVENC `constqp qp=30` is NOT equivalent to CRF=30 — produces 10-30x larger files. Use NVENC defaults.
- Parallel NVENC sessions (one per stream) causes GPU contention — serial streaming overlap is faster
