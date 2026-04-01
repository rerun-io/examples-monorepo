# pyvrs-viewer Development Learnings

## pyvrs API (PyPI package: `vrs`)

### SyncVRSReader key attributes/methods
- `reader.stream_ids` — `set[str]`, each string like `"1201-1"` (type_id-instance_id)
- `reader.get_stream_info(stream_id)` — dict with `device_name`, `flavor`, `configuration_records_count`, `data_records_count`, etc.
- `reader.might_contain_images(stream_id)` — bool, checks if stream has image blocks
- `reader.might_contain_audio(stream_id)` — bool
- `reader.stream_tags` — dict of stream_id → tags dict
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

### Image archetype priority (memory efficiency)
1. `rr.EncodedImage(contents=jpeg_bytes)` — for JPEG streams, no decode needed
2. `rr.VideoStream(codec=..., sample=...)` — for H264/H265 streams, raw codec frames
3. `rr.Image(pixel_array)` — for RAW streams only (fallback)

### Timeline setup
- `rr.set_time("timestamp", duration=seconds)` — wall-clock time from VRS
- `rr.set_time("frame_number", sequence=n)` — per-stream frame counter

## H265 Video Encoding

### Encoder selection
- Tries `hevc_nvenc` (NVIDIA hardware) first, falls back to `libx265` (CPU)
- Both are available from conda-forge's `av` package on Linux
- RTX 5090 NVENC confirmed working — encoding is near-instant

### PyAV encoding setup
- Use `av.CodecContext.create(codec_name, 'w')` for containerless encoding
- Use `from fractions import Fraction` (NOT `av.Fraction` — doesn't exist)
- `ctx.max_b_frames = 0` required by Rerun VideoStream
- `ctx.pix_fmt = 'yuv420p'` — all H265 encoders require this
- Convert gray → yuv420p: `av.VideoFrame.from_ndarray(img, format='gray').reformat(format='yuv420p')`
- `bytes(packet)` from `ctx.encode(frame)` gives raw H265 Annex B data
- Must call `ctx.encode(None)` at end to flush buffered frames

### Rerun VideoStream integration
- Log codec once as static: `rr.log(entity, rr.VideoStream(codec=rr.VideoCodec.H265), static=True)`
- Log each packet: `rr.log(entity, rr.VideoStream.from_fields(sample=packet_bytes))`
- Encoder may buffer frames — first few `encode()` calls return empty lists
- Flush packets must also be logged at the end

### Dynamic Blueprint
- `rerun.blueprint` module provides layout containers: `Grid`, `Horizontal`, `Vertical`, `Tabs`
- `rrb.Spatial2DView(origin=entity)` for cameras
- `rrb.TimeSeriesView(origin=entity)` for IMU data
- Send early: `rr.send_blueprint(blueprint, make_active=True, make_default=True)`

## Test Results

### Hot3D Quest VRS (2.7GB input, 2x 1280x1024 mono @ 30fps, 3981 frames/stream)

| Encoder | Encode FPS | Total Time | RRD Size | Reduction |
|---------|-----------|------------|----------|-----------|
| No encoding (JPEG) | — | 37s | 2.7 GB | — |
| **H265 NVENC** (RTX 5090) | **4800 fps** | 37s | **70 MB** | **38x** |
| **AV1 NVENC** (RTX 5090) | **5200 fps** | 37s | **70 MB** | **38x** |
| H265 CPU (libx265) | 530 fps | 61s | **66 MB** | **41x** |

- Pipeline time dominated by VRS reading + JPEG decoding (~36s), encoding is negligible with NVENC
- NVENC 10x faster than CPU (4800 vs 530 fps) with 0.8s vs 6-9s encode time
- AV1 and H265 similar compression on mono SLAM data (high temporal redundancy)

### Hot3D Aria VRS (1.7GB input)

| Encoder | RRD Size | Reduction |
|---------|----------|-----------|
| No encoding (JPEG) | 1.7 GB | — |
| **H265 NVENC** | **129 MB** | **13x** |

- 3 cameras (1408x1408 RGB + 2x 640x480 mono) + 2 IMUs
- 237,185 records total

## Stream Entity Naming
- Quest VRS has unique `flavor` values per stream → used as Rerun entity path
- Aria VRS has shared `flavor` ("device/ariane") for all streams → falls back to stream_id string
- Detection: count flavor occurrences, use flavor only when count == 1

## Gotchas
- `SyncVRSReader` does NOT have a `.streams` attribute — use `.stream_ids` (set of strings)
- `ImageSpec` objects stringify to the format name (e.g., `"jpg"`) — access attributes via `.image_format`, `.width` etc.
- Configuration record timestamps are near-zero (5e-324) — these are sentinel values, not real timestamps
- `VRSBlocks` is not iterable directly — use `blocks[j]` with range(n_blocks)
- pyvrs is on PyPI as `vrs`, NOT on conda-forge
