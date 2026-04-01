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

## Test Results

### Hot3D Quest VRS (2.7GB)
- 2 streams: `1201-1` (camera-slam-left), `1201-2` (camera-slam-right)
- 7,966 records total (1 config + 1 state + 3,981 data per stream)
- All images JPEG encoded (~360KB per frame, 1280x1024 mono)
- Output RRD: 2.7GB
- Entity paths use `flavor` (unique): `camera-slam-left`, `camera-slam-right`

### Hot3D Aria VRS (1.7GB)
- 8 streams: 3 cameras (1201-1, 1201-2, 214-1), 2 IMUs (1202-1, 1202-2), 3 skipped (285-1, 285-2, 286-1)
- 237,185 records total
- Camera images: JPEG encoded
- IMU: accelerometer + gyroscope (no magnetometer in this recording)
- Skipped: Time Domain Mapping (285) and Attention Data (286) — no player implemented
- Entity paths use stream_id directly because `flavor` ("device/ariane") is shared across all streams
- Output RRD: 1.7GB

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
